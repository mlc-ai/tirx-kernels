# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import annotations

import ctypes
import os
import re
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Any
from unittest import SkipTest

import torch

from tvm.ir.type import PointerType, PrimType

_DEEP_GEMM_MODULE_NAME = "deep_gemm"
_SM100_SMEM_CAPACITY = 232448
_CUDA_TENSOR_MAP_DATA_TYPE_FLOAT32 = 7
_CUDA_TENSOR_MAP_DATA_TYPE_TFLOAT32 = 11
_TEST_DIFF_THRESHOLD = 1e-8


def _tf32_hc_cuda_postproc(code: str) -> str:
    original = code
    code, unroll_count = re.subn(
        r"(\n    )#pragma unroll\n(    for \(uint s_2 =)", r"\1#pragma unroll 12\n\2", code, count=1
    )
    dump_dir = os.environ.get("TF32_HC_POSTPROC_DUMP_DIR")
    if dump_dir:
        dump_path = Path(dump_dir)
        dump_path.mkdir(parents=True, exist_ok=True)
        (dump_path / "original.cu").write_text(original)
        (dump_path / "postproc.cu").write_text(code)
        (dump_path / "notes.txt").write_text(f"unroll12={unroll_count}\n")
    return code


@dataclass(frozen=True)
class TF32HCPrenormGemmConfig:
    m: int = 13
    n: int = 24
    k: int = 512
    num_splits: int = 1
    seed: int = 0
    num_sms: int = 148

    @property
    def block_m(self) -> int:
        return 64

    @property
    def block_k(self) -> int:
        return 64

    @property
    def block_n(self) -> int:
        return _align_up(self.n, 16)

    @property
    def num_threads(self) -> int:
        return 256

    @property
    def num_mma_threads(self) -> int:
        return 128

    @property
    def num_cast_and_reduce_threads(self) -> int:
        return 128

    @property
    def swizzle_cd_mode(self) -> int:
        return _get_swizzle_mode(self.block_n, torch.empty((), dtype=torch.float32).element_size())

    @property
    def smem_a_size_per_stage(self) -> int:
        return self.block_m * self.block_k * torch.empty((), dtype=torch.bfloat16).element_size()

    @property
    def smem_b_size_per_stage(self) -> int:
        return self.block_n * self.block_k * torch.empty((), dtype=torch.float32).element_size()

    @property
    def smem_cd_size(self) -> int:
        return self.block_m * self.swizzle_cd_mode

    @property
    def num_stages(self) -> int:
        num_stages = 12
        while num_stages > 0:
            smem_barriers = (num_stages * 4 + 1) * 8
            smem_tmem_ptr = 4
            smem_size = (
                (self.smem_a_size_per_stage + self.smem_b_size_per_stage) * num_stages
                + self.smem_cd_size
                + smem_barriers
                + smem_tmem_ptr
            )
            if smem_size <= _SM100_SMEM_CAPACITY:
                return num_stages
            num_stages -= 1
        raise ValueError("no valid stage count fits SM100 shared memory")

    @property
    def smem_size(self) -> int:
        num_stages = self.num_stages
        return (
            (self.smem_a_size_per_stage + self.smem_b_size_per_stage) * num_stages
            + self.smem_cd_size
            + (num_stages * 4 + 1) * 8
            + 4
        )

    @property
    def grid_blocks(self) -> int:
        return self.num_splits * _ceil_div(self.m, self.block_m)

    @property
    def num_k_blocks(self) -> int:
        return self.k // self.block_k

    @property
    def d_shape(self) -> tuple[int, ...]:
        if self.num_splits == 1:
            return (self.m, self.n)
        return (self.num_splits, self.m, self.n)

    @property
    def sqr_sum_shape(self) -> tuple[int, ...]:
        if self.num_splits == 1:
            return (self.m,)
        return (self.num_splits, self.m)

    def validate(self) -> None:
        if self.m <= 0 or self.n <= 0 or self.k <= 0:
            raise ValueError("m, n, and k must be positive")
        if self.n > 128 or self.n % 8 != 0:
            raise ValueError("DeepGEMM requires n <= 128 and n % 8 == 0")
        if self.k % self.block_k != 0:
            raise ValueError("DeepGEMM requires k % 64 == 0")
        if (
            self.swizzle_cd_mode // torch.empty((), dtype=torch.float32).element_size()
            != self.block_n
        ):
            raise ValueError("DeepGEMM requires swizzle_cd_mode / sizeof(float) == BLOCK_N")
        if self.num_splits <= 0:
            raise ValueError("num_splits must be positive")
        if self.num_sms <= 0:
            raise ValueError("num_sms must be positive")


def _make_config(**kwargs: Any) -> TF32HCPrenormGemmConfig:
    kwargs = {key: value for key, value in kwargs.items() if key != "label"}
    config = TF32HCPrenormGemmConfig(**kwargs)
    config.validate()
    return config


def _align_up(x: int, y: int) -> int:
    return (x + y - 1) // y * y


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _get_swizzle_mode(block_size: int, elem_size: int) -> int:
    for mode in (128, 64, 32, 16):
        if block_size * elem_size % mode == 0:
            return mode
    return 0


def _config_label(config: dict[str, Any]) -> str:
    split = config["num_splits"]
    return f"m{config['m']}_n{config['n']}_k{config['k']}_s{split}"


def _make_case(*, m: int, n: int, k: int, num_splits: int, seed: int) -> dict[str, Any]:
    config = {"m": m, "n": n, "k": k, "num_splits": num_splits, "seed": seed}
    config["label"] = _config_label(config)
    return config


KERNEL_META = {
    "name": "deepgemm_sm100_tf32_hc_prenorm_gemm",
    "category": "deepgemm",
    "compute_capability": 10,
}

DEEPGEMM_TEST_COVERAGE = [
    _make_case(m=m, n=n, k=k, num_splits=num_splits, seed=1000 + seed)
    for seed, (m, n, k, num_splits) in enumerate(
        (m, n, k, num_splits)
        for m in (13, 137, 4096, 8192)
        for n, k in ((24, 28672), (24, 7680), (24, 7168))
        for num_splits in (1, 16)
    )
]

CONFIGS = [
    _make_case(m=13, n=24, k=512, num_splits=1, seed=0),
    _make_case(m=137, n=24, k=1024, num_splits=4, seed=1),
    _make_case(m=13, n=24, k=7168, num_splits=16, seed=2),
    DEEPGEMM_TEST_COVERAGE[0],
]

BENCH_CONFIGS = [
    _make_case(m=13, n=24, k=7168, num_splits=1, seed=2000),
    _make_case(m=137, n=24, k=7680, num_splits=16, seed=2001),
    _make_case(m=4096, n=24, k=7168, num_splits=1, seed=2002),
    _make_case(m=4096, n=24, k=28672, num_splits=16, seed=2003),
]


def load_deep_gemm_hc() -> tuple[Any, str]:
    try:
        import deep_gemm as module

        source = "installed"
    except Exception as exc:
        raise SkipTest(
            f"DeepGEMM HC prenorm GEMM runtime unavailable: {_DEEP_GEMM_MODULE_NAME}: {exc}"
        ) from exc

    if not hasattr(module, "tf32_hc_prenorm_gemm"):
        raise SkipTest("DeepGEMM runtime unavailable: missing tf32_hc_prenorm_gemm")
    return module, source


def _get_num_sms(default: int) -> int:
    if torch.cuda.is_available():
        return int(
            torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        )
    return default


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    deep_gemm, source = load_deep_gemm_hc()
    config = _make_config(**kwargs)
    if torch.cuda.is_available():
        torch.cuda.set_device(torch.cuda.current_device())
    else:
        raise SkipTest("CUDA is required for SM100 TF32 HC prenorm GEMM")
    if torch.cuda.get_device_capability()[0] < 10:
        raise SkipTest("SM100 TF32 HC prenorm GEMM requires compute capability 10.x")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(config.seed)

    runtime_config = TF32HCPrenormGemmConfig(
        **{
            **asdict(config),
            "num_sms": int(
                getattr(deep_gemm, "get_num_sms", lambda: _get_num_sms(config.num_sms))()
            ),
        }
    )
    a = torch.randn((config.m, config.k), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((config.n, config.k), dtype=torch.float32, device="cuda")
    d_deepgemm = torch.empty(config.d_shape, dtype=torch.float32, device="cuda")
    sqr_deepgemm = torch.empty(config.sqr_sum_shape, dtype=torch.float32, device="cuda")
    d_tirx = torch.empty(config.d_shape, dtype=torch.float32, device="cuda")
    sqr_tirx = torch.empty(config.sqr_sum_shape, dtype=torch.float32, device="cuda")
    reference_d = a.float() @ b.T
    reference_sqr = a.float().square().sum(dim=-1)
    return {
        "config": runtime_config,
        "reference_source": source,
        "a": a,
        "b": b,
        "d_deepgemm": d_deepgemm,
        "sqr_deepgemm": sqr_deepgemm,
        "d_tirx": d_tirx,
        "sqr_tirx": sqr_tirx,
        "reference_d": reference_d,
        "reference_sqr": reference_sqr,
        "deep_gemm": deep_gemm,
    }


@dataclass
class TF32HCBenchCase:
    config: TF32HCPrenormGemmConfig
    deep_gemm: Any
    a: torch.Tensor
    b: torch.Tensor
    d_deepgemm: torch.Tensor
    sqr_deepgemm: torch.Tensor
    d_tirx: torch.Tensor
    sqr_tirx: torch.Tensor
    reference_d: torch.Tensor
    reference_sqr: torch.Tensor
    tensor_maps: dict[str, Any]


def get_kernel(**kwargs: Any):
    from tvm.script import tirx as T
    from tvm.tirx.lang.smem_desc import SmemDescriptor
    from tvm.tirx.layout import S, TCol, TileLayout, TLane

    config = _make_config(**kwargs)
    block_m = config.block_m
    block_n = config.block_n
    block_k = config.block_k
    num_splits = config.num_splits
    num_threads = config.num_threads
    num_warps = num_threads // 32
    num_mma_threads = config.num_mma_threads
    num_cast_and_reduce_threads = config.num_cast_and_reduce_threads
    num_mma_warps = num_mma_threads // 32
    num_stages = config.num_stages
    num_cast_stages = 2
    swizzle_a_mode = min(block_k * 2, 128)
    swizzle_b_mode = min(block_k * 4, 128)
    swizzle_cd_mode = config.swizzle_cd_mode
    smem_cd_size = config.smem_cd_size
    smem_a_size_per_stage = config.smem_a_size_per_stage
    smem_b_size_per_stage = config.smem_b_size_per_stage
    smem_a_offset = smem_cd_size
    smem_b_offset = smem_a_offset + num_stages * smem_a_size_per_stage
    smem_barrier_offset = smem_b_offset + num_stages * smem_b_size_per_stage
    num_total_barriers = num_stages * 4 + 1
    full_barrier_base = 0
    full_cast_barrier_base = num_stages
    empty_barrier_base = num_stages * 2
    empty_cast_barrier_base = num_stages * 3
    tmem_full_barrier_idx = num_stages * 4
    smem_tmem_ptr_offset = smem_barrier_offset + num_total_barriers * 8
    smem_total_bytes = smem_tmem_ptr_offset + 4
    if smem_total_bytes != config.smem_size:
        raise ValueError("shared-memory layout size mismatch")
    num_tmem_cols = 256
    block_swizzled_bk = swizzle_b_mode // 4
    num_b_tma_atoms = block_k // block_swizzled_bk
    umma_k = 32 // 4
    d_tmem_start_col = block_k * num_cast_stages
    tmem_layout = TileLayout(S[(128, num_tmem_cols) : (1 @ TLane, 1 @ TCol)])
    b_sdo = 8 * block_swizzled_bk * 4 // 16
    b_swizzle = 3
    num_k_blocks = config.num_k_blocks
    num_k_blocks_per_split = num_k_blocks // num_splits
    remain_k_blocks = num_k_blocks % num_splits

    def cuda_grid_dependency_synchronize():
        T.evaluate(T.ptx.griddepcontrol.wait())

    def tma_load_2d_cluster(dst, bar, tensormap, coord0, coord1):
        T.evaluate(
            T.ptx.cp_async.bulk.tensor.g2c(
                2, dst, bar, T.address_of(tensormap), 0, 1, "evict_normal", coord0, coord1
            )
        )

    def tma_store_2d(src, tensormap, coord0, coord1):
        T.evaluate(
            T.ptx.cp_async.bulk.tensor.s2g(2, src, T.address_of(tensormap), "", coord0, coord1)
        )

    def tma_store_3d(src, tensormap, coord0, coord1, coord2):
        T.evaluate(
            T.ptx.cp_async.bulk.tensor.s2g(
                3, src, T.address_of(tensormap), "", coord0, coord1, coord2
            )
        )

    def mbarrier_init_cta(barrier, arrive_count):
        T.evaluate(T.ptx.mbarrier.init(barrier, arrive_count))

    def mbarrier_arrive_cta(barrier):
        T.evaluate(T.ptx.mbarrier.arrive(barrier))

    def mbarrier_arrive_expect_tx_cta(barrier, transaction_bytes):
        T.evaluate(T.ptx.mbarrier.arrive.expect_tx(barrier, transaction_bytes))

    def get_swizzled_smem_offset(offset, lane_idx, swizzle_mode):
        swizzle_base = T.uint32(16)
        groups_in_swizzle = T.uint32(swizzle_mode // 16)
        bank_group_idx = offset + lane_idx * groups_in_swizzle
        num_bank_groups = T.uint32(8)
        if swizzle_mode == 128:
            row = offset // num_bank_groups + lane_idx
            col = offset
        else:
            row = bank_group_idx // num_bank_groups
            col = bank_group_idx % num_bank_groups
        col = T.bitwise_xor(col, row % groups_in_swizzle)
        return row * T.uint32(128) + col * swizzle_base

    def ldmatrix_x4_b16(d0, d1, d2, d3, smem_src):
        T.evaluate(T.ptx.ldmatrix(False, 4, ".b16", smem_src, d0, d1, d2, d3))

    def st_shared_v4_u32(smem_dst, v0, v1, v2, v3):
        T.evaluate(T.ptx.st(smem_dst, v0, v1, v2, v3, space="shared", ptx_type="u32", vec="v4"))

    def ffma2_rn(a, b, c):
        # ``__ffma2_rn`` emits ``fma.rn.f32x2`` (non-ftz, IEEE-754); verified
        # via nvcc PTX dump on sm_100a.
        out = T.alloc_local((1,), "uint64")
        T.evaluate(T.ptx.fma_f32x2(out.ptr_to([0]), a, b, c, rounding="rn", ftz=False))
        return out[0]

    def elect_one_sync_value():
        return T.ptx.elect_sync()

    def tcgen05_predicated_mma(d_tmem_addr, a_operand, b_desc, i_desc, scale_c, issue):
        T.evaluate(
            T.ptx.tcgen05.mma(
                d_tmem_addr,
                a_operand,
                b_desc,
                i_desc,
                d_dtype="float32",
                a_dtype="tf32",
                b_dtype="tf32",
                use_a_tmem=True,
                cta_group=1,
                enable_input_d=T.cast(scale_c, "uint32"),
                scale_input_d=0,
                pred=issue,
            )
        )

    def tcgen05_predicated_commit(barrier, issue):
        T.evaluate(T.ptx.tcgen05.commit(barrier, cta_group=1, pred=issue))

    def smem_desc_replace_lo(desc_base, lo):
        # Equivalent to `desc.lo = lo` on a 64-bit SmemDescriptor (low 32 bits).
        return T.bitwise_or(
            T.bitwise_and(desc_base, T.bitwise_not(T.uint64(0xFFFFFFFF))), T.cast(lo, "uint64")
        )

    def advance_umma_desc_lo(base, offset, k_idx):
        return base + ((offset + k_idx) * T.uint32(4) >> T.uint32(4))

    def warp_reduce_sum4(value):
        value = value + T.tvm_warp_shuffle_xor(T.uint32(0xFFFFFFFF), value, 2, 32, 32)
        value = value + T.tvm_warp_shuffle_xor(T.uint32(0xFFFFFFFF), value, 1, 32, 32)
        return value

    @T.prim_func
    def sm100_tf32_hc_prenorm_gemm(
        shape_m: T.uint32,
        tensor_map_a: T.TensorMap(),
        tensor_map_b: T.TensorMap(),
        tensor_map_d: T.TensorMap(),
        sqr_sum: T.Buffer((config.num_splits * config.m,), "float32"),
    ):
        T.device_entry()
        # TIRX_TRANSCRIBE_START sm100_tf32_hc_prenorm_gemm_impl
        T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})
        warp_idx = T.warp_id([num_warps])
        lane_idx = T.lane_id([32])
        lane_u32: T.uint32 = T.cast(lane_idx, "uint32")

        if warp_idx == 0:
            if T.ptx.elect_sync():
                T.evaluate(T.ptx.prefetch_tensormap(T.address_of(tensor_map_a)))
                T.evaluate(T.ptx.prefetch_tensormap(T.address_of(tensor_map_b)))
                T.evaluate(T.ptx.prefetch_tensormap(T.address_of(tensor_map_d)))

        smem = T.alloc_buffer([smem_total_bytes], "uint8", scope="shared.dyn", align=1024)
        smem_cd_data: T.let[T.Var(name="smem_cd_data", dtype=PointerType(PrimType("float32")))] = (
            T.reinterpret("handle", smem.ptr_to([0]))
        )
        smem_a_data: T.let[T.Var(name="smem_a_data", dtype=PointerType(PrimType("bfloat16")))] = (
            T.reinterpret("handle", smem.ptr_to([smem_a_offset]))
        )
        smem_b_data: T.let[T.Var(name="smem_b_data", dtype=PointerType(PrimType("float32")))] = (
            T.reinterpret("handle", smem.ptr_to([smem_b_offset]))
        )
        smem_barrier_data: T.let[
            T.Var(name="smem_barrier_data", dtype=PointerType(PrimType("uint64")))
        ] = T.reinterpret("handle", smem.ptr_to([smem_barrier_offset]))
        smem_tmem_ptr_data: T.let[
            T.Var(name="smem_tmem_ptr_data", dtype=PointerType(PrimType("uint32")))
        ] = T.reinterpret("handle", smem.ptr_to([smem_tmem_ptr_offset]))
        smem_cd = T.decl_buffer(
            (smem_cd_size,), "uint8", data=smem.data, scope="shared.dyn", elem_offset=0
        )
        smem_a = T.decl_buffer(
            (num_stages, block_m, block_k),
            "bfloat16",
            data=smem_a_data,
            scope="shared.dyn",
            elem_offset=0,
            align=1024,
        )
        smem_b = T.decl_buffer(
            (num_stages, num_b_tma_atoms, block_n, block_swizzled_bk),
            "float32",
            data=smem_b_data,
            scope="shared.dyn",
            elem_offset=0,
            align=1024,
        )
        smem_barriers = T.decl_buffer(
            (num_total_barriers,),
            "uint64",
            data=smem_barrier_data,
            scope="shared.dyn",
            elem_offset=0,
            align=8,
        )
        tmem_ptr_in_smem = T.decl_buffer(
            (1,), "uint32", data=smem_tmem_ptr_data, scope="shared.dyn", elem_offset=0, align=4
        )
        _tmem = T.decl_buffer(
            (128, num_tmem_cols),
            "float32",
            scope="tmem",
            allocated_addr=tmem_ptr_in_smem[0],
            layout=tmem_layout,
        )

        if warp_idx == 1:
            if T.ptx.elect_sync():
                for init_i in T.unroll(0, num_stages):
                    mbarrier_init_cta(
                        smem_barriers.ptr_to([full_barrier_base + init_i]), T.uint32(1)
                    )
                    mbarrier_init_cta(
                        smem_barriers.ptr_to([full_cast_barrier_base + init_i]),
                        T.uint32(num_cast_and_reduce_threads),
                    )
                    mbarrier_init_cta(
                        smem_barriers.ptr_to([empty_barrier_base + init_i]), T.uint32(1)
                    )
                    mbarrier_init_cta(
                        smem_barriers.ptr_to([empty_cast_barrier_base + init_i]), T.uint32(1)
                    )
                mbarrier_init_cta(smem_barriers.ptr_to([tmem_full_barrier_idx]), T.uint32(1))
                T.ptx.fence.mbarrier_init()
        elif warp_idx == 2:
            T.ptx.tcgen05.alloc(
                T.address_of(tmem_ptr_in_smem[0]), n_cols=num_tmem_cols, cta_group=1
            )
        T.cuda.cta_sync()

        block_idx_raw = T.cast(T.cta_id([config.grid_blocks]), "uint32")
        block_idx = T.tvm_warp_shuffle(T.uint32(0xFFFFFFFF), block_idx_raw, 0, 32, 32)
        m_block_idx: T.uint32 = block_idx // T.uint32(num_splits)
        k_split_idx: T.uint32 = block_idx % T.uint32(num_splits)
        k_offset: T.uint32 = (
            k_split_idx * T.uint32(num_k_blocks_per_split)
            + T.min(k_split_idx, T.uint32(remain_k_blocks))
        ) * T.uint32(block_k)
        m_offset: T.uint32 = shape_m * k_split_idx
        num_total_stages: T.uint32 = T.uint32(num_k_blocks_per_split) + T.cast(
            k_split_idx < T.uint32(remain_k_blocks), "uint32"
        )

        cuda_grid_dependency_synchronize()

        if warp_idx < num_mma_warps:
            if warp_idx == 0:
                if T.ptx.elect_sync():
                    for s in T.serial(T.uint32(0), num_total_stages):
                        stage_idx: T.uint32 = s % T.uint32(num_stages)
                        T.ptx.mbarrier.try_wait(
                            smem_barriers.ptr_to([empty_barrier_base + stage_idx]),
                            ((s // T.uint32(num_stages)) & T.uint32(1)) ^ T.uint32(1),
                        )
                        m_idx: T.uint32 = m_block_idx * T.uint32(block_m)
                        k_idx: T.uint32 = k_offset + s * T.uint32(block_k)
                        tma_load_2d_cluster(
                            smem_a.ptr_to([stage_idx, 0, 0]),
                            smem_barriers.ptr_to([full_barrier_base + stage_idx]),
                            tensor_map_a,
                            k_idx,
                            m_idx,
                        )
                        for b_atom in T.unroll(0, num_b_tma_atoms):
                            tma_load_2d_cluster(
                                smem_b.ptr_to([stage_idx, b_atom, 0, 0]),
                                smem_barriers.ptr_to([full_barrier_base + stage_idx]),
                                tensor_map_b,
                                k_idx + T.uint32(b_atom * block_swizzled_bk),
                                T.uint32(0),
                            )
                        mbarrier_arrive_expect_tx_cta(
                            smem_barriers.ptr_to([full_barrier_base + stage_idx]),
                            T.uint32(smem_a_size_per_stage + smem_b_size_per_stage),
                        )

            if warp_idx == 1:
                desc_i: T.uint32
                T.ptx.tcgen05.encode_instr_descriptor(
                    T.address_of(desc_i),
                    d_dtype="float32",
                    a_dtype="tf32",
                    b_dtype="tf32",
                    M=block_m,
                    N=block_n,
                    K=umma_k,
                    trans_a=False,
                    trans_b=False,
                    n_cta_groups=1,
                )
                desc_b = SmemDescriptor()
                desc_b.init(smem_b.ptr_to([0, 0, 0, 0]), ldo=0, sdo=b_sdo, swizzle=b_swizzle)
                b_desc_base_lo: T.uint32 = T.cast(desc_b.desc, "uint32")
                b_desc_lo: T.uint32 = T.if_then_else(
                    lane_u32 < T.uint32(num_stages),
                    b_desc_base_lo + lane_u32 * T.uint32(smem_b_size_per_stage // 16),
                    T.uint32(0),
                )
                mma_elect_once: T.uint32 = elect_one_sync_value()
                for s in T.serial(T.uint32(0), num_total_stages):
                    stage_idx: T.uint32 = s % T.uint32(num_stages)
                    cast_stage_idx: T.uint32 = s % T.uint32(num_cast_stages)
                    T.ptx.mbarrier.try_wait(
                        smem_barriers.ptr_to([full_cast_barrier_base + cast_stage_idx]),
                        (s // T.uint32(num_cast_stages)) & T.uint32(1),
                    )
                    b_desc_shuffled_lo: T.uint32 = T.tvm_warp_shuffle(
                        T.uint32(0xFFFFFFFF), b_desc_lo, stage_idx, 32, 32
                    )
                    for mma_k in T.serial(T.uint32(0), T.uint32(block_k // umma_k), unroll=True):
                        mma_col: T.uint32 = mma_k * T.uint32(umma_k)
                        atom_idx: T.uint32 = mma_col // T.uint32(block_swizzled_bk)
                        in_atom_idx: T.uint32 = mma_col % T.uint32(block_swizzled_bk)
                        b_offset: T.uint32 = atom_idx * T.uint32(block_n * block_swizzled_bk)
                        b_desc_lo_i: T.uint32 = advance_umma_desc_lo(
                            b_desc_shuffled_lo, b_offset, in_atom_idx
                        )
                        b_desc_i: T.uint64 = smem_desc_replace_lo(desc_b.desc, b_desc_lo_i)
                        tcgen05_predicated_mma(
                            T.cuda.get_tmem_addr(T.uint32(0), 0, d_tmem_start_col),
                            T.cuda.get_tmem_addr(
                                T.uint32(0), 0, cast_stage_idx * T.uint32(block_k) + mma_col
                            ),
                            b_desc_i,
                            desc_i,
                            (s != T.uint32(0)) | (mma_k != T.uint32(0)),
                            mma_elect_once,
                        )
                    tcgen05_predicated_commit(
                        smem_barriers.ptr_to([empty_cast_barrier_base + cast_stage_idx]),
                        mma_elect_once,
                    )
                    tcgen05_predicated_commit(
                        smem_barriers.ptr_to([empty_barrier_base + stage_idx]), mma_elect_once
                    )
                tcgen05_predicated_commit(
                    smem_barriers.ptr_to([tmem_full_barrier_idx]), mma_elect_once
                )

            T.ptx.mbarrier.try_wait(smem_barriers.ptr_to([tmem_full_barrier_idx]), 0)
            values = T.alloc_local((4,), "uint32")
            for epi_i in T.unroll(0, block_n // 4):
                T.ptx.tcgen05.ld(
                    T.uint32(0),
                    values[0],
                    values[1],
                    values[2],
                    values[3],
                    shape="32x32b",
                    num=4,
                    row=0,
                    col=d_tmem_start_col + epi_i * 4,
                )
                T.ptx.tcgen05.wait.ld()
                smem_offset: T.uint32 = T.cast(warp_idx, "uint32") * T.uint32(
                    block_m // 4 * swizzle_cd_mode
                ) + get_swizzled_smem_offset(T.uint32(epi_i), lane_u32, swizzle_cd_mode)
                if lane_u32 < T.uint32(16):
                    st_shared_v4_u32(
                        smem_cd.ptr_to([smem_offset]), values[0], values[1], values[2], values[3]
                    )
                T.cuda.warp_sync()

            T.ptx.fence.proxy_async("shared::cta")
            T.ptx.bar.sync(0, num_mma_threads)
            if warp_idx == 0:
                if T.ptx.elect_sync():
                    if num_splits == 1:
                        tma_store_2d(
                            smem_cd.ptr_to([0]),
                            tensor_map_d,
                            T.uint32(0),
                            m_block_idx * T.uint32(block_m),
                        )
                    else:
                        tma_store_3d(
                            smem_cd.ptr_to([0]),
                            tensor_map_d,
                            T.uint32(0),
                            m_block_idx * T.uint32(block_m),
                            k_split_idx,
                        )
                    T.ptx.cp_async.bulk.commit_group()
            if warp_idx == 1:
                T.ptx.tcgen05.dealloc(T.uint32(0), n_cols=num_tmem_cols, cta_group=1)
        else:
            sub_warp_idx: T.uint32 = T.cast(
                T.cast(warp_idx, "int32") - T.int32(num_mma_warps), "uint32"
            )
            uint32_values = T.alloc_local((2, block_k // 8), "uint32")
            fp32x2_values = T.alloc_local((2, block_k // 8), "uint64")
            store_values = T.alloc_local((4,), "uint32")
            sum0: T.uint64 = T.cuda.make_float2(T.float32(0), T.float32(0))
            sum1: T.uint64 = T.cuda.make_float2(T.float32(0), T.float32(0))
            for s in T.serial(T.uint32(0), num_total_stages, unroll=True):
                stage_idx: T.uint32 = s % T.uint32(num_stages)
                T.ptx.mbarrier.try_wait(
                    smem_barriers.ptr_to([full_barrier_base + stage_idx]),
                    (s // T.uint32(num_stages)) & T.uint32(1),
                )
                for load_i in T.unroll(0, block_k // 8 // 2):
                    i = load_i * 2
                    swizzled_offset: T.uint32 = get_swizzled_smem_offset(
                        T.uint32(i) + lane_u32 // T.uint32(16),
                        lane_u32 % T.uint32(16),
                        swizzle_a_mode,
                    )
                    smem_linear_byte: T.uint32 = (
                        T.cast(stage_idx, "uint32") * T.uint32(smem_a_size_per_stage)
                        + sub_warp_idx * T.uint32(block_m // 4 * swizzle_a_mode)
                        + swizzled_offset
                    )
                    ldmatrix_x4_b16(
                        T.address_of(uint32_values[0, i]),
                        T.address_of(uint32_values[1, i]),
                        T.address_of(uint32_values[0, i + 1]),
                        T.address_of(uint32_values[1, i + 1]),
                        smem.ptr_to([T.uint32(smem_a_offset) + smem_linear_byte]),
                    )
                cast_stage_idx: T.uint32 = s % T.uint32(num_cast_stages)
                T.ptx.mbarrier.try_wait(
                    smem_barriers.ptr_to([empty_cast_barrier_base + cast_stage_idx]),
                    ((s // T.uint32(num_cast_stages)) & T.uint32(1)) ^ T.uint32(1),
                )
                for i in T.unroll(0, block_k // 8):
                    fp32x2_values[0, i] = T.cuda.bfloat1622float2(uint32_values[0, i])
                    fp32x2_values[1, i] = T.cuda.bfloat1622float2(uint32_values[1, i])
                    sum0 = ffma2_rn(fp32x2_values[0, i], fp32x2_values[0, i], sum0)
                    sum1 = ffma2_rn(fp32x2_values[1, i], fp32x2_values[1, i], sum1)
                    store_values[0] = T.cuda.float_as_uint(T.cuda.float2_x(fp32x2_values[0, i]))
                    store_values[1] = T.cuda.float_as_uint(T.cuda.float2_y(fp32x2_values[0, i]))
                    store_values[2] = T.cuda.float_as_uint(T.cuda.float2_x(fp32x2_values[1, i]))
                    store_values[3] = T.cuda.float_as_uint(T.cuda.float2_y(fp32x2_values[1, i]))
                    T.ptx.tcgen05.st(
                        T.uint32(0),
                        store_values[0],
                        store_values[1],
                        store_values[2],
                        store_values[3],
                        shape="16x256b",
                        num=1,
                        row=0,
                        col=cast_stage_idx * block_k + i * 8,
                    )
                T.ptx.tcgen05.wait.st()
                mbarrier_arrive_cta(smem_barriers.ptr_to([full_cast_barrier_base + cast_stage_idx]))

            reduced0: T.float32 = warp_reduce_sum4(T.cuda.float2_x(sum0) + T.cuda.float2_y(sum0))
            reduced1: T.float32 = warp_reduce_sum4(T.cuda.float2_x(sum1) + T.cuda.float2_y(sum1))
            m_idx0: T.uint32 = (
                m_block_idx * T.uint32(block_m)
                + sub_warp_idx * T.uint32(block_m // 4)
                + lane_u32 // T.uint32(4)
            )
            m_idx1: T.uint32 = m_idx0 + T.uint32(8)
            if (lane_u32 % T.uint32(4)) == T.uint32(0):
                if m_idx0 < shape_m:
                    sqr_sum[T.cast(m_offset + m_idx0, "int32")] = reduced0
                if m_idx1 < shape_m:
                    sqr_sum[T.cast(m_offset + m_idx1, "int32")] = reduced1

    return sm100_tf32_hc_prenorm_gemm.with_attr(
        "tirx.kernel_launch_params",
        [
            "blockIdx.x",
            "threadIdx.x",
            "tirx.use_programtic_dependent_launch",
            "tirx.use_dyn_shared_memory",
        ],
    )


def _compile_tirx_tf32_hc_for_config(
    *, m: int, n: int, k: int, num_splits: int, seed: int, num_sms: int
) -> Any:
    import tvm

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    kernel = get_kernel(m=m, n=n, k=k, num_splits=num_splits, seed=seed, num_sms=num_sms)
    previous_postproc = tvm.get_global_func("tvm_callback_cuda_postproc", allow_missing=True)

    @tvm.register_global_func("tvm_callback_cuda_postproc", override=True)
    def _postproc(code: str, target: Any) -> str:
        if previous_postproc is not None:
            code = previous_postproc(code, target)
        if "sm100_tf32_hc_prenorm_gemm_kernel" in code:
            code = _tf32_hc_cuda_postproc(code)
        return code

    try:
        with target:
            mod = tvm.IRModule({"main": kernel})
            return tvm.compile(mod, target=target, tir_pipeline="tirx")
    finally:
        if previous_postproc is not None:
            tvm.register_global_func("tvm_callback_cuda_postproc", previous_postproc, override=True)
        else:
            tvm.register_global_func(
                "tvm_callback_cuda_postproc", lambda code, target: code, override=True
            )


_compile_tirx_tf32_hc_for_config = cache(_compile_tirx_tf32_hc_for_config)


def _compile_tirx_tf32_hc(config: TF32HCPrenormGemmConfig) -> Any:
    return _compile_tirx_tf32_hc_for_config(**asdict(config))


def _encode_driver_tma_desc(
    *,
    desc: Any,
    tensor: torch.Tensor,
    data_type: int,
    global_shape: tuple[int, ...],
    global_strides_bytes: tuple[int, ...],
    box_dim: tuple[int, ...],
    swizzle: int,
) -> None:
    from tirx_kernels.deepgemm import mega_moe

    rank = len(global_shape)
    global_shape_arr = (ctypes.c_uint64 * rank)(*[int(v) for v in global_shape])
    global_strides_arr = (ctypes.c_uint64 * (rank - 1))(*[int(v) for v in global_strides_bytes])
    box_dim_arr = (ctypes.c_uint32 * rank)(*[int(v) for v in box_dim])
    element_strides_arr = (ctypes.c_uint32 * rank)(*[1 for _ in range(rank)])
    result = mega_moe._get_cuda_driver().cuTensorMapEncodeTiled(
        desc.ptr,
        int(data_type),
        ctypes.c_uint32(rank),
        ctypes.c_void_p(int(tensor.data_ptr())),
        global_shape_arr,
        global_strides_arr,
        box_dim_arr,
        element_strides_arr,
        mega_moe._CUDA_TENSOR_MAP_INTERLEAVE_NONE,
        int(swizzle),
        mega_moe._CUDA_TENSOR_MAP_L2_PROMOTION_L2_256B,
        mega_moe._CUDA_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if result != 0:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed with CUresult={result}")


def _encode_tf32_tma_2d_desc(
    *,
    tensor: torch.Tensor,
    gmem_inner_dim: int,
    gmem_outer_dim: int,
    smem_inner_dim: int,
    smem_outer_dim: int,
    gmem_outer_stride: int,
    swizzle_mode: int,
) -> Any:
    from tirx_kernels.deepgemm import mega_moe

    elem_size = int(tensor.element_size())
    if swizzle_mode != 0:
        smem_inner_dim = swizzle_mode // elem_size
    desc = mega_moe._AlignedTensorMap()
    _encode_driver_tma_desc(
        desc=desc,
        tensor=tensor,
        data_type=_CUDA_TENSOR_MAP_DATA_TYPE_TFLOAT32,
        global_shape=(gmem_inner_dim, gmem_outer_dim),
        global_strides_bytes=(int(gmem_outer_stride * elem_size),),
        box_dim=(smem_inner_dim, smem_outer_dim),
        swizzle=mega_moe._tensor_map_swizzle_from_mode(swizzle_mode),
    )
    return desc


def _encode_float32_tma_3d_desc(
    *,
    tensor: torch.Tensor,
    gmem_dim_0: int,
    gmem_dim_1: int,
    gmem_dim_2: int,
    smem_dim_0: int,
    smem_dim_1: int,
    smem_dim_2: int,
    gmem_stride_0: int,
    gmem_stride_1: int,
    swizzle_mode: int,
) -> Any:
    from tirx_kernels.deepgemm import mega_moe

    elem_size = int(tensor.element_size())
    if swizzle_mode != 0:
        smem_dim_0 = swizzle_mode // elem_size
    desc = mega_moe._AlignedTensorMap()
    _encode_driver_tma_desc(
        desc=desc,
        tensor=tensor,
        data_type=_CUDA_TENSOR_MAP_DATA_TYPE_FLOAT32,
        global_shape=(gmem_dim_0, gmem_dim_1, gmem_dim_2),
        global_strides_bytes=(int(gmem_stride_0 * elem_size), int(gmem_stride_1 * elem_size)),
        box_dim=(smem_dim_0, smem_dim_1, smem_dim_2),
        swizzle=mega_moe._tensor_map_swizzle_from_mode(swizzle_mode),
    )
    return desc


def _build_tirx_tensor_maps(data: dict[str, Any]) -> dict[str, Any]:
    import tvm
    from tirx_kernels.deepgemm.mega_moe import _encode_tma_2d_desc

    config: TF32HCPrenormGemmConfig = data["config"]
    encode_tensormap = tvm.get_global_func("runtime.cuTensorMapEncodeTiled")
    a = data["a"]
    b = data["b"]
    d = data["d_tirx"]
    swizzle_ab_mode = _get_swizzle_mode(config.block_k, int(a.element_size()))
    swizzle_b_mode = _get_swizzle_mode(config.block_k, int(b.element_size()))
    maps = {
        "tensor_map_a": _encode_tma_2d_desc(
            encode_tensormap=encode_tensormap,
            tensor=a,
            gmem_inner_dim=config.k,
            gmem_outer_dim=config.m,
            smem_inner_dim=config.block_k,
            smem_outer_dim=config.block_m,
            gmem_outer_stride=int(a.stride(0)),
            swizzle_mode=swizzle_ab_mode,
        ),
        "tensor_map_b": _encode_tf32_tma_2d_desc(
            tensor=b,
            gmem_inner_dim=config.k,
            gmem_outer_dim=config.n,
            smem_inner_dim=config.block_k,
            smem_outer_dim=config.block_n,
            gmem_outer_stride=int(b.stride(0)),
            swizzle_mode=swizzle_b_mode,
        ),
    }
    if config.num_splits == 1:
        maps["tensor_map_d"] = _encode_tma_2d_desc(
            encode_tensormap=encode_tensormap,
            tensor=d,
            gmem_inner_dim=config.n,
            gmem_outer_dim=config.m,
            smem_inner_dim=config.block_n,
            smem_outer_dim=config.block_m,
            gmem_outer_stride=int(d.stride(-2)),
            swizzle_mode=config.swizzle_cd_mode,
        )
    else:
        maps["tensor_map_d"] = _encode_float32_tma_3d_desc(
            tensor=d,
            gmem_dim_0=config.n,
            gmem_dim_1=config.m,
            gmem_dim_2=config.num_splits,
            smem_dim_0=config.block_n,
            smem_dim_1=config.block_m,
            smem_dim_2=1,
            gmem_stride_0=int(d.stride(-2)),
            gmem_stride_1=int(d.stride(-3)),
            swizzle_mode=config.swizzle_cd_mode,
        )
    return maps


def _run_tirx_with_tensor_maps(
    data: dict[str, Any], executable: Any, tensor_maps: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    config: TF32HCPrenormGemmConfig = data["config"]
    executable(
        config.m,
        tensor_maps["tensor_map_a"].ptr,
        tensor_maps["tensor_map_b"].ptr,
        tensor_maps["tensor_map_d"].ptr,
        data["sqr_tirx"].reshape(-1),
    )
    return data["d_tirx"], data["sqr_tirx"]


def _launch_tirx_hc(data: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    return _run_tirx_with_tensor_maps(
        data, _compile_tirx_tf32_hc(data["config"]), _build_tirx_tensor_maps(data)
    )


def _run_deepgemm_hc(data: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    config: TF32HCPrenormGemmConfig = data["config"]
    data["deep_gemm"].tf32_hc_prenorm_gemm(
        data["a"],
        data["b"],
        data["d_deepgemm"],
        data["sqr_deepgemm"],
        num_splits=None if config.num_splits == 1 else config.num_splits,
    )
    return data["d_deepgemm"], data["sqr_deepgemm"]


def _final_outputs(
    d: torch.Tensor, sqr_sum: torch.Tensor, config: TF32HCPrenormGemmConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    if config.num_splits == 1:
        return d, sqr_sum
    return d.sum(dim=0), sqr_sum.sum(dim=0)


def _calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.double()
    y = y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return float((1 - sim).item())


def _assert_correct(
    data: dict[str, Any], d: torch.Tensor, sqr_sum: torch.Tensor, *, name: str
) -> float:
    config: TF32HCPrenormGemmConfig = data["config"]
    final_d, final_sqr = _final_outputs(d, sqr_sum, config)
    diff = max(
        _calc_diff(final_d, data["reference_d"]), _calc_diff(final_sqr, data["reference_sqr"])
    )
    if diff >= _TEST_DIFF_THRESHOLD:
        raise AssertionError(f"{name} diff {diff:.10g} >= {_TEST_DIFF_THRESHOLD}")
    return diff


def _assert_correct_case(
    case: TF32HCBenchCase, d: torch.Tensor, sqr_sum: torch.Tensor, *, name: str
) -> float:
    final_d, final_sqr = _final_outputs(d, sqr_sum, case.config)
    diff = max(_calc_diff(final_d, case.reference_d), _calc_diff(final_sqr, case.reference_sqr))
    if diff >= _TEST_DIFF_THRESHOLD:
        raise AssertionError(f"{name} diff {diff:.10g} >= {_TEST_DIFF_THRESHOLD}")
    return diff


def run_test(**kwargs: Any) -> None:
    data = prepare_data(**kwargs)
    deepgemm_d, deepgemm_sqr = _run_deepgemm_hc(data)
    torch.cuda.synchronize()
    deepgemm_diff = _assert_correct(data, deepgemm_d, deepgemm_sqr, name="DeepGEMM")
    tirx_d, tirx_sqr = _launch_tirx_hc(data)
    torch.cuda.synchronize()
    tirx_diff = _assert_correct(data, tirx_d, tirx_sqr, name="TIRx")
    if tirx_diff > max(deepgemm_diff, _TEST_DIFF_THRESHOLD):
        raise AssertionError(
            f"TIRx diff {tirx_diff:.10g} is worse than DeepGEMM diff {deepgemm_diff:.10g}"
        )


def _make_bench_case(config_kwargs: dict[str, Any]) -> TF32HCBenchCase:
    data = prepare_data(**config_kwargs)
    return TF32HCBenchCase(
        config=data["config"],
        deep_gemm=data["deep_gemm"],
        a=data["a"],
        b=data["b"],
        d_deepgemm=data["d_deepgemm"],
        sqr_deepgemm=data["sqr_deepgemm"],
        d_tirx=data["d_tirx"],
        sqr_tirx=data["sqr_tirx"],
        reference_d=data["reference_d"],
        reference_sqr=data["reference_sqr"],
        tensor_maps=_build_tirx_tensor_maps(data),
    )


def _bench_case_input_bytes(case: TF32HCBenchCase) -> int:
    return sum(
        tensor.nelement() * tensor.element_size()
        for tensor in (case.a, case.b, case.d_tirx, case.sqr_tirx)
    )


def _bench_tirx_case(case: TF32HCBenchCase, executable: Any) -> tuple[torch.Tensor, torch.Tensor]:
    executable(
        case.config.m,
        case.tensor_maps["tensor_map_a"].ptr,
        case.tensor_maps["tensor_map_b"].ptr,
        case.tensor_maps["tensor_map_d"].ptr,
        case.sqr_tirx.reshape(-1),
    )
    return case.d_tirx, case.sqr_tirx


def _bench_deepgemm_case(case: TF32HCBenchCase) -> tuple[torch.Tensor, torch.Tensor]:
    case.deep_gemm.tf32_hc_prenorm_gemm(
        case.a,
        case.b,
        case.d_deepgemm,
        case.sqr_deepgemm,
        num_splits=None if case.config.num_splits == 1 else case.config.num_splits,
    )
    return case.d_deepgemm, case.sqr_deepgemm


def run_bench(**kwargs: Any) -> dict[str, Any]:
    from tvm.tirx.bench import bench

    timer = kwargs.pop("timer", "proton")
    warmup = kwargs.pop("warmup", 10)
    repeat = kwargs.pop("repeat", 30)
    config_kwargs = dict(kwargs)
    executable: Any | None = None

    def make_input() -> tuple[TF32HCBenchCase, int]:
        nonlocal executable
        case = _make_bench_case(config_kwargs)
        if executable is None:
            executable = _compile_tirx_tf32_hc(case.config)
        return case, _bench_case_input_bytes(case)

    sample_case, _ = make_input()
    tirx_d, tirx_sqr = _bench_tirx_case(sample_case, executable)
    torch.cuda.synchronize()
    tirx_diff = _assert_correct_case(sample_case, tirx_d, tirx_sqr, name="TIRx")
    deepgemm_d, deepgemm_sqr = _bench_deepgemm_case(sample_case)
    torch.cuda.synchronize()
    deepgemm_diff = _assert_correct_case(sample_case, deepgemm_d, deepgemm_sqr, name="DeepGEMM")

    def run_tirx(case: TF32HCBenchCase) -> tuple[torch.Tensor, torch.Tensor]:
        if executable is None:
            raise RuntimeError("TIRx executable was not compiled before benchmarking")
        return _bench_tirx_case(case, executable)

    def _deepgemm():
        return _bench_deepgemm_case

    result = bench(
        {"tirx": run_tirx},
        make_input,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        proton_name="deepgemm_sm100_tf32_hc_prenorm_gemm",
        references={"deepgemm": _deepgemm},
    )

    result["max_diff"] = max(tirx_diff, deepgemm_diff)
    result["tirx_diff"] = tirx_diff
    result["deepgemm_diff"] = deepgemm_diff
    return result


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "DEEPGEMM_TEST_COVERAGE",
    "KERNEL_META",
    "TF32HCPrenormGemmConfig",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
]

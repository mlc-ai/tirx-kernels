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

import os
import re
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Any
from unittest import SkipTest

import torch

_DEEP_GEMM_MODULE_NAME = "deep_gemm"
_SM100_SMEM_CAPACITY = 232448
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
    from tvm.backend.cuda.operator.tile_primitive.tma_utils import SwizzleMode, mma_shared_layout
    from tvm.script import tirx as T
    from tvm.script.tirx import tile as Tx
    from tvm.tirx.lang.pipeline import Pipeline, TCGen05Bar
    from tvm.tirx.layout import S, TCol, TileLayout, TLane, laneid, tcgen05_atom_layout

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
    swizzle_b_mode = min(block_k * 4, 128)
    swizzle_cd_mode = config.swizzle_cd_mode
    smem_cd_size = config.smem_cd_size
    smem_a_size_per_stage = config.smem_a_size_per_stage
    smem_b_size_per_stage = config.smem_b_size_per_stage
    # SMEMPool bump-allocates cd | a | b | pipe barriers | tmem_ptr; all data
    # sizes are 1024-multiples so the pooled offsets reproduce the hand layout.
    num_tmem_cols = 256
    block_swizzled_bk = swizzle_b_mode // 4
    num_b_tma_atoms = block_k // block_swizzled_bk
    umma_k = 32 // 4
    d_tmem_start_col = block_k * num_cast_stages
    # Cast-warp per-thread register counts (compile-time Python ints): each of
    # the 128 cast/reduce threads owns ``cast_per_thread`` fp32 A registers in
    # the .16x256b atom; they span 2 Layout-F rows (top/bottom half), cast_pairs
    # packed (row, lane) pairs each.
    cast_per_thread = block_m * block_k // num_cast_and_reduce_threads
    cast_row_w = cast_per_thread // 2  # per-thread elems per Layout-F row
    cast_pairs = cast_row_w // 2  # packed (f32x2) pairs per row
    tmem_layout = TileLayout(S[(128, num_tmem_cols) : (1 @ TLane, 1 @ TCol)])
    num_k_blocks = config.num_k_blocks
    num_k_blocks_per_split = num_k_blocks // num_splits
    remain_k_blocks = num_k_blocks % num_splits

    def cuda_grid_dependency_synchronize():
        T.evaluate(T.ptx.griddepcontrol.wait())

    def fma_sum_of_squares(acc0, acc1, a_flat, row_w, npairs, sc):
        # Plain (non-@T.inline) Python helper: the parser executes this call's
        # Python ``for`` at parse time, unrolling it so each 2-wide pair slice
        # stays a compile-time-static rank-1 region (a TIR loop var or a
        # middle-dim scalar index would make the fma reg path reject the op —
        # non-static extent / "op rank 3 > anchor rank 2"). Fused packed
        # fma.f32x2 accumulate (acc += pair*pair) into the two per-row dual
        # accumulators, structurally mirroring the hand ffma2 sum0/sum1 float2
        # pair (the T.fma f32x2 path emits fma.rz.ftz vs the hand's rn/non-ftz
        # — a sub-ULP sqr difference, well within the kernel's 1e-8 gate). The
        # flat A view is [0:row_w]=row0, [row_w:2*row_w]=row1.
        for p in range(npairs):
            lo0, lo1 = 2 * p, row_w + 2 * p
            sc.fma(acc0, a_flat[lo0 : lo0 + 2], a_flat[lo0 : lo0 + 2], acc0)
            sc.fma(acc1, a_flat[lo1 : lo1 + 2], a_flat[lo1 : lo1 + 2], acc1)

    @T.prim_func
    def sm100_tf32_hc_prenorm_gemm(
        shape_m: T.uint32,
        a: T.Buffer((config.m, config.k), "bfloat16"),
        b: T.Buffer((config.n, config.k), "float32"),
        d: T.Buffer(config.d_shape, "float32"),
        sqr_sum: T.Buffer((config.num_splits * config.m,), "float32"),
    ):
        T.device_entry()
        T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})
        # TIRX_TRANSCRIBE_START sm100_tf32_hc_prenorm_gemm_impl
        warp_idx = T.warp_id([num_warps])
        T.warpgroup_id([num_warps // 4])
        lane_idx = T.lane_id([32])
        lane_u32: T.uint32 = T.cast(lane_idx, "uint32")

        # SMEMPool bump-allocates over a 1024-aligned uint8 arena (matches the
        # hand's __align__(1024) so the 128B-swizzle TMA boxes stay aligned). All
        # data sizes are 1024-multiples so the pooled offsets reproduce the hand
        # layout (cd@0 | a | b | barriers | tmem_ptr) filling config.smem_size.
        smem = T.alloc_buffer([config.smem_size], "uint8", scope="shared.dyn", align=1024)
        pool = T.SMEMPool(ptr=smem.data)
        # Swizzled (block_m, block_n) float32 view of the D-epilogue SMEM staging
        # buffer. The D epilogue stages the register tile into it with a single
        # Tx.copy(smem_cd_mma, d_reg) (swizzle handled by the reg->smem dispatch),
        # then the copy_async(tma) D store reads it back as this 128B-swizzled
        # mma_shared_layout atom.
        smem_cd_mma = pool.alloc(
            (block_m, block_n),
            "float32",
            align=1024,
            layout=mma_shared_layout("float32", SwizzleMode.SWIZZLE_128B_ATOM, (block_m, block_n)),
        )
        # Swizzled (stage, block_m, block_k) view: copy_async(tma) writes it and
        # the cast warp reads it via T.copy -> ldmatrix.x4 into the warpgroup
        # .16x256b register atom (block_k * 2B = 128 B/row = 1 mma_shared_layout
        # 128B atom).
        smem_a_mma = pool.alloc(
            (num_stages, block_m, block_k),
            "bfloat16",
            align=1024,
            layout=mma_shared_layout(
                "bfloat16", SwizzleMode.SWIZZLE_128B_ATOM, (num_stages, block_m, block_k)
            ),
        )
        # (stage, block_n, block_k) 128B-swizzled view: copy_async(tma) writes it
        # and T.gemm_async reads it as a [block_n, block_k] tf32 operand. block_k *
        # 4B = 256 B/row = 2 x 128B atoms (one TMA copy spans both).
        smem_b_mma = pool.alloc(
            (num_stages, block_n, block_k),
            "float32",
            align=1024,
            layout=mma_shared_layout(
                "float32", SwizzleMode.SWIZZLE_128B_ATOM, (num_stages, block_n, block_k)
            ),
        )
        # Three Pipes carve their full/empty mbarriers from the pool (replacing
        # the hand 5-class barrier array + manual init loop); each ctor inits its
        # barriers under a single-leader (thread-0) guard.
        #   smem_pipe : TMA full (expect_tx; cast warps wait) + tcgen05-commit
        #               empty (MMA frees the A/B stage; TMA warp waits).
        #   cast_pipe : 128-thread mbarrier full (cast warps deposit A->TMEM; MMA
        #               waits) + tcgen05-commit empty (MMA frees the TMEM A; cast
        #               warps wait). num_cast_stages deep (was over-allocated to
        #               num_stages by the hand).
        #   tmem_pipe : tcgen05-commit full only (MMA signals D ready; epilogue
        #               waits).
        smem_pipe = Pipeline(
            pool, num_stages, full="tma", empty="tcgen05", init_full=1, init_empty=1
        )
        cast_pipe = Pipeline(
            pool,
            num_cast_stages,
            full="mbar",
            empty="tcgen05",
            init_full=num_cast_and_reduce_threads,
            init_empty=1,
        )
        # One-way "tmem freed" signal (no slot to recycle), so a bare TCGen05Bar
        # rather than a full/empty Pipeline.
        tmem_pipe = TCGen05Bar(pool, 1)
        tmem_pipe.init(1)
        tmem_ptr_in_smem = pool.alloc((1,), "uint32", align=4)
        # TMEMPool owns the single full-256-col tcgen05.alloc (emitted by
        # ``commit()`` in a warp-2 guard) and the matching relinquish+dealloc
        # (``dealloc()`` in a warp-1 guard). ``alloc`` hands back a buffer whose
        # ``allocated_addr`` is the pool's compile-time ``col_start`` (0 for the
        # first/only region), so the TMEM base stays the constant 0 the hand
        # MMA/epilogue used — gemm_async never reloads the base from SMEM (a
        # per-MMA LDS, ~+16% on the latency-bound configs). ``tmem_ptr_in_smem``
        # is the runtime alloc-result slot (written by tcgen05.alloc, otherwise
        # unread since the base is the const 0). sync_after_alloc=False matches
        # the hand kernel, which relies on the cta_sync below.
        tmem_pool = T.TMEMPool(
            None,
            total_cols=num_tmem_cols,
            cta_group=1,
            alloc_warp=2,
            dealloc_warp=1,
            tmem_addr=tmem_ptr_in_smem,
            sync_after_alloc=False,
        )
        _tmem = tmem_pool.alloc((128, num_tmem_cols), "float32", layout=tmem_layout)

        # Pipeline ctors already inited every barrier (thread-0 leader); the fence
        # makes those inits visible before the cta_sync releases the consumers.
        T.ptx.fence.mbarrier_init()
        tmem_pool.commit()  # warp-2-guarded tcgen05.alloc (self-guards via thread_rank)
        T.cuda.cta_sync()

        block_idx: T.uint32 = T.cast(T.cta_id([config.grid_blocks]), "uint32")
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
                        smem_pipe.empty.wait(
                            stage_idx, ((s // T.uint32(num_stages)) & T.uint32(1)) ^ T.uint32(1)
                        )
                        # u32 row/col bases — the copy_async(tma) gmem-layout grouping
                        # now handles unsigned shape extents (no int32 cast needed).
                        m_idx0: T.uint32 = m_block_idx * T.uint32(block_m)
                        k_idx0: T.uint32 = k_offset + s * T.uint32(block_k)
                        # the outer elect_sync narrows to one lane; the copy_async(tma)
                        # runs at the default thread scope the dispatch requires (a
                        # bare op under a manual `if warp==0` emits no lane guard ->
                        # all 32 lanes over-issue the bulk copy). A is loaded
                        # as bf16 (exact in tf32); B as TFLOAT32 so the TMA RN-truncates
                        # fp32->tf32 ON LOAD, matching the tf32 MMA + DeepGEMM (else the
                        # MMA's RZ truncation diverges ~1 tf32 ULP, D off ~5e-4).
                        # block_n=32 > n -> B rows n..block_n are TMA OOB-filled.
                        Tx.copy_async(
                            smem_a_mma[stage_idx],
                            a[m_idx0 : m_idx0 + block_m, k_idx0 : k_idx0 + block_k],
                            dispatch="tma",
                            mbar=smem_pipe.full.ptr_to([stage_idx]),
                            cta_group=1,
                            cache_hint="evict_normal",
                        )
                        Tx.copy_async(
                            smem_b_mma[stage_idx],
                            b[0:block_n, k_idx0 : k_idx0 + block_k],
                            dispatch="tma",
                            mbar=smem_pipe.full.ptr_to([stage_idx]),
                            cta_group=1,
                            cache_hint="evict_normal",
                            tma_dtype="tf32",
                        )
                        smem_pipe.full.arrive(
                            stage_idx,
                            tx_count=T.uint32(smem_a_size_per_stage + smem_b_size_per_stage),
                        )

            if warp_idx == 1:
                for s in T.serial(T.uint32(0), num_total_stages):
                    stage_idx: T.uint32 = s % T.uint32(num_stages)
                    cast_stage_idx: T.uint32 = s % T.uint32(num_cast_stages)
                    cast_pipe.full.wait(
                        cast_stage_idx, (s // T.uint32(num_cast_stages)) & T.uint32(1)
                    )
                    # int32 col base so the A tmem slice extent stays int (a uint32
                    # base makes the extent uint32, which fails gemm_async's A-layout
                    # structural-equality check against the int expected layout).
                    a_col: T.int32 = T.cast(cast_stage_idx * T.uint32(block_k), "int32")
                    # D[block_m, block_n] = (s==0 ? overwrite : +=) A @ B^T, A read from
                    # TMEM (cast-deposited tf32), B from the 128B-swizzled SMEM view. The
                    # tcgen05 dispatch unrolls the umma_k=8 K-tiling internally, so
                    # accum=(s != 0) reproduces the hand scale_c = (s != 0)|(mma_k != 0).
                    Tx.warp.gemm_async(
                        _tmem[0:block_m, d_tmem_start_col : d_tmem_start_col + block_n],
                        _tmem[0:block_m, a_col : a_col + block_k],
                        smem_b_mma[stage_idx],
                        accum=(s != T.uint32(0)),
                        is_AB_tf32=True,
                        dispatch="tcgen05",
                        cta_group=1,
                        # "recompute" encodes each MMA's B descriptor independently
                        # via cvta on the uniform datapath (ULEA), avoiding the
                        # "hoist" 3-op chain (stage_idx*stride -> shift -> add to a
                        # shared descB) that sits on the per-stage critical path
                        # before the first MMA issues. Matches the hand-rolled
                        # shuffle-precompute latency on the latency-bound tail.
                        smem_desc="recompute",
                    )
                    if T.ptx.elect_sync():
                        cast_pipe.empty.arrive(cast_stage_idx)
                        smem_pipe.empty.arrive(stage_idx)
                if T.ptx.elect_sync():
                    tmem_pipe.arrive(0)

            tmem_pipe.wait(0, 0)
            # D epilogue: read the M=64 (Layout-F scattered) tcgen05 accumulator
            # out of TMEM into a warpgroup register tile via the ``.16x256b`` atom
            # (``copy_async(reg, tmem)``), then stage it into the 128B-swizzled
            # ``smem_cd_mma`` view with a single ``Tx.copy(smem, reg)``. The
            # register tile's layout is ``tcgen05_atom_layout``, whose logical
            # (row, col) ↔ (lane, reg) mapping matches the MMA's Layout-F write,
            # so the reg→smem copy is a logical identity that reproduces the
            # hand-rolled ``tcgen05.ld 32x32b`` + ``st.shared.v4`` store
            # bit-exactly (rel D == 0).
            d_reg = T.alloc_buffer(
                (block_m, block_n),
                "float32",
                layout=tcgen05_atom_layout("16x256b", (block_m, block_n), "float32"),
                scope="local",
            )
            Tx.warpgroup.copy_async(
                d_reg, _tmem[0:block_m, d_tmem_start_col : d_tmem_start_col + block_n]
            )
            T.ptx.tcgen05.wait.ld()
            Tx.warpgroup.copy(smem_cd_mma, d_reg)

            T.ptx.fence.proxy_async("shared::cta")
            T.ptx.bar.sync(0, num_mma_threads)
            if warp_idx == 0:
                if T.ptx.elect_sync():
                    # D store: smem_cd_mma (block_m, block_n swizzled) -> gmem D tile.
                    # block_m/block_n may exceed m/n (boundary tile) -> the TMA store
                    # writes only the valid region. elect (lane guard) + default
                    # thread scope (copy_async scope), like the loads.
                    m0: T.uint32 = m_block_idx * T.uint32(block_m)
                    if num_splits == 1:
                        Tx.copy_async(d[m0 : m0 + block_m, 0:block_n], smem_cd_mma, dispatch="tma")
                    else:
                        ks: T.uint32 = k_split_idx
                        Tx.copy_async(
                            d[ks, m0 : m0 + block_m, 0:block_n], smem_cd_mma, dispatch="tma"
                        )
                    T.ptx.cp_async.bulk.commit_group()
            tmem_pool.dealloc()  # warp-1-guarded relinquish + tcgen05.dealloc
        else:
            sub_warp_idx: T.uint32 = T.cast(
                T.cast(warp_idx, "int32") - T.int32(num_mma_warps), "uint32"
            )
            # A cast/deposit register tiles. The bf16 A tile is ldmatrix-loaded
            # (below) into the per-warpgroup ``.16x256b`` tcgen05 register atom —
            # the same Layout-F M=64 distribution the gemm_async A-in-TMEM operand
            # reads — then cast to tf32 (T.cast) and DEPOSITED into TMEM cols
            # [cast_stage*block_k, +block_k) via T.copy_async (reg->tmem
            # tcgen05.st, the A operand gemm_async consumes). ``a_bf16`` is
            # declared with the *fp32* atom layout (one element per 32-bit slot,
            # NOT the dense 2-bf16-per-slot bf16 atom) so a_bf16 and a_fp32 share
            # an identical per-(lane, register) (row, col) mapping — the cast is
            # then a slot-for-slot widen and a_fp32's register order matches both
            # the ldmatrix output AND what the tcgen05.st deposit consumes (rel
            # D == 0). NB the LOAD stays hand ldmatrix: T.copy on this warpgroup
            # atom can't emit ldmatrix (its m8n8 per-warp lane distribution is
            # structurally incompatible with the wid_in_wg+split-laneid atom) and
            # falls to a 16x-scalar-LDS reg path that costs +25% on the latency-
            # bound tail. The square/accumulate runs on the flat ``.local()`` view
            # (per-thread private regs via ``Tx.fill`` / ``Tx.fma``): per-thread
            # fp32 reg index decomposes va = r // 16 (the Layout-F row selector),
            # regs [0,16) feed row m_idx0, regs [16,32) feed row m_idx1 (+8).
            a_bf16 = T.alloc_buffer(
                (block_m, block_k),
                "bfloat16",
                layout=tcgen05_atom_layout("16x256b", (block_m, block_k), "float32"),
                scope="local",
            )
            a_fp32 = T.alloc_buffer(
                (block_m, block_k),
                "float32",
                layout=tcgen05_atom_layout("16x256b", (block_m, block_k), "float32"),
                scope="local",
            )
            # Dual packed sum-of-squares accumulators, mirroring the hand's two
            # float2 (sum0/sum1) accumulators: one 2-wide packed accumulator per
            # Layout-F row this thread owns. ``T.fma`` (packed fma.f32x2)
            # accumulates one packed pair at a time — the FUSED multiply-
            # accumulate is the only no-regression reduce form (playbook: serial
            # T.sum is -15%, un-fused tree -9%).
            sqr0 = T.alloc_local((2,), "float32")
            sqr1 = T.alloc_local((2,), "float32")
            a_flat = a_fp32.local()  # 1D (cast_per_thread,): [0:16]=row0, [16:32]=row1
            # Per-thread private regs: use thread-scope fill/fma (not Tx.warp — warp
            # scope requires a laneid-distributed layout; see elementwise reg dispatch).
            Tx.fill(sqr0, T.float32(0))
            Tx.fill(sqr1, T.float32(0))
            for s in T.serial(T.uint32(0), num_total_stages, unroll=True):
                stage_idx: T.uint32 = s % T.uint32(num_stages)
                cast_stage_idx: T.uint32 = s % T.uint32(num_cast_stages)
                a_col: T.int32 = T.cast(cast_stage_idx * T.uint32(block_k), "int32")
                smem_pipe.full.wait(stage_idx, (s // T.uint32(num_stages)) & T.uint32(1))
                # SMEM->reg A load: T.copy dispatches to ldmatrix.x4 for the
                # warpgroup .16x256b atom (enabled by the ld_stmatrix llvm-slice
                # fix — without it the canon hits "conflicting scopes for thread"
                # and falls to a +25% scalar-LDS path). a_bf16 carries the *fp32*
                # atom layout so the T.cast to a_fp32 is a slot-for-slot widen
                # and the tcgen05.st deposit reads the same (row, col).
                Tx.warpgroup.copy(a_bf16, smem_a_mma[stage_idx])
                cast_pipe.empty.wait(
                    cast_stage_idx, ((s // T.uint32(num_cast_stages)) & T.uint32(1)) ^ T.uint32(1)
                )
                # bf16 -> tf32 with the atom layouts intact so the dest is
                # written in the deposit's native PTX-register order.
                Tx.warpgroup.cast(a_fp32, a_bf16)
                # Fused multiply-accumulate sum-of-squares: sqr{0,1} += a*a
                # per packed pair (8 fma.f32x2 per row/stage, structurally
                # matching the hand ffma2 dual accumulator). Per-row, order-
                # insensitive.
                fma_sum_of_squares(sqr0, sqr1, a_flat, cast_row_w, cast_pairs, Tx)
                Tx.warpgroup.copy_async(_tmem[0:block_m, a_col : a_col + block_k], a_fp32)
                T.ptx.tcgen05.wait.st()
                cast_pipe.full.arrive(cast_stage_idx)

            # Cross-lane sum-of-squares reduce: each Layout-F row is split across
            # the 4 K-lanes (laneid bits 0..1). Stage the per-thread col-reduced
            # partials (sqr{0,1}[0]+[1]) into a 2-row tile whose warp view is
            # (16, 4) = (8 lane//4 x 2 rows, 4 lane%4); Tx.sum(thread_reduce)
            # reduces the lane%4 axis in place (the shfl_xor 2,1 the hand did).
            sqr_part = T.alloc_buffer(
                [2], "float32", scope="local", layout=TileLayout(S[(2,) : (1,)])
            )
            sqr_part[0] = sqr0[0] + sqr0[1]
            sqr_part[1] = sqr1[0] + sqr1[1]
            sqr_warp = sqr_part.view(
                16,
                4,
                layout=TileLayout(S[(1, 1) : (1, 1)])
                .tile(TileLayout(S[(8, 4) : (4 @ laneid, 1 @ laneid)]), (8, 4), (1, 1))
                .tile(TileLayout(S[(2, 1) : (1, 1)]), (2, 1), (8, 4)),
            )
            Tx.warp.sum(sqr_warp, sqr_warp, thread_reduce=True)
            reduced0: T.float32 = sqr_part[0]
            reduced1: T.float32 = sqr_part[1]
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


def _build_tirx_tensor_maps(data: dict[str, Any]) -> dict[str, Any]:
    # A, B and D are all raw gmem buffer params now (copy_async(tma) host-builds
    # every descriptor: A bf16, B TFLOAT32, D fp32 store). No hand tensor maps.
    return {}


def _run_tirx_with_tensor_maps(
    data: dict[str, Any], executable: Any, tensor_maps: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    config: TF32HCPrenormGemmConfig = data["config"]
    executable(config.m, data["a"], data["b"], data["d_tirx"], data["sqr_tirx"].reshape(-1))
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
    executable(case.config.m, case.a, case.b, case.d_tirx, case.sqr_tirx.reshape(-1))
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
    _rounds = kwargs.pop("rounds", 1)
    _round_cooldown_s = kwargs.pop("round_cooldown_s", 1.0)
    config_kwargs = dict(kwargs)
    config = _make_config(**config_kwargs)
    deep_gemm, _ = load_deep_gemm_hc()
    runtime_config = TF32HCPrenormGemmConfig(
        **{
            **asdict(config),
            "num_sms": int(
                getattr(deep_gemm, "get_num_sms", lambda: _get_num_sms(config.num_sms))()
            ),
        }
    )
    executable = _compile_tirx_tf32_hc(runtime_config)
    metrics: dict[str, float] = {}

    def make_input() -> tuple[TF32HCBenchCase, int]:
        case = _make_bench_case(config_kwargs)
        return case, _bench_case_input_bytes(case)

    def validate_case(case: TF32HCBenchCase) -> None:
        tirx_d, tirx_sqr = _bench_tirx_case(case, executable)
        torch.cuda.synchronize()
        metrics["tirx_diff"] = _assert_correct_case(case, tirx_d, tirx_sqr, name="TIRx")
        deepgemm_d, deepgemm_sqr = _bench_deepgemm_case(case)
        torch.cuda.synchronize()
        metrics["deepgemm_diff"] = _assert_correct_case(
            case, deepgemm_d, deepgemm_sqr, name="DeepGEMM"
        )

    def run_tirx(case: TF32HCBenchCase) -> tuple[torch.Tensor, torch.Tensor]:
        return _bench_tirx_case(case, executable)

    result = bench(
        {"deepgemm": _bench_deepgemm_case, "tirx": run_tirx},
        make_input,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=_rounds,
        round_cooldown_s=_round_cooldown_s,
        proton_name="deepgemm_sm100_tf32_hc_prenorm_gemm",
        validate_case=validate_case,
    )

    result["max_diff"] = max(metrics.get("tirx_diff", 0.0), metrics.get("deepgemm_diff", 0.0))
    result.update(metrics)
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

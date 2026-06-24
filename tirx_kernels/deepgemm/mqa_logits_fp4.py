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

import ctypes
from dataclasses import asdict, dataclass
from functools import cache
from typing import Any
from unittest import SkipTest

import torch

from tvm.ir.type import PointerType, PrimType
from tvm.tirx.cuda.op import cuda_func_call

_DEEP_GEMM_MODULE_NAME = "deep_gemm"
_SM100_SMEM_CAPACITY = 232448
_TEST_DIFF_THRESHOLD = 5e-6


@dataclass(frozen=True)
class MQALogitsConfig:
    seq_len: int = 32
    seq_len_kv: int = 256
    num_heads: int = 64
    head_dim: int = 128
    logits_dtype: str = "float32"
    compressed_logits: bool = False
    disable_cp: bool = True
    seed: int = 0
    num_sms: int = 148
    logits_stride_override: int | None = None

    @property
    def block_q(self) -> int:
        return 128 // self.num_heads

    @property
    def block_kv(self) -> int:
        return 256

    @property
    def max_seqlen_k(self) -> int:
        return 0 if not self.compressed_logits else self.seq_len_kv

    @property
    def aligned_seq_len(self) -> int:
        return _align_up(self.seq_len, self.block_q)

    @property
    def logits_stride(self) -> int:
        if self.logits_stride_override is not None:
            return self.logits_stride_override
        if self.compressed_logits:
            return _align_up(self.max_seqlen_k, self.block_kv)
        return _align_up(self.seq_len_kv + self.block_kv, 8)

    def validate(self) -> None:
        if self.num_heads not in (32, 64):
            raise ValueError("num_heads must be 32 or 64")
        if self.head_dim != 128:
            raise ValueError("head_dim must be 128 for the SM100 FP4 MQA logits kernel")
        if 128 % self.num_heads != 0:
            raise ValueError("128 must be divisible by num_heads")
        if self.seq_len <= 0 or self.seq_len_kv <= 0:
            raise ValueError("sequence lengths must be positive")
        if self.logits_dtype not in ("float32", "bfloat16"):
            raise ValueError("logits_dtype must be 'float32' or 'bfloat16'")
        if self.num_sms <= 0:
            raise ValueError("num_sms must be positive")
        if self.logits_stride_override is not None and self.logits_stride_override <= 0:
            raise ValueError("logits_stride_override must be positive when provided")
        if not self.disable_cp and (self.seq_len_kv % self.seq_len != 0 or self.seq_len % 2 != 0):
            raise ValueError(
                "CP-style schedule generation requires seq_len_kv % seq_len == 0 and even seq_len"
            )


def _make_config(**kwargs: Any) -> MQALogitsConfig:
    kwargs = {key: value for key, value in kwargs.items() if key != "label"}
    config = MQALogitsConfig(**kwargs)
    config.validate()
    return config


def _align_up(x: int, y: int) -> int:
    return (x + y - 1) // y * y


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _torch_logits_dtype(dtype: str) -> torch.dtype:
    if dtype == "float32":
        return torch.float32
    if dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported logits_dtype: {dtype}")


def _config_label(config: dict[str, Any]) -> str:
    dtype = "f32" if config["logits_dtype"] == "float32" else "bf16"
    mode = "compressed" if config["compressed_logits"] else "dense"
    cp = "nocp" if config["disable_cp"] else "cp"
    return (
        f"s{config['seq_len']}_skv{config['seq_len_kv']}_"
        f"h{config['num_heads']}_d{config['head_dim']}_{dtype}_{mode}_{cp}"
    )


def _make_case(
    *,
    seq_len: int,
    seq_len_kv: int,
    logits_dtype: str,
    compressed_logits: bool,
    disable_cp: bool,
    seed: int,
) -> dict[str, Any]:
    config = {
        "seq_len": seq_len,
        "seq_len_kv": seq_len_kv,
        "num_heads": 64,
        "head_dim": 128,
        "logits_dtype": logits_dtype,
        "compressed_logits": compressed_logits,
        "disable_cp": disable_cp,
        "seed": seed,
    }
    config["label"] = _config_label(config)
    return config


KERNEL_META = {
    "name": "deepgemm_sm100_fp4_mqa_logits",
    "category": "deepgemm",
    "compute_capability": 10,
}

DEEPGEMM_TEST_COVERAGE = [
    _make_case(
        seq_len=seq_len,
        seq_len_kv=seq_len_kv,
        logits_dtype=logits_dtype,
        compressed_logits=compressed_logits,
        disable_cp=disable_cp,
        seed=1000 + seed,
    )
    for seed, (logits_dtype, compressed_logits, seq_len, seq_len_kv, disable_cp) in enumerate(
        (logits_dtype, compressed_logits, seq_len, seq_len_kv, disable_cp)
        for logits_dtype in ("float32", "bfloat16")
        for compressed_logits in (False, True)
        for seq_len in (2048, 4096)
        for seq_len_kv in (4096, 8192)
        for disable_cp in (False, True)
    )
]

CONFIGS = [
    _make_case(
        seq_len=32,
        seq_len_kv=256,
        logits_dtype="float32",
        compressed_logits=False,
        disable_cp=True,
        seed=0,
    ),
    _make_case(
        seq_len=32,
        seq_len_kv=256,
        logits_dtype="bfloat16",
        compressed_logits=True,
        disable_cp=True,
        seed=1,
    ),
    _make_case(
        seq_len=32,
        seq_len_kv=256,
        logits_dtype="float32",
        compressed_logits=True,
        disable_cp=False,
        seed=2,
    ),
    DEEPGEMM_TEST_COVERAGE[0],
]

BENCH_CONFIGS = DEEPGEMM_TEST_COVERAGE


def load_deep_gemm_mqa() -> tuple[Any, str]:
    try:
        import deep_gemm as module
    except Exception as exc:
        raise SkipTest(
            f"DeepGEMM MQA logits runtime unavailable: {_DEEP_GEMM_MODULE_NAME}: {exc}"
        ) from exc

    if not hasattr(module, "fp8_fp4_mqa_logits"):
        raise SkipTest("DeepGEMM MQA logits runtime unavailable: missing fp8_fp4_mqa_logits")
    return module, "installed"


def _generate_ks_ke(config: MQALogitsConfig) -> tuple[torch.Tensor, torch.Tensor]:
    if config.disable_cp:
        ks = torch.zeros(config.seq_len, dtype=torch.int32, device="cuda")
        ke = torch.arange(config.seq_len, dtype=torch.int32, device="cuda")
        ke = ke + (config.seq_len_kv - config.seq_len)
        return ks, ke

    chunk_size = config.seq_len // 2
    cp_size = config.seq_len_kv // config.seq_len
    cp_id = cp_size // 3
    ks = torch.zeros(config.seq_len, dtype=torch.int32, device="cuda")
    ke = torch.zeros(config.seq_len, dtype=torch.int32, device="cuda")
    for i in range(chunk_size):
        ke[i] = cp_id * chunk_size + i
        ke[i + chunk_size] = (cp_size * 2 - 1 - cp_id) * chunk_size + i
    return ks, ke


def _ref_mqa_logits(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    cu_seq_len_k_start: torch.Tensor,
    cu_seq_len_k_end: torch.Tensor,
) -> torch.Tensor:
    seq_len_kv = kv.shape[0]
    q_f32 = q.float()
    kv_f32 = kv.float()
    mask_lo = torch.arange(0, seq_len_kv, device="cuda")[None, :] >= cu_seq_len_k_start[:, None]
    mask_hi = torch.arange(0, seq_len_kv, device="cuda")[None, :] < cu_seq_len_k_end[:, None]
    mask = mask_lo & mask_hi
    score = torch.einsum("mhd,nd->hmn", q_f32, kv_f32)
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    return logits.masked_fill(~mask, float("-inf"))


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    deep_gemm, source = load_deep_gemm_mqa()
    config = _make_config(**kwargs)
    if torch.cuda.is_available():
        torch.cuda.set_device(torch.cuda.current_device())
    else:
        raise SkipTest("CUDA is required for SM100 FP4 MQA logits")
    if torch.cuda.get_device_capability()[0] < 10:
        raise SkipTest("SM100 FP4 MQA logits requires compute capability 10.x")

    torch.manual_seed(config.seed)
    q = torch.randn(
        config.seq_len, config.num_heads, config.head_dim, device="cuda", dtype=torch.bfloat16
    )
    kv = torch.randn(config.seq_len_kv, config.head_dim, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(config.seq_len, config.num_heads, device="cuda", dtype=torch.float32)
    ks, ke = _generate_ks_ke(config)

    q_fp4 = deep_gemm.utils.per_token_cast_to_fp4(
        q.view(-1, config.head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    q_in = (
        q_fp4[0].view(config.seq_len, config.num_heads, config.head_dim // 2).contiguous(),
        q_fp4[1].view(config.seq_len, config.num_heads).contiguous(),
    )
    kv_fp4 = deep_gemm.utils.per_token_cast_to_fp4(
        kv.view(-1, config.head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    kv_in = (
        kv_fp4[0].view(config.seq_len_kv, config.head_dim // 2).contiguous(),
        kv_fp4[1].view(config.seq_len_kv).contiguous(),
    )

    q_simulated = deep_gemm.utils.cast_back_from_fp4(
        q_fp4[0], q_fp4[1], gran_k=32, use_packed_ue8m0=True
    ).view(config.seq_len, config.num_heads, config.head_dim)
    kv_simulated = deep_gemm.utils.cast_back_from_fp4(
        kv_fp4[0], kv_fp4[1], gran_k=32, use_packed_ue8m0=True
    ).view(config.seq_len_kv, config.head_dim)
    reference = _ref_mqa_logits(
        q_simulated.to(torch.bfloat16), kv_simulated.to(torch.bfloat16), weights, ks, ke
    )
    max_seqlen_k = int((ke - ks).max().item()) if config.compressed_logits else 0
    runtime_config = MQALogitsConfig(
        **{
            **asdict(config),
            "num_sms": int(getattr(deep_gemm, "get_num_sms", lambda: config.num_sms)()),
        }
    )
    return {
        "config": runtime_config,
        "reference_source": source,
        "q": q,
        "kv": kv,
        "q_in": q_in,
        "kv_in": kv_in,
        "weights": weights,
        "cu_seq_len_k_start": ks,
        "cu_seq_len_k_end": ke,
        "max_seqlen_k": max_seqlen_k,
        "reference": reference,
        "deep_gemm": deep_gemm,
    }


def _mqa_fp4_wrelu_reduce_src(num_heads: int) -> str:
    """Inline CUDA for the weighted-ReLU reduce over heads: sum_h relu(accum[h]) * weights[h].

    Native float2 packed-FMA (make_float2 / fmaxf / __ffma2_rn / __fadd2_rn) in two
    accumulators — matches the DeepGEMM MQA-logits epilogue and avoids the uint64-reinterpret
    PRMT / IMAD-heavy path of the make_float2/float2_x/y/fadd2 intrinsics. `__forceinline__`
    + pointers to the register-resident `accum`/`weights` locals lets SROA keep them in
    registers (no spill). Emitted as a kernel-local helper via `T.cuda_func_call` (deduped by
    name) rather than a shared tir intrinsic — it is specific to this kernel's epilogue.
    """
    return (
        f"__forceinline__ __device__ float tvm_builtin_mqa_fp4_wrelu_reduce_{num_heads}("
        "const float* __restrict__ accum, const float* __restrict__ weights) {\n"
        "    float2 s0 = make_float2(0.0f, 0.0f);\n"
        "    float2 s1 = make_float2(0.0f, 0.0f);\n"
        "    #pragma unroll\n"
        f"    for (int j = 0; j < {num_heads}; j += 4) {{\n"
        "        s0 = __ffma2_rn(make_float2(fmaxf(accum[j], 0.0f), fmaxf(accum[j + 1], 0.0f)),"
        " make_float2(weights[j], weights[j + 1]), s0);\n"
        "        s1 = __ffma2_rn(make_float2(fmaxf(accum[j + 2], 0.0f), fmaxf(accum[j + 3], 0.0f)),"
        " make_float2(weights[j + 2], weights[j + 3]), s1);\n"
        "    }\n"
        "    float2 sv = __fadd2_rn(s0, s1);\n"
        "    return sv.x + sv.y;\n"
        "}"
    )


def get_kernel(**kwargs: Any):
    from tvm.script import tirx as T
    from tvm.tirx.cuda.operator.tile_primitive.gemm_async.tcgen05 import sf_tmem_layout
    from tvm.tirx.layout import S, TCol, TileLayout, TLane

    config = _make_config(**kwargs)
    num_heads = config.num_heads
    head_dim = config.head_dim
    block_q = config.block_q
    block_kv = config.block_kv
    num_q_stages = 3
    num_kv_stages = 6
    num_tmem_stages = 3
    num_specialized_threads = 128
    num_math_threads = 256
    num_math_warpgroups = num_math_threads // 128
    num_threads = num_specialized_threads + num_math_threads
    num_warps = num_threads // 32
    spec_warp_start = num_math_warpgroups * 4
    num_utccp_aligned_elems = 128
    umma_m = 128
    umma_n = block_q * num_heads
    umma_k = 64
    num_sfq = _align_up(block_q * num_heads, num_utccp_aligned_elems)
    num_sfkv = _align_up(block_kv, num_utccp_aligned_elems)
    real_num_sfq = block_q * num_heads
    swizzle_alignment = 8 * (head_dim // 2)
    smem_q_size_per_stage = block_q * num_heads * (head_dim // 2)
    smem_sf_q_size_per_stage = num_sfq * 4
    smem_kv_size_per_stage = block_kv * (head_dim // 2)
    smem_sf_kv_size_per_stage = num_sfkv * 4
    smem_weight_size_per_stage = block_q * num_heads * 4
    smem_q_offset = 0
    smem_kv_offset = smem_q_offset + num_q_stages * smem_q_size_per_stage
    smem_sf_offset = smem_kv_offset + num_kv_stages * smem_kv_size_per_stage
    smem_sf_q_offset = smem_sf_offset
    smem_sf_kv_offset = smem_sf_q_offset + num_q_stages * smem_sf_q_size_per_stage
    smem_weights_offset = smem_sf_kv_offset + num_kv_stages * smem_sf_kv_size_per_stage
    smem_barrier_offset = smem_weights_offset + num_q_stages * smem_weight_size_per_stage
    num_total_barriers = (num_q_stages + num_kv_stages + num_tmem_stages) * 2
    full_q_barrier_base = 0
    empty_q_barrier_base = full_q_barrier_base + num_q_stages
    full_kv_barrier_base = empty_q_barrier_base + num_q_stages
    empty_kv_barrier_base = full_kv_barrier_base + num_kv_stages
    full_tmem_barrier_base = empty_kv_barrier_base + num_kv_stages
    empty_tmem_barrier_base = full_tmem_barrier_base + num_tmem_stages
    smem_tmem_ptr_offset = smem_barrier_offset + num_total_barriers * 8
    smem_total_bytes = smem_tmem_ptr_offset + 4
    if smem_total_bytes > _SM100_SMEM_CAPACITY:
        raise ValueError(f"dynamic shared memory {smem_total_bytes} exceeds SM100 capacity")
    num_accum_tmem_cols = block_q * num_heads * num_tmem_stages
    num_sfa_tmem_cols = num_sfq // 32
    num_sfb_tmem_cols = num_sfkv // 32
    num_tmem_cols = 32
    if num_accum_tmem_cols + num_sfa_tmem_cols + num_sfb_tmem_cols > 32:
        num_tmem_cols = 64
    if num_accum_tmem_cols + num_sfa_tmem_cols + num_sfb_tmem_cols > 64:
        num_tmem_cols = 128
    if num_accum_tmem_cols + num_sfa_tmem_cols + num_sfb_tmem_cols > 128:
        num_tmem_cols = 256
    if num_accum_tmem_cols + num_sfa_tmem_cols + num_sfb_tmem_cols > 256:
        num_tmem_cols = 512
    tmem_start_col_of_sfq = num_accum_tmem_cols
    tmem_start_col_of_sfkv = num_accum_tmem_cols + num_sfa_tmem_cols
    sf_tmem_q_layout = sf_tmem_layout(128, SF_K=num_sfq // 32, sf_per_mma=num_sfq // 32)
    sf_tmem_kv_layout = sf_tmem_layout(128, SF_K=num_sfkv // 32, sf_per_mma=num_sfkv // 32)
    tmem_layout = TileLayout(S[(128, num_tmem_cols) : (1 @ TLane, 1 @ TCol)])
    desc_sdo = 8 * (head_dim // 2) // 16
    sf_desc_sdo = 8 * 4 * 4 // 16
    logits_stride = config.logits_stride
    logits_cols = logits_stride
    aligned_seq_len = config.aligned_seq_len
    logits_tir_dtype = "float32" if config.logits_dtype == "float32" else "bfloat16"

    def shared_addr_u32(ptr):
        return T.cuda.cvta_generic_to_shared(ptr)

    def replace_smem_desc_addr(desc, smem_ptr):
        start_addr = T.cast(
            T.bitwise_and(T.shift_right(shared_addr_u32(smem_ptr), T.uint32(4)), T.uint32(0x3FFF)),
            "uint64",
        )
        return T.bitwise_or(T.bitwise_and(desc, T.bitwise_not(T.uint64(0x3FFF))), start_addr)

    def make_runtime_instr_desc_with_sf_id(desc, sfa_id, sfb_id):
        runtime_desc = T.bitwise_and(desc, T.uint32(0x9FFFFFCF))
        runtime_desc = T.bitwise_or(
            runtime_desc, T.shift_left(T.cast(sfa_id, "uint32"), T.uint32(29))
        )
        runtime_desc = T.bitwise_or(
            runtime_desc, T.shift_left(T.cast(sfb_id, "uint32"), T.uint32(4))
        )
        return T.shift_left(T.cast(runtime_desc, "uint64"), T.uint64(32))

    def ffma2_rn(a, b, c):
        # ``__ffma2_rn`` emits ``fma.rn.f32x2`` (non-ftz, IEEE-754); verified
        # via nvcc PTX dump on sm_100a.
        out = T.alloc_local((1,), "uint64")
        T.evaluate(T.ptx.fma_f32x2(out.ptr_to([0]), a, b, c, rounding="rn", ftz=False))
        return out[0]

    def cuda_grid_dependency_synchronize():
        T.evaluate(T.ptx.griddepcontrol.wait())

    def mbarrier_init_cta(barrier_ptr, arrive_count):
        T.evaluate(T.ptx.mbarrier.init(barrier_ptr, arrive_count))

    def mbarrier_wait_cta(barrier_ptr, phase):
        T.evaluate(T.ptx.mbarrier.try_wait(barrier_ptr, phase))

    def mbarrier_arrive_cta(barrier_ptr):
        T.evaluate(T.ptx.mbarrier.arrive(barrier_ptr))

    def mbarrier_arrive_expect_tx_cta(barrier_ptr, transaction_bytes):
        T.evaluate(T.ptx.mbarrier.arrive.expect_tx(barrier_ptr, transaction_bytes))

    def mma_mxf4_block32_ss(desc_a, desc_b, tmem_c, scale_c, desc, tmem_sfa, tmem_sfb):
        # The original inline asm packed i_desc into the high 32 bits of `desc`.
        i_desc_hi: T.uint32 = T.cast(desc >> T.uint64(32), "uint32")
        T.evaluate(
            T.ptx.tcgen05.mma.block_scale(
                tmem_c,
                desc_a,
                desc_b,
                tmem_sfa,
                tmem_sfb,
                i_desc_hi,
                d_dtype="float32",
                a_dtype="e2m1",
                b_dtype="e2m1",
                sfa_dtype="ue8m0",
                sfb_dtype="ue8m0",
                use_a_tmem=False,
                cta_group=1,
                enable_input_d=scale_c,
            )
        )

    @T.prim_func
    def sm100_fp4_mqa_logits(
        seq_len: T.uint32,
        seq_len_kv: T.uint32,
        max_seqlen_k: T.uint32,
        logits_stride: T.uint32,
        cu_seq_len_k_start: T.Buffer((config.seq_len,), "int32"),
        cu_seq_len_k_end: T.Buffer((config.seq_len,), "int32"),
        logits: T.Buffer((aligned_seq_len, logits_cols), config.logits_dtype),
        tensor_map_q: T.TensorMap(),
        tensor_map_sf_q: T.TensorMap(),
        tensor_map_kv: T.TensorMap(),
        tensor_map_sf_kv: T.TensorMap(),
        tensor_map_weights: T.TensorMap(),
    ):
        T.device_entry()
        # TIRX_TRANSCRIBE_START sm100_fp4_mqa_logits
        logits_flat = T.decl_buffer(
            (aligned_seq_len * logits_cols,), logits_tir_dtype, data=logits.data, scope="global"
        )
        sm_idx = T.cta_id([config.num_sms])
        thread_idx = T.thread_id([num_threads])
        warp_idx = T.warp_id([num_warps])
        warpgroup_idx = T.warpgroup_id([num_warps // 4])
        lane_idx = T.lane_id([32])

        if warp_idx == spec_warp_start:
            T.evaluate(T.ptx.prefetch_tensormap(T.address_of(tensor_map_q)))
            T.evaluate(T.ptx.prefetch_tensormap(T.address_of(tensor_map_sf_q)))
            T.evaluate(T.ptx.prefetch_tensormap(T.address_of(tensor_map_weights)))
            T.evaluate(T.ptx.prefetch_tensormap(T.address_of(tensor_map_kv)))
            T.evaluate(T.ptx.prefetch_tensormap(T.address_of(tensor_map_sf_kv)))

        smem = T.alloc_buffer(
            [smem_total_bytes], "uint8", scope="shared.dyn", align=swizzle_alignment
        )
        smem_q_data: T.let[T.Var(name="smem_q_data", dtype=PointerType(PrimType("uint8")))] = (
            T.reinterpret("handle", smem.ptr_to([smem_q_offset]))
        )
        smem_kv_data: T.let[T.Var(name="smem_kv_data", dtype=PointerType(PrimType("uint8")))] = (
            T.reinterpret("handle", smem.ptr_to([smem_kv_offset]))
        )
        smem_sf_q_data: T.let[
            T.Var(name="smem_sf_q_data", dtype=PointerType(PrimType("uint32")))
        ] = T.reinterpret("handle", smem.ptr_to([smem_sf_q_offset]))
        smem_sf_kv_data: T.let[
            T.Var(name="smem_sf_kv_data", dtype=PointerType(PrimType("uint32")))
        ] = T.reinterpret("handle", smem.ptr_to([smem_sf_kv_offset]))
        smem_weights_data: T.let[
            T.Var(name="smem_weights_data", dtype=PointerType(PrimType("float32")))
        ] = T.reinterpret("handle", smem.ptr_to([smem_weights_offset]))
        smem_barrier_data: T.let[
            T.Var(name="smem_barrier_data", dtype=PointerType(PrimType("uint64")))
        ] = T.reinterpret("handle", smem.ptr_to([smem_barrier_offset]))
        smem_tmem_ptr_data: T.let[
            T.Var(name="smem_tmem_ptr_data", dtype=PointerType(PrimType("uint32")))
        ] = T.reinterpret("handle", smem.ptr_to([smem_tmem_ptr_offset]))
        smem_q = T.decl_buffer(
            (num_q_stages, block_q * num_heads, head_dim // 2),
            "uint8",
            data=smem_q_data,
            scope="shared.dyn",
            elem_offset=0,
            align=swizzle_alignment,
        )
        smem_kv = T.decl_buffer(
            (num_kv_stages, block_kv, head_dim // 2),
            "uint8",
            data=smem_kv_data,
            scope="shared.dyn",
            elem_offset=0,
            align=swizzle_alignment,
        )
        smem_sf_q = T.decl_buffer(
            (num_q_stages, num_sfq),
            "uint32",
            data=smem_sf_q_data,
            scope="shared.dyn",
            elem_offset=0,
            align=16,
        )
        smem_sf_kv = T.decl_buffer(
            (num_kv_stages, num_sfkv),
            "uint32",
            data=smem_sf_kv_data,
            scope="shared.dyn",
            elem_offset=0,
            align=16,
        )
        smem_weights = T.decl_buffer(
            (num_q_stages, block_q, num_heads),
            "float32",
            data=smem_weights_data,
            scope="shared.dyn",
            elem_offset=0,
            align=16,
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
        tmem = T.decl_buffer(
            (128, num_tmem_cols),
            "float32",
            scope="tmem",
            allocated_addr=tmem_ptr_in_smem[0],
            layout=tmem_layout,
        )
        sfq_tmem = T.decl_buffer(
            (128, num_sfq // 32),
            "float8_e8m0fnu",
            scope="tmem",
            allocated_addr=tmem_start_col_of_sfq,
            layout=sf_tmem_q_layout,
        )
        sfkv_tmem = T.decl_buffer(
            (128, num_sfkv // 32),
            "float8_e8m0fnu",
            scope="tmem",
            allocated_addr=tmem_start_col_of_sfkv,
            layout=sf_tmem_kv_layout,
        )
        seq_k_start = T.alloc_local((block_q,), "uint32")
        seq_k_end = T.alloc_local((block_q,), "uint32")
        schedule_result = T.alloc_local((2,), "uint32")

        @T.inline
        def mbarrier_wait_phase(barrier_ptr, phase):
            mbarrier_wait_cta(barrier_ptr, phase)

        @T.inline
        def mbarrier_arrive(barrier_ptr):
            mbarrier_arrive_cta(barrier_ptr)

        @T.inline
        def mbarrier_arrive_and_expect_tx(barrier_ptr, num_bytes):
            mbarrier_arrive_expect_tx_cta(barrier_ptr, num_bytes)

        @T.inline
        def tma_load_2d(dst, barrier_ptr, tensor_map, coord0, coord1):
            T.evaluate(
                T.ptx.cp_async.bulk.tensor.g2c(
                    2,
                    dst,
                    barrier_ptr,
                    T.address_of(tensor_map),
                    0,
                    1,
                    "evict_normal",
                    coord0,
                    coord1,
                )
            )

        @T.inline
        def make_sf_desc(desc_sf, smem_ptr):
            T.ptx.tcgen05.encode_matrix_descriptor(
                T.address_of(desc_sf), smem_ptr, ldo=0, sdo=sf_desc_sdo, swizzle=0
            )

        @T.inline
        def make_smem_desc(desc, smem_ptr):
            T.ptx.tcgen05.encode_matrix_descriptor(
                T.address_of(desc), smem_ptr, ldo=0, sdo=desc_sdo, swizzle=2
            )

        @T.inline
        def utccp_required_smem_warp_transpose(buf1d, base_offset):
            # buf1d: 1D uint32 view of the active SF SMEM stage.
            # Transpose the 32x4 region starting at buf1d[base_offset].
            values = T.alloc_local((4,), "uint32")
            for i in T.unroll(0, 4):
                i_u32 = T.uint32(i)
                col = T.bitwise_xor(i_u32, lane_idx >> T.uint32(3)) * T.uint32(32) + lane_idx
                values[i] = T.ptx.ld(
                    buf1d.ptr_to([T.cast(base_offset + col, "int32")]),
                    "uint32",
                    "u32",
                    space="shared",
                )
            T.cuda.warp_sync()
            for i in T.unroll(0, 4):
                i_u32 = T.uint32(i)
                col = lane_idx * T.uint32(4) + T.bitwise_xor(i_u32, lane_idx >> T.uint32(3))
                T.evaluate(
                    T.ptx.st(
                        buf1d.ptr_to([T.cast(base_offset + col, "int32")]),
                        values[i],
                        space="shared",
                        ptx_type="u32",
                    )
                )

        @T.inline
        def store_logits(flat_offset, value):
            if config.logits_dtype == "float32":
                T.ptx.st(logits_flat.ptr_to([flat_offset]), value, space="global", ptx_type="f32")
            else:
                logits_flat[flat_offset] = value

        @T.inline
        def load_schedule(q_idx):
            schedule_start: T.uint32 = T.uint32(0xFFFFFFFF)
            schedule_end: T.uint32 = T.uint32(0)
            for schedule_i in T.unroll(0, block_q):
                row_idx: T.uint32 = T.min(
                    q_idx * T.uint32(block_q) + T.uint32(schedule_i), seq_len - T.uint32(1)
                )
                row_idx_i32: T.int32 = T.cast(row_idx, "int32")
                seq_k_start[schedule_i] = T.min(
                    T.cast(cu_seq_len_k_start[row_idx_i32], "uint32"), seq_len_kv
                )
                seq_k_end[schedule_i] = T.min(
                    T.cast(cu_seq_len_k_end[row_idx_i32], "uint32"), seq_len_kv
                )
                schedule_start = T.min(schedule_start, seq_k_start[schedule_i])
                schedule_end = T.max(schedule_end, seq_k_end[schedule_i])
            schedule_start = schedule_start // T.uint32(4) * T.uint32(4)
            num_kv_blocks = (schedule_end - schedule_start + T.uint32(block_kv - 1)) // T.uint32(
                block_kv
            )
            schedule_result[0] = schedule_start
            schedule_result[1] = num_kv_blocks

        if warp_idx == spec_warp_start + 1:
            if T.ptx.elect_sync():
                for init_i in T.unroll(0, num_q_stages):
                    mbarrier_init_cta(
                        smem_barriers.ptr_to([full_q_barrier_base + init_i]), T.uint32(1)
                    )
                    mbarrier_init_cta(
                        smem_barriers.ptr_to([empty_q_barrier_base + init_i]),
                        T.uint32(num_math_threads + 32),
                    )
                for init_i in T.unroll(0, num_kv_stages):
                    mbarrier_init_cta(
                        smem_barriers.ptr_to([full_kv_barrier_base + init_i]), T.uint32(1)
                    )
                    mbarrier_init_cta(
                        smem_barriers.ptr_to([empty_kv_barrier_base + init_i]), T.uint32(1)
                    )
                for init_i in T.unroll(0, num_tmem_stages):
                    mbarrier_init_cta(
                        smem_barriers.ptr_to([full_tmem_barrier_base + init_i]), T.uint32(1)
                    )
                    mbarrier_init_cta(
                        smem_barriers.ptr_to([empty_tmem_barrier_base + init_i]), T.uint32(128)
                    )
                T.ptx.fence.mbarrier_init()

        if warp_idx == spec_warp_start + 2:
            T.ptx.tcgen05.alloc(
                T.address_of(tmem_ptr_in_smem[0]), n_cols=num_tmem_cols, cta_group=1
            )
        T.cuda.cta_sync()

        num_q_blocks: T.uint32 = (seq_len + T.uint32(block_q - 1)) // T.uint32(block_q)
        cuda_grid_dependency_synchronize()

        if warp_idx == spec_warp_start:
            T.ptx.setmaxnreg(False, 56)
            if T.ptx.elect_sync():
                q_iter_idx: T.uint32 = T.uint32(0)
                q_idx: T.uint32 = sm_idx
                while q_idx < num_q_blocks:
                    q_stage_idx: T.uint32 = q_iter_idx % T.uint32(num_q_stages)
                    q_phase: T.uint32 = (q_iter_idx // T.uint32(num_q_stages)) & T.uint32(1)
                    q_iter_idx = q_iter_idx + T.uint32(1)
                    mbarrier_wait_phase(
                        smem_barriers.ptr_to([empty_q_barrier_base + q_stage_idx]),
                        q_phase ^ T.uint32(1),
                    )
                    tma_load_2d(
                        smem_q.ptr_to([q_stage_idx, 0, 0]),
                        smem_barriers.ptr_to([full_q_barrier_base + q_stage_idx]),
                        tensor_map_q,
                        T.uint32(0),
                        q_idx * T.uint32(block_q * num_heads),
                    )
                    tma_load_2d(
                        smem_sf_q.ptr_to([q_stage_idx, 0]),
                        smem_barriers.ptr_to([full_q_barrier_base + q_stage_idx]),
                        tensor_map_sf_q,
                        T.uint32(0),
                        q_idx * T.uint32(block_q),
                    )
                    tma_load_2d(
                        smem_weights.ptr_to([q_stage_idx, 0, 0]),
                        smem_barriers.ptr_to([full_q_barrier_base + q_stage_idx]),
                        tensor_map_weights,
                        T.uint32(0),
                        q_idx * T.uint32(block_q),
                    )
                    mbarrier_arrive_and_expect_tx(
                        smem_barriers.ptr_to([full_q_barrier_base + q_stage_idx]),
                        smem_q_size_per_stage + real_num_sfq * 4 + smem_weight_size_per_stage,
                    )
                    q_idx = q_idx + T.uint32(config.num_sms)
            T.cuda.warp_sync()
        elif warp_idx == spec_warp_start + 1:
            T.ptx.setmaxnreg(False, 56)
            if T.ptx.elect_sync():
                kv_iter_idx: T.uint32 = T.uint32(0)
                q_idx: T.uint32 = sm_idx
                while q_idx < num_q_blocks:
                    load_schedule(q_idx)
                    kv_start: T.uint32 = schedule_result[0]
                    num_kv_blocks: T.uint32 = schedule_result[1]
                    kv_idx: T.uint32 = T.uint32(0)
                    while kv_idx < num_kv_blocks:
                        kv_stage_idx: T.uint32 = kv_iter_idx % T.uint32(num_kv_stages)
                        kv_phase: T.uint32 = (kv_iter_idx // T.uint32(num_kv_stages)) & T.uint32(1)
                        kv_iter_idx = kv_iter_idx + T.uint32(1)
                        mbarrier_wait_phase(
                            smem_barriers.ptr_to([empty_kv_barrier_base + kv_stage_idx]),
                            kv_phase ^ T.uint32(1),
                        )
                        tma_load_2d(
                            smem_kv.ptr_to([kv_stage_idx, 0, 0]),
                            smem_barriers.ptr_to([full_kv_barrier_base + kv_stage_idx]),
                            tensor_map_kv,
                            T.uint32(0),
                            kv_start + kv_idx * T.uint32(block_kv),
                        )
                        tma_load_2d(
                            smem_sf_kv.ptr_to([kv_stage_idx, 0]),
                            smem_barriers.ptr_to([full_kv_barrier_base + kv_stage_idx]),
                            tensor_map_sf_kv,
                            kv_start + kv_idx * T.uint32(block_kv),
                            T.uint32(0),
                        )
                        mbarrier_arrive_and_expect_tx(
                            smem_barriers.ptr_to([full_kv_barrier_base + kv_stage_idx]),
                            smem_kv_size_per_stage + smem_sf_kv_size_per_stage,
                        )
                        kv_idx = kv_idx + T.uint32(1)
                    q_idx = q_idx + T.uint32(config.num_sms)
        elif warp_idx == spec_warp_start + 2:
            T.ptx.setmaxnreg(False, 56)
            T.cuda.trap_when_assert_failed(tmem_ptr_in_smem[0] == T.uint32(0))
            desc_i: T.uint32
            desc_sf: T.uint64
            desc_a: T.uint64
            desc_b: T.uint64
            T.ptx.tcgen05.encode_instr_descriptor_block_scaled(
                T.address_of(desc_i),
                d_dtype="float32",
                a_dtype="float4_e2m1fn",
                b_dtype="float4_e2m1fn",
                sfa_dtype="float8_e8m0fnu",
                sfb_dtype="float8_e8m0fnu",
                sfa_tmem_addr=0,
                sfb_tmem_addr=0,
                M=umma_m,
                N=umma_n,
                K=umma_k,
                trans_a=False,
                trans_b=False,
                n_cta_groups=1,
            )
            make_sf_desc(desc_sf, T.reinterpret("handle", T.uint64(0)))
            q_iter_idx: T.uint32 = T.uint32(0)
            kv_iter_idx: T.uint32 = T.uint32(0)
            tmem_iter_idx: T.uint32 = T.uint32(0)
            q_idx: T.uint32 = sm_idx
            while q_idx < num_q_blocks:
                load_schedule(q_idx)
                kv_start: T.uint32 = schedule_result[0]
                num_kv_blocks: T.uint32 = schedule_result[1]
                q_stage_idx: T.uint32 = q_iter_idx % T.uint32(num_q_stages)
                q_phase: T.uint32 = (q_iter_idx // T.uint32(num_q_stages)) & T.uint32(1)
                q_iter_idx = q_iter_idx + T.uint32(1)
                mbarrier_wait_phase(
                    smem_barriers.ptr_to([full_q_barrier_base + q_stage_idx]), q_phase
                )
                sfq_stage_ptr: T.let[
                    T.Var(name="sfq_stage_ptr", dtype=PointerType(PrimType("uint32")))
                ] = T.ptr_byte_offset(smem_sf_q.data, q_stage_idx * T.uint32(num_sfq * 4), "uint32")
                sfq_stage = T.decl_buffer(
                    (num_sfq,),
                    "uint32",
                    data=sfq_stage_ptr,
                    scope="shared.dyn",
                    elem_offset=0,
                    align=16,
                )
                for sfq_i in T.unroll(0, num_sfq // num_utccp_aligned_elems):
                    sfq_base = T.uint32(sfq_i * num_utccp_aligned_elems)
                    utccp_required_smem_warp_transpose(sfq_stage, sfq_base)
                    T.ptx.fence.proxy_async("shared::cta")
                    desc_sf = replace_smem_desc_addr(desc_sf, sfq_stage.ptr_to([sfq_base]))
                    if T.ptx.elect_sync():
                        T.ptx.tcgen05.cp(
                            tmem_start_col_of_sfq + sfq_i * 4,
                            desc_sf,
                            shape="32x128b",
                            cta_group=1,
                            multicast="warpx4",
                        )
                    T.cuda.warp_sync()
                kv_idx: T.uint32 = T.uint32(0)
                while kv_idx < num_kv_blocks:
                    kv_stage_idx: T.uint32 = kv_iter_idx % T.uint32(num_kv_stages)
                    kv_phase: T.uint32 = (kv_iter_idx // T.uint32(num_kv_stages)) & T.uint32(1)
                    kv_iter_idx = kv_iter_idx + T.uint32(1)
                    mbarrier_wait_phase(
                        smem_barriers.ptr_to([full_kv_barrier_base + kv_stage_idx]), kv_phase
                    )
                    sfkv_stage_ptr: T.let[
                        T.Var(name="sfkv_stage_ptr", dtype=PointerType(PrimType("uint32")))
                    ] = T.ptr_byte_offset(
                        smem_sf_kv.data, kv_stage_idx * T.uint32(num_sfkv * 4), "uint32"
                    )
                    sfkv_stage = T.decl_buffer(
                        (num_sfkv,),
                        "uint32",
                        data=sfkv_stage_ptr,
                        scope="shared.dyn",
                        elem_offset=0,
                        align=16,
                    )
                    for sfkv_i in T.unroll(0, num_sfkv // num_utccp_aligned_elems):
                        sfkv_base = T.uint32(sfkv_i * num_utccp_aligned_elems)
                        utccp_required_smem_warp_transpose(sfkv_stage, sfkv_base)
                        T.ptx.fence.proxy_async("shared::cta")
                    if T.ptx.elect_sync():
                        for sfkv_i in T.unroll(0, num_sfkv // num_utccp_aligned_elems):
                            sfkv_base = T.uint32(sfkv_i * num_utccp_aligned_elems)
                            desc_sf = replace_smem_desc_addr(
                                desc_sf, sfkv_stage.ptr_to([sfkv_base])
                            )
                            T.ptx.tcgen05.cp(
                                tmem_start_col_of_sfkv + sfkv_i * 4,
                                desc_sf,
                                shape="32x128b",
                                cta_group=1,
                                multicast="warpx4",
                            )
                        for math_wg_i in T.unroll(0, num_math_warpgroups):
                            tmem_stage_idx: T.uint32 = tmem_iter_idx % T.uint32(num_tmem_stages)
                            tmem_phase: T.uint32 = (
                                tmem_iter_idx // T.uint32(num_tmem_stages)
                            ) & T.uint32(1)
                            tmem_iter_idx = tmem_iter_idx + T.uint32(1)
                            tmem_addr: T.uint32 = tmem_stage_idx * T.uint32(umma_n)
                            mbarrier_wait_phase(
                                smem_barriers.ptr_to([empty_tmem_barrier_base + tmem_stage_idx]),
                                tmem_phase ^ T.uint32(1),
                            )
                            for k in T.unroll(0, head_dim // umma_k):
                                runtime_desc_i = make_runtime_instr_desc_with_sf_id(
                                    desc_i, k * 2, k * 2
                                )
                                make_smem_desc(
                                    desc_a,
                                    smem_kv.ptr_to(
                                        [kv_stage_idx, math_wg_i * umma_m, k * umma_k // 2]
                                    ),
                                )
                                make_smem_desc(
                                    desc_b, smem_q.ptr_to([q_stage_idx, 0, k * umma_k // 2])
                                )
                                mma_mxf4_block32_ss(
                                    desc_a,
                                    desc_b,
                                    tmem_addr,
                                    k,
                                    runtime_desc_i,
                                    tmem_start_col_of_sfkv + math_wg_i * 4,
                                    tmem_start_col_of_sfq,
                                )
                            T.ptx.tcgen05.commit(
                                smem_barriers.ptr_to([full_tmem_barrier_base + tmem_stage_idx])
                            )
                    if T.ptx.elect_sync():
                        T.ptx.tcgen05.commit(
                            smem_barriers.ptr_to([empty_kv_barrier_base + kv_stage_idx]),
                            cta_group=1,
                        )
                    kv_idx = kv_idx + T.uint32(1)
                mbarrier_arrive(smem_barriers.ptr_to([empty_q_barrier_base + q_stage_idx]))
                q_idx = q_idx + T.uint32(config.num_sms)
        elif warp_idx == spec_warp_start + 3:
            T.ptx.setmaxnreg(False, 56)
        elif warp_idx < spec_warp_start:
            T.ptx.setmaxnreg(True, 224)
            accum = T.alloc_local((num_heads,), "float32")
            cached_weights = T.alloc_local((block_q, num_heads), "float32")
            # Per-q-row logits base offset (= q_row * logits_stride): invariant across the kv
            # loop, so compute once per q block instead of per (kv_block, q_inner_i) store.
            q_row_offsets = T.alloc_local((block_q,), "uint64")
            q_iter_idx: T.uint32 = T.uint32(0)
            tmem_iter_idx: T.uint32 = T.uint32(0)
            tmem_iter_idx = tmem_iter_idx + T.cast(warpgroup_idx, "uint32")
            q_idx: T.uint32 = sm_idx
            while q_idx < num_q_blocks:
                load_schedule(q_idx)
                kv_start: T.uint32 = schedule_result[0]
                num_kv_blocks: T.uint32 = schedule_result[1]
                q_stage_idx: T.uint32 = q_iter_idx % T.uint32(num_q_stages)
                q_phase: T.uint32 = (q_iter_idx // T.uint32(num_q_stages)) & T.uint32(1)
                q_iter_idx = q_iter_idx + T.uint32(1)
                mbarrier_wait_phase(
                    smem_barriers.ptr_to([full_q_barrier_base + q_stage_idx]), q_phase
                )
                for weight_i in T.unroll(0, block_q):
                    for weight_j in T.unroll(0, num_heads):
                        cached_weights[weight_i, weight_j] = T.ptx.ld(
                            smem_weights.ptr_to([q_stage_idx, weight_i, weight_j]),
                            "float32",
                            "f32",
                            space="shared",
                        )
                for q_off_i in T.unroll(0, block_q):
                    q_row_offsets[q_off_i] = T.cast(
                        q_idx * T.uint32(block_q) + T.uint32(q_off_i), "uint64"
                    ) * T.cast(logits_stride, "uint64")
                kv_idx: T.uint32 = T.uint32(0)
                while kv_idx < num_kv_blocks:
                    kv_offset: T.uint32 = (
                        kv_start + kv_idx * T.uint32(block_kv) + T.cast(thread_idx, "uint32")
                    )
                    tmem_stage_idx: T.uint32 = tmem_iter_idx % T.uint32(num_tmem_stages)
                    tmem_phase: T.uint32 = (tmem_iter_idx // T.uint32(num_tmem_stages)) & T.uint32(
                        1
                    )
                    tmem_iter_idx = tmem_iter_idx + T.uint32(num_math_warpgroups)
                    mbarrier_wait_phase(
                        smem_barriers.ptr_to([full_tmem_barrier_base + tmem_stage_idx]), tmem_phase
                    )
                    for q_inner_i in T.unroll(0, block_q):
                        tmem_addr: T.uint32 = tmem_stage_idx * T.uint32(umma_n) + T.uint32(
                            q_inner_i * num_heads
                        )
                        if num_heads == 32:
                            T.ptx.tcgen05.ld(
                                tmem_addr,
                                accum[0],
                                accum[1],
                                accum[2],
                                accum[3],
                                accum[4],
                                accum[5],
                                accum[6],
                                accum[7],
                                accum[8],
                                accum[9],
                                accum[10],
                                accum[11],
                                accum[12],
                                accum[13],
                                accum[14],
                                accum[15],
                                accum[16],
                                accum[17],
                                accum[18],
                                accum[19],
                                accum[20],
                                accum[21],
                                accum[22],
                                accum[23],
                                accum[24],
                                accum[25],
                                accum[26],
                                accum[27],
                                accum[28],
                                accum[29],
                                accum[30],
                                accum[31],
                                shape="32x32b",
                                num=32,
                            )
                        if num_heads == 64:
                            T.ptx.tcgen05.ld(
                                tmem_addr,
                                accum[0],
                                accum[1],
                                accum[2],
                                accum[3],
                                accum[4],
                                accum[5],
                                accum[6],
                                accum[7],
                                accum[8],
                                accum[9],
                                accum[10],
                                accum[11],
                                accum[12],
                                accum[13],
                                accum[14],
                                accum[15],
                                accum[16],
                                accum[17],
                                accum[18],
                                accum[19],
                                accum[20],
                                accum[21],
                                accum[22],
                                accum[23],
                                accum[24],
                                accum[25],
                                accum[26],
                                accum[27],
                                accum[28],
                                accum[29],
                                accum[30],
                                accum[31],
                                accum[32],
                                accum[33],
                                accum[34],
                                accum[35],
                                accum[36],
                                accum[37],
                                accum[38],
                                accum[39],
                                accum[40],
                                accum[41],
                                accum[42],
                                accum[43],
                                accum[44],
                                accum[45],
                                accum[46],
                                accum[47],
                                accum[48],
                                accum[49],
                                accum[50],
                                accum[51],
                                accum[52],
                                accum[53],
                                accum[54],
                                accum[55],
                                accum[56],
                                accum[57],
                                accum[58],
                                accum[59],
                                accum[60],
                                accum[61],
                                accum[62],
                                accum[63],
                                shape="32x32b",
                                num=64,
                            )
                        T.ptx.tcgen05.wait.ld()
                        if q_inner_i == block_q - 1:
                            mbarrier_arrive(
                                smem_barriers.ptr_to([empty_tmem_barrier_base + tmem_stage_idx])
                            )
                        # Native-float2 weighted-ReLU reduce over heads (DeepGEMM epilogue
                        # shape): sum_h relu(accum[h]) * weights[h]. Kernel-local inline CUDA
                        # (see _mqa_fp4_wrelu_reduce_src) emitted via cuda_func_call — packed-FMA
                        # with native float2, no uint64 reinterpret round-trips.
                        result_f32: T.float32 = cuda_func_call(
                            f"tvm_builtin_mqa_fp4_wrelu_reduce_{num_heads}",
                            T.address_of(accum[0]),
                            T.address_of(cached_weights[q_inner_i, 0]),
                            source_code=_mqa_fp4_wrelu_reduce_src(num_heads),
                            return_type="float32",
                        )
                        result = T.cast(result_f32, logits_tir_dtype)
                        q_offset: T.uint64 = q_row_offsets[q_inner_i]
                        if config.compressed_logits:
                            row_k_start: T.uint32 = seq_k_start[q_inner_i]
                            row_k_end: T.uint32 = seq_k_end[q_inner_i]
                            # Range-guarded store. Build the flat index with per-operand u64 casts
                            # `q_offset + (u64)kv - (u64)rks`, NOT `(u64)(kv - rks)`: the latter
                            # makes ptxas branch around the 16-bit bf16 store (`BSYNC` + unpredicated
                            # `STG.E.U16`), while the per-operand form if-converts the guard to a
                            # predicated `@P STG.E.U16` — matching DeepGEMM's epilogue and ~6% faster
                            # on bf16xcompressed (verified in SASS + /tir-bench). A predicated PTX
                            # store (inline asm) reaches the same instruction but is opaque to ptxas
                            # scheduling, so it cannot overlap with the surrounding tcgen05 ops and
                            # loses that 6%. See memory/knowledge/predicated-narrow-global-store.md.
                            if row_k_start <= kv_offset and kv_offset < row_k_end:
                                store_logits(
                                    q_offset
                                    + T.cast(kv_offset, "uint64")
                                    - T.cast(row_k_start, "uint64"),
                                    result,
                                )
                        else:
                            store_logits(q_offset + T.cast(kv_offset, "uint64"), result)
                        T.cuda.warp_sync()
                    kv_idx = kv_idx + T.uint32(1)
                mbarrier_arrive(smem_barriers.ptr_to([empty_q_barrier_base + q_stage_idx]))
                q_idx = q_idx + T.uint32(config.num_sms)
            T.ptx.bar.sync(8, num_math_threads)
            if warp_idx == 0:
                T.ptx.tcgen05.dealloc(T.uint32(0), n_cols=num_tmem_cols, cta_group=1)

    return sm100_fp4_mqa_logits.with_attr(
        "tirx.kernel_launch_params",
        [
            "blockIdx.x",
            "threadIdx.x",
            "tirx.use_programtic_dependent_launch",
            "tirx.use_dyn_shared_memory",
        ],
    )


def _compile_tirx_mqa_for_config(
    *,
    seq_len: int,
    seq_len_kv: int,
    num_heads: int,
    head_dim: int,
    logits_dtype: str,
    compressed_logits: bool,
    disable_cp: bool,
    num_sms: int,
    logits_stride_override: int | None,
) -> Any:
    import tvm

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    kernel = get_kernel(
        seq_len=seq_len,
        seq_len_kv=seq_len_kv,
        num_heads=num_heads,
        head_dim=head_dim,
        logits_dtype=logits_dtype,
        compressed_logits=compressed_logits,
        disable_cp=disable_cp,
        num_sms=num_sms,
        logits_stride_override=logits_stride_override,
    )
    with target:
        mod = tvm.IRModule({"main": kernel})
        return tvm.compile(mod, target=target, tir_pipeline="tirx")


_compile_tirx_mqa_for_config = cache(_compile_tirx_mqa_for_config)


def _compile_tirx_mqa(config: MQALogitsConfig, max_seqlen_k: int) -> Any:
    logits_stride_override = None
    if config.compressed_logits:
        logits_stride_override = _align_up(max_seqlen_k, config.block_kv)
    return _compile_tirx_mqa_for_config(
        seq_len=config.seq_len,
        seq_len_kv=config.seq_len_kv,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        logits_dtype=config.logits_dtype,
        compressed_logits=config.compressed_logits,
        disable_cp=config.disable_cp,
        num_sms=config.num_sms,
        logits_stride_override=logits_stride_override,
    )


def _encode_fp4_packed_smem_tma_2d_desc(
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

    desc = mega_moe._AlignedTensorMap()
    global_shape = (ctypes.c_uint64 * 2)(int(gmem_inner_dim), int(gmem_outer_dim))
    global_strides = (ctypes.c_uint64 * 1)(int(gmem_outer_stride * tensor.element_size()))
    box_dim = (ctypes.c_uint32 * 2)(int(smem_inner_dim), int(smem_outer_dim))
    element_strides = (ctypes.c_uint32 * 2)(1, 1)
    result = mega_moe._get_cuda_driver().cuTensorMapEncodeTiled(
        desc.ptr,
        13,  # CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B for packed FP4 shared memory.
        ctypes.c_uint32(2),
        ctypes.c_void_p(int(tensor.data_ptr())),
        global_shape,
        global_strides,
        box_dim,
        element_strides,
        mega_moe._CUDA_TENSOR_MAP_INTERLEAVE_NONE,
        mega_moe._tensor_map_swizzle_from_mode(swizzle_mode),
        mega_moe._CUDA_TENSOR_MAP_L2_PROMOTION_L2_256B,
        mega_moe._CUDA_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if result != 0:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed for FP4 align8 with CUresult={result}")
    return desc


def _build_tirx_tensor_maps(data: dict[str, Any]) -> dict[str, Any]:
    import tvm
    from tirx_kernels.deepgemm.mega_moe import _encode_tma_2d_desc, _get_tma_aligned_size

    config: MQALogitsConfig = data["config"]
    q_fp4, sf_q = data["q_in"]
    kv_fp4, sf_kv = data["kv_in"]
    weights = data["weights"]
    encode_tensormap = tvm.get_global_func("runtime.cuTensorMapEncodeTiled")

    return {
        "tensor_map_q": _encode_fp4_packed_smem_tma_2d_desc(
            tensor=q_fp4,
            gmem_inner_dim=config.head_dim,
            gmem_outer_dim=config.seq_len * config.num_heads,
            smem_inner_dim=config.head_dim,
            smem_outer_dim=config.block_q * config.num_heads,
            gmem_outer_stride=int(q_fp4.stride(1)),
            swizzle_mode=config.head_dim // 2,
        ),
        "tensor_map_sf_q": _encode_tma_2d_desc(
            encode_tensormap=encode_tensormap,
            tensor=sf_q,
            gmem_inner_dim=config.num_heads,
            gmem_outer_dim=config.seq_len,
            smem_inner_dim=config.num_heads,
            smem_outer_dim=config.block_q,
            gmem_outer_stride=int(sf_q.stride(0)),
            swizzle_mode=0,
        ),
        "tensor_map_weights": _encode_tma_2d_desc(
            encode_tensormap=encode_tensormap,
            tensor=weights,
            gmem_inner_dim=config.num_heads,
            gmem_outer_dim=config.seq_len,
            smem_inner_dim=config.num_heads,
            smem_outer_dim=config.block_q,
            gmem_outer_stride=int(weights.stride(0)),
            swizzle_mode=0,
        ),
        "tensor_map_kv": _encode_fp4_packed_smem_tma_2d_desc(
            tensor=kv_fp4,
            gmem_inner_dim=config.head_dim,
            gmem_outer_dim=config.seq_len_kv,
            smem_inner_dim=config.head_dim,
            smem_outer_dim=config.block_kv,
            gmem_outer_stride=int(kv_fp4.stride(0)),
            swizzle_mode=config.head_dim // 2,
        ),
        "tensor_map_sf_kv": _encode_tma_2d_desc(
            encode_tensormap=encode_tensormap,
            tensor=sf_kv,
            gmem_inner_dim=_get_tma_aligned_size(config.seq_len_kv, int(sf_kv.element_size())),
            gmem_outer_dim=1,
            smem_inner_dim=config.block_kv,
            smem_outer_dim=1,
            gmem_outer_stride=0,
            swizzle_mode=0,
        ),
    }


def _logits_storage_shape(config: MQALogitsConfig, max_seqlen_k: int) -> tuple[int, int]:
    if config.compressed_logits:
        stride = _align_up(max_seqlen_k, config.block_kv)
    else:
        stride = _align_up(config.seq_len_kv + config.block_kv, 8)
    return config.aligned_seq_len, stride


def _allocate_logits(config: MQALogitsConfig, max_seqlen_k: int) -> torch.Tensor:
    storage_shape = _logits_storage_shape(config, max_seqlen_k)
    return torch.full(
        storage_shape, float("-inf"), device="cuda", dtype=_torch_logits_dtype(config.logits_dtype)
    )


def _prepare_global_barrier(executable: Any) -> None:
    try:
        prepare_global_barrier = executable.mod.get_function("__tvm_prepare_global_barrier")
    except AttributeError:
        prepare_global_barrier = None
    if prepare_global_barrier is not None:
        prepare_global_barrier()


def _prepare_tirx_invocation(
    data: dict[str, Any], logits: torch.Tensor | None = None
) -> dict[str, Any]:
    config: MQALogitsConfig = data["config"]
    if logits is None:
        logits = _allocate_logits(config, data["max_seqlen_k"])
    return {
        "executable": _compile_tirx_mqa(config, data["max_seqlen_k"]),
        "logits": logits,
        "tensor_maps": _build_tirx_tensor_maps(data),
    }


def _run_tirx_invocation(data: dict[str, Any], invocation: dict[str, Any]) -> torch.Tensor:
    config: MQALogitsConfig = data["config"]
    executable = invocation["executable"]
    tensor_maps = invocation["tensor_maps"]
    logits = invocation["logits"]
    _prepare_global_barrier(executable)
    executable.mod(
        config.seq_len,
        config.seq_len_kv,
        data["max_seqlen_k"],
        logits.stride(0),
        data["cu_seq_len_k_start"],
        data["cu_seq_len_k_end"],
        logits,
        tensor_maps["tensor_map_q"].ptr,
        tensor_maps["tensor_map_sf_q"].ptr,
        tensor_maps["tensor_map_kv"].ptr,
        tensor_maps["tensor_map_sf_kv"].ptr,
        tensor_maps["tensor_map_weights"].ptr,
    )
    return logits


def _launch_tirx_mqa(data: dict[str, Any], logits: torch.Tensor | None = None) -> torch.Tensor:
    return _run_tirx_invocation(data, _prepare_tirx_invocation(data, logits))


def _run_deepgemm_mqa(data: dict[str, Any], *, clean_logits: bool) -> torch.Tensor:
    config: MQALogitsConfig = data["config"]
    return data["deep_gemm"].fp8_fp4_mqa_logits(
        q=data["q_in"],
        kv=data["kv_in"],
        weights=data["weights"],
        cu_seq_len_k_start=data["cu_seq_len_k_start"],
        cu_seq_len_k_end=data["cu_seq_len_k_end"],
        clean_logits=clean_logits,
        max_seqlen_k=data["max_seqlen_k"],
        logits_dtype=_torch_logits_dtype(config.logits_dtype),
    )


def _expand_compressed_logits(logits: torch.Tensor, data: dict[str, Any]) -> torch.Tensor:
    config: MQALogitsConfig = data["config"]
    if not config.compressed_logits:
        return logits[: config.seq_len, : config.seq_len_kv]

    expanded = torch.full(
        (config.seq_len, config.seq_len_kv), float("-inf"), device="cuda", dtype=logits.dtype
    )
    ks = data["cu_seq_len_k_start"]
    ke = data["cu_seq_len_k_end"]
    for row_idx in range(config.seq_len):
        start = int(ks[row_idx].item())
        end = int(ke[row_idx].item())
        expanded[row_idx, start:end] = logits[row_idx, : end - start]
    return expanded


def _calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.double()
    y = y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return float((1 - sim).item())


def _assert_correct(data: dict[str, Any], logits: torch.Tensor, *, name: str) -> float:
    reference = data["reference"]
    observed = _expand_compressed_logits(logits, data)
    ref_neginf_mask = reference == float("-inf")
    observed = observed.masked_fill(ref_neginf_mask, 0)
    reference = reference.masked_fill(ref_neginf_mask, 0)
    diff = _calc_diff(observed, reference)
    if diff >= _TEST_DIFF_THRESHOLD:
        raise AssertionError(f"{name} simulated diff {diff:.6g} >= {_TEST_DIFF_THRESHOLD}")
    return diff


def run_test(**kwargs: Any) -> None:
    data = prepare_data(**kwargs)
    config: MQALogitsConfig = data["config"]
    clean_logits = not config.compressed_logits
    deepgemm_logits = _run_deepgemm_mqa(data, clean_logits=clean_logits)
    deepgemm_diff = _assert_correct(data, deepgemm_logits, name="DeepGEMM")
    tirx_logits = _launch_tirx_mqa(data)
    torch.cuda.synchronize()
    tirx_diff = _assert_correct(data, tirx_logits, name="TIRx")
    if tirx_diff > max(deepgemm_diff, _TEST_DIFF_THRESHOLD):
        raise AssertionError(
            f"TIRx diff {tirx_diff:.6g} is worse than DeepGEMM diff {deepgemm_diff:.6g}"
        )


def run_bench(**kwargs: Any) -> dict[str, Any]:
    from tvm.tirx.bench import bench, tensor_bytes

    warmup = kwargs.pop("warmup", 10)
    repeat = kwargs.pop("repeat", 30)
    timer = kwargs.pop("timer", "proton")
    _rounds = kwargs.pop("rounds", 1)
    _round_cooldown_s = kwargs.pop("round_cooldown_s", 1.0)
    config_kwargs = dict(kwargs)
    errors: dict[str, str] = {}

    data = prepare_data(**config_kwargs)
    invocation = _prepare_tirx_invocation(data)
    tirx_logits = _run_tirx_invocation(data, invocation)
    torch.cuda.synchronize()
    tirx_diff = _assert_correct(data, tirx_logits, name="TIRx")

    def make_input() -> tuple[tuple[dict[str, Any], dict[str, Any]], int]:
        data = prepare_data(**config_kwargs)
        invocation = _prepare_tirx_invocation(data)
        return (data, invocation), tensor_bytes(
            data["q_in"],
            data["kv_in"],
            data["weights"],
            data["cu_seq_len_k_start"],
            data["cu_seq_len_k_end"],
            invocation["logits"],
        )

    def _deepgemm():
        return lambda case: _run_deepgemm_mqa(case[0], clean_logits=False)

    result = bench(
        {"tirx": lambda case: _run_tirx_invocation(case[0], case[1])},
        make_input,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=_rounds,
        round_cooldown_s=_round_cooldown_s,
        proton_name="deepgemm_sm100_fp4_mqa_logits",
        references={"deepgemm": _deepgemm},
    )
    result["errors"].update(errors)
    result["max_diff"] = tirx_diff
    return result


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "DEEPGEMM_TEST_COVERAGE",
    "KERNEL_META",
    "MQALogitsConfig",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
]

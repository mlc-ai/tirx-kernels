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

from dataclasses import asdict, dataclass
from functools import cache
from typing import Any
from unittest import SkipTest

import torch

from tvm.backend.cuda.op import cuda_func_call

_DEEP_GEMM_MODULE_NAME = "deep_gemm"
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

CONFIGS = DEEPGEMM_TEST_COVERAGE
BENCH_CONFIGS = CONFIGS


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
    from tvm.backend.cuda.operator.tile_primitive.gemm_async.tcgen05 import (
        sf_smem_layout,
        sf_tmem_layout,
    )
    from tvm.backend.cuda.operator.tile_primitive.tma_utils import SwizzleMode, mma_shared_layout
    from tvm.script import tirx as T
    from tvm.script.tirx import tile as Tx
    from tvm.tirx.lang.pipeline import Pipeline
    from tvm.tirx.layout import S, TCol, TileLayout, TLane, wg_local_layout

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
    smem_kv_size_per_stage = block_kv * (head_dim // 2)
    smem_sf_kv_size_per_stage = num_sfkv * 4
    smem_weight_size_per_stage = block_q * num_heads * 4
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
    # cp deposit dst: sf_per_mma is the cp atom (<= epc=4 so t_col stays a single
    # iter); SF_K//4 K-outer groups absorb the num_sfkv/128 row chunks.
    sf_tmem_kv_layout = sf_tmem_layout(128, SF_K=num_sfkv // 32, sf_per_mma=head_dim // 32)
    # SMEM "post-transpose" layouts for the SF UTCCP deposit: the warp transpose
    # rearranges each 128-uint32 chunk from col-major (c*32+lane) to row-major
    # (lane*4+c). Expressing the post layout this way lets T.permute_layout emit
    # the warp xor-swizzle transpose (replacing the hand-rolled ld/st), and the
    # same bytes feed the SF->TMEM copy. Mirrors fp8_blockwise_gemm's SF post layout.
    sf_smem_q_post_layout = TileLayout(
        S[(num_q_stages, num_sfq // 128, 4, 32) : (num_sfq, 128, 1, 4)]
    )
    sf_smem_kv_post_layout = TileLayout(
        S[(num_kv_stages, num_sfkv // 128, 4, 32) : (num_sfkv, 128, 1, 4)]
    )
    # cp (SMEM->TMEM UTCCP) SMEM-side layouts read by the T.copy_async SF
    # dispatch: the post-transpose bytes ARE the canonical sf_smem_layout, with
    # rows = one warpx4 super-block (128); SF_K = num_sf/32 columns (the
    # num_sf/128 row chunks ride the K-outer dim at stride 512, matching the TMEM
    # dst's K-outer); sf_per_mma = cp atom = head_dim/sf_vec.
    sf_cp_K = head_dim // 32
    sf_smem_q_cp_layout = sf_smem_layout(
        128, SF_K=num_sfq // 32, sf_per_mma=sf_cp_K, pipe_depth=num_q_stages
    )
    sf_smem_kv_cp_layout = sf_smem_layout(
        128, SF_K=num_sfkv // 32, sf_per_mma=sf_cp_K, pipe_depth=num_kv_stages
    )
    tmem_layout = TileLayout(S[(128, num_tmem_cols) : (1 @ TLane, 1 @ TCol)])
    logits_tir_dtype = "float32" if config.logits_dtype == "float32" else "bfloat16"

    def cuda_grid_dependency_synchronize():
        T.evaluate(T.ptx.griddepcontrol.wait())

    @T.prim_func
    def sm100_fp4_mqa_logits(
        seq_len: T.uint32,
        seq_len_kv: T.uint32,
        max_seqlen_k: T.uint32,
        logits_stride: T.uint32,
        cu_seq_len_k_start_h: T.handle,
        cu_seq_len_k_end_h: T.handle,
        logits_h: T.handle,
        q_gmem_h: T.handle,
        sf_q_gmem_h: T.handle,
        kv_gmem_h: T.handle,
        sf_kv_gmem_h: T.handle,
        weights_gmem_h: T.handle,
    ):
        # Option B: seq_len / seq_len_kv are RUNTIME (symbolic), like DeepGEMM — one
        # compiled kernel serves any length, no recompile. Everything structural
        # (head_dim, num_heads, block_*, stages, dtype) stays compile-time JIT. The
        # gmem/logits buffers are match_buffer'd against the runtime lengths; the
        # cuTensorMap descriptors are host-built per launch from these dims.
        # match_buffer must precede device_entry / any let-binding -> cast inline.
        cu_seq_len_k_start = T.match_buffer(cu_seq_len_k_start_h, (seq_len,), "int32")
        cu_seq_len_k_end = T.match_buffer(cu_seq_len_k_end_h, (seq_len,), "int32")
        logits = T.match_buffer(
            logits_h,
            (
                (T.cast(seq_len, "int32") + T.int32(block_q - 1))
                // T.int32(block_q)
                * T.int32(block_q),
                T.cast(logits_stride, "int32"),
            ),
            logits_tir_dtype,
        )
        q_gmem = T.match_buffer(q_gmem_h, (seq_len * num_heads, head_dim // 2), "uint8")
        sf_q_gmem = T.match_buffer(sf_q_gmem_h, (seq_len, num_heads), "uint32")
        kv_gmem = T.match_buffer(kv_gmem_h, (seq_len_kv, head_dim // 2), "uint8")
        sf_kv_gmem = T.match_buffer(sf_kv_gmem_h, (seq_len_kv,), "uint32")
        weights_gmem = T.match_buffer(weights_gmem_h, (seq_len, num_heads), "float32")
        T.device_entry()
        # TIRX_TRANSCRIBE_START sm100_fp4_mqa_logits
        aligned_sl: T.int32 = (
            (T.cast(seq_len, "int32") + T.int32(block_q - 1)) // T.int32(block_q) * T.int32(block_q)
        )
        logits_flat = T.decl_buffer(
            (aligned_sl * T.cast(logits_stride, "int32"),),
            logits_tir_dtype,
            data=logits.data,
            scope="global",
        )
        sm_idx = T.cta_id([config.num_sms])
        thread_idx = T.thread_id([num_threads])
        warp_idx = T.warp_id([num_warps])
        warpgroup_idx = T.warpgroup_id([num_warps // 4])

        # SMEM via SMEMPool (auto-buffer + commit): the bump allocator owns the
        # offsets (no manual smem_*_offset math). q/kv carry an explicit 64B-atom
        # swizzle layout (head_dim//2 = 64 B/row); their fp4 MMA views are
        # .view("float4_e2m1fn") of the same bytes. NOTE: anything that needs a
        # buffer's start pointer must use .ptr_to([0,...]) / .view (which carry
        # elem_offset), NOT .data — under the pool .data is the arena base.
        pool = T.SMEMPool()
        smem_q = pool.alloc(
            (num_q_stages, block_q * num_heads, head_dim // 2),
            "uint8",
            scope="shared.dyn",
            align=swizzle_alignment,
            layout=mma_shared_layout(
                "uint8",
                SwizzleMode.SWIZZLE_64B_ATOM,
                (num_q_stages, block_q * num_heads, head_dim // 2),
            ),
        )
        smem_kv = pool.alloc(
            (num_kv_stages, block_kv, head_dim // 2),
            "uint8",
            scope="shared.dyn",
            align=swizzle_alignment,
            layout=mma_shared_layout(
                "uint8", SwizzleMode.SWIZZLE_64B_ATOM, (num_kv_stages, block_kv, head_dim // 2)
            ),
        )
        smem_sf_q = pool.alloc((num_q_stages, num_sfq), "uint32", align=16)
        # 2D (block_q, num_heads) view for the copy_async(tma) dst. A fresh decl_buffer
        # (default 3D row-major layout), NOT smem_sf_q.view(shape) — the shape-view
        # keeps the source's rank-2 layout, which mis-maps the 3D TMA write.
        smem_sf_q_2d = T.decl_buffer(
            (num_q_stages, block_q, num_heads),
            "uint32",
            data=smem_sf_q.data,
            scope="shared.dyn",
            elem_offset=smem_sf_q.elem_offset,
            align=16,
        )
        smem_sf_kv = pool.alloc((num_kv_stages, num_sfkv), "uint32", align=16)
        smem_weights = pool.alloc((num_q_stages, block_q, num_heads), "float32", align=16)
        # Per-pipeline producer/consumer barrier pairs as Pipeline objects (replacing
        # the flat smem_barriers buffer + manual init/wait/arrive helpers). full =
        # data ready, empty = slot free. The Pipeline constructor allocs both barriers
        # from the pool (same order/offsets as the old flat layout: full_q, empty_q,
        # full_kv, empty_kv, full_tmem, empty_tmem) and runs mbarrier.init (thread 0)
        # with the counts below — so there is no separate init loop; only the
        # fence.mbarrier_init + cta_sync before first use remain.
        #   q_pipe:    TMA load   -> MMA + math consumers; empty freed by every
        #              reader, so empty count = num_math_threads + the MMA warp (32).
        #   kv_pipe:   TMA load   -> MMA consumer; empty freed by tcgen05.commit.
        #   tmem_pipe: MMA commit -> math consumers; empty freed by mbarrier.arrive
        #              (one math warpgroup = 128 threads per tmem stage).
        q_pipe = Pipeline(
            pool,
            num_q_stages,
            full="tma",
            empty="mbar",
            init_full=1,
            init_empty=num_math_threads + 32,
        )
        kv_pipe = Pipeline(
            pool, num_kv_stages, full="tma", empty="tcgen05", init_full=1, init_empty=1
        )
        tmem_pipe = Pipeline(
            pool, num_tmem_stages, full="tcgen05", empty="mbar", init_full=1, init_empty=128
        )
        tmem_ptr_in_smem = pool.alloc((1,), "uint32", align=4)
        pool.commit()
        # TMEM via TMEMPool: the bump allocator owns the column offsets (replacing
        # the hand-computed allocated_addr / tmem_start_col math). TMEMPool.alloc
        # gives a CONSTANT col_start (0-based), so the accumulator stays at a constant
        # 0 base -> the epilogue tcgen05.ld folds the base into the col offset instead
        # of reloading tmem_ptr_in_smem[0] from smem each hot-loop iter (asserted == 0
        # below). The manual tcgen05.alloc/dealloc keep the lifecycle. move_base_to
        # overlaps the SF columns inside the over-declared accumulator span.
        tmem_pool = T.TMEMPool(
            pool, total_cols=num_tmem_cols, cta_group=1, tmem_addr=tmem_ptr_in_smem
        )
        tmem = tmem_pool.alloc(
            (128, num_tmem_cols), "float32", layout=tmem_layout, cols=num_tmem_cols
        )
        tmem_pool.move_base_to(tmem_start_col_of_sfq)
        sfq_tmem = tmem_pool.alloc(
            (128, num_sfq // 32), "float8_e8m0fnu", layout=sf_tmem_q_layout, cols=num_sfq // 32
        )
        tmem_pool.move_base_to(tmem_start_col_of_sfkv)
        sfkv_tmem = tmem_pool.alloc(
            (128, num_sfkv // 32), "float8_e8m0fnu", layout=sf_tmem_kv_layout, cols=num_sfkv // 32
        )
        seq_k_start = T.alloc_local((block_q,), "uint32")
        seq_k_end = T.alloc_local((block_q,), "uint32")
        schedule_result = T.alloc_local((2,), "uint32")

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
                seq_k_start[schedule_i] = T.min(
                    T.cast(cu_seq_len_k_start[row_idx], "uint32"), seq_len_kv
                )
                seq_k_end[schedule_i] = T.min(
                    T.cast(cu_seq_len_k_end[row_idx], "uint32"), seq_len_kv
                )
                schedule_start = T.min(schedule_start, seq_k_start[schedule_i])
                schedule_end = T.max(schedule_end, seq_k_end[schedule_i])
            schedule_start = schedule_start // T.uint32(4) * T.uint32(4)
            num_kv_blocks = (schedule_end - schedule_start + T.uint32(block_kv - 1)) // T.uint32(
                block_kv
            )
            schedule_result[0] = schedule_start
            schedule_result[1] = num_kv_blocks

        # The Pipeline constructors above already ran mbarrier.init (thread 0); this
        # fence makes those inits visible, and the cta_sync below publishes them
        # CTA-wide before the first wait/arrive.
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
                    q_pipe.empty.wait(q_stage_idx, q_phase ^ T.uint32(1))
                    # u32 row base — the copy_async(tma) gmem-layout grouping now
                    # handles unsigned shape extents (no int32 cast needed).
                    q_row0: T.uint32 = q_idx * T.uint32(block_q * num_heads)
                    Tx.copy_async(
                        smem_q[q_stage_idx],
                        q_gmem[q_row0 : q_row0 + block_q * num_heads, :],
                        dispatch="tma",
                        mbar=q_pipe.full.ptr_to([q_stage_idx]),
                        cta_group=1,
                        cache_hint="evict_normal",
                    )
                    q_blk0: T.uint32 = q_idx * T.uint32(block_q)
                    Tx.copy_async(
                        smem_sf_q_2d[q_stage_idx],
                        sf_q_gmem[q_blk0 : q_blk0 + block_q, :],
                        dispatch="tma",
                        mbar=q_pipe.full.ptr_to([q_stage_idx]),
                        cta_group=1,
                        cache_hint="evict_normal",
                    )
                    Tx.copy_async(
                        smem_weights[q_stage_idx],
                        weights_gmem[q_blk0 : q_blk0 + block_q, :],
                        dispatch="tma",
                        mbar=q_pipe.full.ptr_to([q_stage_idx]),
                        cta_group=1,
                        cache_hint="evict_normal",
                    )
                    q_pipe.full.arrive(
                        q_stage_idx,
                        tx_count=smem_q_size_per_stage
                        + real_num_sfq * 4
                        + smem_weight_size_per_stage,
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
                        kv_pipe.empty.wait(kv_stage_idx, kv_phase ^ T.uint32(1))
                        kv_row0: T.uint32 = kv_start + kv_idx * T.uint32(block_kv)
                        Tx.copy_async(
                            smem_kv[kv_stage_idx],
                            kv_gmem[kv_row0 : kv_row0 + block_kv, :],
                            dispatch="tma",
                            mbar=kv_pipe.full.ptr_to([kv_stage_idx]),
                            cta_group=1,
                            cache_hint="evict_normal",
                        )
                        Tx.copy_async(
                            smem_sf_kv[kv_stage_idx, 0:block_kv],
                            sf_kv_gmem[kv_row0 : kv_row0 + block_kv],
                            dispatch="tma",
                            mbar=kv_pipe.full.ptr_to([kv_stage_idx]),
                            cta_group=1,
                            cache_hint="evict_normal",
                        )
                        kv_pipe.full.arrive(
                            kv_stage_idx,
                            tx_count=smem_kv_size_per_stage + smem_sf_kv_size_per_stage,
                        )
                        kv_idx = kv_idx + T.uint32(1)
                    q_idx = q_idx + T.uint32(config.num_sms)
        elif warp_idx == spec_warp_start + 2:
            T.ptx.setmaxnreg(False, 56)
            T.cuda.trap_when_assert_failed(tmem_ptr_in_smem[0] == T.uint32(0))
            desc_i: T.uint32
            # GAP 1: encode the block-scaled instruction descriptor ONCE here
            # (above the q/kv/math_wg loops). The T.gemm_async D-MMA below is
            # passed ``descI=desc_i``; with the gemm_async decouple fix that path
            # copies desc_i to a per-call local and still does the per-ki sf_id
            # ``runtime_instr_desc`` rotation — so the encode_instr_descriptor
            # intrinsic is emitted exactly once for the whole kernel (matching the
            # hand-rolled original) instead of once per gemm_async call (2x: once
            # per math warpgroup).
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
            # REGION D operand views: fp4 over the same packed-uint8 smem bytes.
            # .view down-casts uint8->fp4 (halves the byte count -> head_dim//2 ->
            # head_dim cols, unpacks the 64B swizzle) AND carries elem_offset, so it
            # gives the true buffer start under the pool (unlike reinterpret(.data)).
            smem_kv_fp4 = smem_kv.view("float4_e2m1fn")
            smem_q_fp4 = smem_q.view("float4_e2m1fn")
            # Post-transpose SF views (same uint32 bytes as smem_sf_q/kv, indexed
            # under the row-major post layout). Tx.permute_layout(post, raw) does the
            # in-place warp transpose into these bytes.
            smem_sf_q_post = smem_sf_q.view(num_q_stages, num_sfq, layout=sf_smem_q_post_layout)
            smem_sf_kv_post = smem_sf_kv.view(
                num_kv_stages, num_sfkv, layout=sf_smem_kv_post_layout
            )
            # e8m0 views of the same SF SMEM under the cp (sf_smem) layout, the
            # SMEM source for T.copy_async into the SF TMEM. Declared at the
            # natural (rows=128, SF_K) SF footprint so the copy region matches
            # the (128, SF_K) SF TMEM tile; the lane/super-block packing lives
            # in sf_smem_*_cp_layout, not in the buffer shape.
            smem_sf_q_cp = smem_sf_q.view("float8_e8m0fnu").view(
                num_q_stages, 128, num_sfq // 32, layout=sf_smem_q_cp_layout
            )
            smem_sf_kv_cp = smem_sf_kv.view("float8_e8m0fnu").view(
                num_kv_stages, 128, num_sfkv // 32, layout=sf_smem_kv_cp_layout
            )
            # SFA/SFB TMEM views in the dispatcher-canonical sf_tmem_layout (the
            # gemm validator expects sf_per_mma == sf_mma_k = 2 for fp4+e8m0, not
            # the 8 the cp-side buffers carry). Same TMEM columns (allocated_addr)
            # as the SF cp deposits: SFB(=Q) at tmem_start_col_of_sfq; SFA(=KV) at
            # tmem_start_col_of_sfkv (+math_wg_i*4 chosen per warpgroup below).
            sf_mma_k_dispatch = T.meta_var(umma_k // 32)  # fp4 SF_VEC=32 -> 2 SFs/MMA-K
            sf_K_dispatch = T.meta_var((umma_k // 32) * (head_dim // umma_k))
            sf_tmem_q_mma_layout = T.meta_var(
                sf_tmem_layout(128, SF_K=sf_K_dispatch, sf_per_mma=sf_mma_k_dispatch)
            )
            sf_tmem_kv_mma_layout = T.meta_var(
                sf_tmem_layout(128, SF_K=sf_K_dispatch, sf_per_mma=sf_mma_k_dispatch)
            )
            sfq_tmem_mma = T.decl_buffer(
                (128, sf_K_dispatch),
                "float8_e8m0fnu",
                scope="tmem",
                allocated_addr=tmem_start_col_of_sfq,
                layout=sf_tmem_q_mma_layout,
            )
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
                q_pipe.full.wait(q_stage_idx, q_phase)
                Tx.warp.permute_layout(smem_sf_q_post[q_stage_idx, :], smem_sf_q[q_stage_idx, :])
                T.ptx.fence.proxy_async("shared::cta")
                if T.ptx.elect_sync():
                    Tx.copy_async(sfq_tmem, smem_sf_q_cp[T.cast(q_stage_idx, "int32")], cta_group=1)
                T.cuda.warp_sync()
                kv_idx: T.uint32 = T.uint32(0)
                while kv_idx < num_kv_blocks:
                    kv_stage_idx: T.uint32 = kv_iter_idx % T.uint32(num_kv_stages)
                    kv_phase: T.uint32 = (kv_iter_idx // T.uint32(num_kv_stages)) & T.uint32(1)
                    kv_iter_idx = kv_iter_idx + T.uint32(1)
                    kv_pipe.full.wait(kv_stage_idx, kv_phase)
                    # Transpose per 128-uint32 chunk (P=4, [4,32]) — the shape the
                    # hand-rolled transpose used — not one P=8 [2,4,32] over the whole
                    # 256, so the warp-xor-swizzle addressing needs no j//4 / j%4 split.
                    # Fence PER chunk (transpose0→fence, transpose1→fence), matching
                    # the hand-rolled deposit's interleaving — NOT both transposes
                    # then a single trailing fence. The interleaved form preserves
                    # the SF-deposit→cp→MMA instruction schedule the latency-bound
                    # bf16-dense epilogue depends on; batching the transposes drifts
                    # ptxas scheduling and lengthens the consumer's wait on the MMA
                    # result (measured: consumer-wait long-scoreboard +187 → +104).
                    Tx.warp.permute_layout(
                        smem_sf_kv_post[kv_stage_idx, 0:num_utccp_aligned_elems],
                        smem_sf_kv[kv_stage_idx, 0:num_utccp_aligned_elems],
                    )
                    T.ptx.fence.proxy_async("shared::cta")
                    Tx.warp.permute_layout(
                        smem_sf_kv_post[kv_stage_idx, num_utccp_aligned_elems:num_sfkv],
                        smem_sf_kv[kv_stage_idx, num_utccp_aligned_elems:num_sfkv],
                    )
                    T.ptx.fence.proxy_async("shared::cta")
                    # cp + MMA share ONE elect scope (matching the hand-rolled
                    # deposit): copy_async needs the thread scope, and folding the
                    # MMA issue into it drops a redundant elect.sync per kv-iter and
                    # lets the cp overlap the MMA setup.
                    if T.ptx.elect_sync():
                        Tx.copy_async(
                            sfkv_tmem, smem_sf_kv_cp[T.cast(kv_stage_idx, "int32")], cta_group=1
                        )
                        for math_wg_i in T.unroll(0, num_math_warpgroups):
                            tmem_stage_idx: T.uint32 = tmem_iter_idx % T.uint32(num_tmem_stages)
                            tmem_phase: T.uint32 = (
                                tmem_iter_idx // T.uint32(num_tmem_stages)
                            ) & T.uint32(1)
                            tmem_iter_idx = tmem_iter_idx + T.uint32(1)
                            tmem_addr: T.uint32 = tmem_stage_idx * T.uint32(umma_n)
                            # int32 column base for the C-accumulator slice (the
                            # gemm dispatch's C layout check compares the slice
                            # extent dtype; a uint32 base would promote the N=umma_n
                            # extent to uint32 and fail StructuralEqual vs int32).
                            tmem_col_start: T.int32 = T.cast(tmem_addr, "int32")
                            tmem_pipe.empty.wait(tmem_stage_idx, tmem_phase ^ T.uint32(1))
                            # REGION D: block-scaled fp4 UMMA (KV @ Qᵀ -> TMEM logits
                            # accumulator) as the T.gemm_async tile primitive. It runs
                            # the head_dim//umma_k=2 K-loop internally, builds the A/B
                            # matrix descriptors (swizzle=2), and — with descI hoisted
                            # + passed below — copies desc_i to a per-call local and
                            # rotates the per-ki SF id (k*2) via runtime_instr_desc,
                            # emitting tcgen05.mma.block_scale + enable_input_d=(k!=0).
                            # Operand order (D, A=KV, B=Q) with SFA=KV-scales /
                            # SFB=Q-scales in TMEM (sfa@388+wg*4, sfb@384) matches the
                            # original hand-rolled block-scaled UMMA exactly.
                            sfkv_tmem_mma = T.decl_buffer(
                                (128, sf_K_dispatch),
                                "float8_e8m0fnu",
                                scope="tmem",
                                allocated_addr=tmem_start_col_of_sfkv + math_wg_i * 4,
                                layout=sf_tmem_kv_mma_layout,
                            )
                            Tx.gemm_async(
                                tmem[:, tmem_col_start : tmem_col_start + umma_n],
                                smem_kv_fp4[
                                    kv_stage_idx,
                                    math_wg_i * umma_m : math_wg_i * umma_m + umma_m,
                                    :,
                                ],
                                smem_q_fp4[q_stage_idx, :, :],
                                SFA=sfkv_tmem_mma,
                                SFB=sfq_tmem_mma,
                                accum=False,
                                descI=desc_i,
                                dispatch="tcgen05",
                                cta_group=1,
                                # Recompute the SMEM descriptor per MMA (cvta on the
                                # uniform datapath). The kv-stage index lands in a regular
                                # register here, so the default "hoist" path costs an R2UR
                                # per MMA (~+3% on this latency-bound kernel). See the
                                # gemm_async tcgen05 dispatch for the hoist-vs-recompute
                                # rationale.
                                smem_desc="recompute",
                            )
                            tmem_pipe.full.arrive(tmem_stage_idx)
                    if T.ptx.elect_sync():
                        kv_pipe.empty.arrive(kv_stage_idx, cta_group=1)
                    kv_idx = kv_idx + T.uint32(1)
                q_pipe.empty.arrive(q_stage_idx)
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
                q_pipe.full.wait(q_stage_idx, q_phase)
                if num_kv_blocks > T.uint32(0):
                    Tx.warpgroup.copy(cached_weights, smem_weights[q_stage_idx])
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
                        tmem_phase: T.uint32 = (
                            tmem_iter_idx // T.uint32(num_tmem_stages)
                        ) & T.uint32(1)
                        tmem_iter_idx = tmem_iter_idx + T.uint32(num_math_warpgroups)
                        tmem_pipe.full.wait(tmem_stage_idx, tmem_phase)
                        for q_inner_i in T.unroll(0, block_q):
                            tmem_addr: T.uint32 = tmem_stage_idx * T.uint32(umma_n) + T.uint32(
                                q_inner_i * num_heads
                            )
                            # REGION E: TMEM->register read of the logits accumulator via tile primitive.
                            # accum stays a flat per-thread (num_heads,) buffer for the wrelu reduce below;
                            # here we take a 2-D (128, num_heads):(1@tid_in_wg,1) layout VIEW of the SAME
                            # per-thread bytes so the tmem<->local copy_async dispatch (.32x32b path)
                            # accepts it and re-emits the identical tcgen05.ld.32x32b.x{num_heads}.
                            accum_2d = accum.view(128, num_heads, layout=wg_local_layout(num_heads))
                            # Pass the uint32 column start directly: the tmem<->local
                            # dispatch's divisibility proof is now robust to unsigned
                            # col-starts (elem_per_32b==1 for fp32 is always
                            # divisible), so no per-ld int32 cvt is needed.
                            Tx.warpgroup.copy_async(
                                accum_2d, tmem[:, tmem_addr : tmem_addr + num_heads]
                            )
                            T.ptx.tcgen05.wait.ld()
                            if q_inner_i == block_q - 1:
                                tmem_pipe.empty.arrive(tmem_stage_idx)
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
                                # on bf16xcompressed (verified in SASS + /bench-suite). A predicated PTX
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
                q_pipe.empty.arrive(q_stage_idx)
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
    # Option B: the generated kernel does NOT depend on seq_len / seq_len_kv /
    # disable_cp / logits_stride (all RUNTIME now — see the match_buffer'd gmem
    # buffers + runtime logits_stride). Pass canonical values for those so the
    # compile cache dedups to ONE kernel per *structural* config
    # (num_heads, head_dim, logits_dtype, compressed_logits, num_sms) — exactly like
    # DeepGEMM, which templates on structure and takes the lengths at runtime.
    return _compile_tirx_mqa_for_config(
        seq_len=config.block_q,
        seq_len_kv=config.block_kv,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        logits_dtype=config.logits_dtype,
        compressed_logits=config.compressed_logits,
        disable_cp=True,
        num_sms=config.num_sms,
        logits_stride_override=None,
    )


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
    data: dict[str, Any], logits: torch.Tensor | None = None, *, executable: Any | None = None
) -> dict[str, Any]:
    config: MQALogitsConfig = data["config"]
    if logits is None:
        logits = _allocate_logits(config, data["max_seqlen_k"])
    if executable is None:
        executable = _compile_tirx_mqa(config, data["max_seqlen_k"])
    return {"executable": executable, "logits": logits}


def _run_tirx_invocation(data: dict[str, Any], invocation: dict[str, Any]) -> torch.Tensor:
    config: MQALogitsConfig = data["config"]
    executable = invocation["executable"]
    logits = invocation["logits"]
    # All five GMEM->SMEM bulk loads are TIRx copy_async(dispatch="tma") tile
    # primitives; pass the raw tensors as gmem buffers (the cuTensorMap
    # descriptors are built in-kernel by the copy dispatch). The packed-fp4
    # Q/KV are reinterpreted as uint8 (sub-byte dtypes aren't TMA-addressable).
    q_fp4, sf_q = data["q_in"]
    kv_fp4, sf_kv = data["kv_in"]
    weights = data["weights"]
    q_gmem = (
        q_fp4.reshape(config.seq_len * config.num_heads, config.head_dim // 2)
        .contiguous()
        .view(torch.uint8)
    )
    kv_gmem = kv_fp4.contiguous().view(torch.uint8)
    sf_q_gmem = sf_q.contiguous().view(torch.uint32)
    sf_kv_gmem = sf_kv.contiguous().view(torch.uint32)
    _prepare_global_barrier(executable)
    executable.mod(
        config.seq_len,
        config.seq_len_kv,
        data["max_seqlen_k"],
        logits.stride(0),
        data["cu_seq_len_k_start"],
        data["cu_seq_len_k_end"],
        logits,
        q_gmem,
        sf_q_gmem,
        kv_gmem,
        sf_kv_gmem,
        weights,
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
    from tvm.tirx.bench import bench

    warmup = kwargs.pop("warmup", None)
    repeat = kwargs.pop("repeat", None)
    timer = kwargs.pop("timer", None)  # None inherits the global default (proton)
    _rounds = kwargs.pop("rounds", 1)
    _cooldown_s = kwargs.pop("cooldown_s", 1.0)
    config_kwargs = dict(kwargs)
    tirx_executable = _compile_tirx_mqa(_make_config(**config_kwargs), 0)

    # Allocate inputs once, outside the timed region (Triton-standard pure launch).
    data = prepare_data(**config_kwargs)
    invocation = _prepare_tirx_invocation(data, executable=tirx_executable)

    # Correctness gate before timing (preserves the old validate_case behavior).
    tirx_logits = _run_tirx_invocation(data, invocation)
    torch.cuda.synchronize()
    max_diff = _assert_correct(data, tirx_logits, name="TIRx")

    funcs = {"tirx": lambda: _run_tirx_invocation(data, invocation)}

    def _deepgemm():
        return lambda: _run_deepgemm_mqa(data, clean_logits=False)

    references = {"deepgemm": _deepgemm}

    result = bench(
        funcs,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references=references,
        rounds=_rounds,
        cooldown_s=_cooldown_s,
    )
    result["max_diff"] = max_diff
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

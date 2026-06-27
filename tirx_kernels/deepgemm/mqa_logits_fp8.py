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

_DEEP_GEMM_MODULE_NAME = "deep_gemm"
_SM100_SMEM_CAPACITY = 232448
_TEST_DIFF_THRESHOLD = 5e-6


@dataclass(frozen=True)
class MQALogitsFP8Config:
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
        if self.head_dim not in (32, 64, 128):
            raise ValueError("head_dim must be 32, 64, or 128 for the SM100 FP8 MQA logits kernel")
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


def _make_config(**kwargs: Any) -> MQALogitsFP8Config:
    kwargs = {key: value for key, value in kwargs.items() if key != "label"}
    config = MQALogitsFP8Config(**kwargs)
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
    "name": "deepgemm_sm100_fp8_mqa_logits",
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


def _generate_ks_ke(config: MQALogitsFP8Config) -> tuple[torch.Tensor, torch.Tensor]:
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
    cols = torch.arange(0, seq_len_kv, device="cuda")
    logits = torch.empty((q.shape[0], seq_len_kv), device="cuda", dtype=torch.float32)
    chunk_size = 128
    for chunk_start in range(0, q.shape[0], chunk_size):
        chunk_end = min(chunk_start + chunk_size, q.shape[0])
        score = torch.einsum("mhd,nd->hmn", q_f32[chunk_start:chunk_end], kv_f32)
        chunk_logits = (
            score.relu() * weights[chunk_start:chunk_end].unsqueeze(-1).transpose(0, 1)
        ).sum(dim=0)
        mask_lo = cols[None, :] >= cu_seq_len_k_start[chunk_start:chunk_end, None]
        mask_hi = cols[None, :] < cu_seq_len_k_end[chunk_start:chunk_end, None]
        logits[chunk_start:chunk_end] = chunk_logits.masked_fill(
            ~(mask_lo & mask_hi), float("-inf")
        )
    return logits


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    deep_gemm, source = load_deep_gemm_mqa()
    config = _make_config(**kwargs)
    if torch.cuda.is_available():
        torch.cuda.set_device(torch.cuda.current_device())
    else:
        raise SkipTest("CUDA is required for SM100 FP8 MQA logits")
    if torch.cuda.get_device_capability()[0] < 10:
        raise SkipTest("SM100 FP8 MQA logits requires compute capability 10.x")

    torch.manual_seed(config.seed)
    q = torch.randn(
        config.seq_len, config.num_heads, config.head_dim, device="cuda", dtype=torch.bfloat16
    )
    kv = torch.randn(config.seq_len_kv, config.head_dim, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(config.seq_len, config.num_heads, device="cuda", dtype=torch.float32)
    ks, ke = _generate_ks_ke(config)

    q_in = q.to(torch.float8_e4m3fn).contiguous()
    kv_in = deep_gemm.utils.per_custom_dims_cast_to_fp8(kv, (0,), False)

    q_simulated = q_in.to(torch.bfloat16)
    kv_simulated = (kv_in[0].float() * kv_in[1].unsqueeze(1)).to(torch.bfloat16)
    reference = _ref_mqa_logits(
        q_simulated.to(torch.bfloat16), kv_simulated.to(torch.bfloat16), weights, ks, ke
    )
    max_seqlen_k = int((ke - ks).max().item()) if config.compressed_logits else 0
    runtime_config = MQALogitsFP8Config(
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


def get_kernel(**kwargs: Any):
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
    num_kv_stages = 3
    num_specialized_threads = 128
    num_math_threads = 256
    num_math_warpgroups = num_math_threads // 128
    num_threads = num_specialized_threads + num_math_threads
    num_warps = num_threads // 32
    spec_warp_start = num_math_warpgroups * 4
    umma_m = 128
    umma_k = 32
    umma_n = block_q * num_heads
    smem_q_size_per_stage = block_q * num_heads * head_dim
    smem_weight_size_per_stage = block_q * num_heads * 4
    smem_kv_size_per_stage = block_kv * head_dim
    smem_kv_scale_size_per_stage = block_kv * 4
    # q/kv carry a 128B MMA swizzle layout (head_dim * 1B = 128 B/row); the
    # SWIZZLE_128B_ATOM is 8 rows * 1024 B, so their SMEM base must be 1024-aligned.
    _SWZ = {
        32: SwizzleMode.SWIZZLE_32B_ATOM,
        64: SwizzleMode.SWIZZLE_64B_ATOM,
        128: SwizzleMode.SWIZZLE_128B_ATOM,
    }[head_dim]
    swizzle_alignment = 8 * head_dim
    # Compile-time halving schedule for the epilogue head reduce (log-depth tree of
    # Epilogue weighted-ReLU reduce: result = sum_h relu(accum[h]) * weights[h].
    # WRELU_K = accumulator width (mirrors the hand-rolled two float2 accumulators =
    # 4 fp32 lanes): multiply-accumulate the relu'd heads into a WRELU_K-wide buffer
    # via T.fma (one fused mul+add, packed fma.f32x2) so the weight-mul is NOT a
    # separate pass, then a log-depth tree over the WRELU_K lanes. A plain T.mul +
    # T.sum lowers to 32 FMUL2 + a serial 63-deep FADD chain (~15% slower); the
    # un-fused mul + a log tree is still ~9% slower — only the fused fma matches.
    WRELU_K = 4
    _wrelu_tree = []
    _rn = WRELU_K
    while _rn > 1:
        _wrelu_tree.append((_rn // 2, _rn))
        _rn //= 2

    def emit_wrelu_reduce(relu_buf, acc, wbuf, wbase, sc):
        # Plain Python helper (NOT @T.inline): the parser only accepts TIR loop
        # forms inside a prim_func, but executes a plain call's Python ``for`` as a
        # parse-time unroll, so all slice extents stay compile-time ints. ``sc`` is
        # the cooperation-scope namespace (e.g. ``Tx.warp``) the caller emits under.
        sc.mul(acc, relu_buf[0:WRELU_K], wbuf[wbase : wbase + WRELU_K])
        for g in range(WRELU_K, num_heads, WRELU_K):
            sc.fma(acc, relu_buf[g : g + WRELU_K], wbuf[wbase + g : wbase + g + WRELU_K], acc)
        for tree_h, tree_n in _wrelu_tree:
            sc.add(acc[0:tree_h], acc[0:tree_h], acc[tree_h:tree_n])

    num_tmem_cols = block_q * num_heads * num_math_warpgroups
    if num_tmem_cols > 512:
        raise ValueError(f"tensor memory columns {num_tmem_cols} exceeds SM100 single-CTA limit")
    tmem_layout = TileLayout(S[(128, num_tmem_cols) : (1 @ TLane, 1 @ TCol)])
    logits_tir_dtype = "float32" if config.logits_dtype == "float32" else "bfloat16"

    def lane_id_u32():
        return T.cast(T.ptx.fetch_register(32, "laneid"), "uint32")

    def cuda_grid_dependency_synchronize():
        T.evaluate(T.ptx.griddepcontrol.wait())

    def named_barrier_sync_8(count):
        T.evaluate(T.ptx.bar.sync(8, count))

    @T.prim_func
    def sm100_fp8_mqa_logits(
        seq_len: T.uint32,
        seq_len_kv: T.uint32,
        max_seqlen_k: T.uint32,
        logits_stride: T.uint32,
        cu_seq_len_k_start_h: T.handle,
        cu_seq_len_k_end_h: T.handle,
        logits_h: T.handle,
        q_gmem_h: T.handle,
        kv_gmem_h: T.handle,
        kv_scales_gmem_h: T.handle,
        weights_gmem_h: T.handle,
    ):
        # seq_len / seq_len_kv are RUNTIME (symbolic) like DeepGEMM: ONE compiled
        # kernel serves any length, no recompile. Everything structural (head_dim,
        # num_heads, block_*, stages, dtype, compressed_logits) stays compile-time.
        # The gmem / logits buffers are match_buffer'd against the runtime lengths;
        # copy_async(tma) builds the cuTensorMap host-side per launch from these
        # dims. match_buffer must precede device_entry / any let-binding -> cast inline.
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
            config.logits_dtype,
        )
        q_gmem = T.match_buffer(q_gmem_h, (seq_len * num_heads, head_dim), "uint8")
        kv_gmem = T.match_buffer(kv_gmem_h, (seq_len_kv, head_dim), "uint8")
        kv_scales_gmem = T.match_buffer(kv_scales_gmem_h, (seq_len_kv,), "float32")
        weights_gmem = T.match_buffer(weights_gmem_h, (seq_len, num_heads), "float32")
        T.device_entry()
        T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})
        # TIRX_TRANSCRIBE_START sm100_fp8_mqa_logits
        aligned_sl: T.int32 = (
            (T.cast(seq_len, "int32") + T.int32(block_q - 1)) // T.int32(block_q) * T.int32(block_q)
        )
        logits_flat = T.decl_buffer(
            (aligned_sl * T.cast(logits_stride, "int32"),),
            logits_tir_dtype,
            data=logits.data,
            scope="global",
        )
        num_q_blocks: T.uint32 = (seq_len + T.uint32(block_q - 1)) // T.uint32(block_q)
        sm_idx = T.cta_id([config.num_sms])
        sm_idx_u32: T.let = T.cast(sm_idx, "uint32")
        warp_idx = T.warp_id([num_warps])
        warp_idx_u32: T.let = T.cast(warp_idx, "uint32")
        warpgroup_idx = T.warpgroup_id([num_warps // 4])
        lane_idx = T.lane_id([32])
        lane_idx_u32: T.let = lane_id_u32()

        # SMEM via SMEMPool (bump allocator owns the offsets — no manual
        # smem_*_offset math). q/kv carry the 128B-atom MMA swizzle layout; their
        # fp8 MMA views are .view("float8_e4m3fn") of the same uint8 bytes. NOTE: a
        # buffer's start pointer must come from .ptr_to([0,...]) / indexing (which
        # carry elem_offset), NOT .data — under the pool .data is the arena base.
        pool = T.SMEMPool()
        smem_q = pool.alloc(
            (num_q_stages, block_q * num_heads, head_dim),
            "uint8",
            scope="shared.dyn",
            align=swizzle_alignment,
            layout=mma_shared_layout("uint8", _SWZ, (num_q_stages, block_q * num_heads, head_dim)),
        )
        smem_weights = pool.alloc((num_q_stages, block_q, num_heads), "float32", align=16)
        smem_kv = pool.alloc(
            (num_kv_stages, block_kv, head_dim),
            "uint8",
            scope="shared.dyn",
            align=swizzle_alignment,
            layout=mma_shared_layout("uint8", _SWZ, (num_kv_stages, block_kv, head_dim)),
        )
        smem_kv_scales = pool.alloc((num_kv_stages, block_kv), "float32", align=16)
        # Producer/consumer barrier pairs as Pipeline objects (replacing the flat
        # smem_barriers buffer + manual init/wait/arrive helpers). full = data ready,
        # empty = slot free. Each Pipeline allocs both barriers from the pool and runs
        # mbarrier.init (thread 0) with the counts below, so there is no separate init
        # loop — only the fence.mbarrier_init + cta_sync before first use remain.
        #   q_pipe:    TMA load -> MMA + math consumers; empty freed by every reader
        #              (num_math_threads math + the 32-lane MMA warp).
        #   kv_pipe:   TMA load -> MMA + math consumers; empty freed by the math
        #              warpgroups (num_math_threads).
        #   umma_pipe: MMA commit (TCGen05Bar) -> math consumer; empty freed by
        #              mbarrier.arrive (one math warpgroup = 128 threads per stage).
        q_pipe = Pipeline(
            pool,
            num_q_stages,
            full="tma",
            empty="mbar",
            init_full=1,
            init_empty=num_math_threads + 32,
        )
        kv_pipe = Pipeline(
            pool, num_kv_stages, full="tma", empty="mbar", init_full=1, init_empty=num_math_threads
        )
        umma_pipe = Pipeline(
            pool, num_math_warpgroups, full="tcgen05", empty="mbar", init_full=1, init_empty=128
        )
        tmem_ptr_in_smem = pool.alloc((1,), "uint32", align=4)
        pool.commit()
        # TMEM via TMEMPool: gives a CONSTANT 0-based col_start so the gemm_async /
        # copy_async tmem addressing folds the base into the col offset instead of
        # reloading tmem_ptr_in_smem[0] from SMEM each hot-loop tcgen05.mma /
        # tcgen05.ld (the reload costs ~10% on the latency-bound epilogue read). The
        # manual tcgen05.alloc/dealloc below keep the lifecycle.
        tmem_pool = T.TMEMPool(
            pool, total_cols=num_tmem_cols, cta_group=1, tmem_addr=tmem_ptr_in_smem
        )
        tmem = tmem_pool.alloc(
            (128, num_tmem_cols), "float32", layout=tmem_layout, cols=num_tmem_cols
        )
        seq_k_start = T.alloc_local((block_q,), "uint32")
        seq_k_end = T.alloc_local((block_q,), "uint32")
        schedule_result = T.alloc_local((4,), "uint32")

        @T.inline
        def store_logits(flat_offset, value):
            # Scalar predicated store: the compressed epilogue guards this with a
            # plain `if row_k_start <= kv_offset < row_k_end:` so ptxas emits a
            # predicated `@P STG` (matching DeepGEMM) — the externally-visible logits
            # output is a per-thread scalar write, so TMA/bulk store does not apply
            # (I2: scalar is the correct choice for non-contiguous visible outputs).
            logits_flat[flat_offset] = value

        @T.inline
        def issue_tma_q(stage_idx, q_block_idx):
            # u32 row bases — the copy_async(tma) gmem-layout grouping now handles
            # unsigned shape extents (no int32 cast needed).
            q_row0: T.uint32 = q_block_idx * T.uint32(block_q * num_heads)
            q_blk0: T.uint32 = q_block_idx * T.uint32(block_q)
            Tx.copy_async(
                smem_q[stage_idx],
                q_gmem[q_row0 : q_row0 + block_q * num_heads, :],
                dispatch="tma",
                mbar=q_pipe.full.ptr_to([stage_idx]),
                cta_group=1,
                cache_hint="evict_normal",
            )
            Tx.copy_async(
                smem_weights[stage_idx],
                weights_gmem[q_blk0 : q_blk0 + block_q, :],
                dispatch="tma",
                mbar=q_pipe.full.ptr_to([stage_idx]),
                cta_group=1,
                cache_hint="evict_normal",
            )
            q_pipe.full.arrive(
                stage_idx, tx_count=smem_q_size_per_stage + smem_weight_size_per_stage
            )

        @T.inline
        def load_schedule(q_idx, q_iter_idx, q_iter_offset):
            schedule_start: T.uint32 = T.uint32(0xFFFFFFFF)
            schedule_end: T.uint32 = T.uint32(0)
            for schedule_i in T.unroll(0, block_q):
                row_idx: T.uint32 = T.min(
                    q_idx * T.uint32(block_q) + T.uint32(schedule_i), seq_len - T.uint32(1)
                )
                seq_k_start[schedule_i] = T.ptx.ld(
                    cu_seq_len_k_start.ptr_to([row_idx]), "uint32", "u32", space="global"
                )
                seq_k_end[schedule_i] = T.ptx.ld(
                    cu_seq_len_k_end.ptr_to([row_idx]), "uint32", "u32", space="global"
                )
                schedule_start = T.min(schedule_start, T.min(seq_k_start[schedule_i], seq_len_kv))
                schedule_end = T.max(schedule_end, T.min(seq_k_end[schedule_i], seq_len_kv))
            schedule_start = schedule_start // T.uint32(4) * T.uint32(4)
            schedule_result[0] = (q_iter_idx + q_iter_offset) % T.uint32(num_q_stages)
            schedule_result[1] = (
                (q_iter_idx + q_iter_offset) // T.uint32(num_q_stages)
            ) & T.uint32(1)
            schedule_result[2] = schedule_start
            schedule_result[3] = (
                schedule_end - schedule_start + T.uint32(block_kv - 1)
            ) // T.uint32(block_kv)

        # The Pipeline constructors above already ran mbarrier.init (thread 0); this
        # fence makes those inits visible, and the cta_sync below publishes them
        # CTA-wide before the first wait/arrive.
        T.ptx.fence.mbarrier_init()
        if warp_idx == spec_warp_start + 1:
            T.ptx.tcgen05.alloc(
                T.address_of(tmem_ptr_in_smem[0]), n_cols=num_tmem_cols, cta_group=1
            )
        T.cuda.cta_sync()

        cuda_grid_dependency_synchronize()

        if warp_idx == spec_warp_start:
            T.ptx.setmaxnreg(False, 40)
            q_idx: T.uint32 = sm_idx_u32
            q_iter_idx: T.uint32 = T.uint32(0)
            num_total_kv_blocks: T.uint32 = T.uint32(0)
            if T.ptx.elect_sync():
                if q_idx < num_q_blocks:
                    issue_tma_q(T.uint32(0), q_idx)
            if T.ptx.elect_sync():
                while q_idx < num_q_blocks:
                    load_schedule(q_idx, q_iter_idx, T.uint32(1))
                    q_stage_idx: T.uint32 = schedule_result[0]
                    q_phase: T.uint32 = schedule_result[1]
                    kv_start: T.uint32 = schedule_result[2]
                    num_kv_blocks: T.uint32 = schedule_result[3]
                    q_pipe.empty.wait(q_stage_idx, q_phase ^ T.uint32(1))
                    next_q_idx: T.uint32 = q_idx + T.uint32(config.num_sms)
                    if next_q_idx < num_q_blocks:
                        issue_tma_q(q_stage_idx, next_q_idx)
                    for kv_idx in T.serial(T.uint32(0), num_kv_blocks, unroll=True):
                        kv_stage_idx: T.uint32 = (num_total_kv_blocks + kv_idx) % T.uint32(
                            num_kv_stages
                        )
                        kv_phase: T.uint32 = (
                            (num_total_kv_blocks + kv_idx) // T.uint32(num_kv_stages)
                        ) & T.uint32(1)
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
                            smem_kv_scales[kv_stage_idx, 0:block_kv],
                            kv_scales_gmem[kv_row0 : kv_row0 + block_kv],
                            dispatch="tma",
                            mbar=kv_pipe.full.ptr_to([kv_stage_idx]),
                            cta_group=1,
                            cache_hint="evict_normal",
                        )
                        kv_pipe.full.arrive(
                            kv_stage_idx,
                            tx_count=smem_kv_size_per_stage + smem_kv_scale_size_per_stage,
                        )
                    num_total_kv_blocks = num_total_kv_blocks + num_kv_blocks
                    q_idx = next_q_idx
                    q_iter_idx = q_iter_idx + T.uint32(1)
        elif warp_idx == spec_warp_start + 1:
            T.ptx.setmaxnreg(False, 40)
            tmem_allocated: T.uint32 = T.ptx.ld(
                tmem_ptr_in_smem.ptr_to([0]), "uint32", "u32", space="shared"
            )
            T.cuda.trap_when_assert_failed(tmem_allocated == T.uint32(0))
            # fp8 (e4m3) operand views over the 128B-swizzled uint8 SMEM buffers.
            # T.gemm_async reads them directly; with descI omitted the dense
            # instruction descriptor is constant-folded inside the tcgen05 dispatch,
            # so the hand encode_instr_descriptor / runtime_instr_desc machinery drops.
            smem_q_fp8 = smem_q.view("float8_e4m3fn")
            smem_kv_fp8 = smem_kv.view("float8_e4m3fn")
            q_idx: T.uint32 = sm_idx_u32
            q_iter_idx: T.uint32 = T.uint32(0)
            num_total_kv_blocks: T.uint32 = T.uint32(0)
            while q_idx < num_q_blocks:
                load_schedule(q_idx, q_iter_idx, T.uint32(0))
                q_stage_idx: T.uint32 = schedule_result[0]
                q_phase: T.uint32 = schedule_result[1]
                num_kv_blocks: T.uint32 = schedule_result[3]
                q_pipe.full.wait(q_stage_idx, q_phase)
                kv_idx: T.uint32 = T.uint32(0)
                while kv_idx < num_kv_blocks:
                    kv_stage_idx: T.uint32 = (num_total_kv_blocks + kv_idx) % T.uint32(
                        num_kv_stages
                    )
                    kv_phase: T.uint32 = (
                        (num_total_kv_blocks + kv_idx) // T.uint32(num_kv_stages)
                    ) & T.uint32(1)
                    kv_pipe.full.wait(kv_stage_idx, kv_phase)
                    for math_wg_i in T.unroll(0, num_math_warpgroups):
                        umma_pipe.empty.wait(
                            math_wg_i, ((num_total_kv_blocks + kv_idx) & T.uint32(1)) ^ T.uint32(1)
                        )
                        # D = KV @ Q^T over the full head_dim K in one issue; the
                        # tcgen05 dispatch unrolls the umma_k tiling and accumulates
                        # internally (accum=False overwrites tmem, matching the
                        # hand-rolled enable_input_d=(k!=0) accumulation chain).
                        # Tx.warp. prefix -> warp scope, so the dispatch wraps each
                        # tcgen05.mma in its own elect_sync so a single lane issues
                        # it (a bare thread-scope Tx.gemm_async would leave all 32
                        # lanes issuing the MMA).
                        Tx.warp.gemm_async(
                            tmem[:, math_wg_i * umma_n : math_wg_i * umma_n + umma_n],
                            smem_kv_fp8[
                                kv_stage_idx, math_wg_i * umma_m : math_wg_i * umma_m + umma_m, :
                            ],
                            smem_q_fp8[q_stage_idx, :, :],
                            accum=False,
                            dispatch="tcgen05",
                            cta_group=1,
                            smem_desc="recompute",
                        )
                        if T.ptx.elect_sync():
                            umma_pipe.full.arrive(math_wg_i)
                    kv_idx = kv_idx + T.uint32(1)
                num_total_kv_blocks = num_total_kv_blocks + num_kv_blocks
                q_pipe.empty.arrive(q_stage_idx)
                q_idx = q_idx + T.uint32(config.num_sms)
                q_iter_idx = q_iter_idx + T.uint32(1)
        elif warp_idx == spec_warp_start + 2 or warp_idx == spec_warp_start + 3:
            T.ptx.setmaxnreg(False, 40)
        elif warp_idx < spec_warp_start:
            T.ptx.setmaxnreg(True, 232)
            warpgroup_idx_local: T.int32 = warp_idx // 4
            tmem_start: T.uint32 = T.cast(warpgroup_idx_local, "uint32") * T.uint32(umma_n)
            math_thread_idx: T.uint32 = warp_idx_u32 * T.uint32(32) + lane_idx_u32
            accum = T.alloc_local((num_heads,), "float32")
            # Flat (block_q*num_heads,) so a per-q-row slice
            # cached_weights[q_inner_i*num_heads : +num_heads] is rank-1 — the tile
            # T.mul anchor (accum) is rank-1, and a 2-D [q,:] slice would be rank-2.
            cached_weights = T.alloc_local((block_q * num_heads,), "float32")
            wrelu_acc = T.alloc_local((WRELU_K,), "float32")
            q_idx: T.uint32 = sm_idx_u32
            q_iter_idx: T.uint32 = T.uint32(0)
            num_total_kv_blocks: T.uint32 = T.uint32(0)
            while q_idx < num_q_blocks:
                load_schedule(q_idx, q_iter_idx, T.uint32(0))
                q_stage_idx: T.uint32 = schedule_result[0]
                q_phase: T.uint32 = schedule_result[1]
                kv_start: T.uint32 = schedule_result[2]
                num_kv_blocks: T.uint32 = schedule_result[3]
                q_pipe.full.wait(q_stage_idx, q_phase)
                for weight_i in T.unroll(0, block_q):
                    for weight_j in T.unroll(0, num_heads):
                        cached_weights[weight_i * num_heads + weight_j] = T.ptx.ld(
                            smem_weights.ptr_to([q_stage_idx, weight_i, weight_j]),
                            "float32",
                            "f32",
                            space="shared",
                        )
                kv_idx: T.uint32 = T.uint32(0)
                while kv_idx < num_kv_blocks:
                    kv_stage_idx: T.uint32 = (num_total_kv_blocks + kv_idx) % T.uint32(
                        num_kv_stages
                    )
                    kv_phase: T.uint32 = (
                        (num_total_kv_blocks + kv_idx) // T.uint32(num_kv_stages)
                    ) & T.uint32(1)
                    kv_pipe.full.wait(kv_stage_idx, kv_phase)
                    scale_kv: T.float32 = T.ptx.ld(
                        smem_kv_scales.ptr_to([kv_stage_idx, math_thread_idx]),
                        "float32",
                        "f32",
                        space="shared",
                    )
                    umma_pipe.full.wait(
                        warpgroup_idx_local, (num_total_kv_blocks + kv_idx) & T.uint32(1)
                    )
                    kv_pipe.empty.arrive(kv_stage_idx)
                    kv_offset: T.uint32 = kv_start + kv_idx * T.uint32(block_kv) + math_thread_idx
                    for q_inner_i in T.unroll(0, block_q):
                        tmem_addr: T.uint32 = tmem_start + T.uint32(q_inner_i * num_heads)
                        # TMEM->register read of the logits accumulator via tile
                        # primitive. accum stays a flat per-thread (num_heads,) buffer
                        # for the wrelu reduce below; here take a 2-D (128, num_heads)
                        # view of the SAME per-thread bytes so the tmem<->local
                        # copy_async (.32x32b path) re-emits the identical
                        # tcgen05.ld.32x32b.x{num_heads}. uint32 column start is fine:
                        # the dispatch divisibility proof is robust for fp32 tmem
                        # (elem_per_32b == 1 is always divisible).
                        accum_2d = accum.view(128, num_heads, layout=wg_local_layout(num_heads))
                        Tx.warpgroup.copy_async(
                            accum_2d, tmem[:, tmem_addr : tmem_addr + num_heads]
                        )
                        T.ptx.tcgen05.wait.ld()
                        if q_inner_i == block_q - 1:
                            umma_pipe.empty.arrive(warpgroup_idx_local)
                        # Weighted-ReLU reduce as tile primitives (replacing the hand
                        # make_float2 + fma_f32x2 chain): result = scale_kv * sum_h
                        # max(accum[h], 0) * weights[h]. accum is per-thread private
                        # storage — use thread-scope elementwise (not Tx.warp; warp
                        # scope requires a laneid-distributed layout).
                        Tx.maximum(accum, accum, T.float32(0))
                        emit_wrelu_reduce(
                            accum, wrelu_acc, cached_weights, q_inner_i * num_heads, Tx
                        )
                        result_f32: T.let = scale_kv * wrelu_acc[0]
                        result = T.cast(result_f32, logits_tir_dtype)
                        q_offset: T.uint64 = T.cast(
                            q_idx * T.uint32(block_q) + T.uint32(q_inner_i), "uint64"
                        ) * T.cast(logits_stride, "uint64")
                        if config.compressed_logits:
                            row_k_start: T.uint32 = seq_k_start[q_inner_i]
                            row_k_end: T.uint32 = seq_k_end[q_inner_i]
                            # Build the flat index with per-operand u64 casts
                            # `q_offset + (u64)kv - (u64)rks`, NOT `(u64)(kv - rks)`:
                            # the latter makes ptxas branch around the 16-bit bf16
                            # store (BSYNC + unpredicated STG.E.U16), while the
                            # per-operand form if-converts the guard to a predicated
                            # `@P STG.E.U16` — matching DeepGEMM's epilogue and ~6%
                            # faster on bf16xcompressed (verified in SASS + tir-bench).
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
                num_total_kv_blocks = num_total_kv_blocks + num_kv_blocks
                q_pipe.empty.arrive(q_stage_idx)
                q_idx = q_idx + T.uint32(config.num_sms)
                q_iter_idx = q_iter_idx + T.uint32(1)
            named_barrier_sync_8(T.uint32(num_math_threads))
            if warp_idx == 0:
                T.ptx.tcgen05.dealloc(T.uint32(0), n_cols=num_tmem_cols, cta_group=1)

    sm100_fp8_mqa_logits = sm100_fp8_mqa_logits.with_attr("tirx.persistent_kernel", True)

    return sm100_fp8_mqa_logits.with_attr(
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

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100f"})
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


def _compile_tirx_mqa(config: MQALogitsFP8Config, max_seqlen_k: int) -> Any:
    # The generated kernel no longer depends on seq_len / seq_len_kv / disable_cp /
    # logits_stride (all RUNTIME now via the match_buffer'd gmem buffers + runtime
    # logits_stride). Pass canonical values so the compile cache dedups to ONE
    # kernel per *structural* config (num_heads, head_dim, logits_dtype,
    # compressed_logits, num_sms) — like DeepGEMM, which templates on structure and
    # takes the lengths at runtime.
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


def _logits_storage_shape(config: MQALogitsFP8Config, max_seqlen_k: int) -> tuple[int, int]:
    if config.compressed_logits:
        stride = _align_up(max_seqlen_k, config.block_kv)
    else:
        stride = _align_up(config.seq_len_kv + config.block_kv, 8)
    return config.aligned_seq_len, stride


def _allocate_logits(config: MQALogitsFP8Config, max_seqlen_k: int) -> torch.Tensor:
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
    config: MQALogitsFP8Config = data["config"]
    if logits is None:
        logits = _allocate_logits(config, data["max_seqlen_k"])
    if executable is None:
        executable = _compile_tirx_mqa(config, data["max_seqlen_k"])
    return {"executable": executable, "logits": logits}


def _run_tirx_invocation(data: dict[str, Any], invocation: dict[str, Any]) -> torch.Tensor:
    config: MQALogitsFP8Config = data["config"]
    executable = invocation["executable"]
    logits = invocation["logits"]
    # Raw GMEM buffers (replacing the host-built cuTensorMap descriptors): the
    # copy_async(dispatch="tma") in-kernel builds the descriptor from the buffer
    # layouts. Q/KV move as uint8 (the fp8 e4m3 bytes reinterpreted); KV-scales
    # and weights stay float32.
    kv_fp8, kv_scales = data["kv_in"]
    q_gmem = (
        data["q_in"].view(torch.uint8).reshape(config.seq_len * config.num_heads, config.head_dim)
    )
    kv_gmem = kv_fp8.view(torch.uint8)
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
        kv_gmem,
        kv_scales,
        data["weights"],
    )
    return logits


def _launch_tirx_mqa(data: dict[str, Any], logits: torch.Tensor | None = None) -> torch.Tensor:
    return _run_tirx_invocation(data, _prepare_tirx_invocation(data, logits))


def _run_deepgemm_mqa(data: dict[str, Any], *, clean_logits: bool) -> torch.Tensor:
    config: MQALogitsFP8Config = data["config"]
    return data["deep_gemm"].fp8_fp4_mqa_logits(
        q=(data["q_in"], None),
        kv=data["kv_in"],
        weights=data["weights"],
        cu_seq_len_k_start=data["cu_seq_len_k_start"],
        cu_seq_len_k_end=data["cu_seq_len_k_end"],
        clean_logits=clean_logits,
        max_seqlen_k=data["max_seqlen_k"],
        logits_dtype=_torch_logits_dtype(config.logits_dtype),
    )


def _expand_compressed_logits(logits: torch.Tensor, data: dict[str, Any]) -> torch.Tensor:
    config: MQALogitsFP8Config = data["config"]
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
    config: MQALogitsFP8Config = data["config"]
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
    max_diff: float | None = None
    tirx_executable = _compile_tirx_mqa(_make_config(**config_kwargs), 0)

    def make_input() -> tuple[tuple[dict[str, Any], dict[str, Any]], int]:
        data = prepare_data(**config_kwargs)
        invocation = _prepare_tirx_invocation(data, executable=tirx_executable)
        return (data, invocation), tensor_bytes(
            data["q_in"],
            data["kv_in"],
            data["weights"],
            data["cu_seq_len_k_start"],
            data["cu_seq_len_k_end"],
            invocation["logits"],
        )

    def validate_case(case: tuple[dict[str, Any], dict[str, Any]]) -> None:
        nonlocal max_diff
        data, invocation = case
        tirx_logits = _run_tirx_invocation(data, invocation)
        torch.cuda.synchronize()
        max_diff = _assert_correct(data, tirx_logits, name="TIRx")
        torch.cuda.empty_cache()

    result = bench(
        {
            "tirx": lambda case: _run_tirx_invocation(case[0], case[1]),
            "deepgemm": lambda case: _run_deepgemm_mqa(case[0], clean_logits=False),
        },
        make_input,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=_rounds,
        round_cooldown_s=_round_cooldown_s,
        proton_name="deepgemm_sm100_fp8_mqa_logits",
        validate_case=validate_case,
    )
    result["errors"].update(errors)
    if max_diff is not None:
        result["max_diff"] = max_diff
    return result


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "DEEPGEMM_TEST_COVERAGE",
    "KERNEL_META",
    "MQALogitsFP8Config",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
]

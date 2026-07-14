from __future__ import annotations

import math
from dataclasses import dataclass, fields
from functools import partial
from typing import Any
from unittest import SkipTest

import torch

from tirx_kernels.flashmla._gemm import tcgen05_config
from tirx_kernels.flashmla._mask import pack_valid_mask8
from tirx_kernels.flashmla._tma import tma_config
from tvm.backend.cuda.operator.tile_primitive.tma_utils import SwizzleMode
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.lang.pipeline import MBarrier, TCGen05Bar, TMABar
from tvm.tirx.layout import S, TileLayout, laneid, wid_in_wg

B_H = 64
B_TOPK = 64
D_V = 512
NUM_BUFS = 3
MAX_INIT_VAL = -1.0e30
LOG_2_E = math.log2(math.e)
LN_2 = math.log(2.0)

LAUNCH_TAGS = ("blockIdx.x", "threadIdx.x", "tirx.use_dyn_shared_memory")

BAR_WG0_SYNC = 0
BAR_WG0_WARP02 = 1

BF16_BYTES = 2
Q_ROPE_DIM = 64
WG1_NUM_WARPS = 4
WG1_ROWS_PER_WARP = (B_TOPK // 4) // WG1_NUM_WARPS

# KV gather4 TMA knobs shared by both gather call sites.
_mma_config = partial(tcgen05_config, cta_group=1)
_kv_gather_tma = partial(
    tma_config,
    cta_group=1,
    gather_axis=0,
    dst_gather_axis=1,
    cache_hint=T.uint64(0x14F0000000000000),
)


@dataclass(frozen=True)
class SparseFlashMLAPrefillHead64Config:
    label: str
    s_q: int
    s_kv: int
    topk: int
    d_qk: int = 576
    h_q: int = B_H
    h_kv: int = 1
    d_v: int = D_V
    have_attn_sink: bool = False
    have_topk_length: bool = False
    inject_invalid_indices: bool = False
    seed: int = 0

    def validate(self) -> None:
        if self.h_q != B_H:
            raise ValueError("head64 regular phase1 requires h_q == 64")
        if self.h_kv != 1:
            raise ValueError("head64 regular phase1 requires h_kv == 1")
        if self.d_qk not in (512, 576):
            raise ValueError("d_qk must be 512 or 576")
        if self.d_v != D_V:
            raise ValueError("d_v must be 512")
        if self.topk % B_TOPK != 0:
            raise ValueError("topk must be a multiple of 64")


# Cover the two upstream fwd/head64 phase1 instantiations:
# D_QK=512 and D_QK=576, h_q=64, topk=512 at the scoped s_kv values.
CONFIGS = [
    {
        "label": f"bench_dqk{d_qk}_hq64_s4096_kv{s_kv}_topk512",
        "s_q": 4096,
        "s_kv": s_kv,
        "topk": 512,
        "d_qk": d_qk,
        "h_q": B_H,
        "have_attn_sink": True,
    }
    for d_qk in (512, 576)
    for s_kv in (8192, 32768, 49152, 65536)
]

KERNEL_META = {
    "name": "sparse_flashmla_prefill_head64_phase1",
    "category": "flashmla",
    "compute_capability": 10,
}


def _cfg(**kwargs: Any) -> SparseFlashMLAPrefillHead64Config:
    cfg_fields = {field.name for field in fields(SparseFlashMLAPrefillHead64Config)}
    cfg_kwargs = {key: value for key, value in kwargs.items() if key in cfg_fields}
    if "label" not in cfg_kwargs:
        cfg_kwargs["label"] = "custom"
    cfg = SparseFlashMLAPrefillHead64Config(**cfg_kwargs)
    cfg.validate()
    return cfg


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    cfg = _cfg(**kwargs)
    device = kwargs.get("device", "cuda")
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed)

    q = torch.randn(
        (cfg.s_q, cfg.h_q, cfg.d_qk), device=device, dtype=torch.bfloat16, generator=gen
    )
    kv = torch.randn(
        (cfg.s_kv, cfg.h_kv, cfg.d_qk), device=device, dtype=torch.bfloat16, generator=gen
    )
    out = torch.empty((cfg.s_q, cfg.h_q, cfg.d_v), device=device, dtype=torch.bfloat16)
    max_logits = torch.empty((cfg.s_q, cfg.h_q), device=device, dtype=torch.float32)
    lse = torch.empty((cfg.s_q, cfg.h_q), device=device, dtype=torch.float32)

    indices = torch.randint(
        low=0,
        high=cfg.s_kv,
        size=(cfg.s_q, cfg.h_kv, cfg.topk),
        device=device,
        dtype=torch.int32,
        generator=gen,
    )
    if cfg.inject_invalid_indices:
        indices[:, :, 0] = -1
        indices[:, :, 1] = cfg.s_kv
        indices[:, :, 2] = cfg.s_kv + 17
        indices[:, :, -1] = -7
    attn_sink = (
        torch.randn((cfg.h_q,), device=device, dtype=torch.float32, generator=gen)
        if cfg.have_attn_sink
        else torch.empty((cfg.h_q,), device=device, dtype=torch.float32)
    )
    if cfg.have_topk_length:
        topk_length = torch.randint(
            low=0,
            high=cfg.topk + 1,
            size=(cfg.s_q,),
            device=device,
            dtype=torch.int32,
            generator=gen,
        )
    else:
        topk_length = torch.empty((cfg.s_q,), device=device, dtype=torch.int32)

    sm_scale = 1.0 / math.sqrt(cfg.d_qk)
    return {
        "config": cfg,
        "q": q,
        "kv": kv,
        "indices": indices,
        "attn_sink": attn_sink,
        "topk_length": topk_length,
        "out": out,
        "max_logits": max_logits,
        "lse": lse,
        "sm_scale": sm_scale,
        "sm_scale_div_log2": sm_scale * LOG_2_E,
    }


def _reference_sparse_prefill(
    case: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg: SparseFlashMLAPrefillHead64Config = case["config"]
    q = case["q"].float()
    kv = case["kv"][:, 0, :].float()
    indices = case["indices"][:, 0, :].to(torch.long)
    sm_scale = case["sm_scale"]
    ref_out = torch.zeros((cfg.s_q, cfg.h_q, cfg.d_v), device=q.device, dtype=torch.float32)
    ref_max_logits = torch.full((cfg.s_q, cfg.h_q), -float("inf"), device=q.device)
    ref_lse = torch.full((cfg.s_q, cfg.h_q), float("inf"), device=q.device)

    for s_q_idx in range(cfg.s_q):
        length = int(case["topk_length"][s_q_idx].item()) if cfg.have_topk_length else cfg.topk
        row_indices = indices[s_q_idx]
        pos = torch.arange(cfg.topk, device=q.device)
        valid = (pos < length) & (row_indices >= 0) & (row_indices < cfg.s_kv)
        if not torch.any(valid):
            continue
        selected = row_indices.clamp(0, cfg.s_kv - 1)
        k_full = kv[selected]
        logits = torch.matmul(q[s_q_idx], k_full[:, : cfg.d_qk].T) * sm_scale
        logits[:, ~valid] = -float("inf")
        max_logits = torch.max(logits, dim=-1).values
        exp_logits = torch.exp(logits - max_logits[:, None])
        exp_logits[:, ~valid] = 0.0
        denom = torch.sum(exp_logits, dim=-1)
        if cfg.have_attn_sink:
            sink = case["attn_sink"].float()
            denom_with_sink = denom + torch.exp(sink - max_logits)
        else:
            denom_with_sink = denom
        ref_out[s_q_idx] = torch.matmul(exp_logits, k_full[:, : cfg.d_v]) / denom_with_sink[:, None]
        ref_max_logits[s_q_idx] = max_logits
        ref_lse[s_q_idx] = max_logits + torch.log(denom)
    return ref_out.to(torch.bfloat16), ref_max_logits, ref_lse


def _tirx_args(case: dict[str, Any]) -> tuple[Any, ...]:
    return (
        case["q"],
        case["kv"].reshape(-1),
        case["indices"].reshape(-1),
        case["attn_sink"],
        case["topk_length"],
        case["out"],
        case["max_logits"],
        case["lse"],
    )


def _ring_mod3(value: Any, max_value: int) -> Any:
    if max_value <= 8:
        packed_mod3 = T.uint32(0x10210210)
        shift = T.cast(value, "uint32") * T.uint32(4)
        return T.cast(T.bitwise_and(T.shift_right(packed_mod3, shift), T.uint32(0xF)), "int32")

    max_offset = (max_value // NUM_BUFS) * NUM_BUFS
    result = value - max_offset
    for offset in range(max_offset, 0, -NUM_BUFS):
        result = T.Select(value < offset, value - (offset - NUM_BUFS), result)
    return result


def _ring_phase_parity(value: Any, max_value: int) -> Any:
    if max_value <= 8:
        packed_phase = T.uint32(0x38)
        return T.cast(
            T.bitwise_and(T.shift_right(packed_phase, T.cast(value, "uint32")), T.uint32(1)),
            "int32",
        )

    max_offset = (max_value // NUM_BUFS) * NUM_BUFS
    result = T.int32((max_offset // NUM_BUFS) & 1)
    for offset in range(max_offset, 0, -NUM_BUFS):
        result = T.Select(value < offset, T.int32(((offset - NUM_BUFS) // NUM_BUFS) & 1), result)
    return result


@T.jit
def _kernel(
    q: T.Buffer((s_q, h_q, d_qk), "bfloat16"),
    kv: T.Buffer((s_kv * stride_kv_s_kv,), "bfloat16"),
    indices: T.Buffer((s_q * stride_indices_s_q,), "int32"),
    attn_sink: T.Buffer((h_q,), "float32"),
    topk_length: T.Buffer((s_q,), "int32"),
    out: T.Buffer((s_q, h_q, D_V), "bfloat16"),
    max_logits: T.Buffer((s_q, h_q), "float32"),
    lse: T.Buffer((s_q, h_q), "float32"),
    *,
    s_q: T.constexpr,
    s_kv: T.constexpr,
    topk: T.constexpr,
    d_qk: T.constexpr,
    h_q: T.constexpr,
    stride_kv_s_kv: T.constexpr,
    stride_indices_s_q: T.constexpr,
    have_attn_sink: T.constexpr,
    have_topk_length: T.constexpr,
    sm_scale_div_log2: T.constexpr,
):
    T.device_entry()
    T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})
    # CUDA_TRANSCRIBE_START: sparse_attn_fwd_kernel lines 65-71. One CTA per query row;
    # warp 0 owns Q TMA, warps 0-1 own O TMA, warpgroup 0 also does softmax/epilogue.
    s_q_idx = T.cta_id([s_q])
    warpgroup_idx = T.warpgroup_id([3])
    warp_idx_in_wg = T.warp_id_in_wg([4])
    lane_idx = T.lane_id([32])
    idx_in_warpgroup = T.thread_id_in_wg([128])
    warp_idx: T.let = warpgroup_idx * 4 + warp_idx_in_wg
    topk_len: T.let = topk_length[s_q_idx] if have_topk_length else topk
    max_k_blocks = T.meta_var(topk // B_TOPK)
    num_k_blocks: T.let = T.max((topk_len + B_TOPK - 1) // B_TOPK, 1)
    have_rope = T.meta_var(d_qk == 576)

    # CUDA phase1.cuh:73-78, config.h:111-139. Reserve SharedMemoryPlan offsets now;
    # instantiate bf16 MMA views only at their use sites (unused ones trip BF16 legalization).
    pool = T.SMEMPool()
    u_base = T.meta_var(pool.offset)
    k_rope = pool.alloc_tcgen05_mma_AB(
        (B_TOPK, Q_ROPE_DIM), "bfloat16", swizzle_mode=SwizzleMode.SWIZZLE_64B_ATOM
    )
    # *_tiled_mma refolds stay here: the gemm B-operand descriptor hoists to the decl site.
    k_rope_tiled_mma = k_rope.rearrange("r (h c) -> (h r) c", h=2)
    k_nope = pool.alloc_tcgen05_mma_AB((NUM_BUFS, B_TOPK, D_V), "bfloat16")
    k_nope_tiled_mma = k_nope.rearrange("b r (dc h ci) -> b (h r) (dc ci)", dc=4, h=2, ci=64)
    u_end = T.meta_var(pool.offset)
    # q_nope aliases the last k_nope stage: Q moves to TMEM before that stage is used.
    pool.move_base_to(u_end - B_H * D_V * BF16_BYTES)
    q_nope = pool.alloc_tcgen05_mma_AB((B_H, D_V), "bfloat16")
    # o_smem aliases the front of the region: O is only written in the epilogue.
    pool.move_base_to(u_base)
    o_smem = pool.alloc_tcgen05_mma_AB((B_H, D_V), "bfloat16")
    pool.move_base_to(u_end)

    p_exchange_buf = pool.alloc((4, 32 * (B_TOPK // 2)), "float32")
    s_q_rope_base = T.meta_var(pool.offset)
    q_rope = pool.alloc_tcgen05_mma_AB(
        (B_H, Q_ROPE_DIM), "bfloat16", swizzle_mode=SwizzleMode.SWIZZLE_64B_ATOM
    )
    q_rope_end = T.meta_var(pool.offset)
    # s_smem_gemm aliases q_rope: Q RoPE moves to TMEM before the first S tile is stored.
    pool.move_base_to(s_q_rope_base)
    s_smem_gemm = pool.alloc_tcgen05_mma_AB(
        (B_H, B_TOPK), "bfloat16", swizzle_mode=SwizzleMode.SWIZZLE_NONE
    )
    pool.move_base_to(q_rope_end)

    is_k_valid = pool.alloc((NUM_BUFS, B_TOPK // 8), "int8")
    bar_prologue_q_nope = TMABar(pool, 1)
    bar_prologue_q_rope = TMABar(pool, 1)
    bar_prologue_utccp_nope = TCGen05Bar(pool, 1)
    bar_prologue_utccp_rope = TCGen05Bar(pool, 1)
    bar_qk_nope_done = TCGen05Bar(pool, NUM_BUFS)
    bar_qk_rope_done = TCGen05Bar(pool, 1)
    bar_sv_done = TCGen05Bar(pool, NUM_BUFS)
    bar_kv_nope_ready_part0 = TMABar(pool, NUM_BUFS)
    bar_kv_nope_ready_part1 = TMABar(pool, NUM_BUFS)
    bar_kv_rope_ready = MBarrier(pool, 1)
    bar_p_free = MBarrier(pool, 1)
    bar_so_ready = MBarrier(pool, 1)
    bar_k_valid_ready = MBarrier(pool, NUM_BUFS)
    bar_k_valid_free = MBarrier(pool, NUM_BUFS)
    tmem_start_addr = pool.alloc((1,), "uint32", align=4)
    rowwise_max_buf = pool.alloc((128,), "float32")
    rowwise_li_buf = pool.alloc((128,), "float32")
    pool.commit()

    # CUDA phase1.cuh:77. h_kv is fixed to 1, so the row pointer is
    # params.indices + s_q_idx * params.stride_indices_s_q.
    g_indices_base: T.let = s_q_idx * stride_indices_s_q
    tmem_pool = T.TMEMPool(pool, total_cols=512, cta_group=1, tmem_addr=tmem_start_addr)
    # O accumulator: one alloc. Col halves [0:256)/[256:512) = E lo/hi gemm outputs;
    # reads back as a plain (128, 256) datapath-D tile.
    o_tmem = tmem_pool.alloc_tcgen05_mma_D(
        (B_H, D_V), "float32", M=64, cta_group=1, ws=True, group=(2, 2, 128)
    )
    o_win = o_tmem.rearrange("h (a b c) -> (b h) (a c)", a=2, b=2, c=128)
    q_nope_tmem_bmm = tmem_pool.alloc_tcgen05_mma_A(
        (2, B_H, D_V // 2), "bfloat16", M=64, cta_group=1, ws=True
    )
    q_rope_tmem_bmm = tmem_pool.alloc_tcgen05_mma_A(
        (2, B_H, Q_ROPE_DIM // 2), "bfloat16", M=64, cta_group=1, ws=True
    )
    tmem_p_col = T.meta_var(tmem_pool.offset)
    # .ws logits gemm C: two batched 64x64 lane-half partials.
    tmem_p_bmm = tmem_pool.alloc_tcgen05_mma_D(
        (2, B_H, B_TOPK), "float32", M=64, cta_group=1, ws=True
    )
    mma_p_accumulate: T.uint32 = 0
    mma_o_accumulate: T.uint32 = 0
    mma_smem_desc = T.meta_var("local_hoist" if (d_qk > D_V and s_kv == 8192) else "hoist")

    # CUDA phase1.cuh:100-150.  Warp 0 performs descriptor prefetch, Q TMA
    # launch, prologue barrier init, and TMEM allocation.
    if warp_idx == 0:
        if T.ptx.elect_sync():
            bar_prologue_q_nope.init(1)
            bar_prologue_q_rope.init(1)
            T.ptx.fence.mbarrier_init()

            if have_rope:
                Tx.copy_async(
                    q_rope[:, :],
                    q[s_q_idx : s_q_idx + 1, :, D_V : D_V + Q_ROPE_DIM],
                    **tma_config(
                        mbar=bar_prologue_q_rope.ptr_to([0]), cta_group=1, cache_hint="evict_first"
                    ),
                )

            Tx.copy_async(
                q_nope[:, :],
                q[s_q_idx : s_q_idx + 1, :, 0:D_V],
                **tma_config(
                    mbar=bar_prologue_q_nope.ptr_to([0]), cta_group=1, cache_hint="evict_first"
                ),
            )
            bar_prologue_utccp_rope.init(1)
            bar_prologue_utccp_nope.init(1)
            if bar_qk_nope_done.leader:
                for init_stage in T.unroll(NUM_BUFS):
                    T.ptx.mbarrier.init(bar_qk_nope_done.ptr_to([init_stage]), 1)
                    T.ptx.mbarrier.init(bar_sv_done.ptr_to([init_stage]), 1)
                    T.ptx.mbarrier.init(bar_kv_nope_ready_part0.ptr_to([init_stage]), 1)
                    T.ptx.mbarrier.init(bar_kv_nope_ready_part1.ptr_to([init_stage]), 1)
                    T.ptx.mbarrier.init(bar_k_valid_ready.ptr_to([init_stage]), B_TOPK // 8)
                    T.ptx.mbarrier.init(bar_k_valid_free.ptr_to([init_stage]), 128)
            bar_p_free.init(128)
            bar_so_ready.init(128)
            bar_qk_rope_done.init(1)
            bar_kv_rope_ready.init(64)
            T.ptx.fence.mbarrier_init()

        T.ptx.tcgen05.alloc(T.address_of(tmem_start_addr[0]), n_cols=512, cta_group=1)
        T.cuda.trap_when_assert_failed(tmem_start_addr[0] == T.uint32(0))
        T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)

    T.cuda.cta_sync()

    if warpgroup_idx == 0:
        # CUDA phase1.cuh:152-168.  Scale/exp warpgroup state.
        mi: T.float32 = MAX_INIT_VAL
        li: T.float32 = 0.0
        real_mi: T.float32 = T.float32(-float("inf"))

        # CUDA phase1.cuh:169-244. Scale/exp loop: P TMEM read/mask/reduce, row max,
        # S generation, S shared store, conditional O rescale.
        for k in T.serial(0, num_k_blocks, unroll=False):
            T.ptx.bar.sync(BAR_WG0_WARP02 + T.bitwise_and(warp_idx, T.int32(1)), 64)
            cur_buf: T.int32 = _ring_mod3(k, max_k_blocks)
            cur_phase: T.int32 = _ring_phase_parity(k, max_k_blocks)
            bar_qk_nope_done.wait(cur_buf, cur_phase)
            bar_k_valid_ready.wait(cur_buf, cur_phase)
            T.ptx.tcgen05.fence.after_thread_sync()

            # CUDA common_subroutine.h:75-134 retrieve_mask_and_reduce_p.
            p_frag = T.alloc_tcgen05_ldst_frag("32x32b", (128, (B_TOPK // 2)), "float32")
            p_peer_frag = T.alloc_tcgen05_ldst_frag("32x32b", (128, (B_TOPK // 2)), "float32")
            p = p_frag.local()
            p_peer = p_peer_frag.local()

            @T.inline
            def load_p(lo_dst, hi_dst):
                p_win = tmem_p_bmm.rearrange("b h t -> (b h) t")
                Tx.wg.copy_async(lo_dst[:, :], p_win.chunk((None, 2))[:, 0])
                Tx.wg.copy_async(hi_dst[:, :], p_win.chunk((None, 2))[:, 1])

            if warp_idx < 2:
                load_p(p_frag, p_peer_frag)
            else:
                load_p(p_peer_frag, p_frag)
            T.ptx.tcgen05.wait.ld()
            T.ptx.tcgen05.fence.before_thread_sync()
            bar_p_free.arrive(0)

            valid_word_offset: T.int32 = T.if_then_else(warp_idx >= 2, (B_TOPK // 2) // 32, 0)
            is_k_valid_u32: T.let = is_k_valid.view("uint32")[cur_buf, valid_word_offset]
            for p_i in T.unroll(B_TOPK // 2):
                invalid_p_predicate: T.let = T.bitwise_and(
                    T.shift_right(is_k_valid_u32, T.uint32(p_i)), T.uint32(1)
                ) == T.uint32(0)
                p[p_i] = T.cuda.uint_as_float(
                    T.if_then_else(
                        invalid_p_predicate, T.uint32(0xFF800000), T.cuda.float_as_uint(p[p_i])
                    )
                )

            for exchange_i in T.unroll((B_TOPK // 2) // 4):
                exchange_offset = exchange_i * 32 * 4 + lane_idx * 4
                p_peer_offset: T.let = exchange_i * 4
                Tx.copy(
                    p_exchange_buf[warp_idx ^ 2, exchange_offset : exchange_offset + 4],
                    p_peer[p_peer_offset : p_peer_offset + 4],
                    dispatch="vec_128b",
                )
            T.ptx.bar.sync(BAR_WG0_WARP02 + T.bitwise_and(warp_idx, T.int32(1)), 64)
            for exchange_i in T.unroll((B_TOPK // 2) // 4):
                exchange_offset = exchange_i * 32 * 4 + lane_idx * 4
                p_exchange_tmp = T.alloc_local((4,), "float32")
                Tx.copy(
                    p_exchange_tmp[0:4],
                    p_exchange_buf[warp_idx, exchange_offset : exchange_offset + 4],
                    dispatch="vec_128b",
                )
                p_pair0: T.let = T.cuda.make_float2(p[exchange_i * 4], p[exchange_i * 4 + 1])
                peer_pair0: T.let = T.cuda.make_float2(p_exchange_tmp[0], p_exchange_tmp[1])
                p_add_pair0: T.let = T.ptx.add_f32x2(p_pair0, peer_pair0, dps=False)
                p[exchange_i * 4] = T.cuda.float2_x(p_add_pair0)
                p[exchange_i * 4 + 1] = T.cuda.float2_y(p_add_pair0)
                p_pair1: T.let = T.cuda.make_float2(p[exchange_i * 4 + 2], p[exchange_i * 4 + 3])
                peer_pair1: T.let = T.cuda.make_float2(p_exchange_tmp[2], p_exchange_tmp[3])
                p_add_pair1: T.let = T.ptx.add_f32x2(p_pair1, peer_pair1, dps=False)
                p[exchange_i * 4 + 2] = T.cuda.float2_x(p_add_pair1)
                p[exchange_i * 4 + 3] = T.cuda.float2_y(p_add_pair1)

            bar_k_valid_free.arrive(cur_buf)

            cur_pi_max: T.float32 = T.float32(-float("inf"))
            for p_i in T.unroll(B_TOPK // 2):
                cur_pi_max = T.max(cur_pi_max, p[p_i])
            cur_pi_max = cur_pi_max * sm_scale_div_log2
            rowwise_max_buf[idx_in_warpgroup] = cur_pi_max
            T.ptx.bar.sync(BAR_WG0_SYNC, 128)
            cur_pi_max = T.max(cur_pi_max, rowwise_max_buf[idx_in_warpgroup ^ 64])
            real_mi = T.max(real_mi, cur_pi_max)
            should_scale_o: T.bool = (
                T.ptx.any_sync(T.uint32(0xFFFFFFFF), cur_pi_max - mi > 6.0) != 0
            )
            new_max: T.float32
            scale_for_old: T.float32
            if not should_scale_o:
                scale_for_old = 1.0
                new_max = mi
            else:
                new_max = T.max(cur_pi_max, mi)
                scale_for_old = T.ptx.exp2(mi - new_max)
            mi = new_max

            # S frag: warpgroup-distributed (B_H, B_TOPK) tile.
            s_frag = T.alloc_buffer(
                (B_H, B_TOPK),
                "bfloat16",
                scope="local",
                layout=TileLayout(
                    S[(2, 32, 2, 32) : (1 @ wid_in_wg, 1 @ laneid, 2 @ wid_in_wg, 1)]
                ),
            )
            s_pack = s_frag.local().view("uint32")
            cur_sum_pair: T.uint64 = T.cuda.make_float2(T.float32(0.0), T.float32(0.0))
            neg_new_max_pair: T.let = T.cuda.make_float2(-new_max, -new_max)
            scale_pair: T.let = T.cuda.make_float2(sm_scale_div_log2, sm_scale_div_log2)
            for s_i in T.unroll((B_TOPK // 2) // 2):
                p_pair: T.let = T.cuda.make_float2(p[s_i * 2], p[s_i * 2 + 1])
                fma_pair: T.let = T.ptx.fma_f32x2(p_pair, scale_pair, neg_new_max_pair, dps=False)
                s_x: T.let = T.ptx.exp2(T.cuda.float2_x(fma_pair))
                s_y: T.let = T.ptx.exp2(T.cuda.float2_y(fma_pair))
                s_pair: T.let = T.cuda.make_float2(s_x, s_y)
                cur_sum_pair = T.ptx.add_f32x2(cur_sum_pair, s_pair, dps=False)
                s_pack[s_i] = T.cuda.float22bfloat162_rn(s_x, s_y)
            cur_sum: T.let = T.cuda.float2_x(cur_sum_pair) + T.cuda.float2_y(cur_sum_pair)
            li_tmp: T.float32
            T.ptx.fma_f32(T.address_of(li_tmp), li, scale_for_old, cur_sum)
            li = li_tmp

            if k > 0:
                prev_buf: T.int32 = _ring_mod3(k - 1, max_k_blocks)
                prev_phase: T.int32 = _ring_phase_parity(k - 1, max_k_blocks)
                bar_sv_done.wait(prev_buf, prev_phase)

            # CUDA phase1.cuh:229-232 S store (vectorized by the reg copy path).
            Tx.wg.copy(s_smem_gemm[:, :], s_frag[:, :])
            if (k > 0) & should_scale_o:
                T.ptx.tcgen05.fence.after_thread_sync()
                # CUDA common_subroutine.h:147-168 rescale_O.
                o_rescale_frag = T.alloc_tcgen05_ldst_frag("32x32b", (128, 32), "float32")
                o_rescale = o_rescale_frag.local()
                for chunk_idx in T.unroll((D_V // 2) // 32):
                    Tx.wg.copy_async(
                        o_rescale_frag[:, :], o_win.chunk((None, (D_V // 2) // 32))[:, chunk_idx]
                    )
                    T.ptx.tcgen05.wait.ld()
                    Tx.wg.mul(o_rescale_frag[:, :], o_rescale_frag[:, :], scale_for_old)
                    Tx.wg.copy_async(
                        o_win.chunk((None, (D_V // 2) // 32))[:, chunk_idx], o_rescale_frag[:, :]
                    )
                    T.ptx.tcgen05.wait.st()
                T.ptx.tcgen05.fence.before_thread_sync()

            T.ptx.fence.proxy_async("shared::cta")
            bar_so_ready.arrive(0)

        # CUDA phase1.cuh:246-357.  Epilogue scalar exchange, O TMEM readback,
        # output scaling/bf16 staging, and the two elected-warp O TMA stores.
        if real_mi == T.float32(-float("inf")):
            li = 0.0
            mi = T.float32(-float("inf"))

        rowwise_li_buf[idx_in_warpgroup] = li
        T.ptx.bar.sync(BAR_WG0_SYNC, 128)
        li = li + rowwise_li_buf[idx_in_warpgroup ^ 64]

        if idx_in_warpgroup < B_H:
            cur_lse: T.float32
            cur_lse_log: T.let = T.log(li)
            T.ptx.fma_f32(T.address_of(cur_lse), mi, LN_2, cur_lse_log)
            cur_lse = T.if_then_else(
                cur_lse == T.float32(-float("inf")), T.float32(float("inf")), cur_lse
            )
            max_logits[s_q_idx, idx_in_warpgroup] = real_mi * LN_2
            lse[s_q_idx, idx_in_warpgroup] = cur_lse

        last_k: T.int32 = num_k_blocks - 1
        last_buf: T.int32 = _ring_mod3(last_k, max_k_blocks)
        last_phase: T.int32 = _ring_phase_parity(last_k, max_k_blocks)
        bar_sv_done.wait(last_buf, last_phase)
        T.ptx.tcgen05.fence.after_thread_sync()

        attn_sink_log2: T.let = (
            T.cuda.ldg(attn_sink.ptr_to([idx_in_warpgroup % B_H]), "float32") * LOG_2_E
            if have_attn_sink
            else T.float32(-float("inf"))
        )
        output_scale: T.float32 = T.cuda.fdividef(
            T.float32(1.0), li + T.ptx.exp2(attn_sink_log2 - mi)
        )

        o_epi_frag = T.alloc_tcgen05_ldst_frag("32x32b", (128, 64), "float32")
        o_epi = o_epi_frag.local()
        o_epi_bf16_frag = T.alloc_tcgen05_ldst_frag("32x32b", (128, 64), "bfloat16")
        # O smem viewed the same way as o_tmem: (128, 256) so a (128,64) frag
        # chunk copies straight in (row r = lane-half*64 + h, col = D_V fold).
        o_smem_win = o_smem.rearrange("h (a b c) -> (b h) (a c)", a=2, b=2, c=128)
        have_valid_indices: T.let = T.ptx.any_sync(T.uint32(0xFFFFFFFF), li != 0.0) != 0
        if not have_valid_indices:
            for o_zero_i in T.unroll(64):
                o_epi[o_zero_i] = 0.0
            output_scale = 1.0
        for epi_c in T.unroll(2):
            for epi_k in T.unroll((D_V // 4) // 64):
                if have_valid_indices:
                    # CUDA phase1.cuh:314-317: TMEM O load/fence.
                    Tx.wg.copy_async(
                        o_epi_frag[:, :],
                        o_win.chunk((None, (D_V // 2) // 64))[:, epi_c * 2 + epi_k],
                    )
                    T.ptx.tcgen05.wait.ld()
                Tx.wg.mul(o_epi_frag[:, :], o_epi_frag[:, :], output_scale)
                Tx.wg.cast(o_epi_bf16_frag[:, :], o_epi_frag[:, :])
                Tx.wg.copy(
                    o_smem_win.chunk((None, (D_V // 2) // 64))[:, epi_c * 2 + epi_k],
                    o_epi_bf16_frag[:, :],
                )
                T.ptx.fence.proxy_async("shared::cta")
                T.ptx.bar.sync(BAR_WG0_SYNC, 128)
                if warp_idx == 0:
                    if T.ptx.elect_sync():
                        # CUDA phase1.cuh:335-342: first half O TMA store.
                        epi_chunk_idx: T.let = epi_c * (D_V // 2 // 64) + epi_k
                        Tx.copy_async(
                            out.chunk((None, None, D_V // 64))[s_q_idx, :, epi_chunk_idx],
                            o_smem.chunk((None, D_V // 64))[:, epi_chunk_idx],
                            **tma_config(),
                        )
                if warp_idx == 1:
                    if T.ptx.elect_sync():
                        # CUDA phase1.cuh:343-350: second half O TMA store.
                        epi_chunk_idx: T.let = epi_c * (D_V // 2 // 64) + (D_V // 64 // 4) + epi_k
                        Tx.copy_async(
                            out.chunk((None, None, D_V // 64))[s_q_idx, :, epi_chunk_idx],
                            o_smem.chunk((None, D_V // 64))[:, epi_chunk_idx],
                            **tma_config(),
                        )

        if warp_idx == 0:
            T.ptx.tcgen05.dealloc(T.uint32(0), n_cols=512, cta_group=1)

    elif warpgroup_idx == 1:
        # CUDA phase1.cuh:358-412. KV NoPE producer. Scalar index loads + skip
        # decisions transcribed; gather4 kept at its source-order call site.
        wg1_warp_idx: T.let = warp_idx - 4
        # This warp's 16 interleaved NoPE rows: split the 64-row dim into
        # (stripe, warp, row) and pick this warp, merging stripe x row.
        k_nope_warp = k_nope.tile((1, (-1, WG1_NUM_WARPS, 4)))[:, wg1_warp_idx, :]
        if T.ptx.elect_sync():
            for k in T.serial(0, num_k_blocks, unroll=False):
                selected_idx = T.alloc_local((WG1_ROWS_PER_WARP, 4), "int32")
                max_indices: T.int32 = -1
                min_indices: T.int32 = s_kv
                # This warp's 16 indices from the (local_row, warp, j) split:
                # one strided copy (auto-vectorizes to 4x 128b ld.global.nc).
                idx_block = indices.view(
                    s_q, stride_indices_s_q // B_TOPK, WG1_ROWS_PER_WARP, WG1_NUM_WARPS, 4
                ).sub[s_q_idx, k, :, wg1_warp_idx, :]
                Tx.copy(selected_idx[:, :], idx_block[:, :], cache="nc")
                for local_row in T.unroll(WG1_ROWS_PER_WARP):
                    for j in T.unroll(4):
                        idx: T.let = selected_idx[local_row, j]
                        max_indices = T.max(max_indices, idx)
                        min_indices = T.min(min_indices, idx)

                is_all_rows_invalid: T.let = (min_indices == s_kv) | (max_indices == -1)
                should_skip_tma: T.let = is_all_rows_invalid & (k >= NUM_BUFS)

                if k == 2:
                    bar_prologue_utccp_nope.wait(0, 0)

                cur_buf: T.int32 = _ring_mod3(k, max_k_blocks)
                cur_phase: T.int32 = _ring_phase_parity(k, max_k_blocks)
                bar_sv_done.wait(cur_buf, T.bitwise_xor(cur_phase, T.int32(1)))

                kv_nope_tma = kv.view(
                    s_kv, D_V, layout=TileLayout(S[(s_kv, D_V) : (stride_kv_s_kv, 1)])
                )

                @T.inline
                def gather_nope_part(part_idx, bar):
                    Tx.copy_async(
                        k_nope_warp.chunk((None, None, 2))[cur_buf, :, part_idx],
                        kv_nope_tma.chunk((None, 2))[:, part_idx],
                        **_kv_gather_tma(
                            mbar=bar.ptr_to([cur_buf]),
                            mbarrier_addr=d_qk == D_V and s_kv >= 65536,
                            indexer=[
                                selected_idx[local_row, j]
                                for local_row in range(WG1_ROWS_PER_WARP)
                                for j in range(4)
                            ],
                        ),
                    )

                if not should_skip_tma:
                    gather_nope_part(0, bar_kv_nope_ready_part0)
                    gather_nope_part(1, bar_kv_nope_ready_part1)
                else:
                    tx_bytes = T.uint32(WG1_ROWS_PER_WARP * 4 * (D_V // 2) * BF16_BYTES)
                    T.ptx.mbarrier.complete_tx(bar_kv_nope_ready_part0.ptr_to([cur_buf]), tx_bytes)
                    T.ptx.mbarrier.complete_tx(bar_kv_nope_ready_part1.ptr_to([cur_buf]), tx_bytes)

    else:
        # CUDA phase1.cuh:413-572. MMA warpgroup. Keep issue-thread control flow
        # source-ordered; materialize tcgen05.cp/gemm_async + cp.async paths later.
        if warp_idx == 8:
            if T.ptx.elect_sync():
                if have_rope:
                    bar_prologue_q_rope.arrive(0, tx_count=B_H * (d_qk - D_V) * BF16_BYTES)
                    bar_prologue_q_rope.wait(0, 0)
                    T.ptx.tcgen05.fence.after_thread_sync()
                    q_rope_tmem_cp = q_rope_tmem_bmm.rearrange("b h k -> h (b k)")
                    Tx.copy_async(q_rope_tmem_cp[:, :], q_rope[:, :])
                    bar_prologue_utccp_rope.arrive(0)

                bar_prologue_q_nope.arrive(0, tx_count=B_H * D_V * BF16_BYTES)
                bar_prologue_q_nope.wait(0, 0)
                T.ptx.tcgen05.fence.after_thread_sync()
                q_nope_tmem_cp = q_nope_tmem_bmm.rearrange("b h (dc di) -> h dc b di", di=64)
                Tx.copy_async(
                    q_nope_tmem_cp[:, :, :, :],
                    q_nope.view(B_H, D_V // 128, 2, 64)[:, :, :, :],
                    shape="128x256b",
                    cta_group=1,
                )
                bar_prologue_utccp_nope.arrive(0)

                if have_rope:
                    bar_prologue_utccp_rope.wait(0, 0)

                for k in T.serial(0, num_k_blocks + 1, unroll=False):
                    if k < num_k_blocks:
                        cur_buf: T.int32 = _ring_mod3(k, max_k_blocks)
                        cur_phase: T.int32 = _ring_phase_parity(k, max_k_blocks)
                        bar_p_free.wait(0, T.bitwise_xor(T.bitwise_and(k, T.int32(1)), T.int32(1)))
                        T.ptx.tcgen05.fence.after_thread_sync()

                        if have_rope:
                            bar_kv_rope_ready.wait(0, T.bitwise_and(k, T.int32(1)))
                            T.ptx.tcgen05.fence.after_thread_sync()
                            # CUDA phase1.cuh:489 Q RoPE x K RoPE MMA.
                            mma_p_accumulate = T.uint32(0)
                            Tx.gemm_async(
                                tmem_p_bmm[:, :, :],
                                q_rope_tmem_bmm[:, :, :],
                                k_rope_tiled_mma[:, :],
                                **_mma_config(accum=mma_p_accumulate, smem_desc=mma_smem_desc),
                            )
                            bar_qk_rope_done.arrive(0)

                        if k == 0:
                            bar_prologue_utccp_nope.wait(0, 0)

                        for kv_nope_part_idx in T.unroll(2):
                            tx_bytes: T.let = B_TOPK * (D_V // 2) * BF16_BYTES
                            if kv_nope_part_idx == 0:
                                bar_kv_nope_ready_part0.arrive(cur_buf, tx_count=tx_bytes)
                                bar_kv_nope_ready_part0.wait(cur_buf, cur_phase)
                            else:
                                bar_kv_nope_ready_part1.arrive(cur_buf, tx_count=tx_bytes)
                                bar_kv_nope_ready_part1.wait(cur_buf, cur_phase)
                            T.ptx.tcgen05.fence.after_thread_sync()
                            # CUDA phase1.cuh:505-506 Q NoPE x K NoPE MMA.
                            clear_nope_accum: T.let = (not have_rope) & (kv_nope_part_idx == 0)
                            mma_p_accumulate = T.if_then_else(
                                clear_nope_accum, T.uint32(0), T.uint32(1)
                            )
                            Tx.gemm_async(
                                tmem_p_bmm[:, :, :],
                                q_nope_tmem_bmm.chunk((None, None, 2))[:, :, kv_nope_part_idx],
                                k_nope_tiled_mma.chunk((None, None, 2))[
                                    cur_buf, :, kv_nope_part_idx
                                ],
                                **_mma_config(accum=mma_p_accumulate, smem_desc=mma_smem_desc),
                            )
                        bar_qk_nope_done.arrive(cur_buf)

                    if k > 0:
                        cur_buf_prev: T.int32 = _ring_mod3(k - 1, max_k_blocks)
                        bar_so_ready.wait(0, T.bitwise_and(k - 1, T.int32(1)))
                        T.ptx.tcgen05.fence.after_thread_sync()
                        # CUDA phase1.cuh:521-523 S(i-1) x V(i-1) MMA.
                        mma_o_accumulate = T.if_then_else(k == 1, T.uint32(0), T.uint32(1))

                        @T.inline
                        def gemm_o(dst, col_lo, col_hi):
                            Tx.gemm_async(
                                dst[:, :],
                                s_smem_gemm[:, :],
                                k_nope[cur_buf_prev, :, col_lo:col_hi],
                                transB=True,
                                **_mma_config(accum=mma_o_accumulate, smem_desc=mma_smem_desc),
                            )

                        gemm_o(o_tmem.sub[:, 0 : D_V // 2], 0, D_V // 2)
                        gemm_o(o_tmem.sub[:, D_V // 2 : D_V], D_V // 2, D_V)
                        mma_o_accumulate = T.uint32(1)
                        bar_sv_done.arrive(cur_buf_prev)

        elif warp_idx == 9:
            # CUDA common_subroutine.h:14-44 load_indices_and_generate_mask.
            if lane_idx < B_TOPK // 8:
                lane_indices = T.alloc_local((8,), "int32")
                for k in T.serial(0, num_k_blocks, unroll=False):
                    abs_pos_start: T.let = k * B_TOPK
                    row_base: T.let = g_indices_base + k * B_TOPK + lane_idx * 8
                    Tx.copy(
                        lane_indices[0:8],
                        indices[row_base : row_base + 8],
                        dispatch="vec_256b",
                        cache="nc",
                        l1_evict="L1::no_allocate",
                        l2_evict="L2::evict_normal",
                        prefetch_size="L2::256B",
                    )
                    is_ks_valid_mask: T.int8 = pack_valid_mask8(
                        lane_indices, abs_pos_start, lane_idx, topk_len, s_kv
                    )

                    cur_buf: T.int32 = _ring_mod3(k, max_k_blocks)
                    cur_phase: T.int32 = _ring_phase_parity(k, max_k_blocks)
                    bar_k_valid_free.wait(cur_buf, T.bitwise_xor(cur_phase, T.int32(1)))
                    is_k_valid[cur_buf, lane_idx] = is_ks_valid_mask
                    bar_k_valid_ready.arrive(cur_buf)

        elif (warp_idx == 10) | (warp_idx == 11):
            if have_rope:
                thread_idx: T.let = (warp_idx - 10) * 32 + lane_idx
                group_idx: T.let = thread_idx // 8
                idx_in_group: T.let = thread_idx % 8
                for k in T.serial(0, num_k_blocks, unroll=False):
                    rope_indices = T.alloc_local(((B_TOPK // (64 // 8)),), "int32")
                    for local_row in T.unroll(B_TOPK // (64 // 8)):
                        rope_indices[local_row] = T.cuda.ldg(
                            indices.ptr_to(
                                [g_indices_base + k * B_TOPK + group_idx + local_row * (64 // 8)]
                            ),
                            "int32",
                        )
                    bar_qk_rope_done.wait(
                        0, T.bitwise_xor(T.bitwise_and(k, T.int32(1)), T.int32(1))
                    )
                    for local_row in T.unroll(B_TOPK // (64 // 8)):
                        index = rope_indices[local_row]
                        is_valid_index: T.let = (index >= 0) & (index < s_kv)
                        kv_off: T.let = index * stride_kv_s_kv + D_V + idx_in_group * 8
                        Tx.copy_async(
                            k_rope.chunk((None, Q_ROPE_DIM // 8))[
                                group_idx + local_row * (64 // 8), idx_in_group
                            ],
                            kv[kv_off : kv_off + 8],
                            dispatch="ldgsts",
                            direct=True,
                            prefetch_size=128,
                            predicate=is_valid_index,
                            fill_mode="zero",
                        )
                    T.ptx.cp_async.mbarrier.arrive.noinc(bar_kv_rope_ready.ptr_to([0]))


def get_kernel(**kwargs: Any):
    cfg = _cfg(**kwargs)
    stride_kv_s_kv = int(kwargs.get("stride_kv_s_kv", cfg.d_qk * cfg.h_kv))
    stride_indices_s_q = int(kwargs.get("stride_indices_s_q", cfg.topk * cfg.h_kv))
    kernel = _kernel.specialize(
        s_q=cfg.s_q,
        s_kv=cfg.s_kv,
        topk=cfg.topk,
        d_qk=cfg.d_qk,
        h_q=cfg.h_q,
        stride_kv_s_kv=stride_kv_s_kv,
        stride_indices_s_q=stride_indices_s_q,
        have_attn_sink=cfg.have_attn_sink,
        have_topk_length=cfg.have_topk_length,
        sm_scale_div_log2=(1.0 / math.sqrt(cfg.d_qk)) * LOG_2_E,
    )
    return kernel.with_attr("tirx.kernel_launch_params", list(LAUNCH_TAGS))


def run_test(**kwargs: Any) -> None:
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for sparse FlashMLA phase1")

    from tirx_kernels.runner import compile_kernel

    case = prepare_data(**kwargs)
    cfg: SparseFlashMLAPrefillHead64Config = case["config"]
    prim_func = get_kernel(**kwargs)
    ex = compile_kernel(prim_func)
    ex(*_tirx_args(case))
    torch.cuda.synchronize()
    ref_out, ref_max_logits, ref_lse = _reference_sparse_prefill(case)
    torch.testing.assert_close(case["out"], ref_out, rtol=3.01 / 128, atol=5e-3)
    torch.testing.assert_close(case["max_logits"], ref_max_logits, rtol=2.01 / 65536, atol=1e-6)
    torch.testing.assert_close(case["lse"], ref_lse, rtol=2.01 / 65536, atol=1e-6)
    cfg.validate()


def run_bench(
    *, warmup: int | None = None, repeat: int | None = None, timer: str | None = None, **kwargs: Any
) -> dict[str, Any]:
    _rounds = kwargs.pop("rounds", 1)
    _cooldown_s = kwargs.pop("cooldown_s", 1.0)
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for sparse FlashMLA phase1 benchmark")

    from tirx_kernels.runner import compile_kernel
    from tvm.tirx.bench import bench

    prim_func = get_kernel(**kwargs)
    ex = compile_kernel(prim_func)

    # Allocate inputs once, outside the timed region (Triton-standard pure launch).
    case = prepare_data(**kwargs)
    args = _tirx_args(case)

    funcs = {"tirx": lambda: ex(*args)}

    from tirx_kernels.flashmla._flashmla_bench import flashmla_reference_builder
    from tirx_kernels.flashmla._trtllm_gen_bench import (
        trtllm_gen_config_compatible,
        trtllm_gen_reference_builder,
    )

    references = {"flashmla": lambda: flashmla_reference_builder(case)}
    if trtllm_gen_config_compatible(case["config"]):
        references["trtllm_gen"] = lambda: trtllm_gen_reference_builder(case)

    return bench(
        funcs,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references=references,
        rounds=_rounds,
        cooldown_s=_cooldown_s,
    )


__all__ = ["CONFIGS", "KERNEL_META", "get_kernel", "prepare_data", "run_bench", "run_test"]

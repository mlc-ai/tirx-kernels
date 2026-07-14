from __future__ import annotations

import math
from dataclasses import dataclass, fields
from functools import partial
from typing import Any
from unittest import SkipTest

import torch

from tirx_kernels.flashmla._gemm import tcgen05_config
from tirx_kernels.flashmla._mask import pack_valid_mask8
from tirx_kernels.flashmla._tma import leader_mbar, tma_config
from tvm.backend.cuda.operator.tile_primitive.tma_utils import SwizzleMode
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.lang.pipeline import MBarrier, TCGen05Bar, TMABar
from tvm.tirx.layout import S, TileLayout, laneid, wid_in_wg

B_H = 128
B_TOPK = 128
D_V = 512
NUM_BUFS = 2
NUM_THREADS = 512
MAX_INIT_VAL = -1.0e30
LOG_2_E = math.log2(math.e)
LN_2 = math.log(2.0)

LAUNCH_TAGS = ("blockIdx.x", "clusterCtaIdx.x", "threadIdx.x", "tirx.use_dyn_shared_memory")

BAR_WG0_SYNC = 0

BF16_BYTES = 2
D_TQ = 384
P_TMEM_COLS = B_TOPK // 2
B_EPI = 64
WG1_NUM_WARPS = 4
WG1_ROWS_PER_WARP = (B_TOPK // 2) // 4 // WG1_NUM_WARPS
WG2_NUM_WARPS = 4
WG2_ROWS_PER_PART = (B_TOPK // 2) // 4 // WG2_NUM_WARPS

# KV gather4 TMA knobs shared by the gather call sites.
_mma_config = partial(tcgen05_config, cta_group=2)
_kv_gather_tma = partial(
    tma_config,
    cta_group=2,
    cta_mask=T.uint16(1),
    gather_axis=0,
    dst_gather_axis=0,
    cache_hint=T.uint64(0x14F0000000000000),
)


@dataclass(frozen=True)
class SparseFlashMLAPrefillHead128Config:
    label: str
    s_q: int
    s_kv: int
    topk: int
    d_qk: int
    h_q: int = B_H
    h_kv: int = 1
    d_v: int = D_V
    have_attn_sink: bool = False
    have_topk_length: bool = False
    inject_invalid_indices: bool = False
    seed: int = 0

    def validate(self) -> None:
        if self.h_q != B_H:
            raise ValueError("head128 regular phase1 requires h_q == 128")
        if self.h_kv != 1:
            raise ValueError("head128 regular phase1 requires h_kv == 1")
        if self.d_qk not in (512, 576):
            raise ValueError("d_qk must be 512 or 576")
        if self.d_v != D_V:
            raise ValueError("d_v must be 512")
        if self.topk % B_TOPK != 0:
            raise ValueError("topk must be a multiple of 128")


CONFIGS = [
    {
        "label": f"bench_regular_dqk{d_qk}_hq128_s4096_kv{s_kv}_topk2048",
        "s_q": 4096,
        "s_kv": s_kv,
        "topk": 2048,
        "d_qk": d_qk,
        "h_q": B_H,
        "have_attn_sink": True,
    }
    for d_qk in (512, 576)
    for s_kv in (8192, 32768, 65536)
]

KERNEL_META = {
    "name": "sparse_flashmla_prefill_head128_phase1",
    "category": "flashmla",
    "compute_capability": 10,
}


def _cfg(**kwargs: Any) -> SparseFlashMLAPrefillHead128Config:
    cfg_fields = {field.name for field in fields(SparseFlashMLAPrefillHead128Config)}
    cfg_kwargs = {key: value for key, value in kwargs.items() if key in cfg_fields}
    if "label" not in cfg_kwargs:
        cfg_kwargs["label"] = "custom"
    cfg = SparseFlashMLAPrefillHead128Config(**cfg_kwargs)
    cfg.validate()
    return cfg


def _flashmla_regular_dispatch_reason(cfg: SparseFlashMLAPrefillHead128Config) -> str:
    if cfg.h_q != B_H:
        return "out_of_scope: h_q != 128 dispatches to head64 or unsupported path"
    if cfg.d_qk == 512 and cfg.topk <= 1280:
        return "out_of_scope: sm100 head128 D_QK=512 topk<=1280 dispatches small-topk"
    return f"regular: sm100 head128 run_fwd_phase1_kernel<{cfg.d_qk}>"


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
        "dispatch_reason": _flashmla_regular_dispatch_reason(cfg),
    }


def _reference_sparse_prefill(
    case: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg: SparseFlashMLAPrefillHead128Config = case["config"]
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
    # CUDA_TRANSCRIBE_START: run_fwd_phase1_kernel line 622, then sparse_attn_fwd_kernel_devfunc
    # line 68. One CTA pair per query-row; upstream source-order TMA/MMA/softmax warp layout.
    block_idx = T.cta_id([2 * s_q])
    T.cta_id_in_cluster([2])
    cta_idx: T.let = block_idx % 2
    s_q_idx: T.let = block_idx // 2
    thread_idx = T.thread_id([NUM_THREADS])
    T.warpgroup_id([NUM_THREADS // 128])
    T.warp_id_in_wg([4])
    T.lane_id([32])
    T.thread_id_in_wg([128])
    warp_idx: T.let = T.cuda.__shfl_sync(T.uint32(0xFFFFFFFF), thread_idx // 32, 0, 32)
    lane_idx: T.let = thread_idx % 32
    topk_len: T.let = (
        T.cuda.ldg(topk_length.ptr_to([s_q_idx]), "int32") if have_topk_length else topk
    )
    num_k_blocks: T.let = T.max((topk_len + B_TOPK - 1) // B_TOPK, 1)
    warpgroup_idx: T.let = T.cuda.__shfl_sync(T.uint32(0xFFFFFFFF), thread_idx // 128, 0, 32)
    idx_in_warpgroup: T.let = thread_idx % 128
    d_sq = T.meta_var(d_qk - D_TQ)
    num_sq_tiles = T.meta_var((d_qk - D_TQ) // 64)
    num_qk_tiles = T.meta_var(d_qk // 64)
    mma_smem_desc = T.meta_var(
        "recompute"
        if (d_qk == 512 and s_kv == 8192)
        else "local_hoist"
        if (d_qk == 576 and s_kv != 65536)
        else "hoist"
        if ((d_qk == 512 and s_kv == 32768) or (d_qk == 576 and s_kv == 65536))
        else "encode"
    )

    # CUDA phase1.cuh:84-90, config.h:93-118.  Preserve SharedMemoryPlan's
    # union offsets: q_full, {sq, v, k}, and o alias the same base.
    pool = T.SMEMPool()
    u_base = T.meta_var(pool.offset)
    q_full = pool.alloc_tcgen05_mma_AB((B_H // 2, d_qk), "bfloat16")
    # sQ stays live as q_full's first d_sq cols (a contiguous prefix under the
    # 64-col swizzle chunks); v/k reuse the D_TQ tail once Q has moved to TMEM.
    pool.move_base_to(u_base + (B_H // 2) * d_sq * BF16_BYTES)
    v_smem = pool.alloc_tcgen05_mma_AB((D_V // 2, B_TOPK), "bfloat16")
    k_smem = pool.alloc_tcgen05_mma_AB((B_TOPK // 2, d_qk), "bfloat16")
    u_end = T.meta_var(pool.offset)
    pool.move_base_to(u_base)
    o_smem = pool.alloc_tcgen05_mma_AB((B_H // 2, D_V), "bfloat16")
    pool.move_base_to(u_end)
    s_smem_gemm = pool.alloc_tcgen05_mma_AB(
        (B_H // 2, B_TOPK), "bfloat16", swizzle_mode=SwizzleMode.SWIZZLE_NONE
    )
    is_k_valid = pool.alloc((NUM_BUFS, B_TOPK // 8), "int8")
    bar_prologue_q = TMABar(pool, 1)
    bar_prologue_utccp = TCGen05Bar(pool, 1)
    bar_qk_part_done = TCGen05Bar(pool, NUM_BUFS)
    bar_qk_done = TCGen05Bar(pool, NUM_BUFS)
    bar_sv_part_done = TCGen05Bar(pool, NUM_BUFS)
    bar_sv_done = TCGen05Bar(pool, NUM_BUFS)
    bar_k_part0_ready = TMABar(pool, NUM_BUFS)
    bar_k_part1_ready = TMABar(pool, NUM_BUFS)
    bar_v_part0_ready = TMABar(pool, NUM_BUFS)
    bar_v_part1_ready = TMABar(pool, NUM_BUFS)
    bar_p_free = MBarrier(pool, NUM_BUFS)
    bar_so_ready = MBarrier(pool, NUM_BUFS)
    bar_k_valid_ready = MBarrier(pool, NUM_BUFS)
    bar_k_valid_free = MBarrier(pool, NUM_BUFS)
    tmem_start_addr = pool.alloc((1,), "uint32", align=4)
    rowwise_max_buf = pool.alloc((128,), "float32")
    rowwise_li_buf = pool.alloc((128,), "float32")
    pool.commit()
    kv_tma = kv.view(s_kv, d_qk, layout=TileLayout(S[(s_kv, d_qk) : (stride_kv_s_kv, 1)]))

    g_indices_base: T.let = s_q_idx * stride_indices_s_q
    mma_p_accumulate: T.uint32 = 0
    mma_o_accumulate: T.uint32 = 0

    # CUDA phase1.cuh:87-146.  Warp 0 owns barrier init, Q TMA launch,
    # and the cta_group::2 TMEM allocation.
    if warp_idx == 0:
        if T.ptx.elect_sync():
            bar_prologue_q.init(1)
            bar_prologue_utccp.init(1)
            for init_stage in T.unroll(NUM_BUFS):
                T.ptx.mbarrier.init(bar_qk_part_done.ptr_to([init_stage]), 1)
                T.ptx.mbarrier.init(bar_qk_done.ptr_to([init_stage]), 1)
                T.ptx.mbarrier.init(bar_sv_part_done.ptr_to([init_stage]), 1)
                T.ptx.mbarrier.init(bar_sv_done.ptr_to([init_stage]), 1)
                T.ptx.mbarrier.init(bar_k_part0_ready.ptr_to([init_stage]), 1)
                T.ptx.mbarrier.init(bar_k_part1_ready.ptr_to([init_stage]), 1)
                T.ptx.mbarrier.init(bar_v_part0_ready.ptr_to([init_stage]), 1)
                T.ptx.mbarrier.init(bar_v_part1_ready.ptr_to([init_stage]), 1)
                T.ptx.mbarrier.init(bar_p_free.ptr_to([init_stage]), 128 * 2)
                T.ptx.mbarrier.init(bar_so_ready.ptr_to([init_stage]), 128 * 2)
                T.ptx.mbarrier.init(bar_k_valid_ready.ptr_to([init_stage]), 16)
                T.ptx.mbarrier.init(bar_k_valid_free.ptr_to([init_stage]), 128)
            T.ptx.fence.mbarrier_init()

    T.cuda.cluster_sync()

    if warp_idx == 0:
        if T.ptx.elect_sync():
            Tx.copy_async(
                q_full[:, :],
                q.chunk((None, 2, None))[s_q_idx, cta_idx, :],
                **tma_config(
                    mbar=leader_mbar(bar_prologue_q.ptr_to([0])),
                    cta_group=2,
                    cache_hint=T.uint64(0x12F0000000000000),
                ),
            )

        T.ptx.tcgen05.alloc(T.address_of(tmem_start_addr[0]), n_cols=512, cta_group=2)
        T.cuda.trap_when_assert_failed(tmem_start_addr[0] == T.uint32(0))
        T.ptx.tcgen05.relinquish_alloc_permit(cta_group=2)

    T.cuda.cta_sync()

    tmem_pool = T.TMEMPool(pool, total_cols=512, cta_group=2, tmem_addr=tmem_start_addr)
    # O accumulator: one alloc; logical col halves are the B lo/hi gemm outputs,
    # read back as a (128, D_V//2) datapath-D tile via permute+reshape.
    o_tmem = tmem_pool.alloc_tcgen05_mma_D(
        (B_H // 2, D_V), "float32", M=128, cta_group=2, group=(2, 2, 128)
    )
    tmem_o_lo = o_tmem.sub[:, 0 : D_V // 2]
    tmem_o_hi = o_tmem.sub[:, D_V // 2 : D_V]
    o_win = o_tmem.rearrange("h (a b c) -> (b h) (a c)", a=2, b=2, c=128)
    tmem_p = tmem_pool.alloc_tcgen05_mma_D((B_H // 2, B_TOPK), "float32", M=128, cta_group=2)
    # Qt TMEM at real 128-lane footprint: the 64x128b.warpx2::02_13 copy mirrors rows 0-63 to
    # lane +64, so the alloc declares that replica (R[2:64@TLane]); MMA validates it at the anchor.
    q_tmem = tmem_pool.alloc_tcgen05_mma_A((B_H // 2, D_TQ), "bfloat16", M=128, cta_group=2)
    v_smem_gemm = v_smem.rearrange("(x r) (z kl) -> r (z x kl)", x=2, z=2, kl=64)

    if warpgroup_idx == 0:
        # CUDA phase1.cuh:150-386.  Scale/exp warpgroup and epilogue.
        T.ptx.setmaxnreg(True, 144)
        mi: T.float32 = MAX_INIT_VAL
        li: T.float32 = 0.0
        real_mi: T.float32 = T.float32(-float("inf"))
        scale_pair: T.let = T.cuda.make_float2(sm_scale_div_log2, sm_scale_div_log2)

        for k in T.serial(0, num_k_blocks, unroll=False):
            cur_buf: T.let = k % NUM_BUFS
            cur_phase: T.let = (k // NUM_BUFS) & 1
            bar_qk_done.wait(cur_buf, cur_phase)
            T.ptx.tcgen05.fence.after_thread_sync()

            p_frag = T.alloc_tcgen05_ldst_frag("32x32b", (128, P_TMEM_COLS), "uint32")
            Tx.wg.copy_async(
                p_frag[:, :], tmem_p.rearrange("h (b t) -> (b h) t", b=2).with_dtype("uint32")[:, :]
            )
            p = p_frag.local()
            T.ptx.tcgen05.wait.ld()
            T.ptx.tcgen05.fence.before_thread_sync()
            bar_p_free.arrive(cur_buf, remote=T.uint32(0))

            bar_k_valid_ready.wait(cur_buf, cur_phase)
            valid_word_offset: T.let = T.if_then_else(
                idx_in_warpgroup >= 64, B_TOPK // 8 // 2 // 4, 0
            )
            is_k_valid_lo: T.let = is_k_valid.view("uint32")[cur_buf, valid_word_offset]
            is_k_valid_hi: T.let = is_k_valid.view("uint32")[cur_buf, valid_word_offset + 1]

            @T.inline
            def mask_p_half(valid_word, base):
                for p_i in T.unroll(P_TMEM_COLS // 2):
                    invalid_p_predicate: T.let = T.bitwise_and(
                        T.shift_right(valid_word, T.uint32(p_i)), T.uint32(1)
                    ) == T.uint32(0)
                    p[base + p_i] = T.if_then_else(
                        invalid_p_predicate, T.uint32(0xFF800000), p[base + p_i]
                    )

            mask_p_half(is_k_valid_lo, 0)
            mask_p_half(is_k_valid_hi, P_TMEM_COLS // 2)

            cur_pi_max: T.float32 = T.float32(-float("inf"))
            for p_i in T.unroll(P_TMEM_COLS):
                cur_pi_max = T.max(cur_pi_max, T.cuda.uint_as_float(p[p_i]))
            cur_pi_max = cur_pi_max * sm_scale_div_log2
            bar_k_valid_free.arrive(cur_buf)

            T.ptx.bar.sync(BAR_WG0_SYNC, 128)
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
            li = li * scale_for_old

            # S frag: warpgroup-distributed (B_H//2, B_TOPK) tile. Thread idx owns row h = idx%64
            # and k half [64*(idx//64), +64) in 8-elem chunks (from the packing loop below).
            s_frag = T.alloc_buffer(
                (B_H // 2, B_TOPK),
                "bfloat16",
                scope="local",
                layout=TileLayout(
                    S[(2, 32, 2, B_TOPK // 2) : (1 @ wid_in_wg, 1 @ laneid, 2 @ wid_in_wg, 1)]
                ),
            )
            s_pack = s_frag.local().view("uint32")
            neg_new_max_pair: T.let = T.cuda.make_float2(-new_max, -new_max)
            for s_i in T.unroll(P_TMEM_COLS // 2):
                p_pair: T.let = T.cuda.make_float2(
                    T.cuda.uint_as_float(p[s_i * 2]), T.cuda.uint_as_float(p[s_i * 2 + 1])
                )
                fma_pair: T.let = T.ptx.fma_f32x2(p_pair, scale_pair, neg_new_max_pair, dps=False)
                s_x: T.let = T.ptx.exp2(T.cuda.float2_x(fma_pair))
                s_y: T.let = T.ptx.exp2(T.cuda.float2_y(fma_pair))
                li = li + s_x + s_y
                s_pack[s_i] = T.cuda.float22bfloat162_rn(s_x, s_y)

            if k > 0:
                prev_buf: T.let = (k - 1) % NUM_BUFS
                prev_phase: T.let = ((k - 1) // NUM_BUFS) & 1
                bar_sv_done.wait(prev_buf, prev_phase)

            Tx.wg.copy(s_smem_gemm[:, :], s_frag[:, :])

            if (k > 0) & should_scale_o:
                T.ptx.tcgen05.fence.after_thread_sync()
                o_rescale_frag = T.alloc_tcgen05_ldst_frag("32x32b", (128, 32), "float32")
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
            bar_so_ready.arrive(cur_buf, remote=T.uint32(0))

        if real_mi == T.float32(-float("inf")):
            li = 0.0
            mi = T.float32(-float("inf"))

        rowwise_li_buf[idx_in_warpgroup] = li
        T.ptx.bar.sync(BAR_WG0_SYNC, 128)
        li = li + rowwise_li_buf[idx_in_warpgroup ^ 64]

        if idx_in_warpgroup < B_H // 2:
            global_head: T.let = cta_idx * (B_H // 2) + idx_in_warpgroup
            cur_lse: T.float32
            cur_lse_log: T.let = T.log(li)
            T.ptx.fma_f32(T.address_of(cur_lse), mi, LN_2, cur_lse_log)
            cur_lse = T.if_then_else(
                cur_lse == T.float32(-float("inf")), T.float32(float("inf")), cur_lse
            )
            max_logits[s_q_idx, global_head] = real_mi * LN_2
            lse[s_q_idx, global_head] = cur_lse

        last_k: T.let = num_k_blocks - 1
        last_buf: T.let = last_k % NUM_BUFS
        last_phase: T.let = (last_k // NUM_BUFS) & 1
        bar_sv_done.wait(last_buf, last_phase)
        T.ptx.tcgen05.fence.after_thread_sync()

        attn_sink_log2: T.let = (
            T.cuda.ldg(
                attn_sink.ptr_to([cta_idx * (B_H // 2) + (idx_in_warpgroup % 64)]), "float32"
            )
            * LOG_2_E
            if have_attn_sink
            else T.float32(-float("inf"))
        )
        output_scale: T.float32 = T.cuda.fdividef(
            T.float32(1.0), li + T.ptx.exp2(attn_sink_log2 - mi)
        )
        o_epi_frag = T.alloc_tcgen05_ldst_frag("32x32b", (128, B_EPI), "float32")
        o_epi = o_epi_frag.local()
        have_valid_indices: T.let = T.ptx.any_sync(T.uint32(0xFFFFFFFF), li != 0.0) != 0
        if not have_valid_indices:
            for o_zero_i in T.unroll(B_EPI):
                o_epi[o_zero_i] = 0.0
            output_scale = 1.0
        o_epi_bf16_frag = T.alloc_tcgen05_ldst_frag("32x32b", (128, B_EPI), "bfloat16")
        o_smem_win = o_smem.rearrange("h (b r) -> (b h) r", b=2)
        for epi_k in T.unroll((D_V // 2) // B_EPI):
            if have_valid_indices:
                Tx.wg.copy_async(
                    o_epi_frag[:, :], o_win.chunk((None, (D_V // 2) // B_EPI))[:, epi_k]
                )
                T.ptx.tcgen05.wait.ld()
            Tx.wg.mul(o_epi_frag[:, :], o_epi_frag[:, :], output_scale)
            Tx.wg.cast(o_epi_bf16_frag[:, :], o_epi_frag[:, :])
            Tx.wg.copy(
                o_smem_win.chunk((None, (D_V // 2) // B_EPI))[:, epi_k], o_epi_bf16_frag[:, :]
            )

            T.ptx.fence.proxy_async("shared::cta")
            T.ptx.bar.sync(BAR_WG0_SYNC, 128)
            if warp_idx == 0:
                if T.ptx.elect_sync():
                    Tx.copy_async(
                        out.chunk((None, 2, D_V // B_EPI))[s_q_idx, cta_idx, epi_k],
                        o_smem.chunk((None, D_V // B_EPI))[:, epi_k],
                        **tma_config(),
                    )
            if warp_idx == 1:
                if T.ptx.elect_sync():
                    epi_k2: T.let = epi_k + (D_V // B_EPI // 2)
                    Tx.copy_async(
                        out.chunk((None, 2, D_V // B_EPI))[s_q_idx, cta_idx, epi_k2],
                        o_smem.chunk((None, D_V // B_EPI))[:, epi_k2],
                        **tma_config(),
                    )

        if warp_idx == 0:
            T.ptx.tcgen05.dealloc(T.uint32(0), n_cols=512, cta_group=2)

    elif warpgroup_idx == 1:
        # CUDA phase1.cuh:387-446.  K producer warpgroup.
        T.ptx.setmaxnreg(False, 96)
        wg1_warp_idx: T.let = warp_idx - 4
        if T.ptx.elect_sync():
            for k in T.serial(0, num_k_blocks, unroll=False):
                indices_int4 = T.alloc_local((WG1_ROWS_PER_WARP, 4), "int32")
                max_indices: T.int32 = -1
                min_indices: T.int32 = s_kv

                # This CTA's topk half (cta_idx), split (local_row, warp, j): one
                # strided nc copy (auto-vectorizes to 4x v4 ld.global.nc), like head64.
                idx_block = indices.view(
                    s_q, stride_indices_s_q // B_TOPK, 2, WG1_ROWS_PER_WARP, WG1_NUM_WARPS, 4
                ).sub[s_q_idx, k, cta_idx, :, wg1_warp_idx, :]
                Tx.copy(indices_int4[:, :], idx_block[:, :], cache="nc")
                for local_row in T.unroll(WG1_ROWS_PER_WARP):
                    for j in T.unroll(4):
                        idx: T.let = indices_int4[local_row, j]
                        max_indices = T.max(max_indices, idx)
                        min_indices = T.min(min_indices, idx)

                is_all_rows_invalid: T.let = (min_indices == s_kv) | (max_indices == -1)
                should_skip_tma: T.let = is_all_rows_invalid & (k >= NUM_BUFS)
                cur_buf: T.let = k % NUM_BUFS
                cur_phase: T.let = (k // NUM_BUFS) & 1

                @T.inline
                def gather_k_part(col_start, col_count, tx_dim, bar):
                    if not should_skip_tma:
                        # One wide gather4 (like head64/small_topk); dispatch splits into per-atom
                        # TMAs, ncu-verified = per-64-col loop. Keep sub bounds concrete for swizzle.
                        k_gather_tile = k_smem.sub[
                            :, col_start * 64 : col_start * 64 + col_count * 64
                        ].tile(0, (-1, WG1_NUM_WARPS, 4))[:, wg1_warp_idx, :]
                        Tx.copy_async(
                            k_gather_tile[:, :],
                            kv_tma[:, col_start * 64 : col_start * 64 + col_count * 64],
                            **_kv_gather_tma(
                                mbar=leader_mbar(bar.ptr_to([cur_buf])),
                                indexer=[
                                    indices_int4[row, lane]
                                    for row in range(WG1_ROWS_PER_WARP)
                                    for lane in range(4)
                                ],
                            ),
                        )
                    else:
                        T.ptx.mbarrier.complete_tx(
                            bar.ptr_to([cur_buf]),
                            T.uint32(WG1_ROWS_PER_WARP * 4 * tx_dim * BF16_BYTES),
                            remote=T.uint32(0),
                            pred=T.uint32(1),
                        )

                if k > 0:
                    prev_buf: T.let = (k - 1) % NUM_BUFS
                    prev_phase: T.let = ((k - 1) // NUM_BUFS) & 1
                    bar_qk_part_done.wait(prev_buf, prev_phase)
                gather_k_part(0, num_sq_tiles, d_sq, bar_k_part0_ready)

                if k > 0:
                    prev_buf: T.let = (k - 1) % NUM_BUFS
                    prev_phase: T.let = ((k - 1) // NUM_BUFS) & 1
                    bar_qk_done.wait(prev_buf, prev_phase)
                gather_k_part(num_sq_tiles, num_qk_tiles - num_sq_tiles, D_TQ, bar_k_part1_ready)

    elif warpgroup_idx == 2:
        # CUDA phase1.cuh:447-489.  V producer warpgroup.
        T.ptx.setmaxnreg(False, 96)
        wg2_warp_idx: T.let = warp_idx - 8
        if T.ptx.elect_sync():
            bar_prologue_utccp.wait(0, 0)
            for k in T.serial(0, num_k_blocks, unroll=False):
                cur_buf: T.let = k % NUM_BUFS
                cur_phase: T.let = (k // NUM_BUFS) & 1
                if k > 0:
                    prev_buf: T.let = (k - 1) % NUM_BUFS
                    prev_phase: T.let = ((k - 1) // NUM_BUFS) & 1
                    bar_sv_part_done.wait(prev_buf, prev_phase)

                @T.inline
                def gather_v_part(row_offset, part, token_buf, bar):
                    # V loads all 128 tokens; the two parts map to an extent-2
                    # axis indexed by part. One strided nc copy, like head64.
                    idx_block = indices.view(
                        s_q, stride_indices_s_q // B_TOPK, 2, WG2_ROWS_PER_PART, WG2_NUM_WARPS, 4
                    ).sub[s_q_idx, k, part, :, wg2_warp_idx, :]
                    Tx.copy(token_buf[:, :], idx_block[:, :], cache="nc")
                    # One wide gather4 over all D_V//2 cols (like head64/small_topk); dispatch splits
                    # into per-atom TMAs, ncu-verified equal to a per-64-col loop (see K-gather note).
                    src0: T.let = cta_idx * 256
                    v_gather_tile = v_smem_gemm.tile(0, (2, -1, WG2_NUM_WARPS, 4))[
                        part, :, wg2_warp_idx, :
                    ]
                    Tx.copy_async(
                        v_gather_tile[:, :],
                        kv_tma[:, src0 : src0 + (D_V // 2)],
                        **_kv_gather_tma(
                            mbar=leader_mbar(bar.ptr_to([cur_buf])),
                            indexer=[
                                token_buf[row, lane]
                                for row in range(WG2_ROWS_PER_PART)
                                for lane in range(4)
                            ],
                        ),
                    )

                token_idxs_part0 = T.alloc_local((WG2_ROWS_PER_PART, 4), "int32")
                gather_v_part(0, 0, token_idxs_part0, bar_v_part0_ready)

                if k > 0:
                    prev_buf: T.let = (k - 1) % NUM_BUFS
                    prev_phase: T.let = ((k - 1) // NUM_BUFS) & 1
                    bar_sv_done.wait(prev_buf, prev_phase)
                token_idxs_part1 = T.alloc_local((WG2_ROWS_PER_PART, 4), "int32")
                gather_v_part(WG2_ROWS_PER_PART, 1, token_idxs_part1, bar_v_part1_ready)

    else:
        # CUDA phase1.cuh:490-606.  MMA warp and KV-valid loading warp.
        T.ptx.setmaxnreg(True, 168)
        if (cta_idx == 0) & (warp_idx == 12):
            if T.ptx.elect_sync():
                bar_prologue_q.arrive(0, tx_count=B_H * d_qk * BF16_BYTES)
                bar_prologue_q.wait(0, 0)
                T.ptx.tcgen05.fence.after_thread_sync()
                Tx.copy_async(
                    q_tmem[:, :],
                    q_full[:, d_sq : d_sq + D_TQ],
                    shape="64x128b",
                    cta_group=2,
                    multicast="warpx2::02_13",
                )
                bar_prologue_utccp.arrive(0, cta_group=2, cta_mask=3)

                for k in T.serial(0, num_k_blocks + 1, unroll=False):
                    if k < num_k_blocks:
                        cur_buf: T.let = k % NUM_BUFS
                        cur_phase: T.let = (k // NUM_BUFS) & 1

                        bar_k_part0_ready.arrive(cur_buf, tx_count=B_TOPK * d_sq * BF16_BYTES)
                        bar_k_part0_ready.wait(cur_buf, cur_phase)
                        if k > 0:
                            prev_buf: T.let = (k - 1) % NUM_BUFS
                            prev_phase: T.let = ((k - 1) // NUM_BUFS) & 1
                            bar_p_free.wait(prev_buf, prev_phase)
                        T.ptx.tcgen05.fence.after_thread_sync()

                        mma_p_accumulate = T.uint32(0)
                        if d_sq > 0:
                            sq_smem = q_full.sub[:, :d_sq]
                            Tx.gemm_async(
                                tmem_p[:, :],
                                sq_smem[:, :d_sq],
                                k_smem[:, :d_sq],
                                **_mma_config(accum=mma_p_accumulate, smem_desc=mma_smem_desc),
                            )
                            mma_p_accumulate = T.uint32(1)
                        bar_qk_part_done.arrive(cur_buf, cta_group=2, cta_mask=3)

                        bar_k_part1_ready.arrive(
                            cur_buf, tx_count=B_TOPK * (d_qk - d_sq) * BF16_BYTES
                        )
                        bar_k_part1_ready.wait(cur_buf, cur_phase)
                        T.ptx.tcgen05.fence.after_thread_sync()

                        Tx.gemm_async(
                            tmem_p[:, :],
                            q_tmem[:, :D_TQ],
                            k_smem[:, d_sq : d_sq + D_TQ],
                            **_mma_config(accum=mma_p_accumulate, smem_desc=mma_smem_desc),
                        )
                        mma_p_accumulate = T.uint32(1)
                        bar_qk_done.arrive(cur_buf, cta_group=2, cta_mask=3)

                    if k > 0:
                        cur_buf_prev: T.let = (k - 1) % NUM_BUFS
                        cur_phase_prev: T.let = ((k - 1) // NUM_BUFS) & 1
                        bar_so_ready.wait(cur_buf_prev, cur_phase_prev)

                        bar_v_part0_ready.arrive(
                            cur_buf_prev, tx_count=(B_TOPK // 2) * D_V * BF16_BYTES
                        )
                        bar_v_part0_ready.wait(cur_buf_prev, cur_phase_prev)
                        T.ptx.tcgen05.fence.after_thread_sync()
                        mma_o_accumulate = T.if_then_else(k == 1, T.uint32(0), T.uint32(1))
                        Tx.gemm_async(
                            tmem_o_lo[:, :],
                            s_smem_gemm[:, 0 : B_TOPK // 2],
                            v_smem_gemm[0 : B_TOPK // 2, 0 : D_V // 4],
                            transB=True,
                            **_mma_config(accum=mma_o_accumulate, smem_desc=mma_smem_desc),
                        )
                        Tx.gemm_async(
                            tmem_o_hi[:, :],
                            s_smem_gemm[:, 0 : B_TOPK // 2],
                            v_smem_gemm[0 : B_TOPK // 2, D_V // 4 : D_V // 2],
                            transB=True,
                            **_mma_config(accum=mma_o_accumulate, smem_desc=mma_smem_desc),
                        )
                        mma_o_accumulate = T.uint32(1)
                        bar_sv_part_done.arrive(cur_buf_prev, cta_group=2, cta_mask=3)

                        bar_v_part1_ready.arrive(
                            cur_buf_prev, tx_count=(B_TOPK // 2) * D_V * BF16_BYTES
                        )
                        bar_v_part1_ready.wait(cur_buf_prev, cur_phase_prev)
                        T.ptx.tcgen05.fence.after_thread_sync()
                        Tx.gemm_async(
                            tmem_o_lo[:, :],
                            s_smem_gemm[:, B_TOPK // 2 : B_TOPK],
                            v_smem_gemm[B_TOPK // 2 : B_TOPK, 0 : D_V // 4],
                            transB=True,
                            **_mma_config(accum=mma_o_accumulate, smem_desc=mma_smem_desc),
                        )
                        Tx.gemm_async(
                            tmem_o_hi[:, :],
                            s_smem_gemm[:, B_TOPK // 2 : B_TOPK],
                            v_smem_gemm[B_TOPK // 2 : B_TOPK, D_V // 4 : D_V // 2],
                            transB=True,
                            **_mma_config(accum=mma_o_accumulate, smem_desc=mma_smem_desc),
                        )
                        mma_o_accumulate = T.uint32(1)
                        bar_sv_done.arrive(cur_buf_prev, cta_group=2, cta_mask=3)

        elif warp_idx == 13:
            if lane_idx < B_TOPK // 8:
                lane_indices = T.alloc_local((8,), "int32")
                for k in T.serial(0, num_k_blocks, unroll=False):
                    row_base: T.let = g_indices_base + k * B_TOPK + lane_idx * 8
                    Tx.copy(
                        lane_indices[0:8],
                        indices[row_base : row_base + 8],
                        dispatch="vec_256b",
                        cache="nc",
                        l1_evict="L1::evict_normal",
                        l2_evict="L2::evict_normal",
                        prefetch_size="L2::256B",
                    )
                    abs_pos_start: T.let = k * B_TOPK
                    is_ks_valid_mask: T.let = pack_valid_mask8(
                        lane_indices, abs_pos_start, lane_idx, topk_len, s_kv
                    )
                    cur_buf: T.let = k % NUM_BUFS
                    cur_phase: T.let = (k // NUM_BUFS) & 1
                    bar_k_valid_free.wait(cur_buf, cur_phase ^ 1)
                    is_k_valid[cur_buf, lane_idx] = is_ks_valid_mask
                    bar_k_valid_ready.arrive(cur_buf)


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
        raise SkipTest("CUDA is required for sparse FlashMLA head128 phase1")

    from tirx_kernels.runner import compile_kernel

    case = prepare_data(**kwargs)
    cfg: SparseFlashMLAPrefillHead128Config = case["config"]
    if not case["dispatch_reason"].startswith("regular:"):
        raise SkipTest(case["dispatch_reason"])
    prim_func = get_kernel(**kwargs)
    ex = compile_kernel(prim_func)
    ex(*_tirx_args(case))
    torch.cuda.synchronize()
    ref_out, ref_max_logits, ref_lse = _reference_sparse_prefill(case)
    torch.testing.assert_close(case["out"], ref_out, rtol=4.01 / 128, atol=5e-3)
    torch.testing.assert_close(case["max_logits"], ref_max_logits, rtol=2.01 / 65536, atol=1e-6)
    torch.testing.assert_close(case["lse"], ref_lse, rtol=2.01 / 65536, atol=1e-6)
    cfg.validate()


def run_bench(
    *, warmup: int | None = None, repeat: int | None = None, timer: str | None = None, **kwargs: Any
) -> dict[str, Any]:
    _rounds = kwargs.pop("rounds", 1)
    _cooldown_s = kwargs.pop("cooldown_s", 1.0)
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for sparse FlashMLA head128 phase1 benchmark")

    from tirx_kernels.runner import compile_kernel
    from tvm.tirx.bench import bench

    case = prepare_data(**kwargs)
    if not case["dispatch_reason"].startswith("regular:"):
        raise SkipTest(case["dispatch_reason"])
    prim_func = get_kernel(**kwargs)
    ex = compile_kernel(prim_func)

    # Allocate inputs once, outside the timed region (Triton-standard pure launch).
    args = _tirx_args(case)

    funcs = {"tirx": lambda: ex(*args)}

    from tirx_kernels.flashmla._flashmla_bench import flashmla_reference_builder

    def _flashmla_ref():
        run = flashmla_reference_builder()
        return lambda: run(case)

    return bench(
        funcs,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references={"flashmla": _flashmla_ref},
        rounds=_rounds,
        cooldown_s=_cooldown_s,
    )


__all__ = ["CONFIGS", "KERNEL_META", "get_kernel", "prepare_data", "run_bench", "run_test"]

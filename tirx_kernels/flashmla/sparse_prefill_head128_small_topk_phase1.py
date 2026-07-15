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
from tvm.tirx.cuda.iket import IketProfiler
from tvm.tirx.lang.pipeline import MBarrier, TCGen05Bar, TMABar
from tvm.tirx.layout import S, TileLayout, laneid, wid_in_wg

B_H = 128
B_TOPK = 64
D_QK = 512
D_V = 512
NUM_THREADS = 512
NUM_K_BUFS = 4
NUM_INDEX_BUFS = 4
NUM_WORKER_THREADS = (128 + 4 + (B_TOPK // 8) + 1 + 128) * 2 + 1
MAX_INIT_VAL = -1.0e30
LOG_2_E = math.log2(math.e)
LN_2 = math.log(2.0)

IKET_EVENT_NAMES = (
    "h128-small-q-o",
    "h128-small-kv-gather",
    "h128-small-mma",
    "h128-small-valid-mask",
    "h128-small-clc",
    "h128-small-softmax",
)

LAUNCH_TAGS = (
    "blockIdx.x",
    "clusterCtaIdx.x",
    "threadIdx.x",
    "tirx.use_programtic_dependent_launch",
    "tirx.use_dyn_shared_memory",
)

BF16_BYTES = 2
B_EPI = 64

BAR_WG0_SYNC = 0
BAR_WG2_SYNC = 1
BAR_WG2_WARP02 = 2

WG1_ROWS_PER_WARP = B_TOPK // 4
WG3_ELEMS_PER_THREAD = B_TOPK // 2

# KV gather4 TMA knobs shared by the gather call sites.
_mma_config = partial(tcgen05_config, cta_group=2, smem_desc="local_hoist")
_kv_gather_tma = partial(
    tma_config,
    cta_group=2,
    cta_mask=T.uint16(1),
    gather_axis=0,
    dst_gather_axis=0,
    cache_hint=T.uint64(0x14F0000000000000),
)


@dataclass(frozen=True)
class SparseFlashMLAPrefillHead128SmallTopKConfig:
    label: str
    s_q: int
    s_kv: int
    topk: int
    d_qk: int = D_QK
    h_q: int = B_H
    h_kv: int = 1
    d_v: int = D_V
    have_attn_sink: bool = False
    have_topk_length: bool = False
    inject_invalid_indices: bool = False
    seed: int = 0

    def validate(self) -> None:
        if self.h_q != B_H:
            raise ValueError("head128 small-topk phase1 requires h_q == 128")
        if self.h_kv != 1:
            raise ValueError("head128 small-topk phase1 requires h_kv == 1")
        if self.d_qk != D_QK:
            raise ValueError("head128 small-topk phase1 is scoped to d_qk == 512")
        if self.d_v != D_V:
            raise ValueError("head128 small-topk phase1 requires d_v == 512")
        if self.topk % B_TOPK != 0:
            raise ValueError("small-topk phase1 requires topk to be a multiple of 64")
        if self.topk > 1280:
            raise ValueError("topk > 1280 dispatches outside the small-topk phase1 scope")


CONFIGS = [
    {
        "label": f"bench_smalltopk_dqk512_hq128_s4096_kv{s_kv}_topk1280",
        "s_q": 4096,
        "s_kv": s_kv,
        "topk": 1280,
        "h_q": B_H,
        "have_attn_sink": True,
    }
    for s_kv in (8192, 32768, 65536)
]

KERNEL_META = {
    "name": "sparse_flashmla_prefill_head128_small_topk_phase1",
    "category": "flashmla",
    "compute_capability": 10,
}


def _cfg(**kwargs: Any) -> SparseFlashMLAPrefillHead128SmallTopKConfig:
    cfg_fields = {field.name for field in fields(SparseFlashMLAPrefillHead128SmallTopKConfig)}
    cfg_kwargs = {key: value for key, value in kwargs.items() if key in cfg_fields}
    if "label" not in cfg_kwargs:
        cfg_kwargs["label"] = "custom"
    cfg = SparseFlashMLAPrefillHead128SmallTopKConfig(**cfg_kwargs)
    cfg.validate()
    return cfg


def _flashmla_small_topk_dispatch_reason(cfg: SparseFlashMLAPrefillHead128SmallTopKConfig) -> str:
    if cfg.h_q != B_H:
        return "out_of_scope: h_q != 128 dispatches to head64 or unsupported path"
    if cfg.h_kv != 1:
        return "out_of_scope: h_kv != 1 violates FlashMLA sparse prefill phase1 assumptions"
    if cfg.d_qk != D_QK:
        return "out_of_scope: small-topk head128 supports only D_QK=512"
    if cfg.d_v != D_V:
        return "out_of_scope: d_v != 512"
    if cfg.topk > 1280:
        return "out_of_scope: topk > 1280 dispatches to regular head128 when supported"
    return "small_topk: sm100 head128 run_fwd_for_small_topk_phase1_kernel<Prefill, 512>"


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
        "dispatch_reason": _flashmla_small_topk_dispatch_reason(cfg),
    }


def _reference_sparse_prefill(
    case: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg: SparseFlashMLAPrefillHead128SmallTopKConfig = case["config"]
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
    iket = IketProfiler()
    # CUDA_TRANSCRIBE_START: phase1.cuh:24, scoped to KernelTemplate<Prefill, 512>.
    block_idx = T.cta_id([2 * s_q])
    T.cta_id_in_cluster([2])
    cta_idx: T.let = block_idx % 2
    thread_idx = T.thread_id([NUM_THREADS])
    T.warpgroup_id([NUM_THREADS // 128])
    T.warp_id_in_wg([4])
    T.lane_id([32])
    T.thread_id_in_wg([128])
    warp_idx: T.let = T.cuda.__shfl_sync(T.uint32(0xFFFFFFFF), thread_idx // 32, 0, 32)
    lane_idx: T.let = thread_idx % 32
    warpgroup_idx: T.let = T.cuda.__shfl_sync(T.uint32(0xFFFFFFFF), thread_idx // 128, 0, 32)
    idx_in_warpgroup: T.let = thread_idx % 128

    pool = T.SMEMPool()
    q_smem = pool.alloc_tcgen05_mma_AB((B_H // 2, D_QK), "bfloat16")
    k_smem = pool.alloc_tcgen05_mma_AB((NUM_K_BUFS * B_TOPK, D_QK // 2), "bfloat16")
    s_smem_gemm = pool.alloc_tcgen05_mma_AB(
        (B_H // 2, B_TOPK), "bfloat16", swizzle_mode=SwizzleMode.SWIZZLE_NONE
    )
    p_exchange = pool.alloc((4, (B_H // 2 // 2) * (B_TOPK // 2)), "uint32")
    rowwise_max_buf = pool.alloc((128,), "float32")
    rowwise_li_buf = pool.alloc((128,), "float32")
    is_k_valid = pool.alloc((NUM_INDEX_BUFS, B_TOPK // 8), "int8", align=16)

    bar_sQ_full = TMABar(pool, 1, leader=True)
    bar_tQ_empty = TCGen05Bar(pool, 1, leader=True)
    bar_tQ_full = TCGen05Bar(pool, 1, leader=True)
    bar_tOut_full = TCGen05Bar(pool, 1, leader=True)
    bar_tOut_empty = MBarrier(pool, 1, leader=True)
    bar_KV_full = TMABar(pool, NUM_K_BUFS, leader=True)
    bar_KV_empty = TCGen05Bar(pool, NUM_K_BUFS, leader=True)
    bar_P_empty = MBarrier(pool, 1, leader=True)
    bar_QK_done = TCGen05Bar(pool, 1, leader=True)
    bar_SV_done = TCGen05Bar(pool, 1, leader=True)
    bar_S_O_full = MBarrier(pool, 1, leader=True)
    bar_li_full = MBarrier(pool, 1, leader=True)
    bar_li_empty = MBarrier(pool, 1, leader=True)
    bar_valid_coord_scales_full = MBarrier(pool, NUM_INDEX_BUFS, leader=True)
    bar_valid_coord_scales_empty = MBarrier(pool, NUM_INDEX_BUFS, leader=True)
    bar_clc_full = TMABar(pool, 1, leader=True)
    bar_clc_empty = MBarrier(pool, 1, leader=True)
    clc_response = pool.alloc((4,), "uint32", align=16)
    tmem_start_addr = pool.alloc((1,), "uint32", align=4)
    pool.commit()
    tmem_pool = T.TMEMPool(pool, total_cols=512, cta_group=2, tmem_addr=tmem_start_addr)
    # O accumulator: one alloc; col halves = B lo/hi gemm outputs (physical col 0-127/128-255),
    # read back as a (128, D_V//2) datapath-D tile via permute+reshape.
    o_tmem = tmem_pool.alloc_tcgen05_mma_D(
        (B_H // 2, D_V), "float32", M=128, cta_group=2, group=(2, 2, 128)
    )
    o_win = o_tmem.rearrange("h (a b c) -> (b h) (a c)", a=2, b=2, c=128)
    # Q TMEM: one alloc at real 128-lane footprint, batched [2,M,K] head-dim fold (batch==lane-half);
    # q_tmem_fold[b,h,k]=Q[h,256b+k], lane-half b = contiguous D half [256b,+256) matching K gather.
    q_tmem_fold = tmem_pool.alloc_tcgen05_mma_A(
        (2, B_H // 2, D_QK // 2), "bfloat16", M=128, cta_group=2
    )
    tmem_p_col = T.meta_var(tmem_pool.offset)
    tmem_p = tmem_pool.alloc_tcgen05_mma_D((B_H // 2, B_TOPK * 2), "float32", M=128, cta_group=2)
    k_smem_gemm = k_smem.rearrange(
        "(kh row) (buf kl) -> buf row (kh kl)", kh=(D_QK // 2) // 64, buf=NUM_K_BUFS, kl=64
    )

    if warp_idx == 1:
        if T.ptx.elect_sync():
            bar_sQ_full.init(1)
            bar_tQ_empty.init(1)
            bar_tQ_full.init(1)
            bar_tOut_full.init(1)
            bar_tOut_empty.init(256)
            bar_P_empty.init(256)
            bar_QK_done.init(1)
            bar_SV_done.init(1)
            bar_S_O_full.init(256)
            bar_li_full.init(B_H // 2)
            bar_li_empty.init(128)
            bar_clc_full.init(1)
            bar_clc_empty.init(NUM_WORKER_THREADS)
            T.ptx.fence.mbarrier_init()
    elif warp_idx == 2:
        T.ptx.tcgen05.alloc(T.address_of(tmem_start_addr[0]), n_cols=512, cta_group=2)
        T.cuda.trap_when_assert_failed(tmem_start_addr[0] == T.uint32(0))
        T.ptx.tcgen05.relinquish_alloc_permit(cta_group=2)
    elif warp_idx == 3:
        if T.ptx.elect_sync():
            for init_stage in T.unroll(NUM_K_BUFS):
                T.ptx.mbarrier.init(bar_KV_full.ptr_to([init_stage]), 1)
                T.ptx.mbarrier.init(bar_KV_empty.ptr_to([init_stage]), 1)
            for init_stage in T.unroll(NUM_INDEX_BUFS):
                T.ptx.mbarrier.init(bar_valid_coord_scales_full.ptr_to([init_stage]), B_TOPK // 8)
                T.ptx.mbarrier.init(bar_valid_coord_scales_empty.ptr_to([init_stage]), 128)
            T.ptx.fence.mbarrier_init()

    T.cuda.cluster_sync()

    if warpgroup_idx == 0:
        # CUDA phase1.cuh:192-396. Q fetching and O write-back warpgroup.
        q_o_token = iket.range_start("h128-small-q-o")
        T.ptx.setmaxnreg(True, 160)

        @T.inline
        def issue_q_copy(q_s_q_idx, q_outer_loop_phase):
            if warp_idx == 0:
                if T.ptx.elect_sync():
                    T.ptx.cp_async.bulk.wait_group(0)
                    # Q's head-dim halves interleave per 64-elem chunk, matching the cp fold.
                    q_tma = q.rearrange(
                        "s h (half chunk inner) -> inner h half chunk s",
                        half=2,
                        chunk=D_QK // 64 // 2,
                        inner=64,
                    )
                    q_smem_tma = q_smem.rearrange(
                        "m (chunk c d0) -> d0 m c chunk", chunk=D_QK // 64 // 2, c=2
                    )
                    Tx.copy_async(
                        q_smem_tma[:, :, :, :],
                        q_tma.chunk((None, 2, None, None, None))[:, cta_idx, :, :, q_s_q_idx],
                        **tma_config(
                            mbar=leader_mbar(bar_sQ_full.ptr_to([0])),
                            cta_group=2,
                            cache_hint=T.uint64(0x12F0000000000000),
                        ),
                    )
                    if cta_idx == 0:
                        bar_sQ_full.arrive(0, tx_count=B_H * D_QK * BF16_BYTES)
                        bar_sQ_full.wait(0, q_outer_loop_phase)
                        bar_tQ_empty.wait(0, q_outer_loop_phase ^ 1)
                        T.ptx.tcgen05.fence.after_thread_sync()
                        q_tmem_cp = q_tmem_fold.rearrange("b h (dc di) -> h dc b di", di=64)
                        Tx.copy_async(
                            q_tmem_cp[:, :, :, :],
                            q_smem.view(B_H // 2, D_QK // 128, 2, 64)[:, :, :, :],
                            shape="128x256b",
                            cta_group=2,
                        )
                        bar_tQ_full.arrive(0, cta_group=2, cta_mask=3)

        @T.inline
        def perform_o_copy_out(o_s_q_idx, o_outer_loop_phase, is_last_o: T.constexpr):
            bar_li_full.wait(0, o_outer_loop_phase)
            output_scale: T.let = rowwise_li_buf[idx_in_warpgroup % 64]
            bar_li_empty.arrive(0)

            bar_tOut_full.wait(0, o_outer_loop_phase)
            if is_last_o:
                if T.ptx.elect_sync():
                    T.ptx.griddepcontrol.launch_dependents()

            o_epi_frag = T.alloc_tcgen05_ldst_frag("32x32b", (128, B_EPI), "float32")
            o_epi = o_epi_frag.local()
            o_epi_bf16_frag = T.alloc_tcgen05_ldst_frag("32x32b", (128, B_EPI), "bfloat16")
            q_smem_win = q_smem.rearrange("h (b r) -> (b h) r", b=2)
            for epi_k in T.unroll((D_V // 2) // B_EPI):
                Tx.wg.copy_async(
                    o_epi_frag[:, :], o_win.chunk((None, (D_V // 2) // B_EPI))[:, epi_k]
                )
                T.ptx.tcgen05.wait.ld()
                if epi_k == 0:
                    if is_last_o:
                        bar_tQ_full.wait(0, o_outer_loop_phase)
                    else:
                        bar_tQ_full.wait(0, o_outer_loop_phase ^ 1)
                if epi_k == ((D_V // 2) // B_EPI) - 1:
                    bar_tOut_empty.arrive(0, remote=T.uint32(0))
                Tx.wg.mul(o_epi_frag[:, :], o_epi_frag[:, :], output_scale)
                Tx.wg.cast(o_epi_bf16_frag[:, :], o_epi_frag[:, :])
                Tx.wg.copy(
                    q_smem_win.chunk((None, (D_V // 2) // B_EPI))[:, epi_k], o_epi_bf16_frag[:, :]
                )

            T.ptx.fence.proxy_async("shared::cta")
            T.ptx.bar.sync(BAR_WG0_SYNC, 128)
            if warp_idx == 0:
                if T.ptx.elect_sync():
                    Tx.copy_async(
                        out.chunk((None, 2, None))[o_s_q_idx, cta_idx, :],
                        q_smem[:, :],
                        **tma_config(),
                    )
                    T.ptx.cp_async.bulk.commit_group()

        wg0_job_valid: T.int32 = 1
        wg0_job_block_idx: T.int32 = block_idx
        wg0_outer_loop_phase: T.int32 = 0
        last_valid: T.int32 = 0
        last_s_q_idx: T.int32 = 0
        last_outer_loop_phase: T.int32 = 0

        while wg0_job_valid != 0:
            wg0_s_q_idx: T.let = wg0_job_block_idx // 2
            issue_q_copy(wg0_s_q_idx, wg0_outer_loop_phase)

            if last_valid != 0:
                perform_o_copy_out(last_s_q_idx, last_outer_loop_phase, False)
            else:
                bar_tQ_full.wait(0, wg0_outer_loop_phase)
            last_valid = 1
            last_s_q_idx = wg0_s_q_idx
            last_outer_loop_phase = wg0_outer_loop_phase

            bar_clc_full.wait(0, wg0_outer_loop_phase)
            wg0_next_job: T.let = T.ptx.clc_query_cancel(
                T.address_of(clc_response[0]), use_ld_acquire=True
            )
            T.ptx.mbarrier.arrive(bar_clc_empty.ptr_to([0]), remote=T.uint32(0), pred=True)
            if wg0_next_job == T.uint32(0xFFFFFFFF):
                wg0_job_valid = 0
            else:
                wg0_job_block_idx = T.cast(wg0_next_job, "int32")
            wg0_outer_loop_phase = wg0_outer_loop_phase ^ 1

        if last_valid != 0:
            if warp_idx == 0:
                if T.ptx.elect_sync():
                    T.ptx.cp_async.bulk.wait_group(0)
            T.ptx.bar.sync(BAR_WG0_SYNC, 128)
            perform_o_copy_out(last_s_q_idx, last_outer_loop_phase, True)

        if warp_idx == 0:
            T.ptx.tcgen05.dealloc(T.uint32(0), n_cols=512, cta_group=2)
        iket.range_end(q_o_token)

    elif warpgroup_idx == 1:
        # CUDA phase1.cuh:397-451. Prefill KV gather producer.
        kv_gather_token = iket.range_start("h128-small-kv-gather")
        T.ptx.setmaxnreg(False, 80)
        # Source uses canonical_warp_idx() here, not canonical_warp_idx_sync().
        wg1_warp_idx: T.let = thread_idx // 32 - 4
        if T.ptx.elect_sync():
            wg1_job_valid: T.int32 = 1
            wg1_job_block_idx: T.int32 = block_idx
            wg1_outer_loop_phase: T.int32 = 0
            wg1_rs: T.int32 = 0
            while wg1_job_valid != 0:
                wg1_s_q_idx: T.let = wg1_job_block_idx // 2
                wg1_topk_len: T.let = (
                    T.cuda.ldg(topk_length.ptr_to([wg1_s_q_idx]), "int32")
                    if have_topk_length
                    else topk
                )
                wg1_num_k_blocks: T.let = T.max((wg1_topk_len + B_TOPK - 1) // B_TOPK, 1)
                wg1_g_indices_base: T.let = wg1_s_q_idx * stride_indices_s_q

                for k in T.serial(0, wg1_num_k_blocks, unroll=False):
                    k_buf_idx: T.let = wg1_rs % NUM_K_BUFS
                    k_bar_phase: T.let = (wg1_rs // NUM_K_BUFS) & 1
                    cur_indices = T.alloc_local((WG1_ROWS_PER_WARP,), "int32")
                    for local_row in T.unroll(WG1_ROWS_PER_WARP // 8):
                        row: T.let = local_row * (4 * 8) + wg1_warp_idx * 8
                        row_base: T.let = wg1_g_indices_base + k * B_TOPK + row
                        Tx.copy(
                            cur_indices[local_row * 8 : local_row * 8 + 8],
                            indices[row_base : row_base + 8],
                            dispatch="vec_256b",
                            cache="nc",
                            l1_evict="L1::no_allocate",
                            l2_evict="L2::evict_first",
                            prefetch_size="L2::256B",
                        )
                    bar_KV_empty.wait(k_buf_idx, k_bar_phase ^ 1)
                    k_smem_gemm_cur = k_smem_gemm.sub[k_buf_idx]
                    src_col: T.let = cta_idx * (D_QK // 2)
                    # Rows interleave (row_group, warp, pair, lane4): this warp's 8-row stripes of each
                    # 64-col chunk. Col reshaped to (chunk,64); row picks this warp's rows via rank-preserving tile.
                    k_gather_tile = k_smem_gemm_cur.view(B_TOPK, (D_QK // 2) // 64, 64).tile(
                        0, (-1, 4, 2, 4)
                    )[:, wg1_warp_idx, :, :]
                    kv_tma = kv.view(
                        s_kv, D_QK, layout=TileLayout(S[(s_kv, D_QK) : (stride_kv_s_kv, 1)])
                    )
                    Tx.copy_async(
                        k_gather_tile[:, :, :],
                        kv_tma[:, src_col : src_col + D_QK // 2],
                        **_kv_gather_tma(
                            mbar=leader_mbar(bar_KV_full.ptr_to([k_buf_idx])),
                            indexer=[cur_indices[i] for i in range(WG1_ROWS_PER_WARP)],
                        ),
                    )
                    wg1_rs = wg1_rs + 1

                bar_clc_full.wait(0, wg1_outer_loop_phase)
                wg1_next_job: T.let = T.ptx.clc_query_cancel(
                    T.address_of(clc_response[0]), use_ld_acquire=True
                )
                T.ptx.mbarrier.arrive(bar_clc_empty.ptr_to([0]), remote=T.uint32(0), pred=True)
                if wg1_next_job == T.uint32(0xFFFFFFFF):
                    wg1_job_valid = 0
                else:
                    wg1_job_block_idx = T.cast(wg1_next_job, "int32")
                wg1_outer_loop_phase = wg1_outer_loop_phase ^ 1
        iket.range_end(kv_gather_token)

    elif warpgroup_idx == 2:
        # CUDA phase1.cuh:533-787. UMMA, valid-mask loading, and CLC producer.
        T.ptx.setmaxnreg(False, 80)

        if (warp_idx == 8) & (cta_idx == 0):
            mma_token = iket.range_start("h128-small-mma")
            if T.ptx.elect_sync():
                umma_job_valid: T.int32 = 1
                umma_job_block_idx: T.int32 = block_idx
                umma_outer_loop_phase: T.int32 = 0
                umma_rs: T.int32 = 0
                while umma_job_valid != 0:
                    umma_s_q_idx: T.let = umma_job_block_idx // 2
                    umma_topk_len: T.let = (
                        T.cuda.ldg(topk_length.ptr_to([umma_s_q_idx]), "int32")
                        if have_topk_length
                        else topk
                    )
                    umma_num_k_blocks: T.let = T.max((umma_topk_len + B_TOPK - 1) // B_TOPK, 1)
                    bar_tQ_full.wait(0, umma_outer_loop_phase)

                    for k in T.serial(0, umma_num_k_blocks + 1, unroll=False):
                        if k < umma_num_k_blocks:
                            k_buf_idx: T.let = umma_rs % NUM_K_BUFS
                            k_bar_phase: T.let = (umma_rs // NUM_K_BUFS) & 1
                            p_bar_phase: T.let = umma_rs & 1
                            bar_P_empty.wait(0, p_bar_phase ^ 1)
                            bar_KV_full.arrive(k_buf_idx, tx_count=B_TOPK * D_QK * BF16_BYTES)
                            bar_KV_full.wait(k_buf_idx, k_bar_phase)
                            T.ptx.tcgen05.fence.after_thread_sync()
                            qk_accumulate: T.uint32 = 0
                            Tx.gemm_async(
                                tmem_p[:, :],
                                q_tmem_fold[:, :, :],
                                k_smem_gemm[k_buf_idx, :, :],
                                **_mma_config(accum=qk_accumulate),
                            )
                            qk_accumulate = T.uint32(1)
                            bar_QK_done.arrive(0, cta_group=2, cta_mask=3)
                            if k == umma_num_k_blocks - 1:
                                T.ptx.tcgen05.commit(
                                    bar_tQ_empty.ptr_to([0]), cta_group=2, cta_mask=0
                                )

                        if k > 0:
                            prev_k: T.let = k - 1
                            prev_rs: T.let = umma_rs - 1
                            prev_buf: T.let = prev_rs % NUM_K_BUFS
                            prev_s_o_phase: T.let = prev_rs & 1
                            bar_S_O_full.wait(0, prev_s_o_phase)
                            if prev_k == 0:
                                bar_tOut_empty.wait(0, umma_outer_loop_phase ^ 1)
                            T.ptx.tcgen05.fence.after_thread_sync()
                            o_accumulate: T.uint32 = T.if_then_else(
                                prev_k == 0, T.uint32(0), T.uint32(1)
                            )

                            @T.inline
                            def gemm_o(dst, col_lo, col_hi):
                                Tx.gemm_async(
                                    dst[:, :],
                                    s_smem_gemm[:, :],
                                    k_smem_gemm[prev_buf, :, col_lo:col_hi],
                                    transB=True,
                                    **_mma_config(accum=o_accumulate),
                                )

                            gemm_o(o_tmem.sub[:, 0 : D_V // 2], 0, D_V // 4)
                            gemm_o(o_tmem.sub[:, D_V // 2 : D_V], D_V // 4, D_V // 2)
                            o_accumulate = T.uint32(1)
                            bar_SV_done.arrive(0, cta_group=2, cta_mask=3)
                            bar_KV_empty.arrive(prev_buf, cta_group=2, cta_mask=3)

                        if k != umma_num_k_blocks:
                            umma_rs = umma_rs + 1

                    T.ptx.tcgen05.fence.before_thread_sync()
                    bar_tOut_full.arrive(0, cta_group=2, cta_mask=3)

                    bar_clc_full.wait(0, umma_outer_loop_phase)
                    umma_next_job: T.let = T.ptx.clc_query_cancel(
                        T.address_of(clc_response[0]), use_ld_acquire=True
                    )
                    T.ptx.mbarrier.arrive(bar_clc_empty.ptr_to([0]), remote=T.uint32(0), pred=True)
                    if umma_next_job == T.uint32(0xFFFFFFFF):
                        umma_job_valid = 0
                    else:
                        umma_job_block_idx = T.cast(umma_next_job, "int32")
                    umma_outer_loop_phase = umma_outer_loop_phase ^ 1
            iket.range_end(mma_token)

        elif warp_idx == 9:
            valid_mask_token = iket.range_start("h128-small-valid-mask")
            if lane_idx < B_TOPK // 8:
                lane_indices = T.alloc_local((8,), "int32")
                valid_job_valid: T.int32 = 1
                valid_job_block_idx: T.int32 = block_idx
                valid_outer_loop_phase: T.int32 = 0
                valid_rs: T.int32 = 0
                while valid_job_valid != 0:
                    valid_s_q_idx: T.let = valid_job_block_idx // 2
                    valid_topk_len: T.let = (
                        T.cuda.ldg(topk_length.ptr_to([valid_s_q_idx]), "int32")
                        if have_topk_length
                        else topk
                    )
                    valid_num_k_blocks: T.let = T.max((valid_topk_len + B_TOPK - 1) // B_TOPK, 1)
                    valid_g_indices_base: T.let = valid_s_q_idx * stride_indices_s_q
                    for k in T.serial(0, valid_num_k_blocks, unroll=False):
                        row_base: T.let = valid_g_indices_base + k * B_TOPK + lane_idx * 8
                        Tx.copy(
                            lane_indices[0:8],
                            indices[row_base : row_base + 8],
                            dispatch="vec_256b",
                            cache="nc",
                            l1_evict="L1::no_allocate",
                            l2_evict="L2::evict_normal",
                            prefetch_size="L2::256B",
                        )
                        abs_pos_start: T.let = k * B_TOPK
                        mask: T.let = pack_valid_mask8(
                            lane_indices, abs_pos_start, lane_idx, valid_topk_len, s_kv
                        )
                        index_buf_idx: T.let = valid_rs % NUM_INDEX_BUFS
                        index_bar_phase: T.let = (valid_rs // NUM_INDEX_BUFS) & 1
                        bar_valid_coord_scales_empty.wait(index_buf_idx, index_bar_phase ^ 1)
                        is_k_valid[index_buf_idx, lane_idx] = mask
                        bar_valid_coord_scales_full.arrive(index_buf_idx)
                        valid_rs = valid_rs + 1

                    bar_clc_full.wait(0, valid_outer_loop_phase)
                    valid_next_job: T.let = T.ptx.clc_query_cancel(
                        T.address_of(clc_response[0]), use_ld_acquire=True
                    )
                    T.ptx.mbarrier.arrive(bar_clc_empty.ptr_to([0]), remote=T.uint32(0), pred=True)
                    if valid_next_job == T.uint32(0xFFFFFFFF):
                        valid_job_valid = 0
                    else:
                        valid_job_block_idx = T.cast(valid_next_job, "int32")
                    valid_outer_loop_phase = valid_outer_loop_phase ^ 1
            iket.range_end(valid_mask_token)

        elif warp_idx >= 10:
            clc_token = iket.sentinel_token("h128-small-clc")
            if warp_idx == 10:
                clc_token = iket.range_start("h128-small-clc")
            if T.ptx.elect_sync():
                if warp_idx == 10:
                    clc_job_valid: T.int32 = 1
                    clc_outer_loop_phase: T.int32 = 0
                    while clc_job_valid != 0:
                        if cta_idx == 0:
                            bar_clc_empty.wait(0, clc_outer_loop_phase ^ 1)
                            T.ptx.clc_try_cancel(
                                T.address_of(clc_response[0]), bar_clc_full.ptr_to([0])
                            )
                        bar_clc_full.arrive(0, tx_count=16)

                        bar_clc_full.wait(0, clc_outer_loop_phase)
                        clc_next_job: T.let = T.ptx.clc_query_cancel(
                            T.address_of(clc_response[0]), use_ld_acquire=True
                        )
                        T.ptx.mbarrier.arrive(
                            bar_clc_empty.ptr_to([0]), remote=T.uint32(0), pred=True
                        )
                        if clc_next_job == T.uint32(0xFFFFFFFF):
                            clc_job_valid = 0
                        clc_outer_loop_phase = clc_outer_loop_phase ^ 1
            iket.range_end(clc_token)

    else:
        # CUDA phase1.cuh:788-921. Scale/exp warpgroup.
        softmax_token = iket.range_start("h128-small-softmax")
        T.ptx.setmaxnreg(True, 160)
        local_warp_idx: T.let = warp_idx - 12
        wg3_job_valid: T.int32 = 1
        wg3_job_block_idx: T.int32 = block_idx
        wg3_outer_loop_phase: T.int32 = 0
        wg3_rs: T.int32 = 0
        while wg3_job_valid != 0:
            wg3_s_q_idx: T.let = wg3_job_block_idx // 2
            wg3_topk_len: T.let = (
                T.cuda.ldg(topk_length.ptr_to([wg3_s_q_idx]), "int32") if have_topk_length else topk
            )
            wg3_num_k_blocks: T.let = T.max((wg3_topk_len + B_TOPK - 1) // B_TOPK, 1)
            mi: T.float32 = MAX_INIT_VAL
            li: T.float32 = 0.0
            real_mi: T.float32 = T.float32(-float("inf"))
            scale_pair: T.let = T.cuda.make_float2(sm_scale_div_log2, sm_scale_div_log2)

            for k in T.serial(0, wg3_num_k_blocks, unroll=False):
                k_buf_idx: T.let = wg3_rs % NUM_K_BUFS
                k_bar_phase: T.let = (wg3_rs // NUM_K_BUFS) & 1
                index_buf_idx: T.let = wg3_rs % NUM_INDEX_BUFS
                index_bar_phase: T.let = (wg3_rs // NUM_INDEX_BUFS) & 1
                bar_valid_coord_scales_full.wait(index_buf_idx, index_bar_phase)
                p_frag = T.alloc_tcgen05_ldst_frag("32x32b", (128, WG3_ELEMS_PER_THREAD), "float32")
                p_peer_frag = T.alloc_tcgen05_ldst_frag(
                    "32x32b", (128, WG3_ELEMS_PER_THREAD), "float32"
                )
                p = p_frag.local().view("uint32")
                p_peer = p_peer_frag.local().view("uint32")
                bar_QK_done.wait(0, wg3_rs & 1)
                T.ptx.tcgen05.fence.after_thread_sync()

                @T.inline
                def load_p(lo_dst, hi_dst):
                    # datapath-B P read back as (128, B_TOPK) identity: merge the
                    # two lane-halves into 128 rows.
                    p_win = tmem_p.rearrange("h (b t) -> (b h) t", b=2)
                    Tx.wg.copy_async(lo_dst[:, :], p_win.chunk((None, 2))[:, 0])
                    Tx.wg.copy_async(hi_dst[:, :], p_win.chunk((None, 2))[:, 1])

                if local_warp_idx < 2:
                    load_p(p_frag, p_peer_frag)
                else:
                    load_p(p_peer_frag, p_frag)
                T.ptx.tcgen05.wait.ld()
                T.ptx.tcgen05.fence.before_thread_sync()
                bar_P_empty.arrive(0, remote=T.uint32(0))

                valid_word_offset: T.let = T.if_then_else(
                    local_warp_idx >= 2, WG3_ELEMS_PER_THREAD // 32, 0
                )
                is_k_valid_u32: T.let = is_k_valid.view("uint32")[index_buf_idx, valid_word_offset]
                for p_i in T.unroll(WG3_ELEMS_PER_THREAD):
                    invalid_p_predicate: T.let = T.bitwise_and(
                        T.shift_right(is_k_valid_u32, T.uint32(p_i)), T.uint32(1)
                    ) == T.uint32(0)
                    p[p_i] = T.if_then_else(invalid_p_predicate, T.uint32(0xFF800000), p[p_i])

                for exchange_i in T.unroll(WG3_ELEMS_PER_THREAD // 4):
                    exchange_offset = exchange_i * 32 * 4 + lane_idx * 4
                    p_peer_offset: T.let = exchange_i * 4
                    Tx.copy(
                        p_exchange[local_warp_idx ^ 2, exchange_offset : exchange_offset + 4],
                        p_peer[p_peer_offset : p_peer_offset + 4],
                        dispatch="vec_128b",
                    )
                T.ptx.bar.sync(BAR_WG2_WARP02 + (local_warp_idx & 1), 64)
                for exchange_i in T.unroll(WG3_ELEMS_PER_THREAD // 4):
                    exchange_offset = exchange_i * 32 * 4 + lane_idx * 4
                    p_exchange_tmp = T.alloc_local((4,), "uint32")
                    Tx.copy(
                        p_exchange_tmp[0:4],
                        p_exchange[local_warp_idx, exchange_offset : exchange_offset + 4],
                        dispatch="vec_128b",
                    )
                    p_pair0: T.let = T.cuda.make_float2(
                        T.cuda.uint_as_float(p[exchange_i * 4]),
                        T.cuda.uint_as_float(p[exchange_i * 4 + 1]),
                    )
                    peer_pair0: T.let = T.cuda.make_float2(
                        T.cuda.uint_as_float(p_exchange_tmp[0]),
                        T.cuda.uint_as_float(p_exchange_tmp[1]),
                    )
                    sum_pair0: T.let = T.ptx.add_f32x2(p_pair0, peer_pair0, dps=False)
                    p[exchange_i * 4] = T.cuda.float_as_uint(T.cuda.float2_x(sum_pair0))
                    p[exchange_i * 4 + 1] = T.cuda.float_as_uint(T.cuda.float2_y(sum_pair0))
                    p_pair1: T.let = T.cuda.make_float2(
                        T.cuda.uint_as_float(p[exchange_i * 4 + 2]),
                        T.cuda.uint_as_float(p[exchange_i * 4 + 3]),
                    )
                    peer_pair1: T.let = T.cuda.make_float2(
                        T.cuda.uint_as_float(p_exchange_tmp[2]),
                        T.cuda.uint_as_float(p_exchange_tmp[3]),
                    )
                    sum_pair1: T.let = T.ptx.add_f32x2(p_pair1, peer_pair1, dps=False)
                    p[exchange_i * 4 + 2] = T.cuda.float_as_uint(T.cuda.float2_x(sum_pair1))
                    p[exchange_i * 4 + 3] = T.cuda.float_as_uint(T.cuda.float2_y(sum_pair1))

                cur_pi_max: T.float32 = T.float32(-float("inf"))
                for p_i in T.unroll(WG3_ELEMS_PER_THREAD):
                    cur_pi_max = T.max(cur_pi_max, T.cuda.uint_as_float(p[p_i]))
                cur_pi_max = cur_pi_max * sm_scale_div_log2
                rowwise_max_buf[idx_in_warpgroup] = cur_pi_max
                T.ptx.bar.sync(BAR_WG2_WARP02 + (local_warp_idx & 1), 64)
                cur_pi_max = T.max(cur_pi_max, rowwise_max_buf[idx_in_warpgroup ^ 64])
                real_mi = T.max(real_mi, cur_pi_max)
                should_scale_o: T.let = (
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

                # S frag: warpgroup-distributed (B_H//2, B_TOPK) tile. Thread idx owns row h = idx%64
                # and k half [32*(idx//64), +32) in 8-elem chunks (from the packing loop below).
                s_frag = T.alloc_buffer(
                    (B_H // 2, B_TOPK),
                    "bfloat16",
                    scope="local",
                    layout=TileLayout(
                        S[(2, 32, 2, B_TOPK // 2) : (1 @ wid_in_wg, 1 @ laneid, 2 @ wid_in_wg, 1)]
                    ),
                )
                s_pack = s_frag.local().view("uint32")
                cur_sum_pair: T.uint64 = T.cuda.make_float2(T.float32(0.0), T.float32(0.0))
                neg_new_max_pair: T.let = T.cuda.make_float2(-new_max, -new_max)
                for s_i in T.unroll(WG3_ELEMS_PER_THREAD // 2):
                    p_pair: T.let = T.cuda.make_float2(
                        T.cuda.uint_as_float(p[s_i * 2]), T.cuda.uint_as_float(p[s_i * 2 + 1])
                    )
                    fma_pair: T.let = T.ptx.fma_f32x2(
                        p_pair, scale_pair, neg_new_max_pair, dps=False
                    )
                    s_x: T.let = T.ptx.exp2(T.cuda.float2_x(fma_pair))
                    s_y: T.let = T.ptx.exp2(T.cuda.float2_y(fma_pair))
                    s_pair: T.let = T.cuda.make_float2(s_x, s_y)
                    cur_sum_pair = T.ptx.add_f32x2(cur_sum_pair, s_pair, dps=False)
                    s_pack[s_i] = T.cuda.float22bfloat162_rn(s_x, s_y)
                cur_sum: T.let = T.cuda.float2_x(cur_sum_pair) + T.cuda.float2_y(cur_sum_pair)
                li_tmp: T.float32
                T.ptx.fma_f32(T.address_of(li_tmp), li, scale_for_old, cur_sum)
                li = li_tmp

                bar_SV_done.wait(0, (wg3_rs & 1) ^ 1)
                Tx.wg.copy(s_smem_gemm[:, :], s_frag[:, :])

                if (k > 0) & should_scale_o:
                    T.ptx.tcgen05.fence.after_thread_sync()
                    o_rescale_frag = T.alloc_tcgen05_ldst_frag("32x32b", (128, 32), "float32")
                    for chunk_idx in T.unroll((D_V // 2) // 32):
                        Tx.wg.copy_async(
                            o_rescale_frag[:, :],
                            o_win.chunk((None, (D_V // 2) // 32))[:, chunk_idx],
                        )
                        T.ptx.tcgen05.wait.ld()
                        Tx.wg.mul(o_rescale_frag[:, :], o_rescale_frag[:, :], scale_for_old)
                        Tx.wg.copy_async(
                            o_win.chunk((None, (D_V // 2) // 32))[:, chunk_idx],
                            o_rescale_frag[:, :],
                        )
                        T.ptx.tcgen05.wait.st()
                    T.ptx.tcgen05.fence.before_thread_sync()

                T.ptx.fence.proxy_async("shared::cta")
                bar_S_O_full.arrive(0, remote=T.uint32(0))
                bar_valid_coord_scales_empty.arrive(index_buf_idx)
                wg3_rs = wg3_rs + 1

            if real_mi == T.float32(-float("inf")):
                li = 0.0
                mi = T.float32(-float("inf"))

            bar_li_empty.wait(0, wg3_outer_loop_phase ^ 1)
            rowwise_li_buf[idx_in_warpgroup ^ 64] = li
            T.ptx.bar.sync(BAR_WG2_SYNC, 128)
            li = li + rowwise_li_buf[idx_in_warpgroup]

            if idx_in_warpgroup < B_H // 2:
                head_idx: T.let = cta_idx * (B_H // 2) + idx_in_warpgroup
                attn_sink_log2: T.let = (
                    T.cuda.ldg(attn_sink.ptr_to([head_idx]), "float32") * LOG_2_E
                    if have_attn_sink
                    else T.float32(-float("inf"))
                )
                output_scale: T.let = T.cuda.fdividef(
                    T.float32(1.0), li + T.ptx.exp2(attn_sink_log2 - mi)
                )
                rowwise_li_buf[idx_in_warpgroup] = T.if_then_else(li == 0.0, 0.0, output_scale)
                bar_li_full.arrive(0)
                cur_lse: T.float32
                T.ptx.fma_f32(T.address_of(cur_lse), mi, LN_2, T.log(li))
                cur_lse = T.if_then_else(
                    cur_lse == T.float32(-float("inf")), T.float32(float("inf")), cur_lse
                )
                max_logits[wg3_s_q_idx, head_idx] = real_mi * LN_2
                lse[wg3_s_q_idx, head_idx] = cur_lse

            bar_clc_full.wait(0, wg3_outer_loop_phase)
            wg3_next_job: T.let = T.ptx.clc_query_cancel(
                T.address_of(clc_response[0]), use_ld_acquire=True
            )
            T.ptx.mbarrier.arrive(bar_clc_empty.ptr_to([0]), remote=T.uint32(0), pred=True)
            if wg3_next_job == T.uint32(0xFFFFFFFF):
                wg3_job_valid = 0
            else:
                wg3_job_block_idx = T.cast(wg3_next_job, "int32")
            wg3_outer_loop_phase = wg3_outer_loop_phase ^ 1
        iket.range_end(softmax_token)

    T.cuda.cluster_sync()


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
        raise SkipTest("CUDA is required for sparse FlashMLA head128 small-topk phase1")

    from tirx_kernels.runner import compile_kernel

    case = prepare_data(**kwargs)
    cfg: SparseFlashMLAPrefillHead128SmallTopKConfig = case["config"]
    if not case["dispatch_reason"].startswith("small_topk:"):
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
        raise SkipTest("CUDA is required for sparse FlashMLA head128 small-topk phase1 benchmark")

    from tirx_kernels.runner import compile_kernel
    from tvm.tirx.bench import bench

    case = prepare_data(**kwargs)
    if not case["dispatch_reason"].startswith("small_topk:"):
        raise SkipTest(case["dispatch_reason"])
    prim_func = get_kernel(**kwargs)
    ex = compile_kernel(prim_func)

    # Allocate inputs once, outside the timed region (Triton-standard pure launch).
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

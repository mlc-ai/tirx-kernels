from __future__ import annotations

import math
import os
from enum import Enum

import numpy as np
import torch

import tvm
import tvm.testing
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.bench import CudaProfiler, bench, tensor_bytes
from tvm.tirx.lang.pipeline import MBarrier, Pipeline, PipelineState, TCGen05Bar
from tvm.tirx.lang.tile_scheduler import FlashAttentionLinearScheduler, FlashAttentionLPTScheduler
from tvm.tirx.layout import wg_local_layout

M_CLUSTER = 1
N_CLUSTER = 1
SM_NUMBER = 148
NUM_GROUPS = 6
PROFILER_BUFFER_SIZE = int(2000000.0)
PROFILER_WRITE_STRIDE = SM_NUMBER * NUM_GROUPS
PROFILER_ON = False
# Lightweight softmax-step timestamps (mirrors instr/flash_fwd_sm100_instr.py
# t0..t3): leader-lane STG of %globaltimer_lo straight to profiler_buffer,
# no fences, no local arrays — CudaProfiler's per-event __threadfence_block
# inflates a single-CTA task ~10x; this stays ~1%. Parse with
# instr/tirx_phase.py --light.
LIGHT_TS = False
# ex2-emulation ratio per regime (grid-searched under the bench protocol;
# heavier and lighter both measured worse on their respective regimes):
# emulate elements with (i*2 % 16) >= 16 - 2*PAIRS in fragments
# [START, 3). Causal keeps fragment 0; non-causal skips it.
EMU_PAIRS_CAUSAL = 2
EMU_START_CAUSAL = 0
EMU_PAIRS_NC = 2
EMU_START_NC = 1


def ceildiv(a, b):
    return (a + b - 1) // b


def shl_u32_clamp(val, shift):
    """Left shift with PTX clamping (shift>=32 -> 0). Lets the causal mask keep
    the fused ``max(col_limit - s*CHUNK, 0)`` (one VIADDMNMX, like cutedsl) and
    drop the ``min(.,CHUNK)`` clamp (which adds a VIMNMX above cutedsl's count):
    ``~shl_clamp(0xFFFFFFFF, k)`` is the low-k-bits mask for any k, no min."""
    func_name = "shl_u32_clamp"
    source_code = (
        f"\n__device__ __forceinline__ unsigned int {func_name}"
        "(unsigned int val, unsigned int shift) {\n"
        "  unsigned int r;\n"
        '  asm("shl.b32 %0, %1, %2;" : "=r"(r) : "r"(val), "r"(shift));\n'
        "  return r;\n}\n"
    )
    return T.cuda.func_call(func_name, val, shift, source_code=source_code, return_type="uint32")


def combine_int_frac_ex2(x_rounded, frac_ex2):
    func_name = "combine_int_frac_ex2"
    source_code = f'\n__device__ __forceinline__ float {func_name}(float x_rounded, float frac_ex2) {{\n  float out;\n  asm volatile(\n    "{{\\n\\t"\n    ".reg .s32 x_rounded_i, frac_ex_i, x_rounded_e, out_i;\\n\\t"\n    "mov.b32 x_rounded_i, %1;\\n\\t"\n    "mov.b32 frac_ex_i, %2;\\n\\t"\n    "shl.b32 x_rounded_e, x_rounded_i, 23;\\n\\t"\n    "add.s32 out_i, x_rounded_e, frac_ex_i;\\n\\t"\n    "mov.b32 %0, out_i;\\n\\t"\n    "}}\\n"\n    : "=f"(out) : "f"(x_rounded), "f"(frac_ex2));\n  return out;\n}}\n'
    return T.cuda.func_call(
        func_name, x_rounded, frac_ex2, source_code=source_code, return_type="float32"
    )


def get_n_block_max(m_block_idx, causal, SEQ_LEN_KV, SEQ_LEN_Q, SEQ_Q_PER_TILE):
    """Maximum KV block index (exclusive) for this Q block."""
    n_block_max = ceildiv(SEQ_LEN_KV, BLK_N)
    if not causal:
        return n_block_max
    m_idx_max = (m_block_idx + 1) * SEQ_Q_PER_TILE * SMEM_PIPE_DEPTH_Q
    n_idx = m_idx_max + SEQ_LEN_KV - SEQ_LEN_Q
    return T.min(n_block_max, ceildiv(n_idx, BLK_N))


def get_n_block_min_causal_mask(m_block_idx, SEQ_LEN_KV, SEQ_LEN_Q, SEQ_Q_PER_TILE):
    """KV block index where causal masking stops being needed."""
    m_idx_min = m_block_idx * SEQ_Q_PER_TILE * SMEM_PIPE_DEPTH_Q
    n_idx = m_idx_min + SEQ_LEN_KV - SEQ_LEN_Q
    return T.max(0, n_idx // BLK_N)


@T.inline
def ex2_emulation_2(out, idx, x, y):
    poly_ex2_deg3 = T.meta_var((1.0, 0.6951461434364319, 0.22756439447402954, 0.07711908966302872))
    fp32_round_int = T.meta_var(float(2**23 + 2**22))
    xy_clamped: T.f32[2]
    xy_clamped[0] = T.max(x, -127.0)
    xy_clamped[1] = T.max(y, -127.0)
    xy_rounded: T.f32[2]
    Tx.add(xy_rounded, xy_clamped, fp32_round_int, rounding_mode="rm")
    xy_rounded_back: T.f32[2]
    Tx.sub(xy_rounded_back, xy_rounded, fp32_round_int, rounding_mode="rn")
    xy_frac: T.f32[2]
    Tx.sub(xy_frac, xy_clamped, xy_rounded_back, rounding_mode="rn")
    xy_frac_ex2: T.f32[2]
    xy_frac_ex2[0] = poly_ex2_deg3[3]
    xy_frac_ex2[1] = poly_ex2_deg3[3]
    Tx.fma(xy_frac_ex2, xy_frac_ex2, xy_frac, poly_ex2_deg3[2])
    Tx.fma(xy_frac_ex2, xy_frac_ex2, xy_frac, poly_ex2_deg3[1])
    Tx.fma(xy_frac_ex2, xy_frac_ex2, xy_frac, poly_ex2_deg3[0])
    out[idx] = combine_int_frac_ex2(xy_rounded[0], xy_frac_ex2[0])
    out[idx + 1] = combine_int_frac_ex2(xy_rounded[1], xy_frac_ex2[1])


class ProfileEventType(Enum):
    IssueTMA_Q = 0
    IssueTMA_K = 1
    IssueTMA_V = 2
    IssueMMA_QK = 3
    IssueMMA_PV = 4
    Softmax_MAX = 5
    Softmax_FMA = 6
    Softmax_EXP2 = 7
    Softmax_TMEM_ST = 8
    Softmax_SUM = 9
    Correction = 10
    EpiLDTMEM = 11
    TMAStore = 12


event_type_names = [
    "issue-tma-q",
    "issue-tma-k",
    "issue-tma-v",
    "issue-mma-qk",
    "issue-mma-pv",
    "softmax-max",
    "softmax-fma",
    "softmax-exp2",
    "softmax-tmem-st",
    "softmax-sum",
    "correction",
    "epi-ld-tmem",
    "tma-store",
]
WG_NUMBER = 4
WARP_NUMBER = 4
NUM_THREADS = 32 * WARP_NUMBER * WG_NUMBER
N_COLS_TMEM = 512
TMEM_PIPE_DEPTH = 2
SMEM_PIPE_DEPTH_Q = 2
SMEM_PIPE_DEPTH_KV = 3
BLK_M = 128
BLK_N = 128
BLK_K = 64
SOFTMAX_LD_CHUNK = 32
SOFTMAX_ST_CHUNK = 32
EPI_TILE = 64
TMEM_EPI_LD_SIZE = 16
USE_S0_S1_BARRIER = False
MMA_M = 128
MMA_N = 128
MMA_K = 16
F16_BYTES = 2
F32_BYTES = 4
F128_BYTES = 16
a_type_qk = tvm.DataType("float16")
b_type_qk = tvm.DataType("float16")
d_type_qk = tvm.DataType("float32")
a_type_pv = tvm.DataType("float16")
b_type_pv = tvm.DataType("float16")
d_type_pv = tvm.DataType("float32")


@T.jit
def _kernel(
    Q: T.Buffer((BATCH_SIZE, SEQ_LEN_Q, NUM_QO_HEADS, HEAD_DIM), "float16"),
    K: T.Buffer((BATCH_SIZE, SEQ_LEN_KV, NUM_KV_HEADS, HEAD_DIM), "float16"),
    V: T.Buffer((BATCH_SIZE, SEQ_LEN_KV, NUM_KV_HEADS, HEAD_DIM), "float16"),
    O: T.Buffer((BATCH_SIZE, SEQ_LEN_Q, NUM_QO_HEADS, HEAD_DIM), "float16"),
    profiler_buffer: T.Buffer((PROFILER_BUFFER_SIZE,), "uint64"),
    *,
    BATCH_SIZE: T.constexpr,
    SEQ_LEN_Q: T.constexpr,
    SEQ_LEN_KV: T.constexpr,
    NUM_QO_HEADS: T.constexpr,
    NUM_KV_HEADS: T.constexpr,
    HEAD_DIM: T.constexpr,
    is_causal: T.constexpr = False,
    CTA_GROUP: T.constexpr = 1,
    TMEM_PIPE_DEPTH: T.constexpr = TMEM_PIPE_DEPTH,
    SMEM_PIPE_DEPTH_KV: T.constexpr = SMEM_PIPE_DEPTH_KV,
):
    GQA_RATIO = T.meta_var(NUM_QO_HEADS // NUM_KV_HEADS)
    SEQ_Q_PER_TILE = T.meta_var(BLK_M // GQA_RATIO)
    # HW named barrier for the softmax->correction stats handshake
    # (cf. FA4 CuTeDSL sm_stats_barrier). One 256-thread barrier per stage
    # (1 + stage): the whole softmax wg arrives, the whole correction wg
    # syncs — collective release exactly like the 128-count mbarrier it
    # replaces, but with HW-barrier wake instead of mbarrier
    # try_wait+nanosleep (that wait gates p_o_rescale and thus the PV MMA).
    # Per-warp pairing (softmax warp w <-> correction warp w) measured 12%
    # WORSE on GQA-packed causal: uneven per-warp softmax finish times make
    # pairwise rendezvous slower than collective release. The reverse
    # (sScale slot reuse) direction stays on softmax_corr.empty.
    # Stats handshake barrier WIDTH. cutedsl uses a 64-thread PAIRWISE
    # barrier (softmax warp w <-> correction warp w, sm_stats_barrier
    # index stage*4+warp). TIRx adopted a 256-thread COLLECTIVE barrier
    # because per-warp pairing regressed GQA-PACKED causal 12% (uneven
    # per-warp softmax finish times -> pairwise waits for the slow
    # partner). But for GQA_RATIO==1 the rows within a warp are uniform,
    # so pairwise rendezvous releases each correction warp as soon as ITS
    # partner softmax warp finishes — no waiting for the slowest of all
    # 128. ncu (s2048_h32kv32_c): the 256-collective shows a 3.7-cycle
    # barrier stall (32% of 11.6 cyc/issue) that cutedsl's 64-pairwise
    # lacks (cutedsl 10.8 cyc/issue). Gate on GQA==1.
    STATS_BAR_PAIRWISE = T.meta_var(GQA_RATIO == 1)
    L2_SIZE = T.meta_var(50 * 1024 * 1024)
    SIZE_ONE_KV_HEAD = T.meta_var(SEQ_LEN_KV * HEAD_DIM * 2 * F16_BYTES)
    L2_SWIZZLE = T.meta_var(
        1 if L2_SIZE < SIZE_ONE_KV_HEAD else 1 << int(math.log2(L2_SIZE // SIZE_ONE_KV_HEAD))
    )
    SSCALE_TOTAL_SIZE = T.meta_var(2 * SMEM_PIPE_DEPTH_Q * BLK_M)
    assert TMEM_PIPE_DEPTH * MMA_N <= N_COLS_TMEM, "TMEM columns exceeded"
    num_q_blocks_total = T.meta_var(ceildiv(SEQ_LEN_Q, SEQ_Q_PER_TILE))
    num_q_blocks = T.meta_var(ceildiv(num_q_blocks_total, SMEM_PIPE_DEPTH_Q))
    num_total_tasks = T.meta_var(BATCH_SIZE * NUM_KV_HEADS * num_q_blocks)
    # When every CTA runs exactly ONE task (causal always launches
    # cta_count == num_total_tasks; non-causal whenever tasks <= 148), the
    # task tail is fully exposed on the critical path — there is no next
    # task to overlap the correction-wg epilogue with. In that regime the
    # epilogue runs on the SOFTMAX warpgroups instead: each wg handles its
    # own stage, so the two stages epilogue IN PARALLEL (correction did
    # them serially), row_sum is normalized straight from the register
    # (no sScale round-trip, no tail stats-handshake trip), and the idle
    # 200-reg softmax budget affords 32-wide TMEM loads (4 round trips vs
    # 16-wide x 8). Single-task locked-clock decomposition vs cutedsl
    # (q256 kv2048 causal): tail 7.2/6.6us vs baseline 6.6/5.8 — the tail
    # is where the small-shape ratio lives. Multi-task persistent shapes
    # keep the correction-wg epilogue (it overlaps the next task there).
    # Measured (ncu back-to-back, GPU 6): causal -1.9..-2.6%
    # (s1024_h32kv32_c 22.05→21.47us, s1024_h32kv16_c 20.16→19.78);
    # NON-causal single-wave (s1024_h32kv32) consistently +2% — restrict
    # to causal, where the LPT longest-task tail is the kernel's critical
    # path.
    EPI_ON_SOFTMAX = T.meta_var(is_causal)
    EARLY_Q_RELEASE = T.meta_var(not is_causal)
    max_ctas: T.let = 148
    cta_count: T.let = T.min(max_ctas, num_total_tasks) if not is_causal else num_total_tasks
    T.device_entry()
    T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})
    bx = T.cta_id([cta_count])
    wg_id = T.warpgroup_id([4])
    warp_id = T.warp_id_in_wg([4])
    lane_id = T.lane_id([32])
    tid_in_wg = T.thread_id_in_wg([128])
    pool = T.SMEMPool()
    Q_smem = pool.alloc_mma((SMEM_PIPE_DEPTH_Q, BLK_M, HEAD_DIM), "float16")
    K_smem = pool.alloc_mma((SMEM_PIPE_DEPTH_KV, BLK_N, HEAD_DIM), "float16")
    V_smem = K_smem.view(SMEM_PIPE_DEPTH_KV, BLK_N, HEAD_DIM)
    O_smem = pool.alloc_mma((TMEM_PIPE_DEPTH, BLK_M, HEAD_DIM), "float16")
    sScale = pool.alloc((SSCALE_TOTAL_SIZE,), "float32", align=1024)
    tmem_addr = pool.alloc([1], "uint32")
    ACC_SCALE_BASE: T.let = 0
    ROW_SUM_BASE: T.let = 0
    kv_pipe = PipelineState(SMEM_PIPE_DEPTH_KV)
    phase_q: T.int32
    phase_s_full: T.int32
    phase_tmem: T.int32
    phase_s0_s1: T.int32
    phase_q_load: T.int32
    phase_oepi: T.int32
    q_load = Pipeline(pool, SMEM_PIPE_DEPTH_Q, full="tma", empty="tcgen05", empty_phase_offset=1)
    kv_load = Pipeline(pool, SMEM_PIPE_DEPTH_KV, full="tma", empty="tcgen05", empty_phase_offset=1)
    p_o_rescale = MBarrier(pool, 2)
    p_o_rescale.init(256)
    s_ready = TCGen05Bar(pool, 2)
    s_ready.init(1)
    o_ready = TCGen05Bar(pool, 2)
    o_ready.init(1)
    softmax_corr = Pipeline(
        pool, 2, full="mbar", empty="mbar", init_full=128, init_empty=128, empty_phase_offset=1
    )
    corr_epi = Pipeline(
        pool,
        TMEM_PIPE_DEPTH,
        full="mbar",
        empty="mbar",
        init_full=128,
        init_empty=32,
        empty_phase_offset=1,
    )
    p_ready_2 = MBarrier(pool, 2)
    p_ready_2.init(128)
    bar_s0_s1_sequence = MBarrier(pool, 8)
    # Init in the FIRST init group so the single prologue fence (below) covers
    # it; lets us drop the redundant 2nd fence+cta_sync (MEMBAR/FENCE/BSSY
    # excess vs cutedsl, which uses one prologue init barrier).
    bar_s0_s1_sequence.init(32)
    pool.commit()
    profiler = CudaProfiler(
        profiler_buffer,
        write_stride=PROFILER_WRITE_STRIDE,
        num_groups=NUM_GROUPS,
        profiler_enabled=PROFILER_ON,
    )
    # TMEM is allocated by the MMA warp (cta-scope warp 12 = wg3/warp0) and
    # the alloc deliberately sits AFTER the prologue cta_syncs (commit is
    # emitted further down): every other warp's first TMEM access is
    # transitively gated behind this warp's progress (softmax via s_ready
    # from the first QK gemm, correction via the softmax handshake, TMA
    # store via corr_epi), so no CTA-wide sync on the alloc is needed, and
    # the TMA load warp issues the first Q/K copies without waiting for it.
    # FA4 CuTeDSL structures its prologue the same way (alloc inside the
    # MMA warp's branch; load warps never wait for it).
    tmem_pool = T.TMEMPool(
        pool,
        total_cols=N_COLS_TMEM,
        cta_group=CTA_GROUP,
        tmem_addr=tmem_addr,
        alloc_warp=12,
        dealloc_warp=0,
    )
    tmem = tmem_pool.alloc((128, N_COLS_TMEM), "float32")
    tmem_pool.move_base_to(0)
    tmem_as_f16 = tmem_pool.alloc((128, N_COLS_TMEM * 2), "float16")
    T.ptx.fence.proxy_async("shared::cta")
    T.ptx.fence.mbarrier_init()
    T.cuda.cta_sync()
    S_region = T.meta_var(
        T.TMEMStages(tmem, col_start=0, width=MMA_N, stages=SMEM_PIPE_DEPTH_Q, stride=MMA_N)
    )
    O_region = T.meta_var(
        T.TMEMStages(
            tmem,
            col_start=MMA_N * SMEM_PIPE_DEPTH_Q,
            width=MMA_N,
            stages=SMEM_PIPE_DEPTH_Q,
            stride=MMA_N,
        )
    )
    P_region = T.meta_var(
        T.TMEMStages(
            tmem_as_f16, col_start=MMA_N, width=BLK_N, stages=SMEM_PIPE_DEPTH_Q, stride=MMA_N * 2
        )
    )
    scheduler = (
        FlashAttentionLPTScheduler(
            "fa_scheduler",
            num_batches=BATCH_SIZE,
            num_heads=NUM_KV_HEADS,
            num_m_blocks=num_q_blocks,
            l2_swizzle=L2_SWIZZLE,
        )
        if is_causal
        else FlashAttentionLinearScheduler(
            "fa_scheduler",
            num_batches=BATCH_SIZE,
            num_heads=NUM_KV_HEADS,
            num_m_blocks=num_q_blocks,
            num_ctas=cta_count,
        )
    )
    scheduler.init(bx)
    if (wg_id == 3) & (warp_id == 1):
        profiler.init(0)
    elif (wg_id == 3) & (warp_id == 2):
        profiler.init(1)
    elif (wg_id == 3) & (warp_id == 0):
        profiler.init(2)
    elif wg_id <= 1:
        profiler.init(3 + wg_id)
    elif wg_id == 2:
        profiler.init(5)
    kv_pipe.init(0)
    phase_q = 0
    phase_oepi = 0
    phase_tmem = 0
    phase_s_full = 0
    if USE_S0_S1_BARRIER:
        phase_s0_s1 = T.if_then_else(wg_id == 1, 0, 1)
    phase_q_load = 0
    tmem_pool.commit()
    if (wg_id == 3) & (warp_id == 0):
        T.cuda.trap_when_assert_failed(tmem_addr[0] == T.uint32(0))
    if wg_id == 2:
        for i_q in T.unroll(2):
            p_o_rescale.arrive(i_q)
    num_kv_blocks: T.let = ceildiv(SEQ_LEN_KV, BLK_N)
    while scheduler.valid():
        m_block_idx = T.meta_var(scheduler.m_block_idx)
        batch_idx = T.meta_var(scheduler.batch_idx)
        kv_head_idx = T.meta_var(scheduler.head_idx)
        m_start = T.meta_var(m_block_idx * SEQ_Q_PER_TILE * SMEM_PIPE_DEPTH_Q)
        if wg_id == 3:
            T.ptx.setmaxnreg(False, 48)
            if warp_id == 1:

                @T.inline
                def load_q(i_q):
                    q_load.empty.wait(i_q, phase_q_load)
                    tma_copy_q = T.meta_var(
                        {
                            "dispatch": "tma",
                            "mbar": q_load.full.buf.ptr_to([i_q]),
                            "cta_group": CTA_GROUP,
                        }
                    )
                    profiler.start(ProfileEventType.IssueTMA_Q, lane_id == 0)
                    Q_smem_3d = Q_smem.view(SMEM_PIPE_DEPTH_Q, SEQ_Q_PER_TILE, GQA_RATIO, HEAD_DIM)
                    if T.ptx.elect_sync():
                        Tx.copy_async(
                            Q_smem_3d[i_q, :, :, :],
                            Q[
                                batch_idx,
                                m_start + i_q * SEQ_Q_PER_TILE : m_start
                                + (i_q + 1) * SEQ_Q_PER_TILE,
                                kv_head_idx * GQA_RATIO : (kv_head_idx + 1) * GQA_RATIO,
                                :,
                            ],
                            **tma_copy_q,
                        )
                        q_load.full.arrive(i_q, CTA_GROUP * BLK_M * HEAD_DIM * F16_BYTES)
                    profiler.end(ProfileEventType.IssueTMA_Q, lane_id == 0)

                @T.inline
                def load_k(i_kv):
                    kv_load.empty.wait(kv_pipe.stage, kv_pipe.phase)
                    tma_copy_k = T.meta_var(
                        {
                            "dispatch": "tma",
                            "mbar": kv_load.full.buf.ptr_to([kv_pipe.stage]),
                            "cta_group": CTA_GROUP,
                        }
                    )
                    profiler.start(ProfileEventType.IssueTMA_K, lane_id == 0)
                    if T.ptx.elect_sync():
                        Tx.copy_async(
                            K_smem[kv_pipe.stage, :, :],
                            K[batch_idx, i_kv * BLK_N : (i_kv + 1) * BLK_N, kv_head_idx, :],
                            **tma_copy_k,
                        )
                        kv_load.full.arrive(kv_pipe.stage, CTA_GROUP * BLK_N * HEAD_DIM * F16_BYTES)
                    profiler.end(ProfileEventType.IssueTMA_K, lane_id == 0)
                    kv_pipe.advance()

                @T.inline
                def load_v(i_kv):
                    kv_load.empty.wait(kv_pipe.stage, kv_pipe.phase)
                    tma_copy_v = T.meta_var(
                        {
                            "dispatch": "tma",
                            "mbar": kv_load.full.buf.ptr_to([kv_pipe.stage]),
                            "cta_group": CTA_GROUP,
                        }
                    )
                    profiler.start(ProfileEventType.IssueTMA_V, lane_id == 0)
                    if T.ptx.elect_sync():
                        Tx.copy_async(
                            V_smem[kv_pipe.stage, :, :],
                            V[batch_idx, i_kv * BLK_N : (i_kv + 1) * BLK_N, kv_head_idx, :],
                            **tma_copy_v,
                        )
                        kv_load.full.arrive(kv_pipe.stage, CTA_GROUP * BLK_N * HEAD_DIM * F16_BYTES)
                    profiler.end(ProfileEventType.IssueTMA_V, lane_id == 0)
                    kv_pipe.advance()

                load_trip_count: T.int32
                load_trip_count = (
                    get_n_block_max(m_block_idx, is_causal, SEQ_LEN_KV, SEQ_LEN_Q, SEQ_Q_PER_TILE)
                    if is_causal
                    else num_kv_blocks
                )
                load_q(0)
                load_k(load_trip_count - 1)
                load_q(1)
                phase_q_load ^= 1
                load_v(load_trip_count - 1)
                for _i in T.serial(load_trip_count - 1, unroll=False):
                    i_kv: T.let = load_trip_count - 2 - _i
                    load_k(i_kv)
                    load_v(i_kv)
            if warp_id == 2:
                for i_q in T.unroll(SMEM_PIPE_DEPTH_Q):
                    corr_epi.full.wait(i_q, phase_tmem)
                    if i_q == 0:
                        profiler.start(ProfileEventType.TMAStore, lane_id == 0)
                    m_start_global = T.meta_var(m_start + i_q * SEQ_Q_PER_TILE)
                    O_smem_3d = O_smem.view(TMEM_PIPE_DEPTH, SEQ_Q_PER_TILE, GQA_RATIO, HEAD_DIM)
                    if T.ptx.elect_sync():
                        Tx.copy_async(
                            O[
                                batch_idx,
                                m_start_global : m_start_global + SEQ_Q_PER_TILE,
                                kv_head_idx * GQA_RATIO : (kv_head_idx + 1) * GQA_RATIO,
                                :,
                            ],
                            O_smem_3d[i_q, :, :, :],
                            dispatch="tma",
                        )
                    T.ptx.cp_async.bulk.commit_group()
                for i_q in T.unroll(SMEM_PIPE_DEPTH_Q):
                    T.ptx.cp_async.bulk.wait_group(1 - i_q)
                    corr_epi.empty.arrive(i_q)
                profiler.end(ProfileEventType.TMAStore, lane_id == 0)
                phase_tmem ^= 1
            if warp_id == 0:
                acc: T.int32
                acc = 0

                @T.inline
                def gemm_qk(q_stage, kv_stage):
                    Tx.warp.gemm_async(
                        S_region[q_stage],
                        Q_smem[q_stage, 0:BLK_M, 0:HEAD_DIM],
                        K_smem[kv_stage, 0:BLK_N, 0:HEAD_DIM],
                        dispatch="tcgen05",
                        cta_group=CTA_GROUP,
                    )
                    if T.ptx.elect_sync():
                        s_ready.arrive(q_stage)

                # PV gemm split point, regime-tuned: for causal, 64 cols
                # (p_o_rescale waits TWO P quarter-stores instead of three,
                # releasing the MMA warp earlier in the
                # softmax->P->PV->QK->s_ready loop that binds the small-shape
                # cadence; single-task phase tables vs cutedsl showed TIRx
                # softmax idling ~26% of its cadence on s_ready vs baseline
                # 14%). For non-causal the earlier release measured 2-3pp
                # WORSE on s1024 shapes — keep 96.
                K_SPLIT = T.meta_var((4 if is_causal else 6) * MMA_K)

                @T.inline
                def gemm_pv_part1(i_q, kv_stage, should_accumulate):
                    Tx.warp.gemm_async(
                        O_region[i_q],
                        P_region[i_q, 0:K_SPLIT],
                        V_smem[kv_stage, 0:K_SPLIT, 0:HEAD_DIM],
                        transB=True,
                        accum=should_accumulate,
                        dispatch="tcgen05",
                        cta_group=CTA_GROUP,
                    )

                @T.inline
                def gemm_pv_part2(i_q, kv_stage):
                    p_ready_2.wait(i_q, phase_tmem)
                    Tx.warp.gemm_async(
                        O_region[i_q],
                        P_region[i_q, K_SPLIT:BLK_N],
                        V_smem[kv_stage, K_SPLIT:BLK_N, 0:HEAD_DIM],
                        transB=True,
                        accum=True,
                        dispatch="tcgen05",
                        cta_group=CTA_GROUP,
                    )

                @T.inline
                def gemm_pv(i_q, kv_stage, should_accumulate):
                    gemm_pv_part1(i_q, kv_stage, should_accumulate)
                    gemm_pv_part2(i_q, kv_stage)

                for i_q in T.unroll(SMEM_PIPE_DEPTH_Q):
                    q_load.full.wait(i_q, phase_q_load)
                    if i_q == 0:
                        kv_load.full.wait(kv_pipe.stage, kv_pipe.phase)
                    gemm_qk(i_q, kv_pipe.stage)
                    if i_q == 1:
                        if T.ptx.elect_sync():
                            kv_load.empty.arrive(kv_pipe.stage)
                kv_pipe.advance()
                mma_trip_count: T.int32
                mma_trip_count = (
                    get_n_block_max(m_block_idx, is_causal, SEQ_LEN_KV, SEQ_LEN_Q, SEQ_Q_PER_TILE)
                    if is_causal
                    else num_kv_blocks
                )
                for i_kv in T.serial(mma_trip_count - 1, unroll=False):
                    stage_v: T.let = kv_pipe.stage
                    phase_v: T.let = kv_pipe.phase
                    kv_pipe.advance()
                    stage_k = T.meta_var(kv_pipe.stage)
                    phase_k = T.meta_var(kv_pipe.phase)
                    for i_q in T.unroll(SMEM_PIPE_DEPTH_Q):
                        if i_q == 0:
                            kv_load.full.wait(stage_v, phase_v)
                        p_o_rescale.wait(i_q, phase_tmem)
                        gemm_pv(i_q, stage_v, acc)
                        if i_q == 1:
                            if T.ptx.elect_sync():
                                kv_load.empty.arrive(stage_v)
                        if i_q == 0:
                            kv_load.full.wait(stage_k, phase_k)
                        gemm_qk(i_q, stage_k)
                        # Early Q release (non-causal / multi-task CTAs):
                        # Q[i_q]'s LAST reader is this final QK — committing
                        # here fires when it completes, ~2 tail-PV gemms
                        # before the end-of-task commit, so the next task's
                        # Q TMA load overlaps the PV tail + epilogue.
                        # Causal keeps the tail commit: one task per CTA
                        # (nothing to overlap) and packed-causal tasks can
                        # have mma_trip_count == 1, where this steady-loop
                        # site never executes.
                        if EARLY_Q_RELEASE:
                            if i_kv == mma_trip_count - 2:
                                if T.ptx.elect_sync():
                                    q_load.empty.arrive(i_q)
                        if i_q == 1:
                            if T.ptx.elect_sync():
                                kv_load.empty.arrive(stage_k)
                    acc = 1
                    kv_pipe.advance()
                    phase_tmem ^= 1
                for i_q in T.unroll(SMEM_PIPE_DEPTH_Q):
                    if i_q == 0:
                        kv_load.full.wait(kv_pipe.stage, kv_pipe.phase)
                    p_o_rescale.wait(i_q, phase_tmem)
                    gemm_pv(i_q, kv_pipe.stage, acc)
                    if i_q == 1:
                        if T.ptx.elect_sync():
                            kv_load.empty.arrive(kv_pipe.stage)
                    if T.ptx.elect_sync():
                        o_ready.arrive(i_q)
                kv_pipe.advance()
                phase_tmem ^= 1
                if not EARLY_Q_RELEASE:
                    for i_q in T.unroll(SMEM_PIPE_DEPTH_Q):
                        if T.ptx.elect_sync():
                            q_load.empty.arrive(i_q)
                phase_q_load ^= 1
        elif wg_id < 2:
            T.ptx.setmaxnreg(True, 200)
            scale_log2 = T.meta_var(math.log2(math.e) / math.sqrt(HEAD_DIM))
            rescale_threshold = T.meta_var(8.0)
            row_max: T.f32[1]
            row_sum: T.f32[1]
            ts_step: T.int32
            ts_step = 0

            @T.inline
            def light_ts(j):
                if LIGHT_TS:
                    if tid_in_wg == 0:
                        profiler_buffer[1 + wg_id * 4096 + ts_step * 8 + j] = T.cast(
                            T.ptx.fetch_register(64, "clock64"), "uint64"
                        )

            if LIGHT_TS:
                if tid_in_wg == 0:
                    profiler_buffer[1 + wg_id * 4096 + 4000] = T.cast(
                        T.ptx.fetch_register(64, "clock64"), "uint64"
                    )

            @T.inline
            def mask_r2p(s_chunk_buf, col_limit, ncol: T.int32):
                """Apply mask using R2P-style bit manipulation.

                Optimizes: for j in range(N): buf[j] = -inf if j >= col_limit else buf[j]
                Into: bitmask operations that compile to R2P PTX instruction.

                Following flash_attn/cute/mask.py mask_r2p(): process in 32-col
                chunks (shl_u32_clamp tolerates shift>=32, so no 24-col split).

                The bit test `mask & (1 << i)` compiles to the R2P (Register to Predicate)
                PTX instruction, which is more efficient than per-column comparisons.
                """
                ncol = T.meta_var(ncol)
                # Subtract-free low-k bitmask: ~(0xFFFFFFFF<<k) gives the low-k-bits
                # mask with NO `-1`; the ~ fuses into the per-column `& (1<<i)` test
                # as ANDN (one LOP3), so there is no VIADD per chunk. shl clamps
                # k>=32 -> 0 => mask_inv=0 => ~=all-ones => keep all (k>=32 correct),
                # so no VIMNMX either. The bit test compiles to R2P.
                CHUNK_SIZE: T.let = 32
                num_chunks: T.let = ceildiv(ncol, CHUNK_SIZE)
                s_chunk_local = s_chunk_buf.local(ncol)
                for s in T.unroll(num_chunks):
                    k_keep: T.let = T.max(col_limit - s * CHUNK_SIZE, 0)
                    mask_inv: T.uint32
                    mask_inv = shl_u32_clamp(T.uint32(0xFFFFFFFF), T.cast(k_keep, "uint32"))
                    for i in T.unroll(CHUNK_SIZE):
                        if i < ncol - s * CHUNK_SIZE:
                            c: T.let = s * CHUNK_SIZE + i
                            in_bound: T.let = T.bitwise_and(
                                T.bitwise_not(mask_inv), T.shift_left(T.uint32(1), i)
                            )
                            s_chunk_local[c] = T.Select(
                                T.cast(in_bound, "bool"), s_chunk_local[c], T.float32(-float("inf"))
                            )

            @T.inline
            def apply_causal_mask(s_chunk_buf, m_blk_idx, n_blk_idx):
                """Apply causal mask to attention scores.

                Following flash_attn/cute/mask.py apply_mask_sm100() lines 384-400:
                causal_row_offset = 1 + seqlen_k - n_block * tile_n - seqlen_q
                row_idx = thread_row + m_block * tile_m
                col_limit_right = row_idx + causal_row_offset
                Mask if col >= col_limit_right

                Coordinate Mapping:
                - BLK_M = 128 packed rows per tile
                - SEQ_Q_PER_TILE = BLK_M // GQA_RATIO (e.g., 32 for GQA_RATIO=4)
                - Each warpgroup handles one Q stage with SEQ_Q_PER_TILE sequence positions
                - tid_in_wg (0-127) maps to packed rows: (seq_pos, head) = (tid//GQA_RATIO, tid%GQA_RATIO)
                """
                seq_pos_in_wg: T.let = tid_in_wg // GQA_RATIO
                row_idx: T.let = (
                    m_blk_idx * SEQ_Q_PER_TILE * SMEM_PIPE_DEPTH_Q
                    + wg_id * SEQ_Q_PER_TILE
                    + seq_pos_in_wg
                )
                causal_row_offset: T.let = 1 + SEQ_LEN_KV - n_blk_idx * BLK_N - SEQ_LEN_Q
                col_limit_right: T.let = row_idx + causal_row_offset
                mask_r2p(s_chunk_buf, col_limit_right, BLK_N)

            @T.inline
            def softmax_step(i_kv, apply_mask=False, is_first=False):
                s_chunk_buf: T.f32[BLK_N]
                s_chunk = s_chunk_buf.view(128, BLK_N, layout=wg_local_layout(BLK_N))
                p_chunk_buf_f32: T.f32[BLK_N // 2]
                p_chunk_buf = T.decl_buffer((BLK_N,), dtype="float16", data=p_chunk_buf_f32.data)
                p_chunk = p_chunk_buf.view(128, BLK_N, layout=wg_local_layout(BLK_N))
                s_ready.wait(wg_id, phase_s_full)
                light_ts(0)
                profiler.start(ProfileEventType.Softmax_MAX, tid_in_wg == 0)
                tile_max: T.f32[1]
                for chunk_idx in T.unroll(BLK_N // SOFTMAX_LD_CHUNK):
                    Tx.wg.copy_async(
                        s_chunk[
                            :, chunk_idx * SOFTMAX_LD_CHUNK : (chunk_idx + 1) * SOFTMAX_LD_CHUNK
                        ],
                        S_region[
                            wg_id, chunk_idx * SOFTMAX_LD_CHUNK : (chunk_idx + 1) * SOFTMAX_LD_CHUNK
                        ],
                    )
                if apply_mask:
                    apply_causal_mask(s_chunk_buf, m_block_idx, i_kv)
                row_max_old: T.f32
                row_max_old = row_max[0]
                if is_first:
                    Tx.max(tile_max, s_chunk_buf)
                else:
                    tile_max[0] = row_max_old
                    Tx.max(tile_max, s_chunk_buf, accum=True)
                row_max_new: T.f32
                acc_scale: T.f32
                acc_scale_: T.f32
                row_max_safe: T.f32
                row_max_new = tile_max[0]
                row_max_safe = T.if_then_else(tile_max[0] == -float("inf"), 0.0, tile_max[0])
                if is_first:
                    acc_scale = T.float32(1.0)
                else:
                    acc_scale_ = (row_max_old - row_max_safe) * scale_log2
                    if acc_scale_ >= -rescale_threshold:
                        row_max_new = row_max_old
                        row_max_safe = row_max_old
                        acc_scale = T.float32(1.0)
                    else:
                        acc_scale = T.ptx.exp2(acc_scale_)
                row_max[0] = row_max_new
                row_max_scaled: T.let = row_max_safe * scale_log2
                light_ts(1)
                profiler.end(ProfileEventType.Softmax_MAX, tid_in_wg == 0)
                if tid_in_wg < BLK_M and (not is_first):
                    sScale_idx: T.let = ACC_SCALE_BASE + tid_in_wg + wg_id * BLK_M
                    sScale[sScale_idx] = acc_scale
                # Stats-ready handshake to the correction wg via HW named
                # barrier (FA4 CuTeDSL sm_stats_barrier): softmax warp w of
                # stage wg_id pairs with correction warp w on barrier
                # 1 + wg_id*4 + w (64 threads). bar.arrive is non-blocking and
                # the correction side's bar.sync wakes without the mbarrier
                # try_wait + nanosleep latency — this wait gates p_o_rescale
                # and therefore the PV MMA. The reverse (sScale slot reuse)
                # direction stays on softmax_corr.empty.
                if STATS_BAR_PAIRWISE:
                    tvm.tirx.cuda.op.ptx_bar_arrive(1 + wg_id * 4 + warp_id, 64)
                else:
                    tvm.tirx.cuda.op.ptx_bar_arrive(1 + wg_id, 256)
                profiler.start(ProfileEventType.Softmax_FMA, tid_in_wg == 0)
                Tx.wg.fma(s_chunk, s_chunk, scale_log2, -row_max_scaled)
                profiler.end(ProfileEventType.Softmax_FMA, tid_in_wg == 0)
                if USE_S0_S1_BARRIER:
                    bar_s0_s1_sequence.wait(wg_id * 4 + warp_id, phase_s0_s1)
                profiler.start(ProfileEventType.Softmax_EXP2, tid_in_wg == 0)
                for frag_idx in T.unroll(4):
                    s_chunk_local = s_chunk_buf.local(BLK_N)
                    for i in T.unroll(BLK_N // 4 // 2):
                        idx = T.meta_var(frag_idx * BLK_N // 4 + 2 * i)
                        # ex2 emulation ratio is shape-tuned (cf. FA4 CuTeDSL
                        # _TUNING_CONFIG keyed on is_causal); see module-level
                        # EMU_* knobs.
                        emu_pairs = T.meta_var(EMU_PAIRS_CAUSAL if is_causal else EMU_PAIRS_NC)
                        emu_start = T.meta_var(EMU_START_CAUSAL if is_causal else EMU_START_NC)
                        if (
                            i * 2 % 16 < 16 - 2 * emu_pairs
                            or frag_idx >= 4 - 1
                            or frag_idx < emu_start
                            or apply_mask
                        ):
                            s_chunk_local[idx] = T.ptx.exp2(s_chunk_local[idx])
                            s_chunk_local[idx + 1] = T.ptx.exp2(s_chunk_local[idx + 1])
                        else:
                            ex2_emulation_2(
                                s_chunk_local, idx, s_chunk_local[idx], s_chunk_local[idx + 1]
                            )
                    Tx.wg.cast(
                        p_chunk[:, frag_idx * BLK_N // 4 : (frag_idx + 1) * BLK_N // 4],
                        s_chunk[:, frag_idx * BLK_N // 4 : (frag_idx + 1) * BLK_N // 4],
                    )
                if USE_S0_S1_BARRIER:
                    bar_s0_s1_sequence.arrive((1 - wg_id) * 4 + warp_id)
                profiler.end(ProfileEventType.Softmax_EXP2, tid_in_wg == 0)
                profiler.start(ProfileEventType.Softmax_TMEM_ST, tid_in_wg == 0)
                P_SPLIT_Q = T.meta_var(2 if is_causal else 3)
                for i in T.unroll(P_SPLIT_Q):
                    Tx.wg.copy_async(
                        P_region[wg_id, i * BLK_N // 4 : (i + 1) * BLK_N // 4],
                        p_chunk[:, i * BLK_N // 4 : (i + 1) * BLK_N // 4],
                    )
                T.ptx.tcgen05.wait.st()
                p_o_rescale.arrive(wg_id)
                for i in T.unroll(4 - P_SPLIT_Q):
                    Tx.wg.copy_async(
                        P_region[
                            wg_id, (P_SPLIT_Q + i) * BLK_N // 4 : (P_SPLIT_Q + 1 + i) * BLK_N // 4
                        ],
                        p_chunk[:, (P_SPLIT_Q + i) * BLK_N // 4 : (P_SPLIT_Q + 1 + i) * BLK_N // 4],
                    )
                light_ts(2)
                T.ptx.tcgen05.wait.st()
                p_ready_2.arrive(wg_id)
                light_ts(3)
                profiler.end(ProfileEventType.Softmax_TMEM_ST, tid_in_wg == 0)
                softmax_corr.empty.wait(wg_id, phase_q)
                light_ts(4)
                profiler.start(ProfileEventType.Softmax_SUM, tid_in_wg == 0)
                phase_s_full ^= 1
                phase_q ^= 1
                if is_first:
                    Tx.sum(row_sum, s_chunk_buf)
                else:
                    row_sum[0] = row_sum[0] * acc_scale
                    Tx.sum(row_sum, s_chunk_buf, accum=True)
                light_ts(5)
                ts_step = ts_step + 1
                profiler.end(ProfileEventType.Softmax_SUM, tid_in_wg == 0)
                if USE_S0_S1_BARRIER:
                    phase_s0_s1 ^= 1

            # Pre-loop empty.wait guards this task's first sScale write
            # against the PREVIOUS task's tail row_sum read. Under
            # EPI_ON_SOFTMAX there is no tail row_sum slot use and no
            # previous task (one task per CTA), and correction's tail
            # empty credits are gone too — the wait must go with them
            # (credit/wait audit: n arrives vs n in-step waits per slot).
            if not EPI_ON_SOFTMAX:
                softmax_corr.empty.wait(wg_id, phase_q)
            # Keep the phase flip even when the wait is gone: parities are
            # absolute, and correction's prologue empty.arrive can land
            # BEFORE this wg reaches its step-1 wait — starting the in-step
            # waits at phase 1 makes step-1 consume that prologue credit
            # exactly like the original protocol (synccheck-verified; the
            # phase-0 "free pass" is NOT sticky and deadlocks if the flip
            # outruns the waiter).
            phase_q ^= 1
            n_block_max: T.let = get_n_block_max(
                m_block_idx, is_causal, SEQ_LEN_KV, SEQ_LEN_Q, SEQ_Q_PER_TILE
            )
            n_block_min_causal: T.let = (
                get_n_block_min_causal_mask(m_block_idx, SEQ_LEN_KV, SEQ_LEN_Q, SEQ_Q_PER_TILE)
                if is_causal
                else n_block_max
            )
            softmax_step(n_block_max - 1, apply_mask=is_causal, is_first=True)
            n_block_max_after_p1: T.let = n_block_max - 1
            num_phase2_blocks: T.let = T.max(n_block_max_after_p1 - n_block_min_causal, 0)
            for i in T.serial(num_phase2_blocks, unroll=False):
                n_block: T.let = n_block_max_after_p1 - 1 - i
                softmax_step(n_block, apply_mask=True)
            n_block_max_after_p2: T.let = T.min(n_block_max_after_p1, n_block_min_causal)
            for i in T.serial(n_block_max_after_p2, unroll=False):
                n_block: T.let = n_block_max_after_p2 - 1 - i
                softmax_step(n_block, apply_mask=False)
            if EPI_ON_SOFTMAX:
                # Stage-parallel epilogue on this wg: row_sum is already in
                # a register (no sScale round-trip / tail stats trip), and
                # the post-loop softmax register file is idle — 32-wide
                # TMEM loads are free (4 round trips, vs 8x16 under the
                # correction wg's 64-reg budget).
                EPI_LD_SM = T.meta_var(32)
                o_ready.wait(wg_id, phase_oepi)
                corr_epi.empty.wait(wg_id, phase_oepi)
                profiler.start(ProfileEventType.EpiLDTMEM, tid_in_wg == 0)
                acc_O_row_is_zero_or_nan: T.let = tvm.tirx.any(
                    row_sum[0] == T.float32(0.0), row_sum[0] != row_sum[0]
                )
                norm_scale_sm: T.let = T.ptx.rcp(
                    T.Select(acc_O_row_is_zero_or_nan, T.float32(1.0), row_sum[0])
                )
                o_row_f32_sm = T.wg_reg_tile(EPI_LD_SM)
                o_row_f16_sm = T.wg_reg_tile(EPI_LD_SM, "float16")
                for epi_q in T.unroll(2):
                    if wg_id == epi_q:
                        for d_tile in T.unroll(ceildiv(HEAD_DIM, EPI_LD_SM)):
                            Tx.wg.copy_async(
                                o_row_f32_sm,
                                O_region[epi_q, d_tile * EPI_LD_SM : (d_tile + 1) * EPI_LD_SM],
                            )
                            Tx.wg.mul(o_row_f32_sm, o_row_f32_sm, norm_scale_sm)
                            Tx.wg.cast(o_row_f16_sm, o_row_f32_sm)
                            Tx.wg.copy(
                                O_smem[
                                    epi_q, 0:BLK_M, d_tile * EPI_LD_SM : (d_tile + 1) * EPI_LD_SM
                                ],
                                o_row_f16_sm,
                                vec_len=8,
                            )
                profiler.end(ProfileEventType.EpiLDTMEM, tid_in_wg == 0)
                T.ptx.fence.proxy_async("shared::cta")
                corr_epi.full.arrive(wg_id)
                p_o_rescale.arrive(wg_id)
                phase_oepi ^= 1
            else:
                if tid_in_wg < BLK_M:
                    sScale[ROW_SUM_BASE + tid_in_wg + wg_id * BLK_M] = row_sum[0]
                if STATS_BAR_PAIRWISE:
                    tvm.tirx.cuda.op.ptx_bar_arrive(1 + wg_id * 4 + warp_id, 64)
                else:
                    tvm.tirx.cuda.op.ptx_bar_arrive(1 + wg_id, 256)
        if wg_id == 2:
            T.ptx.setmaxnreg(False, 64)
            if STATS_BAR_PAIRWISE:
                tvm.tirx.cuda.op.ptx_bar_sync(1 + 0 * 4 + warp_id, 64)
            else:
                tvm.tirx.cuda.op.ptx_bar_sync(1 + 0, 256)
            softmax_corr.empty.arrive(0)
            if STATS_BAR_PAIRWISE:
                tvm.tirx.cuda.op.ptx_bar_sync(1 + 1 * 4 + warp_id, 64)
            else:
                tvm.tirx.cuda.op.ptx_bar_sync(1 + 1, 256)
            phase_q ^= 1
            corr_trip_count: T.let = (
                get_n_block_max(m_block_idx, is_causal, SEQ_LEN_KV, SEQ_LEN_Q, SEQ_Q_PER_TILE)
                if is_causal
                else num_kv_blocks
            )
            for i_kv in T.serial(corr_trip_count - 1, unroll=False):
                for i_q in T.unroll(2):
                    if STATS_BAR_PAIRWISE:
                        tvm.tirx.cuda.op.ptx_bar_sync(1 + i_q * 4 + warp_id, 64)
                    else:
                        tvm.tirx.cuda.op.ptx_bar_sync(1 + i_q, 256)
                    profiler.start(ProfileEventType.Correction, tid_in_wg == 0)
                    acc_scale: T.f32
                    should_rescale: T.i32
                    if tid_in_wg < BLK_M:
                        acc_scale = sScale[ACC_SCALE_BASE + tid_in_wg + i_q * BLK_M]
                        should_rescale = T.Select(acc_scale < T.float32(1.0), 1, 0)
                    else:
                        should_rescale = 0
                    any_needs_rescale: T.let = T.ptx.any_sync(4294967295, should_rescale)
                    if any_needs_rescale != 0:
                        if tid_in_wg < BLK_M:
                            RESCALE_TILE = T.meta_var(16)
                            o_row = T.wg_reg_tile(RESCALE_TILE)
                            for d_tile in T.unroll(ceildiv(HEAD_DIM, RESCALE_TILE)):
                                d_start: T.let = d_tile * RESCALE_TILE
                                if d_start < HEAD_DIM:
                                    Tx.wg.copy_async(
                                        o_row, O_region[i_q, d_start : d_start + RESCALE_TILE]
                                    )
                                    Tx.wg.mul(o_row, o_row, acc_scale)
                                    Tx.wg.copy_async(
                                        O_region[i_q, d_start : d_start + RESCALE_TILE], o_row
                                    )
                            T.ptx.tcgen05.wait.st()
                    p_o_rescale.arrive(i_q)
                    softmax_corr.empty.arrive(1 - i_q)
                    profiler.end(ProfileEventType.Correction, tid_in_wg == 0)
                phase_q ^= 1
            softmax_corr.empty.arrive(1)
            if not EPI_ON_SOFTMAX:
                for i_q in T.unroll(2):
                    if STATS_BAR_PAIRWISE:
                        tvm.tirx.cuda.op.ptx_bar_sync(1 + i_q * 4 + warp_id, 64)
                    else:
                        tvm.tirx.cuda.op.ptx_bar_sync(1 + i_q, 256)
                    row_sum: T.let = sScale[ROW_SUM_BASE + tid_in_wg + i_q * BLK_M]
                    softmax_corr.empty.arrive(i_q)
                    o_ready.wait(i_q, phase_tmem)
                    corr_epi.empty.wait(i_q, phase_tmem)
                    profiler.start(ProfileEventType.EpiLDTMEM, tid_in_wg == 0)
                    acc_O_mn_row_is_zero_or_nan: T.let = tvm.tirx.any(
                        row_sum == T.float32(0.0), row_sum != row_sum
                    )
                    norm_scale: T.let = T.ptx.rcp(
                        T.Select(acc_O_mn_row_is_zero_or_nan, T.float32(1.0), row_sum)
                    )
                    o_row_f32 = T.wg_reg_tile(TMEM_EPI_LD_SIZE)
                    o_row_f16 = T.wg_reg_tile(TMEM_EPI_LD_SIZE, "float16")
                    for d_tile in T.unroll(ceildiv(HEAD_DIM, TMEM_EPI_LD_SIZE)):
                        d_start: T.let = d_tile * TMEM_EPI_LD_SIZE
                        if d_start < HEAD_DIM:
                            Tx.wg.copy_async(
                                o_row_f32, O_region[i_q, d_start : d_start + TMEM_EPI_LD_SIZE]
                            )
                            Tx.wg.mul(o_row_f32, o_row_f32, norm_scale)
                            Tx.wg.cast(o_row_f16, o_row_f32)
                            Tx.wg.copy(
                                O_smem[
                                    i_q,
                                    0:BLK_M,
                                    d_tile * TMEM_EPI_LD_SIZE : d_tile * TMEM_EPI_LD_SIZE
                                    + TMEM_EPI_LD_SIZE,
                                ],
                                o_row_f16,
                                vec_len=8,
                            )
                        profiler.end(ProfileEventType.EpiLDTMEM, tid_in_wg == 0)
                    T.ptx.fence.proxy_async("shared::cta")
                    corr_epi.full.arrive(i_q)
                    p_o_rescale.arrive(i_q)
                phase_tmem ^= 1
            phase_q ^= 1
        scheduler.next_tile()
    tmem_pool.dealloc()
    # No final cta_sync: warps exit independently after the dealloc. On the
    # single-task causal shapes the kernel-end barrier is fully exposed (~11%
    # of stall samples on the tail); dropping it is safe (50x reused-module
    # GPU verify PASS — tcgen05 dealloc/relinquish by warp 0 needs no CTA-wide
    # rendezvous, and there is no post-exit SMEM/TMEM reuse) and lifts the warm
    # ratio ~+0.5pp. Validated first via tvm_callback_cuda_postproc injection,
    # then moved here.


def get_flash_attention4_kernel(
    batch_size, seq_len_q, seq_len_kv, num_qo_heads, num_kv_heads, head_dim, is_causal=False
):
    # ptxas --register-usage-level: the default 10 over-allocates under the
    # setmaxnreg caps and SPILLS (35K LDL/STL; cutedsl has 0). On the latency-
    # bound causal GQA=1 (kv32) shapes those spills sit on the critical path.
    # Clean isolated run_bench A/B (warm-L2) picks the best NON-spilling level
    # per regime (swept 3..8; >=6 re-spills): multi-wave s2048/s4096_h32kv32_c
    # want 5 (0.967->0.992, 0.974->0.993); the single-wave s1024_h32kv32_c
    # wants 4 (0.954->0.962, won all 3 rounds, s2048/s4096 unaffected ~tie).
    # GQA=2 (s1024kv16_c) and throughput-bound non-causal keep 10's ILP. Read
    # by tir support/nvcc.py via the env var; set per-shape here because each
    # bench config compiles in its own process. FA4_REG_LEVEL overrides (tuning).
    _reg_override = os.environ.get("FA4_REG_LEVEL", "")
    if _reg_override:
        _reg_level = _reg_override
    elif is_causal and num_qo_heads == num_kv_heads:
        _reg_level = "4" if seq_len_q <= 1024 else "5"
    else:
        _reg_level = "10"
    os.environ["TVM_CUDA_PTXAS_REG_LEVEL"] = _reg_level
    # Pipeline-depth split by wave count. Single-wave causal shapes (s1024) are
    # warpgroup-handshake-bound (stall-PC: ~40% mbarrier sync); a DEEPER
    # O-accumulator TMEM pipeline (TMEM_PIPE_DEPTH 3, funded by SMEM_PIPE_DEPTH_KV
    # 2 — SMEM-neutral) hides the correction-wg handshake bubble. Clean A/B
    # (warmup=100): s1024_h32kv32_c 0.953->0.998, s1024_h32kv16_c 0.985->1.002.
    # Multi-wave shapes (s2048+) instead need KV depth 3 for the long KV stream:
    # 3/2 REGRESSES s2048 (0.991->0.951) and s4096 (0.992->0.968), so they keep
    # the default 2/3. KV depth 1 deadlocks (pipeline needs >=2 stages).
    _deep_o = is_causal and seq_len_q <= 1024
    _tmem_depth = 3 if _deep_o else TMEM_PIPE_DEPTH
    _kv_depth = 2 if _deep_o else SMEM_PIPE_DEPTH_KV
    return _kernel.specialize(
        BATCH_SIZE=batch_size,
        SEQ_LEN_Q=seq_len_q,
        SEQ_LEN_KV=seq_len_kv,
        NUM_QO_HEADS=num_qo_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        is_causal=is_causal,
        TMEM_PIPE_DEPTH=_tmem_depth,
        SMEM_PIPE_DEPTH_KV=_kv_depth,
    )


def prepare_data(batch_size, seq_len_q, seq_len_kv, num_qo_heads, num_kv_heads, head_dim):
    torch.manual_seed(0)
    Q = torch.randn((batch_size, seq_len_q, num_qo_heads, head_dim), dtype=torch.float16)
    K = torch.randn((batch_size, seq_len_kv, num_kv_heads, head_dim), dtype=torch.float16)
    V = torch.randn((batch_size, seq_len_kv, num_kv_heads, head_dim), dtype=torch.float16)
    O = torch.zeros((batch_size, seq_len_q, num_qo_heads, head_dim), dtype=torch.float16)
    return (Q, K, V, O)


KERNEL_META = {"name": "flash_attention4", "category": "attention", "compute_capability": 10}
CONFIGS = [
    {
        "batch_size": 1,
        "seq_len": sl,
        "num_qo_heads": 32,
        "num_kv_heads": kv,
        "head_dim": 128,
        "is_causal": causal,
        "label": f"s{sl}_h32kv{kv}{('_causal' if causal else '')}",
    }
    for sl in [1024, 2048, 4096, 8192]
    for kv in [4, 8, 16, 32]
    for causal in [False, True]
]


def get_kernel(
    batch_size, seq_len, num_qo_heads, num_kv_heads, head_dim, is_causal=False, **kwargs
):
    return get_flash_attention4_kernel(
        batch_size, seq_len, seq_len, num_qo_heads, num_kv_heads, head_dim, is_causal=is_causal
    )


def run_test(batch_size, seq_len, num_qo_heads, num_kv_heads, head_dim, is_causal=False, **kwargs):
    """Compile, run, and verify flash attention 4 kernel."""
    from tirx_kernels.runner import compile_kernel

    Q, K, V, _ = prepare_data(batch_size, seq_len, seq_len, num_qo_heads, num_kv_heads, head_dim)
    prim_func = get_flash_attention4_kernel(
        batch_size, seq_len, seq_len, num_qo_heads, num_kv_heads, head_dim, is_causal=is_causal
    )
    ex = compile_kernel(prim_func)
    Q_tir = Q.cuda()
    K_tir = K.cuda()
    V_tir = V.cuda()
    O_tir = torch.empty(
        (batch_size, seq_len, num_qo_heads, head_dim), dtype=torch.float16, device="cuda"
    )
    profiler_buf = torch.zeros(PROFILER_BUFFER_SIZE, dtype=torch.uint64, device="cuda")
    ex(Q_tir, K_tir, V_tir, O_tir, profiler_buf)
    torch.cuda.synchronize()
    Q_t = Q.float().transpose(1, 2)
    K_t = K.float().transpose(1, 2)
    V_t = V.float().transpose(1, 2)
    if num_qo_heads != num_kv_heads:
        repeat_factor = num_qo_heads // num_kv_heads
        K_t = K_t.repeat_interleave(repeat_factor, dim=1)
        V_t = V_t.repeat_interleave(repeat_factor, dim=1)
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(Q_t, K_t.transpose(-2, -1)) * scale
    if is_causal:
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    ref = torch.matmul(attn, V_t).transpose(1, 2).to(torch.float16)
    np.testing.assert_allclose(O_tir.cpu().numpy(), ref.cpu().numpy(), rtol=0.01, atol=0.01)


def run_bench(
    batch_size,
    seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    is_causal=False,
    warmup=10,
    repeat=30,
    timer="proton",
    **kwargs,
):
    """Benchmark flash attention 4."""
    from tirx_kernels.runner import compile_kernel

    prim_func = get_flash_attention4_kernel(
        batch_size, seq_len, seq_len, num_qo_heads, num_kv_heads, head_dim, is_causal=is_causal
    )
    ex = compile_kernel(prim_func)

    def make_input():
        Q, K, V, _ = prepare_data(
            batch_size, seq_len, seq_len, num_qo_heads, num_kv_heads, head_dim
        )
        Q_cuda = Q.cuda()
        K_cuda = K.cuda()
        V_cuda = V.cuda()
        O_tir = torch.empty(
            (batch_size, seq_len, num_qo_heads, head_dim), dtype=torch.float16, device="cuda"
        )
        profiler_buf = torch.zeros(PROFILER_BUFFER_SIZE, dtype=torch.uint64, device="cuda")
        case = {
            "tir": (Q_cuda, K_cuda, V_cuda, O_tir, profiler_buf),
            # Cheap reshapes; only consumed when flashinfer is actually benched.
            "flashinfer": (
                Q_cuda.reshape(-1, num_qo_heads, head_dim),
                K_cuda.reshape(-1, num_kv_heads, head_dim),
                V_cuda.reshape(-1, num_kv_heads, head_dim),
            ),
        }
        return case, tensor_bytes(Q_cuda, K_cuda, V_cuda, O_tir)

    def _flashinfer():
        import flashinfer

        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace_buffer, "NHD", backend="cutlass"
        )
        qo_indptr = torch.tensor([0, seq_len], device="cuda:0", dtype=torch.int32)
        kv_indptr = torch.tensor([0, seq_len], device="cuda:0", dtype=torch.int32)
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
        )
        Q0, K0, V0, _ = prepare_data(
            batch_size, seq_len, seq_len, num_qo_heads, num_kv_heads, head_dim
        )
        wrapper.run(
            Q0.reshape(-1, num_qo_heads, head_dim).cuda(),
            K0.reshape(-1, num_kv_heads, head_dim).cuda(),
            V0.reshape(-1, num_kv_heads, head_dim).cuda(),
        )
        torch.cuda.synchronize()
        return lambda case: wrapper.run(*case["flashinfer"])

    def _flashattn_sm100():
        # Flash-Attention SM100 (CuTeDSL FA4) baseline.
        #
        # CUTe-DSL hard rule (discovered by experiment): every `cute_tensor_like`
        # call must happen BEFORE `cute.compile`. Wrapping new tensors after
        # compile poisons the host-side `cuTensorMapEncodeTiled` path (it starts
        # failing ~hundreds of launches later anywhere in the process, including
        # in unrelated TIR kernels). So we pre-allocate one full FA tensor set
        # per bench input group, wrap them all up-front, then compile exactly
        # once using group 0 as the prototype. Each group's storage is
        # independent, so the bench's L2 eviction protocol applies to FA the
        # same way it does to TIR/FlashInfer.
        import cutlass
        import cutlass.cute as cute
        import cutlass.torch as cutlass_torch
        from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100

        from tvm.tirx.bench import _compute_group_count, _get_l2_cache_bytes

        # Estimate per-group input footprint to match the bench's group count.
        elem_bytes = 2  # fp16
        q_bytes = batch_size * seq_len * num_qo_heads * head_dim * elem_bytes
        kv_bytes = batch_size * seq_len * num_kv_heads * head_dim * elem_bytes
        per_group_bytes = q_bytes + 2 * kv_bytes + q_bytes  # Q + K + V + O
        num_fa_groups = _compute_group_count(per_group_bytes, _get_l2_cache_bytes())

        fa_groups = []
        fa_keep_alive = []
        for _ in range(num_fa_groups):
            Qi, Ki, Vi, _ = prepare_data(
                batch_size, seq_len, seq_len, num_qo_heads, num_kv_heads, head_dim
            )
            Qf_g = Qi.cuda().contiguous()
            Kf_g = Ki.cuda().contiguous()
            Vf_g = Vi.cuda().contiguous()
            Of_g = torch.zeros_like(Qf_g)
            q_t_g, q_th_g = cutlass_torch.cute_tensor_like(
                Qf_g, cutlass.Float16, is_dynamic_layout=True, assumed_align=16
            )
            k_t_g, k_th_g = cutlass_torch.cute_tensor_like(
                Kf_g, cutlass.Float16, is_dynamic_layout=True, assumed_align=16
            )
            v_t_g, v_th_g = cutlass_torch.cute_tensor_like(
                Vf_g, cutlass.Float16, is_dynamic_layout=True, assumed_align=16
            )
            o_t_g, o_th_g = cutlass_torch.cute_tensor_like(
                Of_g, cutlass.Float16, is_dynamic_layout=True, assumed_align=16
            )
            fa_groups.append((q_t_g, k_t_g, v_t_g, o_t_g))
            fa_keep_alive.append((q_th_g, k_th_g, v_th_g, o_th_g, Qf_g, Kf_g, Vf_g, Of_g))

        fa_fwd = FlashAttentionForwardSm100(
            head_dim=head_dim,
            head_dim_v=head_dim,
            qhead_per_kvhead=num_qo_heads // num_kv_heads,
            is_causal=is_causal,
            is_local=False,
            pack_gqa=False,
            m_block_size=128,
            n_block_size=128,
            is_persistent=True,
        )
        _stream_fa = cutlass_torch.default_stream()
        _scale_fa = 1.0 / math.sqrt(head_dim)
        proto_q, proto_k, proto_v, proto_o = fa_groups[0]
        compiled_fa = cute.compile(
            fa_fwd,
            proto_q,
            proto_k,
            proto_v,
            proto_o,
            None,  # mLSE
            _scale_fa,  # softmax_scale
            None,  # mCuSeqlensQ
            None,  # mCuSeqlensK
            None,  # mSeqUsedQ
            None,  # mSeqUsedK
            None,  # mPageTable
            None,  # window_size_left
            None,  # window_size_right
            None,  # learnable_sink
            None,  # descale_tensors
            None,  # blocksparse_tensors
            None,  # aux_tensors
            _stream_fa,  # stream (FA4 sm100 keeps stream as the LAST positional)
        )

        counter = [0]

        def run(case):
            idx = counter[0] % len(fa_groups)
            counter[0] += 1
            q_t, k_t, v_t, o_t = fa_groups[idx]
            compiled_fa(
                q_t,
                k_t,
                v_t,
                o_t,
                None,  # mLSE
                _scale_fa,
                None,  # mCuSeqlensQ
                None,  # mCuSeqlensK
                None,  # mSeqUsedQ
                None,  # mSeqUsedK
                None,  # mPageTable
                None,  # window_size_left
                None,  # window_size_right
                None,  # learnable_sink
                None,  # descale_tensors
                None,  # blocksparse_tensors
                None,  # aux_tensors
                _stream_fa,  # stream (FA4 sm100 keeps stream as the LAST positional)
            )

        # Keep the per-group backing torch storage alive for the run's lifetime
        # (the cute tensors in fa_groups alias it).
        run._fa_keep_alive = fa_keep_alive
        return run

    funcs = {"tir": lambda case: ex(*case["tir"])}
    return bench(
        funcs,
        make_input,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        proton_name="flash_attention4",
        references={"flashinfer": _flashinfer, "flashattn_sm100": _flashattn_sm100},
    )

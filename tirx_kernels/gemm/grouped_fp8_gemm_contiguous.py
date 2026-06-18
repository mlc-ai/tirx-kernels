from __future__ import annotations

# ruff: noqa: E402,I001

import math
import random

import deep_gemm
import torch
from deep_gemm.testing import calc_diff
from deep_gemm.utils.math import align, per_block_cast_to_fp8, per_token_cast_to_fp8

import tvm
from tvm.backend.loader import load

load("cuda")

from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.bench import bench, tensor_bytes
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import SwizzleMode
from tvm.tirx.lang.pipeline import MBarrier, Pipeline, PipelineState
from tvm.tirx.lang.tile_scheduler import ClusterPersistentScheduler2D


KERNEL_META = {"name": "grouped_fp8_gemm_contiguous", "category": "gemm", "compute_capability": 10}
CONFIGS = [
    {
        "num_groups": 4,
        "expected_m_per_group": 256,
        "N": 512,
        "K": 512,
        "seed": 1,
        "label": "small_g4_m256_n512_k512",
    },
    {
        "num_groups": 8,
        "expected_m_per_group": 256,
        "N": 1024,
        "K": 512,
        "seed": 2,
        "label": "small_g8_m256_n1024_k512",
    },
]
BENCH_CONFIGS = [
    {
        "num_groups": 4,
        "expected_m_per_group": 8192,
        "N": 6144,
        "K": 7168,
        "seed": 1,
        "label": "large_g4_m8192_n6144_k7168",
    },
    {
        "num_groups": 4,
        "expected_m_per_group": 8192,
        "N": 7168,
        "K": 3072,
        "seed": 2,
        "label": "large_g4_m8192_n7168_k3072",
    },
    {
        "num_groups": 4,
        "expected_m_per_group": 8192,
        "N": 4096,
        "K": 4096,
        "seed": 3,
        "label": "large_g4_m8192_n4096_k4096",
    },
    {
        "num_groups": 4,
        "expected_m_per_group": 8192,
        "N": 4096,
        "K": 2048,
        "seed": 4,
        "label": "large_g4_m8192_n4096_k2048",
    },
    {
        "num_groups": 8,
        "expected_m_per_group": 4096,
        "N": 6144,
        "K": 7168,
        "seed": 5,
        "label": "large_g8_m4096_n6144_k7168",
    },
    {
        "num_groups": 8,
        "expected_m_per_group": 4096,
        "N": 7168,
        "K": 3072,
        "seed": 6,
        "label": "large_g8_m4096_n7168_k3072",
    },
    {
        "num_groups": 8,
        "expected_m_per_group": 4096,
        "N": 4096,
        "K": 4096,
        "seed": 7,
        "label": "large_g8_m4096_n4096_k4096",
    },
    {
        "num_groups": 8,
        "expected_m_per_group": 4096,
        "N": 4096,
        "K": 2048,
        "seed": 8,
        "label": "large_g8_m4096_n4096_k2048",
    },
]

DIFF_THRESHOLD = 0.002


def _align(value: int, alignment: int) -> int:
    return math.ceil(value / alignment) * alignment


def _swizzle_mode(block_size: int, elem_size: int) -> int:
    for mode in (128, 64, 32, 16):
        if (block_size * elem_size) % mode == 0:
            return mode
    raise AssertionError("unreachable swizzle mode")


def _deepgemm_num_stages(
    *, swap_ab: bool, block_m: int, block_n: int, load_block_m: int, load_block_n: int
) -> int:
    """Match DeepGEMM SM100 shared-memory stage count for FP8 grouped GEMM."""

    block_k = 128
    swizzle_cd = _swizzle_mode(block_n, 2)
    if swap_ab:
        smem_cd = 16 * block_n * 2 * 2
    else:
        smem_cd = min(block_m, 128) * swizzle_cd * 2
    smem_barriers = 32 * 8 * 3 + 2 * 8 * 2 + 8
    smem_tmem_ptr = 4
    smem_per_stage = (
        load_block_m * block_k
        + load_block_n * block_k
        + _align(block_m, 128) * 4
        + _align(block_n, 128) * 4
    )
    smem_capacity = 232448
    num_stages = (smem_capacity - smem_cd - smem_barriers - smem_tmem_ptr) // smem_per_stage
    return min(num_stages, 32)


def _pack_ue8m0_rows_to_words(sf: torch.Tensor) -> torch.Tensor:
    """Pack [rows, k_blocks] UE8M0 float scales into [k_words, rows] uint32."""

    rows, k_blocks = sf.shape
    del rows
    if sf.dtype != torch.float32:
        raise TypeError(f"expected float32 scales, got {sf.dtype}")
    if k_blocks % 4 != 0:
        raise ValueError(f"k_blocks={k_blocks} must be divisible by 4")
    sf_u8 = (sf.view(torch.int32) >> 23).to(torch.uint8).contiguous()
    return sf_u8.view(torch.uint32).T.contiguous()


def _pack_b_scales_for_tir(sfb: torch.Tensor, n: int) -> torch.Tensor:
    """Pack B block scales from [ceil(N/128), k_blocks] into [k_words, N]."""

    _, k_blocks = sfb.shape
    if k_blocks % 4 != 0:
        raise ValueError(f"k_blocks={k_blocks} must be divisible by 4")
    sf_u8 = (sfb.view(torch.int32) >> 23).to(torch.uint8).contiguous()
    sf_rows = sf_u8.repeat_interleave(128, dim=0)[:n, :].contiguous()
    return sf_rows.view(torch.uint32).T.contiguous()


def _make_deepgemm_prepacked_scales(
    sfa: torch.Tensor, sfb: torch.Tensor, m: int, n: int, k: int, num_groups: int
) -> tuple[torch.Tensor, torch.Tensor]:
    recipe = (1, 128)
    sfb_rows = sfb.repeat_interleave(128, dim=1)[:, :n, :].contiguous()
    sfa_deep = deep_gemm.transform_sf_into_required_layout(sfa, m, k, recipe)
    sfb_deep = deep_gemm.transform_sf_into_required_layout(sfb_rows, n, k, recipe, num_groups)
    return sfa_deep, sfb_deep


def _make_actual_ms(
    num_groups: int, expected_m_per_group: int, seed: int, alignment: int
) -> tuple[list[int], list[int]]:
    random.seed(seed)
    actual = [int(expected_m_per_group * random.uniform(0.7, 1.3)) for _ in range(num_groups)]
    aligned = [align(m, alignment) for m in actual]
    return actual, aligned


def prepare_data(
    num_groups: int,
    expected_m_per_group: int,
    N: int,
    K: int,
    *,
    seed: int = 0,
    device: str = "cuda",
) -> dict:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
    actual_ms, aligned_ms = _make_actual_ms(num_groups, expected_m_per_group, seed, alignment)

    M = sum(aligned_ms)
    if K % 128 != 0:
        raise ValueError(f"K={K} must be divisible by 128")

    A_bf16 = torch.randn((M, K), device=device, dtype=torch.bfloat16)
    B_bf16 = torch.randn((num_groups, N, K), device=device, dtype=torch.bfloat16)
    grouped_layout = torch.empty((M,), device=device, dtype=torch.int32)
    ref = torch.empty((M, N), device=device, dtype=torch.bfloat16)

    start = 0
    for group, (actual_m, aligned_m) in enumerate(zip(actual_ms, aligned_ms)):
        actual_end = start + actual_m
        aligned_end = start + aligned_m
        grouped_layout[start:actual_end] = group
        grouped_layout[actual_end:aligned_end] = -1
        A_bf16[actual_end:aligned_end] = 0
        ref[start:aligned_end] = (A_bf16[start:aligned_end].float() @ B_bf16[group].float().T).to(
            torch.bfloat16
        )
        start = aligned_end

    A_fp8, sfa = per_token_cast_to_fp8(A_bf16, use_ue8m0=True)
    B_fp8_groups = []
    sfb_groups = []
    for group in range(num_groups):
        B_fp8_group, sfb_group = per_block_cast_to_fp8(B_bf16[group], use_ue8m0=True)
        B_fp8_groups.append(B_fp8_group)
        sfb_groups.append(sfb_group)
    B_fp8 = torch.stack(B_fp8_groups)
    sfb = torch.stack(sfb_groups)

    SFA = _pack_ue8m0_rows_to_words(sfa)
    SFB = torch.stack([_pack_b_scales_for_tir(sfb[group], N) for group in range(num_groups)])
    sfa_deep, sfb_deep = _make_deepgemm_prepacked_scales(sfa, sfb, M, N, K, num_groups)

    return {
        "M": M,
        "N": N,
        "K": K,
        "num_groups": num_groups,
        "expected_m_per_group": expected_m_per_group,
        "actual_ms": actual_ms,
        "aligned_ms": aligned_ms,
        "A": (A_fp8, sfa),
        "B": (B_fp8, sfb),
        "A_deepgemm_fast": (A_fp8, sfa_deep),
        "B_deepgemm_fast": (B_fp8, sfb_deep),
        "A_fp8": A_fp8,
        "B_fp8": B_fp8,
        "SFA": SFA,
        "SFB": SFB,
        "D_deepgemm": torch.empty((M, N), device=device, dtype=torch.bfloat16),
        "D_tir": torch.empty((M, N), device=device, dtype=torch.bfloat16),
        "grouped_layout": grouped_layout,
        "ref": ref,
        "alignment": alignment,
    }


@T.jit
def _kernel(
    A: T.Buffer((M, K), "float8_e4m3fn"),
    B: T.Buffer((NUM_GROUPS, N, K), "float8_e4m3fn"),
    SFA: T.Buffer((math.ceil(K / 128) // 4, M), "uint32"),
    SFB: T.Buffer((NUM_GROUPS, math.ceil(K / 128) // 4, N), "uint32"),
    D: T.Buffer((M, N), "bfloat16"),
    GROUPED_LAYOUT: T.Buffer((M,), "int32"),
    *,
    # problem size
    NUM_GROUPS: T.constexpr,
    M: T.constexpr,
    N: T.constexpr,
    K: T.constexpr,
    # block + cluster layout
    SWAP_AB: T.constexpr,
    DG_BLOCK_M: T.constexpr,
    DG_BLOCK_N: T.constexpr,
    LOGICAL_M_CLUSTER: T.constexpr,
    LOGICAL_N_CLUSTER: T.constexpr,
    # tile / MMA sizes
    BLK_K: T.constexpr = 128,
    MMA_K: T.constexpr = 32,
    EPI_TILE: T.constexpr = 32,
    TMEM_LD_SIZE: T.constexpr = 8,
    # pipeline depths
    SMEM_DEPTH: T.constexpr,
    TMEM_DEPTH: T.constexpr = 2,
    # warp / SM / scheduler
    WG_NUMBER: T.constexpr = 2,
    SM_NUMBER: T.constexpr = 148,
    TILE_GROUPS_ROW_SIZE: T.constexpr = 16,
):
    CTA_GROUP = T.meta_var(LOGICAL_M_CLUSTER * LOGICAL_N_CLUSTER)
    M_CLUSTER = T.meta_var(CTA_GROUP)
    N_CLUSTER = T.meta_var(1)
    MMA_N = T.meta_var(DG_BLOCK_M if SWAP_AB else DG_BLOCK_N)
    BLK_M = T.meta_var(DG_BLOCK_M // LOGICAL_N_CLUSTER if SWAP_AB else DG_BLOCK_M)
    BLK_N = T.meta_var(DG_BLOCK_N if SWAP_AB else DG_BLOCK_N // LOGICAL_M_CLUSTER)
    BLK_SFA = T.meta_var(_align(DG_BLOCK_M, 128))
    BLK_SFB = T.meta_var(_align(DG_BLOCK_N, 128))
    K_TILES = T.meta_var(K // BLK_K)
    SFA_post_layout = T.meta_var(
        T.TileLayout(T.S[(SMEM_DEPTH, BLK_SFA // 128, 4, 32) : (BLK_SFA, 128, 1, 4)])
    )
    SFB_post_layout = T.meta_var(
        T.TileLayout(T.S[(SMEM_DEPTH, BLK_SFB // 128, 4, 32) : (BLK_SFB, 128, 1, 4)])
    )
    K_ITERS = T.meta_var(BLK_K // MMA_K)
    SFA_smem_fp8_layout = T.meta_var(SFA_post_layout.unpack(4).broadcast(K_ITERS))
    SFB_smem_fp8_layout = T.meta_var(SFB_post_layout.unpack(4).broadcast(K_ITERS))
    AB_bytes = T.meta_var(BLK_M * BLK_K + BLK_N * BLK_K)  # fp8 A+B operands: 1 byte/elem
    SFAB_bytes = T.meta_var((DG_BLOCK_M + DG_BLOCK_N) * 4)  # SF packed as uint32: 4 B
    SCHED_M_NUM = T.meta_var(math.ceil(N / DG_BLOCK_N) if SWAP_AB else math.ceil(M / DG_BLOCK_M))
    SCHED_N_NUM = T.meta_var(math.ceil(M / DG_BLOCK_M) if SWAP_AB else math.ceil(N / DG_BLOCK_N))
    D_SMEM_M = T.meta_var(16 if SWAP_AB else BLK_M)
    D_SMEM_N = T.meta_var(DG_BLOCK_N if SWAP_AB else EPI_TILE)
    D_SWIZZLE = T.meta_var(
        SwizzleMode.SWIZZLE_128B_ATOM if SWAP_AB else SwizzleMode.SWIZZLE_64B_ATOM
    )
    T.device_entry()
    T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})
    cbx, cby = T.cta_id_in_cluster([M_CLUSTER, N_CLUSTER])
    cluster_rank = T.ptx.fetch_register(32, "cluster_ctarank")
    bx = T.cta_id([SM_NUMBER])
    wg_id = T.warpgroup_id([WG_NUMBER])
    warp_id = T.warp_id_in_wg([4])
    tid_in_wg = T.thread_id_in_wg([128])
    lane_id = T.lane_id([32])
    pool = T.SMEMPool()
    barrier_leader = T.bitwise_and(T.cast(warp_id == 1, "uint32"), T.ptx.elect_sync()) != T.uint32(
        0
    )
    tmem_pool = T.TMEMPool(
        pool,
        total_cols=512,
        cta_group=CTA_GROUP,
        alloc_warp=2,
        dealloc_warp=0,
        sync_after_alloc=False,
    )
    smem_pipe = Pipeline(pool, SMEM_DEPTH, full="tma", empty="tcgen05", leader=barrier_leader)
    trans_done = MBarrier(pool, SMEM_DEPTH, leader=barrier_leader)
    trans_done.init(CTA_GROUP * 32)
    tmem_pipe = Pipeline(
        pool,
        TMEM_DEPTH,
        full="tcgen05",
        empty="mbar",
        init_empty=CTA_GROUP * 128,
        empty_phase_offset=1,
        leader=barrier_leader,
    )
    # ``datapath="D"`` documents that the MMA writes Layout D (M=128 full
    # datapath, identity row→lane) — the downstream ``.16x256b`` M=128
    # epilogue readback structurally checks this and would reject e.g. a
    # Layout F acc here. See PTX ISA §9.7.16.10.5.
    acc_buf = tmem_pool.alloc((128, TMEM_DEPTH * MMA_N), "float32", datapath="D")
    acc = T.meta_var(T.TMEMStages(acc_buf, col_start=0, width=MMA_N, stages=TMEM_DEPTH))
    SFA_tmem = tmem_pool.alloc_sf(
        (TMEM_DEPTH, BLK_SFA, 4 * K_ITERS), "float8_e8m0fnu", sf_per_mma=1, sf_reuse=K_ITERS
    )
    SFB_tmem = tmem_pool.alloc_sf(
        (TMEM_DEPTH, BLK_SFB, 4 * K_ITERS), "float8_e8m0fnu", sf_per_mma=1, sf_reuse=K_ITERS
    )
    pool.move_base_to(1024)
    A_smem = pool.alloc_mma((SMEM_DEPTH, BLK_M, BLK_K), "float8_e4m3fn")
    B_smem = pool.alloc_mma((SMEM_DEPTH, BLK_N, BLK_K), "float8_e4m3fn")
    D_smem = pool.alloc_mma((TMEM_DEPTH, D_SMEM_M, D_SMEM_N), "bfloat16", swizzle_mode=D_SWIZZLE)
    SFA_smem = pool.alloc((SMEM_DEPTH, BLK_SFA), "uint32")
    SFB_smem = pool.alloc((SMEM_DEPTH, BLK_SFB), "uint32")
    pool.commit()
    if barrier_leader:
        T.ptx.fence.mbarrier_init()
    stage: T.int32
    tile_scheduler = ClusterPersistentScheduler2D(
        "tile_scheduler",
        num_m_tiles=SCHED_M_NUM,
        num_n_tiles=SCHED_N_NUM,
        l2_group_size=TILE_GROUPS_ROW_SIZE,
        num_clusters=SM_NUMBER,
    )
    tile_scheduler.init(bx)
    tmem_pool.commit()
    T.cuda.cluster_sync()
    T.evaluate(T.ptx.griddepcontrol.wait())
    T.cuda.trap_when_assert_failed(tmem_pool.addr == 0)

    m_idx = T.meta_var(tile_scheduler.n_idx if SWAP_AB else tile_scheduler.m_idx)
    n_idx = T.meta_var(tile_scheduler.m_idx if SWAP_AB else tile_scheduler.n_idx)

    if wg_id == 0:
        if warp_id == 0:
            tma_cur = PipelineState(SMEM_DEPTH, 1)
            a_m = T.meta_var(
                m_idx * DG_BLOCK_M + cluster_rank * BLK_M if SWAP_AB else m_idx * DG_BLOCK_M
            )
            sf_m = T.meta_var(m_idx * DG_BLOCK_M)
            b_n = T.meta_var(
                n_idx * DG_BLOCK_N if SWAP_AB else n_idx * DG_BLOCK_N + cluster_rank * BLK_N
            )
            sf_n = T.meta_var(n_idx * DG_BLOCK_N)

            @T.inline
            def tma_load(k_tile):
                smem_pipe.empty.wait(tma_cur.stage, tma_cur.phase)
                stage = tma_cur.stage
                k = T.meta_var(k_tile * BLK_K)
                group: T.let = GROUPED_LAYOUT[sf_m]
                tma_copy = T.meta_var(
                    {
                        "dispatch": "tma",
                        "mbar": smem_pipe.full.ptr_to([stage]),
                        "cta_group": 1,
                        "cache_hint": "evict_normal",
                        "prefetch_tensormap": True,
                    }
                )
                Tx.copy_async(A_smem[stage], A[a_m : a_m + BLK_M, k : k + BLK_K], **tma_copy)
                Tx.copy_async(B_smem[stage], B[group, b_n : b_n + BLK_N, k : k + BLK_K], **tma_copy)
                if k_tile % 4 == 0:
                    Tx.copy_async(
                        SFA_smem[stage, 0:DG_BLOCK_M],
                        SFA[k_tile // 4, sf_m : sf_m + DG_BLOCK_M],
                        **tma_copy,
                    )
                    Tx.copy_async(
                        SFB_smem[stage, 0:DG_BLOCK_N],
                        SFB[group, k_tile // 4, sf_n : sf_n + DG_BLOCK_N],
                        **tma_copy,
                    )

                smem_pipe.full.arrive(
                    tma_cur.stage,
                    tx_count=T.if_then_else(k_tile % 4 == 0, AB_bytes + SFAB_bytes, AB_bytes),
                )

            @T.inline
            def tma_iter():
                for k_tile in T.serial(K_TILES):
                    tma_load(k_tile)
                    tma_cur.advance()

            if T.ptx.elect_sync():
                while tile_scheduler.valid():
                    tma_iter()
                    tile_scheduler.next_tile()
        elif warp_id == 2:
            SFA_smem_post = SFA_smem.view(SMEM_DEPTH, BLK_SFA, layout=SFA_post_layout)
            SFB_smem_post = SFB_smem.view(SMEM_DEPTH, BLK_SFB, layout=SFB_post_layout)
            trans_state = PipelineState(SMEM_DEPTH, 0)

            @T.inline
            def transpose(ks, k_tile):
                smem_pipe.full.wait(ks, trans_state.phase)
                if k_tile % 4 == 0:
                    Tx.warp.permute_layout(SFA_smem_post[ks, :], SFA_smem[ks, :])
                    Tx.warp.permute_layout(SFB_smem_post[ks, :], SFB_smem[ks, :])
                    T.ptx.fence.proxy_async("shared::cta")
                trans_done.arrive(ks, cta_id=0)

            @T.inline
            def trans_iter():
                for k_tile in T.serial(K_TILES):
                    transpose(trans_state.stage, k_tile)
                    trans_state.advance()

            while tile_scheduler.valid():
                trans_iter()
                tile_scheduler.next_tile()
        elif warp_id == 1 and cluster_rank == 0:
            SFA_smem_fp8 = SFA_smem.view("float8_e8m0fnu").view(
                SMEM_DEPTH, BLK_SFA, 4 * K_ITERS, layout=SFA_smem_fp8_layout
            )
            SFB_smem_fp8 = SFB_smem.view("float8_e8m0fnu").view(
                SMEM_DEPTH, BLK_SFB, 4 * K_ITERS, layout=SFB_smem_fp8_layout
            )
            tmem_idx: T.int32
            tmem_phase: T.int32
            mma_state = PipelineState(SMEM_DEPTH, 0)
            accum: T.int32

            @T.inline
            def mma(ks, k_tile):
                trans_done.wait(ks, mma_state.phase)
                if k_tile % 4 == 0:
                    Tx.copy_async(SFA_tmem[tmem_idx], SFA_smem_fp8[ks], cta_group=CTA_GROUP)
                    Tx.copy_async(SFB_tmem[tmem_idx], SFB_smem_fp8[ks], cta_group=CTA_GROUP)

                @T.inline
                def gemm_with_sf(sf_off: T.constexpr):
                    if SWAP_AB:
                        Tx.gemm_async(
                            acc[tmem_idx],
                            B_smem[ks],
                            A_smem[ks],
                            SFA=SFB_tmem[tmem_idx, :, sf_off : sf_off + K_ITERS],
                            SFB=SFA_tmem[tmem_idx, :, sf_off : sf_off + K_ITERS],
                            accum=accum,
                            dispatch="tcgen05",
                            cta_group=CTA_GROUP,
                        )
                    else:
                        Tx.gemm_async(
                            acc[tmem_idx],
                            A_smem[ks],
                            B_smem[ks],
                            SFA=SFA_tmem[tmem_idx, :, sf_off : sf_off + K_ITERS],
                            SFB=SFB_tmem[tmem_idx, :, sf_off : sf_off + K_ITERS],
                            accum=accum,
                            dispatch="tcgen05",
                            cta_group=CTA_GROUP,
                        )

                if k_tile % 4 == 0:
                    gemm_with_sf(0)
                elif k_tile % 4 == 1:
                    gemm_with_sf(K_ITERS)
                elif k_tile % 4 == 2:
                    gemm_with_sf(2 * K_ITERS)
                else:
                    gemm_with_sf(3 * K_ITERS)
                accum = 1
                smem_pipe.empty.arrive(ks, cta_group=CTA_GROUP, cta_mask=3)

            @T.inline
            def mma_iter():
                if T.ptx.elect_sync():
                    tmem_idx = tile_scheduler.tile_idx % TMEM_DEPTH
                    tmem_phase = tile_scheduler.tile_idx // TMEM_DEPTH & 1
                    tmem_pipe.empty.wait(tmem_idx, tmem_phase)
                    accum = 0
                    for k_tile in T.serial(K_TILES):
                        mma(mma_state.stage, k_tile)
                        mma_state.advance()
                    tmem_pipe.full.arrive(tmem_idx, cta_group=CTA_GROUP, cta_mask=3)

            while tile_scheduler.valid():
                mma_iter()
                tile_scheduler.next_tile()
    elif wg_id == 1:
        tmem_idx: T.int32
        tmem_phase: T.int32

        # Stream acc -> D_smem -> TMA in EPI-wide slices. SWAP_AB only changes the
        # acc -> D_smem step (stmatrix transpose vs straight copy) and the tiling.
        EPI = T.meta_var(16 if SWAP_AB else EPI_TILE)
        STORE_TILES = T.meta_var(MMA_N // EPI)
        D_TILE_M = T.meta_var(16 if SWAP_AB else DG_BLOCK_M)
        D_TILE_N = T.meta_var(DG_BLOCK_N if SWAP_AB else EPI_TILE)

        @T.inline
        def epilogue():
            swap_frag = T.alloc_tcgen05_ldst_frag("16x256b", (128, 8), "float32")
            swap_bf16 = T.alloc_cast_frag(swap_frag, "bfloat16")
            for ot in T.unroll(STORE_TILES):
                store_iter: T.let = tile_scheduler.tile_idx * STORE_TILES + ot
                stage = store_iter % TMEM_DEPTH
                if store_iter >= TMEM_DEPTH:
                    if warp_id == 0:
                        T.ptx.cp_async.bulk.wait_group(TMEM_DEPTH - 1)
                    T.cuda.warpgroup_sync(10)
                if SWAP_AB:
                    for atom_m in T.unroll(2):
                        col_st: T.let = ot * 16 + atom_m * 8
                        Tx.wg.copy_async(swap_frag[:, :], acc[tmem_idx, col_st : col_st + 8])
                        T.ptx.tcgen05.wait.ld()
                        Tx.wg.cast(swap_bf16, swap_frag)
                        rs = T.meta_var(atom_m * 8)
                        Tx.wg.copy(
                            D_smem[stage, rs : rs + 8, 0:128],
                            swap_bf16.permute(1, 0),
                            dispatch="ldstmatrix",
                        )
                else:
                    for ki in T.unroll(EPI_TILE // TMEM_LD_SIZE):
                        Dreg = T.wg_reg_tile(TMEM_LD_SIZE)
                        acc_n = T.meta_var(ot * EPI_TILE + ki * TMEM_LD_SIZE)
                        Tx.wg.copy_async(Dreg, acc[tmem_idx, acc_n : acc_n + TMEM_LD_SIZE])
                        T.ptx.tcgen05.wait.ld()
                        Dreg_bf16 = T.wg_reg_tile(TMEM_LD_SIZE, dtype="bfloat16")
                        Tx.wg.cast(Dreg_bf16, Dreg)
                        Tx.wg.copy(
                            D_smem[stage, :, ki * TMEM_LD_SIZE : (ki + 1) * TMEM_LD_SIZE], Dreg_bf16
                        )
                if ot == STORE_TILES - 1:
                    tmem_pipe.empty.arrive(tmem_idx, cta_id=0)
                T.ptx.fence.proxy_async("shared::cta")
                T.cuda.warpgroup_sync(10)
                d_m: T.let = m_idx * DG_BLOCK_M + (ot * 16 if SWAP_AB else 0)
                d_n: T.let = n_idx * DG_BLOCK_N + (0 if SWAP_AB else ot * EPI_TILE)
                if warp_id == 0:
                    if T.ptx.elect_sync():
                        Tx.copy_async(
                            D[d_m : d_m + D_TILE_M, d_n : d_n + D_TILE_N],
                            D_smem[stage],
                            dispatch="tma",
                            prefetch_tensormap=True,
                        )
                        T.ptx.cp_async.bulk.commit_group()

        T.cuda.trap_when_assert_failed(tmem_pool.addr == 0)
        while tile_scheduler.valid():
            tmem_idx = tile_scheduler.tile_idx % TMEM_DEPTH
            tmem_phase = tile_scheduler.tile_idx // TMEM_DEPTH & 1
            tmem_pipe.full.wait(tmem_idx, tmem_phase)
            epilogue()
            tile_scheduler.next_tile()
        if tid_in_wg == 0:
            T.ptx.cp_async.bulk.wait_group(0)
        T.cuda.warpgroup_sync(10)
    tmem_pool.dealloc()
    T.cuda.cluster_sync()


def tir_kernel(num_groups: int, M: int, N: int, K: int):
    """Specialize DeepGEMM's SM100 m-grouped contiguous FP8 layout."""

    swap_ab = True
    dg_block_m = 240
    dg_block_n = 128
    log_m = 1
    log_n = 2 if (N // dg_block_n) % 2 == 0 else 1
    smem_pipe_depth = _deepgemm_num_stages(
        swap_ab=swap_ab,
        block_m=dg_block_m,
        block_n=dg_block_n,
        load_block_m=dg_block_m // log_n,
        load_block_n=dg_block_n,
    )
    return _kernel.specialize(
        NUM_GROUPS=num_groups,
        M=M,
        N=N,
        K=K,
        SWAP_AB=swap_ab,
        DG_BLOCK_M=dg_block_m,
        DG_BLOCK_N=dg_block_n,
        SMEM_DEPTH=smem_pipe_depth,
        LOGICAL_M_CLUSTER=log_m,
        LOGICAL_N_CLUSTER=log_n,
    )


def get_kernel(num_groups: int, M: int, N: int, K: int, **kwargs):
    return tir_kernel(num_groups, M, N, K)


def _compile_tir(num_groups: int, M: int, N: int, K: int):
    target = tvm.target.Target("cuda")
    with target:
        return tvm.compile(
            tvm.IRModule({"main": tir_kernel(num_groups, M, N, K)}),
            target=target,
            tir_pipeline="tirx",
        )


def _make_tir_callable(ex, data: dict):
    A = data["A_fp8"]
    B = data["B_fp8"]
    SFA = data["SFA"]
    SFB = data["SFB"]
    D = data["D_tir"]
    grouped_layout = data["grouped_layout"]

    def kernel_fn():
        ex.mod(A, B, SFA, SFB, D, grouped_layout)

    return kernel_fn


def setup(data: dict, num_groups: int, M: int, N: int, K: int):
    assert int(data["alignment"]) == 240
    ex = _compile_tir(num_groups, M, N, K)
    kernel_fn = _make_tir_callable(ex, data)
    kernel_fn()
    return kernel_fn


def setup_deep_gemm(data: dict, *, use_prepacked_scales: bool = True):
    A = data["A_deepgemm_fast"] if use_prepacked_scales else data["A"]
    B = data["B_deepgemm_fast"] if use_prepacked_scales else data["B"]
    D = data["D_deepgemm"]
    grouped_layout = data["grouped_layout"]
    recipe = (1, 1, 128) if use_prepacked_scales else None

    def kernel_fn():
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(A, B, D, grouped_layout, recipe=recipe)

    kernel_fn()
    return kernel_fn


def _check(tag: str, out: torch.Tensor, ref: torch.Tensor) -> float:
    diff = calc_diff(out, ref)
    if diff >= DIFF_THRESHOLD:
        raise AssertionError(f"{tag} diff {diff:.6f} >= {DIFF_THRESHOLD}")
    return diff


def run_test(
    num_groups: int = 4,
    expected_m_per_group: int = 256,
    N: int = 512,
    K: int = 512,
    seed: int = 1,
    **kwargs,
):
    data = prepare_data(num_groups, expected_m_per_group, N, K, seed=seed)
    M = data["M"]
    setup_deep_gemm(data, use_prepacked_scales=True)
    setup(data, num_groups, M, N, K)
    _check("deepgemm", data["D_deepgemm"], data["ref"])
    _check("tir", data["D_tir"], data["ref"])


def run_bench(
    num_groups: int = 4,
    expected_m_per_group: int = 8192,
    N: int = 4096,
    K: int = 2048,
    seed: int = 1,
    *,
    warmup: int = 10,
    repeat: int = 30,
    timer: str = "event",
    deepgemm_fp32_scales: bool = False,
    **kwargs,
):
    use_prepacked_scales = not deepgemm_fp32_scales
    sample = prepare_data(num_groups, expected_m_per_group, N, K, seed=seed)
    M = sample["M"]
    if int(sample["alignment"]) != 240:
        raise AssertionError(
            f"expected DeepGEMM contiguous alignment 240, got {sample['alignment']}"
        )

    ex = _compile_tir(num_groups, M, N, K)
    tir_sample = _make_tir_callable(ex, sample)
    setup_deep_gemm(sample, use_prepacked_scales=use_prepacked_scales)
    tir_sample()
    _check("deepgemm", sample["D_deepgemm"], sample["ref"])
    _check("tir", sample["D_tir"], sample["ref"])

    def make_input():
        data = prepare_data(num_groups, expected_m_per_group, N, K, seed=seed)
        case = {
            "data": data,
            "tir": _make_tir_callable(ex, data),
            "deepgemm": setup_deep_gemm(data, use_prepacked_scales=use_prepacked_scales),
        }
        input_bytes = tensor_bytes(
            data["A_fp8"],
            data["B_fp8"],
            data["SFA"],
            data["SFB"],
            data["grouped_layout"],
            data["D_tir"],
        )
        return case, input_bytes

    result = bench(
        {"deepgemm": lambda case: case["deepgemm"](), "tir": lambda case: case["tir"]()},
        make_input,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        proton_name="grouped_fp8_gemm_contiguous",
        cooldown_s=0.0,
    )
    result.update(
        {
            "M": M,
            "N": N,
            "K": K,
            "num_groups": num_groups,
            "actual_ms": sample["actual_ms"],
            "aligned_ms": sample["aligned_ms"],
            "deepgemm_scale_mode": "prepacked-int32" if use_prepacked_scales else "fp32-api",
            "tir_launches": 1,
        }
    )
    impls = result.get("impls", {})
    if impls.get("deepgemm", 0) > 0 and "tir" in impls:
        result["ratio"] = impls["tir"] / impls["deepgemm"]
    return result

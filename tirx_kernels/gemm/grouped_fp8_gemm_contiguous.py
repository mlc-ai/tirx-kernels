from __future__ import annotations

# ruff: noqa: E402,I001

import math

from tvm.backend.loader import load

load("cuda")

from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import SwizzleMode
from tvm.tirx.lang.pipeline import MBarrier, Pipeline, PipelineState
from tvm.tirx.lang.tile_scheduler import ClusterPersistentScheduler2D


def _align(value: int, alignment: int) -> int:
    return math.ceil(value / alignment) * alignment


def _swizzle_mode(block_size: int, elem_size: int) -> int:
    for mode in (128, 64, 32, 16):
        if (block_size * elem_size) % mode == 0:
            return mode
    raise AssertionError("unreachable swizzle mode")


def _num_smem_stages(
    *, swap_ab: bool, block_m: int, block_n: int, load_block_m: int, load_block_n: int
) -> int:
    """Compute the SM100 shared-memory stage count for the grouped FP8 schedule."""

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


def grouped_fp8_gemm_contiguous(num_groups: int, M: int, N: int, K: int):
    """Return the SM100 m-grouped contiguous FP8 GEMM PrimFunc."""

    swap_ab = True
    block_m = 240
    block_n = 128
    log_m = 1
    log_n = 2 if (N // block_n) % 2 == 0 else 1
    smem_pipe_depth = _num_smem_stages(
        swap_ab=swap_ab,
        block_m=block_m,
        block_n=block_n,
        load_block_m=block_m // log_n,
        load_block_n=block_n,
    )
    return _kernel.specialize(
        NUM_GROUPS=num_groups,
        M=M,
        N=N,
        K=K,
        SWAP_AB=swap_ab,
        DG_BLOCK_M=block_m,
        DG_BLOCK_N=block_n,
        SMEM_DEPTH=smem_pipe_depth,
        LOGICAL_M_CLUSTER=log_m,
        LOGICAL_N_CLUSTER=log_n,
    )

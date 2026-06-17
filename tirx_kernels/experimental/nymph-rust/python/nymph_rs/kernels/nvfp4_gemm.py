"""NVFP4 GEMM (Blackwell sm100 block-scaled fp4) expressed in Nymph IR.

Faithful port of ``tirx_kernels/gemm/nvfp4_gemm.py``. The operands are FP4
(``e2m1``, two values packed per ``uint8`` byte) and the GEMM is block-scaled at
block size 16: one ``e4m3`` scale factor per 16 contiguous K-elements, applied
to BOTH operands, with a final global ``alpha`` rescale in the epilogue. The
port keeps, at the same granularity as the fp16/bf16, FA4, and fp8-blockwise
ports:

- the cluster datapath: ``CTA_GROUP=2`` with ``CLUSTER_M=2`` — the cluster pair
  takes two adjacent M tiles (A split by M across the pair) and shares one
  ``MMA_N = CTA_N * CTA_GROUP = 256`` N band (B split by N across the pair). This
  is the verified ``(m=256, cta_group=2)`` block-scaled MMA;
- the role split: one TMA-load warp, one scale-factor permute warp, one MMA warp
  (issuing from the cluster leader only), and one epilogue warpgroup;
- the pipeline protocol, identical in shape to the fp8 port: ``smem_pipe`` (full
  = TMA arrive-expect-tx per k-tile covering A+B+SFA+SFB, empty = a
  tcgen05_commit multicast to both CTAs), ``trans_done`` (both CTAs' permute
  warps arrive at the leader, ordering the TMA data before the cluster MMA),
  and the single-stage ``tmem_pipe`` (full = tcgen05_commit multicast, empty =
  both CTAs' epilogues arrive at the leader, first wait passes via the +1 phase
  offset);
- the data path: per k-tile (CTA_K=256) the MMA issues ``K_ITERS = CTA_K//MMA_K
  = 4`` block-scaled instructions of ``MMA_K=64``. Each issue covers 4 scale
  blocks (``SF_PER_MMA = MMA_K//16 = 4``) held as one packed-u32 e4m3 cell per
  operand row. SFA/SFB are TMA'd every k-tile, permuted in SMEM, ``tcgen05.cp``'d
  into TMEM, and read per issue. The accumulator (one TMEM stage of MMA_N cols)
  is rescaled by ``alpha`` and cast to bf16 in the epilogue.

Sub-value physical-layout details (SMEM swizzles, the scale permute's byte
shuffle, the e2m1 nibble packing, the SF-cell TMEM broadcast) are modeled
logically, exactly like the sibling GEMM/attention ports. ``alpha`` is applied
as the epilogue rescale; on silicon it is a runtime ``(1,)`` buffer, here a
power-of-two value-model constant (the value test fixes the global scales).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..builder import IRBuilder
from ..nymph_rs import (
    DType,
    FenceKind,
    FenceScope,
    Kernel,
    LaunchShape,
    MBarKind,
    MemorySpace,
    TensorSlice,
    TmemLayout,
    TmemLayoutKind,
)

CTA_M = 128  # per-CTA A tile rows (one M tile)
CTA_N = 128  # per-CTA B tile rows (this CTA's half of the shared N band)
CTA_GROUP = 2
CLUSTER_M = 2
MMA_N = CTA_N * CTA_GROUP  # 256, the shared N band the pair computes together
MMA_M = 256  # the pair's two M tiles
CTA_K = 256  # K per pipeline tile
MMA_K = 64  # block-scaled fp4 MMA instruction K
K_ITERS = CTA_K // MMA_K  # 4 MMA issues per k-tile
SF_BLOCK = 16  # one e4m3 scale per 16 K-elements
SF_PER_MMA = MMA_K // SF_BLOCK  # 4 scale blocks per MMA issue (one packed u32 cell)
SF_CELLS = CTA_K // SF_BLOCK // 4  # packed-u32 scale cells per row per k-tile == K_ITERS
BLK_K_BYTES = CTA_K // 2  # packed fp4 bytes per row per k-tile (2 e2m1 per byte)
EPI_TILE = 64
TMEM_LD_SIZE = 8
ACC_DEPTH = 1  # accumulator TMEM stages (MMA_N=256 fills half of 512; one stage)
D_DEPTH = 2  # D_smem store ring depth (store pacing)
SMEM_DEPTH = 5  # SMEM pipeline depth (mirrors TIRx PIPE_DEPTH)
N_COLS_TMEM = 512
TILE_GROUPS_ROW_SIZE = 16
SM_NUMBER = 148
U32_BYTES = 4


def _ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


@dataclass(frozen=True, slots=True)
class NvFp4GemmConfig:
    m: int = 1024
    n: int = 1024
    k: int = 1024
    # alpha = 1 / (A_global_sf * B_global_sf), applied as the epilogue rescale.
    # A power of two keeps the value model bit-exact; the default 1.0 matches a
    # unit global scale.
    alpha: float = 1.0
    launch_shape: LaunchShape | None = None


def nvfp4_task_config(tasks: int) -> NvFp4GemmConfig:
    """The canonical single-cluster examination setting, mirroring the fp16/bf16
    and fp8 ones: one m tile per CTA (M = CTA_M * CLUSTER_M = 256, two adjacent M
    tiles), k = 16384 (64 k-tiles per task), on ONE cluster pair
    (``launch_shape=(2,)``), with the task count varied through N = MMA_N * tasks
    (each pair-task covers one 256-wide N band)."""
    return NvFp4GemmConfig(
        m=CTA_M * CLUSTER_M, n=MMA_N * tasks, k=16384, alpha=1.0, launch_shape=(2,)
    )


CONFIGS = [
    {"m": s, "n": s, "k": s, "label": f"{s}x{s}x{s}"} for s in [1024, 2048, 4096, 8192, 16384]
]

# Cheap shapes for the protocol + value sweeps: different k-tile counts and tile
# counts, all on the single cluster datapath. Larger squares build/protocol only.
NVFP4_CONFIGS_SUPPORTED = [
    {"m": 256, "n": 256, "k": 256, "label": "256x256x256"},  # 1 k-tile, 1 tile
    {"m": 256, "n": 512, "k": 512, "label": "256x512x512"},  # 2 k-tiles, 2 N tiles
    {"m": 512, "n": 256, "k": 512, "label": "512x256x512"},  # 2 M tiles (2 pairs)
    {"m": 512, "n": 512, "k": 1024, "label": "512x512x1024"},  # 4 k-tiles, 2x2 tiles
    {"m": 1024, "n": 1024, "k": 1024, "label": "1024x1024x1024"},
]


def build_nvfp4_gemm(config: NvFp4GemmConfig = NvFp4GemmConfig()) -> Kernel:
    M, N, K = config.m, config.n, config.k
    _validate_config(config)
    cta_group = CTA_GROUP
    blk_m = CTA_M  # per-CTA A rows (its own M tile)
    blk_n = CTA_N  # per-CTA B rows (its half of the shared N band)
    sched_rows = _ceil_div(M, CTA_M)  # M tiles
    sched_cols = _ceil_div(N, MMA_N)  # N bands
    k_tiles = K // CTA_K
    total_work = sched_rows * sched_cols
    store_tiles = MMA_N // EPI_TILE

    launch_shape = config.launch_shape or (
        max(cta_group, min(SM_NUMBER, total_work) // cta_group * cta_group),
    )
    _validate_launch_shape(launch_shape, cta_group)
    pair_tasks = total_work // cta_group

    # packed fp4 operand tiles, e4m3 scale cells, bf16 output tile
    a_tile_bytes = blk_m * BLK_K_BYTES
    b_tile_bytes = blk_n * BLK_K_BYTES
    sfa_tile_bytes = SF_CELLS * blk_m * U32_BYTES  # per k-tile, this CTA's M rows
    sfb_tile_bytes = SF_CELLS * MMA_N * U32_BYTES  # per k-tile, the full N band
    d_tile_bytes = blk_m * EPI_TILE * 2

    a_off = 0
    b_off = a_off + SMEM_DEPTH * a_tile_bytes
    sfa_off = b_off + SMEM_DEPTH * b_tile_bytes
    sfb_off = sfa_off + SMEM_DEPTH * sfa_tile_bytes
    d_off = sfb_off + SMEM_DEPTH * sfb_tile_bytes
    smem_size_bytes = d_off + D_DEPTH * d_tile_bytes

    k = IRBuilder(
        "nymph_nvfp4_gemm",
        num_warps=8,  # wg0 = tma/permute/mma warps, wg1 = epilogue
        smem_size_bytes=smem_size_bytes,
        launch_shape=launch_shape,
        cluster_shape=(cta_group,),
    )
    # Operands are packed fp4: uint8[rows, K//2] (two e2m1 per byte), exactly the
    # TIRx A_packed/B_packed storage. Scales are packed e4m3 cells (u32).
    a_gmem = k.arg(space=MemorySpace.GMEM, dtype=DType.U8, shape=(M, K // 2))
    b_gmem = k.arg(space=MemorySpace.GMEM, dtype=DType.U8, shape=(N, K // 2))
    sfa_gmem = k.arg(space=MemorySpace.GMEM, dtype=DType.U32, shape=(k_tiles * SF_CELLS, M))
    sfb_gmem = k.arg(space=MemorySpace.GMEM, dtype=DType.U32, shape=(k_tiles * SF_CELLS, N))
    d_gmem = k.arg(space=MemorySpace.GMEM, dtype=DType.BF16, shape=(M, N))

    # Stage-major SMEM rings, indexed by a runtime pipeline stage (the continuous
    # PipelineState seq % depth, never reset per task).
    a_smem = k.tensor(
        space=MemorySpace.SMEM,
        dtype=DType.U8,
        shape=(SMEM_DEPTH, blk_m, BLK_K_BYTES),
        byte_offset=a_off,
    )
    b_smem = k.tensor(
        space=MemorySpace.SMEM,
        dtype=DType.U8,
        shape=(SMEM_DEPTH, blk_n, BLK_K_BYTES),
        byte_offset=b_off,
    )
    # SF SMEM laid issue-major: [stage, issue (k-block group), row]. The cp flattens
    # this row-major and places element r at TMEM (lane r%128, col base + r/128),
    # which is exactly the (row, issue[, N-half]) cell layout the MMA reads back.
    sfa_smem = k.tensor(
        space=MemorySpace.SMEM,
        dtype=DType.U32,
        shape=(SMEM_DEPTH, SF_CELLS, blk_m),
        byte_offset=sfa_off,
    )
    sfb_smem = k.tensor(
        space=MemorySpace.SMEM,
        dtype=DType.U32,
        shape=(SMEM_DEPTH, SF_CELLS, MMA_N),
        byte_offset=sfb_off,
    )
    d_smem = k.tensor(
        space=MemorySpace.SMEM,
        dtype=DType.BF16,
        shape=(D_DEPTH, blk_m, EPI_TILE),
        byte_offset=d_off,
    )

    # TMEM: accumulator (one MMA_N stage) at col 0, then the scale-vector cells.
    sfa_col0 = ACC_DEPTH * MMA_N
    sfb_col0 = sfa_col0 + ACC_DEPTH * SF_CELLS
    tmem_base = k.tensor(
        space=MemorySpace.TMEM,
        dtype=DType.F32,
        shape=(128, N_COLS_TMEM),
        layout=TmemLayout(TmemLayoutKind.LANE_128, col_start=0),
    )
    accum = k.tensor(
        space=MemorySpace.TMEM,
        dtype=DType.F32,
        shape=(128, ACC_DEPTH * MMA_N),
        layout=TmemLayout(TmemLayoutKind.LANE_128, col_start=0),
    )
    sfa_tmem = k.tensor(
        space=MemorySpace.TMEM,
        dtype=DType.U32,
        shape=(128, ACC_DEPTH * SF_CELLS),
        layout=TmemLayout(TmemLayoutKind.LANE_128, col_start=sfa_col0),
    )
    # The full N band's B scales: 256 rows need two TMEM cell-columns per issue
    # (rows 0..127 and 128..255 via the r/128 column advance).
    sfb_tmem = k.tensor(
        space=MemorySpace.TMEM,
        dtype=DType.U32,
        shape=(128, ACC_DEPTH * SF_CELLS * (MMA_N // 128)),
        layout=TmemLayout(TmemLayoutKind.LANE_128, col_start=sfb_col0),
    )

    accum_frag = k.tensor(space=MemorySpace.REG, dtype=DType.F32, shape=(TMEM_LD_SIZE,))
    out_frag = k.tensor(space=MemorySpace.REG, dtype=DType.BF16, shape=(TMEM_LD_SIZE,))
    # The permute partitions each SF buffer's columns across the warp's 32 lanes.
    perm_a_cols = blk_m // 32
    perm_b_cols = MMA_N // 32
    sfa_perm_frag = k.tensor(space=MemorySpace.REG, dtype=DType.U32, shape=(SF_CELLS, perm_a_cols))
    sfb_perm_frag = k.tensor(space=MemorySpace.REG, dtype=DType.U32, shape=(SF_CELLS, perm_b_cols))

    smem_full = k.mbar(kind=MBarKind.TMA, stages=SMEM_DEPTH)
    smem_empty = k.mbar(kind=MBarKind.TCGEN05, stages=SMEM_DEPTH)
    trans_done = k.mbar(kind=MBarKind.THREAD, stages=SMEM_DEPTH)
    tmem_full = k.mbar(kind=MBarKind.TCGEN05, stages=ACC_DEPTH)
    tmem_empty = k.mbar(kind=MBarKind.THREAD, stages=ACC_DEPTH)
    trans_done_leader = k.mbar_ref(trans_done, remote_coord=0)
    tmem_empty_leader = k.mbar_ref(tmem_empty, remote_coord=0)

    cta_id = k.cta_id()
    cta_rank = k.ctaid_in_cluster()
    task_space = k.task_space(grid=(pair_tasks,), fields=("pair_idx",))
    task_scheduler = k.scheduler(task_space)
    task_start = cta_id // cta_group
    task_step = k.launch_cta_count // cta_group

    ab_bytes = a_tile_bytes + b_tile_bytes
    sf_bytes = sfa_tile_bytes + sfb_tile_bytes

    def work_coords(work_idx):
        """ClusterPersistentScheduler2D group-major mapping: rows (M tiles) walk
        within a TILE_GROUPS_ROW_SIZE-row L2 group, groups row-major. Consecutive
        work indices (a cluster pair) land on adjacent M tiles of the same N band.
        Returns (m_idx, n_idx)."""
        if sched_rows <= TILE_GROUPS_ROW_SIZE:
            return work_idx % sched_rows, work_idx // sched_rows
        group_span = TILE_GROUPS_ROW_SIZE * sched_cols
        group_id = work_idx // group_span
        within = work_idx % group_span
        m_idx = group_id * TILE_GROUPS_ROW_SIZE + within % TILE_GROUPS_ROW_SIZE
        n_idx = within // TILE_GROUPS_ROW_SIZE
        return m_idx, n_idx

    with k.kernel_init(warp=0):
        k.tmem_alloc(tmem_base, n_cols=N_COLS_TMEM, cta_group=cta_group)
        for s in range(SMEM_DEPTH):
            k.mbarrier_init(smem_full, count=1, stage=s)
            k.mbarrier_init(smem_empty, count=1, stage=s)
            k.mbarrier_init(trans_done, count=cta_group, stage=s)
        for s in range(ACC_DEPTH):
            k.mbarrier_init(tmem_full, count=1, stage=s)
            k.mbarrier_init(tmem_empty, count=cta_group, stage=s)

    # ---- TMA producer (wg0/warp0) ----
    with k.role(warp=0):
        with k.for_each_task(task_scheduler) as task:
            local_iter = (task.task_id - task_start) // task_step
            work_idx = task.task_id * cta_group + cta_rank
            m_idx, n_idx = work_coords(work_idx)
            a_m = m_idx * CTA_M  # this CTA's own M tile
            b_n = n_idx * MMA_N + cta_rank * CTA_N  # this CTA's half of the N band
            sf_n = n_idx * MMA_N  # the FULL N band's B scales (rank-independent)
            for t in range(k_tiles):
                seq = local_iter * k_tiles + t
                stage = seq % SMEM_DEPTH
                occ = seq // SMEM_DEPTH
                k.mbarrier_wait(smem_empty, stage=stage, phase=(occ + 1) % 2)
                k.mbarrier_arrive_expect_tx(smem_full, bytes=ab_bytes + sf_bytes, stage=stage)
                kb = t * BLK_K_BYTES  # packed-fp4 byte column
                k.tma_load(
                    TensorSlice(
                        tensor=a_smem, offsets=(stage, 0, 0), shape=(1, blk_m, BLK_K_BYTES)
                    ),
                    a_gmem,
                    mbar=smem_full,
                    bytes=a_tile_bytes,
                    coords=(a_m, kb),
                    shape=(1, blk_m, BLK_K_BYTES),
                    gmem_shape=(blk_m, BLK_K_BYTES),
                    mbar_stage=stage,
                )
                k.tma_load(
                    TensorSlice(
                        tensor=b_smem, offsets=(stage, 0, 0), shape=(1, blk_n, BLK_K_BYTES)
                    ),
                    b_gmem,
                    mbar=smem_full,
                    bytes=b_tile_bytes,
                    coords=(b_n, kb),
                    shape=(1, blk_n, BLK_K_BYTES),
                    gmem_shape=(blk_n, BLK_K_BYTES),
                    mbar_stage=stage,
                )
                # SFA: this CTA's M rows; SFB: the full N band. Both load all
                # SF_CELLS issue-cells for this k-tile (gmem rows [t*SF_CELLS, +)).
                k.tma_load(
                    TensorSlice(tensor=sfa_smem, offsets=(stage, 0, 0), shape=(1, SF_CELLS, blk_m)),
                    sfa_gmem,
                    mbar=smem_full,
                    bytes=sfa_tile_bytes,
                    coords=(t * SF_CELLS, a_m),
                    shape=(1, SF_CELLS, blk_m),
                    gmem_shape=(SF_CELLS, blk_m),
                    mbar_stage=stage,
                )
                k.tma_load(
                    TensorSlice(tensor=sfb_smem, offsets=(stage, 0, 0), shape=(1, SF_CELLS, MMA_N)),
                    sfb_gmem,
                    mbar=smem_full,
                    bytes=sfb_tile_bytes,
                    coords=(t * SF_CELLS, sf_n),
                    shape=(1, SF_CELLS, MMA_N),
                    gmem_shape=(SF_CELLS, MMA_N),
                    mbar_stage=stage,
                )

    # ---- scale-factor permute (wg0/warp2) ----
    with k.role(warp=2):
        with k.for_each_task(task_scheduler) as task:
            local_iter = (task.task_id - task_start) // task_step
            for t in range(k_tiles):
                seq = local_iter * k_tiles + t
                stage = seq % SMEM_DEPTH
                occ = seq // SMEM_DEPTH
                k.mbarrier_wait(smem_full, stage=stage, phase=occ % 2)
                # The warp shuffles the packed scale cells into the cp-required
                # physical layout, in place. The byte permutation is below the
                # value model; the read+write of the buffer and the fence are the
                # protocol-relevant part. Each lane owns a contiguous column band
                # of every issue-cell row.
                lane = k.lane_id()
                sfa_slice = TensorSlice(
                    tensor=sfa_smem,
                    offsets=(stage, 0, lane * perm_a_cols),
                    shape=(1, SF_CELLS, perm_a_cols),
                )
                k.reg_load(sfa_perm_frag, sfa_slice)
                k.reg_store(sfa_slice, sfa_perm_frag)
                sfb_slice = TensorSlice(
                    tensor=sfb_smem,
                    offsets=(stage, 0, lane * perm_b_cols),
                    shape=(1, SF_CELLS, perm_b_cols),
                )
                k.reg_load(sfb_perm_frag, sfb_slice)
                k.reg_store(sfb_slice, sfb_perm_frag)
                k.fence(kind=FenceKind.ASYNC_PROXY, scope=FenceScope.CTA)
                k.mbarrier_arrive(trans_done_leader, stage=stage)

    # ---- MMA (wg0/warp1, cluster leader only) ----
    with k.role(warp=1):
        with k.for_each_task(task_scheduler) as task:
            local_iter = (task.task_id - task_start) // task_step
            with k.if_(cta_rank.eq(0)):
                tmem_idx = local_iter % ACC_DEPTH
                k.mbarrier_wait(tmem_empty, stage=tmem_idx, phase=(local_iter // ACC_DEPTH + 1) % 2)
                acc_slice = TensorSlice(
                    tensor=accum, offsets=(0, tmem_idx * MMA_N), shape=(128, MMA_N)
                )
                for t in range(k_tiles):
                    seq = local_iter * k_tiles + t
                    stage = seq % SMEM_DEPTH
                    occ = seq // SMEM_DEPTH
                    k.mbarrier_wait(trans_done, stage=stage, phase=occ % 2)
                    # Copy this k-tile's scale cells SMEM -> TMEM. A: SF_CELLS cells
                    # over this CTA's M rows. B: SF_CELLS issues x 2 N-halves cells.
                    k.tcgen05_cp(
                        TensorSlice(
                            tensor=sfa_tmem, offsets=(0, tmem_idx * SF_CELLS), shape=(128, SF_CELLS)
                        ),
                        TensorSlice(
                            tensor=sfa_smem, offsets=(stage, 0, 0), shape=(1, SF_CELLS, blk_m)
                        ),
                        cta_group=cta_group,
                    )
                    k.tcgen05_cp(
                        TensorSlice(
                            tensor=sfb_tmem,
                            offsets=(0, tmem_idx * SF_CELLS * (MMA_N // 128)),
                            shape=(128, SF_CELLS * (MMA_N // 128)),
                        ),
                        TensorSlice(
                            tensor=sfb_smem, offsets=(stage, 0, 0), shape=(1, SF_CELLS, MMA_N)
                        ),
                        cta_group=cta_group,
                    )
                    for ki in range(K_ITERS):
                        kob = ki * (MMA_K // 2)  # packed-fp4 byte offset for this issue
                        a_op = TensorSlice(
                            tensor=a_smem, offsets=(stage, 0, kob), shape=(1, blk_m, MMA_K // 2)
                        )
                        b_op = TensorSlice(
                            tensor=b_smem, offsets=(stage, 0, kob), shape=(1, blk_n, MMA_K // 2)
                        )
                        sfa_issue = TensorSlice(
                            tensor=sfa_tmem, offsets=(0, tmem_idx * SF_CELLS + ki), shape=(128, 1)
                        )
                        sfb_issue = TensorSlice(
                            tensor=sfb_tmem,
                            offsets=(0, tmem_idx * SF_CELLS * (MMA_N // 128) + ki * (MMA_N // 128)),
                            shape=(128, MMA_N // 128),
                        )
                        k.tcgen05_mma(
                            acc_slice,
                            a_op,
                            b_op,
                            m=MMA_M,
                            n=MMA_N,
                            k=MMA_K,
                            accum=(t > 0 or ki > 0),
                            cta_group=cta_group,
                            sfa=sfa_issue,
                            sfb=sfb_issue,
                            sf_e4m3=True,
                            sf_block=SF_BLOCK,
                            a_fp4=True,
                            b_fp4=True,
                        )
                    k.tcgen05_commit(
                        smem_empty, stage=stage, cta_group=cta_group, multicast_cta_mask=0b11
                    )
                k.tcgen05_commit(
                    tmem_full, stage=tmem_idx, cta_group=cta_group, multicast_cta_mask=0b11
                )

    # ---- epilogue (wg1) ----
    with k.role(warpgroup=1):
        with k.for_each_task(task_scheduler) as task:
            local_iter = (task.task_id - task_start) // task_step
            work_idx = task.task_id * cta_group + cta_rank
            m_idx, n_idx = work_coords(work_idx)
            d_m = m_idx * CTA_M
            d_n = n_idx * MMA_N
            tmem_idx = local_iter % ACC_DEPTH
            k.mbarrier_wait(tmem_full, stage=tmem_idx, phase=(local_iter // ACC_DEPTH) % 2)
            for ot in range(store_tiles):
                store_iter = local_iter * store_tiles + ot
                with k.if_(store_iter >= D_DEPTH):
                    k.cp_async_bulk_wait_group_read(D_DEPTH - 1)
                    k.wg_sync(barrier_id=10)
                d_stage = store_iter % D_DEPTH
                for ki in range(EPI_TILE // TMEM_LD_SIZE):
                    col = tmem_idx * MMA_N + ot * EPI_TILE + ki * TMEM_LD_SIZE
                    k.tcgen05_ld(accum_frag, accum, num=TMEM_LD_SIZE, row=0, col=col)
                    k.tcgen05_wait_ld()
                    # alpha rescale (epilogue global scale), then f32 -> bf16
                    k.reg_mul(accum_frag, accum_frag, config.alpha)
                    k.reg_cvt(out_frag, accum_frag)
                    k.reg_store(
                        TensorSlice(
                            tensor=d_smem,
                            offsets=(d_stage, k.tid_in_wg(), ki * TMEM_LD_SIZE),
                            shape=(1, 1, TMEM_LD_SIZE),
                        ),
                        out_frag,
                    )
                if ot == store_tiles - 1:
                    k.mbarrier_arrive(tmem_empty_leader, stage=tmem_idx)
                k.fence(kind=FenceKind.ASYNC_PROXY, scope=FenceScope.CTA)
                k.wg_sync(barrier_id=10)
                k.tma_store(
                    d_gmem,
                    TensorSlice(tensor=d_smem, offsets=(d_stage, 0, 0), shape=(1, blk_m, EPI_TILE)),
                    coords=(d_m, d_n + ot * EPI_TILE),
                    shape=(1, blk_m, EPI_TILE),
                    gmem_shape=(blk_m, EPI_TILE),
                )
                k.cp_async_bulk_commit_group()
        k.cp_async_bulk_wait_group_read(0)
        k.wg_sync(barrier_id=10)

    with k.kernel_finalize(warp=0):
        k.tmem_dealloc(tmem_base, n_cols=N_COLS_TMEM, cta_group=cta_group)

    return k.build()


def _validate_config(config: NvFp4GemmConfig) -> None:
    for name in ("m", "n", "k"):
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"nvfp4_gemm {name} must be a positive integer")
    if config.k % CTA_K != 0:
        raise ValueError(f"nvfp4_gemm k must be a multiple of {CTA_K}")
    sched_rows = _ceil_div(config.m, CTA_M)
    if sched_rows % CTA_GROUP != 0:
        raise ValueError("nvfp4_gemm requires an even number of M tiles per cluster pair")
    if sched_rows > TILE_GROUPS_ROW_SIZE and sched_rows % TILE_GROUPS_ROW_SIZE != 0:
        raise ValueError("nvfp4_gemm supports tail-only or full-group tilings")


def _validate_launch_shape(launch_shape: LaunchShape, cta_group: int) -> None:
    if not isinstance(launch_shape, tuple) or len(launch_shape) != 1:
        raise ValueError("nvfp4_gemm requires a 1D launch_shape")
    count = launch_shape[0]
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("nvfp4_gemm launch_shape[0] must be a positive integer")
    if count % cta_group != 0:
        raise ValueError("nvfp4_gemm launch_shape[0] must be divisible by cta_group")

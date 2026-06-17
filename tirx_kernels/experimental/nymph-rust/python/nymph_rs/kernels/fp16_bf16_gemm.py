"""Static-shape fp16/bf16 GEMM expressed in clean Nymph IR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..builder import IRBuilder
from ..nymph_rs import (
    DType,
    FenceKind,
    FenceScope,
    Kernel,
    LaunchShape,
    MBarKind,
    MemorySpace,
    SmemSwizzleLayout,
    Swizzle,
    TensorSlice,
    TmemLayout,
    TmemLayoutKind,
)

_ELEMENT_BYTES = {DType.F16: 2, DType.BF16: 2}


@dataclass(frozen=True, slots=True)
class Fp16Bf16GemmConfig:
    m: int = 512
    n: int = 256
    k: int = 64
    dtype: DType = DType.F16
    block_m: int = 128
    block_n: int = 128
    block_k: int = 64
    mma_m: Literal[256] = 256
    mma_n: int = 256
    mma_k: Literal[16] = 16
    cta_group: Literal[1, 2] = 2
    num_warps: int = 12
    epilogue_maxnreg: int = 64
    launch_shape: LaunchShape | None = None


def build_fp16_bf16_gemm(config: Fp16Bf16GemmConfig = Fp16Bf16GemmConfig()) -> Kernel:
    _validate_config(config)
    num_m_tiles = _ceil_div(config.m, config.cta_group * 2 * config.block_m)
    num_n_tiles = _ceil_div(config.n, config.cta_group * config.block_n)
    num_k_tiles = _ceil_div(config.k, config.block_k)
    num_tasks = num_m_tiles * num_n_tiles
    launch_shape = (
        config.launch_shape if config.launch_shape is not None else (num_tasks * config.cta_group,)
    )
    _validate_1d_launch_schedule(launch_shape, config.cta_group)

    elem_bytes = _ELEMENT_BYTES[config.dtype]
    a_tile_bytes = config.block_m * config.block_k * elem_bytes
    b_tile_bytes = config.block_n * config.block_k * elem_bytes
    c_tile_bytes = config.block_m * config.mma_n * elem_bytes
    a_offsets = (0, a_tile_bytes)
    b_offset = 2 * a_tile_bytes
    c_offsets = (b_offset + b_tile_bytes, b_offset + b_tile_bytes + c_tile_bytes)
    smem_size_bytes = 2 * a_tile_bytes + b_tile_bytes + 2 * c_tile_bytes

    k = IRBuilder(
        "nymph_fp16_bf16_gemm",
        num_warps=config.num_warps,
        smem_size_bytes=smem_size_bytes,
        launch_shape=launch_shape,
        cluster_shape=(config.cta_group,),
    )
    a_gmem = k.arg(space=MemorySpace.GMEM, dtype=config.dtype, shape=(config.m, config.k))
    b_gmem = k.arg(space=MemorySpace.GMEM, dtype=config.dtype, shape=(config.n, config.k))
    c_gmem = k.arg(space=MemorySpace.GMEM, dtype=config.dtype, shape=(config.m, config.n))

    smem_layout = SmemSwizzleLayout(Swizzle.B128)
    a_smem_tiles = tuple(
        k.tensor(
            space=MemorySpace.SMEM,
            dtype=config.dtype,
            shape=(config.block_m, config.block_k),
            layout=smem_layout,
            byte_offset=byte_offset,
        )
        for byte_offset in a_offsets
    )
    b_smem = k.tensor(
        space=MemorySpace.SMEM,
        dtype=config.dtype,
        shape=(config.block_n, config.block_k),
        layout=smem_layout,
        byte_offset=b_offset,
    )
    c_smem_tiles = tuple(
        k.tensor(
            space=MemorySpace.SMEM,
            dtype=config.dtype,
            shape=(config.block_m, config.mma_n),
            byte_offset=byte_offset,
        )
        for byte_offset in c_offsets
    )
    # Layout A (m=256, cta_group=2): each consumer's whole (256, mma_n) result is
    # split across the CTA pair by M (128 rows per CTA), so the per-CTA accumulator
    # is (block_m=128 lanes, NUM_CONSUMER * mma_n cols) — consumer c occupies the
    # mma_n-wide column band [c*mma_n, (c+1)*mma_n).
    accum = k.tensor(
        space=MemorySpace.TMEM,
        dtype=DType.F32,
        shape=(config.block_m, 2 * config.mma_n),
        layout=TmemLayout(TmemLayoutKind.LANE_128, col_start=0),
    )
    accum_frag = k.tensor(space=MemorySpace.REG, dtype=DType.F32, shape=(8,))
    out_frag = k.tensor(space=MemorySpace.REG, dtype=config.dtype, shape=(8,))

    # Single-buffer producer/consumer pipeline. Each tensor buffer has a full
    # (producer -> consumer, "data ready") and an empty (consumer -> producer,
    # "buffer free to overwrite") mbarrier; waits use an explicit phase derived
    # from the iteration index so the first use passes without priming. Without
    # the empty handshake the TMA would overrun the single buffer across k-tiles.
    # For cta_group=2 the operand release must come from the MMA pipeline
    # (tcgen05_commit, multicast to both CTAs) — a thread arrive can't stand
    # for the cluster op's reads of BOTH CTAs' tiles.
    empty_kind = MBarKind.TCGEN05 if config.cta_group == 2 else MBarKind.THREAD
    a_full_tiles = tuple(k.mbar(kind=MBarKind.TMA, stages=1) for _ in range(2))
    a_empty_tiles = tuple(k.mbar(kind=empty_kind, stages=1) for _ in range(2))
    b_full = k.mbar(kind=MBarKind.TMA, stages=1)
    b_empty = k.mbar(kind=empty_kind, stages=1)
    mma_done = k.mbar(kind=MBarKind.TCGEN05, stages=1)
    accum_empty = k.mbar(kind=MBarKind.THREAD, stages=1)
    peer_accum_empty = k.mbar_ref(accum_empty, remote_coord=1) if config.cta_group == 2 else None
    # The cluster MMA reads the PEER's operand tiles too: the leader must
    # observe the peer's TMA completions before issuing, or the issue races
    # with the peer's loads.
    if config.cta_group == 2:
        peer_a_full_tiles = tuple(k.mbar_ref(a, remote_coord=1) for a in a_full_tiles)
        peer_b_full = k.mbar_ref(b_full, remote_coord=1)
    else:
        peer_a_full_tiles = None
        peer_b_full = None
    cta_id = k.cta_id()
    cta_local_id = k.ctaid_in_cluster()
    task_start = cta_id // config.cta_group
    task_step = k.launch_cta_count // config.cta_group
    task_space = k.task_space(grid=(num_n_tiles, num_m_tiles), fields=("n_tile", "m_tile"))
    task_scheduler = k.scheduler(task_space)
    a_bytes = a_tile_bytes
    b_bytes = b_tile_bytes
    k_groups_per_tile = config.block_k // config.mma_k
    col_blocks = config.mma_n // 8

    with k.kernel_init(warp=0):
        k.tmem_alloc(accum, n_cols=2 * config.mma_n, cta_group=config.cta_group)
        for a_full in a_full_tiles:
            k.mbarrier_init(a_full, count=1)
        for a_empty in a_empty_tiles:
            k.mbarrier_init(a_empty, count=1)
        k.mbarrier_init(b_full, count=1)
        # B is consumed by both MMA warps, so both free it each k-tile (count=2).
        k.mbarrier_init(b_empty, count=2)
        # Both consumer MMA warps commit to mma_done each task, so one phase
        # completes per task only when both arrivals land (count=2).
        k.mbarrier_init(mma_done, count=2)
        # Both epilogue warpgroups free the accumulator each task (count=2).
        k.mbarrier_init(accum_empty, count=2)

    with k.role(warp=11):
        with k.for_each_task(task_scheduler) as task:
            m_tile = task.m_tile
            n_tile = task.n_tile
            # Barrier phases track THIS stream's local task-stream iteration (each
            # CTA may run a different subset of tasks), not the global task index.
            local_iter = (task.task_id - task_start) // task_step
            for k_tile in range(num_k_tiles):
                k_coord = k_tile * config.block_k
                # Producer waits for the buffer to be free. The empty phase makes
                # the first local iteration pass without priming.
                empty_phase = (local_iter * num_k_tiles + k_tile + 1) % 2
                for consumer in range(2):
                    m_coord = (
                        m_tile * 2 * config.cta_group + cta_local_id + consumer * config.cta_group
                    ) * config.block_m
                    k.mbarrier_wait(a_empty_tiles[consumer], phase=empty_phase)
                    k.mbarrier_arrive_expect_tx(a_full_tiles[consumer], bytes=a_bytes)
                    k.tma_load(
                        a_smem_tiles[consumer],
                        a_gmem,
                        mbar=a_full_tiles[consumer],
                        bytes=a_bytes,
                        coords=(m_coord, k_coord),
                        shape=(config.block_m, config.block_k),
                    )
                k.mbarrier_wait(b_empty, phase=empty_phase)
                k.mbarrier_arrive_expect_tx(b_full, bytes=b_bytes)
                k.tma_load(
                    b_smem,
                    b_gmem,
                    mbar=b_full,
                    bytes=b_bytes,
                    coords=((n_tile * config.cta_group + cta_local_id) * config.block_n, k_coord),
                    shape=(config.block_n, config.block_k),
                )

    for mma_warp, a_smem in enumerate(a_smem_tiles):
        with k.role(warp=8 + mma_warp):
            with k.for_each_task(task_scheduler) as task:
                local_iter = (task.task_id - task_start) // task_step
                # Wait until the previous task's epilogue has drained the
                # accumulator before overwriting it. (iter+1) phase makes the
                # first local iteration pass immediately (no prior epilogue).
                accum_empty_phase = (local_iter + 1) % 2
                k.mbarrier_wait(accum_empty, phase=accum_empty_phase)
                if peer_accum_empty is not None:
                    with k.if_(cta_local_id.eq(0)):
                        k.mbarrier_wait(peer_accum_empty, phase=accum_empty_phase)
                first_mma = True
                for _k_tile in range(num_k_tiles):
                    full_phase = (local_iter * num_k_tiles + _k_tile) % 2
                    k.mbarrier_wait(a_full_tiles[mma_warp], phase=full_phase)
                    k.mbarrier_wait(b_full, phase=full_phase)
                    if peer_a_full_tiles is not None:
                        with k.if_(cta_local_id.eq(0)):
                            k.mbarrier_wait(peer_a_full_tiles[mma_warp], phase=full_phase)
                            k.mbarrier_wait(peer_b_full, phase=full_phase)
                    for k_group in range(k_groups_per_tile):
                        k_offset = k_group * config.mma_k
                        n_offset = mma_warp * config.mma_n
                        k.tcgen05_mma(
                            accum[:, n_offset : n_offset + config.mma_n],
                            a_smem[:, k_offset : k_offset + config.mma_k],
                            b_smem[:, k_offset : k_offset + config.mma_k],
                            m=config.mma_m,
                            n=config.mma_n,
                            k=config.mma_k,
                            accum=not first_mma,
                            cta_group=config.cta_group,
                        )
                        first_mma = False
                    # Free the A/B buffers for the producer's next k-tile: a
                    # tcgen05_commit from the MMA issuer (the pipeline's proof
                    # the tiles were read; a thread arrive right after the
                    # issue would release them while the engine may still be
                    # reading), multicast to both CTAs for cta_group=2.
                    if config.cta_group == 2:
                        with k.if_(cta_local_id.eq(0)):
                            k.tcgen05_commit(
                                a_empty_tiles[mma_warp],
                                cta_group=config.cta_group,
                                multicast_cta_mask=0b11,
                            )
                            k.tcgen05_commit(
                                b_empty, cta_group=config.cta_group, multicast_cta_mask=0b11
                            )
                    else:
                        k.tcgen05_commit(a_empty_tiles[mma_warp], cta_group=1)
                        k.tcgen05_commit(b_empty, cta_group=1)
                # The cta_group=2 MMA is a single cluster op the leader (even CTA)
                # drives — it computes the whole 256xN product and writes BOTH
                # CTAs' accumulator halves. Only the leader commits, multicasting
                # mma_done to both CTAs so each epilogue is gated until its half is
                # written; a per-CTA commit would let the odd epilogue read its
                # half before the leader wrote it.
                with k.if_(cta_local_id.eq(0)):
                    k.tcgen05_commit(mma_done, cta_group=config.cta_group, multicast_cta_mask=0b11)

    for epilogue_wg, c_smem in enumerate(c_smem_tiles):
        with k.role(warpgroup=epilogue_wg, maxnreg=config.epilogue_maxnreg):
            with k.for_each_task(task_scheduler) as task:
                m_tile = task.m_tile
                n_tile = task.n_tile
                local_iter = (task.task_id - task_start) // task_step
                k.mbarrier_wait(mma_done, phase=local_iter % 2)
                # The previous task's bulk store still READS c_smem until its
                # group drains — wait before overwriting (guarded off the
                # first task, where no group exists yet).
                with k.if_(local_iter >= 1):
                    k.cp_async_bulk_wait_group_read(0)
                with k.for_loop(stop=col_blocks) as col_block:
                    col = col_block * 8
                    tmem_col = epilogue_wg * config.mma_n + col
                    # One warp-collective ld spreads the 128 accumulator rows across
                    # the warpgroup's 128 threads (row=0 is the taddr-base corner):
                    # thread tid_in_wg loads row tid_in_wg, columns tmem_col..+7.
                    k.tcgen05_ld(accum_frag, accum, num=8, row=0, col=tmem_col)
                    k.tcgen05_wait_ld()
                    k.reg_cvt(out_frag, accum_frag)
                    k.reg_store(
                        TensorSlice(tensor=c_smem, offsets=(k.tid_in_wg(), col), shape=(1, 8)),
                        out_frag,
                    )
                # The accumulator is fully drained; let the next task's MMA reuse it.
                k.mbarrier_arrive(accum_empty)
                k.fence(kind=FenceKind.ASYNC_PROXY, scope=FenceScope.CTA)
                k.tma_store(
                    c_gmem,
                    c_smem,
                    coords=(
                        (
                            m_tile * 2 * config.cta_group
                            + cta_local_id
                            + epilogue_wg * config.cta_group
                        )
                        * config.block_m,
                        # Layout A keeps the whole N band in both CTAs (D is split by
                        # M), so the column origin is the tile's N, independent of cbx.
                        n_tile * config.mma_n,
                    ),
                    shape=(config.block_m, config.mma_n),
                )
                k.cp_async_bulk_commit_group()

    with k.kernel_finalize(warp=0):
        k.tmem_dealloc(accum, n_cols=2 * config.mma_n, cta_group=config.cta_group)

    return k.build()


def _validate_config(config: Fp16Bf16GemmConfig) -> None:
    if config.dtype not in _ELEMENT_BYTES:
        raise ValueError("fp16_bf16_gemm dtype must be f16 or bf16")
    for name in ("m", "n", "k", "block_m", "block_n", "block_k", "mma_m", "mma_n"):
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"fp16_bf16_gemm {name} must be a positive integer")
    if config.mma_k != 16:
        raise ValueError("fp16_bf16_gemm mma_k must be 16")
    if config.cta_group not in (1, 2):
        raise ValueError("fp16_bf16_gemm cta_group must be 1 or 2")
    # cta_group=2 distributes the MMA across the pair: each CTA supplies m/cta_group
    # rows of A and n/cta_group rows of B, which is exactly one SMEM tile.
    if config.block_m != config.mma_m // config.cta_group:
        raise ValueError("fp16_bf16_gemm block_m must equal mma_m // cta_group")
    if config.block_n != config.mma_n // config.cta_group:
        raise ValueError("fp16_bf16_gemm block_n must equal mma_n // cta_group")
    if config.block_n % 8 != 0:
        raise ValueError("fp16_bf16_gemm block_n must be a multiple of 8")
    if config.block_k % config.mma_k != 0:
        raise ValueError("fp16_bf16_gemm block_k must be a multiple of mma_k")
    if config.m % (2 * config.cta_group * config.block_m) != 0:
        raise ValueError("fp16_bf16_gemm m must be divisible by 2 * cta_group * block_m")
    if config.n % (config.cta_group * config.block_n) != 0:
        raise ValueError("fp16_bf16_gemm n must be divisible by cta_group * block_n")
    if config.k % config.block_k != 0:
        raise ValueError("fp16_bf16_gemm k must be divisible by block_k")
    if config.num_warps < 12:
        raise ValueError("fp16_bf16_gemm num_warps must be at least 12")
    for name in ("epilogue_maxnreg",):
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1 or value % 8 != 0:
            raise ValueError(f"fp16_bf16_gemm {name} must be a positive multiple of 8")


def _validate_1d_launch_schedule(launch_shape: LaunchShape, cta_group: int) -> None:
    """Validate the 1D persistent task schedule used by this GEMM."""

    if not isinstance(launch_shape, tuple) or len(launch_shape) != 1:
        raise ValueError("fp16_bf16_gemm requires a 1D launch_shape")
    launch_cta_count = launch_shape[0]
    if (
        not isinstance(launch_cta_count, int)
        or isinstance(launch_cta_count, bool)
        or launch_cta_count < 1
    ):
        raise ValueError("fp16_bf16_gemm launch_shape[0] must be a positive integer")
    if launch_cta_count % cta_group != 0:
        raise ValueError("fp16_bf16_gemm launch_shape[0] must be divisible by cta_group")


def _ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs

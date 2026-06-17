"""Chunked Gated Delta Net (GDN) prefill expressed in Nymph IR.

Instruction-faithful port of the FlashInfer Blackwell SM100 CuteDSL chunked GDN
prefill kernel (``~/flashinfer/.../blackwell/gated_delta_net_chunked.py``),
reproducing its exact 12-warp specialization and per-``cute.gemm``/``cute.copy``
instruction selection (PTX/SASS evidence + per-call-site table in
``docs/kernels/gdn_prefill/INSTR.md``). 12 warps (gated_delta_net_chunked.py:232-239):
CG0=0-3 (T-pairwise, kk_epi, qk_epi, inverse), CG1=4-7 (new_v, kv_update, qkv),
MMA=8, TMA=9, gate/beta=10, epilogue=11. The 7 GEMMs lower to
``tcgen05.mma.cta_group::1.kind::f16`` (UTCHMMA); M=64 read-backs to
``tcgen05.ld.16x256b`` (the CUTLASS scatter datapath, mirrored cell-for-cell by
nymph's ``tcgen05_ld(shape="16x256b")``) and M=128 to ``.32x32b``; bulk transfers
to TMA (UTMALDG/UTMASTG).

The **WY inverse is the flashinfer hierarchical blockwise inverse using the
warp-level ``mma.sync`` (SM80 HMMA, the ``mma_sync`` IR node) + ``ldmatrix``** —
GJ-invert the 8 diagonal 8×8 blocks (forward substitution), then merge 8→16→32→64
filling each lower off-diagonal ``newC = -Qinv·C·Pinv`` via two ``mma_sync`` +
``ldmatrix``-loaded operands, ``stmatrix``-stored result. The 7-GEMM epilogue
operand staging (readback→SMEM) uses ``stmatrix`` (STSM) — each ``.16x256b``
tile is the mma m8n8 fragment. GEMM3/4 read the recurrent state ``S`` **directly
from TMEM** as the ``B`` operand (``S^T`` via ``trans_b``), no SMEM copy — nymph's
``tcgen05_mma`` accepts an f32 TMEM operand. See ``docs/kernels/gdn_prefill/INSTR.md``.

Algorithm = FLA ``chunk_gated_delta_rule`` fwd (validated cell-for-cell against
the recurrent reference, ``tests/kernels/_gdn_oracle.py``). Per chunk (BT=64):
the gate warp applies ``log2`` to the raw gate then an inclusive warp prefix-sum
(flashinfer parity), so gcs=Σ log2(gate) and T[i,j]=exp2(gcs[i]-gcs[j])=Π gate;
the 7 GEMMs and the gating/inverse/epilogue glue (see ``docs/kernels/gdn_prefill/INSTR.md``
and the project memory for the full op map). State S[K,V] carried in TMEM.
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

HQK = 4
HV = 8
GVA_RATIO = HV // HQK
K_DIM = 128
V_DIM = 128
BT = 64
SM_COUNT = 148
NUM_THREADS = 128
N_COLS_TMEM = 512
LD_NUM = 8  # tcgen05.ld.16x256b.x8 -> 32 regs (4*num)

# FlashInfer 12-warp specialization (gated_delta_net_chunked.py:232-239).
CG0_WARPS = (0, 1, 2, 3)  # compute group 0: T-pairwise, kk_epi, qk_epi, WY inverse
CG1_WARPS = (4, 5, 6, 7)  # compute group 1: new_v_epi, kv_update_epi, qkv_epilogue
MMA_WARP = 8  # issues all 7 tcgen05 GEMMs
TMA_WARP = 9  # TMA-load q/k/v
GATE_WARP = 10  # load gate/beta + log2 + warp prefix-sum cumsum
EPI_WARP = 11  # output store
NUM_WARPS = 12


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _sl(t, offs, shape):
    return TensorSlice(tensor=t, offsets=offs, shape=shape)


@dataclass(frozen=True, slots=True)
class GdnPrefillConfig:
    num_seqs: int = 1
    seqlen: int = 128  # fixed-length: per-seq length (varlen=False). varlen: max for sizing.
    scale: float = 1.0 / (K_DIM**0.5)
    launch_shape: LaunchShape | None = None
    varlen: bool = False  # True: cu_seqlens arg, runtime per-tile chunk count, OOB-masked tails
    # head config (flashinfer parametrizes these): num_k_heads = min(num_q,num_v) is derived
    # (k/v paired into the kv head); max(num_q,num_v) must be a multiple of the min — both
    # GVA (num_v≥num_q) and GQA (num_q>num_v) are supported. head_dim is fixed at 128 /
    # block_size BT=64 (the only flashinfer-tested values, baked into the TMEM/readback).
    num_q_heads: int = 4
    num_v_heads: int = 8
    # io dtype: "bfloat16" or "float16" (flashinfer tests both). Drives every 16-bit
    # operand (q/k/v/out, the GEMM operands, the WY-inverse mma.sync); accumulators stay
    # f32. mma.sync is first-class for both via its ab_dtype.
    io_dtype: str = "bfloat16"


def _io_dt(config: GdnPrefillConfig) -> DType:
    return DType.F16 if config.io_dtype == "float16" else DType.BF16


def gdn_prefill_task_config(num_seqs: int, seqlen: int) -> GdnPrefillConfig:
    return GdnPrefillConfig(num_seqs=num_seqs, seqlen=seqlen)


# Fixed-length workload coverage: chunk counts 1/2/4/8/16/32 (seqlen = BT·n_chunks)
# crossed with batch sizes, including num_work = num_seqs·HV > SM_COUNT so the
# persistent grid-stride runs MULTIPLE tiles per CTA. The per-chunk barriers/buffers
# carry across tiles via cumulative pipeline-state counters (gc / gc_pos = CUTLASS
# PipelineState, never per-tile reset), with the chunk_free handoff spanning tile
# boundaries. Every shape builds + protocol-checks; the value sweep (CONFIGS_SUPPORTED)
# runs the cheaper subset cell-exact (incl. multi-tile/CTA with odd chunks/tile).
CONFIGS = [
    {"num_seqs": ns, "seqlen": T, "label": f"ns{ns}_t{T}"}
    for ns, T in [
        (1, 64),
        (1, 128),
        (1, 192),
        (1, 256),
        (1, 512),
        (1, 1024),
        (1, 2048),
        (2, 128),
        (2, 256),
        (2, 512),
        (4, 128),
        (4, 256),
        (8, 64),
        (20, 64),
        (20, 192),
        (32, 128),
        (48, 64),  # num_work > SM_COUNT: multi-tile/CTA
    ]
]

# Cheaper fixed-length shapes for the cell-exact value sweep: 1–4 chunks, small batch,
# plus multi-tile/CTA (num_work > SM_COUNT) with odd chunks/tile (the phase-carry case).
CONFIGS_SUPPORTED = [
    {"num_seqs": ns, "seqlen": T, "label": f"ns{ns}_t{T}"}
    for ns, T in [
        (1, 64),
        (1, 128),
        (1, 192),
        (1, 256),
        (2, 128),
        (2, 256),
        (4, 128),
        (20, 64),
        (20, 192),
    ]
]

# Head configs = flashinfer test_prefill_delta_rule.py num_q/k/v_heads (head_dim=128).
# num_k_heads = min(num_q,num_v) is derived (k/v paired into the kv head). Listed as
# (num_q_heads, num_v_heads): GVA (num_v≥num_q) AND GQA (num_q>num_v) — the kernel
# iterates max(num_q,num_v) effective heads. num_work = num_eff·num_seqs may exceed
# SM_COUNT (multi-tile). Covers ALL 8 flashinfer configs: (1,1,1)(4,1,1)(3,3,3)(6,2,2)
# (1,1,2)(2,2,4)(16,16,32)(16,16,64) -> (num_q,num_v) below + default (4,8).
HEAD_CONFIGS = [
    {"num_q_heads": q, "num_v_heads": v, "label": f"h{q}q{v}v"}
    for q, v in [(1, 1), (4, 1), (3, 3), (6, 2), (1, 2), (2, 4), (16, 32), (16, 64), (4, 8)]
]

# Varlen (cu_seqlens) batches: mixed lengths, non-BT-multiple tails, sub-BT and
# multi-chunk sequences, packed back-to-back.
VARLEN_CONFIGS = [
    {"seqlens": s, "label": "v_" + "_".join(map(str, s))}
    for s in [
        [64, 128],
        [128, 64, 192],
        [64, 64],
        [70, 130],
        [6, 100],
        [200],
        [13, 64, 191],
        [256, 320],
        [64, 64, 64, 64],
        [320, 7, 100],
    ]
] + [
    # multi-tile/CTA varlen (num_work = num_seqs·HV > SM_COUNT): mixed + non-BT-multiple.
    {
        "seqlens": [
            70,
            64,
            130,
            6,
            192,
            100,
            64,
            128,
            7,
            200,
            64,
            191,
            64,
            256,
            13,
            64,
            128,
            320,
            100,
            70,
        ],
        "label": "v_mt20",
    }
]


def build_gdn_prefill(config: GdnPrefillConfig = GdnPrefillConfig()) -> Kernel:
    NS, T = config.num_seqs, config.seqlen
    _validate_config(config)
    # Head model (flashinfer gated_delta_net_chunked.py:405-472): iterate NEFF =
    # max(h_q,h_v) effective heads. gate/beta/out/state are per effective head `eh`; the
    # state recurrence is per eh. Only the q/k/v LOADS use shared-vs-unique heads — k/v are
    # paired into the kv head HK = min(h_q,h_v), with HR = NEFF//HK heads per group:
    #   GVA (h_v≥h_q): q,k load eh//HR (shared);  v loads eh (unique)
    #   GQA (h_q>h_v): q loads eh (unique);        k,v load eh//HR (shared)
    H_Q, H_V = config.num_q_heads, config.num_v_heads
    HK = min(H_Q, H_V)  # num_k_heads (k/v are paired into the kv head)
    NEFF = max(H_Q, H_V)  # effective / output heads (= num_o_heads)
    HR = NEFF // HK  # heads per kv group
    IS_GQA = H_Q > H_V
    iod = _io_dt(config)  # io / 16-bit-operand dtype (bf16 or f16)
    n_chunks = _ceil_div(T, BT)
    total_t = NS * T
    num_work = NS * NEFF
    launch_shape = config.launch_shape or (min(SM_COUNT, num_work),)

    sizes = dict(
        k=2 * BT * K_DIM * 2,
        q=BT * K_DIM * 2,
        v=BT * V_DIM * 2,  # k double-buffered (2 stages)
        vnewt=V_DIM * BT * 2,
        attn=BT * BT * 2,
        tmpt=V_DIM * BT * 2,
        out=BT * V_DIM * 2,
        m=BT * BT * 4,
        gcs=BT * 4,
        beta=BT * 4,
        dcs=BT * BT * 2,  # DC scratch for the hierarchical-inverse merges (mma.sync)
    )
    off, offs = 0, {}
    for name, nbytes in sizes.items():
        offs[name] = off
        off += nbytes
    # A_inv aliases NV (vnewt): flashinfer's sAinv holds A_inv then overwrites with NV.
    # Disjoint lifetimes here — A_inv (CG0 inverse → GEMM5 A operand) is dead before
    # _read128_vnew writes NV into vnewt; ainv (8KB) fits in vnewt (16KB).
    offs["ainv"] = offs["vnewt"]

    k = IRBuilder(
        "nymph_gdn_prefill", num_warps=NUM_WARPS, smem_size_bytes=off, launch_shape=launch_shape
    )

    q_g = k.arg(space=MemorySpace.GMEM, dtype=iod, shape=(total_t, H_Q, K_DIM))
    k_g = k.arg(space=MemorySpace.GMEM, dtype=iod, shape=(total_t, HK, K_DIM))
    v_g = k.arg(space=MemorySpace.GMEM, dtype=iod, shape=(total_t, H_V, V_DIM))
    gate_g = k.arg(space=MemorySpace.GMEM, dtype=DType.F32, shape=(total_t, NEFF))
    beta_g = k.arg(space=MemorySpace.GMEM, dtype=DType.F32, shape=(total_t, NEFF))
    out_g = k.arg(space=MemorySpace.GMEM, dtype=iod, shape=(total_t, NEFF, V_DIM))
    state_g = k.arg(space=MemorySpace.GMEM, dtype=DType.F32, shape=(NS, NEFF, K_DIM, V_DIM))
    # varlen: cu_seqlens[NS+1] gives each sequence's token range; num_chunks varies per tile.
    cu_g = (
        k.arg(space=MemorySpace.GMEM, dtype=DType.I32, shape=(NS + 1,)) if config.varlen else None
    )

    def task_geom(task):
        # per-tile (seq, head) geometry: token base + chunk count + the q/k/v load heads.
        # eh = effective head (gate/beta/out/state); shared = eh//HR = the kv-group head.
        # For varlen tok_base/chunk-count are RUNTIME (k.scalar from cu_seqlens, usable as
        # the for_loop bound); for fixed-length they are compile-time.
        work = task.field("work")
        seq = work // NEFF
        eh = work % NEFF
        shared = eh // HR
        q_head = eh if IS_GQA else shared
        k_head = shared
        v_head = shared if IS_GQA else eh
        if config.varlen:
            base = k.scalar(initial=_sl(cu_g, (seq,), (1,)))
            nxt = k.scalar(initial=_sl(cu_g, (seq + 1,), (1,)))
            slen = nxt - base  # seqlen_b (runtime)
            nch = (slen + (BT - 1)) // BT  # ceil_div(seqlen_b, BT)
            return seq, eh, q_head, k_head, v_head, base, nch, slen
        return seq, eh, q_head, k_head, v_head, seq * config.seqlen, n_chunks, config.seqlen

    def sm(name, dt, shape):
        return k.tensor(space=MemorySpace.SMEM, dtype=dt, shape=shape, byte_offset=offs[name])

    # K double-buffered: single [2*BT, K_DIM] buffer; stage = (c%2)*BT row offset
    # (compile-time for fixed-length, runtime for varlen).
    k_s = sm("k", iod, (2 * BT, K_DIM))
    q_s = sm("q", iod, (BT, K_DIM))
    v_s = sm("v", iod, (BT, V_DIM))
    vnewt_s = sm("vnewt", iod, (V_DIM, BT))
    ainv_s = sm("ainv", iod, (BT, BT))
    attn_s = sm("attn", iod, (BT, BT))
    tmpt_s = sm("tmpt", iod, (V_DIM, BT))
    out_s = sm("out", iod, (BT, V_DIM))
    m_s = sm("m", DType.F32, (BT, BT))
    dcs_s = sm("dcs", iod, (BT, BT))  # hierarchical-inverse DC scratch
    gcs_s = sm("gcs", DType.F32, (BT,))
    beta_s = sm("beta", DType.F32, (BT,))

    # TMEM regions matching flashinfer's separate allocations (gated_delta_net_chunked.py
    # :319-330): state S (f32), state_inp (fp16 undecayed-S copy for GEMM3/4), q_state acc
    # (GEMM4/6), shared_acc (GEMM1/2/3/5), shared_inp (fp16 delta/NV for GEMM5/6/7).
    tmem_base = k.tensor(
        space=MemorySpace.TMEM,
        dtype=DType.F32,
        shape=(128, N_COLS_TMEM),
        layout=TmemLayout(TmemLayoutKind.LANE_128, col_start=0),
    )
    s_tmem = k.tensor(
        space=MemorySpace.TMEM,
        dtype=DType.F32,
        shape=(128, V_DIM),
        layout=TmemLayout(TmemLayoutKind.LANE_128, col_start=0),
    )  # 0-127
    state_inp = k.tensor(
        space=MemorySpace.TMEM,
        dtype=iod,
        shape=(128, V_DIM),
        layout=TmemLayout(TmemLayoutKind.LANE_128, col_start=128),
    )  # 128-191 (64 cols bf16)
    qstate_tmem = k.tensor(
        space=MemorySpace.TMEM,
        dtype=DType.F32,
        shape=(64, V_DIM),
        layout=TmemLayout(TmemLayoutKind.LANE_64_UPPER, col_start=192),
    )  # 192-319
    # shared_acc = 2 stages of 64 cols (flashinfer 64-col × 2): kk→stage0, qk→stage1
    # (pipeline); the V-output GEMMs (ks/nv, [BT,V_DIM]) use the contiguous union view.
    acc_s0 = k.tensor(
        space=MemorySpace.TMEM,
        dtype=DType.F32,
        shape=(64, BT),
        layout=TmemLayout(TmemLayoutKind.LANE_64_UPPER, col_start=320),
    )  # stage 0: 320-383
    acc_s1 = k.tensor(
        space=MemorySpace.TMEM,
        dtype=DType.F32,
        shape=(64, BT),
        layout=TmemLayout(TmemLayoutKind.LANE_64_UPPER, col_start=384),
    )  # stage 1: 384-447
    acc_tmem = k.tensor(
        space=MemorySpace.TMEM,
        dtype=DType.F32,
        shape=(64, V_DIM),
        layout=TmemLayout(TmemLayoutKind.LANE_64_UPPER, col_start=320),
    )  # union view (ks/nv)
    shared_inp = k.tensor(
        space=MemorySpace.TMEM,
        dtype=iod,
        shape=(V_DIM, BT),
        layout=TmemLayout(TmemLayoutKind.LANE_128, col_start=448),
    )  # stage A: delta(G5)→NV(G6)
    shared_inp_b = k.tensor(
        space=MemorySpace.TMEM,
        dtype=iod,
        shape=(V_DIM, BT),
        layout=TmemLayout(TmemLayoutKind.LANE_128, col_start=480),
    )  # stage B: vnew_gated(G7)

    def reg(dt, shape):
        return k.tensor(space=MemorySpace.REG, dtype=dt, shape=shape)

    r1 = reg(DType.F32, (1,))
    r2 = reg(DType.F32, (1,))
    r3 = reg(DType.F32, (1,))
    racc = reg(DType.F32, (1,))
    rb16 = reg(iod, (1,))
    rb16b = reg(iod, (1,))
    glast = reg(DType.F32, (1,))  # gcs[BT-1] (chunk's total log-decay)
    frag = reg(DType.F32, (32,))  # .16x256b.x8 readback = 32 regs
    frag32 = reg(DType.F32, (64,))  # .32x32b readback (state, thread=row, 64 cols)
    zrow = reg(DType.F32, (V_DIM,))
    # hierarchical-inverse (mma.sync) fragments
    imm_a = reg(DType.U32, (2,))  # A = Qinv (m16k8 packed-bf16, m-broadcast)
    imm_b = reg(DType.U32, (1,))  # B = C / Pinv
    imm_acc = reg(DType.F32, (4,))  # mma accumulator
    imm_accb = reg(iod, (4,))  # accumulator -> bf16 (store / next-A)
    sttile = reg(iod, (2,))  # one m8n8 tile (2 bf16 = 1 stmatrix word) for reg->SMEM staging
    sttile2 = reg(iod, (2,))  # second tile (vnew gated/ungated emit two stmatrix tiles)
    sinp_reg = reg(iod, (128,))  # state_inp bf16 staging fragment (128 halves = 64 cells)

    CG0_T = 1  # mbarrier_arrive is one arrival per producer cohort (NOT per-thread)
    CG1_T = 1
    GATE_T = 1
    # (barrier name -> (kind, arrival count)) for the 12-warp producer/consumer pipeline.
    KSTAGES = 2  # K is double-buffered (flashinfer smem_k_stages=2): prefetch next chunk
    bar_spec = {
        "tk": (MBarKind.TMA, 1, KSTAGES),  # K full (2 stages)
        "k_free": (MBarKind.THREAD, 1, KSTAGES),  # K empty (MMA frees stage after GEMM7)
        "tq": (MBarKind.TMA, 1),
        "tv": (MBarKind.TMA, 1),
        "tg": (MBarKind.TMA, 1),
        "tb": (MBarKind.TMA, 1),
        "d_kk": (MBarKind.TCGEN05, 1),
        "d_qk": (MBarKind.TCGEN05, 1),
        "d_ks": (MBarKind.TCGEN05, 1),
        "d_qs": (MBarKind.TCGEN05, 1),
        "d_nv": (MBarKind.TCGEN05, 1),
        "d_oi": (MBarKind.TCGEN05, 1),
        "d_ds": (MBarKind.TCGEN05, 1),
        "gate_ready0": (MBarKind.THREAD, GATE_T),
        "gate_ready1": (MBarKind.THREAD, GATE_T),  # gate->CG0 / gate->CG1
        "ainv_ready": (MBarKind.THREAD, CG0_T),  # CG0 -> MMA
        "qkv_ready": (MBarKind.THREAD, CG0_T),  # CG0 -> MMA
        "sT_ready": (MBarKind.THREAD, CG1_T),  # CG1 -> MMA (GEMM3/4)
        "delta_ready": (MBarKind.THREAD, CG1_T),  # CG1 -> MMA (GEMM5)
        "vnew_ready": (MBarKind.THREAD, CG1_T),  # CG1 -> MMA (GEMM6)
        "ktvng_ready": (MBarKind.THREAD, CG1_T),  # CG1 -> MMA (GEMM7)
        "o_ready": (MBarKind.THREAD, CG1_T),  # CG1 -> epilogue(11)
        # per-GEMM acc-free (reader -> MMA, so the shared acc_tmem can be reused)
        "f_kk": (MBarKind.THREAD, CG0_T),
        "f_qk": (MBarKind.THREAD, CG0_T),
        "f_ks": (MBarKind.THREAD, CG1_T),
        "f_qs": (MBarKind.THREAD, CG1_T),
        "f_nv": (MBarKind.THREAD, CG1_T),
        "f_oi": (MBarKind.THREAD, CG1_T),
        "chunk_free": (MBarKind.THREAD, CG1_T + 1),  # CG1 + epilogue (both free chunk-c buffers)
    }
    bars = {
        nm: k.mbar(kind=spec[0], stages=(spec[2] if len(spec) > 2 else 1))
        for nm, spec in bar_spec.items()
    }

    sched = k.scheduler(k.task_space(grid=(num_work,), fields=("work",)))

    with k.kernel_init(warp=0):
        k.tmem_alloc(tmem_base, n_cols=N_COLS_TMEM)
        for nm, spec in bar_spec.items():
            stg = spec[2] if len(spec) > 2 else 1
            for s in range(stg):
                k.mbarrier_init(bars[nm], count=spec[1], stage=s)

    _emit(
        k,
        config,
        n_chunks,
        sched,
        task_geom,
        (q_g, k_g, v_g, gate_g, beta_g, out_g, state_g),
        (k_s, q_s, v_s, vnewt_s, ainv_s, attn_s, tmpt_s, out_s, m_s, gcs_s, beta_s, dcs_s),
        (s_tmem, state_inp, qstate_tmem, acc_tmem, acc_s0, acc_s1, shared_inp, shared_inp_b),
        (
            r1,
            r2,
            r3,
            racc,
            rb16,
            rb16b,
            glast,
            frag,
            frag32,
            zrow,
            imm_a,
            imm_b,
            imm_acc,
            imm_accb,
            sttile,
            sttile2,
            sinp_reg,
        ),
        bars,
    )

    with k.kernel_finalize(warp=0):
        k.tmem_dealloc(tmem_base, n_cols=N_COLS_TMEM)
    return k.build()


def _emit(k, config, n_chunks, sched, task_geom, args, sm, tm, rg, bars):
    NEFF = max(config.num_q_heads, config.num_v_heads)  # final-state store: per effective head
    iod = _io_dt(config)  # io / 16-bit-operand dtype (bf16 or f16)
    q_g, k_g, v_g, gate_g, beta_g, out_g, state_g = args
    (k_s, q_s, v_s, vnewt_s, ainv_s, attn_s, tmpt_s, out_s, m_s, gcs_s, beta_s, dcs_s) = sm
    s_tmem, state_inp, qstate_tmem, acc_tmem, acc_s0, acc_s1, shared_inp, shared_inp_b = tm
    (
        r1,
        r2,
        r3,
        racc,
        rb16,
        rb16b,
        glast,
        frag,
        frag32,
        zrow,
        imm_a,
        imm_b,
        imm_acc,
        imm_accb,
        sttile,
        sttile2,
        sinp_reg,
    ) = rg
    scale = config.scale

    def ph(c):
        return c % 2  # single-buffered barrier phase (chunk loop is Python-unrolled)

    def fence_pub(bid):  # publish generic SMEM writes to the MMA's async proxy
        k.fence(kind=FenceKind.ASYNC_PROXY, scope=FenceScope.CTA)
        k.wg_sync(barrier_id=bid)

    def issue(
        dst, a, b, m, n, kk, done, accum0=False, trans_a=False, trans_b=False, a_row0=0, b_row0=0
    ):
        # cute.gemm -> tcgen05.mma.cta_group::1.kind::f16 (UTCHMMA); MMA warp issues+commits.
        # trans_a/trans_b read the operand transposed (slice [k, m] / [k, n]): GEMM3/4
        # take the TMEM state S as B=S^T; GEMM7 takes k_s as A=Kᵀ via the MMA transpose.
        # a_row0/b_row0 = base row offset into the operand tensor (for the K double-buffer
        # stage = (c%2)*BT, which is compile-time for fixed-length and runtime for varlen).
        for g in range(kk // 16):
            a_sl = (
                _sl(a, (a_row0 + g * 16, 0), (16, m))
                if trans_a
                else _sl(a, (a_row0, g * 16), (m, 16))
            )
            b_sl = (
                _sl(b, (b_row0 + g * 16, 0), (16, n))
                if trans_b
                else _sl(b, (b_row0, g * 16), (n, 16))
            )
            k.tcgen05_mma(
                dst,
                a_sl,
                b_sl,
                m=m,
                n=n,
                k=16,
                accum=(accum0 or g != 0),
                trans_a=trans_a,
                trans_b=trans_b,
            )
        k.tcgen05_commit(bars[done])

    def rss(dst_s, fac_neg_beta, mask_strict, tid, lane, warp, dst_bf16=False, acc=None):
        # kk_epi / qk_epi (CG0): read M=64 acc via tcgen05.ld.16x256b, scale by the
        # T-pairwise factor exp2(gcs[i]-gcs[j]) and (-beta[i] | scale), mask, store.
        k.tcgen05_ld(
            frag, acc if acc is not None else acc_tmem, shape="16x256b", num=LD_NUM, row=0, col=0
        )
        k.tcgen05_wait_ld()
        for va in range(2):
            row = (lane // 4) + 8 * va + 16 * warp
            for vb in range(8):
                for v0p in range(2):
                    r = v0p + 2 * va + 4 * vb
                    col = v0p + 2 * (lane % 4) + 8 * vb
                    k.reg_load(r1, _sl(gcs_s, (row,), (1,)))
                    k.reg_load(r2, _sl(gcs_s, (col,), (1,)))
                    k.reg_sub(r1, r1, r2)
                    k.reg_unary(r1, r1, op="exp2")  # T=exp2(gcs[i]-gcs[j]) (log2-units)
                    k.reg_mul(r1, r1, _sl(frag, (r,), (1,)))
                    if fac_neg_beta:
                        k.reg_load(r2, _sl(beta_s, (row,), (1,)))
                        k.reg_mul(r2, r2, -1.0)
                        k.reg_mul(r1, r1, r2)
                    else:
                        k.reg_mul(r1, r1, float(scale))
                    if dst_bf16:
                        # W_qkv (attn_s) is a GEMM6 operand → stage the m8n8 tile via
                        # stmatrix (STSM), matching flashinfer's qk_epi r2s store.
                        k.reg_cvt(_sl(sttile, (v0p,), (1,)), r1)
                        if v0p == 1:
                            _stm(k, dst_s, 8 * va + 16 * warp, 8 * vb, sttile, lane, trans=False)
                    else:
                        k.reg_store(_sl(dst_s, (row, col), (1, 1)), r1)  # m_s f32 (inverse input)
        k.wg_sync(barrier_id=10)
        with k.if_(tid < BT):
            for j in range(BT):
                cond = (tid <= j) if mask_strict else (tid < j)
                with k.if_(cond):
                    if dst_bf16:
                        k.reg_fill(rb16, 0.0)
                        k.reg_store(_sl(dst_s, (tid, j), (1, 1)), rb16)
                    else:
                        k.reg_fill(r3, 0.0)
                        k.reg_store(_sl(dst_s, (tid, j), (1, 1)), r3)
        k.wg_sync(barrier_id=10)

    # ============== TMA-load warp (9): cp.async.bulk.tensor (UTMALDG) ==============
    with k.role(warp=TMA_WARP):
        gc = k.scalar(initial=0)  # cumulative chunk index, carried across the persistent
        with k.for_each_task(
            sched
        ) as task:  # tile loop (= CUTLASS PipelineState, never per-tile reset)
            seq, eh, q_head, k_head, v_head, tok_base, NCH, slen = task_geom(task)
            with k.for_loop(stop=NCH) as c:  # runtime chunk loop (varlen num_chunks)
                gtok = tok_base + c * BT
                # K (double-buffered): stage gc%2; wait that stage is free (chunk gc-2's
                # GEMM7 freed it) then load — overlaps the previous chunk's compute.
                kstg, kocc = gc % 2, gc // 2
                k.mbarrier_wait(bars["k_free"], stage=kstg, phase=(kocc + 1) % 2)
                k.mbarrier_arrive_expect_tx(bars["tk"], bytes=BT * K_DIM * 2, stage=kstg)
                k.tma_load(
                    _sl(k_s, (kstg * BT, 0), (BT, K_DIM)),
                    k_g,
                    mbar=bars["tk"],
                    bytes=BT * K_DIM * 2,
                    coords=(gtok, k_head, 0),
                    mbar_stage=kstg,
                    shape=(BT, K_DIM),
                    gmem_shape=(BT, 1, K_DIM),
                )
                # q, v (single-buffered): gated by chunk_free (chunk gc-1 freed the buffers —
                # gc>0 so the wait spans tile boundaries, handing buffers off across tasks).
                with k.if_(gc > 0):
                    k.mbarrier_wait(bars["chunk_free"], phase=(gc - 1) % 2)
                for nm, dst, src, hd, cols in (
                    ("tq", q_s, q_g, q_head, K_DIM),
                    ("tv", v_s, v_g, v_head, V_DIM),
                ):
                    k.mbarrier_arrive_expect_tx(bars[nm], bytes=BT * cols * 2)
                    k.tma_load(
                        _sl(dst, (0, 0), (BT, cols)),
                        src,
                        mbar=bars[nm],
                        bytes=BT * cols * 2,
                        coords=(gtok, hd, 0),
                        shape=(BT, cols),
                        gmem_shape=(BT, 1, cols),
                    )
                k.scalar_store(gc, gc + 1)  # advance the pipeline state

    # ============== gate/beta warp (10): load gate/beta + cumsum -> gate_ready ==============
    with k.role(warp=GATE_WARP):
        lane = k.lane_id()
        gc = k.scalar(initial=0)  # cumulative chunk index (pipeline state across tiles)
        with k.for_each_task(sched) as task:
            seq, eh, q_head, k_head, v_head, tok_base, NCH, slen = task_geom(task)
            with k.for_loop(stop=NCH) as c:
                gtok = tok_base + c * BT
                with k.if_(gc > 0):
                    k.mbarrier_wait(bars["chunk_free"], phase=ph(gc - 1))
                k.mbarrier_arrive_expect_tx(bars["tg"], bytes=BT * 4)
                k.tma_load(
                    gcs_s,
                    gate_g,
                    mbar=bars["tg"],
                    bytes=BT * 4,
                    coords=(gtok, eh),
                    shape=(BT,),
                    gmem_shape=(BT, 1),
                )
                k.mbarrier_arrive_expect_tx(bars["tb"], bytes=BT * 4)
                k.tma_load(
                    beta_s,
                    beta_g,
                    mbar=bars["tb"],
                    bytes=BT * 4,
                    coords=(gtok, eh),
                    shape=(BT,),
                    gmem_shape=(BT, 1),
                )
                k.mbarrier_wait(bars["tg"], phase=ph(gc))
                if (
                    config.varlen
                ):  # OOB tokens (global pos >= seqlen_b) -> gate=1 (log2=0, no decay)
                    for half in range(2):
                        i = lane + half * 32
                        with k.if_((c * BT + i) >= slen):
                            k.reg_fill(r1, 1.0)
                            k.reg_store(_sl(gcs_s, (i,), (1,)), r1)
                    k.warp_sync()
                # gate warp: raw gate a -> log2(a) in place (flashinfer @1896), then the
                # inclusive cumsum gcs[i]=Σ log2(a[l]) -> T[i,j]=exp2(gcs[i]-gcs[j])=Πa.
                for half in range(2):
                    i = lane + half * 32
                    k.reg_load(r1, _sl(gcs_s, (i,), (1,)))
                    k.reg_unary(r1, r1, op="log2")
                    k.reg_store(_sl(gcs_s, (i,), (1,)), r1)
                k.warp_sync()
                # inclusive cumsum over BT=64, 32 lanes x 2 elems (Hillis-Steele)
                for step in range(6):
                    ov = 1 << step
                    for half in range(2):
                        i = lane + half * 32
                        dreg = r2 if half == 0 else racc
                        with k.if_(i >= ov):
                            k.reg_load(r1, _sl(gcs_s, (i - ov,), (1,)))
                            k.reg_load(dreg, _sl(gcs_s, (i,), (1,)))
                            k.reg_add(dreg, dreg, r1)
                    k.warp_sync()
                    for half in range(2):
                        i = lane + half * 32
                        dreg = r2 if half == 0 else racc
                        with k.if_(i >= ov):
                            k.reg_store(_sl(gcs_s, (i,), (1,)), dreg)
                    k.warp_sync()
                k.mbarrier_wait(
                    bars["tb"], phase=ph(gc)
                )  # beta must be loaded before consumers read it
                if config.varlen:  # OOB tokens -> beta=0 (no state update, no delta contribution)
                    for half in range(2):
                        i = lane + half * 32
                        with k.if_((c * BT + i) >= slen):
                            k.reg_fill(r1, 0.0)
                            k.reg_store(_sl(beta_s, (i,), (1,)), r1)
                    k.warp_sync()
                k.mbarrier_arrive(bars["gate_ready0"])
                k.mbarrier_arrive(bars["gate_ready1"])
                k.scalar_store(gc, gc + 1)  # advance pipeline state

    # ============== MMA warp (8): issues all 7 GEMMs (UTCHMMA) ==============
    with k.role(warp=MMA_WARP):
        # gc = every-chunk pipeline state; gc_pos = the GEMM3/4 pipeline (used only on
        # chunk>0 — S_prev exists), advancing on its own cadence (= flashinfer's separate
        # per-pipeline PipelineState). Both carried across the persistent tile loop.
        gc = k.scalar(initial=0)
        gc_pos = k.scalar(initial=0)
        with k.for_each_task(sched) as task:
            seq, eh, q_head, k_head, v_head, tok_base, NCH, slen = task_geom(task)
            with k.for_loop(stop=NCH) as c:
                ks_row = (gc % 2) * BT  # K double-buffer stage row offset
                k.mbarrier_wait(bars["tk"], stage=gc % 2, phase=(gc // 2) % 2)
                issue(
                    acc_s0, k_s, k_s, BT, BT, K_DIM, "d_kk", a_row0=ks_row, b_row0=ks_row
                )  # GEMM1 W_kk -> stage 0
                k.mbarrier_wait(bars["f_kk"], phase=ph(gc))
                k.mbarrier_wait(bars["tq"], phase=ph(gc))
                issue(
                    acc_s1, q_s, k_s, BT, BT, K_DIM, "d_qk", b_row0=ks_row
                )  # GEMM2 W_qk -> stage 1
                k.mbarrier_wait(bars["f_qk"], phase=ph(gc))
                with k.if_(c > 0):  # chunk 0: S_prev=0, skip GEMM3/4 (flashinfer is_first_chunk)
                    # GEMM3/4 read the fp16 state_inp (TMEM) as B=Sᵀ (trans_b); GEMM4 → q_state.
                    k.mbarrier_wait(bars["sT_ready"], phase=ph(gc_pos))
                    issue(
                        acc_tmem,
                        k_s,
                        state_inp,
                        BT,
                        V_DIM,
                        K_DIM,
                        "d_ks",
                        trans_b=True,
                        a_row0=ks_row,
                    )  # GEMM3 K@S
                    k.mbarrier_wait(bars["f_ks"], phase=ph(gc_pos))
                    issue(
                        qstate_tmem, q_s, state_inp, BT, V_DIM, K_DIM, "d_qs", trans_b=True
                    )  # GEMM4 Q@S → q_state
                    k.mbarrier_wait(bars["f_qs"], phase=ph(gc_pos))
                k.mbarrier_wait(bars["ainv_ready"], phase=ph(gc))
                k.mbarrier_wait(bars["delta_ready"], phase=ph(gc))
                with k.if_(c.eq(0)):  # chunk 0: S=0 → delta=v; read v_s directly (vᵀ via trans_b)
                    issue(acc_tmem, ainv_s, v_s, BT, V_DIM, BT, "d_nv", trans_b=True)
                with k.if_(c > 0):
                    issue(
                        acc_tmem, ainv_s, shared_inp, BT, V_DIM, BT, "d_nv"
                    )  # GEMM5 VNEW (delta from TMEM)
                k.mbarrier_wait(bars["f_nv"], phase=ph(gc))
                k.mbarrier_wait(bars["qkv_ready"], phase=ph(gc))
                k.mbarrier_wait(bars["vnew_ready"], phase=ph(gc))
                issue(
                    acc_tmem, attn_s, shared_inp, BT, V_DIM, BT, "d_oi"
                )  # GEMM6 O_intra (NV from TMEM)
                k.mbarrier_wait(bars["f_oi"], phase=ph(gc))
                k.mbarrier_wait(bars["ktvng_ready"], phase=ph(gc))
                issue(
                    s_tmem,
                    k_s,
                    shared_inp_b,
                    K_DIM,
                    V_DIM,
                    BT,
                    "d_ds",
                    accum0=True,
                    trans_a=True,
                    a_row0=ks_row,
                )  # GEMM7 dS
                # GEMM7 was the last K[c] consumer → free this K stage for chunk gc+2.
                k.mbarrier_arrive(bars["k_free"], stage=gc % 2)
                k.scalar_store(gc, gc + 1)
                with k.if_(c > 0):
                    k.scalar_store(gc_pos, gc_pos + 1)

    # ============== compute group 0 (warps 0-3): kk_epi, qk_epi, WY inverse ==============
    with k.role(warpgroup=0):
        tid = k.tid_in_wg()
        lane = tid % 32
        warp = tid // 32
        gc = k.scalar(initial=0)  # cumulative chunk index (pipeline state across tiles)
        with k.for_each_task(sched) as task:
            seq, eh, q_head, k_head, v_head, tok_base, NCH, slen = task_geom(task)
            with k.for_loop(stop=NCH) as c:
                with k.if_(gc > 0):
                    k.mbarrier_wait(bars["chunk_free"], phase=ph(gc - 1))
                k.mbarrier_wait(bars["gate_ready0"], phase=ph(gc))
                # kk_epi: W_kk -> M_kk
                k.mbarrier_wait(bars["d_kk"], phase=ph(gc))
                rss(m_s, True, True, tid, lane, warp, acc=acc_s0)
                k.mbarrier_arrive(bars["f_kk"])
                # qk_epi: W_qk -> W_qkv (attn_s) — done BEFORE the inverse so qkv_ready
                # fires GEMM6 early while the inverse overlaps (flashinfer @2431-2464).
                k.mbarrier_wait(bars["d_qk"], phase=ph(gc))
                rss(attn_s, False, False, tid, lane, warp, dst_bf16=True, acc=acc_s1)
                k.mbarrier_arrive(bars["f_qk"])
                fence_pub(10)
                k.mbarrier_arrive(bars["qkv_ready"])
                # ===== WY inverse: A_inv = (I+M)⁻¹ — flashinfer's hierarchical blockwise
                # inverse using the warp-level mma.sync (SM80 HMMA) + ldmatrix. m_s already
                # holds M = -beta·KKT·T = -L (negated strict-lower, diag 0).
                # Stage 1 (GJ): invert the 8 diagonal 8×8 blocks (f32 forward-sub on m_s,
                # NO extra negation since m_s is already -L). thread = col j of its block.
                for r_ in range(1, 8):
                    with k.if_((tid < BT) & ((tid % 8) < r_)):
                        base = (tid // 8) * 8
                        jc = tid % 8
                        k.reg_fill(racc, 0.0)
                        for mm in range(r_):
                            k.reg_load(r1, _sl(m_s, (base + r_, base + mm), (1, 1)))
                            k.reg_load(r2, _sl(m_s, (base + mm, base + jc), (1, 1)))
                            k.reg_fma(racc, r1, r2, racc)
                    k.wg_sync(barrier_id=10)
                    with k.if_((tid < BT) & ((tid % 8) < r_)):
                        base = (tid // 8) * 8
                        jc = tid % 8
                        k.reg_load(r1, _sl(m_s, (base + r_, base + jc), (1, 1)))
                        k.reg_add(r1, r1, racc)
                        k.reg_store(_sl(m_s, (base + r_, base + jc), (1, 1)), r1)
                    k.wg_sync(barrier_id=10)
                # cvt m_s -> ainv_s (bf16) with unit diagonal (the matrix for the merges)
                with k.if_(tid < BT):
                    for c in range(BT):
                        k.reg_load(r1, _sl(m_s, (tid, c), (1, 1)))
                        k.reg_cvt(rb16, r1)
                        k.reg_store(_sl(ainv_s, (tid, c), (1, 1)), rb16)
                    k.reg_fill(rb16, 1.0)
                    k.reg_store(_sl(ainv_s, (tid, tid), (1, 1)), rb16)
                k.wg_sync(barrier_id=10)
                fence_pub(10)

                def _ldA(src, R, C):  # A = src[R:R+8,C:C+8] non-trans, m16 broadcast
                    k.ldmatrix(
                        _sl(imm_a, (0,), (1,)),
                        _sl(src, (R + lane % 8, C), (1, 8)),
                        num=1,
                        trans=False,
                    )
                    k.reg_store(_sl(imm_a, (1,), (1,)), _sl(imm_a, (0,), (1,)))

                def _ldB(src, R, C):
                    k.ldmatrix(imm_b, _sl(src, (R + lane % 8, C), (1, 8)), num=1, trans=False)

                def _store8(dst, R, C, neg):
                    # The accumulator's top 8×8 (ri 0,1) IS one m8n8 tile in the mma
                    # C/D layout (row=lane//4, col=2(lane%4)+ri) — exactly stmatrix's
                    # fragment, so stage it back via stmatrix (STSM), matching flashinfer.
                    for ri in range(2):
                        if neg:
                            k.reg_mul(_sl(imm_acc, (ri,), (1,)), _sl(imm_acc, (ri,), (1,)), -1.0)
                        k.reg_cvt(_sl(imm_accb, (ri,), (1,)), _sl(imm_acc, (ri,), (1,)))
                    k.stmatrix(
                        _sl(dst, (R + lane % 8, C), (1, 8)),
                        _sl(imm_accb, (0,), (2,)),
                        num=1,
                        trans=False,
                    )

                def _merge(R, Cc, b, w):  # newC = -Qinv·C·Pinv (m_s=-L → store neg=False)
                    tb = b // 8
                    with k.if_((tid // 32).eq(w)):
                        for mi in range(tb):
                            for ni in range(tb):
                                k.reg_fill(imm_acc, 0.0)
                                for ki in range(tb):
                                    _ldA(ainv_s, R + b + mi * 8, Cc + b + ki * 8)
                                    _ldB(ainv_s, R + b + ki * 8, Cc + ni * 8)
                                    k.mma_sync(
                                        imm_acc, imm_a, imm_b, imm_acc, m=16, n=8, k=8, ab_dtype=iod
                                    )
                                _store8(dcs_s, w * 16 + mi * 8, ni * 8, False)
                        k.warp_sync()
                        k.fence(kind=FenceKind.ASYNC_PROXY, scope=FenceScope.CTA)
                        for mi in range(tb):
                            for ni in range(tb):
                                k.reg_fill(imm_acc, 0.0)
                                for ki in range(tb):
                                    _ldA(dcs_s, w * 16 + mi * 8, ki * 8)
                                    _ldB(ainv_s, R + ki * 8, Cc + ni * 8)
                                    k.mma_sync(
                                        imm_acc, imm_a, imm_b, imm_acc, m=16, n=8, k=8, ab_dtype=iod
                                    )
                                _store8(ainv_s, R + b + mi * 8, Cc + ni * 8, neg=False)

                for t in range(4):
                    _merge(t * 16, t * 16, 8, t)
                k.wg_sync(barrier_id=10)
                fence_pub(10)
                for t in range(2):
                    _merge(t * 32, t * 32, 16, t)
                k.wg_sync(barrier_id=10)
                fence_pub(10)
                _merge(0, 0, 32, 0)
                k.wg_sync(barrier_id=10)
                # fold beta[j] into A_inv columns (for GEMM5 VNEW = A_inv·diagβ @ vᵀ)
                with k.if_(tid < BT):
                    for j in range(BT):
                        k.reg_load(r2, _sl(beta_s, (j,), (1,)))
                        k.reg_cvt(rb16b, r2)
                        k.reg_load(rb16, _sl(ainv_s, (tid, j), (1, 1)))
                        k.reg_mul(rb16, rb16, rb16b)
                        k.reg_store(_sl(ainv_s, (tid, j), (1, 1)), rb16)
                k.wg_sync(barrier_id=10)
                fence_pub(10)
                k.mbarrier_arrive(bars["ainv_ready"])
                k.scalar_store(gc, gc + 1)  # advance pipeline state

    # ============== compute group 1 (warps 4-7): new_v, qkv, kv_update ==============
    with k.role(warpgroup=1):
        tid = k.tid_in_wg()
        lane = tid % 32
        warp = tid // 32
        gc = k.scalar(initial=0)  # every-chunk pipeline state (across tiles)
        gc_pos = k.scalar(initial=0)  # GEMM3/4 pipeline state (chunk>0 only)
        with k.for_each_task(sched) as task:
            seq, eh, q_head, k_head, v_head, tok_base, NCH, slen = task_geom(task)
            k.reg_fill(zrow, 0.0)  # per-tile state reset: S_prev = 0 (each sequence)
            k.tcgen05_st(s_tmem, zrow, num=V_DIM, row=0, col=0)
            k.tcgen05_wait_st()
            with k.for_loop(stop=NCH) as c:
                with k.if_(gc > 0):
                    k.mbarrier_wait(bars["chunk_free"], phase=ph(gc - 1))
                k.mbarrier_wait(bars["gate_ready1"], phase=ph(gc))
                k.mbarrier_wait(bars["tv"], phase=ph(gc))  # CG1 reads v_s (delta)
                k.reg_load(glast, _sl(gcs_s, (BT - 1,), (1,)))
                # delta operand (deltaT -> tmpt_s) + o_inter (-> out_s)
                with k.if_(c > 0):
                    # state_inp = fp16 copy of the UNDECAYED S_prev (TMEM), the GEMM3/4
                    # operand (flashinfer state_inp r2t, tcgen05.st.32x32b bf16). Then decay
                    # the main state s_tmem -> Phi*S_prev for GEMM7 — both at the chunk TOP
                    # (flashinfer @3345-3399 stage-then-decay), so GEMM3/4 read a stable copy.
                    k.reg_unary(r1, glast, op="exp2")  # Phi = exp2(glast) (gcs log2-units)
                    for half in range(2):
                        k.tcgen05_ld(frag32, s_tmem, shape="32x32b", num=64, row=0, col=half * 64)
                        k.tcgen05_wait_ld()
                        for cc in range(64):
                            k.reg_cvt(
                                _sl(sinp_reg, (half * 64 + cc,), (1,)), _sl(frag32, (cc,), (1,))
                            )
                            k.reg_mul(_sl(frag32, (cc,), (1,)), _sl(frag32, (cc,), (1,)), r1)
                        k.tcgen05_st(
                            s_tmem, frag32, num=64, row=0, col=half * 64
                        )  # decayed main state
                    k.tcgen05_st(
                        state_inp, _sl(sinp_reg, (0,), (64,)), num=64, row=0, col=0
                    )  # fp16 copy
                    k.tcgen05_wait_st()
                    k.wg_sync(barrier_id=11)
                    k.mbarrier_arrive(bars["sT_ready"])  # state_inp ready for GEMM3/4
                    k.mbarrier_wait(bars["d_ks"], phase=ph(gc_pos))
                    _read128_delta(
                        k, acc_tmem, tmpt_s, v_s, gcs_s, frag, rb16, rb16b, sttile, r1, lane, warp
                    )
                    k.wg_sync(barrier_id=11)
                    k.mbarrier_arrive(bars["f_ks"])
                    k.mbarrier_wait(bars["d_qs"], phase=ph(gc_pos))
                    _read128_ointer(
                        k, qstate_tmem, out_s, gcs_s, scale, frag, r1, sttile, lane, warp
                    )
                    k.wg_sync(barrier_id=11)
                    k.mbarrier_arrive(bars["f_qs"])
                    # stage deltaᵀ (tmpt_s SMEM) -> shared_inp (TMEM fp16, row-major .32x32b)
                    # so GEMM5 reads its B operand from TMEM (flashinfer shared_inp).
                    with k.if_(tid < V_DIM):
                        for j in range(BT):
                            k.reg_load(_sl(sinp_reg, (j,), (1,)), _sl(tmpt_s, (tid, j), (1, 1)))
                        k.tcgen05_st(
                            shared_inp, _sl(sinp_reg, (0,), (BT // 2,)), num=BT // 2, row=0, col=0
                        )
                    k.tcgen05_wait_st()
                    k.wg_sync(barrier_id=11)
                with k.if_(c.eq(0)):
                    # chunk 0: S=0 → o_inter=0 (zero out_s); GEMM5 reads v_s directly
                    # (delta=v), so no v transpose needed here.
                    with k.if_(tid < BT):
                        for dv in range(V_DIM):
                            k.reg_fill(rb16, 0.0)
                            k.reg_store(_sl(out_s, (tid, dv), (1, 1)), rb16)
                    k.wg_sync(barrier_id=11)
                fence_pub(11)
                k.mbarrier_arrive(bars["delta_ready"])
                # new_v_epi: VNEW -> vnewt_s (ungated) + tmpt_s (kgate-scaled, for GEMM7)
                k.mbarrier_wait(bars["d_nv"], phase=ph(gc))
                _read128_vnew(
                    k,
                    acc_tmem,
                    vnewt_s,
                    tmpt_s,
                    gcs_s,
                    glast,
                    frag,
                    sttile,
                    sttile2,
                    r1,
                    lane,
                    warp,
                )
                k.wg_sync(barrier_id=11)
                k.mbarrier_arrive(bars["f_nv"])
                # stage NV (vnewt_s) -> shared_inp (GEMM6 B) and vnew_gated (tmpt_s) ->
                # shared_inp_b (GEMM7 B), both fp16 TMEM (row-major .32x32b). shared_inp
                # reused from delta (consumed by GEMM5 d_nv already).
                with k.if_(tid < V_DIM):
                    for j in range(BT):
                        k.reg_load(_sl(sinp_reg, (j,), (1,)), _sl(vnewt_s, (tid, j), (1, 1)))
                    k.tcgen05_st(
                        shared_inp, _sl(sinp_reg, (0,), (BT // 2,)), num=BT // 2, row=0, col=0
                    )
                    for j in range(BT):
                        k.reg_load(_sl(sinp_reg, (j,), (1,)), _sl(tmpt_s, (tid, j), (1, 1)))
                    k.tcgen05_st(
                        shared_inp_b, _sl(sinp_reg, (0,), (BT // 2,)), num=BT // 2, row=0, col=0
                    )
                k.tcgen05_wait_st()
                k.wg_sync(barrier_id=11)
                fence_pub(11)
                k.mbarrier_arrive(bars["vnew_ready"])
                # GEMM7 reads K directly (A=Kᵀ via the MMA transpose) and tmpt_s
                # (vnew_gated, B); both ready now. State already decayed at chunk top.
                k.mbarrier_arrive(bars["ktvng_ready"])
                # qkv_epilogue: O_intra -> o = o_inter + O_intra -> out_s
                k.mbarrier_wait(bars["d_oi"], phase=ph(gc))
                _read128_store_out(k, acc_tmem, out_s, frag, rb16, sttile, lane, warp)
                k.wg_sync(barrier_id=11)
                k.mbarrier_arrive(bars["f_oi"])
                fence_pub(11)
                k.mbarrier_arrive(bars["o_ready"])
                # kv_update_epi: wait GEMM7 (dS into S) — S now holds this chunk's new state.
                k.mbarrier_wait(bars["d_ds"], phase=ph(gc))
                k.mbarrier_arrive(bars["chunk_free"])
                k.scalar_store(gc, gc + 1)
                with k.if_(c > 0):
                    k.scalar_store(gc_pos, gc_pos + 1)
            # store the FINAL state S -> state_g ONCE, after the chunk loop, via scalar
            # reg->GMEM (st.global) — matching flashinfer _store_final_state's autovec_copy
            # (NOT per-chunk, NOT TMA). thread tid = state row dk; .32x32b is thread=row.
            work = task.field("work")
            seq = work // NEFF
            eh = work % NEFF
            for half in range(2):
                k.tcgen05_ld(frag32, s_tmem, shape="32x32b", num=64, row=0, col=half * 64)
                k.tcgen05_wait_ld()
                k.reg_store(_sl(state_g, (seq, eh, tid, half * 64), (1, 1, 1, 64)), frag32)

    # ============== epilogue warp (11): output store (UTMASTG) ==============
    if not config.varlen:
        with k.role(warp=EPI_WARP, elected=True):
            gc = k.scalar(initial=0)  # cumulative chunk index (pipeline state across tiles)
            with k.for_each_task(sched) as task:
                seq, eh, q_head, k_head, v_head, tok_base, NCH, slen = task_geom(task)
                with k.for_loop(stop=NCH) as c:  # runtime chunk loop
                    k.mbarrier_wait(bars["o_ready"], phase=gc % 2)
                    gtok = tok_base + c * BT
                    k.tma_store(
                        out_g,
                        _sl(out_s, (0, 0), (BT, V_DIM)),
                        coords=(gtok, eh, 0),
                        shape=(BT, V_DIM),
                        gmem_shape=(BT, 1, V_DIM),
                    )
                    k.cp_async_bulk_commit_group()
                    k.cp_async_bulk_wait_group_read(0)
                    k.mbarrier_arrive(bars["chunk_free"])  # out_s freed for next chunk's o_inter
                    k.scalar_store(gc, gc + 1)
    else:
        # varlen: the partial last chunk's OOB rows must NOT be stored (they'd overrun
        # into the next packed sequence). Full-warp scalar store, predicated to valid
        # rows (global pos < seqlen_b); boundary-tile checklist guidance.
        with k.role(warp=EPI_WARP):
            gc = k.scalar(initial=0)
            with k.for_each_task(sched) as task:
                seq, eh, q_head, k_head, v_head, tok_base, NCH, slen = task_geom(task)
                lane = k.lane_id()
                with k.for_loop(stop=NCH) as c:
                    k.mbarrier_wait(bars["o_ready"], phase=gc % 2)
                    gtok = tok_base + c * BT
                    for half in range(2):
                        row = lane + half * 32
                        with k.if_((c * BT + row) < slen):  # valid token only
                            for dv in range(V_DIM):
                                k.reg_load(rb16, _sl(out_s, (row, dv), (1, 1)))
                                k.reg_store(_sl(out_g, (gtok + row, eh, dv), (1, 1, 1)), rb16)
                    k.warp_sync()
                    with k.if_(lane.eq(0)):
                        k.mbarrier_arrive(bars["chunk_free"])
                    k.scalar_store(gc, gc + 1)


def _read128(k, acc, frag):
    # yields (row, col, r) for a 64×128 acc read via two tcgen05.ld.16x256b blocks.
    for blk in range(2):
        k.tcgen05_ld(frag, acc, shape="16x256b", num=8, row=0, col=blk * 64)
        k.tcgen05_wait_ld()
        for va in range(2):
            for vb in range(8):
                for v0p in range(2):
                    yield blk, va, vb, v0p


def _stm(k, dst, row_base, col_base, sttile, lane, trans):
    # stmatrix one m8n8 tile (sttile = 2 bf16 = 1 word) → dst. The .16x256b readback's
    # per-(va,vb) tile IS the mma m8n8 C/D fragment (row=lane//4, col=2(lane%4)+v0p),
    # so stmatrix stages it reg→SMEM (STSM), matching flashinfer's epilogue r2s stores.
    if trans:  # transposed store dst[col, row] (stmatrix.trans writes row=mma_col,col=mma_row)
        k.stmatrix(_sl(dst, (col_base + lane % 8, row_base), (1, 8)), sttile, num=1, trans=True)
    else:
        k.stmatrix(_sl(dst, (row_base + lane % 8, col_base), (1, 8)), sttile, num=1, trans=False)


def _read128_store_out(k, acc, dst_s, frag, rb16, sttile, lane, warp):
    # o[row,col] = O_intra[row,col] + o_inter (pre-staged in dst_s by GEMM4). bf16 add,
    # then stmatrix the tile back (STSM). o_inter is read per-element (data dep).
    for blk, va, vb, v0p in _read128(k, acc, frag):
        r = v0p + 2 * va + 4 * vb
        row = (lane // 4) + 8 * va + 16 * warp
        col = blk * 64 + v0p + 2 * (lane % 4) + 8 * vb
        k.reg_cvt(rb16, _sl(frag, (r,), (1,)))  # O_intra → bf16
        k.reg_load(_sl(sttile, (v0p,), (1,)), _sl(dst_s, (row, col), (1, 1)))  # o_inter
        k.reg_add(_sl(sttile, (v0p,), (1,)), _sl(sttile, (v0p,), (1,)), rb16)
        if v0p == 1:
            _stm(k, dst_s, 8 * va + 16 * warp, blk * 64 + 8 * vb, sttile, lane, trans=False)


def _read128_delta(k, acc, tmpt_s, v_s, gcs_s, frag, rb16, rb16b, sttile, r1, lane, warp):
    # delta[i,dv] = v[i,dv] - exp2(gcs[i])·KH[i,dv]; store deltaᵀ → tmpt_s[dv,i] (stmatrix.trans).
    for blk, va, vb, v0p in _read128(k, acc, frag):
        r = v0p + 2 * va + 4 * vb
        row = (lane // 4) + 8 * va + 16 * warp  # i
        col = blk * 64 + v0p + 2 * (lane % 4) + 8 * vb  # dv
        k.reg_load(r1, _sl(gcs_s, (row,), (1,)))
        k.reg_unary(r1, r1, op="exp2")  # dexp[i] (gcs log2-units)
        k.reg_mul(r1, r1, _sl(frag, (r,), (1,)))  # dexp·KH (f32)
        k.reg_cvt(rb16b, r1)  # → bf16
        k.reg_load(rb16, _sl(v_s, (row, col), (1, 1)))  # v (bf16)
        k.reg_sub(_sl(sttile, (v0p,), (1,)), rb16, rb16b)  # delta = v - dexp·KH (bf16)
        if v0p == 1:
            _stm(k, tmpt_s, 8 * va + 16 * warp, blk * 64 + 8 * vb, sttile, lane, trans=True)


def _read128_ointer(k, acc, out_s, gcs_s, scale, frag, r1, sttile, lane, warp):
    # o_inter[i,dv] = exp2(gcs[i])·scale·QS[i,dv]; pre-stage into out_s (bf16) via stmatrix.
    for blk, va, vb, v0p in _read128(k, acc, frag):
        r = v0p + 2 * va + 4 * vb
        row = (lane // 4) + 8 * va + 16 * warp
        col = blk * 64 + v0p + 2 * (lane % 4) + 8 * vb
        k.reg_load(r1, _sl(gcs_s, (row,), (1,)))
        k.reg_unary(r1, r1, op="exp2")  # dexp[i] (gcs log2-units)
        k.reg_mul(r1, r1, _sl(frag, (r,), (1,)))
        k.reg_mul(r1, r1, float(scale))
        k.reg_cvt(_sl(sttile, (v0p,), (1,)), r1)
        if v0p == 1:
            _stm(k, out_s, 8 * va + 16 * warp, blk * 64 + 8 * vb, sttile, lane, trans=False)


def _read128_vnew(k, acc, vnewt_s, vng_s, gcs_s, glast, frag, sttile, sttile2, r1, lane, warp):
    # VNEW readback → vnewt_s[dv,i]=vnew (ungated, GEMM6) AND vng_s[dv,i]=vnew·exp2(glast-gcs[i])
    # (gated, GEMM7); both transposed → stmatrix.trans (one m8n8 tile per (va,vb)).
    for blk, va, vb, v0p in _read128(k, acc, frag):
        r = v0p + 2 * va + 4 * vb
        k.reg_cvt(_sl(sttile, (v0p,), (1,)), _sl(frag, (r,), (1,)))  # ungated
        row = (lane // 4) + 8 * va + 16 * warp  # i (token)
        k.reg_load(r1, _sl(gcs_s, (row,), (1,)))  # gcs[i]
        k.reg_sub(r1, glast, r1)  # glast - gcs[i]   (glast is REG)
        k.reg_unary(r1, r1, op="exp2")  # kgate[i] (gcs log2-units)
        k.reg_mul(r1, r1, _sl(frag, (r,), (1,)))  # · vnew
        k.reg_cvt(_sl(sttile2, (v0p,), (1,)), r1)  # gated
        if v0p == 1:
            rb = 8 * va + 16 * warp
            cb = blk * 64 + 8 * vb
            _stm(k, vnewt_s, rb, cb, sttile, lane, trans=True)
            _stm(k, vng_s, rb, cb, sttile2, lane, trans=True)


def _validate_config(config: GdnPrefillConfig) -> None:
    if config.num_seqs < 1 or config.seqlen < 1:
        raise ValueError("gdn_prefill num_seqs/seqlen must be positive")
    hq, hv = config.num_q_heads, config.num_v_heads
    if hq < 1 or hv < 1 or max(hq, hv) % min(hq, hv) != 0:
        raise ValueError(
            "gdn_prefill needs num_q_heads,num_v_heads >= 1 with max a multiple of min "
            f"(GVA h_v≥h_q or GQA h_q>h_v); got num_q={hq}, num_v={hv}."
        )
    if not config.varlen and config.seqlen % BT != 0:
        raise ValueError(f"gdn_prefill fixed-length requires seqlen a multiple of {BT}")
    # Varlen (cu_seqlens) path: the chunk loop is a runtime for_loop with per-tile
    # num_chunks_b = ceil_div(seqlen_b, BT) loaded from cu_seqlens (k.scalar). Arbitrary
    # lengths supported — the partial last chunk's OOB tokens are masked to no-ops (gate=1,
    # beta=0; K/Q/V additionally TMA-OOB-zero-filled) and the epilogue stores only valid
    # rows. cu_seqlens is the i32[NS+1] arg, passed at interpret time.

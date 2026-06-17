"""Flash-Attention SM100 BACKWARD (dQ/dK/dV) expressed in Nymph IR.

Instruction-faithful port of ~/flash-attention/flash_attn/cute/flash_bwd_sm100.py
(class FlashAttentionBackwardSm100, arch=100). See the working docs next to this file:
  - docs/kernels/flash_bwd_sm100/MAP.md   — full structural map (roles, GEMMs, copies, layout, sched)
  - docs/kernels/flash_bwd_sm100/INSTR.md — real B200 PTX per-op instruction inventory

CONFIG MAP (interface.py:1332): head_dim>=128 -> 2-CTA, head_dim<128 -> 1-CTA. This file
targets **MILESTONE-1 = hd64, 1-CTA, non-causal, dense** (cluster_size=1). Extensions
(causal -> varlen -> deterministic -> GQA -> hd128 2-CTA) come after.

16-warp specialization (FlashAttentionBackwardSm100.__init__):
  warps 0-3   reduce  : dQacc_reduce — t2r read dQ TMEM, r2s SMEM, tma_reduce_add -> dQaccum
  warps 4-11  compute : 2 wg — recompute P=exp2(S·scale_log2 − LSE), dS=P∘(dP−dPsum)
  warp  12    mma     : the 5 tcgen05 GEMMs
  warp  13    load    : TMA Q/K/V/dO + bulk LSE/dPsum
  warp  14    relay   : (2-CTA only — idle in 1-CTA)
  warp  15    empty   : register donor

The 5 GEMMs (acc in TMEM f32; tcgen05.mma.cta_group::1.kind::f16):
  S  = K·Qᵀ    (A=sK SMEM, B=sQ SMEM)        -> tmem_S
  dP = V·dOᵀ   (A=sV SMEM, B=sdOt SMEM)       -> tmem_dP
  dV = Pᵀ·dO   (A=P TMEM,  B=sdO SMEM, accum) -> tmem_dV
  dK = dSᵀ·Q   (A=dS TMEM, B=sQt SMEM, accum) -> tmem_dK
  dQ = dS·K    (A=dS,      B=K)               -> tmem_dQ  (reduce-add to global dQaccum)
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from math import gcd as _math_gcd

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
from ..nymph_rs import (
    max as scalar_max,  # runtime ScalarExpr max(a,b) — used to clamp m_block_min >= 0
)

# ---- architecture constants (flashattn FlashAttentionBackwardSm100 for hd64 1-CTA) ----
TILE_M = 128  # Q tile (m_block_size)
TILE_N = 128  # KV tile (n_block_size)
SM_COUNT = 148
N_COLS_TMEM = 512  # tcgen05 TMEM columns (sm_100)


def _dq_reduce_cfg(hdim, use_2cta, is_causal, deterministic):
    """flashattn's per-config dQ_reduce_ncol / sdQaccum_stage (flash_bwd_sm100.py:245-258).
    The dQ reduce SMEM (sdQaccum) is tile_m · dQ_reduce_ncol · sdQaccum_stage f32 — a
    `sdQaccum_stage`-deep double buffer of `dQ_reduce_ncol`-col slices (NOT the full tile)."""
    if use_2cta and hdim == 192:
        ncol = 32 if is_causal else 24
        nslot = 1 if is_causal else 2
    elif use_2cta:
        ncol = 16 if deterministic else 8
        nslot = 2 if deterministic else 4
    else:
        ncol = 32
        nslot = 64 // ncol  # = 2
    return ncol, nslot


# 16-warp role IDs (FlashAttentionBackwardSm100.__init__:141-146)
REDUCE_WARPS = (0, 1, 2, 3)
COMPUTE_WARPS = (4, 5, 6, 7, 8, 9, 10, 11)
MMA_WARP = 12
LOAD_WARP = 13
RELAY_WARP = 14
EMPTY_WARP = 15
NUM_WARPS = 16

LOG2_E = 1.4426950408889634  # softmax_scale_log2 = softmax_scale · log2(e) (I3: exp2 not exp)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _sl(t, offs, shape):
    return TensorSlice(tensor=t, offsets=offs, shape=shape)


_VALID_TC = (128, 64, 32, 16, 8, 4, 2, 1)  # legal 32x32b .x{N} counts (tcgen05_datapath.rs:61)


def _tc_chunks(num: int) -> list[int]:
    """Decompose a tcgen05.ld column count into valid 32x32b .x{N} pieces, largest-first
    (64 -> [64]; 48 -> [32,16]; 96 -> [64,32]; 192 -> [128,64])."""
    out, r = [], num
    for v in _VALID_TC:
        while r >= v:
            out.append(v)
            r -= v
    assert r == 0, f"cannot tile tcgen05 num={num}"
    return out


def _tc_ld(k, frag, src, num, col=0, base=0, row=0):
    """t2r read of `num` f32 TMEM cols [col, col+num) into frag[base:base+num], chunked into
    valid 32x32b .x{N} instructions. hd64/hd128 have power-of-2 head dims -> a single .x{N};
    hd96/hd192 split a head-dim read (48->32+16, 96->64+32, 192->128+64). The chunks cover the
    SAME column span and write contiguous frag slots, so value + protocol are identical to a
    single read; flashattn's PTX likewise covers a non-power-of-2 span with this same valid set
    (hd96 bwd PTX = 32x32b.x16 + .x32). One tcgen05_wait_ld after the call drains all chunks."""
    off = 0
    for c in _tc_chunks(num):
        k.tcgen05_ld(
            _sl(frag, (base + off,), (c,)), src, shape="32x32b", num=c, row=row, col=col + off
        )
        off += c


@dataclass(frozen=True, slots=True)
class FlashBwdSm100Config:
    # workload
    batch: int = 1
    seqlen_q: int = 128
    seqlen_k: int = 128
    num_head: int = 1  # num_head_q
    num_head_kv: int | None = (
        None  # GQA: #kv-heads. None => == num_head (MHA, G=1). G = num_head // num_head_kv.
    )
    head_dim: int = 64  # <128 -> 1-CTA path
    head_dim_v: int | None = None
    is_causal: bool = False
    scale: float | None = None  # default 1/sqrt(head_dim)
    launch_shape: LaunchShape | None = None
    # VARLEN (cu_seqlens packed variable-length sequences). When True, seqlen_q/seqlen_k are
    # reinterpreted as a MAX-for-sizing upper bound (gdn convention) — the packed buffers are
    # over-allocated to batch*seqlen rows, and each sequence's real token range + per-seq tile
    # count come from a runtime cu_seqlens lookup. Two cu_seqlens GMEM args (cu_q_g/cu_k_g,
    # shape (batch+1,), i32) are appended LAST so the fixed-length arg order is unchanged.
    varlen: bool = False
    # DETERMINISTIC (bit-reproducible dQ / GQA-dKV accumulation). When True, each cross-CTA reduce-add
    # into a shared output tile is SERIALIZED into a fixed order by a per-tile GMEM i32 semaphore: the
    # writer with lock-value v spins until the counter == v (gmem_wait_eq), does its TMA reduce-add,
    # full-drains it, then bumps the counter to v+1 (gmem_atomic_add, order=release) to release the
    # next writer. The add STAYS a reduce-add — determinism comes only from the fixed order. This makes
    # the order EXIST, so the protocol checker flips the cross-CTA reduce-add Inconclusive -> Passed.
    # dQ lock-value = n_block; GQA dKV lock-value = head % G. New semaphore GMEM args are appended LAST
    # (fixed arg order). When False the IR is BYTE-IDENTICAL to the non-deterministic path. The dense
    # non-causal milestone targets 1-CTA hd64; gated off for 2-CTA (out of milestone scope).
    deterministic: bool = False

    @property
    def head_dim_v_(self) -> int:
        return self.head_dim_v if self.head_dim_v is not None else self.head_dim

    @property
    def num_head_kv_(self) -> int:
        # #kv-heads. Default (None) => MHA (== num_head, G=1).
        return self.num_head_kv if self.num_head_kv is not None else self.num_head

    @property
    def gqa_group(self) -> int:
        # G = qhead_per_kvhead = num_q_heads // num_kv_heads. G==1 => MHA (byte-identical IR).
        assert self.num_head % self.num_head_kv_ == 0, "num_head must be divisible by num_head_kv"
        return self.num_head // self.num_head_kv_

    @property
    def softmax_scale(self) -> float:
        return self.scale if self.scale is not None else (self.head_dim**-0.5)


def _tile_hdim(d: int) -> int:
    return _ceil_div(d, 16) * 16  # pad to k_block_size (multiple of 16)


def _use_2cta(config: FlashBwdSm100Config) -> bool:
    # CONFIG MAP (interface.py:1332): head_dim>=128 -> 2-CTA (cluster_size=2). hd<128 -> 1-CTA.
    return config.head_dim >= 128


def _derived(config: FlashBwdSm100Config) -> dict:
    """Compile-time derived geometry (mirrors __init__ / _setup_attributes)."""
    tile_hdim = _tile_hdim(config.head_dim)
    tile_hdimv = _tile_hdim(config.head_dim_v_)
    # __init__:82 — hd<=128 (symmetric) or the (192,128) DeepSeek shape.
    assert tile_hdim <= 128 or (tile_hdim == 192 and tile_hdimv == 128), (
        f"unsupported head_dim/head_dim_v = {tile_hdim}/{tile_hdimv}"
    )
    use_2cta = _use_2cta(config)
    is_hd192 = tile_hdim == 192 and tile_hdimv == 128  # DeepSeek (2-CTA, dedicated TMEM layout)
    assert not is_hd192 or use_2cta, "hd192 must use 2-CTA (__init__:94)"
    cg = 2 if use_2cta else 1  # cta_group_size
    # MMA tilers (M, N, K) — __init__:97-108. Under 2-CTA the kv (M) dimension is
    # cluster-wide (cg*TILE_N=256, split across the CTA pair, Layout A); dQ's K
    # (reduction) is cluster-wide instead (stage 2).
    d = dict(
        tile_hdim=tile_hdim,
        tile_hdimv=tile_hdimv,
        cta_group=cg,
        use_2cta=use_2cta,
        is_hd192=is_hd192,
        mma_kq=(cg * TILE_N, TILE_M, tile_hdim),  # S  = K·Qᵀ
        mma_vdo=(cg * TILE_N, TILE_M, tile_hdimv),  # dP = V·dOᵀ
        mma_pdo=(cg * TILE_N, tile_hdimv, TILE_M),  # dV = Pᵀ·dO
        mma_dsq=(cg * TILE_N, tile_hdim, TILE_M),  # dK = dSᵀ·Q
        mma_dsk=(TILE_M, tile_hdim, cg * TILE_N),  # dQ = dS·K  (K cluster-wide)
    )
    if is_hd192:
        # hd192 DeepSeek dedicated TMEM layout — __init__:183-191. The general 2-CTA layout
        # (dK at dP+tile_m=384, +tile_hdim=192 -> 576) overflows 512, so hd192 repacks:
        # dV[0,128) dK[128,320) S/P[320,448) dP/dS[384,512) dQ[416,512). The S/dP, S/dQ and
        # dP/dQ spans overlap by design — the pipeline keeps their lifetimes disjoint (S read
        # before dP/dQ overwrite those cols), exactly as flashattn does.
        d["tmem_dV"] = 0
        d["tmem_dK"] = d["tmem_dV"] + tile_hdimv  # 128
        d["tmem_S"] = d["tmem_dK"] + tile_hdim  # 320
        d["tmem_P"] = d["tmem_S"]  # overlaps S
        d["tmem_dP"] = N_COLS_TMEM - TILE_M  # 384
        d["tmem_dS"] = d["tmem_dP"]  # overlaps dP
        d["tmem_dQ"] = N_COLS_TMEM - tile_hdim // 2  # 416 (half-hdim per CTA; ends at 512)
    elif use_2cta:
        # 2-CTA TMEM offsets — __init__:194-204 (general 2-CTA path; map §4).
        # S/P overlap at 0; dV at tile_n; dP/dS at dV+hdimv; dQ at half-hdim into S;
        # dK at dP+tile_m. n_cols = full 512 (sm_100), alloc cta_group=2.
        d["tmem_S"] = 0
        d["tmem_P"] = 0  # overlaps S
        d["tmem_dV"] = TILE_N  # 128
        d["tmem_dP"] = d["tmem_dV"] + tile_hdimv  # 256
        d["tmem_dQ"] = d["tmem_S"] + tile_hdim // 2  # 64 (2-CTA-specific; overlaps S)
        d["tmem_dK"] = d["tmem_dP"] + TILE_M  # 384
        d["tmem_dS"] = d["tmem_dP"]  # overlaps dP (256)
    else:
        # TMEM offsets — __init__:193-204 (general 1-CTA path; S/P, dS/dP, dQ/dP overlap)
        d["tmem_S"] = 0
        d["tmem_P"] = 0  # overlaps S
        d["tmem_dV"] = d["tmem_S"] + TILE_N  # 128
        d["tmem_dP"] = d["tmem_dV"] + tile_hdimv  # 128 + hdimv
        d["tmem_dQ"] = d["tmem_dP"]  # 1-CTA: dQ overlaps dP
        d["tmem_dK"] = d["tmem_dP"] + TILE_M  # dP + 128
        d["tmem_dS"] = d["tmem_dP"]  # overlaps dP
    # capacity: every accumulator's column extent must fit the 512-col TMEM (sm_100).
    extent = max(
        d["tmem_S"] + TILE_M,
        d["tmem_dV"] + tile_hdimv,
        d["tmem_dP"] + TILE_M,
        d["tmem_dK"] + tile_hdim,
        d["tmem_dQ"] + (tile_hdim // cg),
    )
    assert extent <= N_COLS_TMEM, f"TMEM overflow: extent {extent} > {N_COLS_TMEM}"
    return d


# Shape configs to cover (per-shape config alignment). Milestone-1 starts at the first.
CONFIGS = [
    {
        "batch": 1,
        "num_head": 1,
        "seqlen_q": 128,
        "seqlen_k": 128,
        "head_dim": 64,
        "label": "b1h1s128_hd64",
    },
    {
        "batch": 1,
        "num_head": 2,
        "seqlen_q": 256,
        "seqlen_k": 256,
        "head_dim": 64,
        "label": "b1h2s256_hd64",
    },
    {
        "batch": 2,
        "num_head": 2,
        "seqlen_q": 256,
        "seqlen_k": 128,
        "head_dim": 64,
        "label": "b2h2_sq256sk128_hd64",
    },
]

# hd128 2-CTA configs (head_dim>=128 -> cluster_size=2). One cluster = 2 adjacent kv-tiles
# of TILE_N=128 (Sk=256), one Q-tile (Sq=128). The CTA pair (leader=even) issues the
# cluster-wide cta_group::2 GEMMs (S/dP/dV/dK) over the 256-row kv band.
CONFIGS_2CTA = [
    {
        "batch": 1,
        "num_head": 1,
        "seqlen_q": 128,
        "seqlen_k": 256,
        "head_dim": 128,
        "label": "b1h1_sq128sk256_hd128",
    }
]


def build_flash_bwd_sm100(config: FlashBwdSm100Config = FlashBwdSm100Config()) -> Kernel:
    """FA-bwd kernel (milestone-1: hd64 1-CTA non-causal dense). 16-warp specialized.

    DRAFT (Phase 4): structure + all 5 GEMMs + load/compute/reduce/epilogue roles wired
    per docs/kernels/flash_bwd_sm100/MAP.md; targets build + trace. The compute t2r/r2t fragment
    layout + per-instruction PTX alignment + value-sim are being tightened next."""
    d = _derived(config)
    use_2cta = d["use_2cta"]
    is_hd192 = d["is_hd192"]  # DeepSeek (192,128): dedicated TMEM layout + S→dP→dK→dV→dQ order
    cg = d["cta_group"]
    B, H = config.batch, config.num_head
    Hkv = config.num_head_kv_  # #kv-heads (== H for MHA)
    G = config.gqa_group  # qhead_per_kvhead = H // Hkv (1 for MHA)
    gqa = G > 1  # GQA gate; G==1 path stays byte-identical
    # varlen-GQA must route dK/dV through the head_kv reduce-add (flashattn use_tma_store=
    # not(qhead_per_kvhead==1 and varlen_k)); the varlen scalar-store epilogue writes per
    # q-head and overwrites — wrong for GQA. Guard the unimplemented combo (reviewer D1).
    Sq, Sk = config.seqlen_q, config.seqlen_k
    Dq, Dv = config.head_dim, config.head_dim_v_
    hdim, hdimv = d["tile_hdim"], d["tile_hdimv"]
    Tq, Tk = B * Sq, B * Sk
    n_mb = _ceil_div(Sq, TILE_M)  # m-blocks (Q tiles) per task
    n_nb = _ceil_div(Sk, TILE_N)  # n-blocks (KV tiles) = CTAs per (head,batch)
    real_n_nb = n_nb  # the TRUE #kv-tiles (compile-time); n_nb may grow below for the grid
    # 2-CTA: each "task" is one CLUSTER (a pair of adjacent kv-tiles). Under the
    # cluster scheduler the two CTAs of a pair share one task; the kernel launches
    # num_tasks*cg CTAs (fp16_bf16_gemm pattern). n_cluster = #kv-tile-pairs.
    if use_2cta:
        if config.varlen:
            # VARLEN 2-CTA: the grid is sized over the OVER-ALLOCATED upper bound (Sk = maxlen),
            # so n_nb = ceil(maxlen/TILE_N) may be ODD. The cluster grid pairs adjacent kv-tiles,
            # so ROUND n_nb UP to a multiple of cg — the extra padding kv-tile(s) lie past EVERY
            # sequence's slen_k (>= maxlen), so the cluster/tile validity predicates mask them out
            # exactly like the partial-cluster peer (cluster_valid runs the cluster MMA with the
            # padding tile's dS=0; tile_valid suppresses its dK/dV store and the per-row dQ
            # predicate handles partial Q-tiles). Each real sequence's true tile count is recovered
            # at runtime from cu_seqlens (n_nb_b), so the padded grid tiles cost a masked-out
            # cluster, never a wrong write. (Fixed-len 2-CTA keeps the exact-divisibility assert.)
            n_nb = ((n_nb + cg - 1) // cg) * cg
        elif n_nb % cg != 0:
            # DENSE 2-CTA, tile-aligned-but-not-cluster-aligned Sk (odd #kv-tiles, e.g. Sk=384 ->
            # 3 tiles). The cluster grid pairs adjacent kv-tiles, so ROUND n_nb UP to a cg multiple
            # for the GRID — the last cluster gets one real kv-tile + one PADDING kv-tile whose kv
            # range [real_n_nb*TILE_N, n_nb*TILE_N) is entirely past Sk. This is the COMPILE-TIME
            # analog of the varlen partial cluster: real_n_nb (= ceil(Sk/TILE_N)) replaces the
            # runtime n_nb_b. The padding tile runs the cluster cta_group::2 MMA/compute/exchange in
            # LOCKSTEP (so the leader's peer barriers stay balanced) under the cluster predicate
            # (nb-cta_in_cluster < real_n_nb); its P is forced 0 by a tile-level mask (nb>=real_n_nb)
            # so dS/dV/dK/dQ get a 0 contribution (CRITICAL for batch>1, where the padding tile's K
            # load reads the NEXT batch's K, not zero); its per-CTA dK/dV store + dQ reduce are
            # suppressed by the per-CTA tile_valid (nb<real_n_nb). Sk that IS a cg*TILE_N multiple
            # keeps n_nb==real_n_nb -> dense_pad is off -> BYTE-IDENTICAL to before.
            n_nb = ((n_nb + cg - 1) // cg) * cg
    n_cluster = n_nb // cg if use_2cta else n_nb
    # DENSE 2-CTA padding-tile mode: active only when the grid was rounded up past the true tile
    # count (odd #kv-tiles), never under varlen (which has its own runtime n_nb_b machinery).
    dense_pad = use_2cta and not config.varlen and (real_n_nb != n_nb)
    num_work = n_cluster * H * B  # cluster tasks (each handled by a CTA pair under 2-CTA)
    scale_log2 = config.softmax_scale * LOG2_E
    iod = DType.BF16

    # n_mb is COMPILE-TIME (Python int) -> the m-loop is Python-unrolled. The per-m-block
    # operands sQ/sdO/sLSE/sdPsum are DOUBLE-BUFFERED (2 SMEM stages, stage row = mb%2 ·TILE_M)
    # so mb0->stage0, mb1->stage1 never alias (n_mb<=2): the load->mma/compute WAR is removed
    # without any operand empty barrier. sK/sV stay single (loaded once per task).
    # 1-CTA double-buffers the per-mb operands (Q_stage=2). 2-CTA uses flashattn's Q_stage=1 /
    # dO_stage=1 (flash_bwd_sm100.py:239-240) — SINGLE-stage operands, the load->mma reload WAR
    # serialized by the empty barriers (q_free/do_free/lse_free/dps_free), NOT double-buffering.
    NSTAGE = 1 if use_2cta else (2 if n_mb > 1 else 1)  # operand SMEM/barrier stages
    # dQ reduce staging — flashattn's exact per-config (dQ_reduce_ncol, sdQaccum_stage).
    RDQ_NCOL, RDQ_NSLOT = _dq_reduce_cfg(hdim, use_2cta, config.is_causal, config.deterministic)
    # dK/dV epilogue CHUNK staging — flashattn's dK_reduce_ncol / sdK_epi_tile[1]
    # (flash_bwd_sm100.py:264,415,440). The epilogue is CHUNKED: each compute wg owns hdim_half
    # cols and stores them in num_epi_stages pieces of RNCOL cols each, through a SINGLE
    # [TILE_N, RNCOL] slice PER WG (flashattn sdK is per-wg via sdKV[None,(None,)wg_idx]
    # @3888/3890 — single-buffered within a wg, full-drained between chunks @4015). The two wgs
    # use DISJOINT column halves of the buffer. The chunk width is dtype-keyed exactly as
    # flashattn: GQA f32 -> gcd(128//4, hdim/2)=gcd(32,..) (=dK_reduce_ncol); MHA bf16 ->
    # gcd(128//2, hdim/2)=gcd(64,..) (=epi_tile[1]). Per-wg row = RNCOL*dkv_bytes = 128 B either
    # way, so f32 total = tile_n*RNCOL*NUM_CWG*4 = bf16 total = tile_n*RNCOL*NUM_CWG*2 = 32 KB ==
    # bf16 sK, so the sdK→sK / sdV→sV alias no longer grows them past 227 KB.
    NUM_CWG = 2  # two compute warpgroups (each owns a wg-half)
    _dkv_bytes = 4 if gqa else 2  # f32 (GQA reduce) / bf16 (MHA store)
    DK_RNCOL = _math_gcd(128 // _dkv_bytes, hdim // 2)  # GQA:32  MHA:64 (chunk width, dK)
    DV_RNCOL = _math_gcd(128 // _dkv_bytes, hdimv // 2)  # GQA:32  MHA:64 (chunk width, dV)
    # ---- VALUE-CORRECT 2-CTA B-operand split (map §2; THE FIX) ----
    # tcgen05.mma.cta_group::2 splits BOTH operands across the CTA pair: A by M (each
    # CTA owns kv-rows [v*tile_n,(v+1)*tile_n)), B by N (each CTA owns N/2 of the B
    # operand). The interpreter's read_operand_ctas reads the SAME (offset, shape) box
    # from EACH CTA and CONCATENATES along axis 0 -> for value-correctness each CTA must
    # load its OWN half of every operand into the SAME SMEM byte offset (the GEMM
    # convention; fp16_bf16_gemm.py does this via a cta_local_id-shifted coordinate).
    # A (K/V, P/dS-TMEM) is already per-kv-tile. The four B operands split along DIFFERENT
    # axes (the N axis of each GEMM): S/dP split q (N=tile_m), dV/dK split head-dim (N=hdim).
    # So each 2-CTA B buffer is sized N/2 along its N axis and each CTA loads its own half.
    TM_H = TILE_M // cg  # q half (S/dP B N-half) = 64
    HD_H = hdim // cg  # d half  (dK B N-half)  = 64
    HDV_H = hdimv // cg  # dv half (dV B N-half)  = 64
    # ---- SMEM layout (bytes): operands + epilogue staging ----
    if use_2cta:
        # 2-CTA B operands carry only this CTA's N/2 half (so the cluster MMA concat over
        # the pair reconstructs the full N). sQ (S B): q-rows split -> TM_H rows, full d.
        # sdO (dV B): dv-cols split -> full q rows (contraction), HDV_H cols.
        sizes = dict(
            sK=TILE_N * hdim * 2,
            sV=TILE_N * hdimv * 2,
            sQ=TM_H * hdim * 2,  # S  B: q-half rows, full d (contraction)
            sdO=TILE_M * HDV_H * 2,  # dV B: full q rows (contract), dv-half cols
            # NOTE: no sdS here — 2-CTA's dK reads dS from TMEM (tdS) and dQ reads it from the
            # cluster-wide sdS_full (assembled via the exchange), so the [TILE_N,TILE_M] SMEM sdS
            # is 1-CTA-only. (flashattn's single sdS == my sdS_full, the dQ A-operand buffer.)
            sLSE=NSTAGE * TILE_M * 4,
            sdPsum=NSTAGE * TILE_M * 4,
            sdQ=TILE_M
            * RDQ_NCOL
            * RDQ_NSLOT
            * 4,  # dQ reduce: RDQ_NSLOT-deep double buffer of RDQ_NCOL-col slices
            # GQA: f32 (4B) staging — the reduce-add accumulates the G q-heads into the shared
            # kv-head's f32 rows (tma_reduce_add needs an f32 src). MHA: bf16 plain store. The
            # tensor dtype is sdkv_dtype (F32 if gqa). CHUNKED (flashattn dK_reduce_ncol): the
            # epilogue stores ONE [TILE_N, RNCOL] slice at a time; the buffer holds NUM_CWG such
            # slices (one per compute wg, DISJOINT halves — single-buffered within a wg). f32 GQA
            # hd128 = 128*32*2*4 = 32 KB == the bf16 sK -> aliasing sdK→sK is FREE (no growth).
            sdK=TILE_N * DK_RNCOL * NUM_CWG * (4 if gqa else 2),
            sdV=TILE_N * DV_RNCOL * NUM_CWG * (4 if gqa else 2),
        )
        # sdOt (dP B): q-half rows, full dv (contraction). sQt (dK B): full q rows
        # (contraction), d-half cols. Loaded by a transposed TMA descriptor; each CTA's
        # own half lands at the same SMEM offset.
        sizes["sdOt"] = TM_H * hdimv * 2  # dP B: q-half rows, full dv (contraction)
        sizes["sQt"] = TILE_M * HD_H * 2  # dK B: full q rows (contract), d-half cols
        # dQ datapath (the dS cross-CTA exchange + dQ MMA, map §3). The dQ MMA is
        # cta_group::2, m=TILE_M=128 (Layout B 2x2): A=dS split by M (q) -> 64 q-rows/CTA,
        # B=sKt split by N (d) -> HD_H d-cols/CTA, with the K (kv) FULL 256 on each CTA.
        # So per CTA:
        #   sdS_full = [cg*TILE_N kv, TM_H q]  full 256-kv (both halves via exchange),
        #              this CTA's own q-half. dS A-operand for dQ (trans_a reads [16kv, TM_H]).
        #   sKt      = [cg*TILE_N kv, HD_H d]  full 256-kv (this CTA's d-half, via a
        #              transposed TMA descriptor of K). dQ B-operand (trans_b reads [16kv,HD_H]).
        #   sdS_xchg = [TILE_N kv, TM_H q]  this CTA's own kv-tile dS for the PEER's q-half
        #              (the export staging buffer s2clustered into the peer's sdS_full slot).
        sizes["sKt"] = cg * TILE_N * HD_H * 2  # dQ B: full kv, this CTA's d-half
        sizes["sdS_full"] = cg * TILE_N * TM_H * 2  # dQ A: full kv (exchanged), this CTA's q-half
        sizes["sdS_xchg"] = TILE_N * TM_H * 2  # dS export half (this kv-tile, peer's q-half)
    else:
        sizes = dict(
            sK=TILE_N * hdim * 2,
            sV=TILE_N * hdimv * 2,
            sQ=NSTAGE * TILE_M * hdim * 2,
            sdO=NSTAGE * TILE_M * hdimv * 2,
            sdS=TILE_N * TILE_M * 2,  # dS staged bf16 for dQ/dK B-operand path
            sLSE=NSTAGE * TILE_M * 4,
            sdPsum=NSTAGE * TILE_M * 4,
            sdQ=TILE_M
            * RDQ_NCOL
            * RDQ_NSLOT
            * 4,  # dQ reduce: RDQ_NSLOT-deep double buffer of RDQ_NCOL-col slices
            # epilogue dK/dV staging. GQA: f32 (4B) — the reduce-add accumulates the G
            # q-heads' dK/dV into the shared kv-head's f32 rows (tma_reduce_add needs f32 src).
            # MHA: bf16 (2B) plain tma_store. CHUNKED (flashattn dK_reduce_ncol): ONE
            # [TILE_N, RNCOL] slice staged at a time; the buffer holds NUM_CWG such slices (one
            # per compute wg, DISJOINT halves), so the f32 GQA staging stays the bf16 sQ/sdO size
            # and the sdK→sQ / sdV→sdO alias does not grow them.
            sdK=TILE_N * DK_RNCOL * NUM_CWG * (4 if gqa else 2),
            sdV=TILE_N * DV_RNCOL * NUM_CWG * (4 if gqa else 2),
        )
    # SMEM aliasing — faithfully model flashattn's physical buffer reuse (flash_bwd_sm100.py:
    # 749-770 + the SharedStorage struct). Each alias maps a buffer onto another with a DISJOINT
    # lifetime; nymph models it by giving the same byte_offset (like [[gdn_prefill]]
    # offs["ainv"]=offs["vnewt"]). The reuse carries a WAR that the protocol checker verifies via
    # the listed happens-before, so modeling it makes the test STRONGER, not weaker.
    #   - dK/dV epilogue reuses the dead input operands: the S/dP/dK/dV MMAs have consumed the
    #     operands before the (last-block-only) dK/dV store runs; HB = dk_done / dv_done.
    #       2-CTA: sdK←sK (S=K·Qᵀ done), sdV←sV (dP=V·dOᵀ done)   (:761-762)
    #       1-CTA: sdK←sQ (last sQ read = dK=dSᵀ·Q), sdV←sdO (last sdO read = dV=Pᵀ·dO)  (:756-757)
    #   - hd192: dS cross-CTA exchange staging reuses the dQ-reduce buffer; HB = dQaccum_empty
    #     (+ a local s2cluster source drain). (:770)
    aliases = {"sdK": ("sK" if use_2cta else "sQ"), "sdV": ("sV" if use_2cta else "sdO")}
    if is_hd192:
        aliases["sdS_xchg"] = "sdQ"
        # hd192: sQt_size = sdOt_size = 0 (flash_bwd_sm100.py:766) — the transposed dK/dP
        # B-operands time-multiplex the Q buffer (pipeline_Qt=pipeline_Q, :1266). The MMA order
        # is S(reads sQ) -> dP(reads sdOt) -> dK(reads sQt), so the sQ buffer is reloaded twice:
        # Q -> dOt -> Qt, each reload gated by the prior consumption (s_ready, then dp_ready),
        # both multicast to the pair. sQt(24KB)/sdOt(16KB) both fit sQ(24KB).
        aliases["sdOt"] = "sQ"
        aliases["sQt"] = "sQ"
    # An alias TARGET is allocated to hold the LARGER of its own data and the aliased buffer —
    # exactly flashattn's generic `sQ_alloc_bytes`/`sdO_alloc_bytes = max(operand, epilogue)`
    # (a single-stage bf16 sQ is smaller than the f32 GQA dK that reuses it, so sQ grows).
    alloc = dict(sizes)
    for alias, target in aliases.items():
        alloc[target] = max(alloc[target], sizes[alias])
    off, offs = 0, {}
    for nm, nb in alloc.items():
        if nm in aliases:
            continue  # aliased below (not bump-allocated)
        offs[nm] = off
        off += nb
    for alias, target in aliases.items():
        offs[alias] = offs[target]

    # INVARIANT-OVERRIDE I1a (F11): nymph uses a 1D PERSISTENT grid (min(SM_COUNT,num_work)
    # CTAs + for_each_task) instead of flashattn's SingleTileScheduler 3D NON-persistent grid
    # (num_block, num_head, num_batch), 1 tile/CTA (tile_scheduler.py:241). The work->tile decode
    # (task_geom: nb + n_nb*(head + H*batch)) is coverage-equivalent (num_work = n_nb·H·B); the
    # persistent form is the nymph convention.
    # 2-CTA launch: num_work cluster-tasks * cg CTAs, capped at SM_COUNT (rounded down
    # to a cg multiple so the 1D cluster schedule stays whole — fp16_bf16_gemm pattern).
    if use_2cta:
        default_launch = (max(cg, (min(SM_COUNT, num_work * cg) // cg) * cg),)
    else:
        default_launch = (min(SM_COUNT, num_work),)
    k = IRBuilder(
        "flash_bwd_sm100",
        num_warps=NUM_WARPS,
        smem_size_bytes=off,
        launch_shape=config.launch_shape or default_launch,
        cluster_shape=(cg,) if use_2cta else (1,),
    )

    # ---- args (token-major [T, H, D]; LSE/dPsum [T, H]; dQ f32) ----
    # GQA: K/V/dK/dV are KV-HEADED (Hkv heads); Q/dO/LSE/dPsum/dQ stay Q-HEADED (H heads).
    # Under GQA the G q-heads sharing a kv-head reduce-add their dK/dV into that kv-head's rows,
    # so dk_g/dv_g become FP32 accumulators (like dq_g); for MHA (G==1) they stay bf16 plain stores.
    dkv_dtype = DType.F32 if gqa else iod
    q_g = k.arg(space=MemorySpace.GMEM, dtype=iod, shape=(Tq, H, Dq))
    k_g = k.arg(space=MemorySpace.GMEM, dtype=iod, shape=(Tk, Hkv, Dq))
    v_g = k.arg(space=MemorySpace.GMEM, dtype=iod, shape=(Tk, Hkv, Dv))
    do_g = k.arg(space=MemorySpace.GMEM, dtype=iod, shape=(Tq, H, Dv))
    lse_g = k.arg(space=MemorySpace.GMEM, dtype=DType.F32, shape=(Tq, H))
    dpsum_g = k.arg(space=MemorySpace.GMEM, dtype=DType.F32, shape=(Tq, H))
    dq_g = k.arg(space=MemorySpace.GMEM, dtype=DType.F32, shape=(Tq, H, Dq))  # dQaccum (reduce-add)
    dk_g = k.arg(space=MemorySpace.GMEM, dtype=dkv_dtype, shape=(Tk, Hkv, Dq))
    dv_g = k.arg(space=MemorySpace.GMEM, dtype=dkv_dtype, shape=(Tk, Hkv, Dv))
    # VARLEN (C2): cu_seqlens for Q and K, shape (B+1,), i32 — appended LAST so the
    # fixed-length arg order / binding contract is unchanged when varlen=False. cu_*[b] is the
    # start token of sequence b in the packed buffer; cu_*[b+1]-cu_*[b] is its length.
    if config.varlen:
        cu_q_g = k.arg(space=MemorySpace.GMEM, dtype=DType.I32, shape=(B + 1,))
        cu_k_g = k.arg(space=MemorySpace.GMEM, dtype=DType.I32, shape=(B + 1,))
    else:
        cu_q_g = cu_k_g = None

    # DETERMINISTIC semaphores (appended LAST, after cu_*). One i32 counter per shared output
    # tile, zero-initialized (init 0 == the first writer's lock-value, so it passes immediately).
    #   dQ sem  (B, H, ceil(Sq/TILE_M), cg) — per (batch, q-head, m_block); the contending writers
    #           are the n_nb KV-tile CTAs, serialized in ascending n_block.  cg=1 for the 1-CTA milestone.
    #   dKV sem (B, Hkv, ceil(Sk/TILE_N), 2) — GQA only; per (batch, kv-head, n_block); the contending
    #           writers are the G q-heads of the group, serialized in ascending head % G.
    deterministic = config.deterministic
    n_mb_sem = _ceil_div(Sq, TILE_M)
    n_nb_sem = _ceil_div(Sk, TILE_N)
    if deterministic:
        dq_sem_g = k.arg(space=MemorySpace.GMEM, dtype=DType.I32, shape=(B, H, n_mb_sem, cg))
        if gqa:
            dk_sem_g = k.arg(space=MemorySpace.GMEM, dtype=DType.I32, shape=(B, Hkv, n_nb_sem, 2))
            dv_sem_g = k.arg(space=MemorySpace.GMEM, dtype=DType.I32, shape=(B, Hkv, n_nb_sem, 2))
        else:
            dk_sem_g = dv_sem_g = None
    else:
        dq_sem_g = dk_sem_g = dv_sem_g = None

    def sm(name, dt, shape):
        return k.tensor(space=MemorySpace.SMEM, dtype=dt, shape=shape, byte_offset=offs[name])

    sK = sm("sK", iod, (TILE_N, hdim))
    sV = sm("sV", iod, (TILE_N, hdimv))
    if use_2cta:
        # VALUE-CORRECT half-sized B operands (each CTA holds its own N/2 half):
        #   sQ  (S  B): q-rows split  -> (TM_H, hdim)   q-half rows, full d  (contraction)
        #   sdO (dV B): dv-cols split -> (TILE_M, HDV_H) full q rows (contract), dv-half cols
        sQ = sm("sQ", iod, (TM_H, hdim))
        sdO = sm("sdO", iod, (TILE_M, HDV_H))
    else:
        # double-buffered operands: stage row = (mb%2)*TILE_M / element offset (mb%2)*TILE_M
        sQ = sm("sQ", iod, (NSTAGE * TILE_M, hdim))
        sdO = sm("sdO", iod, (NSTAGE * TILE_M, hdimv))
    sLSE = sm("sLSE", DType.F32, (NSTAGE * TILE_M,))
    sdPsum = sm("sdPsum", DType.F32, (NSTAGE * TILE_M,))
    sdQ = sm("sdQ", DType.F32, (TILE_M, RDQ_NCOL * RDQ_NSLOT))  # RDQ_NSLOT-deep reduce buffer
    # GQA epilogue staging is f32 (reduce-add into the shared kv-head's f32 accumulators);
    # MHA staging is bf16 (plain tma_store).
    sdkv_dtype = DType.F32 if gqa else iod
    # CHUNKED staging buffers: NUM_CWG disjoint [TILE_N, RNCOL] slices (one per compute wg, NOT
    # the full [TILE_N, hdim] tile). wg wg_idx owns cols [wg_idx*RNCOL, (wg_idx+1)*RNCOL); the
    # epilogue loops num_epi_stages chunks per wg, each REUSING that wg's single slice (full-
    # drained between chunks). f32 GQA = 32 KB == bf16 sK -> the sdK→sK alias stays free.
    sdK = sm("sdK", sdkv_dtype, (TILE_N, DK_RNCOL * NUM_CWG))
    sdV = sm("sdV", sdkv_dtype, (TILE_N, DV_RNCOL * NUM_CWG))
    if use_2cta:
        # Transposed B operands carry only this CTA's N/2 half (loaded by a transposed
        # TMA descriptor, each CTA's own half at the same SMEM offset):
        #   sdOt (dP B): q-rows split  -> (TM_H, hdimv)  q-half rows, full dv (contraction)
        #   sQt  (dK B): d-cols split  -> (TILE_M, HD_H) full q rows (contract), d-half cols
        sdOt = sm("sdOt", iod, (TM_H, hdimv))  # transposed dO (dP B-operand)
        sQt = sm("sQt", iod, (TILE_M, HD_H))  # transposed Q  (dK B-operand)
        # dQ datapath (the dS exchange): full-kv operands, this CTA's q/d half.
        sKt = sm("sKt", iod, (cg * TILE_N, HD_H))  # dQ B: full kv, this CTA's d-half
        sdS_full = sm(
            "sdS_full", iod, (cg * TILE_N, TM_H)
        )  # dQ A: full kv (exchanged), this CTA's q-half
        sdS_xchg = sm(
            "sdS_xchg", iod, (TILE_N, TM_H)
        )  # dS export half (this kv-tile, peer's q-half)
        sdS = None  # 2-CTA dQ uses sdS_full; sdS is 1-CTA-only
    else:
        sdOt = sQt = sKt = sdS_full = sdS_xchg = None
        sdS = sm("sdS", iod, (TILE_N, TILE_M))  # 1-CTA dQ A-operand (dS, trans_a)

    # ---- TMEM (sub-views at the derived column offsets; all 128-row -> LANE_128) ----
    def tmem(dt, shape, col):
        return k.tensor(
            space=MemorySpace.TMEM,
            dtype=dt,
            shape=shape,
            layout=TmemLayout(TmemLayoutKind.LANE_128, col_start=col),
        )

    tmem_base = tmem(DType.F32, (128, N_COLS_TMEM), 0)
    tS = tmem(DType.F32, (TILE_N, TILE_M), d["tmem_S"])
    tP = tmem(iod, (TILE_N, TILE_M), d["tmem_P"])
    tdV = tmem(DType.F32, (TILE_N, hdimv), d["tmem_dV"])
    tdP = tmem(DType.F32, (TILE_N, TILE_M), d["tmem_dP"])
    tdQ = tmem(DType.F32, (TILE_M, hdim), d["tmem_dQ"])
    tdK = tmem(DType.F32, (TILE_N, hdim), d["tmem_dK"])
    tdS = tmem(iod, (TILE_N, TILE_M), d["tmem_dS"])

    def reg(dt, shape):
        return k.tensor(space=MemorySpace.REG, dtype=dt, shape=shape)

    # ---- mbarriers (producer/consumer pipeline) ----
    # Per-m-block operand TMA barriers (tq/tdo/tlse/tdps) get NSTAGE stages so the load warp
    # can fill mb1's stage while mma/compute still consume mb0's; load arrives stage=mb%2 and
    # mma/compute wait stage=mb%2, phase=(mb//2)%2 (=0 for n_mb<=2). tk/tv are once-per-task.
    # The single-stage cross-role barriers (s_ready/dp_ready/p_ready/ds_ready/dq_done) fire once
    # per mb; consumers wait phase=mb%2 (producer flips parity each mb). dv_done/dk_done fire
    # only on the LAST mb (dV/dK accumulate) -> single-stage, phase=0.
    # spec = (kind, arrive_count[, stages])
    bar_spec = {
        "tk": (MBarKind.TMA, 1),
        "tv": (MBarKind.TMA, 1),
        "tq": (MBarKind.TMA, 1, NSTAGE),
        "tdo": (MBarKind.TMA, 1, NSTAGE),
        "tlse": (MBarKind.TMA, 1, NSTAGE),
        "tdps": (MBarKind.TMA, 1, NSTAGE),
        "s_ready": (MBarKind.TCGEN05, 1),  # mma S -> compute
        "dp_ready": (MBarKind.TCGEN05, 1),  # mma dP -> compute
        "p_ready": (MBarKind.THREAD, 2),  # compute P -> mma (dV)   (BOTH compute wgs arrive)
        "ds_ready": (MBarKind.THREAD, 2),  # compute dS -> mma (dK/dQ) (BOTH compute wgs arrive)
        "dv_done": (MBarKind.TCGEN05, 1),
        "dk_done": (MBarKind.TCGEN05, 1),
        "dq_done": (MBarKind.TCGEN05, 1),  # mma dQ -> reduce
        "dq_free": (
            MBarKind.THREAD,
            1,
        ),  # reduce -> mma: tdQ[mb] read done, mma may write tdP[mb+1]
        # NEW-3/F5: operand EMPTY barriers (per-stage) — the consumer arrives after its LAST
        # read of a stage; the load warp WAITS *_free[st] before reloading that stage. This
        # guards the load->consume WAR that makes n_mb>2 (stage reuse at mb and mb+2) correct.
        # q_free/do_free: consumer = mma (single arrival). lse_free/dps_free: consumer = BOTH
        # compute warpgroups (2 arrivals). Mirror of GDN k_free (gdn_prefill.py).
        "q_free": (MBarKind.THREAD, 1, NSTAGE),  # mma (last read = dK) -> load
        "do_free": (MBarKind.THREAD, 1, NSTAGE),  # mma (last read = dV) -> load
        "lse_free": (MBarKind.THREAD, 2, NSTAGE),  # both compute wgs (P loop done) -> load
        "dps_free": (MBarKind.THREAD, 2, NSTAGE),  # both compute wgs (dS loop done) -> load
    }
    if use_2cta:
        # Transposed-operand TMA barriers (NSTAGE stages — single-stage 2-CTA operands, so the
        # load->mma reload WAR is serialized by qt_free/dot_free below, NOT double-buffered).
        bar_spec["tqt"] = (MBarKind.TMA, 1, NSTAGE)  # transposed Q -> dK B
        bar_spec["tdot"] = (MBarKind.TMA, 1, NSTAGE)  # transposed dO -> dP B
        bar_spec["tkt"] = (MBarKind.TMA, 1)  # transposed K -> dQ B (sKt) — once per task
        # NEW (2-CTA single-stage operand frees, multi-Q-tile): each of the 4 SEPARATE 2-CTA B
        # operands has exactly ONE reader mma, so it needs its OWN empty/free barrier (1-CTA's
        # q_free@gdK / do_free@gdV fire after the LAST of two readers — wrong for 2-CTA where
        # sQ/sQt/sdO/sdOt are read by exactly one mma each). q_free (sQ) moves to gS; do_free
        # (sdO) stays at gdV (gdV is sdO's only reader). qt_free (sQt) is arrived by gdK, dot_free
        # (sdOt) by gdP; the load warp waits each before reloading that operand for the next mb.
        bar_spec["qt_free"] = (MBarKind.THREAD, 1, NSTAGE)  # mma dK (sQt read) -> load
        bar_spec["dot_free"] = (MBarKind.THREAD, 1, NSTAGE)  # mma dP (sdOt read) -> load
        # dS cross-CTA exchange + dQ-GEMM release (map §3):
        # dS_cluster_full  (TMA-kind, tx-counting): the PEER's s2cluster of its dS half
        #   lands here (complete_tx of stage_copy_bytes); waited by the relay warp.
        # dS_cluster_leader (THREAD, count 2): relay (both CTAs) -> leader MMA; gates dQ.
        bar_spec["dS_cluster_full"] = (MBarKind.TMA, 1)
        bar_spec["dS_cluster_leader"] = (MBarKind.THREAD, 2)  # both CTAs' relays arrive
        # dS_free (multi-Q-tile WAR): sdS_full is SINGLE-buffer — gdQ(mb-1) (the leader's cluster
        # dQ MMA) reads it as the A-operand, then compute(mb) overwrites it. flashattn's
        # pipeline_dS.consumer_release (the dQ MMA, flash_bwd_sm100.py:2541) releases the dS empty
        # so the producer (compute) may re-acquire. We model it: the leader (gdQ consumer) arrives
        # dS_free on BOTH CTAs (local + peer), both compute wgs wait it before re-writing sdS_full
        # for the next mb. count=1 = the single leader arrival per CTA. (n_mb==1: never waited.)
        bar_spec["dS_free"] = (MBarKind.THREAD, 1)
        if not is_hd192 and n_mb > 1:
            # General 2-CTA multi-Q-tile: dQ OVERLAPS S (tmem_dQ=64, tmem_S=0). The pipeline
            # issues gS(mb) then gdQ(mb-1) in the SAME iteration, so gdQ(mb-1) overwrites the
            # S-overlap cols of tS(mb) that gS(mb) just wrote. compute(mb) must finish reading
            # tS(mb) into registers BEFORE gdQ(mb-1) clobbers it. s_cons = "compute consumed S(mb)"
            # (arrived by both compute wgs right after the S t2r); the leader's gdQ(mb-1) waits
            # s_cons[mb] (local + peer — the cluster MMA writes both CTAs' tdQ over both CTAs' tS).
            # count=2 = both compute wgs arrive. (hd192 is n_mb=1; 1-CTA has no S/dQ overlap.)
            bar_spec["s_cons"] = (MBarKind.THREAD, 2)
        if is_hd192:
            # hd192 ONLY: S/dP overlap TMEM cols [384,448) (dedicated layout). The compute
            # warpgroups signal "S fully read into rmem" so the dP MMA may overwrite those
            # cols — flashattn's hd192 `pipeline_S_P.consumer_release` right after the S t2r
            # (flash_bwd_sm100.py:3060-3065). count=2 = both compute wgs arrive.
            bar_spec["s_free"] = (MBarKind.THREAD, 2)
            # hd192 ONLY: sdS_xchg aliases sdQ (above). The reduce warp arrives after its
            # reduce-add drained sdQ (flash_bwd_sm100.py:3670); the exporting compute wg waits
            # before overwriting the region with the NEXT block's dS export (3258-3262). count=1
            # = the single reduce warpgroup arrives. Single-stage, phase keyed on _eph(mb) like
            # dq_free (the compute wg waits the PREVIOUS block's phase, cross-mb WAR).
            bar_spec["dQaccum_empty"] = (MBarKind.THREAD, 1)
    bars = {
        nm: k.mbar(kind=spec[0], stages=(spec[2] if len(spec) > 2 else 1))
        for nm, spec in bar_spec.items()
    }

    # ---- 2-CTA cluster gating + peer mbars (map §1; fp16_bf16_gemm pattern) ----
    # The leader (even) CTA of each pair issues the cluster-wide cta_group::2 GEMMs and
    # reads BOTH CTAs' operand tiles; before issuing it must observe the PEER's TMA-full
    # mbars (mbar_ref remote_coord=1), or the cluster MMA races the peer's loads.
    is_leader = k.ctaid_in_cluster().eq(0) if use_2cta else None
    if use_2cta:
        # peer-full waits for the leader cluster MMA: it reads BOTH CTAs' operands —
        # the TMA-loaded ones (K/V/Q/dO/Qt/dOt) AND the compute-produced TMEM-A ones
        # (P for dV, dS for dK). For the TMEM-A operands the leader must observe the
        # PEER compute warps' r2t (p_ready/ds_ready arrive) before issuing the MMA.
        _peer_names = ["tk", "tv", "tq", "tdo", "tqt", "tdot", "tkt", "p_ready", "ds_ready"]
        if is_hd192:
            # hd192: the leader's dP MMA writes BOTH CTAs' tmem_dP, overwriting the PEER's S
            # cols [384,448) — so it must also observe the PEER compute wgs' s_free (S read).
            _peer_names.append("s_free")
            if n_mb > 1:
                # hd192 multi-Q-tile: the leader's dQ MMA writes BOTH CTAs' tdQ (dedicated layout
                # dQ cols [416,512) overlap S cols [320,448) at [416,448)). gS(mb+1) rewrites tS
                # over BOTH CTAs' dQ-overlap cols, so the leader's gS must observe the PEER reduce's
                # dq_free (tdQ(mb) drained) before clobbering it — flashattn's cluster
                # pipeline_dQ.sync_object_empty.wait (flash_bwd_sm100.py:2410). (hd192's straight
                # order needs no s_cons: gdQ's dS_cluster_leader wait already orders gdQ(mb) after
                # compute(mb) read S(mb), and gS(mb+1) covers the gS-side WAR via dq_free.)
                _peer_names.append("dq_free")
        if not is_hd192 and n_mb > 1:
            # general 2-CTA multi-Q-tile: the leader's dQ MMA writes BOTH CTAs' tdQ, overwriting
            # BOTH CTAs' S-overlap cols AND BOTH CTAs' dQ accumulator — observe the PEER compute
            # wgs' s_cons (S consumed) AND the PEER reduce's dq_free (tdQ drained) too.
            _peer_names.append("s_cons")
            _peer_names.append("dq_free")
        peer_bars = {nm: k.mbar_ref(bars[nm], remote_coord=1) for nm in _peer_names}
        # Operand-free barriers (q_free/do_free/qt_free/dot_free): the LEADER's cluster MMA reads
        # BOTH CTAs' B-operand SMEM (each CTA's own N/2 half loaded at the same offset), so the
        # free signal that releases the load to reload must come from the leader and reach BOTH
        # CTAs. flashattn does this with the pipeline's cluster producer_mask consumer_release.
        # We model it as a leader-only arrive on the LOCAL barrier + a remote arrive on the peer's
        # (multi-Q-tile only — at n_mb=1 the load never reloads, so this is harmless either way).
        peer_free = {
            nm: k.mbar_ref(bars[nm], remote_coord=1)
            for nm in ("q_free", "do_free", "qt_free", "dot_free", "dS_free")
        }
    else:
        peer_bars = {}
        peer_free = {}

    sched = k.scheduler(k.task_space(grid=(num_work,), fields=("work",)))

    cta_in_cluster = k.ctaid_in_cluster() if use_2cta else 0

    with k.kernel_init(warp=0):
        k.tmem_alloc(tmem_base, n_cols=N_COLS_TMEM, cta_group=cg)
        for nm, spec in bar_spec.items():
            stg = spec[2] if len(spec) > 2 else 1
            for s in range(stg):
                k.mbarrier_init(bars[nm], count=spec[1], stage=s)

    def task_geom(task):
        # 1-CTA: `work` indexes (nb, head, batch) directly. 2-CTA: `work` indexes a
        # CLUSTER (cluster_nb, head, batch); this CTA's kv-tile = cluster_nb*cg + ctaid.
        work = task.field("work")
        c_nb = work % n_cluster
        hb = work // n_cluster
        head = hb % H
        batch = hb // H
        if use_2cta:
            nb = c_nb * cg + cta_in_cluster  # this CTA's own kv-tile
        else:
            nb = c_nb
        return batch, head, nb

    # ---- CAUSAL m_block_min tile-skip (CAUSAL_MAP §C2/§3) -------------------------
    # flashattn's bwd does NOT issue fully-above-diagonal causal tiles: for a kv-tile nb,
    # only Q-tiles mb >= m_block_min(nb) are attended (block_info.get_m_block_min_max:57-71):
    #     m_block_min(nb) = max(0, (nb*TILE_N + Sq - Sk) // TILE_M)
    #                     = max(0, (nb*TILE_N - OFF) // TILE_M),  OFF = Sk - Sq.
    # All four per-task roles restrict the m-loop start to m_block_min IN LOCKSTEP, so a
    # skipped (mb,nb) tile issues NO work on ANY role (no GEMM, no load, no dQ reduce) and
    # the cross-role mbarrier arrive/wait counts stay balanced.
    #
    # nymph realizes this WITHOUT a data-dependent trip count: the m-loop stays Python-
    # unrolled over the full range(n_mb), but each per-mb body is GUARDED by the runtime
    # predicate `mb >= mmin` (mb compile-time int, mmin a ScalarValue from the runtime nb).
    # Because mb >= 0 always, `mb >= max(0, X)` is exactly `mb >= X`, so the max() is dropped.
    #
    # Phase bookkeeping under skipping: the single-stage cross-role barriers carry a per-mb
    # phase. A skipped leading block must NOT shift the parity seen by later blocks, so every
    # phase is counted over the EXECUTED iteration index e(mb) = mb - mmin (the e-th executed
    # block is mb = mmin + e; the first executed block has e=0 -> phase 0 = the fresh barrier
    # parity). Producer and consumer compute the SAME (mb - mmin)%2 from the SAME nb, so they
    # stay balanced. The operand double-buffer stage/occupancy likewise key off e (not mb).
    #
    # NON-CAUSAL: mmin is the Python int 0, so e(mb) == mb, `_eph`/`_est` collapse to plain
    # Python ints and `_skip` is a nullcontext -> the emitted IR is byte-identical to before.
    # The skip is applied only on the 1-CTA path (the causal milestone); 2-CTA stays as-is.
    #
    # mmin may be a ScalarValue (runtime nb) OR a Python int (when the persistent scheduler
    # has a single task per CTA, `task.field` constant-folds nb to a literal — e.g. num_work==1).
    # The guard helpers below accept either: a Python-int mmin yields compile-time bool
    # predicates (emit-or-discard the body), a ScalarValue mmin yields runtime `k.if_` guards.
    OFF = Sk - Sq
    causal_skip = config.is_causal and not use_2cta

    @contextmanager
    def _discard():
        # swallow a guarded body whose compile-time predicate is False (emit nothing).
        scratch: list = []
        k._body_stack.append(scratch)
        try:
            yield
        finally:
            k._body_stack.pop()

    def _guard(pred):
        # pred is a Python bool (compile-time) or a ScalarValue (runtime).
        if isinstance(pred, (bool, int)):
            return nullcontext() if pred else _discard()
        return k.if_(pred)

    def m_block_min(nb, vg=None):
        # mmin = max(0, (nb*TILE_N - off)//TILE_M) when causal-skipping, else Python int 0. When
        # nb is a constant-folded literal AND off is compile-time this is a Python int; when nb is
        # a ScalarValue OR off is runtime (varlen) it is a runtime ScalarExpr.
        #
        # VARLEN+CAUSAL (D2): the causal offset is PER-SEQUENCE — off_b = slen_k_b − slen_q_b
        # (a runtime ScalarExpr from cu_seqlens), NOT the compile-time (Sk−Sq). For
        # self-attention off_b==0 (==compile-time OFF when Sk==Sq) but for cross-attention
        # varlen the per-seq off differs, so route the runtime vg.off_b. The runtime FloorDiv
        # rounds toward −inf (matches Python //), so mmin is consistent with the int path.
        #
        # The max(0,...) is flashattn's get_m_block_min clamp. It is REQUIRED here, not just for
        # the guard: the phase/stage helpers _eph/_est key on the EXECUTED index e = mb − mmin, so
        # a NEGATIVE mmin (off_b > 0, i.e. slen_k > slen_q) would push e past the fresh-barrier
        # parity/stage and desync the cross-role handshake. (For the GUARD alone `mb >= max(0,X)`
        # ≡ `mb >= X` since mb >= 0, so clamping the bound never changes which tiles execute.)
        # The clamp is a NO-OP for every non-varlen causal config (OFF = Sk−Sq >= 0 there), so the
        # existing causal IR is unchanged; it only bites for varlen cross-attention (off_b > 0).
        if not causal_skip:
            return 0
        off = vg.off_b if (varlen and vg is not None) else OFF
        mmin = (nb * TILE_N - off) // TILE_M
        # Preserve the compile-time int path (byte-identical IR) when mmin folded to a literal;
        # use the runtime ScalarExpr max otherwise.
        return max(0, mmin) if isinstance(mmin, int) else scalar_max(0, mmin)

    def _ge(mb, mmin):  # mb >= mmin  (Python bool if mmin is int, else ScalarValue)
        return mb >= mmin

    def _gt(mb, mmin):  # mb > mmin
        return mb > mmin

    def _eq(mb, mmin):  # mb == mmin  (Eq needs the named .eq() for a ScalarValue mmin)
        return mb == mmin if isinstance(mmin, int) else mmin.eq(mb)

    # ---- VARLEN per-sequence runtime geometry (C3/C4; gdn_prefill.py:216-222 idiom) -------
    # Under varlen the packed buffers are over-allocated to Tq=B*Sq / Tk=B*Sk rows (Sq/Sk =
    # max-for-sizing), the task grid stays sized over that upper bound (num_work unchanged), and
    # each sequence's REAL token range + per-seq tile counts are recovered at runtime from
    # cu_seqlens. Mirror of gdn: load cu_*[batch], cu_*[batch+1] into scalars at task start;
    #   base_q = cu_q[batch]      slen_q = cu_q[batch+1]-cu_q[batch]   (runtime)
    #   n_mb_b = ceil_div(slen_q, TILE_M)   n_nb_b = ceil_div(slen_k, TILE_N)   (runtime)
    # The token bases become base_q + mb*TILE_M / base_k + nb*TILE_N (vs batch*Sq / batch*Sk).
    # n_mb_b/n_nb_b bound the EXECUTED tiles: the per-mb body is gated `mb < n_mb_b` and the
    # whole-task body `nb < n_nb_b`, both in LOCKSTEP across roles (like the causal mmin skip),
    # so a skipped (mb,nb) tile issues no work on ANY role and the mbarrier counts stay balanced.
    varlen = config.varlen

    class _VG:  # per-task runtime varlen geometry (scalars valid inside the role's for_each_task)
        __slots__ = ("base_k", "base_q", "n_mb_b", "n_nb_b", "off_b", "slen_k", "slen_q")

    def varlen_geom(batch):
        # load the per-sequence cu_seqlens window into runtime scalars + derive tile counts.
        vg = _VG()
        vg.base_q = k.scalar(initial=_sl(cu_q_g, (batch,), (1,)))
        nxt_q = k.scalar(initial=_sl(cu_q_g, (batch + 1,), (1,)))
        vg.slen_q = nxt_q - vg.base_q
        vg.base_k = k.scalar(initial=_sl(cu_k_g, (batch,), (1,)))
        nxt_k = k.scalar(initial=_sl(cu_k_g, (batch + 1,), (1,)))
        vg.slen_k = nxt_k - vg.base_k
        vg.n_mb_b = (vg.slen_q + (TILE_M - 1)) // TILE_M
        vg.n_nb_b = (vg.slen_k + (TILE_N - 1)) // TILE_N
        vg.off_b = vg.slen_k - vg.slen_q  # per-seq causal OFF (= Sk-Sq); 0 for self-attn
        return vg

    def _skip(mb, mmin, vg=None, nb=None, cluster=False):
        # guard the per-mb body. Causal: `mb >= mmin`. Varlen: ALSO `mb < n_mb_b AND nb < n_nb_b`
        # (runtime). When nb >= n_nb_b (this CTA's kv-tile past the sequence) NO mb executes on
        # ANY role -> no GEMM/commit/store/reduce; only the once-per-task K/V load/wait runs
        # (balanced load<->mma, harmless overscan). When mb >= n_mb_b the partial Q range past the
        # sequence is dropped. All roles compute the SAME predicate from the SAME (nb, n_mb_b,
        # n_nb_b), so the cross-role mbarrier arrive/wait counts stay balanced (= the causal skip).
        #
        # VARLEN 2-CTA (cluster=True): a 2-CTA cluster covers cg=2 kv-tiles (nb = c_nb*cg +
        # cta_in_cluster). When a sequence has an ODD kv-tile count the leader's tile (nb=c_nb*cg)
        # is valid but the peer's tile (nb=c_nb*cg+1) is past-sequence. The per-CTA `nb < n_nb_b`
        # predicate would then DIVERGE across the pair: the leader runs the cluster MMA/compute/
        # exchange, the peer skips them -> the peer never arrives the cross-CTA handshakes
        # (p_ready/ds_ready/dS_cluster_full/q_free/...) -> the leader's cluster cta_group::2 MMA
        # waits forever -> DEADLOCK. The CLUSTER predicate `(nb - cta_in_cluster) < n_nb_b` is
        # cluster-UNIFORM (leader nb=c_nb*cg and peer nb=c_nb*cg+1 both give nb-cta_in_cluster=
        # c_nb*cg), so BOTH CTAs run the cluster MMA + compute + exchange in lockstep even when the
        # peer's own tile is past-sequence. The past tile's K/V is overscan; the compute's varlen
        # OOB mask (kv-row >= slen_k -> P=0) makes the past tile contribute 0 (harmless). The
        # per-CTA OUTPUT (dK/dV store, dQ reduce-add) stays gated by the per-CTA `nb < n_nb_b`
        # (cluster=False) so the past-sequence CTA writes nothing. For 1-CTA cta_in_cluster=0, so
        # `nb - 0 == nb` and cluster=True is a NO-OP (identical predicate).
        #
        # DENSE 2-CTA PADDING TILE (dense_pad, non-varlen): the SAME cluster-vs-tile predicate split
        # as varlen, but with the COMPILE-TIME real_n_nb in place of the runtime n_nb_b. cluster=True
        # bodies (the MMA/compute/exchange that the leader's cta_group::2 peer barriers depend on) run
        # under the cluster-UNIFORM (nb-cta_in_cluster < real_n_nb): the leader's tile (nb=c_nb*cg) and
        # the padding peer tile (nb=c_nb*cg+1) both give nb-cta_in_cluster=c_nb*cg < real_n_nb when the
        # cluster has any real tile, so BOTH CTAs run the cluster body in lockstep (no deadlock). The
        # padding tile's P is masked to 0 (compute), so it contributes 0. cluster=False bodies (the
        # per-CTA dK/dV store + dQ reduce) gate on the per-CTA (nb < real_n_nb) so the padding CTA
        # writes nothing (its rows are past Sk and would overrun dk_g/dv_g/dq_g). For the LAST odd
        # cluster: the real leader tile passes the cluster predicate (so it runs + stores), the padding
        # peer passes the cluster predicate (runs the cluster MMA, P=0) but FAILS the per-CTA tile gate
        # (stores nothing). There is no partial Q-tile here (tile-aligned Sk -> whole-OOB padding tile),
        # so n_mb is unconstrained (mb<n_mb always). dense_pad is off for cg-aligned Sk -> nullcontext.
        ctxs = []
        if causal_skip:
            ctxs.append(_guard(_ge(mb, mmin)))
        if varlen and vg is not None:
            if cluster and use_2cta:
                ctxs.append(k.if_((mb < vg.n_mb_b) & ((nb - cta_in_cluster) < vg.n_nb_b)))
            else:
                ctxs.append(k.if_((mb < vg.n_mb_b) & (nb < vg.n_nb_b)))
        elif dense_pad and nb is not None:
            if cluster:
                ctxs.append(k.if_((nb - cta_in_cluster) < real_n_nb))
            else:
                ctxs.append(k.if_(nb < real_n_nb))
        if not ctxs:
            return nullcontext()
        if len(ctxs) == 1:
            return ctxs[0]

        @contextmanager
        def _both():
            with ctxs[0]:
                with ctxs[1]:
                    yield

        return _both()

    def _eph(mb, mmin):
        # executed-iteration phase e%2 = (mb - mmin)%2. Python int when mmin is a literal.
        return (mb - mmin) % 2

    def _est(mb, mmin):
        # executed-iteration operand stage e%NSTAGE and occupancy e//NSTAGE.
        e = mb - mmin
        return e % NSTAGE, e // NSTAGE

    af = FenceKind.ASYNC_PROXY

    def gemm(
        dst, a, b, m, n, kk, done, *, trans_a=False, trans_b=False, accum=False, a_row0=0, b_row0=0
    ):
        # tcgen05.mma.cta_group::N.kind::f16 — per-instruction MMA-K=16, loop kk//16.
        # nymph default is dst[m,n]=Σ_k A[m,k]·B[n,k] (A·Bᵀ); trans_a/b slice the operand
        # as [16,m]/[16,n] (contraction-major). Mirrors gdn_prefill issue().
        # a_row0/b_row0 = base row offset into the operand (the double-buffer stage row
        # (mb%2)*TILE_M for sQ/sdO). accum=True keeps the accumulator across m-blocks
        # (dV/dK accumulate); done=None skips the consumer commit (commit only on the LAST mb).
        #
        # 2-CTA (cg=2): the M (kv) dimension is cluster-wide (m=256, Layout A — split
        # 128 rows/CTA across the pair). tcgen05.mma.cta_group::2 splits BOTH operands:
        # A by M (kv), B by N. The interpreter's read_operand_ctas reads the SAME box from
        # each CTA and CONCATENATES along axis 0 -> value-correctness requires each CTA's
        # SMEM box to already hold its OWN half (A: this CTA's kv-tile; B: this CTA's N/2).
        # The load warp now arranges exactly that (per-CTA coord-shifted half loads, no
        # multicast), so the half-width B slice below is VALUE-CORRECT for dV/dK (and S/dP).
        if use_2cta:
            # PER-CTA operand slices carry m//cg rows of A and n//cg of B; the MMA's m/n
            # params stay cluster-wide. The buffers are sized to exactly the per-CTA half,
            # so the box at offset 0 IS this CTA's own slab. (a_m = this CTA's 128 kv-rows.)
            a_m = m // cg  # per-CTA A row-extent (128)
            n_b = n // cg  # per-CTA B half-extent (model concatenates)
            # The cta_group::2 MMA is a single cluster op the leader drives; the interpreter
            # no-ops the odd CTA's issue (execute_mma early-return for ctaid&1==1), but the
            # COMMIT must be leader-only (it multicasts the done-mbar to both CTAs — a
            # per-CTA commit would double-arrive). So issue on both, commit on leader.
            for g in range(kk // 16):
                a_sl = (
                    _sl(a, (a_row0 + g * 16, 0), (16, a_m))
                    if trans_a
                    else _sl(a, (a_row0, g * 16), (a_m, 16))
                )
                b_sl = (
                    _sl(b, (b_row0 + g * 16, 0), (16, n_b))
                    if trans_b
                    else _sl(b, (b_row0, g * 16), (n_b, 16))
                )
                k.tcgen05_mma(
                    dst,
                    a_sl,
                    b_sl,
                    m=m,
                    n=n,
                    k=16,
                    accum=(accum or g != 0),
                    trans_a=trans_a,
                    trans_b=trans_b,
                    cta_group=cg,
                )
            if done is not None:
                with k.if_(is_leader):
                    k.tcgen05_commit(bars[done], cta_group=cg, multicast_cta_mask=0b11)
            return
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
                accum=(accum or g != 0),
                trans_a=trans_a,
                trans_b=trans_b,
            )
        if done is not None:
            k.tcgen05_commit(bars[done])

    def gemm_acc(mb, mmin, *args, **kwargs):
        # Accumulating GEMM (dV/dK) whose zero-init vs accumulate depends on whether `mb` is
        # the FIRST EXECUTED m-block. Non-causal: accum=(mb>0) (compile-time, single issue).
        # CAUSAL skip: the first executed block is mb==mmin (runtime), so emit a runtime split
        # — accum=False guarded `mb==mmin` (zero-init), accum=True guarded `mb>mmin`. The two
        # guards are mutually exclusive within the enclosing `mb>=mmin`, so exactly one issue
        # (and exactly one done-commit) runs per executed block. This mirrors flashattn's
        # runtime `zero_init = not accumulate_dK` (accumulate_dK flips after the first iter).
        if not causal_skip:
            gemm(*args, accum=(mb > 0), **kwargs)
            return
        # mmin may be a Python int (literal nb) or a ScalarValue (runtime nb). _guard/_eq/_gt
        # collapse to a single compile-time-selected issue when mmin is a literal, or a runtime
        # split (zero-init guarded mb==mmin, accumulate guarded mb>mmin) when mmin is runtime.
        with _guard(_eq(mb, mmin)):
            gemm(*args, accum=False, **kwargs)  # first executed block -> zero-init
        with _guard(_gt(mb, mmin)):
            gemm(*args, accum=True, **kwargs)  # subsequent executed blocks -> accumulate

    # ============== load warp (13): TMA Q/K/V/dO + bulk LSE/dPsum ==============
    with k.role(warp=LOAD_WARP):
        with k.for_each_task(sched) as task:
            batch, head, nb = task_geom(task)
            # GQA: K/V are shared across the G q-heads of a group -> load from the kv-head
            # head_kv = head // G (== head for MHA). Q/dO/LSE/dPsum stay per-q-head (`head`).
            head_kv = head // G if gqa else head
            vg = varlen_geom(batch) if varlen else None  # VARLEN: per-seq runtime geometry
            mmin = m_block_min(nb, vg)  # CAUSAL: per-task m-loop start (Python 0 if not)
            ktok = (vg.base_k if varlen else batch * Sk) + nb * TILE_N
            # K, V (the n-tile) — loaded once per task. NB: when nb >= n_nb_b (this CTA's kv-tile
            # is past the sequence) the K/V load/wait still runs on BOTH load+mma (balanced); only
            # the OUTPUT-producing per-mb bodies are skipped (_skip gates mb<n_mb_b AND nb<n_nb_b),
            # so the unused K/V read is harmless (TMA clamps the buffer-end overscan).
            k.mbarrier_arrive_expect_tx(bars["tk"], bytes=TILE_N * hdim * 2)
            k.tma_load(
                _sl(sK, (0, 0), (TILE_N, hdim)),
                k_g,
                mbar=bars["tk"],
                bytes=TILE_N * hdim * 2,
                coords=(ktok, head_kv, 0),
                shape=(TILE_N, hdim),
                gmem_shape=(TILE_N, 1, hdim),
            )
            k.mbarrier_arrive_expect_tx(bars["tv"], bytes=TILE_N * hdimv * 2)
            k.tma_load(
                _sl(sV, (0, 0), (TILE_N, hdimv)),
                v_g,
                mbar=bars["tv"],
                bytes=TILE_N * hdimv * 2,
                coords=(ktok, head_kv, 0),
                shape=(TILE_N, hdimv),
                gmem_shape=(TILE_N, 1, hdimv),
            )
            if use_2cta:
                # dQ B-operand sKt = K transposed (full cluster kv, this CTA's d-half). K is the
                # kv (SAME for all mb), so flashattn loads it ONCE per task (pipeline_Kt), not
                # per-mb — reloading it inside the m-loop is a WAR that deadlocks at n_mb>1. Hoist
                # it next to sK/sV. dQ B = K, N-axis = d (contraction = kv): FULL cluster kv (both
                # tiles), this CTA's own d-half [v*HD_H,(v+1)*HD_H) -> sKt (cg*TILE_N, HD_H), over
                # the WHOLE 256-kv slab (cluster kv base = the pair's first tile token). GQA: K is
                # kv-headed (shared across the group) -> head_kv (q-headed operands use `head`).
                cl_ktok = (vg.base_k if varlen else batch * Sk) + (nb - cta_in_cluster) * TILE_N
                k.mbarrier_arrive_expect_tx(bars["tkt"], bytes=cg * TILE_N * HD_H * 2)
                k.tma_load(
                    _sl(sKt, (0, 0), (cg * TILE_N, HD_H)),
                    k_g,
                    mbar=bars["tkt"],
                    bytes=cg * TILE_N * HD_H * 2,
                    coords=(cl_ktok, head_kv, cta_in_cluster * HD_H),
                    shape=(cg * TILE_N, HD_H),
                    gmem_shape=(cg * TILE_N, 1, HD_H),
                )
            for mb in range(n_mb):
                with _skip(mb, mmin, vg, nb, cluster=True):  # CAUSAL: skip above-diagonal Q-tiles;
                    # VARLEN: skip mb>=n_mb_b / nb>=n_nb_b (past-sequence tiles). cluster=True: the load
                    # is balanced with the cluster MMA (K/V/operand loads feed the cta_group::2 GEMMs),
                    # so under 2-CTA both CTAs of a cluster load in lockstep (cluster predicate); a peer
                    # whose own kv-tile is past-sequence still loads its overscan operands so the
                    # cluster MMA's per-mb waits stay balanced.
                    qtok = (vg.base_q if varlen else batch * Sq) + mb * TILE_M
                    # operand stage/occupancy key off the EXECUTED iteration e=mb-mmin (CAUSAL),
                    # so a skipped leading block does not shift the double-buffer parity.
                    st, occ = _est(mb, mmin)  # operand SMEM/barrier stage, stage occupancy
                    rM = st * TILE_M  # stage row offset (2D operands) / elem offset (1D)
                    # NEW-3/F5: wait the consumer freed this stage before reloading it. phase
                    # (occ+1)%2 makes the FIRST fill of each stage (occ=0 -> phase=1, parity=0)
                    # proceed immediately; later reloads block on the prior fill's consume. The 4
                    # operand stages share one stage index, so one wait per operand barrier suffices.
                    k.mbarrier_wait(bars["q_free"], stage=st, phase=(occ + 1) % 2)
                    if use_2cta:
                        # S B = Q, N-axis = q. Each CTA loads its OWN N/2 q-rows
                        # [v*TM_H,(v+1)*TM_H) into sQ[0:TM_H] (the same SMEM box on both CTAs);
                        # the cluster MMA concat over the pair reconstructs the full q-tile.
                        k.mbarrier_arrive_expect_tx(bars["tq"], bytes=TM_H * hdim * 2, stage=st)
                        k.tma_load(
                            _sl(sQ, (0, 0), (TM_H, hdim)),
                            q_g,
                            mbar=bars["tq"],
                            mbar_stage=st,
                            bytes=TM_H * hdim * 2,
                            coords=(qtok + cta_in_cluster * TM_H, head, 0),
                            shape=(TM_H, hdim),
                            gmem_shape=(TM_H, 1, hdim),
                        )
                    else:
                        k.mbarrier_arrive_expect_tx(bars["tq"], bytes=TILE_M * hdim * 2, stage=st)
                        k.tma_load(
                            _sl(sQ, (rM, 0), (TILE_M, hdim)),
                            q_g,
                            mbar=bars["tq"],
                            mbar_stage=st,
                            bytes=TILE_M * hdim * 2,
                            coords=(qtok, head, 0),
                            shape=(TILE_M, hdim),
                            gmem_shape=(TILE_M, 1, hdim),
                        )
                    k.mbarrier_wait(bars["do_free"], stage=st, phase=(occ + 1) % 2)
                    if use_2cta:
                        # dV B = dO, N-axis = dv (head dim); contraction = q. Each CTA loads its
                        # OWN N/2 dv-cols [v*HDV_H,(v+1)*HDV_H) for the FULL q-tile into the same
                        # sdO box; the concat over the pair gives the full dv extent.
                        k.mbarrier_arrive_expect_tx(bars["tdo"], bytes=TILE_M * HDV_H * 2, stage=st)
                        k.tma_load(
                            _sl(sdO, (0, 0), (TILE_M, HDV_H)),
                            do_g,
                            mbar=bars["tdo"],
                            mbar_stage=st,
                            bytes=TILE_M * HDV_H * 2,
                            coords=(qtok, head, cta_in_cluster * HDV_H),
                            shape=(TILE_M, HDV_H),
                            gmem_shape=(TILE_M, 1, HDV_H),
                        )
                    else:
                        k.mbarrier_arrive_expect_tx(bars["tdo"], bytes=TILE_M * hdimv * 2, stage=st)
                        k.tma_load(
                            _sl(sdO, (rM, 0), (TILE_M, hdimv)),
                            do_g,
                            mbar=bars["tdo"],
                            mbar_stage=st,
                            bytes=TILE_M * hdimv * 2,
                            coords=(qtok, head, 0),
                            shape=(TILE_M, hdimv),
                            gmem_shape=(TILE_M, 1, hdimv),
                        )
                    # F12: LSE/dPsum are 1D per-row vectors. flashattn loads them via a non-tensor
                    # 1D bulk copy (CopyBulkG2SOp -> cp.async.bulk.shared::cluster.global, INSTR.md
                    # row 27); nymph has no separate non-tensor-bulk primitive, so tma_load models the
                    # same bulk-async G2S transfer (equivalent bytes/ordering, tensor-descriptor form).
                    k.mbarrier_wait(bars["lse_free"], stage=st, phase=(occ + 1) % 2)
                    k.mbarrier_arrive_expect_tx(bars["tlse"], bytes=TILE_M * 4, stage=st)
                    k.tma_load(
                        _sl(sLSE, (rM,), (TILE_M,)),
                        lse_g,
                        mbar=bars["tlse"],
                        mbar_stage=st,
                        bytes=TILE_M * 4,
                        coords=(qtok, head),
                        shape=(TILE_M,),
                        gmem_shape=(TILE_M, 1),
                    )
                    k.mbarrier_wait(bars["dps_free"], stage=st, phase=(occ + 1) % 2)
                    k.mbarrier_arrive_expect_tx(bars["tdps"], bytes=TILE_M * 4, stage=st)
                    k.tma_load(
                        _sl(sdPsum, (rM,), (TILE_M,)),
                        dpsum_g,
                        mbar=bars["tdps"],
                        mbar_stage=st,
                        bytes=TILE_M * 4,
                        coords=(qtok, head),
                        shape=(TILE_M,),
                        gmem_shape=(TILE_M, 1),
                    )
                    if use_2cta:
                        # Transposed B operands sdOt (dP) / sQt (dK): a SEPARATE TMA descriptor
                        # fills the buffer with THIS CTA's own N/2 half (the cluster MMA concat
                        # over the pair reconstructs the full N). No multicast — each CTA loads
                        # its own slab into the same SMEM offset.
                        #   dP B = dO, N-axis = q (contraction = dv): split q-rows
                        #     [v*TM_H,(v+1)*TM_H), full dv -> sdOt (TM_H, hdimv).
                        if is_hd192:
                            # hd192: sdOt aliases the sQ buffer (Q->dOt time-mux). The S MMA has read
                            # sQ once s_ready is committed (multicast to the pair), so the buffer is
                            # free to reload as dOt. sQ is loaded above (this mb) before this wait, and
                            # the leader's gS reads both CTAs' sQ -> the multicast s_ready frees both.
                            # (hd192 is n_mb=1 only; the multi-Q-tile dot_free WAR below is skipped.)
                            k.mbarrier_wait(bars["s_ready"], phase=_eph(mb, mmin))
                        else:
                            # NEW (multi-Q-tile): sdOt is single-stage; wait gdP freed it before
                            # reloading for the next mb. Same phase pattern as q_free/do_free above.
                            k.mbarrier_wait(bars["dot_free"], stage=st, phase=(occ + 1) % 2)
                        k.mbarrier_arrive_expect_tx(bars["tdot"], bytes=TM_H * hdimv * 2, stage=st)
                        k.tma_load(
                            _sl(sdOt, (0, 0), (TM_H, hdimv)),
                            do_g,
                            mbar=bars["tdot"],
                            mbar_stage=st,
                            bytes=TM_H * hdimv * 2,
                            coords=(qtok + cta_in_cluster * TM_H, head, 0),
                            shape=(TM_H, hdimv),
                            gmem_shape=(TM_H, 1, hdimv),
                        )
                        #   dK B = Q, N-axis = d (contraction = q): full q-rows, split d-cols
                        #     [v*HD_H,(v+1)*HD_H) -> sQt (TILE_M, HD_H).
                        if is_hd192:
                            # hd192: sQt aliases the same sQ buffer (dOt->Qt time-mux). The dP MMA has
                            # read sdOt once dp_ready is committed (multicast), freeing it for Qt.
                            k.mbarrier_wait(bars["dp_ready"], phase=_eph(mb, mmin))
                        else:
                            # NEW (multi-Q-tile): sQt is single-stage; wait gdK freed it before reload.
                            k.mbarrier_wait(bars["qt_free"], stage=st, phase=(occ + 1) % 2)
                        k.mbarrier_arrive_expect_tx(bars["tqt"], bytes=TILE_M * HD_H * 2, stage=st)
                        k.tma_load(
                            _sl(sQt, (0, 0), (TILE_M, HD_H)),
                            q_g,
                            mbar=bars["tqt"],
                            mbar_stage=st,
                            bytes=TILE_M * HD_H * 2,
                            coords=(qtok, head, cta_in_cluster * HD_H),
                            shape=(TILE_M, HD_H),
                            gmem_shape=(TILE_M, 1, HD_H),
                        )

    # ============== mma warp (12): the 5 tcgen05 GEMMs ==============
    # F9 — per-GEMM operand major modes (flashattn _get_tiled_mma @268-320). nymph's
    # gemm() reproduces (A_major, B_major) via the trans_a/trans_b slice form:
    #   S  : A=K-major(sK)  B=K-major(sQ)            -> no trans  (contract over d)
    #   dP : A=K-major(sV)  B=K-major(sdO)           -> no trans  (contract over d)
    #   dV : A=K-major(tP)  B=MN-major(sdO)          -> trans_b   (contract over q=TILE_M)
    #   dK : A=K-major(tdS) B=MN-major(sQ)           -> trans_b   (contract over q)
    #   dQ : A=MN-major(sdS) B=MN-major(sK)          -> trans_a+trans_b (contract over kv=TILE_N)
    # (dV/dK A from TMEM = use_smem_dS_for_mma_dK=False; dQ A=dS from SMEM sdS.)
    # F1 — faithful 1-CTA MMA issue order (flashattn flash_bwd_sm100.py @2588-2700):
    #   PROLOGUE (mb 0):          S[0] ; dP[0] ; dV[0](zero_init)
    #   MAIN LOOP (mb 1..n_mb-1): S[mb] ; dK[mb-1] ; dQ[mb-1] ; dP[mb] ; dV[mb](accum)
    #   TAIL (mb n_mb-1):         dK[n_mb-1] ; dQ[n_mb-1]
    # S/dP/dV are issued one m-block AHEAD; dK/dQ consume the PREVIOUS block's dS (a 1-iter
    # software-pipeline skew — flashattn handle_Q/handle_Q_next @2635/2648). The dependency
    # barriers are UNCHANGED (s_ready/dp_ready/p_ready/ds_ready/dv_done/dk_done/dq_done/dq_free).
    # Non-causal: each GEMM keeps the per-block PHASE = (its block index)%2 regardless of issue
    # position; dV/dK accumulate from their 2nd issue (mb>=1). For n_mb=1 this reduces to
    # S,dP,dV,dK,dQ (the flat single-tile order) — main loop has 0 iters, so single-tile tests
    # are unaffected.
    # CAUSAL skip (1-CTA): each gS/gdP/gdV/gdK/gdQ body is wrapped in `with _skip(mb, mmin)` so
    # a fully-above-diagonal Q-tile (mb < mmin) issues NO GEMM/commit. Phases count the EXECUTED
    # index e=mb-mmin (so the first executed block hits the fresh barrier parity 0), and the
    # zero-init-vs-accumulate split for dV/dK keys off the runtime "first executed" predicate
    # (mb==mmin) via gemm_acc — flashattn's `zero_init = not accumulate_dK`. The prologue calls
    # gS(0)/gdP(0)/gdV(0) and tail gdK(n_mb-1)/gdQ(n_mb-1) stay as-is; the per-body _skip guard
    # makes the FIRST EXECUTED block (mb==mmin) act as the prologue at runtime wherever it lands.
    # mma_m = the M (kv) extent of S/dP/dV/dK: cluster-wide (256) under 2-CTA, else TILE_N.
    mma_m = cg * TILE_N

    with k.role(warp=MMA_WARP):
        with k.for_each_task(sched) as task:
            batch, head, nb = task_geom(task)
            vg = varlen_geom(batch) if varlen else None  # VARLEN: per-seq runtime geometry
            mmin = m_block_min(nb, vg)  # CAUSAL: per-task m-loop start (Python 0 if not)
            k.mbarrier_wait(bars["tk"], phase=0)
            k.mbarrier_wait(bars["tv"], phase=0)
            if use_2cta:
                # Leader reads BOTH CTAs' K/V (A-operands) — observe the peer's K/V loads.
                with k.if_(is_leader):
                    k.mbarrier_wait(peer_bars["tk"], phase=0)
                    k.mbarrier_wait(peer_bars["tv"], phase=0)
                # sKt (dQ B-operand = K transposed) is loaded ONCE per task (hoisted next to
                # sK/sV) — wait it ONCE here (phase=0), not per-mb. The dQ MMA is cta_group::2
                # with B split by N=d across the pair, so the leader reads BOTH CTAs' sKt halves
                # -> also wait the peer's tkt. (Replaces the per-mb tkt wait that was in gdQ.)
                k.mbarrier_wait(bars["tkt"], phase=0)
                with k.if_(is_leader):
                    k.mbarrier_wait(peer_bars["tkt"], phase=0)

            # CAUSAL: every per-mb body is wrapped in `with _skip(mb, mmin)` so a skipped
            # (above-diagonal) Q-tile issues no GEMM/commit; phases count the executed index.
            def gS(mb):  # S[kv,q] = Σ_d K·Qᵀ  -> tS  (waits tq[mb], commits s_ready[mb])
                with _skip(
                    mb, mmin, vg, nb, cluster=True
                ):  # 2-CTA: cluster MMA -> cluster predicate
                    st, occ = _est(mb, mmin)
                    rM = st * TILE_M
                    if is_hd192:
                        # hd192: dQ overlaps S at TMEM cols [416,448) (dedicated layout). Before S
                        # overwrites them the previous block's dQ must be drained by the reduce warp
                        # — flashattn's `pipeline_dQ.sync_object_empty.wait` before the S MMA
                        # (flash_bwd_sm100.py:2410). The FIRST executed block has no prior dQ. In the
                        # straight per-mb order this gS(mb) dq_free wait ALSO covers the gdQ accumulator
                        # WAR (gS(mb) precedes gdQ(mb), so gdQ(mb)'s tdQ write is transitively after the
                        # reduce drained tdQ(mb-1)) — so hd192's gdQ needs no separate dq_free/s_cons
                        # wait. The leader's cluster dQ MMA wrote BOTH CTAs' tdQ -> the leader's gS must
                        # also observe the PEER reduce's dq_free before clobbering the peer's overlap.
                        if causal_skip:
                            with _guard(_gt(mb, mmin)):
                                k.mbarrier_wait(bars["dq_free"], phase=_eph(mb - 1, mmin))
                        elif mb > 0:
                            k.mbarrier_wait(bars["dq_free"], phase=_eph(mb - 1, mmin))
                            if n_mb > 1:
                                with k.if_(is_leader):
                                    k.mbarrier_wait(peer_bars["dq_free"], phase=_eph(mb - 1, mmin))
                    elif use_2cta and mb >= 2:
                        # General 2-CTA: dQ overlaps S (tmem_dQ=64, tmem_S=0 -> cols [64,128) shared);
                        # flashattn gates the S MMA on pipeline_dQ.sync_object_empty (flash_bwd_sm100.py
                        # :2516). The 2-CTA MMA order swaps dQ past dP (gS(mb);gdK(mb-1);gdP(mb);
                        # gdQ(mb-1);gdV(mb)), so the dQ occupying tdQ when gS(mb) runs is gdQ(mb-2) from
                        # the PREVIOUS iteration (not mb-1, which is issued later this iter). The first
                        # two S blocks have no prior dQ to drain: gS(0) is the prologue, gS(1) runs
                        # before any gdQ. So only mb>=2 waits, at the parity the reduce produces for
                        # the (mb-2)-th drained dQ block = _eph(mb-2). (Causal 2-CTA is out of scope.)
                        # This single dq_free gate ALSO covers the tdQ accumulator WAR (gdQ(mb-1) writes
                        # tdQ, last read by reduce): gS(mb) is before gdQ(mb-1) in program order, so the
                        # gdQ(mb-1) write is transitively after the reduce drain (flashattn's single
                        # pipeline_dQ.empty gate). The leader writes BOTH CTAs' tdQ -> wait peer dq_free.
                        k.mbarrier_wait(bars["dq_free"], phase=_eph(mb - 2, mmin))
                        with k.if_(is_leader):
                            k.mbarrier_wait(peer_bars["dq_free"], phase=_eph(mb - 2, mmin))
                    k.mbarrier_wait(bars["tq"], stage=st, phase=occ % 2)
                    if use_2cta:
                        with k.if_(is_leader):
                            k.mbarrier_wait(peer_bars["tq"], phase=occ % 2)
                    gemm(
                        _sl(tS, (0, 0), (TILE_N, TILE_M)),
                        sK,
                        sQ,
                        mma_m,
                        TILE_M,
                        hdim,
                        "s_ready",
                        b_row0=rM,
                    )
                    if use_2cta and not is_hd192:
                        # general 2-CTA: sQ (S B-operand) is read ONLY by gS -> gS frees its stage.
                        # (1-CTA's sQ is read by gS AND gdK, so 1-CTA frees q_free at gdK after the LAST
                        # reader. hd192 TIME-MUXES sQ as Q->dOt->Qt within one mb; sQ's LAST user is gdK
                        # (reads sQt=sQ), so hd192 frees q_free at gdK, NOT here — else the next mb's Q
                        # load could clobber sQt(mb) while gdK(mb) is still reading it.)
                        # The leader's cluster MMA read BOTH CTAs' sQ -> the leader frees BOTH (local +
                        # peer multicast); a per-CTA arrive would let cta1 reload sQ before the leader
                        # finished reading it (cross-CTA WAR).
                        with k.if_(is_leader):
                            k.mbarrier_arrive(bars["q_free"], stage=st)
                            k.mbarrier_arrive(peer_free["q_free"], stage=st)

            def gdP(mb):  # dP[kv,q] = Σ_d V·dOᵀ -> tdP. 2-CTA B = sdOt (transposed dO).
                with _skip(
                    mb, mmin, vg, nb, cluster=True
                ):  # 2-CTA: cluster MMA -> cluster predicate
                    st, occ = _est(mb, mmin)
                    rM = st * TILE_M
                    if is_hd192:
                        # hd192: dP overwrites S's overlapping cols [384,448), so it waits the compute
                        # warpgroups' s_free (S fully read) — flashattn's `pipeline_S_P.sync_object_
                        # empty.wait` before the dP MMA (flash_bwd_sm100.py:2424). The dP/dQ-overlap
                        # WAR (tdP[mb] vs prev dQ) is already covered: gS waited dq_free this block, so
                        # the prior dQ is drained before dP issues.
                        k.mbarrier_wait(bars["s_free"], phase=_eph(mb, mmin))
                        # the leader's cta_group::2 dP MMA writes the PEER CTA's tmem_dP too, which
                        # overlaps the PEER's S cols — wait the peer compute wgs' s_free as well.
                        with k.if_(is_leader):
                            k.mbarrier_wait(peer_bars["s_free"], phase=_eph(mb, mmin))
                    elif not use_2cta:
                        # 1-CTA only: dP and dQ OVERLAP TMEM (tmem_dQ == tmem_dP), so gdP(mb) must not
                        # overwrite tdQ(mb-1) before the reduce drained it. dq_free gates against the
                        # PREVIOUS executed block's dQ read of tdQ. The FIRST executed block has no prior
                        # dQ, so it does not wait. Non-causal: mb>0 (compile-time). CAUSAL: the runtime
                        # predicate mb>mmin, at the previous executed block's phase _eph(mb-1).
                        # (General 2-CTA: dQ overlaps S, not dP -> the dq_free gate is on gS, not here.)
                        if causal_skip:
                            with _guard(_gt(mb, mmin)):
                                k.mbarrier_wait(bars["dq_free"], phase=_eph(mb - 1, mmin))
                        elif mb > 0:
                            k.mbarrier_wait(bars["dq_free"], phase=_eph(mb - 1, mmin))
                    k.mbarrier_wait(bars["tdo"], stage=st, phase=occ % 2)
                    if use_2cta:
                        k.mbarrier_wait(bars["tdot"], phase=occ % 2)
                        with k.if_(is_leader):
                            k.mbarrier_wait(peer_bars["tdo"], phase=occ % 2)
                            k.mbarrier_wait(peer_bars["tdot"], phase=occ % 2)
                        # dP B-operand from the dedicated transposed buffer sdOt (rM=0, single).
                        gemm(
                            _sl(tdP, (0, 0), (TILE_N, TILE_M)),
                            sV,
                            sdOt,
                            mma_m,
                            TILE_M,
                            hdimv,
                            "dp_ready",
                        )
                        # 2-CTA: sdOt (dP B-operand) is read ONLY by gdP -> the leader frees BOTH CTAs'
                        # stage (local + peer multicast) so the load warp may reload sdOt for the next
                        # mb (single-stage WAR serialize; leader read both CTAs' sdOt halves).
                        with k.if_(is_leader):
                            k.mbarrier_arrive(bars["dot_free"], stage=st)
                            k.mbarrier_arrive(peer_free["dot_free"], stage=st)
                    else:
                        gemm(
                            _sl(tdP, (0, 0), (TILE_N, TILE_M)),
                            sV,
                            sdO,
                            TILE_N,
                            TILE_M,
                            hdimv,
                            "dp_ready",
                            b_row0=rM,
                        )

            def gdV(mb):  # dV[kv,dv] = Σ_q Pᵀ·dO -> tdV (accum after 1st executed; waits p_ready)
                with _skip(
                    mb, mmin, vg, nb, cluster=True
                ):  # 2-CTA: cluster MMA -> cluster predicate
                    st, occ = _est(mb, mmin)
                    rM = st * TILE_M
                    k.mbarrier_wait(bars["p_ready"], phase=_eph(mb, mmin))
                    if use_2cta:
                        # leader reads the PEER's P (TMEM-A from both CTAs) -> wait peer p_ready.
                        with k.if_(is_leader):
                            k.mbarrier_wait(peer_bars["p_ready"], phase=_eph(mb, mmin))
                    gemm_acc(
                        mb,
                        mmin,
                        _sl(tdV, (0, 0), (TILE_N, hdimv)),
                        tP,
                        sdO,
                        mma_m,
                        hdimv,
                        TILE_M,
                        "dv_done",
                        trans_b=True,
                        b_row0=rM,
                    )
                    # NEW-3/F5: gdV is the LAST reader of sdO[st] (gdP read it first) -> free the stage.
                    if use_2cta:
                        # leader read BOTH CTAs' sdO -> leader frees both (local + peer multicast).
                        with k.if_(is_leader):
                            k.mbarrier_arrive(bars["do_free"], stage=st)
                            k.mbarrier_arrive(peer_free["do_free"], stage=st)
                    else:
                        k.mbarrier_arrive(bars["do_free"], stage=st)

            def gdK(mb):  # dK[kv,d] = Σ_q dSᵀ·Q -> tdK (accum after 1st executed; waits ds_ready)
                # dK uses THIS CTA's dS rows (no cross-CTA dep) — buildable in stage 1.
                # 2-CTA B = sQt (transposed Q).
                with _skip(
                    mb, mmin, vg, nb, cluster=True
                ):  # 2-CTA: cluster MMA -> cluster predicate
                    st, occ = _est(mb, mmin)
                    rM = st * TILE_M
                    k.mbarrier_wait(bars["ds_ready"], phase=_eph(mb, mmin))
                    if use_2cta:
                        k.mbarrier_wait(bars["tqt"], phase=occ % 2)
                        with k.if_(is_leader):
                            k.mbarrier_wait(peer_bars["tqt"], phase=occ % 2)
                            # leader reads the PEER's dS (TMEM-A from both CTAs) -> wait peer ds_ready.
                            k.mbarrier_wait(peer_bars["ds_ready"], phase=_eph(mb, mmin))
                        gemm_acc(
                            mb,
                            mmin,
                            _sl(tdK, (0, 0), (TILE_N, hdim)),
                            tdS,
                            sQt,
                            mma_m,
                            hdim,
                            TILE_M,
                            "dk_done",
                            trans_b=True,
                        )
                        if is_hd192:
                            # hd192: sQt ALIASES sQ (time-mux Q->dOt->Qt). gdK is the LAST sQ user, so
                            # gdK frees q_free here (general 2-CTA frees it at gS — separate buffers).
                            # The next mb's Q load waits q_free before reloading sQ. Leader frees BOTH
                            # CTAs (local + peer multicast); a per-CTA arrive would let the peer reload
                            # sQ before the leader finished reading sQt. (qt_free aliases this WAR for
                            # hd192 — sQt is sQ — so the separate qt_free below would be redundant; the
                            # load's hd192 Qt reload waits dp_ready, not qt_free, so qt_free is unused.)
                            with k.if_(is_leader):
                                k.mbarrier_arrive(bars["q_free"], stage=st)
                                k.mbarrier_arrive(peer_free["q_free"], stage=st)
                        else:
                            # general 2-CTA: sQt (dK B-operand) is read ONLY by gdK -> the leader frees
                            # BOTH CTAs' stage (local + peer multicast) so the load warp may reload sQt
                            # for the next mb (single-stage WAR serialize; leader read both halves).
                            with k.if_(is_leader):
                                k.mbarrier_arrive(bars["qt_free"], stage=st)
                                k.mbarrier_arrive(peer_free["qt_free"], stage=st)
                    else:
                        gemm_acc(
                            mb,
                            mmin,
                            _sl(tdK, (0, 0), (TILE_N, hdim)),
                            tdS,
                            sQ,
                            TILE_N,
                            hdim,
                            TILE_M,
                            "dk_done",
                            trans_b=True,
                            b_row0=rM,
                        )
                        # NEW-3/F5 (1-CTA): gdK is the LAST reader of sQ[st] (gS read it first) ->
                        # free the stage. (2-CTA's sQ is read only by gS, which frees q_free itself.)
                        k.mbarrier_arrive(bars["q_free"], stage=st)

            def gdQ(
                mb,
            ):  # dQ[q,d] = Σ_kv dS·K -> tdQ (consumes ds_ready[mb] via gdK; commits dq_done[mb])
                with _skip(
                    mb, mmin, vg, nb, cluster=True
                ):  # 2-CTA: cluster MMA -> cluster predicate
                    if use_2cta:
                        # 2-CTA dQ (map §3): the FULL cluster-wide kv (256) dS lands in sdS_full
                        # via the cross-CTA exchange (each CTA holds its own q-half, full kv). The
                        # leader gates the cluster MMA on dS_cluster_leader (relay -> both halves
                        # present). dQ = dS·K is cta_group::2, m=TILE_M=128 (Layout B): A=sdS_full
                        # split by M(q) -> 64 q-rows/CTA, B=sKt split by N(d) -> HD_H d-cols/CTA,
                        # with K=kv looped over the FULL 256 (kk=cg*TILE_N). The gemm() helper's
                        # 2-CTA branch slices a_m=m//cg=64 / n_b=n//cg=64 per CTA exactly.
                        # sKt is loaded ONCE per task (K is the same for all mb) and waited ONCE at
                        # task start in the mma warp — no per-mb tkt wait here (that was a WAR that
                        # deadlocked at n_mb>1).
                        if not is_hd192 and n_mb > 1 and mb > 0:
                            # tdQ accumulator WAR: gdQ(mb) overwrites tdQ, last read by the reduce warp
                            # draining tdQ(mb-1). Wait dq_free[mb-1]. The leader writes BOTH CTAs' tdQ ->
                            # wait local + peer dq_free (each CTA's reduce drains its own 64 q-rows).
                            # gdQ(0) has no prior reduce. dq_free is single-stage but at most ONE dQ is
                            # outstanding at the reduce at a time (the reduce drains tdQ(mb-1) before
                            # gdQ(mb) can overwrite), so the parity is unambiguous per (mb-1) drain.
                            # (hd192: the STRAIGHT order has gS(mb) before gdQ(mb), and gS(mb) already
                            # waits dq_free[mb-1], so gdQ(mb)'s tdQ write is transitively after the
                            # reduce drain — no separate dq_free wait here. The dQ/S-overlap WAR is
                            # likewise covered: gdQ's dS_cluster_leader wait orders it after compute(mb)
                            # read S(mb). So hd192 skips both the dq_free and the s_cons wait below.)
                            k.mbarrier_wait(bars["dq_free"], phase=_eph(mb - 1, mmin))
                            with k.if_(is_leader):
                                k.mbarrier_wait(peer_bars["dq_free"], phase=_eph(mb - 1, mmin))
                        if not is_hd192 and n_mb > 1 and mb + 1 < n_mb:
                            # dQ OVERLAPS S — gdQ(mb) writes tdQ over the S-overlap cols of tS(mb+1)
                            # that gS(mb+1) (issued just before this gdQ in the same iteration) wrote.
                            # Wait s_cons[mb+1] (both compute wgs read tS(mb+1) into rmem) so this
                            # overwrite is after that read. The leader writes BOTH CTAs' tdQ -> wait
                            # local + peer s_cons. The tail gdQ(n_mb-1) has no subsequent gS -> skip.
                            #
                            # VARLEN: s_cons[mb+1] is ARRIVED by compute(mb+1), which runs only when
                            # mb+1 < n_mb_b (the partial-Q-tile skip). When gS(mb+1)/compute(mb+1) are
                            # skipped (mb+1 past the sequence's Q-tiles) there is NO tS(mb+1) write to
                            # order against AND no s_cons[mb+1] arrival — so the wait must be dropped or
                            # it deadlocks. Guard it on the runtime mb+1<n_mb_b (compile-time True for
                            # non-varlen, where n_mb_b≡n_mb). This keeps the s_cons producer (compute)
                            # and consumer (gdQ) balanced under the partial-Q-tile skip.
                            with k.if_((mb + 1) < vg.n_mb_b) if varlen else nullcontext():
                                k.mbarrier_wait(bars["s_cons"], phase=_eph(mb + 1, mmin))
                                with k.if_(is_leader):
                                    k.mbarrier_wait(peer_bars["s_cons"], phase=_eph(mb + 1, mmin))
                        with k.if_(is_leader):
                            # gate on the exchanged dS (relay arrived dS_cluster_leader once both
                            # halves landed cluster-wide). count=2 -> both CTAs' relays released.
                            k.mbarrier_wait(bars["dS_cluster_leader"], phase=mb % 2)
                        gemm(
                            _sl(tdQ, (0, 0), (TILE_M, hdim)),
                            sdS_full,
                            sKt,
                            TILE_M,
                            hdim,
                            cg * TILE_N,
                            "dq_done",
                            trans_a=True,
                            trans_b=True,
                        )
                        if n_mb > 1:
                            # sdS_full is single-buffer — the leader's cluster dQ MMA just read BOTH
                            # CTAs' sdS_full halves. Free it (local + peer multicast) so compute(mb+1)
                            # may overwrite sdS_full for the next block (cross-mb WAR, flashattn's
                            # pipeline_dS.consumer_release). Leader-only arrive reaches both CTAs.
                            with k.if_(is_leader):
                                k.mbarrier_arrive(bars["dS_free"])
                                k.mbarrier_arrive(peer_free["dS_free"])
                        return
                    gemm(
                        _sl(tdQ, (0, 0), (TILE_M, hdim)),
                        sdS,
                        sK,
                        TILE_M,
                        hdim,
                        TILE_N,
                        "dq_done",
                        trans_a=True,
                        trans_b=True,
                    )

            if is_hd192:
                # hd192 DeepSeek: the dedicated TMEM layout overlaps S/dP/dS/dQ, which forces a
                # DIFFERENT order than the general 2-CTA software-pipeline — flashattn issues all
                # five GEMMs in straight per-m-block order S → dP → dK → dV → dQ with NO cross-mb
                # pipelining (flash_bwd_sm100.py:2407-2455). dK precedes dV (dK reads TMEM dS;
                # dV reads P; the overlap choreography is gated by the per-GEMM empty/full waits:
                # gS↦dq_free, gdP↦s_free, gdK/gdQ↦ds_ready, gdV↦p_ready).
                for mb in range(n_mb):
                    gS(mb)
                    gdP(mb)
                    gdK(mb)
                    gdV(mb)
                    gdQ(mb)
            elif use_2cta:
                # The cluster-wide GEMMs (S/dP/dV/dK) + the cluster MMA datapath run ONLY on
                # the leader CTA (map §1: the even CTA issues all cta_group::2 MMAs; the odd
                # CTA's MMA warp idles). gemm()/gdQ() already self-gate their MMA issues on
                # is_leader where they touch the leader-only commits; the per-GEMM peer-full
                # waits above are leader-gated. The compute/reduce roles run on both CTAs.
                # 2-CTA MMA order = flashattn use_2cta main loop S, dK, dP, dQ, dV
                # (flash_bwd_sm100.py:2511-2551): dQ is issued AFTER dP (the dQ/dP swap vs
                # the 1-CTA order) because dQ must wait dS_cluster_leader — the cross-CTA dS
                # exchange handshake — so it is deferred past the dP issue. (Main loop is
                # empty at n_mb=1; the swap bites only for multi-Q-tile 2-CTA.)
                gS(0)
                gdP(0)
                gdV(0)
                for mb in range(1, n_mb):
                    gS(mb)
                    gdK(mb - 1)
                    gdP(mb)
                    gdQ(mb - 1)
                    gdV(mb)
                gdK(n_mb - 1)
                gdQ(n_mb - 1)
            else:
                # PROLOGUE (block 0): S, dP, dV
                gS(0)
                gdP(0)
                gdV(0)
                # MAIN LOOP (block mb = 1..n_mb-1): S[mb], dK[mb-1], dQ[mb-1], dP[mb], dV[mb]
                for mb in range(1, n_mb):
                    gS(mb)
                    gdK(mb - 1)
                    gdQ(mb - 1)
                    gdP(mb)
                    gdV(mb)
                # TAIL (last block): dK[n_mb-1], dQ[n_mb-1]
                gdK(n_mb - 1)
                gdQ(n_mb - 1)
            # mma warp is pure GEMM-issue (F4): the dK/dV epilogue (t2r + scale + TMA
            # store) runs on the COMPUTE warpgroup, matching flashattn's epilogue role.

    # ============== compute warps (4-11): P=exp2(S·scale−LSE), dS=P∘(dP−dPsum) ==============
    # Split across BOTH compute warpgroups (flashattn's 2-warpgroup compute): wg1 (warps 4-7)
    # owns q-cols 0..63 (col_base=0), wg2 (warps 8-11) owns q-cols 64..127 (col_base=64). Each
    # wg does NCOL=64 of the 128 q-columns; the union reproduces the single-wg result. The two
    # wgs rendezvous on k.named_barrier(barrier_id=1, num_warps=8) at the three TMEM-overlap
    # boundaries flashattn guards with compute_sync_barrier (P@S, dS@dP read, dS smem store).
    NCOL = 64

    def compute_softmax_ds(col_base, tid, task, per_mb_tail=None):
        # NCOL-wide P=exp2(S·scale−LSE) then dS=P∘(dP−dPsum) for q-cols [col_base, col_base+NCOL).
        # bf16 TMEM packs 2 bf16/col -> NCOL bf16 cols = NCOL//2 b32 cols at TMEM col col_base//2.
        # per_mb_tail(mb): wg1 supplies the dv_done/dk_done wait + (last-mb) dK/dV epilogue; it
        # MUST run interleaved each mb (not as a separate loop) because dv_done/dk_done are
        # single-stage mma->compute barriers that can't buffer two un-waited arrivals.
        fragS = reg(DType.F32, (NCOL,))
        fragP = reg(iod, (NCOL,))
        fragdP = reg(DType.F32, (NCOL,))
        fragdS = reg(iod, (NCOL,))
        rlse = reg(DType.F32, (1,))
        rnlse = reg(DType.F32, (1,))
        rdps = reg(DType.F32, (1,))
        rt = reg(DType.F32, (1,))
        rP = reg(DType.F32, (NCOL,))
        cscale = reg(DType.F32, (1,))
        # F7: packed-pair (f32x2) temps for the dS residual (sub_packed_f32x2 + mul_packed_f32x2).
        rdps2 = reg(DType.F32, (2,))
        rt2 = reg(DType.F32, (2,))
        cbc = col_base // 2
        rninf = reg(DType.F32, (1,))  # VARLEN: -inf fill for OOB-row/col masking (→ P=0)
        # CAUSAL: this CTA's kv-tile index (for the per-tile key_start base). Pure scalar
        # arithmetic on the work field; only needed when masking (keeps non-causal IR unchanged).
        nb = None
        if config.is_causal or varlen or dense_pad:
            _batch, _, nb = task_geom(task)
        vg = varlen_geom(_batch) if varlen else None  # VARLEN: per-seq runtime geometry
        mmin = m_block_min(nb, vg) if causal_skip else 0  # CAUSAL skip: per-task m-loop start
        for mb in range(n_mb):
            with _skip(mb, mmin, vg, nb, cluster=True):  # CAUSAL: skip above-diagonal Q-tiles;
                # VARLEN: skip mb>=n_mb_b / nb>=n_nb_b. Both compute wgs guard identically on the same
                # (nb,n_mb_b,n_nb_b), so the cross-wg named_barrier/wg_sync and all cross-role mbarriers
                # stay balanced (same lockstep property as the causal skip). cluster=True (2-CTA): both
                # CTAs of a cluster compute the SAME predicate (nb-cta_in_cluster), so the compute body
                # (P/dS + the dS cross-CTA exchange/relay + p_ready/ds_ready arrivals that feed the
                # cluster MMA) runs in lockstep on BOTH CTAs even when the peer's own kv-tile is
                # past-sequence — the past tile's P is masked to 0 by the varlen OOB mask so it
                # contributes 0 dS/dV/dK/dQ. (The per-CTA dK/dV STORE in the tail uses the per-CTA
                # tile_valid predicate so the past CTA writes nothing — see make_dk_dv_tail.)
                st, occ = _est(mb, mmin)
                ph = _eph(mb, mmin)
                rM = st * TILE_M
                k.reg_fill(cscale, scale_log2)
                # P[kv,q] = exp2(S[kv,q]·scale_log2 − LSE[q])  (LSE per q-col)
                k.mbarrier_wait(bars["s_ready"], phase=ph)
                k.tcgen05_ld(fragS, tS, shape="32x32b", num=NCOL, row=0, col=col_base)
                k.tcgen05_wait_ld()
                if is_hd192:
                    # hd192: S fully read into rmem (wait_ld drained the t2r). Signal s_free so the
                    # dP MMA may overwrite S's overlapping cols [384,448) — flashattn issues this
                    # release RIGHT HERE (after the S t2r, before computing P; flash_bwd_sm100.py:
                    # 3060-3065), much earlier than the general path's "after P write". Both compute
                    # wgs arrive (count=2); the dP MMA (gdP) waits s_free.
                    k.fence(kind=af, scope=FenceScope.CTA)
                    k.mbarrier_arrive(bars["s_free"])
                elif use_2cta and n_mb > 1:
                    # general 2-CTA multi-Q-tile: S fully read into rmem (wait_ld drained the t2r).
                    # Signal s_cons so the leader's gdQ(mb-1) may overwrite tS(mb)'s S/dQ-overlap cols
                    # ([64,128); tmem_dQ=64 over tmem_S=0). Both compute wgs arrive (count=2). The fence
                    # orders the t2r read before the arrive so the WAR has a happens-before witness.
                    k.fence(kind=af, scope=FenceScope.CTA)
                    k.mbarrier_arrive(bars["s_cons"])
                if dense_pad:
                    # DENSE 2-CTA PADDING-TILE mask (THE critical correctness item for batch>1). For an
                    # odd-#kv-tile dense Sk the last cluster has one PADDING tile (nb >= real_n_nb) whose
                    # kv range is entirely past Sk. Its K/V load is overscan: for batch>1 it reads coords
                    # batch*Sk + real_n_nb*TILE_N == (batch+1)*Sk = the NEXT batch's K (NOT zero), so the
                    # padding tile's S/P would be a real (wrong) score. Force P=0 on the WHOLE tile by
                    # masking the ENTIRE fragS row to -inf when nb >= real_n_nb (compile-time tile-level
                    # condition; nb is a runtime scalar). exp2(-inf)=0 -> dS=0 -> dV/dK/dQ get a 0
                    # contribution from the padding tile (so dQ's cluster sum dS(valid)·K(valid) +
                    # dS(pad)·K(pad) drops the pad term). Tile-aligned Sk -> the padding tile is
                    # WHOLE-OOB (never partial), so the whole row masks with no per-column predicate.
                    k.reg_fill(rninf, float("-inf"))
                    with k.if_(nb >= real_n_nb):
                        for c in range(NCOL):
                            k.reg_fill(_sl(fragS, (c,), (1,)), rninf)
                if varlen:
                    # VARLEN OOB-tail validity mask (C11; THE critical correctness item). The TMA
                    # clamp only zero-fills at the WHOLE-buffer end, NOT interior per-sequence
                    # boundaries, so a partial last tile reads the NEXT sequence's tokens into
                    # fragS. Mask BEFORE P=exp2 so masked S=-inf -> P=0 -> dS=0 -> dV/dK/dQ get a
                    # 0 contribution from OOB (q,kv) pairs. fragS = [kv-row=tid, q-col=col_base+j]:
                    #   - KV-row OOB:  nb*TILE_N + tid   >= slen_k  -> mask the whole NCOL-wide row.
                    #   - Q-col  OOB:  mb*TILE_M + col_base+j >= slen_q -> mask that column.
                    k.reg_fill(rninf, float("-inf"))
                    with k.if_((nb * TILE_N + tid) >= vg.slen_k):
                        for c in range(NCOL):
                            k.reg_fill(_sl(fragS, (c,), (1,)), rninf)
                    for c in range(NCOL):
                        with k.if_((mb * TILE_M + col_base + c) >= vg.slen_q):
                            k.reg_fill(_sl(fragS, (c,), (1,)), rninf)
                if config.is_causal:
                    # CAUSAL mask on the transposed bwd fragment fragS = [kv-row=tid, q-col=j],
                    # j in [0,NCOL) -> q-col col_base+j. Apply BEFORE P=exp2(S·scale−LSE): masked
                    # S=-inf -> P=exp2(-inf)=0 (then dS=P∘(dP−dPsum)=0 inherits the zero, so dV/dK/dQ
                    # get exact-zero contributions from masked (q,kv) pairs). swap_qk=True ->
                    #   q = query_start + (j/group_size) = (mb*TILE_M+col_base) + j   (within-seq q)
                    #   k = key_start   + tid            = (nb*TILE_N-(Sk-Sq)) + tid  (within-seq kv − OFF)
                    # mask when k>q  <=>  (nb*TILE_N+tid) > (mb*TILE_M+col_base+j) + (Sk-Sq)
                    # i.e. kv_local > q_local + (Sk−Sq) — exactly the oracle's masked region.
                    # Tile-local tid/j make the per-sequence base (batch) cancel, so no batch term
                    # in query_start/key_start (a batch*Sk vs batch*Sq mismatch would otherwise add
                    # a spurious batch*(Sk−Sq) offset). The mask handles the DIAGONAL/straddling tile;
                    # the fully-above-diagonal tiles (mb < mmin) never reach here — the m_block_min
                    # loop-skip (the `with _skip(mb, mmin)` guard above) drops them entirely (the perf
                    # lever, CAUSAL_MAP §C2/§3). On any executed tile the mask is still applied (it
                    # self-no-ops on interior tiles), matching flashattn's "always call mask_fn".
                    #
                    # off = the causal offset (Sk−Sq):
                    #   NON-VARLEN: the COMPILE-TIME (Sk − Sq).
                    #   VARLEN (D2): the PER-SEQUENCE runtime off_b = slen_k_b − slen_q_b (a ScalarExpr
                    #     from cu_seqlens), NOT the compile-time (Sk−Sq)=maxlen−maxlen=0. For
                    #     self-attention off_b==0 (== the compile-time 0), but for cross-attention
                    #     varlen the per-seq off differs, so routing vg.off_b is what makes the mask
                    #     correct per sequence. reg_causal_mask's key_start accepts a runtime ScalarExpr.
                    off = vg.off_b if varlen else (Sk - Sq)
                    k.reg_causal_mask(
                        fragS,
                        fragS,
                        query_start=mb * TILE_M + col_base,
                        key_start=nb * TILE_N - off,
                        group_size=1,
                        swap_qk=True,
                    )
                k.mbarrier_wait(bars["tlse"], stage=st, phase=occ % 2)
                for c in range(NCOL):
                    # P = exp2(S·scale − LSE): one FMA on a pre-negated LSE (I4 / flashattn
                    # fma((S),(scale),(-lse))), not a separate mul + sub.
                    k.reg_load(rlse, _sl(sLSE, (rM + col_base + c,), (1,)))
                    k.reg_unary(rnlse, rlse, op="neg")
                    k.reg_fma(rt, _sl(fragS, (c,), (1,)), cscale, rnlse)
                    k.reg_unary(_sl(rP, (c,), (1,)), rt, op="exp2")
                    k.reg_cvt(_sl(fragP, (c,), (1,)), _sl(rP, (c,), (1,)))
                # NEW-3/F5: this wg has read all its sLSE[st] cols -> free the stage (count=2,
                # both wgs arrive; the load warp reloads sLSE[st] only after both wgs are done).
                k.mbarrier_arrive(bars["lse_free"], stage=st)
                # (a0) S is fully read into registers (fence drains the t2r) — rendezvous BEFORE
                # any wg writes P into tmem: P@S overlap means one wg writing P[col_base//2] would
                # clobber the other wg's still-unread S cols (flashattn line ~3136-3139).
                k.fence(kind=af, scope=FenceScope.CTA)
                k.named_barrier(barrier_id=1, num_warps=8)
                k.tcgen05_st(tP, _sl(fragP, (0,), (NCOL // 2,)), num=NCOL // 2, row=0, col=cbc)
                k.tcgen05_wait_st()
                k.fence(kind=af, scope=FenceScope.CTA)
                # (a) P r2t-write into tmem fenced+visible — rendezvous both wgs so the full P tile
                # is in tmem before either signals p_ready / the mma reads P for dV (flashattn ~3148).
                k.named_barrier(barrier_id=1, num_warps=8)
                k.mbarrier_arrive(bars["p_ready"])
                # dS[kv,q] = P[kv,q]·(dP[kv,q] − dPsum[q])  (dPsum per q-col)
                k.mbarrier_wait(bars["dp_ready"], phase=ph)
                k.tcgen05_ld(fragdP, tdP, shape="32x32b", num=NCOL, row=0, col=col_base)
                k.tcgen05_wait_ld()
                k.fence(kind=af, scope=FenceScope.CTA)
                # (b) dP t2r-read fenced; dS overwrites dP's tmem cols (dS@dP overlap) — rendezvous
                # both wgs so neither writes dS into a col the other is still reading dP from.
                k.named_barrier(barrier_id=1, num_warps=8)
                k.mbarrier_wait(bars["tdps"], stage=st, phase=occ % 2)
                # F7: dS = P·(dP − dPsum) in PACKED f32x2 pairs (flashattn sub_packed_f32x2 +
                # mul_packed_f32x2): 2-wide reg slices express the packed-pair op. dPsum is per-q-col
                # (cols c, c+1 = two consecutive q-cols) so a 2-wide load pairs them naturally.
                for c in range(0, NCOL, 2):
                    k.reg_load(rdps2, _sl(sdPsum, (rM + col_base + c,), (2,)))
                    k.reg_sub(rt2, _sl(fragdP, (c,), (2,)), rdps2)  # (dP − dPsum) packed
                    k.reg_mul(rt2, _sl(rP, (c,), (2,)), rt2)  # P·(…) packed
                    k.reg_cvt(_sl(fragdS, (c,), (2,)), rt2)
                # NEW-3/F5: this wg has read all its sdPsum[st] cols -> free the stage (count=2).
                k.mbarrier_arrive(bars["dps_free"], stage=st)
                k.tcgen05_st(tdS, _sl(fragdS, (0,), (NCOL // 2,)), num=NCOL // 2, row=0, col=cbc)
                k.tcgen05_wait_st()
                # stage dS to SMEM sdS (dQ B-operand path: A=sdS trans_a) — each wg writes its cols.
                # F8: vectorized slice copy of the whole NCOL-wide fragdS row into sdS[tid, col_base:
                # col_base+NCOL] (one r2s, mirrors flashattn autovec_copy) instead of a scalar c-loop.
                if use_2cta:
                    if is_hd192:
                        # hd192 has TWO single-buffer dS-export WARs against the PREVIOUS executed block:
                        #   (1) sdS_full WAR — gdQ(mb-1) read sdS_full (the dQ A-operand); compute(mb)
                        #       overwrites it (the local KEPT-half reg_store below). Wait dS_free, the
                        #       leader gdQ's consumer_release (flashattn pipeline_dS.producer_acquire at
                        #       flash_bwd_sm100.py:3228). Applies at multi-Q-tile only (n_mb>1).
                        #   (2) sdS_xchg/sdQ alias WAR — sdS_xchg ALIASES sdQ; the reduce warp drains sdQ
                        #       (the dQ reduce-add) then arrives dQaccum_empty (flashattn:3258-3262).
                        #       compute(mb) must wait it before re-exporting dS into sdS_xchg.
                        # Both are single-stage, phase-keyed on the PREVIOUS block (_eph(mb-1)); the first
                        # executed block (mb==0) has no prior dQ/reduce. Both compute wgs wait.
                        if n_mb > 1:
                            if causal_skip:
                                with _guard(_gt(mb, mmin)):
                                    k.mbarrier_wait(bars["dS_free"], phase=_eph(mb - 1, mmin))
                            elif mb > 0:
                                k.mbarrier_wait(bars["dS_free"], phase=_eph(mb - 1, mmin))
                        if causal_skip:
                            with _guard(_gt(mb, mmin)):
                                k.mbarrier_wait(bars["dQaccum_empty"], phase=_eph(mb - 1, mmin))
                        elif mb > 0:
                            k.mbarrier_wait(bars["dQaccum_empty"], phase=_eph(mb - 1, mmin))
                    elif n_mb > 1 and mb > 0:
                        # General 2-CTA multi-Q-tile: sdS_full is single-buffer and read by the PREVIOUS
                        # block's leader dQ MMA (gdQ(mb-1)). Wait dS_free (the leader's consumer_release)
                        # before overwriting sdS_full/sdS_xchg for THIS block. The first executed block
                        # (mb==0) has no prior dQ. Both compute wgs wait the single leader arrival; phase
                        # tracks gdQ(mb-1)'s release = _eph(mb-1). (Causal 2-CTA is out of scope.)
                        k.mbarrier_wait(bars["dS_free"], phase=_eph(mb - 1, mmin))
                    # dS cross-CTA exchange producer (map §3). The dQ MMA (cta_group::2, m=128,
                    # Layout B) splits dS by M=q (64 q-rows/CTA) with the kv (K) FULL 256 on each
                    # CTA. CTA c keeps q-cols [c*64,(c+1)*64) and needs BOTH kv-tiles' dS for those
                    # cols. This wg computed THIS CTA's kv-tile dS for q-cols [col_base,col_base+64):
                    #   - if those are THIS CTA's KEPT q-cols (col_base == c*64) -> store into the
                    #     local sdS_full slot for this CTA's kv-tile (rows [c*TILE_N, +TILE_N)).
                    #   - else (col_base == (c^1)*64, the PEER's q-cols) -> stage into sdS_xchg and
                    #     s2cluster into the PEER's sdS_full at this CTA's kv-tile slot.
                    # cta c = col_base//NCOL when KEPT (col_base 0 -> c0, col_base 64 -> c1).
                    kept = is_leader if col_base == 0 else (~is_leader)
                    full_row0 = (col_base // NCOL) * TILE_N  # this kv-tile slot in sdS_full
                    with k.if_(kept & (tid < TILE_N)):
                        k.reg_store(
                            _sl(sdS_full, (full_row0 + tid, 0), (1, NCOL)),
                            _sl(fragdS, (0,), (NCOL,)),
                        )
                    with k.if_((~kept) & (tid < TILE_N)):
                        k.reg_store(_sl(sdS_xchg, (tid, 0), (1, NCOL)), _sl(fragdS, (0,), (NCOL,)))
                    # The exporting wg (the one that wrote sdS_xchg) rendezvous + async-proxy fences
                    # the generic SMEM write, then thread0 ships its half to the peer.
                    k.wg_sync(barrier_id=(2 if col_base == 0 else 3))
                    k.fence(kind=af, scope=FenceScope.CTA)  # generic SMEM write -> async bulk copy
                    peer = 1 - cta_in_cluster
                    xchg_bytes = TILE_N * NCOL * 2
                    # peer's sdS_full slot for THIS CTA's kv-tile = rows [c*TILE_N, +TILE_N).
                    dst_row0 = cta_in_cluster * TILE_N
                    with k.if_((~kept) & tid.eq(0)):
                        k.mbarrier_arrive_expect_tx(
                            k.mbar_ref(bars["dS_cluster_full"], remote_coord=peer), bytes=xchg_bytes
                        )
                        k.cp_async_bulk_s2cluster(
                            _sl(sdS_full, (dst_row0, 0), (TILE_N, NCOL)),
                            _sl(sdS_xchg, (0, 0), (TILE_N, NCOL)),
                            mbar=k.mbar_ref(bars["dS_cluster_full"], remote_coord=peer),
                            bytes=xchg_bytes,
                        )
                        if is_hd192 or n_mb > 1:
                            # Drain the s2cluster's SHARED-source (sdS_xchg) read locally so the NEXT
                            # mb's reg_store to sdS_xchg is provably after-drain (else the next-block
                            # write overlaps the in-flight async source -> async_group_source_overwrite).
                            # hd192: also load-bearing for the sdS_xchg/sdQ alias (reduce overwrites it).
                            # n_mb==1: no reload, so the drain is unnecessary (kept minimal).
                            k.cp_async_bulk_commit_group()
                            k.cp_async_bulk_wait_group_read(0)
                else:
                    with k.if_(tid < TILE_N):
                        k.reg_store(
                            _sl(sdS, (tid, col_base), (1, NCOL)), _sl(fragdS, (0,), (NCOL,))
                        )
                    k.wg_sync(barrier_id=(2 if col_base == 0 else 3))  # per-wg distinct barrier_id
                    k.fence(kind=af, scope=FenceScope.CTA)
                # (c) dS r2s-store to smem fenced — rendezvous both wgs before either signals
                # ds_ready (the full sdS / tdS dS tile is complete only once both wgs stored).
                k.named_barrier(barrier_id=1, num_warps=8)
                k.mbarrier_arrive(bars["ds_ready"])
                if per_mb_tail is not None:
                    per_mb_tail(mb)

    # ---- dK/dV epilogue (NEW-1: dual-warpgroup hdim-column split, flashattn
    # epilogue_dK_or_dV_tma @3852 with split_wg @2718). BOTH compute warpgroups run the
    # epilogue, each over HALF the hdim columns: wg1 (q-col_base 0) -> hdim-cols [0, hdim/2),
    # wg2 (q-col_base 64) -> hdim-cols [hdim/2, hdim). split_wg divides BOTH the t2r source
    # (num=hdim/2 at col=half_base — f32 TMEM is 1 col/f32) AND the dest (r2s sdK/sdV[tid,
    # half_base+c] + a per-wg-half TMA store). The union of the two halves = the full dK/dV.
    # Each wg uses its OWN wg_sync barrier_id (flashattn barrier_id + wg_idx).
    hdim_half = hdim // 2
    hdimv_half = hdimv // 2
    # CHUNKED epilogue (flashattn epilogue_dK_or_dV_tma @3853): each wg's hdim_half cols are
    # stored in num_epi_stages pieces of RNCOL cols (= flashattn num_epi_stages = (hdim/2)/RNCOL;
    # 1 for hd64 and MHA hd128, 2 for GQA hd128 / hd192-dV, 3 for GQA hd192-dK). The wg's single
    # [TILE_N, RNCOL] SMEM slice (at col wg_idx*RNCOL) is REUSED for each chunk and stored/reduced
    # at global col half_base+j*RNCOL, with a per-chunk commit+drain (flashattn
    # cp_async_bulk_wait_group(0) @4015-4017) before the slice is overwritten by the next chunk.
    NEPI_K = max(1, hdim_half // DK_RNCOL)
    NEPI_V = max(1, hdimv_half // DV_RNCOL)

    def make_dk_dv_tail(
        col_base, tid, task, rt, fragK, fragV, rkb, cscale_s, mmin, vg=None, nb=None
    ):
        half_base = (col_base // NCOL) * hdim_half  # q-col half 0/64 -> hdim half 0/hdim_half
        half_base_v = (col_base // NCOL) * hdimv_half
        wg_idx = col_base // NCOL  # compute-wg index (0/1)
        wg_bar = 2 if col_base == 0 else 3  # per-wg distinct wg_sync barrier_id

        def _stage_chunk(sbuf, frag, j, rncol, scale_it):
            # r2s ONE RNCOL-wide chunk (chunk j of this wg's frag) into THIS wg's single SMEM
            # slice (cols [wg_idx*rncol, +rncol)). The two compute wgs use DISJOINT slices, so
            # they never collide; chunks WITHIN a wg reuse the same slice (drained between).
            # GQA: store UNSCALED f32 (scale moves to postprocess). MHA: dK·scale + bf16 cvt
            # (dV: cvt only). frag holds the wg's whole hdim_half cols; chunk j covers
            # frag[j*rncol : (j+1)*rncol] -> sbuf[:, wg_idx*rncol : +rncol].
            slot = wg_idx * rncol
            with k.if_(tid < TILE_N):
                if gqa:
                    for c in range(rncol):
                        k.reg_store(
                            _sl(sbuf, (tid, slot + c), (1, 1)), _sl(frag, (j * rncol + c,), (1,))
                        )
                elif scale_it:
                    for c in range(rncol):
                        k.reg_mul(rt, _sl(frag, (j * rncol + c,), (1,)), cscale_s)
                        k.reg_cvt(rkb, rt)
                        k.reg_store(_sl(sbuf, (tid, slot + c), (1, 1)), rkb)
                else:
                    for c in range(rncol):
                        k.reg_cvt(rkb, _sl(frag, (j * rncol + c,), (1,)))
                        k.reg_store(_sl(sbuf, (tid, slot + c), (1, 1)), rkb)
            k.wg_sync(barrier_id=wg_bar)
            k.fence(kind=af, scope=FenceScope.CTA)

        def _chunk_drain(j, nchunks):
            # flashattn drains the chunk's async store before this wg's slice is reused by the
            # next chunk (cp_async_bulk_wait_group(0) @4015, only when not the LAST chunk) + a
            # wg_sync so the 128 threads observe the drain before overwriting the slice. Only
            # thread 0 issued the bulk store, so the commit/drain is thread-0-only (flashattn's
            # leader_warp @4005); the wg_sync is the cohort rendezvous (all 128 threads).
            if j < nchunks - 1:
                with k.if_(tid.eq(0)):
                    k.cp_async_bulk_commit_group()
                    k.cp_async_bulk_wait_group_read(0)
                k.wg_sync(barrier_id=wg_bar)

        def _store_epilogue():
            # ---- this wg's hdim-column half: per-chunk t2r-read, stage, store/reduce ----
            # flashattn re-does the t2r per chunk; here the full hdim_half is t2r-read once into
            # fragK/fragV (value-identical — the column span is the same) and each chunk slices
            # frag[j*rncol:(j+1)*rncol]. GQA stages UNSCALED f32; MHA dK·scale + bf16, dV bf16.
            k.reg_fill(cscale_s, config.softmax_scale)
            _tc_ld(k, fragV, tdV, hdimv_half, col=half_base_v)
            k.tcgen05_wait_ld()
            _tc_ld(k, fragK, tdK, hdim_half, col=half_base)
            k.tcgen05_wait_ld()
            batch, head, nb_ = task_geom(task)
            head_kv = head // G if gqa else head  # GQA: dK/dV land in the shared kv-head rows
            if varlen:
                # VARLEN dK/dV store (C12): a fixed-tile TMA store of the partial last KV-tile
                # would write rows >= slen_k into the NEXT packed sequence's dK/dV (the TMA clamp
                # only squashes at the buffer end, not interior boundaries). Use a PREDICATED
                # SCALAR store of this wg's hdim-column half, only for valid rows
                # (base_k+nb*TILE_N+row within the sequence, i.e. nb*TILE_N+row < slen_k) —
                # INVARIANT I2 (externally-visible outputs -> scalar writes on boundary tiles).
                #
                # 2-CTA partial cluster (tile_valid = nb < n_nb_b): the compute body + tail run
                # under the CLUSTER predicate (cluster=True), so the past-sequence PEER CTA (whose
                # OWN kv-tile nb >= n_nb_b) ALSO reaches this store (it ran the cluster MMA/compute
                # in lockstep so the handshakes stay balanced). It must write NOTHING — its kv-tile
                # rows belong to the NEXT packed sequence. The per-row predicate already covers it
                # (nb >= n_nb_b => nb*TILE_N >= ceil(slen_k/TILE_N)*TILE_N >= slen_k, so
                # nb*TILE_N+tid >= slen_k for all tid -> no row stores). The explicit per-CTA
                # tile_valid (nb < n_nb_b) guard below makes the whole-past-tile suppression a
                # named, robust gate rather than an emergent property of the per-row arithmetic.
                ktok = vg.base_k + nb * TILE_N
                if gqa:
                    # GQA varlen: dK/dV are f32 reduce-add accumulators in the SHARED kv-head
                    # (head_kv). The OOB-kv rows (kv >= slen_k) of sdK/sdV are ALREADY 0 — the
                    # compute masks P=0 for kv >= slen_k, so dV=Pᵀ·dO and dS (hence dK) are 0 on
                    # those rows. So a per-chunk tma_reduce_add adds 0 to the OOB GMEM rows: at an
                    # INTERIOR sequence boundary that is +0 into the next sequence's rows (harmless,
                    # NOT an overwrite like the MHA store), and at the buffer end the TMA clamp
                    # squashes it. So GQA varlen uses the SAME atomic reduce-add as dense GQA (the
                    # per-CTA tile_valid gate just skips a whole past-sequence tile = all-zero add).
                    with k.if_(nb < vg.n_nb_b):
                        for j in range(NEPI_V):
                            _stage_chunk(sdV, fragV, j, DV_RNCOL, scale_it=False)
                            with k.if_(tid.eq(0)):
                                k.tma_reduce_add(
                                    dv_g,
                                    _sl(sdV, (0, wg_idx * DV_RNCOL), (TILE_N, DV_RNCOL)),
                                    coords=(ktok, head_kv, half_base_v + j * DV_RNCOL),
                                    shape=(TILE_N, DV_RNCOL),
                                    gmem_shape=(TILE_N, 1, DV_RNCOL),
                                )
                            _chunk_drain(j, NEPI_V)
                        for j in range(NEPI_K):
                            _stage_chunk(sdK, fragK, j, DK_RNCOL, scale_it=False)
                            with k.if_(tid.eq(0)):
                                k.tma_reduce_add(
                                    dk_g,
                                    _sl(sdK, (0, wg_idx * DK_RNCOL), (TILE_N, DK_RNCOL)),
                                    coords=(ktok, head_kv, half_base + j * DK_RNCOL),
                                    shape=(TILE_N, DK_RNCOL),
                                    gmem_shape=(TILE_N, 1, DK_RNCOL),
                                )
                            _chunk_drain(j, NEPI_K)
                        with k.if_(tid.eq(0)):
                            k.cp_async_bulk_commit_group()
                            k.cp_async_bulk_wait_group_read(0)
                    return
                # MHA varlen: predicated SCALAR store, one chunk at a time. The chunk is staged
                # into this wg's slice (dK·scale + bf16) then scalar-read back per valid row.
                with k.if_(nb < vg.n_nb_b):  # per-CTA tile_valid: past-seq CTA writes nothing
                    for j in range(NEPI_V):
                        _stage_chunk(sdV, fragV, j, DV_RNCOL, scale_it=False)
                        slot = wg_idx * DV_RNCOL
                        with k.if_(tid < TILE_N):
                            with k.if_((nb * TILE_N + tid) < vg.slen_k):
                                for c in range(DV_RNCOL):
                                    k.reg_load(rkb, _sl(sdV, (tid, slot + c), (1, 1)))
                                    k.reg_store(
                                        _sl(
                                            dv_g,
                                            (ktok + tid, head, half_base_v + j * DV_RNCOL + c),
                                            (1, 1, 1),
                                        ),
                                        rkb,
                                    )
                    for j in range(NEPI_K):
                        _stage_chunk(sdK, fragK, j, DK_RNCOL, scale_it=True)
                        slot = wg_idx * DK_RNCOL
                        with k.if_(tid < TILE_N):
                            with k.if_((nb * TILE_N + tid) < vg.slen_k):
                                for c in range(DK_RNCOL):
                                    k.reg_load(rkb, _sl(sdK, (tid, slot + c), (1, 1)))
                                    k.reg_store(
                                        _sl(
                                            dk_g,
                                            (ktok + tid, head, half_base + j * DK_RNCOL + c),
                                            (1, 1, 1),
                                        ),
                                        rkb,
                                    )
                return
            # DENSE 2-CTA PADDING TILE: the per-CTA dK/dV store is SUPPRESSED for the padding CTA
            # (nb_ >= real_n_nb) — its TILE_N kv-rows start at real_n_nb*TILE_N >= Sk, so a fixed-tile
            # TMA store would overrun the dk_g/dv_g buffer (for batch>1, into the NEXT batch's rows;
            # for the last batch, past the buffer end). The padding CTA reached here only because it
            # ran the cluster MMA/compute in lockstep (cluster predicate); its sdK/sdV are 0 anyway
            # (P=0), but the explicit tile_valid gate makes "write nothing" a named, robust guard
            # (mirrors the varlen per-CTA tile_valid above) instead of relying on a TMA clamp. It also
            # skips the deterministic-GQA semaphore acquire/release for the padding tile (it owns no
            # output cell). Wraps the WHOLE store + any sem ops. Off (nullcontext) when not dense_pad.
            _pad_gate = k.if_(nb_ < real_n_nb) if dense_pad else nullcontext()
            with _pad_gate:
                # dK/dV epilogue on the compute role — each wg writes ITS column half. The
                # dk_g/dv_g gmem tensor is [Tk, Hkv, hdim]; the 3rd coord is the hdim offset.
                ktok = batch * Sk + nb_ * TILE_N
                wg_slot = 0 if col_base == 0 else 1  # the trailing-2 sem slot = per-wg column half
                # DETERMINISTIC GQA: acquire this q-head's turn (lock-value = head % G) on the
                # shared kv-head's (batch, head_kv, nb, wg) cell ONCE BEFORE the chunked dK/dV
                # reduce-add loop (flashattn acquires once per (batch,head_kv,nb,wg) call @3958-3962,
                # NOT per chunk), so the G q-heads of a group add into dk_g/dv_g[head_kv] one at a
                # time. cell rank-4 = (batch, head_kv, nb, wg).
                if deterministic and gqa:
                    with k.if_(tid.eq(0)):
                        k.gmem_wait_eq(
                            dk_sem_g, coords=(batch, head_kv, nb_, wg_slot), value=head % G
                        )
                        k.gmem_wait_eq(
                            dv_sem_g, coords=(batch, head_kv, nb_, wg_slot), value=head % G
                        )
                    k.wg_sync(barrier_id=wg_bar)
                # GQA: per-chunk ATOMIC reduce-add into the shared kv-head's f32 rows (dk_g/dv_g are
                # zero-init output args, the G reduce-adds sum to dK[head_kv]). MHA: per-chunk plain
                # tma_store (dK·scale+bf16 staged, dV bf16 staged). Each chunk covers DK/DV_RNCOL cols
                # at global col half_base+j*RNCOL from this wg's SMEM slice, drained between chunks.
                for j in range(NEPI_V):
                    _stage_chunk(sdV, fragV, j, DV_RNCOL, scale_it=False)
                    with k.if_(tid.eq(0)):
                        cv = wg_idx * DV_RNCOL
                        if gqa:
                            k.tma_reduce_add(
                                dv_g,
                                _sl(sdV, (0, cv), (TILE_N, DV_RNCOL)),
                                coords=(ktok, head_kv, half_base_v + j * DV_RNCOL),
                                shape=(TILE_N, DV_RNCOL),
                                gmem_shape=(TILE_N, 1, DV_RNCOL),
                            )
                        else:
                            k.tma_store(
                                dv_g,
                                _sl(sdV, (0, cv), (TILE_N, DV_RNCOL)),
                                coords=(ktok, head, half_base_v + j * DV_RNCOL),
                                shape=(TILE_N, DV_RNCOL),
                                gmem_shape=(TILE_N, 1, DV_RNCOL),
                            )
                    _chunk_drain(j, NEPI_V)
                for j in range(NEPI_K):
                    _stage_chunk(sdK, fragK, j, DK_RNCOL, scale_it=True)
                    with k.if_(tid.eq(0)):
                        ck = wg_idx * DK_RNCOL
                        if gqa:
                            k.tma_reduce_add(
                                dk_g,
                                _sl(sdK, (0, ck), (TILE_N, DK_RNCOL)),
                                coords=(ktok, head_kv, half_base + j * DK_RNCOL),
                                shape=(TILE_N, DK_RNCOL),
                                gmem_shape=(TILE_N, 1, DK_RNCOL),
                            )
                        else:
                            k.tma_store(
                                dk_g,
                                _sl(sdK, (0, ck), (TILE_N, DK_RNCOL)),
                                coords=(ktok, head, half_base + j * DK_RNCOL),
                                shape=(TILE_N, DK_RNCOL),
                                gmem_shape=(TILE_N, 1, DK_RNCOL),
                            )
                    _chunk_drain(j, NEPI_K)
                with k.if_(tid.eq(0)):
                    k.cp_async_bulk_commit_group()
                    k.cp_async_bulk_wait_group_read(0)
                # DETERMINISTIC GQA: after ALL chunked dK/dV reduce-adds have FULLY drained, release
                # the next q-head in the group (head%G -> head%G+1). The release fence orders the
                # completed adds before the bump (flashattn arrive_inc @4030-4035, once per call).
                if deterministic and gqa:
                    k.wg_sync(barrier_id=wg_bar)
                    with k.if_(tid.eq(0)):
                        k.fence(kind=FenceKind.MEMORY, scope=FenceScope.GPU)
                        k.gmem_atomic_add(
                            dk_sem_g,
                            coords=(batch, head_kv, nb_, wg_slot),
                            value=1,
                            order="release",
                        )
                        k.gmem_atomic_add(
                            dv_sem_g,
                            coords=(batch, head_kv, nb_, wg_slot),
                            value=1,
                            order="release",
                        )

        def dk_dv_tail(mb):
            # dv_done/dk_done commit once per EXECUTED block (the mma gemm_acc runs under the
            # same _skip guard); wait at the executed-iteration phase. The epilogue still fires
            # on the LAST EXECUTED block (n_mb-1 fixed-len; n_mb_b-1 runtime under varlen).
            ph = _eph(mb, mmin)
            # dV/dK accumulate in TMEM across m-blocks; mma commits dv_done/dk_done EVERY mb
            # (to drain the tP/tdS operand reads before the next mb overwrites those cols).
            # BOTH compute wgs wait/consume them (the epilogue is now dual-wg, NEW-1).
            k.mbarrier_wait(bars["dv_done"], phase=ph)
            k.mbarrier_wait(bars["dk_done"], phase=ph)
            if varlen:
                # VARLEN: the store fires on the last EXECUTED block mb == n_mb_b-1 (runtime). This
                # tail body only runs for executed mb (called inside compute's _skip guard), so the
                # last executed mb is exactly n_mb_b-1. Guard the store-block at runtime.
                with k.if_(vg.n_mb_b.eq(mb + 1)):
                    _store_epilogue()
            elif mb == n_mb - 1:
                _store_epilogue()

        return dk_dv_tail

    with k.role(warpgroup=1):  # compute wg1: warps 4-7, q-cols 0..63 / dK/dV hdim-cols [0, hdim/2)
        tid = k.tid_in_wg()  # 0..127 = kv-row of S/P/dP/dS [TILE_N, TILE_M]
        rt = reg(DType.F32, (1,))
        fragK = reg(DType.F32, (hdim_half,))
        fragV = reg(DType.F32, (hdimv_half,))
        rkb = reg(iod, (1,))
        cscale_s = reg(DType.F32, (1,))
        with k.for_each_task(sched) as task:
            b_, _, nb_ = task_geom(task)
            vg_ = varlen_geom(b_) if varlen else None
            mmin_ = m_block_min(nb_, vg_) if causal_skip else 0
            tail = make_dk_dv_tail(0, tid, task, rt, fragK, fragV, rkb, cscale_s, mmin_, vg_, nb_)
            compute_softmax_ds(0, tid, task, per_mb_tail=tail)

    with k.role(
        warpgroup=2
    ):  # compute wg2: warps 8-11, q-cols 64..127 / dK/dV hdim-cols [hdim/2, hdim)
        tid = k.tid_in_wg()
        rt = reg(DType.F32, (1,))
        fragK = reg(DType.F32, (hdim_half,))
        fragV = reg(DType.F32, (hdimv_half,))
        rkb = reg(iod, (1,))
        cscale_s = reg(DType.F32, (1,))
        with k.for_each_task(sched) as task:
            b_, _, nb_ = task_geom(task)
            vg_ = varlen_geom(b_) if varlen else None
            mmin_ = m_block_min(nb_, vg_) if causal_skip else 0
            tail = make_dk_dv_tail(64, tid, task, rt, fragK, fragV, rkb, cscale_s, mmin_, vg_, nb_)
            compute_softmax_ds(64, tid, task, per_mb_tail=tail)

    # ============== reduce warps (0-3): dQ TMEM -> SMEM -> tma_reduce_add(dQaccum) ==============
    # NEW-2: STAGED dQ reduce (flashattn dQacc_reduce @3592-3648). flashattn issues ONE full
    # t2r of all tile_hdim cols into the fragment, releases pipeline_dQ, then loops
    # sdQaccum_stage = tile_hdim // dQ_reduce_ncol (=2 for hd64, dQ_reduce_ncol=32) stages: each
    # stage r2s-stores its 32-col fragment slice into SMEM and issues a cpasync_reduce_bulk_add_f32
    # of tile_m·32·4 bytes. The two 32-col reduce-adds together cover all 64 cols (= one full-tile
    # reduce, value-identical) but match flashattn's 2× reduce-add instruction count.
    RDQ_STAGES = hdim // RDQ_NCOL  # = 2 for hd64 (RDQ_NCOL is module-level)
    assert hdim % RDQ_NCOL == 0
    with k.role(warpgroup=0):
        tid = k.tid_in_wg()
        fragdQ = reg(DType.F32, (hdim,))
        rzero = reg(DType.F32, (1,))
        with k.for_each_task(sched) as task:
            batch, head, nb = task_geom(task)
            vg = varlen_geom(batch) if varlen else None  # VARLEN: per-seq runtime geometry
            mmin = m_block_min(nb, vg) if causal_skip else 0  # CAUSAL skip: per-task m-loop start
            for mb in range(n_mb):
                with _skip(mb, mmin, vg, nb, cluster=True):  # CAUSAL: skip above-diagonal; VARLEN:
                    # skip past-seq. cluster=True (2-CTA): the reduce body waits dq_done (the leader's
                    # cluster gdQ commits it MULTICAST to both CTAs) and arrives dq_free (the leader's
                    # next-iter gS/gdQ waits peer_bars["dq_free"]). Both must stay balanced across the
                    # pair, so the reduce body runs in lockstep on BOTH CTAs even when the peer's own
                    # kv-tile is past-sequence. The peer holds VALID dQ for its q-row half (the cluster
                    # dQ MMA summed over the full 256 kv with the past tile's dS=0), and the q-rows are
                    # the shared Q-tile (valid for mb<n_mb_b), so the peer's reduce-add OUTPUT is
                    # correct — it writes its own valid q-rows, gated by the D1 per-row predicate
                    # (mb*TILE_M+tid < slen_q). (dQ q-rows are cluster-shared, so unlike dK/dV there is
                    # no per-CTA tile_valid output gate: the peer writes valid q-rows, not overrun.)
                    ph = _eph(mb, mmin)
                    qtok = (vg.base_q if varlen else batch * Sq) + mb * TILE_M
                    k.mbarrier_wait(bars["dq_done"], phase=ph)
                    if use_2cta:
                        # 2-CTA dQ accumulator is Layout B (cta_group::2, m=128): each CTA holds
                        # 64 q-rows (q = cta*64 + lane), with the hdim split across the lane halves
                        # — lanes [0,64) hold d[0,DQ_COL) at cols [0,DQ_COL); lanes [64,128) hold
                        # d[DQ_COL,2·DQ_COL) at cols [0,DQ_COL). DQ_COL = hdim//2 (64 for hd128, 96
                        # for hd192). One t2r of DQ_COL cols (row=0,col=0) gives, per thread:
                        #   tid<64  : frag[c] = dQ[q=tid,      d=c]            c in [0,DQ_COL)
                        #   tid>=64 : frag[c] = dQ[q=tid-64,   d=DQ_COL+c]     c in [0,DQ_COL)
                        # cluster_reduce_dQ=False: each CTA reduces its OWN 64 q-rows
                        # [cta*64, cta*64+64) into dq_g (no cross-CTA dQ reduce).
                        DQ_COL = hdim // 2
                        _tc_ld(k, fragdQ, tdQ, DQ_COL, col=0)
                        k.tcgen05_wait_ld()
                        k.mbarrier_arrive(bars["dq_free"])
                        if deterministic:
                            # 2-CTA cross-CLUSTER serialization: the clusters reduce-add into the same
                            # dQ Q-rows in ascending cluster order. Turn = cluster index (work %
                            # n_cluster); the cell is keyed by cta_in_cluster (the pair's two CTAs own
                            # DISJOINT q-rows, so they never serialize against each other — only against
                            # the same-slot CTA of the next cluster). flashattn dQacc_reduce @3619.
                            c_turn = task.field("work") % n_cluster
                            with k.if_(tid.eq(0)):
                                k.gmem_wait_eq(
                                    dq_sem_g, coords=(batch, head, mb, cta_in_cluster), value=c_turn
                                )
                            k.wg_sync(barrier_id=1)
                        # reduce-add this CTA's 64 q-rows into dq_g, one RDQ_NCOL-col slice at a time
                        # through a SINGLE rotating sdQ slot. The frag is lane-split (tid<TM_H holds
                        # global d[0,DQ_COL), tid>=TM_H holds d[DQ_COL,2·DQ_COL)); RDQ_NCOL divides
                        # DQ_COL so each slice lies wholly in one lane-half — that half scatters its 32
                        # cols into sdQ[:, 0:32] (rows [0,TM_H)), reduce-add to dq_g at d=cbase. The
                        # prior slice's reduce-add drains + the wg_sync below order the slot reuse.
                        # RDQ_NSLOT-deep double buffer (flashattn sdQaccum_stage): slice s lands in
                        # slot s%NSLOT at sdQ col slot·RDQ_NCOL; thread-0 keeps NSLOT-1 reduce-adds in
                        # flight (wait_group_read(NSLOT-1)). The per-iter wg_sync makes thread-0's drain
                        # of slot s-NSLOT visible before this CTA's threads overwrite that slot.
                        cl_qtok = qtok + cta_in_cluster * TM_H
                        for s in range(RDQ_STAGES):
                            cbase = s * RDQ_NCOL
                            col0 = (s % RDQ_NSLOT) * RDQ_NCOL
                            if cbase < DQ_COL:  # low half: tid<TM_H, frag[cbase:]
                                with k.if_(tid < TM_H):
                                    for c in range(RDQ_NCOL):
                                        k.reg_store(
                                            _sl(sdQ, (tid, col0 + c), (1, 1)),
                                            _sl(fragdQ, (cbase + c,), (1,)),
                                        )
                            else:  # high half: tid>=TM_H, frag[cbase-DQ_COL:]
                                with k.if_(tid >= TM_H):
                                    for c in range(RDQ_NCOL):
                                        k.reg_store(
                                            _sl(sdQ, (tid - TM_H, col0 + c), (1, 1)),
                                            _sl(fragdQ, (cbase - DQ_COL + c,), (1,)),
                                        )
                            k.wg_sync(barrier_id=1)
                            k.fence(kind=af, scope=FenceScope.CTA)
                            with k.if_(tid.eq(0)):
                                k.tma_reduce_add(
                                    dq_g,
                                    _sl(sdQ, (0, col0), (TM_H, RDQ_NCOL)),
                                    coords=(cl_qtok, head, cbase),
                                    shape=(TM_H, RDQ_NCOL),
                                    gmem_shape=(TM_H, 1, RDQ_NCOL),
                                )
                                k.cp_async_bulk_commit_group()
                                k.cp_async_bulk_wait_group_read(RDQ_NSLOT - 1)
                            k.wg_sync(barrier_id=1)
                        # drain the tail (the last NSLOT-1 in-flight reduce-adds) before the buffer is
                        # reused next task / freed for the hd192 dS exchange. When NSLOT==1 the in-loop
                        # wait_group_read(0) already drained every reduce-add, so a tail drain here would
                        # be a wait_group with no un-drained commit (the checker flags it as a missing
                        # release witness) — skip it.
                        if RDQ_NSLOT > 1:
                            with k.if_(tid.eq(0)):
                                k.cp_async_bulk_wait_group_read(0)
                        if deterministic:
                            # release the next cluster: the reduce-adds have FULLY drained, so the
                            # release fence + atomic_add publishes a state that already includes this
                            # cluster's completed contribution (cluster c_turn+1's wait_eq observes it).
                            k.wg_sync(barrier_id=1)
                            with k.if_(tid.eq(0)):
                                k.fence(kind=FenceKind.MEMORY, scope=FenceScope.GPU)
                                k.gmem_atomic_add(
                                    dq_sem_g,
                                    coords=(batch, head, mb, cta_in_cluster),
                                    value=1,
                                    order="release",
                                )
                        if is_hd192:
                            # hd192: sdQ fully drained — free the aliased region for the next block's dS
                            # export (flash_bwd_sm100.py:3670). Warp-uniform arrive (the wg_sync makes
                            # thread-0's drain visible to the cohort first).
                            k.wg_sync(barrier_id=1)
                            k.mbarrier_arrive(bars["dQaccum_empty"])
                        continue
                    # ONE full t2r of all hdim cols into the fragment (flashattn cute.copy @3595).
                    _tc_ld(k, fragdQ, tdQ, hdim, col=0)
                    k.tcgen05_wait_ld()
                    # tdQ[mb] is now read into registers -> let mma overwrite these cols (tdP[mb+1]).
                    k.mbarrier_arrive(bars["dq_free"])
                    # DETERMINISTIC: acquire this Q-tile's turn BEFORE draining the reduce-add's
                    # predecessors. lock-value = nb (ascending n_block); the n_nb KV-tile CTAs add
                    # into dq_g[(batch,head,mb)] one at a time. cell = (batch,head,mb,cta_in_cluster).
                    # flashattn dQacc_reduce @3619-3638 (wait_eq + reduce_sync_barrier rendezvous).
                    if deterministic:
                        with k.if_(tid.eq(0)):
                            k.gmem_wait_eq(
                                dq_sem_g, coords=(batch, head, mb, cta_in_cluster), value=nb
                            )
                        k.wg_sync(barrier_id=1)  # rendezvous behind thread-0's acquire
                    # staged reduce loop: each stage r2s + reduce-add of its own 32-col slice.
                    # VARLEN dQ reduce-add tail (C13 / D1): the bulk tma_reduce_add is a fixed-TILE_M
                    # async op that thread 0 issues — it cannot carry a per-row predicate, so a partial
                    # last Q-tile's OOB rows (mb*TILE_M+tid >= slen_q) would reduce-add into the NEXT
                    # packed sequence's VALID dQ rows (the interior boundary the TMA clamp does NOT
                    # squash). Value-correctness no longer RESTS on the compute OOB-mask chain making
                    # those rows incidentally bit-exact 0 (masked fragS=-inf -> P=0 -> dS=0 -> dQ=0):
                    # instead we make the 0-addend EXPLICIT by zeroing the OOB rows of the sdQ staging
                    # buffer before the reduce-add. Each thread owns row `tid`, so it stages its own
                    # fragdQ for a valid row and an explicit 0.0 for an OOB row. (D1: the staged zero
                    # is now load-bearing, not incidental; the FINAL sequence's tail still overruns the
                    # buffer end where the TMA clamp squashes it.)
                    # RDQ_NSLOT-deep double buffer (flashattn sdQaccum_stage): stage s writes its
                    # RDQ_NCOL-col frag slice into slot s%NSLOT at sdQ col slot·RDQ_NCOL (cbase survives
                    # only in the gmem coords); thread-0 keeps NSLOT-1 reduce-adds in flight. The
                    # per-iter wg_sync makes thread-0's drain of slot s-NSLOT visible before that slot
                    # is overwritten (WAR on the async-store source SMEM).
                    for s in range(RDQ_STAGES):
                        cbase = s * RDQ_NCOL
                        col0 = (s % RDQ_NSLOT) * RDQ_NCOL
                        if varlen:
                            # D1: stage fragdQ for valid rows, an EXPLICIT 0.0 for OOB rows (so the
                            # bulk reduce-add's 0-addend on OOB rows is load-bearing, not incidental).
                            k.reg_fill(rzero, 0.0)
                            with k.if_(tid < TILE_M):
                                with k.if_((mb * TILE_M + tid) < vg.slen_q):
                                    for c in range(RDQ_NCOL):
                                        k.reg_store(
                                            _sl(sdQ, (tid, col0 + c), (1, 1)),
                                            _sl(fragdQ, (cbase + c,), (1,)),
                                        )
                                with k.if_((mb * TILE_M + tid) >= vg.slen_q):
                                    for c in range(RDQ_NCOL):
                                        k.reg_store(_sl(sdQ, (tid, col0 + c), (1, 1)), rzero)
                        else:
                            with k.if_(tid < TILE_M):
                                for c in range(RDQ_NCOL):
                                    k.reg_store(
                                        _sl(sdQ, (tid, col0 + c), (1, 1)),
                                        _sl(fragdQ, (cbase + c,), (1,)),
                                    )
                        k.wg_sync(barrier_id=1)
                        k.fence(kind=af, scope=FenceScope.CTA)
                        with k.if_(tid.eq(0)):
                            k.tma_reduce_add(
                                dq_g,
                                _sl(sdQ, (0, col0), (TILE_M, RDQ_NCOL)),
                                coords=(qtok, head, cbase),
                                shape=(TILE_M, RDQ_NCOL),
                                gmem_shape=(TILE_M, 1, RDQ_NCOL),
                            )
                            k.cp_async_bulk_commit_group()
                            k.cp_async_bulk_wait_group_read(RDQ_NSLOT - 1)
                        k.wg_sync(barrier_id=1)
                    # drain the tail (last NSLOT-1 in-flight reduce-adds) before the buffer is reused
                    # next task and before the deterministic release publishes "my dQ is complete".
                    # (1-CTA is always NSLOT==2, so this runs; guarded for the same no-witness reason
                    # as the 2-CTA path.)
                    if RDQ_NSLOT > 1:
                        with k.if_(tid.eq(0)):
                            k.cp_async_bulk_wait_group_read(0)
                    k.wg_sync(barrier_id=1)
                    # DETERMINISTIC: after ALL reduce-adds have FULLY drained
                    # (cp_async_bulk_wait_group_read(0) above, read=False-equivalent — the store is
                    # complete, not just the read buffer freed), signal the NEXT KV-tile CTA. The
                    # release fence + atomic_add publishes a memory state that already includes this
                    # CTA's completed contribution, so CTA nb+1's wait_eq observes it (flashattn
                    # arrive_inc @3672-3695; the membar in red.release orders the add before the bump).
                    if deterministic:
                        with k.if_(tid.eq(0)):
                            k.fence(kind=FenceKind.MEMORY, scope=FenceScope.GPU)
                            k.gmem_atomic_add(
                                dq_sem_g,
                                coords=(batch, head, mb, cta_in_cluster),
                                value=1,
                                order="release",
                            )

    # relay (14): idle in 1-CTA. 2-CTA — ACTIVE (map §1, source relay() @1630-1672): per
    # m-iter it waits dS_cluster_full (the peer's s2cluster dS half landed in OUR sdS_full)
    # then elect_one -> mbarrier_arrive(dS_cluster_leader on the LEADER, remote_coord=0) to
    # release the leader MMA's dQ GEMM. Runs on BOTH CTAs: each CTA's relay arrives the
    # leader's (cluster coord 0) dS_cluster_leader (count 2 -> both arrivals release the dQ
    # GEMM). empty (15): reg donor, idle.
    with k.role(warp=RELAY_WARP):
        with k.for_each_task(sched) as task:
            if use_2cta:
                # VARLEN: the relay's dS_cluster_full wait + dS_cluster_leader arrive must stay
                # balanced with the compute body's s2cluster export, which now runs under the
                # cluster `_skip` guard. So the relay per-mb body runs under the SAME cluster
                # predicate (mb<n_mb_b AND (nb-cta_in_cluster)<n_nb_b): for a skipped (past-Q-tile
                # or past-cluster) mb the compute does NO dS export -> the relay must NOT wait
                # dS_cluster_full or arrive dS_cluster_leader (else it waits forever / over-arrives
                # the leader's dQ gate). Non-varlen: _skip is a nullcontext -> every mb runs (the
                # existing dense behavior). batch/nb/vg/mmin are recovered here for the guard.
                _b, _h, _nb = task_geom(task)
                _vg = varlen_geom(_b) if varlen else None
                _mmin = m_block_min(_nb, _vg) if causal_skip else 0
                for mb in range(n_mb):
                    with _skip(mb, _mmin, _vg, _nb, cluster=True):
                        # wait our half landed (peer's s2cluster -> dS_cluster_full), then release
                        # the leader's dQ GEMM. The arrive is WARP-UNIFORM (one per relay warp = one
                        # per CTA; both CTAs' relays target the LEADER's barrier via remote_coord=0
                        # -> count 2). NB: mbarrier_arrive must NOT be tid-guarded — barrier ops are
                        # warp-cohort-uniform; a tid==0 guard makes the cohort diverge and the arrive
                        # never lands (deadlock).
                        k.mbarrier_wait(bars["dS_cluster_full"], phase=mb % 2)
                        k.mbarrier_arrive(k.mbar_ref(bars["dS_cluster_leader"], remote_coord=0))
    with k.role(warp=EMPTY_WARP):
        with k.for_each_task(sched) as task:
            pass

    with k.kernel_finalize(warp=0):
        k.tmem_dealloc(tmem_base, n_cols=N_COLS_TMEM, cta_group=cg)
    return k.build()

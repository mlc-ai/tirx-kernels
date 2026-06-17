"""End-to-end tests for the FP8 blockwise GEMM port: protocol (trace +
check_protocol) and value correctness against a numpy UE8M0 dequantization
reference, cell-for-cell.

The inputs are integer-valued so the whole pipeline is exact: the per-token /
per-block UE8M0 quantization recovers the original integers on dequantization
(power-of-two scales), the f32 accumulation of integer products is exact, and
both sides round to bf16 identically — so the comparison is `mism == 0`, the
same bar as the fp16/bf16 GEMM port.
"""

import numpy as np
import nymph_rs as nr
import pytest
from nymph_rs.kernels import Fp8BlockwiseGemmConfig, build_fp8_blockwise_gemm
from nymph_rs.kernels.fp8_blockwise_gemm import (
    CONFIGS,
    FP8_CONFIGS_SUPPORTED,
    fp8_blockwise_task_config,
)


def _ue8m0_cast(x: np.ndarray, *, per_block: bool) -> tuple[np.ndarray, np.ndarray]:
    """DeepGEMM-style UE8M0 quantization: per (row, 128-k-chunk) for A, per
    (128-row, 128-k) block for B. Returns (quantized e4m3-exact values, biased
    exponent bytes of shape (rows, k_chunks))."""
    rows, k = x.shape
    chunks = x.reshape(rows, k // 128, 128)
    if per_block:
        # deep_gemm's per_block_cast pads the row dim to a 128 multiple before
        # taking block amaxes, then slices the per-row scales back to N rows.
        padded_rows = (rows + 127) // 128 * 128
        padded = np.zeros((padded_rows, k // 128, 128), dtype=np.float32)
        padded[:rows] = np.abs(chunks)
        amax = padded.reshape(padded_rows // 128, 128, k // 128, 128).max(axis=(1, 3))
        amax = np.repeat(amax, 128, axis=0)[:rows]
    else:
        amax = np.abs(chunks).max(axis=2)
    amax = np.maximum(amax, 1e-4)
    exp = np.ceil(np.log2(amax / 448.0))
    scale = np.exp2(exp)
    quant = (chunks / scale[:, :, None]).reshape(rows, k).astype(np.float32)
    byte = (exp + 127).astype(np.uint8)
    return quant, byte


def _pack_sf(byte: np.ndarray) -> np.ndarray:
    """(rows, k_chunks) u8 -> (k_chunks // 4, rows) u32, 4 little-endian bytes
    per word (the TIRx `sfa_pack` layout)."""
    rows, chunks = byte.shape
    u32 = byte.reshape(rows, chunks // 4, 4).astype(np.uint32)
    packed = u32[:, :, 0] | (u32[:, :, 1] << 8) | (u32[:, :, 2] << 16) | (u32[:, :, 3] << 24)
    return np.ascontiguousarray(packed.T)


def _round_bf16(x: np.ndarray) -> np.ndarray:
    bits = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF_0000
    return rounded.astype(np.uint32).view(np.float32)


def _prepare(m: int, n: int, k: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = rng.integers(-2, 3, size=(m, k)).astype(np.float32)
    b = rng.integers(-2, 3, size=(n, k)).astype(np.float32)
    a_q, a_byte = _ue8m0_cast(a, per_block=False)
    b_q, b_byte = _ue8m0_cast(b, per_block=True)
    a_de = (
        a_q.reshape(m, k // 128, 128) * np.exp2(a_byte.astype(np.float32) - 127)[:, :, None]
    ).reshape(m, k)
    b_de = (
        b_q.reshape(n, k // 128, 128) * np.exp2(b_byte.astype(np.float32) - 127)[:, :, None]
    ).reshape(n, k)
    assert np.array_equal(a_de, a) and np.array_equal(b_de, b)
    ref = _round_bf16(a_de @ b_de.T)
    return a_q, b_q, _pack_sf(a_byte), _pack_sf(b_byte), ref


def test_fp8_blockwise_gemm_builds_supported_configs():
    for cfg in FP8_CONFIGS_SUPPORTED:
        kernel = build_fp8_blockwise_gemm(
            Fp8BlockwiseGemmConfig(m=cfg["m"], n=cfg["n"], k=cfg["k"])
        )
        assert len(kernel.args) == 5


def test_fp8_blockwise_gemm_builds_all_tirx_configs():
    # Every TIRx CONFIGS shape resolves and builds — including the partial-tile
    # squares (TMA tensormap OOB clamping) and the odd-EPI 8192^3 tiling.
    for cfg in CONFIGS:
        kernel = build_fp8_blockwise_gemm(
            Fp8BlockwiseGemmConfig(m=cfg["m"], n=cfg["n"], k=cfg["k"])
        )
        assert len(kernel.args) == 5


def test_fp8_blockwise_gemm_protocol_passes_canonical_task_shape():
    # The canonical setting: the 16384^3 task shape (swap (240,128), 128
    # k-tiles per task) on one cluster pair, task count varied through N.
    # Multi-task continuity matters here: the pipeline stage ring and the mbar
    # phases run on the CONTINUOUS issue sequence (TIRx PipelineState) — a
    # per-task stage reset deadlocks whenever k_tiles % smem_depth != 0.
    kernel = build_fp8_blockwise_gemm(fp8_blockwise_task_config(6))
    report = nr.check_protocol(kernel)
    assert report["status"] == "Passed", report["diagnostics"][:2]
    for entry in report["pass_summary"]:
        assert entry["status"] in ("Passed", "Skipped"), entry


def test_fp8_blockwise_gemm_value_canonical_task_shape():
    cfg = fp8_blockwise_task_config(2)
    m, n, k = cfg.m, cfg.n, cfg.k
    kernel = build_fp8_blockwise_gemm(cfg)
    a_t, b_t, sfa_t, sfb_t, d_t = kernel.args
    a_q, b_q, sfa_pack, sfb_pack, ref = _prepare(m, n, k)
    out = nr.interpret(kernel, {a_t: a_q, b_t: b_q, sfa_t: sfa_pack, sfb_t: sfb_pack})
    d = np.asarray(out[d_t.id], dtype=np.float32).reshape(m, n)
    assert np.isfinite(d).all()
    assert int((d != ref).sum()) == 0


@pytest.mark.parametrize("cfg", FP8_CONFIGS_SUPPORTED, ids=lambda c: c["label"])
def test_fp8_blockwise_gemm_protocol_passes(cfg):
    # The examination setting is one cluster pair walking every task
    # (launch_shape=(2,)) — the protocol object is the single cluster's
    # behavior, the problem shape only sets the task list.
    kernel = build_fp8_blockwise_gemm(
        Fp8BlockwiseGemmConfig(m=cfg["m"], n=cfg["n"], k=cfg["k"], launch_shape=(2,))
    )
    report = nr.check_protocol(kernel)
    assert report["status"] == "Passed", report["diagnostics"]
    for entry in report["pass_summary"]:
        assert entry["status"] in ("Passed", "Skipped"), entry


@pytest.mark.parametrize("cfg", FP8_CONFIGS_SUPPORTED, ids=lambda c: c["label"])
def test_fp8_blockwise_gemm_value_matches_numpy_reference(cfg):
    m, n, k = cfg["m"], cfg["n"], cfg["k"]
    kernel = build_fp8_blockwise_gemm(Fp8BlockwiseGemmConfig(m=m, n=n, k=k, launch_shape=(2,)))
    a_t, b_t, sfa_t, sfb_t, d_t = kernel.args
    a_q, b_q, sfa_pack, sfb_pack, ref = _prepare(m, n, k)
    out = nr.interpret(kernel, {a_t: a_q, b_t: b_q, sfa_t: sfa_pack, sfb_t: sfb_pack})
    d = np.asarray(out[d_t.id], dtype=np.float32).reshape(m, n)
    assert np.isfinite(d).all()
    mism = int((d != ref).sum())
    assert mism == 0, f"{mism} mismatched cells"


def test_fp8_blockwise_gemm_value_is_deterministic():
    m = n = k = 1024
    kernel = build_fp8_blockwise_gemm(Fp8BlockwiseGemmConfig(m=m, n=n, k=k))
    a_t, b_t, sfa_t, sfb_t, d_t = kernel.args
    a_q, b_q, sfa_pack, sfb_pack, _ = _prepare(m, n, k, seed=7)
    inputs = {a_t: a_q, b_t: b_q, sfa_t: sfa_pack, sfb_t: sfb_pack}
    first = np.asarray(nr.interpret(kernel, inputs)[d_t.id], dtype=np.float32)
    second = np.asarray(nr.interpret(kernel, inputs)[d_t.id], dtype=np.float32)
    assert np.array_equal(first, second)


def test_fp8_blockwise_gemm_rejects_unsupported_shapes():
    # k must be a multiple of 512 (4 packed k-tiles per scale word)
    with pytest.raises(ValueError, match="multiple of 512"):
        build_fp8_blockwise_gemm(Fp8BlockwiseGemmConfig(m=1024, n=1024, k=256))

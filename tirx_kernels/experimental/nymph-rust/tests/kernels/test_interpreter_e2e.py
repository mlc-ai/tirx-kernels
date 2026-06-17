"""End-to-end interpreter tests: the fp16/bf16 GEMM through the Rust value
simulator, checked against a numpy reference cell-for-cell.

The GEMM exercises the whole interpreter pipeline — TMA load/store, cta_group=2
MMA, the TMEM collective + scratchpad, mbarrier handshakes (event-driven wake),
cooperative sync, the register ALU/cvt, and the direct-mutation runner — so a
cell-exact match here is a strong correctness signal for the port as a whole.
"""

import numpy as np
import nymph_rs as nr
import pytest
from nymph_rs.kernels import build_fp16_bf16_gemm
from nymph_rs.kernels.fp16_bf16_gemm import Fp16Bf16GemmConfig

try:
    from nymph_rs import DType
except Exception:  # pragma: no cover
    DType = None


def _inputs(cfg, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(-2, 3, size=(cfg.m, cfg.k)).astype(np.float32)
    b = rng.integers(-2, 3, size=(cfg.n, cfg.k)).astype(np.float32)
    return a, b


def _np_dtype(cfg):
    name = str(cfg.dtype)
    if "BF16" in name or "bf16" in name:
        ml = pytest.importorskip("ml_dtypes")
        return ml.bfloat16
    return np.float16


def _reference(a, b, cfg):
    return (a.astype(np.float64) @ b.T).astype(_np_dtype(cfg)).astype(np.float64)


@pytest.mark.parametrize("dtype_name", ["F16", "BF16"])
def test_gemm_matches_numpy_reference(dtype_name):
    if DType is None:
        pytest.skip("DType binding unavailable")
    cfg = Fp16Bf16GemmConfig(dtype=getattr(DType, dtype_name))
    a, b = _inputs(cfg)
    kernel = build_fp16_bf16_gemm(cfg)
    a_t, b_t, c_t = kernel.args
    outputs = nr.interpret(kernel, {a_t: a, b_t: b})
    raw = outputs[c_t.id]
    assert raw.dtype == (np.float16 if dtype_name == "F16" else np.float32)
    c = np.asarray(raw, np.float64).reshape(cfg.m, cfg.n)
    ref = _reference(a, b, cfg)
    assert int(np.sum(c != ref)) == 0


def test_gemm_completes_and_is_deterministic():
    cfg = Fp16Bf16GemmConfig()
    a, b = _inputs(cfg, seed=3)
    kernel = build_fp16_bf16_gemm(cfg)
    a_t, b_t, c_t = kernel.args
    first = np.asarray(nr.interpret(kernel, {a_t: a, b_t: b})[c_t.id], np.float64)
    second = np.asarray(nr.interpret(kernel, {a_t: a, b_t: b})[c_t.id], np.float64)
    assert np.array_equal(first, second)

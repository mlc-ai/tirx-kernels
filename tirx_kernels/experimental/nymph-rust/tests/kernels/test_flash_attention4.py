import numpy as np
import nymph_rs as nr
import pytest
from nymph_rs.kernels.flash_attention4 import CONFIGS, FlashAttention4Config, build_flash_attention4


def _cfg(entry, **kwargs):
    values = dict(
        batch_size=entry["batch_size"],
        seq_len=entry["seq_len"],
        num_qo_heads=entry["num_qo_heads"],
        num_kv_heads=entry["num_kv_heads"],
        head_dim=entry["head_dim"],
        is_causal=entry["is_causal"],
    )
    values.update(kwargs)
    return FlashAttention4Config(**values)


@pytest.mark.parametrize("entry", CONFIGS, ids=[entry["label"] for entry in CONFIGS])
def test_flash_attention4_builds_all_bench_configs(entry):
    kernel = build_flash_attention4(_cfg(entry, launch_shape=(1,)))
    assert len(kernel.args) == 4


def test_flash_attention4_protocol_smoke_min_canonical():
    kernel = build_flash_attention4(
        FlashAttention4Config(seq_len=1024, num_kv_heads=4, is_causal=False, launch_shape=(1,))
    )
    report = nr.check_protocol(kernel)
    assert report["status"] == "Passed"


def test_flash_attention4_protocol_smoke_min_causal():
    kernel = build_flash_attention4(
        FlashAttention4Config(seq_len=1024, num_kv_heads=4, is_causal=True, launch_shape=(1,))
    )
    report = nr.check_protocol(kernel)
    assert report["status"] == "Passed"


def _reference_attention(q, k, v, cfg: FlashAttention4Config):
    gqa_ratio = cfg.num_qo_heads // cfg.num_kv_heads
    q_ref = q.astype(np.float32).reshape(1, cfg.seq_len, cfg.num_kv_heads, gqa_ratio, cfg.head_dim)
    q_ref = q_ref.transpose(0, 2, 3, 1, 4)
    k_ref = k.astype(np.float32).transpose(0, 2, 1, 3)
    v_ref = v.astype(np.float32).transpose(0, 2, 1, 3)
    scores = np.einsum("bhgmd,bhnd->bhgmn", q_ref, k_ref) / np.sqrt(cfg.head_dim, dtype=np.float32)
    if cfg.is_causal:
        rows = np.arange(cfg.seq_len)[:, None]
        cols = np.arange(cfg.seq_len)[None, :]
        scores = np.where(cols > rows, -np.inf, scores)
    scores -= scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    ref = np.einsum("bhgmn,bhnd->bhgmd", probs, v_ref)
    ref = ref.transpose(0, 3, 1, 2, 4).reshape(1, cfg.seq_len, cfg.num_qo_heads, cfg.head_dim)
    return ref.astype(np.float16).astype(np.float32)


@pytest.mark.parametrize("is_causal", [False, True], ids=["noncausal", "causal"])
def test_flash_attention4_value_matches_numpy_reference_small(is_causal):
    cfg = FlashAttention4Config(seq_len=128, num_kv_heads=4, is_causal=is_causal, launch_shape=(1,))
    kernel = build_flash_attention4(cfg)
    q_t, k_t, v_t, o_t = kernel.args
    rng = np.random.default_rng(2)
    q = (rng.normal(size=(1, cfg.seq_len, cfg.num_qo_heads, cfg.head_dim)) * 0.25).astype(
        np.float16
    )
    k = (rng.normal(size=(1, cfg.seq_len, cfg.num_kv_heads, cfg.head_dim)) * 0.25).astype(
        np.float16
    )
    v = (rng.normal(size=(1, cfg.seq_len, cfg.num_kv_heads, cfg.head_dim)) * 0.25).astype(
        np.float16
    )

    out = np.asarray(nr.interpret(kernel, {q_t: q, k_t: k, v_t: v})[o_t.id], dtype=np.float32)
    ref = _reference_attention(q, k, v, cfg)

    np.testing.assert_allclose(out, ref, atol=7e-4, rtol=3e-2)


def test_flash_attention4_value_matches_numpy_reference_causal_multiblock_gqa1():
    cfg = FlashAttention4Config(
        seq_len=256, num_qo_heads=32, num_kv_heads=32, is_causal=True, launch_shape=(1,)
    )
    kernel = build_flash_attention4(cfg)
    q_t, k_t, v_t, o_t = kernel.args
    rng = np.random.default_rng(3)
    q = (rng.normal(size=(1, cfg.seq_len, cfg.num_qo_heads, cfg.head_dim)) * 0.25).astype(
        np.float16
    )
    k = (rng.normal(size=(1, cfg.seq_len, cfg.num_kv_heads, cfg.head_dim)) * 0.25).astype(
        np.float16
    )
    v = (rng.normal(size=(1, cfg.seq_len, cfg.num_kv_heads, cfg.head_dim)) * 0.25).astype(
        np.float16
    )

    out = np.asarray(nr.interpret(kernel, {q_t: q, k_t: k, v_t: v})[o_t.id], dtype=np.float32)
    ref = _reference_attention(q, k, v, cfg)

    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, ref, atol=7e-4, rtol=3e-2)

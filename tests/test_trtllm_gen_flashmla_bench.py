from __future__ import annotations

from types import SimpleNamespace
from unittest import SkipTest, mock

import pytest
import torch

from tirx_kernels.flashmla._trtllm_gen_bench import (
    PAGE_SIZE,
    is_sm100_device,
    make_identity_paged_kv_buffer,
    prepare_trtllm_block_tables,
    probe_trtllm_gen_launch,
    trtllm_gen_config_compatible,
    trtllm_gen_incompatible_reason,
)


def _config(**overrides) -> SimpleNamespace:
    values = {
        "s_q": 1,
        "s_kv": 2048,
        "d_qk": 576,
        "d_v": 512,
        "topk": 512,
        "h_q": 64,
        "h_kv": 1,
        "have_attn_sink": False,
        "have_topk_length": False,
        "inject_invalid_indices": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_trtllm_gen_config_compatibility_matrix() -> None:
    assert trtllm_gen_config_compatible(_config())
    assert trtllm_gen_config_compatible(_config(topk=2048, h_q=128))
    assert "d_qk=512" in trtllm_gen_incompatible_reason(_config(d_qk=512))
    assert "topk=1280" in trtllm_gen_incompatible_reason(_config(topk=1280, h_q=128))
    assert "inject_invalid_indices=True" in trtllm_gen_incompatible_reason(
        _config(inject_invalid_indices=True)
    )
    assert "have_topk_length=True" in trtllm_gen_incompatible_reason(_config(have_topk_length=True))


def test_probe_trtllm_gen_launch_wraps_tactic_runtime_error() -> None:
    def _launch() -> None:
        raise RuntimeError("failed to find a valid tactic")

    with pytest.raises(SkipTest, match="tactic unavailable"):
        probe_trtllm_gen_launch(_launch)


def test_probe_trtllm_gen_launch_propagates_value_error() -> None:
    def _launch() -> None:
        raise ValueError("wiring error")

    with pytest.raises(ValueError, match="wiring error"):
        probe_trtllm_gen_launch(_launch)

    def _runtime_error() -> None:
        raise RuntimeError("cuda illegal memory access")

    with pytest.raises(RuntimeError, match="cuda illegal memory access"):
        probe_trtllm_gen_launch(_runtime_error)


def test_identity_paged_kv_shapes_and_flat_view() -> None:
    kv_paged, kv_flat = make_identity_paged_kv_buffer(100, 576, device="cpu", dtype=torch.bfloat16)
    assert kv_paged.shape == (2, 1, PAGE_SIZE, 576)
    assert kv_flat.shape == (100, 1, 576)

    kv_flat[0, 0, 3] = torch.tensor(1.25, dtype=torch.bfloat16)
    kv_flat[99, 0, 7] = torch.tensor(-2.5, dtype=torch.bfloat16)
    page, off = divmod(99, PAGE_SIZE)
    assert page == 1 and off == 35
    assert kv_paged[0, 0, 0, 3].item() == pytest.approx(1.25, rel=0, abs=1e-3)
    assert kv_paged[page, 0, off, 7].item() == pytest.approx(-2.5, rel=0, abs=1e-3)


def test_identity_slot_mapping_is_linear() -> None:
    s_kv = 250
    num_pages = (s_kv + PAGE_SIZE - 1) // PAGE_SIZE
    for slot in (0, 63, 64, 127, 249):
        page, off = divmod(slot, PAGE_SIZE)
        assert page * PAGE_SIZE + off == slot
        assert page < num_pages


def test_validate_trtllm_sparse_indices_rejects_invalid_entries() -> None:
    valid = torch.tensor([[[10, 20, 30, 40]]], dtype=torch.int32)
    block_tables = prepare_trtllm_block_tables(valid, s_kv=256, topk=4)
    torch.testing.assert_close(block_tables, valid)

    invalid = torch.tensor([[[10, 20, -1, 300]]], dtype=torch.int32)
    with pytest.raises(ValueError, match="invalid values"):
        prepare_trtllm_block_tables(invalid, s_kv=256, topk=4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_trtllm_gen_backend_does_not_fall_back_to_other_backends() -> None:
    from tirx_kernels.flashmla._trtllm_gen_bench import prepare_trtllm_gen_launch
    from tirx_kernels.flashmla.sparse_prefill_head64_phase1 import prepare_data

    if not is_sm100_device():
        pytest.skip("SM100 required")

    case = prepare_data(
        label="backend_probe",
        s_q=2,
        s_kv=512,
        topk=512,
        d_qk=576,
        h_q=64,
        have_attn_sink=False,
        seed=0,
    )
    try:
        launch, _ = prepare_trtllm_gen_launch(case, probe=False)
    except SkipTest as exc:
        pytest.skip(str(exc))

    with (
        mock.patch(
            "flashinfer.mla._core.xqa_batch_decode_with_kv_cache_mla",
            side_effect=AssertionError("xqa fallback"),
        ),
        mock.patch(
            "flashinfer.cute_dsl.attention.cute_dsl_mla_decode",
            side_effect=AssertionError("cute-dsl fallback"),
        ),
    ):
        launch()
    torch.cuda.synchronize()


def _max_abs_rel_error(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    abs_err = (actual.float() - expected.float()).abs()
    max_abs = float(abs_err.max().item())
    max_rel = float((abs_err / expected.float().abs().clamp_min(1e-6)).max().item())
    return max_abs, max_rel


def _assert_sparse_prefill_refs_match_tirx(
    *, tirx_out: torch.Tensor, flashmla_out: torch.Tensor, trtllm_out: torch.Tensor, label: str
) -> None:
    rtol, atol = 0.02, 0.01
    pairs = (("flashmla", flashmla_out), ("trtllm_gen", trtllm_out))
    for name, out in pairs:
        max_abs, max_rel = _max_abs_rel_error(out, tirx_out)
        print(f"{label} {name} vs tirx: max_abs={max_abs:.6g} max_rel={max_rel:.6g}")
        torch.testing.assert_close(out, tirx_out, rtol=rtol, atol=atol)

    flash_trtllm_abs, flash_trtllm_rel = _max_abs_rel_error(trtllm_out, flashmla_out)
    print(
        f"{label} trtllm_gen vs flashmla: "
        f"max_abs={flash_trtllm_abs:.6g} max_rel={flash_trtllm_rel:.6g}"
    )
    torch.testing.assert_close(trtllm_out, flashmla_out, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not is_sm100_device(), reason="SM100 required")
def test_trtllm_gen_matches_tirx_and_flashmla_head64() -> None:
    from tirx_kernels.flashmla._flashmla_bench import run_flashmla_sparse_prefill
    from tirx_kernels.flashmla._trtllm_gen_bench import run_trtllm_gen_sparse_prefill
    from tirx_kernels.flashmla.sparse_prefill_head64_phase1 import (
        _tirx_args,
        get_kernel,
        prepare_data,
    )
    from tirx_kernels.runner import compile_kernel

    kwargs = dict(
        label="correctness_hq64",
        s_q=32,
        s_kv=8192,
        topk=512,
        d_qk=576,
        h_q=64,
        have_attn_sink=True,
        seed=3,
    )
    case = prepare_data(**kwargs)
    ex = compile_kernel(get_kernel(**kwargs))
    ex(*_tirx_args(case))
    torch.cuda.synchronize()
    tirx_out = case["out"].clone()

    flashmla_out = run_flashmla_sparse_prefill(case)
    trtllm_out = run_trtllm_gen_sparse_prefill(case)
    _assert_sparse_prefill_refs_match_tirx(
        tirx_out=tirx_out, flashmla_out=flashmla_out, trtllm_out=trtllm_out, label="hq64/topk512"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not is_sm100_device(), reason="SM100 required")
def test_trtllm_gen_matches_tirx_and_flashmla_head128() -> None:
    from tirx_kernels.flashmla._flashmla_bench import run_flashmla_sparse_prefill
    from tirx_kernels.flashmla._trtllm_gen_bench import run_trtllm_gen_sparse_prefill
    from tirx_kernels.flashmla.sparse_prefill_head128_phase1 import (
        _tirx_args,
        get_kernel,
        prepare_data,
    )
    from tirx_kernels.runner import compile_kernel

    kwargs = dict(
        label="correctness_hq128",
        s_q=32,
        s_kv=8192,
        topk=2048,
        d_qk=576,
        h_q=128,
        have_attn_sink=True,
        seed=4,
    )
    case = prepare_data(**kwargs)
    ex = compile_kernel(get_kernel(**kwargs))
    ex(*_tirx_args(case))
    torch.cuda.synchronize()
    tirx_out = case["out"].clone()

    flashmla_out = run_flashmla_sparse_prefill(case)
    trtllm_out = run_trtllm_gen_sparse_prefill(case)
    _assert_sparse_prefill_refs_match_tirx(
        tirx_out=tirx_out, flashmla_out=flashmla_out, trtllm_out=trtllm_out, label="hq128/topk2048"
    )

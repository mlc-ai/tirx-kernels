from __future__ import annotations

import pytest

from tirx_kernels.flashmla.flash_mla_sparse_fwd import (
    CONFIGS,
    KERNEL_META,
    dispatch_reason,
    select_kernel,
)


def test_sparse_flashmla_prefill_dispatches_by_shape() -> None:
    assert (
        select_kernel(h_q=64, h_kv=1, d_qk=512, d_v=512, topk=512)
        == "sparse_flashmla_prefill_head64_phase1"
    )
    assert (
        select_kernel(h_q=128, h_kv=1, d_qk=512, d_v=512, topk=1280)
        == "sparse_flashmla_prefill_head128_small_topk_phase1"
    )
    assert (
        select_kernel(h_q=128, h_kv=1, d_qk=512, d_v=512, topk=1408)
        == "sparse_flashmla_prefill_head128_phase1"
    )
    assert (
        select_kernel(h_q=128, h_kv=1, d_qk=576, d_v=512, topk=1280)
        == "sparse_flashmla_prefill_head128_phase1"
    )


def test_sparse_flashmla_prefill_dispatch_rejects_out_of_scope_shapes() -> None:
    with pytest.raises(ValueError, match="h_q == 64 or 128"):
        select_kernel(h_q=96, h_kv=1, d_qk=512, d_v=512, topk=512)
    with pytest.raises(ValueError, match="h_kv == 1"):
        select_kernel(h_q=64, h_kv=2, d_qk=512, d_v=512, topk=512)
    with pytest.raises(ValueError, match="d_qk == 512 or 576"):
        select_kernel(h_q=64, h_kv=1, d_qk=640, d_v=512, topk=512)


def test_sparse_flashmla_prefill_configs_cover_three_impls() -> None:
    selected = {select_kernel(**cfg) for cfg in CONFIGS}
    assert selected == {
        "sparse_flashmla_prefill_head64_phase1",
        "sparse_flashmla_prefill_head128_phase1",
        "sparse_flashmla_prefill_head128_small_topk_phase1",
    }
    labels = [cfg["label"] for cfg in CONFIGS]
    assert len(labels) == len(set(labels))
    assert KERNEL_META["name"] == "flash_mla_sparse_fwd"
    assert "small-topk" in dispatch_reason(h_q=128, h_kv=1, d_qk=512, d_v=512, topk=1280)

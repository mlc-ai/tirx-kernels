from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import Any


def _import_flash_mla():
    path = os.environ.get("FLASH_MLA_PATH", os.path.expanduser("~/FlashMLA"))
    if path not in sys.path:
        sys.path.insert(0, path)
    import flash_mla

    return flash_mla


def run_flashmla_sparse_prefill(case: dict[str, Any]):
    flash_mla = _import_flash_mla()
    cfg = case["config"]
    out, _, _ = flash_mla.flash_mla_sparse_fwd(
        case["q"],
        case["kv"],
        case["indices"],
        case["sm_scale"],
        d_v=cfg.d_v,
        attn_sink=case["attn_sink"] if cfg.have_attn_sink else None,
        topk_length=case["topk_length"] if cfg.have_topk_length else None,
    )
    return out


def flashmla_reference_builder(case: dict[str, Any]) -> Callable[[], Any]:
    _import_flash_mla()
    return lambda: run_flashmla_sparse_prefill(case)

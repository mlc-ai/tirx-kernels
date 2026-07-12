"""TIRx entry for FlashMLA's sparse prefill Python API.

FlashMLA exposes sparse prefill as ``flash_mla.flash_mla_sparse_fwd`` in Python,
which calls the extension symbol ``sparse_prefill_fwd`` and dispatches through
``sparse_attn_prefill_interface`` in ``csrc/api/sparse_fwd.h``. This module keeps
that public API name for the TIRx registry entry and forwards to the three
ported SM100 phase1 implementations by shape.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

from tirx_kernels.flashmla import sparse_prefill_head64_phase1 as _head64
from tirx_kernels.flashmla import sparse_prefill_head128_phase1 as _head128
from tirx_kernels.flashmla import sparse_prefill_head128_small_topk_phase1 as _small_topk

KERNEL_META = {"name": "flash_mla_sparse_fwd", "category": "flashmla", "compute_capability": 10}

_HEAD64_NAME = _head64.KERNEL_META["name"]
_HEAD128_NAME = _head128.KERNEL_META["name"]
_SMALL_TOPK_NAME = _small_topk.KERNEL_META["name"]


def _config(cfg: dict[str, Any], *, h_q: int, d_qk: int | None = None) -> dict[str, Any]:
    out = dict(cfg)
    out.setdefault("h_q", h_q)
    out.setdefault("h_kv", 1)
    out.setdefault("d_v", 512)
    if d_qk is not None:
        out.setdefault("d_qk", d_qk)
    return out


CONFIGS = [
    *[_config(cfg, h_q=_head64.B_H) for cfg in _head64.CONFIGS],
    *[_config(cfg, h_q=_head128.B_H) for cfg in _head128.CONFIGS],
    *[_config(cfg, h_q=_small_topk.B_H, d_qk=_small_topk.D_QK) for cfg in _small_topk.CONFIGS],
]
BENCH_CONFIGS = CONFIGS


def _required_int(kwargs: dict[str, Any], name: str) -> int:
    value = kwargs.get(name)
    if value is None:
        raise ValueError(f"{name} is required for sparse FlashMLA prefill dispatch")
    return int(value)


def _optional_int(kwargs: dict[str, Any], name: str, default: int) -> int:
    value = kwargs.get(name, default)
    if value is None:
        raise ValueError(f"{name} must not be None for sparse FlashMLA prefill dispatch")
    return int(value)


def _select_impl(**kwargs: Any) -> tuple[str, ModuleType, str]:
    h_q = _required_int(kwargs, "h_q")
    d_qk = _required_int(kwargs, "d_qk")
    topk = _required_int(kwargs, "topk")
    h_kv = _optional_int(kwargs, "h_kv", 1)
    d_v = _optional_int(kwargs, "d_v", 512)

    if h_kv != 1:
        raise ValueError("sparse FlashMLA prefill TIRx ports currently require h_kv == 1")
    if d_v != 512:
        raise ValueError("sparse FlashMLA prefill TIRx ports currently require d_v == 512")
    if d_qk not in (512, 576):
        raise ValueError("sparse FlashMLA prefill supports d_qk == 512 or 576")
    if topk <= 0:
        raise ValueError("sparse FlashMLA prefill requires topk > 0")

    if h_q == 64:
        return _HEAD64_NAME, _head64, "sm100 h_q=64 dispatches to head64 phase1"
    if h_q != 128:
        raise ValueError("sparse FlashMLA prefill supports h_q == 64 or 128")

    # Matches FlashMLA csrc/api/sparse_fwd.h for the scoped SM100 prefill ports:
    # small-topk supports head128 + d_qk=512, otherwise the regular head128 path is used.
    if d_qk == 512 and topk <= 1280:
        return (
            _SMALL_TOPK_NAME,
            _small_topk,
            "sm100 h_q=128 d_qk=512 topk<=1280 dispatches to head128 small-topk phase1",
        )
    return _HEAD128_NAME, _head128, "sm100 h_q=128 dispatches to regular head128 phase1"


def select_kernel(**kwargs: Any) -> str:
    name, _mod, _reason = _select_impl(**kwargs)
    return name


def dispatch_reason(**kwargs: Any) -> str:
    _name, _mod, reason = _select_impl(**kwargs)
    return reason


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    name, mod, reason = _select_impl(**kwargs)
    case = mod.prepare_data(**kwargs)
    if "dispatch_reason" in case:
        case["implementation_dispatch_reason"] = case["dispatch_reason"]
    case["dispatch_reason"] = reason
    case["dispatch_kernel"] = name
    return case


def get_kernel(**kwargs: Any):
    _name, mod, _reason = _select_impl(**kwargs)
    return mod.get_kernel(**kwargs)


def run_test(**kwargs: Any) -> None:
    _name, mod, _reason = _select_impl(**kwargs)
    mod.run_test(**kwargs)


def run_bench(
    *, warmup: int | None = None, repeat: int | None = None, timer: str | None = None, **kwargs: Any
) -> dict[str, Any]:
    name, mod, reason = _select_impl(**kwargs)
    result = mod.run_bench(warmup=warmup, repeat=repeat, timer=timer, **kwargs)
    if isinstance(result, dict):
        result.setdefault("dispatch_kernel", name)
        result.setdefault("dispatch_reason", reason)
    return result


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "dispatch_reason",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
    "select_kernel",
]

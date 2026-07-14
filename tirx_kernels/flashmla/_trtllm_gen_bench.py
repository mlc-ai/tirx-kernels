from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest import SkipTest

import torch

PAGE_SIZE = 64
D_QK = 576
D_V = 512
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = 128
SUPPORTED_TOPKS = frozenset({512, 1024, 2048})
SUPPORTED_H_Q = frozenset({64, 128})
TRTLLM_GEN_WORKSPACE_BYTES = 128 * 1024 * 1024

_TACTIC_UNAVAILABLE_MARKERS = (
    "no valid tactic",
    "no available tactic",
    "tactic unavailable",
    "failed to find a valid tactic",
    "kernel not found",
)


def is_sm100_device(device: torch.device | str | None = None) -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(device)
    return major == 10 and minor == 0


def trtllm_gen_incompatible_reason(cfg: Any) -> str | None:
    if cfg.d_qk != D_QK:
        return f"d_qk={cfg.d_qk} (TRTLLM-Gen sparse MLA requires {D_QK})"
    if cfg.d_v != D_V:
        return f"d_v={cfg.d_v} (TRTLLM-Gen sparse MLA requires {D_V})"
    if cfg.topk not in SUPPORTED_TOPKS:
        return f"topk={cfg.topk} (supported topk: {sorted(SUPPORTED_TOPKS)})"
    if cfg.h_q not in SUPPORTED_H_Q:
        return f"h_q={cfg.h_q} (supported h_q: {sorted(SUPPORTED_H_Q)})"
    if cfg.h_kv != 1:
        return f"h_kv={cfg.h_kv} (TRTLLM-Gen sparse MLA requires h_kv == 1)"
    if cfg.have_topk_length:
        return "have_topk_length=True is outside the TRTLLM-Gen sparse contract"
    if cfg.inject_invalid_indices:
        return "inject_invalid_indices=True is outside the TRTLLM-Gen sparse contract"
    return None


def trtllm_gen_config_compatible(cfg: Any) -> bool:
    return trtllm_gen_incompatible_reason(cfg) is None


def make_identity_paged_kv_buffer(
    s_kv: int, d_qk: int, *, device: torch.device | str, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    num_pages = (s_kv + PAGE_SIZE - 1) // PAGE_SIZE
    kv_paged = torch.zeros((num_pages, 1, PAGE_SIZE, d_qk), device=device, dtype=dtype)
    kv_flat = kv_paged.view(-1, 1, d_qk)[:s_kv]
    return kv_paged, kv_flat


def prepare_trtllm_block_tables(indices: torch.Tensor, *, s_kv: int, topk: int) -> torch.Tensor:
    block_tables = indices[:, :, :topk].contiguous()
    flat = block_tables.view(-1)
    invalid = (flat < 0) | (flat >= s_kv)
    if torch.any(invalid):
        bad = flat[invalid][:8].tolist()
        raise ValueError(
            "TRTLLM-Gen sparse MLA requires all block_tables entries in "
            f"[0, s_kv={s_kv}); got invalid values including {bad}"
        )
    return block_tables


def _load_trtllm_decode(device: torch.device):
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for TRTLLM-Gen sparse MLA baseline")
    if not is_sm100_device(device):
        major, minor = torch.cuda.get_device_capability(device)
        raise SkipTest(f"TRTLLM-Gen baseline requires SM100 (10.0), got SM{major}{minor}")
    try:
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
    except ImportError as exc:
        raise SkipTest(f"FlashInfer TRTLLM-Gen decode is unavailable: {exc}") from exc
    return trtllm_batch_decode_with_kv_cache_mla


def probe_trtllm_gen_launch(launch: Callable[[], None]) -> None:
    try:
        launch()
        torch.cuda.synchronize()
    except RuntimeError as exc:
        if any(marker in str(exc).lower() for marker in _TACTIC_UNAVAILABLE_MARKERS):
            raise SkipTest(f"TRTLLM-Gen sparse MLA tactic unavailable: {exc}") from exc
        raise


def prepare_trtllm_gen_launch(
    case: dict[str, Any], *, probe: bool = True
) -> tuple[Callable[[], None], torch.Tensor]:
    cfg = case["config"]
    reason = trtllm_gen_incompatible_reason(cfg)
    if reason is not None:
        raise ValueError(reason)

    device = case["q"].device
    decode = _load_trtllm_decode(device)
    kv_paged, kv_flat = make_identity_paged_kv_buffer(
        cfg.s_kv, cfg.d_qk, device=device, dtype=case["kv"].dtype
    )
    kv_flat.copy_(case["kv"])

    query = case["q"].view(cfg.s_q, 1, cfg.h_q, cfg.d_qk)
    block_tables = prepare_trtllm_block_tables(case["indices"], s_kv=cfg.s_kv, topk=cfg.topk)
    seq_lens = torch.full((cfg.s_q,), cfg.s_kv, dtype=torch.int32, device=device)
    workspace = torch.zeros(TRTLLM_GEN_WORKSPACE_BYTES, dtype=torch.int8, device=device)
    out = torch.empty((cfg.s_q, 1, cfg.h_q, D_V), dtype=torch.bfloat16, device=device)

    launch_kwargs = {
        "query": query,
        "kv_cache": kv_paged,
        "workspace_buffer": workspace,
        "qk_nope_head_dim": QK_NOPE_HEAD_DIM,
        "kv_lora_rank": KV_LORA_RANK,
        "qk_rope_head_dim": QK_ROPE_HEAD_DIM,
        "block_tables": block_tables,
        "seq_lens": seq_lens,
        "max_seq_len": cfg.s_kv,
        "sparse_mla_top_k": cfg.topk,
        "out": out,
        "bmm1_scale": case["sm_scale"],
        "bmm2_scale": 1.0,
        "sinks": case["attn_sink"] if cfg.have_attn_sink else None,
        "backend": "trtllm-gen",
    }

    def launch() -> None:
        decode(**launch_kwargs)

    if probe:
        probe_trtllm_gen_launch(launch)
    return launch, out


def run_trtllm_gen_sparse_prefill(case: dict[str, Any]) -> torch.Tensor:
    launch, out = prepare_trtllm_gen_launch(case)
    launch()
    torch.cuda.synchronize()
    return out.squeeze(1)


def trtllm_gen_reference_builder(case: dict[str, Any]) -> Callable[[], None]:
    launch, _ = prepare_trtllm_gen_launch(case)
    return launch

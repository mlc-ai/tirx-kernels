# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import ctypes
from dataclasses import asdict, dataclass
from functools import cache
from importlib.util import find_spec
from typing import Any
from unittest import SkipTest

import torch

from tvm.ir.type import PointerType, PrimType

_DEEP_GEMM_MODULE_NAME = "deep_gemm"
_SM100_SMEM_CAPACITY = 232448
_TEST_DIFF_THRESHOLD = 5e-6
_CONTEXT_PATTERNS = ("random_2d", "sglang_fixed", "sglang_ragged")


@dataclass(frozen=True)
class PagedMQALogitsFP8Config:
    batch_size: int = 1
    next_n: int = 1
    max_num_pages: int = 4
    num_pages: int = 128
    num_heads: int = 64
    head_dim: int = 128
    page_size: int = 64
    logits_dtype: str = "float32"
    seed: int = 0
    num_sms: int = 148
    context_lens_2d: bool = True
    varlen: bool = False
    indices_pair_stride: int = 1
    context_pattern: str = "random_2d"

    @property
    def max_context_len(self) -> int:
        return self.max_num_pages * self.page_size

    @property
    def split_kv(self) -> int:
        return 256

    @property
    def block_kv(self) -> int:
        return self.page_size

    @property
    def logits_stride(self) -> int:
        return _align_up(self.max_context_len, self.split_kv)

    def validate(self) -> None:
        if self.batch_size <= 0 or self.next_n <= 0:
            raise ValueError("batch_size and next_n must be positive")
        if self.num_heads not in (32, 64):
            raise ValueError("num_heads must be 32 or 64")
        if self.head_dim not in (32, 64, 128):
            raise ValueError("head_dim must be 32, 64, or 128")
        if self.page_size not in (32, 64):
            raise ValueError("page_size must match DeepGEMM block_kv 32 or 64")
        if self.split_kv % self.page_size != 0:
            raise ValueError("split_kv must be divisible by page_size")
        if self.max_num_pages <= 0 or self.num_pages < self.max_num_pages:
            raise ValueError("num_pages must cover max_num_pages")
        if self.logits_dtype not in ("float32", "bfloat16"):
            raise ValueError("logits_dtype must be 'float32' or 'bfloat16'")
        if not self.context_lens_2d:
            raise ValueError("DeepGEMM paged FP8 API currently requires 2D context_lens")
        if self.varlen and self.next_n != 1:
            raise ValueError("DeepGEMM varlen paged mode requires next_n == 1")
        if self.indices_pair_stride <= 0:
            raise ValueError("indices_pair_stride must be positive")
        if self.context_pattern not in _CONTEXT_PATTERNS:
            raise ValueError(
                f"context_pattern must be one of {_CONTEXT_PATTERNS}, got {self.context_pattern!r}"
            )


def _make_config(**kwargs: Any) -> PagedMQALogitsFP8Config:
    kwargs = {key: value for key, value in kwargs.items() if key != "label"}
    config = PagedMQALogitsFP8Config(**kwargs)
    config.validate()
    return config


def _align_up(x: int, y: int) -> int:
    return (x + y - 1) // y * y


def _torch_logits_dtype(dtype: str) -> torch.dtype:
    if dtype == "float32":
        return torch.float32
    if dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported logits_dtype: {dtype}")


def _config_label(config: dict[str, Any]) -> str:
    dtype = "f32" if config["logits_dtype"] == "float32" else "bf16"
    mode = "varlen" if config.get("varlen", False) else "fixed"
    context_suffix = {"random_2d": "", "sglang_fixed": "_sgfixed", "sglang_ragged": "_sgragged"}[
        config.get("context_pattern", "random_2d")
    ]
    return (
        f"b{config['batch_size']}_n{config['next_n']}_mp{config['max_num_pages']}_"
        f"ps{config['page_size']}_h{config['num_heads']}_d{config['head_dim']}_{dtype}_{mode}"
        f"{context_suffix}"
    )


def _make_case(
    *,
    batch_size: int,
    next_n: int,
    max_num_pages: int,
    num_pages: int,
    page_size: int,
    logits_dtype: str,
    seed: int,
    num_heads: int = 64,
    head_dim: int = 128,
    varlen: bool = False,
    context_pattern: str = "random_2d",
) -> dict[str, Any]:
    config = {
        "batch_size": batch_size,
        "next_n": next_n,
        "max_num_pages": max_num_pages,
        "num_pages": num_pages,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "page_size": page_size,
        "logits_dtype": logits_dtype,
        "seed": seed,
        "varlen": varlen,
        "context_pattern": context_pattern,
    }
    config["label"] = _config_label(config)
    return config


KERNEL_META = {
    "name": "deepgemm_sm100_fp8_paged_mqa_logits",
    "category": "deepgemm",
    "compute_capability": 10,
}

DSA_INDEXER_LIKE_COVERAGE = [
    _make_case(
        batch_size=batch_size,
        next_n=1,
        max_num_pages=max_num_pages,
        num_pages=max(11923, max_num_pages),
        page_size=64,
        logits_dtype=logits_dtype,
        seed=2000 + seed,
    )
    for seed, (batch_size, max_num_pages, logits_dtype) in enumerate(
        (batch_size, max_num_pages, logits_dtype)
        for logits_dtype in ("float32", "bfloat16")
        for batch_size in (1, 2, 4, 8, 16)
        for max_num_pages in (1, 8, 32, 128)
    )
]

# Complete upstream defaults from
# benchmark/kernels/deepseek/benchmark_cute_dsl_fp8_paged_mqa_logits.py:
#   batch_size = (1, 2, 4, 6, 8, 10, 12, 14, 16)
#   next_n = (1, 2, 4, 6)
#   context_len = (4096, 10240, 32768, 81920, 131072)
#   num_heads = 32, head_dim = 128, block_kv = 64, output_dtype = float32
#   varlen = False, use_cuda_graph = True
# The Cartesian product contains 9 * 4 * 5 = 180 configs. Extending the same
# grid to num_heads = (32, 64) contains 360 configs.
#
# SGLANG_BENCH_CONFIGS is currently a curated 80-config kernel-only subset:
#   decode: H=(32,64) x B=(1,2,4,8,16) x every context_len = 50
#   target verify: H=(32,64) x next_n=(2,4,6) x the five paired (B, pages)
#                  points below = 30
# max_num_pages is context_len / page_size, with page_size fixed at 64.
_SGLANG_CONTEXT_PAGES = (64, 160, 512, 1280, 2048)
_SGLANG_TARGET_VERIFY_POINTS = ((1, 64), (2, 2048), (6, 160), (10, 512), (16, 1280))

_SGLANG_DECODE_BENCH_CONFIGS = [
    _make_case(
        batch_size=batch_size,
        next_n=1,
        max_num_pages=max_num_pages,
        num_pages=max(11923, batch_size * max_num_pages),
        num_heads=num_heads,
        page_size=64,
        logits_dtype="float32",
        seed=3000 + seed,
        context_pattern="sglang_fixed",
    )
    for seed, (num_heads, batch_size, max_num_pages) in enumerate(
        (num_heads, batch_size, max_num_pages)
        for num_heads in (32, 64)
        for batch_size in (1, 2, 4, 8, 16)
        for max_num_pages in _SGLANG_CONTEXT_PAGES
    )
]

_SGLANG_TARGET_VERIFY_BENCH_CONFIGS = [
    _make_case(
        batch_size=batch_size,
        next_n=next_n,
        max_num_pages=max_num_pages,
        num_pages=max(11923, batch_size * max_num_pages),
        num_heads=num_heads,
        page_size=64,
        logits_dtype="float32",
        seed=4000 + seed,
        context_pattern="sglang_fixed",
    )
    for seed, (num_heads, next_n, batch_size, max_num_pages) in enumerate(
        (num_heads, next_n, batch_size, max_num_pages)
        for num_heads in (32, 64)
        for next_n in (2, 4, 6)
        for batch_size, max_num_pages in _SGLANG_TARGET_VERIFY_POINTS
    )
]

SGLANG_BENCH_CONFIGS = _SGLANG_DECODE_BENCH_CONFIGS + _SGLANG_TARGET_VERIFY_BENCH_CONFIGS

CONFIGS = [
    _make_case(
        batch_size=1,
        next_n=1,
        max_num_pages=4,
        num_pages=128,
        page_size=64,
        logits_dtype="float32",
        seed=0,
    ),
    _make_case(
        batch_size=2,
        next_n=1,
        max_num_pages=4,
        num_pages=128,
        page_size=64,
        logits_dtype="bfloat16",
        seed=1,
    ),
    _make_case(
        batch_size=2,
        next_n=3,
        max_num_pages=4,
        num_pages=128,
        page_size=64,
        logits_dtype="float32",
        seed=2,
    ),
    _make_case(
        batch_size=1,
        next_n=1,
        max_num_pages=2,
        num_pages=128,
        num_heads=32,
        page_size=64,
        logits_dtype="float32",
        seed=10,
        context_pattern="sglang_fixed",
    ),
    _make_case(
        batch_size=2,
        next_n=2,
        max_num_pages=16,
        num_pages=128,
        num_heads=32,
        page_size=64,
        logits_dtype="float32",
        seed=11,
        context_pattern="sglang_fixed",
    ),
    _make_case(
        batch_size=4,
        next_n=3,
        max_num_pages=64,
        num_pages=256,
        num_heads=64,
        page_size=64,
        logits_dtype="float32",
        seed=12,
        context_pattern="sglang_ragged",
    ),
    _make_case(
        batch_size=2,
        next_n=4,
        max_num_pages=64,
        num_pages=128,
        num_heads=32,
        page_size=64,
        logits_dtype="float32",
        seed=13,
        context_pattern="sglang_fixed",
    ),
    _make_case(
        batch_size=1,
        next_n=5,
        max_num_pages=16,
        num_pages=128,
        num_heads=64,
        page_size=64,
        logits_dtype="float32",
        seed=14,
        context_pattern="sglang_fixed",
    ),
    _make_case(
        batch_size=2,
        next_n=6,
        max_num_pages=64,
        num_pages=128,
        num_heads=32,
        page_size=64,
        logits_dtype="float32",
        seed=15,
        context_pattern="sglang_ragged",
    ),
]

BENCH_CONFIGS = DSA_INDEXER_LIKE_COVERAGE + SGLANG_BENCH_CONFIGS


def load_deep_gemm_paged_mqa() -> tuple[Any, str]:
    try:
        import deep_gemm as module
    except Exception as exc:
        raise SkipTest(
            f"DeepGEMM paged MQA logits runtime unavailable: {_DEEP_GEMM_MODULE_NAME}: {exc}"
        ) from exc

    if not hasattr(module, "fp8_fp4_paged_mqa_logits"):
        raise SkipTest("DeepGEMM runtime unavailable: missing fp8_fp4_paged_mqa_logits")
    if not hasattr(module, "get_paged_mqa_logits_metadata"):
        raise SkipTest("DeepGEMM runtime unavailable: missing get_paged_mqa_logits_metadata")
    return module, "installed"


def _make_context_lens(config: PagedMQALogitsFP8Config) -> torch.Tensor:
    max_context_len = config.max_context_len
    if config.context_pattern == "sglang_fixed":
        lens = torch.full((config.batch_size, 1), max_context_len, dtype=torch.int32, device="cuda")
        if config.next_n > 1:
            lens = (
                lens
                - config.next_n
                + torch.arange(1, config.next_n + 1, dtype=torch.int32, device="cuda")[None, :]
            )
    elif config.context_pattern == "sglang_ragged":
        low = max(config.next_n, config.page_size, int(0.7 * max_context_len))
        last_token_lens = torch.randint(
            low=low,
            high=max_context_len + 1,
            size=(config.batch_size, 1),
            dtype=torch.int32,
            device="cuda",
        )
        if config.next_n == 1:
            lens = last_token_lens
        else:
            lens = (
                last_token_lens
                - config.next_n
                + torch.arange(1, config.next_n + 1, dtype=torch.int32, device="cuda")[None, :]
            )
    elif max_context_len == config.page_size:
        lens = torch.full(
            (config.batch_size, config.next_n), max_context_len, dtype=torch.int32, device="cuda"
        )
    else:
        last_token_lens = torch.randint(
            low=max(1, config.page_size // 2),
            high=max_context_len + 1,
            size=(config.batch_size, 1),
            dtype=torch.int32,
            device="cuda",
        )
        if config.next_n == 1:
            lens = last_token_lens
        else:
            lens = (
                (last_token_lens + 1) * torch.rand(config.batch_size, config.next_n, device="cuda")
            ).to(torch.int32)
            lens[:, -1] = last_token_lens[:, 0]
    lens = torch.maximum(lens, torch.ones_like(lens))
    return lens.contiguous()


def _make_block_table(config: PagedMQALogitsFP8Config) -> torch.Tensor:
    page_ids = torch.arange(config.num_pages, dtype=torch.int32, device="cuda")
    rows = []
    for batch_idx in range(config.batch_size):
        start = (batch_idx * config.max_num_pages) % config.num_pages
        rows.append(page_ids.roll(-start)[: config.max_num_pages])
    return torch.stack(rows, dim=0).contiguous()


def _make_indices(config: PagedMQALogitsFP8Config) -> torch.Tensor | None:
    if not config.varlen:
        return None
    indices = torch.arange(config.batch_size, dtype=torch.int32, device="cuda")
    if config.indices_pair_stride > 1:
        indices = indices // config.indices_pair_stride
    return indices.contiguous()


def _make_fused_kv_cache(
    config: PagedMQALogitsFP8Config, *, keep_dequant: bool
) -> tuple[torch.Tensor, torch.Tensor | None]:
    kv_bf16 = torch.randn(
        config.num_pages, config.page_size, config.head_dim, device="cuda", dtype=torch.bfloat16
    ).clamp_(-2.0, 2.0)
    scales = kv_bf16.abs().float().amax(dim=2, keepdim=True).clamp(1e-4) / 448.0
    kv_fp8 = (kv_bf16 * (1.0 / scales)).to(torch.float8_e4m3fn).contiguous()
    kv_dequant = (kv_fp8.float() * scales).to(torch.bfloat16) if keep_dequant else None
    scales = scales.squeeze(-1).contiguous()

    fused = torch.empty(
        (config.num_pages, config.page_size, 1, config.head_dim + 4),
        dtype=torch.uint8,
        device="cuda",
    )
    fused_flat = fused.view(config.num_pages, config.page_size * (config.head_dim + 4))
    fused_flat[:, : config.page_size * config.head_dim].copy_(
        kv_fp8.view(torch.uint8).reshape(config.num_pages, config.page_size * config.head_dim)
    )
    fused_flat[:, config.page_size * config.head_dim :].copy_(
        scales.view(torch.uint8).reshape(config.num_pages, config.page_size * 4)
    )
    return fused.contiguous(), kv_dequant


def _ref_paged_mqa_logits(
    q: torch.Tensor,
    kv_dequant: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    config: PagedMQALogitsFP8Config,
) -> torch.Tensor:
    q_f32 = q.float()
    kv_f32 = kv_dequant.float()
    weights_f32 = weights.view(config.batch_size, config.next_n, config.num_heads).float()
    output = torch.full(
        (config.batch_size * config.next_n, config.max_context_len),
        float("-inf"),
        device="cuda",
        dtype=torch.float32,
    )
    for batch_idx in range(config.batch_size):
        for next_idx in range(config.next_n):
            row = batch_idx * config.next_n + next_idx
            context_len = int(context_lens[batch_idx, next_idx].item())
            for page_col in range((context_len + config.page_size - 1) // config.page_size):
                page_id = int(block_table[batch_idx, page_col].item())
                token_start = page_col * config.page_size
                token_end = min(token_start + config.page_size, context_len)
                kv_tile = kv_f32[page_id, : token_end - token_start]
                score = torch.einsum("hd,td->ht", q_f32[batch_idx, next_idx], kv_tile)
                logits = (score.relu() * weights_f32[batch_idx, next_idx, :, None]).sum(dim=0)
                output[row, token_start:token_end] = logits
    return output


def _prepare_data(config: PagedMQALogitsFP8Config, *, compute_reference: bool) -> dict[str, Any]:
    deep_gemm, source = load_deep_gemm_paged_mqa()
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for SM100 FP8 paged MQA logits")
    if torch.cuda.get_device_capability()[0] < 10:
        raise SkipTest("SM100 FP8 paged MQA logits requires compute capability 10.x")

    torch.manual_seed(config.seed)
    runtime_config = PagedMQALogitsFP8Config(
        **{
            **asdict(config),
            "num_sms": int(getattr(deep_gemm, "get_num_sms", lambda: config.num_sms)()),
        }
    )
    q_bf16 = torch.randn(
        config.batch_size,
        config.next_n,
        config.num_heads,
        config.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    ).clamp_(-2.0, 2.0)
    q_fp8 = q_bf16.to(torch.float8_e4m3fn).contiguous()
    fused_kv_cache, kv_dequant = _make_fused_kv_cache(config, keep_dequant=compute_reference)
    weights = torch.randn(
        config.batch_size * config.next_n, config.num_heads, device="cuda", dtype=torch.float32
    ).contiguous()
    context_lens = _make_context_lens(config)
    block_table = _make_block_table(config)
    indices = _make_indices(config)
    schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(
        context_lens, config.page_size, runtime_config.num_sms, indices
    )
    data = {
        "config": runtime_config,
        "reference_source": source,
        "q": q_fp8,
        "fused_kv_cache": fused_kv_cache,
        "weights": weights,
        "context_lens": context_lens,
        "block_table": block_table,
        "indices": indices,
        "schedule_meta": schedule_meta,
        "deep_gemm": deep_gemm,
    }
    if compute_reference:
        assert kv_dequant is not None
        data["reference"] = _ref_paged_mqa_logits(
            q_fp8.to(torch.bfloat16), kv_dequant, weights, context_lens, block_table, config
        )
    return data


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    return _prepare_data(_make_config(**kwargs), compute_reference=True)


def get_kernel(**kwargs: Any):
    from tvm.script import tirx as T
    from tvm.tirx.layout import S, TCol, TileLayout, TLane

    config = _make_config(**kwargs)
    num_heads = config.num_heads
    head_dim = config.head_dim
    page_size = config.page_size
    k_pad_odd_n = (not config.varlen) and (config.next_n % 2 == 1) and (config.next_n >= 3)
    next_n_atom = 2 if (config.varlen or config.next_n >= 2) else 1
    num_next_n_atoms = _align_up(config.next_n, next_n_atom) // next_n_atom
    num_q_stages = 3
    num_kv_stages = 4
    split_kv = config.split_kv
    num_blocks_per_split = split_kv // page_size
    num_specialized_threads = 128
    num_specialized_registers = 56
    num_math_registers = 224
    umma_m = 128
    umma_k = 32
    umma_n = next_n_atom * num_heads
    num_math_warpgroups = split_kv // umma_m
    num_math_threads = num_math_warpgroups * 128
    num_threads = num_specialized_threads + num_math_threads
    num_warps = num_threads // 32
    spec_warp_start = num_math_warpgroups * 4
    smem_alignment = head_dim * 8
    desc_sdo = 8 * head_dim // 16
    desc_swizzle = {32: 1, 64: 2, 128: 3}[head_dim]
    smem_q_size_per_stage = next_n_atom * num_heads * head_dim
    smem_kv_size_per_stage = split_kv * head_dim
    smem_kv_scale_size_per_stage = split_kv * 4
    smem_weight_size_per_stage = next_n_atom * num_heads * 4
    smem_q_offset = 0
    smem_kv_offset = smem_q_offset + smem_q_size_per_stage * num_q_stages
    smem_kv_scales_offset = smem_kv_offset + smem_kv_size_per_stage * num_kv_stages
    smem_weights_offset = smem_kv_scales_offset + smem_kv_scale_size_per_stage * num_kv_stages
    smem_barrier_offset = smem_weights_offset + smem_weight_size_per_stage * num_q_stages
    num_total_barriers = num_q_stages * 2 + num_kv_stages * 2 + num_math_warpgroups * 2
    full_q_barrier_base = 0
    empty_q_barrier_base = full_q_barrier_base + num_q_stages
    full_kv_barrier_base = empty_q_barrier_base + num_q_stages
    empty_kv_barrier_base = full_kv_barrier_base + num_kv_stages
    full_umma_barrier_base = empty_kv_barrier_base + num_kv_stages
    empty_umma_barrier_base = full_umma_barrier_base + num_math_warpgroups
    smem_tmem_ptr_offset = smem_barrier_offset + num_total_barriers * 8
    smem_total_bytes = smem_tmem_ptr_offset + 4
    if smem_total_bytes > _SM100_SMEM_CAPACITY:
        raise ValueError(f"dynamic shared memory {smem_total_bytes} exceeds SM100 capacity")
    num_tmem_cols = next_n_atom * num_heads * num_math_warpgroups
    if num_tmem_cols > 512:
        raise ValueError("tensor memory columns exceed SM100 single-CTA limit")
    tmem_layout = TileLayout(S[(128, num_tmem_cols) : (1 @ TLane, 1 @ TCol)])
    logits_tir_dtype = "float32" if config.logits_dtype == "float32" else "bfloat16"
    cache_hint_sm90_evict_normal = "evict_normal"
    cache_hint_sm100_evict_normal = "evict_normal"
    cache_policy_evict_normal = T.uint64(1152921504606846976)
    has_cache_policy_evict_normal = 1
    tma_unicast_cta_mask = 0
    tma_no_cta_group_modifier = -1
    q_tma_block_inner = head_dim
    q_tma_swizzle_mode = head_dim
    q_tma_dtype_size = 1
    q_tma_block_inner_atom = (
        q_tma_block_inner if q_tma_swizzle_mode == 0 else q_tma_swizzle_mode // q_tma_dtype_size
    )
    q_tma_num_inner_atoms = q_tma_block_inner // q_tma_block_inner_atom
    weights_tma_block_inner = next_n_atom * num_heads
    weights_tma_swizzle_mode = 0
    weights_tma_dtype_size = 4
    weights_tma_block_inner_atom = (
        weights_tma_block_inner
        if weights_tma_swizzle_mode == 0
        else weights_tma_swizzle_mode // weights_tma_dtype_size
    )
    weights_tma_num_inner_atoms = weights_tma_block_inner // weights_tma_block_inner_atom
    kv_tma_block_inner = head_dim
    kv_tma_swizzle_mode = 0
    kv_tma_dtype_size = 1
    kv_tma_block_inner_atom = (
        kv_tma_block_inner if kv_tma_swizzle_mode == 0 else kv_tma_swizzle_mode // kv_tma_dtype_size
    )
    kv_tma_num_inner_atoms = kv_tma_block_inner // kv_tma_block_inner_atom
    kv_scales_tma_block_inner = page_size
    kv_scales_tma_swizzle_mode = 0
    kv_scales_tma_dtype_size = 4
    kv_scales_tma_block_inner_atom = (
        kv_scales_tma_block_inner
        if kv_scales_tma_swizzle_mode == 0
        else kv_scales_tma_swizzle_mode // kv_scales_tma_dtype_size
    )
    kv_scales_tma_num_inner_atoms = kv_scales_tma_block_inner // kv_scales_tma_block_inner_atom

    def atom_to_token_idx_expr(q_atom_idx):
        if config.varlen:
            return q_atom_idx
        if k_pad_odd_n:
            return q_atom_idx // T.uint32(num_next_n_atoms) * T.uint32(
                config.next_n
            ) + q_atom_idx % T.uint32(num_next_n_atoms) * T.uint32(next_n_atom)
        return q_atom_idx * T.uint32(next_n_atom)

    def atom_to_block_table_row_expr(q_atom_idx):
        if config.varlen:
            return q_atom_idx
        return q_atom_idx // T.uint32(num_next_n_atoms)

    def get_num_kv_expr(q_atom_idx, runtime_batch_size, context_lens_flat, indices):
        if config.varlen:
            is_paired = T.And(
                q_atom_idx + T.uint32(1) < runtime_batch_size,
                indices[T.cast(q_atom_idx, "int32")]
                == indices[T.cast(q_atom_idx + T.uint32(1), "int32")],
            )
            ctx_len: T.uint32 = T.if_then_else(
                is_paired,
                T.cast(context_lens_flat[T.cast(q_atom_idx + T.uint32(1), "int32")], "uint32"),
                T.cast(context_lens_flat[T.cast(q_atom_idx, "int32")], "uint32"),
            )
            return (ctx_len + T.uint32(page_size - 1)) // T.uint32(page_size)
        q_idx: T.uint32 = q_atom_idx // T.uint32(num_next_n_atoms)
        lens_idx: T.uint32 = q_idx * T.uint32(config.next_n) + T.uint32(config.next_n - 1)
        ctx_len: T.uint32 = T.cast(context_lens_flat[T.cast(lens_idx, "int32")], "uint32")
        return (ctx_len + T.uint32(page_size - 1)) // T.uint32(page_size)

    def get_atom_advance_expr(q_atom_idx, bound, indices):
        if config.varlen:
            return T.if_then_else(
                T.And(
                    q_atom_idx + T.uint32(1) < bound,
                    indices[T.cast(q_atom_idx, "int32")]
                    == indices[T.cast(q_atom_idx + T.uint32(1), "int32")],
                ),
                T.uint32(2),
                T.uint32(1),
            )
        return T.uint32(1)

    def should_refresh_num_kv_expr(q_atom_idx):
        if config.varlen:
            return T.bool(True)
        return q_atom_idx % T.uint32(num_next_n_atoms) == T.uint32(0)

    def exist_q_atom_idx_expr(q_atom_idx, end_q_atom_idx, end_kv_idx):
        return T.Or(
            q_atom_idx < end_q_atom_idx,
            T.And(q_atom_idx == end_q_atom_idx, T.uint32(0) < end_kv_idx),
        )

    def lane_id_u32():
        return T.cast(T.ptx.fetch_register(32, "laneid"), "uint32")

    def fmaxf_noftz(a, b):
        return T.ptx.max_f32(a, b)

    def ffma2_rn_noftz(a, b, c):
        out = T.alloc_local((1,), "uint64")
        T.evaluate(T.ptx.fma_f32x2(out.ptr_to([0]), a, b, c, rounding="rn", ftz=False))
        return out[0]

    def fadd2_rn_noftz(a, b):
        out = T.alloc_local((1,), "uint64")
        T.evaluate(T.ptx.add_f32x2(out.ptr_to([0]), a, b, rounding="rn", ftz=False))
        return out[0]

    def fadd_rn_noftz(a, b):
        out = T.alloc_local((1,), "float32")
        T.evaluate(T.ptx.add_f32(out.ptr_to([0]), a, b, rounding="rn", ftz=False))
        return out[0]

    def fmul_rn_noftz(a, b):
        out = T.alloc_local((1,), "float32")
        T.evaluate(T.ptx.mul_f32(out.ptr_to([0]), a, b, rounding="rn", ftz=False))
        return out[0]

    def cuda_grid_dependency_synchronize():
        T.evaluate(T.ptx.griddepcontrol.wait())

    def mbarrier_init_cta(barrier_ptr, arrive_count):
        T.evaluate(T.ptx.mbarrier.init(barrier_ptr, arrive_count))

    def mbarrier_wait_cta(barrier_ptr, phase):
        T.evaluate(T.ptx.mbarrier.try_wait(barrier_ptr, phase))

    def mbarrier_arrive_cta(barrier_ptr):
        T.evaluate(T.ptx.mbarrier.arrive(barrier_ptr))

    def mbarrier_arrive_expect_tx_cta(barrier_ptr, transaction_bytes):
        T.evaluate(T.ptx.mbarrier.arrive.expect_tx(barrier_ptr, transaction_bytes))

    @T.prim_func
    def sm100_fp8_paged_mqa_logits(
        batch_size: T.uint32,
        logits_stride: T.uint32,
        block_table_stride: T.uint32,
        context_lens: T.Buffer((config.batch_size, config.next_n), "int32"),
        logits: T.Buffer(
            (config.batch_size * config.next_n, config.logits_stride), logits_tir_dtype
        ),
        block_table: T.Buffer((config.batch_size, config.max_num_pages), "int32"),
        indices: T.Buffer((config.batch_size,), "int32"),
        schedule_meta: T.Buffer((config.num_sms + 1, 2), "int32"),
        tensor_map_q: T.TensorMap(),
        tensor_map_kv: T.TensorMap(),
        tensor_map_kv_scales: T.TensorMap(),
        tensor_map_weights: T.TensorMap(),
    ):
        T.device_entry()
        # TIRX_TRANSCRIBE_START sm100_fp8_paged_mqa_logits
        T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})
        logits_flat = T.decl_buffer(
            (config.batch_size * config.next_n * config.logits_stride,),
            logits_tir_dtype,
            data=logits.data,
            scope="global",
        )
        context_lens_flat = T.decl_buffer(
            (config.batch_size * config.next_n,), "int32", data=context_lens.data, scope="global"
        )
        block_table_flat = T.decl_buffer(
            (config.batch_size * config.max_num_pages,),
            "int32",
            data=block_table.data,
            scope="global",
        )
        schedule_meta_flat = T.decl_buffer(
            ((config.num_sms + 1) * 2,), "int32", data=schedule_meta.data, scope="global"
        )
        sm_idx = T.cta_id([config.num_sms])
        sm_idx_u32: T.let = T.cast(sm_idx, "uint32")
        warp_idx = T.warp_id([num_warps])
        warp_idx_u32: T.let = T.cast(warp_idx, "uint32")
        warpgroup_idx = T.warpgroup_id([num_warps // 4])
        lane_idx = T.lane_id([32])
        lane_idx_u32: T.let = lane_id_u32()

        if warp_idx == spec_warp_start:
            T.evaluate(T.ptx.prefetch_tensormap(T.address_of(tensor_map_q)))
            T.evaluate(T.ptx.prefetch_tensormap(T.address_of(tensor_map_kv)))
            T.evaluate(T.ptx.prefetch_tensormap(T.address_of(tensor_map_kv_scales)))
            T.evaluate(T.ptx.prefetch_tensormap(T.address_of(tensor_map_weights)))

        T.static_assert(smem_q_size_per_stage % smem_alignment == 0, "Unaligned TMA swizzling")
        T.static_assert(smem_kv_size_per_stage % smem_alignment == 0, "Unaligned TMA swizzling")

        smem = T.alloc_buffer([smem_total_bytes], "uint8", scope="shared.dyn", align=smem_alignment)
        smem_q_data: T.let[
            T.Var(name="smem_q_data", dtype=PointerType(PrimType("float8_e4m3fn")))
        ] = T.reinterpret("handle", smem.ptr_to([smem_q_offset]))
        smem_kv_data: T.let[
            T.Var(name="smem_kv_data", dtype=PointerType(PrimType("float8_e4m3fn")))
        ] = T.reinterpret("handle", smem.ptr_to([smem_kv_offset]))
        smem_kv_scales_data: T.let[
            T.Var(name="smem_kv_scales_data", dtype=PointerType(PrimType("float32")))
        ] = T.reinterpret("handle", smem.ptr_to([smem_kv_scales_offset]))
        smem_weights_data: T.let[
            T.Var(name="smem_weights_data", dtype=PointerType(PrimType("float32")))
        ] = T.reinterpret("handle", smem.ptr_to([smem_weights_offset]))
        smem_barrier_data: T.let[
            T.Var(name="smem_barrier_data", dtype=PointerType(PrimType("uint64")))
        ] = T.reinterpret("handle", smem.ptr_to([smem_barrier_offset]))
        smem_tmem_ptr_data: T.let[
            T.Var(name="smem_tmem_ptr_data", dtype=PointerType(PrimType("uint32")))
        ] = T.reinterpret("handle", smem.ptr_to([smem_tmem_ptr_offset]))
        smem_q = T.decl_buffer(
            (num_q_stages, next_n_atom * num_heads, head_dim),
            "float8_e4m3fn",
            data=smem_q_data,
            scope="shared.dyn",
            elem_offset=0,
            align=smem_alignment,
        )
        smem_kv = T.decl_buffer(
            (num_kv_stages, split_kv, head_dim),
            "float8_e4m3fn",
            data=smem_kv_data,
            scope="shared.dyn",
            elem_offset=0,
            align=smem_alignment,
        )
        smem_kv_scales = T.decl_buffer(
            (num_kv_stages, split_kv),
            "float32",
            data=smem_kv_scales_data,
            scope="shared.dyn",
            elem_offset=0,
            align=16,
        )
        smem_weights = T.decl_buffer(
            (num_q_stages, next_n_atom, num_heads),
            "float32",
            data=smem_weights_data,
            scope="shared.dyn",
            elem_offset=0,
            align=16,
        )
        smem_barriers = T.decl_buffer(
            (num_total_barriers,),
            "uint64",
            data=smem_barrier_data,
            scope="shared.dyn",
            elem_offset=0,
            align=8,
        )
        tmem_ptr_in_smem = T.decl_buffer(
            (1,), "uint32", data=smem_tmem_ptr_data, scope="shared.dyn", elem_offset=0, align=4
        )
        tmem = T.decl_buffer(
            (128, num_tmem_cols),
            "float32",
            scope="tmem",
            allocated_addr=tmem_ptr_in_smem[0],
            layout=tmem_layout,
        )
        fetch_result = T.alloc_local((4,), "uint32")
        scheduler_result = T.alloc_local((7,), "uint32")

        @T.inline
        def mbarrier_wait_phase(barrier_ptr, phase):
            mbarrier_wait_cta(barrier_ptr, phase)

        @T.inline
        def mbarrier_arrive(barrier_ptr):
            mbarrier_arrive_cta(barrier_ptr)

        @T.inline
        def mbarrier_arrive_and_expect_tx(barrier_ptr, num_bytes):
            mbarrier_arrive_expect_tx_cta(barrier_ptr, num_bytes)

        @T.inline
        def get_q_pipeline(q_iter_idx):
            fetch_result[0] = q_iter_idx % T.uint32(num_q_stages)
            fetch_result[1] = (q_iter_idx // T.uint32(num_q_stages)) & T.uint32(1)

        @T.inline
        def get_kv_pipeline(kv_iter_idx):
            fetch_result[2] = kv_iter_idx % T.uint32(num_kv_stages)
            fetch_result[3] = (kv_iter_idx // T.uint32(num_kv_stages)) & T.uint32(1)

        @T.inline
        def tma_load_2d_q(dst, barrier_ptr, tensor_map, coord0, coord1):
            T.static_assert(
                cache_hint_sm90_evict_normal == cache_hint_sm100_evict_normal, "Invalid cache hint"
            )
            T.static_assert(q_tma_num_inner_atoms == 1, "Unsupported split TMA atom")
            T.evaluate(
                T.call_intrin(
                    "",
                    "tirx.ptx_cp_async_bulk_tensor_global_to_cluster",
                    2,
                    dst,
                    barrier_ptr,
                    T.address_of(tensor_map),
                    tma_unicast_cta_mask,
                    tma_no_cta_group_modifier,
                    cache_policy_evict_normal,
                    has_cache_policy_evict_normal,
                    coord0,
                    coord1,
                )
            )

        @T.inline
        def tma_load_2d_weights(dst, barrier_ptr, tensor_map, coord0, coord1):
            T.static_assert(
                cache_hint_sm90_evict_normal == cache_hint_sm100_evict_normal, "Invalid cache hint"
            )
            T.static_assert(weights_tma_num_inner_atoms == 1, "Unsupported split TMA atom")
            T.evaluate(
                T.call_intrin(
                    "",
                    "tirx.ptx_cp_async_bulk_tensor_global_to_cluster",
                    2,
                    dst,
                    barrier_ptr,
                    T.address_of(tensor_map),
                    tma_unicast_cta_mask,
                    tma_no_cta_group_modifier,
                    cache_policy_evict_normal,
                    has_cache_policy_evict_normal,
                    coord0,
                    coord1,
                )
            )

        @T.inline
        def tma_load_3d_kv(dst, barrier_ptr, tensor_map, coord0, coord1, coord2):
            T.static_assert(
                cache_hint_sm90_evict_normal == cache_hint_sm100_evict_normal, "Invalid cache hint"
            )
            T.static_assert(kv_tma_num_inner_atoms == 1, "Unsupported split TMA atom")
            T.evaluate(
                T.call_intrin(
                    "",
                    "tirx.ptx_cp_async_bulk_tensor_global_to_cluster",
                    3,
                    dst,
                    barrier_ptr,
                    T.address_of(tensor_map),
                    tma_unicast_cta_mask,
                    tma_no_cta_group_modifier,
                    cache_policy_evict_normal,
                    has_cache_policy_evict_normal,
                    coord0,
                    coord1,
                    coord2,
                )
            )

        @T.inline
        def tma_load_2d_kv_scales(dst, barrier_ptr, tensor_map, coord0, coord1):
            T.static_assert(
                cache_hint_sm90_evict_normal == cache_hint_sm100_evict_normal, "Invalid cache hint"
            )
            T.static_assert(kv_scales_tma_num_inner_atoms == 1, "Unsupported split TMA atom")
            T.evaluate(
                T.call_intrin(
                    "",
                    "tirx.ptx_cp_async_bulk_tensor_global_to_cluster",
                    2,
                    dst,
                    barrier_ptr,
                    T.address_of(tensor_map),
                    tma_unicast_cta_mask,
                    tma_no_cta_group_modifier,
                    cache_policy_evict_normal,
                    has_cache_policy_evict_normal,
                    coord0,
                    coord1,
                )
            )

        @T.inline
        def make_smem_desc(desc, smem_ptr):
            T.ptx.tcgen05.encode_matrix_descriptor(
                T.address_of(desc), smem_ptr, ldo=0, sdo=desc_sdo, swizzle=desc_swizzle
            )

        @T.inline
        def issue_tma_q(stage_idx, tma_q_atom_idx):
            if T.ptx.elect_sync():
                q_token_idx: T.uint32 = atom_to_token_idx_expr(tma_q_atom_idx)
                tma_load_2d_q(
                    smem_q.ptr_to([stage_idx, 0, 0]),
                    smem_barriers.ptr_to([full_q_barrier_base + stage_idx]),
                    tensor_map_q,
                    T.uint32(0),
                    q_token_idx * T.uint32(num_heads),
                )
                tma_load_2d_weights(
                    smem_weights.ptr_to([stage_idx, 0, 0]),
                    smem_barriers.ptr_to([full_q_barrier_base + stage_idx]),
                    tensor_map_weights,
                    T.uint32(0),
                    q_token_idx,
                )
                mbarrier_arrive_and_expect_tx(
                    smem_barriers.ptr_to([full_q_barrier_base + stage_idx]),
                    smem_q_size_per_stage + smem_weight_size_per_stage,
                )

        @T.inline
        def fetch_next_task(
            current_q_atom_idx_arg,
            current_kv_idx_arg,
            current_num_kv_arg,
            end_q_atom_idx_arg,
            end_kv_idx_arg,
        ):
            scheduler_result[0] = current_q_atom_idx_arg
            scheduler_result[1] = current_kv_idx_arg
            scheduler_result[2] = current_num_kv_arg
            scheduler_result[4] = current_q_atom_idx_arg
            scheduler_result[5] = current_kv_idx_arg
            scheduler_result[6] = current_num_kv_arg
            if T.And(
                current_q_atom_idx_arg == end_q_atom_idx_arg, current_kv_idx_arg == end_kv_idx_arg
            ):
                scheduler_result[3] = T.uint32(0)
            else:
                scheduler_result[5] = current_kv_idx_arg + T.uint32(num_blocks_per_split)
                if scheduler_result[5] >= current_num_kv_arg:
                    scheduler_result[5] = T.uint32(0)
                    scheduler_result[4] = current_q_atom_idx_arg + get_atom_advance_expr(
                        current_q_atom_idx_arg, end_q_atom_idx_arg, indices
                    )
                    if T.And(
                        should_refresh_num_kv_expr(scheduler_result[4]),
                        exist_q_atom_idx_expr(
                            scheduler_result[4], end_q_atom_idx_arg, end_kv_idx_arg
                        ),
                    ):
                        scheduler_result[6] = get_num_kv_expr(
                            scheduler_result[4], batch_size, context_lens_flat, indices
                        )
                scheduler_result[3] = T.uint32(1)

        if warp_idx == spec_warp_start:
            if T.ptx.elect_sync():
                for init_i in T.unroll(0, num_q_stages):
                    mbarrier_init_cta(
                        smem_barriers.ptr_to([full_q_barrier_base + init_i]), T.uint32(1)
                    )
                    mbarrier_init_cta(
                        smem_barriers.ptr_to([empty_q_barrier_base + init_i]),
                        T.uint32(num_math_threads + 32),
                    )
                for init_i in T.unroll(0, num_kv_stages):
                    mbarrier_init_cta(
                        smem_barriers.ptr_to([full_kv_barrier_base + init_i]), T.uint32(1)
                    )
                    mbarrier_init_cta(
                        smem_barriers.ptr_to([empty_kv_barrier_base + init_i]),
                        T.uint32(num_math_threads),
                    )
                T.ptx.fence.mbarrier_init()
        if warp_idx == spec_warp_start + 1:
            if T.ptx.elect_sync():
                for init_i in T.unroll(0, num_math_warpgroups):
                    mbarrier_init_cta(
                        smem_barriers.ptr_to([full_umma_barrier_base + init_i]), T.uint32(1)
                    )
                    mbarrier_init_cta(
                        smem_barriers.ptr_to([empty_umma_barrier_base + init_i]), T.uint32(128)
                    )
                T.ptx.fence.mbarrier_init()
            T.ptx.tcgen05.alloc(
                T.address_of(tmem_ptr_in_smem[0]), n_cols=num_tmem_cols, cta_group=1
            )
        T.cuda.cta_sync()

        cuda_grid_dependency_synchronize()

        if warp_idx == spec_warp_start:
            T.ptx.setmaxnreg(False, num_specialized_registers)
            current_q_atom_idx: T.uint32 = T.cast(
                schedule_meta_flat[T.cast(sm_idx_u32 * T.uint32(2), "int32")], "uint32"
            )
            current_kv_idx: T.uint32 = T.cast(
                schedule_meta_flat[T.cast(sm_idx_u32 * T.uint32(2) + T.uint32(1), "int32")],
                "uint32",
            ) * T.uint32(num_blocks_per_split)
            end_q_atom_idx: T.uint32 = T.cast(
                schedule_meta_flat[T.cast((sm_idx_u32 + T.uint32(1)) * T.uint32(2), "int32")],
                "uint32",
            )
            end_kv_idx: T.uint32 = T.cast(
                schedule_meta_flat[
                    T.cast((sm_idx_u32 + T.uint32(1)) * T.uint32(2) + T.uint32(1), "int32")
                ],
                "uint32",
            ) * T.uint32(num_blocks_per_split)
            current_num_kv: T.uint32 = get_num_kv_expr(
                current_q_atom_idx, batch_size, context_lens_flat, indices
            )
            q_iter_idx: T.uint32 = T.uint32(0)
            kv_iter_idx: T.uint32 = T.uint32(0)
            q_stage_idx: T.uint32 = T.uint32(0)
            q_phase: T.uint32 = T.uint32(0)
            q_atom_idx: T.uint32 = batch_size * T.uint32(num_next_n_atoms)
            kv_idx: T.uint32 = T.uint32(0)
            num_kv: T.uint32 = T.uint32(0)
            next_q_atom_idx: T.uint32 = current_q_atom_idx
            next_kv_idx: T.uint32 = current_kv_idx
            next_num_kv: T.uint32 = current_num_kv
            fetch_next_task(
                current_q_atom_idx, current_kv_idx, current_num_kv, end_q_atom_idx, end_kv_idx
            )
            next_q_atom_idx = scheduler_result[0]
            next_kv_idx = scheduler_result[1]
            next_num_kv = scheduler_result[2]
            fetched_next_task: T.bool = scheduler_result[3] != T.uint32(0)
            current_q_atom_idx = scheduler_result[4]
            current_kv_idx = scheduler_result[5]
            current_num_kv = scheduler_result[6]
            if fetched_next_task:
                issue_tma_q(T.uint32(0), next_q_atom_idx)
                q_iter_idx = T.uint32(1)

            kv_block_idx_ptr: T.uint32 = T.uint32(32)
            kv_block_idx_storage: T.uint32 = T.uint32(0)

            while fetched_next_task:
                next_advance: T.uint32 = get_atom_advance_expr(next_q_atom_idx, batch_size, indices)
                prefetch_q: T.bool = T.And(
                    q_atom_idx != next_q_atom_idx,
                    exist_q_atom_idx_expr(
                        next_q_atom_idx + next_advance, end_q_atom_idx, end_kv_idx
                    ),
                )
                if q_atom_idx != next_q_atom_idx:
                    kv_block_idx_ptr = T.uint32(32)
                q_atom_idx = next_q_atom_idx
                kv_idx = next_kv_idx
                num_kv = next_num_kv

                if kv_block_idx_ptr == T.uint32(32):
                    kv_block_idx_ptr = T.uint32(0)
                    block_table_offset: T.uint64 = T.cast(
                        atom_to_block_table_row_expr(q_atom_idx), "uint64"
                    ) * T.cast(block_table_stride, "uint64")
                    block_table_index: T.uint64 = block_table_offset + T.cast(
                        kv_idx + lane_idx_u32, "uint64"
                    )
                    kv_block_idx_storage = T.if_then_else(
                        kv_idx + lane_idx_u32 < num_kv,
                        T.cast(block_table_flat[T.cast(block_table_index, "int64")], "uint32"),
                        T.uint32(0),
                    )
                T.cuda.warp_sync()
                T.static_assert(32 % num_blocks_per_split == 0, "Invalid `UMMA_M`")

                if prefetch_q:
                    get_q_pipeline(q_iter_idx)
                    q_stage_idx = fetch_result[0]
                    q_phase = fetch_result[1]
                    q_iter_idx = q_iter_idx + T.uint32(1)
                    mbarrier_wait_phase(
                        smem_barriers.ptr_to([empty_q_barrier_base + q_stage_idx]),
                        q_phase ^ T.uint32(1),
                    )
                    issue_tma_q(q_stage_idx, q_atom_idx + next_advance)

                kv_block_idx = T.alloc_local((num_blocks_per_split,), "uint32")
                for block_i in T.unroll(0, num_blocks_per_split):
                    kv_block_idx[block_i] = T.cuda.__shfl_sync(
                        T.uint32(0xFFFFFFFF),
                        kv_block_idx_storage,
                        kv_block_idx_ptr + T.uint32(block_i),
                        32,
                    )
                kv_block_idx_ptr = kv_block_idx_ptr + T.uint32(num_blocks_per_split)

                get_kv_pipeline(kv_iter_idx)
                kv_stage_idx: T.uint32 = fetch_result[2]
                kv_phase: T.uint32 = fetch_result[3]
                kv_iter_idx = kv_iter_idx + T.uint32(1)
                mbarrier_wait_phase(
                    smem_barriers.ptr_to([empty_kv_barrier_base + kv_stage_idx]),
                    kv_phase ^ T.uint32(1),
                )

                if T.ptx.elect_sync():
                    for block_i in T.unroll(0, num_blocks_per_split):
                        tma_load_3d_kv(
                            smem_kv.ptr_to([kv_stage_idx, block_i * page_size, 0]),
                            smem_barriers.ptr_to([full_kv_barrier_base + kv_stage_idx]),
                            tensor_map_kv,
                            T.uint32(0),
                            T.uint32(0),
                            kv_block_idx[block_i],
                        )
                        tma_load_2d_kv_scales(
                            smem_kv_scales.ptr_to([kv_stage_idx, block_i * page_size]),
                            smem_barriers.ptr_to([full_kv_barrier_base + kv_stage_idx]),
                            tensor_map_kv_scales,
                            T.uint32(0),
                            kv_block_idx[block_i],
                        )
                    mbarrier_arrive_and_expect_tx(
                        smem_barriers.ptr_to([full_kv_barrier_base + kv_stage_idx]),
                        smem_kv_size_per_stage + smem_kv_scale_size_per_stage,
                    )

                next_q_atom_idx = current_q_atom_idx
                next_kv_idx = current_kv_idx
                next_num_kv = current_num_kv
                fetch_next_task(
                    current_q_atom_idx, current_kv_idx, current_num_kv, end_q_atom_idx, end_kv_idx
                )
                next_q_atom_idx = scheduler_result[0]
                next_kv_idx = scheduler_result[1]
                next_num_kv = scheduler_result[2]
                fetched_next_task = scheduler_result[3] != T.uint32(0)
                current_q_atom_idx = scheduler_result[4]
                current_kv_idx = scheduler_result[5]
                current_num_kv = scheduler_result[6]
        elif warp_idx == spec_warp_start + 1:
            T.ptx.setmaxnreg(False, num_specialized_registers)
            current_q_atom_idx: T.uint32 = T.cast(
                schedule_meta_flat[T.cast(sm_idx_u32 * T.uint32(2), "int32")], "uint32"
            )
            current_kv_idx: T.uint32 = T.cast(
                schedule_meta_flat[T.cast(sm_idx_u32 * T.uint32(2) + T.uint32(1), "int32")],
                "uint32",
            ) * T.uint32(num_blocks_per_split)
            end_q_atom_idx: T.uint32 = T.cast(
                schedule_meta_flat[T.cast((sm_idx_u32 + T.uint32(1)) * T.uint32(2), "int32")],
                "uint32",
            )
            end_kv_idx: T.uint32 = T.cast(
                schedule_meta_flat[
                    T.cast((sm_idx_u32 + T.uint32(1)) * T.uint32(2) + T.uint32(1), "int32")
                ],
                "uint32",
            ) * T.uint32(num_blocks_per_split)
            current_num_kv: T.uint32 = get_num_kv_expr(
                current_q_atom_idx, batch_size, context_lens_flat, indices
            )
            q_iter_idx: T.uint32 = T.uint32(0)
            kv_iter_idx: T.uint32 = T.uint32(0)
            q_stage_idx: T.uint32 = T.uint32(0)
            q_phase: T.uint32 = T.uint32(0)
            tmem_allocated: T.uint32 = T.ptx.ld(
                tmem_ptr_in_smem.ptr_to([0]), "uint32", "u32", space="shared"
            )
            T.cuda.trap_when_assert_failed(tmem_allocated == T.uint32(0))
            desc_i: T.uint32
            desc_a: T.uint64
            desc_b: T.uint64
            T.ptx.tcgen05.encode_instr_descriptor(
                T.address_of(desc_i),
                d_dtype="float32",
                a_dtype="float8_e4m3fn",
                b_dtype="float8_e4m3fn",
                M=umma_m,
                N=umma_n,
                K=umma_k,
                trans_a=False,
                trans_b=False,
                n_cta_groups=1,
            )
            runtime_instr_desc: T.uint64 = T.shift_left(T.cast(desc_i, "uint64"), T.uint64(32))
            runtime_instr_desc_hi: T.uint32 = T.cast(
                T.shift_right(runtime_instr_desc, T.uint64(32)), "uint32"
            )
            q_atom_idx: T.uint32 = batch_size * T.uint32(num_next_n_atoms)
            kv_idx: T.uint32 = T.uint32(0)
            next_q_atom_idx: T.uint32 = current_q_atom_idx
            next_kv_idx: T.uint32 = current_kv_idx
            next_num_kv: T.uint32 = current_num_kv
            fetch_next_task(
                current_q_atom_idx, current_kv_idx, current_num_kv, end_q_atom_idx, end_kv_idx
            )
            next_q_atom_idx = scheduler_result[0]
            next_kv_idx = scheduler_result[1]
            next_num_kv = scheduler_result[2]
            fetched_next_task: T.bool = scheduler_result[3] != T.uint32(0)
            current_q_atom_idx = scheduler_result[4]
            current_kv_idx = scheduler_result[5]
            current_num_kv = scheduler_result[6]
            umma_phase: T.uint32 = T.uint32(1)
            while fetched_next_task:
                if q_atom_idx != next_q_atom_idx:
                    if q_iter_idx > T.uint32(0):
                        mbarrier_arrive(
                            smem_barriers.ptr_to(
                                [
                                    empty_q_barrier_base
                                    + (q_iter_idx - T.uint32(1)) % T.uint32(num_q_stages)
                                ]
                            )
                        )
                    get_q_pipeline(q_iter_idx)
                    q_stage_idx = fetch_result[0]
                    q_phase = fetch_result[1]
                    q_iter_idx = q_iter_idx + T.uint32(1)
                    mbarrier_wait_phase(
                        smem_barriers.ptr_to([full_q_barrier_base + q_stage_idx]), q_phase
                    )
                q_atom_idx = next_q_atom_idx
                kv_idx = next_kv_idx

                get_kv_pipeline(kv_iter_idx)
                kv_stage_idx: T.uint32 = fetch_result[2]
                kv_phase: T.uint32 = fetch_result[3]
                kv_iter_idx = kv_iter_idx + T.uint32(1)
                mbarrier_wait_phase(
                    smem_barriers.ptr_to([full_kv_barrier_base + kv_stage_idx]), kv_phase
                )
                for math_wg_i in T.unroll(0, num_math_warpgroups):
                    mbarrier_wait_phase(
                        smem_barriers.ptr_to([empty_umma_barrier_base + math_wg_i]), umma_phase
                    )
                    T.ptx.tcgen05.fence.after_thread_sync()
                    T.static_assert(head_dim % umma_k == 0, "Invalid head dim")
                    for k in T.unroll(0, head_dim // umma_k):
                        make_smem_desc(
                            desc_a, smem_kv.ptr_to([kv_stage_idx, math_wg_i * umma_m, k * umma_k])
                        )
                        make_smem_desc(desc_b, smem_q.ptr_to([q_stage_idx, 0, k * umma_k]))
                        if T.ptx.elect_sync():
                            T.ptx.tcgen05.mma(
                                T.uint32(math_wg_i * umma_n),
                                desc_a,
                                desc_b,
                                runtime_instr_desc_hi,
                                T.uint32(0),
                                T.uint32(0),
                                T.uint32(0),
                                T.uint32(0),
                                d_dtype="float32",
                                a_dtype="float8_e4m3fn",
                                b_dtype="float8_e4m3fn",
                                use_a_tmem=False,
                                cta_group=1,
                                enable_input_d=T.uint32(k),
                            )
                    if T.ptx.elect_sync():
                        T.ptx.tcgen05.commit(
                            smem_barriers.ptr_to([full_umma_barrier_base + math_wg_i])
                        )
                umma_phase = umma_phase ^ T.uint32(1)
                next_q_atom_idx = current_q_atom_idx
                next_kv_idx = current_kv_idx
                next_num_kv = current_num_kv
                fetch_next_task(
                    current_q_atom_idx, current_kv_idx, current_num_kv, end_q_atom_idx, end_kv_idx
                )
                next_q_atom_idx = scheduler_result[0]
                next_kv_idx = scheduler_result[1]
                next_num_kv = scheduler_result[2]
                fetched_next_task = scheduler_result[3] != T.uint32(0)
                current_q_atom_idx = scheduler_result[4]
                current_kv_idx = scheduler_result[5]
                current_num_kv = scheduler_result[6]
        elif T.Or(warp_idx == spec_warp_start + 2, warp_idx == spec_warp_start + 3):
            T.ptx.setmaxnreg(False, num_specialized_registers)
        elif warp_idx < spec_warp_start:
            T.ptx.setmaxnreg(True, num_math_registers)
            current_q_atom_idx: T.uint32 = T.cast(
                schedule_meta_flat[T.cast(sm_idx_u32 * T.uint32(2), "int32")], "uint32"
            )
            current_kv_idx: T.uint32 = T.cast(
                schedule_meta_flat[T.cast(sm_idx_u32 * T.uint32(2) + T.uint32(1), "int32")],
                "uint32",
            ) * T.uint32(num_blocks_per_split)
            end_q_atom_idx: T.uint32 = T.cast(
                schedule_meta_flat[T.cast((sm_idx_u32 + T.uint32(1)) * T.uint32(2), "int32")],
                "uint32",
            )
            end_kv_idx: T.uint32 = T.cast(
                schedule_meta_flat[
                    T.cast((sm_idx_u32 + T.uint32(1)) * T.uint32(2) + T.uint32(1), "int32")
                ],
                "uint32",
            ) * T.uint32(num_blocks_per_split)
            current_num_kv: T.uint32 = get_num_kv_expr(
                current_q_atom_idx, batch_size, context_lens_flat, indices
            )
            q_iter_idx: T.uint32 = T.uint32(0)
            kv_iter_idx: T.uint32 = T.uint32(0)
            q_stage_idx: T.uint32 = T.uint32(0)
            q_phase: T.uint32 = T.uint32(0)
            math_warpgroup_idx: T.int32 = warpgroup_idx
            tmem_start: T.uint32 = T.cast(math_warpgroup_idx, "uint32") * T.uint32(umma_n)
            math_thread_idx: T.uint32 = warp_idx_u32 * T.uint32(32) + lane_idx_u32
            cached_weights = T.alloc_local((next_n_atom, num_heads), "float32")
            q_atom_idx: T.uint32 = batch_size * T.uint32(num_next_n_atoms)
            next_q_atom_idx: T.uint32 = current_q_atom_idx
            next_kv_idx: T.uint32 = current_kv_idx
            next_num_kv: T.uint32 = current_num_kv
            fetch_next_task(
                current_q_atom_idx, current_kv_idx, current_num_kv, end_q_atom_idx, end_kv_idx
            )
            next_q_atom_idx = scheduler_result[0]
            next_kv_idx = scheduler_result[1]
            next_num_kv = scheduler_result[2]
            fetched_next_task: T.bool = scheduler_result[3] != T.uint32(0)
            current_q_atom_idx = scheduler_result[4]
            current_kv_idx = scheduler_result[5]
            current_num_kv = scheduler_result[6]
            umma_phase: T.uint32 = T.uint32(0)
            is_paired_atom: T.bool = T.bool(False)
            T.static_assert(num_heads % 8 == 0, "Invalid head")

            @T.inline
            def reduce_and_store(num_iters_c, kv_offset_arg, scale_kv_arg):
                accum = T.alloc_local((num_heads,), "float32")
                T.static_assert(num_heads == 32 or num_heads == 64, "Unsupported TMEM load size")
                for q_inner_i in T.unroll(0, num_iters_c):
                    tmem_addr: T.uint32 = tmem_start + T.uint32(q_inner_i * num_heads)
                    if num_heads == 32:
                        T.ptx.tcgen05.ld(
                            tmem_addr,
                            accum[0],
                            accum[1],
                            accum[2],
                            accum[3],
                            accum[4],
                            accum[5],
                            accum[6],
                            accum[7],
                            accum[8],
                            accum[9],
                            accum[10],
                            accum[11],
                            accum[12],
                            accum[13],
                            accum[14],
                            accum[15],
                            accum[16],
                            accum[17],
                            accum[18],
                            accum[19],
                            accum[20],
                            accum[21],
                            accum[22],
                            accum[23],
                            accum[24],
                            accum[25],
                            accum[26],
                            accum[27],
                            accum[28],
                            accum[29],
                            accum[30],
                            accum[31],
                            shape="32x32b",
                            num=32,
                        )
                    if num_heads == 64:
                        T.ptx.tcgen05.ld(
                            tmem_addr,
                            accum[0],
                            accum[1],
                            accum[2],
                            accum[3],
                            accum[4],
                            accum[5],
                            accum[6],
                            accum[7],
                            accum[8],
                            accum[9],
                            accum[10],
                            accum[11],
                            accum[12],
                            accum[13],
                            accum[14],
                            accum[15],
                            accum[16],
                            accum[17],
                            accum[18],
                            accum[19],
                            accum[20],
                            accum[21],
                            accum[22],
                            accum[23],
                            accum[24],
                            accum[25],
                            accum[26],
                            accum[27],
                            accum[28],
                            accum[29],
                            accum[30],
                            accum[31],
                            accum[32],
                            accum[33],
                            accum[34],
                            accum[35],
                            accum[36],
                            accum[37],
                            accum[38],
                            accum[39],
                            accum[40],
                            accum[41],
                            accum[42],
                            accum[43],
                            accum[44],
                            accum[45],
                            accum[46],
                            accum[47],
                            accum[48],
                            accum[49],
                            accum[50],
                            accum[51],
                            accum[52],
                            accum[53],
                            accum[54],
                            accum[55],
                            accum[56],
                            accum[57],
                            accum[58],
                            accum[59],
                            accum[60],
                            accum[61],
                            accum[62],
                            accum[63],
                            shape="32x32b",
                            num=64,
                        )
                    T.ptx.tcgen05.wait.ld()
                    sum_0: T.uint64 = T.cuda.make_float2(T.float32(0), T.float32(0))
                    sum_1: T.uint64 = T.cuda.make_float2(T.float32(0), T.float32(0))
                    for head_j_group in T.unroll(0, num_heads // 4):
                        head_j = head_j_group * 4
                        a0 = T.cuda.make_float2(
                            fmaxf_noftz(accum[head_j], T.float32(0)),
                            fmaxf_noftz(accum[head_j + 1], T.float32(0)),
                        )
                        b0 = T.cuda.make_float2(
                            cached_weights[q_inner_i, head_j], cached_weights[q_inner_i, head_j + 1]
                        )
                        sum_0 = ffma2_rn_noftz(a0, b0, sum_0)
                        a1 = T.cuda.make_float2(
                            fmaxf_noftz(accum[head_j + 2], T.float32(0)),
                            fmaxf_noftz(accum[head_j + 3], T.float32(0)),
                        )
                        b1 = T.cuda.make_float2(
                            cached_weights[q_inner_i, head_j + 2],
                            cached_weights[q_inner_i, head_j + 3],
                        )
                        sum_1 = ffma2_rn_noftz(a1, b1, sum_1)
                    sum_v: T.let = fadd2_rn_noftz(sum_0, sum_1)
                    result_f32: T.let = fmul_rn_noftz(
                        scale_kv_arg, fadd_rn_noftz(T.cuda.float2_x(sum_v), T.cuda.float2_y(sum_v))
                    )
                    result = T.cast(result_f32, logits_tir_dtype)
                    logits_flat[
                        T.cast(kv_offset_arg, "uint64")
                        + T.cast(q_inner_i, "uint64") * T.cast(logits_stride, "uint64")
                        + T.cast(math_thread_idx, "uint64")
                    ] = result
                    T.cuda.warp_sync()
                T.ptx.tcgen05.fence.before_thread_sync()
                mbarrier_arrive(
                    smem_barriers.ptr_to([empty_umma_barrier_base + math_warpgroup_idx])
                )

            while fetched_next_task:
                if q_atom_idx != next_q_atom_idx:
                    if q_iter_idx > T.uint32(0):
                        mbarrier_arrive(
                            smem_barriers.ptr_to(
                                [
                                    empty_q_barrier_base
                                    + (q_iter_idx - T.uint32(1)) % T.uint32(num_q_stages)
                                ]
                            )
                        )
                    get_q_pipeline(q_iter_idx)
                    q_stage_idx = fetch_result[0]
                    q_phase = fetch_result[1]
                    q_iter_idx = q_iter_idx + T.uint32(1)
                    mbarrier_wait_phase(
                        smem_barriers.ptr_to([full_q_barrier_base + q_stage_idx]), q_phase
                    )
                    for weight_i in T.unroll(0, next_n_atom):
                        for weight_j in T.unroll(0, num_heads):
                            cached_weights[weight_i, weight_j] = T.ptx.ld(
                                smem_weights.ptr_to([q_stage_idx, weight_i, weight_j]),
                                "float32",
                                "f32",
                                space="shared",
                            )
                    if config.varlen:
                        is_paired_atom = get_atom_advance_expr(
                            next_q_atom_idx, batch_size, indices
                        ) == T.uint32(2)
                q_atom_idx = next_q_atom_idx
                kv_idx: T.uint32 = next_kv_idx
                kv_offset: T.uint64 = T.cast(atom_to_token_idx_expr(q_atom_idx), "uint64") * T.cast(
                    logits_stride, "uint64"
                ) + T.cast(kv_idx * T.uint32(page_size), "uint64")
                get_kv_pipeline(kv_iter_idx)
                kv_stage_idx: T.uint32 = fetch_result[2]
                kv_phase: T.uint32 = fetch_result[3]
                kv_iter_idx = kv_iter_idx + T.uint32(1)
                mbarrier_wait_phase(
                    smem_barriers.ptr_to([full_kv_barrier_base + kv_stage_idx]), kv_phase
                )
                scale_kv: T.float32 = T.ptx.ld(
                    smem_kv_scales.ptr_to([kv_stage_idx, math_thread_idx]),
                    "float32",
                    "f32",
                    space="shared",
                )
                mbarrier_wait_phase(
                    smem_barriers.ptr_to([full_umma_barrier_base + math_warpgroup_idx]), umma_phase
                )
                T.ptx.tcgen05.fence.after_thread_sync()
                umma_phase = umma_phase ^ T.uint32(1)
                mbarrier_arrive(smem_barriers.ptr_to([empty_kv_barrier_base + kv_stage_idx]))
                if config.varlen:
                    if is_paired_atom:
                        reduce_and_store(next_n_atom, kv_offset, scale_kv)
                    else:
                        reduce_and_store(1, kv_offset, scale_kv)
                elif k_pad_odd_n:
                    if q_atom_idx % T.uint32(num_next_n_atoms) == T.uint32(num_next_n_atoms - 1):
                        reduce_and_store(1, kv_offset, scale_kv)
                    else:
                        reduce_and_store(next_n_atom, kv_offset, scale_kv)
                else:
                    reduce_and_store(next_n_atom, kv_offset, scale_kv)
                next_q_atom_idx = current_q_atom_idx
                next_kv_idx = current_kv_idx
                next_num_kv = current_num_kv
                fetch_next_task(
                    current_q_atom_idx, current_kv_idx, current_num_kv, end_q_atom_idx, end_kv_idx
                )
                next_q_atom_idx = scheduler_result[0]
                next_kv_idx = scheduler_result[1]
                next_num_kv = scheduler_result[2]
                fetched_next_task = scheduler_result[3] != T.uint32(0)
                current_q_atom_idx = scheduler_result[4]
                current_kv_idx = scheduler_result[5]
                current_num_kv = scheduler_result[6]
            T.ptx.bar.sync(8, num_math_threads)
            if warp_idx == 0:
                T.ptx.tcgen05.dealloc(T.uint32(0), n_cols=num_tmem_cols, cta_group=1)

    return sm100_fp8_paged_mqa_logits.with_attr(
        "tirx.kernel_launch_params",
        [
            "blockIdx.x",
            "threadIdx.x",
            "tirx.use_programtic_dependent_launch",
            "tirx.use_dyn_shared_memory",
        ],
    )


def _compile_tirx_paged_mqa_for_config(
    *,
    batch_size: int,
    next_n: int,
    max_num_pages: int,
    num_pages: int,
    num_heads: int,
    head_dim: int,
    page_size: int,
    logits_dtype: str,
    num_sms: int,
    context_lens_2d: bool,
    varlen: bool,
    indices_pair_stride: int,
) -> Any:
    import tvm

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100f"})
    kernel = get_kernel(
        batch_size=batch_size,
        next_n=next_n,
        max_num_pages=max_num_pages,
        num_pages=num_pages,
        num_heads=num_heads,
        head_dim=head_dim,
        page_size=page_size,
        logits_dtype=logits_dtype,
        num_sms=num_sms,
        context_lens_2d=context_lens_2d,
        varlen=varlen,
        indices_pair_stride=indices_pair_stride,
    )
    with target:
        mod = tvm.IRModule({"main": kernel})
        return tvm.compile(mod, target=target, tir_pipeline="tirx")


_compile_tirx_paged_mqa_for_config = cache(_compile_tirx_paged_mqa_for_config)


def _compile_tirx_paged_mqa(config: PagedMQALogitsFP8Config) -> Any:
    return _compile_tirx_paged_mqa_for_config(
        batch_size=config.batch_size,
        next_n=config.next_n,
        max_num_pages=config.max_num_pages,
        num_pages=config.num_pages,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        page_size=config.page_size,
        logits_dtype=config.logits_dtype,
        num_sms=config.num_sms,
        context_lens_2d=config.context_lens_2d,
        varlen=config.varlen,
        indices_pair_stride=config.indices_pair_stride,
    )


def _run_deepgemm_paged_mqa(data: dict[str, Any], *, clean_logits: bool = False) -> torch.Tensor:
    config: PagedMQALogitsFP8Config = data["config"]
    return data["deep_gemm"].fp8_fp4_paged_mqa_logits(
        q=(data["q"], None),
        kv_cache=data["fused_kv_cache"],
        weights=data["weights"],
        context_lens=data["context_lens"],
        block_table=data["block_table"],
        schedule_meta=data["schedule_meta"],
        max_context_len=config.max_context_len,
        clean_logits=clean_logits,
        logits_dtype=_torch_logits_dtype(config.logits_dtype),
        indices=data["indices"],
    )


def _sglang_cutedsl_available() -> bool:
    return find_spec("sglang") is not None and find_spec("cutlass") is not None


def _make_sglang_cutedsl_runner(data: dict[str, Any]) -> Any:
    config: PagedMQALogitsFP8Config = data["config"]
    if config.context_pattern == "random_2d" and config.next_n > 1:
        raise ValueError(
            "SGLang CuTeDSL requires causal context lengths when next_n > 1; "
            "use context_pattern='sglang_fixed' or 'sglang_ragged'"
        )

    from sglang.jit_kernel.dsa.cutedsl_paged_mqa_logits import (
        CuteDSLPagedMQALogitsRunner,
        pick_dsl_expand,
    )

    expand_factor, atom = pick_dsl_expand(
        config.next_n,
        batch_size=config.batch_size,
        max_ctx=config.max_context_len,
        num_sms=config.num_sms,
        num_heads=config.num_heads,
    )
    expanded_batch = config.batch_size * expand_factor
    q = data["q"].reshape(expanded_batch, atom, config.num_heads, config.head_dim)
    context_lens = data["context_lens"][:, -1].contiguous()
    block_table = data["block_table"]
    if expand_factor > 1:
        context_lens = context_lens.repeat_interleave(expand_factor)
        block_table = block_table.repeat_interleave(expand_factor, dim=0)
    block_table = block_table.contiguous()
    schedule_meta = data["deep_gemm"].get_paged_mqa_logits_metadata(
        context_lens.unsqueeze(-1), config.page_size, config.num_sms
    )
    output_dtype = _torch_logits_dtype(config.logits_dtype)

    def _run():
        return CuteDSLPagedMQALogitsRunner.forward(
            q,
            data["fused_kv_cache"],
            data["weights"],
            context_lens,
            block_table,
            schedule_meta,
            config.max_context_len,
            epi_dtype=torch.float32,
            acc_dtype=torch.float32,
            output_dtype=output_dtype,
        )

    return _run


def _allocate_logits(config: PagedMQALogitsFP8Config) -> torch.Tensor:
    return torch.full(
        (config.batch_size * config.next_n, config.logits_stride),
        float("-inf"),
        device="cuda",
        dtype=_torch_logits_dtype(config.logits_dtype),
    )


def _encode_tma_3d_desc(
    *,
    encode_tensormap: Any,
    tensor: torch.Tensor,
    gmem_inner_dim: int,
    gmem_mid_dim: int,
    gmem_outer_dim: int,
    smem_inner_dim: int,
    smem_mid_dim: int,
    smem_outer_dim: int,
    gmem_mid_stride: int,
    gmem_outer_stride: int,
    swizzle_mode: int,
    tensor_dtype: Any | None = None,
) -> Any:
    from tirx_kernels.deepgemm import mega_moe

    elem_size = int(tensor.element_size())
    if swizzle_mode != 0:
        smem_inner_dim = swizzle_mode // elem_size
    desc = mega_moe._AlignedTensorMap()
    encode_tensormap(
        desc.ptr,
        mega_moe._torch_dtype_to_tvm_dtype(tensor) if tensor_dtype is None else tensor_dtype,
        3,
        ctypes.c_void_p(int(tensor.data_ptr())),
        int(gmem_inner_dim),
        int(gmem_mid_dim),
        int(gmem_outer_dim),
        int(gmem_mid_stride * elem_size),
        int(gmem_outer_stride * elem_size),
        int(smem_inner_dim),
        int(smem_mid_dim),
        int(smem_outer_dim),
        1,
        1,
        1,
        0,
        mega_moe._tensor_map_swizzle_from_mode(swizzle_mode),
        3,
        0,
    )
    return desc


def _build_tirx_tensor_maps(data: dict[str, Any]) -> dict[str, Any]:
    import tvm
    from tirx_kernels.deepgemm.mega_moe import _encode_tma_2d_desc

    config: PagedMQALogitsFP8Config = data["config"]
    q_fp8 = data["q"]
    fused = data["fused_kv_cache"]
    weights = data["weights"]
    encode_tensormap = tvm.get_global_func("runtime.cuTensorMapEncodeTiled")
    kv_flat = fused.view(torch.uint8).view(
        config.num_pages, config.page_size * (config.head_dim + 4)
    )
    kv_fp8 = (
        kv_flat[:, : config.page_size * config.head_dim]
        .view(torch.float8_e4m3fn)
        .reshape(config.num_pages, config.page_size, config.head_dim)
    )
    kv_scales = kv_flat[:, config.page_size * config.head_dim :].view(torch.float32)

    return {
        "tensor_map_q": _encode_tma_2d_desc(
            encode_tensormap=encode_tensormap,
            tensor=q_fp8,
            gmem_inner_dim=config.head_dim,
            gmem_outer_dim=config.batch_size * config.next_n * config.num_heads,
            smem_inner_dim=config.head_dim,
            smem_outer_dim=(2 if (config.varlen or config.next_n >= 2) else 1) * config.num_heads,
            gmem_outer_stride=int(q_fp8.stride(2)),
            swizzle_mode=config.head_dim,
            tensor_dtype="float8_e4m3fn",
        ),
        "tensor_map_kv": _encode_tma_3d_desc(
            encode_tensormap=encode_tensormap,
            tensor=kv_fp8,
            gmem_inner_dim=config.head_dim,
            gmem_mid_dim=config.page_size,
            gmem_outer_dim=config.num_pages,
            smem_inner_dim=config.head_dim,
            smem_mid_dim=config.page_size,
            smem_outer_dim=1,
            gmem_mid_stride=int(kv_fp8.stride(1)),
            gmem_outer_stride=int(kv_fp8.stride(0)),
            swizzle_mode=config.head_dim,
            tensor_dtype="float8_e4m3fn",
        ),
        "tensor_map_kv_scales": _encode_tma_2d_desc(
            encode_tensormap=encode_tensormap,
            tensor=kv_scales,
            gmem_inner_dim=config.page_size,
            gmem_outer_dim=config.num_pages,
            smem_inner_dim=config.page_size,
            smem_outer_dim=1,
            gmem_outer_stride=int(kv_scales.stride(0)),
            swizzle_mode=0,
        ),
        "tensor_map_weights": _encode_tma_2d_desc(
            encode_tensormap=encode_tensormap,
            tensor=weights,
            gmem_inner_dim=config.num_heads,
            gmem_outer_dim=config.batch_size * config.next_n,
            smem_inner_dim=config.num_heads,
            smem_outer_dim=2 if (config.varlen or config.next_n >= 2) else 1,
            gmem_outer_stride=int(weights.stride(0)),
            swizzle_mode=0,
        ),
    }


def _prepare_global_barrier(executable: Any) -> None:
    try:
        prepare_global_barrier = executable.mod.get_function("__tvm_prepare_global_barrier")
    except AttributeError:
        prepare_global_barrier = None
    if prepare_global_barrier is not None:
        prepare_global_barrier()


def _prepare_tirx_invocation(
    data: dict[str, Any], logits: torch.Tensor | None = None, *, executable: Any | None = None
) -> dict[str, Any]:
    config: PagedMQALogitsFP8Config = data["config"]
    if logits is None:
        logits = _allocate_logits(config)
    if executable is None:
        executable = _compile_tirx_paged_mqa(config)
    return {
        "executable": executable,
        "logits": logits,
        "tensor_maps": _build_tirx_tensor_maps(data),
    }


def _run_tirx_invocation(data: dict[str, Any], invocation: dict[str, Any]) -> torch.Tensor:
    config: PagedMQALogitsFP8Config = data["config"]
    executable = invocation["executable"]
    tensor_maps = invocation["tensor_maps"]
    logits = invocation["logits"]
    indices = data["indices"]
    if indices is None:
        indices = torch.empty((config.batch_size,), dtype=torch.int32, device="cuda")
    _prepare_global_barrier(executable)
    executable.mod(
        config.batch_size,
        config.logits_stride,
        data["block_table"].stride(0),
        data["context_lens"],
        logits,
        data["block_table"],
        indices,
        data["schedule_meta"],
        tensor_maps["tensor_map_q"].ptr,
        tensor_maps["tensor_map_kv"].ptr,
        tensor_maps["tensor_map_kv_scales"].ptr,
        tensor_maps["tensor_map_weights"].ptr,
    )
    return logits


def _launch_tirx_paged_mqa(
    data: dict[str, Any], logits: torch.Tensor | None = None
) -> torch.Tensor:
    return _run_tirx_invocation(data, _prepare_tirx_invocation(data, logits))


def _calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x[:, : y.shape[1]].double()
    y = y.double()
    mask = y == float("-inf")
    x = x.masked_fill(mask, 0)
    y = y.masked_fill(mask, 0)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        return float("inf")
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return float((1 - sim).item())


def _calc_valid_diff(x: torch.Tensor, y: torch.Tensor, context_lens: torch.Tensor) -> float:
    expected_rows = context_lens.numel()
    if x.ndim != 2 or y.ndim != 2:
        raise AssertionError(f"expected rank-2 logits, got {x.shape=} and {y.shape=}")
    if x.shape[0] != expected_rows or y.shape[0] != expected_rows:
        raise AssertionError(
            f"logits row mismatch: expected {expected_rows}, got {x.shape[0]} and {y.shape[0]}"
        )
    required_width = int(context_lens.max().item())
    if x.shape[1] < required_width or y.shape[1] < required_width:
        raise AssertionError(
            f"logits width must cover {required_width}, got {x.shape[1]} and {y.shape[1]}"
        )

    width = min(x.shape[1], y.shape[1])
    valid = torch.arange(width, device=context_lens.device)[None, :] < context_lens.reshape(-1, 1)
    x = x[:, :width].double().masked_fill(~valid, 0)
    y = y[:, :width].double().masked_fill(~valid, 0)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        return float("inf")
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return float((1 - sim).item())


def _assert_correct(data: dict[str, Any], logits: torch.Tensor, *, name: str) -> float:
    reference = data["reference"]
    diff = _calc_diff(logits, reference)
    if diff >= _TEST_DIFF_THRESHOLD:
        raise AssertionError(f"{name} simulated diff {diff:.6g} >= {_TEST_DIFF_THRESHOLD}")
    return diff


def _assert_valid_correct(
    data: dict[str, Any], logits: torch.Tensor, reference: torch.Tensor, *, name: str
) -> float:
    diff = _calc_valid_diff(logits, reference, data["context_lens"])
    if diff >= _TEST_DIFF_THRESHOLD:
        raise AssertionError(f"{name} valid-logits diff {diff:.6g} >= {_TEST_DIFF_THRESHOLD}")
    return diff


def run_test(**kwargs: Any) -> None:
    data = prepare_data(**kwargs)
    config: PagedMQALogitsFP8Config = data["config"]
    deepgemm_logits = _run_deepgemm_paged_mqa(data, clean_logits=False)
    deepgemm_diff = _assert_correct(data, deepgemm_logits, name="DeepGEMM")
    tirx_logits = _launch_tirx_paged_mqa(data)
    torch.cuda.synchronize()
    tirx_diff = _assert_correct(data, tirx_logits, name="TIRx")
    if tirx_diff > max(deepgemm_diff, _TEST_DIFF_THRESHOLD):
        raise AssertionError(
            f"TIRx diff {tirx_diff:.6g} is worse than DeepGEMM diff {deepgemm_diff:.6g}"
        )
    if config.context_pattern.startswith("sglang_") and _sglang_cutedsl_available():
        cutedsl_runner = _make_sglang_cutedsl_runner(data)
        cutedsl_logits = cutedsl_runner()
        torch.cuda.synchronize()
        _assert_correct(data, cutedsl_logits, name="SGLang CuTeDSL")


def run_bench(**kwargs: Any) -> dict[str, Any]:
    from tvm.tirx.bench import bench

    # Tiny (~8-11µs) paged kernel: event timing is launch-jitter-noisy (sporadic
    # 10-13% ratio spread) and ~2x inflated by launch overhead. timer=None inherits the
    # global default (proton) -> pure per-kernel GPU time (~4.5µs, verified stable).
    timer = kwargs.pop("timer", None)
    # warmup/repeat: no hardcoded default here; pass through (None = defer to the
    # timer's own default; the graph timers ignore them anyway). Overridable via the
    # suite/CLI when a specific case needs a longer rep.
    warmup = kwargs.pop("warmup", None)
    repeat = kwargs.pop("repeat", None)
    _rounds = kwargs.pop("rounds", 1)
    _round_cooldown_s = kwargs.pop("round_cooldown_s", 1.0)
    config_kwargs = dict(kwargs)
    config = _make_config(**config_kwargs)
    tirx_executable = _compile_tirx_paged_mqa(config)

    # Allocate inputs once, outside the timed region (Triton-standard pure launch).
    # The independent Python reference is intentionally omitted here: it iterates
    # page-by-page and is prohibitively slow for SGLang's 131K-context sweep.
    data = _prepare_data(config, compute_reference=False)
    invocation = _prepare_tirx_invocation(data, executable=tirx_executable)
    deepgemm_logits = _run_deepgemm_paged_mqa(data, clean_logits=False)
    tirx_logits = _run_tirx_invocation(data, invocation)
    torch.cuda.synchronize()
    max_diff = _assert_valid_correct(data, tirx_logits, deepgemm_logits, name="TIRx vs DeepGEMM")
    torch.cuda.empty_cache()

    def _deepgemm():
        return lambda: _run_deepgemm_paged_mqa(data, clean_logits=False)

    def _sglang_cutedsl():
        cutedsl_runner = _make_sglang_cutedsl_runner(data)
        cutedsl_logits = cutedsl_runner()
        torch.cuda.synchronize()
        _assert_valid_correct(
            data, cutedsl_logits, deepgemm_logits, name="SGLang CuTeDSL vs DeepGEMM"
        )
        return cutedsl_runner

    result = bench(
        {"tirx": lambda: _run_tirx_invocation(data, invocation)},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=_rounds,
        round_cooldown_s=_round_cooldown_s,
        references={"deepgemm": _deepgemm, "sglang_cutedsl": _sglang_cutedsl},
    )
    result["max_diff"] = max_diff
    return result


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "DSA_INDEXER_LIKE_COVERAGE",
    "KERNEL_META",
    "SGLANG_BENCH_CONFIGS",
    "PagedMQALogitsFP8Config",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
]

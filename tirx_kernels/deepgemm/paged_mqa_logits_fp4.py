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
import os
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Any
from unittest import SkipTest

import torch

from tvm.ir.type import PointerType, PrimType
from tvm.backend.cuda.op import cuda_func_call

_DEEP_GEMM_MODULE_NAME = "deep_gemm"
_SM100_SMEM_CAPACITY = 232448
_TEST_DIFF_THRESHOLD = 5e-6


def _mxf4_block32_mma_src() -> str:
    return r"""
__forceinline__ __device__ void tvm_builtin_tcgen05_mma_mxf4_block32_ss(
    uint32_t tmem_c,
    uint64_t desc_a,
    uint64_t desc_b,
    uint32_t i_desc,
    uint32_t scale_c,
    uint32_t tmem_sfa,
    uint32_t tmem_sfb) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32 "
        "[%0], %1, %2, %3, [%5], [%6], p;\n"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(i_desc), "r"(scale_c),
          "r"(tmem_sfa), "r"(tmem_sfb));
}
"""


def _opaque_warp_id_src() -> str:
    return r"""
__forceinline__ __device__ int tvm_builtin_opaque_warp_id(int x) {
    int y;
    asm volatile("mov.u32 %0, %1;" : "=r"(y) : "r"(x));
    return y;
}
"""


def _opaque_sm_idx_u32_src() -> str:
    return r"""
__forceinline__ __device__ unsigned int tvm_builtin_opaque_sm_idx_u32(unsigned int x) {
    unsigned int y;
    asm volatile("mov.u32 %0, %1;" : "=r"(y) : "r"(x));
    return y;
}
"""


def _paged_mqa_logits_fp4_cuda_postproc(code: str) -> str:
    if "sm100_fp4_paged_mqa_logits_kernel" not in code:
        return code

    original = code
    replacements = {
        "int* __restrict__ block_table_flat_ptr": "const int* __restrict__ block_table_flat_ptr",
        "int* __restrict__ context_lens_flat_ptr": "const int* __restrict__ context_lens_flat_ptr",
        "int* __restrict__ indices_ptr": "const int* __restrict__ indices_ptr",
        "int* __restrict__ indices_flat_ptr": "const int* __restrict__ indices_flat_ptr",
        "int* __restrict__ schedule_meta_u32_flat_ptr": (
            "const int* __restrict__ schedule_meta_u32_flat_ptr"
        ),
    }
    for old, new in replacements.items():
        code = code.replace(old, new)

    code = code.replace(
        "((uint*)schedule_meta_u32_flat_ptr)", "((const uint*)schedule_meta_u32_flat_ptr)"
    )
    code = code.replace(
        "((unsigned int*)schedule_meta_u32_flat_ptr)",
        "((const unsigned int*)schedule_meta_u32_flat_ptr)",
    )

    dump_dir = os.environ.get("PAGED_MQA_FP4_POSTPROC_DUMP_DIR")
    if dump_dir:
        path = Path(dump_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "original.cu").write_text(original)
        (path / "postproc.cu").write_text(code)

    return code


@dataclass(frozen=True)
class PagedMQALogitsFP4Config:
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
        if self.head_dim != 128:
            raise ValueError("head_dim must be 128 for the SM100 FP4 paged MQA logits kernel")
        if self.page_size not in (32, 64):
            raise ValueError("page_size must match DeepGEMM block_kv 32 or 64")
        if self.split_kv % self.page_size != 0:
            raise ValueError("split_kv must be divisible by page_size")
        if self.max_num_pages <= 0 or self.num_pages < self.max_num_pages:
            raise ValueError("num_pages must cover max_num_pages")
        if self.logits_dtype not in ("float32", "bfloat16"):
            raise ValueError("logits_dtype must be 'float32' or 'bfloat16'")
        if not self.context_lens_2d:
            raise ValueError("DeepGEMM paged FP4 API currently requires 2D context_lens")
        if self.varlen and self.next_n != 1:
            raise ValueError("DeepGEMM varlen paged mode requires next_n == 1")
        if self.indices_pair_stride <= 0:
            raise ValueError("indices_pair_stride must be positive")


def _make_config(**kwargs: Any) -> PagedMQALogitsFP4Config:
    kwargs = {key: value for key, value in kwargs.items() if key != "label"}
    config = PagedMQALogitsFP4Config(**kwargs)
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
    return (
        f"b{config['batch_size']}_n{config['next_n']}_mp{config['max_num_pages']}_"
        f"ps{config['page_size']}_h{config['num_heads']}_d{config['head_dim']}_{dtype}_{mode}"
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
    varlen: bool = False,
    indices_pair_stride: int = 1,
) -> dict[str, Any]:
    config = {
        "batch_size": batch_size,
        "next_n": next_n,
        "max_num_pages": max_num_pages,
        "num_pages": num_pages,
        "num_heads": 64,
        "head_dim": 128,
        "page_size": page_size,
        "logits_dtype": logits_dtype,
        "seed": seed,
        "varlen": varlen,
        "indices_pair_stride": indices_pair_stride,
    }
    config["label"] = _config_label(config)
    return config


KERNEL_META = {
    "name": "deepgemm_sm100_fp4_paged_mqa_logits",
    "category": "deepgemm",
    "compute_capability": 10,
}

DSA_INDEXER_LIKE_COVERAGE = [
    _make_case(
        batch_size=batch_size,
        next_n=1,
        max_num_pages=max_num_pages,
        num_pages=max(11923, max_num_pages),
        page_size=page_size,
        logits_dtype=logits_dtype,
        seed=2000 + seed,
    )
    for seed, (batch_size, max_num_pages, page_size, logits_dtype) in enumerate(
        (batch_size, max_num_pages, page_size, logits_dtype)
        for logits_dtype in ("float32", "bfloat16")
        for page_size in (32, 64)
        for batch_size in (1, 2, 4, 8, 16)
        for max_num_pages in (1, 8, 32, 128)
    )
]

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
        batch_size=2,
        next_n=2,
        max_num_pages=4,
        num_pages=128,
        page_size=32,
        logits_dtype="bfloat16",
        seed=3,
    ),
    _make_case(
        batch_size=4,
        next_n=1,
        max_num_pages=4,
        num_pages=128,
        page_size=64,
        logits_dtype="float32",
        seed=4,
        varlen=True,
    ),
]

BENCH_CONFIGS = DSA_INDEXER_LIKE_COVERAGE


def load_deep_gemm_paged_mqa() -> tuple[Any, str]:
    try:
        import deep_gemm as module
    except Exception as exc:
        raise SkipTest(
            f"DeepGEMM FP4 paged MQA logits runtime unavailable: {_DEEP_GEMM_MODULE_NAME}: {exc}"
        ) from exc

    if not hasattr(module, "fp8_fp4_paged_mqa_logits"):
        raise SkipTest("DeepGEMM runtime unavailable: missing fp8_fp4_paged_mqa_logits")
    if not hasattr(module, "get_paged_mqa_logits_metadata"):
        raise SkipTest("DeepGEMM runtime unavailable: missing get_paged_mqa_logits_metadata")
    return module, "installed"


def _make_context_lens(config: PagedMQALogitsFP4Config) -> torch.Tensor:
    max_context_len = config.max_context_len
    if max_context_len == config.page_size:
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


def _make_block_table(config: PagedMQALogitsFP4Config) -> torch.Tensor:
    page_ids = torch.arange(config.num_pages, dtype=torch.int32, device="cuda")
    rows = []
    for batch_idx in range(config.batch_size):
        start = (batch_idx * config.max_num_pages) % config.num_pages
        rows.append(page_ids.roll(-start)[: config.max_num_pages])
    return torch.stack(rows, dim=0).contiguous()


def _make_indices(config: PagedMQALogitsFP4Config) -> torch.Tensor | None:
    if not config.varlen:
        return None
    indices = torch.arange(config.batch_size, dtype=torch.int32, device="cuda")
    if config.indices_pair_stride > 1:
        indices = indices // config.indices_pair_stride
    return indices.contiguous()


def _make_fused_kv_cache(
    config: PagedMQALogitsFP4Config, deep_gemm: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    kv_bf16 = torch.randn(
        config.num_pages, config.page_size, 1, config.head_dim, device="cuda", dtype=torch.bfloat16
    ).clamp_(-2.0, 2.0)
    kv_fp4 = deep_gemm.utils.per_token_cast_to_fp4(
        kv_bf16.view(-1, config.head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    kv_packed = kv_fp4[0].view(config.num_pages, config.page_size, config.head_dim // 2)
    kv_scales = kv_fp4[1].view(config.num_pages, config.page_size)
    kv_dequant = deep_gemm.utils.cast_back_from_fp4(
        kv_fp4[0], kv_fp4[1], gran_k=32, use_packed_ue8m0=True
    ).view(config.num_pages, config.page_size, 1, config.head_dim)
    fused = torch.empty(
        (config.num_pages, config.page_size, 1, config.head_dim // 2 + 4),
        dtype=torch.uint8,
        device="cuda",
    )
    fused_flat = fused.view(config.num_pages, config.page_size * (config.head_dim // 2 + 4))
    fused_flat[:, : config.page_size * config.head_dim // 2].copy_(
        kv_packed.view(torch.uint8).reshape(
            config.num_pages, config.page_size * config.head_dim // 2
        )
    )
    fused_flat[:, config.page_size * config.head_dim // 2 :].copy_(
        kv_scales.view(torch.uint8).reshape(config.num_pages, config.page_size * 4)
    )
    return fused.contiguous(), kv_dequant.view(
        config.num_pages, config.page_size, config.head_dim
    ).to(torch.bfloat16)


def _ref_paged_mqa_logits(
    q: torch.Tensor,
    kv_dequant: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    config: PagedMQALogitsFP4Config,
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


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    deep_gemm, source = load_deep_gemm_paged_mqa()
    config = _make_config(**kwargs)
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for SM100 FP4 paged MQA logits")
    if torch.cuda.get_device_capability()[0] < 10:
        raise SkipTest("SM100 FP4 paged MQA logits requires compute capability 10.x")

    torch.manual_seed(config.seed)
    runtime_config = PagedMQALogitsFP4Config(
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
    q_fp4 = deep_gemm.utils.per_token_cast_to_fp4(
        q_bf16.view(-1, config.head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    q_in = (
        q_fp4[0].view(config.batch_size, config.next_n, config.num_heads, config.head_dim // 2),
        q_fp4[1].view(config.batch_size, config.next_n, config.num_heads),
    )
    q_in = (q_in[0].contiguous(), q_in[1].contiguous())
    q_simulated = deep_gemm.utils.cast_back_from_fp4(
        q_fp4[0], q_fp4[1], gran_k=32, use_packed_ue8m0=True
    ).view(config.batch_size, config.next_n, config.num_heads, config.head_dim)
    fused_kv_cache, kv_dequant = _make_fused_kv_cache(config, deep_gemm)
    weights = torch.randn(
        config.batch_size * config.next_n, config.num_heads, device="cuda", dtype=torch.float32
    ).contiguous()
    context_lens = _make_context_lens(config)
    block_table = _make_block_table(config)
    indices = _make_indices(config)
    schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(
        context_lens, config.page_size, runtime_config.num_sms, indices
    )
    reference = _ref_paged_mqa_logits(
        q_simulated.to(torch.bfloat16), kv_dequant, weights, context_lens, block_table, config
    )
    return {
        "config": runtime_config,
        "reference_source": source,
        "q": q_bf16,
        "q_in": q_in,
        "fused_kv_cache": fused_kv_cache,
        "weights": weights,
        "context_lens": context_lens,
        "block_table": block_table,
        "indices": indices,
        "schedule_meta": schedule_meta,
        "reference": reference,
        "deep_gemm": deep_gemm,
    }


def get_kernel(**kwargs: Any):
    from tvm.script import tirx as T
    from tvm.backend.cuda.operator.tile_primitive.gemm_async.tcgen05 import sf_tmem_layout
    from tvm.tirx.layout import S, TCol, TileLayout, TLane

    config = _make_config(**kwargs)
    num_heads = config.num_heads
    head_dim = config.head_dim
    page_size = config.page_size
    k_pad_odd_n = (not config.varlen) and (config.next_n % 2 == 1) and (config.next_n >= 3)
    next_n_atom = 2 if (config.varlen or config.next_n >= 2) else 1
    num_next_n_atoms = _align_up(config.next_n, next_n_atom) // next_n_atom
    num_q_stages = 3
    num_kv_stages = 10
    num_tmem_stages = 3
    split_kv = config.split_kv
    num_blocks_per_split = split_kv // page_size
    num_specialized_threads = 128
    num_specialized_registers = 56
    num_math_registers = 224
    num_utccp_aligned_elems = 128
    umma_m = 128
    umma_k = 64
    umma_n = next_n_atom * num_heads
    num_math_warpgroups = split_kv // umma_m
    num_math_threads = num_math_warpgroups * 128
    num_threads = num_specialized_threads + num_math_threads
    num_warps = num_threads // 32
    spec_warp_start = num_math_warpgroups * 4
    num_sfq_atom = _align_up(next_n_atom * num_heads, num_utccp_aligned_elems)
    num_sfkv = _align_up(split_kv, num_utccp_aligned_elems)
    real_num_sfq_atom = next_n_atom * num_heads
    smem_alignment = 8 * (head_dim // 2)
    desc_sdo = 8 * (head_dim // 2) // 16
    sf_desc_sdo = 8 * 4 * 4 // 16
    smem_q_size_per_stage = next_n_atom * num_heads * (head_dim // 2)
    smem_sf_q_size_per_stage = num_sfq_atom * 4
    smem_kv_size_per_stage = split_kv * (head_dim // 2)
    smem_sf_kv_size_per_stage = num_sfkv * 4
    smem_weight_size_per_stage = next_n_atom * num_heads * 4
    smem_q_offset = 0
    smem_kv_offset = smem_q_offset + smem_q_size_per_stage * num_q_stages
    smem_sf_offset = smem_kv_offset + smem_kv_size_per_stage * num_kv_stages
    smem_sf_q_offset = smem_sf_offset
    smem_sf_kv_offset = smem_sf_q_offset + smem_sf_q_size_per_stage * num_q_stages
    smem_weights_offset = smem_sf_kv_offset + smem_sf_kv_size_per_stage * num_kv_stages
    smem_barrier_offset = smem_weights_offset + smem_weight_size_per_stage * num_q_stages
    num_total_barriers = num_q_stages * 2 + num_kv_stages * 2 + num_tmem_stages * 2
    full_q_barrier_base = 0
    empty_q_barrier_base = full_q_barrier_base + num_q_stages
    full_kv_barrier_base = empty_q_barrier_base + num_q_stages
    empty_kv_barrier_base = full_kv_barrier_base + num_kv_stages
    full_tmem_barrier_base = empty_kv_barrier_base + num_kv_stages
    empty_tmem_barrier_base = full_tmem_barrier_base + num_tmem_stages
    smem_tmem_ptr_offset = smem_barrier_offset + num_total_barriers * 8
    smem_total_bytes = smem_tmem_ptr_offset + 4
    if smem_total_bytes > _SM100_SMEM_CAPACITY:
        raise ValueError(f"dynamic shared memory {smem_total_bytes} exceeds SM100 capacity")
    num_accum_tmem_cols = next_n_atom * num_heads * num_tmem_stages
    num_sfa_tmem_cols = num_sfq_atom // 32
    num_sfb_tmem_cols = num_sfkv // 32
    num_requested_tmem_cols = num_accum_tmem_cols + num_sfa_tmem_cols + num_sfb_tmem_cols
    num_tmem_cols = 32
    if num_requested_tmem_cols > 32:
        num_tmem_cols = 64
    if num_requested_tmem_cols > 64:
        num_tmem_cols = 128
    if num_requested_tmem_cols > 128:
        num_tmem_cols = 256
    if num_requested_tmem_cols > 256:
        num_tmem_cols = 512
    tmem_start_col_of_sfq = num_accum_tmem_cols
    tmem_start_col_of_sfkv = num_accum_tmem_cols + num_sfa_tmem_cols
    sf_tmem_q_layout = sf_tmem_layout(128, SF_K=num_sfq_atom // 32, sf_per_mma=num_sfq_atom // 32)
    sf_tmem_kv_layout = sf_tmem_layout(128, SF_K=num_sfkv // 32, sf_per_mma=num_sfkv // 32)
    tmem_layout = TileLayout(S[(128, num_tmem_cols) : (1 @ TLane, 1 @ TCol)])
    logits_tir_dtype = "float32" if config.logits_dtype == "float32" else "bfloat16"
    cache_hint_sm90_evict_normal = "evict_normal"
    cache_hint_sm100_evict_normal = "evict_normal"
    cache_policy_evict_normal = T.uint64(1152921504606846976)
    has_cache_policy_evict_normal = 1
    tma_unicast_cta_mask = 0
    tma_no_cta_group_modifier = -1
    q_tma_block_inner = head_dim // 2
    q_tma_swizzle_mode = head_dim // 2
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
    kv_tma_block_inner = head_dim // 2
    kv_tma_swizzle_mode = head_dim // 2
    kv_tma_dtype_size = 1
    kv_tma_block_inner_atom = (
        kv_tma_block_inner if kv_tma_swizzle_mode == 0 else kv_tma_swizzle_mode // kv_tma_dtype_size
    )
    kv_tma_num_inner_atoms = kv_tma_block_inner // kv_tma_block_inner_atom
    sf_q_tma_block_inner = next_n_atom * num_heads
    sf_q_tma_swizzle_mode = 0
    sf_q_tma_dtype_size = 4
    sf_q_tma_block_inner_atom = (
        sf_q_tma_block_inner
        if sf_q_tma_swizzle_mode == 0
        else sf_q_tma_swizzle_mode // sf_q_tma_dtype_size
    )
    sf_q_tma_num_inner_atoms = sf_q_tma_block_inner // sf_q_tma_block_inner_atom
    sf_kv_tma_block_inner = page_size
    sf_kv_tma_swizzle_mode = 0
    sf_kv_tma_dtype_size = 4
    sf_kv_tma_block_inner_atom = (
        sf_kv_tma_block_inner
        if sf_kv_tma_swizzle_mode == 0
        else sf_kv_tma_swizzle_mode // sf_kv_tma_dtype_size
    )
    sf_kv_tma_num_inner_atoms = sf_kv_tma_block_inner // sf_kv_tma_block_inner_atom

    def atom_to_token_idx_expr(q_atom_idx):
        if config.varlen:
            return q_atom_idx
        if k_pad_odd_n:
            return q_atom_idx // T.uint32(num_next_n_atoms) * T.uint32(
                config.next_n
            ) + q_atom_idx % T.uint32(num_next_n_atoms) * T.uint32(next_n_atom)
        if next_n_atom == 1:
            return q_atom_idx
        return q_atom_idx * T.uint32(next_n_atom)

    def atom_to_block_table_row_expr(q_atom_idx):
        if config.varlen:
            return q_atom_idx
        if num_next_n_atoms == 1:
            return q_atom_idx
        return q_atom_idx // T.uint32(num_next_n_atoms)

    def get_num_kv_expr(q_atom_idx, runtime_batch_size, context_lens_flat, indices):
        # Local one-page specialization: CUDA get_num_kv still loads context_lens,
        # but this config family can only have one positive BLOCK_KV page.
        if not config.varlen and config.max_num_pages == 1:
            return T.uint32(1)
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
        else:
            if num_next_n_atoms == 1:
                q_idx: T.uint32 = q_atom_idx
            else:
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
        if num_next_n_atoms == 1:
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

    def shared_addr_u32(ptr):
        return T.cuda.cvta_generic_to_shared(ptr)

    def replace_smem_desc_addr(desc, smem_ptr):
        start_addr = T.cast(
            T.bitwise_and(T.shift_right(shared_addr_u32(smem_ptr), T.uint32(4)), T.uint32(0x3FFF)),
            "uint64",
        )
        return T.bitwise_or(T.bitwise_and(desc, T.bitwise_not(T.uint64(0x3FFF))), start_addr)

    def make_runtime_instr_desc_with_sf_id(desc, sfa_id, sfb_id):
        runtime_desc = T.bitwise_and(desc, T.uint32(0x9FFFFFCF))
        runtime_desc = T.bitwise_or(
            runtime_desc, T.shift_left(T.cast(sfa_id, "uint32"), T.uint32(29))
        )
        runtime_desc = T.bitwise_or(
            runtime_desc, T.shift_left(T.cast(sfb_id, "uint32"), T.uint32(4))
        )
        return T.shift_left(T.cast(runtime_desc, "uint64"), T.uint64(32))

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

    def mma_mxf4_block32_ss(desc_a, desc_b, tmem_c, scale_c, desc, tmem_sfa, tmem_sfb):
        i_desc_hi: T.uint32 = T.cast(desc >> T.uint64(32), "uint32")
        T.evaluate(
            cuda_func_call(
                "tvm_builtin_tcgen05_mma_mxf4_block32_ss",
                tmem_c,
                desc_a,
                desc_b,
                i_desc_hi,
                scale_c,
                tmem_sfa,
                tmem_sfb,
                source_code=_mxf4_block32_mma_src(),
            )
        )

    @T.prim_func
    def sm100_fp4_paged_mqa_logits(
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
        tensor_map_sf_q: T.TensorMap(),
        tensor_map_kv: T.TensorMap(),
        tensor_map_sf_kv: T.TensorMap(),
        tensor_map_weights: T.TensorMap(),
    ):
        T.device_entry()
        # TIRX_TRANSCRIBE_START sm100_fp4_paged_mqa_logits
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
        schedule_meta_u32_flat = T.decl_buffer(
            ((config.num_sms + 1) * 2,), "uint32", data=schedule_meta.data, scope="global"
        )
        sm_idx = T.cta_id([config.num_sms])
        sm_idx_u32: T.uint32 = cuda_func_call(
            "tvm_builtin_opaque_sm_idx_u32",
            T.cast(sm_idx, "uint32"),
            source_code=_opaque_sm_idx_u32_src(),
            return_type="uint32",
        )
        warp_idx = T.warp_id([num_warps])
        warp_idx_u32: T.let = T.cast(warp_idx, "uint32")
        warp_idx_presync: T.int32 = cuda_func_call(
            "tvm_builtin_opaque_warp_id",
            warp_idx,
            source_code=_opaque_warp_id_src(),
            return_type="int32",
        )
        warpgroup_idx = T.warpgroup_id([num_warps // 4])
        lane_idx = T.lane_id([32])
        lane_idx_u32: T.uint32 = T.cast(lane_idx, "uint32")

        if warp_idx_presync == spec_warp_start:
            T.evaluate(T.ptx.prefetch_tensormap(T.address_of(tensor_map_q)))
            T.evaluate(T.ptx.prefetch_tensormap(T.address_of(tensor_map_sf_q)))
            T.evaluate(T.ptx.prefetch_tensormap(T.address_of(tensor_map_weights)))
            T.evaluate(T.ptx.prefetch_tensormap(T.address_of(tensor_map_kv)))
            T.evaluate(T.ptx.prefetch_tensormap(T.address_of(tensor_map_sf_kv)))

        T.static_assert(
            num_specialized_threads == 128 and num_math_threads % 128 == 0, "Invalid threads"
        )
        T.static_assert(
            split_kv == num_math_warpgroups * umma_m and split_kv % num_utccp_aligned_elems == 0,
            "Invalid `SPLIT_KV`",
        )
        T.static_assert(split_kv == page_size * num_blocks_per_split, "Invalid `SPLIT_KV`")
        T.static_assert(smem_q_size_per_stage % smem_alignment == 0, "Unaligned TMA swizzling")
        T.static_assert(smem_kv_size_per_stage % smem_alignment == 0, "Unaligned TMA swizzling")
        T.static_assert(num_requested_tmem_cols <= 512, "Too many tensor memory")
        T.static_assert(num_tmem_cols <= 512, "Too many tensor memory")

        smem = T.alloc_buffer([smem_total_bytes], "uint8", scope="shared.dyn", align=smem_alignment)
        smem_q_data: T.let[T.Var(name="smem_q_data", dtype=PointerType(PrimType("uint8")))] = (
            T.reinterpret("handle", smem.ptr_to([smem_q_offset]))
        )
        smem_kv_data: T.let[T.Var(name="smem_kv_data", dtype=PointerType(PrimType("uint8")))] = (
            T.reinterpret("handle", smem.ptr_to([smem_kv_offset]))
        )
        smem_sf_q_data: T.let[
            T.Var(name="smem_sf_q_data", dtype=PointerType(PrimType("uint32")))
        ] = T.reinterpret("handle", smem.ptr_to([smem_sf_q_offset]))
        smem_sf_kv_data: T.let[
            T.Var(name="smem_sf_kv_data", dtype=PointerType(PrimType("uint32")))
        ] = T.reinterpret("handle", smem.ptr_to([smem_sf_kv_offset]))
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
            (num_q_stages, next_n_atom * num_heads, head_dim // 2),
            "uint8",
            data=smem_q_data,
            scope="shared.dyn",
            elem_offset=0,
            align=smem_alignment,
        )
        smem_kv = T.decl_buffer(
            (num_kv_stages, split_kv, head_dim // 2),
            "uint8",
            data=smem_kv_data,
            scope="shared.dyn",
            elem_offset=0,
            align=smem_alignment,
        )
        smem_sf_q = T.decl_buffer(
            (num_q_stages, num_sfq_atom),
            "uint32",
            data=smem_sf_q_data,
            scope="shared.dyn",
            elem_offset=0,
            align=16,
        )
        smem_sf_kv = T.decl_buffer(
            (num_kv_stages, num_sfkv),
            "uint32",
            data=smem_sf_kv_data,
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
        sfq_tmem = T.decl_buffer(
            (128, num_sfq_atom // 32),
            "float8_e8m0fnu",
            scope="tmem",
            allocated_addr=tmem_start_col_of_sfq,
            layout=sf_tmem_q_layout,
        )
        sfkv_tmem = T.decl_buffer(
            (128, num_sfkv // 32),
            "float8_e8m0fnu",
            scope="tmem",
            allocated_addr=tmem_start_col_of_sfkv,
            layout=sf_tmem_kv_layout,
        )
        fetch_result = T.alloc_local((4,), "uint32")
        q_pipeline_iter = T.alloc_local((1,), "uint32")
        kv_pipeline_iter = T.alloc_local((1,), "uint32")
        tmem_pipeline_iter = T.alloc_local((1,), "uint32")

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
        def advance_q_pipeline(step):
            current_idx: T.uint32 = q_pipeline_iter[0]
            fetch_result[0] = current_idx % T.uint32(num_q_stages)
            fetch_result[1] = (current_idx // T.uint32(num_q_stages)) & T.uint32(1)
            q_pipeline_iter[0] = current_idx + step

        @T.inline
        def advance_kv_pipeline(step):
            current_idx: T.uint32 = kv_pipeline_iter[0]
            fetch_result[2] = current_idx % T.uint32(num_kv_stages)
            fetch_result[3] = (current_idx // T.uint32(num_kv_stages)) & T.uint32(1)
            kv_pipeline_iter[0] = current_idx + step

        @T.inline
        def advance_tmem_pipeline(step):
            current_idx: T.uint32 = tmem_pipeline_iter[0]
            fetch_result[0] = current_idx % T.uint32(num_tmem_stages)
            fetch_result[1] = (current_idx // T.uint32(num_tmem_stages)) & T.uint32(1)
            tmem_pipeline_iter[0] = current_idx + step

        @T.inline
        def utccp_required_smem_warp_transpose(buf1d, base_offset):
            values = T.alloc_local((4,), "uint32")
            for i in T.unroll(0, 4):
                i_u32 = T.uint32(i)
                col = (
                    T.bitwise_xor(i_u32, lane_idx_u32 >> T.uint32(3)) * T.uint32(32) + lane_idx_u32
                )
                values[i] = T.ptx.ld(
                    buf1d.ptr_to([T.cast(base_offset + col, "int32")]),
                    "uint32",
                    "u32",
                    space="shared",
                )
            T.cuda.warp_sync()
            for i in T.unroll(0, 4):
                i_u32 = T.uint32(i)
                col = lane_idx_u32 * T.uint32(4) + T.bitwise_xor(i_u32, lane_idx_u32 >> T.uint32(3))
                T.evaluate(
                    T.ptx.st(
                        buf1d.ptr_to([T.cast(base_offset + col, "int32")]),
                        values[i],
                        space="shared",
                        ptx_type="u32",
                    )
                )

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
        def tma_load_2d_sf_q(dst, barrier_ptr, tensor_map, coord0, coord1):
            T.static_assert(
                cache_hint_sm90_evict_normal == cache_hint_sm100_evict_normal, "Invalid cache hint"
            )
            T.static_assert(sf_q_tma_num_inner_atoms == 1, "Unsupported split TMA atom")
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
        def tma_load_2d_sf_kv(dst, barrier_ptr, tensor_map, coord0, coord1):
            T.static_assert(
                cache_hint_sm90_evict_normal == cache_hint_sm100_evict_normal, "Invalid cache hint"
            )
            T.static_assert(sf_kv_tma_num_inner_atoms == 1, "Unsupported split TMA atom")
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
        def make_sf_desc(desc_sf, smem_ptr):
            T.ptx.tcgen05.encode_matrix_descriptor(
                T.address_of(desc_sf), smem_ptr, ldo=0, sdo=sf_desc_sdo, swizzle=0
            )

        @T.inline
        def make_smem_desc(desc, smem_ptr):
            T.ptx.tcgen05.encode_matrix_descriptor(
                T.address_of(desc), smem_ptr, ldo=0, sdo=desc_sdo, swizzle=2
            )

        @T.inline
        def issue_tma_q(stage_idx, tma_q_atom_idx):
            q_token_idx: T.uint32 = atom_to_token_idx_expr(tma_q_atom_idx)
            tma_load_2d_q(
                smem_q.ptr_to([stage_idx, 0, 0]),
                smem_barriers.ptr_to([full_q_barrier_base + stage_idx]),
                tensor_map_q,
                T.uint32(0),
                q_token_idx * T.uint32(num_heads),
            )
            tma_load_2d_sf_q(
                smem_sf_q.ptr_to([stage_idx, 0]),
                smem_barriers.ptr_to([full_q_barrier_base + stage_idx]),
                tensor_map_sf_q,
                T.uint32(0),
                q_token_idx,
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
                smem_q_size_per_stage + real_num_sfq_atom * 4 + smem_weight_size_per_stage,
            )

        if T.And(warp_idx_presync == spec_warp_start, T.ptx.elect_sync() != T.uint32(0)):
            for init_i in T.unroll(0, num_q_stages):
                mbarrier_init_cta(smem_barriers.ptr_to([full_q_barrier_base + init_i]), T.uint32(1))
                mbarrier_init_cta(
                    smem_barriers.ptr_to([empty_q_barrier_base + init_i]),
                    T.uint32(num_math_threads + 32),
                )
            T.ptx.fence.mbarrier_init()
        if T.And(warp_idx_presync == spec_warp_start + 1, T.ptx.elect_sync() != T.uint32(0)):
            for init_i in T.unroll(0, num_kv_stages):
                mbarrier_init_cta(
                    smem_barriers.ptr_to([full_kv_barrier_base + init_i]), T.uint32(1)
                )
                mbarrier_init_cta(
                    smem_barriers.ptr_to([empty_kv_barrier_base + init_i]), T.uint32(1)
                )
            T.ptx.fence.mbarrier_init()
        if warp_idx_presync == spec_warp_start + 2:
            if T.ptx.elect_sync():
                for init_i in T.unroll(0, num_tmem_stages):
                    mbarrier_init_cta(
                        smem_barriers.ptr_to([full_tmem_barrier_base + init_i]), T.uint32(1)
                    )
                    mbarrier_init_cta(
                        smem_barriers.ptr_to([empty_tmem_barrier_base + init_i]), T.uint32(128)
                    )
                T.ptx.fence.mbarrier_init()
            T.ptx.tcgen05.alloc(
                T.address_of(tmem_ptr_in_smem[0]), n_cols=num_tmem_cols, cta_group=1
            )
        T.cuda.cta_sync()

        cuda_grid_dependency_synchronize()

        q_pipeline_iter[0] = T.uint32(0)
        kv_pipeline_iter[0] = T.uint32(0)
        tmem_pipeline_iter[0] = T.uint32(0)

        if warp_idx == spec_warp_start:
            T.ptx.setmaxnreg(False, num_specialized_registers)
            if T.ptx.elect_sync():
                current_pack = schedule_meta_u32_flat.vload(
                    [T.cast(sm_idx_u32 * T.uint32(2), "int32")], dtype="uint32x2"
                )
                end_pack = schedule_meta_u32_flat.vload(
                    [T.cast((sm_idx_u32 + T.uint32(1)) * T.uint32(2), "int32")], dtype="uint32x2"
                )
                current_q_atom_idx: T.uint32 = T.Shuffle([current_pack], [0])
                current_kv_idx: T.uint32 = T.Shuffle([current_pack], [1]) * T.uint32(
                    num_blocks_per_split
                )
                end_q_atom_idx: T.uint32 = T.Shuffle([end_pack], [0])
                end_kv_idx: T.uint32 = T.Shuffle([end_pack], [1]) * T.uint32(num_blocks_per_split)
                current_num_kv: T.uint32 = get_num_kv_expr(
                    current_q_atom_idx, batch_size, context_lens_flat, indices
                )
                last_q_atom_idx: T.uint32 = batch_size * T.uint32(num_next_n_atoms)
                q_atom_idx: T.uint32 = current_q_atom_idx
                fetched_next_task: T.bool = T.bool(True)
                if T.And(current_q_atom_idx == end_q_atom_idx, current_kv_idx == end_kv_idx):
                    fetched_next_task = T.bool(False)
                else:
                    current_kv_idx = current_kv_idx + T.uint32(num_blocks_per_split)
                    if current_kv_idx >= current_num_kv:
                        current_kv_idx = T.uint32(0)
                        current_q_atom_idx = current_q_atom_idx + get_atom_advance_expr(
                            current_q_atom_idx, end_q_atom_idx, indices
                        )
                        if T.And(
                            should_refresh_num_kv_expr(current_q_atom_idx),
                            exist_q_atom_idx_expr(current_q_atom_idx, end_q_atom_idx, end_kv_idx),
                        ):
                            current_num_kv = get_num_kv_expr(
                                current_q_atom_idx, batch_size, context_lens_flat, indices
                            )
                while fetched_next_task:
                    if q_atom_idx != last_q_atom_idx:
                        advance_q_pipeline(T.uint32(1))
                        q_stage_idx: T.uint32 = fetch_result[0]
                        q_phase: T.uint32 = fetch_result[1]
                        mbarrier_wait_phase(
                            smem_barriers.ptr_to([empty_q_barrier_base + q_stage_idx]),
                            q_phase ^ T.uint32(1),
                        )
                        issue_tma_q(q_stage_idx, q_atom_idx)
                    last_q_atom_idx = q_atom_idx
                    q_atom_idx = current_q_atom_idx
                    if T.And(current_q_atom_idx == end_q_atom_idx, current_kv_idx == end_kv_idx):
                        fetched_next_task = T.bool(False)
                    else:
                        fetched_next_task = T.bool(True)
                        current_kv_idx = current_kv_idx + T.uint32(num_blocks_per_split)
                        if current_kv_idx >= current_num_kv:
                            current_kv_idx = T.uint32(0)
                            current_q_atom_idx = current_q_atom_idx + get_atom_advance_expr(
                                current_q_atom_idx, end_q_atom_idx, indices
                            )
                            if T.And(
                                should_refresh_num_kv_expr(current_q_atom_idx),
                                exist_q_atom_idx_expr(
                                    current_q_atom_idx, end_q_atom_idx, end_kv_idx
                                ),
                            ):
                                current_num_kv = get_num_kv_expr(
                                    current_q_atom_idx, batch_size, context_lens_flat, indices
                                )
            T.cuda.warp_sync()
        elif warp_idx == spec_warp_start + 1:
            T.ptx.setmaxnreg(False, num_specialized_registers)
            current_pack = schedule_meta_u32_flat.vload(
                [T.cast(sm_idx_u32 * T.uint32(2), "int32")], dtype="uint32x2"
            )
            end_pack = schedule_meta_u32_flat.vload(
                [T.cast((sm_idx_u32 + T.uint32(1)) * T.uint32(2), "int32")], dtype="uint32x2"
            )
            current_q_atom_idx: T.uint32 = T.Shuffle([current_pack], [0])
            current_kv_idx: T.uint32 = T.Shuffle([current_pack], [1]) * T.uint32(
                num_blocks_per_split
            )
            end_q_atom_idx: T.uint32 = T.Shuffle([end_pack], [0])
            end_kv_idx: T.uint32 = T.Shuffle([end_pack], [1]) * T.uint32(num_blocks_per_split)
            current_num_kv: T.uint32 = get_num_kv_expr(
                current_q_atom_idx, batch_size, context_lens_flat, indices
            )
            kv_block_idx_ptr: T.uint32 = T.uint32(32)
            kv_block_idx_storage: T.uint32 = T.uint32(0)
            last_q_atom_idx: T.uint32 = batch_size * T.uint32(num_next_n_atoms)
            q_atom_idx: T.uint32 = current_q_atom_idx
            kv_idx: T.uint32 = current_kv_idx
            num_kv: T.uint32 = current_num_kv
            fetched_next_task: T.bool = T.bool(True)
            if T.And(current_q_atom_idx == end_q_atom_idx, current_kv_idx == end_kv_idx):
                fetched_next_task = T.bool(False)
            else:
                current_kv_idx = current_kv_idx + T.uint32(num_blocks_per_split)
                if current_kv_idx >= current_num_kv:
                    current_kv_idx = T.uint32(0)
                    current_q_atom_idx = current_q_atom_idx + get_atom_advance_expr(
                        current_q_atom_idx, end_q_atom_idx, indices
                    )
                    if T.And(
                        should_refresh_num_kv_expr(current_q_atom_idx),
                        exist_q_atom_idx_expr(current_q_atom_idx, end_q_atom_idx, end_kv_idx),
                    ):
                        current_num_kv = get_num_kv_expr(
                            current_q_atom_idx, batch_size, context_lens_flat, indices
                        )
            while fetched_next_task:
                if q_atom_idx != last_q_atom_idx:
                    kv_block_idx_ptr = T.uint32(32)
                last_q_atom_idx = q_atom_idx

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
                kv_block_idx = T.alloc_local((num_blocks_per_split,), "int32")
                for block_i in T.unroll(0, num_blocks_per_split):
                    kv_block_idx[block_i] = T.cast(
                        T.cuda.__shfl_sync(
                            T.uint32(0xFFFFFFFF),
                            kv_block_idx_storage,
                            kv_block_idx_ptr + T.uint32(block_i),
                            32,
                        ),
                        "int32",
                    )
                kv_block_idx_ptr = kv_block_idx_ptr + T.uint32(num_blocks_per_split)
                T.static_assert(32 % num_blocks_per_split == 0, "Invalid `SPLIT_KV`")

                advance_kv_pipeline(T.uint32(1))
                kv_stage_idx: T.uint32 = fetch_result[2]
                kv_phase: T.uint32 = fetch_result[3]
                if T.ptx.elect_sync():
                    mbarrier_wait_phase(
                        smem_barriers.ptr_to([empty_kv_barrier_base + kv_stage_idx]),
                        kv_phase ^ T.uint32(1),
                    )
                    for block_i in T.unroll(0, num_blocks_per_split):
                        tma_load_3d_kv(
                            smem_kv.ptr_to([kv_stage_idx, block_i * page_size, 0]),
                            smem_barriers.ptr_to([full_kv_barrier_base + kv_stage_idx]),
                            tensor_map_kv,
                            T.uint32(0),
                            T.uint32(0),
                            T.cast(kv_block_idx[block_i], "uint32"),
                        )
                        tma_load_2d_sf_kv(
                            smem_sf_kv.ptr_to([kv_stage_idx, block_i * page_size]),
                            smem_barriers.ptr_to([full_kv_barrier_base + kv_stage_idx]),
                            tensor_map_sf_kv,
                            T.uint32(0),
                            T.cast(kv_block_idx[block_i], "uint32"),
                        )
                    mbarrier_arrive_and_expect_tx(
                        smem_barriers.ptr_to([full_kv_barrier_base + kv_stage_idx]),
                        smem_kv_size_per_stage + smem_sf_kv_size_per_stage,
                    )
                q_atom_idx = current_q_atom_idx
                kv_idx = current_kv_idx
                num_kv = current_num_kv
                if T.And(current_q_atom_idx == end_q_atom_idx, current_kv_idx == end_kv_idx):
                    fetched_next_task = T.bool(False)
                else:
                    fetched_next_task = T.bool(True)
                    current_kv_idx = current_kv_idx + T.uint32(num_blocks_per_split)
                    if current_kv_idx >= current_num_kv:
                        current_kv_idx = T.uint32(0)
                        current_q_atom_idx = current_q_atom_idx + get_atom_advance_expr(
                            current_q_atom_idx, end_q_atom_idx, indices
                        )
                        if T.And(
                            should_refresh_num_kv_expr(current_q_atom_idx),
                            exist_q_atom_idx_expr(current_q_atom_idx, end_q_atom_idx, end_kv_idx),
                        ):
                            current_num_kv = get_num_kv_expr(
                                current_q_atom_idx, batch_size, context_lens_flat, indices
                            )
        elif warp_idx == spec_warp_start + 2:
            T.ptx.setmaxnreg(False, num_specialized_registers)
            current_pack = schedule_meta_u32_flat.vload(
                [T.cast(sm_idx_u32 * T.uint32(2), "int32")], dtype="uint32x2"
            )
            end_pack = schedule_meta_u32_flat.vload(
                [T.cast((sm_idx_u32 + T.uint32(1)) * T.uint32(2), "int32")], dtype="uint32x2"
            )
            current_q_atom_idx: T.uint32 = T.Shuffle([current_pack], [0])
            current_kv_idx: T.uint32 = T.Shuffle([current_pack], [1]) * T.uint32(
                num_blocks_per_split
            )
            end_q_atom_idx: T.uint32 = T.Shuffle([end_pack], [0])
            end_kv_idx: T.uint32 = T.Shuffle([end_pack], [1]) * T.uint32(num_blocks_per_split)
            current_num_kv: T.uint32 = get_num_kv_expr(
                current_q_atom_idx, batch_size, context_lens_flat, indices
            )
            tmem_allocated: T.uint32 = T.ptx.ld(
                tmem_ptr_in_smem.ptr_to([0]), "uint32", "u32", space="shared"
            )
            T.cuda.trap_when_assert_failed(tmem_allocated == T.uint32(0))
            desc_i: T.uint32
            desc_sf: T.uint64
            desc_a: T.uint64
            desc_b: T.uint64
            T.ptx.tcgen05.encode_instr_descriptor_block_scaled(
                T.address_of(desc_i),
                d_dtype="float32",
                a_dtype="float4_e2m1fn",
                b_dtype="float4_e2m1fn",
                sfa_dtype="float8_e8m0fnu",
                sfb_dtype="float8_e8m0fnu",
                sfa_tmem_addr=0,
                sfb_tmem_addr=0,
                M=umma_m,
                N=umma_n,
                K=umma_k,
                trans_a=False,
                trans_b=False,
                n_cta_groups=1,
            )
            make_sf_desc(desc_sf, T.reinterpret("handle", T.uint64(0)))
            q_stage_idx: T.uint32 = T.uint32(0)
            q_phase: T.uint32 = T.uint32(0)
            last_q_atom_idx: T.uint32 = batch_size * T.uint32(num_next_n_atoms)
            q_atom_idx: T.uint32 = current_q_atom_idx
            fetched_next_task: T.bool = T.bool(True)
            if T.And(current_q_atom_idx == end_q_atom_idx, current_kv_idx == end_kv_idx):
                fetched_next_task = T.bool(False)
            else:
                current_kv_idx = current_kv_idx + T.uint32(num_blocks_per_split)
                if current_kv_idx >= current_num_kv:
                    current_kv_idx = T.uint32(0)
                    current_q_atom_idx = current_q_atom_idx + get_atom_advance_expr(
                        current_q_atom_idx, end_q_atom_idx, indices
                    )
                    if T.And(
                        should_refresh_num_kv_expr(current_q_atom_idx),
                        exist_q_atom_idx_expr(current_q_atom_idx, end_q_atom_idx, end_kv_idx),
                    ):
                        current_num_kv = get_num_kv_expr(
                            current_q_atom_idx, batch_size, context_lens_flat, indices
                        )
            while fetched_next_task:
                if q_atom_idx != last_q_atom_idx:
                    advance_q_pipeline(T.uint32(1))
                    q_stage_idx = fetch_result[0]
                    q_phase = fetch_result[1]
                    if last_q_atom_idx != batch_size * T.uint32(num_next_n_atoms):
                        mbarrier_arrive(
                            smem_barriers.ptr_to(
                                [
                                    empty_q_barrier_base
                                    + (q_stage_idx + T.uint32(num_q_stages - 1))
                                    % T.uint32(num_q_stages)
                                ]
                            )
                        )
                    mbarrier_wait_phase(
                        smem_barriers.ptr_to([full_q_barrier_base + q_stage_idx]), q_phase
                    )
                    sfq_stage_ptr: T.let[
                        T.Var(name="sfq_stage_ptr", dtype=PointerType(PrimType("uint32")))
                    ] = T.ptr_byte_offset(
                        smem_sf_q.data, q_stage_idx * T.uint32(num_sfq_atom * 4), "uint32"
                    )
                    sfq_stage = T.decl_buffer(
                        (num_sfq_atom,),
                        "uint32",
                        data=sfq_stage_ptr,
                        scope="shared.dyn",
                        elem_offset=0,
                        align=16,
                    )
                    for sfq_i in T.unroll(0, num_sfq_atom // num_utccp_aligned_elems):
                        sfq_base = T.uint32(sfq_i * num_utccp_aligned_elems)
                        utccp_required_smem_warp_transpose(sfq_stage, sfq_base)
                        T.ptx.fence.proxy_async("shared::cta")
                        desc_sf = replace_smem_desc_addr(desc_sf, sfq_stage.ptr_to([sfq_base]))
                        if T.ptx.elect_sync():
                            T.ptx.tcgen05.cp(
                                tmem_start_col_of_sfq + sfq_i * 4,
                                desc_sf,
                                shape="32x128b",
                                cta_group=1,
                                multicast="warpx4",
                            )
                        T.cuda.warp_sync()
                last_q_atom_idx = q_atom_idx

                advance_kv_pipeline(T.uint32(1))
                kv_stage_idx: T.uint32 = fetch_result[2]
                kv_phase: T.uint32 = fetch_result[3]
                mbarrier_wait_phase(
                    smem_barriers.ptr_to([full_kv_barrier_base + kv_stage_idx]), kv_phase
                )
                sfkv_stage_ptr: T.let[
                    T.Var(name="sfkv_stage_ptr", dtype=PointerType(PrimType("uint32")))
                ] = T.ptr_byte_offset(
                    smem_sf_kv.data, kv_stage_idx * T.uint32(num_sfkv * 4), "uint32"
                )
                sfkv_stage = T.decl_buffer(
                    (num_sfkv,),
                    "uint32",
                    data=sfkv_stage_ptr,
                    scope="shared.dyn",
                    elem_offset=0,
                    align=16,
                )
                for sfkv_i in T.unroll(0, num_sfkv // num_utccp_aligned_elems):
                    sfkv_base = T.uint32(sfkv_i * num_utccp_aligned_elems)
                    utccp_required_smem_warp_transpose(sfkv_stage, sfkv_base)
                    T.ptx.fence.proxy_async("shared::cta")

                if T.ptx.elect_sync():
                    for sfkv_i in T.unroll(0, num_sfkv // num_utccp_aligned_elems):
                        sfkv_base = T.uint32(sfkv_i * num_utccp_aligned_elems)
                        desc_sf = replace_smem_desc_addr(desc_sf, sfkv_stage.ptr_to([sfkv_base]))
                        T.ptx.tcgen05.cp(
                            tmem_start_col_of_sfkv + sfkv_i * 4,
                            desc_sf,
                            shape="32x128b",
                            cta_group=1,
                            multicast="warpx4",
                        )
                    for math_wg_i in T.unroll(0, num_math_warpgroups):
                        advance_tmem_pipeline(T.uint32(1))
                        tmem_stage_idx: T.uint32 = fetch_result[0]
                        tmem_phase: T.uint32 = fetch_result[1]
                        tmem_addr: T.uint32 = tmem_stage_idx * T.uint32(umma_n)
                        mbarrier_wait_phase(
                            smem_barriers.ptr_to([empty_tmem_barrier_base + tmem_stage_idx]),
                            tmem_phase ^ T.uint32(1),
                        )
                        T.ptx.tcgen05.fence.after_thread_sync()
                        for k in T.unroll(0, head_dim // umma_k):
                            runtime_desc_i = make_runtime_instr_desc_with_sf_id(
                                desc_i, k * 2, k * 2
                            )
                            make_smem_desc(
                                desc_a,
                                smem_kv.ptr_to([kv_stage_idx, math_wg_i * umma_m, k * umma_k // 2]),
                            )
                            make_smem_desc(desc_b, smem_q.ptr_to([q_stage_idx, 0, k * umma_k // 2]))
                            mma_mxf4_block32_ss(
                                desc_a,
                                desc_b,
                                tmem_addr,
                                k,
                                runtime_desc_i,
                                tmem_start_col_of_sfkv + math_wg_i * 4,
                                tmem_start_col_of_sfq,
                            )
                        T.ptx.tcgen05.commit(
                            smem_barriers.ptr_to([full_tmem_barrier_base + tmem_stage_idx])
                        )
                if T.ptx.elect_sync():
                    T.ptx.tcgen05.commit(
                        smem_barriers.ptr_to([empty_kv_barrier_base + kv_stage_idx]), cta_group=1
                    )
                q_atom_idx = current_q_atom_idx
                if T.And(current_q_atom_idx == end_q_atom_idx, current_kv_idx == end_kv_idx):
                    fetched_next_task = T.bool(False)
                else:
                    fetched_next_task = T.bool(True)
                    current_kv_idx = current_kv_idx + T.uint32(num_blocks_per_split)
                    if current_kv_idx >= current_num_kv:
                        current_kv_idx = T.uint32(0)
                        current_q_atom_idx = current_q_atom_idx + get_atom_advance_expr(
                            current_q_atom_idx, end_q_atom_idx, indices
                        )
                        if T.And(
                            should_refresh_num_kv_expr(current_q_atom_idx),
                            exist_q_atom_idx_expr(current_q_atom_idx, end_q_atom_idx, end_kv_idx),
                        ):
                            current_num_kv = get_num_kv_expr(
                                current_q_atom_idx, batch_size, context_lens_flat, indices
                            )
        elif warp_idx == spec_warp_start + 3:
            T.ptx.setmaxnreg(False, num_specialized_registers)
        elif warp_idx < spec_warp_start:
            T.ptx.setmaxnreg(True, num_math_registers)
            current_pack = schedule_meta_u32_flat.vload(
                [T.cast(sm_idx_u32 * T.uint32(2), "int32")], dtype="uint32x2"
            )
            end_pack = schedule_meta_u32_flat.vload(
                [T.cast((sm_idx_u32 + T.uint32(1)) * T.uint32(2), "int32")], dtype="uint32x2"
            )
            current_q_atom_idx: T.uint32 = T.Shuffle([current_pack], [0])
            current_kv_idx: T.uint32 = T.Shuffle([current_pack], [1]) * T.uint32(
                num_blocks_per_split
            )
            end_q_atom_idx: T.uint32 = T.Shuffle([end_pack], [0])
            end_kv_idx: T.uint32 = T.Shuffle([end_pack], [1]) * T.uint32(num_blocks_per_split)
            current_num_kv: T.uint32 = get_num_kv_expr(
                current_q_atom_idx, batch_size, context_lens_flat, indices
            )
            q_stage_idx: T.uint32 = T.uint32(0)
            q_phase: T.uint32 = T.uint32(0)
            math_warpgroup_idx: T.int32 = warpgroup_idx
            math_thread_idx: T.uint32 = warp_idx_u32 * T.uint32(32) + lane_idx_u32
            advance_tmem_pipeline(T.cast(math_warpgroup_idx, "uint32"))
            accum = T.alloc_local((num_heads,), "float32")
            cached_weights = T.alloc_local((next_n_atom, num_heads), "float32")
            last_q_atom_idx: T.uint32 = batch_size * T.uint32(num_next_n_atoms)
            q_atom_idx: T.uint32 = T.uint32(0)
            kv_idx: T.uint32 = T.uint32(0)
            num_kv: T.uint32 = T.uint32(0)
            is_paired_atom: T.bool = T.bool(False)
            T.static_assert(num_heads % 8 == 0, "Invalid head")

            @T.inline
            def reduce_and_store(num_iters_c, kv_offset_arg, tmem_stage_idx_arg):
                T.static_assert(num_heads == 32 or num_heads == 64, "Unsupported TMEM load size")
                for q_inner_i in T.unroll(0, num_iters_c):
                    tmem_addr: T.uint32 = tmem_stage_idx_arg * T.uint32(umma_n) + T.uint32(
                        q_inner_i * num_heads
                    )
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
                            shape="32x32b",
                            num=32,
                        )
                        T.ptx.tcgen05.wait.ld()
                        T.ptx.tcgen05.ld(
                            tmem_addr + T.uint32(num_heads // 2),
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
                            num=32,
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
                    result_f32: T.let = fadd_rn_noftz(
                        T.cuda.float2_x(sum_v), T.cuda.float2_y(sum_v)
                    )
                    result = T.cast(result_f32, logits_tir_dtype)
                    logits_flat[
                        T.cast(kv_offset_arg, "uint64")
                        + T.cast(q_inner_i, "uint64") * T.cast(logits_stride, "uint64")
                    ] = result
                    T.cuda.warp_sync()
                T.ptx.tcgen05.fence.before_thread_sync()
                mbarrier_arrive(
                    smem_barriers.ptr_to([empty_tmem_barrier_base + tmem_stage_idx_arg])
                )

            while True:
                q_atom_idx = current_q_atom_idx
                kv_idx = current_kv_idx
                num_kv = current_num_kv
                if T.And(current_q_atom_idx == end_q_atom_idx, current_kv_idx == end_kv_idx):
                    break
                current_kv_idx = current_kv_idx + T.uint32(num_blocks_per_split)
                if current_kv_idx >= current_num_kv:
                    current_kv_idx = T.uint32(0)
                    current_q_atom_idx = current_q_atom_idx + get_atom_advance_expr(
                        current_q_atom_idx, end_q_atom_idx, indices
                    )
                    if T.And(
                        should_refresh_num_kv_expr(current_q_atom_idx),
                        exist_q_atom_idx_expr(current_q_atom_idx, end_q_atom_idx, end_kv_idx),
                    ):
                        current_num_kv = get_num_kv_expr(
                            current_q_atom_idx, batch_size, context_lens_flat, indices
                        )
                if q_atom_idx != last_q_atom_idx:
                    advance_q_pipeline(T.uint32(1))
                    q_stage_idx = fetch_result[0]
                    q_phase = fetch_result[1]
                    if last_q_atom_idx != batch_size * T.uint32(num_next_n_atoms):
                        mbarrier_arrive(
                            smem_barriers.ptr_to(
                                [
                                    empty_q_barrier_base
                                    + (q_stage_idx + T.uint32(num_q_stages - 1))
                                    % T.uint32(num_q_stages)
                                ]
                            )
                        )
                    mbarrier_wait_phase(
                        smem_barriers.ptr_to([full_q_barrier_base + q_stage_idx]), q_phase
                    )
                    for weight_i in T.unroll(0, next_n_atom):
                        for weight_j in T.unroll(0, num_heads // 4):
                            weight_col = weight_j * 4
                            raw = smem_weights.vload(
                                [q_stage_idx, weight_i, weight_col], dtype="float32x4"
                            )
                            cached_weights[weight_i, weight_col] = T.Shuffle([raw], [0])
                            cached_weights[weight_i, weight_col + 1] = T.Shuffle([raw], [1])
                            cached_weights[weight_i, weight_col + 2] = T.Shuffle([raw], [2])
                            cached_weights[weight_i, weight_col + 3] = T.Shuffle([raw], [3])
                    if config.varlen:
                        is_paired_atom = get_atom_advance_expr(
                            q_atom_idx, batch_size, indices
                        ) == T.uint32(2)
                last_q_atom_idx = q_atom_idx
                kv_offset: T.uint64 = (
                    T.cast(atom_to_token_idx_expr(q_atom_idx), "uint64")
                    * T.cast(logits_stride, "uint64")
                    + T.cast(kv_idx * T.uint32(page_size), "uint64")
                    + T.cast(math_thread_idx, "uint64")
                )
                advance_tmem_pipeline(T.uint32(num_math_warpgroups))
                tmem_stage_idx: T.uint32 = fetch_result[0]
                tmem_phase: T.uint32 = fetch_result[1]
                mbarrier_wait_phase(
                    smem_barriers.ptr_to([full_tmem_barrier_base + tmem_stage_idx]), tmem_phase
                )
                T.ptx.tcgen05.fence.after_thread_sync()
                if config.varlen:
                    if is_paired_atom:
                        reduce_and_store(next_n_atom, kv_offset, tmem_stage_idx)
                    else:
                        reduce_and_store(1, kv_offset, tmem_stage_idx)
                elif k_pad_odd_n:
                    if q_atom_idx % T.uint32(num_next_n_atoms) == T.uint32(num_next_n_atoms - 1):
                        reduce_and_store(1, kv_offset, tmem_stage_idx)
                    else:
                        reduce_and_store(next_n_atom, kv_offset, tmem_stage_idx)
                else:
                    reduce_and_store(next_n_atom, kv_offset, tmem_stage_idx)
            T.ptx.bar.sync(8, num_math_threads)
            if warp_idx == 0:
                T.ptx.tcgen05.dealloc(T.uint32(0), n_cols=num_tmem_cols, cta_group=1)

    return sm100_fp4_paged_mqa_logits.with_attr(
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
    previous_postproc = tvm.get_global_func("tvm_callback_cuda_postproc", allow_missing=True)

    @tvm.register_global_func("tvm_callback_cuda_postproc", override=True)
    def _postproc(code: str, target: Any) -> str:
        if previous_postproc is not None:
            code = previous_postproc(code, target)
        return _paged_mqa_logits_fp4_cuda_postproc(code)

    try:
        with target:
            mod = tvm.IRModule({"main": kernel})
            return tvm.compile(mod, target=target, tir_pipeline="tirx")
    finally:
        if previous_postproc is not None:
            tvm.register_global_func("tvm_callback_cuda_postproc", previous_postproc, override=True)
        else:
            tvm.register_global_func(
                "tvm_callback_cuda_postproc", lambda code, target: code, override=True
            )


_compile_tirx_paged_mqa_for_config = cache(_compile_tirx_paged_mqa_for_config)


def _compile_tirx_paged_mqa(config: PagedMQALogitsFP4Config) -> Any:
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
    config: PagedMQALogitsFP4Config = data["config"]
    return data["deep_gemm"].fp8_fp4_paged_mqa_logits(
        q=data["q_in"],
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


def _allocate_logits(config: PagedMQALogitsFP4Config) -> torch.Tensor:
    return torch.full(
        (config.batch_size * config.next_n, config.logits_stride),
        float("-inf"),
        device="cuda",
        dtype=_torch_logits_dtype(config.logits_dtype),
    )


def _encode_fp4_packed_smem_tma_2d_desc(
    *,
    tensor: torch.Tensor,
    gmem_inner_dim: int,
    gmem_outer_dim: int,
    smem_inner_dim: int,
    smem_outer_dim: int,
    gmem_outer_stride: int,
    swizzle_mode: int,
) -> Any:
    from tirx_kernels.deepgemm import mega_moe

    desc = mega_moe._AlignedTensorMap()
    global_shape = (ctypes.c_uint64 * 2)(int(gmem_inner_dim), int(gmem_outer_dim))
    global_strides = (ctypes.c_uint64 * 1)(int(gmem_outer_stride * tensor.element_size()))
    box_dim = (ctypes.c_uint32 * 2)(int(smem_inner_dim), int(smem_outer_dim))
    element_strides = (ctypes.c_uint32 * 2)(1, 1)
    result = mega_moe._get_cuda_driver().cuTensorMapEncodeTiled(
        desc.ptr,
        13,
        ctypes.c_uint32(2),
        ctypes.c_void_p(int(tensor.data_ptr())),
        global_shape,
        global_strides,
        box_dim,
        element_strides,
        mega_moe._CUDA_TENSOR_MAP_INTERLEAVE_NONE,
        mega_moe._tensor_map_swizzle_from_mode(swizzle_mode),
        mega_moe._CUDA_TENSOR_MAP_L2_PROMOTION_L2_256B,
        mega_moe._CUDA_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if result != 0:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed for FP4 align8 with CUresult={result}")
    return desc


def _encode_fp4_packed_smem_tma_3d_desc(
    *,
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
) -> Any:
    from tirx_kernels.deepgemm import mega_moe

    desc = mega_moe._AlignedTensorMap()
    elem_size = int(tensor.element_size())
    global_shape = (ctypes.c_uint64 * 3)(
        int(gmem_inner_dim), int(gmem_mid_dim), int(gmem_outer_dim)
    )
    global_strides = (ctypes.c_uint64 * 2)(
        int(gmem_mid_stride * elem_size), int(gmem_outer_stride * elem_size)
    )
    box_dim = (ctypes.c_uint32 * 3)(int(smem_inner_dim), int(smem_mid_dim), int(smem_outer_dim))
    element_strides = (ctypes.c_uint32 * 3)(1, 1, 1)
    result = mega_moe._get_cuda_driver().cuTensorMapEncodeTiled(
        desc.ptr,
        13,
        ctypes.c_uint32(3),
        ctypes.c_void_p(int(tensor.data_ptr())),
        global_shape,
        global_strides,
        box_dim,
        element_strides,
        mega_moe._CUDA_TENSOR_MAP_INTERLEAVE_NONE,
        mega_moe._tensor_map_swizzle_from_mode(swizzle_mode),
        mega_moe._CUDA_TENSOR_MAP_L2_PROMOTION_L2_256B,
        mega_moe._CUDA_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if result != 0:
        raise RuntimeError(
            f"cuTensorMapEncodeTiled failed for FP4 3D align8 with CUresult={result}"
        )
    return desc


def _build_tirx_tensor_maps(data: dict[str, Any]) -> dict[str, Any]:
    import tvm
    from tirx_kernels.deepgemm.mega_moe import _encode_tma_2d_desc

    config: PagedMQALogitsFP4Config = data["config"]
    q_fp4, sf_q = data["q_in"]
    fused = data["fused_kv_cache"]
    weights = data["weights"]
    encode_tensormap = tvm.get_global_func("runtime.cuTensorMapEncodeTiled")
    kv_flat = fused.view(torch.uint8).view(
        config.num_pages, config.page_size * (config.head_dim // 2 + 4)
    )
    kv_fp4 = kv_flat[:, : config.page_size * config.head_dim // 2].reshape(
        config.num_pages, config.page_size, config.head_dim // 2
    )
    sf_kv = kv_flat[:, config.page_size * config.head_dim // 2 :].view(torch.int32)
    next_n_atom = 2 if (config.varlen or config.next_n >= 2) else 1

    return {
        "tensor_map_q": _encode_fp4_packed_smem_tma_2d_desc(
            tensor=q_fp4,
            gmem_inner_dim=config.head_dim,
            gmem_outer_dim=config.batch_size * config.next_n * config.num_heads,
            smem_inner_dim=config.head_dim,
            smem_outer_dim=next_n_atom * config.num_heads,
            gmem_outer_stride=int(q_fp4.stride(2)),
            swizzle_mode=config.head_dim // 2,
        ),
        "tensor_map_sf_q": _encode_tma_2d_desc(
            encode_tensormap=encode_tensormap,
            tensor=sf_q,
            gmem_inner_dim=config.num_heads,
            gmem_outer_dim=config.batch_size * config.next_n,
            smem_inner_dim=config.num_heads,
            smem_outer_dim=next_n_atom,
            gmem_outer_stride=int(sf_q.stride(1)),
            swizzle_mode=0,
        ),
        "tensor_map_kv": _encode_fp4_packed_smem_tma_3d_desc(
            tensor=kv_fp4,
            gmem_inner_dim=config.head_dim,
            gmem_mid_dim=config.page_size,
            gmem_outer_dim=config.num_pages,
            smem_inner_dim=config.head_dim,
            smem_mid_dim=config.page_size,
            smem_outer_dim=1,
            gmem_mid_stride=int(kv_fp4.stride(1)),
            gmem_outer_stride=int(kv_fp4.stride(0)),
            swizzle_mode=config.head_dim // 2,
        ),
        "tensor_map_sf_kv": _encode_tma_2d_desc(
            encode_tensormap=encode_tensormap,
            tensor=sf_kv,
            gmem_inner_dim=config.page_size,
            gmem_outer_dim=config.num_pages,
            smem_inner_dim=config.page_size,
            smem_outer_dim=1,
            gmem_outer_stride=int(sf_kv.stride(0)),
            swizzle_mode=0,
        ),
        "tensor_map_weights": _encode_tma_2d_desc(
            encode_tensormap=encode_tensormap,
            tensor=weights,
            gmem_inner_dim=config.num_heads,
            gmem_outer_dim=config.batch_size * config.next_n,
            smem_inner_dim=config.num_heads,
            smem_outer_dim=next_n_atom,
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
    data: dict[str, Any], logits: torch.Tensor | None = None
) -> dict[str, Any]:
    config: PagedMQALogitsFP4Config = data["config"]
    if logits is None:
        logits = _allocate_logits(config)
    return {
        "executable": _compile_tirx_paged_mqa(config),
        "logits": logits,
        "tensor_maps": _build_tirx_tensor_maps(data),
    }


def _run_tirx_invocation(data: dict[str, Any], invocation: dict[str, Any]) -> torch.Tensor:
    config: PagedMQALogitsFP4Config = data["config"]
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
        tensor_maps["tensor_map_sf_q"].ptr,
        tensor_maps["tensor_map_kv"].ptr,
        tensor_maps["tensor_map_sf_kv"].ptr,
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


def run_test(**kwargs: Any) -> None:
    data = prepare_data(**kwargs)
    deepgemm_logits = _run_deepgemm_paged_mqa(data, clean_logits=False)
    deepgemm_diff = _assert_correct(data, deepgemm_logits, name="DeepGEMM")
    tirx_logits = _launch_tirx_paged_mqa(data)
    torch.cuda.synchronize()
    tirx_diff = _assert_correct(data, tirx_logits, name="TIRx")
    if tirx_diff > max(deepgemm_diff, _TEST_DIFF_THRESHOLD):
        raise AssertionError(
            f"TIRx diff {tirx_diff:.6g} is worse than DeepGEMM diff {deepgemm_diff:.6g}"
        )


def run_bench(**kwargs: Any) -> dict[str, Any]:
    from tvm.tirx.bench import bench, bench_impls_mode, tensor_bytes

    warmup = kwargs.pop("warmup", 10)
    repeat = kwargs.pop("repeat", 30)
    timer = kwargs.pop("timer", "proton")
    benchmark_order_mode = kwargs.pop("benchmark_order_mode", "bidirectional")
    config_kwargs = dict(kwargs)

    data = prepare_data(**config_kwargs)
    invocation = _prepare_tirx_invocation(data)
    tirx_logits = _run_tirx_invocation(data, invocation)
    torch.cuda.synchronize()
    tirx_diff = _assert_correct(data, tirx_logits, name="TIRx")
    torch.cuda.empty_cache()

    def make_input() -> tuple[tuple[dict[str, Any], dict[str, Any]], int]:
        data = prepare_data(**config_kwargs)
        invocation = _prepare_tirx_invocation(data)
        return (data, invocation), tensor_bytes(
            data["q_in"],
            data["fused_kv_cache"],
            data["weights"],
            data["context_lens"],
            data["block_table"],
            data["schedule_meta"],
            invocation["logits"],
        )

    def _deepgemm():
        return lambda case: _run_deepgemm_paged_mqa(case[0], clean_logits=False)

    funcs_tirx_first = {"tirx": lambda case: _run_tirx_invocation(case[0], case[1])}
    funcs_deepgemm_first = {"tirx": lambda case: _run_tirx_invocation(case[0], case[1])}

    if bench_impls_mode() == "baseline":
        result = bench(
            funcs_tirx_first,
            make_input,
            warmup=warmup,
            repeat=repeat,
            timer=timer,
            proton_name="deepgemm_sm100_fp4_paged_mqa_logits",
            references={"deepgemm": _deepgemm},
        )
    elif bench_impls_mode() == "ours":
        result = bench(
            funcs_tirx_first,
            make_input,
            warmup=warmup,
            repeat=repeat,
            timer=timer,
            proton_name="deepgemm_sm100_fp4_paged_mqa_logits",
        )
    elif benchmark_order_mode == "tirx_first":
        result = bench(
            funcs_tirx_first,
            make_input,
            warmup=warmup,
            repeat=repeat,
            timer=timer,
            proton_name="deepgemm_sm100_fp4_paged_mqa_logits",
            references={"deepgemm": _deepgemm},
        )
    elif benchmark_order_mode == "deepgemm_first":
        result = bench(
            funcs_deepgemm_first,
            make_input,
            warmup=warmup,
            repeat=repeat,
            timer=timer,
            proton_name="deepgemm_sm100_fp4_paged_mqa_logits",
            references={"deepgemm": _deepgemm},
        )
    elif benchmark_order_mode == "bidirectional":
        tirx_first = bench(
            funcs_tirx_first,
            make_input,
            warmup=warmup,
            repeat=repeat,
            timer=timer,
            proton_name="deepgemm_sm100_fp4_paged_mqa_logits",
            references={"deepgemm": _deepgemm},
        )
        deepgemm_first = bench(
            funcs_deepgemm_first,
            make_input,
            warmup=warmup,
            repeat=repeat,
            timer=timer,
            proton_name="deepgemm_sm100_fp4_paged_mqa_logits",
            references={"deepgemm": _deepgemm},
        )
        result = {
            "impls": {
                "tirx": (tirx_first["impls"]["tirx"] + deepgemm_first["impls"]["tirx"]) / 2,
                "deepgemm": (
                    tirx_first["impls"]["deepgemm"] + deepgemm_first["impls"]["deepgemm"]
                )
                / 2,
            },
            "errors": {**tirx_first.get("errors", {}), **deepgemm_first.get("errors", {})},
            "timer": timer,
            "benchmark_protocol": {
                **tirx_first["benchmark_protocol"],
                "order": ["tirx", "deepgemm", "deepgemm", "tirx"],
                "order_mode": "bidirectional_average",
                "component_runs": [
                    tirx_first["benchmark_protocol"],
                    deepgemm_first["benchmark_protocol"],
                ],
            },
            "component_impls": {
                "tirx_first": tirx_first["impls"],
                "deepgemm_first": deepgemm_first["impls"],
            },
        }
    else:
        raise ValueError(
            "benchmark_order_mode must be one of 'bidirectional', 'tirx_first', or 'deepgemm_first'"
        )
    result["max_diff"] = tirx_diff
    return result


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "DSA_INDEXER_LIKE_COVERAGE",
    "KERNEL_META",
    "PagedMQALogitsFP4Config",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
]

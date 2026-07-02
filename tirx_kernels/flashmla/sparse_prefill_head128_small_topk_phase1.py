from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any
from unittest import SkipTest

import torch

from tirx_kernels.flashmla.sparse_prefill_head128_phase1 import (
    _canonical_warp_idx_sync,
    _encode_tma_desc,
    _fdividef,
    _fma_f32x2,
    _ldg_f32_at,
    _ldg_i32_at,
    _mul_f32x2,
    _shfl_sync_i32,
    _tcgen05_mma_ws_ss_2cta,
    _tcgen05_mma_ws_ts_2cta,
    _tma_gather4_kv_cta_group2,
)
from tvm.script import tirx as T
from tvm.tirx.lang.pipeline import MBarrier, TCGen05Bar, TMABar

B_H = 128
B_TOPK = 64
D_QK = 512
D_V = 512
NUM_THREADS = 512
NUM_K_BUFS = 4
NUM_INDEX_BUFS = 4
NUM_WORKER_THREADS = (128 + 4 + (B_TOPK // 8) + 1 + 128) * 2 + 1
MAX_INIT_VAL = -1.0e30
LOG_2_E = math.log2(math.e)
LN_2 = math.log(2.0)

SMALL_TOPK_HEAD128_LAUNCH_PARAM_TAGS = (
    "blockIdx.x",
    "clusterCtaIdx.x",
    "threadIdx.x",
    "tirx.use_programtic_dependent_launch",
    "tirx.use_dyn_shared_memory",
)

TMEM_COL_O = 0
TMEM_COL_Q = 256
TMEM_COL_P = 384

BF16_BYTES = 2
B_EPI = 64
P_TMEM_ELEMENTS = B_TOPK // 2
K_MAJOR_SWIZZLED_DESC_LDO = 1
Q_DESC_SDO = 64
K_DESC_SDO = 64
S_DESC_LDO = 64
S_DESC_SDO = 8
V_DESC_LDO = 512
V_DESC_SDO = 64

NAMED_BARRIER_WG0_SYNC = 0
NAMED_BARRIER_WG2_SYNC = 1
NAMED_BARRIER_WG2_WARP02_SYNC = 2
NAMED_BARRIER_WG2_WARP13_SYNC = 3

WG1_NUM_WARPS = 4
WG1_ROWS_PER_WARP = B_TOPK // 4
WG3_NUM_ELEMS_PER_THREAD = B_TOPK // 2

_IMPLEMENTATION_COMPLETE = True


@dataclass(frozen=True)
class SparseFlashMLAPrefillHead128SmallTopKConfig:
    label: str
    s_q: int
    s_kv: int
    topk: int
    d_qk: int = D_QK
    h_q: int = B_H
    h_kv: int = 1
    d_v: int = D_V
    have_attn_sink: bool = False
    have_topk_length: bool = False
    inject_invalid_indices: bool = False
    seed: int = 0

    def validate(self) -> None:
        if self.h_q != B_H:
            raise ValueError("head128 small-topk phase1 requires h_q == 128")
        if self.h_kv != 1:
            raise ValueError("head128 small-topk phase1 requires h_kv == 1")
        if self.d_qk != D_QK:
            raise ValueError("head128 small-topk phase1 is scoped to d_qk == 512")
        if self.d_v != D_V:
            raise ValueError("head128 small-topk phase1 requires d_v == 512")
        if self.topk % B_TOPK != 0:
            raise ValueError("small-topk phase1 requires topk to be a multiple of 64")
        if self.topk > 1280:
            raise ValueError("topk > 1280 dispatches outside the small-topk phase1 scope")


CONFIGS = [
    {"label": "smalltopk_dqk512_s1_kv128_topk64", "s_q": 1, "s_kv": 128, "topk": 64},
    {
        "label": "smalltopk_features_dqk512_s17_kv2048_topk1280",
        "s_q": 17,
        "s_kv": 2048,
        "topk": 1280,
        "have_attn_sink": True,
        "have_topk_length": True,
    },
    {
        "label": "smalltopk_invalid_indices_dqk512_s3_kv257_topk192",
        "s_q": 3,
        "s_kv": 257,
        "topk": 192,
        "inject_invalid_indices": True,
    },
]

BENCH_CONFIGS = [
    {
        "label": f"bench_smalltopk_dqk512_hq128_s4096_kv{s_kv}_topk1280",
        "s_q": 4096,
        "s_kv": s_kv,
        "topk": 1280,
        "h_q": B_H,
        "have_attn_sink": True,
    }
    for s_kv in (8192, 32768, 65536)
]

KERNEL_META = {
    "name": "sparse_flashmla_prefill_head128_small_topk_phase1",
    "category": "flashmla",
    "compute_capability": 10,
}


def _cfg(**kwargs: Any) -> SparseFlashMLAPrefillHead128SmallTopKConfig:
    cfg_fields = {field.name for field in fields(SparseFlashMLAPrefillHead128SmallTopKConfig)}
    cfg_kwargs = {key: value for key, value in kwargs.items() if key in cfg_fields}
    if "label" not in cfg_kwargs:
        cfg_kwargs["label"] = "custom"
    cfg = SparseFlashMLAPrefillHead128SmallTopKConfig(**cfg_kwargs)
    cfg.validate()
    return cfg


def _flashmla_small_topk_dispatch_reason(cfg: SparseFlashMLAPrefillHead128SmallTopKConfig) -> str:
    if cfg.h_q != B_H:
        return "out_of_scope: h_q != 128 dispatches to head64 or unsupported path"
    if cfg.h_kv != 1:
        return "out_of_scope: h_kv != 1 violates FlashMLA sparse prefill phase1 assumptions"
    if cfg.d_qk != D_QK:
        return "out_of_scope: small-topk head128 supports only D_QK=512"
    if cfg.d_v != D_V:
        return "out_of_scope: d_v != 512"
    if cfg.topk > 1280:
        return "out_of_scope: topk > 1280 dispatches to regular head128 when supported"
    return "small_topk: sm100 head128 run_fwd_for_small_topk_phase1_kernel<Prefill, 512>"


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    cfg = _cfg(**kwargs)
    device = kwargs.get("device", "cuda")
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed)

    q = torch.randn(
        (cfg.s_q, cfg.h_q, cfg.d_qk), device=device, dtype=torch.bfloat16, generator=gen
    )
    kv = torch.randn(
        (cfg.s_kv, cfg.h_kv, cfg.d_qk), device=device, dtype=torch.bfloat16, generator=gen
    )
    out = torch.empty((cfg.s_q, cfg.h_q, cfg.d_v), device=device, dtype=torch.bfloat16)
    max_logits = torch.empty((cfg.s_q, cfg.h_q), device=device, dtype=torch.float32)
    lse = torch.empty((cfg.s_q, cfg.h_q), device=device, dtype=torch.float32)

    indices = torch.randint(
        low=0,
        high=cfg.s_kv,
        size=(cfg.s_q, cfg.h_kv, cfg.topk),
        device=device,
        dtype=torch.int32,
        generator=gen,
    )
    if cfg.inject_invalid_indices:
        indices[:, :, 0] = -1
        indices[:, :, 1] = cfg.s_kv
        indices[:, :, 2] = cfg.s_kv + 17
        indices[:, :, -1] = -7
    attn_sink = (
        torch.randn((cfg.h_q,), device=device, dtype=torch.float32, generator=gen)
        if cfg.have_attn_sink
        else torch.empty((cfg.h_q,), device=device, dtype=torch.float32)
    )
    if cfg.have_topk_length:
        topk_length = torch.randint(
            low=0,
            high=cfg.topk + 1,
            size=(cfg.s_q,),
            device=device,
            dtype=torch.int32,
            generator=gen,
        )
    else:
        topk_length = torch.empty((cfg.s_q,), device=device, dtype=torch.int32)

    sm_scale = 1.0 / math.sqrt(cfg.d_qk)
    return {
        "config": cfg,
        "q": q,
        "kv": kv,
        "indices": indices,
        "attn_sink": attn_sink,
        "topk_length": topk_length,
        "out": out,
        "max_logits": max_logits,
        "lse": lse,
        "sm_scale": sm_scale,
        "sm_scale_div_log2": sm_scale * LOG_2_E,
        "dispatch_reason": _flashmla_small_topk_dispatch_reason(cfg),
    }


def _reference_sparse_prefill(
    case: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg: SparseFlashMLAPrefillHead128SmallTopKConfig = case["config"]
    q = case["q"].float()
    kv = case["kv"][:, 0, :].float()
    indices = case["indices"][:, 0, :].to(torch.long)
    sm_scale = case["sm_scale"]
    ref_out = torch.zeros((cfg.s_q, cfg.h_q, cfg.d_v), device=q.device, dtype=torch.float32)
    ref_max_logits = torch.full((cfg.s_q, cfg.h_q), -float("inf"), device=q.device)
    ref_lse = torch.full((cfg.s_q, cfg.h_q), float("inf"), device=q.device)

    for s_q_idx in range(cfg.s_q):
        length = int(case["topk_length"][s_q_idx].item()) if cfg.have_topk_length else cfg.topk
        row_indices = indices[s_q_idx]
        pos = torch.arange(cfg.topk, device=q.device)
        valid = (pos < length) & (row_indices >= 0) & (row_indices < cfg.s_kv)
        if not torch.any(valid):
            continue
        selected = row_indices.clamp(0, cfg.s_kv - 1)
        k_full = kv[selected]
        logits = torch.matmul(q[s_q_idx], k_full[:, : cfg.d_qk].T) * sm_scale
        logits[:, ~valid] = -float("inf")
        max_logits = torch.max(logits, dim=-1).values
        exp_logits = torch.exp(logits - max_logits[:, None])
        exp_logits[:, ~valid] = 0.0
        denom = torch.sum(exp_logits, dim=-1)
        if cfg.have_attn_sink:
            sink = case["attn_sink"].float()
            denom_with_sink = denom + torch.exp(sink - max_logits)
        else:
            denom_with_sink = denom
        ref_out[s_q_idx] = torch.matmul(exp_logits, k_full[:, : cfg.d_v]) / denom_with_sink[:, None]
        ref_max_logits[s_q_idx] = max_logits
        ref_lse[s_q_idx] = max_logits + torch.log(denom)
    return ref_out.to(torch.bfloat16), ref_max_logits, ref_lse


def _build_tirx_tensor_maps(case: dict[str, Any]) -> dict[str, Any]:
    cfg: SparseFlashMLAPrefillHead128SmallTopKConfig = case["config"]
    q = case["q"]
    kv = case["kv"]
    out = case["out"]

    return {
        "tensor_map_q": _encode_tma_desc(
            tensor=q,
            global_shape=(64, cfg.h_q, 2, cfg.d_qk // 64 // 2, cfg.s_q),
            global_strides=(int(q.stride(1)), cfg.d_qk // 2, 64, int(q.stride(0))),
            box_dim=(64, cfg.h_q // 2, 2, cfg.d_qk // 64 // 2, 1),
            swizzle_mode=128,
        ),
        "tensor_map_kv": _encode_tma_desc(
            tensor=kv,
            global_shape=(cfg.d_qk, cfg.s_kv),
            global_strides=(int(kv.stride(0)),),
            box_dim=(64, 1),
            swizzle_mode=128,
        ),
        "tensor_map_o": _encode_tma_desc(
            tensor=out,
            global_shape=(64, cfg.h_q, cfg.d_v // 64, cfg.s_q, 1),
            global_strides=(cfg.d_v, 64, cfg.h_q * cfg.d_v, cfg.h_q * cfg.d_v),
            box_dim=(64, cfg.h_q // 2, cfg.d_v // 64, 1, 1),
            swizzle_mode=128,
        ),
    }


def _make_tirx_launch(case: dict[str, Any]) -> dict[str, Any]:
    import ctypes

    tensor_maps = _build_tirx_tensor_maps(case)
    cfg: SparseFlashMLAPrefillHead128SmallTopKConfig = case["config"]
    attn_sink_ptr = (
        ctypes.c_void_p(int(case["attn_sink"].data_ptr()))
        if cfg.have_attn_sink
        else ctypes.c_void_p(0)
    )
    topk_length_ptr = (
        ctypes.c_void_p(int(case["topk_length"].data_ptr()))
        if cfg.have_topk_length
        else ctypes.c_void_p(0)
    )
    return {
        "case": case,
        "tensor_maps": tensor_maps,
        "args": (
            case["q"],
            case["kv"].reshape(-1),
            case["indices"].reshape(-1),
            attn_sink_ptr,
            topk_length_ptr,
            case["out"],
            case["max_logits"],
            case["lse"],
            tensor_maps["tensor_map_q"].ptr,
            tensor_maps["tensor_map_kv"].ptr,
            tensor_maps["tensor_map_o"].ptr,
        ),
    }


def _build_tirx_launches(case: dict[str, Any]) -> list[dict[str, Any]]:
    return [_make_tirx_launch(case)]


def _run_tirx_launches(
    executable: Any, launches: list[dict[str, Any]], *, output_case: dict[str, Any] | None = None
) -> None:
    for launch in launches:
        executable(*launch["args"])


def _tma_5d_cta_group2_nosplit(
    dst_ptr: Any,
    bar_ptr: Any,
    tensor_map_ptr: Any,
    coord0: Any,
    coord1: Any,
    coord2: Any,
    coord3: Any,
    coord4: Any,
    cache_hint: Any,
) -> Any:
    func_name = "sparse_flashmla_small_topk_head128_tma_5d_cta_group2_nosplit"
    source_code = f"""
__device__ __forceinline__ void {func_name}(
    void* dst_ptr, void* bar_ptr, unsigned long long tensor_map_addr,
    int coord0, int coord1, int coord2, int coord3, int coord4,
    unsigned long long cache_hint) {{
  uint32_t smem_addr;
  uint32_t mbar_addr;
  asm volatile(
    "{{\\n\\t"
    ".reg .u64 smem_addr64;\\n\\t"
    "cvta.to.shared.u64 smem_addr64, %1;\\n\\t"
    "cvt.u32.u64 %0, smem_addr64;\\n\\t"
    "}}\\n"
    : "=r"(smem_addr) : "l"(dst_ptr));
  asm volatile(
    "{{\\n\\t"
    ".reg .u64 mbar_addr64;\\n\\t"
    "cvta.to.shared.u64 mbar_addr64, %1;\\n\\t"
    "cvt.u32.u64 %0, mbar_addr64;\\n\\t"
    "}}\\n"
    : "=r"(mbar_addr) : "l"(bar_ptr));
  mbar_addr &= 0xFEFFFFFFu;
  asm volatile(
    "cp.async.bulk.tensor.5d.cta_group::2.shared::cluster.global"
    ".mbarrier::complete_tx::bytes.L2::cache_hint "
    "[%0], [%1, {{%3, %4, %5, %6, %7}}], [%2], %8;\\n"
    :
    : "r"(smem_addr), "l"(tensor_map_addr), "r"(mbar_addr),
      "r"(coord0), "r"(coord1), "r"(coord2), "r"(coord3), "r"(coord4),
      "l"(cache_hint)
    : "memory");
}}
"""
    return T.cuda.func_call(
        func_name,
        dst_ptr,
        bar_ptr,
        tensor_map_ptr,
        coord0,
        coord1,
        coord2,
        coord3,
        coord4,
        cache_hint,
        source_code=source_code,
        return_type="void",
    )


def _ld_shared_u32(src_ptr: Any) -> Any:
    func_name = "sparse_flashmla_small_topk_head128_ld_shared_u32"
    source_code = f"""
__device__ __forceinline__ unsigned int {func_name}(const void* src_ptr) {{
  unsigned int smem_addr;
  asm volatile(
    "{{\\n\\t"
    ".reg .u64 smem_addr64;\\n\\t"
    "cvta.to.shared.u64 smem_addr64, %1;\\n\\t"
    "cvt.u32.u64 %0, smem_addr64;\\n\\t"
    "}}\\n"
    : "=r"(smem_addr) : "l"(src_ptr));
  unsigned int val;
  asm volatile("ld.shared.u32 %0, [%1];\\n" : "=r"(val) : "r"(smem_addr));
  return val;
}}
"""
    return T.cuda.func_call(func_name, src_ptr, source_code=source_code, return_type="uint32")


def _ldg_256_indices_policy(
    dst0: Any,
    dst1: Any,
    dst2: Any,
    dst3: Any,
    dst4: Any,
    dst5: Any,
    dst6: Any,
    dst7: Any,
    src_ptr: Any,
    *,
    func_name: str,
    l2_policy: str,
) -> Any:
    source_code = f"""
__device__ __forceinline__ void {func_name}(
    int* dst0, int* dst1, int* dst2, int* dst3,
    int* dst4, int* dst5, int* dst6, int* dst7, const int* src_ptr) {{
  int raw0, raw1, raw2, raw3, raw4, raw5, raw6, raw7;
  asm volatile(
    "ld.global.nc.L1::no_allocate.L2::{l2_policy}.L2::256B.v8.s32 "
    "{{%0, %1, %2, %3, %4, %5, %6, %7}}, [%8];\\n"
    : "=r"(raw0), "=r"(raw1), "=r"(raw2), "=r"(raw3),
      "=r"(raw4), "=r"(raw5), "=r"(raw6), "=r"(raw7)
    : "l"(src_ptr));
  *dst0 = raw0;
  *dst1 = raw1;
  *dst2 = raw2;
  *dst3 = raw3;
  *dst4 = raw4;
  *dst5 = raw5;
  *dst6 = raw6;
  *dst7 = raw7;
}}
"""
    return T.cuda.func_call(
        func_name,
        dst0,
        dst1,
        dst2,
        dst3,
        dst4,
        dst5,
        dst6,
        dst7,
        src_ptr,
        source_code=source_code,
        return_type="void",
    )


def _ldg_256_indices_evict_first(
    dst0: Any,
    dst1: Any,
    dst2: Any,
    dst3: Any,
    dst4: Any,
    dst5: Any,
    dst6: Any,
    dst7: Any,
    src_ptr: Any,
) -> Any:
    return _ldg_256_indices_policy(
        dst0,
        dst1,
        dst2,
        dst3,
        dst4,
        dst5,
        dst6,
        dst7,
        src_ptr,
        func_name="sparse_flashmla_small_topk_head128_ldg_256_indices_evict_first",
        l2_policy="evict_first",
    )


def _ldg_256_indices_evict_normal(
    dst0: Any,
    dst1: Any,
    dst2: Any,
    dst3: Any,
    dst4: Any,
    dst5: Any,
    dst6: Any,
    dst7: Any,
    src_ptr: Any,
) -> Any:
    return _ldg_256_indices_policy(
        dst0,
        dst1,
        dst2,
        dst3,
        dst4,
        dst5,
        dst6,
        dst7,
        src_ptr,
        func_name="sparse_flashmla_small_topk_head128_ldg_256_indices_evict_normal",
        l2_policy="evict_normal",
    )


def _trigger_programmatic_launch_completion() -> Any:
    func_name = "sparse_flashmla_small_topk_head128_trigger_programmatic_launch_completion"
    source_code = f"""
__device__ __forceinline__ void {func_name}() {{
  asm volatile("griddepcontrol.launch_dependents;":::);
}}
"""
    return T.cuda.func_call(func_name, source_code=source_code, return_type="void")


def _clc_query_cancel_acquire_x(response_ptr: Any) -> Any:
    func_name = "sparse_flashmla_small_topk_head128_clc_query_cancel_acquire_x"
    source_code = f"""
__device__ __forceinline__ unsigned int {func_name}(void* response_ptr) {{
  unsigned int response_addr = (unsigned int)__cvta_generic_to_shared(response_ptr);
  unsigned int first_ctaid_x;
  asm volatile(
    "{{\\n\\t"
    ".reg .pred canceled;\\n\\t"
    ".reg .b128 response;\\n\\t"
    "ld.acquire.cta.shared.b128 response, [%1];\\n\\t"
    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 canceled, response;\\n\\t"
    "mov.u32 %0, 0xffffffff;\\n\\t"
    "@canceled clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, response;\\n\\t"
    "}}\\n"
    : "=r"(first_ctaid_x) : "r"(response_addr) : "memory");
  return first_ctaid_x;
}}
"""
    return T.cuda.func_call(func_name, response_ptr, source_code=source_code, return_type="uint32")


def _mbarrier_arrive_remote_unpred(bar_ptr: Any, cta_id: Any) -> Any:
    func_name = "sparse_flashmla_small_topk_head128_mbarrier_arrive_remote_unpred"
    source_code = f"""
__device__ __forceinline__ void {func_name}(void* bar_ptr, unsigned int cta_id) {{
  unsigned int smem_addr;
  asm volatile(
    "{{\\n\\t"
    ".reg .u64 smem_addr64;\\n\\t"
    "cvta.to.shared.u64 smem_addr64, %1;\\n\\t"
    "cvt.u32.u64 %0, smem_addr64;\\n\\t"
    "}}\\n"
    : "=r"(smem_addr) : "l"(bar_ptr));
  asm volatile(
    "{{\\n\\t"
    ".reg .b32 remAddr32;\\n\\t"
    "mapa.shared::cluster.u32 remAddr32, %0, %1;\\n\\t"
    "mbarrier.arrive.shared::cluster.b64 _, [remAddr32];\\n\\t"
    "}}\\n"
    :: "r"(smem_addr), "r"(cta_id) : "memory");
}}
"""
    return T.cuda.func_call(func_name, bar_ptr, cta_id, source_code=source_code, return_type="void")


def _clc_try_cancel_multicast(response_ptr: Any, bar_ptr: Any) -> Any:
    func_name = "sparse_flashmla_small_topk_head128_clc_try_cancel_multicast"
    source_code = f"""
__device__ __forceinline__ void {func_name}(void* response_ptr, void* bar_ptr) {{
  unsigned int response_addr;
  unsigned int mbarrier_addr;
  asm volatile(
    "{{\\n\\t"
    ".reg .u64 response_addr64;\\n\\t"
    "cvta.to.shared.u64 response_addr64, %1;\\n\\t"
    "cvt.u32.u64 %0, response_addr64;\\n\\t"
    "}}\\n"
    : "=r"(response_addr) : "l"(response_ptr));
  asm volatile(
    "{{\\n\\t"
    ".reg .u64 mbarrier_addr64;\\n\\t"
    "cvta.to.shared.u64 mbarrier_addr64, %1;\\n\\t"
    "cvt.u32.u64 %0, mbarrier_addr64;\\n\\t"
    "}}\\n"
    : "=r"(mbarrier_addr) : "l"(bar_ptr));
  asm volatile(
    "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes"
    ".multicast::cluster::all.b128 [%0], [%1];\\n"
    :: "r"(response_addr), "r"(mbarrier_addr));
}}
"""
    return T.cuda.func_call(
        func_name, response_ptr, bar_ptr, source_code=source_code, return_type="void"
    )


@T.jit
def _kernel(
    q: T.Buffer((s_q, h_q, d_qk), "bfloat16"),
    kv: T.Buffer((s_kv * stride_kv_s_kv,), "bfloat16"),
    indices: T.Buffer((s_q * stride_indices_s_q,), "int32"),
    attn_sink: T.handle("float32"),
    topk_length: T.handle("int32"),
    out: T.Buffer((s_q, h_q, D_V), "bfloat16"),
    max_logits: T.Buffer((s_q, h_q), "float32"),
    lse: T.Buffer((s_q, h_q), "float32"),
    tensor_map_q: T.TensorMap(),
    tensor_map_kv: T.TensorMap(),
    tensor_map_o: T.TensorMap(),
    *,
    s_q: T.constexpr,
    s_kv: T.constexpr,
    topk: T.constexpr,
    d_qk: T.constexpr,
    h_q: T.constexpr,
    stride_kv_s_kv: T.constexpr,
    stride_indices_s_q: T.constexpr,
    have_attn_sink: T.constexpr,
    have_topk_length: T.constexpr,
    sm_scale_div_log2: T.constexpr,
):
    T.device_entry()
    T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})
    # CUDA_TRANSCRIBE_START: phase1.cuh:24, scoped to KernelTemplate<Prefill, 512>.
    block_idx = T.cta_id([2 * s_q])
    T.cta_id_in_cluster([2])
    cta_idx: T.let = block_idx % 2
    thread_idx = T.thread_id([NUM_THREADS])
    warp_idx: T.let = _canonical_warp_idx_sync()
    lane_idx: T.let = thread_idx % 32
    warpgroup_idx: T.let = _shfl_sync_i32(thread_idx // 128)
    idx_in_warpgroup: T.let = thread_idx % 128

    if thread_idx == 0:
        T.evaluate(T.ptx.prefetch_tensormap(T.address_of(tensor_map_q)))
        T.evaluate(T.ptx.prefetch_tensormap(T.address_of(tensor_map_o)))
        T.evaluate(T.ptx.prefetch_tensormap(T.address_of(tensor_map_kv)))

    pool = T.SMEMPool()
    q_smem = pool.alloc_mma((B_H // 2, D_QK), "bfloat16")
    k_smem = pool.alloc_mma((NUM_K_BUFS * B_TOPK, D_QK // 2), "bfloat16")
    s_smem = pool.alloc(((B_H // 2) * B_TOPK,), "bfloat16")
    p_exchange = pool.alloc((4, (B_H // 2 // 2) * (B_TOPK // 2)), "uint32")
    rowwise_max_buf = pool.alloc((128,), "float32")
    rowwise_li_buf = pool.alloc((128,), "float32")
    is_k_valid = pool.alloc((NUM_INDEX_BUFS, B_TOPK // 8), "int8", align=16)

    bar_sQ_full = TMABar(pool, 1, leader=True)
    bar_tQ_empty = TCGen05Bar(pool, 1, leader=True)
    bar_tQ_full = TCGen05Bar(pool, 1, leader=True)
    bar_tOut_full = TCGen05Bar(pool, 1, leader=True)
    bar_tOut_empty = MBarrier(pool, 1, leader=True)
    bar_KV_full = TMABar(pool, NUM_K_BUFS, leader=True)
    bar_KV_empty = TCGen05Bar(pool, NUM_K_BUFS, leader=True)
    bar_P_empty = MBarrier(pool, 1, leader=True)
    bar_QK_done = TCGen05Bar(pool, 1, leader=True)
    bar_SV_done = TCGen05Bar(pool, 1, leader=True)
    bar_S_O_full = MBarrier(pool, 1, leader=True)
    bar_li_full = MBarrier(pool, 1, leader=True)
    bar_li_empty = MBarrier(pool, 1, leader=True)
    bar_valid_coord_scales_full = MBarrier(pool, NUM_INDEX_BUFS, leader=True)
    bar_valid_coord_scales_empty = MBarrier(pool, NUM_INDEX_BUFS, leader=True)
    bar_clc_full = TMABar(pool, 1, leader=True)
    bar_clc_empty = MBarrier(pool, 1, leader=True)
    clc_response = pool.alloc((4,), "uint32", align=16)
    tmem_start_addr = pool.alloc((1,), "uint32", align=4)
    pool.commit()

    if warp_idx == 1:
        if T.ptx.elect_sync():
            bar_sQ_full.init(1)
            bar_tQ_empty.init(1)
            bar_tQ_full.init(1)
            bar_tOut_full.init(1)
            bar_tOut_empty.init(256)
            bar_P_empty.init(256)
            bar_QK_done.init(1)
            bar_SV_done.init(1)
            bar_S_O_full.init(256)
            bar_li_full.init(B_H // 2)
            bar_li_empty.init(128)
            bar_clc_full.init(1)
            bar_clc_empty.init(NUM_WORKER_THREADS)
            T.ptx.fence.mbarrier_init()
    elif warp_idx == 2:
        T.ptx.tcgen05.alloc(T.address_of(tmem_start_addr[0]), n_cols=512, cta_group=2)
        T.cuda.trap_when_assert_failed(tmem_start_addr[0] == T.uint32(0))
        T.ptx.tcgen05.relinquish_alloc_permit(cta_group=2)
    elif warp_idx == 3:
        if T.ptx.elect_sync():
            for init_stage in T.unroll(NUM_K_BUFS):
                T.ptx.mbarrier.init(bar_KV_full.ptr_to([init_stage]), 1)
                T.ptx.mbarrier.init(bar_KV_empty.ptr_to([init_stage]), 1)
            for init_stage in T.unroll(NUM_INDEX_BUFS):
                T.ptx.mbarrier.init(bar_valid_coord_scales_full.ptr_to([init_stage]), B_TOPK // 8)
                T.ptx.mbarrier.init(bar_valid_coord_scales_empty.ptr_to([init_stage]), 128)
            T.ptx.fence.mbarrier_init()

    T.cuda.cluster_sync()

    if warpgroup_idx == 0:
        # CUDA phase1.cuh:192-396. Q fetching and O write-back warpgroup.
        T.ptx.setmaxnreg(True, 160)

        @T.inline
        def issue_q_copy(q_s_q_idx, q_outer_loop_phase):
            if warp_idx == 0:
                if T.ptx.elect_sync():
                    T.ptx.cp_async.bulk.wait_group(0)
                    T.evaluate(
                        _tma_5d_cta_group2_nosplit(
                            q_smem.ptr_to([0, 0]),
                            bar_sQ_full.ptr_to([0]),
                            T.address_of(tensor_map_q),
                            T.uint32(0),
                            cta_idx * (B_H // 2),
                            T.uint32(0),
                            T.uint32(0),
                            q_s_q_idx,
                            T.uint64(0x12F0000000000000),
                        )
                    )
                    if cta_idx == 0:
                        bar_sQ_full.arrive(0, tx_count=B_H * D_QK * BF16_BYTES)
                        bar_sQ_full.wait(0, q_outer_loop_phase)
                        bar_tQ_empty.wait(0, q_outer_loop_phase ^ 1)
                        T.ptx.tcgen05.fence.after_thread_sync()
                        q_desc: T.uint64
                        T.ptx.tcgen05.encode_matrix_descriptor(
                            T.address_of(q_desc),
                            q_smem.ptr_to([0, 0]),
                            ldo=K_MAJOR_SWIZZLED_DESC_LDO,
                            sdo=Q_DESC_SDO,
                            swizzle=3,
                        )
                        for tile_idx in T.unroll(D_QK // 64 // 2):
                            for subtile_idx in T.unroll(4):
                                T.ptx.tcgen05.cp(
                                    T.uint32(TMEM_COL_Q + tile_idx * 32 + subtile_idx * 8),
                                    q_desc + T.uint64(tile_idx * 1024 + subtile_idx * 2),
                                    shape="128x256b",
                                    cta_group=2,
                                )
                        bar_tQ_full.arrive(0, cta_group=2, cta_mask=3)

        @T.inline
        def perform_o_copy_out(o_s_q_idx, o_outer_loop_phase, is_last_o: T.constexpr):
            bar_li_full.wait(0, o_outer_loop_phase)
            output_scale: T.let = rowwise_li_buf[idx_in_warpgroup % 64]
            output_scale_pair: T.let = T.cuda.make_float2(output_scale, output_scale)
            bar_li_empty.arrive(0)

            bar_tOut_full.wait(0, o_outer_loop_phase)
            if is_last_o:
                if T.ptx.elect_sync():
                    T.evaluate(_trigger_programmatic_launch_completion())

            o_epi = T.alloc_local((B_EPI,), "float32")
            for epi_k in T.unroll((D_V // 2) // B_EPI):
                T.ptx.tcgen05.ld(
                    T.uint32(0),
                    o_epi[0],
                    o_epi[1],
                    o_epi[2],
                    o_epi[3],
                    o_epi[4],
                    o_epi[5],
                    o_epi[6],
                    o_epi[7],
                    o_epi[8],
                    o_epi[9],
                    o_epi[10],
                    o_epi[11],
                    o_epi[12],
                    o_epi[13],
                    o_epi[14],
                    o_epi[15],
                    o_epi[16],
                    o_epi[17],
                    o_epi[18],
                    o_epi[19],
                    o_epi[20],
                    o_epi[21],
                    o_epi[22],
                    o_epi[23],
                    o_epi[24],
                    o_epi[25],
                    o_epi[26],
                    o_epi[27],
                    o_epi[28],
                    o_epi[29],
                    o_epi[30],
                    o_epi[31],
                    o_epi[32],
                    o_epi[33],
                    o_epi[34],
                    o_epi[35],
                    o_epi[36],
                    o_epi[37],
                    o_epi[38],
                    o_epi[39],
                    o_epi[40],
                    o_epi[41],
                    o_epi[42],
                    o_epi[43],
                    o_epi[44],
                    o_epi[45],
                    o_epi[46],
                    o_epi[47],
                    o_epi[48],
                    o_epi[49],
                    o_epi[50],
                    o_epi[51],
                    o_epi[52],
                    o_epi[53],
                    o_epi[54],
                    o_epi[55],
                    o_epi[56],
                    o_epi[57],
                    o_epi[58],
                    o_epi[59],
                    o_epi[60],
                    o_epi[61],
                    o_epi[62],
                    o_epi[63],
                    shape="32x32b",
                    num=B_EPI,
                    col=TMEM_COL_O + epi_k * B_EPI,
                )
                T.ptx.tcgen05.wait.ld()
                if epi_k == 0:
                    if is_last_o:
                        bar_tQ_full.wait(0, o_outer_loop_phase)
                    else:
                        bar_tQ_full.wait(0, o_outer_loop_phase ^ 1)
                if epi_k == ((D_V // 2) // B_EPI) - 1:
                    bar_tOut_empty.arrive(0, cta_id=T.uint32(0))
                for o_i in T.unroll(B_EPI // 8):
                    o_epi_bf16 = T.alloc_local((4,), "uint32")
                    for o_j in T.unroll(4):
                        o_pair_idx: T.let = o_i * 8 + o_j * 2
                        o_pair: T.let = T.cuda.make_float2(o_epi[o_pair_idx], o_epi[o_pair_idx + 1])
                        o_scaled_pair: T.let = _mul_f32x2(o_pair, output_scale_pair)
                        o_epi_bf16[o_j] = T.cuda.float22bfloat162_rn(
                            T.cuda.float2_x(o_scaled_pair), T.cuda.float2_y(o_scaled_pair)
                        )
                    o_base_col: T.let = (idx_in_warpgroup // 64) * (D_V // 2) + epi_k * B_EPI
                    T.evaluate(
                        T.ptx.st(
                            q_smem.ptr_to([idx_in_warpgroup % 64, o_base_col + o_i * 8]),
                            o_epi_bf16[0],
                            o_epi_bf16[1],
                            o_epi_bf16[2],
                            o_epi_bf16[3],
                            space="shared",
                            ptx_type="u32",
                            vec="v4",
                        )
                    )

            T.ptx.fence.proxy_async("shared::cta")
            T.ptx.bar.sync(NAMED_BARRIER_WG0_SYNC, 128)
            if warp_idx == 0:
                if T.ptx.elect_sync():
                    T.evaluate(
                        T.ptx.cp_async.bulk.tensor.s2g(
                            5,
                            q_smem.ptr_to([0, 0]),
                            T.address_of(tensor_map_o),
                            "",
                            T.uint32(0),
                            cta_idx * (B_H // 2),
                            T.uint32(0),
                            o_s_q_idx,
                            T.uint32(0),
                        )
                    )
                    T.ptx.cp_async.bulk.commit_group()

        wg0_job_valid = T.local_scalar("int32")
        wg0_job_valid = 1
        wg0_job_block_idx = T.local_scalar("int32")
        wg0_job_block_idx = block_idx
        wg0_outer_loop_phase = T.local_scalar("int32")
        wg0_outer_loop_phase = 0
        last_valid = T.local_scalar("int32")
        last_valid = 0
        last_s_q_idx = T.local_scalar("int32")
        last_s_q_idx = 0
        last_outer_loop_phase = T.local_scalar("int32")
        last_outer_loop_phase = 0

        while wg0_job_valid != 0:
            wg0_s_q_idx: T.let = wg0_job_block_idx // 2
            issue_q_copy(wg0_s_q_idx, wg0_outer_loop_phase)

            if last_valid != 0:
                perform_o_copy_out(last_s_q_idx, last_outer_loop_phase, False)
            else:
                bar_tQ_full.wait(0, wg0_outer_loop_phase)
            last_valid = 1
            last_s_q_idx = wg0_s_q_idx
            last_outer_loop_phase = wg0_outer_loop_phase

            bar_clc_full.wait(0, wg0_outer_loop_phase)
            wg0_next_job: T.let = _clc_query_cancel_acquire_x(T.address_of(clc_response[0]))
            T.evaluate(_mbarrier_arrive_remote_unpred(bar_clc_empty.ptr_to([0]), T.uint32(0)))
            if wg0_next_job == T.uint32(0xFFFFFFFF):
                wg0_job_valid = 0
            else:
                wg0_job_block_idx = T.cast(wg0_next_job, "int32")
            wg0_outer_loop_phase = wg0_outer_loop_phase ^ 1

        if last_valid != 0:
            if warp_idx == 0:
                if T.ptx.elect_sync():
                    T.ptx.cp_async.bulk.wait_group(0)
            T.ptx.bar.sync(NAMED_BARRIER_WG0_SYNC, 128)
            perform_o_copy_out(last_s_q_idx, last_outer_loop_phase, True)

        if warp_idx == 0:
            T.ptx.tcgen05.dealloc(T.uint32(0), n_cols=512, cta_group=2)

    elif warpgroup_idx == 1:
        # CUDA phase1.cuh:397-451. Prefill KV gather producer.
        T.ptx.setmaxnreg(False, 80)
        # Source uses canonical_warp_idx() here, not canonical_warp_idx_sync().
        wg1_warp_idx: T.let = thread_idx // 32 - 4
        if T.ptx.elect_sync():
            wg1_job_valid = T.local_scalar("int32")
            wg1_job_valid = 1
            wg1_job_block_idx = T.local_scalar("int32")
            wg1_job_block_idx = block_idx
            wg1_outer_loop_phase = T.local_scalar("int32")
            wg1_outer_loop_phase = 0
            wg1_rs = T.local_scalar("int32")
            wg1_rs = 0
            while wg1_job_valid != 0:
                wg1_s_q_idx: T.let = wg1_job_block_idx // 2
                wg1_topk_len: T.let = (
                    _ldg_i32_at(topk_length, wg1_s_q_idx) if have_topk_length else topk
                )
                wg1_num_k_blocks: T.let = T.max((wg1_topk_len + B_TOPK - 1) // B_TOPK, 1)
                wg1_g_indices_base: T.let = wg1_s_q_idx * stride_indices_s_q

                for k in T.serial(0, wg1_num_k_blocks, unroll=False):
                    k_buf_idx: T.let = wg1_rs % NUM_K_BUFS
                    k_bar_phase: T.let = (wg1_rs // NUM_K_BUFS) & 1
                    cur_indices = T.alloc_local((WG1_ROWS_PER_WARP,), "int32")
                    for local_row in T.unroll(WG1_ROWS_PER_WARP // 8):
                        row: T.let = local_row * (4 * 8) + wg1_warp_idx * 8
                        T.evaluate(
                            _ldg_256_indices_evict_first(
                                cur_indices.ptr_to([local_row * 8 + 0]),
                                cur_indices.ptr_to([local_row * 8 + 1]),
                                cur_indices.ptr_to([local_row * 8 + 2]),
                                cur_indices.ptr_to([local_row * 8 + 3]),
                                cur_indices.ptr_to([local_row * 8 + 4]),
                                cur_indices.ptr_to([local_row * 8 + 5]),
                                cur_indices.ptr_to([local_row * 8 + 6]),
                                cur_indices.ptr_to([local_row * 8 + 7]),
                                indices.ptr_to([wg1_g_indices_base + k * B_TOPK + row]),
                            )
                        )
                    bar_KV_empty.wait(k_buf_idx, k_bar_phase ^ 1)
                    for local_row in T.unroll(WG1_ROWS_PER_WARP // 4):
                        row: T.let = (
                            wg1_warp_idx * 8 + (local_row // 2) * (4 * 8) + (local_row % 2) * 4
                        )
                        for local_col in T.unroll((D_QK // 64) // 2):
                            raw_k_offset: T.let = (
                                k_buf_idx * B_TOPK * (D_QK // 2)
                                + row * 64
                                + local_col * 64 * B_TOPK
                            )
                            T.evaluate(
                                _tma_gather4_kv_cta_group2(
                                    k_smem.access_ptr("w", offset=raw_k_offset),
                                    bar_KV_full.ptr_to([k_buf_idx]),
                                    T.address_of(tensor_map_kv),
                                    local_col * 64 + cta_idx * (D_QK // 2),
                                    cur_indices[local_row * 4 + 0],
                                    cur_indices[local_row * 4 + 1],
                                    cur_indices[local_row * 4 + 2],
                                    cur_indices[local_row * 4 + 3],
                                    T.uint64(0x14F0000000000000),
                                )
                            )
                    wg1_rs = wg1_rs + 1

                bar_clc_full.wait(0, wg1_outer_loop_phase)
                wg1_next_job: T.let = _clc_query_cancel_acquire_x(T.address_of(clc_response[0]))
                T.evaluate(_mbarrier_arrive_remote_unpred(bar_clc_empty.ptr_to([0]), T.uint32(0)))
                if wg1_next_job == T.uint32(0xFFFFFFFF):
                    wg1_job_valid = 0
                else:
                    wg1_job_block_idx = T.cast(wg1_next_job, "int32")
                wg1_outer_loop_phase = wg1_outer_loop_phase ^ 1

    elif warpgroup_idx == 2:
        # CUDA phase1.cuh:533-787. UMMA, valid-mask loading, and CLC producer.
        T.ptx.setmaxnreg(False, 80)

        if (warp_idx == 8) & (cta_idx == 0):
            if T.ptx.elect_sync():
                desc_i_p: T.uint32
                desc_i_o: T.uint32
                T.ptx.tcgen05.encode_instr_descriptor(
                    T.address_of(desc_i_p),
                    d_dtype="float32",
                    a_dtype="bfloat16",
                    b_dtype="bfloat16",
                    M=B_H,
                    N=B_TOPK * 2,
                    K=16,
                    trans_a=False,
                    trans_b=False,
                    n_cta_groups=2,
                )
                T.ptx.tcgen05.encode_instr_descriptor(
                    T.address_of(desc_i_o),
                    d_dtype="float32",
                    a_dtype="bfloat16",
                    b_dtype="bfloat16",
                    M=B_H,
                    N=256,
                    K=16,
                    trans_a=False,
                    trans_b=True,
                    n_cta_groups=2,
                )
                umma_job_valid = T.local_scalar("int32")
                umma_job_valid = 1
                umma_job_block_idx = T.local_scalar("int32")
                umma_job_block_idx = block_idx
                umma_outer_loop_phase = T.local_scalar("int32")
                umma_outer_loop_phase = 0
                umma_rs = T.local_scalar("int32")
                umma_rs = 0
                while umma_job_valid != 0:
                    umma_s_q_idx: T.let = umma_job_block_idx // 2
                    umma_topk_len: T.let = (
                        _ldg_i32_at(topk_length, umma_s_q_idx) if have_topk_length else topk
                    )
                    umma_num_k_blocks: T.let = T.max((umma_topk_len + B_TOPK - 1) // B_TOPK, 1)
                    bar_tQ_full.wait(0, umma_outer_loop_phase)

                    for k in T.serial(0, umma_num_k_blocks + 1, unroll=False):
                        if k < umma_num_k_blocks:
                            k_buf_idx: T.let = umma_rs % NUM_K_BUFS
                            k_bar_phase: T.let = (umma_rs // NUM_K_BUFS) & 1
                            p_bar_phase: T.let = umma_rs & 1
                            bar_P_empty.wait(0, p_bar_phase ^ 1)
                            bar_KV_full.arrive(k_buf_idx, tx_count=B_TOPK * D_QK * BF16_BYTES)
                            bar_KV_full.wait(k_buf_idx, k_bar_phase)
                            T.ptx.tcgen05.fence.after_thread_sync()
                            qk_accumulate = T.alloc_local((1,), "uint32")
                            qk_accumulate[0] = T.uint32(0)
                            k_desc_base: T.uint64
                            T.ptx.tcgen05.encode_matrix_descriptor(
                                T.address_of(k_desc_base),
                                k_smem.access_ptr("r", offset=k_buf_idx * B_TOPK * (D_QK // 2)),
                                ldo=K_MAJOR_SWIZZLED_DESC_LDO,
                                sdo=K_DESC_SDO,
                                swizzle=3,
                            )
                            for qk_k in T.unroll((D_QK // 2) // 16):
                                k_desc_col: T.let = qk_k * 16
                                k_desc: T.let = k_desc_base + T.uint64(
                                    (k_desc_col // 64) * (64 * B_TOPK // 8) + (k_desc_col % 64) // 8
                                )
                                T.evaluate(
                                    _tcgen05_mma_ws_ts_2cta(
                                        T.uint32(TMEM_COL_P),
                                        T.uint32(TMEM_COL_Q + qk_k * 8),
                                        k_desc,
                                        desc_i_p,
                                        qk_accumulate[0],
                                    )
                                )
                                qk_accumulate[0] = T.uint32(1)
                            bar_QK_done.arrive(0, cta_group=2, cta_mask=3)
                            if k == umma_num_k_blocks - 1:
                                T.ptx.tcgen05.commit(
                                    bar_tQ_empty.ptr_to([0]), cta_group=2, cta_mask=0
                                )

                        if k > 0:
                            prev_k: T.let = k - 1
                            prev_rs: T.let = umma_rs - 1
                            prev_buf: T.let = prev_rs % NUM_K_BUFS
                            prev_s_o_phase: T.let = prev_rs & 1
                            bar_S_O_full.wait(0, prev_s_o_phase)
                            if prev_k == 0:
                                bar_tOut_empty.wait(0, umma_outer_loop_phase ^ 1)
                            T.ptx.tcgen05.fence.after_thread_sync()
                            o_accumulate = T.alloc_local((1,), "uint32")
                            o_accumulate[0] = T.if_then_else(prev_k == 0, T.uint32(0), T.uint32(1))
                            s_desc_base: T.uint64
                            v_desc_base: T.uint64
                            T.ptx.tcgen05.encode_matrix_descriptor(
                                T.address_of(s_desc_base),
                                s_smem.ptr_to([0]),
                                ldo=S_DESC_LDO,
                                sdo=S_DESC_SDO,
                                swizzle=0,
                            )
                            T.ptx.tcgen05.encode_matrix_descriptor(
                                T.address_of(v_desc_base),
                                k_smem.access_ptr("r", offset=prev_buf * B_TOPK * (D_QK // 2)),
                                ldo=V_DESC_LDO,
                                sdo=V_DESC_SDO,
                                swizzle=3,
                            )
                            for sv_k in T.unroll(B_TOPK // 16):
                                sv_desc_offset: T.let = sv_k * 16 * 64 // 8
                                s_desc: T.let = s_desc_base + T.uint64(sv_desc_offset)
                                v_desc: T.let = v_desc_base + T.uint64(sv_desc_offset)
                                T.evaluate(
                                    _tcgen05_mma_ws_ss_2cta(
                                        T.uint32(TMEM_COL_O),
                                        s_desc,
                                        v_desc,
                                        desc_i_o,
                                        o_accumulate[0],
                                    )
                                )
                                T.evaluate(
                                    _tcgen05_mma_ws_ss_2cta(
                                        T.uint32(TMEM_COL_O + 128),
                                        s_desc,
                                        v_desc + T.uint64(1024),
                                        desc_i_o,
                                        o_accumulate[0],
                                    )
                                )
                                o_accumulate[0] = T.uint32(1)
                            bar_SV_done.arrive(0, cta_group=2, cta_mask=3)
                            bar_KV_empty.arrive(prev_buf, cta_group=2, cta_mask=3)

                        if k != umma_num_k_blocks:
                            umma_rs = umma_rs + 1

                    T.ptx.tcgen05.fence.before_thread_sync()
                    bar_tOut_full.arrive(0, cta_group=2, cta_mask=3)

                    bar_clc_full.wait(0, umma_outer_loop_phase)
                    umma_next_job: T.let = _clc_query_cancel_acquire_x(
                        T.address_of(clc_response[0])
                    )
                    T.evaluate(
                        _mbarrier_arrive_remote_unpred(bar_clc_empty.ptr_to([0]), T.uint32(0))
                    )
                    if umma_next_job == T.uint32(0xFFFFFFFF):
                        umma_job_valid = 0
                    else:
                        umma_job_block_idx = T.cast(umma_next_job, "int32")
                    umma_outer_loop_phase = umma_outer_loop_phase ^ 1

        elif warp_idx == 9:
            if lane_idx < B_TOPK // 8:
                lane_indices = T.alloc_local((8,), "int32")
                valid_job_valid = T.local_scalar("int32")
                valid_job_valid = 1
                valid_job_block_idx = T.local_scalar("int32")
                valid_job_block_idx = block_idx
                valid_outer_loop_phase = T.local_scalar("int32")
                valid_outer_loop_phase = 0
                valid_rs = T.local_scalar("int32")
                valid_rs = 0
                while valid_job_valid != 0:
                    valid_s_q_idx: T.let = valid_job_block_idx // 2
                    valid_topk_len: T.let = (
                        _ldg_i32_at(topk_length, valid_s_q_idx) if have_topk_length else topk
                    )
                    valid_num_k_blocks: T.let = T.max((valid_topk_len + B_TOPK - 1) // B_TOPK, 1)
                    valid_g_indices_base: T.let = valid_s_q_idx * stride_indices_s_q
                    for k in T.serial(0, valid_num_k_blocks, unroll=False):
                        T.evaluate(
                            _ldg_256_indices_evict_normal(
                                lane_indices.ptr_to([0]),
                                lane_indices.ptr_to([1]),
                                lane_indices.ptr_to([2]),
                                lane_indices.ptr_to([3]),
                                lane_indices.ptr_to([4]),
                                lane_indices.ptr_to([5]),
                                lane_indices.ptr_to([6]),
                                lane_indices.ptr_to([7]),
                                indices.ptr_to([valid_g_indices_base + k * B_TOPK + lane_idx * 8]),
                            )
                        )
                        abs_pos_start: T.let = k * B_TOPK
                        valid0: T.let = (
                            (lane_indices[0] >= 0)
                            & (lane_indices[0] < s_kv)
                            & (abs_pos_start + lane_idx * 8 < valid_topk_len)
                        )
                        valid1: T.let = (
                            (lane_indices[1] >= 0)
                            & (lane_indices[1] < s_kv)
                            & (abs_pos_start + lane_idx * 8 + 1 < valid_topk_len)
                        )
                        valid2: T.let = (
                            (lane_indices[2] >= 0)
                            & (lane_indices[2] < s_kv)
                            & (abs_pos_start + lane_idx * 8 + 2 < valid_topk_len)
                        )
                        valid3: T.let = (
                            (lane_indices[3] >= 0)
                            & (lane_indices[3] < s_kv)
                            & (abs_pos_start + lane_idx * 8 + 3 < valid_topk_len)
                        )
                        valid4: T.let = (
                            (lane_indices[4] >= 0)
                            & (lane_indices[4] < s_kv)
                            & (abs_pos_start + lane_idx * 8 + 4 < valid_topk_len)
                        )
                        valid5: T.let = (
                            (lane_indices[5] >= 0)
                            & (lane_indices[5] < s_kv)
                            & (abs_pos_start + lane_idx * 8 + 5 < valid_topk_len)
                        )
                        valid6: T.let = (
                            (lane_indices[6] >= 0)
                            & (lane_indices[6] < s_kv)
                            & (abs_pos_start + lane_idx * 8 + 6 < valid_topk_len)
                        )
                        valid7: T.let = (
                            (lane_indices[7] >= 0)
                            & (lane_indices[7] < s_kv)
                            & (abs_pos_start + lane_idx * 8 + 7 < valid_topk_len)
                        )
                        mask: T.let = T.cast(
                            T.bitwise_or(
                                T.bitwise_or(
                                    T.bitwise_or(
                                        T.bitwise_or(
                                            T.Select(valid0, T.int32(1), T.int32(0)),
                                            T.Select(valid1, T.int32(2), T.int32(0)),
                                        ),
                                        T.bitwise_or(
                                            T.Select(valid2, T.int32(4), T.int32(0)),
                                            T.Select(valid3, T.int32(8), T.int32(0)),
                                        ),
                                    ),
                                    T.bitwise_or(
                                        T.bitwise_or(
                                            T.Select(valid4, T.int32(16), T.int32(0)),
                                            T.Select(valid5, T.int32(32), T.int32(0)),
                                        ),
                                        T.bitwise_or(
                                            T.Select(valid6, T.int32(64), T.int32(0)),
                                            T.Select(valid7, T.int32(128), T.int32(0)),
                                        ),
                                    ),
                                ),
                                T.int32(0),
                            ),
                            "int8",
                        )
                        index_buf_idx: T.let = valid_rs % NUM_INDEX_BUFS
                        index_bar_phase: T.let = (valid_rs // NUM_INDEX_BUFS) & 1
                        bar_valid_coord_scales_empty.wait(index_buf_idx, index_bar_phase ^ 1)
                        is_k_valid[index_buf_idx, lane_idx] = mask
                        bar_valid_coord_scales_full.arrive(index_buf_idx)
                        valid_rs = valid_rs + 1

                    bar_clc_full.wait(0, valid_outer_loop_phase)
                    valid_next_job: T.let = _clc_query_cancel_acquire_x(
                        T.address_of(clc_response[0])
                    )
                    T.evaluate(
                        _mbarrier_arrive_remote_unpred(bar_clc_empty.ptr_to([0]), T.uint32(0))
                    )
                    if valid_next_job == T.uint32(0xFFFFFFFF):
                        valid_job_valid = 0
                    else:
                        valid_job_block_idx = T.cast(valid_next_job, "int32")
                    valid_outer_loop_phase = valid_outer_loop_phase ^ 1

        elif warp_idx >= 10:
            if T.ptx.elect_sync():
                if warp_idx == 10:
                    clc_job_valid = T.local_scalar("int32")
                    clc_job_valid = 1
                    clc_outer_loop_phase = T.local_scalar("int32")
                    clc_outer_loop_phase = 0
                    while clc_job_valid != 0:
                        if cta_idx == 0:
                            bar_clc_empty.wait(0, clc_outer_loop_phase ^ 1)
                            T.evaluate(
                                _clc_try_cancel_multicast(
                                    T.address_of(clc_response[0]), bar_clc_full.ptr_to([0])
                                )
                            )
                        bar_clc_full.arrive(0, tx_count=16)

                        bar_clc_full.wait(0, clc_outer_loop_phase)
                        clc_next_job: T.let = _clc_query_cancel_acquire_x(
                            T.address_of(clc_response[0])
                        )
                        T.evaluate(
                            _mbarrier_arrive_remote_unpred(bar_clc_empty.ptr_to([0]), T.uint32(0))
                        )
                        if clc_next_job == T.uint32(0xFFFFFFFF):
                            clc_job_valid = 0
                        clc_outer_loop_phase = clc_outer_loop_phase ^ 1

    else:
        # CUDA phase1.cuh:788-921. Scale/exp warpgroup.
        T.ptx.setmaxnreg(True, 160)
        local_warp_idx: T.let = warp_idx - 12
        wg3_job_valid = T.local_scalar("int32")
        wg3_job_valid = 1
        wg3_job_block_idx = T.local_scalar("int32")
        wg3_job_block_idx = block_idx
        wg3_outer_loop_phase = T.local_scalar("int32")
        wg3_outer_loop_phase = 0
        wg3_rs = T.local_scalar("int32")
        wg3_rs = 0
        while wg3_job_valid != 0:
            wg3_s_q_idx: T.let = wg3_job_block_idx // 2
            wg3_topk_len: T.let = (
                _ldg_i32_at(topk_length, wg3_s_q_idx) if have_topk_length else topk
            )
            wg3_num_k_blocks: T.let = T.max((wg3_topk_len + B_TOPK - 1) // B_TOPK, 1)
            mi = T.local_scalar("float32")
            mi = MAX_INIT_VAL
            li = T.local_scalar("float32")
            li = 0.0
            real_mi = T.local_scalar("float32")
            real_mi = T.float32(-float("inf"))
            s_smem_base: T.let = (
                T.if_then_else(local_warp_idx >= 2, (B_H // 2) * (B_TOPK // 2), 0)
                + (idx_in_warpgroup % 64) * 8
            )
            scale_pair: T.let = T.cuda.make_float2(sm_scale_div_log2, sm_scale_div_log2)

            for k in T.serial(0, wg3_num_k_blocks, unroll=False):
                k_buf_idx: T.let = wg3_rs % NUM_K_BUFS
                k_bar_phase: T.let = (wg3_rs // NUM_K_BUFS) & 1
                index_buf_idx: T.let = wg3_rs % NUM_INDEX_BUFS
                index_bar_phase: T.let = (wg3_rs // NUM_INDEX_BUFS) & 1
                bar_valid_coord_scales_full.wait(index_buf_idx, index_bar_phase)
                p = T.alloc_local((WG3_NUM_ELEMS_PER_THREAD,), "uint32")
                p_peer = T.alloc_local((WG3_NUM_ELEMS_PER_THREAD,), "uint32")
                bar_QK_done.wait(0, wg3_rs & 1)
                T.ptx.tcgen05.fence.after_thread_sync()
                if local_warp_idx < 2:
                    T.ptx.tcgen05.ld(
                        T.uint32(0),
                        p[0],
                        p[1],
                        p[2],
                        p[3],
                        p[4],
                        p[5],
                        p[6],
                        p[7],
                        p[8],
                        p[9],
                        p[10],
                        p[11],
                        p[12],
                        p[13],
                        p[14],
                        p[15],
                        p[16],
                        p[17],
                        p[18],
                        p[19],
                        p[20],
                        p[21],
                        p[22],
                        p[23],
                        p[24],
                        p[25],
                        p[26],
                        p[27],
                        p[28],
                        p[29],
                        p[30],
                        p[31],
                        shape="32x32b",
                        num=WG3_NUM_ELEMS_PER_THREAD,
                        col=TMEM_COL_P,
                    )
                    T.ptx.tcgen05.ld(
                        T.uint32(0),
                        p_peer[0],
                        p_peer[1],
                        p_peer[2],
                        p_peer[3],
                        p_peer[4],
                        p_peer[5],
                        p_peer[6],
                        p_peer[7],
                        p_peer[8],
                        p_peer[9],
                        p_peer[10],
                        p_peer[11],
                        p_peer[12],
                        p_peer[13],
                        p_peer[14],
                        p_peer[15],
                        p_peer[16],
                        p_peer[17],
                        p_peer[18],
                        p_peer[19],
                        p_peer[20],
                        p_peer[21],
                        p_peer[22],
                        p_peer[23],
                        p_peer[24],
                        p_peer[25],
                        p_peer[26],
                        p_peer[27],
                        p_peer[28],
                        p_peer[29],
                        p_peer[30],
                        p_peer[31],
                        shape="32x32b",
                        num=WG3_NUM_ELEMS_PER_THREAD,
                        col=TMEM_COL_P + WG3_NUM_ELEMS_PER_THREAD,
                    )
                else:
                    T.ptx.tcgen05.ld(
                        T.uint32(0),
                        p_peer[0],
                        p_peer[1],
                        p_peer[2],
                        p_peer[3],
                        p_peer[4],
                        p_peer[5],
                        p_peer[6],
                        p_peer[7],
                        p_peer[8],
                        p_peer[9],
                        p_peer[10],
                        p_peer[11],
                        p_peer[12],
                        p_peer[13],
                        p_peer[14],
                        p_peer[15],
                        p_peer[16],
                        p_peer[17],
                        p_peer[18],
                        p_peer[19],
                        p_peer[20],
                        p_peer[21],
                        p_peer[22],
                        p_peer[23],
                        p_peer[24],
                        p_peer[25],
                        p_peer[26],
                        p_peer[27],
                        p_peer[28],
                        p_peer[29],
                        p_peer[30],
                        p_peer[31],
                        shape="32x32b",
                        num=WG3_NUM_ELEMS_PER_THREAD,
                        col=TMEM_COL_P,
                    )
                    T.ptx.tcgen05.ld(
                        T.uint32(0),
                        p[0],
                        p[1],
                        p[2],
                        p[3],
                        p[4],
                        p[5],
                        p[6],
                        p[7],
                        p[8],
                        p[9],
                        p[10],
                        p[11],
                        p[12],
                        p[13],
                        p[14],
                        p[15],
                        p[16],
                        p[17],
                        p[18],
                        p[19],
                        p[20],
                        p[21],
                        p[22],
                        p[23],
                        p[24],
                        p[25],
                        p[26],
                        p[27],
                        p[28],
                        p[29],
                        p[30],
                        p[31],
                        shape="32x32b",
                        num=WG3_NUM_ELEMS_PER_THREAD,
                        col=TMEM_COL_P + WG3_NUM_ELEMS_PER_THREAD,
                    )
                T.ptx.tcgen05.wait.ld()
                T.ptx.tcgen05.fence.before_thread_sync()
                bar_P_empty.arrive(0, cta_id=T.uint32(0))

                valid_word_offset: T.let = T.if_then_else(
                    local_warp_idx >= 2, WG3_NUM_ELEMS_PER_THREAD // 8, 0
                )
                is_k_valid_u32: T.let = _ld_shared_u32(
                    is_k_valid.ptr_to([index_buf_idx, valid_word_offset])
                )
                for p_i in T.unroll(WG3_NUM_ELEMS_PER_THREAD):
                    invalid_p_predicate: T.let = T.bitwise_and(
                        T.shift_right(is_k_valid_u32, T.uint32(p_i)), T.uint32(1)
                    ) == T.uint32(0)
                    p[p_i] = T.if_then_else(invalid_p_predicate, T.uint32(0xFF800000), p[p_i])

                for exchange_i in T.unroll(WG3_NUM_ELEMS_PER_THREAD // 4):
                    exchange_offset: T.let = exchange_i * 32 * 4 + lane_idx * 4
                    T.evaluate(
                        T.ptx.st(
                            p_exchange.ptr_to([local_warp_idx ^ 2, exchange_offset]),
                            p_peer[exchange_i * 4 + 0],
                            p_peer[exchange_i * 4 + 1],
                            p_peer[exchange_i * 4 + 2],
                            p_peer[exchange_i * 4 + 3],
                            space="shared",
                            ptx_type="u32",
                            vec="v4",
                        )
                    )
                T.ptx.bar.sync(NAMED_BARRIER_WG2_WARP02_SYNC + (local_warp_idx & 1), 64)
                for exchange_i in T.unroll(WG3_NUM_ELEMS_PER_THREAD // 4):
                    exchange_offset: T.let = exchange_i * 32 * 4 + lane_idx * 4
                    p_exchange_tmp = T.alloc_local((4,), "uint32")
                    T.evaluate(
                        T.ptx.ld(
                            p_exchange.ptr_to([local_warp_idx, exchange_offset]),
                            "uint32",
                            "u32",
                            dst=p_exchange_tmp.ptr_to([0]),
                            space="shared",
                            vec="v4",
                        )
                    )
                    p_pair0: T.let = T.cuda.make_float2(
                        T.cuda.uint_as_float(p[exchange_i * 4]),
                        T.cuda.uint_as_float(p[exchange_i * 4 + 1]),
                    )
                    peer_pair0: T.let = T.cuda.make_float2(
                        T.cuda.uint_as_float(p_exchange_tmp[0]),
                        T.cuda.uint_as_float(p_exchange_tmp[1]),
                    )
                    sum_pair0 = T.alloc_local((1,), "uint64")
                    T.ptx.add_f32x2(sum_pair0.ptr_to([0]), p_pair0, peer_pair0)
                    p[exchange_i * 4] = T.cuda.float_as_uint(T.cuda.float2_x(sum_pair0[0]))
                    p[exchange_i * 4 + 1] = T.cuda.float_as_uint(T.cuda.float2_y(sum_pair0[0]))
                    p_pair1: T.let = T.cuda.make_float2(
                        T.cuda.uint_as_float(p[exchange_i * 4 + 2]),
                        T.cuda.uint_as_float(p[exchange_i * 4 + 3]),
                    )
                    peer_pair1: T.let = T.cuda.make_float2(
                        T.cuda.uint_as_float(p_exchange_tmp[2]),
                        T.cuda.uint_as_float(p_exchange_tmp[3]),
                    )
                    sum_pair1 = T.alloc_local((1,), "uint64")
                    T.ptx.add_f32x2(sum_pair1.ptr_to([0]), p_pair1, peer_pair1)
                    p[exchange_i * 4 + 2] = T.cuda.float_as_uint(T.cuda.float2_x(sum_pair1[0]))
                    p[exchange_i * 4 + 3] = T.cuda.float_as_uint(T.cuda.float2_y(sum_pair1[0]))

                cur_pi_max = T.local_scalar("float32")
                cur_pi_max = T.float32(-float("inf"))
                for p_i in T.unroll(WG3_NUM_ELEMS_PER_THREAD):
                    cur_pi_max = T.max(cur_pi_max, T.cuda.uint_as_float(p[p_i]))
                cur_pi_max = cur_pi_max * sm_scale_div_log2
                rowwise_max_buf[idx_in_warpgroup] = cur_pi_max
                T.ptx.bar.sync(NAMED_BARRIER_WG2_WARP02_SYNC + (local_warp_idx & 1), 64)
                cur_pi_max = T.max(cur_pi_max, rowwise_max_buf[idx_in_warpgroup ^ 64])
                real_mi = T.max(real_mi, cur_pi_max)
                should_scale_o: T.let = (
                    T.ptx.any_sync(T.uint32(0xFFFFFFFF), cur_pi_max - mi > 6.0) != 0
                )
                new_max = T.local_scalar("float32")
                scale_for_old = T.local_scalar("float32")
                if not should_scale_o:
                    scale_for_old = 1.0
                    new_max = mi
                else:
                    new_max = T.max(cur_pi_max, mi)
                    scale_for_old = T.ptx.exp2(mi - new_max)
                mi = new_max

                s_pack = T.alloc_local((WG3_NUM_ELEMS_PER_THREAD // 2,), "uint32")
                cur_sum_pair = T.local_scalar("uint64")
                cur_sum_pair = T.cuda.make_float2(T.float32(0.0), T.float32(0.0))
                neg_new_max_pair: T.let = T.cuda.make_float2(-new_max, -new_max)
                for s_i in T.unroll(WG3_NUM_ELEMS_PER_THREAD // 2):
                    p_pair: T.let = T.cuda.make_float2(
                        T.cuda.uint_as_float(p[s_i * 2]), T.cuda.uint_as_float(p[s_i * 2 + 1])
                    )
                    fma_pair: T.let = _fma_f32x2(p_pair, scale_pair, neg_new_max_pair)
                    s_x: T.let = T.ptx.exp2(T.cuda.float2_x(fma_pair))
                    s_y: T.let = T.ptx.exp2(T.cuda.float2_y(fma_pair))
                    s_pair: T.let = T.cuda.make_float2(s_x, s_y)
                    sum_pair_tmp = T.alloc_local((1,), "uint64")
                    T.ptx.add_f32x2(sum_pair_tmp.ptr_to([0]), cur_sum_pair, s_pair)
                    cur_sum_pair = sum_pair_tmp[0]
                    s_pack[s_i] = T.cuda.float22bfloat162_rn(s_x, s_y)
                cur_sum: T.let = T.cuda.float2_x(cur_sum_pair) + T.cuda.float2_y(cur_sum_pair)
                li_tmp = T.alloc_local((1,), "float32")
                T.ptx.fma_f32(T.address_of(li_tmp[0]), li, scale_for_old, cur_sum)
                li = li_tmp[0]

                bar_SV_done.wait(0, (wg3_rs & 1) ^ 1)
                for s_store_i in T.unroll(WG3_NUM_ELEMS_PER_THREAD // 8):
                    s_store_offset: T.let = s_smem_base + s_store_i * 8 * (B_H // 2)
                    T.evaluate(
                        T.ptx.st(
                            s_smem.ptr_to([s_store_offset]),
                            s_pack[s_store_i * 4 + 0],
                            s_pack[s_store_i * 4 + 1],
                            s_pack[s_store_i * 4 + 2],
                            s_pack[s_store_i * 4 + 3],
                            space="shared",
                            ptx_type="u32",
                            vec="v4",
                        )
                    )

                if (k > 0) & should_scale_o:
                    T.ptx.tcgen05.fence.after_thread_sync()
                    scale_for_old_pair: T.let = T.cuda.make_float2(scale_for_old, scale_for_old)
                    o_rescale = T.alloc_local((32,), "float32")
                    for chunk_idx in T.unroll((D_V // 2) // 32):
                        T.ptx.tcgen05.ld(
                            T.uint32(0),
                            o_rescale[0],
                            o_rescale[1],
                            o_rescale[2],
                            o_rescale[3],
                            o_rescale[4],
                            o_rescale[5],
                            o_rescale[6],
                            o_rescale[7],
                            o_rescale[8],
                            o_rescale[9],
                            o_rescale[10],
                            o_rescale[11],
                            o_rescale[12],
                            o_rescale[13],
                            o_rescale[14],
                            o_rescale[15],
                            o_rescale[16],
                            o_rescale[17],
                            o_rescale[18],
                            o_rescale[19],
                            o_rescale[20],
                            o_rescale[21],
                            o_rescale[22],
                            o_rescale[23],
                            o_rescale[24],
                            o_rescale[25],
                            o_rescale[26],
                            o_rescale[27],
                            o_rescale[28],
                            o_rescale[29],
                            o_rescale[30],
                            o_rescale[31],
                            shape="32x32b",
                            num=32,
                            col=TMEM_COL_O + chunk_idx * 32,
                        )
                        T.ptx.tcgen05.wait.ld()
                        for o_i in T.unroll(16):
                            o_pair: T.let = T.cuda.make_float2(
                                o_rescale[o_i * 2], o_rescale[o_i * 2 + 1]
                            )
                            o_pair_tmp: T.let = _mul_f32x2(o_pair, scale_for_old_pair)
                            o_rescale[o_i * 2] = T.cuda.float2_x(o_pair_tmp)
                            o_rescale[o_i * 2 + 1] = T.cuda.float2_y(o_pair_tmp)
                        T.ptx.tcgen05.st(
                            T.uint32(0),
                            o_rescale[0],
                            o_rescale[1],
                            o_rescale[2],
                            o_rescale[3],
                            o_rescale[4],
                            o_rescale[5],
                            o_rescale[6],
                            o_rescale[7],
                            o_rescale[8],
                            o_rescale[9],
                            o_rescale[10],
                            o_rescale[11],
                            o_rescale[12],
                            o_rescale[13],
                            o_rescale[14],
                            o_rescale[15],
                            o_rescale[16],
                            o_rescale[17],
                            o_rescale[18],
                            o_rescale[19],
                            o_rescale[20],
                            o_rescale[21],
                            o_rescale[22],
                            o_rescale[23],
                            o_rescale[24],
                            o_rescale[25],
                            o_rescale[26],
                            o_rescale[27],
                            o_rescale[28],
                            o_rescale[29],
                            o_rescale[30],
                            o_rescale[31],
                            shape="32x32b",
                            num=32,
                            col=TMEM_COL_O + chunk_idx * 32,
                        )
                        T.ptx.tcgen05.wait.st()
                    T.ptx.tcgen05.fence.before_thread_sync()

                T.ptx.fence.proxy_async("shared::cta")
                bar_S_O_full.arrive(0, cta_id=T.uint32(0))
                bar_valid_coord_scales_empty.arrive(index_buf_idx)
                wg3_rs = wg3_rs + 1

            if real_mi == T.float32(-float("inf")):
                li = 0.0
                mi = T.float32(-float("inf"))

            bar_li_empty.wait(0, wg3_outer_loop_phase ^ 1)
            rowwise_li_buf[idx_in_warpgroup ^ 64] = li
            T.ptx.bar.sync(NAMED_BARRIER_WG2_SYNC, 128)
            li = li + rowwise_li_buf[idx_in_warpgroup]

            if idx_in_warpgroup < B_H // 2:
                head_idx: T.let = cta_idx * (B_H // 2) + idx_in_warpgroup
                attn_sink_log2: T.let = (
                    _ldg_f32_at(attn_sink, head_idx) * LOG_2_E
                    if have_attn_sink
                    else T.float32(-float("inf"))
                )
                output_scale: T.let = _fdividef(
                    T.float32(1.0), li + T.ptx.exp2(attn_sink_log2 - mi)
                )
                rowwise_li_buf[idx_in_warpgroup] = T.if_then_else(li == 0.0, 0.0, output_scale)
                bar_li_full.arrive(0)
                cur_lse = T.local_scalar("float32")
                T.ptx.fma_f32(T.address_of(cur_lse), mi, LN_2, T.log(li))
                cur_lse = T.if_then_else(
                    cur_lse == T.float32(-float("inf")), T.float32(float("inf")), cur_lse
                )
                max_logits[wg3_s_q_idx, head_idx] = real_mi * LN_2
                lse[wg3_s_q_idx, head_idx] = cur_lse

            bar_clc_full.wait(0, wg3_outer_loop_phase)
            wg3_next_job: T.let = _clc_query_cancel_acquire_x(T.address_of(clc_response[0]))
            T.evaluate(_mbarrier_arrive_remote_unpred(bar_clc_empty.ptr_to([0]), T.uint32(0)))
            if wg3_next_job == T.uint32(0xFFFFFFFF):
                wg3_job_valid = 0
            else:
                wg3_job_block_idx = T.cast(wg3_next_job, "int32")
            wg3_outer_loop_phase = wg3_outer_loop_phase ^ 1

    T.cuda.cluster_sync()


def get_kernel(**kwargs: Any):
    cfg = _cfg(**kwargs)
    stride_kv_s_kv = int(kwargs.get("stride_kv_s_kv", cfg.d_qk * cfg.h_kv))
    stride_indices_s_q = int(kwargs.get("stride_indices_s_q", cfg.topk * cfg.h_kv))
    kernel = _kernel.specialize(
        s_q=cfg.s_q,
        s_kv=cfg.s_kv,
        topk=cfg.topk,
        d_qk=cfg.d_qk,
        h_q=cfg.h_q,
        stride_kv_s_kv=stride_kv_s_kv,
        stride_indices_s_q=stride_indices_s_q,
        have_attn_sink=cfg.have_attn_sink,
        have_topk_length=cfg.have_topk_length,
        sm_scale_div_log2=(1.0 / math.sqrt(cfg.d_qk)) * LOG_2_E,
    )
    return kernel.with_attr("tirx.kernel_launch_params", list(SMALL_TOPK_HEAD128_LAUNCH_PARAM_TAGS))


def run_test(**kwargs: Any) -> None:
    if not _IMPLEMENTATION_COMPLETE:
        raise SkipTest("sparse FlashMLA head128 small-topk phase1 transcription is not complete")
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for sparse FlashMLA head128 small-topk phase1")

    from tirx_kernels.runner import compile_kernel

    case = prepare_data(**kwargs)
    cfg: SparseFlashMLAPrefillHead128SmallTopKConfig = case["config"]
    if not case["dispatch_reason"].startswith("small_topk:"):
        raise SkipTest(case["dispatch_reason"])
    prim_func = get_kernel(**kwargs)
    ex = compile_kernel(prim_func)
    launches = _build_tirx_launches(case)
    _run_tirx_launches(ex, launches, output_case=case)
    torch.cuda.synchronize()
    ref_out, ref_max_logits, ref_lse = _reference_sparse_prefill(case)
    torch.testing.assert_close(case["out"], ref_out, rtol=4.01 / 128, atol=5e-3)
    torch.testing.assert_close(case["max_logits"], ref_max_logits, rtol=2.01 / 65536, atol=1e-6)
    torch.testing.assert_close(case["lse"], ref_lse, rtol=2.01 / 65536, atol=1e-6)
    cfg.validate()


def run_bench(
    *, warmup: int | None = None, repeat: int | None = None, timer: str | None = None, **kwargs: Any
) -> dict[str, Any]:
    _rounds = kwargs.pop("rounds", 1)
    _round_cooldown_s = kwargs.pop("round_cooldown_s", 1.0)
    if not _IMPLEMENTATION_COMPLETE:
        raise SkipTest("sparse FlashMLA head128 small-topk phase1 transcription is not complete")
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for sparse FlashMLA head128 small-topk phase1 benchmark")

    from tirx_kernels.runner import compile_kernel
    from tvm.tirx.bench import bench

    case = prepare_data(**kwargs)
    if not case["dispatch_reason"].startswith("small_topk:"):
        raise SkipTest(case["dispatch_reason"])
    prim_func = get_kernel(**kwargs)
    ex = compile_kernel(prim_func)

    # Allocate inputs once, outside the timed region (Triton-standard pure launch).
    case["launches"] = _build_tirx_launches(case)

    funcs = {"tirx": lambda: _run_tirx_launches(ex, case["launches"])}

    from tirx_kernels.flashmla._flashmla_bench import flashmla_reference_builder

    def _flashmla_ref():
        run = flashmla_reference_builder()
        return lambda: run(case)

    return bench(
        funcs,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references={"flashmla": _flashmla_ref},
        rounds=_rounds,
        round_cooldown_s=_round_cooldown_s,
    )


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
]

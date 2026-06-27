from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any
from unittest import SkipTest

import torch

from tvm.backend.cuda.operator.tile_primitive.tma_utils import SwizzleMode
from tvm.script import tirx as T
from tvm.tirx.lang.pipeline import MBarrier, TCGen05Bar, TMABar

B_H = 64
B_TOPK = 64
D_V = 512
NUM_BUFS = 3
NUM_THREADS = 384
MAX_INIT_VAL = -1.0e30
LOG_2_E = math.log2(math.e)
LN_2 = math.log(2.0)

HEAD64_LAUNCH_PARAM_TAGS = ("blockIdx.x", "threadIdx.x", "tirx.use_dyn_shared_memory")

TMEM_COL_O = 0
TMEM_COL_Q = 256
TMEM_COL_Q_ROPE = 384
TMEM_COL_P = 400
TILED_MMA_P = (B_H, 128, 16, TMEM_COL_P)
TILED_MMA_O = (B_H, 256, 16, TMEM_COL_O)

NAMED_BARRIER_WG0_SYNC = 0
NAMED_BARRIER_WG0_WARP02_SYNC = 1

BF16_BYTES = 2
Q_ROPE_DIM = 64
SHARED_U_BYTES = (B_TOPK * 64 + 2 * B_TOPK * D_V + B_H * D_V) * BF16_BYTES
K_ROPE_BYTES = B_TOPK * Q_ROPE_DIM * BF16_BYTES
K_NOPE_OFFSET_BYTES = K_ROPE_BYTES
Q_NOPE_OFFSET_BYTES = (B_TOPK * 64 + 2 * B_TOPK * D_V) * BF16_BYTES
Q_NOPE_TMA_BYTES = B_H * D_V * BF16_BYTES
# tcgen05 shared-memory descriptors store stride fields in uint128_t units;
# these values match CuTe make_umma_desc<UMMA::Major::K> for the source layouts.
K_MAJOR_SWIZZLED_DESC_LDO = 1
Q_NOPE_DESC_SDO = 64
Q_ROPE_TMA_BYTES = B_H * Q_ROPE_DIM * BF16_BYTES
Q_ROPE_DESC_SDO = 32
S_DESC_LDO = 64
S_DESC_SDO = 8
V_DESC_LDO = 512
V_DESC_SDO = 64
P_EXCHANGE_BYTES = 4 * 32 * (B_TOPK // 2) * 4
S_Q_ROPE_BYTES = B_H * B_TOPK * BF16_BYTES
WG1_NUM_WARPS = 4
WG1_NUM_LOCAL_ROWS_PER_WARP = (B_TOPK // 4) // WG1_NUM_WARPS

_IMPLEMENTATION_COMPLETE = True


@dataclass(frozen=True)
class SparseFlashMLAPrefillHead64Config:
    label: str
    s_q: int
    s_kv: int
    topk: int
    d_qk: int = 576
    h_q: int = B_H
    h_kv: int = 1
    d_v: int = D_V
    have_attn_sink: bool = False
    have_topk_length: bool = False
    inject_invalid_indices: bool = False
    seed: int = 0

    def validate(self) -> None:
        if self.h_q != B_H:
            raise ValueError("head64 regular phase1 requires h_q == 64")
        if self.h_kv != 1:
            raise ValueError("head64 regular phase1 requires h_kv == 1")
        if self.d_qk not in (512, 576):
            raise ValueError("d_qk must be 512 or 576")
        if self.d_v != D_V:
            raise ValueError("d_v must be 512")
        if self.topk % B_TOPK != 0:
            raise ValueError("topk must be a multiple of 64")


CONFIGS = [
    {"label": "smoke_dqk576_s1_kv128_topk128", "s_q": 1, "s_kv": 128, "topk": 128, "d_qk": 576},
    {"label": "smoke_dqk512_s1_kv128_topk128", "s_q": 1, "s_kv": 128, "topk": 128, "d_qk": 512},
    {
        "label": "features_dqk576_s62_kv592_topk128",
        "s_q": 62,
        "s_kv": 592,
        "topk": 128,
        "d_qk": 576,
        "have_attn_sink": True,
        "have_topk_length": True,
    },
    {
        "label": "features_dqk512_s62_kv592_topk128",
        "s_q": 62,
        "s_kv": 592,
        "topk": 128,
        "d_qk": 512,
        "have_attn_sink": True,
        "have_topk_length": True,
    },
    {
        "label": "ring_dqk576_s3_kv640_topk512",
        "s_q": 3,
        "s_kv": 640,
        "topk": 512,
        "d_qk": 576,
        "have_attn_sink": True,
    },
    {
        "label": "ring_dqk512_s3_kv640_topk512",
        "s_q": 3,
        "s_kv": 640,
        "topk": 512,
        "d_qk": 512,
        "have_attn_sink": True,
    },
    {
        "label": "invalid_indices_dqk576_s17_kv192_topk128",
        "s_q": 17,
        "s_kv": 192,
        "topk": 128,
        "d_qk": 576,
        "inject_invalid_indices": True,
    },
    {
        "label": "invalid_indices_dqk512_s17_kv192_topk128",
        "s_q": 17,
        "s_kv": 192,
        "topk": 128,
        "d_qk": 512,
        "inject_invalid_indices": True,
    },
]

# Cover the two upstream fwd/head64 phase1 instantiations:
# D_QK=512 and D_QK=576, h_q=64, topk=512 at the scoped s_kv values.
BENCH_CONFIGS = [
    {
        "label": f"bench_dqk{d_qk}_hq64_s4096_kv{s_kv}_topk512",
        "s_q": 4096,
        "s_kv": s_kv,
        "topk": 512,
        "d_qk": d_qk,
        "h_q": B_H,
        "have_attn_sink": True,
    }
    for d_qk in (512, 576)
    for s_kv in (8192, 32768, 49152, 65536)
]

KERNEL_META = {
    "name": "sparse_flashmla_prefill_head64_phase1",
    "category": "attention",
    "compute_capability": 10,
}


def _cfg(**kwargs: Any) -> SparseFlashMLAPrefillHead64Config:
    cfg_fields = {field.name for field in fields(SparseFlashMLAPrefillHead64Config)}
    cfg_kwargs = {key: value for key, value in kwargs.items() if key in cfg_fields}
    if "label" not in cfg_kwargs:
        cfg_kwargs["label"] = "custom"
    cfg = SparseFlashMLAPrefillHead64Config(**cfg_kwargs)
    cfg.validate()
    return cfg


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
    }


def _reference_sparse_prefill(
    case: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg: SparseFlashMLAPrefillHead64Config = case["config"]
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


def _encode_tma_desc(
    *,
    tensor: torch.Tensor,
    global_shape: tuple[int, ...],
    global_strides: tuple[int, ...],
    box_dim: tuple[int, ...],
    swizzle_mode: int,
) -> Any:
    import ctypes

    import tvm
    from tirx_kernels.deepgemm import mega_moe

    rank = len(global_shape)
    if len(global_strides) != rank - 1:
        raise ValueError("TensorMap global_strides must have rank - 1 entries")
    if len(box_dim) != rank:
        raise ValueError("TensorMap box_dim must have rank entries")

    elem_size = int(tensor.element_size())
    encode_tensormap = tvm.get_global_func("runtime.cuTensorMapEncodeTiled")
    desc = mega_moe._AlignedTensorMap()
    encode_tensormap(
        desc.ptr,
        mega_moe._torch_dtype_to_tvm_dtype(tensor),
        rank,
        ctypes.c_void_p(int(tensor.data_ptr())),
        *[int(v) for v in global_shape],
        *[int(v * elem_size) for v in global_strides],
        *[int(v) for v in box_dim],
        *([1] * rank),
        mega_moe._CUDA_TENSOR_MAP_INTERLEAVE_NONE,
        mega_moe._tensor_map_swizzle_from_mode(swizzle_mode),
        mega_moe._CUDA_TENSOR_MAP_L2_PROMOTION_L2_256B,
        mega_moe._CUDA_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    return desc


def _build_tirx_tensor_maps(case: dict[str, Any]) -> dict[str, Any]:
    import tvm
    from tirx_kernels.deepgemm.mega_moe import _encode_tma_2d_desc

    cfg: SparseFlashMLAPrefillHead64Config = case["config"]
    q = case["q"]
    q_rope = q[:, :, D_V:]
    kv = case["kv"]
    out = case["out"]
    encode_tensormap = tvm.get_global_func("runtime.cuTensorMapEncodeTiled")

    return {
        "tensor_map_q_nope": _encode_tma_desc(
            tensor=q,
            global_shape=(D_V, cfg.h_q, cfg.s_q),
            global_strides=(int(q.stride(1)), int(q.stride(0))),
            box_dim=(64, B_H, 1),
            swizzle_mode=128,
        ),
        "tensor_map_q_rope": _encode_tma_desc(
            tensor=q_rope,
            global_shape=(Q_ROPE_DIM, cfg.h_q, cfg.s_q),
            global_strides=(int(q_rope.stride(1)), int(q_rope.stride(0))),
            box_dim=(32, B_H, 1),
            swizzle_mode=64,
        ),
        "tensor_map_o": _encode_tma_desc(
            tensor=out,
            global_shape=(cfg.d_v, cfg.h_q, cfg.s_q),
            global_strides=(int(out.stride(1)), int(out.stride(0))),
            box_dim=(64, B_H, 1),
            swizzle_mode=128,
        ),
        "tensor_map_kv_nope": _encode_tma_2d_desc(
            encode_tensormap=encode_tensormap,
            tensor=kv,
            gmem_inner_dim=D_V,
            gmem_outer_dim=cfg.s_kv,
            smem_inner_dim=64,
            smem_outer_dim=1,
            gmem_outer_stride=int(kv.stride(0)),
            swizzle_mode=128,
        ),
    }


def _make_tirx_launch(case: dict[str, Any]) -> dict[str, Any]:
    tensor_maps = _build_tirx_tensor_maps(case)
    return {
        "case": case,
        "tensor_maps": tensor_maps,
        "args": (
            case["q"],
            case["kv"].reshape(-1),
            case["indices"].reshape(-1),
            case["attn_sink"],
            case["topk_length"],
            case["out"],
            case["max_logits"],
            case["lse"],
            tensor_maps["tensor_map_q_nope"].ptr,
            tensor_maps["tensor_map_q_rope"].ptr,
            tensor_maps["tensor_map_o"].ptr,
            tensor_maps["tensor_map_kv_nope"].ptr,
        ),
    }


def _build_tirx_launches(case: dict[str, Any]) -> list[dict[str, Any]]:
    return [_make_tirx_launch(case)]


def _run_tirx_launches(
    executable: Any, launches: list[dict[str, Any]], *, output_case: dict[str, Any] | None = None
) -> None:
    for launch in launches:
        executable(*launch["args"])


def _tirx_benchmark_tensors(
    case: dict[str, Any], launches: list[dict[str, Any]]
) -> tuple[Any, ...]:
    return (
        case["q"],
        case["kv"],
        case["indices"],
        case["attn_sink"],
        case["topk_length"],
        case["out"],
        case["max_logits"],
        case["lse"],
    )


def _mbarrier_complete_tx(bar_ptr: Any, transaction_bytes: Any) -> Any:
    func_name = "sparse_flashmla_mbarrier_complete_tx"
    source_code = f"""
__device__ __forceinline__ void {func_name}(void* bar_ptr, unsigned int transaction_bytes) {{
  unsigned int smem_addr;
  asm volatile(
    "{{\\n\\t"
    ".reg .u64 smem_addr64;\\n\\t"
    "cvta.to.shared.u64 smem_addr64, %1;\\n\\t"
    "cvt.u32.u64 %0, smem_addr64;\\n\\t"
    "}}\\n"
    : "=r"(smem_addr) : "l"(bar_ptr));
  asm volatile(
    "mbarrier.complete_tx.shared::cluster.relaxed.cluster.b64 [%0], %1;\\n"
    :: "r"(smem_addr), "r"(transaction_bytes) : "memory");
}}
"""
    return T.cuda.func_call(
        func_name, bar_ptr, transaction_bytes, source_code=source_code, return_type="void"
    )


def _cpasync_barrier_arrive_noinc(bar_ptr: Any) -> Any:
    func_name = "sparse_flashmla_cpasync_barrier_arrive_noinc"
    source_code = f"""
__device__ __forceinline__ void {func_name}(void* bar_ptr) {{
  unsigned int smem_addr;
  asm volatile(
    "{{\\n\\t"
    ".reg .u64 smem_addr64;\\n\\t"
    "cvta.to.shared.u64 smem_addr64, %1;\\n\\t"
    "cvt.u32.u64 %0, smem_addr64;\\n\\t"
    "}}\\n"
    : "=r"(smem_addr) : "l"(bar_ptr));
  asm volatile(
    "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\\n"
    :: "r"(smem_addr) : "memory");
}}
"""
    return T.cuda.func_call(func_name, bar_ptr, source_code=source_code, return_type="void")


def _ldg_int4_indices(dst0: Any, dst1: Any, dst2: Any, dst3: Any, src_ptr: Any) -> Any:
    func_name = "sparse_flashmla_ldg_int4_indices"
    source_code = f"""
__device__ __forceinline__ void {func_name}(
    int* dst0, int* dst1, int* dst2, int* dst3, const int* src_ptr) {{
  int4 v = __ldg(reinterpret_cast<const int4*>(src_ptr));
  *dst0 = v.x;
  *dst1 = v.y;
  *dst2 = v.z;
  *dst3 = v.w;
}}
"""
    return T.cuda.func_call(
        func_name, dst0, dst1, dst2, dst3, src_ptr, source_code=source_code, return_type="void"
    )


def _ldg_256_indices(
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
    func_name = "sparse_flashmla_ldg_256_indices"
    source_code = f"""
__device__ __forceinline__ void {func_name}(
    int* dst0, int* dst1, int* dst2, int* dst3,
    int* dst4, int* dst5, int* dst6, int* dst7, const int* src_ptr) {{
  uint64_t raw0, raw1, raw2, raw3;
  asm volatile(
    "ld.global.nc.L1::no_allocate.L2::evict_normal.L2::256B.v4.u64 "
    "{{%0, %1, %2, %3}}, [%4];\\n"
    : "=l"(raw0), "=l"(raw1), "=l"(raw2), "=l"(raw3)
    : "l"(src_ptr));
  int vals[8];
  reinterpret_cast<uint64_t*>(vals)[0] = raw0;
  reinterpret_cast<uint64_t*>(vals)[1] = raw1;
  reinterpret_cast<uint64_t*>(vals)[2] = raw2;
  reinterpret_cast<uint64_t*>(vals)[3] = raw3;
  *dst0 = vals[0];
  *dst1 = vals[1];
  *dst2 = vals[2];
  *dst3 = vals[3];
  *dst4 = vals[4];
  *dst5 = vals[5];
  *dst6 = vals[6];
  *dst7 = vals[7];
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


def _tma_gather4_kv_nope(
    dst_ptr: Any,
    bar_ptr: Any,
    tensor_map_ptr: Any,
    col_idx: Any,
    row_idx0: Any,
    row_idx1: Any,
    row_idx2: Any,
    row_idx3: Any,
    cache_hint: Any,
) -> Any:
    func_name = "sparse_flashmla_tma_gather4_kv_nope"
    source_code = f"""
__device__ __forceinline__ void {func_name}(
    void* dst_ptr, void* bar_ptr, unsigned long long tensor_map_addr, int col_idx,
    int row_idx0, int row_idx1, int row_idx2, int row_idx3, unsigned long long cache_hint) {{
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
  asm volatile(
    "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
    ".mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint "
    "[%0], [%1, {{%2, %3, %4, %5, %6}}], [%7], %8;\\n"
    :
    : "r"(smem_addr), "l"(tensor_map_addr), "r"(col_idx),
      "r"(row_idx0), "r"(row_idx1), "r"(row_idx2), "r"(row_idx3),
      "r"(mbar_addr), "l"(cache_hint)
    : "memory");
}}
"""
    return T.cuda.func_call(
        func_name,
        dst_ptr,
        bar_ptr,
        tensor_map_ptr,
        col_idx,
        row_idx0,
        row_idx1,
        row_idx2,
        row_idx3,
        cache_hint,
        source_code=source_code,
        return_type="void",
    )


def _ld_shared_float4(dst0: Any, dst1: Any, dst2: Any, dst3: Any, src_ptr: Any) -> Any:
    func_name = "sparse_flashmla_ld_shared_float4"
    source_code = f"""
__device__ __forceinline__ void {func_name}(
    float* dst0, float* dst1, float* dst2, float* dst3, const float* src_ptr) {{
  unsigned int smem_addr;
  asm volatile(
    "{{\\n\\t"
    ".reg .u64 smem_addr64;\\n\\t"
    "cvta.to.shared.u64 smem_addr64, %1;\\n\\t"
    "cvt.u32.u64 %0, smem_addr64;\\n\\t"
    "}}\\n"
    : "=r"(smem_addr) : "l"(src_ptr));
  float x0, x1, x2, x3;
  asm volatile(
    "ld.shared.v4.f32 {{%0, %1, %2, %3}}, [%4];\\n"
    : "=f"(x0), "=f"(x1), "=f"(x2), "=f"(x3)
    : "r"(smem_addr));
  *dst0 = x0;
  *dst1 = x1;
  *dst2 = x2;
  *dst3 = x3;
}}
"""
    return T.cuda.func_call(
        func_name, dst0, dst1, dst2, dst3, src_ptr, source_code=source_code, return_type="void"
    )


def _st_shared_b128_float4(dst_ptr: Any, x0: Any, x1: Any, x2: Any, x3: Any) -> Any:
    func_name = "sparse_flashmla_st_shared_b128_float4"
    source_code = f"""
__device__ __forceinline__ void {func_name}(
    void* dst_ptr, float x0, float x1, float x2, float x3) {{
  struct alignas(16) Float4Pack {{
    float x;
    float y;
    float z;
    float w;
  }};
  Float4Pack pack{{x0, x1, x2, x3}};
  __int128_t val = *reinterpret_cast<__int128_t*>(&pack);
  asm volatile("st.shared.b128 [%0], %1;" :: "l"(__cvta_generic_to_shared(dst_ptr)), "q"(val));
}}
"""
    return T.cuda.func_call(
        func_name, dst_ptr, x0, x1, x2, x3, source_code=source_code, return_type="void"
    )


def _tcgen05_mma_ws_ts(
    d_tmem_addr: Any, a_tmem_addr: Any, b_desc: Any, i_desc: Any, scale_c: Any
) -> Any:
    func_name = "sparse_flashmla_tcgen05_mma_ws_ts"
    source_code = f"""
__device__ __forceinline__ void {func_name}(
    unsigned int d_tmem_addr, unsigned int a_tmem_addr, unsigned long long b_desc,
    unsigned int i_desc, unsigned int scale_c) {{
  asm volatile(
    "{{\\n\\t"
    ".reg .pred p;\\n\\t"
    "setp.ne.b32 p, %4, 0;\\n\\t"
    "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [%1], %2, %3, p, 0;\\n\\t"
    "}}\\n"
    :
    : "r"(d_tmem_addr), "r"(a_tmem_addr), "l"(b_desc), "r"(i_desc), "r"(scale_c));
}}
"""
    return T.cuda.func_call(
        func_name,
        d_tmem_addr,
        a_tmem_addr,
        b_desc,
        i_desc,
        scale_c,
        source_code=source_code,
        return_type="void",
    )


def _tcgen05_mma_ws_ss(
    d_tmem_addr: Any, a_desc: Any, b_desc: Any, i_desc: Any, scale_c: Any
) -> Any:
    func_name = "sparse_flashmla_tcgen05_mma_ws_ss"
    source_code = f"""
__device__ __forceinline__ void {func_name}(
    unsigned int d_tmem_addr, unsigned long long a_desc, unsigned long long b_desc,
    unsigned int i_desc, unsigned int scale_c) {{
  asm volatile(
    "{{\\n\\t"
    ".reg .pred p;\\n\\t"
    "setp.ne.b32 p, %4, 0;\\n\\t"
    "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], %1, %2, %3, p, 0;\\n\\t"
    "}}\\n"
    :
    : "r"(d_tmem_addr), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(scale_c));
}}
"""
    return T.cuda.func_call(
        func_name,
        d_tmem_addr,
        a_desc,
        b_desc,
        i_desc,
        scale_c,
        source_code=source_code,
        return_type="void",
    )


def _fdividef(x: Any, y: Any) -> Any:
    func_name = "sparse_flashmla_fdividef"
    source_code = f"""
__device__ __forceinline__ float {func_name}(float x, float y) {{
  return __fdividef(x, y);
}}
"""
    return T.cuda.func_call(func_name, x, y, source_code=source_code, return_type="float32")


def _ring_mod3(value: Any, max_value: int) -> Any:
    if max_value <= 8:
        packed_mod3 = T.uint32(0x10210210)
        shift = T.cast(value, "uint32") * T.uint32(4)
        return T.cast(T.bitwise_and(T.shift_right(packed_mod3, shift), T.uint32(0xF)), "int32")

    max_offset = (max_value // NUM_BUFS) * NUM_BUFS
    result = value - max_offset
    for offset in range(max_offset, 0, -NUM_BUFS):
        result = T.Select(value < offset, value - (offset - NUM_BUFS), result)
    return result


def _ring_phase_parity(value: Any, max_value: int) -> Any:
    if max_value <= 8:
        packed_phase = T.uint32(0x38)
        return T.cast(
            T.bitwise_and(T.shift_right(packed_phase, T.cast(value, "uint32")), T.uint32(1)),
            "int32",
        )

    max_offset = (max_value // NUM_BUFS) * NUM_BUFS
    result = T.int32((max_offset // NUM_BUFS) & 1)
    for offset in range(max_offset, 0, -NUM_BUFS):
        result = T.Select(value < offset, T.int32(((offset - NUM_BUFS) // NUM_BUFS) & 1), result)
    return result


@T.jit
def _kernel(
    q: T.Buffer((s_q, h_q, d_qk), "bfloat16"),
    kv: T.Buffer((s_kv * stride_kv_s_kv,), "bfloat16"),
    indices: T.Buffer((s_q * stride_indices_s_q,), "int32"),
    attn_sink: T.Buffer((h_q,), "float32"),
    topk_length: T.Buffer((s_q,), "int32"),
    out: T.Buffer((s_q, h_q, D_V), "bfloat16"),
    max_logits: T.Buffer((s_q, h_q), "float32"),
    lse: T.Buffer((s_q, h_q), "float32"),
    tensor_map_q_nope: T.TensorMap(),
    tensor_map_q_rope: T.TensorMap(),
    tensor_map_o: T.TensorMap(),
    tensor_map_kv_nope: T.TensorMap(),
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
    # CUDA_TRANSCRIBE_START: sparse_attn_fwd_kernel lines 65-71.
    # Transcription note: match upstream FlashMLA phase1's one-CTA-per-query-row launch.
    # Transcription note: preserve upstream source-order roles; warp 0 owns Q TMA,
    # warps 0-1 own O TMA, and warpgroup 0 also owns softmax/epilogue CUDA-core work.
    s_q_idx = T.cta_id([s_q])
    warpgroup_idx = T.warpgroup_id([3])
    warp_idx_in_wg = T.warp_id_in_wg([4])
    lane_idx = T.lane_id([32])
    idx_in_warpgroup = T.thread_id_in_wg([128])
    warp_idx: T.let = warpgroup_idx * 4 + warp_idx_in_wg
    topk_len: T.let = topk_length[s_q_idx] if have_topk_length else topk
    max_k_blocks = T.meta_var(topk // B_TOPK)
    num_k_blocks: T.let = T.max((topk_len + B_TOPK - 1) // B_TOPK, 1)
    have_rope = T.meta_var(d_qk == 576)

    # CUDA phase1.cuh:73-78, config.h:111-139.  Preserve the SharedMemoryPlan
    # storage offsets now, but instantiate bf16 MMA views only when their CUDA
    # use sites are transcribed.  Declaring unused bf16 views trips TIRx BF16
    # storage legalization before there is any copy/MMA/store to bind them to.
    pool = T.SMEMPool()
    u_base = T.meta_var(pool.offset)
    k_rope = pool.alloc_mma(
        (B_TOPK, Q_ROPE_DIM), "bfloat16", swizzle_mode=SwizzleMode.SWIZZLE_64B_ATOM
    )
    pool.move_base_to(u_base + K_NOPE_OFFSET_BYTES)
    k_nope = pool.alloc_mma((NUM_BUFS, B_TOPK, D_V), "bfloat16")
    pool.move_base_to(u_base + K_NOPE_OFFSET_BYTES)
    k_nope_tiled_mma = pool.alloc_mma((NUM_BUFS, B_TOPK * 2, D_V // 2), "bfloat16")
    pool.move_base_to(u_base + Q_NOPE_OFFSET_BYTES)
    q_nope = pool.alloc_mma((B_H, D_V), "bfloat16")
    pool.move_base_to(u_base)
    o_smem = pool.alloc_mma((B_H, D_V), "bfloat16")
    pool.move_base_to(u_base + SHARED_U_BYTES)

    p_exchange_buf = pool.alloc((4, 32 * (B_TOPK // 2)), "float32")
    s_q_rope_base = T.meta_var(pool.offset)
    q_rope = pool.alloc_mma(
        (B_H, Q_ROPE_DIM), "bfloat16", swizzle_mode=SwizzleMode.SWIZZLE_64B_ATOM
    )
    pool.move_base_to(s_q_rope_base)
    s_q_rope_s = pool.alloc((B_H * B_TOPK,), "bfloat16")
    pool.move_base_to(s_q_rope_base + S_Q_ROPE_BYTES)

    is_k_valid = pool.alloc((NUM_BUFS, B_TOPK // 8), "int8")
    bar_prologue_q_nope = TMABar(pool, 1)
    bar_prologue_q_rope = TMABar(pool, 1)
    bar_prologue_utccp_nope = TCGen05Bar(pool, 1)
    bar_prologue_utccp_rope = TCGen05Bar(pool, 1)
    bar_qk_nope_done = TCGen05Bar(pool, NUM_BUFS)
    bar_qk_rope_done = TCGen05Bar(pool, 1)
    bar_sv_done = TCGen05Bar(pool, NUM_BUFS)
    bar_kv_nope_ready_part0 = TMABar(pool, NUM_BUFS)
    bar_kv_nope_ready_part1 = TMABar(pool, NUM_BUFS)
    bar_kv_rope_ready = MBarrier(pool, 1)
    bar_p_free = MBarrier(pool, 1)
    bar_so_ready = MBarrier(pool, 1)
    bar_k_valid_ready = MBarrier(pool, NUM_BUFS)
    bar_k_valid_free = MBarrier(pool, NUM_BUFS)
    tmem_start_addr = pool.alloc((1,), "uint32", align=4)
    rowwise_max_buf = pool.alloc((128,), "float32")
    rowwise_li_buf = pool.alloc((128,), "float32")
    pool.commit()

    # CUDA phase1.cuh:77. h_kv is fixed to 1, so the row pointer is
    # params.indices + s_q_idx * params.stride_indices_s_q.
    g_indices_base: T.let = s_q_idx * stride_indices_s_q

    # CUDA phase1.cuh:79-98.  CuTe forges typed TMEM fragments by writing fixed
    # column starts into fragment pointers.  The aliases below preserve those
    # fixed starts while sharing the one physical 512-column TMEM allocation.
    tiled_mma_P = T.meta_var(TILED_MMA_P)
    tiled_mma_O = T.meta_var(TILED_MMA_O)
    tiled_mma_p_m = T.meta_var(tiled_mma_P[0])
    tiled_mma_p_n = T.meta_var(tiled_mma_P[1])
    tiled_mma_p_k = T.meta_var(tiled_mma_P[2])
    tiled_mma_p_col = T.meta_var(tiled_mma_P[3])
    tiled_mma_o_m = T.meta_var(tiled_mma_O[0])
    tiled_mma_o_n = T.meta_var(tiled_mma_O[1])
    tiled_mma_o_k = T.meta_var(tiled_mma_O[2])
    tiled_mma_o_col = T.meta_var(tiled_mma_O[3])
    tiled_mma_p_accumulate = T.alloc_local((1,), "uint32")
    tiled_mma_o_accumulate = T.alloc_local((1,), "uint32")
    tiled_mma_p_accumulate[0] = T.uint32(0)
    tiled_mma_o_accumulate[0] = T.uint32(0)

    # CUDA phase1.cuh:100-150.  Warp 0 performs descriptor prefetch, Q TMA
    # launch, prologue barrier init, and TMEM allocation.
    if warp_idx == 0:
        if T.ptx.elect_sync():
            if have_rope:
                T.evaluate(T.ptx.prefetch_tensormap(T.address_of(tensor_map_q_rope)))
            T.evaluate(T.ptx.prefetch_tensormap(T.address_of(tensor_map_q_nope)))

            bar_prologue_q_nope.init(1)
            bar_prologue_q_rope.init(1)
            T.ptx.fence.mbarrier_init()

            if have_rope:
                for q_rope_tma_tile in T.unroll(Q_ROPE_DIM // 32):
                    T.evaluate(
                        T.ptx.cp_async.bulk.tensor.g2c(
                            3,
                            q_rope.ptr_to([0, q_rope_tma_tile * 32]),
                            bar_prologue_q_rope.ptr_to([0]),
                            T.address_of(tensor_map_q_rope),
                            0,
                            1,
                            "evict_first",
                            q_rope_tma_tile * 32,
                            T.uint32(0),
                            s_q_idx,
                        )
                    )

            for q_nope_tma_tile in T.unroll(D_V // 64):
                T.evaluate(
                    T.ptx.cp_async.bulk.tensor.g2c(
                        3,
                        q_nope.ptr_to([0, q_nope_tma_tile * 64]),
                        bar_prologue_q_nope.ptr_to([0]),
                        T.address_of(tensor_map_q_nope),
                        0,
                        1,
                        "evict_first",
                        q_nope_tma_tile * 64,
                        T.uint32(0),
                        s_q_idx,
                    )
                )

            T.evaluate(T.ptx.prefetch_tensormap(T.address_of(tensor_map_o)))
            T.evaluate(T.ptx.prefetch_tensormap(T.address_of(tensor_map_kv_nope)))

            bar_prologue_utccp_rope.init(1)
            bar_prologue_utccp_nope.init(1)
            if bar_qk_nope_done.leader:
                for init_stage in T.unroll(NUM_BUFS):
                    T.ptx.mbarrier.init(bar_qk_nope_done.ptr_to([init_stage]), 1)
                    T.ptx.mbarrier.init(bar_sv_done.ptr_to([init_stage]), 1)
                    T.ptx.mbarrier.init(bar_kv_nope_ready_part0.ptr_to([init_stage]), 1)
                    T.ptx.mbarrier.init(bar_kv_nope_ready_part1.ptr_to([init_stage]), 1)
                    T.ptx.mbarrier.init(bar_k_valid_ready.ptr_to([init_stage]), B_TOPK // 8)
                    T.ptx.mbarrier.init(bar_k_valid_free.ptr_to([init_stage]), 128)
            bar_p_free.init(128)
            bar_so_ready.init(128)
            bar_qk_rope_done.init(1)
            bar_kv_rope_ready.init(64)
            T.ptx.fence.mbarrier_init()

        T.ptx.tcgen05.alloc(T.address_of(tmem_start_addr[0]), n_cols=512, cta_group=1)
        T.cuda.trap_when_assert_failed(tmem_start_addr[0] == T.uint32(0))
        T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)

    T.cuda.cta_sync()

    if warpgroup_idx == 0:
        # CUDA phase1.cuh:152-168.  Scale/exp warpgroup state.
        mi = T.local_scalar("float32")
        mi = MAX_INIT_VAL
        li = T.local_scalar("float32")
        li = 0.0
        real_mi = T.local_scalar("float32")
        real_mi = T.float32(-float("inf"))
        s_smem_lane_offset: T.int32 = (
            lane_idx * 8
            + T.bitwise_and(warp_idx, T.int32(1)) * (B_H // 2) * 8
            + (warp_idx // 2) * B_H * (B_TOPK // 2)
        )
        num_elems_per_thread = T.meta_var(B_TOPK // 2)

        # CUDA phase1.cuh:169-244.  Scale/exp loop with helper bodies stepped
        # into source order: P TMEM read/mask/reduce, row max, S generation,
        # S shared store, and conditional O rescale.
        for k in T.serial(0, num_k_blocks, unroll=False):
            T.ptx.bar.sync(NAMED_BARRIER_WG0_WARP02_SYNC + T.bitwise_and(warp_idx, T.int32(1)), 64)
            cur_buf: T.int32 = _ring_mod3(k, max_k_blocks)
            cur_phase: T.int32 = _ring_phase_parity(k, max_k_blocks)
            bar_qk_nope_done.wait(cur_buf, cur_phase)
            bar_k_valid_ready.wait(cur_buf, cur_phase)
            T.ptx.tcgen05.fence.after_thread_sync()

            # CUDA common_subroutine.h:75-134 retrieve_mask_and_reduce_p.
            p = T.alloc_local((num_elems_per_thread,), "float32")
            p_peer = T.alloc_local((num_elems_per_thread,), "float32")
            if warp_idx < 2:
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
                    num=num_elems_per_thread,
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
                    num=num_elems_per_thread,
                    col=TMEM_COL_P + num_elems_per_thread,
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
                    num=num_elems_per_thread,
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
                    num=num_elems_per_thread,
                    col=TMEM_COL_P + num_elems_per_thread,
                )
            T.ptx.tcgen05.wait.ld()
            T.ptx.tcgen05.fence.before_thread_sync()
            bar_p_free.arrive(0)

            valid_word_offset: T.int32 = T.if_then_else(warp_idx >= 2, num_elems_per_thread // 8, 0)
            is_k_valid_u32: T.let = T.ptx.ld(
                is_k_valid.ptr_to([cur_buf, valid_word_offset]), "uint32", "u32", space="shared"
            )
            for p_i in T.unroll(num_elems_per_thread):
                invalid_p_predicate: T.let = T.bitwise_and(
                    T.shift_right(is_k_valid_u32, T.uint32(p_i)), T.uint32(1)
                ) == T.uint32(0)
                p[p_i] = T.cuda.uint_as_float(
                    T.if_then_else(
                        invalid_p_predicate, T.uint32(0xFF800000), T.cuda.float_as_uint(p[p_i])
                    )
                )

            for exchange_i in T.unroll(num_elems_per_thread // 4):
                exchange_offset: T.let = exchange_i * 32 * 4 + lane_idx * 4
                T.evaluate(
                    _st_shared_b128_float4(
                        p_exchange_buf.ptr_to([warp_idx ^ 2, exchange_offset]),
                        p_peer[exchange_i * 4],
                        p_peer[exchange_i * 4 + 1],
                        p_peer[exchange_i * 4 + 2],
                        p_peer[exchange_i * 4 + 3],
                    )
                )
            T.ptx.bar.sync(NAMED_BARRIER_WG0_WARP02_SYNC + T.bitwise_and(warp_idx, T.int32(1)), 64)
            for exchange_i in T.unroll(num_elems_per_thread // 4):
                exchange_offset: T.let = exchange_i * 32 * 4 + lane_idx * 4
                p_exchange_tmp = T.alloc_local((4,), "float32")
                T.evaluate(
                    _ld_shared_float4(
                        p_exchange_tmp.ptr_to([0]),
                        p_exchange_tmp.ptr_to([1]),
                        p_exchange_tmp.ptr_to([2]),
                        p_exchange_tmp.ptr_to([3]),
                        p_exchange_buf.ptr_to([warp_idx, exchange_offset]),
                    )
                )
                p_pair0: T.let = T.cuda.make_float2(p[exchange_i * 4], p[exchange_i * 4 + 1])
                peer_pair0: T.let = T.cuda.make_float2(p_exchange_tmp[0], p_exchange_tmp[1])
                p_add_pair0 = T.alloc_local((1,), "uint64")
                T.ptx.add_f32x2(p_add_pair0.ptr_to([0]), p_pair0, peer_pair0)
                p[exchange_i * 4] = T.cuda.float2_x(p_add_pair0[0])
                p[exchange_i * 4 + 1] = T.cuda.float2_y(p_add_pair0[0])
                p_pair1: T.let = T.cuda.make_float2(p[exchange_i * 4 + 2], p[exchange_i * 4 + 3])
                peer_pair1: T.let = T.cuda.make_float2(p_exchange_tmp[2], p_exchange_tmp[3])
                p_add_pair1 = T.alloc_local((1,), "uint64")
                T.ptx.add_f32x2(p_add_pair1.ptr_to([0]), p_pair1, peer_pair1)
                p[exchange_i * 4 + 2] = T.cuda.float2_x(p_add_pair1[0])
                p[exchange_i * 4 + 3] = T.cuda.float2_y(p_add_pair1[0])

            bar_k_valid_free.arrive(cur_buf)

            cur_pi_max = T.local_scalar("float32")
            cur_pi_max = T.float32(-float("inf"))
            for p_i in T.unroll(num_elems_per_thread):
                cur_pi_max = T.max(cur_pi_max, p[p_i])
            cur_pi_max = cur_pi_max * sm_scale_div_log2
            rowwise_max_buf[idx_in_warpgroup] = cur_pi_max
            T.ptx.bar.sync(NAMED_BARRIER_WG0_SYNC, 128)
            cur_pi_max = T.max(cur_pi_max, rowwise_max_buf[idx_in_warpgroup ^ 64])
            real_mi = T.max(real_mi, cur_pi_max)
            should_scale_o = T.local_scalar("bool")
            should_scale_o = T.ptx.any_sync(T.uint32(0xFFFFFFFF), cur_pi_max - mi > 6.0) != 0
            new_max = T.local_scalar("float32")
            scale_for_old = T.local_scalar("float32")
            if not should_scale_o:
                scale_for_old = 1.0
                new_max = mi
            else:
                new_max = T.max(cur_pi_max, mi)
                scale_for_old = T.ptx.exp2(mi - new_max)
            mi = new_max

            s_pack = T.alloc_local((num_elems_per_thread // 2,), "uint32")
            cur_sum_pair = T.local_scalar("uint64")
            cur_sum_pair = T.cuda.make_float2(T.float32(0.0), T.float32(0.0))
            neg_new_max_pair: T.let = T.cuda.make_float2(-new_max, -new_max)
            scale_pair: T.let = T.cuda.make_float2(sm_scale_div_log2, sm_scale_div_log2)
            for s_i in T.unroll(num_elems_per_thread // 2):
                p_pair: T.let = T.cuda.make_float2(p[s_i * 2], p[s_i * 2 + 1])
                fma_pair = T.alloc_local((1,), "uint64")
                T.ptx.fma_f32x2(fma_pair.ptr_to([0]), p_pair, scale_pair, neg_new_max_pair)
                s_x: T.let = T.ptx.exp2(T.cuda.float2_x(fma_pair[0]))
                s_y: T.let = T.ptx.exp2(T.cuda.float2_y(fma_pair[0]))
                s_pair: T.let = T.cuda.make_float2(s_x, s_y)
                sum_pair_tmp = T.alloc_local((1,), "uint64")
                T.ptx.add_f32x2(sum_pair_tmp.ptr_to([0]), cur_sum_pair, s_pair)
                cur_sum_pair = sum_pair_tmp[0]
                s_pack[s_i] = T.cuda.float22bfloat162_rn(s_x, s_y)
            cur_sum: T.let = T.cuda.float2_x(cur_sum_pair) + T.cuda.float2_y(cur_sum_pair)
            li_tmp = T.alloc_local((1,), "float32")
            T.ptx.fma_f32(li_tmp.ptr_to([0]), li, scale_for_old, cur_sum)
            li = li_tmp[0]

            if k > 0:
                prev_buf: T.int32 = _ring_mod3(k - 1, max_k_blocks)
                prev_phase: T.int32 = _ring_phase_parity(k - 1, max_k_blocks)
                bar_sv_done.wait(prev_buf, prev_phase)

            # CUDA phase1.cuh:229-232 vectorized uint128_t stores to sS_base.
            for s_store_i in T.unroll(num_elems_per_thread // 8):
                s_store_offset: T.let = s_smem_lane_offset + B_H * 8 * s_store_i
                T.evaluate(
                    T.ptx.st(
                        s_q_rope_s.ptr_to([s_store_offset]),
                        s_pack[s_store_i * 4],
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
                # CUDA common_subroutine.h:147-168 rescale_O.
                rescale_o_d_v = T.meta_var(D_V)
                chunk_size = T.meta_var(32)
                tmem_col_start = T.meta_var(TMEM_COL_O)
                scale_for_old_pair: T.let = T.cuda.make_float2(scale_for_old, scale_for_old)
                o_rescale = T.alloc_local((32,), "float32")
                for chunk_idx in T.unroll((rescale_o_d_v // 2) // chunk_size):
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
                        num=chunk_size,
                        col=tmem_col_start + chunk_idx * chunk_size,
                    )
                    T.ptx.tcgen05.wait.ld()
                    for o_i in T.unroll(chunk_size // 2):
                        o_pair: T.let = T.cuda.make_float2(
                            o_rescale[o_i * 2], o_rescale[o_i * 2 + 1]
                        )
                        o_pair_tmp = T.alloc_local((1,), "uint64")
                        T.ptx.mul_f32x2(o_pair_tmp.ptr_to([0]), o_pair, scale_for_old_pair)
                        o_rescale[o_i * 2] = T.cuda.float2_x(o_pair_tmp[0])
                        o_rescale[o_i * 2 + 1] = T.cuda.float2_y(o_pair_tmp[0])
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
                        num=chunk_size,
                        col=tmem_col_start + chunk_idx * chunk_size,
                    )
                    T.ptx.tcgen05.wait.st()
                T.ptx.tcgen05.fence.before_thread_sync()

            T.ptx.fence.proxy_async("shared::cta")
            bar_so_ready.arrive(0)

        # CUDA phase1.cuh:246-357.  Epilogue scalar exchange, O TMEM readback,
        # output scaling/bf16 staging, and the two elected-warp O TMA stores.
        if real_mi == T.float32(-float("inf")):
            li = 0.0
            mi = T.float32(-float("inf"))

        rowwise_li_buf[idx_in_warpgroup] = li
        T.ptx.bar.sync(NAMED_BARRIER_WG0_SYNC, 128)
        li = li + rowwise_li_buf[idx_in_warpgroup ^ 64]

        if idx_in_warpgroup < B_H:
            cur_lse = T.local_scalar("float32")
            cur_lse_log: T.let = T.log(li)
            T.ptx.fma_f32(T.address_of(cur_lse), mi, LN_2, cur_lse_log)
            cur_lse = T.if_then_else(
                cur_lse == T.float32(-float("inf")), T.float32(float("inf")), cur_lse
            )
            max_logits[s_q_idx, idx_in_warpgroup] = real_mi * LN_2
            lse[s_q_idx, idx_in_warpgroup] = cur_lse

        last_k: T.int32 = num_k_blocks - 1
        last_buf: T.int32 = _ring_mod3(last_k, max_k_blocks)
        last_phase: T.int32 = _ring_phase_parity(last_k, max_k_blocks)
        bar_sv_done.wait(last_buf, last_phase)
        T.ptx.tcgen05.fence.after_thread_sync()

        attn_sink_log2: T.let = (
            T.cuda.ldg(attn_sink.ptr_to([idx_in_warpgroup % B_H]), "float32") * LOG_2_E
            if have_attn_sink
            else T.float32(-float("inf"))
        )
        output_scale = T.local_scalar("float32")
        output_scale = _fdividef(T.float32(1.0), li + T.ptx.exp2(attn_sink_log2 - mi))

        b_epi = T.meta_var(64)
        o_epi = T.alloc_local((b_epi,), "float32")
        have_valid_indices: T.let = T.ptx.any_sync(T.uint32(0xFFFFFFFF), li != 0.0) != 0
        if not have_valid_indices:
            for o_zero_i in T.unroll(b_epi):
                o_epi[o_zero_i] = 0.0
            output_scale = 1.0
        output_scale_pair: T.let = T.cuda.make_float2(output_scale, output_scale)
        o_epi_pair = T.alloc_local((1,), "uint64")
        for epi_c in T.unroll(2):
            for epi_k in T.unroll((D_V // 4) // b_epi):
                if have_valid_indices:
                    # CUDA phase1.cuh:314-317: TMEM O load/fence.
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
                        num=b_epi,
                        col=TMEM_COL_O + epi_c * 128 + epi_k * b_epi,
                    )
                    T.ptx.tcgen05.wait.ld()
                for o_i in T.unroll(b_epi // 8):
                    o_epi_bf16 = T.alloc_local((4,), "uint32")
                    for o_j in T.unroll(4):
                        o_pair_idx: T.let = o_i * 8 + o_j * 2
                        o_pair: T.let = T.cuda.make_float2(o_epi[o_pair_idx], o_epi[o_pair_idx + 1])
                        T.ptx.mul_f32x2(o_epi_pair.ptr_to([0]), o_pair, output_scale_pair)
                        o_epi[o_pair_idx] = T.cuda.float2_x(o_epi_pair[0])
                        o_epi[o_pair_idx + 1] = T.cuda.float2_y(o_epi_pair[0])
                        o_epi_bf16[o_j] = T.cuda.float22bfloat162_rn(
                            o_epi[o_pair_idx], o_epi[o_pair_idx + 1]
                        )
                    o_store_source_offset: T.let = (
                        epi_c * (D_V // 2) + (idx_in_warpgroup // B_H) * (D_V // 4) + epi_k * b_epi
                    )
                    o_base_col: T.let = o_i * 8 + o_store_source_offset
                    T.evaluate(
                        T.ptx.st(
                            o_smem.ptr_to([idx_in_warpgroup % B_H, o_base_col]),
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
                        # CUDA phase1.cuh:335-342: first half O TMA store.
                        epi_chunk_idx: T.let = epi_c * (D_V // 2 // b_epi) + epi_k
                        T.evaluate(
                            T.ptx.cp_async.bulk.tensor.s2g(
                                3,
                                o_smem.ptr_to([0, epi_chunk_idx * b_epi]),
                                T.address_of(tensor_map_o),
                                "",
                                T.uint32(epi_chunk_idx * b_epi),
                                T.uint32(0),
                                s_q_idx,
                            )
                        )
                if warp_idx == 1:
                    if T.ptx.elect_sync():
                        # CUDA phase1.cuh:343-350: second half O TMA store.
                        epi_chunk_idx: T.let = (
                            epi_c * (D_V // 2 // b_epi) + (D_V // b_epi // 4) + epi_k
                        )
                        T.evaluate(
                            T.ptx.cp_async.bulk.tensor.s2g(
                                3,
                                o_smem.ptr_to([0, epi_chunk_idx * b_epi]),
                                T.address_of(tensor_map_o),
                                "",
                                T.uint32(epi_chunk_idx * b_epi),
                                T.uint32(0),
                                s_q_idx,
                            )
                        )

        if warp_idx == 0:
            T.ptx.tcgen05.dealloc(T.uint32(0), n_cols=512, cta_group=1)

    elif warpgroup_idx == 1:
        # CUDA phase1.cuh:358-412.  KV NoPE producer.  Scalar index loads and
        # skip decisions are transcribed; gather4 requires explicit TensorMap
        # ABI plumbing and is left at the exact source-order call site.
        wg1_warp_idx: T.let = warp_idx - 4
        if T.ptx.elect_sync():
            for k in T.serial(0, num_k_blocks, unroll=False):
                selected_idx0 = T.alloc_local((WG1_NUM_LOCAL_ROWS_PER_WARP,), "int32")
                selected_idx1 = T.alloc_local((WG1_NUM_LOCAL_ROWS_PER_WARP,), "int32")
                selected_idx2 = T.alloc_local((WG1_NUM_LOCAL_ROWS_PER_WARP,), "int32")
                selected_idx3 = T.alloc_local((WG1_NUM_LOCAL_ROWS_PER_WARP,), "int32")
                max_indices = T.local_scalar("int32")
                min_indices = T.local_scalar("int32")
                max_indices = -1
                min_indices = s_kv

                for local_row in T.unroll(WG1_NUM_LOCAL_ROWS_PER_WARP):
                    row_base: T.let = (
                        g_indices_base
                        + k * B_TOPK
                        + local_row * WG1_NUM_WARPS * 4
                        + wg1_warp_idx * 4
                    )
                    T.evaluate(
                        _ldg_int4_indices(
                            selected_idx0.ptr_to([local_row]),
                            selected_idx1.ptr_to([local_row]),
                            selected_idx2.ptr_to([local_row]),
                            selected_idx3.ptr_to([local_row]),
                            indices.ptr_to([row_base]),
                        )
                    )
                    idx0: T.let = selected_idx0[local_row]
                    idx1: T.let = selected_idx1[local_row]
                    idx2: T.let = selected_idx2[local_row]
                    idx3: T.let = selected_idx3[local_row]
                    selected_idx0[local_row] = idx0
                    selected_idx1[local_row] = idx1
                    selected_idx2[local_row] = idx2
                    selected_idx3[local_row] = idx3
                    local_max: T.let = T.max(T.max(idx0, idx1), T.max(idx2, idx3))
                    max_indices = T.max(max_indices, local_max)
                    local_min: T.let = T.min(T.min(idx0, idx1), T.min(idx2, idx3))
                    min_indices = T.min(min_indices, local_min)

                is_all_rows_invalid: T.let = (min_indices == s_kv) | (max_indices == -1)
                should_skip_tma: T.let = is_all_rows_invalid & (k >= NUM_BUFS)

                if k == 2:
                    bar_prologue_utccp_nope.wait(0, 0)

                cur_buf: T.int32 = _ring_mod3(k, max_k_blocks)
                cur_phase: T.int32 = _ring_phase_parity(k, max_k_blocks)
                bar_sv_done.wait(cur_buf, T.bitwise_xor(cur_phase, T.int32(1)))

                if not should_skip_tma:
                    part_idx0 = T.meta_var(0)
                    for local_row in T.unroll(WG1_NUM_LOCAL_ROWS_PER_WARP):
                        for local_col_inner in T.unroll((D_V // 2) // 64):
                            local_col: T.let = part_idx0 * ((D_V // 2) // 64) + local_col_inner
                            smem_row: T.let = wg1_warp_idx * 4 + local_row * WG1_NUM_WARPS * 4
                            raw_k_nope_offset: T.let = (
                                cur_buf * B_TOPK * D_V + smem_row * 64 + local_col * B_TOPK * 64
                            )
                            T.evaluate(
                                _tma_gather4_kv_nope(
                                    k_nope.access_ptr("w", offset=raw_k_nope_offset),
                                    bar_kv_nope_ready_part0.ptr_to([cur_buf]),
                                    T.address_of(tensor_map_kv_nope),
                                    local_col * 64,
                                    selected_idx0[local_row],
                                    selected_idx1[local_row],
                                    selected_idx2[local_row],
                                    selected_idx3[local_row],
                                    T.uint64(0x14F0000000000000),
                                )
                            )
                    part_idx1 = T.meta_var(1)
                    for local_row in T.unroll(WG1_NUM_LOCAL_ROWS_PER_WARP):
                        for local_col_inner in T.unroll((D_V // 2) // 64):
                            local_col: T.let = part_idx1 * ((D_V // 2) // 64) + local_col_inner
                            smem_row: T.let = wg1_warp_idx * 4 + local_row * WG1_NUM_WARPS * 4
                            raw_k_nope_offset: T.let = (
                                cur_buf * B_TOPK * D_V + smem_row * 64 + local_col * B_TOPK * 64
                            )
                            T.evaluate(
                                _tma_gather4_kv_nope(
                                    k_nope.access_ptr("w", offset=raw_k_nope_offset),
                                    bar_kv_nope_ready_part1.ptr_to([cur_buf]),
                                    T.address_of(tensor_map_kv_nope),
                                    local_col * 64,
                                    selected_idx0[local_row],
                                    selected_idx1[local_row],
                                    selected_idx2[local_row],
                                    selected_idx3[local_row],
                                    T.uint64(0x14F0000000000000),
                                )
                            )
                else:
                    for part_idx in T.unroll(2):
                        tx_bytes = T.uint32(
                            WG1_NUM_LOCAL_ROWS_PER_WARP * 4 * (D_V // 2) * BF16_BYTES
                        )
                        if part_idx == 0:
                            T.evaluate(
                                _mbarrier_complete_tx(
                                    bar_kv_nope_ready_part0.ptr_to([cur_buf]), tx_bytes
                                )
                            )
                        else:
                            T.evaluate(
                                _mbarrier_complete_tx(
                                    bar_kv_nope_ready_part1.ptr_to([cur_buf]), tx_bytes
                                )
                            )

    else:
        # CUDA phase1.cuh:413-572.  MMA warpgroup.  Keep the warp-specialized
        # control flow source-ordered; materialize tcgen05.cp/gemm_async and
        # cp.async data paths after the exact SMEM/TMEM views are introduced.
        if warp_idx == 8:
            if T.ptx.elect_sync():
                q_nope_desc: T.uint64
                q_rope_desc: T.uint64
                T.ptx.tcgen05.encode_matrix_descriptor(
                    T.address_of(q_nope_desc),
                    q_nope.ptr_to([0, 0]),
                    ldo=K_MAJOR_SWIZZLED_DESC_LDO,
                    sdo=Q_NOPE_DESC_SDO,
                    swizzle=3,
                )
                T.ptx.tcgen05.encode_matrix_descriptor(
                    T.address_of(q_rope_desc),
                    q_rope.ptr_to([0, 0]),
                    ldo=K_MAJOR_SWIZZLED_DESC_LDO,
                    sdo=Q_ROPE_DESC_SDO,
                    swizzle=2,
                )
                desc_i_p_rope: T.uint32
                desc_i_p_nope: T.uint32
                desc_i_o: T.uint32
                T.ptx.tcgen05.encode_instr_descriptor(
                    T.address_of(desc_i_p_rope),
                    d_dtype="float32",
                    a_dtype="bfloat16",
                    b_dtype="bfloat16",
                    M=tiled_mma_p_m,
                    N=tiled_mma_p_n,
                    K=tiled_mma_p_k,
                    trans_a=False,
                    trans_b=False,
                    n_cta_groups=1,
                )
                T.ptx.tcgen05.encode_instr_descriptor(
                    T.address_of(desc_i_p_nope),
                    d_dtype="float32",
                    a_dtype="bfloat16",
                    b_dtype="bfloat16",
                    M=tiled_mma_p_m,
                    N=tiled_mma_p_n,
                    K=tiled_mma_p_k,
                    trans_a=False,
                    trans_b=False,
                    n_cta_groups=1,
                )
                T.ptx.tcgen05.encode_instr_descriptor(
                    T.address_of(desc_i_o),
                    d_dtype="float32",
                    a_dtype="bfloat16",
                    b_dtype="bfloat16",
                    M=tiled_mma_o_m,
                    N=tiled_mma_o_n,
                    K=tiled_mma_o_k,
                    trans_a=False,
                    trans_b=True,
                    n_cta_groups=1,
                )
                if have_rope:
                    bar_prologue_q_rope.arrive(0, tx_count=B_H * (d_qk - D_V) * BF16_BYTES)
                    bar_prologue_q_rope.wait(0, 0)
                    T.ptx.tcgen05.fence.after_thread_sync()
                    for subtile_idx in T.unroll(2):
                        T.ptx.tcgen05.cp(
                            T.uint32(TMEM_COL_Q_ROPE + subtile_idx * 8),
                            q_rope_desc + T.uint64(subtile_idx * 2),
                            shape="128x256b",
                            cta_group=1,
                        )
                    bar_prologue_utccp_rope.arrive(0)

                bar_prologue_q_nope.arrive(0, tx_count=B_H * D_V * BF16_BYTES)
                bar_prologue_q_nope.wait(0, 0)
                T.ptx.tcgen05.fence.after_thread_sync()
                for tile_idx in T.unroll(D_V // 64 // 2):
                    for subtile_idx in T.unroll(4):
                        T.ptx.tcgen05.cp(
                            T.uint32(TMEM_COL_Q + tile_idx * 32 + subtile_idx * 8),
                            q_nope_desc + T.uint64(tile_idx * 1024 + subtile_idx * 2),
                            shape="128x256b",
                            cta_group=1,
                        )
                bar_prologue_utccp_nope.arrive(0)

                if have_rope:
                    bar_prologue_utccp_rope.wait(0, 0)

                for k in T.serial(0, num_k_blocks + 1, unroll=False):
                    if k < num_k_blocks:
                        cur_buf: T.int32 = _ring_mod3(k, max_k_blocks)
                        cur_phase: T.int32 = _ring_phase_parity(k, max_k_blocks)
                        bar_p_free.wait(0, T.bitwise_xor(T.bitwise_and(k, T.int32(1)), T.int32(1)))
                        T.ptx.tcgen05.fence.after_thread_sync()

                        if have_rope:
                            bar_kv_rope_ready.wait(0, T.bitwise_and(k, T.int32(1)))
                            T.ptx.tcgen05.fence.after_thread_sync()
                            # CUDA phase1.cuh:489 Q RoPE x K RoPE MMA.
                            tiled_mma_p_accumulate[0] = T.uint32(0)
                            for rope_k in T.unroll(Q_ROPE_DIM // 32):
                                k_rope_desc: T.uint64
                                T.ptx.tcgen05.encode_matrix_descriptor(
                                    T.address_of(k_rope_desc),
                                    k_rope.ptr_to([0, rope_k * 16]),
                                    ldo=K_MAJOR_SWIZZLED_DESC_LDO,
                                    sdo=Q_ROPE_DESC_SDO,
                                    swizzle=2,
                                )
                                T.evaluate(
                                    _tcgen05_mma_ws_ts(
                                        T.uint32(tiled_mma_p_col),
                                        T.uint32(TMEM_COL_Q_ROPE + rope_k * 8),
                                        k_rope_desc,
                                        desc_i_p_rope,
                                        tiled_mma_p_accumulate[0],
                                    )
                                )
                                tiled_mma_p_accumulate[0] = T.uint32(1)
                            bar_qk_rope_done.arrive(0)

                        if k == 0:
                            bar_prologue_utccp_nope.wait(0, 0)

                        for kv_nope_part_idx in T.unroll(2):
                            tx_bytes: T.let = B_TOPK * (D_V // 2) * BF16_BYTES
                            if kv_nope_part_idx == 0:
                                bar_kv_nope_ready_part0.arrive(cur_buf, tx_count=tx_bytes)
                                bar_kv_nope_ready_part0.wait(cur_buf, cur_phase)
                            else:
                                bar_kv_nope_ready_part1.arrive(cur_buf, tx_count=tx_bytes)
                                bar_kv_nope_ready_part1.wait(cur_buf, cur_phase)
                            T.ptx.tcgen05.fence.after_thread_sync()
                            # CUDA phase1.cuh:505-506 Q NoPE x K NoPE MMA.
                            clear_nope_accum: T.let = (not have_rope) & (kv_nope_part_idx == 0)
                            tiled_mma_p_accumulate[0] = T.if_then_else(
                                clear_nope_accum, T.uint32(0), T.uint32(1)
                            )
                            for nope_k in T.unroll((D_V // 4) // 16):
                                k_nope_desc: T.uint64
                                T.ptx.tcgen05.encode_matrix_descriptor(
                                    T.address_of(k_nope_desc),
                                    k_nope_tiled_mma.ptr_to(
                                        [cur_buf, 0, kv_nope_part_idx * (D_V // 4) + nope_k * 16]
                                    ),
                                    ldo=K_MAJOR_SWIZZLED_DESC_LDO,
                                    sdo=Q_NOPE_DESC_SDO,
                                    swizzle=3,
                                )
                                T.evaluate(
                                    _tcgen05_mma_ws_ts(
                                        T.uint32(tiled_mma_p_col),
                                        T.uint32(TMEM_COL_Q + kv_nope_part_idx * 64 + nope_k * 8),
                                        k_nope_desc,
                                        desc_i_p_nope,
                                        tiled_mma_p_accumulate[0],
                                    )
                                )
                                tiled_mma_p_accumulate[0] = T.uint32(1)
                        bar_qk_nope_done.arrive(cur_buf)

                    if k > 0:
                        cur_buf_prev: T.int32 = _ring_mod3(k - 1, max_k_blocks)
                        bar_so_ready.wait(0, T.bitwise_and(k - 1, T.int32(1)))
                        T.ptx.tcgen05.fence.after_thread_sync()
                        # CUDA phase1.cuh:521-523 S(i-1) x V(i-1) MMA.
                        tiled_mma_o_accumulate[0] = T.if_then_else(k == 1, T.uint32(0), T.uint32(1))
                        for sv_k in T.unroll(B_TOPK // 16):
                            s_desc: T.uint64
                            v_desc: T.uint64
                            v_desc_hi: T.uint64
                            T.ptx.tcgen05.encode_matrix_descriptor(
                                T.address_of(s_desc),
                                s_q_rope_s.ptr_to([sv_k * 16 * B_H]),
                                ldo=S_DESC_LDO,
                                sdo=S_DESC_SDO,
                                swizzle=0,
                            )
                            T.ptx.tcgen05.encode_matrix_descriptor(
                                T.address_of(v_desc),
                                k_nope.ptr_to([cur_buf_prev, sv_k * 16, 0]),
                                ldo=V_DESC_LDO,
                                sdo=V_DESC_SDO,
                                swizzle=3,
                            )
                            T.evaluate(
                                _tcgen05_mma_ws_ss(
                                    T.uint32(tiled_mma_o_col),
                                    s_desc,
                                    v_desc,
                                    desc_i_o,
                                    tiled_mma_o_accumulate[0],
                                )
                            )
                            T.ptx.tcgen05.encode_matrix_descriptor(
                                T.address_of(v_desc_hi),
                                k_nope.ptr_to([cur_buf_prev, sv_k * 16, D_V // 2]),
                                ldo=V_DESC_LDO,
                                sdo=V_DESC_SDO,
                                swizzle=3,
                            )
                            T.evaluate(
                                _tcgen05_mma_ws_ss(
                                    T.uint32(tiled_mma_o_col + 128),
                                    s_desc,
                                    v_desc_hi,
                                    desc_i_o,
                                    tiled_mma_o_accumulate[0],
                                )
                            )
                            tiled_mma_o_accumulate[0] = T.uint32(1)
                        bar_sv_done.arrive(cur_buf_prev)

        elif warp_idx == 9:
            # CUDA common_subroutine.h:14-44 load_indices_and_generate_mask.
            if lane_idx < B_TOPK // 8:
                lane_indices = T.alloc_local((8,), "int32")
                for k in T.serial(0, num_k_blocks, unroll=False):
                    abs_pos_start: T.let = k * B_TOPK
                    T.evaluate(
                        _ldg_256_indices(
                            lane_indices.ptr_to([0]),
                            lane_indices.ptr_to([1]),
                            lane_indices.ptr_to([2]),
                            lane_indices.ptr_to([3]),
                            lane_indices.ptr_to([4]),
                            lane_indices.ptr_to([5]),
                            lane_indices.ptr_to([6]),
                            lane_indices.ptr_to([7]),
                            indices.ptr_to([g_indices_base + k * B_TOPK + lane_idx * 8]),
                        )
                    )
                    valid0: T.let = (
                        (lane_indices[0] >= 0)
                        & (lane_indices[0] < s_kv)
                        & (abs_pos_start + lane_idx * 8 < topk_len)
                    )
                    valid1: T.let = (
                        (lane_indices[1] >= 0)
                        & (lane_indices[1] < s_kv)
                        & (abs_pos_start + lane_idx * 8 + 1 < topk_len)
                    )
                    valid2: T.let = (
                        (lane_indices[2] >= 0)
                        & (lane_indices[2] < s_kv)
                        & (abs_pos_start + lane_idx * 8 + 2 < topk_len)
                    )
                    valid3: T.let = (
                        (lane_indices[3] >= 0)
                        & (lane_indices[3] < s_kv)
                        & (abs_pos_start + lane_idx * 8 + 3 < topk_len)
                    )
                    valid4: T.let = (
                        (lane_indices[4] >= 0)
                        & (lane_indices[4] < s_kv)
                        & (abs_pos_start + lane_idx * 8 + 4 < topk_len)
                    )
                    valid5: T.let = (
                        (lane_indices[5] >= 0)
                        & (lane_indices[5] < s_kv)
                        & (abs_pos_start + lane_idx * 8 + 5 < topk_len)
                    )
                    valid6: T.let = (
                        (lane_indices[6] >= 0)
                        & (lane_indices[6] < s_kv)
                        & (abs_pos_start + lane_idx * 8 + 6 < topk_len)
                    )
                    valid7: T.let = (
                        (lane_indices[7] >= 0)
                        & (lane_indices[7] < s_kv)
                        & (abs_pos_start + lane_idx * 8 + 7 < topk_len)
                    )
                    is_ks_valid_mask: T.int8 = T.cast(
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
                        "int8",
                    )

                    cur_buf: T.int32 = _ring_mod3(k, max_k_blocks)
                    cur_phase: T.int32 = _ring_phase_parity(k, max_k_blocks)
                    bar_k_valid_free.wait(cur_buf, T.bitwise_xor(cur_phase, T.int32(1)))
                    is_k_valid[cur_buf, lane_idx] = is_ks_valid_mask
                    bar_k_valid_ready.arrive(cur_buf)

        elif (warp_idx == 10) | (warp_idx == 11):
            if have_rope:
                thread_idx: T.let = (warp_idx - 10) * 32 + lane_idx
                group_size = T.meta_var(8)
                num_groups = T.meta_var(64 // group_size)
                rows_per_thread = T.meta_var(B_TOPK // num_groups)
                group_idx: T.let = thread_idx // group_size
                idx_in_group: T.let = thread_idx % group_size
                for k in T.serial(0, num_k_blocks, unroll=False):
                    rope_indices = T.alloc_local((rows_per_thread,), "int32")
                    for local_row in T.unroll(rows_per_thread):
                        rope_indices[local_row] = T.cuda.ldg(
                            indices.ptr_to(
                                [g_indices_base + k * B_TOPK + group_idx + local_row * num_groups]
                            ),
                            "int32",
                        )
                    bar_qk_rope_done.wait(
                        0, T.bitwise_xor(T.bitwise_and(k, T.int32(1)), T.int32(1))
                    )
                    for local_row in T.unroll(rows_per_thread):
                        index = rope_indices[local_row]
                        is_valid_index: T.let = (index >= 0) & (index < s_kv)
                        T.evaluate(
                            T.ptx.cp_async(
                                k_rope.ptr_to(
                                    [group_idx + local_row * num_groups, idx_in_group * 8]
                                ),
                                kv.ptr_to([index * stride_kv_s_kv + D_V + idx_in_group * 8]),
                                16,
                                prefetch_size=128,
                                predicate=is_valid_index,
                                fill_mode="zero",
                            )
                        )
                    T.evaluate(_cpasync_barrier_arrive_noinc(bar_kv_rope_ready.ptr_to([0])))


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
    return kernel.with_attr("tirx.kernel_launch_params", list(HEAD64_LAUNCH_PARAM_TAGS))


def run_test(**kwargs: Any) -> None:
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for sparse FlashMLA phase1")

    from tirx_kernels.runner import compile_kernel

    case = prepare_data(**kwargs)
    cfg: SparseFlashMLAPrefillHead64Config = case["config"]
    prim_func = get_kernel(**kwargs)
    ex = compile_kernel(prim_func)
    launches = _build_tirx_launches(case)
    _run_tirx_launches(ex, launches, output_case=case)
    torch.cuda.synchronize()
    ref_out, ref_max_logits, ref_lse = _reference_sparse_prefill(case)
    torch.testing.assert_close(case["out"], ref_out, rtol=3.01 / 128, atol=5e-3)
    torch.testing.assert_close(case["max_logits"], ref_max_logits, rtol=2.01 / 65536, atol=1e-6)
    torch.testing.assert_close(case["lse"], ref_lse, rtol=2.01 / 65536, atol=1e-6)
    cfg.validate()


def run_bench(
    *, warmup: int = 10, repeat: int = 30, timer: str = "proton", **kwargs: Any
) -> dict[str, Any]:
    _rounds = kwargs.pop("rounds", 1)
    _round_cooldown_s = kwargs.pop("round_cooldown_s", 1.0)
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for sparse FlashMLA phase1 benchmark")

    from tirx_kernels.runner import compile_kernel
    from tvm.tirx.bench import bench, tensor_bytes

    prim_func = get_kernel(**kwargs)
    ex = compile_kernel(prim_func)

    def make_input() -> tuple[dict[str, Any], int]:
        case = prepare_data(**kwargs)
        launches = _build_tirx_launches(case)
        case["launches"] = launches
        input_bytes = tensor_bytes(*_tirx_benchmark_tensors(case, launches))
        return case, input_bytes

    from tirx_kernels.attention._flashmla_bench import flashmla_reference_builder

    return bench(
        {"tirx": lambda case: _run_tirx_launches(ex, case["launches"])},
        make_input,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=_rounds,
        round_cooldown_s=_round_cooldown_s,
        proton_name=KERNEL_META["name"],
        references={"flashmla": flashmla_reference_builder},
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

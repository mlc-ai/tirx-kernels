from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any
from unittest import SkipTest

import torch

from tvm.script import tirx as T
from tvm.tirx.lang.pipeline import MBarrier, TCGen05Bar, TMABar

B_H = 128
B_TOPK = 128
D_V = 512
NUM_BUFS = 2
NUM_THREADS = 512
MAX_INIT_VAL = -1.0e30
LOG_2_E = math.log2(math.e)
LN_2 = math.log(2.0)

HEAD128_LAUNCH_PARAM_TAGS = (
    "blockIdx.x",
    "clusterCtaIdx.x",
    "threadIdx.x",
    "tirx.use_dyn_shared_memory",
)

TMEM_COL_O = 0
TMEM_COL_P = 256
TMEM_COL_Q = 320

NAMED_BARRIER_WG0_SYNC = 0

BF16_BYTES = 2
D_TQ = 384
NUM_TQ_TILES = D_TQ // 64
Q_FULL_DESC_SDO = 64
SQ_DESC_SDO = 64
K_DESC_SDO = 64
K_MAJOR_SWIZZLED_DESC_LDO = 1
S_DESC_LDO = 64
S_DESC_SDO = 8
V_DESC_LDO = 1024
V_DESC_SDO = 64
O_DESC_SDO = 64
P_TMEM_ELEMENTS = B_TOPK // 2
B_EPI = 64
WG1_NUM_WARPS = 4
WG1_NUM_LOCAL_ROWS_PER_WARP = (B_TOPK // 2) // 4 // WG1_NUM_WARPS
WG2_NUM_WARPS = 4
WG2_NUM_LOCAL_ROWS_PER_PART = (B_TOPK // 2) // 4 // WG2_NUM_WARPS

_IMPLEMENTATION_COMPLETE = True


@dataclass(frozen=True)
class SparseFlashMLAPrefillHead128Config:
    label: str
    s_q: int
    s_kv: int
    topk: int
    d_qk: int
    h_q: int = B_H
    h_kv: int = 1
    d_v: int = D_V
    have_attn_sink: bool = False
    have_topk_length: bool = False
    inject_invalid_indices: bool = False
    seed: int = 0

    def validate(self) -> None:
        if self.h_q != B_H:
            raise ValueError("head128 regular phase1 requires h_q == 128")
        if self.h_kv != 1:
            raise ValueError("head128 regular phase1 requires h_kv == 1")
        if self.d_qk not in (512, 576):
            raise ValueError("d_qk must be 512 or 576")
        if self.d_v != D_V:
            raise ValueError("d_v must be 512")
        if self.topk % B_TOPK != 0:
            raise ValueError("topk must be a multiple of 128")


CONFIGS = [
    {
        "label": "regular_dqk512_s1_kv2048_topk1408",
        "s_q": 1,
        "s_kv": 2048,
        "topk": 1408,
        "d_qk": 512,
    },
    {
        "label": "regular_dqk576_s1_kv2048_topk1408",
        "s_q": 1,
        "s_kv": 2048,
        "topk": 1408,
        "d_qk": 576,
    },
    {
        "label": "regular_features_dqk512_s17_kv4096_topk1536",
        "s_q": 17,
        "s_kv": 4096,
        "topk": 1536,
        "d_qk": 512,
        "have_attn_sink": True,
        "have_topk_length": True,
    },
    {
        "label": "regular_features_dqk576_s17_kv4096_topk1536",
        "s_q": 17,
        "s_kv": 4096,
        "topk": 1536,
        "d_qk": 576,
        "have_attn_sink": True,
        "have_topk_length": True,
    },
    {
        "label": "regular_invalid_indices_dqk512_s3_kv2304_topk1408",
        "s_q": 3,
        "s_kv": 2304,
        "topk": 1408,
        "d_qk": 512,
        "inject_invalid_indices": True,
    },
    {
        "label": "regular_invalid_indices_dqk576_s3_kv2304_topk1408",
        "s_q": 3,
        "s_kv": 2304,
        "topk": 1408,
        "d_qk": 576,
        "inject_invalid_indices": True,
    },
]

BENCH_CONFIGS = [
    {
        "label": f"bench_regular_dqk{d_qk}_hq128_s4096_kv{s_kv}_topk2048",
        "s_q": 4096,
        "s_kv": s_kv,
        "topk": 2048,
        "d_qk": d_qk,
        "h_q": B_H,
        "have_attn_sink": True,
    }
    for d_qk in (512, 576)
    for s_kv in (8192, 32768, 65536)
]

KERNEL_META = {
    "name": "sparse_flashmla_prefill_head128_phase1",
    "category": "flashmla",
    "compute_capability": 10,
}


def _cfg(**kwargs: Any) -> SparseFlashMLAPrefillHead128Config:
    cfg_fields = {field.name for field in fields(SparseFlashMLAPrefillHead128Config)}
    cfg_kwargs = {key: value for key, value in kwargs.items() if key in cfg_fields}
    if "label" not in cfg_kwargs:
        cfg_kwargs["label"] = "custom"
    cfg = SparseFlashMLAPrefillHead128Config(**cfg_kwargs)
    cfg.validate()
    return cfg


def _flashmla_regular_dispatch_reason(cfg: SparseFlashMLAPrefillHead128Config) -> str:
    if cfg.h_q != B_H:
        return "out_of_scope: h_q != 128 dispatches to head64 or unsupported path"
    if cfg.d_qk == 512 and cfg.topk <= 1280:
        return "out_of_scope: sm100 head128 D_QK=512 topk<=1280 dispatches small-topk"
    return f"regular: sm100 head128 run_fwd_phase1_kernel<{cfg.d_qk}>"


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
        "dispatch_reason": _flashmla_regular_dispatch_reason(cfg),
    }


def _reference_sparse_prefill(
    case: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg: SparseFlashMLAPrefillHead128Config = case["config"]
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

    cfg: SparseFlashMLAPrefillHead128Config = case["config"]
    q = case["q"]
    kv = case["kv"]
    out = case["out"]
    encode_tensormap = tvm.get_global_func("runtime.cuTensorMapEncodeTiled")

    return {
        "tensor_map_q": _encode_tma_desc(
            tensor=q,
            global_shape=(cfg.d_qk, cfg.h_q, cfg.s_q),
            global_strides=(int(q.stride(1)), int(q.stride(0))),
            box_dim=(64, B_H // 2, 1),
            swizzle_mode=128,
        ),
        "tensor_map_o": _encode_tma_desc(
            tensor=out,
            global_shape=(cfg.d_v, cfg.h_q, cfg.s_q),
            global_strides=(int(out.stride(1)), int(out.stride(0))),
            box_dim=(64, B_H // 2, 1),
            swizzle_mode=128,
        ),
        "tensor_map_kv": _encode_tma_2d_desc(
            encode_tensormap=encode_tensormap,
            tensor=kv,
            gmem_inner_dim=cfg.d_qk,
            gmem_outer_dim=cfg.s_kv,
            smem_inner_dim=64,
            smem_outer_dim=1,
            gmem_outer_stride=int(kv.stride(0)),
            swizzle_mode=128,
        ),
    }


def _make_tirx_launch(case: dict[str, Any]) -> dict[str, Any]:
    import ctypes

    tensor_maps = _build_tirx_tensor_maps(case)
    cfg: SparseFlashMLAPrefillHead128Config = case["config"]
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
            tensor_maps["tensor_map_o"].ptr,
            tensor_maps["tensor_map_kv"].ptr,
        ),
    }


def _build_tirx_launches(case: dict[str, Any]) -> list[dict[str, Any]]:
    return [_make_tirx_launch(case)]


def _run_tirx_launches(
    executable: Any, launches: list[dict[str, Any]], *, output_case: dict[str, Any] | None = None
) -> None:
    for launch in launches:
        executable(*launch["args"])


def _mbarrier_complete_tx(bar_ptr: Any, dst_cta_id: Any, transaction_bytes: Any, pred: Any) -> Any:
    func_name = "sparse_flashmla_head128_mbarrier_complete_tx"
    source_code = f"""
__device__ __forceinline__ void {func_name}(
    void* bar_ptr, unsigned int dst_cta_id, unsigned int transaction_bytes, unsigned int pred) {{
  unsigned int smem_addr;
  asm volatile(
    "{{\\n\\t"
    ".reg .u64 smem_addr64;\\n\\t"
    "cvta.to.shared.u64 smem_addr64, %1;\\n\\t"
    "cvt.u32.u64 %0, smem_addr64;\\n\\t"
    "}}\\n"
    : "=r"(smem_addr) : "l"(bar_ptr));
  unsigned int remote_addr;
  asm volatile(
    "mapa.shared::cluster.u32 %0, %1, %2;\\n"
    : "=r"(remote_addr) : "r"(smem_addr), "r"(dst_cta_id));
  asm volatile(
    "{{\\n\\t"
    ".reg .pred p;\\n\\t"
    "setp.eq.u32 p, %2, 1;\\n\\t"
    "@p mbarrier.complete_tx.shared::cluster.relaxed.cluster.b64 [%1], %0;\\n\\t"
    "}}\\n"
    :: "r"(transaction_bytes), "r"(remote_addr), "r"(pred) : "memory");
}}
"""
    return T.cuda.func_call(
        func_name,
        bar_ptr,
        dst_cta_id,
        transaction_bytes,
        pred,
        source_code=source_code,
        return_type="void",
    )


def _ldg_int4_indices(dst0: Any, dst1: Any, dst2: Any, dst3: Any, src_ptr: Any) -> Any:
    func_name = "sparse_flashmla_head128_ldg_int4_indices"
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


def _int4_max(x: Any, y: Any, z: Any, w: Any) -> Any:
    return T.max(T.max(x, y), T.max(z, w))


def _int4_min(x: Any, y: Any, z: Any, w: Any) -> Any:
    return T.min(T.min(x, y), T.min(z, w))


def _tmem_ld_32dp32bNx_btopk_half(tmem_col: Any, p_float: Any) -> Any:
    # CUDA helper boundary: ku::tmem_ld_32dp32bNx<B_TOPK/2>(tmem_col, p).
    return T.ptx.tcgen05.ld(
        T.uint32(0),
        p_float[0],
        p_float[1],
        p_float[2],
        p_float[3],
        p_float[4],
        p_float[5],
        p_float[6],
        p_float[7],
        p_float[8],
        p_float[9],
        p_float[10],
        p_float[11],
        p_float[12],
        p_float[13],
        p_float[14],
        p_float[15],
        p_float[16],
        p_float[17],
        p_float[18],
        p_float[19],
        p_float[20],
        p_float[21],
        p_float[22],
        p_float[23],
        p_float[24],
        p_float[25],
        p_float[26],
        p_float[27],
        p_float[28],
        p_float[29],
        p_float[30],
        p_float[31],
        p_float[32],
        p_float[33],
        p_float[34],
        p_float[35],
        p_float[36],
        p_float[37],
        p_float[38],
        p_float[39],
        p_float[40],
        p_float[41],
        p_float[42],
        p_float[43],
        p_float[44],
        p_float[45],
        p_float[46],
        p_float[47],
        p_float[48],
        p_float[49],
        p_float[50],
        p_float[51],
        p_float[52],
        p_float[53],
        p_float[54],
        p_float[55],
        p_float[56],
        p_float[57],
        p_float[58],
        p_float[59],
        p_float[60],
        p_float[61],
        p_float[62],
        p_float[63],
        shape="32x32b",
        num=P_TMEM_ELEMENTS,
        col=tmem_col,
    )


def _tma_3d_cta_group2_nosplit(
    dst_ptr: Any,
    bar_ptr: Any,
    tensor_map_ptr: Any,
    coord0: Any,
    coord1: Any,
    coord2: Any,
    cache_hint: Any,
) -> Any:
    func_name = "sparse_flashmla_head128_tma_3d_cta_group2_nosplit"
    source_code = f"""
__device__ __forceinline__ void {func_name}(
    void* dst_ptr, void* bar_ptr, unsigned long long tensor_map_addr,
    int coord0, int coord1, int coord2, unsigned long long cache_hint) {{
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
    "cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global"
    ".mbarrier::complete_tx::bytes.L2::cache_hint "
    "[%0], [%1, {{%3, %4, %5}}], [%2], %6;\\n"
    :
    : "r"(smem_addr), "l"(tensor_map_addr), "r"(mbar_addr),
      "r"(coord0), "r"(coord1), "r"(coord2), "l"(cache_hint)
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
        cache_hint,
        source_code=source_code,
        return_type="void",
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
    func_name = "sparse_flashmla_head128_ldg_256_indices"
    source_code = f"""
__device__ __forceinline__ void {func_name}(
    int* dst0, int* dst1, int* dst2, int* dst3,
    int* dst4, int* dst5, int* dst6, int* dst7, const int* src_ptr) {{
  int raw0, raw1, raw2, raw3, raw4, raw5, raw6, raw7;
  asm volatile(
    "ld.global.nc.L1::evict_normal.L2::evict_normal.L2::256B.v8.s32 "
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


def _canonical_warp_idx_sync() -> Any:
    func_name = "sparse_flashmla_head128_canonical_warp_idx_sync"
    source_code = f"""
__device__ __forceinline__ int {func_name}() {{
  return __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
}}
"""
    return T.cuda.func_call(func_name, source_code=source_code, return_type="int32")


def _shfl_sync_i32(value: Any) -> Any:
    func_name = "sparse_flashmla_head128_shfl_sync_i32"
    source_code = f"""
__device__ __forceinline__ int {func_name}(int value) {{
  return __shfl_sync(0xffffffff, value, 0);
}}
"""
    return T.cuda.func_call(func_name, value, source_code=source_code, return_type="int32")


def _ld_shared_u32(src_ptr: Any) -> Any:
    func_name = "sparse_flashmla_head128_ld_shared_u32"
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


def _ldg_i32_at(base_ptr: Any, idx: Any) -> Any:
    func_name = "sparse_flashmla_head128_ldg_i32_at"
    source_code = f"""
__device__ __forceinline__ int {func_name}(const void* base_ptr, int idx) {{
  auto ptr = reinterpret_cast<const int*>(base_ptr) + idx;
  return __ldg(ptr);
}}
"""
    return T.cuda.func_call(func_name, base_ptr, idx, source_code=source_code, return_type="int32")


def _ldg_f32_at(base_ptr: Any, idx: Any) -> Any:
    func_name = "sparse_flashmla_head128_ldg_f32_at"
    source_code = f"""
__device__ __forceinline__ float {func_name}(const void* base_ptr, int idx) {{
  auto ptr = reinterpret_cast<const float*>(base_ptr) + idx;
  return __ldg(ptr);
}}
"""
    return T.cuda.func_call(
        func_name, base_ptr, idx, source_code=source_code, return_type="float32"
    )


def _tma_gather4_kv_cta_group2(
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
    func_name = "sparse_flashmla_head128_tma_gather4_kv_cta_group2"
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
  mbar_addr &= 0xFEFFFFFFu;
  asm volatile(
    "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
    ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint "
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


def _tcgen05_mma_ws_ts_2cta(
    d_tmem_addr: Any, a_tmem_addr: Any, b_desc: Any, i_desc: Any, scale_c: Any
) -> Any:
    func_name = "sparse_flashmla_head128_tcgen05_mma_ws_ts_2cta"
    source_code = f"""
__device__ __forceinline__ void {func_name}(
    unsigned int d_tmem_addr, unsigned int a_tmem_addr, unsigned long long b_desc,
    unsigned int i_desc, unsigned int scale_c) {{
  uint32_t mask[8] = {{0, 0, 0, 0, 0, 0, 0, 0}};
  asm volatile(
    "{{\\n\\t"
    ".reg .pred p;\\n\\t"
    "setp.ne.b32 p, %4, 0;\\n\\t"
    "tcgen05.mma.cta_group::2.kind::f16 "
    "[%0], [%1], %2, %3, {{%5, %6, %7, %8, %9, %10, %11, %12}}, p;\\n\\t"
    "}}\\n"
    :
    : "r"(d_tmem_addr), "r"(a_tmem_addr), "l"(b_desc), "r"(i_desc), "r"(scale_c),
      "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]),
      "r"(mask[4]), "r"(mask[5]), "r"(mask[6]), "r"(mask[7]));
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


def _tcgen05_mma_ws_ss_2cta(
    d_tmem_addr: Any, a_desc: Any, b_desc: Any, i_desc: Any, scale_c: Any
) -> Any:
    func_name = "sparse_flashmla_head128_tcgen05_mma_ws_ss_2cta"
    source_code = f"""
__device__ __forceinline__ void {func_name}(
    unsigned int d_tmem_addr, unsigned long long a_desc, unsigned long long b_desc,
    unsigned int i_desc, unsigned int scale_c) {{
  uint32_t mask[8] = {{0, 0, 0, 0, 0, 0, 0, 0}};
  asm volatile(
    "{{\\n\\t"
    ".reg .pred p;\\n\\t"
    "setp.ne.b32 p, %4, 0;\\n\\t"
    "tcgen05.mma.cta_group::2.kind::f16 "
    "[%0], %1, %2, %3, {{%5, %6, %7, %8, %9, %10, %11, %12}}, p;\\n\\t"
    "}}\\n"
    :
    : "r"(d_tmem_addr), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(scale_c),
      "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]),
      "r"(mask[4]), "r"(mask[5]), "r"(mask[6]), "r"(mask[7]));
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
    func_name = "sparse_flashmla_head128_fdividef"
    source_code = f"""
__device__ __forceinline__ float {func_name}(float x, float y) {{
  return __fdividef(x, y);
}}
"""
    return T.cuda.func_call(func_name, x, y, source_code=source_code, return_type="float32")


def _fma_f32x2(a: Any, b: Any, c: Any) -> Any:
    func_name = "sparse_flashmla_head128_fma_f32x2"
    source_code = f"""
__device__ __forceinline__ unsigned long long {func_name}(
    unsigned long long a, unsigned long long b, unsigned long long c) {{
  unsigned long long d;
  asm volatile(
      "fma.rn.f32x2 %0, %1, %2, %3;\\n"
      : "=l"(d) : "l"(a), "l"(b), "l"(c));
  return d;
}}
"""
    return T.cuda.func_call(func_name, a, b, c, source_code=source_code, return_type="uint64")


def _mul_f32x2(a: Any, b: Any) -> Any:
    func_name = "sparse_flashmla_head128_mul_f32x2"
    source_code = f"""
__device__ __forceinline__ unsigned long long {func_name}(
    unsigned long long a, unsigned long long b) {{
  unsigned long long c;
  asm volatile(
      "mul.f32x2 %0, %1, %2;\\n"
      : "=l"(c) : "l"(a), "l"(b));
  return c;
}}
"""
    return T.cuda.func_call(func_name, a, b, source_code=source_code, return_type="uint64")


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
    tensor_map_o: T.TensorMap(),
    tensor_map_kv: T.TensorMap(),
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
    # CUDA_TRANSCRIBE_START: run_fwd_phase1_kernel line 622, then sparse_attn_fwd_kernel_devfunc line 68.
    # Transcription note: match upstream FlashMLA phase1's one CTA pair per query-row launch.
    # Transcription note: preserve upstream source-order roles and mixed TMA/MMA/softmax warp layout.
    block_idx = T.cta_id([2 * s_q])
    T.cta_id_in_cluster([2])
    cta_idx: T.let = block_idx % 2
    s_q_idx: T.let = block_idx // 2
    thread_idx = T.thread_id([NUM_THREADS])
    warp_idx: T.let = _canonical_warp_idx_sync()
    lane_idx: T.let = thread_idx % 32
    topk_len: T.let = _ldg_i32_at(topk_length, s_q_idx) if have_topk_length else topk
    num_k_blocks: T.let = T.max((topk_len + B_TOPK - 1) // B_TOPK, 1)
    warpgroup_idx: T.let = _shfl_sync_i32(thread_idx // 128)
    idx_in_warpgroup: T.let = thread_idx % 128
    d_sq = T.meta_var(d_qk - D_TQ)
    num_sq_tiles = T.meta_var((d_qk - D_TQ) // 64)
    num_qk_tiles = T.meta_var(d_qk // 64)
    shared_u_elems = T.meta_var((B_H // 2) * d_sq + (D_V // 2) * B_TOPK + (B_TOPK // 2) * d_qk)

    if thread_idx == 0:
        T.evaluate(T.ptx.prefetch_tensormap(T.address_of(tensor_map_q)))
        T.evaluate(T.ptx.prefetch_tensormap(T.address_of(tensor_map_o)))
        T.evaluate(T.ptx.prefetch_tensormap(T.address_of(tensor_map_kv)))

    # CUDA phase1.cuh:84-90, config.h:93-118.  Preserve SharedMemoryPlan's
    # union offsets: q_full, {sq, v, k}, and o alias the same base.
    pool = T.SMEMPool()
    u_base = T.meta_var(pool.offset)
    q_full = pool.alloc_mma((B_H // 2, d_qk), "bfloat16")
    pool.move_base_to(u_base)
    sq_smem = pool.alloc_mma((B_H // 2, d_sq), "bfloat16")
    pool.move_base_to(u_base + (B_H // 2) * d_sq * BF16_BYTES)
    v_smem = pool.alloc_mma((D_V // 2, B_TOPK), "bfloat16")
    pool.move_base_to(u_base + ((B_H // 2) * d_sq + (D_V // 2) * B_TOPK) * BF16_BYTES)
    k_smem = pool.alloc_mma((B_TOPK // 2, d_qk), "bfloat16")
    pool.move_base_to(u_base)
    o_smem = pool.alloc_mma((B_H // 2, D_V), "bfloat16")
    pool.move_base_to(u_base + shared_u_elems * BF16_BYTES)
    s_smem = pool.alloc(((B_H // 2) * B_TOPK,), "bfloat16")
    p_smem = pool.alloc(((B_H // 2) * B_TOPK,), "float32")
    is_k_valid = pool.alloc((NUM_BUFS, B_TOPK // 8), "int8")
    bar_prologue_q = TMABar(pool, 1)
    bar_prologue_utccp = TCGen05Bar(pool, 1)
    bar_qk_part_done = TCGen05Bar(pool, NUM_BUFS)
    bar_qk_done = TCGen05Bar(pool, NUM_BUFS)
    bar_sv_part_done = TCGen05Bar(pool, NUM_BUFS)
    bar_sv_done = TCGen05Bar(pool, NUM_BUFS)
    bar_k_part0_ready = TMABar(pool, NUM_BUFS)
    bar_k_part1_ready = TMABar(pool, NUM_BUFS)
    bar_v_part0_ready = TMABar(pool, NUM_BUFS)
    bar_v_part1_ready = TMABar(pool, NUM_BUFS)
    bar_p_free = MBarrier(pool, NUM_BUFS)
    bar_so_ready = MBarrier(pool, NUM_BUFS)
    bar_k_valid_ready = MBarrier(pool, NUM_BUFS)
    bar_k_valid_free = MBarrier(pool, NUM_BUFS)
    tmem_start_addr = pool.alloc((1,), "uint32", align=4)
    rowwise_max_buf = pool.alloc((128,), "float32")
    rowwise_li_buf = pool.alloc((128,), "float32")
    pool.commit()

    g_indices_base: T.let = s_q_idx * stride_indices_s_q
    tP_col = T.meta_var(TMEM_COL_P)
    tQr_col = T.meta_var(TMEM_COL_Q)
    tO_col = T.meta_var(TMEM_COL_O)
    tiled_mma_p_accumulate = T.alloc_local((1,), "uint32")
    tiled_mma_o_accumulate = T.alloc_local((1,), "uint32")
    tiled_mma_p_accumulate[0] = T.uint32(0)
    tiled_mma_o_accumulate[0] = T.uint32(0)

    # CUDA phase1.cuh:87-146.  Warp 0 owns barrier init, Q TMA launch,
    # and the cta_group::2 TMEM allocation.
    if warp_idx == 0:
        if T.ptx.elect_sync():
            bar_prologue_q.init(1)
            bar_prologue_utccp.init(1)
            for init_stage in T.unroll(NUM_BUFS):
                T.ptx.mbarrier.init(bar_qk_part_done.ptr_to([init_stage]), 1)
                T.ptx.mbarrier.init(bar_qk_done.ptr_to([init_stage]), 1)
                T.ptx.mbarrier.init(bar_sv_part_done.ptr_to([init_stage]), 1)
                T.ptx.mbarrier.init(bar_sv_done.ptr_to([init_stage]), 1)
                T.ptx.mbarrier.init(bar_k_part0_ready.ptr_to([init_stage]), 1)
                T.ptx.mbarrier.init(bar_k_part1_ready.ptr_to([init_stage]), 1)
                T.ptx.mbarrier.init(bar_v_part0_ready.ptr_to([init_stage]), 1)
                T.ptx.mbarrier.init(bar_v_part1_ready.ptr_to([init_stage]), 1)
                T.ptx.mbarrier.init(bar_p_free.ptr_to([init_stage]), 128 * 2)
                T.ptx.mbarrier.init(bar_so_ready.ptr_to([init_stage]), 128 * 2)
                T.ptx.mbarrier.init(bar_k_valid_ready.ptr_to([init_stage]), 16)
                T.ptx.mbarrier.init(bar_k_valid_free.ptr_to([init_stage]), 128)
            T.ptx.fence.mbarrier_init()

    T.cuda.cluster_sync()

    if warp_idx == 0:
        if T.ptx.elect_sync():
            for q_tma_tile in T.unroll(num_qk_tiles):
                T.evaluate(
                    _tma_3d_cta_group2_nosplit(
                        q_full.ptr_to([0, q_tma_tile * 64]),
                        bar_prologue_q.ptr_to([0]),
                        T.address_of(tensor_map_q),
                        q_tma_tile * 64,
                        cta_idx * (B_H // 2),
                        s_q_idx,
                        T.uint64(0x12F0000000000000),
                    )
                )

        T.ptx.tcgen05.alloc(T.address_of(tmem_start_addr[0]), n_cols=512, cta_group=2)
        T.cuda.trap_when_assert_failed(tmem_start_addr[0] == T.uint32(0))
        T.ptx.tcgen05.relinquish_alloc_permit(cta_group=2)

    T.cuda.cta_sync()

    if warpgroup_idx == 0:
        # CUDA phase1.cuh:150-386.  Scale/exp warpgroup and epilogue.
        T.ptx.setmaxnreg(True, 144)
        mi = T.local_scalar("float32")
        mi = MAX_INIT_VAL
        li = T.local_scalar("float32")
        li = 0.0
        real_mi = T.local_scalar("float32")
        real_mi = T.float32(-float("inf"))
        scale_pair: T.let = T.cuda.make_float2(sm_scale_div_log2, sm_scale_div_log2)

        for k in T.serial(0, num_k_blocks, unroll=False):
            cur_buf: T.let = k % NUM_BUFS
            cur_phase: T.let = (k // NUM_BUFS) & 1
            bar_qk_done.wait(cur_buf, cur_phase)
            T.ptx.tcgen05.fence.after_thread_sync()

            # CUDA source keeps `float2 p[32]` and aliases it as
            # `float *p_float = reinterpret_cast<float *>(p)`. The upstream
            # helper immediately casts the same storage to uint32_t* for the
            # tcgen05.ld operands, so TIRx keeps the raw 32-bit lane payloads and
            # applies the p_float view only at float use sites.
            p = T.alloc_local((P_TMEM_ELEMENTS,), "uint32")
            _tmem_ld_32dp32bNx_btopk_half(tP_col, p)
            T.ptx.tcgen05.wait.ld()
            T.ptx.tcgen05.fence.before_thread_sync()
            bar_p_free.arrive(cur_buf, cta_id=T.uint32(0))

            bar_k_valid_ready.wait(cur_buf, cur_phase)
            valid_word_offset: T.let = T.if_then_else(idx_in_warpgroup >= 64, B_TOPK // 8 // 2, 0)
            is_k_valid_lo: T.let = _ld_shared_u32(is_k_valid.ptr_to([cur_buf, valid_word_offset]))
            is_k_valid_hi: T.let = _ld_shared_u32(
                is_k_valid.ptr_to([cur_buf, valid_word_offset + 4])
            )
            for p_i in T.unroll(P_TMEM_ELEMENTS // 2):
                invalid_p_predicate: T.let = T.bitwise_and(
                    T.shift_right(is_k_valid_lo, T.uint32(p_i)), T.uint32(1)
                ) == T.uint32(0)
                p[p_i] = T.if_then_else(invalid_p_predicate, T.uint32(0xFF800000), p[p_i])
            for p_i in T.unroll(P_TMEM_ELEMENTS // 2):
                invalid_p_predicate: T.let = T.bitwise_and(
                    T.shift_right(is_k_valid_hi, T.uint32(p_i)), T.uint32(1)
                ) == T.uint32(0)
                p[p_i + P_TMEM_ELEMENTS // 2] = T.if_then_else(
                    invalid_p_predicate, T.uint32(0xFF800000), p[p_i + P_TMEM_ELEMENTS // 2]
                )

            cur_pi_max = T.local_scalar("float32")
            cur_pi_max = T.float32(-float("inf"))
            for p_i in T.unroll(P_TMEM_ELEMENTS):
                cur_pi_max = T.max(cur_pi_max, T.cuda.uint_as_float(p[p_i]))
            cur_pi_max = cur_pi_max * sm_scale_div_log2
            bar_k_valid_free.arrive(cur_buf)

            T.ptx.bar.sync(NAMED_BARRIER_WG0_SYNC, 128)
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
            li = li * scale_for_old

            s_pack = T.alloc_local((P_TMEM_ELEMENTS // 2,), "uint32")
            neg_new_max_pair: T.let = T.cuda.make_float2(-new_max, -new_max)
            for s_i in T.unroll(P_TMEM_ELEMENTS // 2):
                p_pair: T.let = T.cuda.make_float2(
                    T.cuda.uint_as_float(p[s_i * 2]), T.cuda.uint_as_float(p[s_i * 2 + 1])
                )
                fma_pair: T.let = _fma_f32x2(p_pair, scale_pair, neg_new_max_pair)
                s_x: T.let = T.ptx.exp2(T.cuda.float2_x(fma_pair))
                s_y: T.let = T.ptx.exp2(T.cuda.float2_y(fma_pair))
                li = li + s_x + s_y
                s_pack[s_i] = T.cuda.float22bfloat162_rn(s_x, s_y)

            if k > 0:
                prev_buf: T.let = (k - 1) % NUM_BUFS
                prev_phase: T.let = ((k - 1) // NUM_BUFS) & 1
                bar_sv_done.wait(prev_buf, prev_phase)

            for s_store_i in T.unroll(P_TMEM_ELEMENTS // 8):
                s_store_offset: T.let = (
                    (idx_in_warpgroup % 64) * 8
                    + (idx_in_warpgroup // 64) * ((B_H // 2) * (B_TOPK // 2))
                    + s_store_i * (B_H // 2) * 8
                )
                T.evaluate(
                    T.ptx.st(
                        s_smem.ptr_to([s_store_offset]),
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
                        col=tO_col + chunk_idx * 32,
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
                        col=tO_col + chunk_idx * 32,
                    )
                    T.ptx.tcgen05.wait.st()
                T.ptx.tcgen05.fence.before_thread_sync()

            T.ptx.fence.proxy_async("shared::cta")
            bar_so_ready.arrive(cur_buf, cta_id=T.uint32(0))

        if real_mi == T.float32(-float("inf")):
            li = 0.0
            mi = T.float32(-float("inf"))

        rowwise_li_buf[idx_in_warpgroup] = li
        T.ptx.bar.sync(NAMED_BARRIER_WG0_SYNC, 128)
        li = li + rowwise_li_buf[idx_in_warpgroup ^ 64]

        if idx_in_warpgroup < B_H // 2:
            global_head: T.let = cta_idx * (B_H // 2) + idx_in_warpgroup
            cur_lse = T.local_scalar("float32")
            cur_lse_log: T.let = T.log(li)
            T.ptx.fma_f32(T.address_of(cur_lse), mi, LN_2, cur_lse_log)
            cur_lse = T.if_then_else(
                cur_lse == T.float32(-float("inf")), T.float32(float("inf")), cur_lse
            )
            max_logits[s_q_idx, global_head] = real_mi * LN_2
            lse[s_q_idx, global_head] = cur_lse

        last_k: T.let = num_k_blocks - 1
        last_buf: T.let = last_k % NUM_BUFS
        last_phase: T.let = (last_k // NUM_BUFS) & 1
        bar_sv_done.wait(last_buf, last_phase)
        T.ptx.tcgen05.fence.after_thread_sync()

        attn_sink_log2: T.let = (
            _ldg_f32_at(attn_sink, cta_idx * (B_H // 2) + (idx_in_warpgroup % 64)) * LOG_2_E
            if have_attn_sink
            else T.float32(-float("inf"))
        )
        output_scale = T.local_scalar("float32")
        output_scale = _fdividef(T.float32(1.0), li + T.ptx.exp2(attn_sink_log2 - mi))
        o_epi = T.alloc_local((B_EPI,), "float32")
        have_valid_indices: T.let = T.ptx.any_sync(T.uint32(0xFFFFFFFF), li != 0.0) != 0
        if not have_valid_indices:
            for o_zero_i in T.unroll(B_EPI):
                o_epi[o_zero_i] = 0.0
            output_scale = 1.0
        output_scale_pair: T.let = T.cuda.make_float2(output_scale, output_scale)
        for epi_k in T.unroll((D_V // 2) // B_EPI):
            if have_valid_indices:
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
                    col=tO_col + epi_k * B_EPI,
                )
                T.ptx.tcgen05.wait.ld()
            for o_i in T.unroll(B_EPI // 8):
                o_epi_bf16 = T.alloc_local((4,), "uint32")
                for o_j in T.unroll(4):
                    o_pair_idx: T.let = o_i * 8 + o_j * 2
                    o_pair: T.let = T.cuda.make_float2(o_epi[o_pair_idx], o_epi[o_pair_idx + 1])
                    o_epi_pair: T.let = _mul_f32x2(o_pair, output_scale_pair)
                    o_epi_bf16[o_j] = T.cuda.float22bfloat162_rn(
                        T.cuda.float2_x(o_epi_pair), T.cuda.float2_y(o_epi_pair)
                    )
                o_base_col: T.let = (idx_in_warpgroup // 64) * (D_V // 2) + epi_k * B_EPI + o_i * 8
                T.evaluate(
                    T.ptx.st(
                        o_smem.ptr_to([idx_in_warpgroup % 64, o_base_col]),
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
                            3,
                            o_smem.ptr_to([0, epi_k * B_EPI]),
                            T.address_of(tensor_map_o),
                            "",
                            T.uint32(epi_k * B_EPI),
                            cta_idx * (B_H // 2),
                            s_q_idx,
                        )
                    )
            if warp_idx == 1:
                if T.ptx.elect_sync():
                    epi_k2: T.let = epi_k + (D_V // B_EPI // 2)
                    T.evaluate(
                        T.ptx.cp_async.bulk.tensor.s2g(
                            3,
                            o_smem.ptr_to([0, epi_k2 * B_EPI]),
                            T.address_of(tensor_map_o),
                            "",
                            T.uint32(epi_k2 * B_EPI),
                            cta_idx * (B_H // 2),
                            s_q_idx,
                        )
                    )

        if warp_idx == 0:
            T.ptx.tcgen05.dealloc(T.uint32(0), n_cols=512, cta_group=2)

    elif warpgroup_idx == 1:
        # CUDA phase1.cuh:387-446.  K producer warpgroup.
        T.ptx.setmaxnreg(False, 96)
        wg1_warp_idx: T.let = warp_idx - 4
        if T.ptx.elect_sync():
            for k in T.serial(0, num_k_blocks, unroll=False):
                indices_int4 = T.alloc_local((WG1_NUM_LOCAL_ROWS_PER_WARP, 4), "int32")
                max_indices = T.local_scalar("int32")
                min_indices = T.local_scalar("int32")
                max_indices = -1
                min_indices = s_kv

                for local_row in T.unroll(WG1_NUM_LOCAL_ROWS_PER_WARP):
                    row_base: T.let = (
                        g_indices_base
                        + k * B_TOPK
                        + cta_idx * (B_TOPK // 2)
                        + (local_row * WG1_NUM_WARPS + wg1_warp_idx) * 4
                    )
                    T.evaluate(
                        _ldg_int4_indices(
                            indices_int4.ptr_to([local_row, 0]),
                            indices_int4.ptr_to([local_row, 1]),
                            indices_int4.ptr_to([local_row, 2]),
                            indices_int4.ptr_to([local_row, 3]),
                            indices.ptr_to([row_base]),
                        )
                    )
                    local_max: T.let = _int4_max(
                        indices_int4[local_row, 0],
                        indices_int4[local_row, 1],
                        indices_int4[local_row, 2],
                        indices_int4[local_row, 3],
                    )
                    local_min: T.let = _int4_min(
                        indices_int4[local_row, 0],
                        indices_int4[local_row, 1],
                        indices_int4[local_row, 2],
                        indices_int4[local_row, 3],
                    )
                    max_indices = T.max(max_indices, local_max)
                    min_indices = T.min(min_indices, local_min)

                is_all_rows_invalid: T.let = (min_indices == s_kv) | (max_indices == -1)
                should_skip_tma: T.let = is_all_rows_invalid & (k >= NUM_BUFS)
                cur_buf: T.let = k % NUM_BUFS
                cur_phase: T.let = (k // NUM_BUFS) & 1

                if k > 0:
                    prev_buf: T.let = (k - 1) % NUM_BUFS
                    prev_phase: T.let = ((k - 1) // NUM_BUFS) & 1
                    bar_qk_part_done.wait(prev_buf, prev_phase)
                if not should_skip_tma:
                    for local_row in T.unroll(WG1_NUM_LOCAL_ROWS_PER_WARP):
                        for local_col in T.unroll(num_sq_tiles):
                            raw_k_offset: T.let = (
                                wg1_warp_idx * 4 * 64
                                + local_row * (4 * WG1_NUM_WARPS) * 64
                                + local_col * ((B_TOPK // 2) * 64)
                            )
                            T.evaluate(
                                _tma_gather4_kv_cta_group2(
                                    k_smem.access_ptr("w", offset=raw_k_offset),
                                    bar_k_part0_ready.ptr_to([cur_buf]),
                                    T.address_of(tensor_map_kv),
                                    local_col * 64,
                                    indices_int4[local_row, 0],
                                    indices_int4[local_row, 1],
                                    indices_int4[local_row, 2],
                                    indices_int4[local_row, 3],
                                    T.uint64(0x14F0000000000000),
                                )
                            )
                else:
                    T.evaluate(
                        _mbarrier_complete_tx(
                            bar_k_part0_ready.ptr_to([cur_buf]),
                            T.uint32(0),
                            T.uint32(WG1_NUM_LOCAL_ROWS_PER_WARP * 4 * d_sq * BF16_BYTES),
                            T.uint32(1),
                        )
                    )

                if k > 0:
                    prev_buf: T.let = (k - 1) % NUM_BUFS
                    prev_phase: T.let = ((k - 1) // NUM_BUFS) & 1
                    bar_qk_done.wait(prev_buf, prev_phase)
                if not should_skip_tma:
                    for local_row in T.unroll(WG1_NUM_LOCAL_ROWS_PER_WARP):
                        for local_col_inner in T.unroll(num_qk_tiles - num_sq_tiles):
                            local_col: T.let = num_sq_tiles + local_col_inner
                            raw_k_offset: T.let = (
                                wg1_warp_idx * 4 * 64
                                + local_row * (4 * WG1_NUM_WARPS) * 64
                                + local_col * ((B_TOPK // 2) * 64)
                            )
                            T.evaluate(
                                _tma_gather4_kv_cta_group2(
                                    k_smem.access_ptr("w", offset=raw_k_offset),
                                    bar_k_part1_ready.ptr_to([cur_buf]),
                                    T.address_of(tensor_map_kv),
                                    local_col * 64,
                                    indices_int4[local_row, 0],
                                    indices_int4[local_row, 1],
                                    indices_int4[local_row, 2],
                                    indices_int4[local_row, 3],
                                    T.uint64(0x14F0000000000000),
                                )
                            )
                else:
                    T.evaluate(
                        _mbarrier_complete_tx(
                            bar_k_part1_ready.ptr_to([cur_buf]),
                            T.uint32(0),
                            T.uint32(WG1_NUM_LOCAL_ROWS_PER_WARP * 4 * D_TQ * BF16_BYTES),
                            T.uint32(1),
                        )
                    )

    elif warpgroup_idx == 2:
        # CUDA phase1.cuh:447-489.  V producer warpgroup.
        T.ptx.setmaxnreg(False, 96)
        wg2_warp_idx: T.let = warp_idx - 8
        if T.ptx.elect_sync():
            bar_prologue_utccp.wait(0, 0)
            for k in T.serial(0, num_k_blocks, unroll=False):
                cur_buf: T.let = k % NUM_BUFS
                cur_phase: T.let = (k // NUM_BUFS) & 1
                if k > 0:
                    prev_buf: T.let = (k - 1) % NUM_BUFS
                    prev_phase: T.let = ((k - 1) // NUM_BUFS) & 1
                    bar_sv_part_done.wait(prev_buf, prev_phase)
                for local_row in T.unroll(WG2_NUM_LOCAL_ROWS_PER_PART):
                    token_idxs = T.alloc_local((4,), "int32")
                    row_base: T.let = (
                        g_indices_base + k * B_TOPK + (local_row * WG2_NUM_WARPS + wg2_warp_idx) * 4
                    )
                    T.evaluate(
                        _ldg_int4_indices(
                            token_idxs.ptr_to([0]),
                            token_idxs.ptr_to([1]),
                            token_idxs.ptr_to([2]),
                            token_idxs.ptr_to([3]),
                            indices.ptr_to([row_base]),
                        )
                    )
                    for local_col in T.unroll((D_V // 2) // 64):
                        raw_v_offset: T.let = (
                            wg2_warp_idx * 4 * 64
                            + local_row * (4 * WG2_NUM_WARPS) * 64
                            + local_col * (B_TOPK * 64)
                        )
                        T.evaluate(
                            _tma_gather4_kv_cta_group2(
                                v_smem.access_ptr("w", offset=raw_v_offset),
                                bar_v_part0_ready.ptr_to([cur_buf]),
                                T.address_of(tensor_map_kv),
                                local_col * 64 + cta_idx * 256,
                                token_idxs[0],
                                token_idxs[1],
                                token_idxs[2],
                                token_idxs[3],
                                T.uint64(0x14F0000000000000),
                            )
                        )

                if k > 0:
                    prev_buf: T.let = (k - 1) % NUM_BUFS
                    prev_phase: T.let = ((k - 1) // NUM_BUFS) & 1
                    bar_sv_done.wait(prev_buf, prev_phase)
                for local_row_inner in T.unroll(WG2_NUM_LOCAL_ROWS_PER_PART):
                    local_row: T.let = WG2_NUM_LOCAL_ROWS_PER_PART + local_row_inner
                    token_idxs = T.alloc_local((4,), "int32")
                    row_base: T.let = (
                        g_indices_base + k * B_TOPK + (local_row * WG2_NUM_WARPS + wg2_warp_idx) * 4
                    )
                    T.evaluate(
                        _ldg_int4_indices(
                            token_idxs.ptr_to([0]),
                            token_idxs.ptr_to([1]),
                            token_idxs.ptr_to([2]),
                            token_idxs.ptr_to([3]),
                            indices.ptr_to([row_base]),
                        )
                    )
                    for local_col in T.unroll((D_V // 2) // 64):
                        raw_v_offset: T.let = (
                            wg2_warp_idx * 4 * 64
                            + local_row * (4 * WG2_NUM_WARPS) * 64
                            + local_col * (B_TOPK * 64)
                        )
                        T.evaluate(
                            _tma_gather4_kv_cta_group2(
                                v_smem.access_ptr("w", offset=raw_v_offset),
                                bar_v_part1_ready.ptr_to([cur_buf]),
                                T.address_of(tensor_map_kv),
                                local_col * 64 + cta_idx * 256,
                                token_idxs[0],
                                token_idxs[1],
                                token_idxs[2],
                                token_idxs[3],
                                T.uint64(0x14F0000000000000),
                            )
                        )

    else:
        # CUDA phase1.cuh:490-606.  MMA warp and KV-valid loading warp.
        T.ptx.setmaxnreg(True, 168)
        if (cta_idx == 0) & (warp_idx == 12):
            if T.ptx.elect_sync():
                desc_i_p_sq: T.uint32
                desc_i_p_tq: T.uint32
                desc_i_o: T.uint32
                T.ptx.tcgen05.encode_instr_descriptor(
                    T.address_of(desc_i_p_sq),
                    d_dtype="float32",
                    a_dtype="bfloat16",
                    b_dtype="bfloat16",
                    M=B_H,
                    N=B_TOPK,
                    K=16,
                    trans_a=False,
                    trans_b=False,
                    n_cta_groups=2,
                )
                T.ptx.tcgen05.encode_instr_descriptor(
                    T.address_of(desc_i_p_tq),
                    d_dtype="float32",
                    a_dtype="bfloat16",
                    b_dtype="bfloat16",
                    M=B_H,
                    N=B_TOPK,
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

                q_tq_desc: T.uint64
                T.ptx.tcgen05.encode_matrix_descriptor(
                    T.address_of(q_tq_desc),
                    q_full.ptr_to([0, d_sq]),
                    ldo=K_MAJOR_SWIZZLED_DESC_LDO,
                    sdo=Q_FULL_DESC_SDO,
                    swizzle=3,
                )
                bar_prologue_q.arrive(0, tx_count=B_H * d_qk * BF16_BYTES)
                bar_prologue_q.wait(0, 0)
                T.ptx.tcgen05.fence.after_thread_sync()
                for tile_idx in T.unroll(NUM_TQ_TILES):
                    for subtile_idx in T.unroll(8):
                        T.ptx.tcgen05.cp(
                            T.uint32(tQr_col + tile_idx * 32 + subtile_idx * 4),
                            q_tq_desc + T.uint64(tile_idx * ((B_H // 2) * 128 // 16) + subtile_idx),
                            shape="64x128b",
                            cta_group=2,
                            multicast="warpx2::02_13",
                        )
                bar_prologue_utccp.arrive(0, cta_group=2, cta_mask=3)

                for k in T.serial(0, num_k_blocks + 1, unroll=False):
                    if k < num_k_blocks:
                        cur_buf: T.let = k % NUM_BUFS
                        cur_phase: T.let = (k // NUM_BUFS) & 1

                        bar_k_part0_ready.arrive(cur_buf, tx_count=B_TOPK * d_sq * BF16_BYTES)
                        bar_k_part0_ready.wait(cur_buf, cur_phase)
                        if k > 0:
                            prev_buf: T.let = (k - 1) % NUM_BUFS
                            prev_phase: T.let = ((k - 1) // NUM_BUFS) & 1
                            bar_p_free.wait(prev_buf, prev_phase)
                        T.ptx.tcgen05.fence.after_thread_sync()

                        tiled_mma_p_accumulate[0] = T.uint32(0)
                        for sq_k in T.unroll(d_sq // 16):
                            sq_desc: T.uint64
                            sk_desc: T.uint64
                            T.ptx.tcgen05.encode_matrix_descriptor(
                                T.address_of(sq_desc),
                                sq_smem.ptr_to([0, sq_k * 16]),
                                ldo=K_MAJOR_SWIZZLED_DESC_LDO,
                                sdo=SQ_DESC_SDO,
                                swizzle=3,
                            )
                            T.ptx.tcgen05.encode_matrix_descriptor(
                                T.address_of(sk_desc),
                                k_smem.ptr_to([0, sq_k * 16]),
                                ldo=K_MAJOR_SWIZZLED_DESC_LDO,
                                sdo=K_DESC_SDO,
                                swizzle=3,
                            )
                            T.evaluate(
                                _tcgen05_mma_ws_ss_2cta(
                                    T.uint32(tP_col),
                                    sq_desc,
                                    sk_desc,
                                    desc_i_p_sq,
                                    tiled_mma_p_accumulate[0],
                                )
                            )
                            tiled_mma_p_accumulate[0] = T.uint32(1)
                        bar_qk_part_done.arrive(cur_buf, cta_group=2, cta_mask=3)

                        bar_k_part1_ready.arrive(
                            cur_buf, tx_count=B_TOPK * (d_qk - d_sq) * BF16_BYTES
                        )
                        bar_k_part1_ready.wait(cur_buf, cur_phase)
                        T.ptx.tcgen05.fence.after_thread_sync()

                        for tq_k in T.unroll(D_TQ // 16):
                            sk_desc: T.uint64
                            T.ptx.tcgen05.encode_matrix_descriptor(
                                T.address_of(sk_desc),
                                k_smem.ptr_to([0, d_sq + tq_k * 16]),
                                ldo=K_MAJOR_SWIZZLED_DESC_LDO,
                                sdo=K_DESC_SDO,
                                swizzle=3,
                            )
                            T.evaluate(
                                _tcgen05_mma_ws_ts_2cta(
                                    T.uint32(tP_col),
                                    T.uint32(tQr_col + tq_k * 8),
                                    sk_desc,
                                    desc_i_p_tq,
                                    tiled_mma_p_accumulate[0],
                                )
                            )
                            tiled_mma_p_accumulate[0] = T.uint32(1)
                        bar_qk_done.arrive(cur_buf, cta_group=2, cta_mask=3)

                    if k > 0:
                        cur_buf_prev: T.let = (k - 1) % NUM_BUFS
                        cur_phase_prev: T.let = ((k - 1) // NUM_BUFS) & 1
                        bar_so_ready.wait(cur_buf_prev, cur_phase_prev)

                        bar_v_part0_ready.arrive(
                            cur_buf_prev, tx_count=(B_TOPK // 2) * D_V * BF16_BYTES
                        )
                        bar_v_part0_ready.wait(cur_buf_prev, cur_phase_prev)
                        T.ptx.tcgen05.fence.after_thread_sync()
                        tiled_mma_o_accumulate[0] = T.if_then_else(k == 1, T.uint32(0), T.uint32(1))
                        for sv_k in T.unroll((B_TOPK // 2) // 16):
                            s_desc: T.uint64
                            v_desc: T.uint64
                            o_accumulate: T.let = tiled_mma_o_accumulate[0]
                            T.ptx.tcgen05.encode_matrix_descriptor(
                                T.address_of(s_desc),
                                s_smem.ptr_to([sv_k * 16 * (B_H // 2)]),
                                ldo=S_DESC_LDO,
                                sdo=S_DESC_SDO,
                                swizzle=0,
                            )
                            T.ptx.tcgen05.encode_matrix_descriptor(
                                T.address_of(v_desc),
                                v_smem.access_ptr("r", offset=sv_k * 16 * 64),
                                ldo=V_DESC_LDO,
                                sdo=V_DESC_SDO,
                                swizzle=3,
                            )
                            T.evaluate(
                                _tcgen05_mma_ws_ss_2cta(
                                    T.uint32(tO_col), s_desc, v_desc, desc_i_o, o_accumulate
                                )
                            )
                            T.evaluate(
                                _tcgen05_mma_ws_ss_2cta(
                                    T.uint32(tO_col + 128),
                                    s_desc,
                                    v_desc + T.uint64(2048),
                                    desc_i_o,
                                    o_accumulate,
                                )
                            )
                            tiled_mma_o_accumulate[0] = T.uint32(1)
                        bar_sv_part_done.arrive(cur_buf_prev, cta_group=2, cta_mask=3)

                        bar_v_part1_ready.arrive(
                            cur_buf_prev, tx_count=(B_TOPK // 2) * D_V * BF16_BYTES
                        )
                        bar_v_part1_ready.wait(cur_buf_prev, cur_phase_prev)
                        T.ptx.tcgen05.fence.after_thread_sync()
                        for sv_k in T.unroll((B_TOPK // 2) // 16):
                            s_desc: T.uint64
                            v_desc: T.uint64
                            o_accumulate: T.let = tiled_mma_o_accumulate[0]
                            s_part_offset: T.let = (B_H // 2) * (B_TOPK // 2)
                            v_part_offset: T.let = (B_TOPK // 2) * 64
                            T.ptx.tcgen05.encode_matrix_descriptor(
                                T.address_of(s_desc),
                                s_smem.ptr_to([s_part_offset + sv_k * 16 * (B_H // 2)]),
                                ldo=S_DESC_LDO,
                                sdo=S_DESC_SDO,
                                swizzle=0,
                            )
                            T.ptx.tcgen05.encode_matrix_descriptor(
                                T.address_of(v_desc),
                                v_smem.access_ptr("r", offset=v_part_offset + sv_k * 16 * 64),
                                ldo=V_DESC_LDO,
                                sdo=V_DESC_SDO,
                                swizzle=3,
                            )
                            T.evaluate(
                                _tcgen05_mma_ws_ss_2cta(
                                    T.uint32(tO_col), s_desc, v_desc, desc_i_o, o_accumulate
                                )
                            )
                            T.evaluate(
                                _tcgen05_mma_ws_ss_2cta(
                                    T.uint32(tO_col + 128),
                                    s_desc,
                                    v_desc + T.uint64(2048),
                                    desc_i_o,
                                    o_accumulate,
                                )
                            )
                            tiled_mma_o_accumulate[0] = T.uint32(1)
                        bar_sv_done.arrive(cur_buf_prev, cta_group=2, cta_mask=3)

        elif warp_idx == 13:
            if lane_idx < B_TOPK // 8:
                lane_indices = T.alloc_local((8,), "int32")
                for k in T.serial(0, num_k_blocks, unroll=False):
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
                    abs_pos_start: T.let = k * B_TOPK
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
                    is_ks_valid_mask: T.let = T.cast(
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
                    cur_buf: T.let = k % NUM_BUFS
                    cur_phase: T.let = (k // NUM_BUFS) & 1
                    bar_k_valid_free.wait(cur_buf, cur_phase ^ 1)
                    is_k_valid[cur_buf, lane_idx] = is_ks_valid_mask
                    bar_k_valid_ready.arrive(cur_buf)


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
    return kernel.with_attr("tirx.kernel_launch_params", list(HEAD128_LAUNCH_PARAM_TAGS))


def run_test(**kwargs: Any) -> None:
    if not _IMPLEMENTATION_COMPLETE:
        raise SkipTest("sparse FlashMLA head128 phase1 transcription is not complete")
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for sparse FlashMLA head128 phase1")

    from tirx_kernels.runner import compile_kernel

    case = prepare_data(**kwargs)
    cfg: SparseFlashMLAPrefillHead128Config = case["config"]
    if not case["dispatch_reason"].startswith("regular:"):
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
        raise SkipTest("sparse FlashMLA head128 phase1 transcription is not complete")
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for sparse FlashMLA head128 phase1 benchmark")

    from tirx_kernels.runner import compile_kernel
    from tvm.tirx.bench import bench

    case = prepare_data(**kwargs)
    if not case["dispatch_reason"].startswith("regular:"):
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

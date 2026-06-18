from __future__ import annotations

import os
import random

import torch

import tvm
from tirx_kernels.gemm.grouped_fp8_gemm_contiguous import grouped_fp8_gemm_contiguous
from tvm.tirx.bench import bench, tensor_bytes

KERNEL_META = {"name": "grouped_fp8_gemm_contiguous", "category": "gemm", "compute_capability": 10}
CONFIGS = [
    {
        "num_groups": 4,
        "expected_m_per_group": 256,
        "N": 512,
        "K": 512,
        "seed": 1,
        "label": "small_g4_m256_n512_k512",
    },
    {
        "num_groups": 8,
        "expected_m_per_group": 256,
        "N": 1024,
        "K": 512,
        "seed": 2,
        "label": "small_g8_m256_n1024_k512",
    },
]
BENCH_CONFIGS = [
    {
        "num_groups": 4,
        "expected_m_per_group": 8192,
        "N": 6144,
        "K": 7168,
        "seed": 1,
        "label": "large_g4_m8192_n6144_k7168",
    },
    {
        "num_groups": 4,
        "expected_m_per_group": 8192,
        "N": 7168,
        "K": 3072,
        "seed": 2,
        "label": "large_g4_m8192_n7168_k3072",
    },
    {
        "num_groups": 4,
        "expected_m_per_group": 8192,
        "N": 4096,
        "K": 4096,
        "seed": 3,
        "label": "large_g4_m8192_n4096_k4096",
    },
    {
        "num_groups": 4,
        "expected_m_per_group": 8192,
        "N": 4096,
        "K": 2048,
        "seed": 4,
        "label": "large_g4_m8192_n4096_k2048",
    },
    {
        "num_groups": 8,
        "expected_m_per_group": 4096,
        "N": 6144,
        "K": 7168,
        "seed": 5,
        "label": "large_g8_m4096_n6144_k7168",
    },
    {
        "num_groups": 8,
        "expected_m_per_group": 4096,
        "N": 7168,
        "K": 3072,
        "seed": 6,
        "label": "large_g8_m4096_n7168_k3072",
    },
    {
        "num_groups": 8,
        "expected_m_per_group": 4096,
        "N": 4096,
        "K": 4096,
        "seed": 7,
        "label": "large_g8_m4096_n4096_k4096",
    },
    {
        "num_groups": 8,
        "expected_m_per_group": 4096,
        "N": 4096,
        "K": 2048,
        "seed": 8,
        "label": "large_g8_m4096_n4096_k2048",
    },
]

DIFF_THRESHOLD = 0.002
CONTIGUOUS_M_ALIGNMENT = 240


def get_kernel(num_groups: int, M: int, N: int, K: int, **kwargs):
    return grouped_fp8_gemm_contiguous(num_groups, M, N, K)


def _compile_kernel(num_groups: int, M: int, N: int, K: int):
    target = tvm.target.Target("cuda")
    with target:
        return tvm.compile(
            tvm.IRModule({"main": grouped_fp8_gemm_contiguous(num_groups, M, N, K)}),
            target=target,
            tir_pipeline="tirx",
        )


def _make_kernel_callable(ex, data: dict):
    A = data["A_fp8"]
    B = data["B_fp8"]
    SFA = data["SFA"]
    SFB = data["SFB"]
    D = data["D_tir"]
    grouped_layout = data["grouped_layout"]

    def kernel_fn():
        ex.mod(A, B, SFA, SFB, D, grouped_layout)

    return kernel_fn


def _setup_tir(data: dict, num_groups: int, M: int, N: int, K: int):
    if int(data["alignment"]) != CONTIGUOUS_M_ALIGNMENT:
        raise AssertionError(
            f"expected grouped contiguous alignment {CONTIGUOUS_M_ALIGNMENT}, got {data['alignment']}"
        )
    ex = _compile_kernel(num_groups, M, N, K)
    kernel_fn = _make_kernel_callable(ex, data)
    kernel_fn()
    return kernel_fn


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    bits = x.abs().float().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return (exp.clamp(1, 254) << 23).view(torch.float32)


def _per_token_cast_to_fp8(x: torch.Tensor, gran_k: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    if x.dim() != 2:
        raise ValueError(f"expected 2D tensor, got rank {x.dim()}")
    M, K = x.shape
    padded_K = _align(K, gran_k)
    if padded_K == K:
        x_padded = x
    else:
        x_padded = torch.zeros((M, padded_K), dtype=x.dtype, device=x.device)
        x_padded[:, :K] = x

    x_view = x_padded.view(M, padded_K // gran_k, gran_k)
    scale = x_view.abs().float().amax(dim=2).clamp(1e-4) / 448.0
    scale = _ceil_to_ue8m0(scale)
    x_fp8 = (x_view * scale.reciprocal().unsqueeze(2)).to(torch.float8_e4m3fn)
    return x_fp8.view(M, padded_K)[:, :K].contiguous(), scale


def _per_block_cast_to_fp8(x: torch.Tensor, gran_m: int = 128, gran_k: int = 128):
    if x.dim() != 2:
        raise ValueError(f"expected 2D tensor, got rank {x.dim()}")
    N, K = x.shape
    padded_N = _align(N, gran_m)
    padded_K = _align(K, gran_k)
    x_padded = torch.zeros((padded_N, padded_K), dtype=x.dtype, device=x.device)
    x_padded[:N, :K] = x

    x_view = x_padded.view(padded_N // gran_m, gran_m, padded_K // gran_k, gran_k)
    scale = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4) / 448.0
    scale = _ceil_to_ue8m0(scale)
    x_fp8 = (x_view * scale.reciprocal()).to(torch.float8_e4m3fn)
    return x_fp8.view(padded_N, padded_K)[:N, :K].contiguous(), scale.view(
        padded_N // gran_m, padded_K // gran_k
    )


def _pack_ue8m0_rows_to_words(scale: torch.Tensor) -> torch.Tensor:
    rows, k_blocks = scale.shape
    del rows
    if scale.dtype != torch.float32:
        raise TypeError(f"expected float32 scales, got {scale.dtype}")
    if k_blocks % 4 != 0:
        raise ValueError(f"k_blocks={k_blocks} must be divisible by 4")
    scale_u8 = (scale.view(torch.int32) >> 23).to(torch.uint8).contiguous()
    return scale_u8.view(torch.uint32).T.contiguous()


def _pack_b_scales_for_tir(scale: torch.Tensor, N: int) -> torch.Tensor:
    _, k_blocks = scale.shape
    if k_blocks % 4 != 0:
        raise ValueError(f"k_blocks={k_blocks} must be divisible by 4")
    scale_u8 = (scale.view(torch.int32) >> 23).to(torch.uint8).contiguous()
    scale_rows = scale_u8.repeat_interleave(128, dim=0)[:N, :].contiguous()
    return scale_rows.view(torch.uint32).T.contiguous()


def _make_actual_ms(
    num_groups: int, expected_m_per_group: int, seed: int
) -> tuple[list[int], list[int]]:
    random.seed(seed)
    actual = [int(expected_m_per_group * random.uniform(0.7, 1.3)) for _ in range(num_groups)]
    aligned = [_align(m, CONTIGUOUS_M_ALIGNMENT) for m in actual]
    return actual, aligned


def _dequant_a(A_fp8: torch.Tensor, scale: torch.Tensor, K: int) -> torch.Tensor:
    return (A_fp8.float().view(A_fp8.shape[0], K // 128, 128) * scale.float().unsqueeze(2)).view(
        A_fp8.shape[0], K
    )


def _dequant_b(B_fp8: torch.Tensor, scale: torch.Tensor, N: int, K: int) -> torch.Tensor:
    scale_rows = scale.float().repeat_interleave(128, dim=0)[:N, :]
    return (B_fp8.float().view(N, K // 128, 128) * scale_rows.unsqueeze(2)).view(N, K)


def _compute_reference(data: dict, out: torch.Tensor | None = None) -> torch.Tensor:
    M = data["M"]
    N = data["N"]
    K = data["K"]
    ref = (
        out
        if out is not None
        else torch.empty((M, N), device=data["A_fp8"].device, dtype=torch.bfloat16)
    )
    start = 0
    for group, aligned_m in enumerate(data["aligned_ms"]):
        end = start + aligned_m
        A = _dequant_a(data["A_fp8"][start:end], data["A_scale"][start:end], K)
        B = _dequant_b(data["B_fp8"][group], data["B_scale"][group], N, K)
        ref[start:end] = (A @ B.T).to(torch.bfloat16)
        start = end
    return ref


def _calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.double()
    y = y.double()
    denominator = (x * x + y * y).sum()
    if denominator.item() == 0:
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return float(1 - sim)


def prepare_data(
    num_groups: int,
    expected_m_per_group: int,
    N: int,
    K: int,
    *,
    seed: int = 0,
    device: str = "cuda",
) -> dict:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    actual_ms, aligned_ms = _make_actual_ms(num_groups, expected_m_per_group, seed)
    M = sum(aligned_ms)
    if K % 128 != 0:
        raise ValueError(f"K={K} must be divisible by 128")

    A_bf16 = torch.randn((M, K), device=device, dtype=torch.bfloat16)
    B_bf16 = torch.randn((num_groups, N, K), device=device, dtype=torch.bfloat16)
    grouped_layout = torch.empty((M,), device=device, dtype=torch.int32)

    start = 0
    for group, (actual_m, aligned_m) in enumerate(zip(actual_ms, aligned_ms)):
        actual_end = start + actual_m
        aligned_end = start + aligned_m
        grouped_layout[start:actual_end] = group
        grouped_layout[actual_end:aligned_end] = -1
        A_bf16[actual_end:aligned_end] = 0
        start = aligned_end

    A_fp8, A_scale = _per_token_cast_to_fp8(A_bf16)
    B_fp8_groups = []
    B_scale_groups = []
    for group in range(num_groups):
        B_fp8_group, B_scale_group = _per_block_cast_to_fp8(B_bf16[group])
        B_fp8_groups.append(B_fp8_group)
        B_scale_groups.append(B_scale_group)
    B_fp8 = torch.stack(B_fp8_groups)
    B_scale = torch.stack(B_scale_groups)

    data = {
        "M": M,
        "N": N,
        "K": K,
        "num_groups": num_groups,
        "expected_m_per_group": expected_m_per_group,
        "actual_ms": actual_ms,
        "aligned_ms": aligned_ms,
        "A_fp8": A_fp8,
        "B_fp8": B_fp8,
        "A_scale": A_scale,
        "B_scale": B_scale,
        "SFA": _pack_ue8m0_rows_to_words(A_scale),
        "SFB": torch.stack(
            [_pack_b_scales_for_tir(B_scale[group], N) for group in range(num_groups)]
        ),
        "D_tir": torch.empty((M, N), device=device, dtype=torch.bfloat16),
        "D_torch": torch.empty((M, N), device=device, dtype=torch.bfloat16),
        "grouped_layout": grouped_layout,
        "alignment": CONTIGUOUS_M_ALIGNMENT,
    }
    data["ref"] = _compute_reference(data)
    return data


def setup_torch_reference(data: dict):
    def ref_fn():
        _compute_reference(data, data["D_torch"])

    ref_fn()
    return ref_fn


def _check(tag: str, out: torch.Tensor, ref: torch.Tensor) -> float:
    diff = _calc_diff(out, ref)
    if diff >= DIFF_THRESHOLD:
        raise AssertionError(f"{tag} diff {diff:.6f} >= {DIFF_THRESHOLD}")
    return diff


def run_test(
    num_groups: int = 4,
    expected_m_per_group: int = 256,
    N: int = 512,
    K: int = 512,
    seed: int = 1,
    **kwargs,
):
    data = prepare_data(num_groups, expected_m_per_group, N, K, seed=seed)
    _setup_tir(data, num_groups, data["M"], N, K)
    _check("tir", data["D_tir"], data["ref"])


def _selected_impls() -> str:
    value = os.environ.get("TIRX_BENCH_IMPLS", "all")
    if value not in {"all", "ours", "baseline"}:
        raise ValueError(f"unsupported TIRX_BENCH_IMPLS={value!r}")
    return value


def run_bench(
    num_groups: int = 4,
    expected_m_per_group: int = 8192,
    N: int = 4096,
    K: int = 2048,
    seed: int = 1,
    *,
    warmup: int = 10,
    repeat: int = 30,
    timer: str = "event",
    **kwargs,
):
    sample = prepare_data(num_groups, expected_m_per_group, N, K, seed=seed)
    ex = _compile_kernel(num_groups, sample["M"], N, K)
    tir_sample = _make_kernel_callable(ex, sample)
    tir_sample()
    _check("tir", sample["D_tir"], sample["ref"])

    impl_mode = _selected_impls()
    funcs = {}
    if impl_mode in {"all", "ours"}:
        funcs["tir"] = lambda case: case["tir"]()
    if impl_mode in {"all", "baseline"}:
        funcs["torch-reference"] = lambda case: case["torch-reference"]()

    def make_input():
        data = prepare_data(num_groups, expected_m_per_group, N, K, seed=seed)
        case = {"data": data, "tir": _make_kernel_callable(ex, data)}
        if impl_mode in {"all", "baseline"}:
            case["torch-reference"] = setup_torch_reference(data)
        input_bytes = tensor_bytes(
            data["A_fp8"],
            data["B_fp8"],
            data["SFA"],
            data["SFB"],
            data["grouped_layout"],
            data["D_tir"],
        )
        return case, input_bytes

    result = bench(
        funcs,
        make_input,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        proton_name="grouped_fp8_gemm_contiguous",
        cooldown_s=0.0,
    )
    result.update(
        {
            "M": sample["M"],
            "N": N,
            "K": K,
            "num_groups": num_groups,
            "actual_ms": sample["actual_ms"],
            "aligned_ms": sample["aligned_ms"],
            "reference": "local torch dequantized fp8 matmul",
            "tir_launches": 1,
        }
    )
    impls = result.get("impls", {})
    if impls.get("torch-reference", 0) > 0 and "tir" in impls:
        result["ratio"] = impls["tir"] / impls["torch-reference"]
    return result

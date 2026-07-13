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
from __future__ import annotations

import inspect
import os

import pytest

from tirx_kernels.deepgemm.mega_moe import (
    MegaMoeConfig,
    _aggregate_rank_results,
    _bench_megamoe_mode,
    _cuda_compile_mode,
    _get_block_config_for_mega_moe,
    _get_mega_moe_cuda_compile_mode,
    _get_num_bytes_per_pull,
    _get_num_experts_per_wave_for_mega_moe,
    _run_worker,
    get_deepgemm_launch_config,
    get_deepgemm_workspace_layout,
    run_bench,
)


def _rank_result(tirx: list[float], deepgemm: list[float]) -> dict:
    return {
        "status": "OK",
        "impls": {"tirx": sum(tirx) / len(tirx), "deepgemm": sum(deepgemm) / len(deepgemm)},
        "round_samples": {"tirx": tirx, "deepgemm": deepgemm},
        "errors": {},
        "deepgemm_max_abs_diff": 0.0,
    }


def test_aggregate_rank_results_takes_slowest_rank_per_round() -> None:
    result = _aggregate_rank_results(
        [
            (0, _rank_result([10.0, 30.0, 20.0], [8.0, 12.0, 7.0])),
            (1, _rank_result([20.0, 15.0, 40.0], [9.0, 11.0, 10.0])),
        ]
    )

    assert result["round_samples"] == {"deepgemm": [9.0, 12.0, 10.0], "tirx": [20.0, 30.0, 40.0]}
    assert result["impls"] == {"deepgemm": pytest.approx(31.0 / 3.0), "tirx": 30.0}
    assert result["rank_results"][1]["round_samples"]["tirx"] == [20.0, 15.0, 40.0]


def test_aggregate_rank_results_rejects_mismatched_round_counts() -> None:
    with pytest.raises(RuntimeError, match="different round counts"):
        _aggregate_rank_results(
            [(0, _rank_result([10.0, 20.0], [8.0, 9.0])), (1, _rank_result([11.0], [7.0]))]
        )


@pytest.mark.parametrize(
    ("num_tokens", "expected_block_m", "expected_block_k"),
    [
        (64, 16, 256),
        (544, 16, 256),
        (545, 32, 128),
        (1056, 32, 128),
        (1057, 64, 128),
        (8192, 192, 128),
    ],
)
def test_block_config_matches_deepgemm_thresholds(
    num_tokens: int, expected_block_m: int, expected_block_k: int
) -> None:
    _, block_m, _, block_k, _ = _get_block_config_for_mega_moe(
        num_ranks=1, num_experts=384, num_tokens=num_tokens, num_topk=6
    )
    assert (block_m, block_k) == (expected_block_m, expected_block_k)


@pytest.mark.parametrize(
    ("hidden", "expected"), [(1024, 1024), (4096, 4096), (7168, 3584), (16384, 4096)]
)
def test_pull_chunk_size_matches_deepgemm(hidden: int, expected: int) -> None:
    assert _get_num_bytes_per_pull(hidden) == expected


def test_wave_heuristic_allows_a_partial_tail() -> None:
    assert (
        _get_num_experts_per_wave_for_mega_moe(
            num_experts_per_rank=3,
            num_tokens=577,
            num_topk=3,
            intermediate_hidden=3072,
            block_m=192,
            block_n=128,
            num_sms=148,
            num_ring_tokens=2304,
            num_max_tokens_per_rank=768,
            num_ranks=1,
        )
        == 2
    )


def test_decode_launch_uses_bk256_and_chunked_pull(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TIRX_DEEPGEMM_NUM_SMS_OVERRIDE", "148")
    launch = get_deepgemm_launch_config(
        MegaMoeConfig(
            num_processes=1,
            num_max_tokens_per_rank=64,
            num_tokens=64,
            hidden=7168,
            intermediate_hidden=3072,
            num_experts=384,
            num_topk=6,
        )
    )
    assert launch.block_k == 256
    assert launch.num_bytes_per_pull == 3584


@pytest.mark.parametrize(
    (
        "num_processes",
        "num_tokens",
        "expected_ring_tokens",
        "expected_sf_ring_tokens",
        "expected_max_pool_tokens",
        "expected_experts_per_wave",
        "expected_block_m",
        "expected_block_k",
    ),
    [
        (1, 64, 147456, 2359296, 75648, 8, 16, 256),
        (2, 64, 78336, 1253376, 41472, 8, 16, 256),
        (4, 64, 46080, 737280, 27648, 8, 16, 256),
        (6, 64, 38400, 614400, 26112, 8, 16, 256),
        (1, 8192, 197760, 3164160, 124032, 8, 192, 128),
        (2, 8192, 175104, 2801664, 138240, 4, 192, 128),
        (4, 8192, 239616, 3833856, 221184, 3, 192, 128),
        (6, 8192, 328704, 5259264, 316416, 2, 192, 128),
    ],
)
def test_bounded_ring_layout_matches_deepgemm_strict_matrix(
    monkeypatch: pytest.MonkeyPatch,
    num_processes: int,
    num_tokens: int,
    expected_ring_tokens: int,
    expected_sf_ring_tokens: int,
    expected_max_pool_tokens: int,
    expected_experts_per_wave: int,
    expected_block_m: int,
    expected_block_k: int,
) -> None:
    monkeypatch.setenv("TIRX_DEEPGEMM_NUM_SMS_OVERRIDE", "148")
    config = MegaMoeConfig(
        num_processes=num_processes,
        num_max_tokens_per_rank=num_tokens,
        num_tokens=num_tokens,
        hidden=7168,
        intermediate_hidden=3072,
        num_experts=384,
        num_topk=6,
    )
    workspace = get_deepgemm_workspace_layout(config)
    launch = get_deepgemm_launch_config(config)

    assert workspace.num_ring_tokens == expected_ring_tokens
    assert workspace.num_ring_blocks == expected_ring_tokens // 8
    assert workspace.num_sf_ring_tokens == expected_sf_ring_tokens
    assert workspace.num_max_pool_tokens == expected_max_pool_tokens
    assert launch.num_experts_per_wave == expected_experts_per_wave
    assert (launch.block_m, launch.block_k) == (expected_block_m, expected_block_k)

    counter_bytes = workspace.num_ring_blocks * 4
    assert workspace.l1_empty_count_offset == workspace.l1_full_count_offset + counter_bytes
    assert workspace.l2_full_count_offset == workspace.l1_empty_count_offset + counter_bytes
    assert workspace.l2_empty_count_offset == workspace.l2_full_count_offset + counter_bytes
    assert workspace.src_token_topk_idx_offset == workspace.l2_empty_count_offset + counter_bytes


def test_mega_moe_cuda_compile_mode_defaults_to_nvcc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TIRX_MEGAMOE_CUDA_COMPILE_MODE", raising=False)
    monkeypatch.delenv("TVM_CUDA_COMPILE_MODE", raising=False)
    assert _get_mega_moe_cuda_compile_mode() == "nvcc"

    monkeypatch.setenv("TVM_CUDA_COMPILE_MODE", "nvrtc")
    assert _get_mega_moe_cuda_compile_mode() == "nvrtc"

    monkeypatch.setenv("TIRX_MEGAMOE_CUDA_COMPILE_MODE", "nvcc")
    assert _get_mega_moe_cuda_compile_mode() == "nvcc"


def test_cuda_compile_mode_context_restores_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TVM_CUDA_COMPILE_MODE", "nvrtc")
    with _cuda_compile_mode("nvcc"):
        assert os.environ["TVM_CUDA_COMPILE_MODE"] == "nvcc"
    assert os.environ["TVM_CUDA_COMPILE_MODE"] == "nvrtc"


def test_mega_moe_bench_inherits_shared_defaults() -> None:
    parameters = inspect.signature(run_bench).parameters
    assert parameters["warmup"].default is None
    assert parameters["repeat"].default is None
    assert parameters["timer"].default is None
    assert "bench_tk" not in inspect.getsource(_run_worker)


def test_megamoe_timer_wraps_deepgemm_bench_kineto() -> None:
    calls = []

    def fake_barrier() -> None:
        pass

    def fake_between_impls() -> None:
        pass

    def fake_bench_kineto(fn, kernel_names, num_tests=30, barrier=None) -> tuple[float, float]:
        assert kernel_names in (
            ("mega_moe_kernel", "sm100_fp8_fp4_mega_moe_impl"),
            ("sm100_fp8_fp4_mega_moe_impl", "mega_moe_kernel"),
        )
        assert barrier is fake_barrier
        fn()
        return (0.001, 0.001)

    result = _bench_megamoe_mode(
        {"tirx": lambda: calls.append("tirx"), "deepgemm": lambda: calls.append("deepgemm")},
        {"tirx": "mega_moe_kernel", "deepgemm": "sm100_fp8_fp4_mega_moe_impl"},
        fake_bench_kineto,
        fake_barrier,
        fake_between_impls,
        rounds=2,
    )

    assert calls == ["tirx", "deepgemm", "deepgemm", "tirx"]
    assert result["timer"] == "megamoe"
    assert result["round_samples"] == {"tirx": [1000.0, 1000.0], "deepgemm": [1000.0, 1000.0]}
    assert result["benchmark_protocol"]["num_tests"] == 30
    assert result["benchmark_protocol"]["round_orders"] == [
        ["tirx", "deepgemm"],
        ["deepgemm", "tirx"],
    ]

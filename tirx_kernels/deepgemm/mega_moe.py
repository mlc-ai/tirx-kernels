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

import ctypes
import ctypes.util
import math
import os
import random
import socket
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import cache
from typing import Any
from unittest import SkipTest

import torch
import torch.multiprocessing as mp

from tvm.ir.type import PointerType, PrimType

_DEEP_GEMM_MODULE_NAME = "deep_gemm"
DEEPGEMM_SYM_BUFFER_MAX_RANKS = 72


@dataclass(frozen=True)
class MegaMoeConfig:
    num_processes: int = 1
    num_max_tokens_per_rank: int = 128
    num_tokens: int = 96
    hidden: int = 1024
    intermediate_hidden: int = 512
    num_experts: int = 8
    num_topk: int = 2
    activation_clamp: float = 10.0
    fast_math: int = 1

    def validate(self) -> None:
        if self.num_processes <= 0:
            raise ValueError("num_processes must be positive")
        if self.num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        if self.num_tokens > self.num_max_tokens_per_rank:
            raise ValueError("num_tokens must not exceed num_max_tokens_per_rank")
        if self.hidden % 128 != 0 or self.intermediate_hidden % 128 != 0:
            raise ValueError("hidden and intermediate_hidden must be multiples of 128")
        if self.intermediate_hidden > 4096:
            raise ValueError(
                "intermediate_hidden must satisfy DeepGEMM L2_SHAPE_K <= 64 * L1_OUT_BLOCK_N"
            )
        if self.num_experts % self.num_processes != 0:
            raise ValueError("num_experts must be divisible by num_processes")
        if self.num_topk <= 0 or self.num_topk > self.num_experts:
            raise ValueError("num_topk must be in [1, num_experts]")

    @property
    def num_experts_per_rank(self) -> int:
        return self.num_experts // self.num_processes


@dataclass
class MegaMoeCase:
    config: MegaMoeConfig
    rank_idx: int
    num_ranks: int
    group: Any
    deep_gemm: Any
    symm_buffer: Any
    x_fp8: tuple[torch.Tensor, torch.Tensor]
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    raw_l1_weights: tuple[torch.Tensor, torch.Tensor]
    raw_l2_weights: tuple[torch.Tensor, torch.Tensor]
    transformed_l1_weights: tuple[torch.Tensor, torch.Tensor]
    transformed_l2_weights: tuple[torch.Tensor, torch.Tensor]
    workspace_layout: DeepGemmWorkspaceLayout
    symm_buffer_layout: DeepGemmSymmBufferLayout
    dispatch_reference: DispatchReference
    pull_reference: PullReference | None
    scheduler_reference: SchedulerReference | None


@dataclass
class TirxMegaMoeLaunchContext:
    config: MegaMoeConfig
    rank_idx: int
    num_ranks: int
    symm_buffer: Any
    transformed_l1_weights: tuple[torch.Tensor, torch.Tensor]
    transformed_l2_weights: tuple[torch.Tensor, torch.Tensor]
    workspace_layout: DeepGemmWorkspaceLayout
    symm_buffer_layout: DeepGemmSymmBufferLayout


@dataclass(frozen=True)
class DeepGemmLaunchConfig:
    num_sms: int
    num_ctas_per_cluster: int
    block_m: int
    block_n: int
    block_k: int
    load_block_m: int
    load_block_n: int
    store_block_m: int
    num_dispatch_threads: int
    num_non_epilogue_threads: int
    num_epilogue_threads: int
    num_experts_per_wave: int
    num_topk: int
    hidden: int
    intermediate_hidden: int

    @property
    def num_dispatch_warps(self) -> int:
        return self.num_dispatch_threads // 32

    @property
    def num_non_epilogue_warps(self) -> int:
        return self.num_non_epilogue_threads // 32

    @property
    def num_epilogue_warps(self) -> int:
        return self.num_epilogue_threads // 32

    @property
    def num_total_warps(self) -> int:
        return self.num_dispatch_warps + self.num_non_epilogue_warps + self.num_epilogue_warps

    @property
    def num_threads(self) -> int:
        return self.num_total_warps * 32

    @property
    def num_warpgroups(self) -> int:
        return self.num_total_warps // 4

    @property
    def num_threads_per_cta(self) -> int:
        return self.num_threads

    @property
    def num_warps_per_cta(self) -> int:
        return self.num_total_warps

    @property
    def num_warpgroups_per_cta(self) -> int:
        return self.num_warpgroups

    @property
    def num_tokens_per_warp(self) -> int:
        return 32 // self.num_topk

    @property
    def num_activate_lanes(self) -> int:
        return self.num_tokens_per_warp * self.num_topk

    @property
    def load_a_warp_idx(self) -> int:
        return self.num_dispatch_warps

    @property
    def load_b_warp_idx(self) -> int:
        return self.num_dispatch_warps + 1

    @property
    def mma_issue_warp_idx(self) -> int:
        return self.num_dispatch_warps + 2

    @property
    def reserved_non_epilogue_warp_idx(self) -> int:
        return self.num_dispatch_warps + 3

    @property
    def epilogue_warp_start_idx(self) -> int:
        return self.num_dispatch_warps + self.num_non_epilogue_warps


@dataclass(frozen=True)
class DeepGemmWorkspaceLayout:
    num_ranks: int
    num_experts: int
    num_experts_per_rank: int
    num_max_tokens_per_rank: int
    num_topk: int
    block_m: int
    num_max_recv_tokens_per_expert: int
    num_max_pool_tokens: int
    num_max_pool_blocks: int
    num_padded_sf_pool_tokens: int
    token_src_metadata_bytes: int
    barrier_offset: int
    expert_send_count_offset: int
    expert_recv_count_offset: int
    expert_recv_count_sum_offset: int
    l1_arrival_count_offset: int
    l2_arrival_mask_offset: int
    src_token_topk_idx_offset: int
    token_src_metadata_offset: int
    total_bytes: int


@dataclass(frozen=True)
class DispatchReference:
    expert_send_count: torch.Tensor
    expert_recv_count: torch.Tensor
    expert_recv_count_sum: torch.Tensor
    src_token_topk_idx: torch.Tensor


@dataclass(frozen=True)
class DeepGemmSymmBufferLayout:
    workspace_bytes: int
    input_token_offset: int
    input_sf_offset: int
    input_topk_idx_offset: int
    input_topk_weights_offset: int
    l1_token_offset: int
    l1_sf_offset: int
    l1_topk_weights_offset: int
    l2_token_offset: int
    l2_sf_offset: int
    combine_token_offset: int
    total_bytes: int


@dataclass(frozen=True)
class PullReference:
    expert_pool_block_offset: torch.Tensor
    pool_src_rank_idx: torch.Tensor
    pool_src_token_idx: torch.Tensor
    pool_src_topk_idx: torch.Tensor
    pool_src_token_topk_idx: torch.Tensor
    l1_arrival_count: torch.Tensor


@dataclass(frozen=True)
class SchedulerBlockReference:
    block_idx_seed: int
    phase: str
    local_expert_idx: int
    num_k_blocks: int
    m_block_idx: int
    n_block_idx: int
    pool_block_offset: int
    valid_m: int


@dataclass(frozen=True)
class SchedulerReference:
    expert_num_tokens: torch.Tensor
    expert_pool_block_offset: torch.Tensor
    blocks: tuple[SchedulerBlockReference, ...]
    num_linear1_blocks: int
    num_linear2_blocks: int


@dataclass
class TirxMegaMoeInvocation:
    executable: Any
    y: torch.Tensor
    symm_buffer_offsets: tuple[int, ...]
    tensor_maps: dict[str, _AlignedTensorMap]


@dataclass
class TirxMegaMoePrepared:
    context: TirxMegaMoeLaunchContext
    invocation: TirxMegaMoeInvocation


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _align_down(value: int, alignment: int) -> int:
    return (value // alignment) * alignment


# Mirror of deep_gemm::layout constants from
# deep_gemm/include/deep_gemm/layout/mega_moe.cuh. The shared symm buffer is
# allocated by deep_gemm host code; the upstream layout is block_m-agnostic
# (uses LCM of the candidate set for token alignment and Min for the barrier-
# array sizing) so the same buffer is reusable across all candidate block_m's.
_K_CANDIDATE_BLOCK_M: tuple[int, ...] = (8, 16, 32, 64, 96, 128, 192)
_K_MIN_CANDIDATE_BLOCK_M = 8
_K_MAX_CANDIDATE_BLOCK_M = 192
_K_LCM_CANDIDATE_BLOCK_M = 384


def _get_num_max_pool_tokens(
    *, num_ranks: int, num_max_tokens_per_rank: int, num_topk: int, num_experts_per_rank: int
) -> int:
    num_max_recv_tokens = num_ranks * num_max_tokens_per_rank
    num_max_experts_per_token = min(num_topk, num_experts_per_rank)
    return _align_up(
        num_max_recv_tokens * num_max_experts_per_token
        + num_experts_per_rank * (_K_MAX_CANDIDATE_BLOCK_M - 1),
        _K_LCM_CANDIDATE_BLOCK_M,
    )


def _get_aligned_num_max_tokens_per_rank(config: MegaMoeConfig) -> int:
    return _align_up(config.num_max_tokens_per_rank, _K_LCM_CANDIDATE_BLOCK_M)


def _get_num_max_padded_sf_pool_tokens(num_max_pool_tokens: int) -> int:
    return max(
        (num_max_pool_tokens // block_m) * _align_up(block_m, 128)
        for block_m in _K_CANDIDATE_BLOCK_M
    )


def get_deepgemm_workspace_layout(config: MegaMoeConfig) -> DeepGemmWorkspaceLayout:
    launch = get_deepgemm_launch_config(config)
    num_ranks = config.num_processes
    num_experts = config.num_experts
    num_experts_per_rank = config.num_experts_per_rank
    aligned_num_max_tokens_per_rank = _get_aligned_num_max_tokens_per_rank(config)
    num_max_recv_tokens_per_expert = num_ranks * aligned_num_max_tokens_per_rank
    num_max_pool_tokens = _get_num_max_pool_tokens(
        num_ranks=num_ranks,
        num_max_tokens_per_rank=aligned_num_max_tokens_per_rank,
        num_topk=config.num_topk,
        num_experts_per_rank=num_experts_per_rank,
    )
    num_max_pool_blocks = num_max_pool_tokens // _K_MIN_CANDIDATE_BLOCK_M
    num_padded_sf_pool_tokens = _get_num_max_padded_sf_pool_tokens(num_max_pool_tokens)

    barrier_offset = 0
    cursor = 32

    expert_send_count_offset = cursor
    cursor += num_experts * 8

    expert_recv_count_offset = cursor
    cursor += num_experts * 8

    expert_recv_count_sum_offset = cursor
    cursor += num_experts_per_rank * 8

    l1_arrival_count_offset = cursor
    cursor += _align_up(num_max_pool_blocks, 2) * 4

    l2_arrival_mask_offset = cursor
    cursor += num_max_pool_blocks * 8

    src_token_topk_idx_offset = cursor
    cursor += num_experts_per_rank * num_ranks * num_max_recv_tokens_per_expert * 4

    token_src_metadata_offset = cursor
    token_src_metadata_bytes = num_max_pool_tokens * 12
    cursor += token_src_metadata_bytes

    total_bytes = _align_up(cursor, 16)
    return DeepGemmWorkspaceLayout(
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_experts_per_rank=num_experts_per_rank,
        num_max_tokens_per_rank=aligned_num_max_tokens_per_rank,
        num_topk=config.num_topk,
        block_m=launch.block_m,
        num_max_recv_tokens_per_expert=num_max_recv_tokens_per_expert,
        num_max_pool_tokens=num_max_pool_tokens,
        num_max_pool_blocks=num_max_pool_blocks,
        num_padded_sf_pool_tokens=num_padded_sf_pool_tokens,
        token_src_metadata_bytes=token_src_metadata_bytes,
        barrier_offset=barrier_offset,
        expert_send_count_offset=expert_send_count_offset,
        expert_recv_count_offset=expert_recv_count_offset,
        expert_recv_count_sum_offset=expert_recv_count_sum_offset,
        l1_arrival_count_offset=l1_arrival_count_offset,
        l2_arrival_mask_offset=l2_arrival_mask_offset,
        src_token_topk_idx_offset=src_token_topk_idx_offset,
        token_src_metadata_offset=token_src_metadata_offset,
        total_bytes=total_bytes,
    )


def get_deepgemm_symm_buffer_layout(config: MegaMoeConfig) -> DeepGemmSymmBufferLayout:
    workspace = get_deepgemm_workspace_layout(config)
    aligned_num_max_tokens_per_rank = workspace.num_max_tokens_per_rank
    cursor = workspace.total_bytes

    input_token_offset = cursor
    cursor += aligned_num_max_tokens_per_rank * config.hidden

    input_sf_offset = cursor
    cursor += aligned_num_max_tokens_per_rank * (config.hidden // 32)

    input_topk_idx_offset = cursor
    cursor += aligned_num_max_tokens_per_rank * config.num_topk * 8

    input_topk_weights_offset = cursor
    cursor += aligned_num_max_tokens_per_rank * config.num_topk * 4

    l1_token_offset = cursor
    cursor += workspace.num_max_pool_tokens * config.hidden

    l1_sf_offset = cursor
    cursor += workspace.num_padded_sf_pool_tokens * (config.hidden // 32)

    l1_topk_weights_offset = cursor
    cursor += workspace.num_max_pool_tokens * 4

    l2_token_offset = cursor
    cursor += workspace.num_max_pool_tokens * config.intermediate_hidden

    l2_sf_offset = cursor
    cursor += workspace.num_padded_sf_pool_tokens * (config.intermediate_hidden // 32)

    combine_token_offset = cursor
    cursor += config.num_topk * aligned_num_max_tokens_per_rank * config.hidden * 2

    return DeepGemmSymmBufferLayout(
        workspace_bytes=workspace.total_bytes,
        input_token_offset=input_token_offset,
        input_sf_offset=input_sf_offset,
        input_topk_idx_offset=input_topk_idx_offset,
        input_topk_weights_offset=input_topk_weights_offset,
        l1_token_offset=l1_token_offset,
        l1_sf_offset=l1_sf_offset,
        l1_topk_weights_offset=l1_topk_weights_offset,
        l2_token_offset=l2_token_offset,
        l2_sf_offset=l2_sf_offset,
        combine_token_offset=combine_token_offset,
        total_bytes=cursor,
    )


def _tensor_offset_bytes(base: torch.Tensor, view: torch.Tensor) -> int:
    return int(view.data_ptr()) - int(base.data_ptr())


def validate_runtime_symm_buffer_layout(
    *, symm_buffer: Any, layout: DeepGemmSymmBufferLayout, config: MegaMoeConfig
) -> None:
    workspace = get_deepgemm_workspace_layout(config)
    actual_offsets = {
        "input_token_offset": _tensor_offset_bytes(symm_buffer.buffer, symm_buffer.x),
        "input_sf_offset": _tensor_offset_bytes(symm_buffer.buffer, symm_buffer.x_sf),
        "input_topk_idx_offset": _tensor_offset_bytes(symm_buffer.buffer, symm_buffer.topk_idx),
        "input_topk_weights_offset": _tensor_offset_bytes(
            symm_buffer.buffer, symm_buffer.topk_weights
        ),
        "l1_token_offset": _tensor_offset_bytes(symm_buffer.buffer, symm_buffer.l1_acts),
        "l1_sf_offset": _tensor_offset_bytes(symm_buffer.buffer, symm_buffer.l1_acts_sf),
        "l2_token_offset": _tensor_offset_bytes(symm_buffer.buffer, symm_buffer.l2_acts),
        "l2_sf_offset": _tensor_offset_bytes(symm_buffer.buffer, symm_buffer.l2_acts_sf),
    }
    expected_offsets = {
        "input_token_offset": layout.input_token_offset,
        "input_sf_offset": layout.input_sf_offset,
        "input_topk_idx_offset": layout.input_topk_idx_offset,
        "input_topk_weights_offset": layout.input_topk_weights_offset,
        "l1_token_offset": layout.l1_token_offset,
        "l1_sf_offset": layout.l1_sf_offset,
        "l2_token_offset": layout.l2_token_offset,
        "l2_sf_offset": layout.l2_sf_offset,
    }
    for key, expected in expected_offsets.items():
        actual = actual_offsets[key]
        if actual != expected:
            raise ValueError(
                f"DeepGEMM symm buffer offset mismatch for {key}: expected {expected}, got {actual}"
            )

    expected_shapes = {
        "x": (workspace.num_max_tokens_per_rank, config.hidden),
        "x_sf": (workspace.num_max_tokens_per_rank, config.hidden // 128),
        "topk_idx": (workspace.num_max_tokens_per_rank, config.num_topk),
        "topk_weights": (workspace.num_max_tokens_per_rank, config.num_topk),
        "l1_acts": (workspace.num_max_pool_tokens, config.hidden),
        "l1_acts_sf": (workspace.num_padded_sf_pool_tokens, config.hidden // 128),
        "l2_acts": (workspace.num_max_pool_tokens, config.intermediate_hidden),
        "l2_acts_sf": (workspace.num_padded_sf_pool_tokens, config.intermediate_hidden // 128),
    }
    for name, expected_shape in expected_shapes.items():
        actual_shape = tuple(getattr(symm_buffer, name).shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"DeepGEMM symm buffer shape mismatch for {name}: "
                f"expected {expected_shape}, got {actual_shape}"
            )

    if int(symm_buffer.buffer.nbytes) != layout.total_bytes:
        raise ValueError(
            f"DeepGEMM symm buffer total size mismatch: "
            f"expected {layout.total_bytes}, got {int(symm_buffer.buffer.nbytes)}"
        )


def build_dispatch_reference(
    *, topk_idx: torch.Tensor, rank_idx: int, layout: DeepGemmWorkspaceLayout
) -> DispatchReference:
    num_tokens, num_topk = topk_idx.shape
    if num_topk != layout.num_topk:
        raise ValueError("topk shape does not match workspace layout")

    expert_send_count = torch.zeros(layout.num_experts, dtype=torch.int64, device=topk_idx.device)
    expert_recv_count = torch.zeros(
        (layout.num_ranks, layout.num_experts_per_rank), dtype=torch.int64, device=topk_idx.device
    )
    expert_recv_count_sum = torch.zeros(
        layout.num_experts_per_rank, dtype=torch.int64, device=topk_idx.device
    )
    src_token_topk_idx = torch.full(
        (layout.num_experts_per_rank, layout.num_ranks, layout.num_max_recv_tokens_per_expert),
        -1,
        dtype=torch.int32,
        device=topk_idx.device,
    )

    for token_idx in range(num_tokens):
        for topk_slot in range(layout.num_topk):
            expert_idx = int(topk_idx[token_idx, topk_slot].item())
            if expert_idx < 0:
                continue
            dst_rank_idx = expert_idx // layout.num_experts_per_rank
            dst_local_expert_idx = expert_idx % layout.num_experts_per_rank
            dst_slot_idx = int(expert_send_count[expert_idx].item())
            if dst_slot_idx >= layout.num_max_recv_tokens_per_expert:
                raise ValueError("dispatch reference overflowed expert recv capacity")
            token_topk_idx = token_idx * layout.num_topk + topk_slot
            src_token_topk_idx[dst_local_expert_idx, rank_idx, dst_slot_idx] = token_topk_idx
            expert_send_count[expert_idx] += 1
            expert_recv_count[rank_idx, dst_local_expert_idx] += 1
            expert_recv_count_sum[dst_local_expert_idx] += 1

    return DispatchReference(
        expert_send_count=expert_send_count,
        expert_recv_count=expert_recv_count,
        expert_recv_count_sum=expert_recv_count_sum,
        src_token_topk_idx=src_token_topk_idx,
    )


def _select_round_robin_rank(rank_counts: list[int], token_idx_in_expert: int) -> tuple[int, int]:
    remaining = list(rank_counts)
    offset = 0
    slot_idx = token_idx_in_expert
    while True:
        active_ranks = [rank_idx for rank_idx, value in enumerate(remaining) if value > 0]
        if not active_ranks:
            raise ValueError("round-robin pull encountered no active ranks")
        length = min(remaining[rank_idx] for rank_idx in active_ranks)
        num_round_tokens = length * len(active_ranks)
        if slot_idx < num_round_tokens:
            slot_idx_in_round = slot_idx % len(active_ranks)
            return active_ranks[slot_idx_in_round], offset + (slot_idx // len(active_ranks))
        slot_idx -= num_round_tokens
        offset += length
        for rank_idx in active_ranks:
            remaining[rank_idx] -= length


def build_pull_reference(
    *, dispatch_reference: DispatchReference, layout: DeepGemmWorkspaceLayout
) -> PullReference:
    device = dispatch_reference.expert_recv_count.device
    expert_pool_block_offset = torch.zeros(
        layout.num_experts_per_rank, dtype=torch.int32, device=device
    )
    pool_src_rank_idx = torch.full(
        (layout.num_max_pool_tokens,), -1, dtype=torch.int32, device=device
    )
    pool_src_token_idx = torch.full(
        (layout.num_max_pool_tokens,), -1, dtype=torch.int32, device=device
    )
    pool_src_topk_idx = torch.full(
        (layout.num_max_pool_tokens,), -1, dtype=torch.int32, device=device
    )
    pool_src_token_topk_idx = torch.full(
        (layout.num_max_pool_tokens,), -1, dtype=torch.int32, device=device
    )
    l1_arrival_count = torch.zeros(layout.num_max_pool_blocks, dtype=torch.int32, device=device)

    running_pool_block_offset = 0
    expert_recv_count = dispatch_reference.expert_recv_count.cpu()
    expert_recv_count_sum = dispatch_reference.expert_recv_count_sum.cpu()
    src_token_topk_idx = dispatch_reference.src_token_topk_idx.cpu()

    for local_expert_idx in range(layout.num_experts_per_rank):
        expert_pool_block_offset[local_expert_idx] = running_pool_block_offset
        num_tokens = int(expert_recv_count_sum[local_expert_idx].item())
        rank_counts = [
            int(expert_recv_count[rank_idx, local_expert_idx].item())
            for rank_idx in range(layout.num_ranks)
        ]
        for token_idx_in_expert in range(num_tokens):
            src_rank_idx, token_idx_in_rank = _select_round_robin_rank(
                rank_counts, token_idx_in_expert
            )
            token_topk_linear_idx = int(
                src_token_topk_idx[local_expert_idx, src_rank_idx, token_idx_in_rank].item()
            )
            if token_topk_linear_idx < 0:
                raise ValueError("dispatch pull reference encountered an uninitialized source slot")
            pool_token_idx = running_pool_block_offset * layout.block_m + token_idx_in_expert
            pool_src_rank_idx[pool_token_idx] = src_rank_idx
            pool_src_token_idx[pool_token_idx] = token_topk_linear_idx // layout.num_topk
            pool_src_topk_idx[pool_token_idx] = token_topk_linear_idx % layout.num_topk
            pool_src_token_topk_idx[pool_token_idx] = token_topk_linear_idx
            l1_arrival_count[running_pool_block_offset + token_idx_in_expert // layout.block_m] += 1
        running_pool_block_offset += _align_up(num_tokens, layout.block_m) // layout.block_m

    return PullReference(
        expert_pool_block_offset=expert_pool_block_offset,
        pool_src_rank_idx=pool_src_rank_idx,
        pool_src_token_idx=pool_src_token_idx,
        pool_src_topk_idx=pool_src_topk_idx,
        pool_src_token_topk_idx=pool_src_token_topk_idx,
        l1_arrival_count=l1_arrival_count,
    )


def build_scheduler_reference(
    *,
    config: MegaMoeConfig,
    launch: DeepGemmLaunchConfig,
    dispatch_reference: DispatchReference,
    pull_reference: PullReference,
) -> SchedulerReference:
    expert_num_tokens = dispatch_reference.expert_recv_count_sum.to(torch.int32).cpu()
    expert_pool_block_offset = pull_reference.expert_pool_block_offset.to(torch.int32).cpu()
    num_l1_block_ns = (config.intermediate_hidden * 2) // launch.block_n
    num_l2_block_ns = config.hidden // launch.block_n
    num_l1_block_ks = config.hidden // launch.block_k
    num_l2_block_ks = config.intermediate_hidden // launch.block_k

    def get_num_tokens(expert_idx: int) -> int:
        if expert_idx >= config.num_experts_per_rank:
            return 0
        return int(expert_num_tokens[expert_idx].item())

    def get_pool_block_offset(expert_idx: int) -> int:
        if expert_idx >= config.num_experts_per_rank:
            return int(expert_pool_block_offset[-1].item())
        return int(expert_pool_block_offset[expert_idx].item())

    blocks: list[SchedulerBlockReference] = []
    num_linear1_blocks = 0
    num_linear2_blocks = 0

    for block_idx_seed in range(launch.num_sms):
        next_phase = "Linear1"
        current_local_expert_idx = 0
        current_num_tokens = get_num_tokens(0)
        current_pool_block_offset = get_pool_block_offset(0)
        block_idx = block_idx_seed

        def get_wave_expert_end_idx() -> int:
            return _align_up(current_local_expert_idx + 1, launch.num_experts_per_wave)

        def get_current_num_m_blocks() -> int:
            return (current_num_tokens + launch.block_m - 1) // launch.block_m

        def advance_expert_idx() -> tuple[int, int, int]:
            next_local_expert_idx = current_local_expert_idx + 1
            next_pool_block_offset = current_pool_block_offset + get_current_num_m_blocks()
            next_num_tokens = get_num_tokens(next_local_expert_idx)
            return next_local_expert_idx, next_num_tokens, next_pool_block_offset

        def set_expert_idx(expert_idx: int) -> tuple[int, int, int]:
            return expert_idx, get_num_tokens(expert_idx), get_pool_block_offset(expert_idx)

        while True:
            if current_local_expert_idx >= config.num_experts_per_rank:
                break
            if next_phase == "Linear1":
                wave_end_expert_idx = get_wave_expert_end_idx()
                found_block = False
                while current_local_expert_idx < wave_end_expert_idx:
                    num_m_blocks = get_current_num_m_blocks()
                    m_block_idx = block_idx // num_l1_block_ns
                    if m_block_idx < num_m_blocks:
                        n_block_idx = block_idx - m_block_idx * num_l1_block_ns
                        valid_m = min(
                            current_num_tokens - m_block_idx * launch.block_m, launch.block_m
                        )
                        blocks.append(
                            SchedulerBlockReference(
                                block_idx_seed=block_idx_seed,
                                phase="Linear1",
                                local_expert_idx=current_local_expert_idx,
                                num_k_blocks=num_l1_block_ks,
                                m_block_idx=m_block_idx,
                                n_block_idx=n_block_idx,
                                pool_block_offset=current_pool_block_offset,
                                valid_m=max(valid_m, 0),
                            )
                        )
                        num_linear1_blocks += 1
                        block_idx += launch.num_sms
                        found_block = True
                        break
                    block_idx -= num_m_blocks * num_l1_block_ns
                    (current_local_expert_idx, current_num_tokens, current_pool_block_offset) = (
                        advance_expert_idx()
                    )
                if found_block:
                    continue
                next_phase = "Linear2"
                (current_local_expert_idx, current_num_tokens, current_pool_block_offset) = (
                    set_expert_idx(
                        _align_down(
                            max(current_local_expert_idx - 1, 0), launch.num_experts_per_wave
                        )
                    )
                )
            else:
                wave_end_expert_idx = get_wave_expert_end_idx()
                found_block = False
                while current_local_expert_idx < wave_end_expert_idx:
                    num_m_blocks = get_current_num_m_blocks()
                    if block_idx < num_m_blocks * num_l2_block_ns:
                        m_block_idx = block_idx // num_l2_block_ns
                        n_block_idx = block_idx - m_block_idx * num_l2_block_ns
                        valid_m = min(
                            current_num_tokens - m_block_idx * launch.block_m, launch.block_m
                        )
                        blocks.append(
                            SchedulerBlockReference(
                                block_idx_seed=block_idx_seed,
                                phase="Linear2",
                                local_expert_idx=current_local_expert_idx,
                                num_k_blocks=num_l2_block_ks,
                                m_block_idx=m_block_idx,
                                n_block_idx=n_block_idx,
                                pool_block_offset=current_pool_block_offset,
                                valid_m=max(valid_m, 0),
                            )
                        )
                        num_linear2_blocks += 1
                        block_idx += launch.num_sms
                        found_block = True
                        break
                    block_idx -= num_m_blocks * num_l2_block_ns
                    (current_local_expert_idx, current_num_tokens, current_pool_block_offset) = (
                        advance_expert_idx()
                    )
                if found_block:
                    continue
                next_phase = "Linear1"

    return SchedulerReference(
        expert_num_tokens=expert_num_tokens.to(dispatch_reference.expert_recv_count_sum.device),
        expert_pool_block_offset=expert_pool_block_offset.to(
            dispatch_reference.expert_recv_count_sum.device
        ),
        blocks=tuple(blocks),
        num_linear1_blocks=num_linear1_blocks,
        num_linear2_blocks=num_linear2_blocks,
    )


def _get_block_config_for_mega_moe(
    *, num_ranks: int, num_experts: int, num_tokens: int, num_topk: int
) -> tuple[int, int, int, int]:
    """Pick `(cluster_size, block_m, store_block_m, num_epilogue_warpgroups)` by
    expected tokens-per-expert. Mirrors `get_block_config_for_mega_moe` in
    `csrc/jit_kernels/heuristics/mega_moe.hpp` so the schedule tracks upstream's
    per-batch-size tuning instead of always picking the prefill-sized 192."""
    expected = float(num_tokens) * num_ranks * num_topk / num_experts
    if expected <= 8.5:
        cfg = (2, 16, 8, 2)
    elif expected <= 16.5:
        cfg = (2, 32, 16, 2)
    elif expected <= 32.5:
        cfg = (2, 64, 32, 1)
    elif expected <= 64.5:
        cfg = (2, 96, 16, 2)
    elif expected <= 96.5:
        cfg = (2, 128, 32, 2)
    else:
        cfg = (2, 192, 32, 2)
    assert cfg[1] in _K_CANDIDATE_BLOCK_M
    return cfg


def _get_num_experts_per_wave_for_mega_moe(
    *,
    num_experts_per_rank: int,
    num_tokens: int,
    num_topk: int,
    intermediate_hidden: int,
    block_m: int,
    block_n: int,
    num_sms: int,
) -> int:
    expected_tokens_per_expert = num_tokens * num_topk / num_experts_per_rank
    if expected_tokens_per_expert < 1:
        # Most experts don't have tokens; calculate all experts at once.
        return num_experts_per_rank

    imbalance_factor = 2
    # L1 GEMM emits gate+up fused (2 * intermediate_hidden wide), not just intermediate_hidden.
    num_m_blocks = (math.ceil(expected_tokens_per_expert) + block_m - 1) // block_m
    num_n_blocks = (2 * intermediate_hidden) // block_n
    num_l1_blocks_per_expert = num_m_blocks * num_n_blocks
    if num_l1_blocks_per_expert > 0:
        num_experts_per_wave = (
            imbalance_factor * num_sms + num_l1_blocks_per_expert - 1
        ) // num_l1_blocks_per_expert
    else:
        num_experts_per_wave = 1
    num_experts_per_wave = min(num_experts_per_wave, num_experts_per_rank)
    while (
        num_experts_per_wave < num_experts_per_rank
        and num_experts_per_rank % num_experts_per_wave != 0
    ):
        num_experts_per_wave += 1
    return num_experts_per_wave


def _get_num_sms_for_mega_moe() -> int:
    override = os.environ.get("TIRX_DEEPGEMM_NUM_SMS_OVERRIDE")
    if override is not None:
        return int(override)
    if torch.cuda.is_available():
        return int(
            torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        )
    raise RuntimeError(
        "MegaMoE launch requires CUDA to infer kNumSMs; set TIRX_DEEPGEMM_NUM_SMS_OVERRIDE to override"
    )


def get_deepgemm_launch_config(config: MegaMoeConfig) -> DeepGemmLaunchConfig:
    block_n = 128
    block_k = 128
    num_sms = _get_num_sms_for_mega_moe()
    if num_sms <= 1:
        raise ValueError("MegaMoE launch must satisfy DeepGEMM kNumSMs > 1")
    if num_sms % 2 != 0:
        raise ValueError("MegaMoE launch must satisfy DeepGEMM kNumSMs % 2 == 0")
    num_ctas_per_cluster_env = os.environ.get("TIRX_DEEPGEMM_NUM_CTAS_PER_CLUSTER_OVERRIDE")
    cluster_size, block_m, store_block_m, num_epilogue_wgs = _get_block_config_for_mega_moe(
        num_ranks=config.num_processes,
        num_experts=config.num_experts,
        num_tokens=config.num_tokens,
        num_topk=config.num_topk,
    )
    if num_ctas_per_cluster_env is not None and int(num_ctas_per_cluster_env) != cluster_size:
        raise ValueError(
            f"MegaMoE must use DeepGEMM-equivalent num_ctas_per_cluster={cluster_size}"
        )
    load_block_m = block_m // 2
    launch = DeepGemmLaunchConfig(
        num_sms=num_sms,
        num_ctas_per_cluster=cluster_size,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        load_block_m=load_block_m,
        load_block_n=block_n,
        store_block_m=store_block_m,
        num_dispatch_threads=128,
        num_non_epilogue_threads=128,
        num_epilogue_threads=num_epilogue_wgs * 128,
        num_experts_per_wave=_get_num_experts_per_wave_for_mega_moe(
            num_experts_per_rank=config.num_experts_per_rank,
            num_tokens=config.num_tokens,
            num_topk=config.num_topk,
            intermediate_hidden=config.intermediate_hidden,
            block_m=block_m,
            block_n=block_n,
            num_sms=num_sms,
        ),
        num_topk=config.num_topk,
        hidden=config.hidden,
        intermediate_hidden=config.intermediate_hidden,
    )
    wg_block_m = launch.block_m // num_epilogue_wgs
    atom_m = 8
    if launch.num_epilogue_warps != num_epilogue_wgs * 4:
        raise ValueError("MegaMoE launch num_epilogue_warps must equal kNumEpilogueWarpgroups * 4")
    if launch.epilogue_warp_start_idx % 4 != 0 or launch.num_epilogue_warps % 4 != 0:
        raise ValueError("MegaMoE launch must satisfy DeepGEMM epilogue warpgroup alignment")
    if launch.block_m % num_epilogue_wgs != 0:
        raise ValueError("MegaMoE launch must satisfy BLOCK_M % kNumEpilogueWarpgroups == 0")
    if wg_block_m % launch.store_block_m != 0:
        raise ValueError("MegaMoE launch must satisfy WG_BLOCK_M % STORE_BLOCK_M == 0")
    if launch.store_block_m % atom_m != 0:
        raise ValueError("MegaMoE launch must satisfy STORE_BLOCK_M % ATOM_M == 0")
    # Upstream relaxed `WG_BLOCK_M % 32 == 0` to a runtime lane-bound guard in the
    # SF weight-cache load (see kernel body). Keep only the `atom_m | 32` part here.
    if 32 % atom_m != 0:
        raise ValueError("MegaMoE launch must satisfy 32 % ATOM_M == 0")
    if launch.block_n != 128:
        raise ValueError("MegaMoE launch must satisfy BLOCK_N == 128")
    if launch.block_k != launch.block_n:
        raise ValueError("MegaMoE launch must satisfy BLOCK_K == BLOCK_N")
    if config.num_experts_per_rank % launch.num_experts_per_wave != 0:
        raise ValueError("MegaMoE launch must satisfy kNumExpertsPerRank % kNumExpertsPerWave == 0")
    if (config.intermediate_hidden * 2 // launch.block_n) % 2 != 0:
        raise ValueError("MegaMoE launch must satisfy kNumL1BlockNs % 2 == 0")
    if (config.hidden // launch.block_n) % 2 != 0:
        raise ValueError("MegaMoE launch must satisfy kNumL2BlockNs % 2 == 0")
    return launch


def get_tirx_launch_param_tags() -> list[str]:
    return ["blockIdx.x", "clusterCtaIdx.x", "threadIdx.x", "tirx.use_dyn_shared_memory"]


def get_tirx_dynamic_shared_memory_bytes(config: MegaMoeConfig) -> int:
    launch = get_deepgemm_launch_config(config)
    num_experts = config.num_experts
    hidden = config.hidden
    intermediate_hidden = config.intermediate_hidden
    l1_out_block_n = launch.block_n // 2
    sf_block_m = _align_up(launch.block_m, 128)
    num_epilogue_stages = 2
    sm100_smem_capacity = 232448
    shared_alignment = 1024
    num_epilogue_wgs = launch.num_epilogue_warps // 4
    smem_expert_count_size = _align_up(num_experts * 4, shared_alignment)
    smem_send_buffer_size = _align_up(hidden * launch.num_dispatch_warps, shared_alignment)
    smem_dispatch_size = smem_expert_count_size + smem_send_buffer_size
    smem_cd_l1_size = num_epilogue_wgs * launch.store_block_m * l1_out_block_n * 2
    smem_cd_l2_size = num_epilogue_wgs * launch.store_block_m * launch.block_n * 2
    smem_cd_size = max(smem_cd_l1_size, smem_cd_l2_size)
    smem_a_size_per_stage = launch.load_block_m * launch.block_k
    smem_b_size_per_stage = launch.load_block_n * launch.block_k
    smem_sfa_size_per_stage = sf_block_m * 4
    smem_sfb_size_per_stage = launch.block_n * 4
    smem_amax_reduction_size = launch.store_block_m * launch.num_epilogue_warps * 4
    smem_tmem_ptr_size = 4
    smem_per_stage = (
        smem_a_size_per_stage
        + smem_b_size_per_stage
        + smem_sfa_size_per_stage
        + smem_sfb_size_per_stage
        + 16
    )
    smem_fixed = (
        smem_dispatch_size
        + smem_cd_size
        + smem_amax_reduction_size
        + (launch.num_dispatch_warps + num_epilogue_stages * 2 + launch.num_epilogue_warps * 2) * 8
        + smem_tmem_ptr_size
    )
    num_stages = max(2, (sm100_smem_capacity - smem_fixed) // smem_per_stage)
    num_total_barriers = (
        launch.num_dispatch_warps
        + num_stages * 2
        + num_epilogue_stages * 2
        + launch.num_epilogue_warps * 2
    )
    smem_expert_count_offset = 0
    smem_send_buffer_offset = smem_expert_count_offset + smem_expert_count_size
    smem_gemm_base_offset = smem_send_buffer_offset + smem_send_buffer_size
    smem_cd_offset = smem_gemm_base_offset
    smem_a_offset = smem_cd_offset + smem_cd_size
    smem_b_offset = smem_a_offset + num_stages * smem_a_size_per_stage
    smem_sfa_offset = smem_b_offset + num_stages * smem_b_size_per_stage
    smem_sfb_offset = smem_sfa_offset + num_stages * smem_sfa_size_per_stage
    smem_amax_reduction_offset = smem_sfb_offset + num_stages * smem_sfb_size_per_stage
    smem_barrier_offset = _align_up(smem_amax_reduction_offset + smem_amax_reduction_size, 8)
    smem_tmem_ptr_offset = smem_barrier_offset + num_total_barriers * 8
    return smem_tmem_ptr_offset + smem_tmem_ptr_size


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _get_tma_aligned_size(x: int, element_size: int) -> int:
    return _align_up(x * element_size, 16) // element_size


def _tensor_map_swizzle_from_mode(mode: int, base: int = 0) -> int:
    if base != 0:
        if base == 32 and mode == 128:
            return 4
        raise ValueError(f"Unsupported tensor map swizzle base={base}, mode={mode}")
    if mode in (0, 16):
        return 0
    if mode == 32:
        return 1
    if mode == 64:
        return 2
    if mode == 128:
        return 3
    raise ValueError(f"Unsupported tensor map swizzle mode={mode}")


def _torch_dtype_to_tvm_dtype(t: torch.Tensor) -> str:
    if t.dtype == torch.int8:
        return "int8"
    if t.dtype == torch.uint8:
        return "uint8"
    if t.dtype == torch.int32:
        return "int32"
    if t.dtype == torch.uint32:
        return "uint32"
    if t.dtype == torch.float32:
        return "float32"
    if t.dtype == torch.bfloat16:
        return "bfloat16"
    if t.dtype == torch.float8_e4m3fn:
        return "float8_e4m3fn"
    raise TypeError(f"Unsupported tensor dtype for TMA descriptor: {t.dtype}")


class _AlignedTensorMap:
    def __init__(self) -> None:
        self._storage = ctypes.create_string_buffer(128 + 64)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 63) & ~63)


_CUDA_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B = 14
_CUDA_TENSOR_MAP_INTERLEAVE_NONE = 0
_CUDA_TENSOR_MAP_L2_PROMOTION_L2_256B = 3
_CUDA_TENSOR_MAP_FLOAT_OOB_FILL_NONE = 0
_CUDA_DRIVER = None


def _get_cuda_driver() -> ctypes.CDLL:
    global _CUDA_DRIVER
    if _CUDA_DRIVER is None:
        libcuda_path = ctypes.util.find_library("cuda") or "libcuda.so.1"
        driver = ctypes.CDLL(libcuda_path)
        driver.cuInit.argtypes = [ctypes.c_uint]
        driver.cuInit.restype = ctypes.c_int
        result = driver.cuInit(0)
        if result != 0:
            raise RuntimeError(f"cuInit failed with CUresult={result}")
        driver.cuTensorMapEncodeTiled.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        driver.cuTensorMapEncodeTiled.restype = ctypes.c_int
        _CUDA_DRIVER = driver
    return _CUDA_DRIVER


def _encode_fp4_align16_tma_2d_desc(
    *,
    desc: _AlignedTensorMap,
    tensor: torch.Tensor,
    gmem_inner_dim: int,
    gmem_outer_dim: int,
    smem_inner_dim: int,
    smem_outer_dim: int,
    gmem_outer_stride_bytes: int,
    swizzle: int,
) -> None:
    global_shape = (ctypes.c_uint64 * 2)(int(gmem_inner_dim), int(gmem_outer_dim))
    global_strides = (ctypes.c_uint64 * 1)(int(gmem_outer_stride_bytes))
    box_dim = (ctypes.c_uint32 * 2)(int(smem_inner_dim), int(smem_outer_dim))
    element_strides = (ctypes.c_uint32 * 2)(1, 1)
    result = _get_cuda_driver().cuTensorMapEncodeTiled(
        desc.ptr,
        _CUDA_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
        ctypes.c_uint32(2),
        ctypes.c_void_p(int(tensor.data_ptr())),
        global_shape,
        global_strides,
        box_dim,
        element_strides,
        _CUDA_TENSOR_MAP_INTERLEAVE_NONE,
        int(swizzle),
        _CUDA_TENSOR_MAP_L2_PROMOTION_L2_256B,
        _CUDA_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    if result != 0:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed for FP4 align16 with CUresult={result}")


def _encode_tma_2d_desc(
    *,
    encode_tensormap: Any,
    tensor: torch.Tensor,
    gmem_inner_dim: int,
    gmem_outer_dim: int,
    smem_inner_dim: int,
    smem_outer_dim: int,
    gmem_outer_stride: int,
    swizzle_mode: int,
    swizzle_base: int = 0,
    tensor_dtype: Any | None = None,
) -> _AlignedTensorMap:
    elem_size = int(tensor.element_size())
    if swizzle_mode != 0:
        smem_inner_dim = swizzle_mode // elem_size
    desc = _AlignedTensorMap()
    swizzle = _tensor_map_swizzle_from_mode(swizzle_mode, swizzle_base)
    if tensor_dtype == "float4_e2m1fn":
        _encode_fp4_align16_tma_2d_desc(
            desc=desc,
            tensor=tensor,
            gmem_inner_dim=gmem_inner_dim,
            gmem_outer_dim=gmem_outer_dim,
            smem_inner_dim=smem_inner_dim,
            smem_outer_dim=smem_outer_dim,
            gmem_outer_stride_bytes=int(gmem_outer_stride * elem_size),
            swizzle=swizzle,
        )
    else:
        encode_tensormap(
            desc.ptr,
            _torch_dtype_to_tvm_dtype(tensor) if tensor_dtype is None else tensor_dtype,
            2,
            ctypes.c_void_p(int(tensor.data_ptr())),
            int(gmem_inner_dim),
            int(gmem_outer_dim),
            int(gmem_outer_stride * elem_size),
            int(smem_inner_dim),
            int(smem_outer_dim),
            1,
            1,
            0,
            swizzle,
            3,
            0,
        )
    return desc


def _encode_tma_sf_desc(
    *,
    encode_tensormap: Any,
    tensor: torch.Tensor,
    shape_mn: int,
    shape_k: int,
    block_mn: int,
    gran_k: int,
    num_groups: int,
) -> _AlignedTensorMap:
    aligned_shape_mn = _get_tma_aligned_size(shape_mn, int(tensor.element_size()))
    packed_shape_k = _ceil_div(shape_k, gran_k * (1 if tensor.dtype == torch.float32 else 4))
    return _encode_tma_2d_desc(
        encode_tensormap=encode_tensormap,
        tensor=tensor,
        gmem_inner_dim=aligned_shape_mn,
        gmem_outer_dim=packed_shape_k * num_groups,
        smem_inner_dim=block_mn,
        smem_outer_dim=1,
        gmem_outer_stride=aligned_shape_mn,
        swizzle_mode=0,
    )


def _build_tirx_tensor_maps(
    *,
    case: MegaMoeCase,
    l1_acts: torch.Tensor,
    l2_acts: torch.Tensor,
    l1_weights: torch.Tensor,
    l1_weights_sf: torch.Tensor,
    l2_weights: torch.Tensor,
    l2_weights_sf: torch.Tensor,
) -> dict[str, _AlignedTensorMap]:
    import tvm

    encode_tensormap = tvm.get_global_func("runtime.cuTensorMapEncodeTiled")
    launch = get_deepgemm_launch_config(case.config)
    workspace = case.workspace_layout
    gran_k = 32
    swizzle_acts_mode = 128
    swizzle_weights_mode = 128
    num_experts_per_rank = case.config.num_experts_per_rank
    sf_block_m = _align_up(launch.block_m, 128)

    return {
        "tensor_map_l1_acts": _encode_tma_2d_desc(
            encode_tensormap=encode_tensormap,
            tensor=l1_acts,
            gmem_inner_dim=case.config.hidden,
            gmem_outer_dim=workspace.num_max_pool_tokens,
            smem_inner_dim=launch.block_k,
            smem_outer_dim=launch.load_block_m,
            gmem_outer_stride=int(l1_acts.stride(-2)),
            swizzle_mode=swizzle_acts_mode,
            tensor_dtype="float8_e4m3fn",
        ),
        "tensor_map_l1_acts_sf": _encode_tma_sf_desc(
            encode_tensormap=encode_tensormap,
            tensor=case.symm_buffer.l1_acts_sf,
            shape_mn=workspace.num_padded_sf_pool_tokens,
            shape_k=case.config.hidden,
            block_mn=sf_block_m,
            gran_k=gran_k,
            num_groups=1,
        ),
        "tensor_map_l1_weights": _encode_tma_2d_desc(
            encode_tensormap=encode_tensormap,
            tensor=l1_weights,
            gmem_inner_dim=case.config.hidden,
            gmem_outer_dim=num_experts_per_rank * case.config.intermediate_hidden * 2,
            smem_inner_dim=launch.block_k,
            smem_outer_dim=launch.load_block_n,
            gmem_outer_stride=int(l1_weights.stride(-2)),
            swizzle_mode=swizzle_weights_mode,
            tensor_dtype="float4_e2m1fn",
        ),
        "tensor_map_l1_weights_sf": _encode_tma_sf_desc(
            encode_tensormap=encode_tensormap,
            tensor=l1_weights_sf,
            shape_mn=case.config.intermediate_hidden * 2,
            shape_k=case.config.hidden,
            block_mn=launch.block_n,
            gran_k=gran_k,
            num_groups=num_experts_per_rank,
        ),
        "tensor_map_l1_output": _encode_tma_2d_desc(
            encode_tensormap=encode_tensormap,
            tensor=l2_acts,
            gmem_inner_dim=case.config.intermediate_hidden,
            gmem_outer_dim=workspace.num_max_pool_tokens,
            smem_inner_dim=launch.block_n // 2,
            smem_outer_dim=launch.store_block_m,
            gmem_outer_stride=int(l2_acts.stride(-2)),
            swizzle_mode=swizzle_acts_mode // 2,
            tensor_dtype="float8_e4m3fn",
        ),
        "tensor_map_l2_acts": _encode_tma_2d_desc(
            encode_tensormap=encode_tensormap,
            tensor=l2_acts,
            gmem_inner_dim=case.config.intermediate_hidden,
            gmem_outer_dim=workspace.num_max_pool_tokens,
            smem_inner_dim=launch.block_k,
            smem_outer_dim=launch.load_block_m,
            gmem_outer_stride=int(l2_acts.stride(-2)),
            swizzle_mode=swizzle_acts_mode,
            tensor_dtype="float8_e4m3fn",
        ),
        "tensor_map_l2_acts_sf": _encode_tma_sf_desc(
            encode_tensormap=encode_tensormap,
            tensor=case.symm_buffer.l2_acts_sf,
            shape_mn=workspace.num_padded_sf_pool_tokens,
            shape_k=case.config.intermediate_hidden,
            block_mn=sf_block_m,
            gran_k=gran_k,
            num_groups=1,
        ),
        "tensor_map_l2_weights": _encode_tma_2d_desc(
            encode_tensormap=encode_tensormap,
            tensor=l2_weights,
            gmem_inner_dim=case.config.intermediate_hidden,
            gmem_outer_dim=num_experts_per_rank * case.config.hidden,
            smem_inner_dim=launch.block_k,
            smem_outer_dim=launch.load_block_n,
            gmem_outer_stride=int(l2_weights.stride(-2)),
            swizzle_mode=swizzle_weights_mode,
            tensor_dtype="float4_e2m1fn",
        ),
        "tensor_map_l2_weights_sf": _encode_tma_sf_desc(
            encode_tensormap=encode_tensormap,
            tensor=l2_weights_sf,
            shape_mn=case.config.hidden,
            shape_k=case.config.intermediate_hidden,
            block_mn=launch.block_n,
            gran_k=gran_k,
            num_groups=num_experts_per_rank,
        ),
    }


def load_deep_gemm_mega() -> tuple[Any, str]:
    try:
        import deep_gemm as module
    except Exception as exc:
        raise SkipTest(
            f"DeepGEMM mega_moe runtime unavailable: {_DEEP_GEMM_MODULE_NAME}: {exc}"
        ) from exc
    if not hasattr(module, "fp8_fp4_mega_moe"):
        raise SkipTest("DeepGEMM mega_moe runtime unavailable: missing fp8_fp4_mega_moe")
    return module, "installed"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def _distributed_env(port: int):
    old_master_addr = os.environ.get("MASTER_ADDR")
    old_master_port = os.environ.get("MASTER_PORT")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    try:
        yield
    finally:
        if old_master_addr is None:
            os.environ.pop("MASTER_ADDR", None)
        else:
            os.environ["MASTER_ADDR"] = old_master_addr
        if old_master_port is None:
            os.environ.pop("MASTER_PORT", None)
        else:
            os.environ["MASTER_PORT"] = old_master_port


def _cast_grouped_weights_to_fp4(
    deep_gemm: Any, bf16_weights: torch.Tensor
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    num_groups, n, k = bf16_weights.shape
    weights = []
    scales = []
    for group_idx in range(num_groups):
        weight, scale = deep_gemm.utils.per_token_cast_to_fp4(
            bf16_weights[group_idx], use_ue8m0=True, gran_k=32
        )
        weights.append(weight)
        scales.append(scale)
    packed_weights = torch.stack(weights, dim=0).contiguous()
    raw_scales = torch.stack(scales, dim=0).contiguous()
    transformed_scales = deep_gemm.transform_sf_into_required_layout(
        raw_scales, n, k, (1, 32), num_groups
    )
    return (packed_weights, raw_scales), (packed_weights, transformed_scales)


def create_case(
    deep_gemm: Any, config: MegaMoeConfig, group: Any, rank_idx: int, num_ranks: int
) -> MegaMoeCase:
    torch.manual_seed(rank_idx)
    random.seed(rank_idx)

    symm_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group,
        config.num_experts,
        config.num_max_tokens_per_rank,
        config.num_topk,
        config.hidden,
        config.intermediate_hidden,
    )
    num_tokens = config.num_tokens
    num_experts_per_rank = config.num_experts // num_ranks

    x = torch.randn((num_tokens, config.hidden), dtype=torch.bfloat16, device="cuda")
    l1_weights = torch.randn(
        (num_experts_per_rank, config.intermediate_hidden * 2, config.hidden),
        dtype=torch.bfloat16,
        device="cuda",
    )
    l2_weights = torch.randn(
        (num_experts_per_rank, config.hidden, config.intermediate_hidden),
        dtype=torch.bfloat16,
        device="cuda",
    )
    scores = torch.randn((num_tokens, config.num_experts), dtype=torch.float32, device="cuda")
    topk_weights, topk_idx = torch.topk(scores, config.num_topk, dim=-1, largest=True, sorted=False)

    x_fp8 = deep_gemm.utils.per_token_cast_to_fp8(
        x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    raw_l1_weights, transformed_l1_input = _cast_grouped_weights_to_fp4(deep_gemm, l1_weights)
    raw_l2_weights, transformed_l2_input = _cast_grouped_weights_to_fp4(deep_gemm, l2_weights)
    transformed_l1_weights, transformed_l2_weights = deep_gemm.transform_weights_for_mega_moe(
        transformed_l1_input, transformed_l2_input
    )
    workspace_layout = get_deepgemm_workspace_layout(config)
    symm_buffer_layout = get_deepgemm_symm_buffer_layout(config)
    validate_runtime_symm_buffer_layout(
        symm_buffer=symm_buffer, layout=symm_buffer_layout, config=config
    )
    dispatch_reference = build_dispatch_reference(
        topk_idx=topk_idx, rank_idx=rank_idx, layout=workspace_layout
    )
    pull_reference = None
    scheduler_reference = None
    if num_ranks == 1:
        pull_reference = build_pull_reference(
            dispatch_reference=dispatch_reference, layout=workspace_layout
        )
        scheduler_reference = build_scheduler_reference(
            config=config,
            launch=get_deepgemm_launch_config(config),
            dispatch_reference=dispatch_reference,
            pull_reference=pull_reference,
        )

    return MegaMoeCase(
        config=config,
        rank_idx=rank_idx,
        num_ranks=num_ranks,
        group=group,
        deep_gemm=deep_gemm,
        symm_buffer=symm_buffer,
        x_fp8=x_fp8,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        raw_l1_weights=raw_l1_weights,
        raw_l2_weights=raw_l2_weights,
        transformed_l1_weights=transformed_l1_weights,
        transformed_l2_weights=transformed_l2_weights,
        workspace_layout=workspace_layout,
        symm_buffer_layout=symm_buffer_layout,
        dispatch_reference=dispatch_reference,
        pull_reference=pull_reference,
        scheduler_reference=scheduler_reference,
    )


def _copy_inputs_into_symm_buffer(case: MegaMoeCase) -> None:
    num_tokens = case.config.num_tokens
    case.symm_buffer.x[:num_tokens].copy_(case.x_fp8[0])
    case.symm_buffer.x_sf[:num_tokens].copy_(case.x_fp8[1])
    case.symm_buffer.topk_idx[:num_tokens].copy_(case.topk_idx)
    case.symm_buffer.topk_weights[:num_tokens].copy_(case.topk_weights)


def run_deepgemm_reference(case: MegaMoeCase) -> torch.Tensor:
    _copy_inputs_into_symm_buffer(case)
    y = torch.empty(
        (case.config.num_tokens, case.config.hidden), dtype=torch.bfloat16, device="cuda"
    )
    case.deep_gemm.fp8_fp4_mega_moe(
        y,
        case.transformed_l1_weights,
        case.transformed_l2_weights,
        case.symm_buffer,
        activation_clamp=case.config.activation_clamp,
        fast_math=bool(case.config.fast_math),
    )
    return y


def run_naive_reference(case: MegaMoeCase) -> torch.Tensor:
    from deep_gemm.utils.math import cast_back_from_fp4, ceil_to_ue8m0, unpack_ue8m0_from_int

    x_fp8, x_sf_packed = case.x_fp8
    x_sf = unpack_ue8m0_from_int(x_sf_packed)
    x = x_fp8.float() * x_sf.repeat_interleave(32, dim=1)
    l1_weights = []
    l2_weights = []
    for expert_idx in range(case.config.num_experts):
        l1_weights.append(
            cast_back_from_fp4(
                case.raw_l1_weights[0][expert_idx],
                case.raw_l1_weights[1][expert_idx],
                gran_k=32,
                use_packed_ue8m0=False,
            )
        )
        l2_weights.append(
            cast_back_from_fp4(
                case.raw_l2_weights[0][expert_idx],
                case.raw_l2_weights[1][expert_idx],
                gran_k=32,
                use_packed_ue8m0=False,
            )
        )
    l1_weights = torch.stack(l1_weights, dim=0).float()
    l2_weights = torch.stack(l2_weights, dim=0).float()
    y = torch.zeros(
        (case.config.num_tokens, case.config.hidden), dtype=torch.float32, device="cuda"
    )
    for token_idx in range(case.config.num_tokens):
        for topk_slot in range(case.config.num_topk):
            expert_idx = int(case.topk_idx[token_idx, topk_slot].item())
            if expert_idx < 0:
                continue
            topk_weight = case.topk_weights[token_idx, topk_slot]
            l1 = torch.matmul(l1_weights[expert_idx], x[token_idx].float())
            half_hidden = case.config.intermediate_hidden
            gate = torch.clamp(l1[:half_hidden], max=case.config.activation_clamp)
            up = torch.clamp(
                l1[half_hidden:],
                min=-case.config.activation_clamp,
                max=case.config.activation_clamp,
            )
            acts = torch.nn.functional.silu(gate) * up * topk_weight
            sf = ceil_to_ue8m0(acts.abs().view(-1, 32).amax(dim=1).clamp_min(1e-4) / 448.0)
            acts = (
                (acts.view(-1, 32) * (1.0 / sf.unsqueeze(1))).to(torch.float8_e4m3fn).float()
                * sf.unsqueeze(1)
            ).view(-1)
            y[token_idx] += torch.matmul(l2_weights[expert_idx], acts.float())
    return y.bfloat16()


def _max_abs_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    abs_diff = (lhs.float() - rhs.float()).abs()
    return 0.0 if abs_diff.numel() == 0 else float(abs_diff.max().item())


def _is_optional_math_reference_error(exc: RuntimeError) -> bool:
    message = str(exc)
    return (
        "CUBLAS_STATUS_ALLOC_FAILED" in message
        or "cublasCreate(handle)" in message
        or "CUDA out of memory" in message
        or "out of memory" in message.lower()
    )


def get_kernel(
    *,
    num_processes: int,
    num_max_tokens_per_rank: int,
    num_tokens: int,
    hidden: int,
    intermediate_hidden: int,
    num_experts: int,
    num_topk: int,
    activation_clamp: float = 10.0,
    fast_math: int = 1,
    emit_nvl_barrier_timeout_printf: bool = True,
):
    from tvm.script import tirx as T
    from tvm.backend.cuda.operator.tile_primitive.gemm_async.tcgen05 import sf_tmem_layout
    from tvm.backend.cuda.operator.tile_primitive.tma_utils import SwizzleMode, mma_shared_layout
    from tvm.tirx.layout import S, TCol, TileLayout, TLane

    runtime_config = MegaMoeConfig(
        num_processes=num_processes,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_tokens=num_tokens,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        num_experts=num_experts,
        num_topk=num_topk,
        activation_clamp=activation_clamp,
        fast_math=fast_math,
    )
    kernel_config = get_deepgemm_launch_config(runtime_config)
    workspace_layout = get_deepgemm_workspace_layout(runtime_config)
    symm_buffer_layout = get_deepgemm_symm_buffer_layout(runtime_config)
    num_experts_per_rank = num_experts // num_processes
    num_experts_per_lane = (num_experts_per_rank + 31) // 32
    num_ranks_per_lane = (num_processes + 31) // 32
    num_warps_per_warpgroup = 4
    num_l1_block_ns = (intermediate_hidden * 2) // kernel_config.block_n
    num_l2_block_ns = hidden // kernel_config.block_n
    num_l1_block_ks = hidden // kernel_config.block_k
    num_l2_block_ks = intermediate_hidden // kernel_config.block_k
    l1_out_block_n = kernel_config.block_n // 2
    sf_block_m = _align_up(kernel_config.block_m, 128)
    umma_m = 256
    umma_n = kernel_config.block_m
    umma_k = 32
    num_sfa_utccp_chunks = sf_block_m // 128
    num_sfb_utccp_chunks = kernel_config.block_n // 128
    num_epilogue_stages = 2
    num_tma_store_stages = 2
    num_dispatch_registers = 48
    num_non_epilogue_registers = 40
    num_epilogue_registers = 208
    num_accum_tmem_cols = kernel_config.block_m * num_epilogue_stages
    num_sfa_tmem_cols = sf_block_m // 32
    num_sfb_tmem_cols = kernel_config.block_n // 32
    num_tmem_cols = 32
    if num_accum_tmem_cols + num_sfa_tmem_cols + num_sfb_tmem_cols > 32:
        num_tmem_cols = 64
    if num_accum_tmem_cols + num_sfa_tmem_cols + num_sfb_tmem_cols > 64:
        num_tmem_cols = 128
    if num_accum_tmem_cols + num_sfa_tmem_cols + num_sfb_tmem_cols > 128:
        num_tmem_cols = 256
    if num_accum_tmem_cols + num_sfa_tmem_cols + num_sfb_tmem_cols > 256:
        num_tmem_cols = 512

    if not 32 <= num_tmem_cols <= 512:
        raise ValueError("Invalid tensor memory columns")

    tcgen05_cta_mask = (1 << 2) - 1

    def unpack_ue8m0_scale(packed_word, lane_idx):
        packed_u32 = T.cast(packed_word, "uint32")
        lane_u32 = T.cast(lane_idx, "uint32")
        exp_bits = T.bitwise_and(
            T.shift_right(packed_u32, T.bitwise_and(lane_u32, T.uint32(3)) * T.uint32(8)),
            T.uint32(0xFF),
        )
        return T.cuda.uint_as_float(T.shift_left(exp_bits, T.uint32(23)))

    def scale_pack_fp8x4_e4m3(v0, v1, v2, v3, s0, s1):
        sf_inv = T.cuda.make_float2(s0, s1)
        upper = T.cuda.fmul2_rn(T.cuda.make_float2(v0, v1), sf_inv)
        lower = T.cuda.fmul2_rn(T.cuda.make_float2(v2, v3), sf_inv)
        return T.cuda.fp8x4_e4m3_from_float4(
            T.cuda.float2_x(upper),
            T.cuda.float2_y(upper),
            T.cuda.float2_x(lower),
            T.cuda.float2_y(lower),
        )

    def red_or_rel_gpu_u64(address, value):
        return T.ptx.red_scalar(
            address, value, sem="release", scope="gpu", space="global", op="or", ptx_type="b64"
        )

    def red_add_rel_sys_s32(address, value):
        return T.ptx.red_scalar(
            address, value, sem="release", scope="sys", space="global", op="add", ptx_type="s32"
        )

    def load_acq_gpu_u64(address):
        return T.ptx.ld_acquire(address, "uint64", "u64", scope="gpu", space="global")

    def load_volatile_u64(address):
        return T.ptx.ld_volatile(address, "uint64", "u64", space="global")

    def load_acq_sys_s32(address):
        return T.ptx.ld_acquire(address, "int32", "s32", scope="sys", space="global")

    def atomic_add_rel_u32(address, value):
        return T.ptx.atom_scalar(
            address, value, sem="release", scope="gpu", space="global", op="add", ptx_type="u32"
        )

    def load_acq_u32(address):
        return T.ptx.ld_acquire(address, "uint32", "b32", scope="gpu", space="global")

    def grid_sync_done_u32(new_value, old_value):
        return T.cast(
            T.bitwise_and(T.bitwise_xor(new_value, old_value), T.uint32(0x80000000)) != T.uint32(0),
            "uint32",
        )

    def load_f32(address):
        return T.ptx.ld(address, "float32", "f32", space="global")

    def uint32_bits_to_float(bits):
        return T.cuda.uint_as_float(bits)

    def float_bits(x):
        return T.cuda.float_as_uint(x)

    def e4m3_scale_bits_from_scaled(scaled):
        bits = float_bits(scaled)
        exp_bits = T.shift_right(bits, T.uint32(23))
        man_bits = T.bitwise_and(bits, T.uint32((1 << 23) - 1))
        mantissa_carry = T.Select(man_bits != T.uint32(0), T.int32(1), T.int32(0))
        log2_ceil = T.cast(exp_bits, "int32") - T.int32(127) + mantissa_carry
        sf_bits = T.shift_left(T.cast(log2_ceil + T.int32(127), "uint32"), T.uint32(23))
        sf_inv_bits = T.shift_left(T.cast(T.int32(127) - log2_ceil, "uint32"), T.uint32(23))
        return sf_bits, sf_inv_bits

    def get_e4m3_sf_and_sf_inv(amax_x, amax_y):
        scaled = T.cuda.fmul2_rn(
            T.cuda.make_float2(amax_x, amax_y),
            T.cuda.make_float2(T.float32(1.0 / 448.0), T.float32(1.0 / 448.0)),
        )
        sf_bits_x, sf_inv_bits_x = e4m3_scale_bits_from_scaled(T.cuda.float2_x(scaled))
        sf_bits_y, sf_inv_bits_y = e4m3_scale_bits_from_scaled(T.cuda.float2_y(scaled))
        return (
            uint32_bits_to_float(sf_bits_x),
            uint32_bits_to_float(sf_bits_y),
            uint32_bits_to_float(sf_inv_bits_x),
            uint32_bits_to_float(sf_inv_bits_y),
        )

    kernel_activation_clamp = float(activation_clamp)
    kernel_fast_math = bool(fast_math)
    kernel_emit_nvl_barrier_timeout_printf = bool(emit_nvl_barrier_timeout_printf)
    use_activation_clamp = math.isfinite(kernel_activation_clamp)

    def warp_reduce_max_4(x):
        x = T.max(x, T.tvm_warp_shuffle_xor(0xFFFFFFFF, x, 4, 32, 32))
        x = T.max(x, T.tvm_warp_shuffle_xor(0xFFFFFFFF, x, 8, 32, 32))
        return T.max(x, T.tvm_warp_shuffle_xor(0xFFFFFFFF, x, 16, 32, 32))

    def get_swizzled_sf_row_idx(row_idx):
        row_idx_u32 = T.cast(row_idx, "uint32")
        return T.cast(
            T.bitwise_and(row_idx_u32, T.uint32(0xFFFFFF80))
            + T.shift_left(T.bitwise_and(row_idx_u32, T.uint32(31)), T.uint32(2))
            + T.shift_right(T.bitwise_and(row_idx_u32, T.uint32(127)), T.uint32(5)),
            "int32",
        )

    def transform_sf_token_idx(token_idx_in_expert):
        token_idx_u32 = T.cast(token_idx_in_expert, "uint32")
        idx = token_idx_u32 % T.uint32(kernel_config.block_m)
        return T.cast(
            token_idx_u32 // T.uint32(kernel_config.block_m) * T.uint32(sf_block_m)
            + T.bitwise_and(idx, T.uint32(0xFFFFFF80))
            + T.shift_left(T.bitwise_and(idx, T.uint32(31)), T.uint32(2))
            + T.bitwise_and(T.shift_right(idx, T.uint32(5)), T.uint32(3)),
            "int32",
        )

    def sync_unaligned(barrier_idx, num_threads):
        return T.ptx.bar.sync(barrier_idx, num_threads)

    def prefetch_tensormap(tensor_map):
        return T.ptx.prefetch_tensormap(T.address_of(tensor_map))

    def lds128(src_ptr, dst_ptr):
        return T.ptx.ld(src_ptr, "uint32", "u32", dst=dst_ptr, space="shared", vec="v4")

    def sts128(dst_ptr, r0, r1, r2, r3):
        return T.ptx.st(dst_ptr, r0, r1, r2, r3, space="shared", vec="v4", ptx_type="b32")

    def mbarrier_arrive_local(barrier_ptr):
        return T.ptx.mbarrier.arrive(barrier_ptr)

    def mbarrier_arrive_and_set_tx(barrier_ptr, num_bytes):
        return T.ptx.mbarrier.arrive.expect_tx(barrier_ptr, num_bytes)

    def mbarrier_wait_phase(barrier_ptr, phase):
        return T.ptx.mbarrier.try_wait(barrier_ptr, phase)

    def mbarrier_test_wait_phase(barrier_ptr, phase):
        return T.ptx.mbarrier_test_wait_parity(barrier_ptr, phase, space="shared::cta")

    def shared_addr_u32(ptr):
        return T.cuda.cvta_generic_to_shared(ptr)

    def replace_smem_desc_addr(desc, smem_ptr):
        start_addr = T.cast(
            T.bitwise_and(T.shift_right(shared_addr_u32(smem_ptr), T.uint32(4)), T.uint32(0x3FFF)),
            "uint64",
        )
        return T.bitwise_or(T.bitwise_and(desc, T.bitwise_not(T.uint64(0x3FFF))), start_addr)

    def tma_load_1d(dst_ptr, src_ptr, barrier_ptr, num_bytes):
        return T.ptx.cp_async_bulk_g2s_cluster(
            dst_ptr, src_ptr, num_bytes, barrier_ptr, cache_hint="evict_first"
        )

    def tma_store_1d(dst_ptr, src_ptr, num_bytes):
        return T.ptx.cp_async_bulk_s2g(dst_ptr, src_ptr, num_bytes, cache_hint="evict_normal")

    def tma_store_fence():
        return T.ptx.fence.proxy_async("shared::cta")

    def fence_barrier_init():
        return T.ptx.fence.mbarrier_init()

    def tma_store_arrive():
        return T.ptx.cp_async.bulk.commit_group()

    def tma_store_wait(num_prior_groups):
        if num_prior_groups == 0:
            return T.ptx.cp_async.bulk.wait_group(0, read=False)
        if num_prior_groups == 1:
            return T.ptx.cp_async.bulk.wait_group(1, read=False)
        raise ValueError("Unsupported TMA store wait distance")

    def tma_store_2d(src, tensormap, coord0, coord1):
        return T.ptx.cp_async.bulk.tensor.s2g(2, src, T.address_of(tensormap), "", coord0, coord1)

    def sm100_tma_2sm_load_2d_addr(dst, bar, tensormap_addr, coord0, coord1):
        bar_addr = T.cuda.sm100_tma_2sm_mbarrier_addr(bar)
        T.evaluate(
            T.ptx.cp_async.bulk.tensor.g2c_bar_addr(
                2, dst, bar_addr, tensormap_addr, 1, 2, "evict_normal", coord0, coord1
            )
        )

    def sm100_tma_2sm_load_2d(dst, bar, tensormap, coord0, coord1):
        sm100_tma_2sm_load_2d_addr(dst, bar, T.address_of(tensormap), coord0, coord1)

    @T.inline
    def sm100_tma_2sm_load_2d_select(
        dst, bar, tensor_map_l1, tensor_map_l2, block_phase_value, coord0, coord1
    ):
        sm100_tma_2sm_load_2d_addr(
            dst,
            bar,
            T.Select(
                block_phase_value == 1, T.address_of(tensor_map_l1), T.address_of(tensor_map_l2)
            ),
            coord0,
            coord1,
        )

    def stg128_symm(peer_base, byte_offset, r0, r1, r2, r3):
        return T.ptx.st(
            peer_ptr(peer_base, byte_offset),
            r0,
            r1,
            r2,
            r3,
            space="global",
            vec="v4",
            ptx_type="b32",
        )

    def ptr_to_u64(ptr):
        return T.reinterpret("uint64", ptr)

    def peer_ptr(peer_base, byte_offset):
        return T.reinterpret("handle", peer_base + byte_offset)

    def peer_store_u32(peer_base, byte_offset, value):
        return T.ptx.st(peer_ptr(peer_base, byte_offset), value, space="global", ptx_type="u32")

    def peer_store_u64(peer_base, byte_offset, value):
        return T.ptx.st(peer_ptr(peer_base, byte_offset), value, space="global", ptx_type="u64")

    def st_shared_u32(ptr, value):
        return T.ptx.st(ptr, value, space="shared", ptx_type="u32")

    def st_shared_bulk(ptr, num_bytes):
        return T.ptx.st_bulk(ptr, num_bytes, weak=True, space="shared::cta")

    def peer_atomic_add_u64(peer_base, byte_offset, value):
        return T.ptx.atom_scalar(
            peer_ptr(peer_base, byte_offset),
            value,
            scope="sys",
            space="global",
            op="add",
            ptx_type="u64",
        )

    def peer_red_add_rel_sys_s32(peer_base, byte_offset, value):
        return T.ptx.red_scalar(
            peer_ptr(peer_base, byte_offset),
            value,
            sem="release",
            scope="sys",
            space="global",
            op="add",
            ptx_type="s32",
        )

    def peer_load_u32(peer_base, byte_offset):
        return T.ptx.ld(peer_ptr(peer_base, byte_offset), "uint32", "u32", space="global")

    def peer_load_f32(peer_base, byte_offset):
        return T.ptx.ld(peer_ptr(peer_base, byte_offset), "float32", "f32", space="global")

    def tma_load_1d_symm(dst_ptr, peer_base, byte_offset, barrier_ptr, num_bytes):
        return T.ptx.cp_async_bulk_g2s_cluster(
            dst_ptr,
            peer_ptr(peer_base, byte_offset),
            num_bytes,
            barrier_ptr,
            cache_hint="evict_first",
        )

    def ballot_sync(mask, pred):
        return T.cuda.ballot_sync(mask, pred)

    def ffs_u32(value):
        return T.cuda.ffs_u32(value)

    def reduce_add_sync_u32(mask, value):
        return T.cuda.reduce_add_sync_u32(mask, value)

    def reduce_min_sync_u32(mask, value):
        return T.cuda.reduce_min_sync_u32(mask, value)

    def fns_b32(mask, base, offset):
        return T.ptx.fns_b32(mask, base, offset)

    def red_add_gpu_u32(address, value):
        return T.ptx.red_scalar(
            address, value, scope="gpu", space="global", op="add", ptx_type="u32"
        )

    def cuda_clock64():
        return T.cuda.clock64()

    def bf16x2_lo(packed):
        return T.cast(T.bitwise_and(packed, T.uint32(0xFFFF)), "uint16")

    def bf16x2_hi(packed):
        return T.cast(
            T.bitwise_and(T.shift_right(packed, T.uint32(16)), T.uint32(0xFFFF)), "uint16"
        )

    def stmatrix_fp8x4_trans(smem_ptr, local_ptr):
        return T.ptx.stmatrix(True, 1, ".b8", smem_ptr, local_ptr, shape="m16n8", space="shared")

    def cast_into_bf16_and_pack(v0, v1):
        return T.cuda.float22bfloat162_rn(v0, v1)

    def bf16_to_f32(x):
        return T.ptx.add_rn_f32_bf16(T.float32(0.0), x)

    def swiglu_sigmoid_gate(gate_value):
        denom = T.float32(1.0) + T.exp(-gate_value)
        if kernel_fast_math:
            return gate_value * T.ptx.rcp(denom)
        return gate_value / denom

    @T.inline
    def swiglu_pair_store(out, out_idx, gate0, gate1, up0, up1, weight0, weight1):
        bf16_gate = cast_into_bf16_and_pack(gate0, gate1)
        bf16_up = cast_into_bf16_and_pack(up0, up1)

        if use_activation_clamp:
            activation_clamp_value = T.float32(kernel_activation_clamp)
            clamp_pos = cast_into_bf16_and_pack(activation_clamp_value, activation_clamp_value)
            clamp_neg = cast_into_bf16_and_pack(-activation_clamp_value, -activation_clamp_value)
            bf16_gate = T.cuda.hmin2(bf16_gate, clamp_pos)
            bf16_up = T.cuda.hmax2(bf16_up, clamp_neg)
            bf16_up = T.cuda.hmin2(bf16_up, clamp_pos)

        gate = T.cuda.bfloat1622float2(bf16_gate)
        gate_x = T.cuda.float2_x(gate)
        gate_y = T.cuda.float2_y(gate)
        neg_gate_exp = T.cuda.make_float2(T.exp(-gate_x), T.exp(-gate_y))
        denom = T.cuda.fadd2_rn(T.cuda.make_float2(T.float32(1.0), T.float32(1.0)), neg_gate_exp)
        if kernel_fast_math:
            gate = T.cuda.fmul2_rn(
                gate,
                T.cuda.make_float2(
                    T.ptx.rcp(T.cuda.float2_x(denom)), T.ptx.rcp(T.cuda.float2_y(denom))
                ),
            )
        else:
            gate = T.cuda.make_float2(
                gate_x / T.cuda.float2_x(denom), gate_y / T.cuda.float2_y(denom)
            )

        up = T.cuda.bfloat1622float2(bf16_up)
        weights = T.cuda.make_float2(weight0, weight1)
        result = T.cuda.fmul2_rn(T.cuda.fmul2_rn(gate, up), weights)
        out[out_idx, 0] = T.cuda.float2_x(result)
        out[out_idx, 1] = T.cuda.float2_y(result)

    def make_runtime_instr_desc_with_sf_id(desc, sfa_id, sfb_id):
        runtime_desc = T.bitwise_and(desc, T.uint32(0x9FFFFFCF))
        runtime_desc = T.bitwise_or(
            runtime_desc, T.shift_left(T.cast(sfa_id, "uint32"), T.uint32(29))
        )
        runtime_desc = T.bitwise_or(
            runtime_desc, T.shift_left(T.cast(sfb_id, "uint32"), T.uint32(4))
        )
        return runtime_desc

    def advance_umma_desc_lo(desc, base_lo, mn_offset, k_offset):
        return T.bitwise_or(
            T.bitwise_and(desc, T.shift_left(T.uint64(0xFFFFFFFF), T.uint64(32))),
            T.cast(base_lo + T.cast((mn_offset + k_offset) // f128_bytes, "uint32"), "uint64"),
        )

    def scheduler_get_num_tokens_expr(expert_idx, lane_idx, stored_num_tokens_per_expert):
        scheduler_num_tokens_expr = T.int32(0)
        for expert_lane_idx in range(num_experts_per_lane):
            scheduler_num_tokens_expr = T.Select(
                expert_idx == expert_lane_idx * 32 + lane_idx,
                T.cast(stored_num_tokens_per_expert[expert_lane_idx], "int32"),
                scheduler_num_tokens_expr,
            )
        expert_lane_idx_u32 = T.cast(expert_idx, "uint32") % T.uint32(32)
        return T.tvm_warp_shuffle(
            T.uint32(0xFFFFFFFF),
            scheduler_num_tokens_expr,
            T.cast(expert_lane_idx_u32, "int32"),
            32,
            32,
        )

    def scheduler_get_pool_block_offset_expr(expert_idx, lane_idx, stored_num_tokens_per_expert):
        scheduler_pool_block_offset_expr = T.int32(0)
        for expert_lane_idx in range(num_experts_per_lane):
            expert_num_blocks_u32 = (
                stored_num_tokens_per_expert[expert_lane_idx] + T.uint32(kernel_config.block_m - 1)
            ) // T.uint32(kernel_config.block_m)
            scheduler_pool_block_offset_expr = scheduler_pool_block_offset_expr + T.Select(
                expert_lane_idx * 32 + lane_idx < expert_idx,
                T.cast(expert_num_blocks_u32, "int32"),
                T.int32(0),
            )
        return T.cast(
            reduce_add_sync_u32(
                T.uint32(0xFFFFFFFF), T.cast(scheduler_pool_block_offset_expr, "uint32")
            ),
            "int32",
        )

    def scheduler_get_wave_expert_end_idx_expr(current_local_expert_idx):
        current_local_expert_idx_u32 = T.cast(current_local_expert_idx + T.int32(1), "uint32")
        wave_expert_end_idx_u32 = (
            (current_local_expert_idx_u32 + T.uint32(kernel_config.num_experts_per_wave - 1))
            // T.uint32(kernel_config.num_experts_per_wave)
            * T.uint32(kernel_config.num_experts_per_wave)
        )
        return T.cast(wave_expert_end_idx_u32, "int32")

    def scheduler_get_current_num_m_blocks_expr(current_num_tokens):
        current_num_tokens_u32 = T.cast(current_num_tokens, "uint32")
        current_num_m_blocks_u32 = (
            current_num_tokens_u32 + T.uint32(kernel_config.block_m - 1)
        ) // T.uint32(kernel_config.block_m)
        return T.cast(current_num_m_blocks_u32, "int32")

    def symm_rank_offset_expr(symm_rank_offsets, mapped_rank_idx):
        if num_processes == 1:
            return symm_rank_offsets[0]
        mapped_rank_idx_u32 = T.cast(mapped_rank_idx, "uint32")
        rank_offset = symm_rank_offsets[0]
        for rank in range(1, num_processes):
            rank_offset = T.Select(
                mapped_rank_idx_u32 == T.uint32(rank), symm_rank_offsets[rank], rank_offset
            )
        return rank_offset

    def symm_rank_base_expr(sym_buffer_base, symm_rank_offsets, mapped_rank_idx):
        return sym_buffer_base + T.cast(
            symm_rank_offset_expr(symm_rank_offsets, mapped_rank_idx), "uint64"
        )

    sm100_smem_capacity = 232448
    shared_alignment = 1024
    f32_bytes = 4
    f128_bytes = 16
    num_epilogue_wgs = kernel_config.num_epilogue_warps // 4
    wg_block_m = kernel_config.block_m // num_epilogue_wgs
    atom_m = 8
    num_atoms_per_store = kernel_config.store_block_m // atom_m
    num_rows_per_warp = kernel_config.store_block_m // 8
    num_bank_group_bytes = 16
    num_hidden_bytes = hidden * 2
    num_elems_per_uint4 = 4
    num_chunk_slots = 3
    num_max_registers_for_buffer = 128
    swizzle_cd_mode = 128
    a_desc_sdo = 8 * kernel_config.block_k // f128_bytes
    b_desc_sdo = 8 * kernel_config.block_k // f128_bytes
    sf_desc_sdo = 8 * 4 * f32_bytes // f128_bytes
    smem_expert_count_size = _align_up(num_experts * 4, shared_alignment)
    smem_send_buffer_size = _align_up(hidden * kernel_config.num_dispatch_warps, shared_alignment)
    smem_dispatch_size = smem_expert_count_size + smem_send_buffer_size
    smem_cd_l1_size = (
        (kernel_config.num_epilogue_warps // 4)
        * kernel_config.store_block_m
        * l1_out_block_n
        * num_tma_store_stages
    )
    smem_cd_l2_size = (
        (kernel_config.num_epilogue_warps // 4)
        * kernel_config.store_block_m
        * kernel_config.block_n
        * 2
    )
    smem_cd_size = max(smem_cd_l1_size, smem_cd_l2_size)
    smem_a_size_per_stage = kernel_config.load_block_m * kernel_config.block_k
    smem_b_size_per_stage = kernel_config.load_block_n * kernel_config.block_k
    smem_sfa_size_per_stage = sf_block_m * 4
    smem_sfb_size_per_stage = kernel_config.block_n * 4
    full_a_expect_tx_leader_bytes: int = smem_a_size_per_stage * 2 + (smem_sfa_size_per_stage * 2)
    full_b_expect_tx_leader_bytes: int = smem_b_size_per_stage + (smem_sfb_size_per_stage * 2)
    smem_amax_reduction_size = kernel_config.store_block_m * kernel_config.num_epilogue_warps * 4
    smem_tmem_ptr_size = 4
    smem_per_stage = (
        smem_a_size_per_stage
        + smem_b_size_per_stage
        + smem_sfa_size_per_stage
        + smem_sfb_size_per_stage
        + 16
    )
    smem_fixed = (
        smem_dispatch_size
        + smem_cd_size
        + smem_amax_reduction_size
        + (
            kernel_config.num_dispatch_warps
            + num_epilogue_stages * 2
            + kernel_config.num_epilogue_warps * 2
        )
        * 8
        + smem_tmem_ptr_size
    )
    num_stages = max(2, (sm100_smem_capacity - smem_fixed) // smem_per_stage)
    if (
        smem_cd_size % shared_alignment != 0
        or smem_a_size_per_stage % shared_alignment != 0
        or smem_b_size_per_stage % shared_alignment != 0
    ):
        raise ValueError("Shared memory of CD/A/B must be aligned to 1024 bytes")
    if num_stages > 32:
        raise ValueError("Too many stages")
    if (
        num_dispatch_registers * kernel_config.num_dispatch_threads
        + num_non_epilogue_registers * kernel_config.num_non_epilogue_threads
        + num_epilogue_registers * kernel_config.num_epilogue_threads
        > 64512
    ):
        raise ValueError("Too many registers")
    smem_a_layout = mma_shared_layout(
        "int8",
        SwizzleMode.SWIZZLE_128B_ATOM,
        (num_stages, kernel_config.load_block_m, kernel_config.block_k),
    )
    smem_b_layout = mma_shared_layout(
        "uint8",
        SwizzleMode.SWIZZLE_128B_ATOM,
        (num_stages, kernel_config.load_block_n, kernel_config.block_k),
    )
    num_total_barriers = (
        kernel_config.num_dispatch_warps
        + num_stages * 2
        + num_epilogue_stages * 2
        + kernel_config.num_epilogue_warps * 2
    )
    dispatch_barrier_base = 0
    full_barrier_base = dispatch_barrier_base + kernel_config.num_dispatch_warps
    empty_barrier_base = full_barrier_base + num_stages
    tmem_full_barrier_base = empty_barrier_base + num_stages
    tmem_empty_barrier_base = tmem_full_barrier_base + num_epilogue_stages
    combine_barrier_base = tmem_empty_barrier_base + num_epilogue_stages
    dispatch_sync_barrier_idx = 0
    dispatch_with_epilogue_sync_barrier_idx = 1
    epilogue_full_sync_barrier_idx = 2
    epilogue_wg_sync_barrier_start_idx = 3
    before_dispatch_pull_barrier_tag = 1
    before_combine_reduce_barrier_tag = 2
    after_workspace_clean_barrier_tag = 3
    num_nvlink_barrier_timeout_cycles = 30 * 2000000000
    dispatch_grid_sync_index = 0
    epilogue_grid_sync_index = 1
    smem_expert_count_offset = 0
    smem_send_buffer_offset = smem_expert_count_offset + smem_expert_count_size
    smem_gemm_base_offset = smem_send_buffer_offset + smem_send_buffer_size
    smem_cd_offset = smem_gemm_base_offset
    smem_a_offset = smem_cd_offset + smem_cd_size
    smem_b_offset = smem_a_offset + num_stages * smem_a_size_per_stage
    smem_sfa_offset = smem_b_offset + num_stages * smem_b_size_per_stage
    smem_sfb_offset = smem_sfa_offset + num_stages * smem_sfa_size_per_stage
    smem_amax_reduction_offset = smem_sfb_offset + num_stages * smem_sfb_size_per_stage
    smem_barrier_offset = _align_up(smem_amax_reduction_offset + smem_amax_reduction_size, 8)
    smem_tmem_ptr_offset = smem_barrier_offset + num_total_barriers * 8
    smem_total_bytes = smem_tmem_ptr_offset + smem_tmem_ptr_size
    num_chunks = (
        1
        if num_chunk_slots * kernel_config.num_epilogue_warps * num_hidden_bytes
        <= smem_barrier_offset
        and hidden <= 32 * num_max_registers_for_buffer
        else 2
    )
    num_chunk_bytes = num_hidden_bytes // num_chunks
    num_chunk_uint4 = num_chunk_bytes // 16
    num_uint4_per_lane = num_chunk_uint4 // 32
    if hidden % num_chunks != 0:
        raise ValueError("Hidden must be divisible by number of chunks")
    if num_chunk_slots * kernel_config.num_epilogue_warps * num_chunk_bytes > smem_barrier_offset:
        raise ValueError("Hidden is too large")
    if num_chunk_bytes % 16 != 0:
        raise ValueError("Combine chunk must be TMA-aligned (16 bytes)")
    if num_chunk_bytes % 16 != 0:
        raise ValueError("Combine chunk must be divisible by 16 bytes")
    if num_chunk_uint4 % 32 != 0:
        raise ValueError("Combine chunk must be a multiple of 32 16-byte elements (one per lane)")
    if num_topk > 32:
        raise ValueError("Top-k must fit in a single warp")

    @T.prim_func
    def mega_moe(
        y_ptr: T.handle,
        symm_buffer_ptr: T.handle,
        symm_rank_offset_0: T.int64,
        symm_rank_offset_1: T.int64,
        symm_rank_offset_2: T.int64,
        symm_rank_offset_3: T.int64,
        symm_rank_offset_4: T.int64,
        symm_rank_offset_5: T.int64,
        symm_rank_offset_6: T.int64,
        symm_rank_offset_7: T.int64,
        symm_rank_offset_8: T.int64,
        symm_rank_offset_9: T.int64,
        symm_rank_offset_10: T.int64,
        symm_rank_offset_11: T.int64,
        symm_rank_offset_12: T.int64,
        symm_rank_offset_13: T.int64,
        symm_rank_offset_14: T.int64,
        symm_rank_offset_15: T.int64,
        symm_rank_offset_16: T.int64,
        symm_rank_offset_17: T.int64,
        symm_rank_offset_18: T.int64,
        symm_rank_offset_19: T.int64,
        symm_rank_offset_20: T.int64,
        symm_rank_offset_21: T.int64,
        symm_rank_offset_22: T.int64,
        symm_rank_offset_23: T.int64,
        symm_rank_offset_24: T.int64,
        symm_rank_offset_25: T.int64,
        symm_rank_offset_26: T.int64,
        symm_rank_offset_27: T.int64,
        symm_rank_offset_28: T.int64,
        symm_rank_offset_29: T.int64,
        symm_rank_offset_30: T.int64,
        symm_rank_offset_31: T.int64,
        symm_rank_offset_32: T.int64,
        symm_rank_offset_33: T.int64,
        symm_rank_offset_34: T.int64,
        symm_rank_offset_35: T.int64,
        symm_rank_offset_36: T.int64,
        symm_rank_offset_37: T.int64,
        symm_rank_offset_38: T.int64,
        symm_rank_offset_39: T.int64,
        symm_rank_offset_40: T.int64,
        symm_rank_offset_41: T.int64,
        symm_rank_offset_42: T.int64,
        symm_rank_offset_43: T.int64,
        symm_rank_offset_44: T.int64,
        symm_rank_offset_45: T.int64,
        symm_rank_offset_46: T.int64,
        symm_rank_offset_47: T.int64,
        symm_rank_offset_48: T.int64,
        symm_rank_offset_49: T.int64,
        symm_rank_offset_50: T.int64,
        symm_rank_offset_51: T.int64,
        symm_rank_offset_52: T.int64,
        symm_rank_offset_53: T.int64,
        symm_rank_offset_54: T.int64,
        symm_rank_offset_55: T.int64,
        symm_rank_offset_56: T.int64,
        symm_rank_offset_57: T.int64,
        symm_rank_offset_58: T.int64,
        symm_rank_offset_59: T.int64,
        symm_rank_offset_60: T.int64,
        symm_rank_offset_61: T.int64,
        symm_rank_offset_62: T.int64,
        symm_rank_offset_63: T.int64,
        symm_rank_offset_64: T.int64,
        symm_rank_offset_65: T.int64,
        symm_rank_offset_66: T.int64,
        symm_rank_offset_67: T.int64,
        symm_rank_offset_68: T.int64,
        symm_rank_offset_69: T.int64,
        symm_rank_offset_70: T.int64,
        symm_rank_offset_71: T.int64,
        tensor_map_l1_acts: T.TensorMap(),
        tensor_map_l1_acts_sf: T.TensorMap(),
        tensor_map_l1_weights: T.TensorMap(),
        tensor_map_l1_weights_sf: T.TensorMap(),
        tensor_map_l1_output: T.TensorMap(),
        tensor_map_l2_acts: T.TensorMap(),
        tensor_map_l2_acts_sf: T.TensorMap(),
        tensor_map_l2_weights: T.TensorMap(),
        tensor_map_l2_weights_sf: T.TensorMap(),
        num_tokens: T.int32,
        rank_idx: T.int32,
    ):
        y = T.match_buffer(y_ptr, (num_tokens, hidden), "bfloat16")
        symm_buffer = T.match_buffer(symm_buffer_ptr, (symm_buffer_layout.total_bytes,), "int8")
        T.device_entry()
        T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})
        symm_rank_offsets = T.meta_var(
            (
                symm_rank_offset_0,
                symm_rank_offset_1,
                symm_rank_offset_2,
                symm_rank_offset_3,
                symm_rank_offset_4,
                symm_rank_offset_5,
                symm_rank_offset_6,
                symm_rank_offset_7,
                symm_rank_offset_8,
                symm_rank_offset_9,
                symm_rank_offset_10,
                symm_rank_offset_11,
                symm_rank_offset_12,
                symm_rank_offset_13,
                symm_rank_offset_14,
                symm_rank_offset_15,
                symm_rank_offset_16,
                symm_rank_offset_17,
                symm_rank_offset_18,
                symm_rank_offset_19,
                symm_rank_offset_20,
                symm_rank_offset_21,
                symm_rank_offset_22,
                symm_rank_offset_23,
                symm_rank_offset_24,
                symm_rank_offset_25,
                symm_rank_offset_26,
                symm_rank_offset_27,
                symm_rank_offset_28,
                symm_rank_offset_29,
                symm_rank_offset_30,
                symm_rank_offset_31,
                symm_rank_offset_32,
                symm_rank_offset_33,
                symm_rank_offset_34,
                symm_rank_offset_35,
                symm_rank_offset_36,
                symm_rank_offset_37,
                symm_rank_offset_38,
                symm_rank_offset_39,
                symm_rank_offset_40,
                symm_rank_offset_41,
                symm_rank_offset_42,
                symm_rank_offset_43,
                symm_rank_offset_44,
                symm_rank_offset_45,
                symm_rank_offset_46,
                symm_rank_offset_47,
                symm_rank_offset_48,
                symm_rank_offset_49,
                symm_rank_offset_50,
                symm_rank_offset_51,
                symm_rank_offset_52,
                symm_rank_offset_53,
                symm_rank_offset_54,
                symm_rank_offset_55,
                symm_rank_offset_56,
                symm_rank_offset_57,
                symm_rank_offset_58,
                symm_rank_offset_59,
                symm_rank_offset_60,
                symm_rank_offset_61,
                symm_rank_offset_62,
                symm_rank_offset_63,
                symm_rank_offset_64,
                symm_rank_offset_65,
                symm_rank_offset_66,
                symm_rank_offset_67,
                symm_rank_offset_68,
                symm_rank_offset_69,
                symm_rank_offset_70,
                symm_rank_offset_71,
            )
        )
        sym_buffer_base = ptr_to_u64(symm_buffer.ptr_to([0]))
        thread_idx = T.thread_id([kernel_config.num_threads_per_cta])
        cta_idx_in_cluster = T.cta_id_in_cluster([kernel_config.num_ctas_per_cluster])
        sm_idx = T.cta_id([kernel_config.num_sms])
        wg_id = T.warpgroup_id([kernel_config.num_warpgroups_per_cta])
        warp_id = T.warp_id_in_wg([num_warps_per_warpgroup])
        lane_idx = T.lane_id([32])
        flat_warp_idx = wg_id * num_warps_per_warpgroup + warp_id
        if flat_warp_idx == 0:
            T.evaluate(prefetch_tensormap(tensor_map_l1_acts))
            T.evaluate(prefetch_tensormap(tensor_map_l1_acts_sf))
            T.evaluate(prefetch_tensormap(tensor_map_l1_weights))
            T.evaluate(prefetch_tensormap(tensor_map_l1_weights_sf))
            T.evaluate(prefetch_tensormap(tensor_map_l1_output))
            T.evaluate(prefetch_tensormap(tensor_map_l2_acts))
            T.evaluate(prefetch_tensormap(tensor_map_l2_acts_sf))
            T.evaluate(prefetch_tensormap(tensor_map_l2_weights))
            T.evaluate(prefetch_tensormap(tensor_map_l2_weights_sf))
        input_topk_idx_data: T.let[
            T.Var(name="input_topk_idx_data", dtype=PointerType(PrimType("int64")))
        ] = T.reinterpret("handle", symm_buffer.ptr_to([symm_buffer_layout.input_topk_idx_offset]))
        l1_acts_sf_data: T.let[
            T.Var(name="l1_acts_sf_data", dtype=PointerType(PrimType("int32")))
        ] = T.reinterpret("handle", symm_buffer.ptr_to([symm_buffer_layout.l1_sf_offset]))
        workspace_expert_send_count_data: T.let[
            T.Var(name="workspace_expert_send_count_data", dtype=PointerType(PrimType("uint64")))
        ] = T.reinterpret("handle", symm_buffer.ptr_to([workspace_layout.expert_send_count_offset]))
        workspace_grid_sync_count_data: T.let[
            T.Var(name="workspace_grid_sync_count_data", dtype=PointerType(PrimType("uint32")))
        ] = T.reinterpret("handle", symm_buffer.ptr_to([workspace_layout.barrier_offset]))
        workspace_nvl_barrier_counter_data: T.let[
            T.Var(name="workspace_nvl_barrier_counter_data", dtype=PointerType(PrimType("uint32")))
        ] = T.reinterpret("handle", symm_buffer.ptr_to([workspace_layout.barrier_offset + 16]))
        workspace_nvl_barrier_signal_data: T.let[
            T.Var(name="workspace_nvl_barrier_signal_data", dtype=PointerType(PrimType("int32")))
        ] = T.reinterpret("handle", symm_buffer.ptr_to([workspace_layout.barrier_offset + 20]))
        workspace_expert_recv_count_data: T.let[
            T.Var(name="workspace_expert_recv_count_data", dtype=PointerType(PrimType("uint64")))
        ] = T.reinterpret("handle", symm_buffer.ptr_to([workspace_layout.expert_recv_count_offset]))
        workspace_expert_recv_count_sum_data: T.let[
            T.Var(
                name="workspace_expert_recv_count_sum_data", dtype=PointerType(PrimType("uint64"))
            )
        ] = T.reinterpret(
            "handle", symm_buffer.ptr_to([workspace_layout.expert_recv_count_sum_offset])
        )
        workspace_src_token_topk_idx_data: T.let[
            T.Var(name="workspace_src_token_topk_idx_data", dtype=PointerType(PrimType("uint32")))
        ] = T.reinterpret(
            "handle", symm_buffer.ptr_to([workspace_layout.src_token_topk_idx_offset])
        )
        workspace_token_src_metadata_data: T.let[
            T.Var(name="workspace_token_src_metadata_data", dtype=PointerType(PrimType("uint32")))
        ] = T.reinterpret(
            "handle", symm_buffer.ptr_to([workspace_layout.token_src_metadata_offset])
        )
        workspace_l1_arrival_count_data: T.let[
            T.Var(name="workspace_l1_arrival_count_data", dtype=PointerType(PrimType("uint32")))
        ] = T.reinterpret("handle", symm_buffer.ptr_to([workspace_layout.l1_arrival_count_offset]))
        workspace_l2_arrival_mask_data: T.let[
            T.Var(name="workspace_l2_arrival_mask_data", dtype=PointerType(PrimType("uint64")))
        ] = T.reinterpret("handle", symm_buffer.ptr_to([workspace_layout.l2_arrival_mask_offset]))
        l1_topk_weights_data: T.let[
            T.Var(name="l1_topk_weights_data", dtype=PointerType(PrimType("float32")))
        ] = T.reinterpret("handle", symm_buffer.ptr_to([symm_buffer_layout.l1_topk_weights_offset]))
        l2_acts_sf_data: T.let[
            T.Var(name="l2_acts_sf_data", dtype=PointerType(PrimType("int32")))
        ] = T.reinterpret("handle", symm_buffer.ptr_to([symm_buffer_layout.l2_sf_offset]))
        combine_tokens_data: T.let[
            T.Var(name="combine_tokens_data", dtype=PointerType(PrimType("uint16")))
        ] = T.reinterpret("handle", symm_buffer.ptr_to([symm_buffer_layout.combine_token_offset]))
        input_topk_idx = T.decl_buffer(
            (workspace_layout.num_max_tokens_per_rank, num_topk),
            "int64",
            data=input_topk_idx_data,
            scope="global",
            elem_offset=0,
        )
        l1_acts = T.decl_buffer(
            (workspace_layout.num_max_pool_tokens, hidden),
            "int8",
            data=symm_buffer.data,
            scope="global",
            elem_offset=symm_buffer_layout.l1_token_offset,
        )
        l1_acts_sf = T.decl_buffer(
            (hidden // 128, workspace_layout.num_padded_sf_pool_tokens),
            "int32",
            data=symm_buffer.data,
            scope="global",
            elem_offset=symm_buffer_layout.l1_sf_offset // 4,
        )
        workspace_expert_send_count = T.decl_buffer(
            (num_experts,),
            "uint64",
            data=workspace_expert_send_count_data,
            scope="global",
            elem_offset=0,
        )
        workspace_grid_sync_count = T.decl_buffer(
            (4,), "uint32", data=workspace_grid_sync_count_data, scope="global", elem_offset=0
        )
        workspace_nvl_barrier_counter = T.decl_buffer(
            (1,), "uint32", data=workspace_nvl_barrier_counter_data, scope="global", elem_offset=0
        )
        workspace_nvl_barrier_signal = T.decl_buffer(
            (2,), "int32", data=workspace_nvl_barrier_signal_data, scope="global", elem_offset=0
        )
        workspace_expert_recv_count = T.decl_buffer(
            (num_processes, num_experts_per_rank),
            "uint64",
            data=workspace_expert_recv_count_data,
            scope="global",
            elem_offset=0,
        )
        workspace_expert_recv_count_sum = T.decl_buffer(
            (num_experts_per_rank,),
            "uint64",
            data=workspace_expert_recv_count_sum_data,
            scope="global",
            elem_offset=0,
        )
        workspace_src_token_topk_idx = T.decl_buffer(
            (num_experts_per_rank, num_processes, workspace_layout.num_max_recv_tokens_per_expert),
            "uint32",
            data=workspace_src_token_topk_idx_data,
            scope="global",
            elem_offset=0,
        )
        workspace_token_src_metadata = T.decl_buffer(
            (workspace_layout.num_max_pool_tokens, 3),
            "uint32",
            data=workspace_token_src_metadata_data,
            scope="global",
            elem_offset=0,
        )
        workspace_l1_arrival_count = T.decl_buffer(
            (workspace_layout.num_max_pool_blocks,),
            "uint32",
            data=workspace_l1_arrival_count_data,
            scope="global",
            elem_offset=0,
        )
        workspace_l2_arrival_mask = T.decl_buffer(
            (workspace_layout.num_max_pool_blocks,),
            "uint64",
            data=workspace_l2_arrival_mask_data,
            scope="global",
            elem_offset=0,
        )
        l1_topk_weights = T.decl_buffer(
            (workspace_layout.num_max_pool_tokens,),
            "float32",
            data=l1_topk_weights_data,
            scope="global",
            elem_offset=0,
        )
        l2_acts = T.decl_buffer(
            (workspace_layout.num_max_pool_tokens, intermediate_hidden),
            "int8",
            data=symm_buffer.data,
            scope="global",
            elem_offset=symm_buffer_layout.l2_token_offset,
        )
        l2_sf_buffer = T.decl_buffer(
            (intermediate_hidden // 128 * workspace_layout.num_padded_sf_pool_tokens * 4,),
            "int8",
            data=symm_buffer.data,
            scope="global",
            elem_offset=symm_buffer_layout.l2_sf_offset,
        )
        combine_tokens = T.decl_buffer(
            (num_topk, workspace_layout.num_max_tokens_per_rank, hidden),
            "uint16",
            data=combine_tokens_data,
            scope="global",
            elem_offset=0,
        )

        smem = T.alloc_buffer([smem_total_bytes], "uint8", scope="shared.dyn")
        smem_expert_count_data: T.let[
            T.Var(name="smem_expert_count_data", dtype=PointerType(PrimType("int32")))
        ] = T.reinterpret("handle", smem.ptr_to([smem_expert_count_offset]))
        smem_send_buffer_data: T.let[
            T.Var(name="smem_send_buffer_data", dtype=PointerType(PrimType("int8")))
        ] = T.reinterpret("handle", smem.ptr_to([smem_send_buffer_offset]))
        smem_a_data: T.let[T.Var(name="smem_a_data", dtype=PointerType(PrimType("int8")))] = (
            T.reinterpret("handle", smem.ptr_to([smem_a_offset]))
        )
        smem_b_data: T.let[T.Var(name="smem_b_data", dtype=PointerType(PrimType("uint8")))] = (
            T.reinterpret("handle", smem.ptr_to([smem_b_offset]))
        )
        smem_sfa_data: T.let[T.Var(name="smem_sfa_data", dtype=PointerType(PrimType("int32")))] = (
            T.reinterpret("handle", smem.ptr_to([smem_sfa_offset]))
        )
        smem_sfb_data: T.let[T.Var(name="smem_sfb_data", dtype=PointerType(PrimType("int32")))] = (
            T.reinterpret("handle", smem.ptr_to([smem_sfb_offset]))
        )
        smem_amax_reduction_data: T.let[
            T.Var(name="smem_amax_reduction_data", dtype=PointerType(PrimType("float32")))
        ] = T.reinterpret("handle", smem.ptr_to([smem_amax_reduction_offset]))
        smem_cd_data: T.let[T.Var(name="smem_cd_data", dtype=PointerType(PrimType("uint8")))] = (
            T.reinterpret("handle", smem.ptr_to([smem_cd_offset]))
        )
        smem_barrier_data: T.let[
            T.Var(name="smem_barrier_data", dtype=PointerType(PrimType("uint64")))
        ] = T.reinterpret("handle", smem.ptr_to([smem_barrier_offset]))
        smem_tmem_ptr_data: T.let[
            T.Var(name="smem_tmem_ptr_data", dtype=PointerType(PrimType("uint32")))
        ] = T.reinterpret("handle", smem.ptr_to([smem_tmem_ptr_offset]))
        smem_expert_count = T.decl_buffer(
            (num_experts,),
            "int32",
            data=smem_expert_count_data,
            scope="shared.dyn",
            elem_offset=0,
            align=shared_alignment,
        )
        smem_send_buffers = T.decl_buffer(
            (kernel_config.num_dispatch_warps, hidden),
            "int8",
            data=smem_send_buffer_data,
            scope="shared.dyn",
            elem_offset=0,
            align=shared_alignment,
        )
        smem_a = T.decl_buffer(
            (num_stages, kernel_config.load_block_m, kernel_config.block_k),
            "int8",
            data=smem_a_data,
            scope="shared.dyn",
            elem_offset=0,
            align=shared_alignment,
            layout=smem_a_layout,
        )
        smem_a_fp8 = T.decl_buffer(
            (num_stages, kernel_config.load_block_m, kernel_config.block_k),
            "float8_e4m3fn",
            data=smem_a_data,
            scope="shared.dyn",
            elem_offset=0,
            align=shared_alignment,
            layout=smem_a_layout,
        )
        smem_b = T.decl_buffer(
            (num_stages, kernel_config.load_block_n, kernel_config.block_k),
            "uint8",
            data=smem_b_data,
            scope="shared.dyn",
            elem_offset=0,
            align=shared_alignment,
            layout=smem_b_layout,
        )
        smem_sfa_i32 = T.decl_buffer(
            (num_stages, sf_block_m, 1),
            "int32",
            data=smem_sfa_data,
            scope="shared.dyn",
            elem_offset=0,
            align=16,
        )
        smem_sfa = T.decl_buffer(
            (num_stages, sf_block_m),
            "uint32",
            data=smem_sfa_data,
            scope="shared.dyn",
            elem_offset=0,
            align=16,
        )
        smem_sfb_i32 = T.decl_buffer(
            (num_stages, kernel_config.block_n, 1),
            "int32",
            data=smem_sfb_data,
            scope="shared.dyn",
            elem_offset=0,
            align=16,
        )
        smem_sfb = T.decl_buffer(
            (num_stages, kernel_config.block_n),
            "uint32",
            data=smem_sfb_data,
            scope="shared.dyn",
            elem_offset=0,
            align=16,
        )
        smem_amax_reduction = T.decl_buffer(
            (kernel_config.num_epilogue_warps * kernel_config.store_block_m,),
            "float32",
            data=smem_amax_reduction_data,
            scope="shared.dyn",
            elem_offset=0,
            align=16,
        )
        smem_cd_l1 = T.decl_buffer(
            (num_tma_store_stages, num_epilogue_wgs, kernel_config.store_block_m, l1_out_block_n),
            "int8",
            data=smem_cd_data,
            scope="shared.dyn",
            elem_offset=0,
            align=16,
        )
        smem_cd_l2 = T.decl_buffer(
            (num_epilogue_wgs, kernel_config.store_block_m, kernel_config.block_n),
            "uint16",
            data=smem_cd_data,
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
        combine_chunks = T.decl_buffer(
            (num_chunk_slots, kernel_config.num_epilogue_warps, num_chunk_uint4, 4),
            "uint32",
            data=smem.data,
            scope="shared.dyn",
            elem_offset=0,
            align=16,
        )
        tmem = T.decl_buffer(
            (128, num_tmem_cols),
            "float32",
            scope="tmem",
            allocated_addr=tmem_ptr_in_smem[0],
            layout=TileLayout(S[(128, num_tmem_cols) : (1 @ TLane, 1 @ TCol)]),
        )
        sfa_tmem = T.decl_buffer(
            (128, sf_block_m // 32),
            "float8_e8m0fnu",
            scope="tmem",
            allocated_addr=num_accum_tmem_cols,
            layout=sf_tmem_layout(128, SF_K=sf_block_m // 32, sf_per_mma=sf_block_m // 32),
        )
        sfb_tmem = T.decl_buffer(
            (128, kernel_config.block_n // 32),
            "float8_e8m0fnu",
            scope="tmem",
            allocated_addr=num_accum_tmem_cols + num_sfa_tmem_cols,
            layout=sf_tmem_layout(
                128, SF_K=kernel_config.block_n // 32, sf_per_mma=kernel_config.block_n // 32
            ),
        )
        desc_a: T.uint64
        desc_b: T.uint64
        desc_sf: T.uint64
        desc_i: T.uint32
        runtime_desc_i: T.uint32
        a_desc_lo: T.uint32
        b_desc_lo: T.uint32
        a_desc_base_lo: T.uint32
        b_desc_base_lo: T.uint32
        dispatch_token_iter: T.int32
        dispatch_token_topk_idx: T.int32
        dispatch_expert_idx: T.int32
        dispatch_dst_rank_idx: T.int32
        dispatch_dst_local_expert_idx: T.int32
        dispatch_dst_slot_idx: T.int32
        pull_local_expert_idx: T.int32
        pull_num_tokens: T.int32
        pull_pool_block_offset: T.int32
        pull_src_token_topk_idx: T.int32
        pull_src_token_idx: T.int32
        pull_src_topk_idx: T.int32
        pull_pool_token_idx: T.int32
        pull_mbarrier_phase: T.int32
        epilogue_value: T.float32
        combine_accum: T.float32
        gate_accum: T.float32
        up_accum: T.float32
        expected_l2_mask: T.uint64
        current_l2_mask: T.uint64
        current_l1_arrival_count: T.uint32
        scheduler_next_phase: T.int32
        scheduler_current_local_expert_idx: T.int32
        scheduler_current_num_tokens: T.int32
        scheduler_current_pool_block_offset: T.int32
        scheduler_block_idx: T.int32
        scheduler_wave_end_expert_idx: T.int32
        scheduler_num_m_blocks: T.int32
        scheduler_found_block: T.int32
        scheduler_cached_status: T.uint64
        scheduler_block_phase: T.int32
        scheduler_local_expert_idx: T.int32
        scheduler_num_k_blocks_local: T.int32
        scheduler_m_block_idx: T.int32
        scheduler_n_block_idx: T.int32
        scheduler_pool_block_idx: T.int32
        scheduler_valid_m: T.int32
        current_expert_idx: T.int32
        old_expert_idx: T.int32
        expert_start_idx: T.int32
        expert_end_idx: T.int32
        token_idx_in_rank: T.int32
        token_idx_in_expert: T.int32
        current_rank_in_expert_idx: T.int32
        rank_count_mask: T.uint32
        num_active_ranks: T.int32
        active_lane_count: T.int32
        num_actives_in_lane: T.int32
        min_in_lane: T.uint32
        min_active_count: T.int32
        round_token_count: T.int32
        slot_idx_in_round: T.int32
        round_offset: T.int32
        barrier_status: T.int32
        barrier_signal_phase: T.int32
        barrier_signal_sign: T.int32
        barrier_target: T.int32
        epilogue_thread_idx: T.int32
        sf_row_idx: T.int32
        epilogue_wg_idx: T.int32
        grid_sync_old_value: T.uint32
        grid_sync_new_value: T.uint32
        nvl_barrier_start_clock: T.uint64
        pipeline_stage_idx: T.int32
        pipeline_phase: T.int32
        k_block_idx: T.int32
        k_idx: T.int32
        k_idx_packed: T.int32
        m_idx: T.int32
        n_idx: T.int32
        pool_block_idx: T.int32
        valid_m: T.int32
        sfa_m_idx: T.int32
        stored_num_tokens_per_expert = T.alloc_local((num_experts_per_lane,), "uint32")
        stored_rank_counts = T.alloc_local((num_ranks_per_lane,), "uint32")
        remaining_rank_counts = T.alloc_local((num_ranks_per_lane,), "uint32")
        combine_phase_local: T.int32
        combine_load_stage_idx: T.int32
        combine_total_mask: T.uint32
        combine_slot_mask: T.uint32
        combine_slot_idx: T.int32
        combine_chunk_offset_elems: T.int32
        combine_do_reduce: T.int32
        combine_next_do_reduce: T.int32

        @T.inline
        def workspace_grid_sync(counter_idx, sync_num_threads, sync_barrier_idx, sync_thread_idx):
            T.ptx.bar.sync(sync_barrier_idx, sync_num_threads)
            if sync_thread_idx == 0:
                if sm_idx == 0:
                    grid_sync_old_value = atomic_add_rel_u32(
                        workspace_grid_sync_count.ptr_to([counter_idx]),
                        T.uint32(0x80000000 - (kernel_config.num_sms - 1)),
                    )
                else:
                    grid_sync_old_value = atomic_add_rel_u32(
                        workspace_grid_sync_count.ptr_to([counter_idx]), T.uint32(1)
                    )
                grid_sync_new_value = load_acq_u32(workspace_grid_sync_count.ptr_to([counter_idx]))
                while grid_sync_done_u32(grid_sync_new_value, grid_sync_old_value) == T.uint32(0):
                    grid_sync_new_value = load_acq_u32(
                        workspace_grid_sync_count.ptr_to([counter_idx])
                    )
            T.ptx.bar.sync(sync_barrier_idx, sync_num_threads)

        @T.inline
        def nvlink_barrier(
            counter_idx,
            barrier_tag,
            sync_num_threads,
            sync_barrier_idx,
            sync_thread_idx,
            sync_prologue,
            sync_epilogue,
        ):
            if num_processes == 1:
                workspace_grid_sync(
                    counter_idx, sync_num_threads, sync_barrier_idx, sync_thread_idx
                )
            else:
                if sync_prologue != 0:
                    workspace_grid_sync(
                        counter_idx, sync_num_threads, sync_barrier_idx, sync_thread_idx
                    )
                if sm_idx == 0:
                    barrier_status = T.cast(
                        T.bitwise_and(workspace_nvl_barrier_counter[0], T.uint32(3)), "int32"
                    )
                    barrier_signal_phase = T.bitwise_and(barrier_status, T.int32(1))
                    barrier_signal_sign = T.shift_right(barrier_status, T.int32(1))
                    if sync_thread_idx < T.int32(num_processes):
                        barrier_target = T.int32(1)
                        if barrier_signal_sign != 0:
                            barrier_target = T.int32(-1)
                        peer_red_add_rel_sys_s32(
                            symm_rank_base_expr(
                                sym_buffer_base, symm_rank_offsets, sync_thread_idx
                            ),
                            T.uint64(workspace_layout.barrier_offset + 20)
                            + T.cast(barrier_signal_phase * 4, "uint64"),
                            barrier_target,
                        )
                    T.ptx.bar.sync(sync_barrier_idx, sync_num_threads)
                    if sync_thread_idx == 0:
                        red_add_gpu_u32(workspace_nvl_barrier_counter.ptr_to([0]), T.uint32(1))
                        barrier_target = T.int32(num_processes)
                        if barrier_signal_sign != 0:
                            barrier_target = T.int32(0)
                        barrier_status = load_acq_sys_s32(
                            workspace_nvl_barrier_signal.ptr_to([barrier_signal_phase])
                        )
                        nvl_barrier_start_clock = cuda_clock64()
                        while barrier_status != barrier_target:
                            if cuda_clock64() - nvl_barrier_start_clock >= T.uint64(
                                num_nvlink_barrier_timeout_cycles
                            ):
                                if kernel_emit_nvl_barrier_timeout_printf:
                                    T.cuda.printf(
                                        "DeepGEMM NVLink barrier timeout (30s): "
                                        "rank=%d, counter=%d, signal=%d, target=%d, "
                                        "phase=%d, sign=%d, tag=%d\n",
                                        rank_idx,
                                        T.cast(workspace_nvl_barrier_counter[0], "int32"),
                                        load_acq_sys_s32(
                                            workspace_nvl_barrier_signal.ptr_to(
                                                [barrier_signal_phase]
                                            )
                                        ),
                                        barrier_target,
                                        barrier_signal_phase,
                                        barrier_signal_sign,
                                        barrier_tag,
                                    )
                                T.cuda.trap_when_assert_failed(False)
                            barrier_status = load_acq_sys_s32(
                                workspace_nvl_barrier_signal.ptr_to([barrier_signal_phase])
                            )
                if sync_epilogue != 0:
                    workspace_grid_sync(
                        counter_idx, sync_num_threads, sync_barrier_idx, sync_thread_idx
                    )

        @T.inline
        def dispatch_nvlink_barrier_before_pull(thread_idx_in_scope):
            nvlink_barrier(
                dispatch_grid_sync_index,
                before_dispatch_pull_barrier_tag,
                kernel_config.num_dispatch_threads,
                dispatch_sync_barrier_idx,
                thread_idx_in_scope,
                0,
                1,
            )

        @T.inline
        def dispatch_nvlink_barrier_after_workspace_clean(thread_idx_in_scope):
            nvlink_barrier(
                dispatch_grid_sync_index,
                after_workspace_clean_barrier_tag,
                kernel_config.num_dispatch_threads,
                dispatch_sync_barrier_idx,
                thread_idx_in_scope,
                1,
                0,
            )

        @T.inline
        def epilogue_nvlink_barrier_before_combine_reduce(thread_idx_in_scope):
            nvlink_barrier(
                epilogue_grid_sync_index,
                before_combine_reduce_barrier_tag,
                kernel_config.num_epilogue_threads,
                epilogue_full_sync_barrier_idx,
                thread_idx_in_scope,
                1,
                1,
            )

        @T.inline
        def scheduler_fetch_expert_recv_count():
            for expert_lane_idx in T.serial(0, num_experts_per_lane):
                dispatch_expert_idx = expert_lane_idx * 32 + lane_idx
                scheduler_cached_status = T.uint64(0)
                if dispatch_expert_idx < num_experts_per_rank:
                    while (
                        T.cast(T.shift_right(scheduler_cached_status, 32), "int32")
                        != kernel_config.num_sms * num_processes
                    ):
                        scheduler_cached_status = load_volatile_u64(
                            workspace_expert_recv_count_sum.ptr_to([dispatch_expert_idx])
                        )
                stored_num_tokens_per_expert[expert_lane_idx] = T.cast(
                    T.bitwise_and(scheduler_cached_status, T.uint64(0xFFFFFFFF)), "uint32"
                )
            T.cuda.warp_sync()

        @T.inline
        def scheduler_set_expert_idx(expert_idx):
            scheduler_current_local_expert_idx = expert_idx
            scheduler_current_num_tokens = scheduler_get_num_tokens_expr(
                expert_idx, lane_idx, stored_num_tokens_per_expert
            )
            scheduler_current_pool_block_offset = scheduler_get_pool_block_offset_expr(
                expert_idx, lane_idx, stored_num_tokens_per_expert
            )

        @T.inline
        def scheduler_advance_expert_idx():
            scheduler_current_pool_block_offset = (
                scheduler_current_pool_block_offset
                + scheduler_get_current_num_m_blocks_expr(scheduler_current_num_tokens)
            )
            scheduler_current_local_expert_idx = scheduler_current_local_expert_idx + T.int32(1)
            scheduler_current_num_tokens = scheduler_get_num_tokens_expr(
                scheduler_current_local_expert_idx, lane_idx, stored_num_tokens_per_expert
            )

        @T.inline
        def scheduler_init():
            scheduler_next_phase = T.int32(1)
            scheduler_block_idx = T.cast(sm_idx, "int32")
            scheduler_fetch_expert_recv_count()
            if num_experts_per_rank > 0:
                scheduler_set_expert_idx(T.int32(0))
            else:
                scheduler_current_local_expert_idx = T.int32(0)
                scheduler_current_num_tokens = T.int32(0)
                scheduler_current_pool_block_offset = T.int32(0)

        @T.inline
        def scheduler_next_block():
            scheduler_block_phase = T.int32(0)
            scheduler_local_expert_idx = T.int32(0)
            scheduler_num_k_blocks_local = T.int32(0)
            scheduler_m_block_idx = T.int32(0)
            scheduler_n_block_idx = T.int32(0)
            scheduler_pool_block_idx = T.int32(0)
            scheduler_valid_m = T.int32(0)
            while True:
                if T.cast(scheduler_current_local_expert_idx, "uint32") >= T.uint32(
                    num_experts_per_rank
                ):
                    break
                if scheduler_next_phase == T.int32(1):
                    scheduler_wave_end_expert_idx = scheduler_get_wave_expert_end_idx_expr(
                        scheduler_current_local_expert_idx
                    )
                    scheduler_found_block = T.int32(0)
                    while T.cast(scheduler_current_local_expert_idx, "uint32") < T.cast(
                        scheduler_wave_end_expert_idx, "uint32"
                    ):
                        scheduler_num_m_blocks = scheduler_get_current_num_m_blocks_expr(
                            scheduler_current_num_tokens
                        )
                        scheduler_block_idx_u32 = T.cast(scheduler_block_idx, "uint32")
                        scheduler_m_block_idx = T.cast(
                            scheduler_block_idx_u32 // T.uint32(num_l1_block_ns), "int32"
                        )
                        if T.cast(scheduler_m_block_idx, "uint32") < T.cast(
                            scheduler_num_m_blocks, "uint32"
                        ):
                            scheduler_block_phase = T.int32(1)
                            scheduler_local_expert_idx = scheduler_current_local_expert_idx
                            scheduler_num_k_blocks_local = T.int32(num_l1_block_ks)
                            scheduler_n_block_idx = (
                                scheduler_block_idx
                                - scheduler_m_block_idx * T.int32(num_l1_block_ns)
                            )
                            scheduler_pool_block_idx = (
                                scheduler_current_pool_block_offset + scheduler_m_block_idx
                            )
                            scheduler_valid_m = T.min(
                                scheduler_current_num_tokens
                                - scheduler_m_block_idx * kernel_config.block_m,
                                kernel_config.block_m,
                            )
                            scheduler_block_idx = scheduler_block_idx + kernel_config.num_sms
                            scheduler_found_block = T.int32(1)
                            break
                        scheduler_block_idx = (
                            scheduler_block_idx - scheduler_num_m_blocks * T.int32(num_l1_block_ns)
                        )
                        scheduler_advance_expert_idx()
                    if T.cast(scheduler_found_block, "uint32") != T.uint32(0):
                        break
                    scheduler_next_phase = T.int32(2)
                    if T.cast(scheduler_current_local_expert_idx, "uint32") > T.uint32(0):
                        scheduler_current_local_expert_idx = (
                            scheduler_current_local_expert_idx - T.int32(1)
                        )
                    scheduler_current_local_expert_idx_u32 = T.cast(
                        scheduler_current_local_expert_idx, "uint32"
                    )
                    scheduler_wave_base_idx_u32 = (
                        scheduler_current_local_expert_idx_u32
                        // T.uint32(kernel_config.num_experts_per_wave)
                        * T.uint32(kernel_config.num_experts_per_wave)
                    )
                    scheduler_set_expert_idx(T.cast(scheduler_wave_base_idx_u32, "int32"))
                else:
                    scheduler_wave_end_expert_idx = scheduler_get_wave_expert_end_idx_expr(
                        scheduler_current_local_expert_idx
                    )
                    scheduler_found_block = T.int32(0)
                    while T.cast(scheduler_current_local_expert_idx, "uint32") < T.cast(
                        scheduler_wave_end_expert_idx, "uint32"
                    ):
                        scheduler_num_m_blocks = scheduler_get_current_num_m_blocks_expr(
                            scheduler_current_num_tokens
                        )
                        if T.cast(scheduler_block_idx, "uint32") < (
                            T.cast(scheduler_num_m_blocks, "uint32") * T.uint32(num_l2_block_ns)
                        ):
                            scheduler_block_phase = T.int32(2)
                            scheduler_local_expert_idx = scheduler_current_local_expert_idx
                            scheduler_num_k_blocks_local = T.int32(num_l2_block_ks)
                            scheduler_block_idx_u32 = T.cast(scheduler_block_idx, "uint32")
                            scheduler_m_block_idx = T.cast(
                                scheduler_block_idx_u32 // T.uint32(num_l2_block_ns), "int32"
                            )
                            scheduler_n_block_idx = (
                                scheduler_block_idx
                                - scheduler_m_block_idx * T.int32(num_l2_block_ns)
                            )
                            scheduler_pool_block_idx = (
                                scheduler_current_pool_block_offset + scheduler_m_block_idx
                            )
                            scheduler_valid_m = T.min(
                                scheduler_current_num_tokens
                                - scheduler_m_block_idx * kernel_config.block_m,
                                kernel_config.block_m,
                            )
                            scheduler_block_idx = scheduler_block_idx + kernel_config.num_sms
                            scheduler_found_block = T.int32(1)
                            break
                        scheduler_block_idx = (
                            scheduler_block_idx - scheduler_num_m_blocks * T.int32(num_l2_block_ns)
                        )
                        scheduler_advance_expert_idx()
                    if T.cast(scheduler_found_block, "uint32") != T.uint32(0):
                        break
                    scheduler_next_phase = T.int32(1)

        @T.inline
        def scheduler_bind_block_args():
            block_phase = scheduler_block_phase
            local_expert_idx = scheduler_local_expert_idx
            num_k_blocks = scheduler_num_k_blocks_local
            m_block_idx = scheduler_m_block_idx
            n_block_idx = scheduler_n_block_idx
            pool_block_idx = scheduler_pool_block_idx
            valid_m = scheduler_valid_m

        @T.inline
        def update_get_valid_m_true():
            valid_m_u32 = T.cast(valid_m, "uint32")
            get_valid_m_true_u32 = (valid_m_u32 + T.uint32(15)) // T.uint32(16) * T.uint32(16)
            get_valid_m_true = T.cast(get_valid_m_true_u32, "int32")
            get_valid_m_true_half = T.cast(get_valid_m_true_u32 // T.uint32(2), "int32")
            get_valid_m_true_eighth = T.cast(get_valid_m_true_u32 // T.uint32(8), "int32")

        @T.inline
        def advance_pipeline():
            pipeline_stage_idx = (
                T.int32(0)
                if pipeline_stage_idx == T.int32(num_stages - 1)
                else pipeline_stage_idx + T.int32(1)
            )
            if pipeline_stage_idx == 0:
                pipeline_phase = pipeline_phase ^ T.int32(1)

        @T.inline
        def barrier_wait(barrier_ptr, phase):
            T.ptx.mbarrier.try_wait(barrier_ptr, phase)

        @T.inline
        def tmem_empty_barrier_arrive_cta0(tmem_empty_barrier_ptr):
            T.ptx.mbarrier.arrive(tmem_empty_barrier_ptr, cta_id=0, pred=True)

        @T.inline
        def umma_arrive_multicast_2x1sm(barrier_ptr):
            if T.ptx.elect_sync():
                T.ptx.tcgen05.commit(
                    barrier_ptr,
                    cta_group=kernel_config.num_ctas_per_cluster,
                    cta_mask=tcgen05_cta_mask,
                )

        @T.inline
        def umma_arrive(barrier_ptr):
            umma_arrive_multicast_2x1sm(barrier_ptr)

        @T.inline
        def empty_barrier_arrive(do_tmem_full_arrive, empty_barrier_ptr, tmem_full_barrier_ptr):
            umma_arrive(empty_barrier_ptr)
            if do_tmem_full_arrive:
                umma_arrive(tmem_full_barrier_ptr)
            T.cuda.warp_sync()

        @T.inline
        def empty_barrier_arrive_current(do_tmem_full_arrive):
            empty_barrier_arrive(
                do_tmem_full_arrive,
                smem_barriers.ptr_to([empty_barrier_base + pipeline_stage_idx]),
                smem_barriers.ptr_to([tmem_full_barrier_base + accum_stage_idx]),
            )

        @T.inline
        def fence_view_async_tmem_load():
            T.ptx.tcgen05.wait.ld()

        @T.inline
        def warpgroup_reg_dealloc(num_registers):
            T.ptx.setmaxnreg(False, num_registers)

        @T.inline
        def warpgroup_reg_alloc(num_registers):
            T.ptx.setmaxnreg(True, num_registers)

        @T.inline
        def tma_copy_2d_multicast(dst_ptr, barrier_ptr, tensor_map_ptr, coord0, coord1):
            sm100_tma_2sm_load_2d(dst_ptr, barrier_ptr, tensor_map_ptr, coord0, coord1)

        @T.inline
        def tma_copy_2d_multicast_select(
            dst_ptr,
            barrier_ptr,
            tensor_map_l1_ptr,
            tensor_map_l2_ptr,
            block_phase_value,
            coord0,
            coord1,
        ):
            sm100_tma_2sm_load_2d_select(
                dst_ptr,
                barrier_ptr,
                tensor_map_l1_ptr,
                tensor_map_l2_ptr,
                block_phase_value,
                coord0,
                coord1,
            )

        @T.inline
        def full_barrier_arrive_and_expect_tx(full_barrier_ptr, transaction_bytes):
            T.ptx.mbarrier.arrive.expect_tx(full_barrier_ptr, transaction_bytes)

        @T.inline
        def full_barrier_arrive_cta0(full_barrier_ptr):
            T.ptx.mbarrier.arrive(full_barrier_ptr, cta_id=0, pred=True)

        @T.inline
        def make_instr_desc_block_scaled():
            T.ptx.tcgen05.encode_instr_descriptor_block_scaled(
                T.address_of(desc_i),
                d_dtype="float32",
                a_dtype="float4_e2m1fn",
                b_dtype="float8_e4m3fn",
                sfa_dtype="float8_e8m0fnu",
                sfb_dtype="float8_e8m0fnu",
                sfa_tmem_addr=0,
                sfb_tmem_addr=0,
                M=umma_m,
                N=umma_n,
                K=umma_k,
                trans_a=False,
                trans_b=False,
                n_cta_groups=kernel_config.num_ctas_per_cluster,
            )

        @T.inline
        def make_sf_desc():
            T.ptx.tcgen05.encode_matrix_descriptor(
                T.address_of(desc_sf), smem_sfa.ptr_to([0, 0]), ldo=0, sdo=sf_desc_sdo, swizzle=0
            )

        @T.inline
        def make_umma_desc_a():
            T.ptx.tcgen05.encode_matrix_descriptor(
                T.address_of(desc_a), smem_a_fp8.ptr_to([0, 0, 0]), ldo=0, sdo=a_desc_sdo, swizzle=3
            )

        @T.inline
        def make_umma_desc_b():
            T.ptx.tcgen05.encode_matrix_descriptor(
                T.address_of(desc_b), smem_b.ptr_to([0, 0, 0]), ldo=0, sdo=b_desc_sdo, swizzle=3
            )

        @T.inline
        def utccp_copy(tmem_addr, sf_desc):
            T.ptx.tcgen05.cp(
                tmem_addr,
                sf_desc,
                shape="32x128b",
                cta_group=kernel_config.num_ctas_per_cluster,
                multicast="warpx4",
            )

        @T.inline
        def sm100_u8x4_stsm_t_copy(fp8x4_values_ptr, smem_ptr):
            stmatrix_fp8x4_trans(smem_ptr, fp8x4_values_ptr)

        @T.inline
        def sm90_u32x4_stsm_t_copy(packed_values_buf, smem_ptr):
            T.ptx.stmatrix(
                True,
                4,
                ".b16",
                smem_ptr,
                packed_values_buf.ptr_to([0]),
                packed_values_buf.ptr_to([1]),
                packed_values_buf.ptr_to([2]),
                packed_values_buf.ptr_to([3]),
            )

        @T.inline
        def sm90_tma_store_2d_copy(src_ptr, tensor_map, coord0, coord1):
            T.evaluate(tma_store_2d(src_ptr, tensor_map, coord0, coord1))

        @T.inline
        def red_or_rel_gpu(address, value):
            red_or_rel_gpu_u64(address, value)

        @T.inline
        def store_token_src_metadata(pool_token_idx, src_rank_idx, src_token_idx, src_topk_idx):
            workspace_token_src_metadata[pool_token_idx, 0] = T.cast(src_rank_idx, "uint32")
            workspace_token_src_metadata[pool_token_idx, 1] = T.cast(src_token_idx, "uint32")
            workspace_token_src_metadata[pool_token_idx, 2] = T.cast(src_topk_idx, "uint32")

        # Relaxed arrive — no prior memory effect needs to be released to peers
        # before TMEM alloc + mbarrier init below. Wait still .acquire (default).
        T.ptx.barrier.cluster.arrive(sem="relaxed", aligned=True)
        T.ptx.barrier.cluster.wait(acquire=True, aligned=True)
        full_barrier_init_count: T.int32 = 2 * 2
        tmem_empty_barrier_init_count: T.int32 = 2 * kernel_config.num_epilogue_threads
        is_reserved_non_epilogue_warp = (
            flat_warp_idx == kernel_config.reserved_non_epilogue_warp_idx
        )
        if flat_warp_idx == 0:
            if T.ptx.elect_sync():
                T.evaluate(st_shared_bulk(smem_expert_count.ptr_to([0]), T.uint32(num_experts * 4)))
        elif flat_warp_idx == 1:
            dispatch_expert_idx = lane_idx
            while dispatch_expert_idx < kernel_config.num_dispatch_warps:
                T.ptx.mbarrier.init(
                    smem_barriers.ptr_to([dispatch_barrier_base + dispatch_expert_idx]), 1
                )
                dispatch_expert_idx = dispatch_expert_idx + 32
            T.evaluate(fence_barrier_init())
        elif flat_warp_idx == 2:
            if T.ptx.elect_sync():
                dispatch_expert_idx = T.int32(0)
                while dispatch_expert_idx < num_stages:
                    T.ptx.mbarrier.init(
                        smem_barriers.ptr_to([full_barrier_base + dispatch_expert_idx]),
                        full_barrier_init_count,
                    )
                    T.ptx.mbarrier.init(
                        smem_barriers.ptr_to([empty_barrier_base + dispatch_expert_idx]), 1
                    )
                    dispatch_expert_idx = dispatch_expert_idx + 1
                dispatch_expert_idx = T.int32(0)
                while dispatch_expert_idx < num_epilogue_stages:
                    T.ptx.mbarrier.init(
                        smem_barriers.ptr_to([tmem_full_barrier_base + dispatch_expert_idx]), 1
                    )
                    T.ptx.mbarrier.init(
                        smem_barriers.ptr_to([tmem_empty_barrier_base + dispatch_expert_idx]),
                        tmem_empty_barrier_init_count,
                    )
                    dispatch_expert_idx = dispatch_expert_idx + 1
                dispatch_expert_idx = T.int32(0)
                while dispatch_expert_idx < kernel_config.num_epilogue_warps * 2:
                    T.ptx.mbarrier.init(
                        smem_barriers.ptr_to([combine_barrier_base + dispatch_expert_idx]), 1
                    )
                    dispatch_expert_idx = dispatch_expert_idx + 1
            T.evaluate(fence_barrier_init())
        elif flat_warp_idx == kernel_config.num_dispatch_warps - 1:
            T.ptx.tcgen05.alloc(
                T.address_of(tmem_ptr_in_smem[0]),
                n_cols=num_tmem_cols,
                cta_group=kernel_config.num_ctas_per_cluster,
            )
        # `fence_barrier_init` above is `.release.cluster`, so the cluster
        # arrive here doesn't need release semantics. `.wait.acquire` pairs
        # with that release to make mbarrier init visible to other CTAs.
        T.ptx.barrier.cluster.arrive(sem="relaxed", aligned=True)
        T.ptx.barrier.cluster.wait(acquire=True, aligned=True)
        if flat_warp_idx < kernel_config.num_dispatch_warps:
            warpgroup_reg_dealloc(num_dispatch_registers)
            dispatch_token_iter = (
                sm_idx * kernel_config.num_dispatch_warps + flat_warp_idx
            ) * kernel_config.num_tokens_per_warp
            while T.cast(dispatch_token_iter, "uint32") < T.cast(num_tokens, "uint32"):
                if lane_idx < kernel_config.num_activate_lanes:
                    lane_idx_u32 = T.cast(lane_idx, "uint32")
                    token_idx = dispatch_token_iter + T.cast(
                        lane_idx_u32 // T.uint32(num_topk), "int32"
                    )
                    if T.cast(token_idx, "uint32") < T.cast(num_tokens, "uint32"):
                        topk_idx = T.cast(lane_idx_u32 % T.uint32(num_topk), "int32")
                        dispatch_expert_idx = T.cast(input_topk_idx[token_idx, topk_idx], "int32")
                        if dispatch_expert_idx >= 0:
                            T.evaluate(
                                T.cuda.atomic_add(
                                    smem_expert_count.ptr_to([dispatch_expert_idx]), 1
                                )
                            )
                T.cuda.warp_sync()
                dispatch_token_iter = (
                    dispatch_token_iter
                    + kernel_config.num_sms
                    * kernel_config.num_dispatch_warps
                    * kernel_config.num_tokens_per_warp
                )

            T.ptx.bar.sync(dispatch_sync_barrier_idx, kernel_config.num_dispatch_threads)
            dispatch_expert_idx = flat_warp_idx * 32 + lane_idx
            while T.cast(dispatch_expert_idx, "uint32") < T.uint32(num_experts):
                send_value = T.bitwise_or(
                    T.uint64(1 << 32), T.cast(smem_expert_count[dispatch_expert_idx], "uint64")
                )
                smem_expert_count[dispatch_expert_idx] = T.cast(
                    T.ptx.atom_scalar(
                        workspace_expert_send_count.ptr_to([dispatch_expert_idx]),
                        send_value,
                        space="global",
                        op="add",
                        ptx_type="u64",
                    ),
                    "int32",
                )
                dispatch_expert_idx = dispatch_expert_idx + kernel_config.num_dispatch_threads
            T.ptx.bar.sync(dispatch_sync_barrier_idx, kernel_config.num_dispatch_threads)

        if flat_warp_idx < kernel_config.num_dispatch_warps:
            dispatch_token_iter = (
                sm_idx * kernel_config.num_dispatch_warps + flat_warp_idx
            ) * kernel_config.num_tokens_per_warp
            while T.cast(dispatch_token_iter, "uint32") < T.cast(num_tokens, "uint32"):
                if lane_idx < kernel_config.num_activate_lanes:
                    lane_idx_u32 = T.cast(lane_idx, "uint32")
                    token_idx = dispatch_token_iter + T.cast(
                        lane_idx_u32 // T.uint32(num_topk), "int32"
                    )
                    if T.cast(token_idx, "uint32") < T.cast(num_tokens, "uint32"):
                        topk_idx = T.cast(lane_idx_u32 % T.uint32(num_topk), "int32")
                        dispatch_token_topk_idx = token_idx * num_topk + topk_idx
                        dispatch_expert_idx = T.cast(input_topk_idx[token_idx, topk_idx], "int32")
                        if dispatch_expert_idx >= 0:
                            dispatch_expert_idx_u32 = T.cast(dispatch_expert_idx, "uint32")
                            dispatch_dst_rank_idx = T.cast(
                                dispatch_expert_idx_u32 // T.uint32(num_experts_per_rank), "int32"
                            )
                            dispatch_dst_local_expert_idx = T.cast(
                                dispatch_expert_idx_u32 % T.uint32(num_experts_per_rank), "int32"
                            )
                            dispatch_dst_slot_idx = T.cuda.atomic_add(
                                smem_expert_count.ptr_to([dispatch_expert_idx]), 1
                            )
                            peer_store_u32(
                                symm_rank_base_expr(
                                    sym_buffer_base, symm_rank_offsets, dispatch_dst_rank_idx
                                ),
                                T.uint64(
                                    workspace_layout.src_token_topk_idx_offset
                                    + (
                                        dispatch_dst_local_expert_idx
                                        * num_processes
                                        * workspace_layout.num_max_recv_tokens_per_expert
                                        + rank_idx * workspace_layout.num_max_recv_tokens_per_expert
                                        + dispatch_dst_slot_idx
                                    )
                                    * 4
                                ),
                                T.cast(dispatch_token_topk_idx, "uint32"),
                            )
                T.cuda.warp_sync()
                dispatch_token_iter = (
                    dispatch_token_iter
                    + kernel_config.num_sms
                    * kernel_config.num_dispatch_warps
                    * kernel_config.num_tokens_per_warp
                )
        if flat_warp_idx < kernel_config.num_dispatch_warps:
            workspace_grid_sync(
                0,
                kernel_config.num_dispatch_threads,
                dispatch_sync_barrier_idx,
                flat_warp_idx * 32 + lane_idx,
            )
        if sm_idx == 0 and flat_warp_idx < kernel_config.num_dispatch_warps:
            dispatch_expert_idx = flat_warp_idx * 32 + lane_idx
            while T.cast(dispatch_expert_idx, "uint32") < T.uint32(num_experts):
                dispatch_expert_idx_u32 = T.cast(dispatch_expert_idx, "uint32")
                dispatch_dst_rank_idx = T.cast(
                    dispatch_expert_idx_u32 // T.uint32(num_experts_per_rank), "int32"
                )
                dispatch_dst_local_expert_idx = T.cast(
                    dispatch_expert_idx_u32 % T.uint32(num_experts_per_rank), "int32"
                )
                scheduler_cached_status = workspace_expert_send_count[dispatch_expert_idx]
                peer_store_u64(
                    symm_rank_base_expr(sym_buffer_base, symm_rank_offsets, dispatch_dst_rank_idx),
                    T.uint64(
                        workspace_layout.expert_recv_count_offset
                        + (rank_idx * num_experts_per_rank + dispatch_dst_local_expert_idx) * 8
                    ),
                    T.bitwise_and(scheduler_cached_status, T.uint64(0xFFFFFFFF)),
                )
                peer_atomic_add_u64(
                    symm_rank_base_expr(sym_buffer_base, symm_rank_offsets, dispatch_dst_rank_idx),
                    T.uint64(
                        workspace_layout.expert_recv_count_sum_offset
                        + dispatch_dst_local_expert_idx * 8
                    ),
                    scheduler_cached_status,
                )
                dispatch_expert_idx = dispatch_expert_idx + kernel_config.num_dispatch_threads
        if flat_warp_idx < kernel_config.num_dispatch_warps:
            T.ptx.bar.sync(dispatch_sync_barrier_idx, kernel_config.num_dispatch_threads)
        if flat_warp_idx < kernel_config.num_dispatch_warps:
            dispatch_nvlink_barrier_before_pull(flat_warp_idx * 32 + lane_idx)
            T.evaluate(
                sync_unaligned(
                    dispatch_with_epilogue_sync_barrier_idx,
                    kernel_config.num_dispatch_threads + kernel_config.num_epilogue_threads,
                )
            )
        if flat_warp_idx < kernel_config.num_dispatch_warps:
            pull_mbarrier_phase = T.int32(0)
            scheduler_fetch_expert_recv_count()
            current_expert_idx = T.int32(-1)
            old_expert_idx = T.int32(-1)
            expert_start_idx = T.int32(0)
            expert_end_idx = T.int32(0)
            pull_pool_block_offset = T.int32(0)
            dispatch_token_iter = sm_idx * kernel_config.num_dispatch_warps + flat_warp_idx
            while True:
                old_expert_idx = current_expert_idx
                while T.cast(dispatch_token_iter, "uint32") >= T.cast(expert_end_idx, "uint32"):
                    current_expert_idx = current_expert_idx + T.int32(1)
                    if T.cast(current_expert_idx, "uint32") >= T.uint32(num_experts_per_rank):
                        break
                    expert_token_count_u32 = T.cast(expert_end_idx - expert_start_idx, "uint32")
                    expert_num_blocks_u32 = (
                        expert_token_count_u32 + T.uint32(kernel_config.block_m - 1)
                    ) // T.uint32(kernel_config.block_m)
                    pull_pool_block_offset = pull_pool_block_offset + T.cast(
                        expert_num_blocks_u32, "int32"
                    )
                    expert_start_idx = expert_end_idx
                    expert_end_idx = expert_end_idx + scheduler_get_num_tokens_expr(
                        current_expert_idx, lane_idx, stored_num_tokens_per_expert
                    )
                if T.cast(current_expert_idx, "uint32") >= T.uint32(num_experts_per_rank):
                    break
                if old_expert_idx != current_expert_idx:
                    old_expert_idx = current_expert_idx
                    for rank_lane_idx in T.unroll(0, num_ranks_per_lane):
                        dispatch_dst_rank_idx = rank_lane_idx * 32 + lane_idx
                        stored_rank_counts[rank_lane_idx] = T.Select(
                            T.cast(dispatch_dst_rank_idx, "uint32") < T.uint32(num_processes),
                            T.cast(
                                workspace_expert_recv_count[
                                    dispatch_dst_rank_idx, current_expert_idx
                                ],
                                "uint32",
                            ),
                            T.uint32(0),
                        )
                token_idx_in_expert = dispatch_token_iter - expert_start_idx
                dispatch_dst_slot_idx = token_idx_in_expert
                round_offset = T.int32(0)
                for rank_lane_idx in T.unroll(0, num_ranks_per_lane):
                    remaining_rank_counts[rank_lane_idx] = stored_rank_counts[rank_lane_idx]
                while True:
                    min_in_lane = T.uint32(0xFFFFFFFF)
                    num_actives_in_lane = T.int32(0)
                    for rank_lane_idx in T.unroll(0, num_ranks_per_lane):
                        if remaining_rank_counts[rank_lane_idx] > T.uint32(0):
                            num_actives_in_lane = num_actives_in_lane + T.int32(1)
                            min_in_lane = T.min(min_in_lane, remaining_rank_counts[rank_lane_idx])
                    num_active_ranks = T.cast(
                        reduce_add_sync_u32(
                            T.uint32(0xFFFFFFFF), T.cast(num_actives_in_lane, "uint32")
                        ),
                        "int32",
                    )
                    min_active_count = T.cast(
                        reduce_min_sync_u32(T.uint32(0xFFFFFFFF), min_in_lane), "int32"
                    )
                    round_token_count = min_active_count * num_active_ranks
                    if T.cast(dispatch_dst_slot_idx, "uint32") < T.cast(
                        round_token_count, "uint32"
                    ):
                        dispatch_dst_slot_idx_u32 = T.cast(dispatch_dst_slot_idx, "uint32")
                        num_active_ranks_u32 = T.cast(num_active_ranks, "uint32")
                        slot_idx_in_round = T.cast(
                            dispatch_dst_slot_idx_u32 % num_active_ranks_u32, "int32"
                        )
                        num_seen_ranks = T.int32(0)
                        current_rank_in_expert_idx = T.int32(0)
                        for rank_lane_idx in T.unroll(0, num_ranks_per_lane):
                            rank_count_mask = ballot_sync(
                                T.uint32(0xFFFFFFFF),
                                remaining_rank_counts[rank_lane_idx] > T.uint32(0),
                            )
                            active_lane_count = T.cast(T.popcount(rank_count_mask), "int32")
                            if T.cast(slot_idx_in_round, "uint32") >= T.cast(
                                num_seen_ranks, "uint32"
                            ) and T.cast(slot_idx_in_round, "uint32") < T.cast(
                                num_seen_ranks + active_lane_count, "uint32"
                            ):
                                current_rank_in_expert_idx = rank_lane_idx * 32 + T.cast(
                                    fns_b32(
                                        rank_count_mask,
                                        T.uint32(0),
                                        slot_idx_in_round - num_seen_ranks + T.int32(1),
                                    ),
                                    "int32",
                                )
                            num_seen_ranks = num_seen_ranks + active_lane_count
                        token_idx_in_rank = round_offset + T.cast(
                            dispatch_dst_slot_idx_u32 // num_active_ranks_u32, "int32"
                        )
                        break
                    dispatch_dst_slot_idx = dispatch_dst_slot_idx - round_token_count
                    round_offset = round_offset + min_active_count
                    for rank_lane_idx in T.unroll(0, num_ranks_per_lane):
                        remaining_rank_counts[rank_lane_idx] = remaining_rank_counts[
                            rank_lane_idx
                        ] - T.min(
                            remaining_rank_counts[rank_lane_idx], T.cast(min_active_count, "uint32")
                        )
                pull_src_token_topk_idx = T.cast(
                    workspace_src_token_topk_idx[
                        current_expert_idx, current_rank_in_expert_idx, token_idx_in_rank
                    ],
                    "int32",
                )
                pull_src_token_topk_idx_u32 = T.cast(pull_src_token_topk_idx, "uint32")
                pull_src_token_idx = T.cast(
                    pull_src_token_topk_idx_u32 // T.uint32(num_topk), "int32"
                )
                pull_src_topk_idx = T.cast(
                    pull_src_token_topk_idx_u32 % T.uint32(num_topk), "int32"
                )
                if T.ptx.elect_sync():
                    tma_load_1d_symm(
                        smem_send_buffers.ptr_to([flat_warp_idx, 0]),
                        symm_rank_base_expr(
                            sym_buffer_base, symm_rank_offsets, current_rank_in_expert_idx
                        ),
                        T.uint64(
                            symm_buffer_layout.input_token_offset + pull_src_token_idx * hidden
                        ),
                        smem_barriers.ptr_to([dispatch_barrier_base + flat_warp_idx]),
                        hidden,
                    )
                T.cuda.warp_sync()
                pull_pool_token_idx = (
                    pull_pool_block_offset * kernel_config.block_m + token_idx_in_expert
                )
                sf_row_idx = pull_pool_block_offset * sf_block_m + transform_sf_token_idx(
                    token_idx_in_expert
                )
                dispatch_dst_rank_idx = lane_idx
                while dispatch_dst_rank_idx < hidden // 128:
                    l1_acts_sf[dispatch_dst_rank_idx, sf_row_idx] = T.cast(
                        peer_load_u32(
                            symm_rank_base_expr(
                                sym_buffer_base, symm_rank_offsets, current_rank_in_expert_idx
                            ),
                            T.uint64(
                                symm_buffer_layout.input_sf_offset
                                + (pull_src_token_idx * (hidden // 128) + dispatch_dst_rank_idx) * 4
                            ),
                        ),
                        "int32",
                    )
                    dispatch_dst_rank_idx = dispatch_dst_rank_idx + 32
                T.cuda.warp_sync()
                if T.ptx.elect_sync():
                    l1_topk_weights[pull_pool_token_idx] = peer_load_f32(
                        symm_rank_base_expr(
                            sym_buffer_base, symm_rank_offsets, current_rank_in_expert_idx
                        ),
                        T.uint64(
                            symm_buffer_layout.input_topk_weights_offset
                            + pull_src_token_topk_idx * 4
                        ),
                    )
                    mbarrier_arrive_and_set_tx(
                        smem_barriers.ptr_to([dispatch_barrier_base + flat_warp_idx]), hidden
                    )
                    mbarrier_wait_phase(
                        smem_barriers.ptr_to([dispatch_barrier_base + flat_warp_idx]),
                        pull_mbarrier_phase,
                    )
                    pull_mbarrier_phase = pull_mbarrier_phase ^ T.int32(1)
                    tma_store_1d(
                        T.address_of(l1_acts[pull_pool_token_idx, 0]),
                        smem_send_buffers.ptr_to([flat_warp_idx, 0]),
                        hidden,
                    )
                    store_token_src_metadata(
                        pull_pool_token_idx,
                        current_rank_in_expert_idx,
                        pull_src_token_idx,
                        pull_src_topk_idx,
                    )
                    T.evaluate(tma_store_arrive())
                    T.evaluate(tma_store_wait(0))
                    T.evaluate(
                        atomic_add_rel_u32(
                            workspace_l1_arrival_count.ptr_to(
                                [
                                    pull_pool_block_offset
                                    + T.cast(
                                        T.cast(token_idx_in_expert, "uint32")
                                        // T.uint32(kernel_config.block_m),
                                        "int32",
                                    )
                                ]
                            ),
                            T.uint32(1),
                        )
                    )
                T.cuda.warp_sync()
                dispatch_token_iter = (
                    dispatch_token_iter + kernel_config.num_sms * kernel_config.num_dispatch_warps
                )
            T.evaluate(
                sync_unaligned(
                    dispatch_with_epilogue_sync_barrier_idx,
                    kernel_config.num_dispatch_threads + kernel_config.num_epilogue_threads,
                )
            )
            T.ptx.bar.sync(dispatch_sync_barrier_idx, kernel_config.num_dispatch_threads)
            if sm_idx == 0:
                dispatch_expert_idx = thread_idx
                while dispatch_expert_idx < num_experts:
                    workspace_expert_send_count[dispatch_expert_idx] = T.uint64(0)
                    dispatch_expert_idx = dispatch_expert_idx + kernel_config.num_dispatch_threads
            else:
                pull_local_expert_idx = sm_idx - 1
                while pull_local_expert_idx < num_experts_per_rank:
                    pull_num_tokens = scheduler_get_num_tokens_expr(
                        pull_local_expert_idx, lane_idx, stored_num_tokens_per_expert
                    )
                    pull_num_tokens_u32 = T.cast(pull_num_tokens, "uint32")
                    scheduler_num_m_blocks = T.cast(
                        (pull_num_tokens_u32 + T.uint32(kernel_config.block_m - 1))
                        // T.uint32(kernel_config.block_m),
                        "int32",
                    )
                    pull_pool_block_offset = scheduler_get_pool_block_offset_expr(
                        pull_local_expert_idx, lane_idx, stored_num_tokens_per_expert
                    )
                    T.ptx.bar.sync(dispatch_sync_barrier_idx, kernel_config.num_dispatch_threads)
                    if thread_idx == 0:
                        workspace_expert_recv_count_sum[pull_local_expert_idx] = T.uint64(0)
                    dispatch_dst_rank_idx = thread_idx
                    while dispatch_dst_rank_idx < T.int32(num_processes):
                        workspace_expert_recv_count[
                            dispatch_dst_rank_idx, pull_local_expert_idx
                        ] = T.uint64(0)
                        dispatch_dst_rank_idx = (
                            dispatch_dst_rank_idx + kernel_config.num_dispatch_threads
                        )
                    dispatch_dst_slot_idx = thread_idx
                    while dispatch_dst_slot_idx < scheduler_num_m_blocks:
                        workspace_l1_arrival_count[
                            pull_pool_block_offset + dispatch_dst_slot_idx
                        ] = T.uint32(0)
                        workspace_l2_arrival_mask[
                            pull_pool_block_offset + dispatch_dst_slot_idx
                        ] = T.uint64(0)
                        dispatch_dst_slot_idx = (
                            dispatch_dst_slot_idx + kernel_config.num_dispatch_threads
                        )
                    pull_local_expert_idx = pull_local_expert_idx + (kernel_config.num_sms - 1)
            dispatch_nvlink_barrier_after_workspace_clean(flat_warp_idx * 32 + lane_idx)
        scheduler_iter_idx = T.local_scalar("int32")
        current_iter_idx = T.local_scalar("int32")
        accum_stage_idx = T.local_scalar("int32")
        accum_phase = T.local_scalar("int32")
        block_phase = T.local_scalar("int32")
        local_expert_idx = T.local_scalar("int32")
        num_k_blocks = T.local_scalar("int32")
        m_block_idx = T.local_scalar("int32")
        n_block_idx = T.local_scalar("int32")
        get_valid_m_true = T.local_scalar("int32")
        get_valid_m_true_half = T.local_scalar("int32")
        get_valid_m_true_eighth = T.local_scalar("int32")
        shape_k = T.local_scalar("int32")
        shape_n = T.local_scalar("int32")
        shape_sfa_k = T.local_scalar("int32")
        shape_sfb_k = T.local_scalar("int32")
        scheduler_iter_idx = 0
        current_iter_idx = 0
        accum_stage_idx = 0
        accum_phase = 0
        block_phase = 0
        local_expert_idx = 0
        num_k_blocks = 0
        m_block_idx = 0
        n_block_idx = 0
        get_valid_m_true = 0
        get_valid_m_true_half = 0
        get_valid_m_true_eighth = 0
        shape_k = 0
        shape_n = 0
        shape_sfa_k = 0
        shape_sfb_k = 0

        if flat_warp_idx == kernel_config.load_a_warp_idx:
            warpgroup_reg_dealloc(num_non_epilogue_registers)
            scheduler_init()
            scheduler_iter_idx = 0
            pipeline_stage_idx = T.int32(0)
            pipeline_phase = T.int32(0)
            while True:
                scheduler_next_block()
                scheduler_bind_block_args()
                if block_phase == T.int32(0):
                    break
                scheduler_iter_idx_u32 = T.cast(scheduler_iter_idx, "uint32")
                accum_stage_idx = T.cast(
                    scheduler_iter_idx_u32 % T.uint32(num_epilogue_stages), "int32"
                )
                accum_phase = T.cast(
                    T.bitwise_and(
                        scheduler_iter_idx_u32 // T.uint32(num_epilogue_stages), T.uint32(1)
                    ),
                    "int32",
                )
                shape_k = T.Select(
                    block_phase == T.int32(2), T.int32(intermediate_hidden), T.int32(hidden)
                )
                shape_k_u32 = T.cast(shape_k, "uint32")
                shape_sfa_k = T.cast(
                    (shape_k_u32 + T.uint32(kernel_config.block_k - 1))
                    // T.uint32(kernel_config.block_k),
                    "int32",
                )
                pull_pool_block_offset = pool_block_idx
                if block_phase == T.int32(1):
                    current_l1_arrival_count = load_acq_u32(
                        workspace_l1_arrival_count.ptr_to([pull_pool_block_offset])
                    )
                    while current_l1_arrival_count != T.cast(valid_m, "uint32"):
                        current_l1_arrival_count = load_acq_u32(
                            workspace_l1_arrival_count.ptr_to([pull_pool_block_offset])
                        )
                else:
                    # Wait for ALL 2*num_k_blocks L2 mask bits before entering the inner k loop.
                    # Upstream removed the per-k-block on-demand wait — when num_experts_per_wave
                    # is large enough that L1 finishes before L2 starts (the common case), the
                    # inner check is dead overhead. Written as `((1<<n)<<n) - 1` instead of
                    # `(1<<(2n)) - 1` to avoid UB when n == 32.
                    num_k_blocks_u64 = T.cast(num_k_blocks, "uint64")
                    expected_full_l2_mask = T.shift_left(
                        T.shift_left(T.uint64(1), num_k_blocks_u64), num_k_blocks_u64
                    ) - T.uint64(1)
                    current_l2_mask = load_acq_gpu_u64(
                        workspace_l2_arrival_mask.ptr_to([pull_pool_block_offset])
                    )
                    while current_l2_mask != expected_full_l2_mask:
                        current_l2_mask = load_acq_gpu_u64(
                            workspace_l2_arrival_mask.ptr_to([pull_pool_block_offset])
                        )
                for k_block_idx in T.serial(0, num_k_blocks):
                    barrier_wait(
                        smem_barriers.ptr_to([empty_barrier_base + pipeline_stage_idx]),
                        pipeline_phase ^ T.int32(1),
                    )
                    pool_block_idx = scheduler_pool_block_idx
                    m_idx = pool_block_idx * kernel_config.block_m
                    k_idx = k_block_idx * kernel_config.block_k
                    sfa_m_idx = pool_block_idx * sf_block_m
                    sfa_k_idx = k_block_idx
                    if cta_idx_in_cluster != 0:
                        update_get_valid_m_true()
                        m_idx += get_valid_m_true_half
                    if T.ptx.elect_sync():
                        full_barrier_ptr = smem_barriers.ptr_to(
                            [full_barrier_base + pipeline_stage_idx]
                        )
                        tma_copy_2d_multicast_select(
                            smem_a.ptr_to([pipeline_stage_idx, 0, 0]),
                            full_barrier_ptr,
                            tensor_map_l1_acts,
                            tensor_map_l2_acts,
                            block_phase,
                            k_idx,
                            m_idx,
                        )
                        tma_copy_2d_multicast_select(
                            smem_sfa_i32.ptr_to([pipeline_stage_idx, 0, 0]),
                            full_barrier_ptr,
                            tensor_map_l1_acts_sf,
                            tensor_map_l2_acts_sf,
                            block_phase,
                            sfa_m_idx,
                            sfa_k_idx,
                        )
                        if cta_idx_in_cluster == 0:
                            full_barrier_arrive_and_expect_tx(
                                full_barrier_ptr, full_a_expect_tx_leader_bytes
                            )
                        else:
                            full_barrier_arrive_cta0(full_barrier_ptr)
                    T.cuda.warp_sync()
                    advance_pipeline()
                scheduler_iter_idx = scheduler_iter_idx + 1
        elif flat_warp_idx == kernel_config.load_b_warp_idx:
            warpgroup_reg_dealloc(num_non_epilogue_registers)
            scheduler_init()
            scheduler_iter_idx = 0
            pipeline_stage_idx = T.int32(0)
            pipeline_phase = T.int32(0)
            while True:
                scheduler_next_block()
                scheduler_bind_block_args()
                if block_phase == T.int32(0):
                    break
                shape_k = T.Select(
                    block_phase == T.int32(2), T.int32(intermediate_hidden), T.int32(hidden)
                )
                shape_n = T.Select(
                    block_phase == T.int32(2), T.int32(hidden), T.int32(intermediate_hidden * 2)
                )
                shape_k_u32 = T.cast(shape_k, "uint32")
                shape_sfb_k = T.cast(
                    (shape_k_u32 + T.uint32(kernel_config.block_k - 1))
                    // T.uint32(kernel_config.block_k),
                    "int32",
                )
                for k_block_idx in T.serial(0, num_k_blocks):
                    barrier_wait(
                        smem_barriers.ptr_to([empty_barrier_base + pipeline_stage_idx]),
                        pipeline_phase ^ T.int32(1),
                    )
                    n_idx = local_expert_idx * shape_n + n_block_idx * kernel_config.block_n
                    k_idx = k_block_idx * kernel_config.block_k
                    sfb_n_idx = n_block_idx * kernel_config.block_n
                    sfb_k_idx = local_expert_idx * shape_sfb_k + k_block_idx
                    if T.ptx.elect_sync():
                        full_barrier_ptr = smem_barriers.ptr_to(
                            [full_barrier_base + pipeline_stage_idx]
                        )
                        tma_copy_2d_multicast_select(
                            smem_b.ptr_to([pipeline_stage_idx, 0, 0]),
                            full_barrier_ptr,
                            tensor_map_l1_weights,
                            tensor_map_l2_weights,
                            block_phase,
                            k_idx,
                            n_idx,
                        )
                        tma_copy_2d_multicast_select(
                            smem_sfb_i32.ptr_to([pipeline_stage_idx, 0, 0]),
                            full_barrier_ptr,
                            tensor_map_l1_weights_sf,
                            tensor_map_l2_weights_sf,
                            block_phase,
                            sfb_n_idx,
                            sfb_k_idx,
                        )
                        if cta_idx_in_cluster == 0:
                            full_barrier_arrive_and_expect_tx(
                                full_barrier_ptr, full_b_expect_tx_leader_bytes
                            )
                        else:
                            full_barrier_arrive_cta0(full_barrier_ptr)
                    T.cuda.warp_sync()
                    advance_pipeline()
                scheduler_iter_idx = scheduler_iter_idx + 1
        elif flat_warp_idx == kernel_config.mma_issue_warp_idx:
            warpgroup_reg_dealloc(num_non_epilogue_registers)
            if cta_idx_in_cluster == 0:
                make_instr_desc_block_scaled()
                make_sf_desc()
                make_umma_desc_a()
                make_umma_desc_b()
                a_desc_lo = T.Select(
                    lane_idx < T.int32(num_stages),
                    T.cast(T.bitwise_and(desc_a, T.uint64(0xFFFFFFFF)), "uint32")
                    + T.cast(lane_idx * (smem_a_size_per_stage // f128_bytes), "uint32"),
                    T.uint32(0),
                )
                b_desc_lo = T.Select(
                    lane_idx < T.int32(num_stages),
                    T.cast(T.bitwise_and(desc_b, T.uint64(0xFFFFFFFF)), "uint32")
                    + T.cast(lane_idx * (smem_b_size_per_stage // f128_bytes), "uint32"),
                    T.uint32(0),
                )
                scheduler_init()
                current_iter_idx = 0
                pipeline_stage_idx = T.int32(0)
                pipeline_phase = T.int32(0)
                while True:
                    scheduler_next_block()
                    scheduler_bind_block_args()
                    if block_phase == T.int32(0):
                        break
                    current_iter_idx_u32 = T.cast(current_iter_idx, "uint32")
                    accum_stage_idx = T.cast(
                        current_iter_idx_u32 % T.uint32(num_epilogue_stages), "int32"
                    )
                    accum_phase = T.cast(
                        T.bitwise_and(
                            current_iter_idx_u32 // T.uint32(num_epilogue_stages), T.uint32(1)
                        ),
                        "int32",
                    )
                    current_iter_idx = current_iter_idx + T.int32(1)
                    update_get_valid_m_true()
                    desc_i = T.bitwise_or(
                        T.bitwise_and(desc_i, T.uint32(0xFF81FFFF)),
                        T.shift_left(T.cast(get_valid_m_true_eighth, "uint32"), T.uint32(17)),
                    )
                    barrier_wait(
                        smem_barriers.ptr_to([tmem_empty_barrier_base + accum_stage_idx]),
                        accum_phase ^ T.int32(1),
                    )
                    for k_block_idx in T.serial(0, num_k_blocks):
                        full_wait_phase: T.int32 = pipeline_phase
                        mbarrier_wait_phase(
                            smem_barriers.ptr_to([full_barrier_base + pipeline_stage_idx]),
                            full_wait_phase,
                        )
                        a_desc_base_lo = T.tvm_warp_shuffle(
                            T.uint32(0xFFFFFFFF), a_desc_lo, pipeline_stage_idx, 32, 32
                        )
                        b_desc_base_lo = T.tvm_warp_shuffle(
                            T.uint32(0xFFFFFFFF), b_desc_lo, pipeline_stage_idx, 32, 32
                        )
                        if T.ptx.elect_sync():
                            for sfa_chunk_idx in T.unroll(0, num_sfa_utccp_chunks):
                                desc_sf = replace_smem_desc_addr(
                                    desc_sf,
                                    smem_sfa.ptr_to([pipeline_stage_idx, sfa_chunk_idx * 128]),
                                )
                                utccp_copy(sfa_tmem.allocated_addr[0] + sfa_chunk_idx * 4, desc_sf)
                            for sfb_chunk_idx in T.unroll(0, num_sfb_utccp_chunks):
                                desc_sf = replace_smem_desc_addr(
                                    desc_sf,
                                    smem_sfb.ptr_to([pipeline_stage_idx, sfb_chunk_idx * 128]),
                                )
                                utccp_copy(sfb_tmem.allocated_addr[0] + sfb_chunk_idx * 4, desc_sf)
                            for k_idx in T.unroll(0, kernel_config.block_k // umma_k):
                                runtime_desc_i = make_runtime_instr_desc_with_sf_id(
                                    desc_i, k_idx, k_idx
                                )
                                desc_a = advance_umma_desc_lo(
                                    desc_a, a_desc_base_lo, T.int32(0), k_idx * umma_k
                                )
                                desc_b = advance_umma_desc_lo(
                                    desc_b, b_desc_base_lo, T.int32(0), k_idx * umma_k
                                )
                                T.ptx.tcgen05.mma.block_scale(
                                    accum_stage_idx * umma_n,
                                    desc_b,
                                    desc_a,
                                    sfb_tmem.allocated_addr[0],
                                    sfa_tmem.allocated_addr[0],
                                    runtime_desc_i,
                                    d_dtype="float32",
                                    a_dtype="float4_e2m1fn",
                                    b_dtype="float8_e4m3fn",
                                    sfa_dtype="float8_e8m0fnu",
                                    sfb_dtype="float8_e8m0fnu",
                                    use_a_tmem=False,
                                    cta_group=kernel_config.num_ctas_per_cluster,
                                    enable_input_d=T.Or(
                                        k_block_idx > T.int32(0), k_idx > T.int32(0)
                                    ),
                                )
                        T.cuda.warp_sync()
                        empty_barrier_arrive_current(k_block_idx == num_k_blocks - T.int32(1))
                        advance_pipeline()
                if current_iter_idx > 0:
                    previous_iter_idx_u32 = T.cast(current_iter_idx - T.int32(1), "uint32")
                    accum_phase = T.cast(
                        T.bitwise_and(
                            previous_iter_idx_u32 // T.uint32(num_epilogue_stages), T.uint32(1)
                        ),
                        "int32",
                    )
                    barrier_wait(
                        smem_barriers.ptr_to(
                            [
                                tmem_empty_barrier_base
                                + T.cast(
                                    previous_iter_idx_u32 % T.uint32(num_epilogue_stages), "int32"
                                )
                            ]
                        ),
                        accum_phase,
                    )
        elif is_reserved_non_epilogue_warp:
            warpgroup_reg_dealloc(num_non_epilogue_registers)
            pass

        elif T.cast(flat_warp_idx, "uint32") >= T.uint32(kernel_config.epilogue_warp_start_idx):
            warpgroup_reg_alloc(num_epilogue_registers)
            swiglu_values = T.alloc_local((num_atoms_per_store * 2, 2), "float32")
            amax_values = T.alloc_local((num_atoms_per_store, 2), "float32")
            values = T.alloc_local((8,), "uint32")
            epilogue_fp8_packed = T.alloc_local((1,), "uint32")
            weights = T.alloc_local((2,), "float32")
            wp_amax = T.alloc_local((2,), "float32")
            sf = T.alloc_local((2,), "float32")
            sf_inv = T.alloc_local((2,), "float32")
            epilogue_bf16_packed = T.alloc_local((4,), "uint32")
            tmem_addr = T.alloc_local((1,), "uint32")
            reduced = T.alloc_local((num_uint4_per_lane * num_elems_per_uint4, 2), "float32")
            T.cuda.trap_when_assert_failed(tmem_ptr_in_smem[0] == T.uint32(0))
            epilogue_warp_idx = flat_warp_idx - kernel_config.epilogue_warp_start_idx
            epilogue_warp_idx_u32 = T.cast(epilogue_warp_idx, "uint32")
            epilogue_wg_idx = T.cast(epilogue_warp_idx_u32 // T.uint32(4), "int32")
            warp_idx_in_wg = T.cast(epilogue_warp_idx_u32 % T.uint32(4), "int32")
            T.evaluate(
                sync_unaligned(
                    dispatch_with_epilogue_sync_barrier_idx,
                    kernel_config.num_dispatch_threads + kernel_config.num_epilogue_threads,
                )
            )
            scheduler_init()
            current_iter_idx = 0
            while True:
                scheduler_next_block()
                scheduler_bind_block_args()
                if block_phase == T.int32(0):
                    break
                current_iter_idx_u32 = T.cast(current_iter_idx, "uint32")
                accum_stage_idx = T.cast(
                    current_iter_idx_u32 % T.uint32(num_epilogue_stages), "int32"
                )
                accum_phase = T.cast(
                    T.bitwise_and(
                        current_iter_idx_u32 // T.uint32(num_epilogue_stages), T.uint32(1)
                    ),
                    "int32",
                )
                current_iter_idx = current_iter_idx + T.int32(1)
                barrier_wait(
                    smem_barriers.ptr_to([tmem_full_barrier_base + accum_stage_idx]), accum_phase
                )
                pull_pool_block_offset = pool_block_idx
                pull_pool_token_idx = pool_block_idx * kernel_config.block_m
                valid_rows_in_wg = T.max(
                    T.min(valid_m - epilogue_wg_idx * wg_block_m, wg_block_m), T.int32(0)
                )
                if block_phase == T.int32(1):
                    m_idx = pull_pool_token_idx
                    n_idx = n_block_idx * l1_out_block_n
                    # Declared outside the `for s` loop so the per-32-rows weight cache persists
                    # across store iters. When wg_block_m is not a multiple of 32 (e.g., 48 for
                    # block_m=96) the load gate `(j*atom_m) % 32 == 0` fires at a different
                    # rhythm than the s loop, so resetting per-s would leave the cache zero on
                    # iterations between loads.
                    stored_cached_weight = T.float32(0.0)
                    for s in T.serial(0, wg_block_m // kernel_config.store_block_m, unroll=True):
                        if s * kernel_config.store_block_m >= valid_rows_in_wg:
                            tmem_empty_barrier_arrive_cta0(
                                smem_barriers.ptr_to([tmem_empty_barrier_base + accum_stage_idx])
                            )
                            break
                        for i in T.unroll(0, num_atoms_per_store):
                            j = s * num_atoms_per_store + i
                            if (j * atom_m) % 32 == 0:
                                # Lanes whose row falls past wg_block_m must skip the load — the
                                # warp-shuffle source lanes below are always < wg_block_m, so leaving
                                # OOB lanes' stored_cached_weight stale is fine. (Matches upstream's
                                # runtime guard for non-32-aligned wg_block_m.)
                                if wg_block_m % 32 == 0:
                                    l1_topk_weight_ptr = l1_topk_weights.ptr_to(
                                        [
                                            m_idx
                                            + epilogue_wg_idx * wg_block_m
                                            + j * atom_m
                                            + lane_idx
                                        ]
                                    )
                                    stored_cached_weight = load_f32(l1_topk_weight_ptr)
                                else:
                                    if T.cast(j * atom_m + lane_idx, "uint32") < T.uint32(
                                        wg_block_m
                                    ):
                                        l1_topk_weight_ptr = l1_topk_weights.ptr_to(
                                            [
                                                m_idx
                                                + epilogue_wg_idx * wg_block_m
                                                + j * atom_m
                                                + lane_idx
                                            ]
                                        )
                                        stored_cached_weight = load_f32(l1_topk_weight_ptr)
                            weights[0] = T.tvm_warp_shuffle(
                                T.uint32(0xFFFFFFFF),
                                stored_cached_weight,
                                (j * atom_m) % 32 + (lane_idx % 4) * 2,
                                32,
                                32,
                            )
                            weights[1] = T.tvm_warp_shuffle(
                                T.uint32(0xFFFFFFFF),
                                stored_cached_weight,
                                (j * atom_m) % 32 + (lane_idx % 4) * 2 + 1,
                                32,
                                32,
                            )
                            tmem_addr[0] = T.cast(
                                accum_stage_idx * umma_n
                                + epilogue_wg_idx * wg_block_m
                                + j * atom_m,
                                "uint32",
                            )
                            T.ptx.tcgen05.ld(
                                tmem_addr[0],
                                values[0],
                                values[1],
                                values[2],
                                values[3],
                                shape="16x256b",
                                num=1,
                            )
                            T.ptx.tcgen05.ld(
                                T.bitwise_or(tmem_addr[0], T.uint32(0x00100000)),
                                values[4],
                                values[5],
                                values[6],
                                values[7],
                                shape="16x256b",
                                num=1,
                            )
                            fence_view_async_tmem_load()
                            if j == wg_block_m // atom_m - 1:
                                tmem_empty_barrier_arrive_cta0(
                                    smem_barriers.ptr_to(
                                        [tmem_empty_barrier_base + accum_stage_idx]
                                    )
                                )
                            for k in T.unroll(0, 2):
                                swiglu_pair_store(
                                    swiglu_values,
                                    i * 2 + k,
                                    uint32_bits_to_float(values[k * 4]),
                                    uint32_bits_to_float(values[k * 4 + 1]),
                                    uint32_bits_to_float(values[k * 4 + 2]),
                                    uint32_bits_to_float(values[k * 4 + 3]),
                                    weights[0],
                                    weights[1],
                                )
                            amax_values[i, 0] = warp_reduce_max_4(
                                T.max(
                                    T.max(swiglu_values[i * 2, 0], -swiglu_values[i * 2, 0]),
                                    T.max(
                                        swiglu_values[i * 2 + 1, 0], -swiglu_values[i * 2 + 1, 0]
                                    ),
                                )
                            )
                            amax_values[i, 1] = warp_reduce_max_4(
                                T.max(
                                    T.max(swiglu_values[i * 2, 1], -swiglu_values[i * 2, 1]),
                                    T.max(
                                        swiglu_values[i * 2 + 1, 1], -swiglu_values[i * 2 + 1, 1]
                                    ),
                                )
                            )
                            if lane_idx < 4:
                                amax_reduction_idx = (
                                    epilogue_warp_idx * (kernel_config.store_block_m // 2)
                                    + i * (atom_m // 2)
                                    + lane_idx
                                ) * 2
                                smem_amax_reduction[amax_reduction_idx] = amax_values[i, 0]
                                smem_amax_reduction[amax_reduction_idx + 1] = amax_values[i, 1]
                            T.cuda.warp_sync()
                        tma_stage_idx = s % num_tma_store_stages
                        T.evaluate(tma_store_wait(1))
                        T.ptx.bar.sync(epilogue_wg_sync_barrier_start_idx + epilogue_wg_idx, 128)
                        for i in T.unroll(0, num_atoms_per_store):
                            j = s * num_atoms_per_store + i
                            amax_reduction_idx = (
                                (epilogue_warp_idx ^ 1) * (kernel_config.store_block_m // 2)
                                + i * (atom_m // 2)
                                + (lane_idx % 4)
                            ) * 2
                            wp_amax[0] = smem_amax_reduction[amax_reduction_idx]
                            wp_amax[1] = smem_amax_reduction[amax_reduction_idx + 1]
                            amax_values[i, 0] = T.max(amax_values[i, 0], wp_amax[0])
                            amax_values[i, 1] = T.max(amax_values[i, 1], wp_amax[1])
                            sf_x, sf_y, sf_inv_x, sf_inv_y = get_e4m3_sf_and_sf_inv(
                                amax_values[i, 0], amax_values[i, 1]
                            )
                            sf[0] = sf_x
                            sf[1] = sf_y
                            sf_inv[0] = sf_inv_x
                            sf_inv[1] = sf_inv_y
                            epilogue_fp8_packed[0] = scale_pack_fp8x4_e4m3(
                                swiglu_values[i * 2, 0],
                                swiglu_values[i * 2, 1],
                                swiglu_values[i * 2 + 1, 0],
                                swiglu_values[i * 2 + 1, 1],
                                sf_inv[0],
                                sf_inv[1],
                            )
                            row = lane_idx
                            col = warp_idx_in_wg
                            smem_ptr = smem_cd_l1.ptr_to(
                                [
                                    tma_stage_idx,
                                    epilogue_wg_idx,
                                    i * atom_m + row,
                                    (col ^ (row // 2)) * num_bank_group_bytes,
                                ]
                            )
                            sm100_u8x4_stsm_t_copy(epilogue_fp8_packed.ptr_to([0]), smem_ptr)
                            if warp_idx_in_wg % 2 == 0 and lane_idx < 4:
                                # Factored form of upstream 891d57b: token_base_idx is < BLOCK_M so
                                # `m_block_idx * BLOCK_M` factors out as `m_block_idx * SF_BLOCK_M`
                                # past `transform_sf_token_idx` (which is bitwise-independent in
                                # that range). `lane_idx * 2` only touches bits 0..2 of the input
                                # (token_base_idx is a multiple of atom_m=8), so its contribution
                                # collapses to a constant `lane_idx * 8` (= `(lane_idx*2) << 2`).
                                # Eliminates one mul + the residual modulo work in the original
                                # composed form.
                                token_base_idx = (
                                    epilogue_wg_idx * wg_block_m
                                    + s * kernel_config.store_block_m
                                    + i * atom_m
                                )
                                sf_pool_token_idx = (
                                    scheduler_current_pool_block_offset * sf_block_m
                                    + m_block_idx * sf_block_m
                                    + transform_sf_token_idx(token_base_idx)
                                    + lane_idx * T.int32(8)
                                )
                                mn_stride = workspace_layout.num_padded_sf_pool_tokens * 4
                                k_idx = n_block_idx * 2 + warp_idx_in_wg // 2
                                k_uint_idx = k_idx // 4
                                byte_idx = k_idx % 4
                                sf_addr = (
                                    k_uint_idx * mn_stride
                                    + sf_pool_token_idx * T.int32(4)
                                    + byte_idx
                                )
                                sf_bits = float_bits(sf[0])
                                sf_bits_hi = float_bits(sf[1])
                                l2_sf_buffer[sf_addr] = T.cast(
                                    T.shift_right(sf_bits, T.uint32(23)), "int8"
                                )
                                l2_sf_buffer[sf_addr + T.int32(16)] = T.cast(
                                    T.shift_right(sf_bits_hi, T.uint32(23)), "int8"
                                )
                        T.cuda.warp_sync()
                        T.ptx.bar.sync(epilogue_wg_sync_barrier_start_idx + epilogue_wg_idx, 128)
                        if (warp_idx_in_wg == 0) & T.ptx.elect_sync() != 0:
                            T.evaluate(tma_store_fence())
                            sm90_tma_store_2d_copy(
                                smem_cd_l1.ptr_to([tma_stage_idx, epilogue_wg_idx, 0, 0]),
                                tensor_map_l1_output,
                                n_idx,
                                m_idx
                                + epilogue_wg_idx * wg_block_m
                                + s * kernel_config.store_block_m,
                            )
                            T.evaluate(tma_store_arrive())
                        T.cuda.warp_sync()
                    T.evaluate(tma_store_wait(0))
                    T.ptx.bar.sync(
                        epilogue_full_sync_barrier_idx, kernel_config.num_epilogue_threads
                    )
                    if (epilogue_warp_idx == 0) & T.ptx.elect_sync() != 0:
                        expected_l2_mask = T.shift_left(T.uint64(1), T.cast(n_block_idx, "uint64"))
                        red_or_rel_gpu(
                            workspace_l2_arrival_mask.ptr_to([pool_block_idx]), expected_l2_mask
                        )
                    T.cuda.warp_sync()
                else:
                    n_idx = n_block_idx * kernel_config.block_n
                    for s in T.serial(0, wg_block_m // kernel_config.store_block_m, unroll=True):
                        if s * kernel_config.store_block_m >= valid_rows_in_wg:
                            tmem_empty_barrier_arrive_cta0(
                                smem_barriers.ptr_to([tmem_empty_barrier_base + accum_stage_idx])
                            )
                            break
                        for i in T.unroll(0, num_atoms_per_store):
                            j = s * num_atoms_per_store + i
                            tmem_addr[0] = T.cast(
                                accum_stage_idx * umma_n
                                + epilogue_wg_idx * wg_block_m
                                + j * atom_m,
                                "uint32",
                            )
                            T.ptx.tcgen05.ld(
                                tmem_addr[0],
                                values[0],
                                values[1],
                                values[2],
                                values[3],
                                shape="16x256b",
                                num=1,
                            )
                            T.ptx.tcgen05.ld(
                                T.bitwise_or(tmem_addr[0], T.uint32(0x00100000)),
                                values[4],
                                values[5],
                                values[6],
                                values[7],
                                shape="16x256b",
                                num=1,
                            )
                            fence_view_async_tmem_load()
                            if i == 0 and s > 0:
                                T.ptx.bar.sync(
                                    epilogue_wg_sync_barrier_start_idx + epilogue_wg_idx, 128
                                )
                            if (
                                s == wg_block_m // kernel_config.store_block_m - 1
                                and i == kernel_config.store_block_m // atom_m - 1
                            ):
                                tmem_empty_barrier_arrive_cta0(
                                    smem_barriers.ptr_to(
                                        [tmem_empty_barrier_base + accum_stage_idx]
                                    )
                                )
                            epilogue_bf16_packed[0] = cast_into_bf16_and_pack(
                                uint32_bits_to_float(values[0]), uint32_bits_to_float(values[1])
                            )
                            epilogue_bf16_packed[1] = cast_into_bf16_and_pack(
                                uint32_bits_to_float(values[2]), uint32_bits_to_float(values[3])
                            )
                            epilogue_bf16_packed[2] = cast_into_bf16_and_pack(
                                uint32_bits_to_float(values[4]), uint32_bits_to_float(values[5])
                            )
                            epilogue_bf16_packed[3] = cast_into_bf16_and_pack(
                                uint32_bits_to_float(values[6]), uint32_bits_to_float(values[7])
                            )
                            row = lane_idx % 8
                            col = (epilogue_warp_idx % 2) * 4 + lane_idx // 8
                            smem_ptr = smem.ptr_to(
                                [
                                    smem_cd_offset
                                    + epilogue_wg_idx
                                    * kernel_config.store_block_m
                                    * kernel_config.block_n
                                    * 2
                                    + (warp_idx_in_wg // 2)
                                    * kernel_config.store_block_m
                                    * swizzle_cd_mode
                                    + i * atom_m * swizzle_cd_mode
                                    + row * (num_bank_group_bytes * 8)
                                    + (col ^ row) * num_bank_group_bytes
                                ]
                            )
                            sm90_u32x4_stsm_t_copy(epilogue_bf16_packed, smem_ptr)
                        T.ptx.bar.sync(epilogue_wg_sync_barrier_start_idx + epilogue_wg_idx, 128)
                        row_in_atom = (warp_idx_in_wg * 2 + lane_idx // 16) % atom_m
                        bank_group_idx = lane_idx % 8
                        lane_col_offset = (lane_idx % 16) * 8
                        for j in T.unroll(0, num_rows_per_warp):
                            row_in_store = j * 8 + warp_idx_in_wg * 2 + lane_idx // 16
                            m_idx_in_block = (
                                epilogue_wg_idx * wg_block_m
                                + s * kernel_config.store_block_m
                                + row_in_store
                            )
                            if T.cast(m_idx_in_block, "uint32") < T.cast(valid_m, "uint32"):
                                src_metadata_idx = pull_pool_token_idx + m_idx_in_block
                                dst_rank_idx_u32 = workspace_token_src_metadata[src_metadata_idx, 0]
                                dst_token_idx_u32 = workspace_token_src_metadata[
                                    src_metadata_idx, 1
                                ]
                                dst_topk_idx_u32 = workspace_token_src_metadata[src_metadata_idx, 2]
                                dst_rank_idx = T.cast(dst_rank_idx_u32, "int32")
                                dst_token_base_offset = T.cast(
                                    dst_topk_idx_u32, "uint64"
                                ) * T.uint64(
                                    workspace_layout.num_max_tokens_per_rank * hidden * 2
                                ) + T.cast(dst_token_idx_u32, "uint64") * T.uint64(hidden * 2)
                                dst_col_byte_offset = T.cast(n_idx * 2, "uint64")
                                lane_byte_offset = T.cast((lane_idx % 16) * 16, "uint64")
                                dst_ptr = (
                                    dst_token_base_offset + dst_col_byte_offset + lane_byte_offset
                                )
                                smem_ptr = smem.ptr_to(
                                    [
                                        smem_cd_offset
                                        + epilogue_wg_idx
                                        * kernel_config.store_block_m
                                        * kernel_config.block_n
                                        * 2
                                        + ((lane_idx % 16) // 8)
                                        * kernel_config.store_block_m
                                        * swizzle_cd_mode
                                        + row_in_store * swizzle_cd_mode
                                        + (bank_group_idx ^ row_in_atom) * num_bank_group_bytes
                                    ]
                                )
                                lds128(smem_ptr, epilogue_bf16_packed.ptr_to([0]))
                                dst_peer_base = symm_rank_base_expr(
                                    sym_buffer_base, symm_rank_offsets, dst_rank_idx
                                )
                                dst_ptr = (
                                    T.cast(symm_buffer_layout.combine_token_offset, "uint64")
                                    + dst_ptr
                                )
                                stg128_symm(
                                    dst_peer_base,
                                    dst_ptr,
                                    epilogue_bf16_packed[0],
                                    epilogue_bf16_packed[1],
                                    epilogue_bf16_packed[2],
                                    epilogue_bf16_packed[3],
                                )
                    T.ptx.bar.sync(
                        epilogue_full_sync_barrier_idx, kernel_config.num_epilogue_threads
                    )
            if epilogue_warp_idx == 0:
                T.ptx.tcgen05.dealloc(
                    T.uint32(0), n_cols=num_tmem_cols, cta_group=kernel_config.num_ctas_per_cluster
                )
            epilogue_thread_idx = epilogue_warp_idx * 32 + lane_idx
            epilogue_nvlink_barrier_before_combine_reduce(epilogue_thread_idx)
            T.evaluate(
                sync_unaligned(
                    dispatch_with_epilogue_sync_barrier_idx,
                    kernel_config.num_dispatch_threads + kernel_config.num_epilogue_threads,
                )
            )
            token_idx = sm_idx * kernel_config.num_epilogue_warps + epilogue_warp_idx
            combine_phase = T.int32(0)
            load_stage_idx = T.int32(0)
            while T.cast(token_idx, "uint32") < T.cast(num_tokens, "uint32"):
                stored_topk_slot_idx = T.int32(-1)
                if lane_idx < num_topk:
                    stored_topk_slot_idx = T.cast(input_topk_idx[token_idx, lane_idx], "int32")
                total_mask = ballot_sync(T.uint32(0xFFFFFFFF), stored_topk_slot_idx >= T.int32(0))
                for chunk in T.unroll(0, num_chunks):
                    mask = total_mask
                    chunk_byte_offset = chunk * num_chunk_bytes
                    chunk_offset_elems = chunk_byte_offset // 2
                    for reduced_idx in T.unroll(0, num_uint4_per_lane * num_elems_per_uint4):
                        reduced[reduced_idx, 0] = T.float32(0.0)
                        reduced[reduced_idx, 1] = T.float32(0.0)
                    do_reduce = T.int32(0)
                    if mask != T.uint32(0):
                        slot_idx = ffs_u32(mask) - T.int32(1)
                        mask = T.bitwise_xor(
                            mask, T.shift_left(T.uint32(1), T.cast(slot_idx, "uint32"))
                        )
                        if T.ptx.elect_sync():
                            src_ptr = combine_tokens.ptr_to(
                                [slot_idx, token_idx, chunk_offset_elems]
                            )
                            load_barrier_ptr = smem_barriers.ptr_to(
                                [combine_barrier_base + epilogue_warp_idx * 2 + load_stage_idx]
                            )
                            load_buffer_ptr = combine_chunks.ptr_to(
                                [load_stage_idx, epilogue_warp_idx, 0, 0]
                            )
                            tma_load_1d(load_buffer_ptr, src_ptr, load_barrier_ptr, num_chunk_bytes)
                            mbarrier_arrive_and_set_tx(load_barrier_ptr, num_chunk_bytes)
                        do_reduce = T.int32(1)
                    T.cuda.warp_sync()
                    while do_reduce != T.int32(0):
                        next_do_reduce = T.int32(0)
                        if mask != T.uint32(0):
                            slot_idx = ffs_u32(mask) - T.int32(1)
                            mask = T.bitwise_xor(
                                mask, T.shift_left(T.uint32(1), T.cast(slot_idx, "uint32"))
                            )
                            if T.ptx.elect_sync():
                                src_ptr = combine_tokens.ptr_to(
                                    [slot_idx, token_idx, chunk_offset_elems]
                                )
                                load_barrier_ptr = smem_barriers.ptr_to(
                                    [
                                        combine_barrier_base
                                        + epilogue_warp_idx * 2
                                        + (load_stage_idx ^ T.int32(1))
                                    ]
                                )
                                prefetch_buffer_ptr = combine_chunks.ptr_to(
                                    [load_stage_idx ^ T.int32(1), epilogue_warp_idx, 0, 0]
                                )
                                tma_load_1d(
                                    prefetch_buffer_ptr, src_ptr, load_barrier_ptr, num_chunk_bytes
                                )
                                mbarrier_arrive_and_set_tx(load_barrier_ptr, num_chunk_bytes)
                            next_do_reduce = T.int32(1)
                        T.cuda.warp_sync()
                        mbarrier_wait_phase(
                            smem_barriers.ptr_to(
                                [combine_barrier_base + epilogue_warp_idx * 2 + load_stage_idx]
                            ),
                            combine_phase,
                        )
                        for j in T.unroll(0, num_uint4_per_lane):
                            load_ptr = combine_chunks.ptr_to(
                                [load_stage_idx, epilogue_warp_idx, j * 32 + lane_idx, 0]
                            )
                            lds128(load_ptr, epilogue_bf16_packed.ptr_to([0]))
                            for elem_idx in T.unroll(0, num_elems_per_uint4):
                                reduced[j * num_elems_per_uint4 + elem_idx, 0] = (
                                    T.ptx.add_rn_f32_bf16(
                                        reduced[j * num_elems_per_uint4 + elem_idx, 0],
                                        bf16x2_lo(epilogue_bf16_packed[elem_idx]),
                                    )
                                )
                                reduced[j * num_elems_per_uint4 + elem_idx, 1] = (
                                    T.ptx.add_rn_f32_bf16(
                                        reduced[j * num_elems_per_uint4 + elem_idx, 1],
                                        bf16x2_hi(epilogue_bf16_packed[elem_idx]),
                                    )
                                )
                        combine_phase = combine_phase ^ load_stage_idx
                        load_stage_idx = load_stage_idx ^ T.int32(1)
                        do_reduce = next_do_reduce
                    for j in T.unroll(0, num_uint4_per_lane):
                        for elem_idx in T.unroll(0, num_elems_per_uint4):
                            epilogue_bf16_packed[elem_idx] = cast_into_bf16_and_pack(
                                reduced[j * num_elems_per_uint4 + elem_idx, 0],
                                reduced[j * num_elems_per_uint4 + elem_idx, 1],
                            )
                        if j == 0:
                            T.evaluate(tma_store_wait(0))
                            T.cuda.warp_sync()
                        combine_store_ptr = combine_chunks.ptr_to(
                            [2, epilogue_warp_idx, j * 32 + lane_idx, 0]
                        )
                        sts128(
                            combine_store_ptr,
                            epilogue_bf16_packed[0],
                            epilogue_bf16_packed[1],
                            epilogue_bf16_packed[2],
                            epilogue_bf16_packed[3],
                        )
                    T.cuda.warp_sync()
                    if T.ptx.elect_sync():
                        T.evaluate(tma_store_fence())
                        dst_ptr = T.address_of(y[token_idx, chunk_offset_elems])
                        combine_store_ptr = combine_chunks.ptr_to([2, epilogue_warp_idx, 0, 0])
                        tma_store_1d(dst_ptr, combine_store_ptr, num_chunk_bytes)
                        T.evaluate(tma_store_arrive())
                T.cuda.warp_sync()
                token_idx = token_idx + kernel_config.num_sms * kernel_config.num_epilogue_warps

    return mega_moe.with_attr("tirx.kernel_launch_params", get_tirx_launch_param_tags())


def _get_tirx_kernel_for_context(case: MegaMoeCase | TirxMegaMoeLaunchContext) -> Any:
    return get_kernel(
        num_processes=case.config.num_processes,
        num_max_tokens_per_rank=case.config.num_max_tokens_per_rank,
        num_tokens=case.config.num_tokens,
        hidden=case.config.hidden,
        intermediate_hidden=case.config.intermediate_hidden,
        num_experts=case.config.num_experts,
        num_topk=case.config.num_topk,
        activation_clamp=case.config.activation_clamp,
        fast_math=case.config.fast_math,
    )


def _get_tirx_kernel_for_case(case: MegaMoeCase) -> Any:
    return _get_tirx_kernel_for_context(case)


def _view_symm_matrix(
    case: MegaMoeCase | TirxMegaMoeLaunchContext, offset: int, rows: int, cols: int
) -> torch.Tensor:
    return case.symm_buffer.buffer.narrow(0, offset, rows * cols).view(rows, cols)


@cache
def _compile_tirx_mega_moe_for_config(
    *,
    num_processes: int,
    num_max_tokens_per_rank: int,
    num_tokens: int,
    hidden: int,
    intermediate_hidden: int,
    num_experts: int,
    num_topk: int,
    activation_clamp: float,
    fast_math: int,
    emit_nvl_barrier_timeout_printf: bool = True,
) -> Any:
    import tvm

    kernel = get_kernel(
        num_processes=num_processes,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_tokens=num_tokens,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        num_experts=num_experts,
        num_topk=num_topk,
        activation_clamp=activation_clamp,
        fast_math=fast_math,
        emit_nvl_barrier_timeout_printf=emit_nvl_barrier_timeout_printf,
    )
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    mod = tvm.IRModule({"main": kernel})
    return tvm.compile(mod, target=target, tir_pipeline="tirx")


def _compile_tirx_mega_moe(case: MegaMoeCase | TirxMegaMoeLaunchContext) -> Any:
    config = case.config
    return _compile_tirx_mega_moe_for_config(
        num_processes=config.num_processes,
        num_max_tokens_per_rank=config.num_max_tokens_per_rank,
        num_tokens=config.num_tokens,
        hidden=config.hidden,
        intermediate_hidden=config.intermediate_hidden,
        num_experts=config.num_experts,
        num_topk=config.num_topk,
        activation_clamp=config.activation_clamp,
        fast_math=config.fast_math,
    )


def _require_mega_moe_tuple(name: str, value: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be a 2-tuple of tensors")
    first, second = value
    if not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor):
        raise TypeError(f"{name} must contain tensors")
    return first, second


def _make_symm_buffer_offsets(case: MegaMoeCase | TirxMegaMoeLaunchContext) -> tuple[int, ...]:
    buffer_ptrs = tuple(int(ptr) for ptr in case.symm_buffer.handle.buffer_ptrs)
    if len(buffer_ptrs) > DEEPGEMM_SYM_BUFFER_MAX_RANKS:
        raise ValueError(
            "TIRx MegaMoE supports at most "
            f"{DEEPGEMM_SYM_BUFFER_MAX_RANKS} symmetric-buffer ranks, got {len(buffer_ptrs)}"
        )
    local_base = int(case.symm_buffer.buffer.data_ptr())
    expected_local_base = buffer_ptrs[case.rank_idx]
    if local_base != expected_local_base:
        raise ValueError(
            "sym_buffer.buffer data_ptr does not match handle.buffer_ptrs[rank_idx]: "
            f"{local_base} != {expected_local_base}"
        )
    offsets = tuple(ptr - local_base for ptr in buffer_ptrs)
    return offsets + (0,) * (DEEPGEMM_SYM_BUFFER_MAX_RANKS - len(offsets))


def _make_tirx_mega_moe_launch_context(
    *,
    y: torch.Tensor,
    l1_weights: tuple[torch.Tensor, torch.Tensor],
    l2_weights: tuple[torch.Tensor, torch.Tensor],
    sym_buffer: Any,
    recipe: tuple[int, int, int],
    activation: str,
    activation_clamp: float | None,
    fast_math: bool,
) -> TirxMegaMoeLaunchContext:
    if tuple(recipe) != (1, 1, 32):
        raise NotImplementedError("TIRx MegaMoE currently supports recipe=(1, 1, 32) only")
    if activation != "swiglu":
        raise NotImplementedError("TIRx MegaMoE currently supports activation='swiglu' only")
    if y.dtype != torch.bfloat16:
        raise TypeError(f"y must have dtype torch.bfloat16, got {y.dtype}")
    if not y.is_cuda:
        raise ValueError("y must be a CUDA tensor")
    if not y.is_contiguous():
        raise ValueError("y must be contiguous")
    if y.dim() != 2:
        raise ValueError(f"y must be 2D, got shape {tuple(y.shape)}")

    l1_weights = _require_mega_moe_tuple("l1_weights", l1_weights)
    l2_weights = _require_mega_moe_tuple("l2_weights", l2_weights)
    for tensor_name, tensor in (
        ("l1_weights[0]", l1_weights[0]),
        ("l1_weights[1]", l1_weights[1]),
        ("l2_weights[0]", l2_weights[0]),
        ("l2_weights[1]", l2_weights[1]),
    ):
        if not tensor.is_cuda:
            raise ValueError(f"{tensor_name} must be a CUDA tensor")
    num_ranks = int(sym_buffer.group.size())
    rank_idx = int(sym_buffer.group.rank())
    config = MegaMoeConfig(
        num_processes=num_ranks,
        num_max_tokens_per_rank=int(sym_buffer.num_max_tokens_per_rank),
        num_tokens=int(y.shape[0]),
        hidden=int(y.shape[1]),
        intermediate_hidden=int(sym_buffer.intermediate_hidden),
        num_experts=int(sym_buffer.num_experts),
        num_topk=int(sym_buffer.num_topk),
        activation_clamp=math.inf if activation_clamp is None else float(activation_clamp),
        fast_math=int(bool(fast_math)),
    )
    config.validate()
    if int(sym_buffer.hidden) != config.hidden:
        raise ValueError(
            f"y hidden dimension {config.hidden} does not match sym_buffer.hidden {sym_buffer.hidden}"
        )

    workspace_layout = get_deepgemm_workspace_layout(config)
    symm_buffer_layout = get_deepgemm_symm_buffer_layout(config)
    validate_runtime_symm_buffer_layout(
        symm_buffer=sym_buffer, layout=symm_buffer_layout, config=config
    )
    return TirxMegaMoeLaunchContext(
        config=config,
        rank_idx=rank_idx,
        num_ranks=num_ranks,
        symm_buffer=sym_buffer,
        transformed_l1_weights=l1_weights,
        transformed_l2_weights=l2_weights,
        workspace_layout=workspace_layout,
        symm_buffer_layout=symm_buffer_layout,
    )


def _prepare_tirx_invocation(
    case: MegaMoeCase | TirxMegaMoeLaunchContext, y: torch.Tensor | None = None
) -> TirxMegaMoeInvocation:
    l1_weights = case.transformed_l1_weights[0]
    l1_weights_sf = case.transformed_l1_weights[1].permute(0, 2, 1)
    l2_weights = case.transformed_l2_weights[0]
    l2_weights_sf = case.transformed_l2_weights[1].permute(0, 2, 1)
    symm_layout = case.symm_buffer_layout
    l1_acts = _view_symm_matrix(
        case,
        symm_layout.l1_token_offset,
        case.workspace_layout.num_max_pool_tokens,
        case.config.hidden,
    )
    l1_acts_sf = case.symm_buffer.l1_acts_sf.transpose(0, 1)
    l2_acts = _view_symm_matrix(
        case,
        symm_layout.l2_token_offset,
        case.workspace_layout.num_max_pool_tokens,
        case.config.intermediate_hidden,
    )
    l2_acts_sf = case.symm_buffer.l2_acts_sf.transpose(0, 1)
    tensor_maps = _build_tirx_tensor_maps(
        case=case,
        l1_acts=l1_acts,
        l2_acts=l2_acts,
        l1_weights=l1_weights,
        l1_weights_sf=l1_weights_sf,
        l2_weights=l2_weights,
        l2_weights_sf=l2_weights_sf,
    )
    if y is None:
        y = torch.empty(
            (case.config.num_tokens, case.config.hidden), dtype=torch.bfloat16, device="cuda"
        )
    return TirxMegaMoeInvocation(
        executable=_compile_tirx_mega_moe(case),
        y=y,
        symm_buffer_offsets=_make_symm_buffer_offsets(case),
        tensor_maps=tensor_maps,
    )


def _prepare_global_barrier(executable: Any) -> None:
    try:
        prepare_global_barrier = executable.mod.get_function("__tvm_prepare_global_barrier")
    except AttributeError:
        prepare_global_barrier = None
    if prepare_global_barrier is not None:
        prepare_global_barrier()


def _launch_tirx_mega_moe(
    case: MegaMoeCase | TirxMegaMoeLaunchContext, invocation: TirxMegaMoeInvocation
) -> None:
    tensor_maps = invocation.tensor_maps
    _prepare_global_barrier(invocation.executable)
    invocation.executable.mod(
        invocation.y,
        case.symm_buffer.buffer,
        *invocation.symm_buffer_offsets,
        tensor_maps["tensor_map_l1_acts"].ptr,
        tensor_maps["tensor_map_l1_acts_sf"].ptr,
        tensor_maps["tensor_map_l1_weights"].ptr,
        tensor_maps["tensor_map_l1_weights_sf"].ptr,
        tensor_maps["tensor_map_l1_output"].ptr,
        tensor_maps["tensor_map_l2_acts"].ptr,
        tensor_maps["tensor_map_l2_acts_sf"].ptr,
        tensor_maps["tensor_map_l2_weights"].ptr,
        tensor_maps["tensor_map_l2_weights_sf"].ptr,
        case.config.num_tokens,
        case.rank_idx,
    )


def run_tirx_mega_moe(case: MegaMoeCase) -> torch.Tensor:
    _copy_inputs_into_symm_buffer(case)
    y = torch.empty(
        (case.config.num_tokens, case.config.hidden), dtype=torch.bfloat16, device="cuda"
    )
    fp8_fp4_mega_moe(
        y,
        case.transformed_l1_weights,
        case.transformed_l2_weights,
        case.symm_buffer,
        activation_clamp=case.config.activation_clamp,
        fast_math=bool(case.config.fast_math),
    )
    return y


def prepare_tirx_fp8_fp4_mega_moe(
    y: torch.Tensor,
    l1_weights: tuple[torch.Tensor, torch.Tensor],
    l2_weights: tuple[torch.Tensor, torch.Tensor],
    sym_buffer: Any,
    recipe: tuple[int, int, int] = (1, 1, 32),
    activation: str = "swiglu",
    activation_clamp: float | None = None,
    fast_math: bool = True,
) -> TirxMegaMoePrepared:
    context = _make_tirx_mega_moe_launch_context(
        y=y,
        l1_weights=l1_weights,
        l2_weights=l2_weights,
        sym_buffer=sym_buffer,
        recipe=recipe,
        activation=activation,
        activation_clamp=activation_clamp,
        fast_math=fast_math,
    )
    return TirxMegaMoePrepared(context=context, invocation=_prepare_tirx_invocation(context, y=y))


def launch_prepared_tirx_fp8_fp4_mega_moe(prepared: TirxMegaMoePrepared) -> None:
    _launch_tirx_mega_moe(prepared.context, prepared.invocation)


def fp8_fp4_mega_moe(
    y: torch.Tensor,
    l1_weights: tuple[torch.Tensor, torch.Tensor],
    l2_weights: tuple[torch.Tensor, torch.Tensor],
    sym_buffer: Any,
    recipe: tuple[int, int, int] = (1, 1, 32),
    activation: str = "swiglu",
    activation_clamp: float | None = None,
    fast_math: bool = True,
) -> None:
    prepared = prepare_tirx_fp8_fp4_mega_moe(
        y,
        l1_weights,
        l2_weights,
        sym_buffer,
        recipe=recipe,
        activation=activation,
        activation_clamp=activation_clamp,
        fast_math=fast_math,
    )
    launch_prepared_tirx_fp8_fp4_mega_moe(prepared)


fp8_fp4_mega_moe_tirx = fp8_fp4_mega_moe


def _cleanup_case(case: MegaMoeCase | None) -> None:
    if case is not None:
        case.symm_buffer.destroy()


def _cleanup_distinct_cases(*cases: MegaMoeCase | None) -> None:
    destroyed: set[int] = set()
    for case in cases:
        if case is None:
            continue
        key = id(case.symm_buffer)
        if key in destroyed:
            continue
        case.symm_buffer.destroy()
        destroyed.add(key)


def _destroy_process_group() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _run_worker(local_rank: int, cfg_dict: dict[str, Any], mode: str) -> dict[str, Any]:
    worker_kwargs = dict(cfg_dict)
    warmup = int(worker_kwargs.pop("warmup", 10))
    repeat = int(worker_kwargs.pop("repeat", 30))
    timer = str(worker_kwargs.pop("timer", "proton"))
    rounds = int(worker_kwargs.pop("rounds", 1))
    round_cooldown_s = float(worker_kwargs.pop("round_cooldown_s", 1.0))
    config = MegaMoeConfig(**worker_kwargs)
    config.validate()

    if config.num_processes > torch.cuda.device_count():
        raise SkipTest(
            f"Requested {config.num_processes} processes, but only "
            f"{torch.cuda.device_count()} CUDA devices are visible"
        )

    deep_gemm, source = load_deep_gemm_mega()
    case = None
    dg_case = None
    tirx_case = None
    default_device_before = torch.get_default_device()
    cuda_device_before = (
        torch.cuda.current_device()
        if torch.cuda.is_available() and torch.cuda.is_initialized()
        else None
    )
    try:
        if (
            hasattr(torch.distributed, "destroy_process_group")
            and torch.distributed.is_initialized()
        ):
            _destroy_process_group()
        rank_idx, num_ranks, group = deep_gemm.utils.dist.init_dist(
            local_rank, config.num_processes
        )

        if mode == "test":
            case = create_case(deep_gemm, config, group, rank_idx, num_ranks)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            y_ref = run_deepgemm_reference(case)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            checksum = float(y_ref.float().sum().item())
            y_math = None
            naive_reference_error = None
            if config.num_processes == 1:
                try:
                    y_math = run_naive_reference(case)
                except RuntimeError as exc:
                    if _is_optional_math_reference_error(exc):
                        naive_reference_error = str(exc)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        raise
            try:
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                y_tir = run_tirx_mega_moe(case)
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
            except NotImplementedError as exc:
                return {
                    "status": "SKIP",
                    "reason": str(exc),
                    "reference_source": source,
                    "reference_checksum": checksum,
                    "num_tokens": config.num_tokens,
                }
            deepgemm_max_abs_diff = _max_abs_diff(y_tir, y_ref)
            max_abs_diff = (
                _max_abs_diff(y_tir, y_math) if y_math is not None else deepgemm_max_abs_diff
            )
            return {
                "status": "OK",
                "reference_source": source,
                "reference_checksum": checksum,
                "max_abs_diff": max_abs_diff,
                "deepgemm_max_abs_diff": deepgemm_max_abs_diff,
                "naive_reference_error": naive_reference_error,
            }

        if mode == "bench":
            from tvm.tirx.bench import bench, tensor_bytes

            dg_case = create_case(deep_gemm, config, group, rank_idx, num_ranks)
            tirx_case = create_case(deep_gemm, config, group, rank_idx, num_ranks)
            _copy_inputs_into_symm_buffer(dg_case)
            _copy_inputs_into_symm_buffer(tirx_case)
            y_deepgemm = torch.empty(
                (config.num_tokens, config.hidden), dtype=torch.bfloat16, device="cuda"
            )
            tirx_invocation = _prepare_tirx_invocation(tirx_case)

            def deepgemm_step() -> None:
                dg_case.deep_gemm.fp8_fp4_mega_moe(
                    y_deepgemm,
                    dg_case.transformed_l1_weights,
                    dg_case.transformed_l2_weights,
                    dg_case.symm_buffer,
                    activation_clamp=dg_case.config.activation_clamp,
                    fast_math=bool(dg_case.config.fast_math),
                )

            def tirx_step() -> None:
                _launch_tirx_mega_moe(tirx_case, tirx_invocation)

            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            deepgemm_step()
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            tirx_step()
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            deepgemm_max_abs_diff = _max_abs_diff(tirx_invocation.y, y_deepgemm)
            if deepgemm_max_abs_diff != 0.0:
                raise AssertionError(f"TIRx diff={deepgemm_max_abs_diff}")

            def make_input():
                dg_case = create_case(deep_gemm, config, group, rank_idx, num_ranks)
                tirx_case = create_case(deep_gemm, config, group, rank_idx, num_ranks)
                _copy_inputs_into_symm_buffer(dg_case)
                _copy_inputs_into_symm_buffer(tirx_case)
                y_deepgemm = torch.empty(
                    (config.num_tokens, config.hidden), dtype=torch.bfloat16, device="cuda"
                )
                tirx_invocation = _prepare_tirx_invocation(tirx_case)
                return (dg_case, tirx_case, y_deepgemm, tirx_invocation), tensor_bytes(
                    dg_case.symm_buffer,
                    dg_case.transformed_l1_weights,
                    dg_case.transformed_l2_weights,
                    y_deepgemm,
                    tirx_invocation.y,
                )

            def run_deepgemm(case) -> None:
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                dg_case, _, y_deepgemm, _ = case
                dg_case.deep_gemm.fp8_fp4_mega_moe(
                    y_deepgemm,
                    dg_case.transformed_l1_weights,
                    dg_case.transformed_l2_weights,
                    dg_case.symm_buffer,
                    activation_clamp=dg_case.config.activation_clamp,
                    fast_math=bool(dg_case.config.fast_math),
                )

            def run_tirx(case) -> None:
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                _, tirx_case, _, tirx_invocation = case
                _launch_tirx_mega_moe(tirx_case, tirx_invocation)

            def _deepgemm():
                return run_deepgemm

            session_name = f"deepgemm_mega_moe_rank{rank_idx}_{os.getpid()}_{time.time_ns()}"
            bench_result = bench(
                {"tirx": run_tirx},
                make_input,
                warmup=warmup,
                repeat=repeat,
                timer=timer,
                proton_name=session_name,
                references={"deepgemm": _deepgemm},
                rounds=rounds,
                round_cooldown_s=round_cooldown_s,
            )
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            impls = bench_result["impls"]
            missing = {"deepgemm", "tirx"} - set(impls)
            if missing:
                raise RuntimeError(f"Proton did not report timings for: {sorted(missing)}")
            return {
                "status": "OK",
                "reference_source": source,
                "deepgemm_max_abs_diff": deepgemm_max_abs_diff,
                "impls": {"deepgemm": float(impls["deepgemm"]), "tirx": float(impls["tirx"])},
                "errors": bench_result["errors"],
            }

        raise ValueError(f"Unsupported mode: {mode}")
    finally:
        try:
            _cleanup_distinct_cases(case, dg_case, tirx_case)
            _destroy_process_group()
        finally:
            torch.set_default_device(default_device_before)
            if cuda_device_before is not None:
                torch.cuda.set_device(cuda_device_before)


def _worker_entry(
    local_rank: int, cfg_dict: dict[str, Any], mode: str, result_queue: mp.SimpleQueue | None
) -> None:
    result = _run_worker(local_rank, cfg_dict, mode)
    if result_queue is not None:
        result_queue.put((local_rank, result))


def _aggregate_rank_results(rank_results: list[tuple[int, dict[str, Any]]]) -> dict[str, Any]:
    rank_results = sorted(rank_results, key=lambda item: item[0])
    results = [result for _, result in rank_results]
    for result in results:
        if result["status"] == "SKIP":
            return result
    first = results[0]
    if "impls" in first:
        impl_names = sorted({name for result in results for name in result.get("impls", {})})
        impls = {
            name: max(float(result["impls"][name]) for result in results if name in result["impls"])
            for name in impl_names
        }
        errors = {}
        for result in results:
            errors.update(result.get("errors", {}))
        return {
            **first,
            "impls": impls,
            "errors": errors,
            "deepgemm_max_abs_diff": max(
                float(result.get("deepgemm_max_abs_diff", 0.0)) for result in results
            ),
            "rank_results": [
                {
                    "rank": rank,
                    "impls": result.get("impls", {}),
                    "deepgemm_max_abs_diff": float(result.get("deepgemm_max_abs_diff", 0.0)),
                }
                for rank, result in rank_results
            ],
        }
    if "deepgemm_max_abs_diff" not in first:
        return first
    return {
        **first,
        "max_abs_diff": max(float(result["max_abs_diff"]) for result in results),
        "deepgemm_max_abs_diff": max(float(result["deepgemm_max_abs_diff"]) for result in results),
        "rank_results": [
            {
                "rank": rank,
                "max_abs_diff": float(result["max_abs_diff"]),
                "deepgemm_max_abs_diff": float(result["deepgemm_max_abs_diff"]),
                "reference_checksum": float(result["reference_checksum"]),
            }
            for rank, result in rank_results
        ],
    }


def _run_distributed(config: MegaMoeConfig, mode: str, **kwargs) -> dict[str, Any]:
    cfg_dict = {**asdict(config), **kwargs}
    if config.num_processes > torch.cuda.device_count():
        raise SkipTest(
            f"Requested {config.num_processes} processes, but only "
            f"{torch.cuda.device_count()} CUDA devices are visible"
        )
    if config.num_processes == 1:
        last_error = None
        for _ in range(32):
            port = _find_free_port()
            try:
                with _distributed_env(port):
                    return _run_worker(0, cfg_dict, mode)
            except Exception as exc:
                message = str(exc)
                if "EADDRINUSE" not in message and "address already in use" not in message:
                    raise
                last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError(
            "Unable to allocate a free TCP port for single-process distributed init."
        )

    port = _find_free_port()
    with _distributed_env(port):
        ctx = mp.get_context("spawn")
        result_queue = ctx.SimpleQueue()
        mp.spawn(
            _worker_entry,
            args=(cfg_dict, mode, result_queue),
            nprocs=config.num_processes,
            join=True,
        )
        rank_results = [result_queue.get() for _ in range(config.num_processes)]
        return _aggregate_rank_results(rank_results)


KERNEL_META = {
    "name": "deepgemm_fp8_fp4_mega_moe",
    "category": "deepgemm",
    "compute_capability": 10,
}

# One case per block_m bucket in `_get_block_config_for_mega_moe` so a per-PR
# sm100a run covers all heuristic-selected block_m paths (16, 32, 64, 96, 128, 192).
# Each tpe (= tokens * ranks * topk / experts) is set just below the bucket boundary.
CONFIGS = [
    {
        "num_processes": 1,
        "num_max_tokens_per_rank": 4,
        "num_tokens": 2,
        "hidden": 1024,
        "intermediate_hidden": 512,
        "num_experts": 2,
        "num_topk": 1,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": "p1_tok2_h1024_i512_e2_k1_bm16",
    },
    {
        "num_processes": 1,
        "num_max_tokens_per_rank": 16,
        "num_tokens": 16,
        "hidden": 1024,
        "intermediate_hidden": 512,
        "num_experts": 2,
        "num_topk": 2,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": "p1_tok16_h1024_i512_e2_k2_bm32",
    },
    {
        "num_processes": 1,
        "num_max_tokens_per_rank": 32,
        "num_tokens": 32,
        "hidden": 1024,
        "intermediate_hidden": 512,
        "num_experts": 2,
        "num_topk": 2,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": "p1_tok32_h1024_i512_e2_k2_bm64",
    },
    {
        "num_processes": 1,
        "num_max_tokens_per_rank": 64,
        "num_tokens": 64,
        "hidden": 1024,
        "intermediate_hidden": 512,
        "num_experts": 2,
        "num_topk": 2,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": "p1_tok64_h1024_i512_e2_k2_bm96",
    },
    {
        "num_processes": 1,
        "num_max_tokens_per_rank": 96,
        "num_tokens": 96,
        "hidden": 1024,
        "intermediate_hidden": 512,
        "num_experts": 2,
        "num_topk": 2,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": "p1_tok96_h1024_i512_e2_k2_bm128",
    },
    {
        "num_processes": 1,
        "num_max_tokens_per_rank": 192,
        "num_tokens": 192,
        "hidden": 1024,
        "intermediate_hidden": 512,
        "num_experts": 2,
        "num_topk": 2,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": "p1_tok192_h1024_i512_e2_k2_bm192",
    },
]


BENCH_CONFIGS = [
    {
        "num_processes": 1,
        "num_max_tokens_per_rank": 8,
        "num_tokens": 8,
        "hidden": 1024,
        "intermediate_hidden": 512,
        "num_experts": 24,
        "num_topk": 2,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": "t8_h1024_i512_e24_k2_g1",
    },
    {
        "num_processes": 1,
        "num_max_tokens_per_rank": 16,
        "num_tokens": 16,
        "hidden": 2048,
        "intermediate_hidden": 1024,
        "num_experts": 48,
        "num_topk": 2,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": "t16_h2048_i1024_e48_k2_g1",
    },
    {
        "num_processes": 1,
        "num_max_tokens_per_rank": 64,
        "num_tokens": 64,
        "hidden": 4096,
        "intermediate_hidden": 1536,
        "num_experts": 96,
        "num_topk": 4,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": "t64_h4096_i1536_e96_k4_g1",
    },
    {
        "num_processes": 1,
        "num_max_tokens_per_rank": 64,
        "num_tokens": 32,
        "hidden": 7168,
        "intermediate_hidden": 3072,
        "num_experts": 384,
        "num_topk": 6,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": "t32_m64_h7168_i3072_e384_k6_g1",
    },
    {
        "num_processes": 1,
        "num_max_tokens_per_rank": 8192,
        "num_tokens": 8192,
        "hidden": 7168,
        "intermediate_hidden": 3072,
        "num_experts": 64,
        "num_topk": 6,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": "t8192_h7168_i3072_e64_k6_g1",
    },
    {
        "num_processes": 1,
        "num_max_tokens_per_rank": 8192,
        "num_tokens": 8192,
        "hidden": 7168,
        "intermediate_hidden": 3072,
        "num_experts": 192,
        "num_topk": 6,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": "t8192_h7168_i3072_e192_k6_g1",
    },
    {
        "num_processes": 1,
        "num_max_tokens_per_rank": 8192,
        "num_tokens": 8192,
        "hidden": 7168,
        "intermediate_hidden": 3072,
        "num_experts": 384,
        "num_topk": 6,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": "t8192_h7168_i3072_e384_k6_g1",
    },
    {
        "num_processes": 4,
        "num_max_tokens_per_rank": 256,
        "num_tokens": 256,
        "hidden": 7168,
        "intermediate_hidden": 3072,
        "num_experts": 384,
        "num_topk": 6,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": "t256_h7168_i3072_e384_k6_g4",
    },
    {
        "num_processes": 4,
        "num_max_tokens_per_rank": 1024,
        "num_tokens": 1024,
        "hidden": 7168,
        "intermediate_hidden": 3072,
        "num_experts": 384,
        "num_topk": 6,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": "t1024_h7168_i3072_e384_k6_g4",
    },
    {
        "num_processes": 6,
        "num_max_tokens_per_rank": 8192,
        "num_tokens": 8192,
        "hidden": 7168,
        "intermediate_hidden": 3072,
        "num_experts": 384,
        "num_topk": 6,
        "activation_clamp": 10.0,
        "fast_math": 1,
        "label": "t8192_h7168_i3072_e384_k6_g6",
    },
]


def _make_config(
    num_processes=1,
    num_max_tokens_per_rank=4,
    num_tokens=2,
    hidden=1024,
    intermediate_hidden=512,
    num_experts=2,
    num_topk=1,
    activation_clamp=10.0,
    fast_math=1,
) -> MegaMoeConfig:
    config = MegaMoeConfig(
        num_processes=num_processes,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_tokens=num_tokens,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        num_experts=num_experts,
        num_topk=num_topk,
        activation_clamp=activation_clamp,
        fast_math=fast_math,
    )
    config.validate()
    return config


def prepare_data(
    num_processes=1,
    num_max_tokens_per_rank=4,
    num_tokens=2,
    hidden=1024,
    intermediate_hidden=512,
    num_experts=2,
    num_topk=1,
    activation_clamp=10.0,
    fast_math=1,
) -> dict[str, Any]:
    return {
        "config": _make_config(
            num_processes=num_processes,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            num_tokens=num_tokens,
            hidden=hidden,
            intermediate_hidden=intermediate_hidden,
            num_experts=num_experts,
            num_topk=num_topk,
            activation_clamp=activation_clamp,
            fast_math=fast_math,
        )
    }


def _assert_correctness_result(result: dict[str, Any]) -> None:
    if result["status"] == "SKIP":
        raise SkipTest(
            f"{result['reason']} DeepGEMM reference source={result['reference_source']} "
            f"checksum={result['reference_checksum']:.4f}"
        )
    assert result["deepgemm_max_abs_diff"] == 0.0, (
        f"Expected bitwise parity with DeepGEMM reference, got deepgemm_max_abs_diff="
        f"{result['deepgemm_max_abs_diff']} (naive_max_abs_diff={result['max_abs_diff']})"
    )


def check_correctness(
    outputs: dict[str, Any],
    num_processes=1,
    num_max_tokens_per_rank=4,
    num_tokens=2,
    hidden=1024,
    intermediate_hidden=512,
    num_experts=2,
    num_topk=1,
    activation_clamp=10.0,
    fast_math=1,
) -> None:
    result = outputs.get("result")
    if result is None:
        result = _run_distributed(
            _make_config(
                num_processes=num_processes,
                num_max_tokens_per_rank=num_max_tokens_per_rank,
                num_tokens=num_tokens,
                hidden=hidden,
                intermediate_hidden=intermediate_hidden,
                num_experts=num_experts,
                num_topk=num_topk,
                activation_clamp=activation_clamp,
                fast_math=fast_math,
            ),
            "test",
        )
    _assert_correctness_result(result)


def run_test(
    num_processes=1,
    num_max_tokens_per_rank=4,
    num_tokens=2,
    hidden=1024,
    intermediate_hidden=512,
    num_experts=2,
    num_topk=1,
    activation_clamp=10.0,
    fast_math=1,
):
    config = _make_config(
        num_processes=num_processes,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_tokens=num_tokens,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        num_experts=num_experts,
        num_topk=num_topk,
        activation_clamp=activation_clamp,
        fast_math=fast_math,
    )
    result = _run_distributed(config, "test")
    _assert_correctness_result(result)


def run_bench(
    num_processes=1,
    num_max_tokens_per_rank=128,
    num_tokens=96,
    hidden=1024,
    intermediate_hidden=512,
    num_experts=8,
    num_topk=2,
    activation_clamp=10.0,
    fast_math=1,
    *,
    warmup=10,
    repeat=30,
    timer="proton",
    **kwargs,
):
    config = _make_config(
        num_processes=num_processes,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_tokens=num_tokens,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        num_experts=num_experts,
        num_topk=num_topk,
        activation_clamp=activation_clamp,
        fast_math=fast_math,
    )
    return _run_distributed(config, "bench", warmup=warmup, repeat=repeat, timer=timer, **kwargs)

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

"""MOE-only support helpers for the event-dependency megakernel benchmark."""

from typing import Literal

import numpy as np
import tvm_ffi

import tvm
from tirx_kernels.megakernel.kernels import GroupGEMMTileSM100
from tirx_kernels.megakernel.utils.config import JobType, KernelConfig
from tirx_kernels.megakernel.utils.dynamic_scheduler import DynamicTileScheduler, MPMCQueueHost
from tirx_kernels.megakernel.utils.static_scheduler import StaticTileScheduler
from tirx_kernels.megakernel.utils.utils import ceildiv, pack_into_32bit
from tvm.script import tirx as T


def push_moe_tasks(
    central_queue: list[tuple[int, int, int, int]],
    batch_size: int,
    config: dict,
    insert_wait_etensor_init: bool = False,
):
    """Append the static task sequence for the fused MOE megakernel."""
    moe_blk_m = 128
    gating_blk_m = 128
    for m_idx in range(ceildiv(batch_size, gating_blk_m)):
        for k_idx in range(config["GATING_SPLIT_K_FACTOR"]):
            central_queue.append((m_idx, 0, k_idx, JobType.MOE_GATING.value))
    if insert_wait_etensor_init:
        for i in range(KernelConfig.SM_NUMBER):
            central_queue.append((i, 0, 0, JobType.WAIT_ETENSOR_INIT.value))
    for m_idx in range(KernelConfig.SM_NUMBER):
        central_queue.append((m_idx, 0, 0, JobType.MOE_TOPK_SOFTMAX.value))
    central_queue.append((0, 0, 0, JobType.MOE_ALIGN.value))
    for m_idx in range(KernelConfig.SM_NUMBER):
        central_queue.append((m_idx, 0, 0, JobType.MOE_COUNT_AND_SORT.value))

    max_num_tokens_padded = get_max_num_tokens_padded(
        batch_size, config["NUM_EXPERTS_PER_TOK"], config["NUM_EXPERTS"], moe_blk_m
    )
    for m_idx in range(max_num_tokens_padded // moe_blk_m):
        for n_idx in range(config["INTERMEDIATE_SIZE"] * 2 // GroupGEMMTileSM100.BLK_N):
            central_queue.append((m_idx, n_idx, 0, JobType.MOE_GROUP_GEMM_GATE_UP_SILU.value))
    for m_idx in range(max_num_tokens_padded // moe_blk_m):
        for n_idx in range(config["HIDDEN_SIZE"] // GroupGEMMTileSM100.BLK_N):
            central_queue.append((m_idx, n_idx, 0, JobType.MOE_GROUP_GEMM_DOWN.value))


@tvm_ffi.register_global_func("tirx.megakernel.get_max_num_tokens_padded")
def get_max_num_tokens_padded(batch_size, topk, num_experts, moe_blk_m):
    if isinstance(batch_size, int):
        if batch_size * topk < num_experts:
            return batch_size * topk * moe_blk_m
        return (num_experts + ceildiv(batch_size * topk - num_experts, moe_blk_m)) * moe_blk_m
    return T.if_then_else(
        batch_size * topk < num_experts,
        batch_size * topk * moe_blk_m,
        (num_experts + ceildiv(batch_size * topk - num_experts, moe_blk_m)) * moe_blk_m,
    )


def get_max_blocks_padded_relaxed(batch_size, topk, num_experts, moe_blk_m):
    return batch_size * topk // moe_blk_m + (num_experts + 1)


def generate_exec_queue_moe(
    batch_size: int, config: dict, etensor_num: int, scheduler: Literal["static", "dynamic"]
):
    if scheduler == "static":
        exec_queue = np.zeros(
            (KernelConfig.SM_NUMBER, StaticTileScheduler.MAX_TASKS), dtype=np.int32
        )
        central_queue = []
        for i in range(etensor_num):
            central_queue.append((i, 0, 0, JobType.INIT_ETENSOR.value))
        push_moe_tasks(central_queue, batch_size, config, insert_wait_etensor_init=True)

        tile_idx = 0
        while central_queue:
            for bx in range(KernelConfig.SM_NUMBER):
                if central_queue:
                    exec_queue[bx, tile_idx] = pack_into_32bit(*central_queue.pop(0))
                else:
                    exec_queue[bx, tile_idx] = pack_into_32bit(-1, -1, -1, JobType.END.value)
            tile_idx += 1
        for bx in range(KernelConfig.SM_NUMBER):
            exec_queue[bx, tile_idx] = pack_into_32bit(-1, -1, -1, JobType.END.value)
        return tvm.runtime.tensor(exec_queue, device=tvm.cuda(0))

    if scheduler == "dynamic":
        exec_queue = MPMCQueueHost(DynamicTileScheduler.MAX_TASKS)
        gating_blk_m = 128
        for i in range(etensor_num):
            exec_queue.enqueue(JobType.INIT_ETENSOR.value, i, 0, 0)
        for m in range(ceildiv(batch_size, gating_blk_m)):
            for k in range(config["GATING_SPLIT_K_FACTOR"]):
                exec_queue.enqueue(JobType.MOE_GATING.value, m, 0, k)
        return exec_queue

    raise ValueError(f"Unsupported scheduler: {scheduler}")

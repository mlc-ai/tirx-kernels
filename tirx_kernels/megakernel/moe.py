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

from unittest import SkipTest

import numpy as np
import torch

import tvm
from tirx_kernels.megakernel.kernels import (
    CountAndSortExpertTokens,
    GemmTile,
    GroupGEMMSiluTile,
    GroupGEMMTileSM100,
    MOEAlignTile,
    TopkSoftmaxTile,
)
from tirx_kernels.megakernel.utils import dynamic_scheduler, static_scheduler
from tirx_kernels.megakernel.utils.base import MegaKernelWrapper
from tirx_kernels.megakernel.utils.config import (
    MEGAKERNEL_MOE_BENCH_CONFIG,
    JobType,
    KernelConfig,
    ProfileEventType,
)
from tirx_kernels.megakernel.utils.dynamic_scheduler import DynamicTileScheduler
from tirx_kernels.megakernel.utils.static_scheduler import StaticTileScheduler
from tirx_kernels.megakernel.utils.support import (
    generate_exec_queue_moe,
    get_max_blocks_padded_relaxed,
    get_max_num_tokens_padded,
)
from tirx_kernels.megakernel.utils.utils import ceildiv, f_init_const, get_source
from tvm.script import tirx as Tx

# TODO: fix abnormal slowness of batch-attn on the first tile

KERNEL_META = {"name": "megakernel_moe", "category": "megakernel", "compute_capability": 10}

_SUPPORTED_SCHEDULERS = ("static", "dynamic", "unfused")
_STATIC_QUEUE_SCHEDULERS = ("static", "unfused")
_TEST_BATCH_SIZES = (1, 128)
_BENCH_BATCH_SIZES = (1, 8, 32, 128, 512, 1024, 2048, 4096)

CONFIGS = [
    {
        "label": f"moe_a3b_bs{batch_size}_{scheduler}",
        "batch_size": batch_size,
        "scheduler": scheduler,
        "world_size": 1,
    }
    for batch_size in _TEST_BATCH_SIZES
    for scheduler in _SUPPORTED_SCHEDULERS
]

BENCH_CONFIGS = [
    {"label": f"moe_a3b_bs{batch_size}_all", "batch_size": batch_size, "world_size": 1}
    for batch_size in _BENCH_BATCH_SIZES
]

_COMPILE_CACHE = {}


class MegaKernelMOE(MegaKernelWrapper):
    MOE_M_PAD_SIZE = 128
    GATING_BLK_M = 128

    def __init__(self, config, world_size, profiler_on):
        super().__init__(config, 1, profiler_on)
        self.world_size = world_size
        self.HIDDEN_SIZE = config.get("HIDDEN_SIZE", None)
        self.INTERMEDIATE_SIZE = config.get("INTERMEDIATE_SIZE", None)
        self.NUM_EXPERTS = config.get("NUM_EXPERTS", None)
        self.NUM_EXPERTS_PER_TOK = config.get("NUM_EXPERTS_PER_TOK", None)
        self.GATING_SPLIT_K_FACTOR = config.get("GATING_SPLIT_K_FACTOR", None)

    def _set_tiles(self, batch_size, low_batch):
        self.gate = self._add_tile(
            GemmTile(
                self.NUM_EXPERTS,
                self.HIDDEN_SIZE,
                "float16",
                "float16",
                self.GATING_SPLIT_K_FACTOR,
                self.GATING_BLK_M,
                self.GATING_BLK_M,
                use_tma_reduce=True,
            ),
            ProfileEventType.MOE_GATING,
        )
        self.topk_softmax = self._add_tile(
            TopkSoftmaxTile(
                self.NUM_EXPERTS, batch_size, self.NUM_EXPERTS_PER_TOK, dtype="float32"
            ),
            ProfileEventType.TOPK_SOFTMAX,
        )
        numel = self.NUM_EXPERTS_PER_TOK * batch_size
        self.align = self._add_tile(
            MOEAlignTile(self.NUM_EXPERTS, numel, self.MOE_M_PAD_SIZE, pad_sorted_token_ids=True),
            ProfileEventType.MOE_ALIGN,
        )
        self.count_and_sort_expert_tokens = self._add_tile(
            CountAndSortExpertTokens(numel, self.HIDDEN_SIZE, self.NUM_EXPERTS_PER_TOK),
            ProfileEventType.COUNT_AND_SORT,
        )
        self.group_gemm_gate_up_silu = self._add_tile(
            GroupGEMMSiluTile(
                self.INTERMEDIATE_SIZE * 2,
                self.HIDDEN_SIZE,
                self.NUM_EXPERTS,
                self.NUM_EXPERTS_PER_TOK,
                numel,
                "float16",
                "float16",
                low_batch=low_batch,
            ),
            ProfileEventType.GROUP_GEMM_GATE_UP_SILU,
        )
        self.group_gemm_down = self._add_tile(
            GroupGEMMTileSM100(
                self.HIDDEN_SIZE,
                self.INTERMEDIATE_SIZE,
                self.NUM_EXPERTS,
                self.NUM_EXPERTS_PER_TOK,
                numel,
                "float16",
                "float16",
                acc_output=True,
                low_batch=low_batch,
            ),
            ProfileEventType.GROUP_GEMM_DOWN,
        )

    def set_tiles(self, batch_size, low_batch):
        self.reset()
        self._set_tiles(batch_size, low_batch)

    def _set_events(
        self,
        batch_size,
        Semaphore: type[static_scheduler.Semaphore | dynamic_scheduler.Semaphore],
        etensor_workspace_global,
        unfused=False,
    ):
        self.evt_gating = self.add_etensor(
            Semaphore,
            etensor_workspace_global,
            shape=[1],
            f_init=f_init_const(
                self.GATING_SPLIT_K_FACTOR * ceildiv(batch_size, self.GATING_BLK_M)
            ),
        )
        self.evt_topk_softmax = self.add_etensor(
            Semaphore,
            etensor_workspace_global,
            shape=[1],
            f_init=f_init_const(KernelConfig.SM_NUMBER),
        )
        self.evt_moe_align = self.add_etensor(
            Semaphore, etensor_workspace_global, shape=[1], f_init=f_init_const(1)
        )
        self.evt_count_and_sort = self.add_etensor(
            Semaphore,
            etensor_workspace_global,
            shape=[1],
            f_init=f_init_const(KernelConfig.SM_NUMBER),
        )
        max_num_tokens_padded = get_max_num_tokens_padded(
            batch_size, self.NUM_EXPERTS_PER_TOK, self.NUM_EXPERTS, self.MOE_M_PAD_SIZE
        )
        max_blocks_padded_relaxed = get_max_blocks_padded_relaxed(
            batch_size, self.NUM_EXPERTS_PER_TOK, self.NUM_EXPERTS, self.MOE_M_PAD_SIZE
        )
        if unfused:
            self.evt_group_gemm_gate_up = self.add_etensor(
                Semaphore,
                etensor_workspace_global,
                shape=[1],
                f_init=f_init_const(
                    max_num_tokens_padded
                    // self.MOE_M_PAD_SIZE
                    * self.INTERMEDIATE_SIZE
                    * 2
                    // GroupGEMMTileSM100.BLK_N
                ),
            )
        else:
            self.evt_group_gemm_gate_up = self.add_etensor(
                Semaphore,
                etensor_workspace_global,
                shape=[max_blocks_padded_relaxed],
                f_init=f_init_const(self.INTERMEDIATE_SIZE * 2 // GroupGEMMTileSM100.BLK_N),
            )
        f_init_group_gemm_down = (
            f_init_const(
                max_num_tokens_padded
                // self.MOE_M_PAD_SIZE
                * self.HIDDEN_SIZE
                // GroupGEMMTileSM100.BLK_N
            )
            if issubclass(Semaphore, static_scheduler.Semaphore)
            else None
        )
        self.evt_group_gemm_down = self.add_etensor(
            Semaphore, etensor_workspace_global, shape=[1], f_init=f_init_group_gemm_down
        )

    def set_events(
        self,
        is_dynamic_sch,
        batch_size,
        Semaphore: type[static_scheduler.Semaphore | dynamic_scheduler.Semaphore],
        etensor_workspace_global,
        unfused=False,
    ):
        self._set_events(batch_size, Semaphore, etensor_workspace_global, unfused=unfused)
        self.set_events_complete(is_dynamic_sch, Semaphore, etensor_workspace_global)
        self.num_etensors[is_dynamic_sch] = len(self.etensor_and_f_init_pairs)

    def _add_tile(self, tile, profiler_event_type, predicate=True):
        self.tile_attr[tile] = (profiler_event_type, predicate)
        subclass = GroupGEMMTileSM100 if isinstance(tile, GemmTile) else tile.__class__
        self.class_list.add(subclass)
        return tile

    @Tx.inline
    def task_impl_moe_gating(self, A, B, output, is_dynamic_sch):
        if is_dynamic_sch:
            self.tile_scheduler.pre_notify_and_push(
                self.evt_gating,
                lambda notify_idx: (1, -1, 0),
                lambda trigger_idx: (
                    lambda push_idx: (
                        JobType.MOE_TOPK_SOFTMAX.value,
                        self.topk_softmax.PERSISTENT_SM_NUMBER,
                        push_idx,
                        0,
                        0,
                    )
                ),
                "warpgroup",
                "warpgroup",
                scope_id=0,
            )
        self.run_tile(
            self.gate,
            self.tile_scheduler.m_idx,
            self.tile_scheduler.n_idx,
            self.tile_scheduler.k_idx,
            A,
            B,
            output,
            self.profiler,
        )
        self.tile_scheduler.notify(
            self.evt_gating, lambda notify_idx: (1, -1, 0), scope="warpgroup", scope_id=0
        )

    @Tx.inline
    def task_impl_moe_topk_softmax(
        self,
        gating_output_global,
        topk_weights_global,
        topk_indices_global,
        is_dynamic_sch,
        renormalize=True,
    ):
        if is_dynamic_sch:
            self.tile_scheduler.pre_notify_and_push(
                self.evt_topk_softmax,
                lambda notify_idx: (1, -1, 0),
                lambda trigger_idx: lambda push_idx: (JobType.MOE_ALIGN.value, 1, 0, 0, 0),
                "thread",
                "thread",
            )
        self.tile_scheduler.wait(self.evt_gating, 0, wait_level="cta")
        self.run_tile(
            self.topk_softmax,
            self.tile_scheduler.m_idx,
            self.tile_scheduler.n_idx,
            self.tile_scheduler.k_idx,
            gating_output_global,
            topk_weights_global,
            topk_indices_global,
            renormalize=renormalize,
        )
        self.tile_scheduler.notify(
            self.evt_topk_softmax, lambda notify_idx: (1, -1, 0), scope="cta"
        )

    @Tx.inline
    def task_impl_moe_align(
        self,
        topk_ids_flattened,
        sorted_token_ids_global,
        expert_ids_global,
        num_tokens_post_pad_global,
        cumsum_buffer_global,
        num_valid_tokens_global,
        down_proj_task_size,
        is_dynamic_sch,
    ):
        tid = Tx.thread_id([KernelConfig.NUM_THREADS])
        if is_dynamic_sch:
            self.tile_scheduler.pre_notify_and_push(
                self.evt_moe_align,
                lambda notify_idx: (1, -1, 0),
                lambda trigger_idx: (
                    lambda push_idx: (
                        JobType.MOE_COUNT_AND_SORT.value,
                        KernelConfig.SM_NUMBER,
                        push_idx,
                        0,
                        0,
                    )
                ),
                "cta",
                "cta",
            )
        self.tile_scheduler.wait(self.evt_topk_softmax, 0, wait_level="cta")
        self.run_tile(
            self.align,
            self.tile_scheduler.m_idx,
            self.tile_scheduler.n_idx,
            self.tile_scheduler.k_idx,
            topk_ids_flattened,
            sorted_token_ids_global,
            expert_ids_global,
            num_tokens_post_pad_global,
            cumsum_buffer_global,
            num_valid_tokens_global,
        )
        Tx.cuda.cta_sync()
        if tid == 0:
            # TODO: make this etensor initialization a task
            if is_dynamic_sch:
                self.evt_group_gemm_down.sem[0] = (
                    (self.evt_group_gemm_down.base + 1)
                    * (num_tokens_post_pad_global[0] // self.MOE_M_PAD_SIZE)
                    * (self.HIDDEN_SIZE // GroupGEMMTileSM100.BLK_N // down_proj_task_size)
                )
        self.tile_scheduler.notify(
            self.evt_moe_align, lambda notify_idx: (1, -1, 0), scope="thread"
        )

    @Tx.inline
    def task_impl_moe_count_and_sort(
        self,
        topk_ids_flattened,
        sorted_token_ids_global,
        cumsum_buffer_global,
        hidden_state_global,
        reordered_hidden_state_global,
        num_tokens_post_pad_global,
        is_dynamic_sch,
    ):
        if is_dynamic_sch:
            n_axis_len = Tx.meta_var(self.INTERMEDIATE_SIZE * 2 // GroupGEMMTileSM100.BLK_N)
            self.tile_scheduler.pre_notify_and_push(
                self.evt_count_and_sort,
                lambda notify_idx: (1, -1, 0),
                lambda trigger_idx: (
                    lambda push_idx: (
                        JobType.MOE_GROUP_GEMM_GATE_UP_SILU.value,
                        num_tokens_post_pad_global[0] // self.MOE_M_PAD_SIZE * n_axis_len,
                        push_idx // n_axis_len,
                        push_idx % n_axis_len,
                        0,
                    )
                ),
                "cta",
                "cta",
            )
        self.tile_scheduler.wait(self.evt_moe_align, 0, wait_level="cta")
        self.run_tile(
            self.count_and_sort_expert_tokens,
            self.tile_scheduler.m_idx,
            self.tile_scheduler.n_idx,
            self.tile_scheduler.k_idx,
            topk_ids_flattened,
            sorted_token_ids_global,
            cumsum_buffer_global,
            hidden_state_global,
            reordered_hidden_state_global,
        )
        self.tile_scheduler.notify(
            self.evt_count_and_sort, lambda notify_idx: (1, -1, 0), scope="cta"
        )

    @Tx.inline
    def task_impl_moe_group_gemm_gate_up_silu(
        self,
        A,
        B,
        output,
        topk_weights_flattened,
        sorted_token_ids_global,
        expert_ids_global,
        num_valid_tokens_global,
        num_tokens_post_pad_global,
        unfused,
        down_proj_task_size,
        is_dynamic_sch,
    ):
        if is_dynamic_sch:
            self.tile_scheduler.pre_notify_and_push(
                self.evt_group_gemm_gate_up,
                lambda notify_idx: (1, -1, self.tile_scheduler.m_idx),
                lambda trigger_idx: (
                    lambda push_idx: (
                        JobType.MOE_GROUP_GEMM_DOWN.value,
                        self.HIDDEN_SIZE // GroupGEMMTileSM100.BLK_N // down_proj_task_size,
                        self.tile_scheduler.m_idx,
                        push_idx,
                        0,
                    )
                ),
                "warp",
                "warp",
            )
        self.tile_scheduler.wait(self.evt_count_and_sort, 0, wait_level="warp")
        if (
            is_dynamic_sch
            or self.tile_scheduler.m_idx < num_tokens_post_pad_global[0] // self.MOE_M_PAD_SIZE
        ):
            self.run_tile(
                self.group_gemm_gate_up_silu,
                self.tile_scheduler.m_idx,
                self.tile_scheduler.n_idx,
                self.tile_scheduler.k_idx,
                A,
                B,
                output,
                expert_ids_global,
                topk_weights_flattened,
                sorted_token_ids_global,
                num_valid_tokens_global,
                self.profiler,
            )
        idx = Tx.meta_var(self.tile_scheduler.m_idx if not unfused else 0)
        self.tile_scheduler.notify(
            self.evt_group_gemm_gate_up,
            lambda notify_idx: (1, -1, idx),
            scope="warpgroup",
            scope_id=0,
        )

    @Tx.inline
    def task_impl_moe_group_gemm_down(
        self,
        A,
        B,
        output,
        expert_ids_global,
        topk_weights_flattened,
        sorted_token_ids_global,
        num_valid_tokens_global,
        num_tokens_post_pad_global,
        unfused,
        down_proj_task_size,
        is_dynamic_sch,
    ):
        if is_dynamic_sch:
            self.tile_scheduler.pre_notify_and_push(
                self.evt_group_gemm_down,
                lambda notify_idx: (1, -1, 0),
                lambda trigger_idx: (
                    lambda push_idx: (JobType.END.value, KernelConfig.SM_NUMBER, 0, 0, 0)
                ),
                "warp",
                "warp",
            )
        wait_idx = Tx.meta_var(self.tile_scheduler.m_idx if not unfused else 0)
        self.tile_scheduler.wait(self.evt_group_gemm_gate_up, wait_idx, wait_level="warp")
        if (
            is_dynamic_sch
            or self.tile_scheduler.m_idx < num_tokens_post_pad_global[0] // self.MOE_M_PAD_SIZE
        ):
            for i in range(down_proj_task_size):
                self.run_tile(
                    self.group_gemm_down,
                    self.tile_scheduler.m_idx,
                    self.tile_scheduler.n_idx * down_proj_task_size + i,
                    self.tile_scheduler.k_idx,
                    A,
                    B,
                    output,
                    expert_ids_global,
                    topk_weights_flattened,
                    sorted_token_ids_global,
                    num_valid_tokens_global,
                    self.profiler,
                )

    # fmt: off
    @Tx.inline
    def fused_body(
        self,
        batch_size,
        hidden_state_global,
        residual_global,
        output_global,
        gate_weight_global,
        grp_gate_up_weight_global,
        grp_down_weight_global,
        gating_output_global,
        topk_weights_global,
        topk_indices_global,
        sorted_token_ids_global,
        expert_ids_global,
        num_valid_tokens_global,
        num_tokens_post_pad_global,
        cumsum_buffer_global,
        reordered_hidden_state_global,
        gate_up_output_global,
        silu_mul_output_global,
        topk_reduce_output_global,
        etensor_workspace_global,
        profiler_buffer,
        exec_queue,
        exec_task,
        exec_head,
        exec_tail,
        down_proj_task_size, # to amortize dynamic scheduling overhead
        low_batch,
        unfused,
        is_dynamic_sch,
        Semaphore: type[static_scheduler.Semaphore | dynamic_scheduler.Semaphore],
        Scheduler: type[static_scheduler.StaticTileScheduler | dynamic_scheduler.DynamicTileScheduler],
    ):
        # initialize tile
        self.set_tiles(batch_size, low_batch)
        self.host_init_all()

        Tx.device_entry()
        cta_id = Tx.cta_id([KernelConfig.SM_NUMBER])
        warp_id = Tx.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER])
        wg_id = Tx.warpgroup_id([KernelConfig.WG_NUMBER])
        tid = Tx.thread_id([KernelConfig.NUM_THREADS])
        tid_in_wg = Tx.thread_id_in_wg([128])
        lane_id = Tx.lane_id([32])
        Tx.alloc_buffer([1], "uint32", scope="local", align=8)
        Tx.alloc_buffer([1], "uint64", scope="local", align=8)
        self.init_profiler(profiler_buffer)
        buf = Tx.alloc_buffer([KernelConfig.MAX_SMEM_SIZE], "uint8", scope="shared.dyn")
        # initialize smem manager
        self.set_smem_manager(KernelConfig.MAX_SMEM_SIZE, 16384, buf.data)

        # initialize device
        self.device_init_all(self.smem_manager)
        self.class_init_all(self.smem_manager)

        # initialize event tensors
        self.set_events(
            is_dynamic_sch,
            batch_size,
            Semaphore,
            etensor_workspace_global,
            unfused,
        )

        # initialize tile scheduler and smem_manager
        if not is_dynamic_sch:
            self.init_tile_scheduler(False, Scheduler, "moe", exec_queue, self.smem_manager)
        else:
            self.init_tile_scheduler(True, Scheduler, exec_task, exec_head, exec_tail, self.smem_manager, self.profiler)
        self.smem_manager.init()

        topk_ids_flattened = topk_indices_global.view(-1)
        topk_weights_flattened = topk_weights_global.view(-1)
        while self.tile_scheduler.valid():
            if self.tile_scheduler.task_type == JobType.MOE_GATING.value:
                self.task_impl_moe_gating(hidden_state_global, gate_weight_global, gating_output_global, is_dynamic_sch)
            elif self.tile_scheduler.task_type == JobType.MOE_TOPK_SOFTMAX.value:
                self.task_impl_moe_topk_softmax(gating_output_global, topk_weights_global, topk_indices_global, is_dynamic_sch, renormalize=False)
            elif self.tile_scheduler.task_type == JobType.MOE_ALIGN.value:
                self.task_impl_moe_align(topk_ids_flattened, sorted_token_ids_global, expert_ids_global, num_tokens_post_pad_global, cumsum_buffer_global, num_valid_tokens_global, down_proj_task_size, is_dynamic_sch)
            elif self.tile_scheduler.task_type == JobType.MOE_COUNT_AND_SORT.value:
                self.task_impl_moe_count_and_sort(topk_ids_flattened, sorted_token_ids_global, cumsum_buffer_global, hidden_state_global, reordered_hidden_state_global, num_tokens_post_pad_global, is_dynamic_sch)
            elif self.tile_scheduler.task_type == JobType.MOE_GROUP_GEMM_GATE_UP_SILU.value:
                self.task_impl_moe_group_gemm_gate_up_silu(reordered_hidden_state_global, grp_gate_up_weight_global, silu_mul_output_global, topk_weights_flattened, sorted_token_ids_global, expert_ids_global, num_valid_tokens_global, num_tokens_post_pad_global, unfused, down_proj_task_size, is_dynamic_sch)
            elif self.tile_scheduler.task_type == JobType.MOE_GROUP_GEMM_DOWN.value:
                self.task_impl_moe_group_gemm_down(silu_mul_output_global, grp_down_weight_global, topk_reduce_output_global, expert_ids_global, topk_weights_flattened, sorted_token_ids_global, num_valid_tokens_global, num_tokens_post_pad_global, unfused, down_proj_task_size, is_dynamic_sch)
            elif self.tile_scheduler.task_type == JobType.INIT_ETENSOR.value:
                self.task_impl_init_etensor(is_dynamic_sch)
            elif self.tile_scheduler.task_type == JobType.WAIT_ETENSOR_INIT.value:
                self.task_impl_wait_etensor_init_complete(is_dynamic_sch)
            else:
                Tx.cuda.trap_when_assert_failed(False)
            self.smem_manager.exit_tile_runtime()
            self.tile_scheduler.next_tile()
        if self.profiler_on:
            self.profiler.finalize(lane_id == 0)
        self.class_finalize_all()

    # fmt: on

    # FIXME: change offset_factor to 0 can make performance better
    #       but it requires change on engine side
    def get_func_static(self, unfused=False):
        compile_batch_size = getattr(self, "_compile_batch_size", 1)

        # fmt: off
        @Tx.prim_func
        def main(
            # input and output
            hidden_state_ptr: Tx.handle, # input: read-only
            residual_ptr: Tx.handle, # input & output: inplace update
            output_ptr: Tx.handle, # output

            # weight
            gate_weight_ptr: Tx.handle, # read-only
            grp_gate_up_weight_ptr: Tx.handle, # read-only
            grp_down_weight_ptr: Tx.handle, # read-only

            # intermediate buffer
            gating_output_ptr: Tx.handle, # intermediate
            topk_weights_ptr: Tx.handle, # intermediate
            topk_indices_ptr: Tx.handle, # intermediate
            sorted_token_ids_ptr: Tx.handle, # intermediate
            expert_ids_ptr: Tx.handle, # intermediate
            num_valid_tokens_ptr: Tx.handle, # intermediate
            num_tokens_post_pad_ptr: Tx.handle, # intermediate
            cumsum_buffer_ptr: Tx.handle, # intermediate
            reordered_hidden_state_ptr: Tx.handle, # intermediate
            gate_up_output_ptr: Tx.handle, # intermediate
            silu_mul_output_ptr: Tx.handle, # intermediate
            topk_reduce_output_ptr: Tx.handle, # intermediate


            # event tensor
            etensor_workspace_ptr: Tx.handle, # not required to reset. Must be 0 before launch.

            # execution queue
            exec_queue_ptr: Tx.handle,
            profiler_buffer: Tx.Buffer((self.PROFILER_BUFFER_SIZE,), "uint64")
        ):
            Tx.func_attr(
                {"global_symbol": "main", "target": Tx.target("cuda")}
            )

            # match buffer
            batch_size = Tx.meta_var(compile_batch_size)

            # input and output
            hidden_state_global = Tx.match_buffer(hidden_state_ptr, [batch_size, self.HIDDEN_SIZE], "float16", scope="global")
            residual_global = Tx.match_buffer(residual_ptr, [batch_size, self.HIDDEN_SIZE], "float16", scope="global")
            output_global = Tx.match_buffer(output_ptr, [batch_size, self.HIDDEN_SIZE], "float16")

            # weight
            gate_weight_global = Tx.match_buffer(gate_weight_ptr, [self.NUM_EXPERTS, self.HIDDEN_SIZE], "float16", scope="global")
            grp_gate_up_weight_global = Tx.match_buffer(grp_gate_up_weight_ptr, [self.NUM_EXPERTS, self.INTERMEDIATE_SIZE * 2, self.HIDDEN_SIZE], "float16", scope="global")
            grp_down_weight_global = Tx.match_buffer(grp_down_weight_ptr, [self.NUM_EXPERTS, self.HIDDEN_SIZE, self.INTERMEDIATE_SIZE], "float16", scope="global")

            # intermediate buffer
            gating_output_global = Tx.match_buffer(gating_output_ptr, [batch_size, self.NUM_EXPERTS], "float32", scope="global")
            topk_weights_global = Tx.match_buffer(topk_weights_ptr, [batch_size, self.NUM_EXPERTS_PER_TOK], "float32", scope="global")
            topk_indices_global = Tx.match_buffer(topk_indices_ptr, [batch_size, self.NUM_EXPERTS_PER_TOK], "int32", scope="global")
            max_num_tokens_padded = Tx.int32()
            sorted_token_ids_global = Tx.match_buffer(sorted_token_ids_ptr, [max_num_tokens_padded], "int32", scope="global")
            expert_ids_global = Tx.match_buffer(expert_ids_ptr, [max_num_tokens_padded // self.MOE_M_PAD_SIZE], "int32", scope="global")
            num_valid_tokens_global = Tx.match_buffer(num_valid_tokens_ptr, [max_num_tokens_padded // self.MOE_M_PAD_SIZE], "int32", scope="global")
            num_tokens_post_pad_global = Tx.match_buffer(num_tokens_post_pad_ptr, [1], "int32", scope="global")
            cumsum_buffer_global = Tx.match_buffer(cumsum_buffer_ptr, [self.NUM_EXPERTS + 1], "int32", scope="global")
            reordered_hidden_state_global = Tx.match_buffer(reordered_hidden_state_ptr, [max_num_tokens_padded, self.HIDDEN_SIZE], "float16", scope="global")
            gate_up_output_global = Tx.match_buffer(gate_up_output_ptr, [max_num_tokens_padded, self.INTERMEDIATE_SIZE * 2], "float16", scope="global")
            silu_mul_output_global = Tx.match_buffer(silu_mul_output_ptr, [max_num_tokens_padded, self.INTERMEDIATE_SIZE], "float16", scope="global")
            topk_reduce_output_global = Tx.match_buffer(topk_reduce_output_ptr, [batch_size, self.HIDDEN_SIZE], "float16", scope="global")

            # event tensor
            etensor_workspace_size = Tx.int32()
            etensor_workspace_global = Tx.match_buffer(etensor_workspace_ptr, [etensor_workspace_size], "int32", scope="global")

            # exec queue
            exec_queue = Tx.match_buffer(exec_queue_ptr, [KernelConfig.SM_NUMBER, StaticTileScheduler.MAX_TASKS], "int32", scope="global")

            @Tx.inline
            def run(low_batch, dynamic_gemm_size):
                num_valid_tokens = Tx.meta_var(num_valid_tokens_global if dynamic_gemm_size else None)
                self.fused_body(
                    batch_size, hidden_state_global, residual_global, output_global, gate_weight_global, grp_gate_up_weight_global, grp_down_weight_global,
                    gating_output_global, topk_weights_global, topk_indices_global, sorted_token_ids_global, expert_ids_global, num_valid_tokens, num_tokens_post_pad_global,
                    cumsum_buffer_global, reordered_hidden_state_global, gate_up_output_global, silu_mul_output_global, topk_reduce_output_global,
                    etensor_workspace_global,
                    profiler_buffer, exec_queue, None, None, None, 1, low_batch, unfused,
                    False, static_scheduler.Semaphore, static_scheduler.StaticTileScheduler
                )

            if compile_batch_size >= 2048:
                run(low_batch=False, dynamic_gemm_size=True)
            elif compile_batch_size >= 512:
                run(low_batch=True, dynamic_gemm_size=True)
            else:
                run(low_batch=True, dynamic_gemm_size=False)
            # fmt: on
        return main

    def get_func_dynamic(self):
        compile_batch_size = getattr(self, "_compile_batch_size", 1)

        # fmt: off
        @Tx.prim_func
        def main(
            # input and output
            hidden_state_ptr: Tx.handle, # input: read-only
            residual_ptr: Tx.handle, # input & output: inplace update
            output_ptr: Tx.handle, # output

            # weight
            gate_weight_ptr: Tx.handle, # read-only
            grp_gate_up_weight_ptr: Tx.handle, # read-only
            grp_down_weight_ptr: Tx.handle, # read-only

            # intermediate buffer
            gating_output_ptr: Tx.handle, # intermediate
            topk_weights_ptr: Tx.handle, # intermediate
            topk_indices_ptr: Tx.handle, # intermediate
            sorted_token_ids_ptr: Tx.handle, # intermediate
            expert_ids_ptr: Tx.handle, # intermediate
            num_valid_tokens_ptr: Tx.handle, # intermediate
            num_tokens_post_pad_ptr: Tx.handle, # intermediate
            cumsum_buffer_ptr: Tx.handle, # intermediate
            reordered_hidden_state_ptr: Tx.handle, # intermediate
            gate_up_output_ptr: Tx.handle, # intermediate
            silu_mul_output_ptr: Tx.handle, # intermediate
            topk_reduce_output_ptr: Tx.handle, # intermediate


            # event tensor
            etensor_workspace_ptr: Tx.handle, # not required to reset. Must be 0 before launch.

            # execution queue
            queue_tasks_ptr: Tx.handle,
            queue_head_ptr: Tx.handle,
            queue_tail_ptr: Tx.handle,
            profiler_buffer: Tx.Buffer((self.PROFILER_BUFFER_SIZE,), "uint64")
        ):
            Tx.func_attr(
                {"global_symbol": "main", "target": Tx.target("cuda")}
            )

            # match buffer
            batch_size = Tx.meta_var(compile_batch_size)

            # input and output
            hidden_state_global = Tx.match_buffer(hidden_state_ptr, [batch_size, self.HIDDEN_SIZE], "float16", scope="global")
            residual_global = Tx.match_buffer(residual_ptr, [batch_size, self.HIDDEN_SIZE], "float16", scope="global")
            output_global = Tx.match_buffer(output_ptr, [batch_size, self.HIDDEN_SIZE], "float16")

            # weight
            gate_weight_global = Tx.match_buffer(gate_weight_ptr, [self.NUM_EXPERTS, self.HIDDEN_SIZE], "float16", scope="global")
            grp_gate_up_weight_global = Tx.match_buffer(grp_gate_up_weight_ptr, [self.NUM_EXPERTS, self.INTERMEDIATE_SIZE * 2, self.HIDDEN_SIZE], "float16", scope="global")
            grp_down_weight_global = Tx.match_buffer(grp_down_weight_ptr, [self.NUM_EXPERTS, self.HIDDEN_SIZE, self.INTERMEDIATE_SIZE], "float16", scope="global")

            # intermediate buffer
            gating_output_global = Tx.match_buffer(gating_output_ptr, [batch_size, self.NUM_EXPERTS], "float32", scope="global")
            topk_weights_global = Tx.match_buffer(topk_weights_ptr, [batch_size, self.NUM_EXPERTS_PER_TOK], "float32", scope="global")
            topk_indices_global = Tx.match_buffer(topk_indices_ptr, [batch_size, self.NUM_EXPERTS_PER_TOK], "int32", scope="global")
            max_num_tokens_padded = Tx.int32()
            sorted_token_ids_global = Tx.match_buffer(sorted_token_ids_ptr, [max_num_tokens_padded], "int32", scope="global")
            expert_ids_global = Tx.match_buffer(expert_ids_ptr, [max_num_tokens_padded // self.MOE_M_PAD_SIZE], "int32", scope="global")
            num_valid_tokens_global = Tx.match_buffer(num_valid_tokens_ptr, [max_num_tokens_padded // self.MOE_M_PAD_SIZE], "int32", scope="global")
            num_tokens_post_pad_global = Tx.match_buffer(num_tokens_post_pad_ptr, [1], "int32", scope="global")
            cumsum_buffer_global = Tx.match_buffer(cumsum_buffer_ptr, [self.NUM_EXPERTS + 1], "int32", scope="global")
            reordered_hidden_state_global = Tx.match_buffer(reordered_hidden_state_ptr, [max_num_tokens_padded, self.HIDDEN_SIZE], "float16", scope="global")
            gate_up_output_global = Tx.match_buffer(gate_up_output_ptr, [max_num_tokens_padded, self.INTERMEDIATE_SIZE * 2], "float16", scope="global")
            silu_mul_output_global = Tx.match_buffer(silu_mul_output_ptr, [max_num_tokens_padded, self.INTERMEDIATE_SIZE], "float16", scope="global")
            topk_reduce_output_global = Tx.match_buffer(topk_reduce_output_ptr, [batch_size, self.HIDDEN_SIZE], "float16", scope="global")

            # event tensor
            etensor_workspace_size = Tx.int32()
            etensor_workspace_global = Tx.match_buffer(etensor_workspace_ptr, [etensor_workspace_size], "int32", scope="global")

            # exec queue
            queue_tasks_global = Tx.match_buffer(queue_tasks_ptr, [DynamicTileScheduler.MAX_TASKS], "int32", scope="global", offset_factor=1)
            queue_head_global = Tx.match_buffer(queue_head_ptr, [1], "int32", scope="global", offset_factor=1)
            queue_tail_global = Tx.match_buffer(queue_tail_ptr, [1], "int32", scope="global", offset_factor=1)

            @Tx.inline
            def run(low_batch, dynamic_gemm_size, down_proj_task_size):
                num_valid_tokens = Tx.meta_var(num_valid_tokens_global if dynamic_gemm_size else None)
                self.fused_body(
                    batch_size, hidden_state_global, residual_global, output_global, gate_weight_global, grp_gate_up_weight_global, grp_down_weight_global,
                    gating_output_global, topk_weights_global, topk_indices_global, sorted_token_ids_global, expert_ids_global, num_valid_tokens, num_tokens_post_pad_global,
                    cumsum_buffer_global, reordered_hidden_state_global, gate_up_output_global, silu_mul_output_global, topk_reduce_output_global,
                    etensor_workspace_global,
                    profiler_buffer, None, queue_tasks_global, queue_head_global, queue_tail_global, down_proj_task_size, low_batch, False,
                    True, dynamic_scheduler.Semaphore, dynamic_scheduler.DynamicTileScheduler
                )

            if compile_batch_size >= 2048:
                run(low_batch=False, dynamic_gemm_size=True, down_proj_task_size=4)
            elif compile_batch_size >= 512:
                run(low_batch=True, dynamic_gemm_size=True, down_proj_task_size=4)
            elif compile_batch_size >= 4:
                run(low_batch=True, dynamic_gemm_size=False, down_proj_task_size=4)
            else:
                run(low_batch=True, dynamic_gemm_size=False, down_proj_task_size=1)
            # fmt: on
        return main


arg_dict = {}


def prepare_data(batch_size, mk: MegaKernelMOE):
    print("start prepare data", flush=True)
    global arg_dict

    def _correct_weight_tensor_view(tensor):
        if mk.world_size == 1:
            return tensor.view(*tensor.shape[1:])
        return tensor

    torch.manual_seed(42)

    # input
    arg_dict["hidden_state"] = torch.randn((batch_size, mk.HIDDEN_SIZE), dtype=torch.float16)
    arg_dict["residual"] = torch.randn((batch_size, mk.HIDDEN_SIZE), dtype=torch.float16)
    # intermediate buffer
    arg_dict["gating_output"] = torch.zeros((batch_size, mk.NUM_EXPERTS), dtype=torch.float32)
    arg_dict["topk_weights"] = torch.zeros(
        (batch_size, mk.NUM_EXPERTS_PER_TOK), dtype=torch.float32
    )
    arg_dict["topk_indices"] = torch.zeros((batch_size, mk.NUM_EXPERTS_PER_TOK), dtype=torch.int32)
    max_num_tokens_padded = get_max_num_tokens_padded(
        batch_size, mk.NUM_EXPERTS_PER_TOK, mk.NUM_EXPERTS, mk.MOE_M_PAD_SIZE
    )
    arg_dict["sorted_token_ids"] = torch.zeros((max_num_tokens_padded,), dtype=torch.int32)
    arg_dict["expert_ids"] = torch.zeros(
        (max_num_tokens_padded // mk.MOE_M_PAD_SIZE,), dtype=torch.int32
    )
    arg_dict["num_valid_tokens"] = torch.zeros(
        (max_num_tokens_padded // mk.MOE_M_PAD_SIZE,), dtype=torch.int32
    )
    arg_dict["num_tokens_post_pad"] = torch.zeros((1,), dtype=torch.int32)
    arg_dict["cumsum_buffer"] = torch.zeros((mk.NUM_EXPERTS + 1,), dtype=torch.int32)
    arg_dict["reordered_hidden_state"] = torch.zeros(
        (max_num_tokens_padded, mk.HIDDEN_SIZE), dtype=torch.float16
    )
    arg_dict["gate_up_output"] = torch.zeros(
        (max_num_tokens_padded, mk.INTERMEDIATE_SIZE * 2), dtype=torch.float16
    )
    arg_dict["silu_mul_output"] = torch.zeros(
        (max_num_tokens_padded, mk.INTERMEDIATE_SIZE), dtype=torch.float16
    )
    arg_dict["topk_reduce_output"] = torch.zeros((batch_size, mk.HIDDEN_SIZE), dtype=torch.float16)

    # weight initialization
    if not hasattr(prepare_data, "weight_initialized"):
        prepare_data.weight_initialized = True
    else:
        return arg_dict
    arg_dict["gate_weight"] = _correct_weight_tensor_view(
        torch.zeros((mk.world_size, mk.NUM_EXPERTS, mk.HIDDEN_SIZE), dtype=torch.float16).cuda()
    )
    torch.nn.init.xavier_normal_(arg_dict["gate_weight"], gain=1.0)
    arg_dict["gate_weight"] = arg_dict["gate_weight"].cpu()
    arg_dict["grp_gate_up_weight"] = _correct_weight_tensor_view(
        torch.zeros(
            (mk.world_size, mk.NUM_EXPERTS, mk.INTERMEDIATE_SIZE * 2, mk.HIDDEN_SIZE),
            dtype=torch.float16,
        ).cuda()
    )
    for i in range(mk.NUM_EXPERTS):
        torch.nn.init.xavier_normal_(arg_dict["grp_gate_up_weight"][i], gain=1.0)
    arg_dict["grp_gate_up_weight"] = arg_dict["grp_gate_up_weight"].cpu()
    w1 = arg_dict["grp_gate_up_weight"]
    arg_dict["grp_up_gate_weight"] = torch.cat(
        (w1[:, mk.INTERMEDIATE_SIZE :, :], w1[:, : mk.INTERMEDIATE_SIZE, :]), dim=1
    ).contiguous()
    new_order_indices = np.stack(
        (
            np.arange(mk.INTERMEDIATE_SIZE).reshape(-1, 16),
            np.arange(mk.INTERMEDIATE_SIZE, mk.INTERMEDIATE_SIZE * 2).reshape(-1, 16),
        ),
        axis=1,
    ).reshape(-1)
    arg_dict["shuffled_grp_gate_up_weight"] = arg_dict["grp_gate_up_weight"][
        :, new_order_indices, :
    ]

    arg_dict["grp_down_weight"] = _correct_weight_tensor_view(
        torch.zeros(
            (mk.world_size, mk.NUM_EXPERTS, mk.HIDDEN_SIZE, mk.INTERMEDIATE_SIZE),
            dtype=torch.float16,
        ).cuda()
    )
    for i in range(mk.NUM_EXPERTS):
        torch.nn.init.xavier_normal_(arg_dict["grp_down_weight"][i], gain=1.0)
    arg_dict["grp_down_weight"] = arg_dict["grp_down_weight"].cpu()
    print("end prepare data", flush=True)
    return arg_dict


def _require_cuda_sm100():
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for MegaKernelMOE")
    if torch.cuda.get_device_capability()[0] < 10:
        raise SkipTest("MegaKernelMOE requires SM100 or newer")
    if not tvm.cuda(0).exist:
        raise SkipTest("TVM CUDA device 0 is not available")
    return torch


def _reset_prepare_data_cache():
    arg_dict.clear()
    if hasattr(prepare_data, "weight_initialized"):
        delattr(prepare_data, "weight_initialized")


def _check_scheduler(scheduler: str):
    if scheduler not in _SUPPORTED_SCHEDULERS:
        raise ValueError(
            f"Unsupported scheduler {scheduler!r}; expected one of {_SUPPORTED_SCHEDULERS}"
        )


def _needs_unfused_reference(scheduler: str) -> bool:
    return scheduler in ("static", "dynamic")


def _compile_moe_schedulers(
    schedulers: tuple[str, ...], batch_size: int, world_size: int, profiler_on: bool
) -> tuple[MegaKernelMOE, dict[str, tvm.runtime.Module]]:
    if world_size != 1:
        raise SkipTest("tirx-kernels MegaKernelMOE benchmark currently supports world_size=1")
    for scheduler in schedulers:
        _check_scheduler(scheduler)

    key = (schedulers, batch_size, world_size, profiler_on)
    cached = _COMPILE_CACHE.get(key)
    if cached is not None:
        return cached

    mk = MegaKernelMOE(
        config=MEGAKERNEL_MOE_BENCH_CONFIG, world_size=world_size, profiler_on=profiler_on
    )
    mk._compile_batch_size = batch_size
    libs = {}
    for scheduler in schedulers:
        _, libs[scheduler] = get_source(mk.get_module(scheduler))
    _COMPILE_CACHE[key] = (mk, libs)
    return mk, libs


def _as_tvm_tensor(value, dev):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous().numpy()
    return tvm.runtime.tensor(value, dev)


def _as_cuda_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cuda()
    return torch.from_numpy(np.asarray(value)).cuda()


def _make_tir_case(
    *,
    batch_size: int,
    mk: MegaKernelMOE,
    lib: tvm.runtime.Module,
    scheduler: str,
    data: dict[str, torch.Tensor],
    launch_slots: int,
):
    dev = tvm.cuda(0)
    max_num_tokens_padded = get_max_num_tokens_padded(
        batch_size, mk.NUM_EXPERTS_PER_TOK, mk.NUM_EXPERTS, mk.MOE_M_PAD_SIZE
    )
    case = {
        "kernel": lib["main"],
        "scheduler": scheduler,
        "cursor": 0,
        "launch_slots": launch_slots,
        "hidden_state": _as_tvm_tensor(data["hidden_state"], dev),
        "output": tvm.runtime.tensor(np.zeros((batch_size, mk.HIDDEN_SIZE), dtype=np.float16), dev),
        "gate_weight": _as_tvm_tensor(data["gate_weight"], dev),
        "shuffled_grp_gate_up_weight": _as_tvm_tensor(data["shuffled_grp_gate_up_weight"], dev),
        "grp_down_weight": _as_tvm_tensor(data["grp_down_weight"], dev),
        "topk_weights": _as_tvm_tensor(data["topk_weights"], dev),
        "topk_indices": _as_tvm_tensor(data["topk_indices"], dev),
        "sorted_token_ids": _as_tvm_tensor(data["sorted_token_ids"], dev),
        "expert_ids": _as_tvm_tensor(data["expert_ids"], dev),
        "num_valid_tokens": _as_tvm_tensor(data["num_valid_tokens"], dev),
        "num_tokens_post_pad": _as_tvm_tensor(data["num_tokens_post_pad"], dev),
        "cumsum_buffer": _as_tvm_tensor(data["cumsum_buffer"], dev),
        "reordered_hidden_state": _as_tvm_tensor(data["reordered_hidden_state"], dev),
        "gate_up_output": _as_tvm_tensor(data["gate_up_output"], dev),
        "silu_mul_output": _as_tvm_tensor(data["silu_mul_output"], dev),
        "etensor_workspace": tvm.runtime.tensor(
            np.zeros([mk.ETENSOR_WORKSPACE_SIZE], dtype=np.int32), dev
        ),
        "profiler_buffer": tvm.runtime.tensor(
            np.zeros([mk.PROFILER_BUFFER_SIZE], dtype=np.uint64), dev
        ),
        "residual": [],
        "gating_output": [],
        "topk_reduce_output": [],
        "graph_reset": {
            "residual_source": _as_cuda_tensor(data["residual"]),
            "residual": [],
            "gating_output": [],
            "topk_reduce_output": [],
        },
    }
    for name in (
        "output",
        "topk_weights",
        "topk_indices",
        "sorted_token_ids",
        "expert_ids",
        "num_valid_tokens",
        "num_tokens_post_pad",
        "cumsum_buffer",
        "reordered_hidden_state",
        "gate_up_output",
        "silu_mul_output",
        "etensor_workspace",
        "profiler_buffer",
    ):
        case["graph_reset"].setdefault("zero", []).append(torch.from_dlpack(case[name]))

    for _ in range(launch_slots):
        residual = _as_tvm_tensor(data["residual"], dev)
        gating_output = _as_tvm_tensor(data["gating_output"], dev)
        topk_reduce_output = _as_tvm_tensor(data["topk_reduce_output"], dev)
        case["residual"].append(residual)
        case["gating_output"].append(gating_output)
        case["topk_reduce_output"].append(topk_reduce_output)
        case["graph_reset"]["residual"].append(torch.from_dlpack(residual))
        case["graph_reset"]["gating_output"].append(torch.from_dlpack(gating_output))
        case["graph_reset"]["topk_reduce_output"].append(torch.from_dlpack(topk_reduce_output))

    if scheduler in _STATIC_QUEUE_SCHEDULERS:
        exec_queue = generate_exec_queue_moe(
            batch_size, mk.config, mk.num_etensors[False], "static"
        )
        case["exec_queue"] = tvm.runtime.tensor(exec_queue, dev)
    else:
        exec_queue = generate_exec_queue_moe(
            batch_size, mk.config, mk.num_etensors[True], "dynamic"
        )
        case["queue_tasks"] = []
        case["queue_head"] = []
        case["queue_tail"] = []
        case["graph_reset"].update(
            {
                "queue_tasks_source": _as_cuda_tensor(exec_queue.tasks.copy()),
                "queue_head_source": _as_cuda_tensor(exec_queue.head.copy()),
                "queue_tail_source": _as_cuda_tensor(exec_queue.tail.copy()),
                "queue_tasks": [],
                "queue_head": [],
                "queue_tail": [],
            }
        )
        for _ in range(launch_slots):
            queue_tasks = tvm.runtime.tensor(exec_queue.tasks.copy(), dev)
            queue_head = tvm.runtime.tensor(exec_queue.head.copy(), dev)
            queue_tail = tvm.runtime.tensor(exec_queue.tail.copy(), dev)
            case["queue_tasks"].append(queue_tasks)
            case["queue_head"].append(queue_head)
            case["queue_tail"].append(queue_tail)
            case["graph_reset"]["queue_tasks"].append(torch.from_dlpack(queue_tasks))
            case["graph_reset"]["queue_head"].append(torch.from_dlpack(queue_head))
            case["graph_reset"]["queue_tail"].append(torch.from_dlpack(queue_tail))

    byte_tensors = [
        case["hidden_state"],
        case["output"],
        case["gate_weight"],
        case["shuffled_grp_gate_up_weight"],
        case["grp_down_weight"],
        case["topk_weights"],
        case["topk_indices"],
        case["sorted_token_ids"],
        case["expert_ids"],
        case["num_valid_tokens"],
        case["num_tokens_post_pad"],
        case["cumsum_buffer"],
        case["reordered_hidden_state"],
        case["gate_up_output"],
        case["silu_mul_output"],
        case["etensor_workspace"],
        case["profiler_buffer"],
        case["residual"][0],
        case["gating_output"][0],
        case["topk_reduce_output"][0],
    ]
    if scheduler in _STATIC_QUEUE_SCHEDULERS:
        byte_tensors.append(case["exec_queue"])
    else:
        byte_tensors.extend([case["queue_tasks"][0], case["queue_head"][0], case["queue_tail"][0]])
    case["byte_tensors"] = byte_tensors
    case["max_num_tokens_padded"] = max_num_tokens_padded
    return case


def _reset_tir_case_for_cuda_graph(case):
    idx = case["cursor"]
    if idx >= case["launch_slots"]:
        raise RuntimeError(
            f"MegaKernelMOE benchmark exhausted launch slots "
            f"({case['launch_slots']}); increase warmup/repeat slot allocation"
        )

    reset = case["graph_reset"]
    for tensor in reset.get("zero", []):
        tensor.zero_()
    reset["residual"][idx].copy_(reset["residual_source"])
    reset["gating_output"][idx].zero_()
    reset["topk_reduce_output"][idx].zero_()
    if case["scheduler"] not in _STATIC_QUEUE_SCHEDULERS:
        reset["queue_tasks"][idx].copy_(reset["queue_tasks_source"])
        reset["queue_head"][idx].copy_(reset["queue_head_source"])
        reset["queue_tail"][idx].copy_(reset["queue_tail_source"])


def _run_tir_case(case):
    idx = case["cursor"]
    if idx >= case["launch_slots"]:
        raise RuntimeError(
            f"MegaKernelMOE benchmark exhausted launch slots "
            f"({case['launch_slots']}); increase warmup/repeat slot allocation"
        )
    kernel = case["kernel"]
    if case["scheduler"] in _STATIC_QUEUE_SCHEDULERS:
        kernel(
            case["hidden_state"],
            case["residual"][idx],
            case["output"],
            case["gate_weight"],
            case["shuffled_grp_gate_up_weight"],
            case["grp_down_weight"],
            case["gating_output"][idx],
            case["topk_weights"],
            case["topk_indices"],
            case["sorted_token_ids"],
            case["expert_ids"],
            case["num_valid_tokens"],
            case["num_tokens_post_pad"],
            case["cumsum_buffer"],
            case["reordered_hidden_state"],
            case["gate_up_output"],
            case["silu_mul_output"],
            case["topk_reduce_output"][idx],
            case["etensor_workspace"],
            case["exec_queue"],
            case["profiler_buffer"],
        )
    else:
        kernel(
            case["hidden_state"],
            case["residual"][idx],
            case["output"],
            case["gate_weight"],
            case["shuffled_grp_gate_up_weight"],
            case["grp_down_weight"],
            case["gating_output"][idx],
            case["topk_weights"],
            case["topk_indices"],
            case["sorted_token_ids"],
            case["expert_ids"],
            case["num_valid_tokens"],
            case["num_tokens_post_pad"],
            case["cumsum_buffer"],
            case["reordered_hidden_state"],
            case["gate_up_output"],
            case["silu_mul_output"],
            case["topk_reduce_output"][idx],
            case["etensor_workspace"],
            case["queue_tasks"][idx],
            case["queue_head"][idx],
            case["queue_tail"][idx],
            case["profiler_buffer"],
        )
    case["cursor"] += 1
    case["last_output"] = case["topk_reduce_output"][idx]
    return case["last_output"]


def _ensure_reference_cuda_case(case, mk: MegaKernelMOE):
    ref_case = case.get("reference_cuda")
    if ref_case is not None:
        return ref_case

    cpu_data = case["cpu_data"]
    ref_case = {
        key: cpu_data[key].clone().cuda()
        for key in (
            "hidden_state",
            "gate_weight",
            "grp_gate_up_weight",
            "grp_up_gate_weight",
            "grp_down_weight",
        )
    }
    ref_case["hidden_state_router"] = ref_case["hidden_state"].to(torch.float32)
    ref_case["gate_weight_router"] = ref_case["gate_weight"].to(torch.float32)
    ref_case["baseline_gating_output"] = torch.empty(
        (case["batch_size"], mk.NUM_EXPERTS), dtype=torch.float32, device="cuda"
    )
    ref_case["baseline_topk_weights"] = torch.empty(
        (case["batch_size"], mk.NUM_EXPERTS_PER_TOK), dtype=torch.float32, device="cuda"
    )
    ref_case["baseline_topk_indices_i64"] = torch.empty(
        (case["batch_size"], mk.NUM_EXPERTS_PER_TOK), dtype=torch.int64, device="cuda"
    )
    ref_case["baseline_topk_indices"] = torch.empty(
        (case["batch_size"], mk.NUM_EXPERTS_PER_TOK), dtype=torch.int32, device="cuda"
    )
    case["reference_cuda"] = ref_case
    return ref_case


def _compute_reference_routing(ref_case, mk: MegaKernelMOE):
    torch.mm(
        ref_case["hidden_state_router"],
        ref_case["gate_weight_router"].T,
        out=ref_case["baseline_gating_output"],
    )
    routing_weights = torch.softmax(ref_case["baseline_gating_output"], dim=-1, dtype=torch.float32)
    torch.topk(
        routing_weights,
        mk.NUM_EXPERTS_PER_TOK,
        dim=-1,
        out=(ref_case["baseline_topk_weights"], ref_case["baseline_topk_indices_i64"]),
    )
    ref_case["baseline_topk_indices"].copy_(ref_case["baseline_topk_indices_i64"])
    return (
        ref_case["baseline_gating_output"],
        ref_case["baseline_topk_weights"],
        ref_case["baseline_topk_indices"],
    )


def _build_sglang_full_reference(mk: MegaKernelMOE):
    try:
        import importlib
        import os

        os.environ.setdefault(
            "SGLANG_MOE_CONFIG_DIR", os.path.join(os.path.dirname(__file__), "sglang_moe_configs")
        )

        triton_compiler = importlib.import_module("triton.compiler.compiler")
        if not hasattr(triton_compiler, "triton_key"):
            triton_key = triton_compiler.get_cache_key.__globals__.get("triton_key")
            if triton_key is not None:
                triton_compiler.triton_key = triton_key

        from sglang.srt.layers.moe import MoeRunnerConfig
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe
        from sglang.srt.layers.moe.topk import StandardTopKOutput
        from sglang.srt.runtime_context import get_server_args
        from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    except (ImportError, AttributeError) as err:
        raise RuntimeError(f"sglang full MoE reference is unavailable: {err}") from err

    try:
        get_server_args()
    except ValueError:
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    moe_config = MoeRunnerConfig(
        num_experts=mk.NUM_EXPERTS,
        num_local_experts=mk.NUM_EXPERTS,
        hidden_size=mk.HIDDEN_SIZE,
        intermediate_size_per_partition=mk.INTERMEDIATE_SIZE,
        top_k=mk.NUM_EXPERTS_PER_TOK,
        params_dtype=torch.float16,
        inplace=False,
    )

    def run(case):
        ref_case = _ensure_reference_cuda_case(case, mk)
        gating_output, topk_weights, topk_indices = _compute_reference_routing(ref_case, mk)
        topk_output = StandardTopKOutput(
            topk_weights=topk_weights, topk_ids=topk_indices, router_logits=gating_output
        )
        out = fused_moe(
            ref_case["hidden_state"],
            ref_case["grp_gate_up_weight"],
            ref_case["grp_down_weight"],
            topk_output,
            moe_config,
        )
        ref_case["sglang_full_output"] = out
        return out

    return run


def _build_flashinfer_full_reference(batch_size: int, mk: MegaKernelMOE):
    try:
        import os

        import flashinfer.fused_moe as fused_moe_lib
        from flashinfer.autotuner import autotune
    except ImportError as err:
        raise RuntimeError(f"flashinfer full MoE reference is unavailable: {err}") from err

    tune_mode = os.environ.get("TIRX_MEGAKERNEL_MOE_FLASHINFER_AUTOTUNE", "1") != "0"
    tune_cache = os.environ.get(
        "TIRX_MEGAKERNEL_MOE_FLASHINFER_CACHE",
        os.path.expanduser("~/.cache/tirx-kernels/megakernel_moe_flashinfer_cutlass.json"),
    )
    tune_cache_dir = os.path.dirname(tune_cache)
    if tune_cache_dir:
        os.makedirs(tune_cache_dir, exist_ok=True)

    tuned = False

    def run_cutlass(ref_case, topk_weights, topk_indices, out):
        fused_moe_lib.cutlass_fused_moe(
            ref_case["hidden_state"],
            topk_indices,
            topk_weights,
            ref_case["grp_up_gate_weight"],
            ref_case["grp_down_weight"],
            ref_case["hidden_state"].dtype,
            quant_scales=[],
            output=out,
            tune_max_num_tokens=batch_size,
        )

    def run(case):
        nonlocal tuned
        ref_case = _ensure_reference_cuda_case(case, mk)
        _, topk_weights, topk_indices = _compute_reference_routing(ref_case, mk)
        out = ref_case.get("flashinfer_full_output")
        if out is None:
            out = torch.zeros_like(ref_case["hidden_state"])
            ref_case["flashinfer_full_output"] = out

        if not tuned:
            with (
                torch.inference_mode(),
                autotune(tune_mode, cache=tune_cache, tuning_buckets=(batch_size,), round_up=True),
            ):
                run_cutlass(ref_case, topk_weights, topk_indices, out)
            tuned = True
        else:
            with torch.inference_mode():
                run_cutlass(ref_case, topk_weights, topk_indices, out)
        return out

    return run


def _torch_moe_reference(data: dict[str, torch.Tensor], mk: MegaKernelMOE):
    hidden_state = data["hidden_state"].cuda().to(torch.float32)
    gate_weight = data["gate_weight"].cuda().to(torch.float32)
    gate_up_weight = data["grp_gate_up_weight"].cuda().to(torch.float32)
    down_weight = data["grp_down_weight"].cuda().to(torch.float32)

    gating_output = hidden_state @ gate_weight.T
    routing_weights = torch.softmax(gating_output, dim=-1)
    topk_weights, topk_indices = torch.topk(routing_weights, mk.NUM_EXPERTS_PER_TOK, dim=-1)

    output = torch.zeros(
        (hidden_state.shape[0], mk.HIDDEN_SIZE), dtype=torch.float32, device="cuda"
    )
    for token_idx in range(hidden_state.shape[0]):
        token = hidden_state[token_idx]
        for route_idx in range(mk.NUM_EXPERTS_PER_TOK):
            expert_idx = int(topk_indices[token_idx, route_idx])
            gate_up = gate_up_weight[expert_idx] @ token
            gate = gate_up[: mk.INTERMEDIATE_SIZE]
            up = gate_up[mk.INTERMEDIATE_SIZE :]
            activated = torch.nn.functional.silu(gate) * up
            output[token_idx] += topk_weights[token_idx, route_idx] * (
                down_weight[expert_idx] @ activated
            )
    return output.cpu().numpy()


def _validate_tir_case(case, mk: MegaKernelMOE, *, check_torch: bool = True):
    out = _run_tir_case(case["tir"]).numpy()
    if not np.isfinite(out).all():
        raise AssertionError("MegaKernelMOE TIR output contains non-finite values")

    if check_torch:
        ref = _torch_moe_reference(case["cpu_data"], mk)
        np.testing.assert_allclose(out, ref, rtol=2e-2, atol=1e-2)

    reference = case.get("tir_reference")
    if reference is not None:
        ref = _run_tir_case(reference).numpy()
        np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-2)
    return out


def _validate_tir_matches_reference(case, mk: MegaKernelMOE, reference_run):
    out = case["tir"].get("last_output")
    if out is None:
        out_np = _validate_tir_case(case, mk)
    else:
        out_np = out.numpy()
    ref_np = reference_run(case).detach().cpu().numpy()
    np.testing.assert_allclose(out_np, ref_np, rtol=2e-2, atol=1e-2)
    abs_diff = np.abs(out_np.astype(np.float32) - ref_np.astype(np.float32))
    return {
        "allclose_rtol": 2e-2,
        "allclose_atol": 1e-2,
        "max_abs": float(abs_diff.max(initial=0.0)),
        "mean_abs": float(abs_diff.mean()),
        "numel": int(abs_diff.size),
    }


def run_test(batch_size: int, scheduler: str, world_size: int = 1, profiler_on: bool = False):
    _require_cuda_sm100()
    _check_scheduler(scheduler)
    compile_schedulers = [scheduler]
    if _needs_unfused_reference(scheduler):
        compile_schedulers.append("unfused")
    mk, libs = _compile_moe_schedulers(
        tuple(compile_schedulers), batch_size, world_size, profiler_on
    )

    _reset_prepare_data_cache()
    data = dict(prepare_data(batch_size, mk))
    case = {
        "batch_size": batch_size,
        "cpu_data": data,
        "tir": _make_tir_case(
            batch_size=batch_size,
            mk=mk,
            lib=libs[scheduler],
            scheduler=scheduler,
            data=data,
            launch_slots=2,
        ),
    }
    if _needs_unfused_reference(scheduler):
        case["tir_reference"] = _make_tir_case(
            batch_size=batch_size,
            mk=mk,
            lib=libs["unfused"],
            scheduler="unfused",
            data=data,
            launch_slots=2,
        )
    _validate_tir_case(case, mk)
    torch.cuda.synchronize()


def _estimate_tir_runtime_us(case) -> float:
    _run_tir_case(case)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        _run_tir_case(case)
    end_event.record()
    torch.cuda.synchronize()
    return max(start_event.elapsed_time(end_event) * 1000.0 / 5.0, 1.0)


def _estimate_bench_launch_slots(
    runtime_us: float, warmup: int | None, repeat: int | None, rounds: int, preflight_launches: int
) -> int:
    """Reserve enough pre-zeroed slots for the pure-launch bench loop.

    The current MoE kernel uses reductions into output workspaces, so the timed
    closure must not reuse a workspace slot unless it has been reset outside the
    timed region.  The new Triton-style bench API takes warmup/repeat as
    millisecond budgets, not fixed iteration counts.  This estimate is only for
    workspace capacity; the actual benchmark protocol remains owned by
    ``tvm.tirx.bench.bench``.
    """
    warmup_budget_ms = 25 if warmup is None else warmup
    repeat_budget_ms = 100 if repeat is None else repeat
    estimate_ms = max(runtime_us / 1000.0, 1e-6)
    n_warmup = max(1, int(warmup_budget_ms / estimate_ms))
    n_repeat = max(1, int(repeat_budget_ms / estimate_ms))
    # bench() calls fn() once, then runs a five-call estimate before the warmup
    # and timed loops. Add a small guard for integer rounding and post-bench
    # validation.
    return preflight_launches + rounds * (1 + 5 + n_warmup + n_repeat) + 16


def run_bench(
    batch_size: int,
    scheduler: str | None = None,
    world_size: int = 1,
    profiler_on: bool = False,
    warmup: int | None = None,
    repeat: int | None = None,
    timer: str | None = None,
    rounds: int = 1,
    round_cooldown_s: float = 1.0,
    **kwargs,
):
    _require_cuda_sm100()
    from tvm.tirx.bench import bench

    schedulers = (scheduler,) if scheduler is not None else _SUPPORTED_SCHEDULERS
    for scheduler_name in schedulers:
        _check_scheduler(scheduler_name)
    mk, libs = _compile_moe_schedulers(tuple(schedulers), batch_size, world_size, profiler_on)

    _reset_prepare_data_cache()
    data = dict(prepare_data(batch_size, mk))
    case = {"batch_size": batch_size, "cpu_data": data}
    launch_slots: dict[str, int] = {}
    tir_runtime_estimate_us: dict[str, float] = {}

    for scheduler_name in schedulers:
        probe_case = _make_tir_case(
            batch_size=batch_size,
            mk=mk,
            lib=libs[scheduler_name],
            scheduler=scheduler_name,
            data=data,
            launch_slots=8,
        )
        tir_runtime_estimate_us[scheduler_name] = _estimate_tir_runtime_us(probe_case)
        launch_slots[scheduler_name] = _estimate_bench_launch_slots(
            tir_runtime_estimate_us[scheduler_name], warmup, repeat, rounds, preflight_launches=1
        )
        if timer == "cudagraph_proton":
            launch_slots[scheduler_name] *= 4
        del probe_case
    torch.cuda.empty_cache()

    tir_cases = {
        scheduler_name: _make_tir_case(
            batch_size=batch_size,
            mk=mk,
            lib=libs[scheduler_name],
            scheduler=scheduler_name,
            data=data,
            launch_slots=launch_slots[scheduler_name],
        )
        for scheduler_name in schedulers
    }
    for scheduler_name in schedulers:
        validation_case = {**case, "tir": tir_cases[scheduler_name]}
        if scheduler_name != "unfused" and "unfused" in tir_cases:
            validation_case["tir_reference"] = tir_cases["unfused"]
        _validate_tir_case(validation_case, mk, check_torch=batch_size <= max(_TEST_BATCH_SIZES))

    sglang_runner = None
    flashinfer_runner = None
    validation = {}

    def build_sglang_full():
        nonlocal sglang_runner
        if sglang_runner is None:
            runner = _build_sglang_full_reference(mk)

            def run():
                return runner(case)

            sglang_runner = run
        return sglang_runner

    def build_flashinfer_full():
        nonlocal flashinfer_runner
        if flashinfer_runner is None:
            runner = _build_flashinfer_full_reference(batch_size, mk)

            def run():
                return runner(case)

            flashinfer_runner = run
        return flashinfer_runner

    def make_tir_runner(tir_case):
        def run_tir():
            return _run_tir_case(tir_case)

        run_tir.graph_reset = lambda: _reset_tir_case_for_cuda_graph(tir_case)
        return run_tir

    tir_impls = {
        ("tir" if scheduler is not None else f"tir_{scheduler_name}"): make_tir_runner(
            tir_cases[scheduler_name]
        )
        for scheduler_name in schedulers
    }

    result = bench(
        tir_impls,
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        references={"sglang_full": build_sglang_full, "flashinfer_full": build_flashinfer_full},
        rounds=rounds,
        round_cooldown_s=round_cooldown_s,
        **kwargs,
    )

    baseline_errors = result.get("errors") or {}
    if baseline_errors:
        details = "; ".join(f"{name}: {error}" for name, error in baseline_errors.items())
        raise RuntimeError(f"MegaKernelMOE benchmark requires both full references: {details}")
    if sglang_runner is None or flashinfer_runner is None:
        raise RuntimeError("MegaKernelMOE benchmark requires both full references")

    for scheduler_name in schedulers:
        impl_name = "tir" if scheduler is not None else f"tir_{scheduler_name}"
        validation_case = {**case, "tir": tir_cases[scheduler_name]}
        validation[f"{impl_name}_vs_flashinfer_full"] = _validate_tir_matches_reference(
            validation_case, mk, lambda _: flashinfer_runner()
        )
        validation[f"{impl_name}_vs_sglang_full"] = _validate_tir_matches_reference(
            validation_case, mk, lambda _: sglang_runner()
        )

    result.setdefault("metadata", {})
    result["metadata"].update(
        {
            "scheduler": scheduler if scheduler is not None else "all",
            "schedulers": list(schedulers),
            "our_impls": list(tir_impls),
            "batch_size": batch_size,
            "world_size": world_size,
            "config": MEGAKERNEL_MOE_BENCH_CONFIG["CONFIG_NAME"],
            "benchmark_scope": "full_moe_router_plus_expert",
            "sglang_router": "torch_fp32_mm_softmax_topk",
            "sglang_weight_layout": "grp_gate_up_weight",
            "flashinfer_router": "torch_fp32_mm_softmax_topk",
            "flashinfer_weight_layout": "grp_up_gate_weight",
            "launch_slots": launch_slots if scheduler is None else launch_slots[scheduler],
            "tir_runtime_estimate_us": (
                tir_runtime_estimate_us if scheduler is None else tir_runtime_estimate_us[scheduler]
            ),
        }
    )
    if validation:
        result["metadata"]["validation"] = validation
    return result

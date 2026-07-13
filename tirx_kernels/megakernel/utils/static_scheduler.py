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

"""Static tile scheduler for megakernel."""

from typing import Literal

import tvm
from tirx_kernels.megakernel.utils.base import SemaphoreBase, TileSchedulerBase
from tirx_kernels.megakernel.utils.config import JobType, KernelConfig
from tirx_kernels.megakernel.utils.utils import any_sync, atomic_add_int32, gt, unpack_from_32bit
from tvm.script import tirx as T


class Semaphore(SemaphoreBase):
    def __init__(self, buffer):
        self.sem = buffer
        self.state = T.alloc_buffer([1], "int32", scope="local", align=4)

    @T.inline
    def semaphore_wait(self, *coord, level: Literal["cta", "warp"] = "cta", mask=0xFFFFFFFF):
        if level == "cta":
            while 1:
                T.ptx.ld_global_acquire(
                    self.state[0], self.sem.access_ptr("r", offset=self.sem.elem_offset_of(coord))
                )
                if T.cuda.syncthreads_and(self.state[0] == 0):
                    break
                T.cuda.nano_sleep(40)
        elif level == "warp":
            warp_id = T.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER])
            lane_id = T.lane_id([32])
            if (mask >> warp_id) & 1 == 1:
                self.state[0] = -1
                while 1:
                    if lane_id == 0:
                        T.ptx.ld_global_acquire(
                            self.state[0],
                            self.sem.access_ptr("r", offset=self.sem.elem_offset_of(coord)),
                        )
                    if any_sync(0xFFFFFFFF, self.state[0] == 0):
                        break
                    T.cuda.nano_sleep(40)
        else:
            assert False

    @T.inline
    def semaphore_notify(self, *coord, rank=-1, release=False):
        # wg is synced
        self.state[0] = atomic_add_int32(
            self.sem.ptr_to(coord), -(self.base + 1), rank, release=release
        )
        if self.state[0] <= 0:
            while 1:
                T.ptx.ld_global_acquire(self.state[0], self.sem.ptr_to(coord))
                if gt(self.state[0], 0):
                    atomic_add_int32(
                        self.sem.ptr_to(coord), -(self.base + 1), rank, release=release
                    )
                    break
                T.cuda.nano_sleep(40)


class StaticTileScheduler(TileSchedulerBase):
    MAX_TASKS = 128

    def __init__(self, prefix: str, exec_queue, smem_manager, debug=False):
        super().__init__()
        self.exec_queue = exec_queue
        self.debug = debug
        self.prefix = prefix
        self.smem_manager = smem_manager

    @T.inline
    def _update_current_m_n_idx(self):
        unpack_from_32bit(
            self.queue_smem[self.tile_idx],
            T.address_of(self.task_type),
            T.address_of(self.m_idx),
            T.address_of(self.n_idx),
            T.address_of(self.k_idx),
        )

    def _alloc(self):
        self.m_idx = T.local_scalar("int32")
        self.n_idx = T.local_scalar("int32")
        self.k_idx = T.local_scalar("int32")
        self.task_type = T.local_scalar("int32")
        self.tile_idx = T.local_scalar("int32")
        self.queue_smem = self.smem_manager.alloc(
            (self.MAX_TASKS,), "int32", align=16, method="persistent"
        )

    @T.inline
    def init(self):
        self._alloc()
        bx = T.cta_id([KernelConfig.SM_NUMBER])
        tid = T.thread_id([KernelConfig.NUM_THREADS])
        self.tile_idx = 0
        for k in T.serial(T.ceildiv(self.MAX_TASKS, KernelConfig.NUM_THREADS)):
            idx = T.meta_var(k * KernelConfig.NUM_THREADS + tid)
            if idx < self.MAX_TASKS:
                self.queue_smem[idx] = self.exec_queue[bx, idx]
        T.tvm_storage_sync("shared")
        self._update_current_m_n_idx()

    def get_idx_and_task_type(self):
        return [self.m_idx, self.n_idx, self.k_idx], self.task_type

    @T.inline
    def next_tile(self):
        self.tile_idx += 1
        self._update_current_m_n_idx()

    @T.inline
    def wait(
        self, evt: Semaphore, *coord, wait_level: Literal["cta", "warp"] = "cta", mask=0xFFFFFFFF
    ):
        evt.semaphore_wait(*coord, level=wait_level, mask=mask)

    @T.inline
    def notify(
        self,
        evt: Semaphore,
        func_notify,
        scope: Literal["thread", "warp", "warpgroup", "cta"] = "thread",
        scope_id=0,
        release=False,
    ):
        # Notes: Here each thread will notify only at most one time,
        #        and the tids of the threads involved among scope in the notification process start from 0 and increment sequentially.
        # Notes: (num, rank, coord) = func_notify(notify_idx), rank=-1 for the local rank
        # Notes: scope_id = -1 represents that each scope will separately notify

        max_notify_num_map = T.meta_var(
            {
                "thread": 1,
                "warp": 32,
                "warpgroup": KernelConfig.NUM_THREADS // KernelConfig.WG_NUMBER,
                "cta": KernelConfig.NUM_THREADS,
            }
        )
        max_scope_id_map = T.meta_var(
            {
                "thread": KernelConfig.NUM_THREADS,
                "warp": KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER,
                "warpgroup": KernelConfig.WG_NUMBER,
                "cta": 1,
            }
        )

        @T.inline
        def sync(scope: Literal["thread", "warp", "warpgroup", "cta"], scope_id=0):
            if scope == "thread":
                pass
            elif scope == "warp":
                T.cuda.warp_sync()
            elif scope == "warpgroup":
                T.ptx.bar.sync(6 + scope_id, 128)
            elif scope == "cta":
                T.tvm_storage_sync("shared")

        wg_id = T.warpgroup_id([KernelConfig.WG_NUMBER])
        warp_id = T.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER])
        tid = T.thread_id([KernelConfig.NUM_THREADS])
        tid_in_wg = T.thread_id_in_wg([KernelConfig.NUM_THREADS // KernelConfig.WG_NUMBER])
        lane_id = T.lane_id([32])
        idx_map = T.meta_var(
            {
                "thread": (tid, 0),
                "warp": (warp_id, lane_id),
                "warpgroup": (wg_id, tid_in_wg),
                "cta": (0, tid),
            }
        )
        idx = T.meta_var(idx_map[scope])
        if self.debug:
            T.cuda.trap_when_assert_failed(scope_id == -1 or scope_id < max_scope_id_map[scope])
        if scope_id == -1 or idx[0] == scope_id:
            sync(scope, scope_id)
            # `func_notify` can emit side effects while constructing dependency exprs.
            # Evaluate it once and then unpack to avoid duplicated rewrites.
            notify_info = T.meta_var(func_notify(idx[1]))
            notify_num = T.meta_var(notify_info[0])
            rank = T.meta_var(notify_info[1])
            coord = T.meta_var(notify_info[2:])
            if self.debug:
                T.cuda.trap_when_assert_failed(notify_num <= max_notify_num_map[scope])
            if idx[1] < notify_num:
                evt.semaphore_notify(*coord, rank=rank, release=release)

    def valid(self):
        return tvm.tirx.all(self.tile_idx < self.MAX_TASKS, self.task_type != JobType.END.value)

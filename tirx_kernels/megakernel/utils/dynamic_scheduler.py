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

"""Dynamic tile scheduler for megakernel."""

from typing import Literal

import numpy as np

from tirx_kernels.megakernel.utils.base import Barriers, SemaphoreBase, TileSchedulerBase
from tirx_kernels.megakernel.utils.config import JobType, KernelConfig, ProfileEventType
from tirx_kernels.megakernel.utils.utils import (
    any_sync,
    atomic_add_int32,
    gt,
    pack_into_32bit,
    stg,
    sts,
    unpack_from_32bit,
    while_ld_global_acquire,
)
from tvm.script import tirx as T
from tvm.tirx.bench import CudaProfiler

# notes: The following applies to decrement=False. The logic for True is similar.
#        For semaphore with expected count = expected_cnt, we set it actual count = expected_cnt * (base + 1).
#        Here base >= expected_cnt is the power of 2 (for efficiency mod in the following and giving convenience for decrement=True).
#        By default, the base will be set to 1 << 16.
#        That means the semaphore will take in two args: Semaphore(expected_cnt, base). For decrement=True, we can set semaphore value in generation.
#        In dynamic scheduling, the semaphore will be notified for two times.
#        The first one happens after the prefetch of the tile but before the corresponding semaphore wait,
#        which will atomic_add the semaphore value by 1.
#        The second one happens after the tile running, which will atomic_add the semaphore value by base.
#        The task pushing will happen after the first semaphore notify, which will be triggered by old_value % base == expected_cnt - 1.
#        In this way, the tasks will be pre-push when the last tile has already been dispatched to the sm, which avoid the dead lock.
#        For semaphore wait, we can still use the condition value == expected_cnt * (base + 1) to distinguish.


class Semaphore(SemaphoreBase):
    def __init__(self, buffer, debug=False):
        self.sem = buffer
        self.state = T.alloc_local([1], "int32")

        # cta-level interface

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
    def semaphore_notify(self, *coord, pre_notify=False, rank=-1, release=False):
        number = T.meta_var(1 if pre_notify else self.base)
        # the old value will be stored in self.state
        self.state[0] = atomic_add_int32(self.sem.ptr_to(coord), -number, rank, release=release)
        if self.state[0] <= 0:
            while 1:
                T.ptx.ld_global_acquire(self.state[0], self.sem.ptr_to(coord))
                if gt(self.state[0], 0):
                    self.state[0] = atomic_add_int32(
                        self.sem.ptr_to(coord), -number, rank, release=release
                    )
                    break
                sleep_time = T.meta_var(800 if pre_notify else 40)
                T.cuda.nano_sleep(sleep_time)

    def is_triggered(self):
        return self.state[0] % self.base == 1


class SchedulerBarrier(Barriers):
    def __init__(self, smem_manager, is_p2c):
        super().__init__(smem_manager, 1, is_p2c)

    @T.inline
    def arrive(self):
        T.ptx.mbarrier.arrive(self.mbar.ptr_to([0]))


@T.meta_class
class MPMCQueue:
    def __init__(
        self,
        capacity: int,
        tasks: T.Buffer,
        head: T.Buffer,
        tail: T.Buffer,
        smem_manager,
        debug=False,
    ):
        # TODO: we currently assume that the queue is infinitely large.
        if capacity & (capacity - 1):
            raise ValueError("capacity must be a power-of-two")
        self.capacity = capacity
        self.mask = capacity - 1
        self.tasks = tasks  # an array of (task_type, m_idx, n_idx, k_idx)
        self.head = head
        self.tail = tail
        self.debug = debug
        self.smem_manager = smem_manager

    def _alloc(self):
        self.head_r = T.local_scalar(dtype="int32")
        self.tail_r = T.local_scalar(dtype="int32")
        self.masked_pos = T.local_scalar(dtype="int32")
        self.idx = T.local_scalar(dtype="int32")
        self.tail_smem = self.smem_manager.alloc(
            (KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER,), "int32", method="persistent"
        )

    @T.inline
    def init(self):
        self._alloc()

    @T.inline
    def enqueue(self, rank, func_push, level: Literal["thread", "warp", "warpgroup", "cta"]):
        if level == "thread":
            task_type, enqueue_num, m_idx, n_idx, k_idx = func_push(0)
            if self.debug:
                T.cuda.trap_when_assert_failed(enqueue_num == 1)  # notes: enqueue_num must be 1
            self.tail_r = atomic_add_int32(
                self.tail.access_ptr("rw", offset=self.tail.elem_offset_of([T.int32(0)])), 1, rank
            )
            self.masked_pos = self.tail_r & self.mask
            task_info = T.meta_var(
                pack_into_32bit(m_idx, n_idx, k_idx, task_type, host=False, debug=self.debug)
            )
            stg(
                task_info,
                self.tasks.access_ptr("rw", offset=self.tasks.elem_offset_of([self.masked_pos])),
                rank,
            )
        else:
            lane_id = T.lane_id([32])
            tid_in_wg = T.thread_id_in_wg([KernelConfig.NUM_THREADS // KernelConfig.WG_NUMBER])
            tid = T.thread_id([KernelConfig.NUM_THREADS])
            wg_id = T.warpgroup_id([KernelConfig.WG_NUMBER])
            warp_id = T.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER])
            idx_map = T.meta_var(
                {
                    "warp": (warp_id, lane_id, 32),
                    "warpgroup": (
                        wg_id,
                        tid_in_wg,
                        KernelConfig.NUM_THREADS // KernelConfig.WG_NUMBER,
                    ),
                    "cta": (0, tid, KernelConfig.NUM_THREADS),
                }
            )
            scope_idx, tid_in_scope, tid_stride = idx_map[level]
            enqueue_num = func_push(0)[1]
            if level == "warp":
                if tid_in_scope == 0:
                    self.tail_r = atomic_add_int32(
                        self.tail.access_ptr("rw", offset=self.tail.elem_offset_of([T.int32(0)])),
                        enqueue_num,
                        rank,
                    )
                self.tail_r = T.tvm_warp_shuffle(0xFFFFFFFF, self.tail_r, 0, 32, 32)
            else:
                if tid_in_scope == 0:
                    self.tail_smem[scope_idx] = atomic_add_int32(
                        self.tail.access_ptr("rw", offset=self.tail.elem_offset_of([T.int32(0)])),
                        enqueue_num,
                        rank,
                    )
                if level == "warpgroup":
                    T.ptx.bar.sync(6 + wg_id, 128)
                elif level == "cta":
                    T.tvm_storage_sync("shared")
                self.tail_r = self.tail_smem[scope_idx]

            self.idx = tid_in_scope
            while self.idx < enqueue_num:
                self.masked_pos = (self.tail_r + self.idx) & self.mask
                task_type, _, m_idx, n_idx, k_idx = func_push(self.idx)
                task_info = T.meta_var(
                    pack_into_32bit(m_idx, n_idx, k_idx, task_type, host=False, debug=self.debug)
                )
                stg(
                    task_info,
                    self.tasks.access_ptr(
                        "rw", offset=self.tasks.elem_offset_of([self.masked_pos])
                    ),
                    rank,
                )
                self.idx += tid_stride

    @T.inline
    def dequeue(self, fetched_task_info):
        self.head_r = T.cuda.atomic_add(
            self.head.access_ptr("rw", offset=self.head.elem_offset_of([T.int32(0)])), 1
        )
        self.masked_pos = self.head_r & self.mask
        while_ld_global_acquire(
            self.tasks.access_ptr("r", offset=self.tasks.elem_offset_of([self.masked_pos])),
            T.address_of(fetched_task_info),
        )
        # FIXME: enable this when we consider capacity issue
        # self.tasks[self.masked_pos, 0] = -1


class DynamicTileScheduler(TileSchedulerBase):
    MAX_TASKS = 32768
    scheduler_warp = 7

    def __init__(
        self,
        tasks: T.Buffer,
        head: T.Buffer,
        tail: T.Buffer,
        smem_manager,
        profiler: CudaProfiler = None,
        debug=False,
    ):
        self.queue = MPMCQueue(
            capacity=self.MAX_TASKS, tasks=tasks, head=head, tail=tail, smem_manager=smem_manager
        )
        self.profiler_on = profiler is not None
        self.profiler = profiler
        self.debug = debug
        self.smem_manager = smem_manager

    def _alloc(self):
        self.task_info = T.local_scalar(dtype="int32")
        self.task_type = T.local_scalar(dtype="int32")
        self.m_idx = T.local_scalar(dtype="int32")
        self.n_idx = T.local_scalar(dtype="int32")
        self.k_idx = T.local_scalar(dtype="int32")
        self.idx = T.local_scalar(dtype="int32")
        self.dequeue_phase = T.local_scalar(dtype="int32")
        self.p2c_dequeue_barrier = SchedulerBarrier(self.smem_manager, is_p2c=True)
        self.c2p_dequeue_barrier = SchedulerBarrier(self.smem_manager, is_p2c=False)
        self.packed_value = self.smem_manager.alloc((1,), "int32", align=16, method="persistent")
        self.semaphore_state = self.smem_manager.alloc(
            (KernelConfig.NUM_THREADS,), "int32", method="persistent"
        )

    @T.inline
    def _dequeue_and_store_packed(self):
        self.queue.dequeue(self.task_info)
        sts(self.task_info, self.packed_value.ptr_to([0]))

    @T.inline
    def _fetch_from_queue(self):
        warp_id = T.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER])
        # fetch from GEMM queue
        if warp_id == self.scheduler_warp:
            if T.ptx.elect_sync():
                self.c2p_dequeue_barrier.wait(0, self.dequeue_phase)
                self._dequeue_and_store_packed()
                self.p2c_dequeue_barrier.arrive()
        self.p2c_dequeue_barrier.wait(0, self.dequeue_phase)
        unpack_from_32bit(
            self.packed_value[0],
            T.address_of(self.task_type),
            T.address_of(self.m_idx),
            T.address_of(self.n_idx),
            T.address_of(self.k_idx),
        )
        self.c2p_dequeue_barrier.arrive()
        self.dequeue_phase = self.dequeue_phase ^ 1

    @T.inline
    def init(self):
        tid = T.thread_id([KernelConfig.NUM_THREADS])
        self._alloc()
        self.queue.init()
        self.dequeue_phase = 0
        if tid == 0:
            self.p2c_dequeue_barrier.init(1)
            self.c2p_dequeue_barrier.init(KernelConfig.NUM_THREADS)
        T.tvm_storage_sync("shared")
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()

    @T.inline
    def next_tile(self):
        lane_id = T.lane_id([32])
        if self.profiler_on:
            self.profiler.start(ProfileEventType.FETCH, lane_id == 0)
        self._fetch_from_queue()
        if self.profiler_on:
            self.profiler.end(ProfileEventType.FETCH, lane_id == 0)

    def get_idx_and_task_type(self):
        return [self.m_idx, self.n_idx, self.k_idx], self.task_type

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
        pre_notify=False,
    ):
        # Notes: Here each thread will notify only at most one time,
        #        and the tids of the threads involved among scope in the notification process start from 0 and increment sequentially.
        # Notes: (notify_num, rank, coord) = func_notify(notify_idx), rank=-1 for the local rank
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
            if not pre_notify:
                sync(scope, scope_id)
            notify_num = T.meta_var(func_notify(idx[1])[0])
            rank = T.meta_var(func_notify(idx[1])[1])
            coord = T.meta_var(func_notify(idx[1])[2:])
            if self.debug:
                T.cuda.trap_when_assert_failed(notify_num <= max_notify_num_map[scope])
            if idx[1] < notify_num:
                evt.semaphore_notify(*coord, pre_notify=pre_notify, rank=rank)

    def _enqueue(self, idx, func_trigger_list, push_level):
        if not isinstance(func_trigger_list, list):
            func_trigger_list = [func_trigger_list]
        for func_trigger in func_trigger_list:
            self.queue.enqueue(-1, func_trigger(idx), push_level)

    @T.inline
    def pre_notify_and_push(
        self,
        evt: Semaphore,
        func_notify,
        func_trigger_list,
        push_level: Literal["thread", "warp", "warpgroup", "cta"],
        scope: Literal["thread", "warp", "warpgroup", "cta"],
        scope_id=0,
    ):
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

        wg_id = T.warpgroup_id([KernelConfig.WG_NUMBER])
        warp_id = T.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER])
        tid = T.thread_id([KernelConfig.NUM_THREADS])
        tid_in_wg = T.thread_id_in_wg([KernelConfig.NUM_THREADS // KernelConfig.WG_NUMBER])
        warp_id_in_wg = T.warp_id_in_wg([KernelConfig.WARP_NUMBER])
        lane_id = T.lane_id([32])
        idx_map = T.meta_var(
            {
                "thread": (tid, 0),
                "warp": (warp_id, lane_id),
                "warpgroup": (wg_id, tid_in_wg),
                "cta": (0, tid),
            }
        )
        idx_in_scope_map = T.meta_var(
            {
                "thread": {"thread": 0},
                "warp": {"thread": lane_id, "warp": 0},
                "warpgroup": {"thread": tid_in_wg, "warp": warp_id_in_wg, "warpgroup": 0},
                "cta": {"thread": tid, "warp": warp_id, "warpgroup": wg_id, "cta": 0},
            }
        )
        stride_in_scope_map = T.meta_var(
            {
                "warp": {"warp": 1},
                "warpgroup": {"warp": KernelConfig.WARP_NUMBER, "warpgroup": 1},
                "cta": {
                    "warp": KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER,
                    "warpgroup": KernelConfig.WG_NUMBER,
                    "cta": 1,
                },
            }
        )
        scope_id_map = T.meta_var({"thread": tid, "warp": warp_id, "warpgroup": wg_id, "cta": 0})
        new_scope_id = T.if_then_else(scope_id == -1, scope_id_map[scope], scope_id)
        idx = T.meta_var(idx_map[scope])
        if self.debug:
            T.cuda.trap_when_assert_failed(scope_id == -1 or scope_id < max_scope_id_map[scope])
        if idx[0] == new_scope_id:
            notify_num = T.meta_var(func_notify(idx[1])[0])
            rank = T.meta_var(func_notify(idx[1])[1])
            coord_notify = T.meta_var(func_notify(idx[1])[2:])
            if self.debug:
                T.cuda.trap_when_assert_failed(notify_num <= max_notify_num_map[scope])
            if idx[1] < notify_num:
                evt.semaphore_notify(*coord_notify, pre_notify=True, rank=rank)
            if self.profiler_on:
                self.profiler.start(ProfileEventType.PUSH, lane_id == 0)
            if scope == "thread":
                if tid == new_scope_id:
                    if push_level == "thread":
                        if evt.is_triggered():
                            self._enqueue(0, func_trigger_list, push_level)
                    else:
                        assert False
            elif scope == "warp":
                if warp_id == new_scope_id:
                    if push_level == "thread":
                        if lane_id < notify_num:
                            if evt.is_triggered():
                                self._enqueue(lane_id, func_trigger_list, push_level)
                    elif push_level == "warp":
                        self.semaphore_state[tid] = evt.state[0]
                        T.cuda.warp_sync()
                        self.idx = idx_in_scope_map[scope][push_level]
                        while self.idx < notify_num:
                            evt.state[0] = self.semaphore_state[new_scope_id * 32 + self.idx]
                            if evt.is_triggered():
                                self._enqueue(self.idx, func_trigger_list, push_level)
                            self.idx += stride_in_scope_map[scope][push_level]
                    else:
                        assert False
            elif scope == "warpgroup":
                if wg_id == new_scope_id:
                    if push_level == "thread":
                        if tid_in_wg < notify_num:
                            if evt.is_triggered():
                                self._enqueue(tid_in_wg, func_trigger_list, push_level)
                    elif push_level == "warp" or push_level == "warpgroup":
                        self.semaphore_state[tid] = evt.state[0]
                        T.cuda.warpgroup_sync(6 + wg_id)
                        self.idx = idx_in_scope_map[scope][push_level]
                        while self.idx < notify_num:
                            evt.state[0] = self.semaphore_state[
                                new_scope_id * KernelConfig.NUM_THREADS // KernelConfig.WG_NUMBER
                                + self.idx
                            ]
                            if evt.is_triggered():
                                self._enqueue(self.idx, func_trigger_list, push_level)
                            self.idx += stride_in_scope_map[scope][push_level]
                    else:
                        assert False
            elif scope == "cta":
                if push_level == "thread":
                    if tid < notify_num:
                        if evt.is_triggered():
                            self._enqueue(tid, func_trigger_list, push_level)
                elif push_level == "warp" or push_level == "warpgroup" or push_level == "cta":
                    self.semaphore_state[tid] = evt.state[0]
                    T.tvm_storage_sync("shared")
                    self.idx = idx_in_scope_map[scope][push_level]
                    while self.idx < notify_num:
                        evt.state[0] = self.semaphore_state[self.idx]
                        if evt.is_triggered():
                            self._enqueue(self.idx, func_trigger_list, push_level)
                        self.idx += stride_in_scope_map[scope][push_level]
                else:
                    assert False
            else:
                assert False
            if self.profiler_on:
                self.profiler.end(ProfileEventType.PUSH, lane_id == 0)

    def valid(self):
        return self.task_type != JobType.END.value


class MPMCQueueHost:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tasks = np.full((capacity,), -1, dtype=np.int32)
        self.head = np.zeros((1,), dtype=np.int32)
        self.tail = np.zeros((1,), dtype=np.int32)
        self.head[0] = 0
        self.tail[0] = 0

    def enqueue(self, task_type, m_idx, n_idx, k_idx):
        pos = self.tail[0] & (self.capacity - 1)
        self.tasks[pos] = pack_into_32bit(m_idx, n_idx, k_idx, task_type)
        self.tail[0] = self.tail[0] + 1

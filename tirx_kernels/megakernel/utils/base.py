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
"""Base abstract classes for megakernel."""

import functools
from typing import Literal

import tvm
from tvm.script import ir as I
from tvm.script import tirx as T
from tvm.tirx import Expr
from tvm.tirx.bench import CudaProfiler
from tvm.tirx.expr import Var

from .config import KernelConfig, ProfileEventType
from .utils import any_sync, f_init_const


@T.meta_class
class Tile:
    """Abstract base class for megakernel tiles."""

    need_init = True

    @classmethod
    def class_init(cls, smem_manager):
        pass

    @classmethod
    def class_finalize(cls):
        pass

    def __init__(self):
        self._instance_id = id(self)

    def __str__(self):
        class_name = self.__class__.__name__
        return f"{class_name}-{self._instance_id:x}"

    def init(self, smem_manager):
        pass

    def host_init(self):
        pass

    def run(self, m_idx, n_idx, k_idx):
        raise NotImplementedError("run is not implemented")

    def prefetch(self, m_idx, n_idx, k_idx):
        raise NotImplementedError("prefetch is not implemented")


@T.meta_class
class Barriers:
    """Mbarrier wrapper class"""

    def __init__(self, smem_manager, pipe_depth, is_p2c, persistent=True):
        self.smem_manager = smem_manager
        self.init_phase = 0 if is_p2c else 1
        self.pipe_depth = pipe_depth
        self.persistent = persistent

    def _alloc(self):
        self.mbar = self.smem_manager.alloc(
            (self.pipe_depth,), "uint64", method="persistent" if self.persistent else "shared"
        )

    @T.inline
    def init(self, threads_num_wait):
        tid = T.thread_id([KernelConfig.NUM_THREADS])
        self._alloc()
        if self.pipe_depth == 1:
            if tid == 0:
                T.ptx.mbarrier.init(self.mbar.ptr_to([0]), threads_num_wait)
        elif tid == 0:
            for i in T.serial(self.pipe_depth):
                T.ptx.mbarrier.init(self.mbar.ptr_to([i]), threads_num_wait)

    @T.inline
    def wait(self, idx, phase):
        T.ptx.mbarrier.try_wait(self.mbar.ptr_to([idx]), self.init_phase ^ phase)


@T.meta_class
class SmemManager:
    """Shared memory manager"""

    def __init__(self, smem_max_bytes, chunk_size, ptr: Var, fusion_mode=False):
        self.smem_max_bytes = smem_max_bytes
        self.chunk_size = chunk_size
        self.chunk_num = smem_max_bytes // chunk_size
        assert self.chunk_num <= 32
        self.ptr = ptr
        self.reguler_pool_allocator = T.SMEMPool(ptr)
        self.persistent_pool_allocator = T.SMEMPool(None if fusion_mode else ptr)
        self.tiles = {}
        self.runtime_tile_chunk_count = {}
        self.bufs = {}
        self.persistent_bufs = {}
        self.cur_tile_name = ""
        self.persistent_pool_allocator.move_base_to(self.chunk_size * self.chunk_num)
        self.fusion_mode = fusion_mode
        self._pool_allocators = {
            "persistent": self.persistent_pool_allocator,
            "shared": self.reguler_pool_allocator,
            "exclusive": self.reguler_pool_allocator,
        }

    @property
    def pool_allocator(self):
        return self.reguler_pool_allocator

    def _inner_alloc(self):
        self.mbar = self.alloc((self.chunk_num,), "uint64", method="persistent")
        self.shared_count = self.alloc((1,), "int32", method="persistent")
        if self.fusion_mode:
            self.cur_phase = T.alloc_local([1], "int32", scope="local.persistent")
            self.reg_count = T.alloc_local([1], "int32", scope="local.persistent")
        else:
            self.cur_phase = T.alloc_local([1], "int32")
            self.reg_count = T.alloc_local([1], "int32")

    @T.inline
    def init(self):
        tid = T.thread_id([KernelConfig.NUM_THREADS])
        self.check_smem_well_formed(debug=False)
        self._inner_alloc()
        self.cur_phase[0] = 1
        if tid == 0:
            for i in T.serial(self.chunk_num):
                T.ptx.mbarrier.init(self.mbar.ptr_to([i]), 1)
            self.shared_count[0] = 0
        T.tvm_storage_sync("shared")
        T.ptx.fence.mbarrier_init()
        T.ptx.fence.proxy_async("shared::cta")

    def alloc(
        self,
        shape,
        dtype="float32",
        strides=None,
        scope="shared.dyn",
        align=1,
        layout="default",
        split=1,
        method: Literal["shared", "exclusive", "persistent"] = "shared",
    ):
        assert "shared" in scope
        pool_allocator = self._pool_allocators[method]
        beg = pool_allocator.offset
        if align > 0:
            beg = (beg + align - 1) // align * align
        if self.fusion_mode and method == "persistent":
            scope = "shared.persistent"
        buf = pool_allocator.alloc(shape, dtype, strides, scope, align, layout)
        end = pool_allocator.offset
        size = end - beg
        assert size % split == 0
        if method == "persistent":
            self.persistent_bufs[buf] = (beg, end)
        else:
            if method == "shared":
                assert len(self.tiles[self.cur_tile_name][1]["exclusive"]) == 0, (
                    "Cannot use both shared and shared/exclusive methods at the same time"
                )
            elif method == "exclusive":
                assert len(self.tiles[self.cur_tile_name][1]["shared"]) == 0, (
                    "Cannot use both shared and shared/exclusive methods at the same time"
                )
            buf_info = (split, beg, size, method)
            self.tiles[self.cur_tile_name][0] = max(
                self.tiles[self.cur_tile_name][0], (end - 1) // self.chunk_size
            )
            self.tiles[self.cur_tile_name][1][method].append(buf_info)
            self.bufs[buf] = buf_info
            if method == "exclusive":
                for split_idx in range(split):
                    beg_chunk_id = (beg + size // split * split_idx) // self.chunk_size
                    end_chunk_id = (beg + size // split * (split_idx + 1) - 1) // self.chunk_size
                    for chunk_id in range(beg_chunk_id, end_chunk_id + 1):
                        self.tiles[self.cur_tile_name][2][chunk_id] += 1
        return buf

    def check_smem_well_formed(self, debug=False):
        for _, buf_info_dict, _ in self.tiles.values():
            checked_exclusive = []
            check_overlap = []
            for method in ["shared", "exclusive"]:
                for split, beg, size, _ in buf_info_dict[method]:
                    end = beg + size
                    assert end <= self.chunk_num * self.chunk_size
                    for beg_other, end_other in check_overlap:
                        assert beg >= end_other or beg_other >= end, (
                            "Overlap detected in smem allocation"
                        )
                    check_overlap.append((beg, end))
                    if method == "exclusive":
                        for split_idx in range(split):
                            beg_chunk_id = (beg + size // split * split_idx) // self.chunk_size
                            end_chunk_id = (
                                beg + size // split * (split_idx + 1) - 1
                            ) // self.chunk_size
                            for beg_id, end_id in checked_exclusive:
                                assert beg_id > end_chunk_id or end_id < beg_chunk_id, (
                                    "Exclusive chunk overlap detected"
                                )
                            checked_exclusive.append((beg_chunk_id, end_chunk_id))
                    else:
                        beg_chunk_id = beg // self.chunk_size
                        end_chunk_id = (end - 1) // self.chunk_size
                        checked_exclusive.append((beg_chunk_id, end_chunk_id))
        for beg_persistent, end_persistent in self.persistent_bufs.values():
            assert (
                beg_persistent >= self.chunk_size * self.chunk_num
                and end_persistent <= self.smem_max_bytes
            )
            for _, beg, size, _ in self.bufs.values():
                assert beg >= end_persistent or beg_persistent >= beg + size
        persistent_buf_list = list(self.persistent_bufs.values())
        for i in range(len(persistent_buf_list)):
            beg_i, end_i = persistent_buf_list[i]
            for j in range(i + 1, len(persistent_buf_list)):
                beg_j, end_j = persistent_buf_list[j]
                assert beg_i >= end_j or beg_j >= end_i
        if debug:
            self._debug_print()

    def _debug_print(self):
        for k, v in self.tiles.items():
            print(k, v)
        for k, v in self.bufs.items():
            print(k, v)
        for k, v in self.persistent_bufs.items():
            print(k, v)

    def set_tile(self, cur_tile: Tile | None):
        if cur_tile is None:
            self.cur_tile_name = "default"
        else:
            self.cur_tile_name = str(cur_tile)
        self.tiles[self.cur_tile_name] = [
            -1,
            {"exclusive": [], "shared": []},
            [0 for _ in range(self.chunk_num)],
        ]
        self.runtime_tile_chunk_count[self.cur_tile_name] = [
            [0 for _ in range(self.chunk_num)] for _ in range(2)
        ]
        self.reguler_pool_allocator.move_base_to(0)

    def _assert_cond(self, cond):
        assert cond

    @T.inline
    def advance(self):
        self.cur_phase[0] = self.cur_phase[0] ^ 1

    def enter_tile_runtime(self, cur_tile: Tile):
        self.cur_tile_name = str(cur_tile)

    def exit_tile_runtime(self):
        self._check_runtime()
        self.cur_tile_name = ""

    def _check_runtime(self):
        pass

    @T.inline
    def wait_all(self, level: Literal["cta", "warpgroup"] = "cta"):
        lane_id = T.lane_id([32])
        warp_id = T.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER])
        wg_id = T.warpgroup_id([KernelConfig.WG_NUMBER])
        if level == "cta":
            if warp_id == 0:
                if lane_id < self.chunk_num:
                    T.ptx.mbarrier.try_wait(self.mbar.ptr_to([lane_id]), self.cur_phase[0])
            T.tvm_storage_sync("shared")
        elif level == "warpgroup":
            if warp_id % KernelConfig.WARP_NUMBER == 0:
                if lane_id < self.chunk_num:
                    T.ptx.mbarrier.try_wait(self.mbar.ptr_to([lane_id]), self.cur_phase[0])
            T.ptx.bar.sync(6 + wg_id, 128)

    @T.inline
    def wait_specific(self, lane_id, buffer, split_idx: int):
        self._assert_cond(buffer in self.bufs and buffer not in self.persistent_bufs)
        self._assert_cond(self.bufs[buffer][3] == "exclusive")
        beg_chunk_id = (
            self.bufs[buffer][1] + self.bufs[buffer][2] // self.bufs[buffer][0] * split_idx
        ) // self.chunk_size
        end_chunk_id = (
            self.bufs[buffer][1]
            + self.bufs[buffer][2] // self.bufs[buffer][0] * (split_idx + 1)
            - 1
        ) // self.chunk_size
        if (lane_id >= beg_chunk_id) & (lane_id <= end_chunk_id):
            T.ptx.mbarrier.try_wait(self.mbar.ptr_to([lane_id]), self.cur_phase[0])

    @T.inline
    def wait_unused(self, lane_id, cur_tile: Tile):
        self._assert_cond(len(self.tiles[self.cur_tile_name][1]["shared"]) == 0)
        if (lane_id < self.chunk_num) & (lane_id > self.tiles[str(cur_tile)][0]):
            T.ptx.mbarrier.try_wait(self.mbar.ptr_to([lane_id]), self.cur_phase[0])

    @T.inline
    def wait_chunk(self, chunk_id):
        T.ptx.mbarrier.try_wait(self.mbar.ptr_to([chunk_id]), self.cur_phase[0])

    @T.inline
    def wait_specific_one_thread(self, buffer, split_idx: int):
        self._assert_cond(buffer in self.bufs and buffer not in self.persistent_bufs)
        self._assert_cond(self.bufs[buffer][3] == "exclusive")
        beg_chunk_id = (
            self.bufs[buffer][1] + self.bufs[buffer][2] // self.bufs[buffer][0] * split_idx
        ) // self.chunk_size
        end_chunk_id = (
            self.bufs[buffer][1]
            + self.bufs[buffer][2] // self.bufs[buffer][0] * (split_idx + 1)
            - 1
        ) // self.chunk_size
        for idx in T.serial(0, end_chunk_id - beg_chunk_id + 1):
            T.ptx.mbarrier.try_wait(self.mbar.ptr_to([beg_chunk_id + idx]), self.cur_phase[0])

    @T.inline
    def arrive_all(self, level: Literal["cta", "warpgroup"] = "cta"):
        lane_id = T.lane_id([32])
        warp_id = T.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER])
        wg_id = T.warpgroup_id([KernelConfig.WG_NUMBER])
        if level == "cta":
            T.tvm_storage_sync("shared")
            if warp_id == 0:
                if lane_id < self.chunk_num:
                    T.ptx.mbarrier.arrive(self.mbar.ptr_to([lane_id]))
        elif level == "warpgroup":
            self.reg_count[0] = 0
            T.ptx.bar.sync(6 + wg_id, 128)
            if warp_id % KernelConfig.WARP_NUMBER == 0:
                if lane_id == 0:
                    self.reg_count[0] = T.cuda.atomic_add(T.address_of(self.shared_count[0]), 1) + 1
                    if self.reg_count[0] == KernelConfig.WG_NUMBER:
                        T.cuda.atomic_add(
                            T.address_of(self.shared_count[0]), -KernelConfig.WG_NUMBER
                        )
                self.reg_count[0] = T.tvm_warp_shuffle(4294967295, self.reg_count[0], 0, 32, 32)
                if self.reg_count[0] == KernelConfig.WG_NUMBER:
                    if lane_id < self.chunk_num:
                        T.ptx.mbarrier.arrive(self.mbar.ptr_to([lane_id]))

    @T.inline
    def arrive_specific(self, lane_id, buffer, split_idx: int):
        self._assert_cond(buffer in self.bufs and buffer not in self.persistent_bufs)
        self._assert_cond(self.bufs[buffer][3] == "exclusive")
        beg_chunk_id = (
            self.bufs[buffer][1] + self.bufs[buffer][2] // self.bufs[buffer][0] * split_idx
        ) // self.chunk_size
        end_chunk_id = (
            self.bufs[buffer][1]
            + self.bufs[buffer][2] // self.bufs[buffer][0] * (split_idx + 1)
            - 1
        ) // self.chunk_size
        if (lane_id >= beg_chunk_id) & (lane_id <= end_chunk_id):
            T.ptx.mbarrier.arrive(self.mbar.ptr_to([lane_id]))

    @T.inline
    def arrive_unused(self, lane_id, cur_tile: Tile):
        self._assert_cond(len(self.tiles[self.cur_tile_name][1]["shared"]) == 0)
        if (lane_id < self.chunk_num) & (lane_id > self.tiles[str(cur_tile)][0]):
            T.ptx.mbarrier.arrive(self.mbar.ptr_to([lane_id]))

    @T.inline
    def arrive_chunk(self, chunk_id):
        T.ptx.mbarrier.arrive(self.mbar.ptr_to([chunk_id]))


@T.meta_class
class SemaphoreBase:
    """Abstract base class for semaphore."""

    base = 1 << 16

    def __init__(self):
        pass

    def semaphore_wait(self, *coord, level, mask):
        raise NotImplementedError

    def semaphore_notify(self, *coord, rank=-1):
        raise NotImplementedError


@T.meta_class
class TileSchedulerBase:
    """Abstract base class for tile schedulers."""

    MAX_TASKS = 128

    def __init__(self):
        pass

    def get_idx_and_task_type(self) -> tuple[list[Expr], Expr]:
        raise NotImplementedError

    @T.inline
    def init(self):
        raise NotImplementedError

    @T.inline
    def next_tile(self):
        raise NotImplementedError

    @T.inline
    def wait(self, evt, *coord, wait_level, mask):
        raise NotImplementedError

    @T.inline
    def notify(self, evt, func_notify, scope, scope_id, release):
        raise NotImplementedError

    @T.inline
    def pre_notify_and_push(self, evt, func_notify, func_trigger_list, push_level, scope, scope_id):
        pass

    def valid(self):
        raise NotImplementedError


class InitETensorTile(Tile):
    VEC_SIZE = 1

    def __init__(self, etensor_and_f_init_pairs):
        super().__init__()
        self.etensor_and_f_init_pairs = etensor_and_f_init_pairs
        self.total_num_etensors = len(etensor_and_f_init_pairs)

    def convert_1d_index_to_nd(self, idx, shape):
        nd_idx = []
        for i in reversed(range(len(shape))):
            nd_idx.append(idx % shape[i])
            idx = idx // shape[i]
        return list(reversed(nd_idx))

    def run(self, m_idx, n_idx, k_idx):
        tid = T.thread_id([KernelConfig.NUM_THREADS])
        if_frames = [T.If(m_idx == i) for i in range(self.total_num_etensors)]
        then_frames = [T.Then() for i in range(self.total_num_etensors)]
        else_frames = [T.Else() for i in range(self.total_num_etensors - 1)]
        idx = T.alloc_local([1], "int32")
        T.buffer_store(idx, tid * self.VEC_SIZE, [0])
        for i in range(self.total_num_etensors):
            if_frames[i].__enter__()
            with then_frames[i]:
                etensor, f_init = self.etensor_and_f_init_pairs[i]
                if f_init is None:
                    T.evaluate(0)
                else:
                    nelem = functools.reduce(lambda x, y: x * y, etensor.shape, 1)
                    etensor_1d = etensor.view(-1)
                    with T.While(idx[0] < nelem):
                        with T.vectorized(self.VEC_SIZE) as v:
                            T.buffer_store(
                                etensor_1d,
                                f_init(*self.convert_1d_index_to_nd(idx[0] + v, etensor.shape))
                                * (SemaphoreBase.base + 1),
                                idx[0] + v,
                            )
                        T.buffer_store(idx, idx[0] + KernelConfig.NUM_THREADS * self.VEC_SIZE, [0])
            if i < self.total_num_etensors - 1:
                else_frames[i].__enter__()
        for i in range(self.total_num_etensors - 1, -1, -1):
            if i < self.total_num_etensors - 1:
                else_frames[i].__exit__(None, None, None)
            if_frames[i].__exit__(None, None, None)


class MegaKernelWrapper:
    """Base class for megakernel wrappers."""

    ETENSOR_WORKSPACE_SIZE = 1024 * 1024
    NUM_GROUPS = KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER
    PROFILER_BUFFER_SIZE = int(10000000.0)
    PROFILER_WRITE_STRIDE = KernelConfig.SM_NUMBER * NUM_GROUPS

    def __init__(self, config: dict = {}, tp_size: int = 1, profiler_on: bool = False):
        self.tp_size = tp_size
        self.config = config
        self.profiler_on = profiler_on
        self.tile_attr = {}
        self.class_list = set()
        self.etensor_and_f_init_pairs = []
        self.num_etensors = {}
        self.etensor_workspace_offset = 0

    def _init_profiler(self, profiler_buffer):
        if self.profiler_on:
            self.profiler = CudaProfiler(
                profiler_buffer, write_stride=self.PROFILER_WRITE_STRIDE, num_groups=self.NUM_GROUPS
            )
        else:
            self.profiler = None

    def _init_tile_scheduler(self, scheduler_class: type[TileSchedulerBase], *args):
        self.tile_scheduler: TileSchedulerBase = scheduler_class(*args)

    def _add_tile(self, tile, profiler_event_type, predicate=True):
        self.tile_attr[tile] = (profiler_event_type, predicate)
        self.class_list.add(tile.__class__)
        return tile

    @T.inline
    def init_profiler(self, profiler_buffer):
        self._init_profiler(profiler_buffer)
        warp_id = T.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER])
        if self.profiler_on:
            self.profiler.init(warp_id)

    def set_smem_manager(self, smem_max_bytes, chunk_size, ptr: Var):
        self.smem_manager = SmemManager(smem_max_bytes, chunk_size, ptr)

    @T.inline
    def init_tile_scheduler(self, is_dynamic_sch, scheduler_class, *args):
        self._init_tile_scheduler(scheduler_class, *args)
        self.tile_scheduler.init()
        if is_dynamic_sch:
            self.tile_scheduler.next_tile()

    @T.inline
    def run_tile(self, tile: Tile, *args, **kwargs):
        event_type = T.meta_var(self.tile_attr[tile][0])
        self.smem_manager.enter_tile_runtime(tile)
        lane_id = T.lane_id([32])
        if self.profiler_on:
            self.profiler.start(event_type, lane_id == 0)
        tile.run(*args, **kwargs)
        if self.profiler_on:
            self.profiler.end(event_type, lane_id == 0)

    @T.inline
    def run_tile_prefetch(self, tile: Tile, *args):
        self.smem_manager.enter_tile_runtime(tile)
        lane_id = T.lane_id([32])
        if self.profiler_on:
            self.profiler.start(ProfileEventType.PREFETCH, lane_id == 0)
        tile.prefetch(*args)
        if self.profiler_on:
            self.profiler.end(ProfileEventType.PREFETCH, lane_id == 0)

    def add_etensor(self, sem_class, etensor_workspace, shape, f_init):
        size = functools.reduce(lambda x, y: x * y, shape, 1)
        etensor_buffer = T.decl_buffer(
            shape, "int32", etensor_workspace.data, elem_offset=self.etensor_workspace_offset
        )
        self.etensor_workspace_offset += size
        etensor = sem_class(etensor_buffer)
        self.etensor_and_f_init_pairs.append((etensor_buffer, f_init))
        return etensor

    def set_events_complete(
        self, is_dynamic_sch, Semaphore: type[SemaphoreBase], etensor_workspace_global
    ):
        if not is_dynamic_sch:
            num_evtensors = len(self.etensor_and_f_init_pairs)
            self.evt_etensor_init_complete = self.add_etensor(
                Semaphore,
                etensor_workspace_global,
                shape=[1],
                f_init=f_init_const(num_evtensors + 1 + KernelConfig.SM_NUMBER),
            )
        else:
            self.evt_etensor_init_complete = None
        self.init_etensor_tile = self._add_tile(
            InitETensorTile(self.etensor_and_f_init_pairs), ProfileEventType.INIT_ETENSOR
        )

    @T.inline
    def task_impl_init_etensor(self, is_dynamic_sch):
        self.run_tile(
            self.init_etensor_tile,
            self.tile_scheduler.m_idx,
            self.tile_scheduler.n_idx,
            self.tile_scheduler.k_idx,
        )
        if self.evt_etensor_init_complete is not None:
            if self.tile_scheduler.m_idx < len(self.etensor_and_f_init_pairs):
                self.tile_scheduler.notify(
                    self.evt_etensor_init_complete,
                    lambda notify_idx: (1, -1, 0),
                    scope="cta",
                    release=True,
                )

    @T.inline
    def task_impl_wait_etensor_init_complete(self, is_dynamic_sch):
        if not is_dynamic_sch:
            warp_id = T.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER])
            lane_id = T.lane_id([32])
            if self.profiler_on:
                self.profiler.start(ProfileEventType.WAIT_ETENSOR_INIT, lane_id == 0)
            state = T.alloc_local([1], "int32")
            state[0] = -1
            while 1:
                if lane_id == 0:
                    T.ptx.ld_global_acquire(
                        state[0], self.evt_etensor_init_complete.sem.ptr_to([0])
                    )
                if any_sync(
                    4294967295,
                    state[0] <= KernelConfig.SM_NUMBER * (SemaphoreBase.base + 1) and state[0] > 0,
                ):
                    if (lane_id == 0) & (warp_id == 0):
                        T.cuda.atomic_add(
                            self.evt_etensor_init_complete.sem.ptr_to([0]),
                            -(SemaphoreBase.base + 1),
                        )
                    break
                T.cuda.nano_sleep(40)
            if self.profiler_on:
                self.profiler.end(ProfileEventType.WAIT_ETENSOR_INIT, lane_id == 0)

    def reset(self):
        self.tile_attr = {}
        self.class_list = set()
        self.etensor_and_f_init_pairs = []
        self.etensor_workspace_offset = 0

    def host_init_all(self):
        for tile, (_, predicate) in self.tile_attr.items():
            if predicate:
                tile.host_init()

    def class_init_all(self, smem_manager: SmemManager):
        for cls in self.class_list:
            if cls.need_init:
                smem_manager.set_tile(cls)
                cls.class_init(smem_manager)

    def class_finalize_all(self):
        for cls in self.class_list:
            if cls.need_init:
                cls.class_finalize()

    def device_init_all(self, smem_manager: SmemManager):
        for tile, (_, predicate) in self.tile_attr.items():
            if predicate:
                smem_manager.set_tile(tile)
                tile.init(smem_manager)

    def get_func_static(self, unfused=False):
        raise NotImplementedError

    def get_func_dynamic(self):
        raise NotImplementedError

    def get_func(self, scheduler: Literal["static", "dynamic", "unfused"]):
        if scheduler == "static" or scheduler == "unfused":
            return self.get_func_static(unfused=scheduler == "unfused")
        elif scheduler == "dynamic":
            return self.get_func_dynamic()
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler}")

    def get_module(self, scheduler: Literal["static", "dynamic", "unfused"]):
        @I.ir_module
        class Module:
            @T.prim_func
            def main():
                pass

        module: tvm.IRModule = Module
        if scheduler == "static" or scheduler == "unfused":
            module.update_func(
                module.get_global_var("main"), self.get_func_static(unfused=scheduler == "unfused")
            )
        elif scheduler == "dynamic":
            module.update_func(module.get_global_var("main"), self.get_func_dynamic())
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler}")
        return module

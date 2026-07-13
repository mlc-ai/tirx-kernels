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
from typing import Literal

import tvm
from tirx_kernels.megakernel.utils.base import Barriers, SmemManager, Tile
from tirx_kernels.megakernel.utils.config import (
    F16_BYTES,
    F32_BYTES,
    KernelConfig,
    ProfileEventType,
)
from tirx_kernels.megakernel.utils.utils import ceildiv, mbarrier_try_wait
from tvm.backend.cuda.operator.tile_primitive.tma_utils import SwizzleMode, mma_shared_layout
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.bench import CudaProfiler
from tvm.tirx.layout import S, TCol, TileLayout, TLane
from tvm.tirx.layout import tid_in_wg as axis_tid_in_wg


class BarTMA2MMA(Barriers):
    @T.inline
    def arrive(self, idx, expected_bytes):
        T.ptx.mbarrier.arrive.expect_tx(self.mbar.ptr_to([idx]), expected_bytes)

    @T.inline
    def arrive_only(self, idx):
        T.ptx.mbarrier.arrive(self.mbar.ptr_to([idx]))


class BarMMA2LD(Barriers):
    @T.inline
    def arrive(self, idx):
        T.ptx.tcgen05.commit(self.mbar.ptr_to([idx]), cta_group=KernelConfig.CTA_GROUP)


class BarMMA2TMA(Barriers):
    @T.inline
    def arrive(self, idx):
        T.ptx.tcgen05.commit(self.mbar.ptr_to([idx]), cta_group=KernelConfig.CTA_GROUP)


class BarLD2MMA(Barriers):
    @T.inline
    def arrive(self, idx):
        T.ptx.mbarrier.arrive(self.mbar.ptr_to([idx]), remote=0, pred=True)


class GemmTile(Tile):
    SMEM_PIPE_DEPTH = 6
    TMEM_PIPE_DEPTH = 2
    MAX_BLK_M, BLK_N, BLK_K = (128, 128, 64)
    MMA_N, MMA_K = (128, 16)
    EPI_TILE = 32
    TMEM_LD_SIZE = 8
    N_COLS = 512
    SWIZZLE = 3
    TMA2MMA_ARRIVE_COUNT = 1
    SMEM_SIZE = (
        SMEM_PIPE_DEPTH * MAX_BLK_M * BLK_K * F16_BYTES
        + SMEM_PIPE_DEPTH * BLK_N * BLK_K * F16_BYTES
        + TMEM_PIPE_DEPTH * EPI_TILE * MMA_N * F32_BYTES
        + 1024
    )
    assert SMEM_SIZE <= 232448
    assert TMEM_PIPE_DEPTH * MMA_N <= 512
    tile_idx = None

    def __init__(
        self,
        N,
        K,
        a_type,
        b_type,
        split_k_factor,
        BLK_M,
        MMA_M,
        out_type=None,
        use_tma_reduce=False,
        low_batch=True,
        prefetch_on=False,
        profiler_on=False,
    ):
        super().__init__()
        self.BLK_M = BLK_M
        self.MMA_M = MMA_M
        self.N = N
        self.K = K
        self.a_type = a_type
        self.b_type = b_type
        assert not (use_tma_reduce and split_k_factor == 1), (
            "use_tma_reduce when split_k_factor == 1 is not supported"
        )
        if out_type is None:
            self.out_type = "float32" if split_k_factor > 1 or use_tma_reduce else "float16"
        else:
            self.out_type = out_type
        self.split_k_factor = split_k_factor
        self.use_tma_reduce = use_tma_reduce
        self.low_batch = low_batch
        self.prefetch_on = prefetch_on
        self.profiler_on = profiler_on
        self.use_cp_async_input = False
        self.use_tma_gather4 = False
        self.TILE_K = ceildiv(ceildiv(self.K, self.split_k_factor), self.BLK_K) * self.BLK_K
        self.PIPE_CIRCLE_NUM = self.TILE_K // self.BLK_K // self.SMEM_PIPE_DEPTH
        self.PIPE_REMAIN_NUM = self.TILE_K // self.BLK_K % self.SMEM_PIPE_DEPTH
        self.M_pad_size = BLK_M
        self.A_layout = mma_shared_layout(
            a_type,
            SwizzleMode.SWIZZLE_128B_ATOM,
            (self.SMEM_PIPE_DEPTH, self.MAX_BLK_M, self.BLK_K),
        )
        self.B_layout = mma_shared_layout(
            b_type, SwizzleMode.SWIZZLE_128B_ATOM, (self.SMEM_PIPE_DEPTH, self.BLK_N, self.BLK_K)
        )
        self.D_layout = T.TileLayout(
            T.S[
                (self.TMEM_PIPE_DEPTH, self.EPI_TILE, self.MMA_N) : (
                    self.EPI_TILE * self.MMA_N,
                    self.MMA_N,
                    1,
                )
            ]
        )

    def _alloc_buffer(self, smem_manager: SmemManager):
        self.smem_manager = smem_manager
        self.A_smem = smem_manager.alloc(
            (self.SMEM_PIPE_DEPTH, self.MAX_BLK_M, self.BLK_K),
            self.a_type,
            layout=self.A_layout,
            align=1024,
            split=self.SMEM_PIPE_DEPTH,
            method="exclusive",
        )
        self.B_smem = smem_manager.alloc(
            (self.SMEM_PIPE_DEPTH, self.BLK_N, self.BLK_K),
            self.b_type,
            layout=self.B_layout,
            align=1024,
            split=self.SMEM_PIPE_DEPTH,
            method="exclusive",
        )
        self.output_smem = smem_manager.alloc(
            (self.TMEM_PIPE_DEPTH, self.EPI_TILE, self.MMA_N),
            self.out_type,
            layout=self.D_layout,
            align=1024,
            method="exclusive",
        )

    def _alloc_local(self, m_idx):
        self.reg = T.alloc_buffer((self.TMEM_LD_SIZE,), "float32", scope="local")
        if self.out_type == "float16":
            self.reg_fp16 = T.alloc_buffer((self.TMEM_LD_SIZE,), self.out_type, scope="local")
        self.tmem_idx = T.local_scalar("int32")
        self.tmem_phase = T.local_scalar("int32")
        self.stage = T.local_scalar("int32")
        self.wait_complete = T.local_scalar("bool")

    @classmethod
    def _alloc_buffer_class_member(cls, smem_manager: SmemManager):
        GemmTile.tmem_addr = smem_manager.alloc([1], "uint32", method="persistent")
        GemmTile.tma2mma_bar = BarTMA2MMA(smem_manager, cls.SMEM_PIPE_DEPTH, True)
        GemmTile.mma2tma_bar = BarMMA2TMA(smem_manager, cls.SMEM_PIPE_DEPTH, False)
        GemmTile.mma2ld_bar = BarMMA2LD(smem_manager, cls.TMEM_PIPE_DEPTH, True)
        GemmTile.ld2mma_bar = BarLD2MMA(smem_manager, cls.TMEM_PIPE_DEPTH, False)
        GemmTile.tile_idx = T.local_scalar("int32")
        GemmTile.phase = T.alloc_buffer((1,), "int32", scope="local")
        GemmTile.tmem = T.decl_buffer(
            (128, 512),
            "float32",
            scope="tmem",
            allocated_addr=0,
            layout=TileLayout(S[(128, 512) : (1 @ TLane, 1 @ TCol)]),
        )

    @classmethod
    @T.inline
    def class_init(cls, smem_manager: SmemManager):
        warp_id = T.warp_id([KernelConfig.WG_NUMBER * KernelConfig.WARP_NUMBER])
        cls._alloc_buffer_class_member(smem_manager)
        cls.tile_idx = 0
        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(cls.tmem_addr[0]), n_cols=cls.N_COLS, cta_group=1)
            T.cuda.warp_sync()
        cls.tma2mma_bar.init(cls.TMA2MMA_ARRIVE_COUNT)
        cls.mma2ld_bar.init(1)
        cls.mma2tma_bar.init(1)
        cls.ld2mma_bar.init(KernelConfig.CTA_GROUP * 128)
        cls.phase[0] = 0
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.tvm_storage_sync("shared")

    @classmethod
    @T.inline
    def class_finalize(cls):
        warp_id = T.warp_id([KernelConfig.WG_NUMBER * KernelConfig.WARP_NUMBER])
        T.tvm_storage_sync("shared")
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(cls.tmem_addr[0], n_cols=cls.N_COLS, cta_group=1)
        T.tvm_storage_sync("shared")

    @T.inline
    def init(self, smem_manager: SmemManager):
        self._alloc_buffer(smem_manager)

    @T.inline
    def host_init(self):
        pass

    @T.inline
    def _tma(self, ks, buf, buf_name: Literal["A", "B"], mn_st, k_st, tma_config, predicate=True):
        if predicate:
            if buf_name == "A":
                Tx.copy_async(
                    self.A_smem[ks, 0 : self.BLK_M, :],
                    buf[mn_st : mn_st + self.BLK_M, k_st : k_st + self.BLK_K],
                    **tma_config,
                )
            elif buf_name == "B":
                Tx.copy_async(
                    self.B_smem[ks, 0 : self.BLK_N, :],
                    buf[mn_st : mn_st + self.BLK_N, k_st : k_st + self.BLK_K],
                    **tma_config,
                )
            else:
                T.cuda.trap_when_assert_failed(False)

    @T.inline
    def _consumer_wg(self, m_idx, n_idx, k_idx, A, B, output, profiler: CudaProfiler):
        tid_in_wg = T.thread_id_in_wg([128])
        warp_id = T.warp_id_in_wg([KernelConfig.WARP_NUMBER])
        lane_id = T.lane_id([32])
        T.cuda.trap_when_assert_failed(self.tmem_addr[0] == 0)
        if warp_id == 0:
            self.smem_manager.wait_specific(lane_id, self.output_smem, 0)
        T.cuda.warpgroup_sync(10)
        self.phase[0] = 0
        self.tmem_idx = self.tile_idx % self.TMEM_PIPE_DEPTH
        self.tmem_phase = self.tile_idx // self.TMEM_PIPE_DEPTH & 1
        self.mma2ld_bar.wait(self.tmem_idx, self.tmem_phase)
        T.ptx.tcgen05.fence.after_thread_sync()
        for ko in T.unroll(self.MMA_M // self.EPI_TILE):
            self.stage = (self.tile_idx * self.MMA_M // self.EPI_TILE + ko) % self.TMEM_PIPE_DEPTH
            if ko >= self.TMEM_PIPE_DEPTH:
                if (lane_id == 0) & (warp_id == 0):
                    T.ptx.cp_async.bulk.wait_group(self.TMEM_PIPE_DEPTH - 1)
                T.cuda.warpgroup_sync(10)
            for ki in T.unroll(self.EPI_TILE // self.TMEM_LD_SIZE):
                reg_wg = self.reg.view(
                    128,
                    self.TMEM_LD_SIZE,
                    layout=TileLayout(S[(128, self.TMEM_LD_SIZE) : (1 @ axis_tid_in_wg, 1)]),
                )
                col_st = T.meta_var(
                    self.tmem_idx * self.M_pad_size + ko * self.EPI_TILE + ki * self.TMEM_LD_SIZE
                )
                Tx.wg.copy_async(reg_wg, self.tmem[:, col_st : col_st + self.TMEM_LD_SIZE])
                T.ptx.tcgen05.wait.ld()
                st = T.meta_var(ki * self.TMEM_LD_SIZE)
                if self.out_type == "float16":
                    reg_wg_fp16 = self.reg_fp16.view(
                        128,
                        self.TMEM_LD_SIZE,
                        layout=TileLayout(S[(128, self.TMEM_LD_SIZE) : (1 @ axis_tid_in_wg, 1)]),
                    )
                    Tx.wg.cast(reg_wg_fp16, reg_wg)
                    Tx.wg.copy(
                        self.output_smem[self.stage, st : st + self.TMEM_LD_SIZE, 0:128],
                        reg_wg_fp16.permute(1, 0),
                        dispatch="ldstmatrix",
                    )
                else:
                    Tx.wg.copy(
                        self.output_smem[self.stage, st : st + self.TMEM_LD_SIZE, 0:128],
                        reg_wg.permute(1, 0),
                    )
            if ko == self.MMA_M // self.EPI_TILE - 1:
                T.ptx.tcgen05.fence.before_thread_sync()
                self.ld2mma_bar.arrive(self.tmem_idx)
            T.ptx.fence.proxy_async("shared::cta")
            T.cuda.warpgroup_sync(10)
            if tid_in_wg == 0:
                m_st = T.meta_var(m_idx * self.M_pad_size + ko * self.EPI_TILE)
                n_st = T.meta_var(n_idx * self.BLK_N)
                tma_config = T.meta_var(
                    {"dispatch": "tma", "cta_group": KernelConfig.CTA_GROUP}
                    | (
                        {"cache_hint": "evict_last" if self.low_batch else ""}
                        if self.split_k_factor > 1
                        else {}
                    )
                    | ({"use_tma_reduce": "add"} if self.use_tma_reduce else {})
                )
                if self.split_k_factor > 1 and (not self.use_tma_reduce):
                    Tx.thread.copy_async(
                        output[k_idx, m_st : m_st + self.EPI_TILE, n_st : n_st + self.BLK_N],
                        self.output_smem[self.stage, :, :],
                        **tma_config,
                    )
                else:
                    Tx.thread.copy_async(
                        output[m_st : m_st + self.EPI_TILE, n_st : n_st + self.BLK_N],
                        self.output_smem[self.stage, :, :],
                        **tma_config,
                    )
                T.ptx.cp_async.bulk.commit_group()
        if tid_in_wg:
            T.ptx.cp_async.bulk.wait_group(0)
        T.cuda.warpgroup_sync(10)
        self.tile_idx += 1
        if warp_id == 0:
            self.smem_manager.arrive_specific(lane_id, self.output_smem, 0)

    @T.inline
    def _run(
        self, m_idx, n_idx, k_idx, A, B, output, profiler: CudaProfiler = None, A_tensormap=None
    ):
        wg_id = T.warpgroup_id([KernelConfig.WG_NUMBER])
        T.warpgroup_id([KernelConfig.WG_NUMBER])
        warp_id = T.warp_id_in_wg([KernelConfig.WARP_NUMBER])
        lane_id = T.lane_id([32])
        tid = T.thread_id([KernelConfig.NUM_THREADS])
        if wg_id == 1:
            if warp_id == 3:
                if self.use_tma_gather4:

                    @T.inline
                    def tma_gather4_stage(ks, k_st, first_stage):
                        self.mma2tma_bar.wait(ks, self.phase[0])
                        B_tma_config = T.meta_var(
                            {
                                "dispatch": "tma",
                                "cta_group": KernelConfig.CTA_GROUP,
                                "mbar": self.tma2mma_bar.mbar.ptr_to([ks]),
                                "cache_hint": "evict_first" if self.low_batch else "",
                            }
                        )
                        A_tma_config = T.meta_var(
                            {
                                "dispatch": "tma",
                                "cta_group": KernelConfig.CTA_GROUP,
                                "mbar": self.tma2mma_bar.mbar.ptr_to([ks]),
                                "cache_hint": "evict_last" if self.low_batch else "",
                                "tile_mode": "tile_gather4",
                                "box_dim": [1, self.BLK_K],
                                "tensormap": A_tensormap,
                            }
                        )
                        if self.profiler_on:
                            profiler.start(ProfileEventType.TMA, lane_id == 0)
                        if first_stage:
                            self.smem_manager.wait_specific_one_thread(self.A_smem, ks)
                        self._tma_gather4_A(ks, A, m_idx, k_st, A_tma_config, tid, lane_id)
                        if not self.prefetch_on and first_stage:
                            self.smem_manager.wait_specific_one_thread(self.B_smem, ks)
                        self._tma(
                            ks,
                            B,
                            "B",
                            n_idx * self.BLK_N,
                            k_st,
                            B_tma_config,
                            predicate=tvm.tirx.Not(self.prefetch_on and first_stage),
                        )
                        if self.profiler_on:
                            profiler.end(ProfileEventType.TMA, lane_id == 0)
                        self.tma2mma_bar.arrive(
                            ks,
                            KernelConfig.CTA_GROUP
                            * self.BLK_K
                            * (self.BLK_M + self.BLK_N)
                            * F16_BYTES,
                        )

                    self._preload_tma_gather4_A(m_idx, lane_id)
                    if T.ptx.elect_sync():
                        k_offset = k_idx * self.TILE_K
                        for ko in T.serial(self.PIPE_CIRCLE_NUM):
                            for ks in T.unroll(self.SMEM_PIPE_DEPTH):
                                tma_gather4_stage(
                                    ks,
                                    (ko * self.SMEM_PIPE_DEPTH + ks) * self.BLK_K + k_offset,
                                    ko == 0,
                                )
                            self.phase[0] = self.phase[0] ^ 1
                        if self.PIPE_REMAIN_NUM > 0:
                            for ks in T.unroll(self.PIPE_REMAIN_NUM):
                                tma_gather4_stage(
                                    ks,
                                    (self.PIPE_CIRCLE_NUM * self.SMEM_PIPE_DEPTH + ks) * self.BLK_K
                                    + k_offset,
                                    self.PIPE_CIRCLE_NUM == 0,
                                )
                            for ks in T.unroll(self.PIPE_REMAIN_NUM, self.SMEM_PIPE_DEPTH):
                                self.mma2tma_bar.wait(ks, self.phase[0])
                                self.tma2mma_bar.arrive_only(ks)
                            self.phase[0] = self.phase[0] ^ 1
                elif self.use_cp_async_input:

                    @T.inline
                    def cp_async_stage(ks, k_st, first_stage):
                        lane_id = T.lane_id([32])
                        bytes_per_B_stage = T.meta_var(
                            KernelConfig.CTA_GROUP * self.BLK_K * self.BLK_N * F16_BYTES
                        )
                        B_tma_config = T.meta_var(
                            {
                                "dispatch": "tma",
                                "cta_group": KernelConfig.CTA_GROUP,
                                "mbar": self.tma2mma_bar.mbar.ptr_to([ks]),
                                "cache_hint": "evict_first" if self.low_batch else "",
                            }
                        )
                        self.mma2tma_bar.wait(ks, self.phase[0])
                        if T.ptx.elect_sync():
                            if self.profiler_on:
                                profiler.start(ProfileEventType.TMA, lane_id == 0)
                            if first_stage:
                                self.smem_manager.wait_specific_one_thread(self.A_smem, ks)
                        self._cp_async_load_A_tile(
                            m_idx, ks, k_st, tid, lane_id, A, self.tma2mma_bar.mbar.ptr_to([ks])
                        )
                        T.cuda.warp_sync()
                        if T.ptx.elect_sync():
                            if not self.prefetch_on and first_stage:
                                self.smem_manager.wait_specific_one_thread(self.B_smem, ks)
                            if self.prefetch_on:
                                self.tma2mma_bar.arrive(
                                    ks, T.if_then_else(first_stage, 0, bytes_per_B_stage)
                                )
                            else:
                                self.tma2mma_bar.arrive(ks, bytes_per_B_stage)
                            self._tma(
                                ks,
                                B,
                                "B",
                                n_idx * self.BLK_N,
                                k_st,
                                B_tma_config,
                                predicate=tvm.tirx.Not(self.prefetch_on and first_stage),
                            )
                            if self.profiler_on:
                                profiler.end(ProfileEventType.TMA, lane_id == 0)
                        T.cuda.warp_sync()

                    k_offset = k_idx * self.TILE_K
                    for ko in T.serial(self.PIPE_CIRCLE_NUM):
                        for ks in T.unroll(self.SMEM_PIPE_DEPTH):
                            cp_async_stage(
                                ks,
                                (ko * self.SMEM_PIPE_DEPTH + ks) * self.BLK_K + k_offset,
                                ko == 0,
                            )
                        self.phase[0] = self.phase[0] ^ 1
                    if self.PIPE_REMAIN_NUM > 0:
                        for ks in T.unroll(self.PIPE_REMAIN_NUM):
                            cp_async_stage(
                                ks,
                                (self.PIPE_CIRCLE_NUM * self.SMEM_PIPE_DEPTH + ks) * self.BLK_K
                                + k_offset,
                                self.PIPE_CIRCLE_NUM == 0,
                            )
                        for ks in T.unroll(self.PIPE_REMAIN_NUM, self.SMEM_PIPE_DEPTH):
                            self.mma2tma_bar.wait(ks, self.phase[0])
                            self.tma2mma_bar.arrive_only(ks)
                            if T.ptx.elect_sync():
                                self.tma2mma_bar.arrive_only(ks)
                        self.phase[0] = self.phase[0] ^ 1
                else:

                    @T.inline
                    def tma_stage(ks, k_st, first_stage):
                        self.mma2tma_bar.wait(ks, self.phase[0])
                        B_tma_config = T.meta_var(
                            {
                                "dispatch": "tma",
                                "cta_group": KernelConfig.CTA_GROUP,
                                "mbar": self.tma2mma_bar.mbar.ptr_to([ks]),
                                "cache_hint": "evict_first" if self.low_batch else "",
                            }
                        )
                        A_tma_config = T.meta_var(
                            {
                                "dispatch": "tma",
                                "cta_group": KernelConfig.CTA_GROUP,
                                "mbar": self.tma2mma_bar.mbar.ptr_to([ks]),
                                "cache_hint": "evict_last" if self.low_batch else "",
                            }
                        )
                        if self.profiler_on:
                            profiler.start(ProfileEventType.TMA, lane_id == 0)
                        if first_stage:
                            self.smem_manager.wait_specific_one_thread(self.A_smem, ks)
                        self._tma(ks, A, "A", m_idx * self.M_pad_size, k_st, A_tma_config)
                        if not self.prefetch_on and first_stage:
                            self.smem_manager.wait_specific_one_thread(self.B_smem, ks)
                        self._tma(
                            ks,
                            B,
                            "B",
                            n_idx * self.BLK_N,
                            k_st,
                            B_tma_config,
                            predicate=tvm.tirx.Not(self.prefetch_on and first_stage),
                        )
                        if self.profiler_on:
                            profiler.end(ProfileEventType.TMA, lane_id == 0)
                        self.tma2mma_bar.arrive(
                            ks,
                            KernelConfig.CTA_GROUP
                            * self.BLK_K
                            * (self.BLK_M + self.BLK_N)
                            * F16_BYTES,
                        )

                    if T.ptx.elect_sync():
                        k_offset = k_idx * self.TILE_K
                        for ko in T.serial(self.PIPE_CIRCLE_NUM):
                            for ks in T.unroll(self.SMEM_PIPE_DEPTH):
                                tma_stage(
                                    ks,
                                    (ko * self.SMEM_PIPE_DEPTH + ks) * self.BLK_K + k_offset,
                                    ko == 0,
                                )
                            self.phase[0] = self.phase[0] ^ 1
                        if self.PIPE_REMAIN_NUM > 0:
                            for ks in T.unroll(self.PIPE_REMAIN_NUM):
                                tma_stage(
                                    ks,
                                    (self.PIPE_CIRCLE_NUM * self.SMEM_PIPE_DEPTH + ks) * self.BLK_K
                                    + k_offset,
                                    self.PIPE_CIRCLE_NUM == 0,
                                )
                            for ks in T.unroll(self.PIPE_REMAIN_NUM, self.SMEM_PIPE_DEPTH):
                                self.mma2tma_bar.wait(ks, self.phase[0])
                                self.tma2mma_bar.arrive_only(ks)
                            self.phase[0] = self.phase[0] ^ 1
                T.ptx.bar.sync(13, 64)
            elif warp_id == 0:

                @T.inline
                def mbar_try_wait(idx, phase):
                    self.wait_complete = mbarrier_try_wait(
                        self.tma2mma_bar.mbar.ptr_to([idx]), self.tma2mma_bar.init_phase ^ phase
                    )

                @T.inline
                def mma_stage(ks, acc):
                    if self.profiler_on:
                        profiler.start(ProfileEventType.MMA, lane_id == 0)
                    Tx.gemm_async(
                        self.tmem[
                            :,
                            self.tmem_idx * self.M_pad_size : self.tmem_idx * self.M_pad_size
                            + self.BLK_N,
                        ],
                        self.B_smem[ks, :, :],
                        self.A_smem[ks, :, :],
                        accum=acc,
                        dispatch="tcgen05",
                        cta_group=KernelConfig.CTA_GROUP,
                    )
                    if self.profiler_on:
                        profiler.end(ProfileEventType.MMA, lane_id == 0)
                    self.mma2tma_bar.arrive(ks)

                if T.ptx.elect_sync():
                    self.tmem_idx = self.tile_idx % self.TMEM_PIPE_DEPTH
                    self.tmem_phase = self.tile_idx // self.TMEM_PIPE_DEPTH & 1
                    self.ld2mma_bar.wait(self.tmem_idx, self.tmem_phase)
                    T.ptx.tcgen05.fence.after_thread_sync()
                    if self.use_cp_async_input:
                        self.wait_complete = False
                    else:
                        mbar_try_wait(0, self.phase[0])
                    for ko in T.serial(self.PIPE_CIRCLE_NUM):
                        for ks in T.unroll(self.SMEM_PIPE_DEPTH):
                            if self.use_cp_async_input:
                                self.tma2mma_bar.wait(ks, self.phase[0])
                            elif not self.wait_complete:
                                self.tma2mma_bar.wait(ks, self.phase[0])
                            if not self.use_cp_async_input and (
                                self.PIPE_REMAIN_NUM > 0
                                or ko != self.PIPE_REMAIN_NUM - 1
                                or ks != self.SMEM_PIPE_DEPTH - 1
                            ):
                                mbar_try_wait(
                                    (ks + 1) % self.SMEM_PIPE_DEPTH,
                                    self.phase[0] ^ (1 if ks == self.SMEM_PIPE_DEPTH - 1 else 0),
                                )
                            mma_stage(ks, not ((ko == 0) & (ks == 0)))
                        self.phase[0] = self.phase[0] ^ 1
                    if self.PIPE_REMAIN_NUM > 0:
                        for ks in T.unroll(self.PIPE_REMAIN_NUM):
                            if self.use_cp_async_input:
                                self.tma2mma_bar.wait(ks, self.phase[0])
                            elif not self.wait_complete:
                                self.tma2mma_bar.wait(ks, self.phase[0])
                            if not self.use_cp_async_input and ks != self.PIPE_REMAIN_NUM - 1:
                                mbar_try_wait((ks + 1) % self.SMEM_PIPE_DEPTH, self.phase[0])
                            mma_stage(ks, not (self.PIPE_CIRCLE_NUM == 0 and ks == 0))
                        self.mma2ld_bar.arrive(self.tmem_idx)
                        for ks in T.unroll(self.PIPE_REMAIN_NUM, self.SMEM_PIPE_DEPTH):
                            self.tma2mma_bar.wait(ks, self.phase[0])
                            self.mma2tma_bar.arrive(ks)
                        self.phase[0] = self.phase[0] ^ 1
                    else:
                        self.mma2ld_bar.arrive(self.tmem_idx)
                self.tile_idx += 1
            elif warp_id == 1:
                self.smem_manager.wait_unused(lane_id, self)
                self.smem_manager.arrive_unused(lane_id, self)
            elif warp_id == 2:
                self.phase[0] = self.phase[0] ^ self.PIPE_CIRCLE_NUM & 1
                if self.PIPE_REMAIN_NUM > 0:
                    self.phase[0] = self.phase[0] ^ 1
                T.ptx.bar.sync(13, 64)
                for ks in T.unroll(self.SMEM_PIPE_DEPTH):
                    self.mma2tma_bar.wait(ks, self.phase[0])
                    self.smem_manager.arrive_specific(lane_id, self.B_smem, ks)
                    self.smem_manager.arrive_specific(lane_id, self.A_smem, ks)
        if wg_id == 0:
            self._consumer_wg(m_idx, n_idx, k_idx, A, B, output, profiler)

    @T.inline
    def prefetch(self, m_idx, n_idx, k_idx, A, B, output, profiler: CudaProfiler):
        self._alloc_local(m_idx)
        wg_id = T.warpgroup_id([KernelConfig.WG_NUMBER])
        warp_id = T.warp_id_in_wg([KernelConfig.WARP_NUMBER])
        lane_id = T.lane_id([32])
        if (wg_id == 1) & (warp_id == 3):
            k_offset = k_idx * self.TILE_K
            if self.PIPE_CIRCLE_NUM > 0:
                for ks in T.unroll(self.SMEM_PIPE_DEPTH):
                    self.stage = ks
                    self.smem_manager.wait_specific(lane_id, self.B_smem, ks)
                    if self.profiler_on:
                        profiler.start(ProfileEventType.TMA, lane_id == 0)
                    if T.ptx.elect_sync():
                        tma_config = T.meta_var(
                            {
                                "dispatch": "tma",
                                "cta_group": KernelConfig.CTA_GROUP,
                                "mbar": self.tma2mma_bar.mbar.ptr_to([ks]),
                                "cache_hint": "evict_first" if self.low_batch else "",
                            }
                        )
                        self._tma(
                            ks,
                            B,
                            "B",
                            n_idx * self.BLK_N,
                            self.stage * self.BLK_K + k_offset,
                            tma_config,
                        )
                    if self.profiler_on:
                        profiler.end(ProfileEventType.TMA, lane_id == 0)

    @T.inline
    def run(
        self, m_idx, n_idx, k_idx, A, B, output, profiler: CudaProfiler = None, A_tensormap=None
    ):
        self._alloc_local(m_idx)
        self._run(m_idx, n_idx, k_idx, A, B, output, profiler, A_tensormap=A_tensormap)
        self.smem_manager.advance()

    def _cp_async_load_A_tile(self, m_idx, ks, stage_k, tid, lane_id, A, mbar):
        raise RuntimeError("cp.async input path is not implemented for this tile.")

    @T.inline
    def _preload_tma_gather4_A(self, m_idx, tid_in_wg):
        pass

    def _tma_gather4_A(self, ks, A, m_idx, k_st, tma_config, tid, lane_id):
        raise RuntimeError("tma gather4 input path is not implemented for this tile.")

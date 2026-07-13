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

"""Configuration for megakernel."""

from enum import Enum

F16_BYTES = 2
F32_BYTES = 4


class KernelConfig:
    # global constant
    M_CLUSTER = 1
    N_CLUSTER = 1
    WG_NUMBER = 2
    WARP_NUMBER = 4
    NUM_THREADS = (32 * WARP_NUMBER) * WG_NUMBER
    SM_NUMBER = 148
    CTA_GROUP = M_CLUSTER
    MAX_SMEM_SIZE = 232448


class JobType(Enum):
    MOE_GATING = 18
    MOE_TOPK_SOFTMAX = 19
    MOE_ALIGN = 20
    MOE_COUNT_AND_SORT = 22
    MOE_GROUP_GEMM_DOWN = 25
    MOE_GROUP_GEMM_GATE_UP_SILU = 27
    INIT_ETENSOR = 28
    WAIT_ETENSOR_INIT = 29
    END = 31


class ProfileEventType(Enum):
    FETCH = 5
    PUSH = 20
    PREFETCH = 26
    TMA = 27
    MMA = 28
    MOE_GATING = 38
    TOPK_SOFTMAX = 39
    MOE_ALIGN = 40
    COUNT_AND_SORT = 41
    SILU_MUL = 43
    GROUP_GEMM_DOWN = 44
    GROUP_GEMM_GATE_UP_SILU = 51
    INIT_ETENSOR = 52
    WAIT_ETENSOR_INIT = 53
    END = 54


MEGAKERNEL_MOE_BENCH_CONFIG = {
    "CONFIG_NAME": "moe_a3b",
    "HIDDEN_SIZE": 2048,
    "INTERMEDIATE_SIZE": 768,
    "NUM_EXPERTS": 128,
    "NUM_EXPERTS_PER_TOK": 8,
    "GATING_SPLIT_K_FACTOR": 4,
}

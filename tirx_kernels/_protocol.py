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
"""Standard kernel interface protocol.

Every kernel module under ``kernels/<category>/`` that wants to be
discoverable by the registry must expose:

Module-level constants
----------------------
KERNEL_META : dict
    Required keys:
    - "name" (str): unique kernel name used by CLI (e.g. "rmsnorm")
    - "category" (str): one of gemm, gemm_comm, attention, normalization, activation, ssm, loss, moe
    - "compute_capability" (int): minimum SM version (e.g. 10 for sm100a)

CONFIGS : list[dict]
    Each dict has a "label" key (str) plus arbitrary kernel-specific
    parameters.  The same config matrix is used by correctness tests and
    benchmark runs.

Functions
---------
get_kernel(**cfg) -> tvm.tirx.PrimFunc | list[tvm.tirx.PrimFunc]
    Return the TIR PrimFunc(s) for this kernel.  Multi-kernel workloads
    (e.g. split-k GEMM with a separate reduce kernel) return a list.

prepare_data(**cfg) -> dict[str, Any]
    Prepare input/output tensors.  Returns a dict mapping argument names
    to tensors (torch.Tensor or numpy.ndarray).

check_correctness(outputs: dict, **cfg) -> None
    Validate kernel outputs against a reference.
    Raise AssertionError on mismatch.

get_baselines(**cfg) -> dict[str, Callable]   (optional)
    Return {name: callable} for baseline implementations used in
    benchmarking (e.g. cublas, flashinfer).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class KernelModule(Protocol):
    """Structural type that a kernel module must satisfy."""

    KERNEL_META: dict[str, Any]
    CONFIGS: list[dict[str, Any]]

    @staticmethod
    def get_kernel(**kwargs: Any) -> Any: ...

    @staticmethod
    def prepare_data(**kwargs: Any) -> dict[str, Any]: ...

    @staticmethod
    def check_correctness(outputs: dict[str, Any], **kwargs: Any) -> None: ...

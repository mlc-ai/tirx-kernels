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

from tirx_kernels.registry import discover_categories, discover_kernels, load_kernel


def test_discover_categories_includes_kernel_dirs() -> None:
    categories = discover_categories()
    assert "gemm" in categories
    assert "attention" in categories
    assert "flashmla" in categories
    assert "bench" not in categories
    assert "bench_suite" not in categories


def test_discover_kernels_finds_known_gemm() -> None:
    kernels = discover_kernels(category="gemm")
    assert "fp16_bf16_gemm" in kernels
    assert "nvfp4_gemm" in kernels


def test_load_kernel_finds_single_module() -> None:
    mod = load_kernel("nvfp4_gemm")
    assert mod.KERNEL_META["name"] == "nvfp4_gemm"


def test_load_kernel_finds_flashmla_unified_entry() -> None:
    mod = load_kernel("flash_mla_sparse_fwd")
    assert mod.KERNEL_META["category"] == "flashmla"

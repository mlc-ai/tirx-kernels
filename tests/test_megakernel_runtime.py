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
"""Runtime tests for MegaKernelMOE."""

import pytest

from tirx_kernels.runner import run_kernel_test


@pytest.mark.parametrize("batch_size", [1, 128])
@pytest.mark.parametrize("scheduler", ["static", "dynamic", "unfused"])
def test_megakernel_moe(batch_size, scheduler):
    run_kernel_test(
        "megakernel_moe", {"batch_size": batch_size, "scheduler": scheduler, "world_size": 1}
    )

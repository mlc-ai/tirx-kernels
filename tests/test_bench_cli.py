# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Tests for the benchmark CLI."""

from types import SimpleNamespace

from tirx_kernels.bench.__main__ import _get_bench_configs


def test_bench_configs_prefer_benchmark_sweep():
    configs = [{"label": "correctness"}]
    bench_configs = [{"label": "performance"}]
    mod = SimpleNamespace(CONFIGS=configs, BENCH_CONFIGS=bench_configs)

    assert _get_bench_configs(mod) is bench_configs


def test_bench_configs_fall_back_to_correctness_sweep():
    configs = [{"label": "correctness"}]
    mod = SimpleNamespace(CONFIGS=configs)

    assert _get_bench_configs(mod) is configs

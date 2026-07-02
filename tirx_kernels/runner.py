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
"""Unified kernel test and benchmark runner.

Each kernel module must provide ``run_test(**config)`` which handles
compile → run → correctness-check internally.  Optionally, it can
provide ``run_bench(**config, warmup, repeat)`` for profiling.

The helpers ``compile_kernel`` and ``proton_bench`` are exposed for
kernel modules to use.
"""

from __future__ import annotations

from typing import Any

import tvm


def compile_kernel(func):
    """Compile a single TIR PrimFunc via the tirx pipeline."""
    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": func})
    return tvm.compile(mod, target=target, tir_pipeline="tirx")


def run_kernel_test(kernel_name: str, config: dict[str, Any], *, registry=None):
    """Run a kernel's correctness test.

    Delegates to ``mod.run_test(**params)``.
    """
    if registry is None:
        from tirx_kernels.registry import discover_kernels

        registry = discover_kernels()

    mod = registry[kernel_name]
    params = {k: v for k, v in config.items() if k != "label"}
    mod.run_test(**params)


def run_kernel_bench(
    kernel_name: str,
    config: dict[str, Any],
    *,
    registry=None,
    warmup: int | None = None,
    repeat: int | None = None,
    timer: str | None = None,
    rounds: int | None = None,
    round_cooldown: float | None = None,
):
    """Run a kernel's benchmark.

    Delegates to ``mod.run_bench(**params, ...)`` if available, otherwise runs
    ``run_test`` without timing. warmup/repeat are only forwarded when explicitly
    provided (CLI ``--warmup/--repeat`` or a per-workload override); otherwise each
    timer uses its own Triton-aligned default inside ``tvm.tirx.bench.bench``.
    """
    if registry is None:
        from tirx_kernels.registry import load_kernel

        registry = {kernel_name: load_kernel(kernel_name)}

    mod = registry[kernel_name]
    params = {k: v for k, v in config.items() if k != "label"}
    label = config.get("label", "default")

    run_bench_fn = getattr(mod, "run_bench", None)
    if run_bench_fn is not None:
        bench_kwargs = dict(params)
        if warmup is not None:
            bench_kwargs["warmup"] = warmup
        if repeat is not None:
            bench_kwargs["repeat"] = repeat
        if timer is not None:
            bench_kwargs["timer"] = timer
        if rounds is not None:
            bench_kwargs["rounds"] = rounds
        if round_cooldown is not None:
            bench_kwargs["round_cooldown_s"] = round_cooldown
        result = run_bench_fn(**bench_kwargs)
        if not isinstance(result, dict):
            result = {}

        result.setdefault("kernel", kernel_name)
        result.setdefault("label", label)
        return result

    # Fallback: just run the test (no profiling)
    mod.run_test(**params)
    return {"kernel": kernel_name, "label": label}

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
"""CLI entry point: python -m tirx_kernels.bench [--kernel <name>] [--config <label>]"""

import argparse
import contextlib
import json
import os
import sys
import tempfile
import traceback
from unittest import SkipTest

from tirx_kernels.registry import discover_kernels, load_kernel
from tirx_kernels.runner import run_kernel_bench


def _gpu_lock():
    """Per-physical-GPU advisory lock around the GPU-measurement phase.

    Enabled by TIR_BENCH_GPU_LOCK=1 (set by tir-bench run.py when it overcommits
    CPU workers): many bench subprocesses import + compile in parallel, but only
    one measures per physical GPU at a time. Keyed by CUDA_VISIBLE_DEVICES; a
    no-op for standalone runs or a multi-GPU mask.
    """
    if os.environ.get("TIR_BENCH_GPU_LOCK") != "1":
        return contextlib.nullcontext()
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu or "," in gpu:
        return contextlib.nullcontext()
    from tvm_ffi.utils import FileLock

    lock_dir = os.environ.get("TIR_BENCH_LOCK_DIR") or tempfile.gettempdir()
    return FileLock(os.path.join(lock_dir, f"tir-bench-gpu-{gpu}.lock"))


def main():
    parser = argparse.ArgumentParser(description="Run kernel benchmarks")
    parser.add_argument("--kernel", type=str, default=None, help="Run only this kernel")
    parser.add_argument("--config", type=str, default=None, help="Run only this config label")
    parser.add_argument("--json", action="store_true", help="Output JSON results")
    parser.add_argument(
        "--json-file",
        type=str,
        default=None,
        help="Write JSON results to this file instead of stdout",
    )
    parser.add_argument("--cc", type=int, default=None, help="Compute capability filter")
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Override the kernel module's benchmark warmup default",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=None,
        help="Override the kernel module's benchmark repeat default",
    )
    parser.add_argument(
        "--timer",
        type=str,
        choices=("proton", "event"),
        default=None,
        help="Override the kernel module's benchmark timer",
    )
    parser.add_argument(
        "--impls",
        type=str,
        choices=("all", "ours", "baseline"),
        default=None,
        help="Which impls to bench: 'all' (default), 'ours' (only our kernel — "
        "skips reference setup/execution; reference times come from the pinned "
        "baseline), or 'baseline' (only reference impls). Sets TIRX_BENCH_IMPLS.",
    )
    args = parser.parse_args()

    if args.impls is not None:
        os.environ["TIRX_BENCH_IMPLS"] = args.impls

    if args.json or args.json_file:
        os.environ["TIRX_BENCH_JSON"] = "1"

    if args.kernel:
        try:
            mod = load_kernel(args.kernel)
        except KeyError:
            print(f"ERROR: kernel '{args.kernel}' not found.", file=sys.stderr)
            sys.exit(1)
        if args.cc is not None and mod.KERNEL_META.get("compute_capability") != args.cc:
            print(
                f"ERROR: kernel '{args.kernel}' compute_capability="
                f"{mod.KERNEL_META.get('compute_capability')} != filter {args.cc}",
                file=sys.stderr,
            )
            sys.exit(1)
        all_kernels = {args.kernel: mod}
    else:
        all_kernels = discover_kernels(min_compute_capability=args.cc)

    # Each kernel's run_bench() manages its own ProtonContext session.
    # No global proton session needed.
    results = []

    for name, mod in sorted(all_kernels.items()):
        configs = getattr(mod, "BENCH_CONFIGS", getattr(mod, "CONFIGS", []))
        for cfg in configs:
            label = cfg.get("label", "default")
            if args.config and label != args.config:
                continue
            try:
                with _gpu_lock():
                    result = run_kernel_bench(
                        name,
                        cfg,
                        registry=all_kernels,
                        warmup=args.warmup,
                        repeat=args.repeat,
                        timer=args.timer,
                    )
                results.append(result)
            except SkipTest as exc:
                results.append(
                    {"kernel": name, "label": label, "status": "SKIP", "reason": str(exc)}
                )
                if not args.json and not args.json_file:
                    print(f"SKIP  {name} [{label}]: {exc}", file=sys.stderr)
            except Exception as e:
                results.append({"kernel": name, "label": label, "status": "FAIL", "error": str(e)})
                if not args.json and not args.json_file:
                    print(f"FAIL  {name} [{label}]: {e}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)

    if args.json_file:
        with open(args.json_file, "w") as f:
            json.dump({"results": results}, f, indent=2)
    elif args.json:
        print(json.dumps({"results": results}, indent=2))
    else:
        # Print summary to stdout for human consumption
        for r in results:
            status = r.get("status", "ok")
            kernel = r.get("kernel", "?")
            label = r.get("label", "?")
            if status == "FAIL":
                print(f"FAIL  {kernel} [{label}]: {r.get('error', '?')}")
            elif status == "SKIP":
                print(f"SKIP  {kernel} [{label}]: {r.get('reason', '?')}")
            else:
                impls = r.get("impls", {})
                impl_str = ", ".join(f"{k}={v:.3f}ms" for k, v in impls.items())
                print(f"OK    {kernel} [{label}]: {impl_str}")


if __name__ == "__main__":
    main()

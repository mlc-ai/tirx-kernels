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
"""CLI entry point: python -m tirx_kernels.test [--kernel <name>] [--config <label>]"""

import argparse
import json
import sys
import traceback
from unittest import SkipTest

from tirx_kernels.registry import discover_kernels
from tirx_kernels.runner import run_kernel_test


def main():
    parser = argparse.ArgumentParser(description="Run kernel correctness tests")
    parser.add_argument("--kernel", type=str, default=None, help="Run only this kernel")
    parser.add_argument("--config", type=str, default=None, help="Run only this config label")
    parser.add_argument("--json", action="store_true", help="Output JSON results")
    parser.add_argument("--cc", type=int, default=None, help="Compute capability filter")
    args = parser.parse_args()

    all_kernels = discover_kernels(min_compute_capability=args.cc)

    if args.kernel:
        if args.kernel not in all_kernels:
            print(
                f"ERROR: kernel '{args.kernel}' not found. Available: {sorted(all_kernels.keys())}"
            )
            sys.exit(1)
        all_kernels = {args.kernel: all_kernels[args.kernel]}

    results = []
    passed = 0
    failed = 0
    skipped = 0

    for name, mod in sorted(all_kernels.items()):
        configs = getattr(mod, "CONFIGS", [])
        for cfg in configs:
            label = cfg.get("label", "default")
            if args.config and label != args.config:
                continue
            try:
                run_kernel_test(name, cfg, registry=all_kernels)
                results.append({"kernel": name, "config": label, "status": "PASS"})
                passed += 1
                if not args.json:
                    print(f"PASS  {name} [{label}]")
            except SkipTest as exc:
                results.append(
                    {"kernel": name, "config": label, "status": "SKIP", "reason": str(exc)}
                )
                skipped += 1
                if not args.json:
                    print(f"SKIP  {name} [{label}]: {exc}")
            except Exception as e:
                results.append({"kernel": name, "config": label, "status": "FAIL", "error": str(e)})
                failed += 1
                if not args.json:
                    print(f"FAIL  {name} [{label}]: {e}")
                    traceback.print_exc()

    if args.json:
        print(
            json.dumps(
                {"passed": passed, "failed": failed, "skipped": skipped, "results": results},
                indent=2,
            )
        )
    else:
        print(f"\n{'=' * 60}")
        print(
            f"Total: {passed + failed + skipped}  "
            f"Passed: {passed}  Failed: {failed}  Skipped: {skipped}"
        )

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()

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
"""Kernel registry — auto-discovers kernel modules with KERNEL_META."""

from __future__ import annotations

import importlib
import logging
import pkgutil
from pathlib import Path
from types import ModuleType

import tirx_kernels

_log = logging.getLogger(__name__)

# CLI / package infra — not kernel categories.
_SKIP_CATEGORIES = frozenset({"bench", "test", "bench_suite", "experimental"})

_KERNEL_CACHE: dict[str, ModuleType] = {}


def _kernels_root() -> Path:
    return Path(tirx_kernels.__file__).resolve().parent


def discover_categories() -> list[str]:
    """Return sorted category subpackage names under the installed ``tirx_kernels`` package."""
    root = _kernels_root()
    out: list[str] = []
    for child in sorted(root.iterdir()):
        if child.name.startswith("_") or child.name in _SKIP_CATEGORIES or not child.is_dir():
            continue
        if (child / "__init__.py").is_file():
            out.append(child.name)
    return out


def _import_kernel_module(category: str, mod_name: str, *, strict: bool) -> ModuleType | None:
    pkg_name = f"tirx_kernels.{category}"
    try:
        mod = importlib.import_module(f"{pkg_name}.{mod_name}")
    except Exception as e:
        if strict:
            raise
        _log.warning("tirx_kernels: failed to import %s.%s: %s", pkg_name, mod_name, e)
        return None
    meta = getattr(mod, "KERNEL_META", None)
    if meta is not None and isinstance(meta, dict) and "name" in meta:
        _KERNEL_CACHE[meta["name"]] = mod
        return mod
    return None


def load_kernel(name: str, *, strict: bool = False) -> ModuleType:
    """Import a single kernel module by ``KERNEL_META['name']``.

    Uses an in-process cache populated by prior ``load_kernel`` / ``discover_kernels``
    calls. Raises ``KeyError`` when no matching kernel is installed.
    """
    if name in _KERNEL_CACHE:
        return _KERNEL_CACHE[name]

    for cat in discover_categories():
        pkg_path = _kernels_root() / cat
        if not pkg_path.is_dir():
            continue
        for _importer, mod_name, is_pkg in pkgutil.iter_modules([str(pkg_path)]):
            if mod_name.startswith("_") or is_pkg:
                continue
            mod = _import_kernel_module(cat, mod_name, strict=strict)
            if mod is not None and mod.KERNEL_META["name"] == name:
                return mod

    raise KeyError(name)


def check_workload_imports(workloads: list[dict], *, strict: bool = True) -> list[str]:
    """Import every unique kernel referenced in a workloads list. Returns kernel names."""
    names = sorted({w["kernel"] for w in workloads})
    for name in names:
        load_kernel(name, strict=strict)
    return names


def _scan_category(category: str, *, strict: bool) -> dict[str, ModuleType]:
    """Import all modules in tirx_kernels/<category>/ that expose KERNEL_META."""
    result: dict[str, ModuleType] = {}
    pkg_path = _kernels_root() / category
    if not pkg_path.is_dir():
        return result
    for _importer, mod_name, is_pkg in pkgutil.iter_modules([str(pkg_path)]):
        if mod_name.startswith("_") or is_pkg:
            continue
        mod = _import_kernel_module(category, mod_name, strict=strict)
        if mod is not None:
            result[mod.KERNEL_META["name"]] = mod
    return result


def discover_kernels(
    *, min_compute_capability: int | None = None, category: str | None = None, strict: bool = False
) -> dict[str, ModuleType]:
    """Return ``{name: module}`` for all registered kernels in the installed package.

    Parameters
    ----------
    min_compute_capability : int, optional
        If given, only include kernels whose ``compute_capability`` is
        *exactly* this value.  (We use exact match because sm100a kernels
        won't run on sm90a, etc.)
    category : str, optional
        If given, only scan this category subdirectory.
    strict : bool, optional
        If True, raise on the first kernel module import failure instead of
        logging a warning and continuing.
    """
    categories = [category] if category else discover_categories()
    result: dict[str, ModuleType] = {}
    for cat in categories:
        result.update(_scan_category(cat, strict=strict))
    if min_compute_capability is not None:
        result = {
            name: mod
            for name, mod in result.items()
            if mod.KERNEL_META.get("compute_capability") == min_compute_capability
        }
    return result


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="List registered kernels")
    parser.add_argument("--list", action="store_true", help="List all kernels")
    parser.add_argument(
        "--format", choices=["text", "json", "benchrun"], default="text", help="Output format"
    )
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--cc", type=int, default=None, help="Compute capability filter")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise on the first kernel import failure (for CI import gates)",
    )
    args = parser.parse_args()

    all_kernels = discover_kernels(
        min_compute_capability=args.cc, category=args.category, strict=args.strict
    )

    if args.format == "json":
        out = []
        for name, mod in sorted(all_kernels.items()):
            configs = getattr(mod, "CONFIGS", [])
            bench_configs = getattr(mod, "BENCH_CONFIGS", configs)
            out.append(
                {
                    "name": name,
                    "meta": mod.KERNEL_META,
                    "configs": configs,
                    "bench_configs": bench_configs,
                }
            )
        print(json.dumps(out, indent=2))
    elif args.format == "benchrun":
        # Output in bench-run.sh format: KERNEL|SIZE|COMMAND
        for name, mod in sorted(all_kernels.items()):
            for cfg in getattr(mod, "BENCH_CONFIGS", getattr(mod, "CONFIGS", [])):
                label = cfg.get("label", "default")
                print(
                    f"{name}|{label}|python -m tirx_kernels.bench --kernel {name} --config {label} --json-file {{json_file}}"
                )
    else:
        for name, mod in sorted(all_kernels.items()):
            meta = mod.KERNEL_META
            n_configs = len(getattr(mod, "CONFIGS", []))
            n_bench_configs = len(getattr(mod, "BENCH_CONFIGS", getattr(mod, "CONFIGS", [])))
            print(
                f"  {name:30s}  {meta.get('category', '?'):12s}  sm{meta.get('compute_capability', '?') * 10}  ({n_configs} configs, {n_bench_configs} bench configs)"
            )

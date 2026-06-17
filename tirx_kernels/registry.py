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
import os
import pkgutil
from pathlib import Path
from types import ModuleType

import tirx_kernels

_log = logging.getLogger(__name__)
_STRICT = os.environ.get("TIRX_KERNELS_STRICT", "").lower() in ("1", "true", "yes")

_BASE_CATEGORIES = ("gemm", "attention")
_EXTRA_CATEGORIES_ENV = "TIRX_KERNELS_EXTRA_CATEGORIES"


def _package_roots() -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()
    for raw in getattr(tirx_kernels, "__path__", []):
        path = Path(str(raw)).resolve()
        if not path.is_dir():
            continue
        text = str(path)
        if text not in seen:
            roots.append(path)
            seen.add(text)
    return roots


def _category_paths(category: str) -> list[Path]:
    paths = []
    for root in _package_roots():
        path = root / category
        if path.is_dir():
            paths.append(path)
    return paths


def _discover_categories() -> list[str]:
    categories = set(_BASE_CATEGORIES)
    raw_extra = os.environ.get(_EXTRA_CATEGORIES_ENV, "")
    for item in raw_extra.split(","):
        item = item.strip()
        if item:
            categories.add(item)
    for root in _package_roots():
        for child in root.iterdir():
            if child.name.startswith("_") or not child.is_dir():
                continue
            if (child / "__init__.py").is_file():
                categories.add(child.name)
    return sorted(categories)


def _ensure_category_package_path(category: str, paths: list[Path]) -> bool:
    try:
        pkg = importlib.import_module(f"tirx_kernels.{category}")
    except Exception as e:
        if _STRICT:
            raise
        _log.warning("tirx_kernels: failed to import category %s: %s", category, e)
        return False
    pkg_path = getattr(pkg, "__path__", None)
    if pkg_path is None:
        return True
    seen = {str(Path(item).resolve()) for item in pkg_path}
    for path in paths:
        text = str(path.resolve())
        if text not in seen:
            pkg_path.append(text)
            seen.add(text)
    return True


def _scan_category(category: str) -> dict[str, ModuleType]:
    """Import all modules in tirx_kernels/<category>/ that expose KERNEL_META."""
    result = {}
    pkg_paths = _category_paths(category)
    if not pkg_paths:
        return result
    if not _ensure_category_package_path(category, pkg_paths):
        return result
    pkg_name = f"tirx_kernels.{category}"
    seen_modules: set[str] = set()
    for pkg_path in pkg_paths:
        for _, mod_name, is_pkg in pkgutil.iter_modules([str(pkg_path)]):
            if mod_name.startswith("_") or is_pkg or mod_name in seen_modules:
                continue
            seen_modules.add(mod_name)
            try:
                mod = importlib.import_module(f"{pkg_name}.{mod_name}")
            except Exception as e:
                if _STRICT:
                    raise
                _log.warning("tirx_kernels: failed to import %s.%s: %s", pkg_name, mod_name, e)
                continue
            meta = getattr(mod, "KERNEL_META", None)
            if meta is not None and isinstance(meta, dict) and "name" in meta:
                result[meta["name"]] = mod
    return result


def discover_kernels(
    *, min_compute_capability: int | None = None, category: str | None = None
) -> dict[str, ModuleType]:
    """Return ``{name: module}`` for all registered kernels.

    Parameters
    ----------
    min_compute_capability : int, optional
        If given, only include kernels whose ``compute_capability`` is
        *exactly* this value.  (We use exact match because sm100a kernels
        won't run on sm90a, etc.)
    category : str, optional
        If given, only scan this category subdirectory.
    """
    categories = [category] if category else _discover_categories()
    result: dict[str, ModuleType] = {}
    for cat in categories:
        result.update(_scan_category(cat))
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
    args = parser.parse_args()

    all_kernels = discover_kernels(min_compute_capability=args.cc, category=args.category)

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

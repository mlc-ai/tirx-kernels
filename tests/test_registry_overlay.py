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

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_kernel(root: Path, category: str, module: str, name: str) -> None:
    category_dir = root / category
    category_dir.mkdir(parents=True, exist_ok=True)
    (category_dir / "__init__.py").write_text("", encoding="utf-8")
    (category_dir / f"{module}.py").write_text(
        textwrap.dedent(
            f"""
            KERNEL_META = {{
                "name": "{name}",
                "category": "{category}",
                "compute_capability": 100,
            }}
            CONFIGS = [{{"label": "tiny"}}]
            """
        ),
        encoding="utf-8",
    )


def _run_with_overlay(overlay_root: Path, code: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_REPO_ROOT)
    env["TIRX_KERNELS_OVERLAY_PATHS"] = str(overlay_root)
    env.pop("TIRX_KERNELS_EXTRA_CATEGORIES", None)
    env.pop("TIRX_KERNELS_STRICT", None)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=_REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def test_overlay_can_add_private_category(tmp_path: Path) -> None:
    overlay_root = tmp_path / "overlay" / "tirx_kernels"
    _write_kernel(overlay_root, "private", "toy", "private_toy")

    proc = _run_with_overlay(
        overlay_root,
        """
        import json
        from tirx_kernels.registry import _discover_categories, discover_kernels

        kernels = discover_kernels(category="private")
        print(json.dumps({
            "categories": _discover_categories(),
            "names": sorted(kernels),
            "category": kernels["private_toy"].KERNEL_META["category"],
        }))
        """,
    )

    data = json.loads(proc.stdout)
    assert "private" in data["categories"]
    assert data["names"] == ["private_toy"]
    assert data["category"] == "private"


def test_overlay_can_extend_existing_category(tmp_path: Path) -> None:
    overlay_root = tmp_path / "overlay" / "tirx_kernels"
    _write_kernel(overlay_root, "gemm", "overlay_toy", "overlay_gemm_toy")

    proc = _run_with_overlay(
        overlay_root,
        """
        import json
        from tirx_kernels.registry import discover_kernels

        kernels = discover_kernels(category="gemm")
        print(json.dumps({"has_overlay": "overlay_gemm_toy" in kernels}))
        """,
    )

    assert json.loads(proc.stdout)["has_overlay"]

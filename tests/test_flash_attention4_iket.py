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
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest
import torch

import tvm
from tirx_kernels.attention import flash_attention4
from tirx_kernels.attention.flash_attention4 import IKET_EVENT_NAMES, get_flash_attention4_kernel
from tvm.tirx.cuda.iket import IketProfiler, IketProfileResult

EXPECTED_RANGES = set(IKET_EVENT_NAMES[:11])
EXPECTED_MARKS = set(IKET_EVENT_NAMES[11:])


def _fa4_func():
    return get_flash_attention4_kernel(1, 1024, 1024, 32, 32, 128, is_causal=True)


def _sources(root) -> str:
    pending = [root]
    sources = []
    while pending:
        module = pending.pop()
        pending.extend(module.imports)
        try:
            sources.append(module.inspect_source())
        except RuntimeError:
            pass
    return "\n".join(sources)


def _require_sm100() -> None:
    if not torch.cuda.is_available() or not tvm.cuda(0).exist:
        pytest.skip("FA4 IKET verification requires CUDA in both PyTorch and TVM")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("FA4 IKET verification requires an SM100 GPU")


def test_flash_attention4_has_four_argument_abi_and_expected_annotations() -> None:
    func = _fa4_func()
    assert len(func.params) == 4

    script = func.script(show_meta=False)
    declarations = set(re.findall(r'T\.cuda\.iket\.(?:mark|range_start)\("([^"]+)"', script))
    assert declarations == set(IKET_EVENT_NAMES)
    assert len(IKET_EVENT_NAMES) == 18
    assert "profiler_buffer" not in script
    assert "clock64" not in script


def test_iket_profiler_is_not_exported_from_generic_bench_module() -> None:
    import tvm.tirx.bench as bench

    assert not hasattr(bench, "IketProfiler")


def test_flash_attention4_module_entry_uses_orchestrator_defaults(
    tmp_path, monkeypatch, capsys
) -> None:
    captured = {}

    def fake_run(main, **kwargs):
        captured.update(main=main, kwargs=kwargs)
        return IketProfileResult(
            output_dir=Path(kwargs["output_dir"]),
            postprocess=kwargs["postprocess"],
            json_traces=(tmp_path / "trace.json",),
            perfetto_traces=(tmp_path / "trace.pftrace",),
            html_reports=(tmp_path / "trace.html",),
        )

    monkeypatch.setattr(flash_attention4.iket, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["flash_attention4"])
    flash_attention4.main()

    assert captured["main"].func is flash_attention4._profile_iket_workload
    assert captured["kwargs"] == {
        "output_dir": "/tmp/fa4-iket",
        "postprocess": "all",
        "clobber": True,
        "timeout": 600.0,
        "keep": False,
        "max_ts_cnt_per_warp": None,
    }
    output = capsys.readouterr().out
    assert "IKET output directory: /tmp/fa4-iket" in output
    assert f"IKET artifact: {tmp_path / 'trace.json'}" in output


def test_flash_attention4_strip_and_official_metadata_on_sm100() -> None:
    _require_sm100()
    torch.empty(1, device="cuda")
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    module = tvm.IRModule({"main": _fa4_func()})

    plain = tvm.compile(module, target=target, tir_pipeline="tirx")
    plain_source = _sources(plain.mod)
    assert "iket" not in plain_source.lower()
    assert "profiler" not in plain_source.lower()

    executable = IketProfiler().compile(module, target=target)
    source = _sources(executable.mod)
    assert "__iket_meta_info" in source
    assert "__tvm_iket_" not in source
    assert "tvm_builtin_iket_prologue" not in source
    assert "tvm_builtin_iket_finalize" not in source

    declaration_names = set(re.findall(r"__iket_evt_decl_([a-z0-9_]+)_\d+_attrs", source))
    assert declaration_names == {name.replace("-", "_") for name in IKET_EVENT_NAMES}


def test_flash_attention4_external_trace_contract() -> None:
    trace_path = os.environ.get("TIRX_FA4_IKET_OFFICIAL_TRACE_JSON")
    if not trace_path:
        pytest.skip("set TIRX_FA4_IKET_OFFICIAL_TRACE_JSON to a locked run-iket trace")

    trace = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    launches = trace["launches"]
    assert len(launches) >= 1
    strings = trace["stringTable"]
    for launch in launches:
        assert launch["kernelName"].startswith("_kernel_kernel")
        assert {strings[item["rangeNameIdx"]] for item in launch["ranges"]} == EXPECTED_RANGES
        assert {strings[item["markerNameIdx"]] for item in launch["markers"]} == EXPECTED_MARKS
        assert all(item["startTs"] <= item["endTs"] for item in launch["ranges"])
        assert all(
            len(item["internalEvents"]) == 2
            and item["internalEvents"][0]["eventId"] == item["internalEvents"][1]["eventId"]
            and item["internalEvents"][0]["timestamp"] == item["startTs"]
            and item["internalEvents"][1]["timestamp"] == item["endTs"]
            for item in launch["ranges"]
        )
        assert all(
            isinstance(item["timestamp"], int) and item["timestamp"] > 0
            for item in launch["markers"]
        )

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
from tirx_kernels.flashmla import flash_mla_sparse_fwd
from tirx_kernels.flashmla import sparse_prefill_head64_phase1 as head64
from tirx_kernels.flashmla import sparse_prefill_head128_phase1 as head128
from tirx_kernels.flashmla import sparse_prefill_head128_small_topk_phase1 as head128_small
from tvm.tirx.cuda.iket import IketProfiler, IketProfileResult

CASES = (
    (head64, {"s_q": 1, "s_kv": 8192, "topk": 512, "d_qk": 576, "h_q": 64}),
    (head128, {"s_q": 1, "s_kv": 8192, "topk": 2048, "d_qk": 576, "h_q": 128}),
    (head128_small, {"s_q": 1, "s_kv": 8192, "topk": 1280, "h_q": 128}),
)
EXPECTED_EVENT_NAMES = {
    head64: frozenset(
        {
            "h64-q-load",
            "h64-softmax-tile",
            "h64-output",
            "h64-kv-nope-load",
            "h64-qk-pv-issue",
            "h64-qk-wait",
            "h64-pv-wait",
            "h64-valid-mask",
            "h64-k-rope-load",
        }
    ),
    head128: frozenset(
        {
            "h128-q-load",
            "h128-softmax-tile",
            "h128-output",
            "h128-k-load",
            "h128-v-load",
            "h128-qk-pv-issue",
            "h128-qk-wait",
            "h128-pv-wait",
            "h128-valid-mask",
        }
    ),
    head128_small: frozenset(
        {
            "h128-small-q-load-output",
            "h128-small-kv-load",
            "h128-small-qk-pv-issue",
            "h128-small-valid-mask",
            "h128-small-clc",
            "h128-small-softmax",
        }
    ),
}
EXPECTED_EVENT_SETS = set(EXPECTED_EVENT_NAMES.values())


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
        pytest.skip("sparse FlashMLA IKET verification requires CUDA in both PyTorch and TVM")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("sparse FlashMLA IKET verification requires an SM100 GPU")


@pytest.mark.parametrize(("module", "config"), CASES)
def test_sparse_flashmla_declares_expected_warp_uniform_ranges(module, config) -> None:
    func = module.get_kernel(**config)
    assert len(func.params) == 8
    script = func.script(show_meta=False)
    declarations = set(re.findall(r'T\.cuda\.iket\.range_start\("([^"]+)"', script))
    assert frozenset(module.IKET_EVENT_NAMES) == EXPECTED_EVENT_NAMES[module]
    assert declarations == EXPECTED_EVENT_NAMES[module]
    assert "profiler_buffer" not in script
    assert "clock64" not in script


def test_sparse_flashmla_module_entry_uses_orchestrator_defaults(
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

    monkeypatch.setattr(flash_mla_sparse_fwd.iket, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["flash_mla_sparse_fwd"])
    flash_mla_sparse_fwd.main()

    assert captured["main"].func is flash_mla_sparse_fwd._profile_iket_workload
    assert captured["kwargs"] == {
        "output_dir": "/tmp/flashmla-iket",
        "postprocess": "all",
        "clobber": True,
        "timeout": 600.0,
        "keep": False,
        "max_ts_cnt_per_warp": None,
    }
    output = capsys.readouterr().out
    assert "IKET output directory: /tmp/flashmla-iket" in output
    assert f"IKET artifact: {tmp_path / 'trace.json'}" in output


@pytest.mark.parametrize(("module", "config"), CASES)
def test_sparse_flashmla_strip_and_official_metadata(module, config) -> None:
    _require_sm100()
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    ir_module = tvm.IRModule({"main": module.get_kernel(**config)})

    plain = tvm.compile(ir_module, target=target, tir_pipeline="tirx")
    plain_source = _sources(plain.mod)
    assert "iket" not in plain_source.lower()

    official = IketProfiler().compile(ir_module, target=target)
    source = _sources(official.mod)
    assert "__iket_meta_info" in source
    declaration_names = set(re.findall(r"__iket_evt_decl_([a-z0-9_]+)_\d+_attrs", source))
    assert declaration_names == {name.replace("-", "_") for name in module.IKET_EVENT_NAMES}


def test_sparse_flashmla_external_trace_contract() -> None:
    trace_path = os.environ.get("TIRX_FLASHMLA_IKET_OFFICIAL_TRACE_JSON")
    if not trace_path:
        pytest.skip("set TIRX_FLASHMLA_IKET_OFFICIAL_TRACE_JSON to a locked run-iket trace")

    trace = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    strings = trace["stringTable"]
    observed = []
    for launch in trace["launches"]:
        names = frozenset(strings[item["rangeNameIdx"]] for item in launch["ranges"])
        if names not in EXPECTED_EVENT_SETS:
            continue
        observed.append(names)
        assert all(item["startTs"] <= item["endTs"] for item in launch["ranges"])
        assert all(
            len(item["internalEvents"]) == 2
            and item["internalEvents"][0]["eventId"] == item["internalEvents"][1]["eventId"]
            and item["internalEvents"][0]["timestamp"] == item["startTs"]
            and item["internalEvents"][1]["timestamp"] == item["endTs"]
            for item in launch["ranges"]
        )

    assert set(observed) == EXPECTED_EVENT_SETS

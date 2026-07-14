import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from tirx_kernels.bench_suite import run
from tirx_kernels.bench_suite.baseline_view import render_markdown
from tirx_kernels.bench_suite.ratio_diff import build_report
from tirx_kernels.megakernel.moe import BENCH_CONFIGS as MEGAKERNEL_MOE_BENCH_CONFIGS


def test_default_workloads_include_full_megakernel_moe_sweep() -> None:
    workloads = run.load_workloads(run.DEFAULT_WORKLOADS)
    megakernel_moe_workloads = [w for w in workloads if w["kernel"] == "megakernel_moe"]

    assert {w["config"] for w in megakernel_moe_workloads} == {
        config["label"] for config in MEGAKERNEL_MOE_BENCH_CONFIGS
    }
    assert all(w["num_gpus"] == 1 for w in megakernel_moe_workloads)
    assert all("timer" not in w for w in megakernel_moe_workloads)


def test_ratio_report_keeps_grouped_tir_schedulers_out_of_references() -> None:
    baseline = {
        "results": [
            {
                "kernel": "megakernel_moe",
                "label": "moe_a3b_bs1_all",
                "status": "ok",
                "impls": {
                    "tir_static": 10.0,
                    "tir_dynamic": 11.0,
                    "tir_unfused": 12.0,
                    "sglang_full": 13.0,
                    "flashinfer_full": 14.0,
                },
            }
        ]
    }
    current = {
        "results": [
            {
                "kernel": "megakernel_moe",
                "label": "moe_a3b_bs1_all",
                "status": "ok",
                "impls": {
                    "tir_static": 10.0,
                    "tir_dynamic": 11.0,
                    "tir_unfused": 12.0,
                    "sglang_full": 13.0,
                    "flashinfer_full": 14.0,
                },
            }
        ]
    }

    report, regressions = build_report(baseline, current)

    assert regressions == 0
    assert "| megakernel_moe | moe_a3b_bs1_all | tir_static | sglang_full |" in report
    assert "| megakernel_moe | moe_a3b_bs1_all | tir_dynamic | sglang_full |" in report
    assert "| megakernel_moe | moe_a3b_bs1_all | tir_unfused | sglang_full |" in report


def test_baseline_view_renders_grouped_implementations_in_one_row() -> None:
    payload = {
        "timestamp": "now",
        "label": "test",
        "git": {},
        "results": [
            {
                "kernel": "megakernel_moe",
                "label": "moe_a3b_bs128_all",
                "status": "ok",
                "impls": {
                    "tir_static": 20.0,
                    "tir_dynamic": 21.0,
                    "tir_unfused": 22.0,
                    "sglang_full": 23.0,
                    "flashinfer_full": 24.0,
                },
            },
            {
                "kernel": "megakernel_moe",
                "label": "moe_a3b_bs1_all",
                "status": "ok",
                "impls": {
                    "tir_static": 10.0,
                    "tir_dynamic": 11.0,
                    "tir_unfused": 12.0,
                    "sglang_full": 13.0,
                    "flashinfer_full": 14.0,
                },
            },
        ],
    }

    markdown = render_markdown(payload, "test.json")

    assert (
        "| config | tir_static (µs) | tir_dynamic (µs) | tir_unfused (µs) | "
        "sglang_full (µs) | flashinfer_full (µs) |" in markdown
    )
    assert "| `moe_a3b_bs1_all` | 10.0000 | 11.0000 | 12.0000 | 13.0000 | 14.0000 |" in markdown
    assert markdown.count("`moe_a3b_bs1_all`") == 1
    assert markdown.index("`moe_a3b_bs1_all`") < markdown.index("`moe_a3b_bs128_all`")


def test_baseline_view_keeps_single_tir_ratio_table() -> None:
    payload = {
        "results": [
            {
                "kernel": "gemm",
                "label": "m1024",
                "status": "ok",
                "impls": {"tir": 10.0, "reference": 12.0},
            }
        ]
    }

    markdown = render_markdown(payload, "test.json")

    assert "| config | ours impl | ours (µs) | ref impl |" in markdown
    assert "| `m1024` | tir | 10.0000 | reference | 12.0000 | 1.200 | — |" in markdown


def test_load_workloads_accepts_multigpu_megamoe(tmp_path: Path) -> None:
    workloads = tmp_path / "workloads.yaml"
    workloads.write_text(
        """
defaults: {}
workloads:
  - kernel: deepgemm_fp8_fp4_mega_moe
    config: t64_m64_h7168_i3072_e384_k6_g6
    timer: megamoe
    num_gpus: 6
"""
    )

    assert run.load_workloads(workloads) == [
        {
            "kernel": "deepgemm_fp8_fp4_mega_moe",
            "config": "t64_m64_h7168_i3072_e384_k6_g6",
            "timer": "megamoe",
            "num_gpus": 6,
        }
    ]


@pytest.mark.parametrize("num_gpus", [0, -1, True, "2"])
def test_load_workloads_rejects_invalid_gpu_count(tmp_path: Path, num_gpus) -> None:
    workloads = tmp_path / "workloads.yaml"
    workloads.write_text(
        f"workloads:\n  - {{kernel: kernel, config: config, num_gpus: {json.dumps(num_gpus)}}}\n"
    )

    with pytest.raises(ValueError, match="num_gpus must be a positive integer"):
        run.load_workloads(workloads)


def test_load_workloads_rejects_megamoe_budget_override(tmp_path: Path) -> None:
    workloads = tmp_path / "workloads.yaml"
    workloads.write_text(
        """
workloads:
  - {kernel: kernel, config: config, timer: megamoe, warmup: 10}
"""
    )

    with pytest.raises(ValueError, match="fixed DeepGEMM protocol"):
        run.load_workloads(workloads)


def test_gpu_pool_acquires_and_releases_multiple_cards_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = run.GpuPool(allowed={"0", "1", "2"})
    monkeypatch.setattr(pool, "_occupied_indices", lambda: set())
    monkeypatch.setattr(pool, "_all_gpus", lambda: [("0", "GPU-0"), ("1", "GPU-1"), ("2", "GPU-2")])
    monkeypatch.setattr(run.random, "sample", lambda population, count: population[:count])

    assert pool.acquire_many(2) == ("0", "1")
    assert pool.acquire() == "2"
    assert pool._owned == {"0", "1", "2"}

    pool.release_many(("0", "1"))
    pool.release("2")
    assert pool._owned == set()


def test_gpu_pool_prioritizes_larger_waiting_claims(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = run.GpuPool(allowed={"0", "1", "2"})
    monkeypatch.setattr(pool, "_occupied_indices", lambda: set())
    monkeypatch.setattr(pool, "_all_gpus", lambda: [("0", "GPU-0"), ("1", "GPU-1"), ("2", "GPU-2")])
    monkeypatch.setattr(run.random, "sample", lambda population, count: population[:count])
    monkeypatch.setattr(run, "POLL_INTERVAL", 0.001)
    with pool._lock:
        pool._owned.add("2")

    results = {}
    large_done = threading.Event()
    single_done = threading.Event()

    def acquire_large() -> None:
        results["large"] = pool.acquire_many(3)
        large_done.set()

    def acquire_single() -> None:
        results["single"] = pool.acquire_many(1)
        single_done.set()

    large_thread = threading.Thread(target=acquire_large)
    single_thread = threading.Thread(target=acquire_single)
    large_thread.start()
    while True:
        with pool._lock:
            if pool._waiters:
                break
        time.sleep(0.001)
    single_thread.start()
    time.sleep(0.01)
    assert not single_done.is_set()

    pool.release("2")
    assert large_done.wait(1)
    assert results["large"] == ("0", "1", "2")
    assert not single_done.is_set()

    pool.release_many(results["large"])
    assert single_done.wait(1)
    pool.release(results["single"])
    large_thread.join()
    single_thread.join()


def test_active_strangers_are_merged_across_assigned_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    active = {"0": {101: 1.0}, "2": {101: 4.0, 202: 3.0}}
    monkeypatch.setattr(
        run, "_active_strangers", lambda gpu_index, our_pids, sm_threshold: active[gpu_index]
    )

    assert run._active_strangers_on_gpus(("0", "2"), {999}, 0.0) == {101: 4.0, 202: 3.0}


def test_run_one_passes_multigpu_assignment_to_megamoe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakePool:
        util_threshold = 0.0

        def __init__(self) -> None:
            self.released = None

        def acquire_many(
            self, count: int, *, cancel_event: threading.Event | None = None
        ) -> tuple[str, ...]:
            assert count == 2
            assert cancel_event is None
            return ("2", "4")

        def release_many(self, indices: tuple[str, ...]) -> None:
            self.released = indices

    pool = FakePool()
    captured = {}

    def fake_run_subprocess_monitored(
        cmd, env, cwd, log_path, gpu_indices, monitor_interval, sm_threshold, cancel_event
    ):
        assert cancel_event is None
        captured.update(cmd=cmd, env=env, gpu_indices=gpu_indices)
        json_path = Path(cmd[cmd.index("--json-file") + 1])
        json_path.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "kernel": "deepgemm_fp8_fp4_mega_moe",
                            "label": "t64_m64_h7168_i3072_e384_k6_g2",
                            "status": "OK",
                            "impls": {"deepgemm": 10.5, "tirx": 10.0},
                            "round_samples": {"deepgemm": [10.0, 11.0], "tirx": [9.5, 10.5]},
                            "errors": {},
                        }
                    ]
                }
            )
        )
        return 0, False, [], False

    monkeypatch.setattr(run, "_run_subprocess_monitored", fake_run_subprocess_monitored)

    record = run.run_one(
        {
            "kernel": "deepgemm_fp8_fp4_mega_moe",
            "config": "t64_m64_h7168_i3072_e384_k6_g2",
            "timer": "megamoe",
            "num_gpus": 2,
        },
        pool,
        tmp_path,
        rounds=2,
        cooldown=0,
    )

    assert record["status"] == "ok"
    assert record["gpu"] == "2,4"
    assert record["gpus"] == ["2", "4"]
    assert record["num_gpus"] == 2
    assert record["impls"] == {"deepgemm": 10.5, "tirx": 10.0}
    assert record["round_samples"] == {"deepgemm": [10.0, 11.0], "tirx": [9.5, 10.5]}
    assert captured["gpu_indices"] == ("2", "4")
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "2,4"
    assert captured["cmd"][captured["cmd"].index("--timer") + 1] == "megamoe"
    assert captured["cmd"][captured["cmd"].index("--cooldown") + 1] == "0"
    assert "--round-cooldown" not in captured["cmd"]
    assert pool.released == ("2", "4")


def test_gpu_pool_wait_is_cancelled_promptly(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = run.GpuPool(allowed={"0"})
    monkeypatch.setattr(pool, "_occupied_indices", lambda: set())
    monkeypatch.setattr(pool, "_all_gpus", lambda: [("0", "GPU-0")])
    with pool._lock:
        pool._owned.add("0")

    cancel_event = threading.Event()
    timer = threading.Timer(0.05, cancel_event.set)
    timer.start()
    try:
        with pytest.raises(run._BenchSuiteCancelled):
            pool.acquire_many(1, cancel_event=cancel_event)
    finally:
        timer.cancel()


def test_monitored_subprocess_is_terminated_on_cancel(tmp_path: Path) -> None:
    cancel_event = threading.Event()
    timer = threading.Timer(0.05, cancel_event.set)
    timer.start()
    started = time.monotonic()
    try:
        returncode, interfered, intruders, cancelled = run._run_subprocess_monitored(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            os.environ.copy(),
            str(tmp_path),
            tmp_path / "subprocess.log",
            (),
            0.01,
            0.0,
            cancel_event,
        )
    finally:
        timer.cancel()

    assert cancelled
    assert returncode != 0
    assert not interfered
    assert intruders == []
    assert time.monotonic() - started < 2


def test_run_scheduled_jobs_stops_after_first_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake_run_one(workload, pool, log_dir, **kwargs):
        calls.append(workload["config"])
        return {
            "kernel": workload["kernel"],
            "config": workload["config"],
            "status": "FAIL",
            "error": "deterministic failure",
        }

    monkeypatch.setattr(run, "run_one", fake_run_one)
    records, retry_log = run.run_scheduled_jobs(
        [{"kernel": "kernel", "config": "fails"}, {"kernel": "kernel", "config": "must_not_start"}],
        object(),
        tmp_path,
        rounds=1,
        cooldown=0,
        bench_aggregate="mean",
        cpu_workers=1,
    )

    assert calls == ["fails"]
    assert retry_log == []
    assert [(record["config"], record["status"], record["attempt"]) for record in records] == [
        ("fails", "FAIL", 1)
    ]


def test_run_scheduled_jobs_retries_interference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = []

    def fake_run_one(workload, pool, log_dir, *, attempt, **kwargs):
        attempts.append(attempt)
        if attempt == 1:
            return {
                "kernel": workload["kernel"],
                "config": workload["config"],
                "status": "INTERFERED",
                "error": "active neighbor",
                "intruder_pids": [123],
            }
        return {"kernel": workload["kernel"], "config": workload["config"], "status": "ok"}

    monkeypatch.setattr(run, "run_one", fake_run_one)
    records, retry_log = run.run_scheduled_jobs(
        [{"kernel": "kernel", "config": "config"}],
        object(),
        tmp_path,
        rounds=1,
        cooldown=0,
        bench_aggregate="mean",
        cpu_workers=1,
    )

    assert attempts == [1, 2]
    assert records[0]["status"] == "ok"
    assert records[0]["attempt"] == 2
    assert retry_log == [("kernel", "config", 1, "intruders [123]")]


def test_run_scheduled_jobs_cancels_inflight_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    active_started = threading.Event()
    calls = []

    def fake_run_one(workload, pool, log_dir, *, cancel_event, **kwargs):
        config = workload["config"]
        calls.append(config)
        if config == "fails":
            assert active_started.wait(1)
            return {
                "kernel": workload["kernel"],
                "config": config,
                "status": "FAIL",
                "error": "deterministic failure",
            }
        if config == "active":
            active_started.set()
            assert cancel_event.wait(1)
            return {"kernel": workload["kernel"], "config": config, "status": "CANCELLED"}
        raise AssertionError(f"unexpectedly started {config}")

    monkeypatch.setattr(run, "run_one", fake_run_one)
    records, retry_log = run.run_scheduled_jobs(
        [
            {"kernel": "kernel", "config": "fails"},
            {"kernel": "kernel", "config": "active"},
            {"kernel": "kernel", "config": "must_not_start"},
        ],
        object(),
        tmp_path,
        rounds=1,
        cooldown=0,
        bench_aggregate="mean",
        cpu_workers=2,
    )

    assert set(calls) == {"fails", "active"}
    assert retry_log == []
    assert [(record["config"], record["status"]) for record in records] == [("fails", "FAIL")]

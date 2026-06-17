"""Standard GEMM performance bench — value mode vs trace mode.

This is the canonical perf methodology for nymph-rust; it runs as part of the test
suite so every run reports the numbers and guards the trace-mode design invariant.
Do not invent ad-hoc benchmarks elsewhere — extend this one.

Methodology (matches STATUS.md "Performance"):
  - Fix one GEMM task tile: m=512, n=256, k=16384, cta_group=2, launch=(2,) — ONE
    persistent cluster that grid-strides over the tasks.
  - Scale the task count by widening n (n = 256*T → T n-tiles → T tasks); launch stays
    (2,) so the single cluster runs all T tasks via ForEachTask(grid_stride).
  - Time value (nr.interpret, with A/B inputs), raw trace (nr.trace, no payload
    inputs and no offline checkers), and full protocol_check (nr.check_protocol,
    trace plus offline checkers) at task counts TASKS. Take the best of REPS runs,
    linear-fit total = a + b*tasks, and report the per-task slope b plus the
    EXTRAPOLATE_TO-task extrapolation a + b*EXTRAPOLATE_TO.
  - Measure clean: do not set NYMPH_STATS (the profiler adds ~3 ms/task).

Design invariant asserted below: trace mode skips the numeric payload (the OpenBLAS
sgemm, the f16 decode, the TMA byte copy) and only computes control/protocol state, so
its per-task time MUST be lower than value mode's. If trace is not faster, the trace
path has a pathological overhead (e.g. retained/allocated events, per-op invalidation)
that defeats the whole point of trace mode — fix it.

Full protocol_check timing is reported separately so checker cost is visible without
being conflated with the raw trace-vs-value invariant.
"""

import time

import numpy as np
import nymph_rs as nr
from nymph_rs.kernels import build_fp16_bf16_gemm
from nymph_rs.kernels.fp16_bf16_gemm import Fp16Bf16GemmConfig

K = 16384
TASKS = [1, 2, 4]
EXTRAPOLATE_TO = 2048
REPS = 2


def _build(num_tasks):
    cfg = Fp16Bf16GemmConfig(m=512, n=256 * num_tasks, k=K, launch_shape=(2,))
    return cfg, build_fp16_bf16_gemm(cfg)


def _best(fn):
    best = float("inf")
    for _ in range(REPS):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def _best_checked(fn):
    best = float("inf")
    for _ in range(REPS):
        t = time.perf_counter()
        report = fn()
        elapsed = time.perf_counter() - t
        assert report["status"] == "Passed", report["status"]
        best = min(best, elapsed)
    return best


def _fit(xs, ys):
    a = np.vstack([np.asarray(xs, float), np.ones(len(xs))]).T
    slope, intercept = np.linalg.lstsq(a, np.asarray(ys, float), rcond=None)[0]
    return slope, intercept


def _measure():
    value_t, trace_t, check_t = [], [], []
    for num_tasks in TASKS:
        cfg, kernel = _build(num_tasks)
        a_t, b_t, _ = kernel.args
        rng = np.random.default_rng(0)
        a = rng.integers(-2, 3, size=(cfg.m, cfg.k)).astype(np.float32)
        b = rng.integers(-2, 3, size=(cfg.n, cfg.k)).astype(np.float32)
        value_t.append(_best(lambda: nr.interpret(kernel, {a_t: a, b_t: b})))
        trace_t.append(_best_checked(lambda: nr.trace(kernel)))
        check_t.append(_best_checked(lambda: nr.check_protocol(kernel)))
    return value_t, trace_t, check_t


def _format(value_t, trace_t, check_t):
    sv, iv = _fit(TASKS, value_t)
    st, it = _fit(TASKS, trace_t)
    sc, ic = _fit(TASKS, check_t)
    lines = [f"\nGEMM perf (m=512, n=256*T, k={K}, cta_group=2, launch=(2,)):"]
    for num_tasks, v, t, c in zip(TASKS, value_t, trace_t, check_t):
        lines.append(
            f"  T={num_tasks:<3d} value={v * 1000:9.1f} ms   trace={t * 1000:9.1f} ms   "
            f"protocol_check={c * 1000:9.1f} ms"
        )
    lines.append(
        f"  value: per-task={sv * 1000:8.2f} ms   {EXTRAPOLATE_TO}-task ~= {iv + EXTRAPOLATE_TO * sv:8.1f} s"
    )
    lines.append(
        f"  trace: per-task={st * 1000:8.2f} ms   {EXTRAPOLATE_TO}-task ~= {it + EXTRAPOLATE_TO * st:8.1f} s"
    )
    lines.append(
        f"  protocol_check: per-task={sc * 1000:8.2f} ms   {EXTRAPOLATE_TO}-task ~= {ic + EXTRAPOLATE_TO * sc:8.1f} s"
    )
    lines.append(f"  trace per-task speedup over value: {sv / st:.2f}x")
    return sv, st, "\n".join(lines)


def test_gemm_value_vs_trace_perf(capsys):
    value_t, trace_t, check_t = _measure()
    sv, st, report = _format(value_t, trace_t, check_t)
    with capsys.disabled():
        print(report)
    assert st < sv, (
        "trace mode must be faster than value mode — it skips the numeric payload "
        "(sgemm / f16 decode / TMA copy) and this benchmark excludes offline checkers. "
        f"A slower trace means a pathological per-op overhead in the trace path.\n{report}"
    )

# bench-suite

Pre-commit regression benchmark for TIRx kernels. Runs the curated workload
sweep in `workloads.yaml` against the **working tree**, assigns GPUs
automatically, and writes run JSON + reports under `.bench-suite/`.

```bash
cd /path/to/tirx-kernels-staging
pip install -e .

export TVM_PATH=/path/to/tvm
export PYTHONPATH="${TVM_PATH}/python"
export TVM_LIBRARY_PATH="${TVM_PATH}/build/lib"
# Do NOT set CUDA_VISIBLE_DEVICES — GPU selection is automatic.
```

Entry point: `python -m tirx_kernels.bench_suite` (same flags as `run.py`).

Import gate (kernels referenced in `workloads.yaml` only):

```bash
python -m tirx_kernels.bench_suite --check-imports
```

## Directory layout

| Kind | Files |
|------|--------|
| **Run** | `run.py`, `workloads.yaml` |
| **Pinned baseline (git)** | `tir.json`, `ref.json`, `ratio.json`, `baseline.md` |
| **Promote / report** | `promote_baseline.py`, `reaggregate_from_logs.py`, `ratio_diff.py`, `baseline_view.py` |

Run artifacts (logs, `runs/*.json`, `reports/*`) live under `.bench-suite/` and are not committed.

## Strategy (TL;DR)

1. **Pinned baseline lives in git** (`tir.json`, `ref.json`, `baseline.md`, `ratio.json`).
2. **One job = one workload** (kernel + config). A worker acquires a GPU, runs **one**
   bench subprocess: compile/prepare once, then **`--rounds N` in-bench** (each round:
   warmup + repeat, paired ours + ref when `--impls all`). Optional `--round-cooldown`
   between rounds.
3. **Retry until ok**: INTERFERED / subprocess failure → job goes back on the
   worker queue (another worker may pick it up later). `SKIP` workloads are not
   retried.
4. **Dynamic free GPU queue** (`--cpu-workers 0` = one worker per probe-OK GPU):
   workers pull jobs from a shared queue; each job `acquire()`s any free card,
   runs one subprocess, then `release()`s. Whoever finishes first grabs the next
   job and the next free GPU — no static workload→GPU binding, no overcommit.
5. **`--impls ours` / `baseline` / `all`** unchanged. Ratio Δ needs `--impls all`
   (paired in one subprocess). Daily iteration uses `--impls ours` vs pinned `tir.json`.

## Baseline files (git-tracked)

| File | Contents | Refresh when |
|------|----------|--------------|
| `tir.json` | Our kernel times only (`tir` / `tirx`) | Kernel changes |
| `ref.json` | Reference impl times only | Env / library upgrades |
| `ratio.json` | Saved ref/ours ratio per workload | Auto on promote / reaggregate |
| `baseline.md` | Human view: ours + ref + ratio | Auto on promote / reaggregate |

Promote through `promote_baseline.py` only (never bare `cp`).

## `--impls` modes

| Mode | When | Report | Promote |
|------|------|--------|---------|
| `ours` (**default**) | Daily kernel work | abs µs vs pin | `--tir` |
| `baseline` | Refresh references | ref abs µs vs pin | `--ref` |
| `all` | Full ratio check | ours + ref + ratio Δ | `--both` |

## Workflows

### Daily: kernel iteration

```bash
python -m tirx_kernels.bench_suite --impls ours
python tirx_kernels/bench_suite/promote_baseline.py \
  .bench-suite/runs/<id>.json --tir
```

Before merge, add `--rounds 5` and promote.

### Refresh references (rare)

```bash
python -m tirx_kernels.bench_suite --impls baseline --rounds 5
python tirx_kernels/bench_suite/reaggregate_from_logs.py --ref
```

Or separate ours + baseline runs, then:

```bash
python -m tirx_kernels.bench_suite --impls ours --rounds 5
python -m tirx_kernels.bench_suite --impls baseline --rounds 5
python tirx_kernels/bench_suite/promote_baseline.py .bench-suite/runs/<ours-id>.json --tir
python tirx_kernels/bench_suite/reaggregate_from_logs.py --ref
```

### Full ratio sweep

```bash
python -m tirx_kernels.bench_suite --impls all --rounds 5
less .bench-suite/reports/latest/bench.md
python tirx_kernels/bench_suite/promote_baseline.py .bench-suite/runs/<id>.json --both
```

Spot-check one workload: `python -m tirx_kernels.bench --kernel ... --config ... --impls all --rounds 5`

## Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--rounds N` | `1` | In-bench rounds (warmup+repeat each) per subprocess |
| `--round-cooldown` | `1.0` | Seconds between in-bench rounds |
| `--bench-aggregate` | `mean` | `mean`, `median`, or `trimmed_mean` over round samples |
| `--cpu-workers` | `0` (= GPU count) | Concurrent workload workers (capped at GPU count) |
| `--impls` | `ours` | `ours`, `baseline`, or `all` |
| `--util-threshold` | `0` | Skip GPUs with SM utilization above this percent |
| `--mem-threshold` | `0` | Skip GPUs with compute-app memory-used percent above this percent |

Promoted baselines often use **trimmed_mean ×5** via `reaggregate_from_logs.py`.

## Ratio rules

- **ref impl** = fastest non-ours impl in baseline, fixed across runs.
- **ratio** = ref/ours (>1 means ours is faster).
- **ratio Δ** in `bench.md` = current ratio vs saved `ratio.json`.

## Outputs

| Path | Description |
|------|-------------|
| `.bench-suite/runs/<id>.json` | Aggregated run results (times in microseconds) |
| `.bench-suite/reports/<id>/bench.md` | Main diff report |
| `.bench-suite/logs/*__<role>_a<N>.log` | Subprocess stdout (N proton trees when `--rounds N`) |

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | No regressions over threshold (or no baseline yet) |
| `2` | Config error |
| `3` | One or more regressions exceeded `--threshold` |

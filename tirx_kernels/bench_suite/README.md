# bench-suite

Pre-commit regression benchmark for TIRx kernels. Runs the curated workload
sweep in `workloads.yaml` against the **working tree**, assigns GPUs
automatically, and writes run JSON + reports under `.bench-suite/`.

```bash
cd /path/to/tirx-kernels
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

### SGLang FP8 paged MQA exploration

The optional SGLang CuTeDSL reference is imported lazily; the normal TIRx and
DeepGEMM paths do not require SGLang. To run the 80-shape SM100 comparison against
SGLang's current production picker, expose a matching SGLang checkout and install
the CUTLASS DSL version required by that checkout:

```bash
export SGLANG_PATH=/path/to/sglang
export PYTHONPATH="${SGLANG_PATH}/python:${PYTHONPATH}"

python -m tirx_kernels.bench_suite \
  --workloads tirx_kernels/bench_suite/workloads_sglang_fp8_paged_mqa.yaml \
  --rounds 5 \
  --bench-aggregate trimmed_mean
```

This is a kernel-only Proton comparison: Q/context reshaping, schedule metadata,
and CuTe JIT compilation happen outside the timed region. Runs are written under
`.bench-suite/`; inspect that run's `errors` and require every row to contain
`tirx`, `deepgemm`, and `sglang_cutedsl`. Do not promote this exploratory sweep to
the pinned baseline until its shape set and winning regions have been reviewed.

## Directory layout

| Kind | Files |
|------|--------|
| **Run** | `run.py`, `workloads.yaml`, `workloads_sglang_fp8_paged_mqa.yaml` |
| **Pinned baseline (git)** | `baseline.json`, `baseline.md` |
| **Promote / report** | `promote_baseline.py`, `ratio_diff.py`, `baseline_view.py` |

Run artifacts (logs, `runs/*.json`, `reports/*`) live under `.bench-suite/` and are not committed.

## Strategy (TL;DR)

1. **Pinned baseline lives in git** (`baseline.json`, `baseline.md`).
2. **One job = one workload** (kernel + config). A worker atomically acquires the
   workload's requested GPU count, then runs **one**
   bench subprocess that always benches our kernel **and** every reference impl:
   compile/prepare once, then **`--rounds N` in-bench** (each round: warmup + repeat).
   `--cooldown` is applied before every implementation in every round.
3. **Retry until ok**: INTERFERED / subprocess failure → job goes back on the
   worker queue (another worker may pick it up later). `SKIP` workloads are not
   retried.
4. **Dynamic free GPU queue** (`--cpu-workers 0` = one worker per probe-OK GPU):
   workers pull jobs from a shared queue; each job atomically claims all required
   free cards, runs one subprocess, then releases the full set. Whoever finishes
   first grabs the next satisfiable job — no static workload→GPU binding and no
   partial multi-GPU claims. Larger waiting claims take priority so single-GPU
   traffic cannot starve 2/4/6-GPU workloads.
5. **Ratio regression report** compares current ref/ours ratio vs the pinned
   `baseline.json` ratio (computed from its ours + ref impls). Promote a run over
   the baseline with `promote_baseline.py`.

## Baseline files (git-tracked)

| File | Contents | Refresh when |
|------|----------|--------------|
| `baseline.json` | Our kernel times + reference impl times per workload | Kernel changes, env / library upgrades |
| `baseline.md` | Human view: ours + ref + ratio | Auto on promote |

Promote through `promote_baseline.py` only (never bare `cp`).

## Workflows

### Daily: kernel iteration

```bash
python -m tirx_kernels.bench_suite
python tirx_kernels/bench_suite/promote_baseline.py \
  .bench-suite/runs/<id>.json --merge
```

Before merge, add `--rounds 5` and promote.

### Refresh the pinned baseline (rare)

```bash
python -m tirx_kernels.bench_suite --rounds 5 --bench-aggregate trimmed_mean
python tirx_kernels/bench_suite/promote_baseline.py .bench-suite/runs/<id>.json --merge
```

`promote_baseline.py <run>.json --merge` patches the ok rows from a run JSON into
`baseline.json` by `(kernel, config)` and regenerates `baseline.md`. Use
`--rounds 5 --bench-aggregate trimmed_mean` for a promoted baseline (drops the
fastest and slowest round).

Spot-check one workload: `python -m tirx_kernels.bench --kernel ... --config ... --rounds 5`

## Workload fields

Each `workloads.yaml` entry requires `kernel` and `config`. Optional fields are
`timer`, `warmup`, `repeat`, and `num_gpus` (default `1`). Multi-GPU jobs receive
the acquired physical indices as an ordered, comma-separated
`CUDA_VISIBLE_DEVICES` value and all assigned cards are monitored for interference.

MegaMoE entries use `timer: megamoe`, which invokes the dedicated DeepGEMM
`bench_kineto` protocol. Do not set `warmup` or `repeat` for this timer because
the protocol fixes its own 30-test schedule.

## Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--rounds N` | `1` | In-bench rounds (warmup+repeat each) per subprocess |
| `--cooldown` | `1.0` | Seconds before every implementation in every round |
| `--bench-aggregate` | `mean` | `mean`, `median`, or `trimmed_mean` over round samples |
| `--cpu-workers` | `0` (= GPU count) | Concurrent workload workers (capped at GPU count) |
| `--util-threshold` | `0` | Skip GPUs with SM utilization above this percent |
| `--mem-threshold` | `0` | Skip GPUs with compute-app memory-used percent above this percent |

Promoted baselines often use **trimmed_mean ×5** (`--rounds 5 --bench-aggregate trimmed_mean`).

## Ratio rules

- **ref impl** = fastest non-ours impl in baseline, fixed across runs.
- **ratio** = ref/ours (>1 means ours is faster).
- **ratio Δ** in `bench.md` = current ratio vs the baseline ratio (computed from
  `baseline.json`'s ours + ref impls).

## Outputs

| Path | Description |
|------|-------------|
| `.bench-suite/runs/<id>.json` | Aggregated run results (times in microseconds) |
| `.bench-suite/reports/<id>/bench.md` | Main diff report |
| `.bench-suite/logs/*__a<N>.log` | Benchmark subprocess stdout for each attempt |

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | No regressions over threshold (or no baseline yet) |
| `2` | Config error |
| `3` | One or more regressions exceeded `--threshold` |

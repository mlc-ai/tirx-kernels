# tir-bench

Pre-commit regression benchmark for TIRx kernels. Runs the curated workload
sweep in `workloads.yaml` against the **working tree**, assigns GPUs
automatically, and writes run JSON + reports under `.tir-bench/`.

```bash
cd /path/to/tirx-kernels-staging
pip install -e .

export TVM_PATH=/path/to/tvm
export PYTHONPATH="${TVM_PATH}/python"
export TVM_LIBRARY_PATH="${TVM_PATH}/build/lib"
# Do NOT set CUDA_VISIBLE_DEVICES — GPU selection is automatic.
```

Entry point: `python -m tirx_kernels.tir_bench` (same flags as `run.py`).

Import gate (kernels referenced in `workloads.yaml` only):

```bash
python -m tirx_kernels.tir_bench --check-imports
```

## Directory layout

| Kind | Files |
|------|--------|
| **Run** | `run.py`, `workloads.yaml` |
| **Pinned baseline (git)** | `tir.json`, `ref.json`, `ratio.json`, `baseline.md` |
| **Promote / report** | `promote_baseline.py`, `reaggregate_from_logs.py`, `ratio_diff.py`, `baseline_view.py` |

Run artifacts (logs, `runs/*.json`, `reports/*`) live under `.tir-bench/` and are not committed.

## Strategy (TL;DR)

1. **Pinned baseline lives in git** under this directory (`tir.json`, `ref.json`,
   `baseline.md`, `ratio.json`) — not in `.tir-bench/reports/`.
2. **Split jobs**: `run.py` benches ours and refs in **separate subprocesses**
   (not paired like `tirx_kernels.bench --impls all`). Daily work uses
   `--impls ours`; ref refresh uses `--impls baseline`.
3. **×5 rounds** before promote; refs are usually reaggregated from logs with
   **trimmed_mean** (drop fastest + slowest ok round).
4. **One report per run**: `reports/<id>/bench.md` — ours ms, ref ms, ratio,
   ratio Δ vs saved `ratio.json`. Human baseline view: `baseline.md`.
5. **Trust ratio Δ** only when spread is low; spot-check suspicious rows with
   paired `python -m tirx_kernels.bench --impls all`.

## Baseline files (git-tracked)

| File | Contents | Refresh when |
|------|----------|--------------|
| `tir.json` | Our kernel times only (`tir` / `tirx`) | Kernel changes |
| `ref.json` | Reference impl times only | Env / library upgrades |
| `ratio.json` | Saved ref/ours ratio per workload | Auto on promote / reaggregate |
| `baseline.md` | Human view: ours + ref + ratio | Auto on promote / reaggregate |

`tir.json` and `ref.json` have **independent update cadences**; they are joined
at diff time. Always promote through `promote_baseline.py` (never bare `cp`).

`sparse_flashmla_*` uses upstream [FlashMLA](https://github.com/deepseek-ai/FlashMLA)
(`flashmla` impl). Set `FLASH_MLA_PATH` to a built tree (default `~/FlashMLA`).

## `--impls` modes

| Mode | When | Report | Promote |
|------|------|--------|---------|
| `ours` (**default**) | Daily kernel work | `bench.md` (ours abs ms vs pin) | `--tir` |
| `baseline` | Refresh references | `bench.md` (ref abs ms vs pin) | `--ref` or `reaggregate_from_logs.py --ref` |
| `all` | Full ratio check (optional) | `bench.md` (ours + ref + ratio Δ) | split promote preferred |

Ratio columns in `bench.md` need **both** ours and ref in the **current** run,
so full ratio Δ normally requires `--impls all`. Day-to-day iteration compares
ours abs ms against pinned `tir.json`; saved ratios live in `ratio.json` /
`baseline.md`.

## Workflows

### Daily: kernel iteration

```bash
python -m tirx_kernels.tir_bench --impls ours
# before merge: add --rounds 5
python tirx_kernels/tir_bench/promote_baseline.py \
  .tir-bench/runs/<id>.json --tir
git add tirx_kernels/tir_bench/tir.json tirx_kernels/tir_bench/baseline.md tirx_kernels/tir_bench/ratio.json && git commit
```

### Refresh references (rare)

```bash
python -m tirx_kernels.tir_bench \
  --impls baseline \
  --rounds 5 \
  --bench-aggregate mean \
  --restable-reps 0

# Prefer trim×5 from logs (keeps flashinfer etc. merged cleanly):
python tirx_kernels/tir_bench/reaggregate_from_logs.py --ref

git add tirx_kernels/tir_bench/ref.json tirx_kernels/tir_bench/baseline.md tirx_kernels/tir_bench/ratio.json && git commit
```

After a split sweep (ours job + baseline job in one `run.py` invocation with
separate roles), promote ours and reaggregate refs:

```bash
python -m tirx_kernels.tir_bench --impls ours --rounds 5
python -m tirx_kernels.tir_bench --impls baseline --rounds 5 --restable-reps 0
python tirx_kernels/tir_bench/promote_baseline.py .tir-bench/runs/<ours-id>.json --tir
python tirx_kernels/tir_bench/reaggregate_from_logs.py --ref
```

### Optional: full ratio report

```bash
python -m tirx_kernels.tir_bench --impls all --rounds 5 --restable-reps 0
less .tir-bench/reports/latest/bench.md
```

Do **not** use full `--impls all` ×5 as the default daily driver (cost vs benefit).
For suspicious ratio Δ, confirm with paired bench:

```bash
python -m tirx_kernels.bench --kernel ... --config ... --impls all --warmup 100 --repeat 30
```

## Multi-round aggregation

Each job is one `(workload, role, round)` subprocess. INTERFERED jobs retry up
to `--max-retry` before the round is discarded.

| Flag | Default | Meaning |
|------|---------|---------|
| `--rounds N` | `1` | Rounds per (workload, role) |
| `--bench-aggregate` | `mean` | `mean`, `median`, or `trimmed_mean` over ok rounds |
| `--max-retry N` | `5` | Max attempts per job on INTERFERED |
| `--min-ok-rounds N` | `1` | Min ok rounds per impl to aggregate |

Promoted baselines typically use **trimmed_mean ×5** (`reaggregate_from_logs.py`).
Run JSON aggregation uses `--bench-aggregate` (default `mean`).

## Ratio rules

- **ref impl** = fastest non-ours impl in baseline, **fixed** across runs.
- **ratio** = ref/ours (>1 means ours is faster).
- **ratio Δ** in `bench.md` = current ratio vs saved `ratio.json`.
- **ours Δ / ref Δ** = abs ms vs pinned `tir.json` / `ref.json`.
- |ratio Δ| > 5% with high spread → treat as inconclusive.
- ⚠ in `bench.md` when ref abs ms drifted >20% vs pin (unstable env).

## Restable phase (`--impls all` only)

After `--impls all`, workloads with |ratio Δ| > `--restable-threshold` (3%) are
re-benched `--restable-reps` rounds (median) and `bench.md` is rewritten.
Skip with `--restable-reps 0` (recommended for baseline refresh).

## Outputs

| Path | Description |
|------|-------------|
| `.tir-bench/runs/<id>.json` | Raw run results |
| `.tir-bench/reports/<id>/summary.md` | Raw per-kernel table |
| `.tir-bench/reports/<id>/bench.md` | **Main diff report** |
| `.tir-bench/reports/latest/` | Symlink → latest run id |
| `baseline.md` | **Pinned baseline (git)** |

Ignore ad-hoc `rebaseline-*.md` under `.tir-bench/reports/` — one-off analysis.

## Helpers

```bash
# Promote
python tirx_kernels/tir_bench/promote_baseline.py .tir-bench/runs/<id>.json --tir
python tirx_kernels/tir_bench/promote_baseline.py .tir-bench/runs/<id>.json --ref
python tirx_kernels/tir_bench/promote_baseline.py .tir-bench/runs/<id>.json --tir --merge

# Re-trim from logs (after ×5 sweep)
python tirx_kernels/tir_bench/reaggregate_from_logs.py --tir --ref

# Render baseline.md / ratio.json from pins
python tirx_kernels/tir_bench/baseline_view.py
python tirx_kernels/tir_bench/ratio_diff.py --refresh-ratio-json

# Regenerate bench.md for an existing run
python tirx_kernels/tir_bench/ratio_diff.py .tir-bench/runs/<id>.json
```

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | No regressions over threshold (or no baseline yet) |
| `2` | Config error |
| `3` | One or more regressions exceeded `--threshold` |

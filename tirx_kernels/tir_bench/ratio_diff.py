#!/usr/bin/env python3
"""Ratio-based regression diff for tir-bench.

For each (kernel, config) we measure with multiple impls, compute the
ratio ref/ours where ref = fastest non-ours impl picked in baseline and
held fixed across runs. Higher ref/ours = ours is faster than ref =
better. Diff that ratio between baseline and current: positive ratio Δ
means we got faster vs ref (improvement), negative means slower
(regression).

Rationale: under GPU contention every impl slows by a similar factor,
so absolute-µs diffs are dominated by that noise. The ratio between
ours and a same-run reference is unchanged by uniform slowdown, so a
moving ratio is a real perf signal. Rows where the reference impl
itself drifted > 20% are flagged ⚠ — workload's environment was
unstable, so the ratio Δ is less trustworthy.

The report lists every comparable workload in a single table, sorted by
ratio Δ from most-improved to most-regressed (positive → negative).
Baseline workloads that were attempted this run but produced no comparable
measurement (failed, interfered, or missing an impl) are listed in a separate
"Not comparable in current run" section so lost coverage is never silent.

Usage:
    python ratio_diff.py [baseline.json] [current.json] [-o PATH]

Importable as `build_report(baseline_path, current)` for use from
`run.py`.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

OUR_IMPLS = {"tir", "tirx"}
DEFAULT_RATIO_THRESHOLD = 1.0
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_LATEST_RUN = REPO_ROOT / ".tir-bench" / "latest.json"
DEFAULT_TIR_BASELINE = HERE / "tir.json"
DEFAULT_REF_BASELINE = HERE / "ref.json"
DEFAULT_RATIO_BASELINE = HERE / "ratio.json"


def _result_key(row: dict) -> tuple[str, str]:
    return row["kernel"], row.get("label") or row.get("config")


def _refs_only(impls: dict[str, float]) -> dict[str, float]:
    return {k: v for k, v in impls.items() if k not in OUR_IMPLS and v > 0}


def join_default_baseline() -> dict:
    """Join the checked-in tir.json + ref.json into one diffable payload.

    The two pinned baselines have independent update cadences and are never
    stored combined; the ref impls are folded into each tir result's `impls`.
    """
    here = Path(__file__).resolve().parent
    tir = json.loads((here / "tir.json").read_text()) if (here / "tir.json").exists() else {}
    ref = json.loads((here / "ref.json").read_text()) if (here / "ref.json").exists() else {}
    ref_idx = {_result_key(r): _refs_only(r.get("impls") or {}) for r in ref.get("results", [])}
    payload = {k: v for k, v in tir.items() if k != "results"}
    payload["results"] = [
        {
            **r,
            "impls": {
                **(r.get("impls") or {}),
                **ref_idx.get((r["kernel"], r.get("label") or r.get("config")), {}),
            },
        }
        for r in tir.get("results", [])
    ]
    return payload


def index(payload: dict) -> dict[tuple[str, str], dict[str, float]]:
    """{(kernel, config) -> {impl -> us}} for ok results."""
    out: dict[tuple[str, str], dict[str, float]] = {}
    for r in payload.get("results") or []:
        if r.get("status") != "ok":
            continue
        key = (r["kernel"], r.get("label") or r.get("config"))
        out[key] = dict(r.get("impls") or {})
    return out


def pick_ref(base_impls: dict[str, float]) -> str | None:
    """Pick the fastest non-ours impl from BASELINE; reused in current to
    keep ref fixed across runs."""
    refs = _refs_only(base_impls)
    if not refs:
        return None
    return min(refs, key=lambda k: refs[k])


def pick_ours(impls: dict[str, float]) -> str | None:
    for name in ("tir", "tirx"):
        if name in impls and impls[name] > 0:
            return name
    return None


def build_ratio_payload(tir_payload: dict, ref_payload: dict) -> dict:
    """Build ratio.json contents from pinned tir + ref (no run JSON needed)."""
    ref_idx = {
        _result_key(r): _refs_only(r.get("impls") or {}) for r in ref_payload.get("results", [])
    }
    out = {
        "timestamp": tir_payload.get("timestamp"),
        "label": tir_payload.get("label"),
        "git": tir_payload.get("git"),
        "kernel_tree": tir_payload.get("kernel_tree"),
        "results": [],
    }
    for r in tir_payload.get("results") or []:
        if r.get("status") != "ok":
            continue
        key = _result_key(r)
        impls = {**(r.get("impls") or {}), **ref_idx.get(key, {})}
        ref = pick_ref(impls)
        ours = pick_ours(impls)
        if ref is None or ours is None:
            continue
        ours_us = impls[ours]
        ref_us = impls[ref]
        if ours_us <= 0 or ref_us <= 0:
            continue
        out["results"].append(
            {
                "kernel": r["kernel"],
                "config": r.get("config"),
                "label": r.get("label") or r.get("config"),
                "status": "ok",
                "ref_impl": ref,
                "ours_impl": ours,
                "ours_us": ours_us,
                "ref_us": ref_us,
                "ratio": ref_us / ours_us,
            }
        )
    out["results"].sort(key=_result_key)
    return out


def ratio_index(payload: dict) -> dict[tuple[str, str], dict]:
    """{(kernel, config) -> ratio row} from ratio.json."""
    out: dict[tuple[str, str], dict] = {}
    for r in payload.get("results") or []:
        if r.get("status") != "ok":
            continue
        key = _result_key(r)
        out[key] = r
    return out


def load_ratio_baseline(path: Path | str | None = None) -> dict | None:
    path = Path(path) if path is not None else DEFAULT_RATIO_BASELINE
    if not path.exists():
        return None
    return json.loads(path.read_text())


def write_ratio_json(
    path: Path | None = None,
    *,
    tir_path: Path | None = None,
    ref_path: Path | None = None,
) -> Path:
    """Regenerate ratio.json from pinned tir.json + ref.json."""
    tir_path = tir_path or DEFAULT_TIR_BASELINE
    ref_path = ref_path or DEFAULT_REF_BASELINE
    tir = json.loads(tir_path.read_text()) if tir_path.exists() else {}
    ref = json.loads(ref_path.read_text()) if ref_path.exists() else {}
    payload = build_ratio_payload(tir, ref)
    out = path or DEFAULT_RATIO_BASELINE
    out.write_text(json.dumps(payload, indent=2))
    return out


def build_report(
    baseline_path: Path | str,
    current: dict | Path | str,
    *,
    threshold_pct: float = DEFAULT_RATIO_THRESHOLD,
    ratio_baseline: dict | Path | str | None = None,
) -> tuple[str, int]:
    """Build the unified bench markdown (ours + ref + ratio vs saved ratio).

    Returns (markdown, n_regressions_below_threshold) for run.py's exit code.
    """
    if isinstance(baseline_path, dict):
        base_payload = baseline_path
        baseline_label = (
            f"{base_payload.get('timestamp')} ({base_payload.get('label') or '-'}) "
            "from tir.json + ref.json"
        )
    else:
        base_payload = json.loads(Path(baseline_path).read_text())
        baseline_label = str(baseline_path)
    if isinstance(current, str | Path):
        cur_payload = json.loads(Path(current).read_text())
        current_label = str(current)
    else:
        cur_payload = current
        current_label = "(in-memory)"

    base = index(base_payload)
    cur = index(cur_payload)

    if ratio_baseline is None:
        ratio_baseline = load_ratio_baseline()
    elif isinstance(ratio_baseline, str | Path):
        ratio_baseline = json.loads(Path(ratio_baseline).read_text())
    saved = ratio_index(ratio_baseline or {})

    # Status of every current-run result, including non-ok rows. index() keeps
    # only status=="ok", so a workload that failed/interfered this run is absent
    # from `cur` and would otherwise vanish from the report with no trace — the
    # only hint being a "comparable" count below the baseline size. Keep the full
    # record so we can explain *why* a baseline workload has no comparable
    # measurement this run instead of silently truncating coverage.
    cur_status: dict[tuple[str, str], dict] = {}
    for r in cur_payload.get("results") or []:
        cur_status[(r["kernel"], r.get("label") or r.get("config"))] = r

    rows: list[tuple[str, str, str, float, float, float, float, float, float, float]] = []
    skipped_no_ref: list[tuple[str, str]] = []
    # Baseline workloads attempted this run but yielding no comparable ratio
    # (failed, interfered, or ok-but-missing an impl). Workloads simply not in
    # this run's scope (e.g. a --filter subset) have no cur_status record and are
    # NOT listed, so filtered runs don't get spammed with the whole baseline.
    not_comparable: list[tuple[str, str, str]] = []
    for key, base_impls in base.items():
        ref = pick_ref(base_impls)
        ours_b = next((i for i in OUR_IMPLS if i in base_impls), None)
        if ref is None or ours_b is None:
            skipped_no_ref.append(key)
            continue
        if key not in cur:
            rec = cur_status.get(key)
            if rec is not None:  # attempted this run but not ok → surface it
                st = rec.get("status") or "?"
                err = (rec.get("error") or "").strip().splitlines()
                not_comparable.append((key[0], key[1], f"{st}: {err[0]}" if err else st))
            continue
        cur_impls = cur[key]
        if ours_b not in cur_impls or ref not in cur_impls:
            missing = ", ".join(i for i in (ours_b, ref) if i not in cur_impls)
            not_comparable.append((key[0], key[1], f"ok but missing impl(s): {missing}"))
            continue
        our_b_us, ref_b_us = base_impls[ours_b], base_impls[ref]
        our_c_us, ref_c_us = cur_impls[ours_b], cur_impls[ref]
        if min(our_b_us, ref_b_us, our_c_us, ref_c_us) <= 0:
            continue
        # ref/ours: higher = ours is faster than ref = better.
        base_ratio = ref_b_us / our_b_us
        cur_ratio = ref_c_us / our_c_us
        saved_row = saved.get(key)
        saved_ratio = saved_row.get("ratio") if saved_row else base_ratio
        if saved_ratio <= 0:
            saved_ratio = base_ratio
        delta_pct = (cur_ratio - saved_ratio) / saved_ratio * 100.0
        ref_drift_pct = (ref_c_us - ref_b_us) / ref_b_us * 100.0
        our_drift_pct = (our_c_us - our_b_us) / our_b_us * 100.0
        rows.append(
            (
                key[0],
                key[1],
                ref,
                our_c_us,
                ref_c_us,
                cur_ratio,
                saved_ratio,
                delta_pct,
                our_drift_pct,
                ref_drift_pct,
            )
        )

    # Positive ratio Δ first (improvements), negative last (regressions).
    rows.sort(key=lambda r: -r[7])

    out = io.StringIO()

    def w(line: str = "") -> None:
        out.write(line + "\n")

    n_regressions = sum(1 for r in rows if r[7] <= -threshold_pct)
    n_improvements = sum(1 for r in rows if r[7] >= threshold_pct)

    ratio_label = (
        f"{ratio_baseline.get('timestamp')} ({ratio_baseline.get('label') or '-'})"
        if ratio_baseline
        else "computed from tir.json + ref.json"
    )

    w("# tir-bench bench report")
    w()
    w(f"- Baseline (abs µs): `{baseline_label}`")
    w(f"- Saved ratios: `{ratio_label}` from ratio.json")
    w(f"- Current run: `{current_label}`")
    w(
        "- Columns: ref/ours ratio (higher = ours faster), ratio Δ vs saved ratio.json, "
        "ours/ref Δ vs pinned tir.json + ref.json abs µs. Sorted by ratio Δ."
    )
    w(
        f"- Summary: {len(rows)} comparable workloads; "
        f"{n_improvements} > +{threshold_pct:g}%, {n_regressions} < -{threshold_pct:g}%"
        + (
            f"; {len(not_comparable)} not comparable in current run (see below)"
            if not_comparable
            else ""
        )
        + ". ⚠ = reference abs µs drifted >20% vs pinned ref (less trustworthy)."
    )
    w()

    if rows:
        w(
            "| kernel | config | ref | ours (µs) | ref (µs) | ratio | saved | "
            "ratio Δ | ours Δ | ref Δ |"
        )
        w("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for k, c, ref, our_us, ref_us, cr, sr, d, our_d, ref_d in rows:
            flag = " ⚠" if abs(ref_d) > 20 else ""
            w(
                f"| {k} | {c} | {ref} | {our_us:.2f} | {ref_us:.2f} | {cr:.3f} | "
                f"{sr:.3f} | {d:+.1f}% | {our_d:+.1f}% | {ref_d:+.1f}%{flag} |"
            )
        w()

    if not_comparable:
        w(f"## Not comparable in current run ({len(not_comparable)})")
        w()
        w(
            "_In baseline with a ref/ours pair, but produced no comparable "
            "measurement this run (failed, interfered, or missing an impl), so "
            "excluded from the ratio table above. Not a perf signal — usually a "
            "contention/OOM artifact — but flagged so lost coverage is never silent._"
        )
        w()
        for k, c, reason in sorted(not_comparable):
            # OOM messages are huge single lines; keep the actionable head.
            reason = reason if len(reason) <= 160 else reason[:157] + "..."
            w(f"- `{k}/{c}` — {reason}")
        w()

    if skipped_no_ref:
        w(f"## Skipped — no comparable ref impl ({len(skipped_no_ref)})")
        w()
        for k, c in skipped_no_ref:
            w(f"- `{k}/{c}`")
        w()

    return out.getvalue(), n_regressions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "current",
        nargs="?",
        default=str(DEFAULT_LATEST_RUN),
        help=f"Current run JSON (default: {DEFAULT_LATEST_RUN})",
    )
    ap.add_argument(
        "--baseline",
        default=None,
        help="Combined baseline JSON; default joins tir.json + ref.json",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_RATIO_THRESHOLD,
        help=f"Ratio regression threshold in percent (default {DEFAULT_RATIO_THRESHOLD:g})",
    )
    ap.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write report path (default: .tir-bench/reports/<run>/bench.md)",
    )
    ap.add_argument(
        "--refresh-ratio-json",
        action="store_true",
        help="Regenerate ratio.json from tir.json + ref.json and exit",
    )
    args = ap.parse_args()

    if args.refresh_ratio_json:
        out = write_ratio_json()
        print(f"[ratio_diff] refreshed {out} ({len(json.loads(out.read_text())['results'])} rows)")
        return

    baseline = join_default_baseline() if args.baseline is None else args.baseline
    if isinstance(baseline, str | Path):
        baseline = json.loads(Path(baseline).read_text())
    report, _ = build_report(baseline, args.current, threshold_pct=args.threshold)
    print(report)

    if args.output is not None:
        out_path = args.output
    else:
        cur_path = Path(args.current).resolve()
        run_id = cur_path.stem
        reports_dir = cur_path.parent.parent / "reports" / run_id
        reports_dir.mkdir(parents=True, exist_ok=True)
        out_path = reports_dir / "bench.md"
    out_path.write_text(report)
    print(f"[ratio_diff] written: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

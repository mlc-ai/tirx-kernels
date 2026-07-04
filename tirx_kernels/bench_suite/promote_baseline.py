#!/usr/bin/env python3
"""Promote a bench-suite run JSON to a checked-in baseline and refresh baseline.md.

See README.md in this directory for the full baseline refresh workflow.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

_BASELINE_AGG_KEYS = ("rounds", "method", "max_retry", "min_ok_rounds", "ok_rounds")


def _result_key(row: dict) -> tuple[str, str]:
    return row["kernel"], row.get("label") or row["config"]


def slim_baseline_row(row: dict) -> dict:
    """Drop per-run metadata; keep only checked-in baseline fields."""
    agg = row.get("aggregated") or {}
    return {
        "kernel": row["kernel"],
        "config": row["config"],
        "label": row.get("label") or row["config"],
        "status": row.get("status", "ok"),
        "impls": dict(row.get("impls") or {}),
        "aggregated": {k: agg[k] for k in _BASELINE_AGG_KEYS if k in agg},
    }


def slim_baseline_doc(doc: dict) -> dict:
    out = {k: v for k, v in doc.items() if k != "results"}
    out["results"] = sorted(
        [slim_baseline_row(r) for r in doc.get("results") or []], key=_result_key
    )
    return out


def _write_baseline(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(slim_baseline_doc(doc), indent=2) + "\n")


def merge_baseline(run_json: Path, baseline_path: Path) -> int:
    """Patch ok rows from run_json into baseline_path by (kernel, config).

    If ``baseline_path`` does not exist yet, the slimmed run is written as a
    fresh baseline."""
    run = json.loads(run_json.read_text())
    if not baseline_path.exists():
        _write_baseline(baseline_path, run)
        print(f"[promote] merge: no existing baseline, wrote {baseline_path.relative_to(HERE)}")
        return 0
    baseline = json.loads(baseline_path.read_text())
    patch = {
        _result_key(r): slim_baseline_row(r)
        for r in run.get("results") or []
        if r.get("status") == "ok"
    }
    if not patch:
        print("[promote] merge: no ok rows in run JSON", file=sys.stderr)
        return 1

    merged: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for row in baseline.get("results") or []:
        key = _result_key(row)
        if key in patch:
            merged.append(patch[key])
            seen.add(key)
        else:
            merged.append(slim_baseline_row(row))
    for key, row in patch.items():
        if key not in seen:
            merged.append(row)

    baseline["results"] = merged
    if run.get("git"):
        baseline["git"] = run["git"]
    if run.get("kernel_tree"):
        baseline["kernel_tree"] = run["kernel_tree"]
    if run.get("baselines"):
        baseline["baselines"] = run["baselines"]
    _write_baseline(baseline_path, baseline)
    print(
        f"[promote] merged {len(patch)} ok row(s) from {run_json} "
        f"-> {baseline_path.relative_to(HERE)}"
    )
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "run_json",
        type=Path,
        nargs="?",
        help="run JSON to promote (e.g. .bench-suite/runs/18.json)",
    )
    ap.add_argument(
        "--merge",
        action="store_true",
        help="patch ok rows from run_json into the existing baseline.json instead of replacing it",
    )
    ap.add_argument(
        "--slim",
        action="store_true",
        help="strip run metadata from the checked-in baseline JSON (no run_json needed)",
    )
    args = ap.parse_args()

    baseline_path = HERE / "baseline.json"

    if args.slim:
        if not baseline_path.exists():
            ap.error(f"baseline not found: {baseline_path}")
        _write_baseline(baseline_path, json.loads(baseline_path.read_text()))
        print(f"[promote] slimmed {baseline_path.relative_to(HERE)}")
    else:
        if not args.run_json:
            ap.error("run_json required (or pass --slim to clean the existing baseline)")
        if not args.run_json.exists():
            ap.error(f"run JSON not found: {args.run_json}")
        if args.merge:
            rc = merge_baseline(args.run_json, baseline_path)
            if rc:
                sys.exit(1)
        else:
            run = json.loads(args.run_json.read_text())
            _write_baseline(baseline_path, run)
            print(f"[promote] {args.run_json} -> {baseline_path.relative_to(HERE)}")

    # Always regenerate the human-facing baseline.md so it never drifts from the
    # JSON baseline. This is the whole reason to promote through this helper.
    subprocess.run(
        [sys.executable, str(HERE / "baseline_view.py")], check=True, stdout=subprocess.DEVNULL
    )
    print(f"[promote] regenerated {(HERE / 'baseline.md').relative_to(HERE)}")


if __name__ == "__main__":
    main()

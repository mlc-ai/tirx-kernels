#!/usr/bin/env python3
"""Promote a tir-bench run JSON to a checked-in baseline and refresh baseline.md.

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
        [slim_baseline_row(r) for r in doc.get("results") or []],
        key=_result_key,
    )
    return out


def _write_baseline(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(slim_baseline_doc(doc), indent=2) + "\n")


def merge_ref_baseline(run_json: Path, ref_path: Path) -> int:
    """Patch ok rows from run_json into ref_path by (kernel, config)."""
    run = json.loads(run_json.read_text())
    ref = json.loads(ref_path.read_text())
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
    for row in ref.get("results") or []:
        key = _result_key(row)
        if key in patch:
            merged.append(patch[key])
            seen.add(key)
        else:
            merged.append(slim_baseline_row(row))
    for key, row in patch.items():
        if key not in seen:
            merged.append(row)

    ref["results"] = merged
    if run.get("git"):
        ref["git"] = run["git"]
    if run.get("kernel_tree"):
        ref["kernel_tree"] = run["kernel_tree"]
    if run.get("baselines"):
        ref["baselines"] = run["baselines"]
    _write_baseline(ref_path, ref)
    print(
        f"[promote] merged {len(patch)} ok row(s) from {run_json} -> {ref_path.relative_to(HERE)}"
    )
    return 0


def merge_tir_baseline(run_json: Path, tir_path: Path) -> int:
    """Patch ok rows from run_json into tir_path by (kernel, config)."""
    return merge_ref_baseline(run_json, tir_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "run_json",
        type=Path,
        nargs="?",
        help="run JSON to promote (e.g. .tir-bench/runs/18.json)",
    )
    ap.add_argument("--tir", action="store_true", help="refresh tir.json (our-kernel baseline)")
    ap.add_argument("--ref", action="store_true", help="refresh ref.json (reference baseline)")
    ap.add_argument(
        "--both", action="store_true", help="refresh both (use for a full --impls all run)"
    )
    ap.add_argument(
        "--merge",
        action="store_true",
        help="patch ok rows into existing baseline(s) instead of replacing",
    )
    ap.add_argument(
        "--slim",
        action="store_true",
        help="strip run metadata from checked-in baseline JSON (no run_json needed)",
    )
    args = ap.parse_args()

    if args.slim:
        if not (args.tir or args.ref or args.both):
            ap.error("pick at least one of --tir / --ref / --both")
        targets = []
        if args.tir or args.both:
            targets.append(HERE / "tir.json")
        if args.ref or args.both:
            targets.append(HERE / "ref.json")
        for path in targets:
            _write_baseline(path, json.loads(path.read_text()))
            print(f"[promote] slimmed {path.relative_to(HERE)}")
    elif not (args.tir or args.ref or args.both):
        ap.error("pick at least one of --tir / --ref / --both")
    elif not args.run_json:
        ap.error("run_json required unless --slim")
    elif not args.run_json.exists():
        ap.error(f"run JSON not found: {args.run_json}")
    elif args.merge:
        rc = 0
        if args.tir or args.both:
            rc |= merge_tir_baseline(args.run_json, HERE / "tir.json")
        if args.ref or args.both:
            rc |= merge_ref_baseline(args.run_json, HERE / "ref.json")
        if rc:
            sys.exit(1)
    else:
        targets = []
        if args.tir or args.both:
            targets.append(HERE / "tir.json")
        if args.ref or args.both:
            targets.append(HERE / "ref.json")

        run = json.loads(args.run_json.read_text())
        for dst in targets:
            _write_baseline(dst, run)
            print(f"[promote] {args.run_json} -> {dst.relative_to(HERE)}")

    if not (args.slim or args.tir or args.ref or args.both):
        return

    # Always regenerate the human-facing baseline.md and ratio.json so they never
    # drift from the JSON baselines. This is the whole reason to promote through
    # this helper.
    subprocess.run(
        [sys.executable, str(HERE / "baseline_view.py")], check=True, stdout=subprocess.DEVNULL
    )
    print(f"[promote] regenerated {(HERE / 'baseline.md').relative_to(HERE)}")
    subprocess.run(
        [sys.executable, str(HERE / "ratio_diff.py"), "--refresh-ratio-json"], check=True
    )


if __name__ == "__main__":
    main()

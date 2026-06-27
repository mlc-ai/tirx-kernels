#!/usr/bin/env python3
"""Re-aggregate pinned tir.json / ref.json from bench logs.

Each successful workload subprocess writes one log with N proton trees (one
per in-bench round). Parses those trees and writes the chosen aggregate
(mean by default) back into the baseline JSON.

Typical use after a full x5 sweep:

    python reaggregate_from_logs.py --tir --ref
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_LOG_DIR = HERE.parents[2] / ".bench_suite" / "logs"
OURS_IMPLS = frozenset({"tir", "tirx"})
FA4_SKIP_BASELINES = frozenset({"flashinfer"})

sys.path.insert(0, str(HERE))
from promote_baseline import _write_baseline  # noqa: E402
from run import aggregate_impl_times  # noqa: E402


def _keep_ref_impl(kernel: str, impl: str) -> bool:
    if impl in OURS_IMPLS:
        return False
    return not (kernel == "flash_attention4" and impl in FA4_SKIP_BASELINES)


# proton-viewer -m avg_time/us prints microseconds (triton/profiler/viewer.py).
PROTON_AVG_TIME_METRIC = "avg_time/us"


def _parse_log_rounds(txt: str, *, kernel: str) -> list[dict[str, float]]:
    """Return [{impl: us}, ...] — one dict per proton ROOT tree in the log."""
    rounds: list[dict[str, float]] = []
    current: dict[str, float] = {}
    impl: str | None = None
    for raw in txt.splitlines():
        line = raw.rstrip()
        if re.match(r"^\d+\.\d+\s+ROOT", line):
            if current:
                rounds.append(current)
            current = {}
            impl = None
            continue
        if line and line[0] in "├└":
            parts = line.split("─", 1)[-1].split()
            if len(parts) >= 2:
                impl = parts[1]
            else:
                impl = None
            continue
        if impl and ("├─" in line or "└─" in line):
            parts = line.split("─", 1)[-1].split()
            if len(parts) >= 2:
                try:
                    us = float(parts[0])
                    current[impl] = max(current.get(impl, 0), us)
                except ValueError:
                    pass
    if current:
        rounds.append(current)
    return rounds


def _collect_rounds(log_dir: Path, role: str) -> dict[tuple[str, str], dict[str, list[float]]]:
    """{(kernel, config): {impl: [us per in-bench round]}} from successful logs."""
    pat = re.compile(rf"^(?P<kernel>.+)__(?P<config>.+)__{role}_a(?P<attempt>\d+)\.log$")
    staging: dict[tuple[str, str], dict[int, list[dict[str, float]]]] = defaultdict(dict)
    for path in sorted(log_dir.glob(f"*__{role}_a*.log")):
        m = pat.match(path.name)
        if not m:
            continue
        key = (m.group("kernel"), m.group("config"))
        attempt = int(m.group("attempt"))
        trees = _parse_log_rounds(path.read_text(), kernel=m.group("kernel"))
        if trees:
            staging[key][attempt] = trees

    by: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for key, attempts in staging.items():
        trees = attempts[max(attempts)]
        for tree in trees:
            for impl, us in tree.items():
                if us > 0:
                    by[key][impl].append(us)
    return by


def _merge_sample_maps(
    *maps: dict[tuple[str, str], dict[str, list[float]]],
) -> dict[tuple[str, str], dict[str, list[float]]]:
    out: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for m in maps:
        for key, impls in m.items():
            for impl, vals in impls.items():
                out[key][impl].extend(vals)
    return out


def _aggregate_rows(
    samples: dict[tuple[str, str], dict[str, list[float]]], *, impl_filter, rounds: int, method: str
) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for key, impl_samples in samples.items():
        qualified: dict[str, float] = {}
        for impl, us_list in impl_samples.items():
            if not impl_filter(impl):
                continue
            if len(us_list) < rounds:
                continue
            qualified[impl] = aggregate_impl_times(us_list[:rounds], method)
        if qualified:
            out[key] = {"impls": qualified, "aggregated": {"rounds": rounds, "method": method}}
    return out


def _patch_baseline(baseline_path: Path, patch: dict[tuple[str, str], dict]) -> tuple[int, int]:
    data = json.loads(baseline_path.read_text())
    updated = 0
    missing = 0
    for row in data.get("results") or []:
        key = (row["kernel"], row.get("label") or row["config"])
        if key not in patch:
            missing += 1
            continue
        p = patch[key]
        row["status"] = "ok"
        row["impls"] = p["impls"]
        row["aggregated"] = p["aggregated"]
        row.pop("error", None)
        updated += 1
    _write_baseline(baseline_path, data)
    return updated, missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help=f"bench log directory (default {DEFAULT_LOG_DIR})",
    )
    ap.add_argument("--tir", action="store_true", help="refresh tir.json (ours logs)")
    ap.add_argument("--ref", action="store_true", help="refresh ref.json (baseline logs)")
    ap.add_argument(
        "--rounds", type=int, default=5, help="Expected in-bench rounds per log (default 5)"
    )
    ap.add_argument(
        "--aggregate",
        choices=("mean", "median", "trimmed_mean"),
        default="mean",
        help="How to combine the N in-bench rounds (default mean)",
    )
    args = ap.parse_args()

    if not (args.tir or args.ref):
        ap.error("pick at least one of --tir / --ref")
    if args.rounds < 1:
        ap.error("--rounds must be >= 1")
    if not args.log_dir.is_dir():
        ap.error(f"log dir not found: {args.log_dir}")

    if args.tir:
        samples = _merge_sample_maps(
            _collect_rounds(args.log_dir, "ours"), _collect_rounds(args.log_dir, "all")
        )
        patch = _aggregate_rows(
            samples,
            impl_filter=lambda n: n in OURS_IMPLS,
            rounds=args.rounds,
            method=args.aggregate,
        )
        n, miss = _patch_baseline(HERE / "tir.json", patch)
        print(f"[reaggregate] tir.json: updated {n} row(s), {miss} without ours logs (unchanged)")

    if args.ref:
        samples = _merge_sample_maps(
            _collect_rounds(args.log_dir, "baseline"), _collect_rounds(args.log_dir, "all")
        )
        patch = _aggregate_rows(
            samples,
            impl_filter=lambda n: n not in OURS_IMPLS,
            rounds=args.rounds,
            method=args.aggregate,
        )
        ref_path = HERE / "ref.json"
        data = json.loads(ref_path.read_text())
        updated = 0
        for row in data.get("results") or []:
            key = (row["kernel"], row.get("label") or row["config"])
            if key not in patch:
                continue
            p = patch[key]
            row.setdefault("impls", {})
            for impl, us in p["impls"].items():
                if not _keep_ref_impl(row["kernel"], impl):
                    continue
                row["impls"][impl] = us
            for impl in OURS_IMPLS:
                row["impls"].pop(impl, None)
            if row["kernel"] == "flash_attention4":
                row["impls"].pop("flashinfer", None)
            row["aggregated"] = p["aggregated"]
            row["status"] = "ok"
            row.pop("error", None)
            updated += 1
        _write_baseline(ref_path, data)
        print(f"[reaggregate] ref.json: patched {updated} row(s) from baseline logs")

    subprocess.run(
        [sys.executable, str(HERE / "baseline_view.py")], check=True, stdout=subprocess.DEVNULL
    )
    print(f"[reaggregate] regenerated {(HERE / 'baseline.md').relative_to(HERE)}")
    subprocess.run(
        [sys.executable, str(HERE / "ratio_diff.py"), "--refresh-ratio-json"], check=True
    )


if __name__ == "__main__":
    main()

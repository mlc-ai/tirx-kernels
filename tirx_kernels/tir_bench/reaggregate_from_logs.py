#!/usr/bin/env python3
"""Re-aggregate pinned tir.json / ref.json from per-round bench logs.

Parses proton ROOT / impl lines in .tir-bench/logs (milliseconds) and writes
trimmed_mean (drop fastest + slowest ok round) back into the baseline JSON.

Typical use after a full x5 sweep without re-benchmarking:

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
DEFAULT_LOG_DIR = HERE.parents[2] / ".tir-bench" / "logs"
OURS_IMPLS = frozenset({"tir", "tirx"})

# Import shared aggregator from run.py (same directory).
sys.path.insert(0, str(HERE))
from run import aggregate_impl_times  # noqa: E402


def _parse_log_ms(path: Path) -> dict[str, float]:
    """Return {impl: ms} from one bench log (proton tree)."""
    txt = path.read_text()
    if not txt.strip():
        return {}
    out: dict[str, float] = {}
    for line in txt.splitlines():
        m = re.match(r"^[├└]─\s+(\d+\.\d+)\s+(\S+)", line)
        if m:
            out[m.group(2)] = float(m.group(1))
    if out:
        return out
    m = re.search(r"^(\d+\.\d+)\s+ROOT", txt, re.M)
    if not m:
        return {}
    ms = float(m.group(1))
    tail = txt.split("ROOT", 1)[-1][:300]
    impl = "tirx" if "tirx" in tail else "tir"
    return {impl: ms}


def _collect_rounds(log_dir: Path, role: str) -> dict[tuple[str, str], dict[str, list[float]]]:
    """{(kernel, config): {impl: [ms per round]}} from *_{role}_r*_a*.log."""
    # round_idx -> attempt -> {impl: ms}  (keep best attempt per round)
    staging: dict[tuple[str, str], dict[int, dict[int, dict[str, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    pat = re.compile(
        rf"^(?P<kernel>.+)__(?P<config>.+)__{role}_r(?P<round>\d+)_a(?P<attempt>\d+)\.log$"
    )
    for path in sorted(log_dir.glob(f"*__{role}_r*_a*.log")):
        m = pat.match(path.name)
        if not m:
            continue
        key = (m.group("kernel"), m.group("config"))
        round_idx = int(m.group("round"))
        attempt = int(m.group("attempt"))
        impl_ms = _parse_log_ms(path)
        if impl_ms:
            staging[key][round_idx][attempt] = impl_ms

    by: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for key, rounds in staging.items():
        for round_idx in sorted(rounds):
            attempts = rounds[round_idx]
            impl_ms = attempts[max(attempts)]
            for impl, ms in impl_ms.items():
                if ms > 0:
                    by[key][impl].append(ms)
    return by


def _aggregate_rows(
    samples: dict[tuple[str, str], dict[str, list[float]]],
    *,
    impl_filter,
    min_ok_rounds: int,
    rounds: int,
    max_retry: int,
) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for key, impl_samples in samples.items():
        qualified: dict[str, float] = {}
        ok_counts: dict[str, int] = {}
        for impl, ms_list in impl_samples.items():
            if not impl_filter(impl):
                continue
            ok_counts[impl] = len(ms_list)
            if ok_counts[impl] >= min_ok_rounds:
                qualified[impl] = aggregate_impl_times(ms_list, "trimmed_mean")
        if qualified:
            out[key] = {
                "impls": {k: v / 1000.0 for k, v in qualified.items()},  # seconds
                "aggregated": {
                    "rounds": rounds,
                    "method": "trimmed_mean",
                    "max_retry": max_retry,
                    "min_ok_rounds": min_ok_rounds,
                    "ok_rounds": ok_counts,
                },
            }
    return out


def _patch_baseline(
    baseline_path: Path,
    patch: dict[tuple[str, str], dict],
) -> tuple[int, int]:
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
    baseline_path.write_text(json.dumps(data, indent=2) + "\n")
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
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--min-ok-rounds", type=int, default=5)
    ap.add_argument("--max-retry", type=int, default=5)
    args = ap.parse_args()

    if not (args.tir or args.ref):
        ap.error("pick at least one of --tir / --ref")
    if not args.log_dir.is_dir():
        ap.error(f"log dir not found: {args.log_dir}")

    if args.tir:
        ours = _collect_rounds(args.log_dir, "ours")
        patch = _aggregate_rows(
            ours,
            impl_filter=lambda n: n in OURS_IMPLS,
            min_ok_rounds=args.min_ok_rounds,
            rounds=args.rounds,
            max_retry=args.max_retry,
        )
        n, miss = _patch_baseline(HERE / "tir.json", patch)
        print(f"[reaggregate] tir.json: updated {n} row(s), {miss} without ours logs (unchanged)")

    if args.ref:
        refs = _collect_rounds(args.log_dir, "baseline")
        patch = _aggregate_rows(
            refs,
            impl_filter=lambda n: n not in OURS_IMPLS,
            min_ok_rounds=args.min_ok_rounds,
            rounds=args.rounds,
            max_retry=args.max_retry,
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
            # Keep impls absent from logs (e.g. old nvfp4 without flashinfer logs).
            for impl, sec in p["impls"].items():
                row["impls"][impl] = sec
            agg = row.setdefault("aggregated", {})
            agg.update(p["aggregated"])
            ok_rounds = agg.setdefault("ok_rounds", {})
            ok_rounds.update(p["aggregated"]["ok_rounds"])
            row["status"] = "ok"
            row.pop("error", None)
            updated += 1
        ref_path.write_text(json.dumps(data, indent=2) + "\n")
        print(f"[reaggregate] ref.json: patched {updated} row(s) from baseline logs")

    subprocess.run(
        [sys.executable, str(HERE / "baseline_view.py")],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    print(f"[reaggregate] regenerated {(HERE / 'baseline.md').relative_to(HERE)}")
    subprocess.run(
        [sys.executable, str(HERE / "ratio_diff.py"), "--refresh-ratio-json"],
        check=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render a single bench_suite run JSON as a human-readable markdown summary.

For each (kernel, config) workload, lists every impl's timing and the
ref/ours ratio (higher = ours beats ref). Sorted by kernel then config.

Usage:
    python baseline_view.py [run.json] [-o PATH]

Default input: tir.json + ref.json in this directory
Default output: baseline.md (next to the baselines)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

OUR_IMPLS = {"tir", "tirx"}


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Combined baseline JSON; default joins tir.json + ref.json",
    )
    ap.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write markdown path (default: baseline.md next to the baselines)",
    )
    args = ap.parse_args()

    if args.input is not None:
        in_path = Path(args.input)
        payload = json.loads(in_path.read_text())
        src_name = in_path.name
        default_out = in_path.with_suffix(".md")
    else:
        tir = json.loads((here / "tir.json").read_text()) if (here / "tir.json").exists() else {}
        ref = json.loads((here / "ref.json").read_text()) if (here / "ref.json").exists() else {}
        ref_idx = {
            (r["kernel"], r.get("label") or r.get("config")): (r.get("impls") or {})
            for r in ref.get("results", [])
        }
        payload = {k: v for k, v in tir.items() if k != "results"}
        payload["results"] = [
            {
                **r,
                "impls": {
                    **ref_idx.get((r["kernel"], r.get("label") or r.get("config")), {}),
                    **(r.get("impls") or {}),
                },
            }
            for r in tir.get("results", [])
        ]
        src_name = "tir.json + ref.json"
        default_out = here / "baseline.md"
    results = payload.get("results") or []

    rows = []
    for r in results:
        if r.get("status") != "ok":
            continue
        impls = r.get("impls") or {}
        ours = next((i for i in OUR_IMPLS if i in impls), None)
        refs = {i: us for i, us in impls.items() if i not in OUR_IMPLS and us > 0}
        ref = min(refs, key=lambda k: refs[k]) if refs else None
        ratio = refs[ref] / impls[ours] if (ours and ref and impls[ours] > 0) else None
        rows.append(
            {
                "kernel": r["kernel"],
                "config": r.get("label") or r.get("config"),
                "impls": impls,
                "ours": ours,
                "ref": ref,
                "ratio": ratio,
            }
        )

    failed = [r for r in results if r.get("status") != "ok"]

    lines: list[str] = []
    lines.append(f"# bench_suite baseline view: `{src_name}`")
    lines.append("")
    lines.append(f"- Timestamp: `{payload.get('timestamp')}`")
    lines.append(f"- Label:     `{payload.get('label')}`")
    lines.append(f"- Git:       `{payload.get('git')}`")
    lines.append(f"- Workloads: {len(rows)} ok, {len(failed)} failed")
    lines.append("")
    lines.append(
        "Each row shows our impl's time (tir/tirx) and every reference "
        "impl, with ref/ours where ref = fastest non-ours impl. "
        "Higher ratio = ours is faster."
    )
    lines.append("")

    # Group by kernel; within a kernel, sort by config name.
    rows.sort(key=lambda r: (r["kernel"], r["config"]))
    cur_kernel = None
    for r in rows:
        if r["kernel"] != cur_kernel:
            cur_kernel = r["kernel"]
            lines.append(f"## {cur_kernel}")
            lines.append("")
            lines.append(
                "| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |"
            )
            lines.append("|---|---|---:|---|---:|---:|---|")
        ours_us = r["impls"].get(r["ours"], float("nan")) if r["ours"] else float("nan")
        ref_us = r["impls"].get(r["ref"], float("nan")) if r["ref"] else float("nan")
        ratio_s = f"{r['ratio']:.3f}" if r["ratio"] is not None else "—"
        others = sorted(
            (i, us) for i, us in r["impls"].items() if i not in OUR_IMPLS and i != r["ref"]
        )
        others_s = ", ".join(f"{i}={us:.4f}" for i, us in others) or "—"
        lines.append(
            f"| `{r['config']}` | {r['ours'] or '—'} | "
            f"{ours_us:.4f} | {r['ref'] or '—'} | {ref_us:.4f} | "
            f"{ratio_s} | {others_s} |"
        )
        # final newline added once per kernel group below
    # Append separator after the loop (the next kernel header takes care of spacing).
    lines.append("")

    if failed:
        lines.append(f"## Failed ({len(failed)})")
        lines.append("")
        for r in failed:
            first = (r.get("error") or "?").splitlines()[0]
            lines.append(f"- `{r['kernel']}/{r.get('label') or r.get('config')}`: {first}")
        lines.append("")

    md = "\n".join(lines)
    out_path = args.output if args.output else default_out
    out_path.write_text(md)
    print(f"[baseline_view] written: {out_path}", file=sys.stderr)
    print(md[:1200])  # head preview to stdout


if __name__ == "__main__":
    main()

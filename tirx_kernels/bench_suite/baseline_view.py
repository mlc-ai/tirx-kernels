#!/usr/bin/env python3
"""Render a single bench-suite run JSON as a human-readable markdown summary.

Grouped workloads use one row per config and one timing column per implementation.
Single-TIR workloads show the TIR timing and ref/ours ratio.

Usage:
    python baseline_view.py [run.json] [-o PATH]

Default input: baseline.json in this directory
Default output: baseline.md (next to the baseline)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    from tirx_kernels.bench_suite.impls import is_our_impl, our_impls
except ModuleNotFoundError:  # Support `python tirx_kernels/bench_suite/baseline_view.py`.
    from impls import is_our_impl, our_impls


def _natural_sort_key(value: str) -> tuple[str | int, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value))


def render_markdown(payload: dict, src_name: str) -> str:
    results = payload.get("results") or []
    ok_results = [r for r in results if r.get("status") == "ok"]
    failed = [r for r in results if r.get("status") != "ok"]

    lines: list[str] = []
    lines.append(f"# bench-suite baseline view: `{src_name}`")
    lines.append("")
    lines.append(f"- Timestamp: `{payload.get('timestamp')}`")
    lines.append(f"- Label:     `{payload.get('label')}`")
    lines.append(f"- Git:       `{payload.get('git')}`")
    lines.append(f"- Workloads: {len(ok_results)} ok, {len(failed)} failed")
    lines.append("")
    lines.append(
        "Grouped workloads show one row per config and one timing column per implementation. "
        "Single-TIR workloads show ref/ours against the fastest reference implementation."
    )
    lines.append("")

    by_kernel: dict[str, list[dict]] = {}
    for result in sorted(
        ok_results, key=lambda r: (r["kernel"], r.get("label") or r.get("config"))
    ):
        by_kernel.setdefault(result["kernel"], []).append(result)

    for kernel, kernel_results in by_kernel.items():
        lines.append(f"## {kernel}")
        lines.append("")
        grouped = any(len(our_impls(result.get("impls") or {})) > 1 for result in kernel_results)
        if grouped:
            kernel_results.sort(
                key=lambda result: _natural_sort_key(result.get("label") or result.get("config"))
            )
            impl_names: list[str] = []
            for result in kernel_results:
                for impl_name in result.get("impls") or {}:
                    if impl_name not in impl_names:
                        impl_names.append(impl_name)
            lines.append(
                "| config | " + " | ".join(f"{impl_name} (µs)" for impl_name in impl_names) + " |"
            )
            lines.append("|---|" + "---:|" * len(impl_names))
            for result in kernel_results:
                impls = result.get("impls") or {}
                timings = [
                    f"{impls[impl_name]:.4f}" if impl_name in impls else "—"
                    for impl_name in impl_names
                ]
                config = result.get("label") or result.get("config")
                lines.append(f"| `{config}` | " + " | ".join(timings) + " |")
        else:
            lines.append(
                "| config | ours impl | ours (µs) | ref impl | ref (µs) | ref/ours | other impls |"
            )
            lines.append("|---|---|---:|---|---:|---:|---|")
            for result in kernel_results:
                impls = result.get("impls") or {}
                refs = {i: us for i, us in impls.items() if not is_our_impl(i) and us > 0}
                ref = min(refs, key=lambda name: refs[name]) if refs else None
                for ours in our_impls(impls) or [None]:
                    ours_us = impls.get(ours, float("nan")) if ours else float("nan")
                    ref_us = impls.get(ref, float("nan")) if ref else float("nan")
                    ratio = ref_us / ours_us if ours and ref and ours_us > 0 else None
                    ratio_s = f"{ratio:.3f}" if ratio is not None else "—"
                    others = sorted(
                        (i, us) for i, us in impls.items() if not is_our_impl(i) and i != ref
                    )
                    others_s = ", ".join(f"{i}={us:.4f}" for i, us in others) or "—"
                    config = result.get("label") or result.get("config")
                    lines.append(
                        f"| `{config}` | {ours or '—'} | {ours_us:.4f} | {ref or '—'} | "
                        f"{ref_us:.4f} | {ratio_s} | {others_s} |"
                    )
        lines.append("")

    if failed:
        lines.append(f"## Failed ({len(failed)})")
        lines.append("")
        for result in failed:
            first = (result.get("error") or "?").splitlines()[0]
            config = result.get("label") or result.get("config")
            lines.append(f"- `{result['kernel']}/{config}`: {first}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Baseline JSON; default is baseline.json in this directory",
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
        baseline_path = here / "baseline.json"
        payload = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
        src_name = "baseline.json"
        default_out = here / "baseline.md"
    md = render_markdown(payload, src_name)
    out_path = args.output if args.output else default_out
    out_path.write_text(md)
    print(f"[baseline_view] written: {out_path}", file=sys.stderr)
    print(md[:1200])  # head preview to stdout


if __name__ == "__main__":
    main()

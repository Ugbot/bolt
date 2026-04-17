#!/usr/bin/env python3
"""perf_check.py — Bolt perf regression gate.

Runs the four Bolt bench binaries (bench_bolt, bench_kernels, bench_tpch_lite,
bench_1brc), parses `ns/op` and `ns/row` metrics out of their stdout, and
compares against baselines in `ci/perf_baselines.json`.

Modes:
    --mode check   (default) — fail (exit 1) if any metric regresses >THRESHOLD.
                                Exits 0 if baselines file is empty (first run).
    --mode record           — write current numbers to the baselines file.
                                Intended for manual runs on trusted hardware.

The CI workflow invokes this with `--mode check` on the Linux runner only.

Invariants (Tiger-Style-inspired):
- Never allocate a baseline on the fly — baselines must be committed.
- Ignore metrics that aren't baselined (new metrics land during perf work).
- Only compare within the same bench and metric name.
- 5% regression tolerance is the single adjustable knob.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# --- Config -----------------------------------------------------------------

REGRESSION_THRESHOLD_PCT = 5.0  # headline metric regression budget

# (bench binary name, extra CLI args)
BENCHES: List[Tuple[str, List[str]]] = [
    ("bench_bolt",      []),
    ("bench_kernels",   []),
    ("bench_tpch_lite", ["--rows", "1000000"]),
    ("bench_1brc",      ["--rows", "10000000", "--stations", "413"]),
]

# Matches lines like:
#   "  filter_gt<int32_t>            0.17 ns/op  (best ...)"
#   "  bmm_* SIMD kernel (compressstore)  0.28 ns/op  ..."
#   "  Q1 groupby              33.40 ns/row  ..."
#   "  per-row: 12.34 ns/row    throughput: ..."
#   "  malloc+free:     158.0 ns/alloc"
_METRIC_LINE = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9][A-Za-z0-9_<>():+*,\-\/ ]*?)"
    r"\s+(?P<value>[0-9]+\.[0-9]+)\s+ns/(?P<unit>op|row|alloc)\b"
)

# --- Data types -------------------------------------------------------------


@dataclass(frozen=True)
class Metric:
    bench: str
    name: str     # unique within bench
    unit: str     # "op" or "row"
    value: float  # ns per unit


# --- Bench discovery + invocation ------------------------------------------


def find_bench(build_dir: Path, name: str) -> Optional[Path]:
    """Locate a bench binary under build_dir. Handles .exe on Windows,
    CMake multi-config layouts (Release/), and flat Ninja layouts."""
    candidates = [
        build_dir / "benchmarks" / name,
        build_dir / "benchmarks" / f"{name}.exe",
        build_dir / "benchmarks" / "Release" / name,
        build_dir / "benchmarks" / "Release" / f"{name}.exe",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def run_bench(binary: Path, extra_args: List[str]) -> str:
    assert binary.is_file(), f"bench binary missing: {binary}"
    # Force utf-8 decode with replacement so box-drawing chars in
    # the bench banner don't blow up cp1252-defaulted Windows hosts.
    result = subprocess.run(
        [str(binary), *extra_args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=600,
    )
    if result.returncode != 0:
        sys.stderr.write(
            f"[perf_check] {binary.name} exited {result.returncode}\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}\n"
        )
        raise SystemExit(2)
    return result.stdout


# --- Parse ------------------------------------------------------------------


def parse_metrics(bench: str, stdout: str) -> List[Metric]:
    out: List[Metric] = []
    seen: set = set()
    for raw in stdout.splitlines():
        m = _METRIC_LINE.match(raw)
        if not m:
            continue
        name = m.group("name").strip().rstrip(":").strip()
        unit = m.group("unit")
        value = float(m.group("value"))
        key = (bench, name, unit)
        if key in seen:
            # Prefer first occurrence (best-of-N bench reports min once).
            continue
        seen.add(key)
        out.append(Metric(bench=bench, name=name, unit=unit, value=value))
    return out


# --- Baseline I/O -----------------------------------------------------------


def load_baselines(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Shape: { bench_name: { "unit:name": ns_value } }
    return {str(k): dict(v) for k, v in data.items()}


def save_baselines(path: Path, metrics: Iterable[Metric]) -> None:
    tree: Dict[str, Dict[str, float]] = {}
    for m in metrics:
        key = f"{m.unit}:{m.name}"
        tree.setdefault(m.bench, {})[key] = m.value
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(tree, f, indent=2, sort_keys=True)
        f.write("\n")


# --- Check ------------------------------------------------------------------


def check_regressions(
    metrics: Iterable[Metric],
    baselines: Dict[str, Dict[str, float]],
    threshold_pct: float,
) -> Tuple[int, int]:
    """Returns (num_compared, num_regressed). Prints a human summary."""
    compared = 0
    regressed = 0
    for m in metrics:
        tree = baselines.get(m.bench, {})
        key = f"{m.unit}:{m.name}"
        baseline = tree.get(key)
        if baseline is None:
            print(f"  [new]   {m.bench}::{m.name}: {m.value:.2f} ns/{m.unit}")
            continue
        assert baseline > 0.0
        delta_pct = (m.value - baseline) / baseline * 100.0
        compared += 1
        tag = "ok   "
        if delta_pct > threshold_pct:
            tag = "REGR "
            regressed += 1
        elif delta_pct < -threshold_pct:
            tag = "WIN  "
        print(
            f"  [{tag}] {m.bench}::{m.name}: "
            f"{m.value:.2f} ns/{m.unit} "
            f"(baseline {baseline:.2f}, {delta_pct:+.1f}%)"
        )
    return compared, regressed


# --- Entry point ------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Bolt perf regression gate")
    parser.add_argument(
        "--mode",
        choices=("check", "record"),
        default="check",
        help="check = fail on regression; record = write current to baselines",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "build" / "release",
        help="CMake build directory containing benchmarks/ subdir",
    )
    parser.add_argument(
        "--baselines",
        type=Path,
        default=Path(__file__).resolve().parent / "perf_baselines.json",
        help="Path to baselines JSON",
    )
    parser.add_argument(
        "--threshold-pct",
        type=float,
        default=REGRESSION_THRESHOLD_PCT,
        help="Regression budget in percent (default 5.0)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Best-of-N samples per metric (default 5). Kills single-run variance.",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        help="Bench name to skip (can repeat)",
    )
    args = parser.parse_args()

    print(f"[perf_check] build_dir = {args.build_dir}")
    print(f"[perf_check] mode      = {args.mode}")
    print(f"[perf_check] threshold = {args.threshold_pct:.1f}%")
    print(f"[perf_check] samples   = {args.samples} (best-of-N per metric)")

    # Merge best-of-N: keep the minimum value observed for each (bench, name, unit).
    best: Dict[Tuple[str, str, str], float] = {}
    missing: List[str] = []
    for name, extra in BENCHES:
        if name in args.skip:
            print(f"[perf_check] skipping {name}")
            continue
        binary = find_bench(args.build_dir, name)
        if binary is None:
            missing.append(name)
            continue
        print(f"[perf_check] running {binary.name} {' '.join(extra)} × {args.samples}")
        for run_i in range(args.samples):
            stdout = run_bench(binary, extra)
            metrics = parse_metrics(name, stdout)
            for m in metrics:
                k = (m.bench, m.name, m.unit)
                if k not in best or m.value < best[k]:
                    best[k] = m.value
        sample_count = sum(1 for k in best if k[0] == name)
        print(f"[perf_check]   best-of-{args.samples} across {sample_count} metrics")

    all_metrics: List[Metric] = [
        Metric(bench=k[0], name=k[1], unit=k[2], value=v)
        for k, v in best.items()
    ]

    if missing:
        sys.stderr.write(
            "[perf_check] missing bench binaries: " + ", ".join(missing) + "\n"
            "[perf_check] build the benches first (cmake --build).\n"
        )
        return 2

    if args.mode == "record":
        save_baselines(args.baselines, all_metrics)
        print(f"[perf_check] recorded {len(all_metrics)} metrics to {args.baselines}")
        return 0

    baselines = load_baselines(args.baselines)
    if not baselines:
        print(
            "[perf_check] baselines file is empty — first-run pass. "
            "Record baselines with `--mode record` and commit the file."
        )
        return 0

    print("[perf_check] comparing vs baselines:")
    compared, regressed = check_regressions(
        all_metrics, baselines, args.threshold_pct
    )
    print(
        f"[perf_check] {compared} metrics compared, "
        f"{regressed} regressed beyond {args.threshold_pct:.1f}%"
    )
    return 1 if regressed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
run_reference.py — one command: pick reference units, detect, interpret, stats.

    python3 scripts/run_reference.py --n 8 --seed 1 --model llama3.2:3b
    python3 scripts/run_reference.py --units 24 82 177     # explicit units
    python3 scripts/run_reference.py --all                 # the full test set

Runs the batch (non-streaming) path with a live progress bar over anomalies,
telemetry sampling in the background, and a final statistics report. Resumable:
re-running continues the interpretation where it stopped.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arad_edge import paths  # noqa: E402
from arad_edge.sampling import describe, sample_units  # noqa: E402


def run(cmd, **kw):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, cwd=ROOT, **kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--units", nargs="*", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--model", default="llama-3.2-3b-instruct")
    ap.add_argument("--backend", default="llamacpp",
                    choices=["llamacpp", "ollama", "dryrun"])
    ap.add_argument("--host", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-telemetry", action="store_true")
    a = ap.parse_args()
    paths.ensure_results()
    py = sys.executable

    det = paths.RESULTS / "test_anomalies.csv"
    units = None
    if not a.all:
        units = (sorted(set(a.units)) if a.units
                 else sample_units(a.n, a.seed))
        print(f"Reference units (seed={a.seed}): {units}\n")
        print(describe(units))

    # 1) detection ---------------------------------------------------------
    cmd = [py, "-m", "arad_edge", "detect", "apply", "--out", str(det)]
    if units:
        cmd += ["--units", *map(str, units)]
    run(cmd, check=True)

    # 2) telemetry (background) -------------------------------------------
    tel = None
    if not a.no_telemetry:
        tel = subprocess.Popen(
            [py, "-m", "arad_edge", "telemetry", "--interval", "2"], cwd=ROOT)

    # 3) interpretation (progress bar over anomalies) ---------------------
    icmd = [py, "-m", "arad_edge", "interpret", "--backend", a.backend,
            "--model", a.model, "--detections", str(det)]
    if a.host:
        icmd += ["--host", a.host]
    if a.limit:
        icmd += ["--limit", str(a.limit)]
    run(icmd, check=True)

    if tel:
        tel.terminate()

    # 4) statistics -------------------------------------------------------
    run([py, "-m", "arad_edge", "stats"], check=True)

    print("\nDone. Deliverables in results/:")
    print("  - test_FD002_with_interpretations.csv   (KB schema + interpretation)")
    print("  - edge_stats_report.md + figures/       (speed, tokens, energy)")


if __name__ == "__main__":
    main()

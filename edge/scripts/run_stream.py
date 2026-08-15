#!/usr/bin/env python3
"""
run_stream.py — live collection-simulation demo on a reference sample.

Starts the collector (test_FD002 replayed as a growing CSV), the online
detection daemon (tailing it, queueing anomalies), the SLM interpreter in
follow-mode (draining the queue), and telemetry — all concurrently — then
finalizes RUL and writes the statistics report (incl. queue depth & staleness).

    python3 scripts/run_stream.py --n 5 --seed 1 --tick 0.5 --model llama3.2:3b
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arad_edge import paths  # noqa: E402
from arad_edge.sampling import sample_units  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--units", nargs="*", type=int, default=None)
    ap.add_argument("--tick", type=float, default=0.5,
                    help="seconds per fleet cycle (offered-load knob)")
    ap.add_argument("--model", default="llama-3.2-3b-instruct")
    ap.add_argument("--host", default=None)
    ap.add_argument("--backend", default="llamacpp",
                    choices=["llamacpp", "ollama", "dryrun"])
    a = ap.parse_args()
    paths.ensure_results()
    py = sys.executable
    R = paths.RESULTS

    units = (sorted(set(a.units)) if a.units else sample_units(a.n, a.seed))
    print(f"Streaming reference units (seed={a.seed}): {units}\n")

    # clean prior stream artifacts
    for f in ("incoming.csv", "collector_log.jsonl", "detections_stream.csv",
              "queue_anomalies.jsonl", "stream_events.jsonl",
              "telemetry.jsonl"):
        (R / f).unlink(missing_ok=True)

    tel = subprocess.Popen([py, "-m", "arad_edge", "telemetry",
                            "--interval", "1"], cwd=ROOT)
    coll = subprocess.Popen([py, "-m", "arad_edge", "collect",
                             "--tick", str(a.tick),
                             "--units", *map(str, units)], cwd=ROOT)
    time.sleep(1.0)
    daemon = subprocess.Popen([py, "-m", "arad_edge", "daemon",
                               "--follow", "--max-idle", "20"], cwd=ROOT)
    # interpreter in the foreground so its progress bar is visible
    icmd1 = [py, "-m", "arad_edge", "interpret", "--backend",
             a.backend, "--model", a.model, "--follow",
             "--max-idle", "60"]
    if a.host:
        icmd1 += ["--host", a.host]
    subprocess.run(icmd1 + [
                    "--detections", str(R / "detections_stream.csv"),
                    "--no-merge"], cwd=ROOT)
    coll.wait()
    daemon.wait()
    tel.terminate()

    subprocess.run([py, "-m", "arad_edge", "daemon", "--finalize"], cwd=ROOT)
    icmd2 = [py, "-m", "arad_edge", "interpret", "--backend",
             a.backend, "--model", a.model, "--follow",
             "--max-idle", "2"]
    if a.host:
        icmd2 += ["--host", a.host]
    subprocess.run(icmd2 + [
                    "--detections", str(R / "detections_stream.csv")],
                   cwd=ROOT)
    subprocess.run([py, "-m", "arad_edge", "stats"], cwd=ROOT)
    print("\nStreaming demo complete. See results/edge_stats_report.md "
          "and results/figures/.")


if __name__ == "__main__":
    main()

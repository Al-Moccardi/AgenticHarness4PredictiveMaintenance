#!/usr/bin/env python3
"""inference_time.py — per-arm inference time from a bench out dir.

  python -m apdm.inference_time --dir results\\arad3\\prog_final
Reads the wall_s logged for every episode.
"""
import argparse, json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
a = ap.parse_args()
eps = [json.loads(l) for l in
       open(Path(a.dir) / "forecast_episodes.jsonl") if l.strip()]
arms = sorted({e["arm"] for e in eps})
print(f"{'arm':<14s} {'n':>3s} {'mean s':>7s} {'median':>7s} "
      f"{'p95':>6s} {'total':>8s}")
for arm in arms:
    w = [float(e.get("wall_s") or 0) for e in eps if e["arm"] == arm]
    print(f"{arm:<14s} {len(w):>3d} {np.mean(w):>7.1f} "
          f"{np.median(w):>7.1f} {np.percentile(w, 95):>6.1f} "
          f"{sum(w)/60:>6.1f} m")

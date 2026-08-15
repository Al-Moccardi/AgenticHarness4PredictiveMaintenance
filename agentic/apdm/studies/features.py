"""Snapshot features. One definition, used by everything.

The SAME features feed (a) the ML baselines, (b) the kNN retrieval index,
and (c) the textual summaries the LLM sees in the P1/P2 arms. This is the
control that makes the comparison fair: no arm gets information another arm
cannot in principle access.

Per snapshot (unit u, cycle c), from the W=20 history:
  for each sensor: last value, window mean, window std, linear slope,
                   z-score of last value vs the healthy reference of the
                   CURRENT regime (train inliers only)
  plus: cycle, current regime, regime-change count in window,
        degradation-state flag (ResPdM k=7 refinement, cluster 6) and cycles
        since entry. No detector-derived counters: the Isolation-Forest-era
        columns of the CSV are ignored by design.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ..data import SENSORS, W, FD002, Snapshot

_PER_SENSOR = ["last", "mean", "std", "slope", "z"]


def feature_names() -> List[str]:
    names = [f"{s}_{k}" for s in SENSORS for k in _PER_SENSOR]
    names += ["cycle", "regime", "regime_changes",
              "state_entered", "cycles_in_state"]
    return names


def snapshot_features(ds: FD002, s: Snapshot) -> np.ndarray:
    h = ds.history(s.unit, s.cycle)
    row = h.iloc[-1]
    regime = int(row["h_clust"])
    ref = ds.regime_ref.get(regime, {})
    t = h["cycle"].to_numpy(dtype=float)
    t = t - t.mean()
    denom = float((t * t).sum()) or 1.0

    vals: List[float] = []
    for sen in SENSORS:
        x = h[sen].to_numpy(dtype=float)
        mu, sd = ref.get(sen, (float(x.mean()), float(x.std() or 1.0)))
        vals += [float(x[-1]), float(x.mean()), float(x.std(ddof=0)),
                 float((t * (x - x.mean())).sum() / denom),
                 float((x[-1] - mu) / (sd or 1.0))]

    entry = ds.state_entered(s.unit, s.cycle)
    vals += [float(s.cycle), float(regime),
             float((h["h_clust"].diff() != 0).sum() - 1 if len(h) > 1 else 0),
             1.0 if entry is not None else 0.0,
             float(s.cycle - entry) if entry is not None else 0.0]
    return np.asarray(vals, dtype=float)


def feature_matrix(ds: FD002, snaps: List[Snapshot]) -> np.ndarray:
    return np.vstack([snapshot_features(ds, s) for s in snaps])


def summary_text(ds: FD002, s: Snapshot, top_k: int = 6) -> str:
    """Compact engineered summary for the P1/P2 prompts. Deliberately built
    from the same quantities as the feature vector -- nothing extra."""
    h = ds.history(s.unit, s.cycle)
    row = h.iloc[-1]
    regime = int(row["h_clust"])
    ref = ds.regime_ref.get(regime, {})
    t = h["cycle"].to_numpy(dtype=float); t -= t.mean()
    denom = float((t * t).sum()) or 1.0

    lines = [f"Unit {s.unit}, cycle {s.cycle}, operating regime {regime}."]
    devs = []
    for sen in SENSORS:
        x = h[sen].to_numpy(dtype=float)
        mu, sd = ref.get(sen, (x.mean(), x.std() or 1.0))
        z = (x[-1] - mu) / (sd or 1.0)
        slope = (t * (x - x.mean())).sum() / denom
        devs.append((abs(z), sen, z, slope, x[-1]))
    devs.sort(reverse=True)
    lines.append("Largest deviations from the healthy reference of this "
                 "regime (z-score, 20-cycle slope):")
    for _, sen, z, sl, last in devs[:top_k]:
        lines.append(f"  {sen}: z={z:+.2f}, slope={sl:+.4f}/cycle, last={last:.2f}")
    entry = ds.state_entered(s.unit, s.cycle)
    lines.append(f"Degradation state (k=7 cluster 6): "
                 + (f"ENTERED at cycle {entry} ({s.cycle - entry} cycles ago)."
                    if entry is not None else "not entered."))
    return "\n".join(lines)

"""Reference-unit sampling.

For a full paper run you interpret every anomaly of all 259 test units (~1,062
events, several hours on the Orin). For pilots, demos, and the streaming
figure you usually want a handful of REPRESENTATIVE units instead. This module
picks them deterministically and explains the choice.

Selection is stratified by each unit's anomaly burden so the sample spans the
range (units with few, moderate, and many anomalies), and — because the
temporal-memory agent needs history — it prefers units that actually accrue
multiple anomalies. `--seed` makes it reproducible.

CLI:
    python -m arad_edge.sample --n 8
    python -m arad_edge.sample --units 7 15 16
    python -m arad_edge.sample --n 6 --detections results/test_anomalies.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from . import paths
from .detector import apply_bundle, load_bundle, load_official_test


def _anomaly_counts(detections: Optional[Path]) -> pd.DataFrame:
    """Per-unit anomaly counts. Uses an existing detections CSV if given,
    otherwise runs the frozen detector on the official test set."""
    if detections and Path(detections).exists():
        d = pd.read_csv(detections)
    else:
        b = load_bundle(str(paths.BUNDLE))
        te = load_official_test(str(paths.TEST_TXT), str(paths.RUL_TXT))
        d = apply_bundle(b, te)
    ucol = "Unit_ID" if "Unit_ID" in d.columns else "unit_ID"
    g = (d[d.anomaly_label == -1].groupby(ucol).size()
         .rename("n_anomalies").reset_index()
         .rename(columns={ucol: "unit"}))
    allu = pd.DataFrame({"unit": sorted(d[ucol].unique())})
    g = allu.merge(g, on="unit", how="left").fillna({"n_anomalies": 0})
    g["n_anomalies"] = g["n_anomalies"].astype(int)
    return g.sort_values("unit").reset_index(drop=True)


def sample_units(n: int, seed: int = 0,
                 detections: Optional[Path] = None,
                 min_anomalies: int = 1) -> List[int]:
    """Stratified sample of `n` units across the anomaly-count distribution,
    preferring units with history (>= min_anomalies)."""
    counts = _anomaly_counts(detections)
    pool = counts[counts.n_anomalies >= min_anomalies].reset_index(drop=True)
    if len(pool) <= n:
        return sorted(pool.unit.tolist())
    # rank by anomaly count, cut into n strata, take the median unit of each
    pool = pool.sort_values("n_anomalies").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    edges = np.linspace(0, len(pool), n + 1).astype(int)
    picked = []
    for i in range(n):
        lo, hi = edges[i], max(edges[i + 1], edges[i] + 1)
        stratum = pool.iloc[lo:hi]
        picked.append(int(stratum.iloc[rng.integers(len(stratum))].unit))
    return sorted(set(picked))


def describe(units: List[int], detections: Optional[Path] = None) -> str:
    counts = _anomaly_counts(detections).set_index("unit")
    lines = ["unit  anomalies",
             "----  ---------"]
    tot = 0
    for u in units:
        na = int(counts.loc[u, "n_anomalies"]) if u in counts.index else 0
        tot += na
        lines.append(f"{u:>4}  {na:>9}")
    lines.append("----  ---------")
    lines.append(f"{'sum':>4}  {tot:>9}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="pick reference test units")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--units", nargs="*", type=int, default=None,
                    help="bypass sampling; just describe these units")
    ap.add_argument("--min-anomalies", type=int, default=1)
    ap.add_argument("--detections", default=None,
                    help="reuse a detections CSV instead of recomputing")
    ap.add_argument("--out", default=None,
                    help="write the chosen unit ids (comma-sep) to this file")
    a = ap.parse_args()

    det = Path(a.detections) if a.detections else None
    units = (sorted(set(a.units)) if a.units
             else sample_units(a.n, a.seed, det, a.min_anomalies))
    print(f"selected {len(units)} reference units "
          f"(seed={a.seed}):\n  {units}\n")
    print(describe(units, det))
    if a.out:
        Path(a.out).write_text(",".join(map(str, units)))
        print(f"\nwrote -> {a.out}")


if __name__ == "__main__":
    main()

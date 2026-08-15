"""Event forecasting: time-to-next-anomaly ONSET as an interval + severity.

WHY ONSET, NOT NEXT-ANOMALY. Measured on this dataset, inter-anomaly gaps
have median 1 cycle (q90=5): anomalies are bursty, so "time to next anomaly"
from inside a burst is trivially ~1. The forecastable event is the ONSET:
for a unit currently QUIET (no anomaly in the last G cycles, G=5), when does
anomalous behaviour begin or resume?

GOLD (future-derived, leakage-safe by the same construction as the phenotype
gold -- inputs use rows <= c, gold uses rows > c):
  T(u,c)   = min{c' > c : anomaly at c'} - c, defined on QUIET snapshots;
             right-censored snapshots (no anomaly before EoL) are excluded
             and the censoring rate is reported.
  SEV(u,c) = severity of that realised event, from OUTCOME: RUL at the event
             mapped to bands 1..5 (>75 -> 1, 51-75 -> 2, 31-50 -> 3,
             16-30 -> 4, <=15 -> 5; bands informed by the RUL-at-event
             quantiles and fixed here, pre-registered).

WHY OUTCOMES AND NOT THE INTERPRETATIONS' GRAVITY SCORES. The gravity scores
are SLM-generated text (extractable from only 30/84 records, values include
0): using them as gold scores one LLM against another LLM's opinion. They
are instead an AUDIT TARGET: `gravity_audit()` correlates them with the
outcome severity, i.e. it tests whether the TIOT interpretation layer's own
severity opinions track reality.

FORECAST FORMAT AND METRICS. A forecast is an interval [lo, hi] (cycles
until onset) plus a severity band 1..5. Interval quality: coverage, mean
width, and the Winkler interval score at alpha=0.2 (width + (2/alpha) *
distance when the truth falls outside -- proper for central 80% intervals).
Severity: exact and +/-1 accuracy, ordinal MAE, quadratic weighted kappa.

TWINS (the non-language bar any agent must beat):
  T1 global    train-fleet onset-gap quantiles (q10, q90), one interval for
               everyone -- the "climatology" forecast;
  T2 learned   HistGradientBoosting quantile regressors (0.1 / 0.9) on the
               75 shared features;
  S1 learned   HistGradientBoosting classifier over the severity bands.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..data import FD002, Snapshot

QUIET_GAP = 5
SEV_BANDS = [(76, 10**9, 1), (51, 75, 2), (31, 50, 3), (16, 30, 4), (0, 15, 5)]
ALPHA = 0.2                     # central 80% interval


def severity_band(rul_at_event: int) -> int:
    for lo, hi, b in SEV_BANDS:
        if lo <= rul_at_event <= hi:
            return b
    return 5


# ------------------------------------------------------------------- gold
@dataclass
class EventGold:
    unit: int
    cycle: int
    rul: int
    t_onset: int            # cycles until the next anomaly (>0)
    event_cycle: int
    severity: int           # 1..5, outcome-derived


def _anomaly_cycles(ds: FD002, unit: int) -> np.ndarray:
    g = ds._by_unit[unit]
    return g.loc[g["anomaly_label"] == -1, "cycle"].to_numpy()


def is_quiet(ds: FD002, unit: int, cycle: int, gap: int = QUIET_GAP) -> bool:
    ac = _anomaly_cycles(ds, unit)
    return not np.any((ac > cycle - gap) & (ac <= cycle))


def event_gold(ds: FD002, unit: int, cycle: int) -> Optional[EventGold]:
    """None if not quiet or right-censored. Gold uses ONLY cycles > c."""
    if not is_quiet(ds, unit, cycle):
        return None
    ac = _anomaly_cycles(ds, unit)
    fut = ac[ac > cycle]
    if not len(fut):
        return None                                       # censored
    ev = int(fut.min())
    return EventGold(unit, cycle, ds.rul(unit, cycle), ev - cycle, ev,
                     severity_band(ds.eol[unit] - ev))


def build_event_dataset(ds: FD002, units: List[int],
                        max_per_unit: int = 0, seed: int = 0
                        ) -> Tuple[List[EventGold], Dict]:
    rng = np.random.default_rng(seed)
    out: List[EventGold] = []
    n_quiet = n_cens = 0
    for u in units:
        golds = []
        for c in range(20, ds.eol[u] + 1):          # W history requirement
            if not is_quiet(ds, u, c):
                continue
            n_quiet += 1
            g = event_gold(ds, u, c)
            if g is None:
                n_cens += 1
            else:
                golds.append(g)
        if max_per_unit and len(golds) > max_per_unit:
            idx = rng.choice(len(golds), size=max_per_unit, replace=False)
            golds = [golds[int(i)] for i in sorted(idx)]
        out.extend(golds)
    stats = {"n_events": len(out), "n_quiet": n_quiet,
             "censoring_rate": round(n_cens / max(n_quiet, 1), 4)}
    return out, stats


# ----------------------------------------------------------------- metrics
def winkler(lo: float, hi: float, t: float, alpha: float = ALPHA) -> float:
    w = max(hi - lo, 0.0)
    if t < lo:
        return w + (2.0 / alpha) * (lo - t)
    if t > hi:
        return w + (2.0 / alpha) * (t - hi)
    return w


def interval_metrics(lo, hi, t) -> Dict[str, float]:
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    t = np.asarray(t, float)
    cover = (t >= lo) & (t <= hi)
    return {"n": int(len(t)),
            "coverage": float(cover.mean()),
            "mean_width": float((hi - lo).mean()),
            "median_width": float(np.median(hi - lo)),
            "winkler": float(np.mean([winkler(a, b, x)
                                      for a, b, x in zip(lo, hi, t)]))}


def qwk(a, b, n_classes: int = 5) -> float:
    """Quadratic weighted kappa on 1..n_classes ordinal labels."""
    a = np.asarray(a, int) - 1
    b = np.asarray(b, int) - 1
    O = np.zeros((n_classes, n_classes))
    for x, y in zip(a, b):
        O[x, y] += 1
    Wm = np.array([[(i - j) ** 2 for j in range(n_classes)]
                   for i in range(n_classes)], float) / (n_classes - 1) ** 2
    ra = O.sum(1, keepdims=True)
    rb = O.sum(0, keepdims=True)
    E = ra @ rb / max(O.sum(), 1)
    den = float((Wm * E).sum())
    return 1.0 - float((Wm * O).sum()) / den if den else 0.0


def severity_metrics(pred, true) -> Dict[str, float]:
    p = np.asarray(pred, int)
    t = np.asarray(true, int)
    return {"n": int(len(t)),
            "sev_acc": float((p == t).mean()),
            "sev_pm1_acc": float((np.abs(p - t) <= 1).mean()),
            "sev_mae": float(np.abs(p - t).mean()),
            "sev_qwk": qwk(p, t)}


# ------------------------------------------------------------------- twins
class EventTwins:
    """Non-language baselines fitted on TRAIN quiet snapshots."""

    def __init__(self, ds: FD002, train_events: List[EventGold],
                 max_fit: int = 20000, seed: int = 0):
        from .features import feature_matrix
        from sklearn.ensemble import (HistGradientBoostingClassifier,
                                      HistGradientBoostingRegressor)
        rng = np.random.default_rng(seed)
        ev = train_events
        if len(ev) > max_fit:
            idx = rng.choice(len(ev), size=max_fit, replace=False)
            ev = [ev[int(i)] for i in idx]
        t = np.array([e.t_onset for e in ev], float)
        self.q_global = (float(np.quantile(t, ALPHA / 2)),
                         float(np.quantile(t, 1 - ALPHA / 2)))
        snaps = [Snapshot(e.unit, e.cycle, e.rul) for e in ev]
        X = feature_matrix(ds, snaps)
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd == 0] = 1
        Xs = (X - self.mu) / self.sd
        self.q_lo = HistGradientBoostingRegressor(
            loss="quantile", quantile=ALPHA / 2, max_iter=250,
            random_state=seed).fit(Xs, t)
        self.q_hi = HistGradientBoostingRegressor(
            loss="quantile", quantile=1 - ALPHA / 2, max_iter=250,
            random_state=seed).fit(Xs, t)
        self.sev = HistGradientBoostingClassifier(
            max_iter=250, random_state=seed).fit(
            Xs, np.array([e.severity for e in ev]))

    def predict(self, ds: FD002, events: List[EventGold]) -> pd.DataFrame:
        from .features import feature_matrix
        snaps = [Snapshot(e.unit, e.cycle, e.rul) for e in events]
        Xs = (feature_matrix(ds, snaps) - self.mu) / self.sd
        lo = np.maximum(1.0, self.q_lo.predict(Xs))
        hi = np.maximum(lo + 1.0, self.q_hi.predict(Xs))
        return pd.DataFrame({
            "unit": [e.unit for e in events],
            "cycle": [e.cycle for e in events],
            "t_onset": [e.t_onset for e in events],
            "severity": [e.severity for e in events],
            "t1_lo": self.q_global[0], "t1_hi": self.q_global[1],
            "t2_lo": lo, "t2_hi": hi,
            "s1_sev": self.sev.predict(Xs).astype(int)})


# ----------------------------------------------------------- gravity audit
def gravity_audit(ds: FD002, interp_dir: Path) -> Dict:
    """Do the TIOT interpretations' own gravity scores track real outcomes?
    Spearman rho between extracted gravity (30/84 records) and the
    outcome-derived severity band at the same (unit, cycle)."""
    rows = []
    for f in sorted(Path(interp_dir).glob("unit_*.json")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            t = str(r.get("interpretation", ""))
            m = (re.search(r"[Gg]ravity\s*[Ss]core\D{0,15}(\d)", t)
                 or re.search(r"[Ss]everity\D{0,15}(\d)", t))
            if not m:
                continue
            u = int(r["unit_ID"])
            cyc = int(r.get("cycle", r.get("cycles", 0)))
            if u not in ds.eol or cyc <= 0:
                continue
            rows.append({"unit": u, "cycle": cyc,
                         "slm_gravity": int(m.group(1)),
                         "outcome_severity":
                             severity_band(ds.eol[u] - cyc)})
    df = pd.DataFrame(rows)
    if len(df) < 5:
        return {"n": len(df), "note": "too few extractable scores"}
    from scipy.stats import spearmanr
    rho, p = spearmanr(df.slm_gravity, df.outcome_severity)
    return {"n": int(len(df)), "spearman_rho": round(float(rho), 3),
            "p_value": round(float(p), 4),
            "slm_gravity_dist": df.slm_gravity.value_counts().to_dict(),
            "outcome_dist": df.outcome_severity.value_counts().to_dict()}

"""Metrics and paired statistics.

S-score exactly as in Moccardi et al. eq. (4) / Saxena et al. 2008:
  diff = pred - true;  diff < 0 -> exp(-diff/13) - 1;  diff >= 0 -> exp(diff/10) - 1
Overestimating remaining life (dangerous) is penalised more sharply.

One elicitation, three scorings: the predicted RUL is scored as regression,
as an induced 3-class health stage (CRITICAL<=30 < MID<=100 < EARLY), and as
an induced fail-within-30 binary. ML and LLM arms are scored identically.
All cross-arm comparisons are PAIRED on identical snapshots: Wilcoxon
signed-rank on |error| for regression, McNemar for classification.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

BUCKETS = ((0, 30, "CRITICAL"), (31, 100, "MID"), (101, 10**9, "EARLY"))


def stage(rul: float) -> str:
    for lo, hi, name in BUCKETS:
        if lo <= rul <= hi:
            return name
    return "EARLY"


def s_score(pred: Sequence[float], true: Sequence[float]) -> float:
    tot = 0.0
    for p, t in zip(pred, true):
        d = p - t
        tot += math.exp(-d / 13.0) - 1.0 if d < 0 else math.exp(d / 10.0) - 1.0
    return tot


def regression_metrics(pred: Sequence[float], true: Sequence[float]) -> Dict[str, float]:
    p = np.asarray(pred, float); t = np.asarray(true, float)
    err = p - t
    ss_res = float((err ** 2).sum())
    ss_tot = float(((t - t.mean()) ** 2).sum()) or 1e-12
    out = {"n": int(len(t)),
           "mse": float((err ** 2).mean()),
           "mae": float(np.abs(err).mean()),
           "rmse": float(np.sqrt((err ** 2).mean())),
           "r2": 1.0 - ss_res / ss_tot,
           "bias": float(err.mean()),
           "s_score": s_score(p, t)}
    crit = t <= 30
    if crit.any():
        out["mae_critical"] = float(np.abs(err[crit]).mean())
        out["s_score_critical"] = s_score(p[crit], t[crit])
        out["overest_rate_critical"] = float((err[crit] > 0).mean())
    return out


def classification_metrics(pred_rul, true_rul) -> Dict[str, float]:
    ps = [stage(x) for x in pred_rul]
    ts = [stage(x) for x in true_rul]
    labels = [b[2] for b in BUCKETS]
    f1s = []
    for lab in labels:
        tp = sum(1 for a, b in zip(ps, ts) if a == lab and b == lab)
        fp = sum(1 for a, b in zip(ps, ts) if a == lab and b != lab)
        fn = sum(1 for a, b in zip(ps, ts) if a != lab and b == lab)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    acc = sum(1 for a, b in zip(ps, ts) if a == b) / len(ts)
    # fail-within-30 binary
    pb = [x <= 30 for x in pred_rul]; tb = [x <= 30 for x in true_rul]
    tp = sum(a and b for a, b in zip(pb, tb))
    fp = sum(a and not b for a, b in zip(pb, tb))
    fn = sum((not a) and b for a, b in zip(pb, tb))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"stage_acc": acc, "stage_macro_f1": float(np.mean(f1s)),
            "fail30_precision": prec, "fail30_recall": rec,
            "fail30_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


# ------------------------------------------------------------- paired tests
def wilcoxon_paired(a_abs_err: Sequence[float], b_abs_err: Sequence[float]) -> Dict:
    """Two-sided Wilcoxon signed-rank on |err|_A - |err|_B (scipy)."""
    from scipy.stats import wilcoxon
    a = np.asarray(a_abs_err, float); b = np.asarray(b_abs_err, float)
    d = a - b
    if np.allclose(d, 0):
        return {"stat": 0.0, "p_value": 1.0, "median_delta": 0.0}
    st = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return {"stat": float(st.statistic), "p_value": float(st.pvalue),
            "median_delta": float(np.median(d))}


def mcnemar(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> Dict:
    n01 = sum(1 for x, y in zip(a_correct, b_correct) if (not x) and y)
    n10 = sum(1 for x, y in zip(a_correct, b_correct) if x and (not y))
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "p_value": 1.0}
    k = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return {"n01": n01, "n10": n10, "p_value": min(1.0, 2 * tail)}


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"),) * 2
    p = k / n; dd = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / dd, (c + m) / dd)


def bootstrap_ci(vals: Sequence[float], n_boot: int = 4000, alpha: float = .05,
                 seed: int = 0) -> Tuple[float, float]:
    v = np.asarray([x for x in vals if x == x], float)
    if not v.size:
        return (float("nan"),) * 2
    rng = np.random.default_rng(seed)
    m = rng.choice(v, (n_boot, v.size), replace=True).mean(axis=1)
    return float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))

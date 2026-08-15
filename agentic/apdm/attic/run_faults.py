"""Answer the two questions that decide whether fault-aware PdM is real on
this dataset, BEFORE any LLM is involved.

Q1  Do the phenotypes carry prognostic information? Mann-Whitney U on
    residual-life-after-state-entry across phenotypes (train), and a
    prognosis-at-entry comparison on TEST units: pooled prior vs
    phenotype-conditioned prior, scored with MAE and the paper-1 S-score.
Q2  How early is the phenotype diagnosable, and how hard is diagnosis?
    Accuracy of the two non-LLM twins -- (a) nearest-centroid on the
    current-window z, (b) logistic regression on the 75 features -- as a
    function of true-RUL bucket. If twin (a) is already ~ceiling everywhere,
    diagnosis is trivial and the agent has nothing to add; if accuracy climbs
    with degradation, there is a lead-time frontier worth fighting over.

  python -m apdm.run_faults
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import FD002, RMAX
from .faults import (build_fault_layer, current_z, gold_phenotype,
                     terminal_z)
from .features import feature_matrix
from .metrics import s_score

ROOT = Path(__file__).resolve().parent.parent
BUCKETS = [(0, 30, "critical"), (31, 100, "mid"), (101, RMAX, "early")]


def main() -> None:
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    ds = FD002(seed=42)
    layer = build_fault_layer(ds, cache=ROOT / "cache" / "faults.pkl")

    print(f"[faults] k={layer.k} phenotypes (silhouette {layer.silhouette:.3f}, "
          f"tail={layer.tail})")
    for p in layer.phenotypes:
        print(f"  P{p.pid} '{p.name}': n={p.n_train_units}  "
              f"sig={p.signature}  comps={p.components}  "
              f"resid_after_entry med={p.residual_after_entry_median:.0f} "
              f"IQR[{p.residual_after_entry_iqr[0]:.0f},"
              f"{p.residual_after_entry_iqr[1]:.0f}]")
        print(f"      {p.interpretation}")

    # ---------------- Q1: prognostic value of the phenotype ---------------
    from scipy.stats import mannwhitneyu, kruskal
    groups = []
    for p in layer.phenotypes:
        r = [ds.eol[u] - ds.state_entered(u, ds.eol[u])
             for u in ds.train_units
             if layer.train_unit_phenotype[u] == p.pid
             and ds.state_entered(u, ds.eol[u]) is not None]
        groups.append(r)
    if layer.k == 2:
        st = mannwhitneyu(groups[0], groups[1], alternative="two-sided")
        test_name, pval = "Mann-Whitney U", float(st.pvalue)
    else:
        st = kruskal(*groups)
        test_name, pval = "Kruskal-Wallis", float(st.pvalue)
    print(f"\n[Q1] residual-after-entry differs across phenotypes? "
          f"{test_name} p={pval:.4f}  medians="
          f"{[float(np.median(g)) for g in groups]}")

    # prognosis-at-entry on TEST units: pooled vs conditioned prior
    rows = []
    for u in ds.test_units:
        e = ds.state_entered(u, ds.eol[u])
        if e is None:
            continue
        true_res = ds.eol[u] - e
        g = gold_phenotype(ds, layer, u)                       # oracle diag
        zc = current_z(ds, u, e)
        pred_diag = layer.assign_z(zc)[0]                      # entry-time diag
        rows.append({
            "unit": u, "entry": e, "true_residual": true_res,
            "gold_phenotype": g, "diag_at_entry": pred_diag,
            "pooled": layer.pooled_residual_median,
            "cond_oracle": layer.phenotypes[g].residual_after_entry_median,
            "cond_diag": layer.phenotypes[pred_diag]
                         .residual_after_entry_median})
    q1 = pd.DataFrame(rows)
    q1.to_csv(out / "faults_q1_prognosis_at_entry.csv", index=False)

    def score(col):
        pred = np.clip(q1[col], 0, RMAX)
        true = np.clip(q1["true_residual"], 0, RMAX)
        return (float(np.abs(pred - true).mean()),
                float(s_score(pred, true)))

    for col, label in [("pooled", "pooled prior"),
                       ("cond_diag", "phenotype prior (entry-time diagnosis)"),
                       ("cond_oracle", "phenotype prior (oracle diagnosis)")]:
        mae, s = score(col)
        print(f"[Q1] prognosis at entry, {label:<42} "
              f"MAE={mae:6.2f}  S={s:10.1f}  (n={len(q1)})")
    agree = float((q1.diag_at_entry == q1.gold_phenotype).mean())
    print(f"[Q1] entry-time nearest-centroid diagnosis accuracy: {agree:.3f}")

    # ---------------- Q2: diagnosability frontier -------------------------
    from sklearn.linear_model import LogisticRegression
    tr_snaps = ds.snapshots(ds.train_units)
    Xtr = feature_matrix(ds, tr_snaps)
    ytr = np.array([layer.train_unit_phenotype[s.unit] for s in tr_snaps])
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1
    lr = LogisticRegression(max_iter=2000).fit(
        (Xtr - mu) / sd, ytr)

    te_snaps = ds.snapshots(ds.test_units)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(te_snaps), size=min(1500, len(te_snaps)),
                     replace=False)
    sample = [te_snaps[int(i)] for i in idx]
    Xte = feature_matrix(ds, sample)
    gold = {u: gold_phenotype(ds, layer, u) for u in ds.test_units}

    recs = []
    for s_, x in zip(sample, Xte):
        zc = current_z(ds, s_.unit, s_.cycle)
        recs.append({
            "unit": s_.unit, "cycle": s_.cycle, "rul": s_.rul,
            "gold": gold[s_.unit],
            "twin_rule": layer.assign_z(zc)[0],
            "twin_lr": int(lr.predict(((x - mu) / sd)[None, :])[0])})
    q2 = pd.DataFrame(recs)
    q2.to_csv(out / "faults_q2_diagnosability.csv", index=False)

    print("\n[Q2] phenotype diagnosis accuracy by true-RUL bucket "
          f"(n={len(q2)} snapshots, majority class = "
          f"{q2.gold.value_counts(normalize=True).max():.3f})")
    print(f"{'bucket':<10}{'n':>6}{'rule (no-learn)':>18}{'logistic':>12}")
    for lo, hi, name in BUCKETS:
        m = q2[(q2.rul >= lo) & (q2.rul <= hi)]
        if not len(m):
            continue
        print(f"{name:<10}{len(m):>6}"
              f"{(m.twin_rule == m.gold).mean():>18.3f}"
              f"{(m.twin_lr == m.gold).mean():>12.3f}")
    print(f"{'overall':<10}{len(q2):>6}"
          f"{(q2.twin_rule == q2.gold).mean():>18.3f}"
          f"{(q2.twin_lr == q2.gold).mean():>12.3f}")

    (out / "faults_layer.json").write_text(layer.library_json())
    print(f"\n[faults] wrote faults_layer.json, faults_q1_*.csv, "
          f"faults_q2_*.csv in {out}")


if __name__ == "__main__":
    main()

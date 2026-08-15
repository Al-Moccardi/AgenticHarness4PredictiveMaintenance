#!/usr/bin/env python3
"""
eval_iforest.py — effectiveness of the regime-specific Isolation Forest detector
================================================================================

The detector is UNSUPERVISED: there is no ground-truth anomaly label to score it
against. The physically defensible reference is the degradation state itself, so
every metric here is anchored on RUL, which the detector never sees:

  A. Score-RUL association      Spearman rho, pooled and per unit (the detector
                                should track degradation monotonically).
  B. Weak-label discrimination  "degraded" = RUL <= TAU_POS vs "healthy" =
                                RUL >= TAU_NEG; ROC-AUC / PR-AUC of -score, with
                                bootstrap CIs and a sweep over TAU_POS.
  C. Baseline comparison        regime-specific IF  vs  global IF (no clustering)
                                vs  Hotelling T^2 (the multi-regime-blind control
                                chart). Paired bootstrap on the AUC differences —
                                this is the evidence for the clustering claim.
  D. Detection lead time        RUL at the first SUSTAINED alarm per unit
                                (m alarms inside a w-cycle window), i.e. how much
                                warning an operator actually gets.
  E. False-alarm behaviour      alarm rate in the healthy zone (RUL >= TAU_NEG),
                                pooled and per regime.
  F. Per-regime consistency     alarm rate and AUC within each operating regime;
                                a regime-blind detector shows large spread here.

Outputs (under --outdir, default results/if_eval/):
    summary.csv          one row per metric with value + CI  (machine readable)
    summary.md           the same, formatted, ready to paste into the paper
    per_unit.csv         per-unit rho, first-alarm cycle, lead time, EoL
    per_regime.csv       per-regime n, alarm rate, AUC
    tau_sweep.csv        AUC/PR-AUC vs the degradation threshold
    roc_curves.csv       ROC points for the three detectors (for plotting)
    lead_times.csv       per-unit lead time distribution (for plotting)
    scores_sample.csv    thinned score-vs-RUL sample (for the scatter panel)

Usage
-----
    python3 scripts/eval_iforest.py                       # train fleet (default)
    python3 scripts/eval_iforest.py --csv results/test_anomalies.csv \
            --outdir results/if_eval_test --label test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler

SENSORS = ["T24", "T30", "T50", "P30", "Nf", "Nc", "Ps30", "phi",
           "NRf", "NRc", "BPR", "htBleed", "W31", "W32"]
IF_FEATURES = SENSORS + ["h_clust", "cycles"]
SEED = 42
RNG = np.random.default_rng(SEED)


# ------------------------------------------------------------------ utilities
def boot_ci(fn, *arrays, n_boot: int = 1000, alpha: float = 0.05
            ) -> Tuple[float, float, float]:
    """Point estimate + percentile bootstrap CI, resampling observations."""
    point = fn(*arrays)
    n = len(arrays[0])
    vals = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        try:
            vals.append(fn(*[a[idx] for a in arrays]))
        except Exception:  # noqa: BLE001  (degenerate resample)
            continue
    if not vals:
        return point, np.nan, np.nan
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def paired_auc_diff(y: np.ndarray, s_a: np.ndarray, s_b: np.ndarray,
                    n_boot: int = 1000) -> Dict[str, float]:
    """Bootstrap the AUC difference between two detectors on identical rows."""
    d0 = roc_auc_score(y, s_a) - roc_auc_score(y, s_b)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], s_a[idx])
                     - roc_auc_score(y[idx], s_b[idx]))
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5]) if len(diffs) else (np.nan,
                                                                  np.nan)
    # two-sided bootstrap p: fraction of resamples crossing zero
    p = (2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
         if len(diffs) else np.nan)
    return {"delta_auc": float(d0), "ci_lo": float(lo), "ci_hi": float(hi),
            "p_boot": float(min(p, 1.0))}


def hotelling_t2(X: np.ndarray) -> np.ndarray:
    """Global Hotelling T^2 with pooled mean/covariance — deliberately blind to
    the operating regime, i.e. the classical multivariate control chart."""
    mu = X.mean(0)
    S = np.cov(X, rowvar=False)
    Si = np.linalg.pinv(S)
    d = X - mu
    return np.einsum("ij,jk,ik->i", d, Si, d)


def first_sustained_alarm(cycles: np.ndarray, alarm: np.ndarray,
                          m: int = 2, w: int = 10) -> int | None:
    """First cycle at which >= m alarms have occurred within a w-cycle window."""
    ac = cycles[alarm]
    for i in range(len(ac)):
        lo = ac[i]
        k = int(((ac >= lo) & (ac < lo + w)).sum())
        if k >= m:
            return int(ac[i + m - 1]) if i + m - 1 < len(ac) else int(ac[-1])
    return None


# ------------------------------------------------------------------ the study
def run(df: pd.DataFrame, args) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    rows: List[Dict] = []

    def add(metric, value, lo=np.nan, hi=np.nan, n=np.nan, note=""):
        rows.append({"metric": metric, "value": value, "ci_lo": lo,
                     "ci_hi": hi, "n": n, "note": note})

    df = df.copy()
    df["alarm"] = (df.anomaly_label == -1).astype(int)
    # score orientation: sklearn decision_function is LOW = anomalous, so the
    # "anomaly evidence" used everywhere below is the negated score.
    df["evidence"] = -df.anomaly_score.values
    X = df[SENSORS].astype(float).values
    Xs = StandardScaler().fit_transform(X)

    # ---- A. score-RUL association ---------------------------------------
    rho, p = stats.spearmanr(df.anomaly_score.values, df.RUL.values)
    add("spearman_score_vs_RUL", round(float(rho), 4), note=f"p={p:.3g}")
    per_unit = []
    for u, g in df.groupby("Unit_ID"):
        r_u = (stats.spearmanr(g.anomaly_score.values, g.RUL.values)[0]
               if g.RUL.nunique() > 2 else np.nan)
        eol = int(g.cycles.max())
        gs = g.sort_values("cycles")
        fa = first_sustained_alarm(gs.cycles.values,
                                   gs.alarm.values.astype(bool),
                                   args.sustain_m, args.sustain_w)
        lead = (int(g.loc[g.cycles == fa, "RUL"].iloc[0])
                if fa is not None and (g.cycles == fa).any() else np.nan)
        # degradation-phase alarm: ignore break-in, i.e. only consider alarms
        # after `burnin_frac` of the unit's life has elapsed
        cut = args.burnin_frac * eol
        gd = gs[gs.cycles >= cut]
        fad = first_sustained_alarm(gd.cycles.values,
                                    gd.alarm.values.astype(bool),
                                    args.sustain_m, args.sustain_w)
        leadd = (int(g.loc[g.cycles == fad, "RUL"].iloc[0])
                 if fad is not None and (g.cycles == fad).any() else np.nan)
        per_unit.append({"unit": int(u), "n_rows": len(g),
                         "n_alarms": int(g.alarm.sum()), "eol_cycle": eol,
                         "spearman_rho": r_u, "first_alarm_cycle": fa,
                         "lead_time_cycles": lead,
                         "lead_frac_of_life": (lead / eol
                                               if lead == lead else np.nan),
                         "first_alarm_cycle_degr": fad,
                         "lead_time_degradation": leadd})
    pu = pd.DataFrame(per_unit)
    out["per_unit"] = pu
    add("spearman_per_unit_median", round(float(pu.spearman_rho.median()), 4),
        n=int(pu.spearman_rho.notna().sum()))
    add("spearman_per_unit_frac_expected_sign",
        round(float((pu.spearman_rho > 0).mean()), 4),
        note="rho>0 = score falls as RUL falls = expected direction")

    # ---- B/C. weak labels + baselines -----------------------------------
    pos = df.RUL <= args.tau_pos
    neg = df.RUL >= args.tau_neg
    sub = df[pos | neg].copy()
    y = pos[pos | neg].astype(int).values
    ev_regime = sub.evidence.values
    # global IF (no clustering) on the same feature matrix
    gif = IsolationForest(n_estimators=100, contamination=0.1,
                         random_state=SEED).fit(df[IF_FEATURES].astype(float).values)
    ev_global = -gif.score_samples(sub[IF_FEATURES].astype(float).values)
    # Hotelling T^2, regime-blind
    t2_all = hotelling_t2(Xs)
    ev_t2 = t2_all[(pos | neg).values]

    detectors = {"regime_IF": ev_regime, "global_IF": ev_global,
                 "hotelling_T2": ev_t2}
    roc_rows, auc_tbl = [], {}
    for name, ev in detectors.items():
        a, lo, hi = boot_ci(roc_auc_score, y, ev, n_boot=args.n_boot)
        ap, alo, ahi = boot_ci(average_precision_score, y, ev,
                               n_boot=args.n_boot)
        auc_tbl[name] = a
        add(f"AUC_{name}", round(a, 4), round(lo, 4), round(hi, 4), n=len(y))
        add(f"PRAUC_{name}", round(ap, 4), round(alo, 4), round(ahi, 4),
            n=len(y), note=f"prevalence={y.mean():.3f}")
        fpr, tpr, _ = roc_curve(y, ev)
        k = max(len(fpr) // 400, 1)          # thin for plotting
        roc_rows += [{"detector": name, "fpr": f, "tpr": t}
                     for f, t in zip(fpr[::k], tpr[::k])]
        prec, rec, _ = precision_recall_curve(y, ev)
        k2 = max(len(rec) // 400, 1)
        roc_rows += [{"detector": name, "recall": r, "precision": p2}
                     for r, p2 in zip(rec[::k2], prec[::k2])]
    out["roc_curves"] = pd.DataFrame(roc_rows)
    for b in ("global_IF", "hotelling_T2"):
        d = paired_auc_diff(y, ev_regime, detectors[b], n_boot=args.n_boot)
        add(f"delta_AUC_regimeIF_vs_{b}", round(d["delta_auc"], 4),
            round(d["ci_lo"], 4), round(d["ci_hi"], 4), n=len(y),
            note=f"bootstrap p={d['p_boot']:.4f}")

    # ---- tau sweep -------------------------------------------------------
    sweep = []
    for tp in args.tau_sweep:
        m_pos, m_neg = df.RUL <= tp, df.RUL >= args.tau_neg
        if m_pos.sum() < 20 or m_neg.sum() < 20:
            continue
        s = df[m_pos | m_neg]
        yy = m_pos[m_pos | m_neg].astype(int).values
        r = {"tau_pos": tp, "n_pos": int(m_pos.sum()),
             "n_neg": int(m_neg.sum()),
             "auc_regime_IF": roc_auc_score(yy, s.evidence.values),
             "prauc_regime_IF": average_precision_score(yy, s.evidence.values)}
        r["auc_global_IF"] = roc_auc_score(
            yy, -gif.score_samples(s[IF_FEATURES].astype(float).values))
        r["auc_hotelling_T2"] = roc_auc_score(yy, t2_all[(m_pos | m_neg).values])
        sweep.append(r)
    out["tau_sweep"] = pd.DataFrame(sweep)

    # ---- D. lead time ----------------------------------------------------
    lt = pu.dropna(subset=["lead_time_cycles"])
    out["lead_times"] = lt[["unit", "eol_cycle", "first_alarm_cycle",
                            "lead_time_cycles", "lead_frac_of_life",
                            "first_alarm_cycle_degr", "lead_time_degradation"]]
    add("units_with_sustained_alarm", int(len(lt)), n=int(len(pu)),
        note=f"m={args.sustain_m} alarms within w={args.sustain_w} cycles")
    if len(lt):
        add("lead_time_median_cycles", float(lt.lead_time_cycles.median()),
            float(lt.lead_time_cycles.quantile(.25)),
            float(lt.lead_time_cycles.quantile(.75)), n=len(lt),
            note="IQR in CI columns")
        for thr in (20, 50):
            add(f"frac_units_lead_ge_{thr}",
                round(float((lt.lead_time_cycles >= thr).mean()), 4), n=len(lt))
        ltd = pu.dropna(subset=["lead_time_degradation"])
        if len(ltd):
            add("lead_time_degradation_median_cycles",
                float(ltd.lead_time_degradation.median()),
                float(ltd.lead_time_degradation.quantile(.25)),
                float(ltd.lead_time_degradation.quantile(.75)), n=len(ltd),
                note=f"break-in excluded: alarms after "
                     f"{args.burnin_frac:.0%} of life; IQR in CI columns")
            add("frac_units_degradation_lead_ge_20",
                round(float((ltd.lead_time_degradation >= 20).mean()), 4),
                n=len(ltd))

    # ---- E/F. false alarms + per regime ---------------------------------
    healthy = df[df.RUL >= args.tau_neg]
    add("alarm_rate_healthy_zone", round(float(healthy.alarm.mean()), 4),
        n=len(healthy), note=f"RUL >= {args.tau_neg}")
    add("alarm_rate_degraded_zone",
        round(float(df[df.RUL <= args.tau_pos].alarm.mean()), 4),
        n=int((df.RUL <= args.tau_pos).sum()), note=f"RUL <= {args.tau_pos}")
    add("alarm_rate_overall", round(float(df.alarm.mean()), 4), n=len(df))

    per_reg = []
    for k, g in df.groupby("h_clust"):
        gp, gn = g.RUL <= args.tau_pos, g.RUL >= args.tau_neg
        auc_k = np.nan
        if gp.sum() >= 10 and gn.sum() >= 10:
            gg = g[gp | gn]
            auc_k = roc_auc_score(gp[gp | gn].astype(int).values,
                                  gg.evidence.values)
        per_reg.append({"regime": int(k), "n_rows": len(g),
                        "alarm_rate": float(g.alarm.mean()),
                        "alarm_rate_healthy": (float(g[gn].alarm.mean())
                                               if gn.sum() else np.nan),
                        "auc": auc_k})
    pr = pd.DataFrame(per_reg)
    out["per_regime"] = pr
    add("per_regime_alarm_rate_spread_pp",
        round(float((pr.alarm_rate.max() - pr.alarm_rate.min()) * 100), 3),
        n=len(pr),
        note="~0 BY CONSTRUCTION: contamination=0.1 is enforced inside each "
             "cluster, so this is not evidence of anything")
    add("per_regime_healthy_alarm_spread_pp",
        round(float((pr.alarm_rate_healthy.max()
                     - pr.alarm_rate_healthy.min()) * 100), 2),
        n=len(pr),
        note="max-min false-alarm rate across regimes (the informative one)")
    add("per_regime_auc_min", round(float(pr.auc.min()), 4), n=len(pr))

    # ---- alarm rate by life decile (for the plot) ------------------------
    df["life_frac"] = 1 - df.RUL / df.groupby("Unit_ID").RUL.transform("max")
    bins = np.linspace(0, 1, 11)
    df["life_bin"] = pd.cut(df.life_frac, bins, include_lowest=True)
    prof = (df.groupby("life_bin", observed=True)
            .agg(alarm_rate=("alarm", "mean"), n=("alarm", "size"),
                 mean_evidence=("evidence", "mean")).reset_index())
    prof["life_bin_mid"] = [iv.mid for iv in prof.life_bin]
    out["life_profile"] = prof.drop(columns="life_bin")

    # ---- cumulative-anomaly dynamics (C_i and F_i vs degradation) -------
    ccol = ("global_cumulative_anomaly_count"
            if "global_cumulative_anomaly_count" in df else None)
    fcol = "global_last_3_freq" if "global_last_3_freq" in df else None
    if ccol:
        cr = []
        for u, g in df.groupby("Unit_ID"):
            if g.RUL.nunique() < 3:
                continue
            r_c = stats.spearmanr(g[ccol], g.RUL)[0]
            r_f = (stats.spearmanr(g[fcol], g.RUL)[0]
                   if fcol and g[fcol].nunique() > 2 else np.nan)
            cr.append({"unit": int(u), "rho_cumcount_vs_RUL": r_c,
                       "rho_last3freq_vs_RUL": r_f,
                       "final_cum_count": int(g[ccol].max()),
                       "eol_cycle": int(g.cycles.max())})
        cdf = pd.DataFrame(cr)
        out["cumulative_corr"] = cdf
        add("spearman_cumcount_vs_RUL_pooled",
            round(float(stats.spearmanr(df[ccol], df.RUL)[0]), 4), n=len(df),
            note="negative = anomalies accumulate as life runs out")
        add("spearman_cumcount_vs_RUL_per_unit_median",
            round(float(cdf.rho_cumcount_vs_RUL.median()), 4), n=len(cdf))
        add("frac_units_cumcount_rho_negative",
            round(float((cdf.rho_cumcount_vs_RUL < 0).mean()), 4), n=len(cdf))
        if fcol:
            add("spearman_last3freq_vs_RUL_pooled",
                round(float(stats.spearmanr(df[fcol], df.RUL)[0]), 4),
                n=len(df), note="anomaly acceleration vs remaining life")
        # representative curves: 8 units spanning the lifetime range
        eol = df.groupby("Unit_ID").cycles.max().sort_values()
        pick = eol.iloc[np.linspace(0, len(eol) - 1, 8).astype(int)].index
        cc = df[df.Unit_ID.isin(pick)][["Unit_ID", "cycles", "RUL", ccol]
                                       ].copy()
        cc.columns = ["unit", "cycle", "RUL", "cum_count"]
        out["cum_curves"] = cc.sort_values(["unit", "cycle"])
        # normalised cumulative count against RUL bucket (all units)
        df["_cumnorm"] = df[ccol] / df.groupby("Unit_ID")[ccol].transform(
            lambda x: max(x.max(), 1))
        bins = [0, 10, 20, 30, 50, 75, 100, 150, 10 ** 6]
        df["_rb"] = pd.cut(df.RUL, bins, right=False)
        prof = (df.groupby("_rb", observed=True)
                .agg(n=("_cumnorm", "size"),
                     cumnorm_mean=("_cumnorm", "mean"),
                     cumnorm_se=("_cumnorm", lambda x: x.std() / max(np.sqrt(len(x)), 1)),
                     freq_mean=((fcol or "_cumnorm"), "mean"),
                     alarm_rate=("alarm", "mean")).reset_index())
        prof["rul_lo"] = [iv.left for iv in prof._rb]
        prof["rul_hi"] = [min(iv.right, 400) for iv in prof._rb]
        out["cum_vs_rul"] = prof.drop(columns="_rb")

    # thinned scatter sample
    s = df.sample(min(len(df), 12000), random_state=SEED)
    out["scores_sample"] = s[["Unit_ID", "cycles", "RUL", "anomaly_score",
                              "alarm", "h_clust"]]

    out["summary"] = pd.DataFrame(rows)
    return out


def write_md(summary: pd.DataFrame, args, path: Path) -> None:
    L = ["# Isolation-Forest effectiveness — RUL-anchored evaluation", "",
         f"Dataset: `{args.csv}` ({args.label}). Degradation weak labels: "
         f"positive = RUL <= {args.tau_pos}, negative = RUL >= {args.tau_neg}. "
         f"Anomaly evidence = -decision_function. Bootstrap: "
         f"{args.n_boot} resamples, 95% percentile CIs. The detector never "
         f"observes RUL.", "",
         "| metric | value | 95% CI | n | note |", "|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        ci = ("" if r.ci_lo != r.ci_lo
              else f"[{r.ci_lo:g}, {r.ci_hi:g}]")
        n = "" if r.n != r.n else f"{int(r.n)}"
        L.append(f"| {r.metric} | {r.value:g} | {ci} | {n} | {r.note} |")
    L += ["", "Reading guide:",
          "- `spearman_score_vs_RUL` is POSITIVE in the expected direction: "
          "decision_function is low for anomalies, so it falls together with "
          "RUL as the unit degrades.",
          "- `delta_AUC_regimeIF_vs_*` > 0 with a CI excluding 0 is the "
          "evidence that regime-specific detection beats the regime-blind "
          "alternatives on identical rows.",
          "- `alarm_rate_healthy_zone` is the operational false-alarm proxy. "
          "Note `per_regime_alarm_rate_spread_pp` is ~0 BY CONSTRUCTION "
          "(contamination is enforced per cluster) and must NOT be reported "
          "as a result; use `per_regime_healthy_alarm_spread_pp` and the "
          "per-regime AUCs, which are free to vary.",
          "- The alarm-rate-vs-life profile is U-shaped: elevated at "
          "break-in, minimal mid-life, sharply rising near end of life. "
          "Report both arms — the early-life bump is real, not noise."]
    path.write_text("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/anomalies_multimodal.csv",
                    help="detections CSV with RUL, h_clust, anomaly_* columns")
    ap.add_argument("--label", default="train fleet, run-to-failure")
    ap.add_argument("--outdir", default="results/if_eval")
    ap.add_argument("--tau-pos", type=int, default=30,
                    help="RUL <= this is 'degraded'")
    ap.add_argument("--tau-neg", type=int, default=100,
                    help="RUL >= this is 'healthy'")
    ap.add_argument("--tau-sweep", type=int, nargs="*",
                    default=[10, 20, 30, 40, 50, 75])
    ap.add_argument("--sustain-m", type=int, default=2)
    ap.add_argument("--sustain-w", type=int, default=10)
    ap.add_argument("--burnin-frac", type=float, default=0.5,
                    help="fraction of life treated as break-in when computing "
                         "the degradation-phase lead time")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    need = {"RUL", "h_clust", "anomaly_score", "anomaly_label", "cycles"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"{a.csv} missing columns: {sorted(miss)}")
    if "Unit_ID" not in df.columns and "unit_ID" in df.columns:
        df["Unit_ID"] = df["unit_ID"]

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[eval] {a.csv}: {len(df)} rows, {df.Unit_ID.nunique()} units, "
          f"alarm rate {(df.anomaly_label == -1).mean():.3f}")
    res = run(df, a)
    for k, v in res.items():
        v.to_csv(out / f"{k}.csv", index=False)
    write_md(res["summary"], a, out / "summary.md")
    (out / "config.json").write_text(json.dumps(vars(a), indent=1))
    print(f"[eval] wrote {len(res)} tables + summary.md -> {out}")
    key = ["spearman_score_vs_RUL", "AUC_regime_IF", "AUC_global_IF",
           "AUC_hotelling_T2", "delta_AUC_regimeIF_vs_global_IF",
           "delta_AUC_regimeIF_vs_hotelling_T2", "lead_time_median_cycles",
           "alarm_rate_healthy_zone"]
    s = res["summary"].set_index("metric")
    print("\nheadlines:")
    for k in key:
        if k in s.index:
            r = s.loc[k]
            ci = ("" if r.ci_lo != r.ci_lo else
                  f"  [{r.ci_lo:g}, {r.ci_hi:g}]")
            print(f"  {k:38s} {r.value:>8g}{ci}   {r.note}")


if __name__ == "__main__":
    main()

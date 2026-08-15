#!/usr/bin/env python3
"""
plot_paper.py — publication figures from the evaluation tables
==============================================================

Reads what `eval_iforest.py` wrote (and, if present, your rule-to-interpretation
anchoring metrics) and renders camera-ready multi-panel figures as vector PDF +
300 dpi PNG, in a consistent journal style: serif type, colour-blind-safe
palette, no chart junk, panel letters, sizes set to Elsevier column widths
(single 90 mm, double 190 mm).

    Figure 1  detector effectiveness
              (a) anomaly score vs RUL density   (b) alarm rate over life
              (c) ROC, three detectors           (d) detection lead time
    Figure 2  where the gain comes from
              (a) per-regime AUC + false alarms  (b) AUC vs degradation threshold
              (c) paired Delta-AUC with 95% CI   (d) per-unit rho distribution
    Figure 3  rule-to-interpretation anchoring   [only if metrics are found]

Usage
-----
    python3 scripts/plot_paper.py
    python3 scripts/plot_paper.py --eval results/if_eval --outdir results/figures/paper
    python3 scripts/plot_paper.py --anchor results/anchoring_metrics.csv
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Dict, Optional

import json

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

MM = 1 / 25.4
W1, W2 = 90 * MM, 190 * MM                   # single / double column
# Okabe-Ito, colour-blind safe
C = {"main": "#0072B2", "alt": "#D55E00", "third": "#009E73",
     "grey": "#666666", "light": "#BBBBBB", "accent": "#CC79A7"}

STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "lines.linewidth": 1.2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "-",
    "legend.frameon": False, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,      # editable text in the PDF
}


def panel(ax, letter: str, title: str = "") -> None:
    ax.text(-0.16, 1.06, f"({letter})", transform=ax.transAxes,
            fontweight="bold", fontsize=9, va="bottom", ha="left")
    if title:
        ax.set_title(title, loc="left", pad=3)


def save(fig, outdir: Path, name: str) -> str:
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{name}.{ext}")
    plt.close(fig)
    return f"{name}.pdf/.png"


def load(evdir: Path) -> Dict[str, pd.DataFrame]:
    d = {}
    for f in evdir.glob("*.csv"):
        d[f.stem] = pd.read_csv(f)
    if "summary" not in d:
        raise SystemExit(f"no summary.csv in {evdir}; run eval_iforest.py first")
    return d


def sval(summary: pd.DataFrame, metric: str, col: str = "value"):
    r = summary[summary.metric == metric]
    return float(r.iloc[0][col]) if len(r) else np.nan


# ------------------------------------------------------------------ figure 1
def figure1(D: Dict[str, pd.DataFrame], outdir: Path) -> str:
    s = D["summary"]
    fig, axes = plt.subplots(2, 2, figsize=(W2, 0.62 * W2))
    (a, b), (c, d) = axes

    # (a) score vs RUL density -------------------------------------------
    sc = D["scores_sample"]
    hb = a.hexbin(sc.RUL, sc.anomaly_score, gridsize=40, bins="log",
                  cmap="Blues", mincnt=1, linewidths=0)
    cb = fig.colorbar(hb, ax=a, pad=0.02, fraction=0.045)
    cb.set_label("observations (log)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    a.axhline(0, color=C["alt"], ls="--", lw=0.9)
    a.text(sc.RUL.max() * 0.98, 0.004, "alarm threshold", color=C["alt"],
           ha="right", va="bottom", fontsize=6.5)
    rho = sval(s, "spearman_score_vs_RUL")
    a.set_xlabel("remaining useful life (cycles)")
    a.set_ylabel("anomaly score")
    a.text(0.97, 0.06, rf"Spearman $\rho$ = {rho:.2f}", transform=a.transAxes,
           ha="right", fontsize=7)
    panel(a, "a", "Score tracks degradation")

    # (b) alarm rate over life -------------------------------------------
    lp = D["life_profile"].sort_values("life_bin_mid")
    b.plot(lp.life_bin_mid, lp.alarm_rate, "o-", color=C["main"], ms=3.2)
    b.fill_between(lp.life_bin_mid, 0, lp.alarm_rate, color=C["main"],
                   alpha=0.12)
    b.set_xlabel("fraction of life elapsed")
    b.set_ylabel("alarm rate")
    b.yaxis.set_major_formatter(PercentFormatter(1.0))
    b.set_xlim(0, 1)
    imin = int(lp.alarm_rate.idxmin())
    b.annotate("break-in", xy=(lp.life_bin_mid.iloc[0],
                               lp.alarm_rate.iloc[0]),
               xytext=(0.16, lp.alarm_rate.max() * 0.42), fontsize=6.5,
               color=C["grey"],
               arrowprops=dict(arrowstyle="->", lw=0.5, color=C["grey"]))
    b.annotate("wear-out", xy=(lp.life_bin_mid.iloc[-1],
                               lp.alarm_rate.iloc[-1]),
               xytext=(0.55, lp.alarm_rate.max() * 0.8), fontsize=6.5,
               color=C["grey"],
               arrowprops=dict(arrowstyle="->", lw=0.5, color=C["grey"]))
    panel(b, "b", "Bathtub profile of alarms")

    # (c) ROC -------------------------------------------------------------
    roc = D["roc_curves"]
    names = {"regime_IF": "regime-specific IF",
             "global_IF": "global IF (no clustering)",
             "hotelling_T2": r"Hotelling $T^2$"}
    cols = {"regime_IF": C["main"], "global_IF": C["third"],
            "hotelling_T2": C["alt"]}
    for k, lab in names.items():
        g = roc[(roc.detector == k) & roc.fpr.notna()].sort_values("fpr")
        if not len(g):
            continue
        au = sval(s, f"AUC_{k}")
        lo = sval(s, f"AUC_{k}", "ci_lo")
        hi = sval(s, f"AUC_{k}", "ci_hi")
        c.plot(g.fpr, g.tpr, color=cols[k], lw=1.3,
               label=f"{lab}\nAUC {au:.3f} [{lo:.3f}, {hi:.3f}]")
    c.plot([0, 1], [0, 1], color=C["light"], ls=":", lw=0.8)
    c.set_xlabel("false-positive rate")
    c.set_ylabel("true-positive rate")
    c.legend(loc="lower right", handlelength=1.4, labelspacing=0.7)
    panel(c, "c", "Discriminating degraded from healthy")

    # (d) lead time -------------------------------------------------------
    lt = D["lead_times"]
    col = ("lead_time_degradation"
           if "lead_time_degradation" in lt and
           lt.lead_time_degradation.notna().sum() > 10 else "lead_time_cycles")
    lt = lt.dropna(subset=[col])
    d.hist(lt[col], bins=28, color=C["main"], alpha=0.75,
           edgecolor="white", linewidth=0.3)
    med = lt[col].median()
    d.axvline(med, color=C["alt"], lw=1.1)
    d.text(med, d.get_ylim()[1] * 0.94, f" median {med:.0f} cycles",
           color=C["alt"], fontsize=6.8, va="top")
    d.set_xlabel("lead time of first sustained alarm (cycles)")
    d.set_ylabel("engine units")
    n_ge = sval(s, "frac_units_degradation_lead_ge_20")
    if n_ge != n_ge:
        n_ge = sval(s, "frac_units_lead_ge_20")
    d.text(0.97, 0.55, f"{n_ge:.0%} of units warned\n$\\geq$20 cycles ahead\n"
                       f"(break-in alarms excluded)",
           transform=d.transAxes, ha="right", fontsize=6.5, color=C["grey"])
    panel(d, "d", "Operational warning horizon")

    fig.tight_layout(w_pad=1.8, h_pad=1.6)
    return save(fig, outdir, "fig1_detector_effectiveness")


# ------------------------------------------------------------------ figure 2
def figure2(D: Dict[str, pd.DataFrame], outdir: Path) -> str:
    s = D["summary"]
    fig, axes = plt.subplots(2, 2, figsize=(W2, 0.62 * W2))
    (a, b), (c, d) = axes

    # (a) per-regime AUC + healthy false alarm ----------------------------
    pr = D["per_regime"].sort_values("regime")
    x = np.arange(len(pr))
    a.bar(x, pr.auc, 0.62, color=C["main"], label="AUC (degraded vs healthy)")
    a.set_ylim(0.5, 1.0)
    a.set_xticks(x)
    a.set_xticklabels([f"R{int(k)}" for k in pr.regime])
    a.set_xlabel("operating regime")
    a.set_ylabel("AUC")
    a2 = a.twinx()
    a2.plot(x, pr.alarm_rate_healthy * 100, "s--", color=C["alt"], ms=3.4,
            lw=1.0, label="false-alarm rate (healthy)")
    a2.set_ylabel("false alarms in healthy zone (%)", color=C["alt"])
    a2.tick_params(axis="y", colors=C["alt"], labelsize=7)
    a2.grid(False)
    a2.spines["right"].set_visible(True)
    a2.spines["right"].set_color(C["alt"])
    a.set_ylim(0.5, 1.02)
    a.text(0.02, 0.955, "bars: AUC", transform=a.transAxes, fontsize=6.5,
           color=C["main"])
    a2.text(0.98, 0.955, "line: false alarms", transform=a2.transAxes,
            fontsize=6.5, color=C["alt"], ha="right")
    panel(a, "a", "Consistency across operating regimes")

    # (b) tau sweep -------------------------------------------------------
    ts = D["tau_sweep"].sort_values("tau_pos")
    for k, lab, col in (("auc_regime_IF", "regime-specific IF", C["main"]),
                        ("auc_global_IF", "global IF", C["third"]),
                        ("auc_hotelling_T2", r"Hotelling $T^2$", C["alt"])):
        if k in ts:
            b.plot(ts.tau_pos, ts[k], "o-", color=col, ms=3.2, label=lab)
    b.set_xlabel(r"degradation threshold $\tau$ (cycles)")
    b.set_ylabel("AUC")
    b.legend(loc="upper right")
    panel(b, "b", "Robust to the label definition")

    # (c) delta AUC forest -------------------------------------------------
    rows = [("vs global IF (no clustering)",
             "delta_AUC_regimeIF_vs_global_IF"),
            (r"vs Hotelling $T^2$", "delta_AUC_regimeIF_vs_hotelling_T2")]
    ys = np.arange(len(rows))[::-1]
    for y, (lab, m) in zip(ys, rows):
        v, lo, hi = (sval(s, m), sval(s, m, "ci_lo"), sval(s, m, "ci_hi"))
        c.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt="o", color=C["main"],
                   ms=5, capsize=2.5, lw=1.2)
        c.text(hi + 0.004, y, f"+{v:.3f}", va="center", fontsize=7,
               color=C["main"])
    c.axvline(0, color=C["grey"], ls="--", lw=0.8)
    c.set_yticks(ys)
    c.set_yticklabels([r[0] for r in rows])
    c.set_xlabel(r"$\Delta$AUC of regime-specific IF (95% bootstrap CI)")
    c.set_ylim(-0.6, len(rows) - 0.4)
    c.grid(axis="y", alpha=0)
    panel(c, "c", "The clustering earns its place")

    # (d) per-unit rho ----------------------------------------------------
    pu = D["per_unit"].dropna(subset=["spearman_rho"])
    d.hist(pu.spearman_rho, bins=30, color=C["main"], alpha=0.75,
           edgecolor="white", linewidth=0.3)
    d.axvline(0, color=C["grey"], ls="--", lw=0.8)
    med = pu.spearman_rho.median()
    d.axvline(med, color=C["alt"], lw=1.1)
    d.text(med, d.get_ylim()[1] * 0.95, f" median {med:.2f}", color=C["alt"],
           fontsize=6.8, va="top")
    frac = sval(s, "spearman_per_unit_frac_expected_sign")
    d.set_xlabel(r"per-unit Spearman $\rho$ (score vs RUL)")
    d.set_ylabel("engine units")
    d.text(0.03, 0.78, f"{frac:.0%} of units in the\nexpected direction",
           transform=d.transAxes, fontsize=6.8, color=C["grey"])
    panel(d, "d", "Holds unit by unit")

    fig.tight_layout(w_pad=2.2, h_pad=1.6)
    return save(fig, outdir, "fig2_regime_and_baselines")



# ------------------------------------------------------------------ figure 3
def figure3_cumulative(D, outdir: Path):
    """Cumulative-anomaly dynamics: the temporal evidence that the detector
    tracks degradation rather than firing at random."""
    if "cum_curves" not in D or "cum_vs_rul" not in D:
        return None
    s = D["summary"]
    fig, axes = plt.subplots(2, 2, figsize=(W2, 0.62 * W2))
    (a, b), (c, d) = axes

    # (a) per-unit cumulative curves -------------------------------------
    cc = D["cum_curves"]
    units = list(dict.fromkeys(cc.unit))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(units)))
    for col, u in zip(cmap, units):
        g = cc[cc.unit == u].sort_values("cycle")
        a.plot(g.cycle, g.cum_count, color=col, lw=1.0)
        a.plot(g.cycle.iloc[-1], g.cum_count.iloc[-1], "o", color=col, ms=2.6)
    a.set_xlabel("operating cycle")
    a.set_ylabel("cumulative anomalies $C_i$")
    a.text(0.03, 0.93, f"{len(units)} representative units\n(dot = failure)",
           transform=a.transAxes, fontsize=6.6, va="top", color=C["grey"])
    panel(a, "a", "Flat in health, steep at wear-out")

    # (b) normalised cumulative count vs RUL ------------------------------
    cv = D["cum_vs_rul"].sort_values("rul_lo")
    xm = (cv.rul_lo + cv.rul_hi.clip(upper=200)) / 2
    b.errorbar(xm, cv.cumnorm_mean, yerr=1.96 * cv.cumnorm_se, fmt="o-",
               color=C["main"], ms=3.4, lw=1.2, capsize=2)
    b.invert_xaxis()
    b.set_xlabel("remaining useful life (cycles, reversed)")
    b.set_ylabel("normalised $C_i$")
    rho = sval(s, "spearman_cumcount_vs_RUL_per_unit_median")
    b.text(0.03, 0.93, rf"per-unit median $\rho$ = {rho:.2f}",
           transform=b.transAxes, fontsize=7, va="top")
    panel(b, "b", "Accumulation rises as life runs out")

    # (c) per-unit correlation distributions -------------------------------
    cr = D["cumulative_corr"]
    c.hist(cr.rho_cumcount_vs_RUL.dropna(), bins=26, color=C["main"],
           alpha=0.8, edgecolor="white", linewidth=0.3, label=r"$C_i$")
    if cr.rho_last3freq_vs_RUL.notna().any():
        c.hist(cr.rho_last3freq_vs_RUL.dropna(), bins=26, color=C["accent"],
               alpha=0.55, edgecolor="white", linewidth=0.3,
               label=r"$F_i$ (frequency)")
    c.axvline(0, color=C["grey"], ls="--", lw=0.8)
    frac = sval(s, "frac_units_cumcount_rho_negative")
    c.set_xlabel(r"per-unit Spearman $\rho$ vs RUL")
    c.set_ylabel("engine units")
    c.legend(loc="upper right")
    c.text(0.03, 0.93, f"{frac:.0%} of units negative\nfor $C_i$",
           transform=c.transAxes, fontsize=6.6, va="top", color=C["grey"])
    panel(c, "c", "Consistent in every unit")

    # (d) anomaly frequency + alarm rate vs RUL ---------------------------
    d.plot(xm, cv.freq_mean, "o-", color=C["accent"], ms=3.4,
           label=r"anomaly frequency $F_i$")
    d.set_ylabel(r"mean $F_i$", color=C["accent"])
    d.tick_params(axis="y", colors=C["accent"], labelsize=7)
    d2 = d.twinx()
    d2.plot(xm, cv.alarm_rate * 100, "s--", color=C["main"], ms=3.2, lw=1.0)
    d2.set_ylabel("alarm rate (%)", color=C["main"])
    d2.tick_params(axis="y", colors=C["main"], labelsize=7)
    d2.grid(False)
    d2.spines["right"].set_visible(True)
    d.invert_xaxis()
    d.set_xlabel("remaining useful life (cycles, reversed)")
    d.text(0.03, 0.93, "circles: $F_i$\nsquares: alarm rate",
           transform=d.transAxes, fontsize=6.4, va="top", color=C["grey"])
    panel(d, "d", "Anomalies also accelerate")

    fig.tight_layout(w_pad=2.2, h_pad=1.6)
    return save(fig, outdir, "fig3_cumulative_dynamics")


# ------------------------------------------------------------------ figure 4
def figure4_detail(D, outdir: Path):
    """Discrimination detail: PR curves, score separation, operating point."""
    s = D["summary"]
    fig, axes = plt.subplots(2, 2, figsize=(W2, 0.62 * W2))
    (a, b), (c, d) = axes
    roc = D["roc_curves"]
    names = {"regime_IF": "regime-specific IF",
             "global_IF": "global IF", "hotelling_T2": r"Hotelling $T^2$"}
    cols = {"regime_IF": C["main"], "global_IF": C["third"],
            "hotelling_T2": C["alt"]}

    # (a) precision-recall -------------------------------------------------
    for k, lab in names.items():
        g = roc[(roc.detector == k) & roc.recall.notna()].sort_values("recall")
        if len(g):
            ap = sval(s, f"PRAUC_{k}")
            a.plot(g.recall, g.precision, color=cols[k], lw=1.3,
                   label=f"{lab}  AP {ap:.3f}")
    prev = 0.225
    m = s[s.metric == "PRAUC_regime_IF"]
    if len(m) and isinstance(m.iloc[0].note, str) and "prevalence" in str(m.iloc[0].note):
        prev = float(str(m.iloc[0].note).split("=")[1])
    a.axhline(prev, color=C["light"], ls=":", lw=0.9)
    a.text(0.02, prev + 0.025, "chance", ha="left", fontsize=6.4,
           color=C["grey"])
    a.set_xlabel("recall")
    a.set_ylabel("precision")
    a.legend(loc="lower left")
    panel(a, "a", "Precision-recall (imbalanced view)")

    # (b) score separation -------------------------------------------------
    sc = D["scores_sample"]
    hi = sc[sc.RUL >= 100].anomaly_score
    lo = sc[sc.RUL <= 30].anomaly_score
    bins = np.linspace(sc.anomaly_score.min(), sc.anomaly_score.max(), 45)
    b.hist(hi, bins=bins, color=C["third"], alpha=0.65, density=True,
           label=r"healthy (RUL$\geq$100)", edgecolor="white", linewidth=0.2)
    b.hist(lo, bins=bins, color=C["alt"], alpha=0.6, density=True,
           label=r"degraded (RUL$\leq$30)", edgecolor="white", linewidth=0.2)
    b.axvline(0, color=C["grey"], ls="--", lw=0.9)
    b.set_xlabel("anomaly score")
    b.set_ylabel("density")
    b.legend(loc="upper left")
    panel(b, "b", "Score distributions separate")

    # (c) alarm rate vs RUL with Wilson CI --------------------------------
    cv = D["cum_vs_rul"].sort_values("rul_lo")
    xm = (cv.rul_lo + cv.rul_hi.clip(upper=200)) / 2
    p_ = cv.alarm_rate.values
    n_ = cv.n.values
    half = 1.96 * np.sqrt(p_ * (1 - p_) / n_)
    c.errorbar(xm, p_ * 100, yerr=half * 100, fmt="o-", color=C["main"],
               ms=3.4, lw=1.2, capsize=2)
    c.invert_xaxis()
    c.set_xlabel("remaining useful life (cycles, reversed)")
    c.set_ylabel("alarm rate (%)")
    panel(c, "c", "Operating characteristic over life")

    # (d) lead-time ECDF ---------------------------------------------------
    lt = D["lead_times"]
    col = ("lead_time_degradation" if "lead_time_degradation" in lt
           and lt.lead_time_degradation.notna().sum() > 10
           else "lead_time_cycles")
    v = np.sort(lt[col].dropna().values)
    d.step(v, np.arange(1, len(v) + 1) / len(v), color=C["main"], lw=1.3)
    for thr in (20, 50):
        f = (v >= thr).mean()
        d.axvline(thr, color=C["light"], ls=":", lw=0.8)
        d.text(thr, 0.04, f" {f:.0%} $\\geq${thr}", fontsize=6.4,
               color=C["grey"], rotation=90, va="bottom")
    d.set_xlabel("lead time (cycles before failure)")
    d.set_ylabel("cumulative share of units")
    d.set_ylim(0, 1.02)
    panel(d, "d", "How many units get enough warning")

    fig.tight_layout(w_pad=2.0, h_pad=1.6)
    return save(fig, outdir, "fig4_discrimination_detail")


# --------------------------------------------------------------- SLM figures
def figure5_slm(sd: Path, outdir: Path):
    """SLM interpretation quality against the rule it was handed."""
    pa = pd.read_csv(sd / "per_anomaly.csv")
    sm = pd.read_csv(sd / "summary.csv").set_index("metric")
    fig, axes = plt.subplots(2, 2, figsize=(W2, 0.62 * W2))
    (a, b), (c, d) = axes

    # (a) grounding rates with CI ------------------------------------------
    order = [("direction_agreement", "direction agreement", True),
             ("format_complete", "format compliance", True),
             ("threshold_anchoring", "threshold anchoring", True),
             ("sensor_precision", "sensor precision", True),
             ("sensor_recall", "sensor recall", True),
             ("direction_contradiction", "direction contradiction", False),
             ("echo_contamination", "echo contamination", False)]
    rows = [(lab, sm.loc[k, "value"], sm.loc[k, "ci_lo"], sm.loc[k, "ci_hi"],
             good) for k, lab, good in order if k in sm.index]
    y = np.arange(len(rows))[::-1]
    for yy, (lab, v, lo, hi, good) in zip(y, rows):
        col = C["main"] if good else C["alt"]
        a.barh(yy, v, 0.62, color=col, alpha=0.9)
        if lo == lo:
            a.plot([lo, hi], [yy, yy], color=C["grey"], lw=1.0)
        a.text(min(max(v, hi if hi == hi else v) + 0.025, 0.98), yy,
               f"{v:.1%}", va="center", fontsize=6.8)
    a.set_yticks(y)
    a.set_yticklabels([r[0] for r in rows])
    a.set_xlim(0, 1.08)
    a.xaxis.set_major_formatter(PercentFormatter(1.0))
    a.set_xlabel("rate (95% CI)")
    a.grid(axis="y", alpha=0)
    panel(a, "a", "Rule grounding and faithfulness")

    # (b) recall vs rule complexity ---------------------------------------
    g = (pa.groupby("n_rule_sensors")
         .agg(n=("sensor_recall", "size"),
              recall=("sensor_recall", "mean"),
              se=("sensor_recall", lambda x: x.std() / max(np.sqrt(len(x)), 1)))
         .reset_index())
    g = g[g.n >= 2]
    b.errorbar(g.n_rule_sensors, g.recall, yerr=1.96 * g.se.fillna(0),
               fmt="o-", color=C["main"], ms=3.6, lw=1.2, capsize=2)
    for _, r in g.iterrows():
        b.text(r.n_rule_sensors, min(r.recall + 0.06, 1.02), f"n={int(r.n)}",
               ha="center", fontsize=5.8, color=C["grey"])
    b.set_xlabel("distinct sensors in the isolation rule")
    b.set_ylabel("sensor recall")
    b.set_ylim(0, 1.12)
    b.yaxis.set_major_formatter(PercentFormatter(1.0))
    panel(b, "b", "Coverage falls as rules grow")

    # (c) gravity vs true RUL ---------------------------------------------
    gg = pa.dropna(subset=["gravity"])
    levels = sorted(gg.gravity.unique())
    data = [gg[gg.gravity == l].RUL.values for l in levels]
    bp = c.boxplot(data, positions=range(len(levels)), widths=0.6,
                   patch_artist=True, medianprops=dict(color="white", lw=1.2),
                   flierprops=dict(marker="o", ms=2, alpha=0.5))
    for patch in bp["boxes"]:
        patch.set_facecolor(C["main"])
        patch.set_alpha(0.85)
        patch.set_edgecolor("none")
    ylo = c.get_ylim()[0]
    for i, l in enumerate(levels):
        c.text(i, ylo, f"n={len(data[i])}", ha="center", fontsize=5.8,
               color=C["grey"], va="bottom")
    c.set_xticks(range(len(levels)))
    c.set_xticklabels([f"{int(l)}" for l in levels])
    c.set_xlabel("gravity score assigned by the SLM (1-5)")
    c.set_ylabel("true RUL (cycles)")
    if "spearman_gravity_vs_RUL" in sm.index:
        rho = sm.loc["spearman_gravity_vs_RUL", "value"]
        pv = (sm.loc["spearman_gravity_vs_RUL_pvalue", "value"]
              if "spearman_gravity_vs_RUL_pvalue" in sm.index else np.nan)
        c.text(0.97, 0.90, rf"$\rho$ = {rho:.2f}" +
               (f", p = {pv:.3f}" if pv == pv else ""),
               transform=c.transAxes, ha="right", fontsize=7)
    panel(c, "c", "Severity opinion vs reality")

    # (d) per-anomaly recall / precision distributions ---------------------
    d.hist(pa.sensor_recall.dropna(), bins=18, color=C["main"], alpha=0.75,
           edgecolor="white", linewidth=0.3, label="recall")
    d.hist(pa.sensor_precision.dropna(), bins=18, color=C["third"],
           alpha=0.6, edgecolor="white", linewidth=0.3, label="precision")
    d.set_xlabel("per-anomaly rate")
    d.set_ylabel("anomalies")
    d.xaxis.set_major_formatter(PercentFormatter(1.0))
    d.legend(loc="upper left")
    panel(d, "d", "Spread across anomalies")

    fig.tight_layout(w_pad=2.0, h_pad=1.6)
    return save(fig, outdir, "fig5_slm_quality")


def figure6_slm_cost(sd: Path, outdir: Path, telemetry: Path | None):
    """Measured edge cost of the interpretation layer."""
    pa = pd.read_csv(sd / "per_anomaly.csv")
    if "wall_s" not in pa or not pa.wall_s.notna().any():
        return None
    tel = None
    if telemetry and telemetry.exists():
        tel = pd.DataFrame([json.loads(l) for l in
                            telemetry.read_text().splitlines() if l.strip()])
    fig, axes = plt.subplots(2, 2, figsize=(W2, 0.62 * W2))
    (a, b), (c, d) = axes

    w = pa.wall_s.dropna()
    a.hist(w, bins=22, color=C["main"], alpha=0.8, edgecolor="white",
           linewidth=0.3)
    for q, lab, col in ((0.5, "p50", C["alt"]), (0.95, "p95", C["accent"])):
        v = w.quantile(q)
        a.axvline(v, color=col, lw=1.1)
        a.text(v, a.get_ylim()[1] * (0.95 if q == 0.5 else 0.8),
               f" {lab} {v:.1f}s", color=col, fontsize=6.6, va="top")
    a.set_xlabel("interpretation latency (s)")
    a.set_ylabel("anomalies")
    panel(a, "a", "Per-anomaly latency on the edge")

    if "gen_tokens" in pa:
        b.scatter(pa.gen_tokens, pa.wall_s, s=12, alpha=0.6, color=C["main"],
                  edgecolors="none")
        ok = pa.dropna(subset=["gen_tokens", "wall_s"])
        if len(ok) > 3:
            k = np.polyfit(ok.gen_tokens, ok.wall_s, 1)
            xs = np.linspace(ok.gen_tokens.min(), ok.gen_tokens.max(), 50)
            b.plot(xs, np.polyval(k, xs), color=C["alt"], lw=1.1)
            if "decode_tps" in pa and pa.decode_tps.notna().any():
                b.text(0.03, 0.93,
                       f"median decode {pa.decode_tps.median():.0f} tok/s",
                       transform=b.transAxes, fontsize=6.6, va="top",
                       color=C["grey"])
        b.set_xlabel("generated tokens")
        b.set_ylabel("latency (s)")
    panel(b, "b", "Latency is decode-bound")

    if "prefill_s" in pa and pa.prefill_s.notna().any():
        idx = np.argsort(pa.wall_s.values)
        pf = pa.prefill_s.values[idx]
        dc = pa.decode_s.values[idx] if "decode_s" in pa else np.zeros_like(pf)
        x = np.arange(len(pf))
        c.bar(x, pf, 1.0, color=C["third"], label="prefill")
        c.bar(x, dc, 1.0, bottom=pf, color=C["main"], label="decode")
        c.set_xlabel("anomalies (sorted by latency)")
        c.set_ylabel("seconds")
        c.legend(loc="upper left")
    panel(c, "c", "Where the time goes")

    if tel is not None and len(tel):
        t0 = tel.t.min()
        tt = (tel.t - t0) / 60
        c2 = d
        if "mem_used_mb" in tel and tel.mem_used_mb.notna().any():
            c2.plot(tt, tel.mem_used_mb / 1024, color=C["main"], lw=1.0)
            c2.set_ylabel("RAM (GiB)", color=C["main"])
            c2.tick_params(axis="y", colors=C["main"], labelsize=7)
        ax2 = c2.twinx()
        if "gpu_pct" in tel and tel.gpu_pct.notna().any():
            ax2.plot(tt, tel.gpu_pct, color=C["alt"], lw=0.8, alpha=0.75)
            ax2.set_ylabel("GPU (%)", color=C["alt"])
            ax2.tick_params(axis="y", colors=C["alt"], labelsize=7)
        elif "temp_c_max" in tel and tel.temp_c_max.notna().any():
            ax2.plot(tt, tel.temp_c_max, color=C["alt"], lw=0.8)
            ax2.set_ylabel("temp (C)", color=C["alt"])
        ax2.grid(False)
        ax2.spines["right"].set_visible(True)
        c2.set_xlabel("time (min)")
        if "temp_c_max" in tel and tel.temp_c_max.notna().any():
            c2.text(0.45, 0.04, f"peak {tel.temp_c_max.max():.0f} C",
                    transform=c2.transAxes, fontsize=6.4, color=C["grey"])
    panel(d, "d", "Device telemetry during the run")

    fig.tight_layout(w_pad=2.2, h_pad=1.6)
    return save(fig, outdir, "fig6_slm_edge_cost")


# ------------------------------------------------------------- anchoring fig
def find_anchor(explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    pats = ["results/**/anchor*.csv", "results/**/*anchoring*.csv",
            "results/**/*grounding*.csv", "results/**/*faithful*.csv",
            "results/**/rule_interp*.csv"]
    for pat in pats:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return Path(hits[0])
    return None


def figure_anchor(path: Path, outdir: Path) -> Optional[str]:
    """Anchoring metrics: rate-like columns in [0,1] are plotted as a ranked
    bar chart with Wilson CIs when a count is available; per-anomaly columns
    get a distribution panel. Column names are discovered, not assumed."""
    df = pd.read_csv(path)
    num = df.select_dtypes("number")
    rate_cols = [c for c in num.columns
                 if num[c].dropna().between(0, 1).all()
                 and num[c].nunique() > 1]
    if not rate_cols:
        print(f"[plot] {path.name}: no rate-like columns found, skipping fig3")
        return None
    per_row = len(df) > 5                     # per-anomaly table vs summary row
    fig, axes = plt.subplots(1, 2 if per_row else 1,
                             figsize=(W2 if per_row else W1, 0.42 * W2))
    axes = np.atleast_1d(axes)

    means = df[rate_cols].mean().sort_values()
    ax = axes[0]
    y = np.arange(len(means))
    ax.barh(y, means.values, 0.6, color=C["main"])
    lab_x = list(means.values)
    if per_row:
        n = len(df)
        for i, c_ in enumerate(means.index):     # Wilson 95%
            p_ = means[c_]
            z = 1.96
            den = 1 + z ** 2 / n
            ctr = (p_ + z ** 2 / (2 * n)) / den
            half = z * np.sqrt(p_ * (1 - p_) / n + z ** 2 / (4 * n ** 2)) / den
            hi_ = min(ctr + half, 1.0)
            ax.plot([max(ctr - half, 0), hi_], [i, i], color=C["grey"], lw=1.0)
            lab_x[i] = max(lab_x[i], hi_)
    for i, (v, x_) in enumerate(zip(means.values, lab_x)):
        ax.text(min(x_ + 0.022, 0.985), i, f"{v:.1%}", va="center",
                fontsize=6.8)
    ax.set_yticks(y)
    ax.set_yticklabels([c_.replace("_", " ") for c_ in means.index])
    ax.set_xlim(0, 1.06)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("rate")
    ax.grid(axis="y", alpha=0)
    panel(ax, "a", "Rule-to-interpretation anchoring")

    if per_row:
        ax2 = axes[1]
        main = means.index[-1]
        vals = df[main].dropna()
        ax2.hist(vals, bins=min(20, max(vals.nunique(), 5)), color=C["third"],
                 alpha=0.8, edgecolor="white", linewidth=0.3)
        ax2.axvline(vals.mean(), color=C["alt"], lw=1.1)
        ax2.set_xlabel(main.replace("_", " "))
        ax2.set_ylabel("anomalies")
        panel(ax2, "b", "Per-anomaly distribution")

    fig.tight_layout(w_pad=2.0)
    return save(fig, outdir, "fig3_anchoring")


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="results/if_eval")
    ap.add_argument("--outdir", default="results/figures/paper")
    ap.add_argument("--slm", default="results/slm_eval")
    ap.add_argument("--telemetry", default="results/telemetry.jsonl")
    ap.add_argument("--anchor", default=None,
                    help="CSV of rule-to-interpretation anchoring metrics "
                         "(auto-discovered under results/ if omitted)")
    a = ap.parse_args()

    plt.rcParams.update(STYLE)
    D = load(Path(a.eval))
    out = Path(a.outdir)
    made = [figure1(D, out), figure2(D, out)]
    for fn in (figure3_cumulative, figure4_detail):
        r = fn(D, out)
        if r:
            made.append(r)
    sd = Path(a.slm)
    if (sd / "per_anomaly.csv").exists():
        made.append(figure5_slm(sd, out))
        r6 = figure6_slm_cost(sd, out, Path(a.telemetry))
        if r6:
            made.append(r6)
    else:
        print(f"[plot] no SLM eval at {sd} (run eval_slm.py) — "
              f"skipping SLM figures")

    ap_path = find_anchor(a.anchor)
    if ap_path:
        f3 = figure_anchor(ap_path, out)
        if f3:
            made.append(f3)
            print(f"[plot] anchoring metrics from {ap_path}")
    else:
        print("[plot] no anchoring-metrics CSV found "
              "(pass --anchor path/to.csv for figure 3)")

    print(f"[plot] wrote -> {out}")
    for m in made:
        print(f"        {m}")


if __name__ == "__main__":
    main()

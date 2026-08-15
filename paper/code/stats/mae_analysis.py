#!/usr/bin/env python3
"""mae_analysis.py — MAE under every useful lens:
by life stage x arm, corrupted-vs-clean tool split (the rescue,
quantified), signal quadrants (reliability x uncertainty), per-unit
heterogeneity, and the paired agent-tool delta geometry.

Source: results/final_prognostic (all four arms) + signals recomputed.
Outputs -> agentic/results/analysis/ (figMAE1, figMAE2, mae_stats.md)
Run from repo root:  python paper/code/stats/mae_analysis.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "agentic"))
from apdm.prognostic.future_progression import (reliability,          # noqa
                                                future_progression)
from apdm.prognostic.forecast import PrecedentFutures                 # noqa

MM = 1 / 25.4
C = {"tool": "#D55E00", "agent": "#0072B2", "agent0": "#56B4E9",
     "b0": "#999999", "third": "#009E73", "grey": "#666666"}
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "legend.frameon": False, "figure.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "axes.unicode_minus": False,
    "font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True})
OUT = ROOT / "agentic/results/analysis"; OUT.mkdir(parents=True,
                                                  exist_ok=True)
RES = ROOT / "agentic/results"
ARMS = {"dl_only": ("CNN-GRU tool", C["tool"]),
        "P7_agent_dl": ("agent + tool", C["agent"]),
        "P7_agent": ("agent (no tool)", C["agent0"]),
        "b0_median": ("precedent median", C["b0"])}


def main():
    m = pd.read_csv(RES / "final_prognostic/forecast_metrics.csv")
    err = m.pivot(index="qid", columns="arm", values="rul_abs_err")
    true = m.pivot(index="qid", columns="arm",
                   values="true_rul")["dl_only"]
    hint = m.pivot(index="qid", columns="arm",
                   values="dl_hint")["P7_agent_dl"]
    unit = m.pivot(index="qid", columns="arm", values="unit")["dl_only"]
    eps = {e["qid"]: e for e in map(json.loads, open(
        RES / "final_prognostic/forecast_episodes.jsonl"))
        if e["arm"] == "P7_agent_dl"}
    pf = PrecedentFutures(ROOT / "agentic/data/vector_store/meta.jsonl")
    rel = pd.Series({q: reliability(e.get("contexts"))["value"]
                     for q, e in eps.items()})
    unc = pd.Series({q: future_progression(
        pf, e.get("contexts"))["aggregate"]["uncertainty"]
        for q, e in eps.items()})

    S = ["# MAE analysis (final_prognostic, 89 anomalies)\n"]
    S.append("## Overall MAE per arm")
    for a, (lab, _) in ARMS.items():
        S.append(f"{lab:>18s}: {err[a].mean():5.1f}  "
                 f"(answered {int(err[a].notna().sum())}/89)")

    bands = [(0, 20, "<20"), (20, 60, "20-60"), (60, 125, "60-124"),
             (125, 126, "cap")]
    S.append("\n## MAE by true-RUL band x arm")
    hdr = f"{'band':>7s} {'n':>3s}" + "".join(
        f"{lab:>18s}" for lab, _ in ARMS.values())
    S.append(hdr)
    for lo, hi, lab in bands:
        k = (true >= lo) & (true < hi)
        row = f"{lab:>7s} {int(k.sum()):>3d}"
        for a in ARMS:
            row += f"{err[a][k].mean():>18.1f}"
        S.append(row)

    corr = hint == 0
    S.append(f"\n## The rescue, quantified (corrupted tool: hint==0, "
             f"n={int(corr.sum())} | clean hint, n={int((~corr).sum())})")
    for name, k in (("corrupted-hint subset", corr),
                    ("clean-hint subset", ~corr)):
        S.append(f"{name}: tool MAE {err['dl_only'][k].mean():5.1f}  |  "
                 f"agent MAE {err['P7_agent_dl'][k].mean():5.1f}  |  "
                 f"paired mean delta "
                 f"{(err['P7_agent_dl'][k]-err['dl_only'][k]).mean():+5.1f}")

    S.append("\n## Signal quadrants (reliability median-split x "
             "uncertainty 0.35-split)")
    rhalf = rel >= rel.median()
    uhalf = unc >= 0.35
    for rn, rk in (("high-rel", rhalf), ("low-rel", ~rhalf)):
        for un, uk in (("low-unc", ~uhalf), ("high-unc", uhalf)):
            k = (rk & uk).reindex(err.index).fillna(False)
            S.append(f"{rn:>8s} x {un:<8s} n={int(k.sum()):2d}  tool "
                     f"{err['dl_only'][k].mean():5.1f}  agent "
                     f"{err['P7_agent_dl'][k].mean():5.1f}")

    S.append("\n## Per-unit MAE (heterogeneity)")
    for u in sorted(unit.unique()):
        k = unit == u
        S.append(f"unit {int(u):>3d} (n={int(k.sum()):2d}): tool "
                 f"{err['dl_only'][k].mean():5.1f}  agent "
                 f"{err['P7_agent_dl'][k].mean():5.1f}")
    (OUT / "mae_stats.md").write_text("\n".join(S) + "\n")
    print("\n".join(S[:28]))

    print(f"[mae_analysis] -> {OUT}")


if __name__ == "__main__":
    main()

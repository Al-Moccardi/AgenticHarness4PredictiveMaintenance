#!/usr/bin/env python3
"""band_focus.py — the 20-60 cycles band under the microscope (n=41):
(1) is agent+tool CONSERVATIVE here? signed-error evidence.
(2) reliability x uncertainty x MAE: a readable 2-panel and a 3-D view.

Outputs -> agentic/results/analysis/ (band2060_stats.md, figBAND1, figBAND2_3d)
Run from repo root:  python paper/code/stats/band_focus.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "agentic"))
from apdm.prognostic.future_progression import (reliability,          # noqa
                                                future_progression)
from apdm.prognostic.forecast import PrecedentFutures                 # noqa

MM = 1 / 25.4
C = {"tool": "#D55E00", "agent": "#0072B2", "grey": "#666666",
     "third": "#009E73"}
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "axes.grid": True, "grid.alpha": 0.25,
    "legend.frameon": False, "figure.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "axes.unicode_minus": False,
    "font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True})
OUT = ROOT / "agentic/results/analysis"; OUT.mkdir(exist_ok=True,
                                                  parents=True)
RES = ROOT / "agentic/results"


def main():
    m = pd.read_csv(RES / "final_prognostic/forecast_metrics.csv")
    err = m.pivot(index="qid", columns="arm", values="rul_err")
    aerr = m.pivot(index="qid", columns="arm", values="rul_abs_err")
    true = m.pivot(index="qid", columns="arm",
                   values="true_rul")["dl_only"]
    band = (true >= 20) & (true < 60)
    eps = {e["qid"]: e for e in map(json.loads, open(
        RES / "final_prognostic/forecast_episodes.jsonl"))
        if e["arm"] == "P7_agent_dl"}
    pf = PrecedentFutures(ROOT / "agentic/data/vector_store/meta.jsonl")
    rel = pd.Series({q: reliability(e.get("contexts"))["value"]
                     for q, e in eps.items()})
    unc = pd.Series({q: future_progression(
        pf, e.get("contexts"))["aggregate"]["uncertainty"]
        for q, e in eps.items()})
    d = pd.DataFrame({"true": true, "rel": rel, "unc": unc,
                      "tool_s": err["dl_only"],
                      "agent_s": err["P7_agent_dl"],
                      "tool_a": aerr["dl_only"],
                      "agent_a": aerr["P7_agent_dl"]})[band]

    S = [f"# The 20-60 band (n={len(d)}, agent answered "
         f"{int(d.agent_s.notna().sum())})\n",
         "## Is agent+tool conservative here? (signed error, cycles)"]
    for lab, col in (("tool", "tool_s"), ("agent+tool", "agent_s")):
        e = d[col].dropna()
        S.append(f"{lab:>10s}: bias {e.mean():+5.1f} | median "
                 f"{e.median():+5.1f} | q05 {e.quantile(.05):+5.1f} | "
                 f"q95 {e.quantile(.95):+5.1f} | under-pred "
                 f"{(e<0).mean():4.0%} | over-pred {(e>0).mean():4.0%} | "
                 f"dangerous(>+20) {(e>20).sum()}")
    sp = lambda a, b: d[a].rank().corr(d[b].rank())
    S += ["\n## Signals vs MAE inside the band (Spearman)",
          f"rho(rel, agent MAE) = {sp('rel','agent_a'):+.2f}   "
          f"rho(unc, agent MAE) = {sp('unc','agent_a'):+.2f}",
          f"rho(rel, tool  MAE) = {sp('rel','tool_a'):+.2f}   "
          f"rho(unc, tool  MAE) = {sp('unc','tool_a'):+.2f}",
          f"rho(rel, unc)       = {sp('rel','unc'):+.2f}"]
    # ---- the >60 range: type of error ----
    hi = true >= 60
    cap = true >= 125
    h = pd.DataFrame({"true": true, "tool_s": err["dl_only"],
                      "agent_s": err["P7_agent_dl"]})[hi]
    S.append(f"\n# The >60 range (n={len(h)}: 60-124 n="
             f"{int(((true>=60)&(true<125)).sum())}, cap n="
             f"{int(cap.sum())})")
    S.append("## Type of error (signed, cycles)")
    for lab, col in (("tool", "tool_s"), ("agent+tool", "agent_s")):
        for sub, k in ((">60 all", h.index),
                       ("60-124", h.index[h.true < 125]),
                       ("cap", h.index[h.true >= 125])):
            e = h.loc[k, col].dropna()
            if not len(e):
                continue
            S.append(f"{lab:>10s} {sub:>7s}: bias {e.mean():+6.1f} | "
                     f"median {e.median():+6.1f} | q05 "
                     f"{e.quantile(.05):+6.1f} | q95 "
                     f"{e.quantile(.95):+6.1f} | under "
                     f"{(e<0).mean():4.0%} | over {(e>0).mean():4.0%} |"
                     f" dangerous(>+20) {(e>20).sum()}")
    (OUT / "band2060_stats.md").write_text("\n".join(S) + "\n")
    print("\n".join(S))

    # ---- figBAND1: reliability x uncertainty heatmap of agent MAE
    fig, a2 = plt.subplots(figsize=(100 * MM, 78 * MM))
    re_ = pd.qcut(d.rel, 3, labels=["low", "mid", "high"])
    ue_ = pd.cut(d.unc, [0, .35, .55, 1.0],
                 labels=["low", "mid", "high"])
    piv = d.assign(R=re_, U=ue_).pivot_table(
        index="U", columns="R", values="agent_a", aggfunc="median",
        observed=False)
    npiv = d.assign(R=re_, U=ue_).pivot_table(
        index="U", columns="R", values="agent_a", aggfunc="count",
        observed=False)
    im = a2.imshow(piv.to_numpy(), cmap="viridis", origin="lower",
                   aspect="auto")
    a2.set_xticks(range(3), piv.columns)
    a2.set_yticks(range(3), piv.index)
    a2.set_xlabel("reliability tertile")
    a2.set_ylabel("uncertainty level")
    a2.set_title("median agent MAE by signal cell (20-60 band)")
    for i2 in range(piv.shape[0]):
        for j2 in range(piv.shape[1]):
            v, n = piv.iloc[i2, j2], npiv.iloc[i2, j2]
            if pd.notna(v):
                a2.text(j2, i2, f"{v:.0f}\n(n={int(n)})", ha="center",
                        va="center", color="white", fontsize=7.5)
    fig.colorbar(im, ax=a2, shrink=0.85, label="MAE (cycles)")
    a2.grid(False)
    fig.savefig(OUT / "figBAND1_signal_heatmap.png")
    fig.savefig(OUT / "figBAND1_signal_heatmap.pdf")
    plt.close(fig)

    # ---- figBAND2: the fancy 3-D view
    dd = d.dropna(subset=["rel", "unc", "agent_a"])
    fig = plt.figure(figsize=(150 * MM, 110 * MM))
    ax = fig.add_subplot(111, projection="3d")
    try:
        ax.plot_trisurf(dd.rel, dd.unc, dd.agent_a, cmap="viridis",
                        alpha=0.45, linewidth=0.1, edgecolor="none")
    except Exception:
        pass
    p = ax.scatter(dd.rel, dd.unc, dd.agent_a, c=dd.agent_a,
                   cmap="viridis", s=34, depthshade=True,
                   edgecolor="k", linewidth=0.3)
    for _, r in dd.iterrows():
        ax.plot([r.rel, r.rel], [r.unc, r.unc], [0, r.agent_a],
                color=C["grey"], lw=0.4, alpha=0.5)
    ax.set_xlabel("reliability", labelpad=6)
    ax.set_ylabel("uncertainty", labelpad=6)
    ax.set_zlabel("agent MAE (cycles)", labelpad=8)
    ax.set_title("reliability $\\times$ uncertainty $\\times$ MAE, "
                 f"20-60 band (n={len(dd)})", pad=12)
    ax.view_init(elev=22, azim=-58)
    fig.colorbar(p, ax=ax, shrink=0.5, pad=0.12,
                 label="MAE (cycles)")
    fig.savefig(OUT / "figBAND2_3d.png")
    fig.savefig(OUT / "figBAND2_3d.pdf")
    plt.close(fig)
    print(f"[band_focus] -> {OUT}")


if __name__ == "__main__":
    main()

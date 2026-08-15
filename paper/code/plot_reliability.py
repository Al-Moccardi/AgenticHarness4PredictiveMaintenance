#!/usr/bin/env python3
"""plot_reliability.py — reliability (similarity to the knowledge base)
vs true RUL and vs absolute error. Works on any bench out dir that has
forecast_episodes.jsonl + forecast_metrics.csv.

  python paper/code/plot_reliability.py                       (published run)
  python paper/code/plot_reliability.py --dir agentic/results/arad3/prog_final
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve()
while not (ROOT / "agentic" / "apdm").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "agentic"))
from apdm.prognostic.future_progression import reliability  # noqa: E402

MM = 1 / 25.4
C = {"main": "#0072B2", "alt": "#D55E00", "grey": "#666666",
     "third": "#009E73"}
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "legend.frameon": False, "figure.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "axes.unicode_minus": False,
    "font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True})


def sp(a, b):
    a, b = pd.Series(a), pd.Series(b)
    m = a.notna() & b.notna()
    return float(a[m].rank().corr(b[m].rank()))


def binned(x, y, edges):
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi) & y.notna()
        if m.sum() >= 3:
            xs.append((lo + hi) / 2)
            ys.append(float(y[m].median()))
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "agentic/results/final_prognostic"))
    a = ap.parse_args()
    d = Path(a.dir)
    if not (d / "forecast_episodes.jsonl").exists():
        raise SystemExit(f"no episodes in {d} - run the prognostic bench first")
    eps = [json.loads(l) for l in open(d / "forecast_episodes.jsonl")]
    eps = [e for e in eps if e["arm"] == "P7_agent_dl"]
    rel = pd.Series({e["qid"]: (reliability(e.get("contexts")) or
                                {}).get("value") for e in eps}, name="rel")
    m = pd.read_csv(d / "forecast_metrics.csv")
    tool = m[m.arm == "dl_only"].set_index("qid")
    ag = m[m.arm == "P7_agent_dl"].set_index("qid")
    df = pd.DataFrame({"rel": rel})
    df["true"] = tool["true_rul"]
    df["tool_err"] = tool["rul_abs_err"]
    df["agent_err"] = ag["rul_abs_err"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(190 * MM * 0.98, 62 * MM))
    # A: reliability vs true RUL
    ax1.scatter(df["true"], df["rel"], s=14, color=C["main"], alpha=0.55)
    xs, ys = binned(df["true"], df["rel"], np.linspace(0, 130, 8))
    ax1.plot(xs, ys, color=C["main"], lw=2, label="binned median")
    ax1.set_xlabel("true RUL (cycles, capped at 125)")
    ax1.set_ylabel("reliability (mean similarity to KB)")
    ax1.set_title("(a) reliability vs remaining life")
    ax1.text(0.03, 0.05, rf"$\rho$ = {sp(df['rel'], df['true']):+.2f}",
             transform=ax1.transAxes, fontsize=8, color=C["grey"])
    ax1.legend(loc="upper right")
    # B: reliability vs absolute error
    ax2.scatter(df["rel"], df["tool_err"], s=14, color=C["alt"], alpha=0.5,
                label=rf"CNN-GRU tool  ($\rho$={sp(df['rel'], df['tool_err']):+.2f})")
    ax2.scatter(df["rel"], df["agent_err"], s=14, color=C["main"],
                alpha=0.5, label=rf"agent  ($\rho$={sp(df['rel'], df['agent_err']):+.2f})")
    e = np.linspace(df["rel"].min(), df["rel"].max(), 6)
    for col, key in ((C["alt"], "tool_err"), (C["main"], "agent_err")):
        xs, ys = binned(df["rel"], df[key], e)
        ax2.plot(xs, ys, color=col, lw=2)
    ax2.set_xlabel("reliability (mean similarity to KB)")
    ax2.set_ylabel("absolute RUL error (cycles)")
    ax2.set_title("(b) reliability vs error")
    ax2.legend(loc="upper left")
    out = ROOT / "paper" / "figures_regen" / "reliability"
    out.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(out / "reliability_scores.csv")
    fig.savefig(out / "figREL_reliability.png")
    fig.savefig(out / "figREL_reliability.pdf")
    print(f"[reliability] n={df.rel.notna().sum()} | "
          f"range [{df.rel.min():.3f}, {df.rel.max():.3f}] | "
          f"rho(rel,true) {sp(df['rel'], df['true']):+.2f} | "
          f"rho(rel,tool_err) {sp(df['rel'], df['tool_err']):+.2f} | "
          f"rho(rel,agent_err) {sp(df['rel'], df['agent_err']):+.2f}")
    print(f"-> {out}/figREL_reliability.png")


if __name__ == "__main__":
    main()

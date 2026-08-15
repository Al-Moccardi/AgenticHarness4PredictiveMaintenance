#!/usr/bin/env python3
"""
paper_figures_panels.py — every curated panel as a STANDALONE figure
(one PNG each), with full-size axes so text, legends and data never
overlap. Filenames map to the composite set:

  fig1a_bias_hist            fig1b_retrieval_fix     fig1c_redesign_slopes
  fig2_patterns
  fig3a_rescue_hero          fig3b_agent_vs_hint     fig3c_tool_vs_agent
  fig4a_paired_forest        fig4b_perunit_dumbbell  fig4c_sscore_ecdf
  fig5a_grounding            fig5b_escalation        fig5c_severity_validity
  fig6a_outlook_confusion    fig6b_trend_chance
  figS1_trajectories (supplement strip)

Run:
  python paper\\code\\figures\\paper_figures_panels.py ^
      --arad results\\arad3 --arad2 results\\arad2 --v1grid results\\patterns
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
_r = Path(__file__).resolve()
while not (_r / 'agentic' / 'apdm').exists():
    _r = _r.parent
_DIAG_DIR = _r / 'agentic/results/final_diagnostic'


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MM = 1 / 25.4
W1, W2 = 90 * MM, 190 * MM
C = {"main": "#0072B2", "alt": "#D55E00", "third": "#009E73",
     "accent": "#CC79A7", "grey": "#666666", "light": "#BBBBBB",
     "gold": "#E69F00"}
def _fonts():
    base = {"font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
            "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
            "legend.fontsize": 7, "axes.linewidth": 0.6,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "grid.alpha": 0.25,
            "legend.frameon": False, "figure.dpi": 300,
            "savefig.bbox": "tight", "pdf.fonttype": 42,
            "axes.unicode_minus": False}
    try:
        plt.rcParams.update({**base, "text.usetex": True,
                             "font.family": "serif"})
        f, a = plt.subplots(figsize=(1, 1)); a.set_title(r"$\rho$ 95\%")
        f.savefig("/tmp/_texprobe2.png"); plt.close(f)
    except Exception:
        plt.rcParams.update({**base, "text.usetex": False,
                             "font.family": "serif",
                             "font.serif": ["cmr10", "DejaVu Serif"],
                             "mathtext.fontset": "cm",
                             "axes.formatter.use_mathtext": True})
_fonts()
USETEX = plt.rcParams.get("text.usetex", False)
RNG = np.random.default_rng(42)
BANDS = [(120, "continue_monitoring"), (60, "schedule_inspection"),
         (25, "plan_maintenance"), (-1, "immediate_shutdown")]
SEV2ACT = {1: "continue_monitoring", 2: "continue_monitoring",
           3: "schedule_inspection", 4: "plan_maintenance",
           5: "immediate_shutdown"}
HKW = re.compile(r"(previous|prior|progression|recurring|earlier|history|"
                 r"count|past|first anomal|accelerat)", re.I)


def band(r):
    for t, a_ in BANDS:
        if r > t:
            return a_
    return "immediate_shutdown"


def jread(p: Path):
    return [json.loads(l) for l in
            p.read_text(encoding="utf-8", errors="ignore").splitlines()
            if l.strip()]


def save(fig, out, name):
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)
    print(f"  [panel] {name}")


def boot(v, n=3000):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    bs = [v[RNG.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return v.mean(), *np.percentile(bs, [2.5, 97.5])


def diag_frame(dir_: Path) -> pd.DataFrame:
    rows = []
    ep = dir_ / "episodes.jsonl"
    if not ep.exists():
        ep = dir_ / "diag" / "episodes.jsonl"
    for e in jread(ep):
        t = e.get("ticket") or {}
        ctx = e.get("contexts") or []
        cited = t.get("cited_precedents") or []
        shown = [f"u{c['unit']}c{c['cycle']}" for c in ctx]
        cr = [c["rul_then"] for c in ctx
              if f"u{c['unit']}c{c['cycle']}" in cited
              and c.get("rul_then") is not None]
        allr = [c["rul_then"] for c in ctx if c.get("rul_then") is not None]
        est = t.get("rul_estimate")
        sev = t.get("severity")
        rows.append(dict(
            pattern=e["pattern"], true=e.get("true_rul"), est=est, sev=sev,
            cited_max=(max(cr) if cr else
                       (max(allr) if allr else np.nan)),
            cit=(float(all(c in shown for c in cited)) if cited
                 else (1.0 if not ctx else 0.0)),
            band_ok=(float(t.get("action") == band(est))
                     if est is not None and t.get("action") else np.nan),
            coher=(float(t.get("action") == SEV2ACT.get(sev))
                   if sev is not None else np.nan),
            hist_gr=(float(bool(HKW.search(
                str(t.get("matched_pattern", "")) + " "
                + str(t.get("diagnosis", "")))))
                if e.get("has_history") else np.nan),
            esc=1.0 if e.get("escalated") else 0.0,
            ac=1.0 if e.get("auto_corrected") else 0.0))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arad", default="agentic/results/final_prognostic")
    ap.add_argument("--arad2", default="agentic/results/v2_rul_coupled_collapse")
    ap.add_argument("--v1grid", default="agentic/results/study_A_pattern_grid")
    ap.add_argument("--out", default=str(_r / "paper/figures_regen/panels"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    A, A2, V1 = Path(a.arad), Path(a.arad2), Path(a.v1grid)

    M = pd.read_csv(Path(a.arad) / "forecast_metrics.csv")
    eps = jread(Path(a.arad) / "forecast_episodes.jsonl")
    hint = {(e["qid"], e["arm"]): e.get("dl_hint") for e in eps}
    M["hint"] = [hint.get((q, ar)) for q, ar in zip(M.qid, M.arm)]
    D3 = diag_frame(_DIAG_DIR)
    has2 = (A2 / "diag" / "episodes.jsonl").exists()
    D2 = diag_frame(A2) if has2 else None

    # ---------------- fig1a — bias histograms --------------------------
    if has2:
        S2 = D2[D2.pattern.isin(["P2_rag", "P5_verifier"])].copy()
        S2["tc"] = S2.true.clip(upper=125)
        lo = S2[S2.tc < 60]
        hi = S2[S2.tc >= 100]
        fig, ax = plt.subplots(figsize=(W1 * 1.25, 0.75 * W1 * 1.25))
        bins = np.linspace(0, 130, 27)
        ax.hist(hi.cited_max.dropna(), bins=bins, alpha=0.65,
                color=C["main"],
                label=f"cases with true RUL $\\geq$ 100 (n={len(hi)})")
        ax.hist(lo.cited_max.dropna(), bins=bins, alpha=0.65,
                color=C["alt"],
                label=f"cases with true RUL $<$ 60 (n={len(lo)})")
        ax.set_xlabel("cited precedent outcome (max rul_then, cycles)")
        ax.set_ylabel("cases")
        ax.legend(loc="upper right")
        ax.set_title("Cosine retrieval: cited outcomes ignore the case",
                     loc="left", pad=6)
        save(fig, out, "fig1a_bias_hist")

        # ------------ fig1b — the retrieval fix ------------------------
        S3 = D3[D3.pattern.isin(["P2_rag", "P5_verifier"])].copy()
        S3["tc"] = S3.true.clip(upper=125)
        fig, ax = plt.subplots(figsize=(W1 * 1.35, 0.80 * W1 * 1.35))
        ax.scatter(S2.tc + RNG.normal(0, 1.2, len(S2)), S2.cited_max,
                   s=11, alpha=0.55, color=C["alt"],
                   label="plain cosine (v2)")
        ax.scatter(S3.tc + RNG.normal(0, 1.2, len(S3)), S3.cited_max,
                   s=11, alpha=0.55, color=C["main"],
                   label="life-stage-aware (v3)")
        for D_, cc in ((S2, C["alt"]), (S3, C["main"])):
            mb = (D_.assign(b=pd.cut(D_.tc, [0, 25, 60, 100, 126]))
                  .groupby("b", observed=True).cited_max.median())
            ax.plot([12, 42, 80, 113][:len(mb)], mb.values, color=cc,
                    lw=1.6, marker="o", ms=3.4)
        r2 = S2[["tc", "cited_max"]].corr(method="spearman").iloc[0, 1]
        r3 = S3[["tc", "cited_max"]].corr(method="spearman").iloc[0, 1]
        ax.text(0.02, 0.97,
                f"$\\rho$(cited outcome, true RUL): "
                f"{r2:.2f} $\\to$ {r3:.2f}",
                transform=ax.transAxes, fontsize=8, va="top")
        ax.set_xlabel("true RUL of the monitored case (cycles)")
        ax.set_ylabel("cited outcome (max rul_then, cycles)")
        ax.set_ylim(-5, 132)
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02),
                  ncols=2, borderaxespad=0)
        ax.set_title("Stage-aware retrieval restores case relevance",
                     loc="left", pad=26)
        save(fig, out, "fig1b_retrieval_fix")

        # ------------ fig1c — redesign slopes --------------------------
        p5o = D2.query("pattern=='P5_verifier'")
        p5n = D3.query("pattern=='P5_verifier'")
        rows = [("escalation", p5o.esc.mean(), p5n.esc.mean()),
                ("action coherence", p5o.band_ok.mean(), p5n.coher.mean()),
                ("history grounded", np.nan, p5n.hist_gr.mean()),
                ("auto-corrected", 0.0, p5n.ac.mean())]
        fig, ax = plt.subplots(figsize=(W1 * 1.35, 0.62 * W1 * 1.35))
        for i, (lab, v2, v3) in enumerate(rows):
            y = len(rows) - 1 - i
            if not np.isnan(v2):
                ax.plot([v2, v3], [y, y], color=C["light"], lw=1.8,
                        zorder=1)
                ax.scatter([v2], [y], s=30, color=C["grey"], zorder=2,
                           label="RUL-coupled (v2)" if i == 0 else None)
            ax.scatter([v3], [y], s=36, color=C["main"], zorder=3,
                       label="retrospective (v3)" if i == 0 else None)
            txt = (f"{v2:.0%} $\\to$ {v3:.0%}" if not np.isnan(v2)
                   else f"{v3:.0%}")
            ax.text(1.04, y, txt, va="center", fontsize=7.5,
                    transform=ax.get_yaxis_transform())
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows][::-1])
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(-0.5, len(rows) - 0.5)
        ax.set_xlabel("rate on the A-RAD agent arm (P5)")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.16),
                  ncols=1)
        ax.set_title("What the retrospective redesign bought", loc="left",
                     pad=6)
        fig.subplots_adjust(right=0.80)
        save(fig, out, "fig1c_redesign_slopes")

    # ---------------- fig2 — patterns (single panel) --------------------
    if (V1 / "episodes.jsonl").exists():
        rows = []
        for e in jread(V1 / "episodes.jsonl"):
            t = e.get("ticket") or {}
            shown = [f"u{c['unit']}c{c['cycle']}" for c in
                     (e.get("contexts") or [])]
            cited = t.get("cited_precedents") or []
            rows.append(dict(
                pattern=e["pattern"],
                cit=(float(all(c in shown for c in cited)) if cited
                     else (1.0 if not shown else 0.0)),
                calls=e.get("llm_calls") or 0,
                tok=e.get("tokens_out") or 0))
        G = (pd.DataFrame(rows).groupby("pattern")
             .agg(cit=("cit", "mean"), calls=("calls", "mean"),
                  tok=("tok", "mean")))
        labs = {"B0_retrieval": "B0 (no LLM)", "P1_direct": "direct",
                "P2_rag": "RAG", "P3_react": "naive ReAct",
                "P4_reflexion": "reflexion", "P5_verifier": "verifier",
                "P6_specialists": "specialists"}
        offs = {"B0_retrieval": (-6, -14), "P2_rag": (8, -13),
                "P1_direct": (9, 2), "P3_react": (9, -3),
                "P4_reflexion": (10, -12), "P5_verifier": (-10, -16),
                "P6_specialists": (11, 4)}
        fig, ax = plt.subplots(figsize=(W1 * 1.45, 0.66 * W1 * 1.45))
        for p, r in G.iterrows():
            cc = (C["alt"] if p == "P3_react" else
                  C["main"] if p == "P5_verifier" else C["grey"])
            ax.scatter(r.calls, r.cit, s=34 + r.tok / 6, color=cc,
                       alpha=0.85, zorder=3)
            ax.annotate(labs.get(p, p), (r.calls, r.cit),
                        textcoords="offset points",
                        xytext=offs.get(p, (7, 5)), fontsize=7.4)
        ax.annotate("autonomy without\nverification collapses",
                    xy=(G.loc["P3_react", "calls"],
                        G.loc["P3_react", "cit"]),
                    xytext=(3.0, 0.55), fontsize=7.4, color=C["alt"],
                    arrowprops=dict(arrowstyle="->", lw=0.9,
                                    color=C["alt"]))
        ax.set_xlabel("LLM calls per case (bubble = generated tokens)")
        ax.set_ylabel("citation validity")
        ax.set_ylim(-0.07, 1.14)
        ax.set_xlim(-0.4, 5.9)
        ax.set_title("Agentic patterns at 3B: structure, not autonomy, "
                     "buys grounding", loc="left", pad=6)
        save(fig, out, "fig2_patterns")

    # ---------------- fig3a — rescue hero -------------------------------
    dl = M[M.arm == "P7_agent_dl"].copy()
    dl["corrupt_flag"] = dl.hint.fillna(0) <= 5
    hero = dl.groupby("unit").corrupt_flag.sum().idxmax()
    fig, ax = plt.subplots(figsize=(W2, 0.42 * W2))
    g = dl[dl.unit == hero].sort_values("cycle")
    tru = M[M.unit == hero].drop_duplicates("cycle").sort_values("cycle")
    bad = g[g.corrupt_flag]
    if len(bad):
        ax.axvspan(bad.cycle.min(), g.cycle.max(), color=C["alt"],
                   alpha=0.08)
        ax.text(bad.cycle.min() + 4, 3.5, "tool corrupted (hint = 0)",
                fontsize=7.4, color=C["alt"])
    ax.plot(tru.cycle, tru.true_rul.clip(upper=125), color=C["grey"],
            lw=1.5, label="true RUL (capped)")
    ax.plot(g.cycle, g.hint.fillna(0), ls="--", lw=1.2, marker="s",
            ms=3.0, color=C["alt"], label="CNN-GRU hint as seen by agent")
    ax.plot(g.cycle, g.rul_pred, lw=1.4, marker="o", ms=3.4,
            color=C["main"], label="agent estimate")
    if len(bad):
        xm = bad.cycle.iloc[len(bad) // 2]
        ym = float(bad.rul_pred.iloc[len(bad) // 2])
        ax.annotate("agent overrides the corrupted tool\n"
                    + ("(97\\% of corrupted cases)" if USETEX else "(97% of corrupted cases)"),
                    xy=(xm, ym), xytext=(xm - 130, ym + 55), fontsize=7.4,
                    arrowprops=dict(arrowstyle="->", lw=0.9,
                                    color=C["grey"]))
    ax.set_xlabel("cycle")
    ax.set_ylabel("RUL (cycles)")
    ax.set_ylim(-6, 132)
    ax.legend(ncols=3, loc="lower left", bbox_to_anchor=(0.0, 1.02),
              borderaxespad=0)
    ax.set_title(f"unit T{hero}: the tool collapses, the agent holds",
                 loc="left", pad=24)
    save(fig, out, "fig3a_rescue_hero")

    # ---------------- fig3b — agent vs hint scatter ----------------------
    fig, ax = plt.subplots(figsize=(W1 * 1.3, 0.82 * W1 * 1.3))
    sc = ax.scatter(dl.hint.fillna(0), dl.rul_pred, c=dl.rul_abs_err,
                    cmap="viridis", s=20, vmax=60)
    ax.plot([0, 125], [0, 125], color=C["light"], ls="--", lw=0.9)
    ax.text(3, 96, "override:\nhint = 0,\nagent near truth", fontsize=7.2,
            color=C["grey"])
    ax.text(70, 12, "follows the tool\nwhen plausible", fontsize=7.2,
            color=C["grey"])
    ax.set_xlabel("CNN-GRU hint (cycles)")
    ax.set_ylabel("agent estimate (cycles)")
    plt.colorbar(sc, ax=ax, label="|error| (cycles)", shrink=0.9)
    ax.set_title("Selective tool-following", loc="left", pad=6)
    save(fig, out, "fig3b_agent_vs_hint")

    # ---------------- fig3c — tool vs agent, same cases ------------------
    fig, ax = plt.subplots(figsize=(W1 * 1.2, 0.78 * W1 * 1.2))
    for i, flag in enumerate([False, True]):
        gg = dl[dl.corrupt_flag == flag]
        te = np.abs(gg.hint.fillna(0) - gg.true_rul.clip(upper=125))
        tm, tlo, thi = boot(te)
        am, alo, ahi = boot(gg.rul_abs_err)
        ax.bar(i - 0.19, tm, width=0.36, color=C["alt"],
               label="tool as seen by agent" if i == 0 else None)
        ax.errorbar(i - 0.19, tm, yerr=[[tm - tlo], [thi - tm]],
                    fmt="none", ecolor="black", capsize=2.6, lw=0.9)
        ax.bar(i + 0.19, am, width=0.36, color=C["main"],
               label="agent" if i == 0 else None)
        ax.errorbar(i + 0.19, am, yerr=[[am - alo], [ahi - am]],
                    fmt="none", ecolor="black", capsize=2.6, lw=0.9)
        ax.text(i - 0.19, thi + 1.6, f"{tm:.0f}", ha="center",
                fontsize=7.4)
        ax.text(i + 0.19, ahi + 1.6, f"{am:.0f}", ha="center",
                fontsize=7.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [f"tool usable\n(n={int((~dl.corrupt_flag).sum())})",
         f"tool corrupted\n(n={int(dl.corrupt_flag.sum())})"])
    ax.set_ylabel("MAE (cycles)")
    ax.set_ylim(0, 48)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncols=2,
              borderaxespad=0)
    ax.set_title("Same cases, tool vs agent", loc="left", pad=24)
    save(fig, out, "fig3c_tool_vs_agent")

    # ---------------- fig4a — paired forest ------------------------------
    pairs = [("P7_agent_dl", "dl_only", "agent+tool vs tool"),
             ("P7_agent_dl", "P7_agent", "value of the DL tool"),
             ("P7_agent_dl", "b0_median", "agent+tool vs floor")]
    fig, ax = plt.subplots(figsize=(W1 * 1.35, 0.55 * W1 * 1.35))
    for j, (x1, x2, lab) in enumerate(pairs):
        x = M[M.arm == x1].set_index("qid").rul_abs_err
        y = M[M.arm == x2].set_index("qid").rul_abs_err
        d = (x - y).dropna().values
        bs = [d[RNG.integers(0, len(d), len(d))].mean()
              for _ in range(3000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        yj = len(pairs) - 1 - j
        ax.errorbar(d.mean(), yj, xerr=[[d.mean() - lo], [hi - d.mean()]],
                    fmt="o", ms=5, color=C["main"], capsize=3.4, lw=1.2)
        ax.text(d.mean(), yj - 0.22, f"{d.mean():+.1f} "
                f"[{lo:+.1f}, {hi:+.1f}]", ha="center", va="top",
                fontsize=7.2, color=C["grey"])
    ax.axvline(0, color=C["light"], lw=0.9, ls="--")
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([p[2] for p in pairs][::-1])
    ax.set_ylim(-0.75, len(pairs) - 0.4)
    ax.set_xlim(-30, 30)
    ax.set_xlabel(r"paired $\Delta$|err| (cycles; $<0$ favours the agent)")
    ax.set_title("Paired per-anomaly bootstrap (n=89)", loc="left", pad=6)
    save(fig, out, "fig4a_paired_forest")

    # ---------------- fig4b — per-unit dumbbell --------------------------
    pu = (M.pivot_table(index="unit", columns="arm",
                        values="rul_abs_err", aggfunc="mean")
          [["dl_only", "P7_agent_dl"]].dropna())
    order = pu.dl_only.sort_values().index
    fig, ax = plt.subplots(figsize=(W1 * 1.25, 0.70 * W1 * 1.25))
    for y, u in enumerate(order):
        d0, d1 = pu.loc[u, "dl_only"], pu.loc[u, "P7_agent_dl"]
        ax.plot([d0, d1], [y, y], color=C["light"], lw=1.7, zorder=1)
        ax.scatter([d0], [y], s=28, color=C["alt"], zorder=2,
                   label="CNN-GRU alone" if y == 0 else None)
        ax.scatter([d1], [y], s=32, color=C["main"], zorder=3,
                   label="Guarded Agent" if y == 0 else None)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"T{u}" for u in order])
    ax.set_ylim(-0.6, len(order) - 0.2)
    ax.set_xlabel("per-unit MAE (cycles)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncols=2,
              borderaxespad=0)
    ax.set_title("Where each wins", loc="left", pad=24)
    save(fig, out, "fig4b_perunit_dumbbell")

    # ---------------- fig4c — S-score ECDF -------------------------------
    ARMC = {"b0_median": C["grey"], "dl_only": C["alt"],
            "P7_agent": C["accent"], "P7_agent_dl": C["main"]}
    ARML = {"b0_median": "b0 (stage-matched median)",
            "dl_only": "CNN-GRU alone", "P7_agent": "agent, no tool",
            "P7_agent_dl": "Guarded Agent"}
    fig, ax = plt.subplots(figsize=(W1 * 1.3, 0.72 * W1 * 1.3))
    for arm in ["b0_median", "dl_only", "P7_agent", "P7_agent_dl"]:
        v = np.sort(M[M.arm == arm].s_score.dropna())
        ax.plot(v, np.linspace(0, 1, len(v)), color=ARMC[arm], lw=1.4,
                label=ARML[arm])
    ax.set_xscale("log")
    ax.set_xlabel("PHM08 S-score (log scale)")
    ax.set_ylabel("ECDF")
    ax.legend(loc="lower right")
    ax.set_title("Asymmetric safety penalty, full distribution",
                 loc="left", pad=6)
    save(fig, out, "fig4c_sscore_ecdf")

    # ---------------- fig5a — grounding bars -----------------------------
    d3 = D3.groupby("pattern").agg(
        cit=("cit", "mean"), coher=("coher", "mean"),
        hist_gr=("hist_gr", "mean"), esc=("esc", "mean"),
        ac=("ac", "mean")).round(3)
    pats = ["B0_retrieval", "P1_direct", "P2_rag", "P5_verifier"]
    labs = ["B0", "direct", "RAG", "A-RAD agent"]
    x = np.arange(len(pats))
    fig, ax = plt.subplots(figsize=(W1 * 1.35, 0.62 * W1 * 1.35))
    for off, col, cc, lab in ((-0.27, "cit", C["main"],
                               "citation validity"),
                              (0.0, "coher", C["third"],
                               "severity-action consistency"),
                              (0.27, "hist_gr", C["gold"],
                               "history-grounded")):
        vals = [d3.loc[p, col] for p in pats]
        ax.bar(x + off, vals, width=0.25, color=cc, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("rate")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncols=2,
              borderaxespad=0, fontsize=6.6)
    ax.set_title("Deterministic grounding (89 cases per arm)",
                 loc="left", pad=28)
    save(fig, out, "fig5a_grounding")

    # ---------------- fig5b — escalation ---------------------------------
    fig, ax = plt.subplots(figsize=(W1 * 1.25, 0.62 * W1 * 1.25))
    if has2:
        e2 = D2.groupby("pattern").esc.mean()
        ax.bar(x - 0.17, [e2.get(p, 0) for p in pats], width=0.32,
               color=C["light"], label="v2 escalated")
    ax.bar(x + (0.17 if has2 else 0), [d3.loc[p, "esc"] for p in pats],
           width=0.32, color=C["main"], label="v3 escalated")
    ax.bar(x + (0.17 if has2 else 0), [d3.loc[p, "ac"] for p in pats],
           width=0.32, bottom=[d3.loc[p, "esc"] for p in pats],
           color=C["third"], label="v3 auto-corrected")
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("rate")
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))
    ax.set_title("Escalation collapses under the retrospective design",
                 loc="left", pad=6)
    save(fig, out, "fig5b_escalation")

    # ---------------- fig5c — severity validity --------------------------
    from scipy import stats as st
    fig, ax = plt.subplots(figsize=(W1 * 1.2, 0.62 * W1 * 1.2))
    rhos = []
    for p in pats:
        g = D3[D3.pattern == p].dropna(subset=["sev"])
        g = g.assign(tc=g.true.clip(upper=125))
        r, pv = (st.spearmanr(g.sev, g.tc) if g.sev.nunique() > 1
                 else (np.nan, 1))
        rhos.append((r, pv))
    ax.bar(x, [r for r, _ in rhos],
           color=[C["main"] if pv < 0.05 else C["light"]
                  for _, pv in rhos])
    for i, (r, pv) in enumerate(rhos):
        ax.text(i, r - 0.02, "$p<0.05$" if pv < 0.05 else "n.s.",
                ha="center", va="top", fontsize=7.2, color=C["grey"])
    ax.axhline(0, color=C["grey"], lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_ylim(-0.50, 0.05)
    ax.set_ylabel(r"Spearman $\rho$(severity, true RUL)")
    ax.set_title("The graded severity signal lives in retrieval",
                 loc="left", pad=6)
    save(fig, out, "fig5c_severity_validity")

    # ---------------- fig6a — outlook confusion --------------------------
    mm = M[M.arm.isin(["P7_agent", "P7_agent_dl"])]
    cm = pd.crosstab(mm.outlook_pred, mm.outlook_real)
    fig, ax = plt.subplots(figsize=(W1 * 1.15, 0.52 * W1 * 1.15))
    ax.imshow(cm.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(cm.shape[1]))
    ax.set_xticklabels(cm.columns, rotation=15)
    ax.set_yticks(range(cm.shape[0]))
    ax.set_yticklabels(cm.index)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm.values[i, j], ha="center", va="center",
                    fontsize=8.5,
                    color="white" if cm.values[i, j] > cm.values.max() / 2
                    else "black")
    ax.set_xlabel("realised outlook")
    ax.set_ylabel("predicted")
    ax.grid(False)
    ax.set_title("Outlook: base-rate collapse", loc="left", pad=6)
    save(fig, out, "fig6a_outlook_confusion")

    # ---------------- fig6b — trends vs chance ---------------------------
    fig, ax = plt.subplots(figsize=(W1 * 1.05, 0.62 * W1 * 1.05))
    acc = mm.groupby("arm").trend_direction_acc.mean()
    ax.bar([0, 1], [acc.get("P7_agent", np.nan),
                    acc.get("P7_agent_dl", np.nan)],
           color=[C["accent"], C["main"]], width=0.55)
    ax.axhline(1 / 3, color=C["alt"], ls="--", lw=1.1)
    ax.text(0.5, 1 / 3 + 0.018, "3-class chance", fontsize=7.4,
            color=C["alt"], ha="center")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["agent, no anchor", "Guarded Agent"])
    ax.set_ylabel("trend-direction accuracy")
    ax.set_ylim(0, 0.6)
    ax.set_title("Per-sensor trends $\\approx$ chance", loc="left", pad=6)
    save(fig, out, "fig6b_trend_chance")

    # ---------------- figS1 — supplement strip ---------------------------
    rich = [u for u in sorted(M.unit.unique())
            if M[(M.unit == u) & (M.arm == "P7_agent_dl")]
            .cycle.nunique() >= 5]
    fig, axes = plt.subplots(1, len(rich), figsize=(W2, 0.30 * W2),
                             sharey=True)
    for ax_, u in zip(np.atleast_1d(axes), rich):
        g = M[M.unit == u]
        tr = g.drop_duplicates("cycle").sort_values("cycle")
        ax_.plot(tr.cycle, tr.true_rul.clip(upper=125), color=C["grey"],
                 lw=1.3, label="true RUL (capped)")
        gg = g[g.arm == "dl_only"].sort_values("cycle")
        ax_.plot(gg.cycle, gg.rul_pred, ls="--", lw=1.0, marker="s",
                 ms=2.4, color=C["alt"], label="CNN-GRU alone")
        gg = g[g.arm == "P7_agent_dl"].sort_values("cycle")
        ax_.plot(gg.cycle, gg.rul_pred, lw=1.1, marker="o", ms=2.6,
                 color=C["main"], label="Guarded Agent")
        ax_.set_title(f"unit T{u}", loc="left", pad=2)
        ax_.set_xlabel("cycle")
    np.atleast_1d(axes)[0].set_ylabel("RUL (cycles)")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, ncols=3, loc="lower center",
               bbox_to_anchor=(0.5, 1.00))
    fig.tight_layout(w_pad=1.6)
    save(fig, out, "figS1_trajectories")

    print(f"[panels] all standalone panels -> {out}")


if __name__ == "__main__":
    main()

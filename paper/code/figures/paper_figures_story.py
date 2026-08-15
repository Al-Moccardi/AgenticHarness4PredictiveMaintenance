#!/usr/bin/env python3
"""
paper_figures_story.py — narrative figures:
  figE0_experiment_map   the whole campaign, start to finish, one card
  figCMP1_rag_vs_agent   classical RAG vs Guarded agent (deterministic)
  figCMP2_rag_vs_agent_ragas  same comparison on judged RAGAS subset
  figCMP3_capability_card what RAG cannot do (capability delta)

Run:
  python paper\\code\\figures\\paper_figures_story.py ^
      --arad results\\arad3 --v1grid results\\patterns
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
     "grey": "#666666", "light": "#BBBBBB", "gold": "#E69F00",
     "red": "#B22222"}


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
        f, a = plt.subplots(figsize=(1, 1))
        a.set_title(r"$\rho$ 95\%")
        f.savefig("/tmp/_texprobe6.png")
        plt.close(f)
        return True
    except Exception:
        plt.rcParams.update({**base, "text.usetex": False,
                             "font.family": "serif",
                             "font.serif": ["cmr10", "DejaVu Serif"],
                             "mathtext.fontset": "cm"})
        return False


USETEX = _fonts()
SEV2ACT = {1: "continue_monitoring", 2: "continue_monitoring",
           3: "schedule_inspection", 4: "plan_maintenance",
           5: "immediate_shutdown"}
HKW = re.compile(r"(previous|prior|progression|recurring|earlier|history|"
                 r"count|past|first anomal|accelerat)", re.I)
EXRC = {"text.usetex": False, "font.family": "serif",
        "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif"}


def pct(x, dec=0):
    s = f"{100 * x:.{dec}f}"
    return (s + r"\%") if USETEX else (s + "%")


def jread(p: Path):
    return [json.loads(l) for l in
            p.read_text(encoding="utf-8", errors="ignore").splitlines()
            if l.strip()]


def save(fig, out, name):
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)
    print(f"  [story] {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arad", default="agentic/results/final_prognostic")
    ap.add_argument("--v1grid", default="agentic/results/study_A_pattern_grid")
    ap.add_argument("--out", default=str(_r / "paper/figures_regen/story"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    A, V1 = Path(a.arad), Path(a.v1grid)
    dg = jread(_DIAG_DIR / "episodes.jsonl")

    def arm_stats(patt):
        g = [e for e in dg if e["pattern"] == patt]
        def cit(e):
            t = e.get("ticket") or {}
            shown = [f"u{c['unit']}c{c['cycle']}" for c in
                     (e.get("contexts") or [])]
            cited = t.get("cited_precedents") or []
            return (float(all(c in shown for c in cited)) if cited
                    else (1.0 if not shown else 0.0))
        cits = np.mean([cit(e) for e in g])
        coh = np.mean([float((e.get("ticket") or {}).get("action")
                       == SEV2ACT.get((e.get("ticket") or {})
                                      .get("severity"))) for e in g])
        hist = np.mean([float(bool(HKW.search(
            str((e.get("ticket") or {}).get("matched_pattern", ""))
            + " " + str((e.get("ticket") or {}).get("diagnosis", ""))))
            ) for e in g if e.get("has_history")])
        rep = np.mean([(e.get("repairs") or 0) for e in g])
        ac = np.mean([bool(e.get("auto_corrected")) for e in g])
        esc = np.mean([bool(e.get("escalated")) for e in g])
        return dict(cit=cits, coh=coh, hist=hist, rep=rep, ac=ac,
                    esc=esc, n=len(g))
    P2 = arm_stats("P2_rag")
    P5 = arm_stats("P5_verifier")

    # ---------------- figCMP1 — deterministic comparison ---------------
    fig, ax = plt.subplots(figsize=(W1 * 1.5, 0.6 * W1 * 1.5))
    names = ["citation\nvalidity", "severity-action\nconsistency",
             "history\ngrounding", "delivered\nquality"]
    ragv = [P2["cit"], P2["coh"], P2["hist"], P2["coh"]]
    agv = [P5["cit"], P5["coh"], P5["hist"],
           P5["coh"] / (1 - P5["esc"])]
    x = np.arange(len(names))
    ax.bar(x - 0.18, ragv, width=0.34, color=C["grey"],
           label="classical RAG (single-shot)")
    ax.bar(x + 0.18, agv, width=0.34, color=C["main"],
           label="Guarded agent (tools + verifier)")
    for xi, (rv, av) in zip(x, zip(ragv, agv)):
        ax.text(xi - 0.18, rv + 0.015, f"{rv:.2f}", ha="center",
                fontsize=6.6)
        ax.text(xi + 0.18, av + 0.015, f"{av:.2f}", ha="center",
                fontsize=6.6)
        ax.annotate("", xy=(xi + 0.18, min(av, 1.06)),
                    xytext=(xi - 0.18, rv),
                    arrowprops=dict(arrowstyle="->", lw=0.7,
                                    color=C["third"], alpha=0.7))
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylim(0, 1.14)
    ax.set_ylabel("rate (89 cases each)")
    ax.legend(ncols=2, fontsize=6.6, loc="lower left",
              bbox_to_anchor=(0.0, 1.02), borderaxespad=0)
    ax.set_title("Classical RAG vs the Guarded diagnostic agent",
                 loc="left", pad=24)
    save(fig, out, "figCMP1_rag_vs_agent")

    # ---------------- figCMP2 — RAGAS comparison (audited) -------------
    mj = V1 / "metrics.jsonl"
    if mj.exists():
        R = pd.DataFrame(jread(mj))
        R = R[R.faithfulness.notna()
              & R.pattern.isin(["P2_rag", "P5_verifier"])]
        fig, ax = plt.subplots(figsize=(W1 * 1.4, 0.58 * W1 * 1.4))
        mets = [("faithfulness", "faithfulness"),
                ("answer_relevancy", "answer\nrelevancy"),
                ("context_precision", "context\nprecision")]
        x = np.arange(len(mets))
        for off, patt, cc, lab in (
                (-0.18, "P2_rag", C["grey"], "classical RAG"),
                (0.18, "P5_verifier", C["main"], "Guarded agent")):
            g = R[R.pattern == patt]
            vals = [g[m].mean() for m, _ in mets]
            ax.bar(x + off, vals, width=0.34, color=cc,
                   label=f"{lab} (n={len(g)})")
            for xi, v in zip(x + off, vals):
                ax.text(xi, v + 0.015, f"{v:.2f}", ha="center",
                        fontsize=6.8)
        ax.set_xticks(x)
        ax.set_xticklabels([n for _, n in mets], fontsize=7.2)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("RAGAS score (llama3.1:8b judge)")
        ax.legend(ncols=2, fontsize=6.6, loc="lower left",
                  bbox_to_anchor=(0.0, 1.02), borderaxespad=0)
        ax.set_title("RAGAS on the audited subset: RAG vs agent",
                     loc="left", pad=24)
        save(fig, out, "figCMP2_rag_vs_agent_ragas")

    # ---------------- figCMP3 — capability delta card -------------------
    rows = [
        ("retrieved precedents in context", "yes", "yes"),
        ("precedent-history tool (read_history)", "-", "yes"),
        ("deterministic progression comparison", "-", "yes"),
        ("citation gate (D4)", "-", "yes"),
        ("severity-action contract gate (D5)", "-", "yes"),
        ("history-reference gate (D6)", "-", "yes"),
        ("deterministic repair rounds", "-",
         f"{P5['rep']:.2f} per case"),
        ("auto-correction", "-", pct(P5["ac"], 1)),
        ("escalation channel to human", "-", pct(P5["esc"], 1)),
    ]
    with plt.rc_context(EXRC):
        fig, ax = plt.subplots(figsize=(W2 * 0.72, 0.40 * W2))
        ax.axis("off")
        ax.text(0, 1.02, "What single-shot RAG cannot do", fontsize=10.5,
                fontweight="bold", va="top")
        ax.text(0.52, 0.92, "classical RAG", fontsize=8, va="top",
                color=C["grey"], fontweight="bold")
        ax.text(0.76, 0.92, "Guarded agent", fontsize=8, va="top",
                color=C["main"], fontweight="bold")
        y = 0.82
        for name, rv, av in rows:
            ax.text(0.0, y, name, fontsize=8, va="top")
            ax.text(0.545, y, rv, fontsize=8, va="top",
                    color=C["light"] if rv == "-" else "black")
            ax.text(0.785, y, av.replace("\\%", "%"), fontsize=8,
                    va="top", color=C["main"])
            y -= 0.088
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        save(fig, out, "figCMP3_capability_card")

    # ---------------- figE0 — the experiment map ------------------------
    stages = [
        ("Study A - agentic pattern grid",
         "7 patterns x 3 styles, 1,691 episodes on FD002",
         "structure, not autonomy, buys grounding; naive ReAct "
         "collapses to 0.29 citation validity"),
        ("Judge audit",
         "43 episodes scored with llama3.1:8b (RAGAS)",
         "3B judges proven too lenient; the campaign adopts "
         "judge-free deterministic gates"),
        ("v2 - RUL-coupled Guarded agent",
         "356 episodes, grounding constraints on anchored RUL",
         "retrieval bias made binding: 97.8% escalation, 7.9% "
         "action coherence"),
        ("v3 - stage-aware retrospective redesign",
         "diagnostic agent: retrospective tools + tiered verifier",
         "escalation 1.1%, coherence 98.9%, history 100%, 0.46 "
         "repairs/case"),
        ("v3 - prognostic study (4 arms x 89)",
         "Guarded forecaster with CNN-GRU anchor + precedent futures",
         "0% dangerous over-predictions; rescue on corrupted tool "
         "-13.9 cycles; wins every EOL zone <100"),
        ("Cross-analyses",
         "EOL sweep, abstention ladder, selective prediction, "
         "pattern inventory",
         "abstention becomes graduated and informative: 16.4 MAE "
         "at 55% coverage via stated ranges"),
    ]
    with plt.rc_context(EXRC):
        fig, ax = plt.subplots(figsize=(W2 * 0.95, 0.52 * W2))
        ax.axis("off")
        ax.text(0, 1.02, "The A-RAD experimental campaign, start to "
                         "finish", fontsize=11, fontweight="bold",
                va="top")
        y = 0.90
        for i, (t1, t2, t3) in enumerate(stages):
            ax.text(0.0, y, f"{i + 1}.", fontsize=9.5,
                    fontweight="bold", color=C["main"], va="top")
            ax.text(0.045, y, t1, fontsize=9, fontweight="bold",
                    va="top")
            ax.text(0.045, y - 0.045, t2, fontsize=7.6,
                    color=C["grey"], va="top")
            ax.text(0.045, y - 0.086, t3, fontsize=7.9,
                    color=C["main"], va="top")
            if i < len(stages) - 1:
                ax.annotate("", xy=(0.016, y - 0.135),
                            xytext=(0.016, y - 0.045),
                            arrowprops=dict(arrowstyle="->", lw=0.9,
                                            color=C["light"]))
            y -= 0.155
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        save(fig, out, "figE0_experiment_map")

    print(f"[story] -> {out}")


if __name__ == "__main__":
    main()

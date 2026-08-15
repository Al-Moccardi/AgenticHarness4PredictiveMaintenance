"""report_patterns — aggregate the grid into tables + camera-ready figures.

  python -m apdm.report_patterns                    # results/patterns/
  python -m apdm.report_patterns --dir results/patterns_smoke
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402

MM = 1 / 25.4
W2 = 190 * MM
C = {"main": "#0072B2", "alt": "#D55E00", "third": "#009E73",
     "grey": "#666666", "accent": "#CC79A7", "light": "#BBBBBB"}
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "legend.frameon": False, "figure.dpi": 300,
    "savefig.bbox": "tight", "pdf.fonttype": 42})

RAGAS = ["faithfulness", "answer_relevancy", "context_precision"]
DET = ["json_valid", "citation_validity", "rul_in_cited_span",
       "action_band_consistency"]


def panel(ax, letter, title=""):
    ax.text(-0.16, 1.06, f"({letter})", transform=ax.transAxes,
            fontweight="bold", fontsize=9, va="bottom")
    if title:
        ax.set_title(title, loc="left", pad=3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(Path(__file__).resolve().parent
                                         .parent / "results/patterns"))
    a = ap.parse_args()
    d = Path(a.dir)
    eps = pd.DataFrame(json.loads(l) for l in
                       (d / "episodes.jsonl").read_text().splitlines()
                       if l.strip())
    mts = pd.DataFrame(json.loads(l) for l in
                       (d / "metrics.jsonl").read_text().splitlines()
                       if l.strip())
    for c_ in RAGAS + DET:
        if c_ in mts:
            mts[c_] = pd.to_numeric(mts[c_], errors="coerce")
    df = eps.merge(mts, on=["qid", "pattern", "style"], how="left")
    df["escalated"] = df.get("escalated").fillna(False).astype(bool)
    df["gate_fail"] = df.gate_violations.apply(
        lambda v: bool(v) if isinstance(v, list) else False)

    g = (df.groupby(["pattern", "style"])
         .agg(n=("qid", "size"),
              faithfulness=("faithfulness", "mean"),
              answer_relevancy=("answer_relevancy", "mean"),
              context_precision=("context_precision", "mean"),
              json_valid=("json_valid", "mean"),
              citation_validity=("citation_validity", "mean"),
              rul_in_cited_span=("rul_in_cited_span", "mean"),
              action_band=("action_band_consistency", "mean"),
              gate_fail=("gate_fail", "mean"),
              escalated=("escalated", "mean"),
              llm_calls=("llm_calls", "mean"),
              tokens_out=("tokens_out", "mean"),
              wall_s=("wall_s", "mean"))
         .reset_index().round(3))
    g.to_csv(d / "summary.csv", index=False)

    lines = ["# Agentic pattern grid — summary", "",
             f"{df.qid.nunique()} queries x {df.pattern.nunique()} patterns "
             f"x styles = {len(df)} episodes. RAGAS-definition metrics with "
             f"a LOCAL judge; deterministic grounding needs no judge.", "",
             g.to_markdown(index=False)]
    (d / "summary.md").write_text("\n".join(lines))

    # ---------------- figure 10: quality grid ---------------------------
    pats = [p for p in ["B0_retrieval", "P1_direct", "P2_rag", "P3_react",
                        "P4_reflexion", "P5_verifier", "P6_specialists"]
            if p in set(df["pattern"])]
    styles = [s for s in ["plain", "cot", "fewshot"] if s in set(df["style"])]
    fig, axes = plt.subplots(2, 2, figsize=(W2, 0.62 * W2))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    piv = g.pivot_table(index="pattern", columns="style",
                        values="faithfulness").reindex(pats)
    im = ax_a.imshow(piv.values.astype(float), cmap="Blues", vmin=0, vmax=1,
                     aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if v == v:
                ax_a.text(j, i, f"{v:.2f}", ha="center", va="center",
                          fontsize=6.5,
                          color="white" if v > 0.6 else "#333")
    ax_a.set_xticks(range(piv.shape[1]))
    ax_a.set_xticklabels(piv.columns)
    ax_a.set_yticks(range(piv.shape[0]))
    ax_a.set_yticklabels([p.replace("_", " ") for p in piv.index],
                         fontsize=6.3)
    ax_a.grid(False)
    fig.colorbar(im, ax=ax_a, fraction=0.045, pad=0.02).ax.tick_params(
        labelsize=6)
    panel(ax_a, "a", "Faithfulness (local judge)")

    x = np.arange(len(pats))
    w = 0.38
    best = g.groupby("pattern").faithfulness.mean().reindex(pats)
    ar = g.groupby("pattern").answer_relevancy.mean().reindex(pats)
    cp = g.groupby("pattern").context_precision.mean().reindex(pats)
    ax_b.bar(x - w / 2, ar.values, w, color=C["third"],
             label="answer relevancy")
    ax_b.bar(x + w / 2, cp.values, w, color=C["accent"],
             label="context precision")
    ax_b.plot(x, best.values, "o-", color=C["main"], ms=3.6,
              label="faithfulness")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([p.split("_")[0] for p in pats])
    ax_b.set_ylim(0, 1.05)
    ax_b.legend(loc="lower right")
    panel(ax_b, "b", "RAGAS profile by pattern (styles pooled)")

    det = g.groupby("pattern")[["citation_validity", "rul_in_cited_span",
                                "action_band"]].mean().reindex(pats)
    for k, (col, lab, cc) in enumerate(
            [("citation_validity", "citations valid", C["main"]),
             ("rul_in_cited_span", "RUL in cited span", C["third"]),
             ("action_band", "action-band consistent", C["alt"])]):
        ax_c.plot(x, det[col].values, "o-", ms=3.2, color=cc, label=lab)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([p.split("_")[0] for p in pats])
    ax_c.set_ylim(0, 1.05)
    ax_c.legend(loc="lower right", fontsize=6)
    panel(ax_c, "c", "Deterministic grounding (no judge)")

    cost = g.groupby("pattern")[["llm_calls", "wall_s"]].mean().reindex(pats)
    f_ = best
    ax_d.scatter(cost.llm_calls, f_.values, s=26, color=C["main"])
    for p, xx, yy in zip(pats, cost.llm_calls, f_.values):
        ax_d.annotate(p.split("_")[0], (xx, yy),
                      textcoords="offset points", xytext=(4, 3),
                      fontsize=6.2, color=C["grey"])
    ax_d.set_xlabel("LLM calls per case")
    ax_d.set_ylabel("faithfulness")
    ax_d.set_ylim(0, 1.05)
    panel(ax_d, "d", "Quality vs agentic cost")

    fig.tight_layout(w_pad=2.2, h_pad=1.6)
    for ext in ("pdf", "png"):
        fig.savefig(d / f"fig10_pattern_grid.{ext}")
    plt.close(fig)

    # ---------------- figure 11: gates + styles -------------------------
    fig, axes = plt.subplots(1, 2, figsize=(W2, 0.34 * W2))
    ax_a, ax_b = axes
    gf = g.groupby("pattern")[["gate_fail", "escalated"]].mean().reindex(pats)
    ax_a.bar(x - 0.19, gf.gate_fail, 0.38, color=C["alt"],
             label="any gate violated (final)")
    ax_a.bar(x + 0.19, gf.escalated.fillna(0), 0.38, color=C["accent"],
             label="escalated to human")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([p.split("_")[0] for p in pats])
    ax_a.legend(fontsize=6)
    ax_a.set_ylabel("rate")
    panel(ax_a, "a", "Verifier outcomes")

    sty = g.groupby("style")[RAGAS].mean().reindex(styles)
    xs = np.arange(len(styles))
    for k, (m_, cc) in enumerate(zip(RAGAS,
                                     [C["main"], C["third"], C["accent"]])):
        ax_b.bar(xs + (k - 1) * 0.27, sty[m_].values, 0.27, color=cc,
                 label=m_.replace("_", " "))
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels(styles)
    ax_b.set_ylim(0, 1.05)
    ax_b.legend(fontsize=6)
    panel(ax_b, "b", "Prompting style (patterns pooled)")
    fig.tight_layout(w_pad=2.2)
    for ext in ("pdf", "png"):
        fig.savefig(d / f"fig11_gates_styles.{ext}")
    plt.close(fig)

    print(g.to_string(index=False))
    print(f"\n[report] summary.csv/.md + fig10/fig11 -> {d}")


if __name__ == "__main__":
    main()

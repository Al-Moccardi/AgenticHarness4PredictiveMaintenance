#!/usr/bin/env python3
"""
paper_figures_core.py — the THREE core contribution figures, LaTeX-typeset.

  figC1_risk_profile   Conservative by construction: signed-error quantile
                       intervals per arm; the unanchored agent over-promises
                       (37% of cases > +20 cycles), the anchored agent
                       never does (0%, q95 = +2).
  figC2_outcome_matrix From a number to an accountable outcome: what each
                       system delivers, with the measured rate for each row.
  figC3_evidence_card  The quantitative case: paired prognostic effects
                       (cycles, 95% CI) and the diagnostic redesign rates.

Typography: text.usetex if a working LaTeX toolchain is present,
otherwise Computer Modern via mathtext (identical look).

Run:
  python paper\\code\\figures\\paper_figures_core.py ^
      --arad results\\arad3 --arad2 results\\arad2
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
     "gold": "#E69F00", "red": "#B22222"}


def setup_fonts() -> bool:
    """True if real LaTeX is active, else Computer Modern mathtext."""
    base = {"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
            "xtick.labelsize": 8, "ytick.labelsize": 8,
            "legend.fontsize": 7.5, "axes.linewidth": 0.6,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "grid.alpha": 0.25,
            "legend.frameon": False, "figure.dpi": 300,
            "savefig.bbox": "tight", "pdf.fonttype": 42,
            "axes.unicode_minus": False}
    try:
        plt.rcParams.update({**base, "text.usetex": True,
                             "font.family": "serif",
                             "text.latex.preamble":
                                 r"\usepackage{amsmath}"})
        f, a = plt.subplots(figsize=(1, 1))
        a.set_title(r"$\rho$ 95\% $\Delta$")
        f.savefig("/tmp/_texprobe.png")
        plt.close(f)
        return True
    except Exception:
        plt.rcParams.update({**base, "text.usetex": False,
                             "font.family": "serif",
                             "font.serif": ["cmr10", "DejaVu Serif"],
                             "mathtext.fontset": "cm",
                             "axes.formatter.use_mathtext": True})
        return False


USETEX = setup_fonts()
RNG = np.random.default_rng(42)


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
    print(f"  [core] {name}  (usetex={USETEX})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arad", default="agentic/results/final_prognostic")
    ap.add_argument("--arad2", default="agentic/results/v2_rul_coupled_collapse")
    ap.add_argument("--out", default=str(_r / "paper/figures_regen/core"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    A, A2 = Path(a.arad), Path(a.arad2)

    M = pd.read_csv(Path(a.arad) / "forecast_metrics.csv")
    eps = jread(Path(a.arad) / "forecast_episodes.jsonl")
    hint = {(e["qid"], e["arm"]): e.get("dl_hint") for e in eps}
    M["hint"] = [hint.get((q, ar)) for q, ar in zip(M.qid, M.arm)]

    # =========== figC1 — conservative by construction ==================
    arms = ["P7_agent", "b0_median", "dl_only", "P7_agent_dl"]
    ALAB = {"b0_median": "precedent floor (b0)",
            "dl_only": "CNN-GRU alone",
            "P7_agent": "agent WITHOUT anchor",
            "P7_agent_dl": r"\textbf{Guarded Agent}" if USETEX
                           else "Guarded Agent"}
    fig, ax = plt.subplots(figsize=(W2 * 0.92, 0.40 * W2))
    ax.axvspan(0, 90, color=C["red"], alpha=0.06)
    ax.text(55, 3.42, "over-prediction: promises life\n"
                      "the asset does not have (unsafe)",
            fontsize=8, color=C["red"], ha="center")
    ax.axvline(0, color=C["grey"], lw=1.0)
    for y, arm in enumerate(arms):
        e = M[M.arm == arm].rul_err.dropna()
        q5, q25, q50, q75, q95 = np.percentile(e, [5, 25, 50, 75, 95])
        safe = (e <= 0).mean()
        danger = (e > 20).mean()
        col = (C["main"] if arm == "P7_agent_dl" else
               C["accent"] if arm == "P7_agent" else
               C["alt"] if arm == "dl_only" else C["grey"])
        ax.plot([q5, q95], [y, y], color=col, lw=1.4, alpha=0.8)
        ax.plot([q25, q75], [y, y], color=col, lw=5, solid_capstyle="butt")
        ax.plot([q50], [y], marker="o", ms=6, color="white",
                markeredgecolor=col, markeredgewidth=1.6, zorder=5)
        ax.text(92, y + 0.13, f"safe-side {pct(safe)}", fontsize=7.6,
                va="center", color=C["grey"])
        ax.text(92, y - 0.17,
                f"over-pred $>$20: {pct(danger, 1)}", fontsize=7.6,
                va="center",
                color=C["red"] if danger > 0.05 else C["third"])
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([ALAB[x] for x in arms])
    ax.set_xlim(-125, 90)
    ax.set_ylim(-0.55, 3.75)
    ax.set_xlabel(r"signed RUL error, predicted $-$ true (cycles): "
                  r"5--95\% span, 25--75\% box, median"
                  if USETEX else
                  "signed RUL error, predicted $-$ true (cycles): "
                  "5-95% span, 25-75% box, median")
    ax.set_title("Deterministic anchoring makes the agent conservative "
                 "by construction", loc="left", pad=8)
    fig.subplots_adjust(right=0.82)
    save(fig, out, "figC1_risk_profile")

    # =========== figC2 — the outcome-extension matrix ==================
    dl = M[M.arm == "P7_agent_dl"].copy()
    dl["corrupt_flag"] = dl.hint.fillna(0) <= 5
    corr = dl[dl.corrupt_flag]
    te = np.abs(corr.hint.fillna(0) - corr.true_rul.clip(upper=125))
    pe = [e for e in eps if e["arm"] == "P7_agent_dl"]
    cv = []
    for e in pe:
        f_ = e.get("forecast") or {}
        shown = [f"u{c['unit']}c{c['cycle']}" for c in
                 (e.get("contexts") or [])]
        cited = f_.get("cited_precedents") or []
        if cited:
            cv.append(float(all(c in shown for c in cited)))
    rows = [
        ("RUL point estimate",
         "MAE 15.6, bias $-$15.6",
         "MAE 22.8, bias $-$22.1 (bounded overhead $+$7.0)"),
        ("uncertainty range",
         "---",
         f"stated in {pct(dl.range_width.notna().mean())} of cases; "
         f"true RUL inside {pct(dl.range_coverage.mean())}"),
        ("precedent citations",
         "---",
         f"{pct(np.mean(cv))} verifiably grounded (n={len(cv)})"),
        ("maintenance action + narrative",
         "---",
         f"structured ticket in {pct(dl.json_valid.mean(), 1)} of cases"),
        ("safe-side behaviour",
         f"{pct((M[M.arm == 'dl_only'].rul_err <= 0).mean())} "
         "(always under)",
         f"{pct((dl.rul_err.dropna() <= 0).mean())}; over-pred $>$20 cycles: "
         f"{pct((dl.rul_err.dropna() > 20).mean())}"),
        ("operation under tool fault",
         f"MAE {te.mean():.1f} (fails silently)",
         f"MAE {corr.rul_abs_err.mean():.1f}, overrides "
         f"{pct((np.abs(corr.rul_pred - corr.hint.fillna(0)) > 5).mean())}"
         " of faults, stays safe-side "
         f"{pct((corr.rul_err <= 0).mean())}"),
    ]
    fig, ax = plt.subplots(figsize=(W2, 0.44 * W2))
    ax.axis("off")
    x0, x1, x2 = 0.00, 0.335, 0.60
    ax.text(x1 + 0.01, 1.00, "CNN-GRU alone", fontsize=9.5,
            fontweight="bold", color=C["alt"], va="top")
    ax.text(x2 + 0.01, 1.00, "Guarded Agent", fontsize=9.5,
            fontweight="bold", color=C["main"], va="top")
    yy = 0.90
    dy = 0.155
    for name, tool, agent in rows:
        ax.axhline(yy + 0.055, xmin=0.0, xmax=1.0, color=C["light"],
                   lw=0.5)
        ax.text(x0, yy, name, fontsize=8.6, va="top",
                fontweight="bold")
        ax.text(x1 + 0.01, yy, tool, fontsize=8.2, va="top",
                color=C["grey"] if tool == "---" else "black", wrap=True)
        ax.text(x2 + 0.01, yy, agent, fontsize=8.2, va="top", wrap=True)
        yy -= dy
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    ax.set_title("The agentic layer extends a point estimate into an "
                 "accountable maintenance outcome", loc="left", pad=6)
    save(fig, out, "figC2_outcome_matrix")

    # =========== figC3 — the evidence card =============================
    def paired(a1, a2, subset=None):
        m = M if subset is None else subset
        x = m[m.arm == a1].set_index("qid").rul_abs_err
        y = m[m.arm == a2].set_index("qid").rul_abs_err
        d = (x - y).dropna().values
        bs = [d[RNG.integers(0, len(d), len(d))].mean()
              for _ in range(3000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return d.mean(), lo, hi

    # rescue as paired: agent |err| - tool-as-seen |err| on corrupted
    ce = corr.set_index("qid")
    dres = (ce.rul_abs_err
            - np.abs(ce.hint.fillna(0)
                     - ce.true_rul.clip(upper=125))).dropna()
    bs = [dres.values[RNG.integers(0, len(dres), len(dres))].mean()
          for _ in range(3000)]
    rlo, rhi = np.percentile(bs, [2.5, 97.5])

    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(W2, 0.30 * W2),
        gridspec_kw={"width_ratios": [1.15, 1]})
    effects = [("agent vs its own tool", *paired("P7_agent_dl",
                                                 "dl_only")),
               ("value of the DL anchor", *paired("P7_agent_dl",
                                                  "P7_agent")),
               ("agent vs precedent floor", *paired("P7_agent_dl",
                                                    "b0_median")),
               ("rescue on corrupted tool", dres.mean(), rlo, rhi)]
    for j, (lab, m_, lo, hi) in enumerate(effects):
        yj = len(effects) - 1 - j
        ax.errorbar(m_, yj, xerr=[[m_ - lo], [hi - m_]], fmt="o", ms=5,
                    color=C["main"], capsize=3.4, lw=1.2)
        ax.text(m_, yj - 0.22, f"{m_:+.1f} [{lo:+.1f}, {hi:+.1f}]",
                ha="center", va="top", fontsize=7.4, color=C["grey"])
    ax.axvline(0, color=C["light"], lw=0.9, ls="--")
    ax.set_yticks(range(len(effects)))
    ax.set_yticklabels([e[0] for e in effects][::-1], fontsize=8)
    ax.set_ylim(-0.8, len(effects) - 0.4)
    ax.set_xlim(-32, 26)
    ax.set_xlabel(r"paired $\Delta$ absolute error "
                  r"(cycles; $<0$ favours the agent)")
    ax.set_title("Prognostic evidence (89 anomalies, 95\\% CI)"
                 if USETEX else
                 "Prognostic evidence (89 anomalies, 95% CI)",
                 loc="left", pad=6)

    if (A2 / "diag" / "episodes.jsonl").exists():
        def esc_rate(dir_, patt="P5_verifier", legacy=False):
            es, n, coh = 0, 0, []
            BANDS_ = [(120, "continue_monitoring"),
                      (60, "schedule_inspection"),
                      (25, "plan_maintenance"),
                      (-1, "immediate_shutdown")]

            def band_(r):
                for t, a_ in BANDS_:
                    if r > t:
                        return a_
                return "immediate_shutdown"
            S2A = {1: "continue_monitoring", 2: "continue_monitoring",
                   3: "schedule_inspection", 4: "plan_maintenance",
                   5: "immediate_shutdown"}
            ep = dir_ / "diag" / "episodes.jsonl"
            if not ep.exists():
                ep = _DIAG_DIR / "episodes.jsonl"
            for e in jread(ep):
                if e["pattern"] != patt:
                    continue
                n += 1
                es += 1 if e.get("escalated") else 0
                t = e.get("ticket") or {}
                if legacy:
                    if t.get("rul_estimate") is not None and t.get(
                            "action"):
                        coh.append(float(t["action"]
                                         == band_(t["rul_estimate"])))
                else:
                    if t.get("severity") is not None:
                        coh.append(float(t.get("action")
                                         == S2A.get(t["severity"])))
            return es / n, float(np.mean(coh))
        e2, c2 = esc_rate(A2, legacy=True)
        e3, c3 = esc_rate(A, legacy=False)
        rows2 = [("escalation", e2, e3, True),
                 ("action coherence", c2, c3, False)]
        for i, (lab, v2, v3, invert) in enumerate(rows2):
            y = 1 - i
            bx.plot([v2, v3], [y, y], color=C["light"], lw=1.8, zorder=1)
            bx.scatter([v2], [y], s=34, color=C["grey"], zorder=2,
                       label="RUL-coupled design (v2)" if i == 0 else None)
            bx.scatter([v3], [y], s=40, color=C["main"], zorder=3,
                       label="retrospective A-RAD (v3)" if i == 0
                       else None)
            bx.annotate("", xy=(v3, y + 0.16), xytext=(v2, y + 0.16),
                        arrowprops=dict(arrowstyle="->", lw=0.9,
                                        color=C["third"]))
            bx.text(1.04, y, f"{pct(v2)} $\\to$ {pct(v3)}", va="center",
                    fontsize=8, transform=bx.get_yaxis_transform())
        bx.set_yticks([0, 1])
        bx.set_yticklabels(["action coherence", "escalation"],
                           fontsize=8)
        bx.set_xlim(-0.04, 1.04)
        bx.set_ylim(-0.55, 1.75)
        bx.set_xlabel("rate on the diagnostic agent arm")
        bx.legend(loc="lower center", bbox_to_anchor=(0.5, 0.02),
                  fontsize=7)
        bx.set_title("Diagnostic evidence (89 cases)", loc="left", pad=6)
        fig.subplots_adjust(right=0.86, wspace=0.55)
    save(fig, out, "figC3_evidence_card")
    print(f"[core] -> {out}")


if __name__ == "__main__":
    main()

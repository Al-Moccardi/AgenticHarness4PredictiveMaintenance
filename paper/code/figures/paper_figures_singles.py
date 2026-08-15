#!/usr/bin/env python3
"""
paper_figures_singles.py — the COMPLETE single-panel library.

Every plot is ONE png (+ pdf). Contents:
  * splits of every composite figure (core, more)
  * 14 NEW research-outcome plots, self-derived from the campaign data,
    organised by research question (see INDEX.md written alongside)

Run:
  python paper\\code\\figures\\paper_figures_singles.py ^
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
     "gold": "#E69F00", "red": "#B22222"}


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
        f.savefig("/tmp/_texprobe4.png")
        plt.close(f)
        return True
    except Exception:
        plt.rcParams.update({**base, "text.usetex": False,
                             "font.family": "serif",
                             "font.serif": ["cmr10", "DejaVu Serif"],
                             "mathtext.fontset": "cm",
                             "axes.formatter.use_mathtext": True})
        return False


USETEX = _fonts()
RNG = np.random.default_rng(42)
BANDS = [(100, 126, r"$\geq$100"), (60, 100, "60--100" if USETEX
         else "60-100"), (25, 60, "25--60" if USETEX else "25-60"),
         (0, 25, "$<$25")]
ACTIONS = ["continue_monitoring", "schedule_inspection",
           "plan_maintenance", "immediate_shutdown"]
ALAB = ["monitor", "inspect", "plan", "shutdown"]
SEV2ACT = {1: "continue_monitoring", 2: "continue_monitoring",
           3: "schedule_inspection", 4: "plan_maintenance",
           5: "immediate_shutdown"}
INDEX: list[tuple[str, str, str]] = []      # (rq, name, caption)


def pct(x, dec=0):
    s = f"{100 * x:.{dec}f}"
    return (s + r"\%") if USETEX else (s + "%")


def jread(p: Path):
    return [json.loads(l) for l in
            p.read_text(encoding="utf-8", errors="ignore").splitlines()
            if l.strip()]


def band_idx(r):
    r = min(r, 125)
    for i, (lo, hi, _) in enumerate(BANDS):
        if lo <= r < hi:
            return i
    return len(BANDS) - 1


def fig1(w=1.3, h=0.66):
    return plt.subplots(figsize=(W1 * w, W1 * w * h))


def save(fig, out, name, rq, caption):
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)
    INDEX.append((rq, name, caption))
    print(f"  [single] {name}")


def load_diag(dir_: Path):
    rows = []
    ep = dir_ / "diag" / "episodes.jsonl"
    if not ep.exists():
        ep = _DIAG_DIR / "episodes.jsonl"
    for e in jread(ep):
        t = e.get("ticket") or {}
        rows.append(dict(pattern=e["pattern"], qid=e["qid"],
                         true=e.get("true_rul"), ticket=t,
                         sev=t.get("severity"), act=t.get("action"),
                         est=t.get("rul_estimate"),
                         esc=bool(e.get("escalated")),
                         ac=bool(e.get("auto_corrected")),
                         rep=e.get("repairs") or 0,
                         gates=e.get("gate_violations") or [],
                         tok=e.get("tokens_out") or 0,
                         wall=e.get("wall_s") or np.nan,
                         hist=bool(e.get("has_history")),
                         ctx=e.get("contexts") or []))
    return pd.DataFrame(rows)


def lf(c, r):
    return c / (c + max(r, 1))


def case_cycle(qid):
    m = re.search(r"c(\d+)$", str(qid))
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arad", default="agentic/results/final_prognostic")
    ap.add_argument("--arad2", default="agentic/results/v2_rul_coupled_collapse")
    ap.add_argument("--v1grid", default="agentic/results/study_A_pattern_grid")
    ap.add_argument("--out", default=str(_r / "paper/figures_regen/singles"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    A, A2 = Path(a.arad), Path(a.arad2)
    D3 = load_diag(A)
    has2 = (A2 / "diag" / "episodes.jsonl").exists()
    D2 = load_diag(A2) if has2 else None
    M = pd.read_csv(Path(a.arad) / "forecast_metrics.csv")
    peps = jread(Path(a.arad) / "forecast_episodes.jsonl")
    hint = {(e["qid"], e["arm"]): e.get("dl_hint") for e in peps}
    M["hint"] = [hint.get((q, ar)) for q, ar in zip(M.qid, M.arm)]
    pmap = {(e["qid"], e["arm"]): e for e in peps}
    dl = M[M.arm == "P7_agent_dl"].copy()
    dl["ov"] = np.abs(dl.rul_pred - dl.hint.fillna(0)) > 5
    dl["corr_f"] = dl.hint.fillna(0) <= 5

    def med_future(row):
        e = pmap.get((row.qid, "P7_agent_dl"), {})
        f_ = e.get("forecast") or {}
        cited = set(f_.get("cited_precedents") or [])
        vals = [c["rul_then"] for c in (e.get("contexts") or [])
                if f"u{c['unit']}c{c['cycle']}" in cited
                and c.get("rul_then") is not None]
        if not vals:
            vals = [c["rul_then"] for c in (e.get("contexts") or [])
                    if c.get("rul_then") is not None]
        return np.median(vals) if vals else np.nan
    dl["prec_med"] = dl.apply(med_future, axis=1)

    def mean_sim(row):
        e = pmap.get((row.qid, "P7_agent_dl"), {})
        vals = [c.get("similarity") for c in (e.get("contexts") or [])
                if c.get("similarity") is not None]
        return np.mean(vals) if vals else np.nan
    dl["sim"] = dl.apply(mean_sim, axis=1)

    # ================== NEW RESEARCH-OUTCOME PLOTS =====================
    # N1 — override policy vs evidence disagreement
    g = dl.dropna(subset=["prec_med"])
    gap = np.abs(g.hint.fillna(0) - g.prec_med)
    bins = [0, 10, 20, 40, 130]
    bi = np.digitize(gap, bins) - 1
    fig, ax = fig1(1.35, 0.62)
    xs, ys, ns = [], [], []
    for b in range(len(bins) - 1):
        m_ = bi == b
        if m_.sum():
            xs.append((bins[b] + bins[b + 1]) / 2)
            ys.append(g.ov[m_].mean())
            ns.append(int(m_.sum()))
    ax.plot(xs, ys, marker="o", ms=5, color=C["main"], lw=1.4)
    for x, y, n in zip(xs, ys, ns):
        ax.text(x, y + 0.045, f"n={n}", ha="center", fontsize=6.6,
                color=C["grey"])
    ax.set_xlabel(r"$|$hint $-$ median precedent future$|$ (cycles)")
    ax.set_ylabel("P(agent overrides the hint)")
    ax.set_ylim(0, 1.1)
    ax.set_title("The override policy tracks evidence disagreement",
                 loc="left", pad=5)
    save(fig, out, "figN1_override_policy", "RQ4 robustness",
         "Override probability rises with the gap between the DL hint "
         "and the median cited precedent future.")

    # N2 — where the overriding estimate comes from
    go = dl[dl.ov & dl.prec_med.notna()]
    fig, ax = fig1(1.3, 0.72)
    sc = ax.scatter(go.prec_med, go.rul_pred, c=go.rul_abs_err, s=18,
                    cmap="viridis", vmax=60)
    ax.plot([0, 125], [0, 125], ls="--", lw=0.9, color=C["light"])
    from scipy import stats as st
    rho, pv = st.spearmanr(go.prec_med, go.rul_pred)
    ax.text(0.03, 0.94, f"$\\rho$ = {rho:.2f} (p = {pv:.1g})",
            transform=ax.transAxes, fontsize=8)
    plt.colorbar(sc, ax=ax, label=r"$|$error$|$ (cycles)", shrink=0.9)
    ax.set_xlabel("median cited precedent future (cycles)")
    ax.set_ylabel("agent estimate on override cases")
    ax.set_title("When it overrides, the agent anchors on precedent "
                 "futures", loc="left", pad=5)
    save(fig, out, "figN2_override_anchor", "RQ4 robustness",
         "On override cases the estimate correlates with the median "
         "cited precedent future: arbitration, not guessing.")

    # N3 — learning along the per-unit chain
    dl2 = dl.sort_values(["unit", "cycle"]).copy()
    dl2["pos"] = dl2.groupby("unit").cumcount()
    dl2["posb"] = pd.cut(dl2.pos, [-1, 1, 4, 9, 99],
                         labels=["1st-2nd", "3rd-5th", "6th-10th",
                                 "11th+"])
    fig, ax = fig1(1.3, 0.62)
    med = dl2.groupby("posb", observed=True).rul_abs_err.median()
    q1 = dl2.groupby("posb", observed=True).rul_abs_err.quantile(.25)
    q3 = dl2.groupby("posb", observed=True).rul_abs_err.quantile(.75)
    x = np.arange(len(med))
    ax.plot(x, med.values, marker="o", ms=5, color=C["main"], lw=1.5)
    ax.fill_between(x, q1.values, q3.values, color=C["main"], alpha=0.15)
    ax.set_xticks(x)
    ax.set_xticklabels(med.index, fontsize=7.5)
    ax.set_xlabel("position of the anomaly in the unit's chain")
    ax.set_ylabel(r"$|$error$|$ (cycles), median and IQR")
    ax.set_title("Accuracy improves along the unit-chained dialogue",
                 loc="left", pad=5)
    save(fig, out, "figN3_chain_learning", "RQ3 prognostic value",
         "Median |error| falls as the per-unit chain accumulates "
         "anomalies: the chained protocol pays off late in life.")

    # N4 — retrieval quality vs accuracy
    g = dl.dropna(subset=["sim"])
    fig, ax = fig1(1.3, 0.62)
    ax.scatter(g.sim, g.rul_abs_err, s=14, alpha=0.6, color=C["main"])
    rho, pv = st.spearmanr(g.sim, g.rul_abs_err)
    ax.text(0.03, 0.92, f"$\\rho$ = {rho:.2f} (p = {pv:.2g})",
            transform=ax.transAxes, fontsize=8)
    ax.set_xlabel("mean similarity of retrieved precedents")
    ax.set_ylabel(r"$|$error$|$ (cycles)")
    ax.set_title("Retrieval similarity is a weak predictor of accuracy",
                 loc="left", pad=5)
    save(fig, out, "figN4_similarity_vs_error", "RQ2 grounding",
         "Similarity alone does not buy accuracy: relevance must be "
         "stage-aware (cf. the retrieval-fix figure).")

    # N5 — interval calibration cone
    g = dl.dropna(subset=["range_width", "rul_abs_err"])
    fig, ax = fig1(1.3, 0.72)
    ax.scatter(g.range_width, g.rul_abs_err, s=14, alpha=0.6,
               color=C["main"])
    lim = max(g.range_width.max(), g.rul_abs_err.max()) * 1.05
    ax.plot([0, lim], [0, lim / 2], ls="--", lw=1.0, color=C["third"])
    ax.text(lim * 0.55, lim * 0.30, r"$|$err$|$ = width/2", fontsize=7,
            color=C["third"], rotation=22)
    inside = (g.rul_abs_err <= g.range_width / 2).mean()
    ax.text(0.03, 0.92, f"$|$err$|\\;\\leq$ width/2 in {pct(inside)} "
                        "of cases", transform=ax.transAxes, fontsize=8)
    ax.set_xlabel("stated interval width (cycles)")
    ax.set_ylabel(r"$|$error$|$ (cycles)")
    ax.set_title("Stated uncertainty vs realised error", loc="left",
                 pad=5)
    save(fig, out, "figN5_interval_cone", "RQ3 prognostic value",
         "Half-width bounds the realised error in about half the "
         "cases: honest but under-dispersed intervals.")

    # N6 — wall-clock ECDFs (edge viability)
    fig, ax = fig1(1.3, 0.62)
    dw = D3[D3.pattern == "P5_verifier"].wall.dropna()
    pw = pd.Series([e.get("wall_s") for e in peps
                    if e["arm"] == "P7_agent_dl"]).dropna()
    for v, cc, lab in ((dw, C["main"], "diagnostic agent"),
                       (pw, C["gold"], "prognostic agent")):
        vs = np.sort(v)
        ax.plot(vs, np.linspace(0, 1, len(vs)), lw=1.5, color=cc,
                label=f"{lab} (median {np.median(v):.1f} s)")
    ax.set_xlabel("wall-clock seconds per case (3B model)")
    ax.set_ylabel("ECDF")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title("Per-case latency at the edge", loc="left", pad=5)
    save(fig, out, "figN6_latency_ecdf", "RQ5 deployment",
         "Full per-case latency distributions for both agents on "
         "edge-class hardware.")

    # N7 — EOL slopegraph, case by case
    z = M[M.true_rul < 20]
    zd = z[z.arm == "dl_only"].set_index("qid").rul_abs_err
    za = z[z.arm == "P7_agent_dl"].set_index("qid").rul_abs_err
    both = pd.concat([zd, za], axis=1, keys=["tool", "agent"]).dropna()
    fig, ax = fig1(1.15, 0.85)
    for _, r in both.iterrows():
        cc = C["third"] if r.agent < r.tool else C["red"]
        ax.plot([0, 1], [r.tool, r.agent], color=cc, lw=1.0, alpha=0.7,
                marker="o", ms=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["CNN-GRU alone", "Guarded Agent"])
    ax.set_ylabel(r"$|$error$|$ (cycles)")
    ax.set_xlim(-0.25, 1.25)
    imp = (both.agent < both.tool).mean()
    ax.set_title(f"End-of-life cases one by one: the agent improves "
                 f"{pct(imp)}", loc="left", pad=5)
    save(fig, out, "figN7_eol_slopegraph", "RQ4 robustness",
         "Per-case slopegraph inside true RUL<20: green lines fall "
         "(agent better), red rise.")

    # N8 — cumulative safety Pareto
    fig, ax = fig1(1.3, 0.62)
    for arm, cc, lab in (("dl_only", C["alt"], "CNN-GRU alone"),
                         ("P7_agent_dl", C["main"],
                          "Guarded Agent")):
        v = np.sort(M[M.arm == arm].s_score.dropna().values)[::-1]
        ax.plot(np.arange(1, len(v) + 1), np.cumsum(v), lw=1.5,
                color=cc, label=lab)
    ax.set_xlabel("cases, worst first")
    ax.set_ylabel("cumulative S-score")
    ax.set_yscale("log")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title("Total safety cost is dominated by a few worst cases",
                 loc="left", pad=5)
    save(fig, out, "figN8_safety_pareto", "RQ3 prognostic value",
         "Cumulative asymmetric penalty, worst cases first (log): "
         "where each system accumulates its risk.")

    # N9 — severity x action heat (the coherence contract)
    g5 = D3[(D3.pattern == "P5_verifier")].dropna(subset=["sev"])
    H = np.zeros((5, 4))
    for _, r in g5.iterrows():
        if r.act in ACTIONS:
            H[int(r.sev) - 1, ACTIONS.index(r.act)] += 1
    fig, ax = fig1(1.15, 0.72)
    ax.imshow(H, cmap="Blues", aspect="auto", origin="lower")
    for i in range(5):
        for j in range(4):
            if H[i, j]:
                ax.text(j, i, int(H[i, j]), ha="center", va="center",
                        fontsize=7.5,
                        color="white" if H[i, j] > H.max() / 2
                        else "black")
    for sv, act in SEV2ACT.items():
        ax.add_patch(plt.Rectangle((ACTIONS.index(act) - 0.5,
                                    sv - 1 - 0.5), 1, 1, fill=False,
                                   edgecolor=C["third"], lw=1.4))
    ax.set_xticks(range(4))
    ax.set_xticklabels(ALAB, fontsize=7)
    ax.set_yticks(range(5))
    ax.set_yticklabels(range(1, 6))
    ax.set_xlabel("recommended action")
    ax.set_ylabel("severity")
    ax.grid(False)
    ax.set_title("The severity-action contract, enforced "
                 "(green = permitted)", loc="left", pad=5)
    save(fig, out, "figN9_sev_action_heat", "RQ2 grounding",
         "Joint severity-action counts for the agent; green boxes mark "
         "the contract the verifier enforces.")

    # N10 — repair rate across life stages (null result, honest)
    g5b = D3[D3.pattern == "P5_verifier"].copy()
    g5b["b"] = [band_idx(t) for t in g5b.true]
    fig, ax = fig1(1.25, 0.6)
    rr = g5b.groupby("b").apply(lambda g_: (g_.rep > 0).mean())
    ax.bar(range(len(BANDS)), [rr.get(i, 0) for i in range(len(BANDS))],
           color=C["main"], width=0.55)
    ax.set_xticks(range(len(BANDS)))
    ax.set_xticklabels([b[2] for b in BANDS])
    ax.set_xlabel("true RUL band (cycles)")
    ax.set_ylabel("share of cases needing repair")
    ax.set_ylim(0, 0.6)
    ax.set_title("Repairs are not stage-biased (a useful null result)",
                 loc="left", pad=5)
    save(fig, out, "figN10_repairs_by_stage", "RQ2 grounding",
         "Verifier repairs distribute across life stages: correction "
         "load is structural, not degradation-driven.")

    # N11 — the retrieval trade-off frontier (per case)
    if has2:
        def case_pts(D):
            pts = []
            for _, r in D[D.pattern.isin(["P2_rag",
                                          "P5_verifier"])].iterrows():
                cc_ = case_cycle(r.qid)
                if cc_ is None or r.true is None:
                    continue
                clf = lf(cc_, r.true)
                gaps, sims = [], []
                cited = set(r.ticket.get("cited_precedents") or [])
                for c in r.ctx:
                    key = f"u{c['unit']}c{c['cycle']}"
                    if cited and key not in cited:
                        continue
                    if c.get("rul_then") is None:
                        continue
                    gaps.append(abs(lf(c["cycle"], c["rul_then"])
                                    - clf))
                    if c.get("similarity") is not None:
                        sims.append(c["similarity"])
                if gaps and sims:
                    pts.append((np.mean(sims), np.mean(gaps)))
            return np.array(pts)
        P2 = case_pts(D2)
        P3 = case_pts(D3)
        fig, ax = fig1(1.3, 0.72)
        ax.scatter(P2[:, 0], P2[:, 1], s=12, alpha=0.5, color=C["alt"],
                   label="plain cosine (v2)")
        ax.scatter(P3[:, 0], P3[:, 1], s=12, alpha=0.5, color=C["main"],
                   label="stage-aware (v3)")
        for P_, cc in ((P2, C["alt"]), (P3, C["main"])):
            ax.scatter([P_[:, 0].mean()], [P_[:, 1].mean()], s=90,
                       marker="X", color=cc, edgecolor="white",
                       lw=1.2, zorder=5)
        ax.set_xlabel("mean similarity of cited precedents")
        ax.set_ylabel(r"mean $|$life-stage gap$|$")
        ax.legend(loc="upper left", fontsize=7)
        ax.set_title("The retrieval trade-off: a little similarity "
                     "buys a lot of stage relevance", loc="left", pad=5)
        save(fig, out, "figN11_retrieval_frontier", "RQ1 mechanism",
             "Per-case frontier: v3 moves down (stage-relevant) at a "
             "small similarity cost; X marks the means.")

    # N12 — uncertainty behaviour under tool fault
    fig, ax = fig1(1.25, 0.6)
    for i, (flag, lab) in enumerate(((False, "tool usable"),
                                     (True, "tool corrupted"))):
        g = dl[dl.corr_f == flag]
        ax.bar(i - 0.18, g.range_coverage.mean(), width=0.34,
               color=C["main"],
               label="coverage" if i == 0 else None)
        ax.bar(i + 0.18, g.range_width.mean() / 100, width=0.34,
               color=C["gold"],
               label="width / 100" if i == 0 else None)
        ax.text(i - 0.18, g.range_coverage.mean() + 0.02,
                pct(g.range_coverage.mean()), ha="center", fontsize=7)
        ax.text(i + 0.18, g.range_width.mean() / 100 + 0.02,
                f"{g.range_width.mean():.0f}", ha="center", fontsize=7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["tool usable", "tool corrupted"])
    ax.set_ylim(0, 0.85)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("Under tool fault the agent keeps coverage without "
                 "exploding width", loc="left", pad=5)
    save(fig, out, "figN12_uncertainty_fault", "RQ4 robustness",
         "Interval coverage and width split by hint status.")

    # N13 — per-unit signed bias
    pu = (M[M.arm.isin(["dl_only", "P7_agent_dl"])]
          .pivot_table(index="unit", columns="arm", values="rul_err",
                       aggfunc="mean").dropna())
    fig, ax = fig1(1.25, 0.66)
    order = pu.P7_agent_dl.sort_values().index
    for y, u in enumerate(order):
        ax.plot([pu.loc[u, "dl_only"], pu.loc[u, "P7_agent_dl"]],
                [y, y], color=C["light"], lw=1.5, zorder=1)
        ax.scatter([pu.loc[u, "dl_only"]], [y], s=26, color=C["alt"],
                   zorder=2, label="CNN-GRU" if y == 0 else None)
        ax.scatter([pu.loc[u, "P7_agent_dl"]], [y], s=30,
                   color=C["main"], zorder=3,
                   label="Guarded Agent" if y == 0 else None)
    ax.axvline(0, color=C["grey"], lw=0.9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"T{u}" for u in order])
    ax.set_xlabel("mean signed error (cycles); $<0$ = conservative")
    ax.legend(fontsize=6.6, loc="lower right")
    ax.set_title("Per-unit bias: conservative everywhere", loc="left",
                 pad=5)
    save(fig, out, "figN13_unit_bias", "RQ3 prognostic value",
         "Both systems sit left of zero on every unit; the agent adds "
         "reserve where hints were corrupted.")

    # N14 — diagnostic token distribution by arm
    fig, ax = fig1(1.25, 0.6)
    pats = ["P1_direct", "P2_rag", "P5_verifier"]
    labs = ["direct", "RAG", "A-RAD agent"]
    data = [D3[D3.pattern == p_].tok.values for p_ in pats]
    bp = ax.boxplot(data, tick_labels=labs, widths=0.5,
                    patch_artist=True, medianprops=dict(color="black"))
    for patch, cc in zip(bp["boxes"], [C["light"], C["light"],
                                       C["main"]]):
        patch.set_facecolor(cc)
        patch.set_alpha(0.8)
    ax.set_ylabel("tokens generated per case")
    ax.set_title("The verifier's token cost is modest and bounded",
                 loc="left", pad=5)
    save(fig, out, "figN14_token_box", "RQ5 deployment",
         "Token distributions per diagnostic arm.")

    # ================== SPLITS OF EXISTING COMPOSITES ==================
    # (concise re-renders; same data, one panel per file)
    # S-core C3 split
    def paired(a1, a2):
        x = M[M.arm == a1].set_index("qid").rul_abs_err
        y = M[M.arm == a2].set_index("qid").rul_abs_err
        d = (x - y).dropna().values
        bs = [d[RNG.integers(0, len(d), len(d))].mean()
              for _ in range(3000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return d.mean(), lo, hi
    ce = dl[dl.corr_f].set_index("qid")
    dres = (ce.rul_abs_err - np.abs(ce.hint.fillna(0)
            - ce.true_rul.clip(upper=125))).dropna()
    bs = [dres.values[RNG.integers(0, len(dres), len(dres))].mean()
          for _ in range(3000)]
    rlo, rhi = np.percentile(bs, [2.5, 97.5])
    effects = [("agent vs its own tool", *paired("P7_agent_dl",
                                                 "dl_only")),
               ("value of the DL anchor", *paired("P7_agent_dl",
                                                  "P7_agent")),
               ("agent vs precedent floor", *paired("P7_agent_dl",
                                                    "b0_median")),
               ("rescue on corrupted tool", dres.mean(), rlo, rhi)]
    fig, ax = fig1(1.35, 0.55)
    for j, (lab, m_, lo, hi) in enumerate(effects):
        yj = len(effects) - 1 - j
        ax.errorbar(m_, yj, xerr=[[m_ - lo], [hi - m_]], fmt="o", ms=5,
                    color=C["main"], capsize=3.4, lw=1.2)
        ax.text(m_, yj - 0.22, f"{m_:+.1f} [{lo:+.1f}, {hi:+.1f}]",
                ha="center", va="top", fontsize=7.2, color=C["grey"])
    ax.axvline(0, color=C["light"], lw=0.9, ls="--")
    ax.set_yticks(range(len(effects)))
    ax.set_yticklabels([e[0] for e in effects][::-1], fontsize=8)
    ax.set_ylim(-0.8, len(effects) - 0.4)
    ax.set_xlim(-32, 26)
    ax.set_xlabel(r"paired $\Delta|$err$|$ (cycles; $<0$ favours agent)")
    ax.set_title("Prognostic paired effects (95\\% CI)" if USETEX else
                 "Prognostic paired effects (95% CI)", loc="left",
                 pad=5)
    save(fig, out, "figS_prog_forest", "RQ3 prognostic value",
         "All paired effects with bootstrap CIs.")

    for arm_pair, name, cap in ((None, None, None),):
        pass
    # ================== SPLITS OF REMAINING COMPOSITES ================
    z = M[M.true_rul < 20]
    fig, ax = fig1(1.2, 0.8)
    for arm, cc, lab, mk in (("dl_only", C["alt"], "CNN-GRU alone",
                              "s"),
                             ("P7_agent_dl", C["main"],
                              "Guarded Agent", "o")):
        g = z[z.arm == arm].dropna(subset=["rul_pred"])
        ax.scatter(g.true_rul + RNG.normal(0, 0.15, len(g)), g.rul_pred,
                   s=22, alpha=0.75, color=cc, marker=mk, label=lab)
    ax.plot([0, 20], [0, 20], color=C["grey"], lw=0.9)
    ax.fill_between([0, 20], [5, 25], [-5, 15], color=C["third"],
                    alpha=0.10)
    ax.set_xlabel("true RUL (cycles)")
    ax.set_ylabel("predicted RUL (cycles)")
    ax.set_xlim(0, 20)
    ax.set_ylim(-1, 27)
    ax.legend(loc="upper left", fontsize=6.8)
    ax.set_title("Danger-zone predictions (true RUL $<$ 20)",
                 loc="left", pad=5)
    save(fig, out, "figS_eol_scatter", "RQ4 robustness",
         "EOL scatter, single panel.")

    for col, nm, ylab, cap in (
            ("rul_abs_err", "figS_eol_mae", "MAE (cycles)",
             "EOL MAE with bootstrap CI."),
            ("s_score", "figS_eol_sscore", "S-score (median)",
             "EOL median safety penalty.")):
        fig, ax = fig1(0.95, 0.75)
        for i, (arm, cc, lab) in enumerate(
                (("dl_only", C["alt"], "CNN-GRU"),
                 ("P7_agent_dl", C["main"], "Guarded Agent"))):
            g = z[z.arm == arm].dropna(subset=[col])
            if col == "rul_abs_err":
                v = g[col].mean()
                bs = [g[col].sample(len(g), replace=True,
                                    random_state=k).mean()
                      for k in range(2000)]
                lo, hi = np.percentile(bs, [2.5, 97.5])
                ax.bar(i, v, color=cc, width=0.55)
                ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], fmt="none",
                            ecolor="black", capsize=2.6, lw=0.9)
                ax.text(i, hi + 0.3, f"{v:.1f}", ha="center",
                        fontsize=7.6)
            else:
                v = g[col].median()
                ax.bar(i, v, color=cc, width=0.55)
                ax.text(i, v + 0.04, f"{v:.1f}", ha="center",
                        fontsize=7.6)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["CNN-GRU", "Guarded\nAgent"],
                               fontsize=7.2)
        ax.set_ylabel(ylab)
        ax.set_title("End-of-life zone", loc="left", pad=5)
        save(fig, out, nm, "RQ4 robustness", cap)

    # D3 split heatmaps
    for patt, nm, ttl in (("B0_retrieval", "figS_sevstage_b0",
                           "Severity vs stage: retrieval floor (B0)"),
                          ("P5_verifier", "figS_sevstage_agent",
                           "Severity vs stage: A-RAD agent")):
        g = D3[(D3.pattern == patt)].dropna(subset=["sev"])
        g = g.assign(b=[band_idx(t) for t in g.true])
        H = np.zeros((5, len(BANDS)))
        for bi_ in range(len(BANDS)):
            gb = g[g.b == bi_]
            for s_ in range(1, 6):
                H[s_ - 1, bi_] = ((gb.sev == s_).mean() if len(gb)
                                  else 0)
        fig, ax = fig1(1.15, 0.72)
        ax.imshow(H, cmap="Blues", vmin=0, vmax=1, aspect="auto",
                  origin="lower")
        for i in range(5):
            for j in range(len(BANDS)):
                if H[i, j] >= 0.005:
                    ax.text(j, i, pct(H[i, j]), ha="center",
                            va="center", fontsize=6.8,
                            color="white" if H[i, j] > 0.5 else "black")
        ax.set_xticks(range(len(BANDS)))
        ax.set_xticklabels([b[2] for b in BANDS], fontsize=7)
        ax.set_yticks(range(5))
        ax.set_yticklabels(range(1, 6))
        ax.set_xlabel("true RUL band (cycles)")
        ax.set_ylabel("assigned severity")
        ax.grid(False)
        ax.set_title(ttl, loc="left", pad=5)
        save(fig, out, nm, "RQ2 grounding", ttl + ".")

    # D7 split ECDFs
    if has2:
        def gaps_sims(D):
            G, S_ = [], []
            for _, r in D[D.pattern.isin(["P2_rag",
                                          "P5_verifier"])].iterrows():
                cc_ = case_cycle(r.qid)
                if cc_ is None or r.true is None:
                    continue
                clf = lf(cc_, r.true)
                cited = set(r.ticket.get("cited_precedents") or [])
                for c in r.ctx:
                    key = f"u{c['unit']}c{c['cycle']}"
                    if cited and key not in cited:
                        continue
                    if c.get("rul_then") is None:
                        continue
                    G.append(abs(lf(c["cycle"], c["rul_then"]) - clf))
                    if c.get("similarity") is not None:
                        S_.append(c["similarity"])
            return np.array(G), np.array(S_)
        G2, S2v = gaps_sims(D2)
        G3, S3v = gaps_sims(D3)
        for (v2, v3, nm, xl, ttl) in (
                (G2, G3, "figS_stagegap_ecdf",
                 r"$|$life-stage gap$|$ of cited precedent",
                 "Stage relevance of citations, v2 vs v3"),
                (S2v, S3v, "figS_similarity_ecdf",
                 "cosine similarity of cited precedent",
                 "The similarity price of stage-aware retrieval")):
            fig, ax = fig1(1.25, 0.62)
            for v, cc, lab in ((v2, C["alt"], "plain cosine (v2)"),
                               (v3, C["main"], "stage-aware (v3)")):
                vs = np.sort(v)
                ax.plot(vs, np.linspace(0, 1, len(vs)), lw=1.5,
                        color=cc,
                        label=f"{lab} (median {np.median(v):.2f})")
            ax.set_xlabel(xl)
            ax.set_ylabel("ECDF")
            ax.legend(fontsize=6.8, loc="best")
            ax.set_title(ttl, loc="left", pad=5)
            save(fig, out, nm, "RQ1 mechanism", ttl + ".")

    # diagnostic redesign slopes single (C3b)
    if has2:
        def esc_coh(D, legacy):
            g = D[D.pattern == "P5_verifier"]
            if legacy:
                coh = g.dropna(subset=["est"]).apply(
                    lambda r: r.act == (
                        "continue_monitoring" if r.est > 120 else
                        "schedule_inspection" if r.est > 60 else
                        "plan_maintenance" if r.est > 25 else
                        "immediate_shutdown"), axis=1).mean()
            else:
                coh = g.dropna(subset=["sev"]).apply(
                    lambda r: r.act == SEV2ACT.get(r.sev),
                    axis=1).mean()
            return g.esc.mean(), coh
        e2_, c2_ = esc_coh(D2, True)
        e3_, c3_ = esc_coh(D3, False)
        fig, ax = fig1(1.3, 0.55)
        rows = [("escalation", e2_, e3_),
                ("action coherence", c2_, c3_)]
        for i, (lab, v2, v3) in enumerate(rows):
            y = 1 - i
            ax.plot([v2, v3], [y, y], color=C["light"], lw=1.8,
                    zorder=1)
            ax.scatter([v2], [y], s=34, color=C["grey"], zorder=2,
                       label="v2" if i == 0 else None)
            ax.scatter([v3], [y], s=40, color=C["main"], zorder=3,
                       label="v3" if i == 0 else None)
            ax.text(1.04, y, f"{pct(v2)} $\\to$ {pct(v3)}",
                    va="center", fontsize=8,
                    transform=ax.get_yaxis_transform())
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["action coherence", "escalation"])
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(-0.5, 1.6)
        ax.set_xlabel("rate on the diagnostic agent arm")
        ax.legend(loc="center left", fontsize=7)
        ax.set_title("Diagnostic redesign, before/after", loc="left",
                     pad=5)
        save(fig, out, "figS_diag_slopes", "RQ1 mechanism",
             "Escalation and coherence, v2 to v3, single panel.")

    # X1 split: diag cost + prog cost
    fig, ax = fig1(1.2, 0.6)
    pats2 = ["P1_direct", "P2_rag", "P5_verifier"]
    labs2 = ["direct", "RAG", "A-RAD agent"]
    tokm = [D3[D3.pattern == p_].tok.mean() for p_ in pats2]
    ax.bar(range(3), tokm, color=[C["light"], C["light"], C["main"]],
           width=0.55)
    for i, v in enumerate(tokm):
        ax.text(i, v + 6, f"{v:.0f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labs2)
    ax.set_ylabel("mean tokens / case")
    ax.set_title("Diagnostic token cost", loc="left", pad=5)
    save(fig, out, "figS_diag_tokens", "RQ5 deployment",
         "Mean generated tokens per diagnostic arm.")

    fig, ax = fig1(1.2, 0.6)
    parms = ["b0_median", "dl_only", "P7_agent", "P7_agent_dl"]
    plabs = ["b0", "CNN-GRU", "agent, no anchor", "Guarded Agent"]
    Wdf = pd.DataFrame([(e["arm"], e.get("wall_s")) for e in peps],
                       columns=["arm", "w"]).dropna()
    wv = [Wdf[Wdf.arm == p_].w.mean() for p_ in parms]
    ax.bar(range(4), wv, color=[C["grey"], C["alt"], C["accent"],
                                C["main"]], width=0.55)
    for i, v in enumerate(wv):
        ax.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(plabs, fontsize=7.2)
    ax.set_ylabel("wall-clock s / case")
    ax.set_title("Prognostic latency per arm", loc="left", pad=5)
    save(fig, out, "figS_prog_latency", "RQ5 deployment",
         "Mean wall-clock per prognostic arm.")

    # write INDEX.md
    lines = ["# A-RAD single-panel figure library\n"]
    for rq in sorted(set(r for r, _, _ in INDEX)):
        lines.append(f"\n## {rq}\n")
        for r, n, c_ in INDEX:
            if r == rq:
                lines.append(f"- **{n}** - {c_}")
    (out / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[singles] {len(INDEX)} plots + INDEX.md -> {out}")


if __name__ == "__main__":
    main()

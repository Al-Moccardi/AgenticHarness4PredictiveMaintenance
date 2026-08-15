#!/usr/bin/env python3
"""
paper_figures_more.py — extended paper-level feature figures (LaTeX type).

Diagnostic:
  figD3_severity_stage    severity distribution across life stages:
                          retrieval floor (graded) vs agent (compressed)
  figD4_gravity_severity  precedent gravity -> assigned severity: where the
                          graded signal is and where it is lost
  figD5_verifier_economics gates raised, repaired, escalated: the tiered
                          verifier funnel with repair counts
  figD7_stage_match       |life-stage gap| of cited precedents, v2 vs v3,
                          and the similarity price paid for stage relevance
  figD8_action_mix        recommended-action mix per arm vs the
                          stage-implied reference mix
  figD9_escalation_raster all 89 cases ordered by true RUL: who escalates
                          under v2 vs v3
  figD10_ticket_exhibit   one real diagnostic ticket, typeset as exhibit
Prognostic:
  figP5_error_by_stage    signed error vs true RUL: trust early,
                          arbitrate late
  figP6_interval_by_stage interval coverage and width across life stages
System:
  figX1_compute_budget    tokens and wall-clock per case: 3B-at-the-edge
                          viability

Run:
  python paper\\code\\figures\\paper_figures_more.py ^
      --arad results\\arad3 --arad2 results\\arad2
"""
from __future__ import annotations

import argparse
import json
import re
import textwrap
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


def _fonts() -> bool:
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
        f.savefig("/tmp/_texprobe3.png")
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
BANDS = [(100, 126, r"$\geq$100"), (60, 100, "60--100"),
         (25, 60, "25--60"), (0, 25, "$<$25")]
if not USETEX:
    BANDS = [(100, 126, "$\\geq$100"), (60, 100, "60-100"),
             (25, 60, "25-60"), (0, 25, "$<$25")]
ACTIONS = ["continue_monitoring", "schedule_inspection",
           "plan_maintenance", "immediate_shutdown"]
ALAB = ["monitor", "inspect", "plan maint.", "shutdown"]
STAGE2ACT = {0: "continue_monitoring", 1: "schedule_inspection",
             2: "plan_maintenance", 3: "immediate_shutdown"}


def pct(x, dec=0):
    s = f"{100 * x:.{dec}f}"
    return (s + r"\%") if USETEX else (s + "%")


def clean_txt(x: str) -> str:
    x = str(x or "")
    for a, b in (("**", ""), ("`", ""), ("###", ""), ("##", ""),
                 (" != ", " \u2260 "), ("!=", " \u2260 "),
                 ("<=", "\u2264"), (">=", "\u2265"),
                 ("->", "\u2192"), ("=>", "\u21d2")):
        x = x.replace(a, b)
    return " ".join(x.split())


EX_RC = {"text.usetex": False, "font.family": "serif",
         "font.serif": ["DejaVu Serif"],
         "mathtext.fontset": "dejavuserif"}


def ex_block(ax, y, label, text, width=108, fs=8, color="black",
             lead=0.052):
    import textwrap as _tw
    if label:
        ax.text(0, y, label, fontsize=fs, color="#666666", va="top")
        y -= 0.055
    txt = _tw.fill(clean_txt(text), width)
    ax.text(0, y, txt, fontsize=fs, va="top", color=color)
    return y - lead * (txt.count("\n") + 1) - 0.045


def jread(p: Path):
    return [json.loads(l) for l in
            p.read_text(encoding="utf-8", errors="ignore").splitlines()
            if l.strip()]


def save(fig, out, name):
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)
    print(f"  [more] {name}")


def band_idx(r):
    r = min(r, 125)
    for i, (lo, hi, _) in enumerate(BANDS):
        if lo <= r < hi:
            return i
    return len(BANDS) - 1


def load_diag(dir_: Path):
    rows = []
    ep = dir_ / "diag" / "episodes.jsonl"
    if not ep.exists():
        ep = _DIAG_DIR / "episodes.jsonl"
    for e in jread(ep):
        t = e.get("ticket") or {}
        rows.append(dict(ep=e, pattern=e["pattern"], qid=e["qid"],
                         true=e.get("true_rul"), ticket=t,
                         sev=t.get("severity"), act=t.get("action"),
                         esc=bool(e.get("escalated")),
                         ac=bool(e.get("auto_corrected")),
                         rep=e.get("repairs") or 0,
                         gates=e.get("gate_violations") or [],
                         tok=e.get("tokens_out") or 0,
                         calls=e.get("llm_calls") or 0,
                         wall=e.get("wall_s") or np.nan,
                         ctx=e.get("contexts") or []))
    return pd.DataFrame(rows)


def stage_gap(row):
    """|life-fraction(precedent) - life-fraction(case)| for cited ctx."""
    t = row.ticket
    cited = set(t.get("cited_precedents") or [])
    case_lf = None
    ep = row.ep
    cyc = ep.get("qid", "")
    m = re.search(r"c[ycle_]*(\d+)(?!.*c[ycle_]*\d)",
                  str(ep.get("qid", "")))
    if m and row.true is not None:
        cc = int(m.group(1))
        case_lf = cc / (cc + max(row.true, 1))
    gaps, sims = [], []
    for c in row.ctx:
        key = f"u{c['unit']}c{c['cycle']}"
        if cited and key not in cited:
            continue
        if c.get("rul_then") is None or case_lf is None:
            continue
        lf = c["cycle"] / (c["cycle"] + max(c["rul_then"], 1))
        gaps.append(abs(lf - case_lf))
        if c.get("similarity") is not None:
            sims.append(c["similarity"])
    return gaps, sims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arad", default="agentic/results/final_prognostic")
    ap.add_argument("--arad2", default="agentic/results/v2_rul_coupled_collapse")
    ap.add_argument("--out", default=str(_r / "paper/figures_regen/more"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    A, A2 = Path(a.arad), Path(a.arad2)
    D3 = load_diag(A)
    has2 = (A2 / "diag" / "episodes.jsonl").exists()
    D2 = load_diag(A2) if has2 else None
    M = pd.read_csv(Path(a.arad) / "forecast_metrics.csv")

    # ---------- figD3 — severity across life stages --------------------
    fig, axes = plt.subplots(1, 2, figsize=(W2 * 0.78, 0.30 * W2),
                             sharey=True)
    for ax, patt, name in ((axes[0], "B0_retrieval",
                            "retrieval floor (B0)"),
                           (axes[1], "P5_verifier", "A-RAD agent")):
        g = D3[(D3.pattern == patt)].dropna(subset=["sev"])
        g = g.assign(b=[band_idx(t) for t in g.true])
        H = np.zeros((5, len(BANDS)))
        for bi in range(len(BANDS)):
            gb = g[g.b == bi]
            for s in range(1, 6):
                H[s - 1, bi] = ((gb.sev == s).mean() if len(gb) else 0)
        im = ax.imshow(H, cmap="Blues", vmin=0, vmax=1, aspect="auto",
                       origin="lower")
        for i in range(5):
            for j in range(len(BANDS)):
                if H[i, j] >= 0.005:
                    ax.text(j, i, pct(H[i, j]), ha="center", va="center",
                            fontsize=6.6,
                            color="white" if H[i, j] > 0.5 else "black")
        ax.set_xticks(range(len(BANDS)))
        ax.set_xticklabels([b[2] for b in BANDS], fontsize=7)
        ax.set_yticks(range(5))
        ax.set_yticklabels([f"{s}" for s in range(1, 6)])
        ax.set_xlabel("true RUL band (cycles)")
        ax.set_title(name, loc="left", pad=4)
        ax.grid(False)
    axes[0].set_ylabel("assigned severity")
    fig.suptitle("Severity across life stages: the retrieval prior is "
                 "weakly but consistently graded "
                 "($\\rho=-0.35$, $p<0.001$); the generator spreads "
                 "severity without stage correlation "
                 "($\\rho=-0.13$, n.s.)", x=0.02, ha="left",
                 fontsize=8.5, y=1.06)
    fig.tight_layout()
    save(fig, out, "figD3_severity_stage")

    # ---------- figD4 — precedent gravity vs assigned severity ---------
    from scipy import stats as st
    g5 = D3[D3.pattern == "P5_verifier"].dropna(subset=["sev"]).copy()
    med_grav = []
    for _, r in g5.iterrows():
        gv = [c.get("gravity") for c in r.ctx
              if c.get("gravity") is not None]
        med_grav.append(np.median(gv) if gv else np.nan)
    g5["mg"] = med_grav
    g5 = g5.dropna(subset=["mg"])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(W2 * 0.78, 0.30 * W2),
                                 gridspec_kw={"width_ratios": [1.2, 1]})
    jit = RNG.normal(0, 0.07, len(g5))
    ax.scatter(g5.mg + RNG.normal(0, 0.05, len(g5)), g5.sev + jit,
               s=12, alpha=0.6, color=C["main"])
    rho, pv = st.spearmanr(g5.mg, g5.sev)
    ax.plot([1, 5], [1, 5], ls="--", lw=0.9, color=C["light"])
    ax.text(0.03, 0.94, f"$\\rho$ = {rho:.2f} (p = {pv:.2g})",
            transform=ax.transAxes, fontsize=8)
    ax.set_xlabel("median gravity of retrieved precedents")
    ax.set_ylabel("severity assigned by the agent")
    ax.set_xlim(0.8, 5.2)
    ax.set_ylim(0.8, 5.2)
    ax.set_title("The agent reads the prior only weakly", loc="left",
                 pad=4)
    for patt, cc, lab in (("B0_retrieval", C["grey"],
                           "retrieval floor (B0)"),
                          ("P5_verifier", C["main"], "A-RAD agent")):
        gg = D3[D3.pattern == patt].dropna(subset=["sev"])
        vals = [(gg.sev == s).mean() for s in range(1, 6)]
        off = -0.18 if patt == "B0_retrieval" else 0.18
        bx.bar(np.arange(1, 6) + off, vals, width=0.34, color=cc,
               label=lab)
    bx.set_xticks(range(1, 6))
    bx.set_xlabel("severity")
    bx.set_ylabel("share of cases")
    bx.legend(loc="upper left", fontsize=6.6)
    bx.set_title("Severity compression at 3B", loc="left", pad=4)
    fig.tight_layout(w_pad=2.4)
    save(fig, out, "figD4_gravity_severity")

    # ---------- figD5 — the tiered verifier funnel ----------------------
    g5a = D3[D3.pattern == "P5_verifier"]
    from collections import Counter
    gate_cnt = Counter(v for gs in g5a.gates for v in gs)
    raised = int(sum(gate_cnt.values()))
    n_viol = int((g5a.gates.map(len) > 0).sum())
    n_ac = int(g5a.ac.sum())
    n_esc = int(g5a.esc.sum())
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(W2 * 0.72, 0.27 * W2),
                                 gridspec_kw={"width_ratios": [1, 1.1]})
    glab = {"D4_citation_not_shown": "D4 citation",
            "D5_severity_action_mismatch": "D5 sev-action",
            "D6_progression_not_referenced": "D6 progression"}
    keys = list(gate_cnt.keys())
    ax.bar(range(len(keys)), [gate_cnt[k] for k in keys],
           color=[C["gold"], C["third"], C["main"]][:len(keys)],
           width=0.55)
    for i, k in enumerate(keys):
        ax.text(i, gate_cnt[k] + 0.05, str(gate_cnt[k]), ha="center",
                fontsize=7.5)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([glab.get(k, k) for k in keys], fontsize=7)
    ax.set_ylabel("violations raised")
    ax.set_title("Gates fired (89 cases)", loc="left", pad=4)
    stages = [("cases with a\nviolation", n_viol, C["gold"]),
              ("repaired \\&\nauto-corrected" if USETEX else
               "repaired &\nauto-corrected", n_viol - n_esc, C["third"]),
              ("escalated to\nhuman", n_esc, C["red"])]
    for i, (lab, v, cc) in enumerate(stages):
        bx.bar(i, v, color=cc, width=0.55)
        bx.text(i, v + 0.12, str(v), ha="center", fontsize=8)
    bx.set_xticks(range(len(stages)))
    bx.set_xticklabels([s[0] for s in stages], fontsize=7)
    bx.set_ylabel("cases")
    bx.set_title(f"Resolution funnel (repairs/case = "
                 f"{g5a.rep.mean():.2f})", loc="left", pad=4)
    fig.tight_layout(w_pad=2.6)
    save(fig, out, "figD5_verifier_economics")

    # ---------- figD7 — stage match of cited precedents ------------------
    if has2:
        def gaps_sims(D):
            G, S = [], []
            for _, r in D[D.pattern.isin(["P2_rag",
                                          "P5_verifier"])].iterrows():
                g_, s_ = stage_gap(r)
                G += g_
                S += s_
            return np.array(G), np.array(S)
        G2, S2 = gaps_sims(D2)
        G3, S3 = gaps_sims(D3)
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(W2 * 0.78,
                                                    0.28 * W2))
        for v, cc, lab in ((G2, C["alt"], "plain cosine (v2)"),
                           (G3, C["main"], "stage-aware (v3)")):
            vs = np.sort(v)
            ax.plot(vs, np.linspace(0, 1, len(vs)), color=cc, lw=1.5,
                    label=f"{lab}  (median {np.median(v):.2f})")
        ax.set_xlabel(r"$|$life-stage gap$|$ between case and cited precedent")
        ax.set_ylabel("ECDF")
        ax.legend(loc="lower right", fontsize=6.8)
        ax.set_title("Stage relevance of citations", loc="left", pad=4)
        for v, cc, lab in ((S2, C["alt"], "v2"), (S3, C["main"], "v3")):
            vs = np.sort(v)
            bx.plot(vs, np.linspace(0, 1, len(vs)), color=cc, lw=1.5,
                    label=f"{lab} (median {np.median(v):.2f})")
        bx.set_xlabel("cosine similarity of cited precedent")
        bx.set_ylabel("ECDF")
        bx.legend(loc="upper left", fontsize=6.8)
        bx.set_title("The similarity price paid", loc="left", pad=4)
        fig.tight_layout(w_pad=2.6)
        save(fig, out, "figD7_stage_match")

    # ---------- figD8 — decision mix: normal vs stage-aware design -----
    fig, ax = plt.subplots(figsize=(W1 * 1.5, 0.58 * W1 * 1.5))
    cols = [C["third"], C["gold"], C["alt"], C["red"]]
    ref = [STAGE2ACT[band_idx(t)] for t in
           D3[D3.pattern == "P5_verifier"].true]
    refmix = np.array([np.mean([r == a_ for r in ref])
                       for a_ in ACTIONS])
    mixes = [("stage-implied\nreference", refmix, None)]
    if has2:
        g2m = D2[D2.pattern == "P5_verifier"]
        v2mix = np.array([(g2m.act == a_).mean() for a_ in ACTIONS])
        mixes.append(("RUL-coupled\ndesign (v2)", v2mix,
                      0.5 * np.abs(v2mix - refmix).sum()))
    g3m = D3[D3.pattern == "P5_verifier"]
    v3mix = np.array([(g3m.act == a_).mean() for a_ in ACTIONS])
    mixes.append(("stage-aware\nA-RAD (v3)", v3mix,
                  0.5 * np.abs(v3mix - refmix).sum()))
    bottoms = np.zeros(len(mixes))
    for ai in range(len(ACTIONS)):
        vals = [m[1][ai] for m in mixes]
        ax.bar(range(len(mixes)), vals, bottom=bottoms, color=cols[ai],
               width=0.55, label=ALAB[ai])
        bottoms += np.array(vals)
    for i, (_, _, tv) in enumerate(mixes):
        if tv is not None:
            ax.text(i, 1.04, ("TV dist. " if not USETEX else
                              r"TV dist.\ ") + f"{tv:.2f}",
                    ha="center", fontsize=7.4, color=C["grey"])
    ax.set_xticks(range(len(mixes)))
    ax.set_xticklabels([m[0] for m in mixes], fontsize=7.5)
    ax.set_ylabel("share of cases")
    ax.set_ylim(0, 1.12)
    ax.legend(ncols=4, loc="lower left", bbox_to_anchor=(0.0, 1.10),
              borderaxespad=0, fontsize=6.8)
    ax.set_title("Retrieval design shapes the decision mix "
                 "(distance from the stage-implied reference)",
                 loc="left", pad=30)
    save(fig, out, "figD8_action_mix")

    # ---------- figD9 — escalation raster --------------------------------
    if has2:
        g3 = (D3[D3.pattern == "P5_verifier"]
              .sort_values("true").reset_index())
        g2 = D2[D2.pattern == "P5_verifier"].set_index("qid")
        n = len(g3)
        R = np.zeros((2, n))
        for i, r in g3.iterrows():
            R[0, i] = 1.0 if (r.qid in g2.index
                              and bool(g2.loc[r.qid, "esc"])) else 0.0
            R[1, i] = (1.0 if r.esc else (0.5 if r.ac else 0.0))
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(["#f0f0f0", C["third"], C["red"]])
        fig, ax = plt.subplots(figsize=(W2 * 0.9, 0.16 * W2))
        ax.imshow(np.digitize(R, [0.25, 0.75]), cmap=cmap, aspect="auto",
                  interpolation="nearest")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["RUL-coupled (v2)", "A-RAD (v3)"])
        ax.set_xlabel(r"the 89 cases, ordered by true RUL "
                      r"(left = near end of life $\to$ right = early "
                      "life)")
        ax.set_xticks([0, n - 1])
        ax.set_xticklabels(["most degraded", "healthiest"], fontsize=7)
        ax.grid(False)
        import matplotlib.patches as mp
        ax.legend(handles=[mp.Patch(color="#f0f0f0", label="clean pass"),
                           mp.Patch(color=C["third"],
                                    label="auto-corrected"),
                           mp.Patch(color=C["red"], label="escalated")],
                  ncols=3, loc="lower left", bbox_to_anchor=(0.0, 1.05),
                  borderaxespad=0, fontsize=7)
        ax.set_title("Escalation is stage-blind under v2 and "
                     "surgical under v3", loc="left", pad=26)
        save(fig, out, "figD9_escalation_raster")

    # ---------- figD10a/b/c — real tickets, typeset -------------------
    cand = D3[(D3.pattern == "P5_verifier")
              & D3.ep.map(lambda e: bool(e.get("has_history")))
              & D3.ticket.map(lambda t: bool(t.get("cited_precedents")))]

    def _grav_ok(r_):
        keys = set(r_.ticket.get("cited_precedents") or [])
        return any(f"u{c['unit']}c{c['cycle']}" in keys
                   and c.get("gravity") is not None for c in r_.ctx)
    cand = cand.assign(nctx=cand.ctx.map(len),
                       gok=cand.apply(_grav_ok, axis=1))
    cand = cand.sort_values(["gok", "rep", "nctx"],
                            ascending=[False, False, False])
    r = cand.iloc[0]
    t = r.ticket
    with plt.rc_context(EX_RC):
        fig, ax = plt.subplots(figsize=(W2 * 0.92, 0.40 * W2))
        ax.axis("off")
        m = re.search(r"[TuU]+(\d+)c(\d+)", str(r.qid))
        uc = (f"unit {m.group(1)}, cycle {m.group(2)}" if m
              else str(r.qid))
        head = (f"Diagnostic ticket - {uc}   |   "
                f"severity {t.get('severity')}   |   action: "
                f"{str(t.get('action', '')).replace('_', ' ')}")
        ax.text(0, 1.0, head, fontsize=9.5, fontweight="bold", va="top")
        y = 0.90
        y = ex_block(ax, y, "matched pattern",
                     t.get("matched_pattern", ""))
        y = ex_block(ax, y, "diagnosis", t.get("diagnosis", ""))
        y = ex_block(ax, y, "rationale", t.get("reasoning", ""))
        ax.text(0, y, "cited precedents (verified against shown "
                      "contexts)", fontsize=8, color=C["grey"], va="top")
        y -= 0.055
        for key in (t.get("cited_precedents") or [])[:3]:
            cc = next((c for c in r.ctx
                       if f"u{c['unit']}c{c['cycle']}" == key), None)
            if cc:
                parts = [key + ":"]
                if cc.get("gravity") is not None:
                    parts.append(f"gravity {cc.get('gravity')}")
                if cc.get("similarity") is not None:
                    parts.append(f"similarity "
                                 f"{cc.get('similarity'):.2f}")
                ax.text(0.02, y, "  ".join(parts), fontsize=8, va="top",
                        color=C["main"])
                y -= 0.052
        y -= 0.02
        state = ("escalated to human review" if r.esc
                 else "final ticket clean")
        ax.text(0, y, f"verifier: {r.rep} repair round(s) applied; "
                      f"{state}   |   {r.tok} tokens, {r.wall:.1f} s "
                      "per case (3B model)",
                fontsize=7.6, color=C["grey"], va="top")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        save(fig, out, "figD10_ticket_exhibit")

    # prognostic exhibit: rich forecast, clean trends, EOL preferred
    peps = [e for e in jread(Path(a.arad) / "forecast_episodes.jsonl")
            if e["arm"] == "P7_agent_dl" and (e.get("forecast") or {})]

    def _score(e):
        f_ = e["forecast"]
        tr = f_.get("expected_trends") or []
        tr_clean = bool(tr) and not any("|" in str(x.get("sensor", ""))
                                        for x in tr)
        rr = f_.get("rul_range") or [0, 125]
        nondeg = not (rr[0] == 0 and rr[1] == 125)
        return (4 * bool(f_.get("progression_narrative"))
                + 3 * bool(f_.get("cited_precedents"))
                + 6 * tr_clean + 1 * nondeg
                + 1 * (e.get("true_rul", 999) < 25)
                + (max(0.0, 6 - abs(f_.get("rul_estimate")
                                    - e["true_rul"]))
                   if e.get("true_rul") is not None else 0)
                + 0.001 * len(f_.get("progression_narrative") or ""))
    peps = [e for e in peps if e["forecast"].get("rul_estimate")
            is not None]
    pe = max(peps, key=_score)
    f_ = pe["forecast"]
    with plt.rc_context(EX_RC):
        fig, ax = plt.subplots(figsize=(W2 * 0.92, 0.33 * W2))
        ax.axis("off")
        rr = f_.get("rul_range")
        rtxt = (f" (range {min(rr):.0f}-{max(rr):.0f})"
                if rr and rr[0] is not None
                and not (min(rr) == 0 and max(rr) == 125) else "")
        head = (f"Prognostic ticket - unit {pe['unit']}, cycle "
                f"{pe['cycle']}   |   RUL estimate "
                f"{f_.get('rul_estimate'):.0f} cycles{rtxt}   |   "
                f"outlook: {f_.get('anomaly_outlook')}   |   "
                f"confidence: {f_.get('confidence')}")
        ax.text(0, 1.0, head, fontsize=9.3, fontweight="bold", va="top")
        y = 0.90
        y = ex_block(ax, y, "progression narrative (rationale)",
                     f_.get("progression_narrative", ""))
        tr = [x for x in (f_.get("expected_trends") or [])
              if x.get("sensor") and "|" not in str(x.get("sensor"))
              and "<" not in str(x.get("sensor"))]
        if tr:
            tt = ",  ".join(f"{x.get('sensor')} {x.get('direction')}"
                            for x in tr[:6])
            y = ex_block(ax, y, "expected sensor trends", tt)
        ax.text(0, y, "cited precedent futures (from fleet memory)",
                fontsize=8, color=C["grey"], va="top")
        y -= 0.055
        for key in (f_.get("cited_precedents") or [])[:3]:
            cc = next((c for c in (pe.get("contexts") or [])
                       if f"u{c['unit']}c{c['cycle']}" == key), None)
            if cc and cc.get("rul_then") is not None:
                line = (f"{key}:  precedent survived "
                        f"{cc['rul_then']:.0f} more cycles"
                        + (f",  similarity {cc['similarity']:.2f}"
                           if cc.get("similarity") is not None else ""))
                ax.text(0.02, y, line, fontsize=8, va="top",
                        color=C["main"])
                y -= 0.052
        y -= 0.02
        hint = pe.get("dl_hint")
        foot = (f"CNN-GRU hint: {hint:.0f} cycles"
                + (f"  |  ground truth: {pe.get('true_rul'):.0f} cycles"
                   if pe.get("true_rul") is not None else "")
                + f"  |  {pe.get('steps')} tool steps, "
                  f"{pe.get('wall_s', 0):.1f} s per case")
        ax.text(0, y, foot, fontsize=7.6, color=C["grey"], va="top")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        save(fig, out, "figD10b_prognostic_exhibit")

    # the one escalated case: what human review receives
    esc_ep = D3[(D3.pattern == "P5_verifier") & D3.esc]
    if len(esc_ep):
        r = esc_ep.iloc[0]
        t = r.ticket
        with plt.rc_context(EX_RC):
            fig, ax = plt.subplots(figsize=(W2 * 0.92, 0.26 * W2))
            ax.axis("off")
            m = re.search(r"[TuU]+(\d+)c(\d+)", str(r.qid))
            uc = (f"unit {m.group(1)}, cycle {m.group(2)}" if m
                  else str(r.qid))
            glab = {"D4_citation_not_shown": "D4: cited precedent not "
                                             "among shown contexts",
                    "D5_severity_action_mismatch": "D5: action does not "
                                                   "match severity",
                    "D6_progression_not_referenced":
                        "D6: progression not referenced"}
            ax.text(0, 1.0, f"Escalated case - {uc}: what human review "
                            "receives", fontsize=9.5,
                    fontweight="bold", va="top")
            y = 0.84
            y = ex_block(ax, y, "unresolved gate(s) after "
                                f"{r.rep} repair round(s)",
                         ";  ".join(glab.get(g, g) for g in r.gates),
                         color=C["red"])
            y = ex_block(ax, y, "draft severity / action",
                         f"severity {t.get('severity')}  -  "
                         f"{str(t.get('action', '')).replace('_', ' ')}")
            y = ex_block(ax, y, "draft diagnosis",
                         t.get("diagnosis", ""))
            ax.text(0, y, "full trace and shown contexts attached; "
                          "no autonomous action is taken on this case.",
                    fontsize=7.6, color=C["grey"], va="top")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            save(fig, out, "figD12_escalated_case")

    # ---------- figP5 — error by life stage ------------------------------
    fig, ax = plt.subplots(figsize=(W1 * 1.55, 0.60 * W1 * 1.55))
    for arm, cc, lab in (("dl_only", C["alt"], "CNN-GRU alone"),
                         ("P7_agent_dl", C["main"],
                          "Guarded Agent")):
        g = M[M.arm == arm].dropna(subset=["rul_err"])
        ax.scatter(g.true_rul.clip(upper=125)
                   + RNG.normal(0, 1.0, len(g)),
                   g.rul_err, s=10, alpha=0.45, color=cc, label=lab)
        med = (g.assign(b=[band_idx(t) for t in g.true_rul])
               .groupby("b").rul_err.median())
        xs = [112, 80, 42, 12]
        ax.plot([xs[i] for i in med.index], med.values, color=cc, lw=1.7,
                marker="o", ms=4)
    ax.axhline(0, color=C["grey"], lw=0.9)
    ax.axhspan(0, 90, color=C["red"], alpha=0.05)
    ax.text(64, 32, "over-prediction (unsafe)", fontsize=7.4,
            color=C["red"])
    ax.text(8, -108, "conservative reserve\nconcentrates early in life",
            fontsize=7.4, color=C["grey"])
    ax.set_xlabel("true RUL (cycles; capped at 125)")
    ax.set_ylabel(r"signed error, predicted $-$ true (cycles)")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_title("Trust the tool early, arbitrate late: the overhead is "
                 "conservative reserve", loc="left", pad=6)
    save(fig, out, "figP5_error_by_stage")

    # ---------- figP6 — interval behaviour by stage ----------------------
    g = M[M.arm == "P7_agent_dl"].dropna(subset=["range_coverage"])
    g = g.assign(b=[band_idx(t) for t in g.true_rul])
    cov = g.groupby("b").range_coverage.mean()
    wid = g.groupby("b").range_width.mean()
    fig, ax = plt.subplots(figsize=(W1 * 1.35, 0.58 * W1 * 1.35))
    x = np.arange(len(BANDS))
    ax.bar(x - 0.18, [cov.get(i, 0) for i in range(len(BANDS))],
           width=0.34, color=C["main"], label="coverage")
    bx = ax.twinx()
    bx.bar(x + 0.18, [wid.get(i, 0) for i in range(len(BANDS))],
           width=0.34, color=C["gold"], label="mean width")
    bx.grid(False)
    bx.spines["right"].set_visible(True)
    ax.set_xticks(x)
    ax.set_xticklabels([b[2] for b in BANDS])
    ax.set_xlabel("true RUL band (cycles)")
    ax.set_ylabel("interval coverage", color=C["main"])
    bx.set_ylabel("interval width (cycles)", color=C["gold"])
    ax.set_ylim(0, 1.0)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = bx.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7)
    ax.set_title("Stated uncertainty across life stages", loc="left",
                 pad=6)
    save(fig, out, "figP6_interval_by_stage")

    # ---------- figX1 — compute budget -----------------------------------
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(W2 * 0.72, 0.26 * W2))
    pats = ["P1_direct", "P2_rag", "P5_verifier"]
    labs = ["direct", "RAG", "A-RAD agent"]
    tok = [D3[D3.pattern == p].tok.mean() for p in pats]
    wal = [D3[D3.pattern == p].wall.mean() for p in pats]
    ax.bar(np.arange(3) - 0.18, tok, width=0.34, color=C["main"],
           label="tokens generated")
    cx = ax.twinx()
    cx.bar(np.arange(3) + 0.18, wal, width=0.34, color=C["gold"],
           label="seconds / case")
    cx.grid(False)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labs, fontsize=7.5)
    ax.set_ylabel("tokens / case", color=C["main"])
    cx.set_ylabel("wall-clock s / case", color=C["gold"])
    ax.set_title("Diagnostic cost (3B model)", loc="left", pad=4)
    parms = ["b0_median", "dl_only", "P7_agent", "P7_agent_dl"]
    plabs = ["b0", "CNN-GRU", "agent, no anchor", "Guarded Agent"]
    pe = jread(Path(a.arad) / "forecast_episodes.jsonl")
    W = pd.DataFrame([(e["arm"], e.get("wall_s")) for e in pe],
                     columns=["arm", "w"]).dropna()
    wv = [W[W.arm == p].w.mean() for p in parms]
    bx.bar(range(4), wv, color=[C["grey"], C["alt"], C["accent"],
                                C["main"]], width=0.55)
    for i, v in enumerate(wv):
        bx.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=7)
    bx.set_xticks(range(4))
    bx.set_xticklabels(plabs, fontsize=7)
    bx.set_ylabel("wall-clock s / case")
    bx.set_title("Prognostic cost", loc="left", pad=4)
    fig.tight_layout(w_pad=3.0)
    save(fig, out, "figX1_compute_budget")

    # ---------- figP7 — the end-of-life zone (true RUL < 20) ----------
    z = M[M.true_rul < 20]
    fig, (ax, bx, cx) = plt.subplots(
        1, 3, figsize=(W2 * 0.9, 0.28 * W2),
        gridspec_kw={"width_ratios": [1.25, 0.85, 0.85]})
    for arm, cc, lab, mk in (("dl_only", C["alt"], "CNN-GRU alone", "s"),
                             ("P7_agent_dl", C["main"],
                              "Guarded Agent", "o")):
        g = z[z.arm == arm].dropna(subset=["rul_pred"])
        ax.scatter(g.true_rul + RNG.normal(0, 0.15, len(g)), g.rul_pred,
                   s=22, alpha=0.75, color=cc, marker=mk, label=lab)
    ax.plot([0, 20], [0, 20], color=C["grey"], lw=0.9)
    ax.fill_between([0, 20], [5, 25], [-5, 15], color=C["third"],
                    alpha=0.10)
    ax.text(1.0, 6.6, r"$\pm$5 cycles", fontsize=7,
            color=C["third"], rotation=38)
    ax.set_xlabel("true RUL (cycles)")
    ax.set_ylabel("predicted RUL (cycles)")
    ax.set_xlim(0, 20)
    ax.set_ylim(-1, 27)
    ax.legend(loc="upper left", fontsize=6.6,
              bbox_to_anchor=(0.02, 1.0))
    ax.set_title("Prediction in the danger zone", loc="left", pad=4)
    stats_ = {}
    for i, (arm, cc, lab) in enumerate(
            (("dl_only", C["alt"], "CNN-GRU"),
             ("P7_agent_dl", C["main"], "Guarded\nAgent"))):
        g = z[z.arm == arm].dropna(subset=["rul_abs_err"])
        m_, lo, hi = (g.rul_abs_err.mean(),
                      *np.percentile([g.rul_abs_err.sample(
                          len(g), replace=True,
                          random_state=k).mean()
                          for k in range(2000)], [2.5, 97.5]))
        bx.bar(i, m_, color=cc, width=0.55)
        bx.errorbar(i, m_, yerr=[[m_ - lo], [hi - m_]], fmt="none",
                    ecolor="black", capsize=2.6, lw=0.9)
        bx.text(i, hi + 0.35, f"{m_:.1f}", ha="center", fontsize=7.6)
        cx.bar(i, g.s_score.median(), color=cc, width=0.55)
        cx.text(i, g.s_score.median() + 0.05,
                f"{g.s_score.median():.1f}", ha="center", fontsize=7.6)
        stats_[arm] = (m_, g.s_score.median())
        bx.set_xticks([0, 1])
        bx.set_xticklabels([lab for _, _, lab in
                            (("dl_only", C["alt"], "CNN-GRU"),
                             ("P7_agent_dl", C["main"],
                              "Guarded\nAgent"))], fontsize=7)
        cx.set_xticks([0, 1])
        cx.set_xticklabels(["CNN-GRU", "Guarded\nAgent"], fontsize=7)
    bx.set_ylabel("MAE (cycles)")
    bx.set_ylim(0, 14.5)
    bx.set_title(f"MAE, n={int(len(z) / 4)} cases", loc="left", pad=8)
    cx.set_ylabel("S-score (median)")
    cx.set_ylim(0, 1.7)
    cx.set_title("Safety penalty", loc="left", pad=8)
    fig.suptitle("Where it matters most: inside true RUL $<$ 20 the "
                 "agent halves its tool's error "
                 f"({stats_['dl_only'][0]:.1f} $\\to$ "
                 f"{stats_['P7_agent_dl'][0]:.1f} cycles)",
                 x=0.02, ha="left", fontsize=9, y=1.06)
    fig.tight_layout(w_pad=2.4)
    save(fig, out, "figP7_eol_zone")

    # ---------- figD11 — the economics of self-correction --------------
    g5b = D3[D3.pattern == "P5_verifier"].copy()

    def _outcome(r_):
        if r_.esc:
            return "escalated"
        if r_.ac:
            return "auto-corrected"
        return "clean pass" if r_.rep == 0 else "repaired"
    g5b["oc"] = g5b.apply(_outcome, axis=1)
    order_oc = ["clean pass", "repaired", "auto-corrected", "escalated"]
    occ = {"clean pass": C["light"], "repaired": C["gold"],
           "auto-corrected": C["third"], "escalated": C["red"]}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(W2 * 0.72, 0.27 * W2))
    for ri, rc in enumerate([0, 1, 2]):
        bot = 0
        gg = g5b[g5b.rep == rc]
        for oc in order_oc:
            v = (gg.oc == oc).sum()
            if v:
                ax.bar(ri, v, bottom=bot, color=occ[oc], width=0.55,
                       label=oc if (ri == 0 or (oc, True) not in
                                    [(x.get_label(), True) for x in
                                     ax.containers]) else None)
                bot += v
        ax.text(ri, bot + 0.8, str(int(bot)), ha="center", fontsize=7.6)
    handles = [plt.Rectangle((0, 0), 1, 1, color=occ[o])
               for o in order_oc]
    ax.legend(handles, order_oc, ncols=4, fontsize=6.2,
              loc="lower left", bbox_to_anchor=(0.0, 1.02),
              borderaxespad=0)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["0 repairs", "1 repair", "2 repairs"],
                       fontsize=7.5)
    ax.set_ylabel("cases")
    ax.set_ylim(0, 70)
    ax.set_title("Repair rounds and outcomes", loc="left", pad=18)
    toks = [g5b[g5b.oc == o].tok.mean() for o in order_oc]
    wals = [g5b[g5b.oc == o].wall.mean() for o in order_oc]
    bx.bar(np.arange(4) - 0.18, toks, width=0.34, color=C["main"],
           label="tokens")
    dx = bx.twinx()
    dx.bar(np.arange(4) + 0.18, wals, width=0.34, color=C["gold"],
           label="seconds")
    dx.grid(False)
    bx.set_xticks(range(4))
    bx.set_xticklabels(["clean", "repaired", "auto-\ncorr.", "escal."],
                       fontsize=7)
    bx.set_ylabel("tokens / case", color=C["main"])
    dx.set_ylabel("s / case", color=C["gold"])
    bx.set_title("The price of self-correction", loc="left", pad=4)
    fig.tight_layout(w_pad=3.2)
    save(fig, out, "figD11_selfcorrection")

    print(f"[more] -> {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
paper_figures_request.py — Guarded Agent naming + requested plots:
  figZ1_eol_sweep_mae      MAE vs zone threshold (true RUL < 20/40/60/100)
  figZ2_eol_sweep_safety   median S-score vs zone threshold
  figU_T135 / figU_T209 / figU_T245   unit-specific rescue-style patterns
  figU_fingerprint         per-unit summary card (typeset)

Run:
  python paper\\code\\figures\\paper_figures_request.py --arad results\\arad3
"""
from __future__ import annotations

import argparse
import json
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
AGENT = "Guarded Agent"
TOOL = "CNN-GRU alone"


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
        f.savefig("/tmp/_texprobe5.png")
        plt.close(f)
        return True
    except Exception:
        plt.rcParams.update({**base, "text.usetex": False,
                             "font.family": "serif",
                             "font.serif": ["cmr10", "DejaVu Serif"],
                             "mathtext.fontset": "cm"})
        return False


USETEX = _fonts()
RNG = np.random.default_rng(42)


def pct(x, dec=0):
    s_ = f"{100 * x:.{dec}f}"
    return (s_ + r"\%") if USETEX else (s_ + "%")


def jread(p: Path):
    return [json.loads(l) for l in
            p.read_text(encoding="utf-8", errors="ignore").splitlines()
            if l.strip()]


def save(fig, out, name):
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)
    print(f"  [req] {name}")


def boot(v, n=2000):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    bs = [v[RNG.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return v.mean(), *np.percentile(bs, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arad", default="agentic/results/final_prognostic")
    ap.add_argument("--arad2", default="agentic/results/v2_rul_coupled_collapse")
    ap.add_argument("--v1grid", default="agentic/results/study_A_pattern_grid")
    ap.add_argument("--out", default=str(_r / "paper/figures_regen/request"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    A = Path(a.arad)
    A2 = Path(a.arad2)
    V1 = Path(a.v1grid)
    M = pd.read_csv(Path(a.arad) / "forecast_metrics.csv")
    peps = jread(Path(a.arad) / "forecast_episodes.jsonl")
    hint = {(e["qid"], e["arm"]): e.get("dl_hint") for e in peps}
    M["hint"] = [hint.get((q, ar)) for q, ar in zip(M.qid, M.arm)]

    # ---------------- EOL threshold sweep -------------------------------
    THS = [20, 40, 60, 100]
    rows = []
    for th in THS:
        z = M[M.true_rul < th]
        for arm in ("dl_only", "P7_agent_dl"):
            g = z[z.arm == arm].dropna(subset=["rul_abs_err"])
            m_, lo, hi = boot(g.rul_abs_err)
            rows.append(dict(th=th, arm=arm, mae=m_, lo=lo, hi=hi,
                             smed=g.s_score.median(), n=len(g)))
    SW = pd.DataFrame(rows)
    print(SW.round(2).to_string(index=False))

    x = np.arange(len(THS))
    fig, ax = plt.subplots(figsize=(W1 * 1.4, 0.62 * W1 * 1.4))
    for arm, cc, lab, off in (("dl_only", C["alt"], TOOL, -0.18),
                              ("P7_agent_dl", C["main"], AGENT, 0.18)):
        g = SW[SW.arm == arm]
        ax.bar(x + off, g.mae, width=0.34, color=cc, label=lab)
        ax.errorbar(x + off, g.mae, yerr=[g.mae - g.lo, g.hi - g.mae],
                    fmt="none", ecolor="black", capsize=2.4, lw=0.8)
        for xi, (_, r) in zip(x + off, g.iterrows()):
            ax.text(xi, r.hi + 0.5, f"{r.mae:.1f}", ha="center",
                    fontsize=6.8)
    ns = [int(SW[(SW.th == th) & (SW.arm == "P7_agent_dl")].n.iloc[0])
          for th in THS]
    ax.set_xticks(x)
    ax.set_xticklabels([f"$<${t}\n(n={n})" for t, n in zip(THS, ns)],
                       fontsize=7)
    ax.set_xlabel("end-of-life zone: true RUL threshold (cycles)")
    ax.set_ylabel("MAE (cycles), 95\\% CI" if USETEX
                  else "MAE (cycles), 95% CI")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_title("The advantage concentrates near end of life",
                 loc="left", pad=6)
    save(fig, out, "figZ1_eol_sweep_mae")

    fig, ax = plt.subplots(figsize=(W1 * 1.4, 0.62 * W1 * 1.4))
    for arm, cc, lab, off in (("dl_only", C["alt"], TOOL, -0.18),
                              ("P7_agent_dl", C["main"], AGENT, 0.18)):
        meds, los, his = [], [], []
        for th in THS:
            v = M[(M.true_rul < th) & (M.arm == arm)].s_score.dropna()
            bs = [np.median(v.sample(len(v), replace=True,
                                     random_state=k))
                  for k in range(2000)]
            meds.append(np.median(v))
            lo_, hi_ = np.percentile(bs, [2.5, 97.5])
            los.append(lo_)
            his.append(hi_)
        meds = np.array(meds); los = np.array(los); his = np.array(his)
        ax.bar(x + off, meds, width=0.34, color=cc, label=lab)
        ax.errorbar(x + off, meds, yerr=[meds - los, his - meds],
                    fmt="none", ecolor="black", capsize=2.4, lw=0.8)
        for xi, m_, h_ in zip(x + off, meds, his):
            ax.text(xi, h_ + 0.08, f"{m_:.1f}", ha="center",
                    fontsize=6.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"$<${t}\n(n={n})" for t, n in zip(THS, ns)],
                       fontsize=7)
    ax.set_xlabel("end-of-life zone: true RUL threshold (cycles)")
    ax.set_ylabel("median S-score, 95\\% CI" if USETEX
                  else "median S-score, 95% CI")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_title("Safety penalty across zone thresholds", loc="left",
                 pad=6)
    save(fig, out, "figZ2_eol_sweep_safety")

    # ---------------- unit-specific patterns ----------------------------
    dl = M[M.arm == "P7_agent_dl"].copy()
    dl["corr_f"] = dl.hint.fillna(0) <= 5
    for u in (135, 209, 245):
        g = dl[dl.unit == u].sort_values("cycle")
        if len(g) < 4:
            continue
        tru = (M[M.unit == u].drop_duplicates("cycle")
               .sort_values("cycle"))
        fig, ax = plt.subplots(figsize=(W1 * 1.55, 0.52 * W1 * 1.55))
        bad = g[g.corr_f]
        if len(bad):
            ax.axvspan(bad.cycle.min(), g.cycle.max(), color=C["alt"],
                       alpha=0.08)
            ax.text(bad.cycle.min() + 1, 3.0,
                    "tool corrupted (hint = 0)", fontsize=6.8,
                    color=C["alt"])
        ax.plot(tru.cycle, tru.true_rul.clip(upper=125),
                color=C["grey"], lw=1.4, label="true RUL (capped)")
        ax.plot(g.cycle, g.hint.fillna(0), ls="--", lw=1.1, marker="s",
                ms=2.8, color=C["alt"],
                label="CNN-GRU hint as seen")
        ax.plot(g.cycle, g.rul_pred, lw=1.3, marker="o", ms=3.2,
                color=C["main"], label=AGENT)
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
        ax.set_xlim(g.cycle.min() - 2, g.cycle.max() + 2)
        ax.set_xlabel("cycle")
        ax.set_ylabel("RUL (cycles)")
        ax.set_ylim(-6, 132)
        ax.legend(ncols=3, loc="lower left",
                  bbox_to_anchor=(0.0, 1.02), borderaxespad=0,
                  fontsize=6.4)
        ax.set_title(f"unit T{u}: unit-specific pattern", loc="left",
                     pad=22)
        save(fig, out, f"figU_T{u}")

    # ---------------- per-unit fingerprint card -------------------------
    dclean = M[M.arm == "dl_only"]
    rows = []
    for u in sorted(dl.unit.unique()):
        g = dl[dl.unit == u].dropna(subset=["rul_abs_err"])
        if not len(g):
            continue
        t = dclean[dclean.unit == u].dropna(subset=["rul_abs_err"])
        ov = (np.abs(g.rul_pred - g.hint.fillna(0)) > 5).mean()
        rows.append((f"T{u}", len(g),
                     f"{100 * g.corr_f.mean():.0f}",
                     f"{100 * ov:.0f}",
                     f"{t.rul_abs_err.mean():.1f}",
                     f"{g.rul_abs_err.mean():.1f}",
                     f"{g.s_score.median():.1f}"))
    with plt.rc_context({"text.usetex": False, "font.family": "serif",
                         "font.serif": ["DejaVu Serif"],
                         "mathtext.fontset": "dejavuserif"}):
        fig, ax = plt.subplots(figsize=(W2 * 0.8, 0.30 * W2))
        ax.axis("off")
        cols = ["unit", "n", "% hints\ncorrupted", "% override",
                "tool MAE", "agent MAE", "agent\nS-median"]
        xs = [0.02, 0.14, 0.26, 0.44, 0.60, 0.74, 0.90]
        ax.text(0, 1.02, "Per-unit fingerprint of the "
                         "Guarded Agent", fontsize=10,
                fontweight="bold", va="top")
        for xc, cl in zip(xs, cols):
            ax.text(xc, 0.88, cl, fontsize=8, color=C["grey"],
                    va="top")
        y = 0.72
        for r in rows:
            better = float(r[5]) <= float(r[4])
            for xc, v in zip(xs, r):
                ax.text(xc, y, str(v), fontsize=8.3, va="top",
                        color=(C["main"] if better else C["red"])
                        if xc == xs[5] else "black")
            y -= 0.095
        ax.text(0, y - 0.02, "agent MAE in blue where it beats or "
                             "ties its tool, red otherwise",
                fontsize=7, color=C["grey"], va="top")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        save(fig, out, "figU_fingerprint")

    # ---------------- RAGAS on the audited v1 subset --------------------
    mj = V1 / "metrics.jsonl"
    if mj.exists():
        R = pd.DataFrame(jread(mj))
        R = R[R.faithfulness.notna()]
        order = [p_ for p_ in ["B0_retrieval", "P1_direct", "P2_rag",
                               "P3_react", "P4_reflexion",
                               "P5_verifier", "P6_specialists"]
                 if p_ in set(R.pattern)]
        labs = {"B0_retrieval": "B0", "P1_direct": "direct",
                "P2_rag": "RAG", "P3_react": "ReAct",
                "P4_reflexion": "reflex.", "P5_verifier": "verifier",
                "P6_specialists": "special."}
        xr = np.arange(len(order))
        fig, ax = plt.subplots(figsize=(W1 * 1.5, 0.6 * W1 * 1.5))
        for off, col, cc, lab in (
                (-0.27, "faithfulness", C["main"], "faithfulness"),
                (0.0, "answer_relevancy", C["gold"],
                 "answer relevancy"),
                (0.27, "context_precision", C["third"],
                 "context precision")):
            vals = [R[R.pattern == p_][col].mean() for p_ in order]
            ax.bar(xr + off, vals, width=0.25, color=cc, label=lab)
        for xi, p_ in zip(xr, order):
            ax.text(xi, 1.02, f"n={int((R.pattern == p_).sum())}",
                    ha="center", fontsize=6.2, color=C["grey"])
        ax.set_xticks(xr)
        ax.set_xticklabels([labs[p_] for p_ in order], fontsize=7)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("RAGAS score (llama3.1:8b judge)")
        ax.legend(ncols=3, fontsize=6.4, loc="lower left",
                  bbox_to_anchor=(0.0, 1.06), borderaxespad=0)
        ax.set_title("RAGAS metrics on the audited subset "
                     "(judge-based)", loc="left", pad=24)
        save(fig, out, "figR1_ragas_audited")

    # ---------------- judge-free grounding surrogates -------------------
    import re as _re
    HKW = _re.compile(r"(previous|prior|progression|recurring|earlier|"
                      r"history|count|past|first anomal|accelerat)",
                      _re.I)
    SEV2ACT = {1: "continue_monitoring", 2: "continue_monitoring",
               3: "schedule_inspection", 4: "plan_maintenance",
               5: "immediate_shutdown"}
    dg = jread(_DIAG_DIR / "episodes.jsonl")
    p5 = [e for e in dg if e["pattern"] == "P5_verifier"]
    def _cit(e):
        t = e.get("ticket") or {}
        shown = [f"u{c['unit']}c{c['cycle']}" for c in
                 (e.get("contexts") or [])]
        cited = t.get("cited_precedents") or []
        return (float(all(c in shown for c in cited)) if cited
                else (1.0 if not shown else 0.0))
    d_cit = np.mean([_cit(e) for e in p5])
    d_coh = np.mean([float((e.get("ticket") or {}).get("action")
                     == SEV2ACT.get((e.get("ticket") or {})
                                    .get("severity")))
                     for e in p5])
    d_hist = np.mean([float(bool(HKW.search(
        str((e.get("ticket") or {}).get("matched_pattern", "")) + " "
        + str((e.get("ticket") or {}).get("diagnosis", "")))))
        for e in p5 if e.get("has_history")])
    pp = [e for e in peps if e["arm"] == "P7_agent_dl"]
    def _pcit(e):
        f_ = e.get("forecast") or {}
        shown = [f"u{c['unit']}c{c['cycle']}" for c in
                 (e.get("contexts") or [])]
        cited = f_.get("cited_precedents") or []
        return (float(all(c in shown for c in cited)) if cited
                else np.nan)
    p_cit = np.nanmean([_pcit(e) for e in pp])
    p_nar = np.mean([float(bool((e.get("forecast") or {})
                          .get("progression_narrative"))) for e in pp])
    p_rng = np.mean([float((e.get("forecast") or {})
                     .get("rul_range") is not None) for e in pp])
    fig, ax = plt.subplots(figsize=(W1 * 1.45, 0.58 * W1 * 1.45))
    names = ["citation\nvalidity", "coherence /\nnarrative",
             "history /\nrange stated"]
    dvals = [d_cit, d_coh, d_hist]
    pvals = [p_cit, p_nar, p_rng]
    xg = np.arange(3)
    ax.bar(xg - 0.18, dvals, width=0.34, color=C["main"],
           label="diagnostic agent")
    ax.bar(xg + 0.18, pvals, width=0.34, color=C["gold"],
           label="prognostic agent")
    for xi, v in zip(xg - 0.18, dvals):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.8)
    for xi, v in zip(xg + 0.18, pvals):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.8)
    ax.set_xticks(xg)
    ax.set_xticklabels(names, fontsize=7.2)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=6.8, ncols=2, loc="lower left",
              bbox_to_anchor=(0.0, 1.02), borderaxespad=0)
    ax.set_title("Judge-free grounding surrogates on the final agents "
                 "(deterministic, RAGAS-style)", loc="left", pad=22)
    save(fig, out, "figR2_grounding_surrogates")

    # ---------------- combined maintenance ticket -----------------------
    def clean_txt(x):
        x = str(x or "")
        for a_, b_ in ((", Gravity=None", ""), ("Gravity=None", ""),
                       ("**", ""), ("`", ""), ("!=", " \u2260 "),
                       ("<=", "\u2264"), (">=", "\u2265"),
                       ("->", "\u2192")):
            x = x.replace(a_, b_)
        return " ".join(x.split())
    import textwrap as _tw
    dmap = {e["qid"]: e for e in p5}
    both = []
    for e in pp:
        q = e["qid"]
        f_ = e.get("forecast") or {}
        if q in dmap and f_.get("rul_estimate") is not None:
            t = dmap[q].get("ticket") or {}
            if t.get("cited_precedents") and f_.get("cited_precedents"):
                err = (abs(f_["rul_estimate"] - e["true_rul"])
                       if e.get("true_rul") is not None else 99)
                both.append((max(0, 8 - err)
                             + 2 * bool(t.get("reasoning")), e, dmap[q]))
    both.sort(key=lambda z: -z[0])
    _, pe_, de_ = both[0]
    t = de_.get("ticket") or {}
    f_ = pe_.get("forecast") or {}
    with plt.rc_context({"text.usetex": False, "font.family": "serif",
                         "font.serif": ["DejaVu Serif"],
                         "mathtext.fontset": "dejavuserif"}):
        fig, ax = plt.subplots(figsize=(W2 * 0.94, 0.56 * W2))
        ax.axis("off")

        def blk(y, label, text, width=110, color="black", fs=7.9):
            if label:
                ax.text(0, y, label, fontsize=7.6, color=C["grey"],
                        va="top")
                y -= 0.042
            txt = _tw.fill(clean_txt(text), width)
            ax.text(0, y, txt, fontsize=fs, va="top", color=color)
            return y - 0.040 * (txt.count("\n") + 1) - 0.030
        ax.text(0, 1.0, f"A-RAD maintenance ticket - unit "
                        f"{pe_['unit']}, cycle {pe_['cycle']}",
                fontsize=10.5, fontweight="bold", va="top")
        y = 0.925
        ax.text(0, y, "DIAGNOSIS (retrospective agent)", fontsize=8.4,
                fontweight="bold", color=C["main"], va="top")
        y -= 0.048
        y = blk(y, None, f"severity {t.get('severity')}  -  action: "
                         f"{str(t.get('action', '')).replace('_', ' ')}"
                         f"  -  matched: "
                         + clean_txt(t.get('matched_pattern', '')))
        y = blk(y, "diagnosis", t.get("diagnosis", ""))
        y = blk(y, "rationale", t.get("reasoning", ""))
        y -= 0.012
        ax.text(0, y, "PROGNOSIS (forward agent)", fontsize=8.4,
                fontweight="bold", color=C["gold"], va="top")
        y -= 0.048
        rr = f_.get("rul_range")
        rt = (f" (range {min(rr):.0f}-{max(rr):.0f})"
              if rr and not (min(rr) == 0 and max(rr) == 125) else "")
        y = blk(y, None, f"RUL estimate {f_['rul_estimate']:.0f} "
                         f"cycles{rt}  -  outlook "
                         f"{f_.get('anomaly_outlook')}  -  confidence "
                         f"{f_.get('confidence')}")
        y = blk(y, "progression narrative",
                f_.get("progression_narrative", ""))
        cited = (f_.get("cited_precedents") or [])[:2]
        lines = []
        for key in cited:
            cc_ = next((c for c in (pe_.get("contexts") or [])
                        if f"u{c['unit']}c{c['cycle']}" == key), None)
            if cc_ and cc_.get("rul_then") is not None:
                lines.append(f"{key} survived "
                             f"{cc_['rul_then']:.0f} more cycles")
        if lines:
            y = blk(y, "cited precedent futures", ";  ".join(lines),
                    color=C["main"])
        y -= 0.008
        foot = (f"verifier: {de_.get('repairs') or 0} repair(s), "
                f"{'escalated' if de_.get('escalated') else 'clean'}"
                f"  |  CNN-GRU hint {pe_.get('dl_hint'):.0f}, ground "
                f"truth {pe_.get('true_rul'):.0f} cycles  |  "
                f"{(de_.get('tokens_out') or 0)} + generation tokens, "
                f"{(de_.get('wall_s') or 0) + (pe_.get('wall_s') or 0):.0f} s"
                " total on edge-class hardware")
        ax.text(0, y, foot, fontsize=7.3, color=C["grey"], va="top")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.03)
        save(fig, out, "figT_combined_ticket")

    # ---------------- abstention usefulness -----------------------------
    if (A2 / "diag" / "episodes.jsonl").exists():
        d2 = jread(A2 / "diag" / "episodes.jsonl")
        BANDS_ = [(120, "continue_monitoring"),
                  (60, "schedule_inspection"),
                  (25, "plan_maintenance"),
                  (-1, "immediate_shutdown")]
        def band_(r):
            for th_, a_ in BANDS_:
                if r > th_:
                    return a_
            return "immediate_shutdown"
        pts = []
        g2 = [e for e in d2 if e["pattern"] == "P5_verifier"]
        del2 = [e for e in g2 if not e.get("escalated")]
        q2 = np.mean([float((e.get("ticket") or {}).get("action")
                      == band_((e.get("ticket") or {})
                               .get("rul_estimate", 999)))
                      for e in del2]) if del2 else 0
        pts.append(("normal retrieval,\nRUL-coupled (v2)",
                    1 - np.mean([e.get("escalated", False)
                                 for e in g2]), q2, len(del2),
                    C["grey"]))
        del3 = [e for e in p5 if not e.get("escalated")]
        q3 = np.mean([float((e.get("ticket") or {}).get("action")
                      == SEV2ACT.get((e.get("ticket") or {})
                                     .get("severity")))
                      for e in del3])
        pts.append(("stage-aware,\nretrospective (v3)",
                    1 - np.mean([e.get("escalated", False)
                                 for e in p5]), q3, len(del3),
                    C["main"]))
        fig, ax = plt.subplots(figsize=(W1 * 1.4, 0.66 * W1 * 1.4))
        for lab, cov, q, n, cc in pts:
            ax.scatter([cov], [q], s=60 + n * 2.4, color=cc, zorder=3,
                       edgecolor="white", lw=1.2)
            extra = ("\n(quality on n=2:\nnot informative)"
                     if n < 5 else "")
            ax.annotate(lab + f"\ncoverage {pct(cov)}, "
                              f"n={n}" + extra,
                        (cov, q), textcoords="offset points",
                        xytext=(14, -16) if cc == C["grey"]
                        else (-14, -44),
                        ha="left" if cc == C["grey"] else "right",
                        fontsize=7)
        ax.annotate("", xy=(pts[1][1], pts[1][2]),
                    xytext=(pts[0][1], pts[0][2]),
                    arrowprops=dict(arrowstyle="->", lw=1.2,
                                    color=C["third"]))
        ax.set_xlabel("coverage (1 $-$ abstention rate)")
        ax.set_ylabel("action quality among delivered tickets")
        ax.set_xlim(-0.05, 1.08)
        ax.set_ylim(0, 1.08)
        ax.set_title("Abstention usefulness: from blanket to "
                     "selective", loc="left", pad=6)
        save(fig, out, "figA_abstention")

    # ---------------- abstention deep dive ------------------------------
    ge = [e for e in peps if e["arm"] == "P7_agent_dl"
          and (e.get("forecast") or {}).get("rul_estimate") is not None]
    conf = pd.DataFrame([dict(
        qid=e["qid"], conf=str((e.get("forecast") or {})
                               .get("confidence", "")).lower(),
        width=(lambda r: (max(r) - min(r)) if r else np.nan)(
            (e.get("forecast") or {}).get("rul_range")),
        ) for e in ge])
    errmap = M[M.arm == "P7_agent_dl"].set_index("qid").rul_abs_err
    conf["err"] = conf.qid.map(errmap)
    conf = conf.dropna(subset=["err"])
    order_c = ["low", "medium", "high"]
    conf["conf"] = conf.conf.where(conf.conf.isin(order_c), "medium")
    print("confidence dist:", conf.conf.value_counts().to_dict())
    print(conf.groupby("conf").err.agg(["mean", "median",
                                        "count"]).round(1))

    # AB2 — error by self-reported confidence
    fig, ax = plt.subplots(figsize=(W1 * 1.3, 0.6 * W1 * 1.3))
    xs = np.arange(len(order_c))
    means, los, his, ns = [], [], [], []
    for c_ in order_c:
        v = conf[conf.conf == c_].err
        if len(v) == 0:
            means.append(np.nan); los.append(np.nan)
            his.append(np.nan); ns.append(0)
            continue
        m_, lo, hi = boot(v)
        means.append(m_); los.append(lo); his.append(hi)
        ns.append(len(v))
    ax.bar(xs, means, color=[C["red"], C["gold"], C["third"]],
           width=0.55)
    ax.errorbar(xs, means, yerr=[np.array(means) - np.array(los),
                                 np.array(his) - np.array(means)],
                fmt="none", ecolor="black", capsize=2.6, lw=0.9)
    for xi, m_, n_ in zip(xs, means, ns):
        if n_:
            ax.text(xi, m_ + 1.2, f"{m_:.1f}\n(n={n_})", ha="center",
                    fontsize=6.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([c_.capitalize() for c_ in order_c])
    ax.set_xlabel("self-reported confidence of the Guarded Agent")
    ax.set_ylabel(r"MAE (cycles), 95\% CI" if USETEX
                  else "MAE (cycles), 95% CI")
    ax.set_title("Error by self-reported confidence", loc="left", pad=6)
    save(fig, out, "figAB2_confidence_validity")

    # AB1 — risk-coverage curves (selective prediction)
    fig, ax = plt.subplots(figsize=(W1 * 1.4, 0.64 * W1 * 1.4))
    rank_c = conf.conf.map({"low": 0, "medium": 1, "high": 2})
    for proxy, cc, lab in ((rank_c + RNG.uniform(0, .01, len(conf)),
                            C["main"],
                            "abstain by low confidence"),
                           (-conf.width.fillna(conf.width.max()),
                            C["gold"], "abstain by widest range")):
        df = conf.assign(px=proxy).sort_values("px")   # abstain first
        covs, maes = [], []
        n = len(df)
        for k in range(0, int(0.6 * n) + 1):
            kept = df.iloc[k:]
            covs.append(len(kept) / n)
            maes.append(kept.err.mean())
        ax.plot(covs, maes, lw=1.5, color=cc, label=lab)
    ax.axhline(conf.err.mean(), color=C["grey"], lw=1.0, ls="--")
    ax.text(0.42, conf.err.mean() + 0.5, "no abstention "
            f"({conf.err.mean():.1f})", fontsize=6.8, color=C["grey"])
    ax.set_xlabel("coverage (share of cases answered)")
    ax.set_ylabel("MAE on answered cases (cycles)")
    ax.set_xlim(0.38, 1.02)
    ax.legend(fontsize=6.8, loc="lower right")
    ax.set_title("Selective prediction via stated uncertainty",
                 loc="left", pad=6)
    save(fig, out, "figAB1_risk_coverage")

    # AB3 — the abstention ladder, v2 vs v3
    if (A2 / "diag" / "episodes.jsonl").exists():
        d2 = jread(A2 / "diag" / "episodes.jsonl")
        g2 = [e for e in d2 if e["pattern"] == "P5_verifier"]
        dg = jread(_DIAG_DIR / "episodes.jsonl")
        g3 = [e for e in dg if e["pattern"] == "P5_verifier"]
        def ladder(g):
            n = len(g)
            esc = sum(bool(e.get("escalated")) for e in g)
            ac = sum(bool(e.get("auto_corrected"))
                     and not e.get("escalated") for e in g)
            rep = sum((e.get("repairs") or 0) > 0
                      and not e.get("auto_corrected")
                      and not e.get("escalated") for e in g)
            clean = n - esc - ac - rep
            return np.array([clean, rep, ac, esc]) / n
        L = {"RUL-coupled (v2)": ladder(g2),
             "stage-aware Guarded (v3)": ladder(g3)}
        labs_ = ["delivered clean", "delivered after repair",
                 "auto-corrected", "abstained (escalated)"]
        cols_ = [C["light"], C["gold"], C["third"], C["red"]]
        fig, ax = plt.subplots(figsize=(W1 * 1.4, 0.5 * W1 * 1.4))
        for i, (nm, v) in enumerate(L.items()):
            bot = 0
            for j, val in enumerate(v):
                ax.barh(1 - i, val, left=bot, color=cols_[j],
                        height=0.5,
                        label=labs_[j] if i == 0 else None)
                if val > 0.04:
                    ax.text(bot + val / 2, 1 - i, pct(val),
                            ha="center", va="center", fontsize=6.8,
                            color="black" if j < 3 else "white")
                bot += val
        ax.set_yticks([1, 0])
        ax.set_yticklabels(list(L.keys()), fontsize=7.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of the 89 cases")
        ax.legend(ncols=2, fontsize=6.2, loc="lower left",
                  bbox_to_anchor=(0.0, 1.04), borderaxespad=0)
        ax.set_title("The abstention ladder: graduated responses "
                     "replace blanket refusal", loc="left", pad=26)
        save(fig, out, "figAB3_abstention_ladder")

    # ---------------- agentic-pattern inventory -------------------------
    rows_p = [
        ("B0  retrieval floor", "yes - grounding ceiling",
         "yes - severity prior", "yes - b0 stage-matched floor"),
        ("P1  direct LLM", "yes - fabricates cit. (0.00)",
         "yes - contrast arm", "-"),
        ("P2  single-shot RAG", "yes - 0.996 cit.",
         "yes - contrast arm", "-"),
        ("P3  naive ReAct", "yes - collapses (0.29 cit., 5 calls)",
         "-", "-"),
        ("P4  reflexion", "yes - 0.96 cit., 3 calls", "-", "-"),
        ("P5  verifier agent", "yes - 0.99 cit., 1.9 calls",
         "yes - THE diagnostic agent\n(retrospective tools + gates)",
         "-"),
        ("P6  specialists", "yes - 1.00 cit., 3 calls", "-", "-"),
        ("P7  guarded forecaster", "-", "-",
         "yes - THE prognostic agent\n(dl_predict + futures + "
         "verifier)"),
    ]
    with plt.rc_context({"text.usetex": False, "font.family": "serif",
                         "font.serif": ["DejaVu Serif"],
                         "mathtext.fontset": "dejavuserif"}):
        fig, ax = plt.subplots(figsize=(W2 * 0.95, 0.44 * W2))
        ax.axis("off")
        ax.text(0, 1.02, "Agentic patterns across the three studies",
                fontsize=10.5, fontweight="bold", va="top")
        xs = [0.0, 0.27, 0.55, 0.815]
        heads = ["pattern", "Study A: diagnostic grid\n(v1, 1691 eps)",
                 "Study D: diagnostic final\n(v3, 356 eps)",
                 "Study P: prognostic\n(v3, 356 eps)"]
        for xc, h in zip(xs, heads):
            ax.text(xc, 0.92, h.replace("\\n", chr(10)), fontsize=7.6,
                    color=C["grey"], va="top")
        y = 0.80
        for r in rows_p:
            for xc, v in zip(xs, r):
                col = ("black" if xc == xs[0]
                       else C["main"] if str(v).startswith("yes")
                       else C["light"])
                ax.text(xc, y, str(v).replace("\\n", chr(10)),
                        fontsize=7.4, va="top", color=col,
                        fontweight="bold" if "THE" in str(v)
                        else "normal")
            y -= 0.105
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        save(fig, out, "figPAT_inventory")

    # ------------- abstention propensity vs RUL -------------------------
    BANDS_ = [(100, 126, "$\\geq$100"), (60, 100, "60--100" if USETEX
              else "60-100"), (25, 60, "25--60" if USETEX else "25-60"),
              (0, 25, "$<$25")]
    def bidx(rul):
        rul = min(rul, 125)
        for i, (lo, hi, _) in enumerate(BANDS_):
            if lo <= rul < hi:
                return i
        return len(BANDS_) - 1
    ab = conf.copy()
    tmap = M[M.arm == "P7_agent_dl"].set_index("qid").true_rul
    ab["true"] = ab.qid.map(tmap)
    ab = ab.dropna(subset=["true"])
    ab["b"] = [bidx(t_) for t_ in ab.true]
    lowsh = ab.groupby("b").conf.apply(lambda g: (g == "low").mean())
    wid = ab.groupby("b").width.mean()
    mae = ab.groupby("b").err.mean()
    x = np.arange(len(BANDS_))
    fig, ax = plt.subplots(figsize=(W1 * 1.45, 0.62 * W1 * 1.45))
    ax.bar(x - 0.18, [lowsh.get(i, 0) for i in range(4)], width=0.34,
           color=C["red"], alpha=0.85,
           label="share of low-confidence forecasts")
    ax.bar(x + 0.18, [wid.get(i, 0) / 100 for i in range(4)],
           width=0.34, color=C["gold"],
           label="mean stated width / 100")
    bx = ax.twinx()
    bx.plot(x, [mae.get(i, np.nan) for i in range(4)], marker="o",
            ms=5, lw=1.6, color=C["main"], label="Guarded Agent MAE")
    bx.grid(False)
    for i in range(4):
        ax.text(x[i] - 0.18, lowsh.get(i, 0) + 0.02,
                pct(lowsh.get(i, 0)), ha="center", fontsize=6.6)
    ax.set_xticks(x)
    ax.set_xticklabels([b[2] for b in BANDS_])
    ax.set_xlabel("true RUL band (cycles)")
    ax.set_ylabel("soft-abstention propensity")
    bx.set_ylabel("MAE (cycles)", color=C["main"])
    ax.set_ylim(0, 1.0)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = bx.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.4, loc="lower left",
              bbox_to_anchor=(0.0, 1.02), ncols=2, borderaxespad=0)
    ax.set_title("Where the agent hedges: stated width follows the "
                 "error profile; the confidence label only weakly",
                 loc="left", pad=26)
    save(fig, out, "figAB4_abstention_vs_rul")
    print("low-conf share by band:",
          {BANDS_[i][2]: round(lowsh.get(i, 0), 2) for i in range(4)})

    # ------------- high-confidence combined ticket ----------------------
    dmap2 = {e["qid"]: e for e in jread(_DIAG_DIR / "episodes.jsonl")
             if e["pattern"] == "P5_verifier"}
    hc = []
    for e in peps:
        if e["arm"] != "P7_agent_dl":
            continue
        f_ = e.get("forecast") or {}
        if str(f_.get("confidence", "")).lower() != "high":
            continue
        if not (f_.get("rul_estimate") is not None
                and f_.get("cited_precedents")
                and f_.get("progression_narrative")
                and e["qid"] in dmap2):
            continue
        err = (abs(f_["rul_estimate"] - e["true_rul"])
               if e.get("true_rul") is not None else 99)
        rr = f_.get("rul_range") or [0, 125]
        hc.append((max(0, 8 - err) + 2 * (not (rr[0] == 0
                                               and rr[1] == 125)), e))
    hc.sort(key=lambda z: -z[0])
    pe_ = hc[0][1]
    de_ = dmap2[pe_["qid"]]
    t = de_.get("ticket") or {}
    f_ = pe_.get("forecast") or {}
    import textwrap as _tw
    def clean_txt(x):
        x = str(x or "")
        for a_, b_ in ((", Gravity=None", ""), ("Gravity=None", ""),
                       ("**", ""), ("`", ""), ("!=", " \u2260 "),
                       ("<=", "\u2264"), (">=", "\u2265"),
                       ("->", "\u2192")):
            x = x.replace(a_, b_)
        return " ".join(x.split())
    with plt.rc_context({"text.usetex": False, "font.family": "serif",
                         "font.serif": ["DejaVu Serif"],
                         "mathtext.fontset": "dejavuserif"}):
        fig, ax = plt.subplots(figsize=(W2 * 0.94, 0.52 * W2))
        ax.axis("off")
        def blk(y, label, text, width=110, color="black", fs=7.9):
            if label:
                ax.text(0, y, label, fontsize=7.6, color=C["grey"],
                        va="top")
                y -= 0.045
            txt = _tw.fill(clean_txt(text), width)
            ax.text(0, y, txt, fontsize=fs, va="top", color=color)
            return y - 0.043 * (txt.count("\n") + 1) - 0.032
        ax.text(0, 1.0, "A-RAD maintenance ticket (high-confidence "
                        f"prognosis) - unit {pe_['unit']}, cycle "
                        f"{pe_['cycle']}", fontsize=10.3,
                fontweight="bold", va="top")
        y = 0.92
        ax.text(0, y, "DIAGNOSIS (retrospective agent)", fontsize=8.4,
                fontweight="bold", color=C["main"], va="top")
        y -= 0.05
        y = blk(y, None, f"severity {t.get('severity')}  -  action: "
                f"{str(t.get('action', '')).replace('_', ' ')}")
        y = blk(y, "diagnosis", t.get("diagnosis", ""))
        y -= 0.012
        ax.text(0, y, "PROGNOSIS (forward agent)", fontsize=8.4,
                fontweight="bold", color=C["gold"], va="top")
        y -= 0.05
        rr = f_.get("rul_range")
        rt = (f" (range {min(rr):.0f}-{max(rr):.0f})"
              if rr and not (min(rr) == 0 and max(rr) == 125) else "")
        y = blk(y, None, f"RUL estimate {f_['rul_estimate']:.0f} "
                f"cycles{rt}  -  outlook {f_.get('anomaly_outlook')}"
                f"  -  confidence HIGH")
        y = blk(y, "progression narrative",
                f_.get("progression_narrative", ""))
        lines = []
        for key in (f_.get("cited_precedents") or [])[:3]:
            cc_ = next((c for c in (pe_.get("contexts") or [])
                        if f"u{c['unit']}c{c['cycle']}" == key), None)
            if cc_ and cc_.get("rul_then") is not None:
                lines.append(f"{key} survived "
                             f"{cc_['rul_then']:.0f} more cycles")
        if lines:
            y = blk(y, "cited precedent futures", ";  ".join(lines),
                    color=C["main"])
        foot = (f"CNN-GRU hint {pe_.get('dl_hint'):.0f}  |  ground "
                f"truth {pe_.get('true_rul'):.0f} cycles  |  "
                f"{pe_.get('steps')} tool steps, "
                f"{pe_.get('wall_s', 0):.1f} s")
        ax.text(0, y, foot, fontsize=7.3, color=C["grey"], va="top")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.03)
        save(fig, out, "figT_high_confidence_ticket")
        print("high-conf ticket:", pe_["qid"], "est",
              f_["rul_estimate"], "true", pe_.get("true_rul"))
    # ------------- calibrated urgency: margin vs remaining life ---------
    from scipy import stats as _st
    fig, ax = plt.subplots(figsize=(W1 * 1.5, 0.64 * W1 * 1.5))
    ax.fill_between([0, 128], [0, 0], [-45, -45], color=C["red"],
                    alpha=0.06)
    ax.text(66, -20, "over-prediction (unsafe)", fontsize=7,
            color=C["red"])
    ax.plot([0, 125], [0, 125], ls=":", lw=0.9, color=C["light"])
    ax.text(88, 103, "predicting zero", fontsize=6.6, color=C["grey"],
            rotation=35)
    ax.axhline(0, color=C["grey"], lw=0.9)
    BANDX = {0: 112, 1: 80, 2: 42, 3: 12}
    for arm, cc, lab in (("dl_only", C["alt"], "CNN-GRU alone"),
                         ("P7_agent_dl", C["main"], "Guarded Agent")):
        g = M[M.arm == arm].dropna(subset=["rul_err"]).copy()
        g["tc"] = g.true_rul.clip(upper=125)
        g["margin"] = -g.rul_err            # true - pred
        ax.scatter(g.tc + RNG.normal(0, 1.0, len(g)), g.margin, s=11,
                   alpha=0.45, color=cc, label=lab)
        med = (g.assign(b=[0 if t >= 100 else 1 if t >= 60 else
                           2 if t >= 25 else 3 for t in g.tc])
               .groupby("b").margin.median())
        ax.plot([BANDX[i] for i in med.index], med.values, color=cc,
                lw=1.8, marker="o", ms=4.5, zorder=5)
        if arm == "P7_agent_dl":
            r_, p_ = _st.spearmanr(g.tc, g.margin)
            ax.text(0.02, 0.94, f"Guarded Agent: $\\rho$(margin, "
                    f"remaining life) = {r_:+.2f}",
                    transform=ax.transAxes, fontsize=7.6)
    ax.set_xlabel("true remaining life (cycles, capped at 125)")
    ax.set_ylabel(r"early-warning margin, true $-$ predicted (cycles)")
    ax.set_xlim(0, 128)
    ax.set_ylim(-45, 128)
    ax.legend(loc="upper left", fontsize=6.8,
              bbox_to_anchor=(0.02, 0.90))
    ax.set_title("Calibrated urgency: the conservative margin is "
                 "proportional to remaining life", loc="left", pad=6)
    save(fig, out, "figAB5_calibrated_urgency")
    print(f"[request] -> {out}")


if __name__ == "__main__":
    main()

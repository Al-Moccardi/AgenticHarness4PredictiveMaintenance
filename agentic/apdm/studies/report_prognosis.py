"""Aggregate the v8 prognostic runs into paper-ready tables, figures and a
HEADLINES.md whose sentences are filled with the measured numbers (and whose
claims are gated on the paired statistics: nothing is called significant
unless the Wilcoxon p < .05 on identical cases).

Inputs : results/prog_predictions_{model}_seed{S}.csv   (one per model)
Outputs: results/prognosis_summary_seed{S}.csv
         results/prognosis_paired_seed{S}.csv
         results/figures/fig_prog_{mae,calibration,scatter}_seed{S}.{png,pdf}
         results/HEADLINES_seed{S}.md

  python -m apdm.report_prognosis --sample-seed 1
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .metrics import bootstrap_ci, wilcoxon_paired, wilson
from .prognosis import ARMS

ROOT = Path(__file__).resolve().parent.parent
LADDER = ["P1_direct", "P2_rag", "P3_react", "P4_reflexion", "P5_verifier",
          "P6_specialists"]
BASELINES = ["B0_retrieval", "B1_zknn"]


# ------------------------------------------------------------------ loading
def load_runs(out: Path, seed: int) -> pd.DataFrame:
    files = sorted(glob.glob(str(out / f"prog_predictions_*_seed{seed}.csv")))
    if not files:
        raise SystemExit(f"[report-prog] no prog_predictions_*_seed{seed}"
                         f".csv under {out}; run apdm.prognosis or "
                         f"apdm.bench_prognosis first")
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    # deterministic baselines are recomputed identically inside every model
    # run: keep one copy, labelled model-free.
    base = df[df.arm.isin(BASELINES)].drop_duplicates(
        subset=["arm", "unit", "cycle"]).copy()
    base["model"] = "(no LLM)"
    df = pd.concat([df[~df.arm.isin(BASELINES)], base], ignore_index=True)
    df["parse_failed"] = df["parse_failed"].astype(bool)
    return df


# ---------------------------------------------------------------- aggregate
def _agg_one(g: pd.DataFrame) -> Dict:
    ok = g[~g.parse_failed]
    lo, hi = wilson(int((~g.parse_failed).sum()), len(g))
    row = {"n": len(g), "n_parsed": len(ok),
           "parse_rate": round(float((~g.parse_failed).mean()), 3),
           "parse_wilson_lo": round(lo, 3), "parse_wilson_hi": round(hi, 3)}
    if len(ok):
        ci = bootstrap_ci(list(ok.abs_err))
        row.update({
            "mae": round(float(ok.abs_err.mean()), 2),
            "mae_ci_lo": round(ci[0], 2), "mae_ci_hi": round(ci[1], 2),
            "rmse": round(float(np.sqrt((ok.err ** 2).mean())), 2),
            "bias": round(float(ok.err.mean()), 2),
            "s_per_case": round(float(ok.s_i.mean()), 2),
            "coverage80": round(float(ok.in_range.mean()), 3),
            "width_mean": round(float(ok.width.mean()), 1),
            "winkler_mean": round(float(ok.winkler.mean()), 1),
            "action_exact": round(float(ok.action_exact.mean()), 3),
            "action_pm1": round(float(ok.action_pm1.mean()), 3)})
        for c, name in (("supported", "faith_supported"),
                        ("contradicted", "faith_contradicted"),
                        ("cited_valid", "cited_valid_rate"),
                        ("post_hoc_violations", "post_hoc_violations_mean")):
            if c in ok and ok[c].notna().any():
                row[name] = round(float(pd.to_numeric(
                    ok[c], errors="coerce").mean()), 3)
        if "escalated" in g and g["escalated"].notna().any():
            e = pd.to_numeric(g["escalated"].map(
                {True: 1, False: 0, "True": 1, "False": 0}),
                errors="coerce")
            row["escalation_rate"] = round(float(e.mean()), 3)
        if "violations_pre" in g and g["violations_pre"].notna().any():
            row["violations_pre_mean"] = round(float(pd.to_numeric(
                g["violations_pre"], errors="coerce").mean()), 2)
    for c in ("llm_calls", "tool_calls", "seconds", "prompt_tokens",
              "completion_tokens", "sim_edge_s", "sim_energy_j"):
        if c in g and g[c].notna().any():
            row[f"{c}_mean"] = round(float(pd.to_numeric(
                g[c], errors="coerce").mean()), 3)
    return row


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, arm), g in df.groupby(["model", "arm"]):
        rows.append({"model": model, "arm": arm, **_agg_one(g)})
    s = pd.DataFrame(rows)
    order = {a: i for i, a in enumerate(BASELINES + LADDER)}
    s["_o"] = s.arm.map(order)
    return s.sort_values(["model", "_o"]).drop(columns="_o")


def paired_vs_baseline(df: pd.DataFrame, baseline: str = "B0_retrieval"
                       ) -> pd.DataFrame:
    b = df[(df.arm == baseline) & (~df.parse_failed)]
    b = b.set_index(["unit", "cycle"])
    rows = []
    for (model, arm), g in df[~df.arm.isin(BASELINES)].groupby(
            ["model", "arm"]):
        ok = g[~g.parse_failed].set_index(["unit", "cycle"])
        common = ok.index.intersection(b.index)
        if len(common) < 8:
            continue
        wa = wilcoxon_paired(list(ok.loc[common, "abs_err"]),
                             list(b.loc[common, "abs_err"]))
        rows.append({"model": model, "arm": arm, "vs": baseline,
                     "n_paired": len(common),
                     "mae_arm": round(float(
                         ok.loc[common, "abs_err"].mean()), 2),
                     "mae_base": round(float(
                         b.loc[common, "abs_err"].mean()), 2),
                     "delta_mae": round(float(
                         ok.loc[common, "abs_err"].mean()
                         - b.loc[common, "abs_err"].mean()), 2),
                     "median_delta": round(wa["median_delta"], 2),
                     "p_wilcoxon": round(wa["p_value"], 4)})
    return pd.DataFrame(rows, columns=["model", "arm", "vs", "n_paired",
                                       "mae_arm", "mae_base", "delta_mae",
                                       "median_delta", "p_wilcoxon"])


# ------------------------------------------------------------------ figures
def make_figures(df: pd.DataFrame, summary: pd.DataFrame, out: Path,
                 seed: int) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    made = []
    models = [m for m in summary.model.unique() if m != "(no LLM)"]
    base = summary[summary.model == "(no LLM)"].set_index("arm")

    # 1 -- MAE by arm and model, baselines as reference lines --------------
    fig, ax = plt.subplots(figsize=(9, 4.4))
    width = 0.8 / max(len(models), 1)
    xs = np.arange(len(LADDER))
    for j, m in enumerate(models):
        sub = summary[summary.model == m].set_index("arm")
        vals = [sub.loc[a, "mae"] if a in sub.index and
                pd.notna(sub.loc[a].get("mae")) else np.nan for a in LADDER]
        ax.bar(xs + j * width, vals, width, label=m)
    for a, ls, lab in (("B0_retrieval", "--", "B0 retrieval-only"),
                       ("B1_zknn", ":", "B1 z-kNN twin")):
        if a in base.index and pd.notna(base.loc[a].get("mae")):
            ax.axhline(base.loc[a, "mae"], ls=ls, c="k", lw=1.2, label=lab)
    ax.set_xticks(xs + (len(models) - 1) * width / 2)
    ax.set_xticklabels([a.split("_", 1)[1] for a in LADDER], fontsize=9)
    ax.set_ylabel("MAE (cycles)")
    ax.set_title(f"RUL MAE by agentic pattern (test anomalies, seed {seed})")
    ax.legend(fontsize=7, ncols=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figdir / f"fig_prog_mae_seed{seed}.{ext}", dpi=180)
    plt.close(fig)
    made.append(f"figures/fig_prog_mae_seed{seed}.png")

    # 2 -- interval calibration: coverage vs width -------------------------
    fig, ax = plt.subplots(figsize=(6, 4.4))
    for _, r in summary.iterrows():
        if pd.isna(r.get("coverage80")) or pd.isna(r.get("width_mean")):
            continue
        mk = "s" if r.model == "(no LLM)" else "o"
        ax.scatter(r.width_mean, r.coverage80, marker=mk, s=42)
        ax.annotate(f"{r.arm.split('_', 1)[1]}"
                    + ("" if r.model == "(no LLM)" else f"\n{r.model}"),
                    (r.width_mean, r.coverage80), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    ax.axhline(0.8, ls="--", c="grey", lw=1)
    ax.text(ax.get_xlim()[1], 0.805, "80% target", ha="right", fontsize=7,
            c="grey")
    ax.set_xlabel("mean interval width (cycles)")
    ax.set_ylabel("empirical coverage of rul_range")
    ax.set_title("Interval calibration: narrow AND covered is the frontier")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figdir / f"fig_prog_calibration_seed{seed}.{ext}",
                    dpi=180)
    plt.close(fig)
    made.append(f"figures/fig_prog_calibration_seed{seed}.png")

    # 3 -- predicted vs true for the best LLM arm + B0 ---------------------
    llm = summary[(summary.model != "(no LLM)")
                  & summary.mae.notna()] if "mae" in summary else None
    if llm is not None and len(llm):
        best = llm.sort_values("mae").iloc[0]
        fig, ax = plt.subplots(figsize=(5.2, 5))
        b0 = df[(df.arm == "B0_retrieval") & (~df.parse_failed)]
        ax.scatter(b0.gold_rul, b0.pred_rul, s=12, alpha=.45,
                   label=f"B0 retrieval-only (MAE "
                         f"{b0.abs_err.mean():.1f})")
        g = df[(df.arm == best.arm) & (df.model == best.model)
               & (~df.parse_failed)]
        ax.scatter(g.gold_rul, g.pred_rul, s=12, alpha=.55,
                   label=f"{best.arm} / {best.model} (MAE "
                         f"{g.abs_err.mean():.1f})")
        ax.plot([0, 125], [0, 125], c="k", lw=1)
        ax.set_xlabel("true RUL (clipped 125)")
        ax.set_ylabel("ticket rul_estimate")
        ax.set_title("Retrieval-grounded estimates vs ground truth")
        ax.legend(fontsize=7)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(figdir / f"fig_prog_scatter_seed{seed}.{ext}",
                        dpi=180)
        plt.close(fig)
        made.append(f"figures/fig_prog_scatter_seed{seed}.png")
    return made


# ---------------------------------------------------------------- headlines
def _fmt_p(p: float) -> str:
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def _verdict(delta: float, p: float) -> str:
    if p < 0.05:
        return ("significantly BETTER than" if delta < 0
                else "significantly WORSE than")
    return "statistically indistinguishable from"


def headlines(df: pd.DataFrame, summary: pd.DataFrame,
              paired: pd.DataFrame, figs: List[str], out: Path,
              seed: int, store_note: str) -> Path:
    L: List[str] = []
    A = L.append
    n_cases = df[df.arm == "B0_retrieval"][["unit", "cycle"]
                                           ].drop_duplicates().shape[0]
    n_models = df[~df.arm.isin(BASELINES)].model.nunique()
    fb = df[df.arm == "P2_rag"]
    fb_share = (float((fb.query_source == "summary_fallback").mean())
                if len(fb) else float("nan"))
    A(f"# A-RAD v8 -- retrieval-augmented prognostic decision layer: "
      f"headline results (seed {seed})")
    A("")
    A(f"Pipeline: test-unit anomaly -> its own edge interpretation as the "
      f"query -> semantic retrieval over the {store_note} -> agentic "
      f"maintenance ticket (RUL estimate, central-80% range, action 1-5).")
    A(f"Universe: {n_cases} stratified test-anomaly cases; {n_models} "
      f"model(s); every LLM arm paired case-by-case against the no-LLM "
      f"retrieval reference (Wilcoxon on |error|). Query fallback share "
      f"(no own interpretation): {fb_share:.0%}."
      if fb_share == fb_share else f"Universe: {n_cases} cases.")
    A("")

    base = summary[summary.model == "(no LLM)"].set_index("arm")
    b0 = base.loc["B0_retrieval"] if "B0_retrieval" in base.index else None
    b1 = base.loc["B1_zknn"] if "B1_zknn" in base.index else None

    # H1 -------------------------------------------------------------------
    A("## H1 -- the headline")
    llm = summary[(summary.model != "(no LLM)") & summary.get(
        "mae", pd.Series(dtype=float)).notna()]
    if len(llm) and b0 is not None:
        best = llm.sort_values("mae").iloc[0]
        pr = paired[(paired.model == best.model) & (paired.arm == best.arm)]
        if not len(pr):
            A(f"Best language arm so far: **{best.arm} / {best.model}**, "
              f"MAE {best.mae:.1f} cycles over {best.n_parsed} parsed "
              f"tickets (retrieval-only B0: {b0['mae']:.1f}). Too few "
              f"paired cases for the Wilcoxon test yet -- run more cases "
              f"before claiming a direction.")
        if len(pr):
            p = pr.iloc[0]
            A(f"The best language arm is **{best.arm} / {best.model}**: "
              f"MAE {best.mae:.1f} cycles [{best.mae_ci_lo:.1f}, "
              f"{best.mae_ci_hi:.1f}] over {best.n_parsed} parsed tickets, "
              f"{_verdict(p.delta_mae, p.p_wilcoxon)} the retrieval-only "
              f"reference B0 (MAE {p.mae_base:.1f} on the same "
              f"{p.n_paired} cases; delta {p.delta_mae:+.1f}, "
              f"{_fmt_p(p.p_wilcoxon)}).")
            A("")
            if p.p_wilcoxon >= 0.05:
                A("Read straight: on this benchmark the language layer's "
                  "estimate does not yet move the needle over its own "
                  "retrieval -- the kNN-twin lesson of the earlier RUL "
                  "study, reproduced at the interpretation level. The "
                  "value the ladder adds must therefore be argued from the "
                  "OTHER columns: calibration, action planning, "
                  "faithfulness and the verifier's repair behaviour.")
            else:
                A("The language layer beats its own retrieval on the same "
                  "evidence: the fusion of precedent outcomes with the "
                  "current-window context is doing real work.")
    A("")

    # H2 -------------------------------------------------------------------
    A("## H2 -- the structure ladder (MAE, per model)")
    A("")
    A("| model | " + " | ".join(a.split("_", 1)[1] for a in LADDER) + " |")
    A("|---" * (len(LADDER) + 1) + "|")
    for m in sorted(x for x in summary.model.unique() if x != "(no LLM)"):
        sub = summary[summary.model == m].set_index("arm")
        cells = []
        for a in LADDER:
            if a in sub.index and pd.notna(sub.loc[a].get("mae")):
                cells.append(f"{sub.loc[a,'mae']:.1f}"
                             f" ({sub.loc[a,'parse_rate']:.0%})")
            else:
                cells.append("--")
        A(f"| {m} | " + " | ".join(cells) + " |")
    if b0 is not None:
        A(f"| *(no LLM)* B0 semantic | " + " | ".join(
            [f"{b0['mae']:.1f}"] + ["--"] * (len(LADDER) - 1)) + " |")
    A("")
    A("Cells are MAE in cycles (protocol-compliance rate in parentheses); "
      "'--' = arm not run or nothing parsed. Compliance is the edge gate: "
      "an arm a small model cannot emit parseably is unusable regardless "
      "of its accuracy.")
    A("")

    # H3 -------------------------------------------------------------------
    A("## H3 -- what the semantic store itself is worth (B0 vs B1)")
    if b0 is not None and b1 is not None:
        d0 = df[(df.arm == "B0_retrieval") & (~df.parse_failed)].set_index(
            ["unit", "cycle"])
        d1 = df[(df.arm == "B1_zknn") & (~df.parse_failed)].set_index(
            ["unit", "cycle"])
        common = d0.index.intersection(d1.index)
        if len(common) >= 8:
            w = wilcoxon_paired(list(d0.loc[common, "abs_err"]),
                                list(d1.loc[common, "abs_err"]))
            A(f"With NO language model in either arm, retrieving by "
              f"interpretation semantics (B0: MAE {b0['mae']:.1f}) is "
              f"{_verdict(b0['mae'] - b1['mae'], w['p_value'])} the "
              f"embedder-free z-space twin (B1: MAE {b1['mae']:.1f}; "
              f"{_fmt_p(w['p_value'])}, n={len(common)}). This prices the "
              f"generated-interpretation representation itself, "
              f"independently of any agent.")
    A("")

    # H4 -------------------------------------------------------------------
    A("## H4 -- does the loop beat handing over the evidence? "
      "(P3 vs P2, per model)")
    for m in sorted(x for x in summary.model.unique() if x != "(no LLM)"):
        s2 = summary[(summary.model == m) & (summary.arm == "P2_rag")]
        s3 = summary[(summary.model == m) & (summary.arm == "P3_react")]
        if len(s2) and len(s3) and pd.notna(s2.iloc[0].get("mae")) \
                and pd.notna(s3.iloc[0].get("mae")):
            d2 = df[(df.model == m) & (df.arm == "P2_rag")
                    & (~df.parse_failed)].set_index(["unit", "cycle"])
            d3 = df[(df.model == m) & (df.arm == "P3_react")
                    & (~df.parse_failed)].set_index(["unit", "cycle"])
            common = d2.index.intersection(d3.index)
            tail = ""
            if len(common) >= 8:
                w = wilcoxon_paired(list(d3.loc[common, "abs_err"]),
                                    list(d2.loc[common, "abs_err"]))
                tail = f" ({_fmt_p(w['p_value'])}, n={len(common)})"
            A(f"- {m}: ReAct MAE {s3.iloc[0].mae:.1f} "
              f"(compliance {s3.iloc[0].parse_rate:.0%}) vs single-shot "
              f"RAG {s2.iloc[0].mae:.1f} "
              f"(compliance {s2.iloc[0].parse_rate:.0%}){tail}.")
    A("")

    # H5 -------------------------------------------------------------------
    A("## H5 -- the verifier as a runtime component")
    for m in sorted(x for x in summary.model.unique() if x != "(no LLM)"):
        s5 = summary[(summary.model == m) & (summary.arm == "P5_verifier")]
        s2 = summary[(summary.model == m) & (summary.arm == "P2_rag")]
        if len(s5) and pd.notna(s5.iloc[0].get("mae")):
            r5, r2 = s5.iloc[0], (s2.iloc[0] if len(s2) else None)
            bits = [f"- {m}: {r5.get('violations_pre_mean', float('nan')):.2f} "
                    f"gate violations per draft; escalation (unrepairable) "
                    f"rate {r5.get('escalation_rate', float('nan')):.0%}"]
            if r2 is not None and pd.notna(r2.get("faith_contradicted")) \
                    and pd.notna(r5.get("faith_contradicted")):
                bits.append(f"; contradicted-claim rate "
                            f"{r2.faith_contradicted:.1%} (P2) -> "
                            f"{r5.faith_contradicted:.1%} (P5)")
            if r2 is not None and pd.notna(r2.get("post_hoc_violations_mean")) \
                    and pd.notna(r5.get("post_hoc_violations_mean")):
                bits.append(f"; residual violations per ticket "
                            f"{r2.post_hoc_violations_mean:.2f} -> "
                            f"{r5.post_hoc_violations_mean:.2f}")
            A("".join(bits) + ".")
    A("")
    A("The gates are the RAD guardrails promoted from evaluation-time "
      "metrics to runtime acceptance checks with one violation-driven "
      "repair; what cannot be repaired is escalated to the human board "
      "rather than silently shipped.")
    A("")

    # H6 -------------------------------------------------------------------
    A("## H6 -- interval calibration (target: 80% coverage, narrow width)")
    cal = summary[summary.get("coverage80",
                              pd.Series(dtype=float)).notna()].copy()
    if len(cal):
        cal["cal_gap"] = (cal.coverage80 - 0.8).abs()
        for _, r in cal.sort_values(["cal_gap", "width_mean"]
                                    ).head(3).iterrows():
            A(f"- {r.arm} / {r.model}: coverage {r.coverage80:.0%} at mean "
              f"width {r.width_mean:.0f} cycles (Winkler "
              f"{r.winkler_mean:.0f}).")
        A("")
        A("(The calibrated-AND-narrow frontier; the no-LLM q10-q90 row "
          "shows what the raw precedent spread buys.)")
    A("")

    # H7 -------------------------------------------------------------------
    A("## H7 -- action planning (outcome-derived severity bands 1-5)")
    act = summary[summary.get("action_pm1",
                              pd.Series(dtype=float)).notna()]
    if len(act):
        bst = act.sort_values("action_pm1", ascending=False).iloc[0]
        A(f"Best action planner: **{bst.arm} / {bst.model}** with "
          f"{bst.action_pm1:.0%} within one band of the outcome-derived "
          f"action ({bst.action_exact:.0%} exact). Bands come from the "
          f"realised remaining life, never from the interpretations' own "
          f"gravity opinions (which showed no outcome correlation, "
          f"rho=0.039, in the v6 audit).")
    A("")

    # H8 -------------------------------------------------------------------
    A("## H8 -- edge capability ranking")
    A("")
    A("| model | compliance (all arms) | best-arm MAE | s/case | "
      "sim Jetson s/case |")
    A("|---|---|---|---|---|")
    for m in sorted(x for x in summary.model.unique() if x != "(no LLM)"):
        sub = summary[summary.model == m]
        comp = (sub.n_parsed.sum() / sub.n.sum()) if sub.n.sum() else np.nan
        okm = sub[sub.get("mae", pd.Series(dtype=float)).notna()]
        bm = okm.mae.min() if len(okm) else np.nan
        sec = sub.get("seconds_mean", pd.Series(dtype=float)).mean()
        sim = sub.get("sim_edge_s_mean", pd.Series(dtype=float)).mean()
        A(f"| {m} | {comp:.0%} | {bm:.1f} | {sec:.1f} | "
          + (f"{sim:.1f} |" if sim == sim else "-- |"))
    A("")

    A("## Notes for the paper")
    A("- S-scores are sums and n-dependent: `s_per_case` in the summary "
      "table is the per-case mean; compare arms only at matched n "
      "(the paired CSV gives matched-case MAE directly).")
    A("- Parse failures are first-class results, never imputed; every "
      "MAE is conditional on the shown compliance rate.")
    A(f"- Query source: {fb_share:.0%} of cases ran on the grounded-summary "
      f"fallback instead of their own interpretation."
      if fb_share == fb_share else "")
    if "HASH" in store_note.upper():
        A("- **WARNING: this report was produced on the HASH fallback "
          "embedder -- pipeline test only, never publish these numbers.**")
    A("")
    A("## Figures")
    for f in figs:
        A(f"![]({f})")
    A("")
    A("## Full aggregate table")
    A("")
    A(summary.to_markdown(index=False))
    A("")

    path = out / f"HEADLINES_seed{seed}.md"
    path.write_text("\n".join(L))
    return path


# --------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-seed", type=int, default=1)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = ROOT / a.out

    df = load_runs(out, a.sample_seed)
    summary = summarise(df)
    summary.to_csv(out / f"prognosis_summary_seed{a.sample_seed}.csv",
                   index=False)
    paired = paired_vs_baseline(df)
    paired.to_csv(out / f"prognosis_paired_seed{a.sample_seed}.csv",
                  index=False)

    store_note = "train-fleet interpretation store"
    info = ROOT / "data" / "vector_store" / "info.json"
    if info.exists():
        j = json.loads(info.read_text())
        store_note = (f"train-fleet interpretation store "
                      f"(n={j.get('n')}, {j.get('embedder')})")

    figs = make_figures(df, summary, out, a.sample_seed)
    hp = headlines(df, summary, paired, figs, out, a.sample_seed, store_note)

    print(f"[report-prog] wrote prognosis_summary_seed{a.sample_seed}.csv, "
          f"prognosis_paired_seed{a.sample_seed}.csv, {len(figs)} figures, "
          f"and {hp.name}")
    with pd.option_context("display.width", 200,
                           "display.max_columns", 24):
        cols = [c for c in ("model", "arm", "n", "parse_rate", "mae",
                            "coverage80", "width_mean", "action_pm1",
                            "faith_contradicted", "seconds_mean")
                if c in summary.columns]
        print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()

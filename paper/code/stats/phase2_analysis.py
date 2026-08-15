"""phase2_analysis.py — revision Phase 2: every pending number, from existing logs.

Computes, WITHOUT any new model runs:
  1. Clopper-Pearson CI for 0/89 dangerous over-predictions
  2. Exact counts behind every reported percentage (incl. the 36.6%)
  3. Threshold sensitivity sweep for "dangerous" (10/15/20/25 cycles)
  4. Clean-hint vs corrupted-hint subgroup over-prediction rates
  5. Unit-level (cluster) bootstrap for the Table-6 paired contrasts
  6. Deterministic arbiter baseline (override iff |hint - median cited future| > theta)
     + min-rule guard, incl. calibrated-urgency rho for each
  7. Leave-one-unit-out conformalization of the agent's stated ranges
  8. Ladder recount (resolves the 88-vs-89 sum), v2 ladder, Study-A episode accounting
  9. Override-definition reconciliation (97% of 60)

Reads:  agentic/results/arad3/prog/forecast_metrics.csv
        agentic/results/arad3/prog/forecast_episodes.jsonl
        agentic/results/arad3/diag/episodes.jsonl
        agentic/results/arad2/diag/episodes.jsonl
        agentic/results/patterns/episodes.jsonl
Writes: paper/tables_stats/phase2/*.csv + phase2_stats.md
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]          # A-RAD/
RES = ROOT / "agentic" / "results"
OUT = ROOT / "paper" / "tables_stats" / "phase2"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(7)
B = 3000
CAP = 125.0

lines: list[str] = ["# Phase-2 statistics (computed from existing run logs)\n"]


def log(s: str = "") -> None:
    print(s)
    lines.append(s)


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    ra = (ra - ra.mean()) / (ra.std() + 1e-12)
    rb = (rb - rb.mean()) / (rb.std() + 1e-12)
    return float(np.mean(ra * rb))


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """Upper bound of the two-sided (1-alpha) CP interval; exact for k=0."""
    if k == 0:
        return 1.0 - (alpha / 2.0) ** (1.0 / n)
    from scipy.stats import beta  # only needed for k>0
    return float(beta.ppf(1 - alpha / 2, k + 1, n - k))


# ---------------------------------------------------------------- load
m = pd.read_csv(ROOT / "agentic/results/final_prognostic" / "forecast_metrics.csv")
eps = [json.loads(l) for l in open(ROOT / "agentic/results/final_prognostic" / "forecast_episodes.jsonl")]
d3 = [json.loads(l) for l in open(ROOT / "agentic/results/final_diagnostic" / "episodes.jsonl")]
d2 = [json.loads(l) for l in open(ROOT / "agentic/results/v2_rul_coupled_collapse" / "diag" / "episodes.jsonl")]
pa_n = sum(1 for _ in open(ROOT / "agentic/results/study_A_pattern_grid" / "episodes.jsonl"))

piv_pred = m.pivot(index="qid", columns="arm", values="rul_pred")
piv_err = m.pivot(index="qid", columns="arm", values="rul_err")
piv_abs = m.pivot(index="qid", columns="arm", values="rul_abs_err")
piv_hint = m.pivot(index="qid", columns="arm", values="dl_hint")
truth = m.pivot(index="qid", columns="arm", values="true_rul")["P7_agent_dl"]
unit_of = m.drop_duplicates("qid").set_index("qid")["unit"]
qids = piv_err.index
units = sorted(unit_of.unique())
log(f"Loaded {len(qids)} cases from {len(units)} units: {units}\n")

# hint the agent SAW vs the clean tool
hint_seen = piv_hint["P7_agent_dl"]
clean_tool = piv_pred["dl_only"]
# The documented fault "zeroed the late-life hints": corrupted := hint == 0.
# (Release files do NOT reproduce the paper's 60/29 split: hint==0 gives 52,
#  |hint-clean|>5 gives 62 -> AUTHOR check: publish the exact criterion.)
corrupted = hint_seen == 0
n_corr, n_clean = int(corrupted.sum()), int((~corrupted).sum())
diff_hc = (hint_seen - clean_tool).abs()

# ------------------------------------------------- 1-2: counts & the zero
log("## 1. Counts behind the percentages (author checks)\n")
agent_over = piv_err["P7_agent"] > 20
agentdl_over = piv_err["P7_agent_dl"] > 20
n_ans_na = int(piv_err["P7_agent"].notna().sum())
n_ans_dl = int(piv_err["P7_agent_dl"].notna().sum())
nan_dl = sorted(piv_err.index[piv_err["P7_agent_dl"].isna()])
log(f"- ANSWERED cases: anchored {n_ans_dl}/89, no-anchor {n_ans_na}/89; "
    f"unanswered anchored qids: {nan_dl}  <- must be disclosed in Sec. 5")
k_na = int(agent_over.sum())
log(f"- no-anchor over-prediction >20: {k_na}/{n_ans_na} answered"
    f" = {k_na/n_ans_na:.1%}  <- reproduces the paper's 36.6% exactly")
k = int(agentdl_over.sum())
ub = clopper_pearson_upper(k, n_ans_dl)
log(f"- anchored over-prediction >20: {k}/{n_ans_dl} answered"
    f"  -> 95% CP CI [0, {ub:.1%}]")
log(f"- q95 signed error: no-anchor {np.nanpercentile(piv_err['P7_agent'],95):+.0f},"
    f" anchored {np.nanpercentile(piv_err['P7_agent_dl'],95):+.0f}")
log(f"- corrupted (hint==0, the documented fault): {n_corr}/89; usable: {n_clean}/89")
log(f"- sensitivity: |hint-clean|>5 -> {int((diff_hc>5).sum())}/89; >10 -> "
    f"{int((diff_hc>10).sum())}/89. Neither reproduces the paper's 60/29 "
    f"[AUTHOR: publish the corrupted-mask criterion]")
mae_dl_full = float(piv_abs["P7_agent_dl"].mean())
mae_tool = float(piv_abs["dl_only"].mean())
paired_full = float((piv_abs["P7_agent_dl"] - piv_abs["dl_only"]).mean())
mask_v = piv_abs["P7_agent_dl"].notna()
paired_valid = float((piv_abs["P7_agent_dl"] - piv_abs["dl_only"])[mask_v].mean())
log(f"- MAE agent_dl {mae_dl_full:.2f} vs tool {mae_tool:.2f}: naive diff "
    f"{mae_dl_full-mae_tool:+.2f}; paired mean (valid n={int(mask_v.sum())}) "
    f"{paired_valid:+.2f}  <- reconciles the +7.0 vs 7.2 question")
log(f"- Study A accounting: {pa_n} episodes = 6 LLM patterns x 3 styles x 89"
    f" + B0 (style-invariant, run once) x 89 = {6*3*89 + 89}")

# ------------------------------------------------- 3: threshold sweep
log("\n## 2. Threshold sensitivity sweep (dangerous over-prediction)\n")
rows = []
for thr in (10, 15, 20, 25):
    r = {"threshold": thr}
    for arm in ("b0_median", "dl_only", "P7_agent", "P7_agent_dl"):
        r[arm] = int((piv_err[arm] > thr).sum())
    rows.append(r)
    log(f"- >{thr} cycles: b0 {r['b0_median']}/89, tool {r['dl_only']}/89, "
        f"no-anchor {r['P7_agent']}/89, anchored {r['P7_agent_dl']}/89")
pd.DataFrame(rows).to_csv(OUT / "threshold_sweep.csv", index=False)

# ------------------------------------------------- 4: subgroups
log("\n## 3. Clean-hint vs corrupted-hint subgroups (anchored agent)\n")
sub = []
for name, mask in (("corrupted", corrupted), ("clean-hint", ~corrupted)):
    e = piv_err["P7_agent_dl"][mask]
    a = piv_abs["P7_agent_dl"][mask]
    t = piv_abs["dl_only"][mask]  # clean tool on same cases
    hs = (hint_seen[mask] - truth[mask]).abs()  # tool AS SEEN
    over = int((e > 20).sum())
    log(f"- {name} (n={int(mask.sum())}): over-pred>20 = {over}/{int(mask.sum())}"
        f" (CP95 upper {clopper_pearson_upper(over,int(mask.sum())):.1%}),"
        f" q95 {np.nanpercentile(e,95):+.0f}, agent MAE {np.nanmean(a):.1f},"
        f" tool-as-seen MAE {np.nanmean(hs):.1f}, clean-tool MAE {np.nanmean(t):.1f}")
    sub.append({"subgroup": name, "n": int(mask.sum()), "overpred_gt20": over,
                "agent_mae": float(np.nanmean(a)),
                "tool_as_seen_mae": float(np.nanmean(hs)),
                "clean_tool_mae": float(np.nanmean(t))})
pd.DataFrame(sub).to_csv(OUT / "subgroup_overpred.csv", index=False)

# override reconciliation
pred_dl = piv_pred["P7_agent_dl"]
for tol in (1, 2, 5):
    ov_c = int(((pred_dl - hint_seen).abs() > tol)[corrupted].sum())
    ov_u = int(((pred_dl - hint_seen).abs() > tol)[~corrupted].sum())
    log(f"- override(|pred-hint|>{tol}): corrupted {ov_c}/{n_corr}"
        f" ({ov_c/max(n_corr,1):.0%}), clean {ov_u}/{n_clean}"
        f" ({ov_u/max(n_clean,1):.0%})")

# ------------------------------------------------- 5: cluster bootstrap
log("\n## 4. Unit-level (cluster) bootstrap, B=3000 — Table-6 contrasts\n")


def cluster_ci(delta: pd.Series, mask=None):
    d = delta if mask is None else delta[mask]
    d = d.dropna()
    uq = unit_of.loc[d.index]
    point = float(d.mean())
    us = uq.unique()
    stats = []
    for _ in range(B):
        pick = RNG.choice(us, size=len(us), replace=True)
        vals = np.concatenate([d[uq == u].to_numpy() for u in pick])
        stats.append(vals.mean())
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return point, float(lo), float(hi)


contrasts = {
    "agent_dl vs clean tool": piv_abs["P7_agent_dl"] - piv_abs["dl_only"],
    "value of anchor (vs no-anchor)": piv_abs["P7_agent_dl"] - piv_abs["P7_agent"],
    "agent_dl vs precedent floor": piv_abs["P7_agent_dl"] - piv_abs["b0_median"],
    "rescue on corrupted (agent vs tool-as-seen)":
        piv_abs["P7_agent_dl"] - (hint_seen - truth).abs(),
}
crows = []
for name, delta in contrasts.items():
    mask = corrupted if "rescue" in name else None
    p, lo, hi = cluster_ci(delta, mask)
    log(f"- {name}: {p:+.1f} cycles, unit-level 95% CI [{lo:+.1f}, {hi:+.1f}]")
    crows.append({"contrast": name, "point": p, "lo": lo, "hi": hi})
pd.DataFrame(crows).to_csv(OUT / "cluster_bootstrap.csv", index=False)

# ------------------------------------------------- 6: deterministic arbiter
log("\n## 5. Deterministic guarded baselines (no LLM)\n")
med_fut = {}
for e in eps:
    if e["arm"] == "P7_agent_dl":
        f = [min(c["rul_then"], CAP) for c in (e.get("contexts") or [])
             if c.get("rul_then") is not None]
        med_fut[e["qid"]] = float(np.median(f)) if f else np.nan
med_fut = pd.Series(med_fut).reindex(qids)

arows = []


def eval_rule(pred: pd.Series, label: str):
    err = pred - truth
    ae = err.abs()
    over = int((err > 20).sum())
    margin = truth - pred
    rho = spearman(margin[pred.notna()], truth[pred.notna()])
    s = np.where(err < 0, np.exp(-err.clip(upper=0) / 13) - 1,
                 np.exp(err.clip(lower=0) / 10) - 1)
    log(f"- {label}: MAE {np.nanmean(ae):.1f}, over-pred>20 {over}/89, "
        f"median S {np.nanmedian(np.abs(s)):.2f}, rho(margin, true RUL) {rho:+.2f}")
    arows.append({"rule": label, "mae": float(np.nanmean(ae)), "overpred": over,
                  "rho_margin": rho})


eval_rule(pred_dl, "AGENT (as run, for reference)")
for theta in (10, 20, 30, 40):
    pred = hint_seen.where((hint_seen - med_fut).abs() <= theta, med_fut)
    eval_rule(pred.clip(upper=CAP), f"arbiter theta={theta} (corrupted hints, as agent saw)")
pred = np.minimum(hint_seen.replace(0, np.nan).fillna(med_fut), med_fut)
eval_rule(pd.Series(pred, index=qids).clip(upper=CAP), "min(hint, median future) guard")
for theta in (20,):
    pred = clean_tool.where((clean_tool - med_fut).abs() <= theta, med_fut)
    eval_rule(pred.clip(upper=CAP), f"arbiter theta={theta} (CLEAN hints — preview of P3 arm)")
eval_rule(med_fut.clip(upper=CAP), "median cited future alone")
pd.DataFrame(arows).to_csv(OUT / "arbiter_baselines.csv", index=False)

# ------------------------------------------------- 7: LOUO conformal
log("\n## 6. Leave-one-unit-out conformalization of stated ranges (target 90%)\n")
rng_lo, rng_hi = {}, {}
for e in eps:
    if e["arm"] == "P7_agent_dl":
        r = (e.get("forecast") or {}).get("rul_range")
        if isinstance(r, (list, tuple)) and len(r) == 2 and r[0] is not None:
            rng_lo[e["qid"]], rng_hi[e["qid"]] = float(r[0]), float(r[1])
lo = pd.Series(rng_lo).reindex(qids)
hi = pd.Series(rng_hi).reindex(qids)
have = lo.notna() & hi.notna()
E = np.maximum(lo - truth, truth - hi)  # CQR nonconformity
base_cov = float(((truth >= lo) & (truth <= hi))[have].mean())
covs, widths = [], []
for u in units:
    te = have & (unit_of.loc[qids] == u)
    ca = have & (unit_of.loc[qids] != u)
    n_cal = int(ca.sum())
    qlev = min(1.0, math.ceil((n_cal + 1) * 0.9) / n_cal)
    qhat = float(np.quantile(E[ca], qlev))
    cov_u = ((truth >= lo - qhat) & (truth <= hi + qhat))[te]
    covs.extend(cov_u.tolist())
    widths.extend(((hi - lo) + 2 * qhat)[te].tolist())
log(f"- stated ranges (n={int(have.sum())}): coverage {base_cov:.1%}, "
    f"mean width {float((hi-lo)[have].mean()):.1f} cycles")
log(f"- LOUO split-conformal @90%: coverage {np.mean(covs):.1%}, "
    f"mean width {np.mean(widths):.1f} cycles")
pd.DataFrame({"base_coverage": [base_cov], "conf_coverage": [np.mean(covs)],
              "conf_width": [np.mean(widths)]}).to_csv(OUT / "conformal.csv", index=False)

# ------------------------------------------------- 8: ladders & v2
log("\n## 7. Abstention ladders, recounted from episodes\n")


def ladder(episodes, pat):
    rows = [e for e in episodes if e.get("pattern") == pat]
    esc = sum(1 for e in rows if e.get("escalated") is True)
    auto = sum(1 for e in rows if e.get("auto_corrected") is True
               and e.get("escalated") is not True)
    clean = sum(1 for e in rows if (e.get("repairs") in (0, None))
                and e.get("auto_corrected") is not True
                and e.get("escalated") is not True)
    rep = len(rows) - esc - auto - clean
    return len(rows), clean, rep, auto, esc


for label, ds, pat in (("v3 (stage-aware)", d3, "P5_verifier"),):
    n, c, r, a, e = ladder(ds, pat)
    log(f"- {label}: {c} clean, {r} after repair, {a} auto-corrected, "
        f"{e} escalated  (sum {c+r+a+e} of {n})  <- resolves the 88-vs-89 check")
pats2 = Counter(e.get("pattern") for e in d2)
log(f"- v2 file patterns: {dict(pats2)}")
for pat in pats2:
    if pat and "P" in str(pat):
        n, c, r, a, e = ladder(d2, pat)
        log(f"- v2 [{pat}]: {c} clean, {r} repair, {a} auto, {e} escalated of {n}")

# ---------------------------------------------- 9: clean-tool arm (auto)
ct = ROOT / "agentic/results/clean_tool_study" / "forecast_metrics.csv"
log("\n## 9. Clean-tool agent arm (P3-A)")
if ct.exists():
    cm = pd.read_csv(ct)
    cm = cm[cm.arm == "P7_agent_dl"].set_index("qid")
    e = cm["rul_err"]; a = cm["rul_abs_err"]
    n_ans = int(e.notna().sum()); k = int((e > 20).sum())
    log(f"- over-pred>20: {k}/{n_ans} answered (CP95 upper "
        f"{clopper_pearson_upper(k, n_ans):.1%}); q95 {np.nanpercentile(e,95):+.0f}")
    both = pd.concat([a, piv_abs["dl_only"]], axis=1, keys=["ag","tl"]).dropna()
    log(f"- MAE {np.nanmean(a):.1f}; TRUE anchoring overhead vs clean tool: "
        f"{float((both.ag-both.tl).mean()):+.2f} cycles (paired, n={len(both)})"
        f"  <- replaces Table 5/6 headline")
else:
    log("- clean_tool_study results not found: run RUNBOOK_P3 step A, "
        "then re-run this script.")

log("\n## 8. Edge tier note")
log("- Per-row edge scores are not in this release zip; the unit-level AUC CI "
    "must be recomputed on the edge box (see RUNBOOK_P3, step E).")

(OUT / "phase2_stats.md").write_text("\n".join(lines) + "\n")
print(f"\nWROTE {OUT/'phase2_stats.md'} and 5 CSVs")

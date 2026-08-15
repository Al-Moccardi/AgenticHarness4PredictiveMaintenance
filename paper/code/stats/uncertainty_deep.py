#!/usr/bin/env python3
"""uncertainty_deep.py — two deep dives, CAP EXCLUDED (true RUL < 125):
(1) RUL vs the agent's signed difference: the conservativeness profile
    across the real (uncapped) life range.
(2) UNCERTAINTY vs MAE (agent+tool), with the anchoring mechanism made
    measurable: at low uncertainty the cited futures agree and pull the
    agent AWAY from the CNN-GRU hint; at high uncertainty the agent
    stays with the hint. Tested via distance-to-hint and
    distance-to-cited-median as functions of uncertainty.

Outputs -> agentic/results/analysis/
  (uncapped_uncertainty.md, figUNC1_rul_vs_error_uncapped,
   figUNC2_uncertainty_mechanism)
Run from repo root:  python paper/code/stats/uncertainty_deep.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "agentic"))
from apdm.prognostic.future_progression import (reliability,          # noqa
                                                future_progression)
from apdm.prognostic.forecast import PrecedentFutures                 # noqa

OUT = ROOT / "agentic/results/analysis"; OUT.mkdir(exist_ok=True,
                                                  parents=True)
RES = ROOT / "agentic/results"


def sp(a, b):
    a, b = pd.Series(a), pd.Series(b)
    m = a.notna() & b.notna()
    return float(a[m].rank().corr(b[m].rank()))


def partial_sp(a, b, c):
    ra, rb, rc = (pd.Series(x).rank() for x in (a, b, c))
    k = ra.notna() & rb.notna() & rc.notna()
    ra, rb, rc = ra[k], rb[k], rc[k]
    ea = ra - np.polyval(np.polyfit(rc, ra, 1), rc)
    eb = rb - np.polyval(np.polyfit(rc, rb, 1), rc)
    return float(pd.Series(ea).corr(pd.Series(eb)))


def qbin(x, y, k=5, fn="median"):
    q = pd.qcut(x, k, duplicates="drop")
    g = y.groupby(q, observed=True)
    mids = [i.mid for i in g.median().index]
    return mids, getattr(g, fn)(), g.count()


def main():
    m = pd.read_csv(RES / "final_prognostic/forecast_metrics.csv")
    P = m[m.arm == "P7_agent_dl"].set_index("qid")
    T = m[m.arm == "dl_only"].set_index("qid")
    eps = {e["qid"]: e for e in map(json.loads, open(
        RES / "final_prognostic/forecast_episodes.jsonl"))
        if e["arm"] == "P7_agent_dl"}
    pf = PrecedentFutures(ROOT / "agentic/data/vector_store/meta.jsonl")
    sig = {q: future_progression(pf, e.get("contexts"))["aggregate"]
           for q, e in eps.items()}
    d = pd.DataFrame({
        "true": T.true_rul, "hint": P.dl_hint,
        "est": P.rul_pred, "agent_s": P.rul_err,
        "agent_a": P.rul_abs_err, "tool_s": T.rul_err,
        "tool_a": T.rul_abs_err,
        "unc": pd.Series({q: s.get("uncertainty")
                          for q, s in sig.items()}),
        "med_fut": pd.Series({q: s.get("median_ttf")
                              for q, s in sig.items()}),
        "rel": pd.Series({q: reliability(
            eps[q].get("contexts"))["value"] for q in eps})})
    d = d[d.true < 125].copy()                     # CAP EXCLUDED
    d["d_hint"] = (d.est - d.hint).abs()
    d["d_fut"] = (d.est - d.med_fut).abs()

    S = [f"# Cap excluded (true RUL < 125): n={len(d)}, agent answered "
         f"{int(d.agent_s.notna().sum())}\n",
         "## 1. RUL vs agent difference — conservativeness profile"]
    for lo, hi, lab in ((0, 20, "$<$20"), (20, 40, "20-39"),
                        (40, 60, "40-59"), (60, 125, "60-124")):
        k = (d.true >= lo) & (d.true < hi)
        e = d.agent_s[k].dropna()
        t = d.tool_s[k].dropna()
        S.append(f"{lab:>7s} n={int(k.sum()):2d}: agent bias "
                 f"{e.mean():+6.1f} (q05 {e.quantile(.05):+6.1f}, q95 "
                 f"{e.quantile(.95):+6.1f}, over-pred {(e>0).mean():3.0%})"
                 f"  |  tool bias {t.mean():+6.1f}")
    e = d.agent_s.dropna()
    S.append(f"UNCAPPED overall: agent bias {e.mean():+.1f}, "
             f"under-pred {(e<0).mean():.0%}, over-pred "
             f"{(e>0).mean():.0%} (max over +{e.max():.0f}), "
             f"dangerous(>+20): {(e>20).sum()}")
    S.append(f"rho(true, agent signed error) = "
             f"{sp(d.true, d.agent_s):+.2f}  "
             "(more life -> deeper under-prediction)")

    S.append("\n## 2. Uncertainty vs MAE (agent+tool) — the mechanism")
    S.append(f"rho(unc, agent MAE) = {sp(d.unc, d.agent_a):+.2f}   "
             f"rho(unc, tool MAE) = {sp(d.unc, d.tool_a):+.2f}")
    S.append(f"rho(unc, agent signed) = {sp(d.unc, d.agent_s):+.2f}  "
             "(higher disagreement -> milder under-bias)")
    S.append("anchoring test:")
    S.append(f"  rho(unc, |est - hint|)         = "
             f"{sp(d.unc, d.d_hint):+.2f}   (consensus pulls away "
             "from the tool)")
    S.append(f"  rho(unc, |est - cited median|) = "
             f"{sp(d.unc, d.d_fut):+.2f}   (divergence releases the "
             "anchor)")
    S.append(f"  control: rho(unc, true) = {sp(d.unc, d.true):+.2f}, "
             f"rho(unc, rel) = {sp(d.unc, d.rel):+.2f}")
    S.append("\n## 3. Stage control + the fixed-anchor reading")
    S.append(f"agent estimate (uncapped): median {d.est.median():.0f}, "
             f"IQR [{d.est.quantile(.25):.0f}, "
             f"{d.est.quantile(.75):.0f}] -> a near-constant "
             "pessimistic prior")
    S.append(f"rho(est, true) = {sp(d.est, d.true):+.2f} | "
             f"rho(est, hint) = {sp(d.est, d.hint):+.2f} | "
             f"rho(est, cited median) = {sp(d.est, d.med_fut):+.2f}")
    S.append(f"partial rho(unc, agent MAE | true) = "
             f"{partial_sp(d.unc, d.agent_a, d.true):+.2f} (raw "
             f"{sp(d.unc, d.agent_a):+.2f}); within 20-40: "
             f"{sp(d.unc[(d.true>=20)&(d.true<40)], d.agent_a[(d.true>=20)&(d.true<40)]):+.2f}, "
             f"within 40-60: "
             f"{sp(d.unc[(d.true>=40)&(d.true<60)], d.agent_a[(d.true>=40)&(d.true<60)]):+.2f}")
    S.append("\n## 4. What the signals mean (for the ticket reader)")
    S.append("RELIABILITY: coverage - how similar this case is to the "
             "knowledge base. UNCERTAINTY: constraint - how much the "
             "known outcomes of those similar cases disagree (0 = "
             "unanimous, 1 = divergent). Neither predicts model error; "
             "they describe the EVIDENCE, not the estimator. High "
             "uncertainty is NOT good: it means the evidence supports "
             "several futures and the outlook range must widen. Low "
             "uncertainty is epistemically best - even though, for THIS "
             "anchored agent, unanimous long futures expose its fixed "
             "prior (hence the negative unc-MAE correlation: anchor "
             "geometry, not virtue of ignorance).")
    S.append("Reading: the estimate is a fixed ~14-16 cycle anchor that "
             "follows only the hint (rho +0.67) and ignores the cited "
             "futures (rho -0.05). Where the cited futures AGREE on a "
             "long survival, the anchor is exposed (large error); where "
             "they DIVERGE, their median collapses toward the anchor and "
             "the error shrinks. The uncertainty-MAE link (-0.24 stage-"
             "controlled) is anchor geometry, not agent adaptivity: the "
             "adaptive-anchoring hypothesis is falsified by "
             "rho(unc, |est-hint|) = -0.12.")
    (OUT / "uncapped_uncertainty.md").write_text("\n".join(S) + "\n")
    print("\n".join(S))

    print(f"[uncertainty_deep] -> {OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""signal_analysis.py — the ticket signals under the microscope:
reliability vs error, the uncertainty distribution, and the
uncertainty <-> reliability <-> MAE relationships.

Sources: results/final_prognostic (paired per-case tool/agent errors +
contexts -> signals recomputed) and results/progression_run (logged
uncertainty on fresh retrieval + horizon errors).
Outputs -> agentic/results/analysis/ (figures PNG+PDF + signal_stats.md)
Run from repo root:  python paper/code/stats/signal_analysis.py
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

OUT = ROOT / "agentic" / "results" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
RES = ROOT / "agentic" / "results"


def sp(a, b):
    a, b = pd.Series(a), pd.Series(b)
    m = a.notna() & b.notna()
    return float(a[m].rank().corr(b[m].rank()))


def binmed(x, y, edges):
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi) & y.notna()
        if m.sum() >= 4:
            xs.append((lo + hi) / 2); ys.append(float(y[m].median()))
    return xs, ys


def main():
    eps = [json.loads(l) for l in
           open(RES / "final_prognostic/forecast_episodes.jsonl")]
    eps = {e["qid"]: e for e in eps if e["arm"] == "P7_agent_dl"}
    pf = PrecedentFutures(ROOT / "agentic/data/vector_store/meta.jsonl")
    m = pd.read_csv(RES / "final_prognostic/forecast_metrics.csv")
    tool = m[m.arm == "dl_only"].set_index("qid")
    ag = m[m.arm == "P7_agent_dl"].set_index("qid")
    rows = {}
    for q, e in eps.items():
        rel = reliability(e.get("contexts"))
        fp = future_progression(pf, e.get("contexts"))
        rows[q] = {"rel": rel["value"],
                   "unc": fp["aggregate"]["uncertainty"]}
    d = pd.DataFrame(rows).T
    d["true"] = tool["true_rul"]
    d["tool_err"] = tool["rul_abs_err"]
    d["agent_err"] = ag["rul_abs_err"]
    # progression-run cross-check + horizon error
    pr = [json.loads(l) for l in
          open(RES / "progression_run/forecast_episodes.jsonl")]
    pr = {e["qid"]: e for e in pr if e["arm"] == "P7_progression"}
    d["unc_run"] = pd.Series({q: ((e.get("signals") or {})
                                  .get("future_progression", {})
                                  .get("aggregate", {}) or {})
                              .get("uncertainty") for q, e in pr.items()})
    d["hz_err"] = pd.Series(
        {q: (abs(float((e.get("forecast") or {}).get(
            "progression_horizon")) - min(float(e["true_rul"]), 125.0))
            if (e.get("forecast") or {}).get("progression_horizon")
            is not None else None) for q, e in pr.items()})

    S = []
    S.append("# Ticket-signal analysis (n = 89 anomalies)\n")
    S.append(f"UNCERTAINTY distribution: min {d.unc.min():.2f}, q25 "
             f"{d.unc.quantile(.25):.2f}, median {d.unc.median():.2f}, "
             f"q75 {d.unc.quantile(.75):.2f}, max {d.unc.max():.2f}")
    S.append(f"RELIABILITY range: [{d.rel.min():.3f}, {d.rel.max():.3f}]"
             f", median {d.rel.median():.3f}")
    S.append(f"stability across runs: rho(unc, unc_progression_run) = "
             f"{sp(d.unc, d.unc_run):+.2f}\n")
    S.append("## Correlations (Spearman)")
    pairs = [("rel", "unc"), ("rel", "tool_err"), ("rel", "agent_err"),
             ("unc", "tool_err"), ("unc", "agent_err"),
             ("unc", "hz_err"), ("unc", "true"), ("rel", "true")]
    for a, b in pairs:
        S.append(f"rho({a:>3s}, {b:<9s}) = {sp(d[a], d[b]):+.2f}")
    (OUT / "signal_stats.md").write_text("\n".join(S) + "\n")
    print("\n".join(S))

    print(f"[signal_analysis] -> {OUT}")


if __name__ == "__main__":
    main()

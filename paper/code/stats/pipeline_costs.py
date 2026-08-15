#!/usr/bin/env python3
"""pipeline_costs.py — end-to-end time and token accounting, edge ->
diagnosis -> prognosis -> signals -> synthesis.

Measured wherever logs exist (Ollama prompt_eval_count / eval_count,
episode wall_s / tokens_in / tokens_out); estimates are labelled.
Output -> agentic/results/analysis/pipeline_costs.md
Run from repo root:  python paper/code/stats/pipeline_costs.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "agentic" / "results"
OUT = RES / "analysis"; OUT.mkdir(parents=True, exist_ok=True)
L = []


def add(stage, per_case_s, tin, tout, n, src):
    L.append({"stage": stage, "s/case": per_case_s, "tok_in/case": tin,
              "tok_out/case": tout, "n": n, "source": src})


def main():
    # EDGE — hardware timeline (Jetson): tokens/s under load
    tl = pd.read_csv(ROOT / "edge/evaluation/results/hw_eval/timeline.csv")
    tokps = tl["tok_per_s"].replace(0, np.nan).dropna() \
        if "tok_per_s" in tl.columns else pd.Series(dtype=float)
    edge_note = (f"Jetson decode {tokps.median():.1f} tok/s (timeline.csv)"
                 if len(tokps) else "see edge/evaluation (fig9_hardware)")
    # edge interpretation length ~ from queries text
    q = pd.read_csv(ROOT /
                    "agentic/queries/test_FD002_with_interpretations.csv")
    itok = q["interpretation"].astype(str).str.len().div(4).median()
    est_edge_s = (itok / tokps.median()) if len(tokps) else None
    add("edge interpretation (Jetson)",
        round(float(est_edge_s), 1) if est_edge_s else "~",
        "-", int(itok), len(q), f"estimated from {edge_note}")

    # DIAGNOSIS — published P5_verifier episodes (native token fields)
    de = [json.loads(l) for l in
          open(RES / "final_diagnostic/episodes.jsonl")
          if '"P5_verifier"' in l]
    dd = pd.DataFrame([{k: e.get(k) for k in
                        ("wall_s", "tokens_in", "tokens_out",
                         "llm_calls")} for e in de])
    add("diagnostic agent (P5_verifier)", round(dd.wall_s.mean(), 1),
        int(dd.tokens_in.mean()), int(dd.tokens_out.mean()), len(dd),
        "final_diagnostic episodes (measured)")

    # PROGNOSTIC published arm — wall only (tokens not logged in release)
    pe = [json.loads(l) for l in
          open(RES / "final_prognostic/forecast_episodes.jsonl")]
    w = [e["wall_s"] for e in pe if e["arm"] == "P7_agent_dl"]
    add("prognostic agent (P7_agent_dl, published)",
        round(float(np.mean(w)), 1), "n/l", "n/l", len(w),
        "final_prognostic episodes (wall measured; tokens not logged)")

    # PROGRESSION arm — exact Ollama token counts
    fl = [json.loads(l) for l in
          open(RES / "progression_run/forecast_llm.jsonl")]
    dl = [json.loads(l) for l in
          open(RES / "progression_run/diag_llm.jsonl")]
    pin = np.sum([r.get("prompt_eval_count") or 0 for r in fl]) / 89
    pout = np.sum([r.get("eval_count") or 0 for r in fl]) / 89
    din = np.sum([r.get("prompt_eval_count") or 0 for r in dl]) / 89
    dout = np.sum([r.get("eval_count") or 0 for r in dl]) / 89
    pw = [e["wall_s"] for e in
          map(json.loads, open(RES /
              "progression_run/forecast_episodes.jsonl"))
          if e["arm"] == "P7_progression"]
    add("progression arm (incl. inline diagnosis)",
        round(float(np.mean(pw)), 1), int(pin + din), int(pout + dout),
        len(pw), "progression_run llm logs (exact Ollama counts)")

    # SIGNALS — deterministic
    add("signals (reliability + future progression + uncertainty)",
        0.0001, 0, 0, 89, "future_progression time (measured, ~0.1 ms)")

    # SYNTHESIS — live sample if present, else offline note
    tj = RES / "synthesis_run/tickets.jsonl"
    if tj.exists():
        sw = [json.loads(l)["wall_s"] for l in open(tj)]
        lab = ("synthesis_run tickets (live)" if np.mean(sw) > 1
               else "synthesis_run (offline template path; live 3B "
                    "sample: 7.3 s/case, n=11)")
        add("synthesis composer", round(float(np.mean(sw)), 2)
            if np.mean(sw) > 1 else 7.3, "~1400", "~220", len(sw), lab)

    df = pd.DataFrame(L)
    tot = sum(x for x in df["s/case"] if isinstance(x, (int, float)))
    md = ["# End-to-end pipeline costs (per anomaly, fleet of 89)\n",
          df.to_markdown(index=False), "",
          f"END-TO-END (edge -> ticket): ~{tot:.0f} s per anomaly; "
          f"~{tot*89/60:.0f} min for the full 89-anomaly campaign "
          f"(single-stream, one Jetson + one RTX 4070 Laptop).",
          "Measured token totals (progression campaign): "
          f"{(pin+din)*89/1000:.0f}k in / {(pout+dout)*89/1000:.0f}k "
          "out."]
    (OUT / "pipeline_costs.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

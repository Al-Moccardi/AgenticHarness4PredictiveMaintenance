#!/usr/bin/env python3
r"""run.py — synthesize the final ticket for every anomaly, from the REAL
pipeline events (edge interpretation, diagnostic ticket, prognostic
forecast, live-computed signals; progression_run fields merged when
present). Collects wall-time per ticket.

  python -m apdm.synthesis.run --backend mock                 (offline)
  python -m apdm.synthesis.run --backend ollama               (live 3B)
  python -m apdm.synthesis.run --stats --out ...              (timing table)
Output: results\synthesis_run\tickets.jsonl + final_tickets.md
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from apdm.synthesis.compose2 import compose, MockComposer
from apdm.prognostic.future_progression import (reliability,
                                                future_progression)
from apdm.prognostic.forecast import PrecedentFutures

AG = Path(__file__).resolve().parents[2]


def gather():
    q = pd.read_csv(AG / "queries/test_FD002_with_interpretations.csv"
                    ).set_index(["Unit_ID", "cycles"])
    diag = {e["qid"]: e for e in map(json.loads, open(
        AG / "results/final_diagnostic/episodes.jsonl"))
        if e.get("pattern") == "P5_verifier"}
    prog = {e["qid"]: e for e in map(json.loads, open(
        AG / "results/final_prognostic/forecast_episodes.jsonl"))
        if e["arm"] == "P7_agent_dl"}
    pr = {}
    prf = AG / "results/progression_run/forecast_episodes.jsonl"
    if prf.exists():
        pr = {e["qid"]: (e.get("forecast") or {}) for e in
              map(json.loads, open(prf)) if e["arm"] == "P7_progression"}
    pf = PrecedentFutures(AG / "data/vector_store/meta.jsonl")
    out = []
    for qid, p in sorted(prog.items(), key=lambda kv: (kv[1]["unit"],
                                                       kv[1]["cycle"])):
        d = (diag.get(qid) or {}).get("ticket") or {}
        fc = dict(p.get("forecast") or {})
        fc["dl_hint"] = p.get("dl_hint")
        fc.pop("expected_trends", None)   # measured-at-chance field: drop
        if qid in pr:
            fc["projected_progression"] = pr[qid].get(
                "projected_progression")
            fc["progression_horizon"] = pr[qid].get("progression_horizon")
        row = q.loc[(int(p["unit"]), int(p["cycle"]))]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        sig = {"reliability": reliability(p.get("contexts")),
               "future_progression": future_progression(
                   pf, p.get("contexts"))}
        out.append({"qid": qid, "unit": p["unit"], "cycle": p["cycle"],
                    "edge": str(row["interpretation"]),
                    "severity": d.get("severity"),
                    "action": d.get("action"),
                    "diagnosis": d.get("diagnosis"),
                    "cited_precedents": d.get("cited_precedents") or [],
                    "prognostic": fc, "signals": sig})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock",
                    choices=["mock", "ollama"])
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(AG / "results/synthesis_run"))
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ep = out / "tickets.jsonl"
    if a.stats:
        rows = [json.loads(l) for l in open(ep)]
        w = [r["wall_s"] for r in rows]
        lad = pd.Series([r["state"] for r in rows]).value_counts().to_dict()
        print(f"synthesis tickets: {len(rows)} | ladder {lad}")
        print(f"wall_s: mean {np.mean(w):.2f}  median {np.median(w):.2f}  "
              f"p95 {np.percentile(w, 95):.2f}  total {sum(w)/60:.1f} min")
        return
    done = {json.loads(l)["qid"] for l in open(ep)} if ep.exists() else set()
    if a.backend == "ollama":
        from apdm.llm import Ollama
        be = Ollama(model=a.model, num_ctx=8192, num_predict=420,
                    log_path=out / "llm.jsonl")
    else:
        be = MockComposer()
    items = gather()
    if a.limit:
        items = items[:a.limit]
    todo = [i for i in items if i["qid"] not in done]
    print(f"[synthesis v3 json-first] {len(items)} tickets | "
          f"todo {len(todo)} | backend={a.backend}")
    md = out / "final_tickets.md"
    with ep.open("a") as f, md.open("a") as g:
        for i, inp in enumerate(todo):
            t0 = time.time()
            text, state, viol = compose(be, inp)
            rec = {"qid": inp["qid"], "unit": inp["unit"],
                   "cycle": inp["cycle"], "state": state,
                   "notes": viol, "ticket_md": text,
                   "wall_s": round(time.time() - t0, 3)}
            f.write(json.dumps(rec, default=str) + "\n"); f.flush()
            g.write(text + "\n\n---\n\n"); g.flush()
            if a.backend == "ollama":
                print(f"  [{i+1}/{len(todo)}] {inp['qid']:>10s} {state:<18s}"
                      f" {rec['wall_s']:5.1f}s", flush=True)
    print(f"[synthesis] -> {ep}\n[synthesis] -> {md}")


if __name__ == "__main__":
    main()

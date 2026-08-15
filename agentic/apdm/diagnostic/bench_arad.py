"""bench_arad — the A-RAD two-agent study (reduced, architecture-first).

Replaces the exploratory 7x3 grid with the system the paper actually
proposes, on the SAME eight edge units:

STUDY D — the DIAGNOSTIC agent vs its floors (4 arms, style: plain)
    B0_retrieval   no LLM, neighbour-median ticket      (prices the LLM)
    P1_direct      LLM without retrieval                (prices the KB)
    P2_rag         single-shot grounded                 (the standard)
    P5_verifier    A-RAD diagnostic agent: retrieve -> ticket ->
                   deterministic gates (incl. G7 span) -> repair -> escalate

STUDY P — the PROGNOSTIC agent WITH TOOLS (4 arms)
    b0_median      no LLM: median rul_then of retrieved precedents
    dl_only        the CNN-GRU number alone
    P7_agent       tool-using agent: search_memory / read_future / finish
    P7_agent_dl    same + the dl_predict tool (CNN-GRU on demand)
  The prognostic agent RECEIVES THE DIAGNOSTIC AGENT'S TICKET for the same
  anomaly (true two-agent pipeline), is unit-chained (sees its own previous
  prognosis; cap 125; monotone), and is scored by the existing deterministic
  evaluator: realised future trends + RUL_FD002-anchored gold.

Output schemas are IDENTICAL to bench_patterns / bench_forecast, so the
whole existing evaluation stack applies unchanged:
    results/arad/diag/episodes.jsonl   -> eval_patterns.py --dir results/arad/diag
    results/arad/prog/forecast_episodes.jsonl
        -> bench_forecast --evaluate/--report --out results/arad/prog
        -> eval_forecast_agent.py --dir results/arad/prog

Run:
    python -m apdm.bench_arad --smoke          # offline plumbing (~1 min)
    python -m apdm.bench_arad                  # both studies (resumable)
    python -m apdm.bench_arad --study diag     # or one at a time
    python -m apdm.bench_arad --study prog
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..prognostic.bench_forecast import _dl_hints
from ..prognostic.forecast import (RUL_CAP, PrecedentFutures, parse_forecast,
                       unit_history)
from ..llm import Backend, Ollama
from ..patterns import (DiagRunner, MockBackend, Runner, case_block,
                       fmt_precedents, gates_d, load_queries)
from ..vector_store import HashEmbedder, OllamaEmbedder, VectorStore

ROOT = Path(__file__).resolve().parents[2]
DIAG_ARMS = ["B0_retrieval", "P1_direct", "P2_rag", "P5_verifier"]
PROG_ARMS = ["b0_median", "dl_only", "P7_agent", "P7_agent_dl"]

F_TICKET = ('{"progression_narrative": "<2-4 sentences>", '
            '"expected_trends": [{"sensor": "<T24|T30|T50|P30|Nf|Nc|Ps30|'
            'phi|NRf|NRc|BPR|htBleed|W31|W32>", "direction": '
            '"up|down|stable"}, ...3-5...], '
            '"anomaly_outlook": "accelerating|steady|sporadic", '
            '"rul_estimate": <int 0-125>, "rul_range": [<lo>, <hi>], '
            '"cited_precedents": ["u<unit>c<cycle>", ...], '
            '"confidence": "low|med|high"}')

TOOL_SYS = ("You are the fleet PROGNOSTIC agent. Forecast THIS SPECIFIC "
            "unit's degradation using tools. RUL is capped at 125 and can "
            "only DECREASE over cycles; stay consistent with your own "
            "previous prognosis when shown. Ground the estimate in the "
            "futures of the precedents you cite. FINISH AS SOON AS your "
            "evidence is sufficient - do not exhaust the step budget. "
            "Respond with ONE JSON object per turn: a tool call or the "
            "finish.")


def _tool_proto(with_dl: bool) -> str:
    tools = ['{"tool": "search_memory", "query": "<text>"}',
             '{"tool": "read_future", "id": "u<unit>c<cycle>"}  '
             '(what happened AFTER that precedent, to failure)']
    if with_dl:
        tools.append('{"tool": "dl_predict"}  (CNN-GRU RUL estimate for '
                     'THIS unit at THIS cycle)')
    tools.append('{"tool": "finish", "forecast": ' + F_TICKET + "}")
    return "TOOLS (choose ONE per turn):\n" + "\n".join(tools)


class MockToolBackend(Backend):
    """Deterministic tool-following mock for offline smoke only."""
    name = "mocktool"
    model = "mocktool"

    def __init__(self, log_path=None):
        self.log_path = log_path

    def generate(self, prompt, system=None):
        ids = re.findall(r"\[(u\d+c\d+)\]", prompt)[:2]
        if "dl_predict" in prompt and "DL MODEL SAYS" not in prompt:
            out = json.dumps({"tool": "dl_predict"})
        elif "read_future" in prompt and "afterwards:" not in prompt and ids:
            out = json.dumps({"tool": "read_future", "id": ids[0]})
        else:
            hint = re.findall(r"DL MODEL SAYS RUL = (\d+)", prompt)
            ruls = [float(x) for x in
                    re.findall(r"rul_then=(\d+\.?\d*)", prompt)]
            est = int(hint[0]) if hint else (int(np.median(ruls))
                                             if ruls else 60)
            out = json.dumps({"tool": "finish", "forecast": {
                "progression_narrative": "mock forecast",
                "expected_trends": [{"sensor": "T50", "direction": "up"},
                                    {"sensor": "phi", "direction": "down"},
                                    {"sensor": "Ps30", "direction": "up"}],
                "anomaly_outlook": "accelerating",
                "rul_estimate": min(est, 125),
                "rul_range": [max(est - 20, 0), min(est + 20, 125)],
                "cited_precedents": ids, "confidence": "med"}})
        self._log(system, prompt, out)
        return out


def prog_tool_agent(be: Backend, q: Dict, run: Runner, pf: PrecedentFutures,
                    queries: List[Dict], diag_ticket: Optional[Dict],
                    dl_hint: Optional[float], with_dl: bool,
                    prev: Optional[Dict], max_steps: int = 3) -> Dict:
    seen: Dict[str, Dict] = {}
    for r in run.retrieve(q):
        seen[f"u{r['unit']}c{r['cycle']}"] = r
    obs = fmt_precedents(list(seen.values()))
    extra: List[str] = []
    trace: List[Dict] = []
    dl_used = False
    proto = _tool_proto(with_dl)
    base = [case_block(q), unit_history(queries, q)]
    if diag_ticket:
        base.append("DIAGNOSTIC AGENT'S TICKET for this anomaly:\n"
                    + json.dumps(diag_ticket))
    if prev:
        el = q["cycle"] - prev["cycle"]
        base.append(f"YOUR PREVIOUS PROGNOSIS for THIS unit at cycle "
                    f"{prev['cycle']} estimated RUL = {prev['rul']:.0f}. "
                    f"New estimate must not exceed that value.")
    for step in range(max_steps):
        nudge = ("" if step < max_steps - 1 else
                 " This is your LAST tool step - prefer finish.")
        p = ("\n\n".join(base)
             + f"\n\nOBSERVATIONS:\n{obs}"
             + ("\n" + "\n".join(extra) if extra else "")
             + f"\n\nStep {step+1}/{max_steps}.{nudge} " + proto)
        out = be.generate(p, system=TOOL_SYS)
        m = re.search(r"\{.*\}", out or "", re.S)
        act = None
        if m:
            try:
                act = json.loads(m.group(0))
            except Exception:  # noqa: BLE001
                act = None
        trace.append({"step": step, "raw": (out or "")[:280]})
        if not isinstance(act, dict):
            fc = parse_forecast(out)
            if fc and fc.get("rul_estimate") is not None:
                return {"forecast": fc, "contexts": list(seen.values()),
                        "trace": trace, "dl_used": dl_used,
                        "steps": step + 1}
            continue
        tool = str(act.get("tool", ""))
        if tool == "finish":
            fc = parse_forecast(json.dumps(act.get("forecast", {})))
            return {"forecast": fc, "contexts": list(seen.values()),
                    "trace": trace, "dl_used": dl_used, "steps": step + 1}
        if tool == "search_memory":
            qq = dict(q)
            qq["interpretation"] = str(act.get("query")
                                       or q["interpretation"])
            for r in run.retrieve(qq):
                seen[f"u{r['unit']}c{r['cycle']}"] = r
            obs = fmt_precedents(list(seen.values()))
        elif tool == "read_future":
            pid = str(act.get("id", ""))
            mm = re.match(r"u(\d+)c(\d+)", pid)
            if mm:
                extra.append(f"FUTURE of {pid}: "
                             + pf.future_of(int(mm.group(1)),
                                            int(mm.group(2))))
            else:
                extra.append(f"(future of {pid}: id not understood)")
        elif tool == "dl_predict" and with_dl:
            dl_used = True
            extra.append("DL MODEL SAYS RUL = "
                         + (f"{dl_hint:.0f}" if dl_hint is not None
                            else "unavailable")
                         + " cycles (CNN-GRU, capped, monotone).")
    # out of steps: force the forecast from gathered evidence
    out = be.generate("\n\n".join(base) + f"\n\nOBSERVATIONS:\n{obs}"
                      + ("\n" + "\n".join(extra) if extra else "")
                      + "\n\nOut of tool steps. OUTPUT the finish JSON now:"
                        "\n" + F_TICKET, system=TOOL_SYS)
    fc = parse_forecast(out)
    return {"forecast": fc, "contexts": list(seen.values()),
            "trace": trace, "dl_used": dl_used, "steps": max_steps + 1,
            "raw_final": (out or "")[:1500]}


# --------------------------------------------------------------- studies
def study_diag(a, run: DiagRunner, queries: List[Dict]) -> Path:
    print("[arad/diag] protocol: RETROSPECTIVE case-matching "
          "(progression vs precedent histories; no RUL, no futures)")
    out = Path(a.out) / "diag"
    out.mkdir(parents=True, exist_ok=True)
    ep = out / "episodes.jsonl"
    done = set()
    if ep.exists():
        for line in ep.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["qid"], r["pattern"], r["style"]))
    todo = [(q, p) for q in queries for p in DIAG_ARMS
            if (q["qid"], p, "plain") not in done]
    print(f"[arad/diag] {len(queries)} queries x {len(DIAG_ARMS)} arms "
          f"({len(done)} done, {len(todo)} to run)")
    be = run.be
    with ep.open("a") as f:
        for i, (q, p) in enumerate(todo):
            tok0 = be.totals() if p != "B0_retrieval" else None
            t0 = time.time()
            has_hist = any(x["unit"] == q["unit"] and x["cycle"] < q["cycle"]
                           for x in queries)
            try:
                r = run.run_diag(p, q, has_hist)
                err = None
            except Exception as e:  # noqa: BLE001
                r, err = {"ticket": None, "contexts": []}, \
                    f"{type(e).__name__}: {e}"
            tok1 = be.totals() if tok0 else None
            shown = [f"u{c['unit']}c{c['cycle']}"
                     for c in r.get("contexts", [])]
            f.write(json.dumps({
                "qid": q["qid"], "pattern": p, "style": "plain",
                "true_rul": q.get("true_rul"),
                "case_text": case_block(q), "ticket": r.get("ticket"),
                "contexts": [{k: c.get(k) for k in
                              ("unit", "cycle", "rul_then", "gravity",
                               "similarity", "text")}
                             for c in r.get("contexts", [])],
                "gate_violations": gates_d(r.get("ticket"), shown, has_hist),
                "has_history": has_hist,
                "steps": r.get("steps"), "repairs": r.get("repairs"),
                "escalated": r.get("escalated"),
                "auto_corrected": r.get("auto_corrected"),
                "trace": r.get("trace"),
                "wall_s": round(time.time() - t0, 3),
                "tokens_in": (tok1["prompt_tokens"] - tok0["prompt_tokens"])
                if tok0 else 0,
                "tokens_out": (tok1["completion_tokens"]
                               - tok0["completion_tokens"]) if tok0 else 0,
                "llm_calls": (tok1["n_calls"] - tok0["n_calls"])
                if tok0 else 0,
                "error": err}, default=str) + "\n")
            f.flush()
            tk = r.get("ticket") or {}
            nv = len(gates_d(r.get("ticket"), shown, has_hist))
            print(f"  [{i+1}/{len(todo)}] {q['qid']:>10s} {p:<13s} "
                  f"{time.time()-t0:6.1f}s sev={tk.get('severity')} "
                  f"act={str(tk.get('action'))[:14]} viol={nv} "
                  f"rep={r.get('repairs') or 0}"
                  f"{' AC' if r.get('auto_corrected') else ''}", flush=True)
    return out


def study_prog(a, run: Runner, queries: List[Dict], be_f: Backend) -> Path:
    out = Path(a.out) / "prog"
    out.mkdir(parents=True, exist_ok=True)
    ep = out / "forecast_episodes.jsonl"
    done = set()
    if ep.exists():
        for line in ep.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["qid"], r["arm"]))
    pf = PrecedentFutures(Path(a.store_dir) / "meta.jsonl")
    hints = _dl_hints(Path(a.dl_hints))
    # the two-agent chain: diagnostic agent's (P5) tickets feed the prognostic
    diag_map: Dict[str, Dict] = {}
    dpath = Path(a.out) / "diag" / "episodes.jsonl"
    if dpath.exists():
        for line in dpath.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r["pattern"] == "P5_verifier" and r.get("ticket"):
                    diag_map[r["qid"]] = r["ticket"]
    print(f"[arad/prog] diagnostic tickets available for "
          f"{len(diag_map)}/{len(queries)} queries")
    if len(hints) == 0 and not a.allow_missing:
        raise SystemExit(
            "[arad/prog] ABORT: 0 DL hints found at the --dl-hints path.\n"
            "  Run first:  python -m apdm.dl_rul --train --epochs 100\n"
            "              python -m apdm.dl_rul --hints\n"
            "  (override only for debugging with --allow-missing)")
    if len(diag_map) == 0 and not a.allow_missing:
        raise SystemExit(
            "[arad/prog] ABORT: 0 diagnostic tickets found under "
            f"{Path(a.out) / 'diag'}.\n"
            "  Run first:  python -m apdm.bench_arad --study diag --out "
            + str(a.out) + "\n  (or copy an existing diag/episodes.jsonl "
            "there; override with --allow-missing)")
    grid = [(q, arm) for q in queries for arm in PROG_ARMS]
    todo = [(q, arm) for (q, arm) in grid if (q["qid"], arm) not in done]
    print(f"[arad/prog] {len(queries)} x {len(PROG_ARMS)} arms "
          f"({len(done)} done, {len(todo)} to run) | dl hints {len(hints)}")
    prev_fc: Dict = {}
    with ep.open("a") as f:
        for i, (q, arm) in enumerate(todo):
            t0 = time.time()
            hint = hints.get((q["unit"], q["cycle"]))
            fc, err, dl_used, steps, raw_final = None, None, None, None, None
            recs = run.retrieve(q)
            try:
                if arm == "b0_median":
                    ruls = [r["rul_then"] for r in recs
                            if r["rul_then"] is not None]
                    est = (min(float(np.median(ruls)), RUL_CAP)
                           if ruls else None)
                    fc = {"rul_estimate": est,
                          "rul_range": [float(np.quantile(ruls, .1)),
                                        min(float(np.quantile(ruls, .9)),
                                            RUL_CAP)] if ruls else None,
                          "expected_trends": [], "anomaly_outlook": None,
                          "cited_precedents":
                              [f"u{r['unit']}c{r['cycle']}" for r in recs]}
                elif arm == "dl_only":
                    fc = ({"rul_estimate": hint, "rul_range": None,
                           "expected_trends": [], "anomaly_outlook": None,
                           "cited_precedents": []}
                          if hint is not None else None)
                else:
                    r7 = prog_tool_agent(
                        be_f, q, run, pf, queries,
                        diag_ticket=diag_map.get(q["qid"]),
                        dl_hint=hint, with_dl=(arm == "P7_agent_dl"),
                        prev=prev_fc.get((arm, q["unit"])))
                    fc, dl_used, steps = (r7["forecast"], r7["dl_used"],
                                          r7["steps"])
                    raw_final = r7.get("raw_final")
                    if fc and fc.get("rul_estimate") is not None:
                        prev_fc[(arm, q["unit"])] = {
                            "cycle": q["cycle"],
                            "rul": min(float(fc["rul_estimate"]), RUL_CAP)}
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
            f.write(json.dumps({
                "qid": q["qid"], "arm": arm, "unit": q["unit"],
                "cycle": q["cycle"], "true_rul": q["true_rul"],
                "dl_hint": hint, "dl_used": dl_used, "steps": steps,
                "diagnosis": diag_map.get(q["qid"]), "forecast": fc,
                "raw_final": raw_final,
                "contexts": [{k: r.get(k) for k in
                              ("unit", "cycle", "rul_then", "similarity")}
                             for r in recs],
                "wall_s": round(time.time() - t0, 2), "error": err},
                default=str) + "\n")
            f.flush()
            est = (fc or {}).get("rul_estimate")
            tr = q.get("true_rul")
            trc = min(float(tr), RUL_CAP) if tr is not None else None
            print(f"  [{i+1}/{len(todo)}] {q['qid']:>10s} {arm:<12s} "
                  f"{time.time()-t0:5.1f}s steps={steps or 0} "
                  f"dl={'Y' if dl_used else '-'} "
                  f"est={est if est is None else round(float(est))} "
                  f"true={trc if trc is None else round(trc)}", flush=True)
    return out


def smoke() -> None:
    import shutil
    tmp = ROOT / "results" / "arad_smoke"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    meta = [json.loads(l) for l in
            (ROOT / "data/vector_store/meta.jsonl").read_text().splitlines()
            if l.strip()][:500]
    emb = HashEmbedder()
    vs = VectorStore(emb.embed([m["text"] for m in meta]), meta, emb.name)
    (tmp / "store").mkdir()
    vs.save(tmp / "store")
    import pandas as pd
    qcsv = ROOT / "queries/test_FD002_with_interpretations.csv"
    qd = pd.read_csv(qcsv)
    an = qd[qd.anomaly_label == -1][["Unit_ID", "cycles"]].head(60)
    pd.DataFrame({"unit_ID": an.Unit_ID, "cycle": an.cycles,
                  "dl_rul_raw": 70.0, "dl_rul": 70.0}
                 ).to_csv(tmp / "hints.csv", index=False)
    argv = ["--allow-missing",
            "--queries", str(qcsv), "--store-dir", str(tmp / "store"),
            "--embedder", "hash", "--backend", "mock",
            "--dl-hints", str(tmp / "hints.csv"), "--limit", "2",
            "--out", str(tmp)]
    a = build_parser().parse_args(argv)
    run = Runner(MockBackend(), VectorStore.load(tmp / "store"),
                 HashEmbedder(), k=a.k, stage_aware=True)
    pf_hist = PrecedentFutures(tmp / "store" / "meta.jsonl")
    from ..prognostic.forecast import unit_history as _uh
    qs = load_queries(Path(a.queries))[:a.limit]
    drun = DiagRunner(MockBackend(), VectorStore.load(tmp / "store"),
                      HashEmbedder(), k=a.k, stage_aware=True, hist=pf_hist,
                      unit_hist_fn=lambda q: _uh(qs, q), queries=qs)
    study_diag(a, drun, qs)
    study_prog(a, run, qs, MockToolBackend())
    de = sum(1 for _ in (tmp / "diag/episodes.jsonl").open())
    pe = sum(1 for _ in (tmp / "prog/forecast_episodes.jsonl").open())
    ok = de == 2 * len(DIAG_ARMS) and pe == 2 * len(PROG_ARMS)
    # prove the downstream evaluators accept these files
    import subprocess, sys
    r = subprocess.run([sys.executable, "-m", "apdm.bench_forecast",
                        "--evaluate", "--out", str(tmp / "prog"),
                        "--detections",
                        str(ROOT / "queries/test_anomalies.csv"),
                        "--test-txt", str(ROOT / "data/test_FD002.txt")],
                       capture_output=True, text=True)
    ok2 = (tmp / "prog/forecast_summary.csv").exists()
    print(f"\nSMOKE: diag {de}/8 prog {pe}/8 evaluator_ok={ok2}")
    print("SMOKE " + ("PASSED" if (ok and ok2) else "FAILED\n"
                      + r.stdout[-500:] + r.stderr[-500:]))


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=str(
        ROOT / "queries/test_FD002_with_interpretations.csv"))
    ap.add_argument("--store-dir", default=str(ROOT / "data/vector_store"))
    ap.add_argument("--dl-hints", default=str(ROOT / "queries/dl_hints.csv"))
    ap.add_argument("--embedder", default="ollama",
                    choices=["ollama", "hash"])
    ap.add_argument("--backend", default="ollama",
                    choices=["ollama", "mock"])
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--study", default="both",
                    choices=["both", "diag", "prog"])
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--num-ctx", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-stage-rerank", dest="stage", action="store_false",
                    default=True, help="ablation: plain cosine retrieval")
    ap.add_argument("--allow-missing", action="store_true",
                    help="debug only: run prog without hints/tickets")
    ap.add_argument("--out", default=str(ROOT / "results/arad"))
    return ap


def main() -> None:
    a = build_parser().parse_args()
    if a.smoke:
        smoke()
        return
    store = VectorStore.load(Path(a.store_dir))
    if a.embedder == "ollama":
        emb = OllamaEmbedder()
        if not store.embedder_name.startswith("ollama"):
            raise SystemExit("store/query embedder mismatch")
    else:
        emb = HashEmbedder()
    if a.backend == "mock":
        be_d, be_f = MockBackend(), MockToolBackend()
    else:
        be_d = Ollama(model=a.model, num_ctx=a.num_ctx, num_predict=400,
                      log_path=Path(a.out) / "diag_llm.jsonl")
        be_f = Ollama(model=a.model, num_ctx=a.num_ctx, num_predict=500,
                      log_path=Path(a.out) / "prog_llm.jsonl")
    pf_hist = PrecedentFutures(Path(a.store_dir) / "meta.jsonl")
    run = Runner(be_d, store, emb, k=a.k, stage_aware=a.stage)
    from ..prognostic.forecast import unit_history as _uh
    drun = DiagRunner(be_d, store, emb, k=a.k, stage_aware=a.stage,
                      hist=pf_hist, queries=None)
    print(f"[arad] retrieval: "
          f"{'life-stage-aware rerank' if a.stage else 'plain cosine'}")
    queries = load_queries(Path(a.queries))
    if a.limit:
        queries = queries[:a.limit]
    print(f"[arad] units={sorted(set(q['unit'] for q in queries))} "
          f"({len(queries)} anomalies)")
    drun.unit_hist_fn = lambda q: _uh(queries, q)
    drun.queries = queries
    if a.study in ("both", "diag"):
        study_diag(a, drun, queries)
    if a.study in ("both", "prog"):
        study_prog(a, run, queries, be_f)
    print("[arad] next: bench_forecast --evaluate/--report --out "
          f"{a.out}/prog ; eval_patterns.py --dir {a.out}/diag")


if __name__ == "__main__":
    main()

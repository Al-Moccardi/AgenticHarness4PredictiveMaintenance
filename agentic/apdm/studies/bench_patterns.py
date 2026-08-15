"""bench_patterns — the agentic-pattern grid, generation + local-RAGAS phases.

  # generation (resumable; ~7 patterns x 3 styles x N queries)
  python -m apdm.bench_patterns --queries path/to/test_FD002_with_interpretations.csv \
      --backend ollama --model llama3.2:3b --limit 20
  # evaluation (local judge + nomic embeddings; resumable)
  python -m apdm.bench_patterns --evaluate --judge-model llama3.2:3b
  # everything, small offline plumbing test (mock LLM + dry judge + hash store)
  python -m apdm.bench_patterns --smoke

Outputs under --out (default results/patterns/):
  episodes.jsonl   one line per (query, pattern, style): ticket, contexts,
                   trace, gates, tokens, latency
  metrics.jsonl    RAGAS-style + deterministic grounding per episode
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..llm import Ollama
from ..patterns import (MockBackend, PATTERNS, STYLES, Runner, case_block,
                       gates, load_queries)
from .ragas_local import DryJudge, evaluate_episode
from ..vector_store import HashEmbedder, OllamaEmbedder, VectorStore

ROOT = Path(__file__).resolve().parent.parent


def _key(qid, pattern, style):
    return f"{qid}|{pattern}|{style}"


def phase_generate(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ep_path = out / "episodes.jsonl"
    done = set()
    if ep_path.exists():
        for line in ep_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add(_key(r["qid"], r["pattern"], r["style"]))

    store = VectorStore.load(Path(a.store_dir))
    if a.embedder == "ollama":
        emb = OllamaEmbedder()
        if not store.embedder_name.startswith("ollama"):
            raise SystemExit(f"store was built with {store.embedder_name}; "
                             f"query embedder must match")
    else:
        emb = HashEmbedder()
    be = (MockBackend(log_path=out / "llm_calls.jsonl") if a.backend == "mock"
          else Ollama(model=a.model, num_ctx=a.num_ctx,
                      num_predict=a.num_predict,
                      log_path=out / "llm_calls.jsonl"))
    run = Runner(be, store, emb, k=a.k)

    queries = load_queries(Path(a.queries))
    if a.limit:
        queries = queries[:a.limit]
    styles = a.styles
    grid = [(q, p, s) for q in queries for p in a.patterns
            for s in (["plain"] if p == "B0_retrieval" else styles)]
    todo = [(q, p, s) for (q, p, s) in grid
            if _key(q["qid"], p, s) not in done]
    print(f"[patterns] units={sorted(set(q['unit'] for q in queries))}")
    print(f"[patterns] {len(queries)} queries x grid -> {len(grid)} episodes "
          f"({len(done)} done, {len(todo)} to run) | store n={len(store.meta)} "
          f"({store.embedder_name})")

    t_all = time.time()
    with ep_path.open("a") as f:
        for i, (q, p, s) in enumerate(todo):
            tok0 = be.totals()
            t0 = time.time()
            try:
                r = run.run(p, q, s)
                err = None
            except Exception as e:  # noqa: BLE001
                r = {"ticket": None, "contexts": []}
                err = f"{type(e).__name__}: {e}"
            tok1 = be.totals()
            shown = [f"u{c['unit']}c{c['cycle']}" for c in r.get("contexts", [])]
            rec = {"qid": q["qid"], "pattern": p, "style": s,
                   "true_rul": q.get("true_rul"),
                   "case_text": case_block(q),
                   "ticket": r.get("ticket"),
                   "contexts": [{k2: c.get(k2) for k2 in
                                 ("unit", "cycle", "rul_then", "gravity",
                                  "similarity", "text")}
                                for c in r.get("contexts", [])],
                   "gate_violations": gates(r.get("ticket"), shown),
                   "steps": r.get("steps"),
                   "repairs": r.get("repairs"),
                   "escalated": r.get("escalated"),
                   "wall_s": round(time.time() - t0, 3),
                   "tokens_in": tok1["prompt_tokens"] - tok0["prompt_tokens"],
                   "tokens_out": (tok1["completion_tokens"]
                                  - tok0["completion_tokens"]),
                   "llm_calls": tok1["n_calls"] - tok0["n_calls"],
                   "error": err}
            f.write(json.dumps(rec, default=str) + "\n")
            f.flush()
            el = time.time() - t_all
            print(f"  [{i+1}/{len(todo)}] {rec['qid']:>10s} {p:<15s} {s:<7s} "
                  f"{rec['wall_s']:6.1f}s  calls={rec['llm_calls']} "
                  f"({el/60:.1f} min elapsed)", flush=True)
    print(f"[patterns] episodes -> {ep_path}")


def phase_evaluate(a) -> None:
    out = Path(a.out)
    ep_path = out / "episodes.jsonl"
    mt_path = out / "metrics.jsonl"
    if not ep_path.exists():
        raise SystemExit("no episodes.jsonl; run generation first")
    eps = [json.loads(l) for l in ep_path.read_text().splitlines()
           if l.strip()]
    done = set()
    if mt_path.exists():
        for line in mt_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add(_key(r["qid"], r["pattern"], r["style"]))
    judge = (DryJudge() if a.judge == "dry"
             else Ollama(model=a.judge_model, num_ctx=a.num_ctx,
                         num_predict=300,
                         log_path=out / "judge_calls.jsonl"))
    emb = HashEmbedder() if a.embedder == "hash" else OllamaEmbedder()
    todo = [e for e in eps if _key(e["qid"], e["pattern"], e["style"])
            not in done]
    print(f"[ragas] {len(eps)} episodes ({len(done)} scored, "
          f"{len(todo)} to score) judge={judge.model}")
    with mt_path.open("a") as f:
        for i, e in enumerate(todo):
            t0 = time.time()
            try:
                m = evaluate_episode(e, judge, emb)
                err = None
            except Exception as ex:  # noqa: BLE001
                m, err = {}, f"{type(ex).__name__}: {ex}"
            f.write(json.dumps({"qid": e["qid"], "pattern": e["pattern"],
                                "style": e["style"], **m,
                                "judge_wall_s": round(time.time() - t0, 2),
                                "error": err}, default=str) + "\n")
            f.flush()
            if (i + 1) % 10 == 0 or i == len(todo) - 1:
                print(f"  scored {i+1}/{len(todo)}", flush=True)
    print(f"[ragas] metrics -> {mt_path}")


def smoke() -> None:
    """Offline end-to-end: hash mini-store + mock LLM + dry judge."""
    import shutil
    tmp = ROOT / "results" / "patterns_smoke"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    # mini store from the shipped meta (texts only; hash embeddings)
    meta = [json.loads(l) for l in
            (ROOT / "data/vector_store/meta.jsonl").read_text().splitlines()
            if l.strip()][:400]
    emb = HashEmbedder()
    vs = VectorStore(emb.embed([m["text"] for m in meta]), meta, emb.name)
    sdir = tmp / "store"
    vs.save(sdir)
    qcsv = None
    for cand in (ROOT / "queries/test_FD002_with_interpretations.csv",
                 ROOT / "results/test_FD002_with_interpretations.csv"):
        if cand.exists():
            qcsv = cand
            break
    if qcsv is None:
        raise SystemExit("smoke needs the edge merged CSV under queries/ "
                         "or results/")
    import sys
    argv = ["--queries", str(qcsv), "--store-dir", str(sdir),
            "--embedder", "hash", "--backend", "mock", "--limit", "2",
            "--out", str(tmp)]
    a = build_parser().parse_args(argv)
    phase_generate(a)
    a = build_parser().parse_args(argv + ["--evaluate", "--judge", "dry"])
    phase_evaluate(a)
    eps = [json.loads(l) for l in (tmp / "episodes.jsonl").read_text()
           .splitlines() if l.strip()]
    mts = [json.loads(l) for l in (tmp / "metrics.jsonl").read_text()
           .splitlines() if l.strip()]
    n_grid = 2 * (1 + 6 * 3)
    ok_n = len(eps) == n_grid and len(mts) == n_grid
    ok_t = all(e["ticket"] for e in eps if e["pattern"] != "P1_direct"
               or True)
    ok_m = all(("faithfulness" in m) for m in mts)
    print(f"\nSMOKE: episodes {len(eps)}/{n_grid} metrics {len(mts)}/{n_grid} "
          f"tickets_ok={ok_t} metrics_ok={ok_m}")
    print("SMOKE " + ("PASSED" if (ok_n and ok_t and ok_m) else "FAILED"))


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries",
                    default=str(ROOT / "queries/"
                                       "test_FD002_with_interpretations.csv"))
    ap.add_argument("--store-dir", default=str(ROOT / "data/vector_store"))
    ap.add_argument("--embedder", default="ollama",
                    choices=["ollama", "hash"])
    ap.add_argument("--backend", default="ollama",
                    choices=["ollama", "mock"])
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--judge", default="ollama", choices=["ollama", "dry"])
    ap.add_argument("--judge-model", default="llama3.2:3b")
    ap.add_argument("--patterns", nargs="+", default=PATTERNS,
                    choices=PATTERNS)
    ap.add_argument("--styles", nargs="+", default=STYLES, choices=STYLES)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--num-predict", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results/patterns_v2"))
    return ap


def main() -> None:
    a = build_parser().parse_args()
    if a.smoke:
        smoke()
    elif a.evaluate:
        phase_evaluate(a)
    else:
        phase_generate(a)


if __name__ == "__main__":
    main()

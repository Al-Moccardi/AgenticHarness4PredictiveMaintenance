# Agentic tier — the fleet coordinator

The laptop side of A-RAD: retrieval-augmented **diagnosis** and **prognosis**
over the fleet's memory, fully local (Ollama; no OpenAI, no auxiliary ML
models loaded — the CNN-GRU is the single deliberate exception, as a hint).

## Components

* **Knowledge base** — `data/vector_store/`: 4,458 interpreted anomalies of
  the TRAIN fleet, embedded with `nomic-embed-text` (768-d, cosine). Frozen,
  read-only; a mismatch between store and query embedder aborts the run.
* **Queries** — `queries/test_FD002_with_interpretations.csv`: the 89 REAL
  interpretations the Jetson wrote for official-test anomalies (8 stratified
  units). Query units are namespaced (`T65`) and never enter the store.
* **`apdm/patterns.py`** — the diagnostic agents. B0 no-LLM floor; P1
  direct; P2 RAG; P3 ReAct (search_memory / read_precedent / finish);
  P4 reflexion; P5 verifier-gated with deterministic gates + repair loop +
  escalation; P6 analyst→diagnostician→reviewer. Styles: plain / CoT /
  few-shot. All emit the same ticket JSON (diagnosis, action, RUL, range,
  cited precedents).
* **`apdm/ragas_local.py`** — RAGAS-definition metrics (faithfulness, answer
  relevancy, context precision) with a LOCAL judge, plus deterministic
  grounding: citation validity, RUL-in-cited-span, action-band consistency.
* **`apdm/forecast.py` + `apdm/bench_forecast.py`** — the PROGNOSTIC agent
  P7: forecasts the SPECIFIC unit from its own anomaly progression, from
  precedents shown WITH THEIR KNOWN FUTURES, with diagnosis-first chaining,
  and (arm `P7_agent_dl`) a CNN-GRU hint it must arbitrate. Scored against
  the realised test future (sensor-trend directions, anomaly outlook —
  evaluator-only) and against `RUL_FD002.txt` gold (MAE, bias, CMAPSS
  S-score, range coverage). Arms: b0_median, dl_only, P7_agent, P7_agent_dl.
* **`apdm/dl_rul.py`** — faithful port of the CPdM CNN-GRU
  (MinMax, window 20, Conv1D(64,3,same)→GRU(100)→Dense(50)→Dense(1),
  Adam 1e-3, seed 42) on the 14 edge sensors.

## Conventions enforced in code

RUL capped at **125** everywhere; per-unit **monotone** forecasts (the agent
sees its own previous prognosis and the bound it implies; hints and the
evaluator apply the same clamp; violations are counted, raw values kept).
Leakage guards: train-only store, namespaced test units, futures only in
evaluators. Everything resumable (`*.jsonl`), seed 42, temperature 0.

## Documentation

`RUNBOOK.md` — the exact command sequence. `README_PATTERNS.md` — both
studies in depth, including how to read the results honestly (local-judge
caveat, the B0 floor logic, when to distrust a lenient judge, what to say if
dl_only beats the agent). `docs/README_v8_legacy.md` — the original v8
package this tier extends (its OpenAI-based benches remain available but are
not part of the local-only study).

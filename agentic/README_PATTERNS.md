# Agentic pattern grid over your fleet vector store — fully local

Overlay for the ARAD-agentic-v8 tree. **No ML models are loaded. No OpenAI is
called.** The coordinator LLM, the RAGAS judge, and the query embedder are all
local (Ollama); the knowledge base is your existing `data/vector_store`
(nomic-embed-text, n=4,458, train units only) — it is loaded read-only, never
rebuilt, never written.

## Install (from your ARAD-agentic-v8 root)

    unzip -o arad-patterns-overlay.zip     # adds 4 files to apdm/ + queries/
    ollama pull llama3.2:3b                # coordinator + judge
    ollama pull nomic-embed-text           # query embedder (must match store)
    python -m apdm.bench_patterns --smoke  # offline plumbing check (~30 s)

## What runs

7 coordination patterns x 3 prompting styles on the SAME cases, every case
grounded in the same frozen store:

    B0_retrieval  no LLM: neighbour median ticket        (the floor)
    P1_direct     no retrieval, case only                (prices the KB)
    P2_rag        case + top-k precedents, single shot
    P3_react      tool loop: search_memory / read_precedent / finish
    P4_reflexion  draft -> self-critique -> revision
    P5_verifier   draft -> deterministic gates -> repair (<=2) -> escalate
    P6_specialists analyst -> diagnostician -> reviewer

    styles: plain | cot (mandatory reasoning-first) | fewshot (one example)

Queries = the 89 REAL interpretations your Jetson wrote for official-test
anomalies (`queries/test_FD002_with_interpretations.csv`): the coordinator is
answering exactly what the edge produced. Query units are namespaced (T65)
and are never inserted into the store.

## Run (laptop, all local)

    # pilot: 8 queries, everything            (~20-40 min with llama3.2:3b)
    python -m apdm.bench_patterns --limit 8
    # full grid: 89 queries x (1 + 6x3) = 1,691 episodes  (resumable)
    python -m apdm.bench_patterns
    # RAGAS phase (local judge; resumable)
    python -m apdm.bench_patterns --evaluate
    # tables + camera-ready figures
    python -m apdm.report_patterns

Useful knobs: `--patterns P2_rag P5_verifier` `--styles plain cot`
`--k 6` `--model llama3.1:8b` (if you have it; better judge too:
`--evaluate --judge-model llama3.1:8b`).

## What you get (results/patterns/)

    episodes.jsonl   per (query, pattern, style): ticket JSON, retrieved
                     precedents, ReAct trace, gate violations, repairs,
                     tokens, latency
    metrics.jsonl    per episode: faithfulness, answer_relevancy,
                     context_precision (RAGAS definitions, LOCAL judge) +
                     deterministic grounding: json_valid, citation_validity,
                     rul_in_cited_span, action_band_consistency
    summary.csv/.md  the pattern x style table
    fig10_pattern_grid.(pdf|png)   faithfulness heatmap, RAGAS profile,
                                   deterministic grounding, quality-vs-cost
    fig11_gates_styles.(pdf|png)   verifier outcomes, prompting-style effect

## How to read it honestly (for the paper)

* These are **RAGAS-definition metrics with a local judge** (state the judge
  model). RAD's OpenAI-judged absolutes are not directly comparable; the
  *ranking* claims (CoT vs plain, RAG vs direct) are.
* The deterministic block needs no judge and is the hard evidence:
  `citation_validity` (cited ids were actually shown), `rul_in_cited_span`
  (the estimate lies inside its own precedents' outcome span), and
  `action_band_consistency` — the coordinator-side answer to the edge
  finding that 92% of SLM recommendations collapse to "shutdown".
* Expected shape of the result: P2 ≥ P1 prices the knowledge base;
  P5 lifts the deterministic block via repairs (report repairs + escalations,
  they are the safety story); P3/P6 pay 3-4x the calls — the quality-vs-cost
  panel shows whether they earn it. If P1 without retrieval matches P2 on
  faithfulness, the judge is too lenient: switch to `--judge-model
  llama3.1:8b` before drawing conclusions.
* B0 has one style row ("plain") by design: no prompt, no LLM.

Smoke mode (`--smoke`) uses a hash mini-store + a mock LLM + a dry judge:
plumbing only, numbers meaningless, never report them.

---

# Part 2 — the PROGNOSTIC agent (P7) + CNN-GRU

Three more modules: `forecast.py` (the agent + the deterministic future
evaluator), `dl_rul.py` (your CNN-GRU, ported faithfully: MinMaxScaler,
window 20, Conv1D(64,k3,same)->GRU(100)->Dense(50)->Dense(1), Adam 1e-3,
seed 42, on the 14 edge sensors), `bench_forecast.py` (the study).

The agent sees, all causal: the case, THIS unit's own anomaly progression,
the retrieved precedents WITH THEIR KNOWN FUTURES (train engines — what
happened to them afterwards, down to their failure), optionally the
diagnostic ticket first (diagnosis→prognosis chaining, on by default), and
— in the `P7_agent_dl` arm — the CNN-GRU RUL hint. It must emit a structured
forecast: progression narrative, 3-5 expected sensor trends (up/down/stable),
anomaly outlook, RUL estimate + range, cited precedents.

Scoring is deterministic and leak-free: the test unit's REAL future (raw
sensors after the query cycle, and the detector's future alarms) is used
ONLY in the scorer, never in a prompt. RUL truth is anchored on your
RUL_FD002.txt (gold + last_cycle − c). Metrics: RUL MAE/bias, your
pipeline's CMAPSS S-score, range coverage, sensor-trend direction accuracy
against realised slopes (dead-band 0.3σ → "stable"), and anomaly-outlook
accuracy against the realised alarm-rate ratio. Arms: `b0_median` (no LLM),
`dl_only` (CNN-GRU alone), `P7_agent`, `P7_agent_dl`.

## Run (laptop, after the pattern grid)

    # 1. train YOUR CNN-GRU on the run-to-failure fleet (TensorFlow)
    python -m apdm.dl_rul --train --epochs 100
    # 2. one RUL hint per edge anomaly (decouples TF from the bench)
    python -m apdm.dl_rul --hints
    # 3. the study: 89 queries x 4 arms (resumable), then scoring + figure
    python -m apdm.bench_forecast --limit 8      # pilot
    python -m apdm.bench_forecast
    python -m apdm.bench_forecast --evaluate
    python -m apdm.bench_forecast --report       # forecast_summary.csv + fig12

Offline plumbing check: `python -m apdm.bench_forecast --smoke`.

Reading guide: (a) MAE + S-score per arm — note the S-score is your
pipeline's asymmetric exponential and is dominated by early-life
underestimates, report median too if reviewers ask; (b) calibration scatter;
(c) the natural-language forecast panel — trend-direction and outlook
accuracy are THE novel numbers (nobody scores NL degradation forecasts
against realised futures); (d) the value-of-agency test: |agent err| −
|CNN-GRU err| per case — negative mass means the agent improves on its own
DL hint by arbitrating it against precedent futures. If dl_only beats
P7_agent_dl, say so honestly: the agent's job is then interpretation and
action, not number-beating.

### RUL convention (cap 125, monotone) — applied everywhere

Training target, DL hints, agent estimates, and the evaluation truth are all
on the piecewise scale `min(RUL, 125)`. Physical monotonicity is enforced
per unit three ways: (1) the agent is chained — it SEES its own previous
prognosis for that unit and the bound it implies; (2) DL hints are clamped
per unit (`mono(t) = min(pred(t), mono(prev) − elapsed)`); (3) the evaluator
applies the same clamp to every arm and reports `mono_violation_rate` — how
often the raw forecast tried to rise. Raw values are kept in
`rul_pred_raw` so the adjustment is auditable.

---

# v2 — performance modifications (driven by the measured v1 failures)

1. **ReAct rebuilt (P3 v2).** The loop is *seeded* with an initial retrieval
   (the agent refines evidence instead of bootstrapping tool use from
   nothing — the measured 3B collapse: 29% of v1 episodes ended with zero
   contexts, 71% of citations fabricated). Tolerant action parsing recovers
   `finish` from raw text; unparseable steps degrade to a search instead of
   aborting; 3 steps + a forced cite-only-the-above final ticket.
2. **Span anchoring everywhere.** Every precedent block now states the
   OUTCOME SPAN (min–max rul_then) and the system prompt makes it a hard
   rule: the estimate must lie inside the span of the cited precedents
   (v1 measured rul_in_cited_span at only 0.18–0.34).
3. **P5 gains gate G7** (estimate outside cited span → violation) and the
   repair message now quotes the allowed numeric interval explicitly.
4. **Revision drift stopped.** P4's reviser and P6's reviewer may no longer
   change the action unless it mismatches the band of the estimate
   (v1: reflexion/specialists dropped band consistency to 0.29–0.33).
5. **Band table in the system prompt** for every pattern.

**Scope**: queries are restricted to the SAME eight stratified units the
edge tier processed (65, 103, 110, 131, 135, 209, 222, 245) — enforced in
`load_queries` and printed at bench start.

**v2 runs write to `results/patterns_v2/`** (new prompts must not resume
into v1 files; your v1 results stay untouched in `results/patterns/` for
the before/after comparison — that comparison IS a paper table).

**Evaluation is now judge-independent**: `evaluation/scripts/eval_patterns.py`
recomputes the deterministic block (citation validity, RUL-in-span,
action-band) from episodes.jsonl for EVERY episode, merges RAGAS only where
judged, writes `gate_composition.csv` and `judge_coverage.txt`.

# agentic-pdm — Do Agentic Patterns Help? LLM Agents vs Machine Learning for RUL Prognostics

A controlled, from-zero study on FD002 (N-CMAPSS lineage) asking one question:
**when an LLM is asked to do the actual prognostic task — estimate remaining
useful life — do agentic patterns (tools, retrieval, iteration, ML
orchestration) close the gap to trained ML models, and where exactly does the
language layer add value?**

Anchoring paper — **Moccardi et al. (ResPdM)** — supplies everything derived:
the protocol (FD002, window 20, RUL clip 125, 208/52 unit split, S-score,
their eq. 4), the k=7 regime refinement whose 7th cluster captures the
failure modality (used here as the degradation-STATE input signal and as a
train-fleet residual-life prior), and the idea the P3 arm operationalises:
their diagnostic-based RUL *adjustment* (eq. 3) hard-codes a conservatism
factor δ; P3 replaces the hand-tuned δ with an agent that sees the same
evidence and adjusts contextually. **Isolation-Forest-era columns in the CSV
(anomaly_label/score, cumulative counters, rule text, SLM interpretations)
are ignored by design** — that prior pipeline is out of scope; the healthy
per-regime reference is defined from ground truth instead (train rows with
true RUL > 100), and smoke test L3 asserts no detector-derived feature
survives.

## Research questions

* **RQ1 (floor):** can an LLM estimate RUL at all from raw or featurized
  windows (P0/P1), relative to naive and reliability-style baselines?
* **RQ2 (pattern ladder):** which agentic ingredient moves the needle —
  featurization (P0→P1), tools+iteration (P1→P2)? The critical control is
  **kNN**: P2's retrieval tool *is* kNN, so P2 ≈ kNN means the language model
  added nothing over its own tool.
* **RQ3 (orchestration):** does the hybrid (P3 = agent supervising XGBoost)
  beat both pure ML and pure agent — specifically on **S-score near EoL**,
  where paper 1's hand-tuned adjustment earned its 17.5% S reduction? Report
  adjustment behaviour: how often the agent moves the ML estimate, in which
  direction, and whether moves correlate with anomaly/state evidence.
* **RQ4 (cost):** calls, tokens, seconds per decision, per arm — the edge
  budget an IoT deployment pays for each pattern.

## Arms (all elicit one integer RUL per snapshot; one strict retry; parse
failures are a reported metric, never imputed)

| arm | what it sees | non-language twin |
|---|---|---|
| P0_raw | raw 20×14 window in the prompt | — |
| P1_featurized | engineered summary (same quantities as ML features) | mlp/xgb inputs |
| P2_agentic | ReAct loop: sensor_summary, sensor_window, similar_cases, degradation_status | **knn** |
| P3_hybrid | P2 tools + ml_predict (XGB estimate + its known MAE) | **xgb** |
| ML: naive_mean, linear_deg, knn, mlp, histgb, xgb | 78 shared features | — |

Anchors on the full 9,470-snapshot test universe (IF-free build, already
run): naive R²≈0.00 · linear_deg 0.51 · knn 0.63 (MAE 17.6) ·
**xgb/histgb 0.83 · mlp 0.84** (MAE 12.0–12.4) — matching paper 1's BiLSTM
(0.83) under the same protocol, which validates the replication and shows the
dropped detector features were not load-bearing. (S-scores are sums; compare
only at matched n.)

## Fairness & leakage controls (tested in `apdm/smoke_test.py`)

Same features feed ML, retrieval and the LLM's summaries; tools are bound to
one (unit, cycle) and structurally cannot read that unit's future (proved by
construction-swap, L1) or retrieve test units (L2); causal counters re-derived
and cross-checked against the CSV (L3). Paired design throughout: every LLM
arm is compared to every ML arm on identical snapshots (Wilcoxon on |err|,
McNemar on induced stage), with parse-failure Wilson CIs.

## Run matrix

```bash
python -m apdm.smoke_test                      # 23 checks, ~1 min, no model
python -m apdm.run_ml --per-bucket 30 --sample-seeds 1 2 3      # ~6 min
# pilot (15 snapshots) to time your box:
python -m apdm.run_llm --backend ollama --model llama3 --sample-seed 1 --limit 15
# full arms, one model (90 snaps × 4 arms; P0/P1 ≈1–2 calls, P2/P3 ≈3–6):
python -m apdm.run_llm --backend ollama --model llama3 --sample-seed 1
python -m apdm.run_llm --backend ollama --model llama3.2:3b --sample-seed 1
python -m apdm.run_llm --backend openai --model gpt-4o-mini --sample-seed 1   # strong-coordinator reference
python -m apdm.report --sample-seed 1
# robustness: repeat with --sample-seed 2 3
```

Budget guide: ~90 snapshots ⇒ P0/P1 ≈ 90–180 calls each; P2/P3 ≈ 300–550
each. At 4–10 s/call on an 8B local model: 0.5–1.5 h per agentic arm.

## Stated hypotheses (pre-registered here, falsifiable by the tables)

H1 P0 ≪ ML (LLMs are weak numeric regressors) — a floor, not a finding.
H2 P1 > P0 but < xgb.  H3 P2 ≈ knn (agent ≈ its retrieval tool) — if P2 > knn
significantly, the LLM is genuinely fusing tools, which would be the surprise
result.  H4 P3 ≈ xgb on MAE but **P3 < xgb on S-score / overestimation in the
critical bucket** — contextual conservatism where paper 1's fixed δ helped.
Any outcome is publishable if the design holds; negative results on H3/H4 are
findings, not failures.

## Threats to validity (state these in the paper)

Sampling: LLM arms see 90 stratified snapshots, not the universe — hence
paired tests and multi-seed subsets, and ML reported on both scopes. Anomaly
labels/k7 state are pseudo-labels used **only as inputs**, never as targets.
S-score is n-dependent; compare at matched n only. Single dataset (FD002);
FD004 transfer is future work. `parse_fail_rate` is a capability result in
itself for small models — report it, don't hide it.

## Layout

```
apdm/data.py        FD002 loader, split, RUL labels, leakage-safe views
apdm/features.py    78 shared features + the P1 textual summary
apdm/ml_models.py   baselines + kNN twin + the P3 ml_predict source
apdm/metrics.py     S-score (paper-1 exact), suite, Wilcoxon/McNemar/Wilson
apdm/tools.py       snapshot-bound tool registry
apdm/agent.py       P0–P3, terminating ReAct loop, strict ANSWER parsing
apdm/run_ml.py      full-universe + canonical-subset ML evaluation
apdm/run_llm.py     LLM arms on the canonical subset (+ report entry)
apdm/smoke_test.py  leakage proofs, metric identities, termination guards
```


## Fault layer (FA-PdM extension) — built, with two decisive pilot results

`apdm/faults.py` derives fault PHENOTYPES from TRAIN units' terminal
signatures (mean z over the last 10 cycles vs each row's regime-healthy
reference; k by silhouette). On FD002: **k=2**, silhouette 0.63 — P0
'Nc-NRc' (core-speed-led, 71 units) and P1 'Ps30-BPR' (core-pressure-led,
137 units). Each phenotype carries a deterministic, data-grounded
interpretation (signature + CMAPSS-schematic physics + onset and
residual-life statistics), exported to `results/faults_layer.json` and
exposed to agents via the `fault_library` tool. Gold for any unit is
future-derived (its own terminal signature -> nearest train centroid);
F-series smoke tests prove inputs are past-only, gold is future-only, and
the layer is train-only.

**Q1 (fault-conditioned prognosis) — NEGATIVE on FD002.** Residual life
after degradation-state entry does not differ across phenotypes
(Mann-Whitney p=0.52; medians 45 vs 53), and phenotype-conditioned priors at
entry do not beat the pooled prior (MAE 35.7 vs 36.3; oracle diagnosis 36.3).
Both phenotypes are the single designed HPC fault on different channels —
same clock. FD002 therefore serves as the study's NEGATIVE CONTROL, and the
pre-registered prediction is that FD004 (two designed fault modes) flips Q1
to significant. Also measured: entry-event priors alone are weak clocks
(MAE ~36 vs XGB's ~12), i.e. event-based prognosis is dominated by
feature-based regression here.

**Q2 (diagnosability frontier) — POSITIVE.** Phenotype diagnosis accuracy by
true-RUL bucket (1500 test snapshots; majority class 0.675):

| bucket | n | rule (nearest-centroid, no learning) | logistic |
|---|---|---|---|
| early (RUL>100) | 686 | 0.649 | 0.688 |
| mid | 552 | 0.717 | 0.902 |
| critical (RUL<=30) | 262 | 0.916 | 0.950 |

The signature emerges with degradation: early diagnosis is barely above
chance for the no-learning rule and the logistic bar is beatable-looking but
non-trivial. Any agentic diagnosis arm must beat BOTH twins per bucket to
claim value; entry-time nearest-centroid accuracy is 0.827.


## Diagnostic / root-cause agents over the interpretation layer (v5)

The successor-paper module: agents built ON TOP of the TIOT interpretation
pipeline, with every output scorable and every arm twinned.

* `apdm/interpret_kb.py` -- the enriched database as retrieval memory:
  4,393 TRAIN anomaly precedents (rule text for all; SLM interpretation
  excerpts for the 65 records of train units 57/146/232). kNN in the same
  regime-referenced z-space as the fault layer (embedder-free). The 19
  interpreted records of test units 58/140 are HELD OUT as qualitative
  references. Guards K1-K3.
* `apdm/diagnosis.py` -- arms D1_featurized (single shot),
  D2_agentic (tools: sensor_summary, fault_library, similar_anomalies,
  degradation_status) and D2_norag (same minus the KB): the ablation that
  DIRECTLY TESTS the TIOT premise -- does retrieving past rules and
  interpretations measurably improve diagnosis? Strict protocol
  (PHENOTYPE / SENSORS / EXPLANATION; parse failures first-class);
  scoring = phenotype accuracy per RUL bucket PAIRED against the two Q2
  twins (McNemar), signature Jaccard / P@3 vs future-derived gold, and
  claim FAITHFULNESS vs current-window evidence (supported / contradicted /
  unverifiable at |z|>=1). Guards D1-D4.
* The prognostic agent role remains GATE, not regressor (locked by the
  P3-vs-xgb result); the deferral task is the next module.

Run:
  python -m apdm.diagnosis --backend ollama --model llama3 --sample-seed 1
  python -m apdm.diagnosis --backend ollama --model llama3 --arms D2_agentic D2_norag --sample-seed 2


## v5.1 — protocol fix + SLM agentic benchmark

**Why v5 D2 failed (llama3, 90 snapshots): a protocol defect, not a
finding.** D1 parsed 90/90 with the explicit three-line template; D2 parsed
4/90 and D2_norag 0/10. Two causes, both fixed:
1. the JSON `final` carried a three-line string with embedded newlines,
   which small models mangle under `format="json"` -> the final is now a
   STRUCTURED object `{"phenotype":..., "sensors":[["T50","high"],...],
   "explanation":...}` with a worked example in the system prompt;
2. every D2 trace reaches forced finalisation (plain text) at the last step,
   and the FORCE prompt said only "output the three labelled lines" while
   D1's TASK spelled out the template -> FORCE now REUSES the exact D1
   template verbatim.
`parse_diag` accepts a structured dict, the plain-text lines, or either
embedded in a JSON blob; direction synonyms (elevated/reduced/rising/...)
are normalised. Guards D1-D4 cover all paths.

**D1 result stands and is the first real diagnostic finding** (llama3, 90
snapshots): parse 90/90, phenotype accuracy 0.744 (twins: rule 0.721,
logistic 0.813 -> the agent sits between them), signature P@3 0.659, and
FAITHFULNESS support 0.585 -- i.e. ~40% of stated (sensor, direction) claims
are unverifiable or contradicted by current-window evidence while the label
is often right. Verification of the explanation is load-bearing, not
decoration.

**`apdm/bench_slm.py` -- successor to TIOT Fig. 7.** That figure ranked edge
SLMs by size and popularity; this one ranks them by AGENTIC CAPABILITY:
protocol compliance (with Wilson CI) as the gating property for edge
deployment, phenotype accuracy against both twins, signature P@3,
faithfulness, and cost per diagnosis -- plus the per-model KB delta
(D2_agentic - D2_norag), which is the direct test of the TIOT premise.
Runs each model as a subprocess and skips models whose predictions already
exist, so an interrupted sweep resumes.

  python -m apdm.bench_slm --models llama3.2:1b llama3.2:3b qwen2.5:3b phi3 mistral llama3 --sample-seed 1


## v6 — event forecasting, outcome severity, gravity audit, Jetson cost model

**Thesis frame (final):** same IoT substrate, same interpretation layer;
treatment = agentic pattern x SLM; tasks = RCA, event forecasting
(onset interval + severity), suggestions; all costed on simulated Jetson
Orin Nano hardware.

**Event target (`apdm/events.py`, `run_events.py`).** "Time to next anomaly"
is degenerate here (inter-anomaly gap median 1 cycle, q90=5: bursts), so the
forecastable event is the ONSET: for a QUIET unit (no anomaly in the last 5
cycles), cycles until anomalous behaviour (re)starts. Onset gaps: median 61,
q10 8, q90 138; censoring 0. Gold is future-derived (guard E1b: deleting the
future removes it). Severity gold is OUTCOME-derived (RUL at the event ->
bands 1..5), never the interpretations' gravity scores.

**Twin bars (1,200 internal-test events).** Intervals at 80% target:
T1 climatology coverage 0.819 / width 130 / Winkler 165.2; T2 learned
quantile-HistGB coverage 0.672 (UNDER-covers) / width 62 / Winkler 123.7 --
the calibrated-AND-narrow frontier is open. Severity: majority acc 0.336 /
QWK 0; learned acc 0.405 / QWK 0.296 / ±1 0.83 -- severity-at-onset is
genuinely hard from current features.

**GRAVITY AUDIT (`gravity_audit.json`).** The TIOT interpretations' own
gravity scores (extractable from 30/84 records; values include 0) correlate
with real outcomes at Spearman rho = 0.039 (p = 0.84): NO relationship.
19/30 audited events were outcome-severity 5; the SLM most often said 3.
Caveats: n=30, 5 units, bands are ours. Consequence: interpretations are an
audit target and retrieval memory -- structurally unusable as gold.

**Jetson layer (`apdm/hardware.py`).** Analytical Orin-Nano cost model
(prefill compute-bound, decode bandwidth-bound, KV vs unified memory) wired
into every backend call via REAL Ollama token counts (prompt_eval_count /
eval_count; chars/3.6 fallback), with prefix-cache estimation, per-item
sim_edge_s / sim_energy_j columns in diagnosis outputs, provenance files,
and a calibrate() hook for measured device triples. All sweep models fit
8 GB at 8k ctx; llama3 5.94/6.40 GB (max ctx 11,686); phi3's full-MHA KV is
3.2 GB at 8k -- context is a hardware decision. Guards H1-H5. sim_ values
are MODEL ESTIMATES until calibrated on the device.

**Next build:** pattern arms P1 RAG / P3 reflection / P4 verifier-gated /
P5 plan-execute / P6 multi-agent for the three tasks, on this substrate.


## v7 — the four-step pipeline complete (generator, vector store, P1_RAG)

Step 1 `apdm/gen_interpretations.py`: TIOT-style interpretation for EVERY
train anomaly (4,393), grounded in the shared z-deviations + IF rule,
few-shot from the real train interpretations, STRUCTURED output
(interpretation / gravity 1-5 / components) so the gravity audit becomes
total instead of 30/84. Resumable JSONL per unit; provenance
source="generated"; generated text is retrieval memory + audit target,
never gold. Step 2 `apdm/vector_store.py`: semantic KB over all
interpretation texts (real + generated), Ollama nomic-embed-text on-device
(hash n-gram fallback strictly for offline pipeline tests, loudly labelled).
Query = the current grounded sensor summary. z-space kNN (interpret_kb)
remains the retrieval twin the semantic store must beat to justify the
generation cost. Step 3: pattern arm D1_rag (P1) wired into diagnosis
alongside P0/P2; hardware simulation active on every call (v6). Step 4:
evaluation stack unchanged -- outcome gold, twins, faithfulness, paired
stats; guards G1/G1b + V1-V3 added (suite: L/F/O/K/D/H/E/G/V, all passing).

Order of operations on the experiment machine:
  ollama pull nomic-embed-text
  python -m apdm.gen_interpretations --backend ollama --model llama3.2:3b --units train
  python -m apdm.vector_store --build --embedder ollama
  python -m apdm.bench_slm --models ... --arms D1_featurized D1_rag D2_agentic D2_norag


## v7.2 — explicit RUL export (paper protocol)

`data/Dataset_with_interpretations_RUL.csv`: the train CSV with explicit
columns `RUL` = clip(EoL_u - cycle, 125) -- the piecewise-linear target of
the prior papers and the gold used by this pipeline since v1 (verified
column == ds.rul() row-wise) -- plus `RUL_raw` (uncapped) and
`split_seed42` (the frozen 208/52 membership). 39.5% of rows sit at the
cap. Note: the event layer's severity bands intentionally use RAW remaining
life at the event (capping there would merge bands 1-2); the official test
file remains the one place where complement-of-cycles is NOT the RUL.


## v8 — the retrieval-augmented prognostic DECISION layer (final experiment line)

The pipeline exactly as specified: IF anomalies + rules → context-aware
natural-language interpretations → frozen 208/52 unit split (seed 42) →
vector store over the TRAIN units' interpretations (nomic, n=4,458, shipped
in `data/vector_store/`) → for each TEST-unit anomaly, its OWN
interpretation is the query; the k most similar train interpretations are
retrieved together with their raw-signal context and their known OUTCOMES
(rul_then), and agents produce a maintenance TICKET:
`{rul_estimate, rul_range (central 80%), action 1-5, cited_precedents,
rationale}` — RUL estimation *and* action planning in one elicitation.

Because the layer is ML-free, the estimate is retrieval-grounded case-based
reasoning, so its honest reference is the SAME retrieval with the language
model off (B0), plus the embedder-free z-space kNN twin (B1) that prices the
semantic representation itself.

**Arms — a ladder of increasing agentic structure** (`apdm/prognosis.py`):
B0_retrieval, B1_zknn (no LLM) · P1_direct · P2_rag · P3_react ·
P4_reflexion · P5_verifier (the RAD guardrails promoted to runtime
acceptance gates: support / range / action / citation / claim-contradiction,
one violation-driven repair, unrepairable → ESCALATED to the human board) ·
P6_specialists (retrieval analyst → prognostics estimator → maintenance
planner).

**Scoring** (internal test split, 986 anomaly cases, stratified samples for
the LLM arms): MAE/bias/per-case S on the estimate; coverage, width and
Winkler (α=.2) on the range; exact and ±1 accuracy on the OUTCOME-derived
action band (v6 severity bands — never the interpretations' own gravity
opinions, ρ=0.039 in the audit); rationale faithfulness vs current-window
z-evidence; parse failures first-class; per-decision cost incl. simulated
Jetson seconds/energy. Every LLM arm is PAIRED case-by-case against B0
(Wilcoxon on |err|); `report_prognosis` writes tables, three figures, and a
HEADLINES_seed*.md whose claims are gated on p<.05.

**Order of operations on the experiment machine:**
```bash
pip install -r requirements.txt
python -m apdm.smoke_prognosis                 # 22 checks, offline, ~1 min
python -m apdm.smoke_test                      # the original suite stays green
# step 0 — the ~967 missing TEST interpretations (queries), same generator:
python -m apdm.gen_interpretations --backend ollama --model llama3.2:3b --units test
# the sweep (row-level resume inside; a finished model is skipped):
python -m apdm.bench_prognosis --models llama3.2:1b llama3.2:3b qwen2.5:3b phi3 mistral llama3 --sample-seed 1
python -m apdm.report_prognosis --sample-seed 1
# robustness seeds / full universe when time allows:
python -m apdm.bench_prognosis --sample-seed 2   # and 3
python -m apdm.prognosis --backend ollama --model <best> --full --sample-seed 1
# optional ablations:
python -m apdm.prognosis --backend ollama --model <best> --query summary --sample-seed 1  # query-representation
python -m apdm.prognosis --backend openai --model gpt-4o-mini --sample-seed 1             # strong-coordinator reference
```
Budget: 90 cases × 8 arms ≈ 100–140 calls for the single-shot arms and
250–450 for P3/P6, per model. The store and cache ship in this archive; the
vector store is TRAIN-only by construction (guard PB1) and must NOT be
rebuilt after test-interpretation generation unless you keep
`--embedder ollama` (the builder filters to train units either way).

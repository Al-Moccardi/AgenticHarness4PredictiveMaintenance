# A-RAD Evaluation Suite — code, results, and how to read them

Everything in this archive was computed on **your real data**: the 260-unit
run-to-failure train fleet (`anomalies_multimodal.csv`, 53,759 rows) for the
detector, and the **89 real Llama-3.2-3B-Instruct-Q4_K_M interpretations**
produced on your Jetson (llama.cpp, temperature 0) for the SLM. Nothing below
is simulated or illustrative; every number cited here appears verbatim in a
CSV under `results/`.

```
scripts/                       run in this order
  eval_iforest.py              detector effectiveness (RUL-anchored)
  eval_iforest_extra.py        deeper IF statistics
  eval_slm.py                  SLM grounding & faithfulness
  eval_slm_extra.py            SLM behaviour (actions, diversity, history)
  eval_hw.py                   hardware statistics
  plot_paper.py                figures 1-6
  plot_extra.py                figures 7-9
results/
  if_eval/  if_eval_extra/  slm_eval/  slm_eval_extra/  hw_eval/
  figures/paper/               fig1..fig9 (PDF vector + 300-dpi PNG)
docs/EVALUATION.md             methodology details
```

Reproduce end-to-end (needs the data CSVs alongside, as on your machine):

```bash
python3 scripts/eval_iforest.py && python3 scripts/eval_iforest_extra.py
python3 scripts/eval_slm.py     && python3 scripts/eval_slm_extra.py
python3 scripts/eval_hw.py
python3 scripts/plot_paper.py   && python3 scripts/plot_extra.py
```

---

## 1. How to read the detector results (figs 1–4, 7)

**The evaluation problem.** The Isolation Forest is unsupervised; CMAPSS has no
anomaly labels, so "accuracy" is undefined. Every metric is therefore anchored
on **RUL, which the detector never observes**: weak labels (degraded = RUL≤30
vs healthy = RUL≥100, mid-life excluded), swept over the threshold so no
conclusion depends on one choice, with bootstrap CIs, against two baselines
computed on **identical rows**.

**The claim the numbers support** — regime-specific detection works and the
clustering is the reason:

| detector | AUC | PR-AUC |
|---|---|---|
| **regime-specific IF (yours)** | **0.904** [0.899, 0.908] | **0.810** |
| global IF, same features, no clustering | 0.809 | 0.611 |
| Hotelling T² (regime-blind control chart) | 0.772 | 0.562 |

ΔAUC vs the global IF is **+0.095** [0.091, 0.099], bootstrap p < 0.001. This
is the quantitative version of the TIOT-LLM Fig. 3 argument (the T² chart that
cannot tell an operating-mode change from a fault): same rows, same features,
the only difference is the regime conditioning, and it is worth ~0.10 AUC.
Say exactly that.

**Operational meaning.** At the deployed 10 % contamination: **1.7 % false
alarms** in the healthy zone, 51 % alarm rate in the degraded zone, median
**16-cycle warning** before failure (degradation phase). Figure 7c shows this
is one point on a dial — 2 %→30 % alarm budget buys 5→52 cycles of lead at
0.2 %→20 % false alarms — so present 16 cycles as *the deployed operating
point*, not a property of the method.

**The temporal signature (fig 3) is your strongest single result.** Per-unit
Spearman ρ(cumulative anomalies, RUL): median **−0.62**, and **260 of 260
units** have the expected sign. Normalised C_i rises 0.045 → 0.72 and the
alarm rate 3.3 % → 84.7 % as RUL falls from ≥150 to <10; the frequency metric
F_i confirms anomalies *accelerate* (ρ = −0.34), matching the physics of
accelerating wear. **Report the per-unit ρ, never the pooled −0.29**: raw
counts are not comparable across units of different lifetimes, so pooling
mixes populations and dilutes a unanimous effect.

**Robustness box-ticks** (fig 7): labels are 97.6 % identical across five
seeds (AUC 0.900–0.929); the best single sensor (Ps30, 0.848) trails the
multivariate IF by +0.056, so the joint rule adds real value; per-regime AUC
spans only 0.885–0.916, so no operating mode is a blind spot.

### The three traps — read before writing the section

1. **Per-regime alarm rate is flat at 10 % BY CONSTRUCTION** (contamination is
   enforced inside each cluster). It is not evidence of anything. Use the
   healthy-zone false-alarm spread (0.87–3.35 pp) and per-regime AUC, which
   are free to vary.
2. **The alarm profile is a bathtub, not a ramp** (fig 1b): 3.8 % at break-in,
   0.24 % mid-life, 68 % in the final decile. Break-in alarms are real but
   inflate the naive lead time (19 vs 16 cycles); cite the degradation-phase
   number and mention the early bump honestly — it is expected physics, and a
   reviewer who spots an unexplained ramp-only story will distrust the rest.
3. **Distance-to-own-centroid alone scores AUC 0.929 — above the IF's 0.904.**
   Do not hide this; a reviewer can compute it in five lines. Frame it as it
   actually is: *regime-conditioning is the driver* (both regime-conditional
   detectors dominate the blind baselines at 0.77–0.81), and the IF's specific
   contribution is the **interpretable split rules the language layer
   requires** — a distance scalar explains nothing and cannot feed the SLM.
   Stated first by you, this strengthens the paper; found by a reviewer, it
   sinks the detector section.

---

## 2. How to read the SLM results (figs 5, 6, 8)

**The evaluation trick.** No human annotation is needed: the rule is a set of
(sensor, operator, threshold) triples and the interpretation is text, so
grounding is checkable deterministically. The model also never sees RUL, so
correlating its gravity score against true RUL is a genuine out-of-band test.

**What the model does well.** Format compliance **100 %** (all six sections,
every time), echo contamination **0 %** (the run-1 defect is dead), direction
agreement **94.9 %** with only **4.3 %** contradictions, threshold anchoring
80 %, sensor precision 77.7 %. Latency **5.6 s median / 6.4 s p95** per
anomaly at 44 tok/s decode. Lexical diversity is healthy (mean pairwise
3-gram Jaccard 0.16, 0.7 % near-duplicates — fig 8c), and when a unit has
prior anomalies the Trend section references history **100 %** of the time.

**The two findings that make this a contribution rather than a benchmark:**

- **Compression, not hallucination.** Sensor recall is 61.4 % — the model
  summarises rather than enumerates — and recall degrades monotonically with
  rule complexity: 100 % → 78 % → 60 % → 41 % for rules spanning 2 → 4 → 6 → 8
  sensors (fig 5b). This reproduces, at the interpretation level, the
  complexity effect RAD measured for strict rule recovery, and it is the
  cleanest argument for keeping edge rules compact. Keep "recall < 100 %"
  and "hallucination" strictly separate: precision is the unsupported-content
  measure, and the harsher `extra_sensor_flag` (76 %) mostly counts sensors
  named as downstream *consequences* ("may raise T50") — legitimate physical
  reasoning, median 1 extra sensor.
- **The action policy collapses (fig 8b) — report this as a finding, not a
  bug.** **92 % of all recommendations are shutdown-class**, including
  gravity-2 cases with RUL > 200. The graded information lives in the gravity
  score (median true RUL 197 / 38 / 36 / 19 for g = 2/3/4/5; ρ = −0.20,
  p = 0.066 — correctly ordered, not yet significant at n = 89), while the
  recommendation text is uniformly maximal, almost certainly inherited from
  the few-shot exemplars' "I recommend stopping the machine" style. This is
  precisely the gap the A-RAD coordinator's outcome-derived action bands are
  designed to close — the edge SLM interprets; calibrated *action* selection
  needs the retrieval layer. That sentence is the bridge between the two
  halves of your paper.

**What not to claim:** do not call ρ(gravity, RUL) = −0.20 significant
(p = 0.066); call it a correctly-ordered weak signal and note the full
1,062-event run will decide it. Do not present the 89-event sample as the
final SLM evaluation — it is the pilot that validates the metrics pipeline.

---

## 3. How to read the hardware results (figs 6, 9)

The run is exactly what an edge deployment should look like: **GPU > 90 % busy
98.2 % of the time** with CPU at 4 % (the workload is on the accelerator);
decode throughput **44.0 tok/s with CV 5.3 % and ~zero drift** over 34 min
(no thermal degradation — peak **69 °C**, 16 °C below throttle); time budget
**94 % decode / 6 % prefill**, the signature of memory-bandwidth-bound SLM
inference; measured model + KV footprint **+8.2 GiB** (the RAM ramp in
fig 9b is the llama.cpp load, not a leak). Sustained **10.6 anomalies/min**
⇒ the full 1,062-event test set costs ≈ **1.7 h**.

**The one correction you must make in the manuscript:** telemetry says
29.8 GiB total RAM with a Tegra GPU — this board is **AGX-Orin-class, not the
Orin Nano 8 GB** the drafts have been naming. 44 tok/s on a 3B Q4 is
AGX-class; a Nano 8 GB would run roughly 2–3× slower and the +8.2 GiB
footprint would not even fit beside the OS. Either state the real board
(`cat /etc/nv_tegra_release`) or re-run on the Nano; do not let the text and
the telemetry contradict each other. Power rails are not exposed on this
image, so energy is unreported — run `sudo tegrastats` during the full run if
you want Watts, or drop the energy claim.

---

## 4. The paragraph the whole archive supports

A regime-conditional detector, frozen from the training fleet, separates
degraded from healthy operation at AUC 0.904 — a +0.095 gain attributable to
the operational clustering itself — with a 1.7 % healthy-zone false-alarm
rate, a 16-cycle median warning, and a cumulative-anomaly signature that
tracks remaining life in every single unit (median ρ = −0.62, 260/260). On
the device, a 3B-parameter SLM turns each isolation rule into a six-section
diagnostic in 5.6 s at 44 tok/s, thermally stable, with 94.9 % directional
faithfulness and zero echo — but it compresses (61 % sensor recall, falling
with rule complexity) and its action policy saturates at "shutdown"
regardless of severity, which is exactly the calibrated-decision gap the
retrieval-augmented coordinator tier exists to close.

*(Numbers in this README trace to: `if_eval/summary.csv`,
`if_eval_extra/summary.csv`, `slm_eval/summary.csv`,
`slm_eval_extra/summary.csv`, `hw_eval/summary.csv`. Figures: vector PDFs for
submission, PNGs for drafts. Sample: detector metrics on 260 run-to-failure
units; SLM metrics on the 89-interpretation Jetson pilot, 8 stratified test
units.)*

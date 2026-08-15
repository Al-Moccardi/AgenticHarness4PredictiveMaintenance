# Isolation-Forest effectiveness + paper figures

Two scripts. `eval_iforest.py` produces the numbers, `plot_paper.py` turns them
into camera-ready figures. Both are standalone (numpy/pandas/scikit-learn/
scipy/matplotlib only) — drop them in `scripts/` and run.

```bash
python3 scripts/eval_iforest.py                    # train fleet, 260 units
python3 scripts/plot_paper.py                      # figures 1-2 (+3 if anchoring metrics found)
python3 scripts/plot_paper.py --anchor results/your_anchoring_metrics.csv
```

## The evaluation problem, and how it is solved

The detector is unsupervised: CMAPSS has no ground-truth anomaly labels, so
"accuracy" is undefined. Every metric here is therefore anchored on **RUL**,
which the detector never observes — this is the only physically defensible
reference, and it makes the evaluation honest rather than circular.

**Weak labels.** degraded = RUL ≤ 30, healthy = RUL ≥ 100, mid-life excluded to
avoid an arbitrary boundary. Anomaly evidence = −decision_function. Everything
is swept over the threshold (`tau_sweep.csv`) so no conclusion rests on one
choice.

**Baselines.** The claim "regime-specific detection matters" is only meaningful
against alternatives, so two are computed on identical rows: a **global IF**
(same features, no clustering) and **Hotelling T²** (the regime-blind
multivariate control chart your related work criticises). Differences are
paired-bootstrapped, giving CIs and a bootstrap p — not two numbers side by side.

## Results obtained on your train fleet (53,759 rows, 260 units)

| quantity | value |
|---|---|
| AUC, regime-specific IF | **0.904** [0.899, 0.908] |
| AUC, global IF (no clustering) | 0.809 [0.804, 0.814] |
| AUC, Hotelling T² | 0.772 [0.766, 0.778] |
| ΔAUC vs global IF | **+0.095** [0.091, 0.099], p < 0.001 |
| ΔAUC vs Hotelling T² | **+0.132** [0.127, 0.137], p < 0.001 |
| PR-AUC, regime IF (prevalence 0.225) | 0.810 vs 0.611 / 0.562 |
| Spearman ρ (score vs RUL), pooled | 0.29 |
| per-unit ρ: median / share correct sign | 0.32 / **88 %** |
| false-alarm rate, healthy zone | **1.7 %** |
| alarm rate, degraded zone | 51.3 % |
| lead time, degradation phase (median, IQR) | **16 cycles** [10, 28] |
| units warned ≥ 20 cycles ahead | 38 % |
| per-regime AUC range | 0.885 – 0.916 |

Three things worth foregrounding in the paper. The **clustering earns its
place**: +0.095 AUC over the identical detector without it, CI far from zero —
that is the quantitative version of your Fig. 3 control-chart argument. The
detector is **regime-fair**: AUC varies only 0.885–0.916 across the six
operating profiles, so no single mode absorbs the alarms. And it is
**operationally usable**: 1.7 % false alarms in the healthy zone with a median
16-cycle warning.

## Two honest caveats the scripts enforce

**Per-regime alarm rate is ~10 % by construction** — `contamination=0.1` is
enforced inside each cluster, so the flat alarm rate across regimes is an
artefact, not a result. The script labels that metric explicitly and reports
`per_regime_healthy_alarm_spread_pp` (0.87 %–3.35 %) and per-regime AUC instead,
which are free to vary. Do not put the flat 10 % in a table as evidence.

**The alarm profile is a bathtub, not a ramp** (figure 1b): 3.8 % at break-in,
0.24 % mid-life, 68 % in the final decile. The early bump is real, so the naive
"first sustained alarm" lead time is inflated by break-in events. The script
therefore also reports `lead_time_degradation_*`, which ignores alarms in the
first 50 % of life (`--burnin-frac`), and the figure uses that. Report the
degradation-phase number — 16 cycles — not the 19 you get by counting break-in.

Note the pooled ρ = 0.29 is lower than the ρ ≈ 0.64 in the RAD paper. That
earlier figure was computed on a different subset; if you cite both, say which
population each refers to, or recompute the old one with this script.

## Figures

`fig1_detector_effectiveness` — (a) score-vs-RUL hexbin density with the alarm
threshold and ρ; (b) the bathtub alarm profile with break-in/wear-out annotated;
(c) ROC for the three detectors with AUC + CI in the legend; (d) degradation-
phase lead-time histogram with median.

`fig2_regime_and_baselines` — (a) per-regime AUC bars + healthy false-alarm line
on a twin axis; (b) AUC vs degradation threshold for all three detectors;
(c) ΔAUC forest plot with bootstrap CIs; (d) per-unit ρ distribution.

`fig3_anchoring` — rendered only when an anchoring-metrics CSV is found
(auto-discovered as `results/**/*anchor*|*grounding*|*faithful*.csv`, or passed
with `--anchor`). Column names are **discovered, not assumed**: numeric columns
whose values lie in [0, 1] are treated as rates, ranked, drawn with Wilson 95 %
intervals when the table is per-anomaly, and the top metric gets a distribution
panel. If your metrics table has a different shape, send me its header and I
will tailor the panel rather than guess.

Style: serif, Okabe-Ito colour-blind-safe palette, Elsevier column widths
(90 mm / 190 mm), vector PDF with embedded editable text (Type 42) + 300 dpi PNG,
panel letters (a)–(d).

## Applying it to the test split

```bash
python3 scripts/eval_iforest.py --csv results/test_anomalies.csv \
    --outdir results/if_eval_test --label "test units, censored" --tau-pos 30
python3 scripts/plot_paper.py --eval results/if_eval_test \
    --outdir results/figures/paper_test
```
Expect weaker absolute numbers there and say why: the official test
trajectories are right-censored, so the degraded stratum is thin. The train
fleet is the correct population for *detector* effectiveness; the test split is
for the streaming/interpretation demonstration.


---

# Part 2 — cumulative-anomaly dynamics + SLM evaluation

## Cumulative anomalies (added to `eval_iforest.py`)

The detector's temporal signature is the strongest evidence that it tracks
physics rather than noise. Metrics, all against RUL which the detector never
sees:

| quantity | value |
|---|---|
| per-unit Spearman ρ(C_i, RUL), median | **−0.62** |
| units with ρ(C_i, RUL) < 0 | **100 % (260/260)** |
| pooled ρ(C_i, RUL) | −0.29 |
| pooled ρ(F_i, RUL) — anomaly acceleration | −0.34 |
| normalised C_i, RUL≥150 → RUL<10 | 0.045 → **0.72** |
| alarm rate, RUL≥150 → RUL<10 | 3.3 % → **84.7 %** |

Report the **per-unit** figure, not the pooled one. Cumulative counts are not
comparable across units (a 380-cycle engine accrues more than a 130-cycle one),
so pooling them mixes populations and dilutes ρ from −0.62 to −0.29. The
per-unit statistic is the correct one and it is unanimous: every single unit
shows the expected sign.

## SLM evaluation (`eval_slm.py`)

Deterministic, no human annotation: the isolation rule gives
(sensor, operator, threshold) triples, and the interpretation is text, so
grounding is checkable by construction. Results on the 89 real
Llama-3.2-3B-Instruct-Q4_K_M interpretations:

| metric | value | reading |
|---|---|---|
| format compliance | **100 %** [95.9, 100] | all six sections, every time |
| echo contamination | **0 %** [0, 4.1] | the run-1 defect is gone |
| direction agreement | **94.9 %** [92.4, 97.3] | high/low matches >/≤ |
| direction contradiction | **4.3 %** [2.2, 6.5] | genuine errors, low |
| threshold anchoring | 80.0 % [74.3, 85.6] | the numeric value is cited |
| sensor precision | 77.7 % [74.7, 80.9] | mentioned sensors that are in the rule |
| sensor recall | 61.4 % [57.7, 65.4] | rule sensors the text mentions |
| ρ(gravity, true RUL) | **−0.20**, p = 0.066 | correct sign, marginal |
| latency p50 / p95 | 5.6 s / 6.4 s | measured on device |
| decode throughput | 44.3 tok/s | |

Three findings worth writing up.

**Faithfulness is high, coverage is partial.** Direction agreement 94.9 % with
only 4.3 % contradictions means the model rarely misstates a sensor's direction
— but recall 61.4 % means it summarises rather than enumerates. That is a
*compression* behaviour, not a hallucination behaviour, and the two must be
reported separately.

**Coverage degrades monotonically with rule complexity** (figure 5b): recall
falls 100 % → 78 % → 60 % → 41 % as the rule spans 2 → 4 → 6 → 8 distinct
sensors. This reproduces, at the interpretation level, the same complexity
effect your RAD paper found for strict rule recovery, and it is the cleanest
argument for keeping rules compact at the edge.

**The severity opinion tracks reality, weakly.** Median true RUL by assigned
gravity: 197 (g=2), 38 (g=3), 36 (g=4), 19 (g=5) — monotone in the right
direction, ρ = −0.20, p = 0.066 on n = 89. Do not call this significant; call it
a correctly-ordered but weak signal, and note it is a marked improvement over
the earlier audit where gravity carried no outcome information. More anomalies
(the full 1,062-event run) would settle it.

`sensor_precision` is the primary unsupported-content measure. The harsher
`extra_sensor_flag` (76 %, ≥1 sensor named that is not in the rule) is reported
but must not be called hallucination: naming a sensor as a downstream
*consequence* ("this may raise T50") is legitimate reasoning. Inspect
`n_extra_sensors` (median 1) before drawing conclusions.

## Figures

| file | content |
|---|---|
| `fig1_detector_effectiveness` | score-vs-RUL density, bathtub alarm profile, ROC, lead time |
| `fig2_regime_and_baselines` | per-regime AUC + false alarms, τ sweep, ΔAUC forest, per-unit ρ |
| `fig3_cumulative_dynamics` | C_i curves per unit, normalised C_i vs RUL, per-unit ρ histograms, F_i + alarm rate |
| `fig4_discrimination_detail` | precision-recall, score separation, alarm rate over life, lead-time ECDF |
| `fig5_slm_quality` | grounding rates with CI, recall vs rule complexity, gravity vs true RUL, per-anomaly spread |
| `fig6_slm_edge_cost` | latency histogram, tokens vs latency, prefill/decode split, device telemetry |

Run order:

```bash
python3 scripts/eval_iforest.py          # IF metrics (train fleet)
python3 scripts/eval_slm.py              # SLM metrics (the 89 interpretations)
python3 scripts/plot_paper.py            # all six figures
```

`plot_paper.py` skips SLM figures gracefully if `results/slm_eval/` is absent,
and still renders the anchoring figure if it finds your own metrics CSV
(`--anchor`). Note figure 6 panel (d) shows RAM climbing over the first ~7
minutes: that is the llama.cpp model load and KV-cache warm-up, not a leak.

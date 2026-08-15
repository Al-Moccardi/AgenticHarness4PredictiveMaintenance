# Edge SLM interpretation — deterministic evaluation against the rule

89 interpretations. Every metric is computed from the isolation-forest rule the model was given; no human annotation. Binary rates carry Wilson 95% intervals, ratio metrics a 2000-sample bootstrap. The model never sees RUL, so `spearman_gravity_vs_RUL` tests whether its severity opinion tracks reality.

| metric | value | 95% CI | n |
|---|---|---|---|
| sensor_recall | 0.6141 | [0.5772, 0.6544] | 89 |
| sensor_precision | 0.7771 | [0.7466, 0.8088] | 89 |
| direction_agreement | 0.9493 | [0.9235, 0.973] | 89 |
| direction_contradiction | 0.0427 | [0.022, 0.0646] | 89 |
| threshold_anchoring | 0.8001 | [0.7432, 0.8558] | 89 |
| hallucinated_sensor | 0.764 | [0.6661, 0.8402] | 89 |
| format_complete | 1 | [0.9586, 1] | 89 |
| has_gravity | 1 | [0.9586, 1] | 89 |
| echo_contamination | 0 | [0, 0.0414] | 89 |
| truncated | 0 | [0, 0.0414] | 89 |
| spearman_gravity_vs_RUL | -0.1955 |  | 89 |
| spearman_gravity_vs_RUL_pvalue | 0.0663 |  | 89 |
| spearman_gravity_vs_anomaly_score | -0.1714 |  | 89 |
| cost_wall_p50 | 5.595 |  | 89 |
| cost_wall_p95 | 6.405 |  | 89 |
| cost_gen_tokens_median | 233 |  | 89 |
| cost_decode_tps_median | 44.3 |  | 89 |
| cost_prefill_tps_median | 1707.7 |  | 89 |
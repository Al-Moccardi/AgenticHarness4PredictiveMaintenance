# Isolation-Forest effectiveness — RUL-anchored evaluation

Dataset: `data/anomalies_multimodal.csv` (train fleet, run-to-failure). Degradation weak labels: positive = RUL <= 30, negative = RUL >= 100. Anomaly evidence = -decision_function. Bootstrap: 200 resamples, 95% percentile CIs. The detector never observes RUL.

| metric | value | 95% CI | n | note |
|---|---|---|---|---|
| spearman_score_vs_RUL | 0.2905 |  |  | p=0 |
| spearman_per_unit_median | 0.3204 |  | 260 |  |
| spearman_per_unit_frac_expected_sign | 0.8769 |  |  | rho>0 = score falls as RUL falls = expected direction |
| AUC_regime_IF | 0.9036 | [0.8992, 0.9076] | 35819 |  |
| PRAUC_regime_IF | 0.8098 | [0.8028, 0.8167] | 35819 | prevalence=0.225 |
| AUC_global_IF | 0.8091 | [0.8041, 0.8142] | 35819 |  |
| PRAUC_global_IF | 0.6105 | [0.6014, 0.6212] | 35819 | prevalence=0.225 |
| AUC_hotelling_T2 | 0.7721 | [0.7678, 0.7773] | 35819 |  |
| PRAUC_hotelling_T2 | 0.5619 | [0.5513, 0.5711] | 35819 | prevalence=0.225 |
| delta_AUC_regimeIF_vs_global_IF | 0.0946 | [0.0904, 0.0989] | 35819 | bootstrap p=0.0000 |
| delta_AUC_regimeIF_vs_hotelling_T2 | 0.1315 | [0.1265, 0.1368] | 35819 | bootstrap p=0.0000 |
| units_with_sustained_alarm | 260 |  | 260 | m=2 alarms within w=10 cycles |
| lead_time_median_cycles | 19 | [12, 46.25] | 260 | IQR in CI columns |
| frac_units_lead_ge_20 | 0.4962 |  | 260 |  |
| frac_units_lead_ge_50 | 0.2346 |  | 260 |  |
| lead_time_degradation_median_cycles | 16 | [10, 28] | 260 | break-in excluded: alarms after 50% of life; IQR in CI columns |
| frac_units_degradation_lead_ge_20 | 0.3846 |  | 260 |  |
| alarm_rate_healthy_zone | 0.0172 |  | 27759 | RUL >= 100 |
| alarm_rate_degraded_zone | 0.5134 |  | 8060 | RUL <= 30 |
| alarm_rate_overall | 0.1001 |  | 53759 |  |
| per_regime_alarm_rate_spread_pp | 0.009 |  | 6 | ~0 BY CONSTRUCTION: contamination=0.1 is enforced inside each cluster, so this is not evidence of anything |
| per_regime_healthy_alarm_spread_pp | 2.49 |  | 6 | max-min false-alarm rate across regimes (the informative one) |
| per_regime_auc_min | 0.8849 |  | 6 |  |
| spearman_cumcount_vs_RUL_pooled | -0.2891 |  | 53759 | negative = anomalies accumulate as life runs out |
| spearman_cumcount_vs_RUL_per_unit_median | -0.6213 |  | 260 |  |
| frac_units_cumcount_rho_negative | 1 |  | 260 |  |
| spearman_last3freq_vs_RUL_pooled | -0.3383 |  | 53759 | anomaly acceleration vs remaining life |

Reading guide:
- `spearman_score_vs_RUL` is POSITIVE in the expected direction: decision_function is low for anomalies, so it falls together with RUL as the unit degrades.
- `delta_AUC_regimeIF_vs_*` > 0 with a CI excluding 0 is the evidence that regime-specific detection beats the regime-blind alternatives on identical rows.
- `alarm_rate_healthy_zone` is the operational false-alarm proxy. Note `per_regime_alarm_rate_spread_pp` is ~0 BY CONSTRUCTION (contamination is enforced per cluster) and must NOT be reported as a result; use `per_regime_healthy_alarm_spread_pp` and the per-regime AUCs, which are free to vary.
- The alarm-rate-vs-life profile is U-shaped: elevated at break-in, minimal mid-life, sharply rising near end of life. Report both arms — the early-life bump is real, not noise.
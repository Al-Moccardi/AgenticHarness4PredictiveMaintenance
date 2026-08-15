# Isolation Forest — additional statistics

| metric | value | note |
|---|---|---|
| top_rule_sensor | NRc | in 53% of anomaly rules |
| rule_sensor_entropy_bits | 3.88 | max 3.91 = uniform |
| best_univariate_sensor | Ps30 | AUC 0.848 |
| delta_auc_IF_vs_best_univariate | 0.0557 | IF 0.904: the multivariate rule adds this |
| seed_label_agreement_mean | 0.9759 | 10 pairs, 5 seeds |
| seed_auc_range | 0.8998-0.9293 | labels 97.6% stable; AUC varies ~±0.015 with the RNG |
| operating_point_10pct | FA 1.7%, lead 16 | the deployed contamination |
| spearman_evidence_vs_centroid_distance | 0.752 | IF evidence correlates with, but is not, regime distance |
| auc_centroid_distance | 0.9291 | vs IF 0.9036. Distance-to-own-centroid is ALSO regime-conditional and slightly beats IF on this weak-label task -- report honestly: regime-conditioning is the driver (both >> blind baselines 0.77-0.81); the IF buys the INTERPRETABLE RULES the language layer requires, which a distance scalar cannot provide |
| alarm_run_len_median_overall | 1.0 | 1604 runs |
| alarm_run_len_median_last20pct_life | 2.0 | sustained bursts near failure vs isolated blips elsewhere |
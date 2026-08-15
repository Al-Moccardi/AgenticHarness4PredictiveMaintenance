# Hardware statistics — edge interpretation run

| metric | value | note |
|---|---|---|
| device_ram_total_gib | 29.8 | Tegra sysfs GPU present -> Jetson AGX-Orin-class (NOT Orin Nano 8GB) |
| telemetry_minutes | 34.2 | 684 samples @ 2.0s |
| cpu_pct_mean | 4.1 |  |
| cpu_pct_max | 20.5 |  |
| ram_gib_baseline | 9.28 | before model load |
| ram_gib_peak | 17.44 |  |
| ram_gib_model_footprint | 8.16 | peak - baseline = llama-server weights + KV |
| gpu_pct_mean | 96.0 |  |
| gpu_pct_p95 | 99.8 |  |
| gpu_busy_frac_gt90 | 0.982 | GPU-bound confirmation |
| temp_c_start | 55.9 |  |
| temp_c_peak | 69.3 | Orin soft throttle ~ >85C: no throttling |
| temp_c_rise | 13.4 |  |
| power_rails | not exposed | INA3221 path absent on this image; use `sudo tegrastats` for W |
| interp_n | 89 |  |
| wall_s_p50 | 5.6 |  |
| wall_s_p95 | 6.4 |  |
| decode_tps_mean | 44.0 |  |
| decode_tps_cv_pct | 5.3 | coefficient of variation: thermal/clock stability |
| decode_tps_drift_per_min | -0.555 | linear drift across the run; ~0 = thermally stable |
| prefill_tps_median | 1708.0 |  |
| throughput_anomalies_per_min | 10.59 | SLM busy-time basis |
| prefill_share_pct | 5.9 | decode dominates -> memory-bandwidth-bound |
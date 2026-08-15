# studies — earlier experimental campaigns (runnable, not part of the live pipeline)

Study A pattern grid (bench_patterns, report_patterns), the v1/v2-era
diagnosis stack (diagnosis, agent, faults, features, interpret_kb, events),
SLM baselines (bench_slm, run_llm, run_ml, ml_models), older prognosis
studies (bench_prognosis, prognosis, report_prognosis, smoke_prognosis,
dl_seed_study), the RAGAS judge audit (ragas_local), official-split runs
(official, run_official), reports and misc (report, metrics, tools,
gen_interpretations, smoke_test).

Their results live in ../../results/ (study_A_pattern_grid,
v2_rul_coupled_collapse). Invoke as: python -m apdm.studies.<name>

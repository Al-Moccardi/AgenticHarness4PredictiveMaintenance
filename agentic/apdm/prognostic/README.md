# Prognostic agent — forecasts WHAT WILL HAPPEN

The CNN-GRU owns the RUL number (given, deterministic). The SLM's job is
the progression hypothesis: "projected_progression" + a structured
"progression_horizon", grounded in the cited precedents' observed
futures. Every ticket carries the deterministic signals
(future_progression.py): RELIABILITY (similarity of the case to the
knowledge base), FUTURE PROGRESSION (each precedent's continuation), and
PROGRESSION UNCERTAINTY in [0,1] (dissimilarity of the cited futures;
89-case spread [0.00, 0.83], median 0.42; lookup cost ~0.1 ms/case).

Files: bench_forecast (the bench, all arms incl. P7_progression),
forecast (prompts p7 / p7_progression + parsers), future_progression
(signals), eval_progression (scorer), inference_time (timings),
dl_rul (CNN-GRU utilities), run (entry), tools (facade).
Results: ../../..../agentic/results/final_prognostic (published),
clean_tool_study (the inversion), progression_run (your next run).

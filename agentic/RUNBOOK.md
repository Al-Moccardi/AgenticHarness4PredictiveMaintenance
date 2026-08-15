# RUNBOOK — agentic tier (laptop, fully local)

## 0. Setup
    pip3 install -r requirements.txt tensorflow
    ollama pull llama3.2:3b            # coordinator + judge
    ollama pull nomic-embed-text       # query embedder (must match the store)

## 1. Offline plumbing checks (~1 min, no model calls)
    python -m apdm.bench_patterns --smoke
    python -m apdm.bench_forecast --smoke

## 2. CNN-GRU (TensorFlow needed only here)
    python -m apdm.dl_rul --train --epochs 100      # run-to-failure fleet, seed 42
    python -m apdm.dl_rul --hints                   # -> queries/dl_hints.csv (capped, monotone)

## 3. Study A — diagnostic pattern grid (v2 prompts -> results/patterns_v2)
    python -m apdm.bench_patterns --limit 8         # pilot first
    python -m apdm.bench_patterns                   # full, resumable
    python -m apdm.bench_patterns --evaluate        # local-judge RAGAS + grounding
    python -m apdm.report_patterns                  # summary + fig10, fig11

## 4. Study B — prognostic agent  (89 x 4 arms, unit-chained, cap-125)
    python -m apdm.bench_forecast --limit 8         # pilot
    python -m apdm.bench_forecast                   # full, resumable
    python -m apdm.bench_forecast --evaluate        # vs realised future + RUL gold
    python -m apdm.bench_forecast --report          # forecast_summary.csv + fig12

## 5. Deep evaluation (paper-grade tables + figures A1-A4)
    python3 evaluation/scripts/eval_patterns.py
    python3 evaluation/scripts/eval_forecast_agent.py

## 6. Outputs
    results/patterns/            episodes.jsonl, metrics.jsonl, summary, fig10-11
    results/forecast/            forecast episodes/metrics/summary, fig12
    evaluation/results/          paired deltas, per-unit views, figA1-A4

Knobs: --patterns / --styles / --arms to subset; --k retrieval depth;
--model llama3.1:8b for a stronger coordinator; --evaluate --judge-model
llama3.1:8b if the 3B judge looks too lenient (P1 ≈ P2 on faithfulness is
the tell). Run long phases under tmux; every phase resumes.

## 3b. REDUCED two-agent study (recommended): bench_arad
    # diagnostic agent vs floors (356 eps ~1.2h) + tool-using prognostic
    # agent chained on the diagnostic ticket (~1.5h) — same 8 edge units
    python -m apdm.bench_arad --smoke
    python -m apdm.bench_arad
    # then the EXISTING evaluators, pointed at the arad dirs:
    python -m apdm.bench_forecast --evaluate --out results\arad\prog
    python -m apdm.bench_forecast --report   --out results\arad\prog
    python evaluation\scripts\eval_patterns.py --dir results\arad\diag
    python evaluation\scripts\eval_forecast_agent.py --dir results\arad\prog

## 6. CNN-GRU seed-robustness study (learning curves + unit-bootstrap CIs)
    python -m apdm.dl_seed_study --epochs 100          # full, ~1-2h GPU
    python -m apdm.dl_seed_study --epochs 40 --patience 6   # faster
Outputs -> evaluation/results/dl_seed_study/: histories.csv,
val_predictions.csv, seed_summary.csv, fig_dl_seeds.(pdf|png).
Fixed validation units (meta-seed 42, the production split); the varying
seed isolates initialisation + SGD stochasticity; MAE CIs are bootstrapped
over UNITS, not rows.

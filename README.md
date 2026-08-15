# AgenticRag — Agentic Retrieval-Augmented Predictive Maintenance with SLMs at the IoT Edge

Flora Amato · Alberto Moccardi · Rajib Chandra Ghosh
IDEAL Lab, DIETI, University of Naples Federico II

## What this is
A three-tier maintenance pipeline on CMAPSS FD002: the EDGE detects and
interprets anomalies and retrieves similar cases from a fleet knowledge
base; the DIAGNOSTIC agent looks into the past and issues a verified
ticket; the PROGNOSTIC side forecasts the future; a SYNTHESIS layer
composes one final gated ticket. Every responsibility was assigned by a
pre-registered experiment, not by preference.

## The decisions (who owns what, and which experiment decided it)
| responsibility            | owner                          | decided by |
|---------------------------|--------------------------------|------------|
| anomaly interpretation    | edge SLM (Jetson)              | edge study |
| diagnosis: severity/action/citations | SLM agent (P5_verifier) | final_diagnostic: 88/89, 1 escalation |
| RUL number                | CNN-GRU tool (deterministic)   | clean_tool_study: agent 50.3 vs tool 15.6 |
| future outlook figures    | deterministic FUTURE PROGRESSION signal (median/range TTF) | progression_run: SLM horizon rho +0.27, evidence-paraphrase |
| progression narrative     | SLM (commentary, cited)        | progression_run: 89/89, 75 templates, 8% modal |
| ticket trust              | RELIABILITY + PROGRESSION UNCERTAINTY (deterministic) | plot_reliability + signal spread [0.00, 0.83] |
| final ticket composition  | SLM composer (JSON-first: model writes prose, code injects facts) | synthesis_run LIVE: 89/89 clean, 2.8 s/ticket |
Recurring finding: at 3B, the model's decision to answer is better
calibrated than its answers (silent forecasts, voluntary-production
selectivity: rho +0.75 voluntary vs +0.27 forced).

## Structure
    edge/                    Tier 1 (self-contained; own README/RUNBOOK; paper/png)
    agentic/
      apdm/
        diagnostic/            the past  — bench, run, tools, README, RUNBOOK, paper/
        prognostic/            the future — bench, forecast, signals, eval, timings,
                               run, tools, README, RUNBOOK, paper/
        synthesis/             the reporting layer — compose (gates+template), run,
                               README, RUNBOOK
        studies/               earlier campaigns (runnable; own README)
        attic/                 parked dead code
        data.py llm.py patterns.py vector_store.py hardware.py   <- shared core
      data/ queries/ models/  knowledge base, query sets, CNN-GRU hints
      results/                ALL experimental data:
        study_A_pattern_grid     1,691-episode prompt-pattern study
        v2_rul_coupled_collapse  the 87/89-escalation mechanism study
        final_diagnostic         published diagnostic run (88/89)
        final_prognostic         published prognostic run
        clean_tool_study         the inversion finding
        progression_run          progression-only run (89/89; horizon retired)
        synthesis_run            all 89 composed final tickets (final_tickets.md)
        analysis/                curated: figBAND1_signal_heatmap,
                                 figBAND2_3d + all stats (.md files:
                                 signals, MAE, bands, uncapped, costs)
    paper/                    shared engine: code/make_figures.py (VERIFIED 21/21),
                              code/figures/, code/stats/, code/plot_reliability.py,
                              figures/ (committed selected set)
    requirements.txt

## Run (see RUNBOOK.md for the full sequence)
    cd C:\Users\Alberto\Desktop\AgenticRag\agentic
    python -m apdm.synthesis.run --backend ollama        <- the one pending live run
    python -m apdm.synthesis.run --stats

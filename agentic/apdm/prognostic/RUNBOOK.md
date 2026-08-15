# RUNBOOK — prognostic agent (PowerShell 5)

STATUS: campaign complete. results\final_prognostic (published run),
results\clean_tool_study (the inversion), results\progression_run
(89/89 hypotheses; the numeric horizon was RETIRED by its own result:
rho(horizon, true RUL) +0.27 vs +0.59 to the cited evidence - the
narrative stays as commentary, the outlook figures come from the
deterministic FUTURE PROGRESSION signal).

Reproduce:
    cd C:\Users\Alberto\Desktop\AgenticRag\agentic
    python -m apdm.prognostic.run                       (progression arm)
    python -m apdm.prognostic.eval_progression --dir results\progression_run
    python -m apdm.prognostic.run --arms b0_median dl_only P7_agent P7_agent_dl --out C:\Users\Alberto\Desktop\AgenticRag\agentic\results\full_rerun
    python -m apdm.prognostic.inference_time --dir results\final_prognostic
    python -m apdm.prognostic.future_progression time
Requires Ollama serving llama3.2:3b + nomic-embed-text.

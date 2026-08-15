# RUNBOOK — AgenticRag (PowerShell 5, one line at a time)

## 0. Environment (once)
    cd C:\Users\Alberto\Desktop\AgenticRag
    pip install -r requirements.txt
    ollama pull nomic-embed-text
    ollama pull llama3.2:3b

## 1. STATUS: what is already done (results in agentic\results\)
- final_diagnostic       diagnostic agent, published (88/89)
- final_prognostic       prognostic bench, published
- clean_tool_study       the inversion experiment
- progression_run        progression-only arm, completed (89/89; the
                         numeric horizon was retired by its result)
- synthesis_run          LIVE, COMPLETE: 89/89 clean tickets from the
                         3B (2.8 s/ticket, 4.1 min). final_tickets.md
                         holds all of them.

## 2. NOTHING PENDING — the experimental campaign is complete.
Formal work-order PDF from any ticket:
    cd C:\Users\Alberto\Desktop\AgenticRag
    python paper\code\make_ticket_pdf.py --qid T65c321
-> results\synthesis_run\workorder_T65c321.pdf

## 3. Reproduce anything else
    Diagnostic rerun:    python -m apdm.diagnostic.run
    Prognostic 4 arms:   python -m apdm.prognostic.run --arms b0_median dl_only P7_agent P7_agent_dl --out C:\Users\Alberto\Desktop\AgenticRag\agentic\results\full_rerun
    Progression arm:     python -m apdm.prognostic.run
    Evaluate progression: python -m apdm.prognostic.eval_progression --dir results\progression_run
    Timings:             python -m apdm.prognostic.inference_time --dir results\final_prognostic
    Signal cost:         python -m apdm.prognostic.future_progression time

## 4. Paper artifacts
    cd C:\Users\Alberto\Desktop\AgenticRag
    python paper\code\make_figures.py                (must end: VERIFIED 21/21)
    python paper\code\stats\phase2_analysis.py
    python paper\code\plot_reliability.py
    python paper\code\stats\signal_analysis.py        (signals vs MAE, uncertainty distribution)
    python paper\code\stats\pipeline_costs.py         (end-to-end time + token accounting)
    python paper\code\stats\mae_analysis.py           (MAE stats: stage, rescue split, quadrants, per-unit)
    python paper\code\stats\band_focus.py             (20-60 band: conservativeness + reliability x uncertainty x MAE)
    python paper\code\stats\uncertainty_deep.py       (cap excluded: RUL vs agent difference + the uncertainty-MAE mechanism)
Per-tier details: edge\RUNBOOK.md, agentic\apdm\diagnostic\RUNBOOK.md,
agentic\apdm\prognostic\RUNBOOK.md, agentic\apdm\synthesis\RUNBOOK.md

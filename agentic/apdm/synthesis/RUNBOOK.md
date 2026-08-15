# RUNBOOK — synthesis layer (PowerShell 5)

## offline check (no Ollama; template path, must end with a ladder print)
    cd C:\Users\Alberto\Desktop\AgenticRag\agentic
    python -m apdm.synthesis.run --backend mock
    python -m apdm.synthesis.run --stats

## live (Ollama serving llama3.2:3b; ~6-10 s per ticket)
    del /q results\synthesis_run\tickets.jsonl 2>NUL
    python -m apdm.synthesis.run --backend ollama
    python -m apdm.synthesis.run --stats
Outputs: results\synthesis_run\tickets.jsonl (state, gates, wall_s per
ticket) and final_tickets.md (all tickets, readable).
Best run AFTER the progression run, so projected_progression is merged in.

# RUNBOOK — diagnostic agent (PowerShell 5)

Published results already in agentic\results\final_diagnostic - nothing
needs rerunning. To reproduce or extend:
    cd C:\Users\Alberto\Desktop\AgenticRag\agentic
    python -m apdm.diagnostic.run --help
    python -m apdm.diagnostic.run
Output -> results\diagnostic_rerun. Requires: pip install -r
..\requirements.txt and Ollama serving llama3.2:3b + nomic-embed-text.

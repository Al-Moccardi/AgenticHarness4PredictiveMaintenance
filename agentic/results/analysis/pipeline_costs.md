# End-to-end pipeline costs (per anomaly, fleet of 89)

| stage                                                    | s/case   | tok_in/case   | tok_out/case   |    n | source                                                                  |
|:---------------------------------------------------------|:---------|:--------------|:---------------|-----:|:------------------------------------------------------------------------|
| edge interpretation (Jetson)                             | ~        | -             | 6              | 1541 | estimated from see edge/evaluation (fig9_hardware)                      |
| diagnostic agent (P5_verifier)                           | 11.5     | 2594          | 211            |   89 | final_diagnostic episodes (measured)                                    |
| prognostic agent (P7_agent_dl, published)                | 16.5     | n/l           | n/l            |   89 | final_prognostic episodes (wall measured; tokens not logged)            |
| progression arm (incl. inline diagnosis)                 | 22.2     | 2422          | 258            |   89 | progression_run llm logs (exact Ollama counts)                          |
| signals (reliability + future progression + uncertainty) | 0.0001   | 0             | 0              |   89 | future_progression time (measured, ~0.1 ms)                             |
| synthesis composer                                       | 7.3      | ~1400         | ~220           |   89 | synthesis_run (offline template path; live 3B sample: 7.3 s/case, n=11) |

END-TO-END (edge -> ticket): ~58 s per anomaly; ~85 min for the full 89-anomaly campaign (single-stream, one Jetson + one RTX 4070 Laptop).
Measured token totals (progression campaign): 216k in / 23k out.

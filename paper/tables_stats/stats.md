# A-RAD final statistics

- retrieval fix: rho(cited outcome, true RUL) 0.05 -> 0.50
- rescue [usable]: n=29, tool MAE=27.7, agent MAE=31.8, override=38%
- rescue [corrupted]: n=60, tool MAE=33.5, agent MAE=18.9, override=97%
- paired P7_agent_dl vs dl_only: +7.0 [+2.7,+11.5]
- paired P7_agent_dl vs P7_agent: -13.4 [-20.8,-6.2]
- paired P7_agent_dl vs b0_median: -18.9 [-25.1,-13.1]
- severity validity B0_retrieval: rho=-0.348 (p=0.00082)
- severity validity P1_direct: rho=-0.220 (p=0.0382)
- severity validity P2_rag: rho=-0.136 (p=0.205)
- severity validity P5_verifier: rho=-0.130 (p=0.225)
- diag steps mean (P5): 1.46 (1 generation + 0.46 repairs)
- steps (no self-termination): 3.99 of 4 budget

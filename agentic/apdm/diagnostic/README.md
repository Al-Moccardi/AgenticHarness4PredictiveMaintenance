# Diagnostic agent — looks into the PAST

Matches the unit's anomaly progression against precedent histories and
produces the verified ticket: severity 1-5, action bound by the severity
contract, verifiable citations (gates D4-D6, bounded repair, escalation).
Published: 88/89 on both postconditions, 1/89 escalated
(results/final_diagnostic). The mechanism study behind the design is
results/v2_rul_coupled_collapse (87/89 escalated); the prompt-pattern
study is results/study_A_pattern_grid (1,691 episodes).

Files: bench_arad (the bench), run (entry), tools (facade over the shared
core; retrieval + diagnosis live in ../patterns.py).

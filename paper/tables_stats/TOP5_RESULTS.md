# A-RAD — Top 5 results of interest

**1. Deterministic anchoring makes the agent conservative by construction.**
Without the CNN-GRU anchor the 3B agent over-promises life in 36.6% of
cases (>20 cycles; q95 = +78). With the anchor: 0.0% dangerous
over-predictions, q95 = +2, safe-side 93% — at a bounded, verified
overhead of +7.0 cycles MAE [+2.8, +11.6] against the clean tool.
The anchor does not just improve accuracy; it flips the risk profile.
→ Fig07 (risk profile), Fig08 (paired effects).

**2. The rescue: measured, selective tool-skepticism.**
On the 60/89 cases whose DL hint was corrupted to ~0, the agent —
never told of the fault — overrode its tool in 97% of cases and cut the
error from 33.5 to 18.9 cycles (paired −13.9 [−14.9, −12.7]), staying
safe-side in 87%. The override policy is systematic: P(override) rises
36% → 84% → 92% with the disagreement between hint and the median
cited precedent future, and the overriding estimate anchors on those
precedent futures. → Fig09 (hero unit T65), Fig10 (policy),
supplement figN2.

**3. Superiority exactly where it matters: the end-of-life zone.**
Inside true RUL < 20 (n=17) the agent halves its own clean tool:
MAE 3.2 vs 10.6, median S-score 0.2 vs 1.3, 88% of cases improved.
Together with (1): conservative reserve early in life, sharpness at
end of life — trust early, arbitrate late. → Fig11, Fig12,
supplement figN7.

**4. The diagnostic mechanism, its fix, and the collapse of escalation.**
Plain cosine retrieval cites outcomes that ignore the case
(ρ(cited outcome, true RUL) = 0.05); grounding constraints make that
bias binding and the RUL-coupled design escalates 97.8% of cases.
Stage-aware retrieval restores case relevance (ρ → 0.50, paying only
0.013 mean similarity) and the retrospective redesign brings
escalation to 1.1% with 12.4% auto-corrected, 0.46 repairs/case, and
action coherence 8% → 99%. → Fig02, Fig03, Fig05.

**5. Severity compression — the graded signal lives in retrieval.**
The agent's severity is stage-agnostic (ρ = −0.13, n.s.; median 3 in
every life band), while the deterministic stage-matched precedent
gravity is weakly but significantly graded (B0: ρ = −0.35, p < 0.001).
Design implication for the discussion: severity prior from precedents,
LLM adjustment on top — the same anchoring lesson as prognosis,
mirrored. → Fig04.

*Boundary notes for honesty (not top-5, but quotable):* outlook
prediction collapses to the base rate (139/139 "accelerating",
0.71 = base rate), per-sensor trends sit at 3-class chance
(0.29–0.31), and 3B models never self-terminate tool loops
(3.98/4 steps used). → Fig15.

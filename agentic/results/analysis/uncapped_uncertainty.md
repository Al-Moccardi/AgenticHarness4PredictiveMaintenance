# Cap excluded (true RUL < 125): n=66, agent answered 64

## 1. RUL vs agent difference — conservativeness profile
  $<$20 n=17: agent bias   +0.6 (q05   -4.2, q95   +8.2, over-pred 38%)  |  tool bias  -10.6
  20-39 n=30: agent bias  -13.7 (q05  -23.6, q95   -5.5, over-pred  0%)  |  tool bias  -18.2
  40-59 n=11: agent bias  -31.1 (q05  -43.5, q95  -19.5, over-pred  0%)  |  tool bias  -21.5
 60-124 n= 8: agent bias  -47.1 (q05  -54.7, q95  -33.4, over-pred  0%)  |  tool bias  -29.4
UNCAPPED overall: agent bias -16.8, under-pred 88%, over-pred 9% (max over +9), dangerous(>+20): 0
rho(true, agent signed error) = -0.99  (more life -> deeper under-prediction)

## 2. Uncertainty vs MAE (agent+tool) — the mechanism
rho(unc, agent MAE) = -0.41   rho(unc, tool MAE) = -0.31
rho(unc, agent signed) = +0.39  (higher disagreement -> milder under-bias)
anchoring test:
  rho(unc, |est - hint|)         = -0.12   (consensus pulls away from the tool)
  rho(unc, |est - cited median|) = -0.49   (divergence releases the anchor)
  control: rho(unc, true) = -0.34, rho(unc, rel) = -0.17

## 3. Stage control + the fixed-anchor reading
agent estimate (uncapped): median 14, IQR [14, 16] -> a near-constant pessimistic prior
rho(est, true) = +0.17 | rho(est, hint) = +0.67 | rho(est, cited median) = -0.05
partial rho(unc, agent MAE | true) = -0.24 (raw -0.41); within 20-40: -0.20, within 40-60: -0.26

## 4. What the signals mean (for the ticket reader)
RELIABILITY: coverage - how similar this case is to the knowledge base. UNCERTAINTY: constraint - how much the known outcomes of those similar cases disagree (0 = unanimous, 1 = divergent). Neither predicts model error; they describe the EVIDENCE, not the estimator. High uncertainty is NOT good: it means the evidence supports several futures and the outlook range must widen. Low uncertainty is epistemically best - even though, for THIS anchored agent, unanimous long futures expose its fixed prior (hence the negative unc-MAE correlation: anchor geometry, not virtue of ignorance).
Reading: the estimate is a fixed ~14-16 cycle anchor that follows only the hint (rho +0.67) and ignores the cited futures (rho -0.05). Where the cited futures AGREE on a long survival, the anchor is exposed (large error); where they DIVERGE, their median collapses toward the anchor and the error shrinks. The uncertainty-MAE link (-0.24 stage-controlled) is anchor geometry, not agent adaptivity: the adaptive-anchoring hypothesis is falsified by rho(unc, |est-hint|) = -0.12.

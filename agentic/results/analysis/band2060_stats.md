# The 20-60 band (n=41, agent answered 41)

## Is agent+tool conservative here? (signed error, cycles)
      tool: bias -19.1 | median -19.9 | q05 -26.9 | q95  -9.6 | under-pred 100% | over-pred   0% | dangerous(>+20) 0
agent+tool: bias -18.3 | median -17.0 | q05 -40.0 | q95  -6.0 | under-pred 100% | over-pred   0% | dangerous(>+20) 0

## Signals vs MAE inside the band (Spearman)
rho(rel, agent MAE) = -0.00   rho(unc, agent MAE) = -0.39
rho(rel, tool  MAE) = +0.14   rho(unc, tool  MAE) = -0.14
rho(rel, unc)       = -0.10

# The >60 range (n=31: 60-124 n=8, cap n=23)
## Type of error (signed, cycles)
      tool >60 all: bias  -13.8 | median   -8.0 | q05  -36.6 | q95   -2.5 | under 100% | over   0% | dangerous(>+20) 0
      tool  60-124: bias  -29.4 | median  -32.6 | q05  -43.3 | q95   -9.3 | under 100% | over   0% | dangerous(>+20) 0
      tool     cap: bias   -8.4 | median   -6.6 | q05  -14.5 | q95   -2.5 | under 100% | over   0% | dangerous(>+20) 0
agent+tool >60 all: bias  -41.9 | median  -34.0 | q05  -93.8 | q95   -6.2 | under 100% | over   0% | dangerous(>+20) 0
agent+tool  60-124: bias  -47.1 | median  -50.0 | q05  -54.7 | q95  -33.4 | under 100% | over   0% | dangerous(>+20) 0
agent+tool     cap: bias  -40.0 | median  -32.0 | q05 -104.0 | q95   -4.8 | under 100% | over   0% | dangerous(>+20) 0

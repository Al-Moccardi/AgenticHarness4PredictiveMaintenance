# MAE analysis (final_prognostic, 89 anomalies)

## Overall MAE per arm
      CNN-GRU tool:  15.6  (answered 89/89)
      agent + tool:  22.8  (answered 83/89)
   agent (no tool):  36.3  (answered 82/89)
  precedent median:  41.2  (answered 89/89)

## MAE by true-RUL band x arm
   band   n      CNN-GRU tool      agent + tool   agent (no tool)  precedent median
    <20  17              10.6               3.2              51.4               8.5
  20-60  41              19.1              18.3              26.7              25.4
 60-124   8              29.4              47.1              32.9              74.7
    cap  23               8.4              40.0              44.5              82.1

## The rescue, quantified (corrupted tool: hint==0, n=52 | clean hint, n=37)
corrupted-hint subset: tool MAE  20.4  |  agent MAE  20.4  |  paired mean delta  +0.5
clean-hint subset: tool MAE   9.0  |  agent MAE  26.7  |  paired mean delta +17.4

## Signal quadrants (reliability median-split x uncertainty 0.35-split)
high-rel x low-unc  n=14  tool  21.7  agent  26.5
high-rel x high-unc n=32  tool  16.4  agent  19.2
 low-rel x low-unc  n= 8  tool  15.5  agent  22.9
 low-rel x high-unc n=35  tool  12.5  agent  25.0

## Per-unit MAE (heterogeneity)
unit  65 (n=50): tool  21.6  agent  25.4
unit 103 (n= 4): tool   6.2  agent   8.3
unit 110 (n= 3): tool   8.0  agent  40.7
unit 131 (n= 1): tool   7.8  agent   nan
unit 135 (n= 9): tool   5.4  agent  33.9
unit 209 (n= 4): tool   3.6  agent  30.2
unit 222 (n= 1): tool   5.5  agent   3.0
unit 245 (n=17): tool  10.7  agent   9.8

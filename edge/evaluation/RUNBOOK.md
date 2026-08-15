# RUNBOOK — edge evaluation (any machine)

This folder now lives INSIDE the edge tier: it evaluates the edge's outputs.
Run everything from the `edge/` root (one level up), so the script defaults
find `data/` (train KB, test files) and `results/` (your Jetson outputs):

    cd edge/
    python3 evaluation/scripts/eval_iforest.py     # etc.

The shipped reference tables and figures (computed on the real 89-event
Jetson run) are preserved under `evaluation/results/`.

Interprets nothing here — that is `README.md`'s job. This is the order of
execution. Inputs expected alongside (as in this repo): the train KB
`anomalies_multimodal.csv`, the edge outputs
(`test_FD002_with_interpretations.csv`, `test_anomalies.csv`,
`edge_stats.csv`, `telemetry.jsonl`), `test_FD002.txt`, `RUL_FD002.txt`.

    pip3 install numpy pandas scikit-learn scipy matplotlib

## 1. Detector effectiveness (train fleet, RUL-anchored)
    python3 evaluation/scripts/eval_iforest.py            # AUC vs baselines, lead time, cumulative
    python3 evaluation/scripts/eval_iforest_extra.py      # sensor usage, seeds, operating curve

## 2. SLM interpretation quality (the 89 real Jetson interpretations)
    python3 evaluation/scripts/eval_slm.py                # grounding, faithfulness, gravity-vs-RUL
    python3 evaluation/scripts/eval_slm_extra.py          # diversity, actions, history use

## 3. Hardware statistics
    python3 evaluation/scripts/eval_hw.py

## 4. Figures 1–9 (vector PDF + 300-dpi PNG)
    python3 evaluation/scripts/plot_paper.py              # figs 1–6 (+ anchoring fig if CSV found)
    python3 evaluation/scripts/plot_extra.py              # figs 7–9

Outputs land under results/<study>/ (summary.csv + summary.md each) and
results/figures/paper/. Optional test-split variant of the detector study:
    python3 evaluation/scripts/eval_iforest.py --csv <test_anomalies.csv> \
        --outdir results/if_eval_test --label "test (censored)"
Read `README.md` before quoting any number: it lists the three detector
traps and the SLM claims that must NOT be made.

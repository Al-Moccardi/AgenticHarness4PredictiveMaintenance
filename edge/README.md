# A-RAD Edge Tier

Frozen **Isolation-Forest anomaly detection** + **SLM natural-language
interpretation** for the NVIDIA **Jetson Orin Nano (8 GB)**, with a streaming
**collection simulator** and full **hardware / pipeline statistics**.

This is the edge tier of the A-RAD predictive-maintenance pipeline. It takes the
official CMAPSS **FD002 test set** as an unknown stream and produces, per detected
anomaly, a diagnostic interpretation in the exact schema of the training
knowledge base — ready to become a query for the (separate) coordinator tier.

```
thing (sensors) ─▶ EDGE (this package: detect + interpret) ─▶ coordinator ─▶ cloud
```

---

## Project layout

```
arad-edge/
├── arad_edge/              the package (one responsibility per module)
│   ├── detector.py         frozen IF bundle: fit / apply / KB-schema export
│   ├── interpreter.py      SLM interpretation (batch + live follow mode)
│   ├── collector.py        collection simulator (test_FD002 → growing CSV)
│   ├── daemon.py           online detection daemon (tails collector, queues)
│   ├── telemetry.py        CPU/RAM/GPU/temp/power sampler (Jetson-aware)
│   ├── stats.py            aggregate logs → report + figures
│   ├── sampling.py         stratified reference-unit picker
│   ├── progress.py         tqdm-or-fallback progress bars
│   ├── paths.py            central project paths
│   └── __main__.py         `python -m arad_edge <command>`
├── models/
│   ├── edge_bundle.joblib  the frozen detector (scaler+centroids+6 IFs+cuts)
│   └── centroids.json      the six training centroids (+scaler stats)
├── data/
│   ├── test_FD002.txt      official NASA FD002 test stream (259 units)
│   ├── RUL_FD002.txt       official RUL anchors
│   └── anomalies_multimodal.csv   training KB (schema reference + few-shot src)
├── config/fewshot_examples.json   2 real TRAIN-unit interpretations
├── notebooks/ARAD_Edge_Model_Training.ipynb   fits & exports the bundle
├── scripts/run_reference.py       sample → detect → interpret → stats
├── scripts/run_stream.py          live collection-simulation demo
├── tests/smoke.py                 20-check offline guard suite
├── expected/test_anomalies.csv    reference detection output (verification)
├── results/                       ← all outputs land here
├── Makefile · pyproject.toml · requirements.txt
```

---

## Quick start (Jetson Orin Nano)

```bash
sudo apt-get update && sudo apt-get install -y python3-pip
pip3 install -r requirements.txt           # or: pip3 install -e .[all]
sudo nvpmodel -m 0 && sudo jetson_clocks   # stable clocks (recommended)

# LLM backend (default): llama.cpp — full build/serve guide in
# docs/LLAMACPP_JETSON.md. Short version:
#   build llama.cpp with CUDA, download Llama-3.2-3B-Instruct Q4_K_M GGUF,
#   run:  llama-server -m <gguf> --port 8080 -c 4096 -ngl 99
# (Ollama still works: --backend ollama)

ARAD_SMOKE_FAST=1 python3 -m arad_edge smoke   # 20 checks, offline
```

The **model bundle is self-healing**: if `models/edge_bundle.joblib` was
written under a different numpy/sklearn than yours, the first load detects it
and deterministically rebuilds the identical model from
`data/anomalies_multimodal.csv` (seed 42), one time, automatically.

### Run on a sample of reference units (recommended first)

Pick representative units, then detect + interpret + collect statistics — with a
live progress bar:

```bash
python3 -m arad_edge sample --n 8 --seed 1        # see which units & why
python3 scripts/run_reference.py --n 8 --seed 1   # batch path (backend: llamacpp)
# explicit units, or the full test set:
python3 scripts/run_reference.py --units 24 82 177
python3 scripts/run_reference.py --all
```

`sample` is stratified by each unit's anomaly burden (spanning few→many) and
prefers units with history, so a small sample still exercises the temporal
context. It is deterministic under `--seed`.

### Streaming (collection-simulation) demo

Replays the test set as a *gradually populating* CSV and runs detection +
interpretation live, measuring queue depth and end-to-end staleness:

```bash
python3 scripts/run_stream.py --n 5 --seed 1 --tick 0.5
```
`--tick` is seconds per fleet cycle — the offered-load knob.

### Make targets

```bash
make smoke              # guard suite
make sample N=8 SEED=1  # reference sample
make run N=8            # scripts/run_reference.py
make stream N=5         # scripts/run_stream.py
make clean              # wipe results/
```

---

## Outputs (in `results/`)

| file | contents |
|---|---|
| `test_anomalies.csv` / `detections_stream.csv` | detections, **exact `anomalies_multimodal.csv` schema** |
| `test_FD002_with_interpretations.csv` | the above **+ `interpretation`** column (the deliverable) |
| `interpretations/unit_*.jsonl` | per-anomaly records (resumable) with full per-call metrics |
| `edge_stats_report.md` + `edge_stats.csv` + `figures/` | speed, tokens, latency, queue, RAM/GPU/power, energy |

**Statistics collected:** detection ms/row and rows/s capacity; interpretation
wall latency p50/p95/p99; prompt/generated token counts; prefill & decode tok/s
(from Ollama's own metrics); queue wait and arrival→done staleness (streaming);
CPU/RAM peak, temperature, power and integrated energy (Jetson INA3221), plus
energy per interpretation.

---

## What is "frozen", and the leakage stance

The detector is fit **once on the training fleet** and applied to the unseen test
units by **pure inference** — assign to the nearest of six frozen centroids,
score with the per-cluster Isolation Forest, threshold at the forest's own fitted
cut. Verified against the training KB: cluster assignment reproduced (2
boundary-tie rows in 53,759), **anomaly labels 100 %**, rule text **99.94 %
byte-equal**. The interpreter prompt never contains RUL — only the rule text,
cumulative counters, and up to three *previous* in-unit anomalies (strictly
causal); few-shot examples come from TRAIN units only. All of this is enforced by
`tests/smoke.py` (20/20).

**Censoring note for the paper:** the official test trajectories stop before
failure, so the test anomaly rate is ~3.1 % (vs the 10 % training contamination,
which is frozen, not re-fit). Fine for the streaming/diagnostic demonstration and
all comparative metrics; note it wherever absolute prognostic error is reported.

## Rebuilding the model

`notebooks/ARAD_Edge_Model_Training.ipynb` fits and exports
`models/edge_bundle.joblib` and `models/centroids.json` from
`data/anomalies_multimodal.csv` and the six training centroids; an executed copy
with outputs is included. With `random_state=42` the forests are deterministic,
so this re-instantiates the exact KB detector (labels 100 %).

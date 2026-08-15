# RUNBOOK — edge tier (NVIDIA Jetson)

Step-by-step. Details and design: `README.md`; llama.cpp specifics:
`docs/LLAMACPP_JETSON.md`.

## 0. One-time setup
    sudo apt-get update && sudo apt-get install -y python3-pip build-essential cmake git curl
    pip3 install -r requirements.txt
    sudo nvpmodel -m 0 && sudo jetson_clocks          # stable clocks

## 1. Build + serve the SLM (llama.cpp, one time)
    export PATH=/usr/local/cuda/bin:$PATH && nvcc --version
    git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
    cd ~/llama.cpp && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87
    cmake --build build --config Release -j$(nproc)
    pip3 install -U "huggingface_hub[cli]" && mkdir -p ~/models
    huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF \
        Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir ~/models
    nohup ~/llama.cpp/build/bin/llama-server -m ~/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
        --host 0.0.0.0 --port 8080 -c 4096 -ngl 99 -t $(nproc) > ~/llama-server.log 2>&1 &
    sleep 8 && grep -i offloaded ~/llama-server.log   # want NN/NN layers on GPU
    curl -s http://localhost:8080/health

## 2. Guard suite (offline, fast mode)
    ARAD_SMOKE_FAST=1 python3 -m arad_edge smoke      # expect 20/20
    # note: on a machine with a different numpy/sklearn the bundle
    # self-heals (one-time deterministic refit, ~1 min) — this is normal.

## 3. Pick reference units, pilot, full batch run
    python3 -m arad_edge sample --n 10 --seed 1 --out results/units.txt
    python3 -m arad_edge interpret --backend llamacpp \
        --detections results/test_anomalies.csv --limit 3        # 3-anomaly pilot
    python3 scripts/run_reference.py \
        --units $(cat results/units.txt | tr ',' ' ') --backend llamacpp
    # whole official test set instead:  --all   (~1,062 anomalies)

## 4. Streaming demo (optional)
    python3 scripts/run_stream.py --units $(cat results/units.txt | tr ',' ' ') \
        --tick 0.5 --backend llamacpp

## 5. Outputs (results/)
    test_anomalies.csv                       detections, exact KB schema
    test_FD002_with_interpretations.csv      + interpretation  ← THE HAND-OFF
    interpretations/unit_*.jsonl             per-anomaly records (resumable)
    edge_stats_report.md + figures/          latency, tokens, telemetry
Copy the hand-off CSV to  ../agentic/queries/  and to the evaluation inputs.

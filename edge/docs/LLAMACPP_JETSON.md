# llama.cpp on the Jetson Orin Nano — build, serve, and run the pipeline

The pipeline's default backend is now **llama.cpp** (`--backend llamacpp`),
talking to `llama-server`'s OpenAI-compatible endpoint at
`http://localhost:8080/v1/chat/completions`. Ollama remains available with
`--backend ollama`.

---

## 1. Build llama.cpp with CUDA (one time, ~15–25 min)

```bash
sudo apt-get update && sudo apt-get install -y build-essential cmake git curl
export PATH=/usr/local/cuda/bin:$PATH        # JetPack's CUDA toolkit
nvcc --version                                # must print a CUDA version

git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cd ~/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87   # 87 = Orin
cmake --build build --config Release -j$(nproc)
```

`build/bin/llama-server` is the binary you'll run. If `nvcc` is missing, install
the CUDA toolkit for your JetPack (`sudo apt-get install -y cuda-toolkit-*`) or
check `/usr/local/cuda*` exists.

## 2. Get the model (Llama-3.2-3B-Instruct, Q4_K_M ≈ 2.0 GB)

Same model family as the knowledge-base generation, so the test-side
interpretation style stays consistent with the KB.

```bash
pip3 install -U "huggingface_hub[cli]"
mkdir -p ~/models
huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF \
    Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir ~/models
```

Alternatives: any `Llama-3.2-3B-Instruct` GGUF at Q4_K_M works (e.g. the
`unsloth/Llama-3.2-3B-Instruct-GGUF` repo). Recent llama.cpp can also fetch
directly: `llama-server -hf bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M ...`
If the HF repo asks you to accept the Llama 3.2 license, do it once in the
browser and re-run the download.

## 3. Serve

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks      # stable clocks (recommended)

~/llama.cpp/build/bin/llama-server \
    -m ~/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    --host 0.0.0.0 --port 8080 \
    -c 4096 -ngl 99 -t $(nproc)
```

Flag meanings, mapped to the pipeline: `-c 4096` = context window (matches the
interpreter's `--num-ctx 4096`); `-ngl 99` = offload all layers to the GPU;
`--port 8080` = the backend's default host. Keep it running under `tmux` or:

```bash
nohup ~/llama.cpp/build/bin/llama-server -m ~/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    --host 0.0.0.0 --port 8080 -c 4096 -ngl 99 -t $(nproc) > ~/llama-server.log 2>&1 &
```

In the startup log, confirm the GPU is actually used — you want a line like
`offloaded 29/29 layers to GPU`. If it says 0 layers, the CUDA build didn't
take (rebuild step 1, check `nvcc`).

## 4. Verify (both must succeed)

```bash
curl -s http://localhost:8080/health
curl -s http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"reply with the single word: ready"}],"max_tokens":8}'
```

## 5. Run the pipeline

```bash
cd ~/Desktop/arad-edge && source .venv/bin/activate

# wipe any dryrun placeholders from earlier attempts
rm -rf results/interpretations results/test_FD002_with_interpretations.csv results/interp_calls.jsonl

# 3-anomaly pilot first (should take ~1 min and produce varied, real text)
python3 -m arad_edge interpret --backend llamacpp --detections results/test_anomalies.csv --limit 3

# the full run on your sampled units (resumable; progress bar)
python3 scripts/run_reference.py --units $(cat results/units.txt | tr ',' ' ') --backend llamacpp

# streaming demo, same backend
python3 scripts/run_stream.py --units $(cat results/units.txt | tr ',' ' ') --tick 0.5 --backend llamacpp
```

You'll know it's real (not dryrun) when: the progress bar advances in
**seconds per anomaly**, `results/edge_stats_report.md` shows
`model(s): llama-3.2-3b-instruct` with nonzero median latency and measured
decode tok/s, and the interpretation text varies per anomaly.

## 6. Troubleshooting

- **Out of memory / model falls to CPU** — lower the context: serve with
  `-c 2048` and run the interpreter with `--num-ctx 2048`; or use the 1B model
  (`bartowski/Llama-3.2-1B-Instruct-GGUF`, Q4_K_M ≈ 0.8 GB) and pass
  `--model llama-3.2-1b-instruct`. Close other GPU users; watch
  `sudo tegrastats`.
- **Connection refused from the pipeline** — the server isn't up or is on a
  different port; re-check step 4, or pass `--host http://<ip>:<port>`.
- **Slow (<5 tok/s decode)** — you're on CPU; see the `offloaded ... layers`
  note in step 3.
- **Build fails on `nvcc`** — `export PATH=/usr/local/cuda/bin:$PATH` and
  ensure the JetPack CUDA toolkit is installed.

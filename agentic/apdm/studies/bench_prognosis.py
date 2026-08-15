"""SLM benchmark for the prognostic decision ladder (v8).

Runs each model as a subprocess over the eight arms (row-level resume lives
inside apdm.prognosis via prog_rows_*.jsonl, so an interrupted model
continues where it stopped; a model whose predictions CSV already exists is
skipped entirely). The two no-LLM baselines are deterministic and identical
across models; report_prognosis de-duplicates them.

  python -m apdm.bench_prognosis --models llama3.2:1b llama3.2:3b \\
      qwen2.5:3b phi3 mistral llama3 --sample-seed 1
  python -m apdm.report_prognosis --sample-seed 1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from .prognosis import ARMS

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["llama3.2:1b", "llama3.2:3b", "qwen2.5:3b",
                             "phi3", "mistral", "llama3"])
    ap.add_argument("--backend", default="ollama")
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--per-bucket", type=int, default=30)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--sample-seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--query", default="auto", choices=["auto", "summary"])
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(exist_ok=True)

    for model in a.models:
        tag = (f"{model.replace(':', '_').replace('/', '_')}"
               f"_seed{a.sample_seed}")
        if (out / f"prog_predictions_{tag}.csv").exists():
            print(f"[bench-prog] {model}: predictions exist, skipping")
            continue
        print(f"\n{'=' * 70}\n[bench-prog] model = {model}\n{'=' * 70}")
        cmd = [sys.executable, "-m", "apdm.prognosis",
               "--backend", a.backend, "--model", model,
               "--arms", *a.arms, "--per-bucket", str(a.per_bucket),
               "--sample-seed", str(a.sample_seed),
               "--k", str(a.k), "--max-steps", str(a.max_steps),
               "--query", a.query, "--out", a.out]
        if a.full:
            cmd.append("--full")
        if a.limit:
            cmd += ["--limit", str(a.limit)]
        t0 = time.time()
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode:
            print(f"[bench-prog] {model} FAILED (rc={r.returncode}); "
                  f"continuing with the next model")
            continue
        print(f"[bench-prog] {model} done in {time.time() - t0:.0f}s")

    print("\n[bench-prog] sweep complete; aggregate with:\n"
          f"  python -m apdm.report_prognosis --sample-seed {a.sample_seed}")


if __name__ == "__main__":
    main()

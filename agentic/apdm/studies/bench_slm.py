"""SLM benchmark for agentic diagnosis -- successor to TIOT's Fig. 7.

That figure ranked edge SLMs by size and community adoption. This one ranks
them by AGENTIC CAPABILITY on the diagnostic task: can the model hold a tool
protocol, and does the interpretation KB help it?

Reported per (model, arm):
  protocol_compliance  1 - parse_fail_rate, with Wilson CI. For an edge
                       deployment this is the gating property: a model that
                       cannot emit a parseable diagnosis is unusable no
                       matter how accurate its prose sounds.
  phenotype_accuracy   vs future-derived gold, and vs the two non-LLM twins
                       (nearest-centroid rule, logistic) on identical
                       snapshots -- computed only over parsed answers, with
                       n_parsed shown so the conditioning stays visible.
  signature_p_at_3     (sensor, direction) precision vs the terminal gold
  faithfulness         supported / contradicted / unverifiable claim rates
  cost                 llm calls, tool calls, seconds per diagnosis
  KB_delta             D2_agentic minus D2_norag on each metric: the direct
                       test of the TIOT premise, per model.

Models are Ollama tags; approximate quantised footprints are printed for the
edge-deployability column (verify against `ollama list` before publishing).

  python -m apdm.bench_slm --models llama3.2:1b llama3.2:3b qwen2.5:3b \\
      phi3 mistral llama3 --per-bucket 30 --sample-seed 1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from .diagnosis import ARMS
from .metrics import wilson

ROOT = Path(__file__).resolve().parent.parent

# Approximate Q4 footprints (GB) for the edge-deployability column.
FOOTPRINT = {"llama3.2:1b": 1.3, "gemma2:2b": 1.6, "qwen2.5:3b": 1.9,
             "llama3.2:3b": 2.0, "phi3": 2.2, "mistral": 4.1,
             "qwen2.5:7b": 4.7, "llama3": 4.7, "llama3.1": 4.7}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["llama3.2:1b", "llama3.2:3b", "qwen2.5:3b",
                             "phi3", "mistral", "llama3"])
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--backend", default="ollama")
    ap.add_argument("--per-bucket", type=int, default=30)
    ap.add_argument("--sample-seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=5)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(exist_ok=True)

    for model in a.models:
        tag = f"{model.replace(':', '_')}_seed{a.sample_seed}"
        if (out / f"diag_predictions_{tag}.csv").exists():
            print(f"[bench] {model}: predictions exist, skipping run")
            continue
        print(f"\n{'=' * 70}\n[bench] model = {model}\n{'=' * 70}")
        cmd = [sys.executable, "-m", "apdm.diagnosis", "--backend", a.backend,
               "--model", model, "--arms", *a.arms,
               "--per-bucket", str(a.per_bucket),
               "--sample-seed", str(a.sample_seed),
               "--max-steps", str(a.max_steps), "--out", a.out]
        if a.limit:
            cmd += ["--limit", str(a.limit)]
        t0 = time.time()
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode:
            print(f"[bench] {model} FAILED (rc={r.returncode}); continuing")
            continue
        print(f"[bench] {model} done in {time.time() - t0:.0f}s")

    frames = []
    for model in a.models:
        f = out / f"diag_predictions_{model.replace(':', '_')}_seed{a.sample_seed}.csv"
        if f.exists():
            frames.append(pd.read_csv(f))
    if not frames:
        raise SystemExit("[bench] no prediction files produced")
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out / f"bench_slm_seed{a.sample_seed}.csv", index=False)

    rows = []
    for (model, arm), g in df.groupby(["model", "arm"]):
        ok = g[~g.parse_failed]
        lo, hi = wilson(len(ok), len(g))
        row = {"model": model, "gb": FOOTPRINT.get(model, float("nan")),
               "arm": arm, "n": len(g), "n_parsed": len(ok),
               "protocol_compliance": round(len(ok) / len(g), 3),
               "pc_ci_lo": round(lo, 3), "pc_ci_hi": round(hi, 3),
               "sec_per_diag": round(float(g.seconds.mean()), 2),
               "sim_edge_s": (round(float(g.sim_edge_s.mean()), 2)
                              if "sim_edge_s" in g else float("nan")),
               "sim_energy_j": (round(float(g.sim_energy_j.mean()), 1)
                                if "sim_energy_j" in g else float("nan")),
               "llm_calls": round(float(g.llm_calls.mean()), 2),
               "tool_calls": round(float(g.tool_calls.mean()), 2)}
        if len(ok):
            row.update({
                "phenotype_acc": round(float(ok.correct.mean()), 3),
                "sig_p_at_3": round(float(ok.sig_p_at_3.mean()), 3),
                "faith_supported": round(float(ok.supported.mean()), 3),
                "faith_contradicted": round(float(ok.contradicted.mean()), 3),
                "faith_unverifiable": round(float(ok.unverifiable.mean()), 3),
                "twin_rule_acc": round(float((ok.twin_rule ==
                                              ok.gold_phenotype).mean()), 3),
                "twin_lr_acc": round(float((ok.twin_lr ==
                                            ok.gold_phenotype).mean()), 3)})
        rows.append(row)
    tab = pd.DataFrame(rows).sort_values(["gb", "model", "arm"])
    tab.to_csv(out / f"bench_slm_summary_seed{a.sample_seed}.csv", index=False)

    print("\n=== SLM agentic-capability benchmark ===")
    cols = ["model", "gb", "arm", "n_parsed", "protocol_compliance",
            "phenotype_acc", "twin_lr_acc", "sig_p_at_3", "faith_supported",
            "faith_contradicted", "sec_per_diag", "sim_edge_s",
            "sim_energy_j"]
    print(tab[[c for c in cols if c in tab.columns]].to_string(index=False))

    print("\n=== KB delta (D2_agentic - D2_norag): the TIOT premise, per model ===")
    piv = tab.set_index(["model", "arm"])
    for model in tab.model.unique():
        try:
            ag = piv.loc[(model, "D2_agentic")]
            nr = piv.loc[(model, "D2_norag")]
        except KeyError:
            continue
        d = lambda k: (float(ag[k]) - float(nr[k])
                       if k in ag and ag[k] == ag[k] and nr[k] == nr[k]
                       else float("nan"))
        print(f"  {model:<14} compliance {d('protocol_compliance'):+.3f}  "
              f"accuracy {d('phenotype_acc'):+.3f}  "
              f"P@3 {d('sig_p_at_3'):+.3f}  "
              f"faithfulness {d('faith_supported'):+.3f}")
    print(f"\n[bench] wrote bench_slm_summary_seed{a.sample_seed}.csv")


if __name__ == "__main__":
    main()

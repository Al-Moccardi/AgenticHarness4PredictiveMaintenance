"""Run the LLM arms on the sampled subset and build the joint report.

  python -m apdm.run_llm --backend ollama --model llama3 \\
      --arms P0_raw P1_featurized P2_agentic P3_hybrid \\
      --per-bucket 30 --sample-seed 1
  python -m apdm.report --sample-seed 1

The report pairs every LLM arm against every ML arm on the identical
snapshots: Wilcoxon signed-rank on |error|, McNemar on induced stage
correctness, plus the metric suite of metrics.py. Parse failures are
reported as a first-class rate and excluded pairwise (both members of a
pair must have a prediction), with n shown for every test.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .agent import ARMS, run_arm
from ..data import FD002
from ..llm import get_backend
from .metrics import (classification_metrics, mcnemar, regression_metrics,
                      stage, wilcoxon_paired, wilson)
from .ml_models import ML_ARMS, train_all

ROOT = Path(__file__).resolve().parent.parent


def run_llm() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="dryrun",
                    choices=["dryrun", "ollama", "openai"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--per-bucket", type=int, default=30)
    ap.add_argument("--sample-seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0,
                    help="use only the first N snapshots of the canonical "
                         "subset (cheap pilot runs)")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    model = a.model or ("gpt-4o-mini" if a.backend == "openai" else "llama3")
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)

    ds = FD002(seed=a.split_seed)
    bundle = train_all(ds, cache=ROOT / "cache" / "ml_v2.pkl")
    xgb_mae = _xgb_full_mae(out)
    # Pairing contract: the ML runner defines the canonical subset for this
    # seed; consume it verbatim so every LLM row has ML twins. Fall back to
    # sampling only if run_ml has not been run yet.
    sub_csv = out / f"ml_predictions_subset_seed{a.sample_seed}.csv"
    if sub_csv.exists():
        from ..data import Snapshot
        sd = pd.read_csv(sub_csv)
        snaps = [Snapshot(int(r.unit), int(r.cycle), int(r.true_rul))
                 for r in sd.itertuples()]
        if a.limit:
            snaps = snaps[: a.limit]
        print(f"[llm] using canonical subset from {sub_csv.name} "
              f"({len(snaps)} snapshots)")
    else:
        snaps = ds.sample_snapshots(ds.test_units, a.per_bucket,
                                    seed=a.sample_seed)
        print("[llm] WARNING: no ml subset file; sampled independently -- "
              "run apdm.run_ml first for paired comparisons")
    tag = f"{model.replace(':', '_')}_seed{a.sample_seed}"
    backend = get_backend(a.backend, model, num_ctx=a.num_ctx,
                          log_path=out / f"llm_calls_{tag}.jsonl")

    rows, traces = [], []
    for arm in a.arms:
        print(f"[llm] === arm {arm} | {model} | {len(snaps)} snapshots ===")
        t0 = time.time()
        for i, s in enumerate(snaps, 1):
            r = run_arm(arm, ds, s, backend, bundle, a.max_steps, xgb_mae)
            rows.append({
                "arm": arm, "model": model, "sample_seed": a.sample_seed,
                "unit": r.unit, "cycle": r.cycle, "true_rul": r.true_rul,
                "pred_rul": r.pred_rul, "parse_failed": r.parse_failed,
                "abs_err": (abs(r.pred_rul - r.true_rul)
                            if r.pred_rul is not None else np.nan),
                "n_llm_calls": r.n_llm_calls, "n_tool_calls": r.n_tool_calls,
                "protocol_errors": r.n_protocol_errors,
                "termination": r.termination, "seconds": round(r.seconds, 2),
                "tools_used": "|".join(r.tools_used),
                "answer": r.answer_text})
            traces.append({"arm": arm, "unit": r.unit, "cycle": r.cycle,
                           "trace": r.trace})
            if i % 15 == 0:
                sub = [x for x in rows if x["arm"] == arm]
                ok = [x for x in sub if not x["parse_failed"]]
                mae = (np.mean([x["abs_err"] for x in ok]) if ok else np.nan)
                print(f"    {i}/{len(snaps)}  parse_ok={len(ok)}/{len(sub)}"
                      f"  running MAE={mae:.1f}  "
                      f"({(time.time()-t0)/i:.1f}s/snap)")
        sub = [x for x in rows if x["arm"] == arm]
        ok = [x for x in sub if not x["parse_failed"]]
        print(f"[llm] {arm}: parse_ok {len(ok)}/{len(sub)}, "
              f"MAE {np.mean([x['abs_err'] for x in ok]):.2f}" if ok
              else f"[llm] {arm}: ALL PARSES FAILED")

    df = pd.DataFrame(rows)
    df.to_csv(out / f"llm_predictions_{tag}.csv", index=False)
    with open(out / f"llm_traces_{tag}.jsonl", "w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t, default=str) + "\n")
    print(f"[llm] wrote llm_predictions_{tag}.csv (+ traces, + call log). "
          f"Backend totals: {backend.totals()}")


def _xgb_full_mae(out: Path) -> float:
    p = out / "ml_metrics.csv"
    if p.exists():
        m = pd.read_csv(p)
        r = m[(m.arm == "xgb") & (m.scope == "full_test")]
        if len(r):
            return float(r.iloc[0]["mae"])
    return 12.4


# ---------------------------------------------------------------- report
def report() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-seed", type=int, default=1)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = ROOT / a.out

    ml = pd.read_csv(out / f"ml_predictions_subset_seed{a.sample_seed}.csv")
    llm_files = sorted(out.glob(f"llm_predictions_*_seed{a.sample_seed}.csv"))
    if not llm_files:
        raise SystemExit("no LLM prediction files for this seed")
    llm = pd.concat([pd.read_csv(f) for f in llm_files], ignore_index=True)

    key = ["unit", "cycle"]
    lines, table = [], []
    for (arm, model), g in llm.groupby(["arm", "model"]):
        merged = g.merge(ml, on=key, suffixes=("", "_ml"))
        if not len(merged):
            print(f"[report] SKIP {arm} [{model}]: no snapshot overlap with "
                  f"the ML subset (was run_llm launched without run_ml?)")
            continue
        ok = merged[~merged.parse_failed & merged.pred_rul.notna()]
        pf = 1 - len(ok) / len(merged)
        lo, hi = wilson(len(merged) - len(ok), len(merged))
        row = {"arm": f"{arm} [{model}]", "n": len(merged),
               "parse_fail_rate": round(pf, 3),
               "pf_ci": f"[{lo:.2f},{hi:.2f}]"}
        if len(ok):
            m = regression_metrics(ok.pred_rul, ok.true_rul)
            c = classification_metrics(ok.pred_rul, ok.true_rul)
            row.update({k: round(m[k], 2) for k in
                        ("mae", "rmse", "r2", "bias", "s_score")})
            row["stage_acc"] = round(c["stage_acc"], 3)
            row["fail30_recall"] = round(c["fail30_recall"], 3)
            row["mean_llm_calls"] = round(float(g.n_llm_calls.mean()), 2)
            row["mean_s_per_snap"] = round(float(g.seconds.mean()), 2)
        table.append(row)

        for ml_arm in ML_ARMS:
            col = f"pred_{ml_arm}"
            if col not in ok or len(ok) < 8:
                continue
            w = wilcoxon_paired(np.abs(ok.pred_rul - ok.true_rul),
                                np.abs(ok[col] - ok.true_rul))
            mc = mcnemar(
                [stage(p) == stage(t) for p, t in zip(ok.pred_rul, ok.true_rul)],
                [stage(p) == stage(t) for p, t in zip(ok[col], ok.true_rul)])
            lines.append({"llm_arm": arm, "model": model, "ml_arm": ml_arm,
                          "n_paired": len(ok),
                          "wilcoxon_p": round(w["p_value"], 4),
                          "median_abs_err_delta": round(w["median_delta"], 2),
                          "mcnemar_p": round(mc["p_value"], 4),
                          "llm_stage_wins": mc["n10"],
                          "ml_stage_wins": mc["n01"]})

    # ML reference rows on the same subset
    truth = ml["true_rul"].to_numpy(float)
    for ml_arm in ML_ARMS:
        m = regression_metrics(ml[f"pred_{ml_arm}"], truth)
        c = classification_metrics(ml[f"pred_{ml_arm}"], truth)
        table.append({"arm": f"[ML] {ml_arm}", "n": len(ml),
                      "parse_fail_rate": 0.0, "pf_ci": "-",
                      **{k: round(m[k], 2) for k in
                         ("mae", "rmse", "r2", "bias", "s_score")},
                      "stage_acc": round(c["stage_acc"], 3),
                      "fail30_recall": round(c["fail30_recall"], 3),
                      "mean_llm_calls": 0, "mean_s_per_snap": 0})

    t = pd.DataFrame(table)
    p = pd.DataFrame(lines)
    t.to_csv(out / f"report_table_seed{a.sample_seed}.csv", index=False)
    p.to_csv(out / f"report_paired_seed{a.sample_seed}.csv", index=False)
    print("\n=== Joint table (matched subset, seed "
          f"{a.sample_seed}; ML also has full-test metrics in "
          "ml_metrics.csv) ===")
    print(t.to_string(index=False))
    print("\n=== Paired tests (LLM vs ML, identical snapshots) ===")
    print(p.to_string(index=False) if len(p) else "(none)")


if __name__ == "__main__":
    import sys
    if sys.argv[0].endswith("run_llm.py") or "run_llm" in sys.argv[0]:
        run_llm()
    else:
        report()

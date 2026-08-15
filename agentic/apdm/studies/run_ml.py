"""Evaluate the ML arms on (a) the full test universe and (b) the sampled
subset the LLM arms will see, and write per-snapshot predictions so every
later comparison is paired.

  python -m apdm.run_ml --per-bucket 30 --sample-seeds 1 2 3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import FD002
from .features import feature_matrix
from .metrics import classification_metrics, regression_metrics
from .ml_models import ML_ARMS, train_all

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-bucket", type=int, default=30)
    ap.add_argument("--sample-seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)

    ds = FD002(seed=a.split_seed)
    bundle = train_all(ds, cache=ROOT / "cache" / "ml_v2.pkl")

    # ---------------- full test universe ---------------------------------
    full = ds.snapshots(ds.test_units)
    t0 = time.time()
    Xf = feature_matrix(ds, full)
    print(f"[ml] featurised {len(full)} test snapshots in {time.time()-t0:.0f}s")
    truth = np.array([s.rul for s in full], float)

    rows, preds_full = [], {}
    for arm in ML_ARMS:
        p = bundle.predict(arm, ds, full, X=Xf)
        preds_full[arm] = p
        m = {**regression_metrics(p, truth), **classification_metrics(p, truth)}
        rows.append({"arm": arm, "scope": "full_test", **m})
        print(f"[ml] {arm:>10} full: MAE {m['mae']:6.2f}  RMSE {m['rmse']:6.2f} "
              f" R2 {m['r2']:.3f}  S {m['s_score']:>10.0f}")

    pd.DataFrame({
        "unit": [s.unit for s in full], "cycle": [s.cycle for s in full],
        "true_rul": truth, **{f"pred_{k}": v for k, v in preds_full.items()},
    }).to_csv(out / "ml_predictions_full.csv", index=False)

    # ---------------- LLM-matched subsets --------------------------------
    for seed in a.sample_seeds:
        sub = ds.sample_snapshots(ds.test_units, a.per_bucket, seed=seed)
        Xs = feature_matrix(ds, sub)
        t = np.array([s.rul for s in sub], float)
        sub_preds = {}
        for arm in ML_ARMS:
            p = bundle.predict(arm, ds, sub, X=Xs)
            sub_preds[arm] = p
            m = {**regression_metrics(p, t), **classification_metrics(p, t)}
            rows.append({"arm": arm, "scope": f"subset_seed{seed}", **m})
        pd.DataFrame({
            "unit": [s.unit for s in sub], "cycle": [s.cycle for s in sub],
            "true_rul": t, **{f"pred_{k}": v for k, v in sub_preds.items()},
        }).to_csv(out / f"ml_predictions_subset_seed{seed}.csv", index=False)
        print(f"[ml] subset seed {seed}: {len(sub)} snapshots written")

    df = pd.DataFrame(rows)
    df.to_csv(out / "ml_metrics.csv", index=False)
    (out / "ml_metrics.json").write_text(json.dumps(rows, indent=1))
    print(f"\n[ml] wrote {out}/ml_metrics.csv and per-snapshot predictions")

    anchor = df[df.scope == "full_test"][
        ["arm", "mse", "mae", "rmse", "r2", "s_score"]]
    print("\nAnchor vs Moccardi et al. Table 1 "
          "(their BiLSTM: MSE 348.98, MAE 13.97, RMSE 18.68, R2 0.83, S 99164):")
    print(anchor.round(2).to_string(index=False))


if __name__ == "__main__":
    main()

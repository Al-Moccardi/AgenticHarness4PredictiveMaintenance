"""Establish the event-forecasting bars BEFORE any LLM runs.

  python -m apdm.run_events [--max-per-unit 40] [--test-sample 1200]

Outputs:
  results/events_test.csv          gold + twin forecasts, one row per event
  results/events_twins.json        interval + severity metrics of the twins
  results/gravity_audit.json       do TIOT gravity scores track outcomes?
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .data import FD002
from .events import (EventTwins, build_event_dataset, gravity_audit,
                     interval_metrics, severity_metrics)

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-unit", type=int, default=40)
    ap.add_argument("--test-sample", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(exist_ok=True)

    ds = FD002(seed=42)

    tr_ev, tr_stats = build_event_dataset(ds, ds.train_units,
                                          max_per_unit=a.max_per_unit,
                                          seed=a.seed)
    te_ev, te_stats = build_event_dataset(ds, ds.test_units,
                                          max_per_unit=0, seed=a.seed)
    rng = np.random.default_rng(a.seed)
    if len(te_ev) > a.test_sample:
        idx = rng.choice(len(te_ev), size=a.test_sample, replace=False)
        te_ev = [te_ev[int(i)] for i in sorted(idx)]

    t_tr = np.array([e.t_onset for e in tr_ev])
    print(f"[events] train events {tr_stats} | onset gap: "
          f"median {np.median(t_tr):.0f}, q10 {np.quantile(t_tr, .1):.0f}, "
          f"q90 {np.quantile(t_tr, .9):.0f}, max {t_tr.max():.0f}")
    print(f"[events] test  events {te_stats} -> evaluating on {len(te_ev)}")
    sev_tr = pd.Series([e.severity for e in tr_ev])
    print(f"[events] train severity distribution: "
          f"{sev_tr.value_counts().sort_index().to_dict()} "
          f"(majority {sev_tr.value_counts(normalize=True).max():.3f})")

    cache = ROOT / "cache" / "event_twins.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            twins = pickle.load(f)
        print("[events] twins loaded from cache")
    else:
        print("[events] fitting twins (features + 3 HistGB fits)...")
        twins = EventTwins(ds, tr_ev, seed=a.seed)
        with open(cache, "wb") as f:
            pickle.dump(twins, f)

    df = twins.predict(ds, te_ev)
    df.to_csv(out / "events_test.csv", index=False)

    res = {}
    for tag, lo, hi in (("T1_global", df.t1_lo, df.t1_hi),
                        ("T2_learned", df.t2_lo, df.t2_hi)):
        m = interval_metrics(lo, hi, df.t_onset)
        res[tag] = m
        print(f"[events] {tag:<11} coverage {m['coverage']:.3f}  "
              f"width {m['mean_width']:.1f}  winkler {m['winkler']:.1f}")
    maj = int(sev_tr.mode().iloc[0])
    for tag, pred in (("S_majority", np.full(len(df), maj)),
                      ("S1_learned", df.s1_sev)):
        m = severity_metrics(pred, df.severity)
        res[tag] = m
        print(f"[events] {tag:<11} acc {m['sev_acc']:.3f}  "
              f"±1 {m['sev_pm1_acc']:.3f}  MAE {m['sev_mae']:.2f}  "
              f"QWK {m['sev_qwk']:.3f}")

    res["train_stats"] = tr_stats
    res["test_stats"] = te_stats
    (out / "events_twins.json").write_text(json.dumps(res, indent=2))

    audit = gravity_audit(ds, ROOT / "data" / "interpretations")
    (out / "gravity_audit.json").write_text(json.dumps(audit, indent=2))
    print(f"\n[events] GRAVITY AUDIT (are TIOT's own severity opinions "
          f"tracking outcomes?): {audit}")
    print(f"[events] wrote events_test.csv, events_twins.json, "
          f"gravity_audit.json")


if __name__ == "__main__":
    main()

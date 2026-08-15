#!/usr/bin/env python3
"""eval_progression.py — scores the progression-only arm.

  python -m apdm.eval_progression --dir results\\arad3\\prog_progression
Reports: production rate, boilerplate share, rho(horizon, true RUL),
rho(horizon, cited median future), horizon MAE, and per-band tracking.
"""
import argparse, json, re
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
a = ap.parse_args()
eps = [json.loads(l) for l in
       open(Path(a.dir) / "forecast_episodes.jsonl") if l.strip()]
eps = [e for e in eps if e["arm"] == "P7_progression"]
if not eps:
    raise SystemExit("no P7_progression episodes in this dir")
rows = []
for e in eps:
    fc = e.get("forecast") or {}
    agg = ((e.get("signals") or {}).get("future_progression", {})
           .get("aggregate", {}) or {})
    rows.append({"qid": e["qid"], "true": min(float(e["true_rul"]), 125.0),
                 "hint": e.get("dl_hint"),
                 "text": fc.get("projected_progression") or "",
                 "hz": fc.get("progression_horizon"),
                 "med_fut": agg.get("median_ttf"),
                 "unc": agg.get("uncertainty"),
                 "conf": fc.get("confidence")})
d = pd.DataFrame(rows).set_index("qid")
sp = lambda x, y: d[x].rank().corr(d[y].rank())
nz = d.text.str.len() > 0
tmpl = Counter(re.sub(r"\\d+", "N", t.lower())[:120] for t in d.text[nz])
print(f"produced hypothesis: {int(nz.sum())}/{len(d)} "
      f"({nz.mean():.0%}) | with horizon: {int(d.hz.notna().sum())}/{len(d)}")
print(f"distinct templates: {len(tmpl)} | modal share: "
      f"{(max(tmpl.values())/max(int(nz.sum()),1)) if tmpl else 0:.0%}")
print(f"rho(horizon, TRUE RUL)             = {sp('hz','true'):+.2f}")
print(f"rho(horizon, cited median future)  = {sp('hz','med_fut'):+.2f}")
print(f"rho(horizon, tool hint)            = {sp('hz','hint'):+.2f}")
err = (d.hz - d.true)
print(f"horizon as a number: MAE {err.abs().mean():.1f}, "
      f"bias {err.mean():+.1f}, over>20 {int((err>20).sum())}"
      f"/{int(err.notna().sum())}")
if d.unc.notna().any():
    lo = d[d.unc <= 0.35]; hi = d[d.unc > 0.35]
    print(f"tracking by uncertainty: low-unc rho "
          f"{lo.hz.rank().corr(lo.true.rank()):+.2f} (n={len(lo)}) | "
          f"high-unc rho {hi.hz.rank().corr(hi.true.rank()):+.2f} "
          f"(n={len(hi)})")
for lab, m in (("<20", d.true < 20), ("20-60", (d.true >= 20) & (d.true < 60)),
               ("60+", d.true >= 60)):
    if m.sum() >= 3:
        print(f"  band {lab:>5s}: n={int(m.sum()):2d} horizon-MAE "
              f"{(d.hz-d.true)[m].abs().mean():6.1f}")

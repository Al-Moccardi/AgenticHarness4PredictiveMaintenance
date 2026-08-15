#!/usr/bin/env python3
"""future_progression.py — the two ticket additions of the final pipeline.

RELIABILITY   similarity of the current case to the fleet knowledge base:
              mean (and top) cosine similarity of the k retrieved
              precedents. High = the memory covers this case well; low =
              novel case, precedent-based conclusions deserve caution.
FUTURE PROGRESSION LOOKUP   for each cited precedent, its OBSERVED
              continuation from fleet memory: the next anomalies as
              (+cycle offset, gravity) relative to the matched point, and
              the cycles it survived. Aggregates: median/range
              time-to-failure and how often gravity escalated. The
              prognostic agent uses it to hypothesize THIS unit's own
              progression ("projected_progression" ticket field).

CLI:  python -m apdm.future_progression time      (measures the lookup cost)
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np

CAP = 125.0


def reliability(recs) -> dict:
    sims = [float(r["similarity"]) for r in (recs or [])
            if r.get("similarity") is not None]
    if not sims:
        return {"value": None, "top": None, "k": 0}
    return {"value": round(float(np.mean(sims)), 3),
            "top": round(float(max(sims)), 3), "k": len(sims)}


def future_progression(pf, recs, max_events: int = 4) -> dict:
    """pf: PrecedentFutures (uses pf.by_unit). Returns structured lookup."""
    cases, ttf, esc = [], [], []
    for r in recs or []:
        u, c = int(r["unit"]), int(r["cycle"])
        rows = [x for x in pf.by_unit.get(u, []) if x["cycle"] > c]
        ev = [{"plus": int(x["cycle"]) - c, "gravity": x.get("gravity")}
              for x in rows[:max_events]]
        rt = r.get("rul_then")
        rec = {"id": f"u{u}c{c}", "rul_then": rt, "n_more": len(rows),
               "events": ev}
        cases.append(rec)
        if rt is not None:
            ttf.append(min(float(rt), CAP))
        gs = [e["gravity"] for e in ev if e.get("gravity") is not None]
        if len(gs) >= 2:
            esc.append(gs[-1] > gs[0])
    agg = {"median_ttf": (round(float(np.median(ttf)), 1) if ttf else None),
           "ttf_range": ([round(min(ttf), 1), round(max(ttf), 1)]
                         if ttf else None),
           "escalating": (f"{sum(esc)}/{len(esc)}" if esc else None)}
    # PROGRESSION UNCERTAINTY: how dissimilar are the cited futures?
    #   0 = they agree (well-constrained hypothesis)
    #   1 = they diverge (one of several plausible paths)
    # 0.7 * dispersion of time-to-failure (robust CV, capped at 1)
    # + 0.3 * disagreement on the gravity trend (1 - majority share)
    u = None
    if len(ttf) >= 2:
        cv = float(np.std(ttf) / max(np.mean(ttf), 1e-9))
        dis = 0.0
        if esc:
            maj = max(sum(esc), len(esc) - sum(esc)) / len(esc)
            dis = 1.0 - maj
        u = round(min(1.0, 0.7 * min(cv, 1.0) + 0.3 * dis), 3)
    agg["uncertainty"] = u
    return {"cases": cases, "aggregate": agg}


def fmt_reliability(rel: dict) -> str:
    if not rel or rel.get("value") is None:
        return ""
    return (f"RELIABILITY: similarity of this case to the fleet knowledge "
            f"base = {rel['value']:.3f} (top {rel['top']:.3f} over "
            f"{rel['k']} precedents). High = well covered by known cases; "
            f"low = novel case: rely less on precedent-based conclusions "
            f"and WIDEN your stated range.")


def fmt_future_progression(fp: dict) -> str:
    if not fp or not fp.get("cases"):
        return ""
    ls = ["FUTURE PROGRESSION OF THE SIMILAR CASES (observed outcomes):"]
    for c in fp["cases"]:
        ev = ", ".join(f"+{e['plus']} (g{e['gravity']})"
                       for e in c["events"]) or "no further anomalies"
        rt = (f"failed {c['rul_then']:.0f} cycles after the match"
              if c["rul_then"] is not None else "end of life unknown")
        ls.append(f"  {c['id']}: next anomalies at {ev}; {rt}.")
    a = fp["aggregate"]
    if a.get("median_ttf") is not None:
        ls.append(f"  across cited cases: median time-to-failure "
                  f"+{a['median_ttf']:.0f} cycles "
                  f"(range {a['ttf_range']}); gravity escalated in "
                  f"{a['escalating'] or 'n/a'}.")
    if a.get("uncertainty") is not None:
        u = a["uncertainty"]
        read = ("the cited futures AGREE: the hypothesis is well "
                "constrained" if u < 0.35 else
                "the cited futures DIVERGE: treat the hypothesis as one of "
                "several plausible paths and WIDEN range and lower "
                "confidence accordingly")
        ls.append(f"  PROGRESSION UNCERTAINTY = {u:.2f} "
                  f"(dissimilarity of the cited futures; {read}).")
    ls.append("Hypothesize THIS unit's own progression in "
              "\"projected_progression\", grounded in these futures.")
    return "\n".join(ls)


def _time():
    from .forecast import PrecedentFutures
    root = Path(__file__).resolve().parents[2]
    pf = PrecedentFutures(root / "data" / "vector_store" / "meta.jsonl")
    eps = [json.loads(l) for l in
           open(root / "results" / "final_prognostic" / "forecast_episodes.jsonl")]
    eps = [e for e in eps if e["arm"] == "P7_agent_dl"]
    t0 = time.perf_counter()
    reps = 50
    for _ in range(reps):
        for e in eps:
            fp = future_progression(pf, e.get("contexts") or [])
            reliability(e.get("contexts") or [])
            fmt_future_progression(fp)
    dt = time.perf_counter() - t0
    per_case = dt / (reps * len(eps))
    print(f"[future_progression] {reps * len(eps)} lookups in {dt:.3f}s "
          f"-> {per_case*1e3:.3f} ms per case "
          f"({per_case*1e6/4:.0f} us per precedent, incl. formatting); "
          f"one full 89-case run: {per_case*89*1e3:.1f} ms total.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "time":
        _time()

"""Step 1: build an interpretation for EVERY anomaly, TIOT-style.

The 84 real SLM interpretations cover 5 units; this generator scales the
enriched database to all TRAIN anomalies (4,393) so the vector store has
fleet-wide coverage. Differences from the original TIOT generation, each
deliberate:

  * GROUNDED INPUT: besides the extracted rule (Splits=...) and cumulative
    counters, the prompt carries the regime-referenced z-deviations from the
    shared feature pipeline -- the same evidence every other arm sees.
  * STRUCTURED OUTPUT: {"interpretation", "gravity" (1-5 int),
    "components" (list)} -- gravity extraction was possible for only 30/84
    of the free-text originals; structured output makes the audit total.
  * FEW-SHOT from the real interpretations of TRAIN units only (57/146/232).
  * PROVENANCE: every record is tagged source="generated" with the model
    name; generated text is retrieval MEMORY and audit target, never gold.
  * RESUMABLE: one JSONL per unit under data/interpretations_generated/;
    existing (unit, cycle) keys are skipped, so interrupted runs continue.

  python -m apdm.gen_interpretations --backend ollama --model llama3.2:3b \\
      --units train --limit 0
Dry-run (offline, deterministic templates -- pipeline testing only):
  python -m apdm.gen_interpretations --backend dryrun --limit 40
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..data import FD002, SENSORS
from .faults import SENSOR_PHYSICS, current_z
from ..llm import get_backend

ROOT = Path(__file__).resolve().parent.parent
GEN_DIR = ROOT / "data" / "interpretations_generated"

SYSTEM = """You are the diagnostic interpretation layer of an IoT predictive
maintenance system for turbofan engines. Given one detected anomaly with its
extracted isolation-forest rule and sensor deviations, write a concise
engineering interpretation.

Reply with ONE JSON object, nothing else:
{"interpretation": "<3-5 sentences: what deviates, plausible physical cause,
expected consequence if unaddressed, recommended check>",
 "gravity": <integer 1-5, 1 trivial - 5 severe>,
 "components": ["<subsystem>", ...]}

Ground every claim in the given evidence; never invent sensor values."""


def _fewshot(ds: FD002, k: int = 2) -> str:
    shots = []
    for f in sorted((ROOT / "data" / "interpretations").glob("unit_*.json")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if int(r["unit_ID"]) in ds.train_units:
                txt = str(r.get("interpretation", ""))[:600]
                if len(txt) > 200:
                    shots.append(f"EXAMPLE INTERPRETATION (style reference):\n"
                                 f"{txt}")
            if len(shots) >= k:
                return "\n\n".join(shots)
    return "\n\n".join(shots)


def anomaly_prompt(ds: FD002, unit: int, cycle: int, fewshot: str) -> str:
    row = ds.row(unit, cycle)
    z = current_z(ds, unit, cycle)
    order = np.argsort(-np.abs(z))[:6]
    devs = "\n".join(
        f"  {SENSORS[i]} ({SENSOR_PHYSICS[SENSORS[i]][1]}): "
        f"z={z[i]:+.2f} vs healthy reference of regime "
        f"{int(row['h_clust'])}" for i in order)
    entry = ds.state_entered(unit, cycle)
    return (f"{fewshot}\n\nANOMALY RECORD\n"
            f"Unit {unit}, cycle {cycle}, operating regime "
            f"{int(row['h_clust'])}.\n"
            f"Isolation-forest rule: {str(row['text'])[:300]}\n"
            f"Largest deviations:\n{devs}\n"
            f"Degradation state: "
            + (f"entered at cycle {entry}." if entry is not None
               else "not entered.")
            + "\n\nWrite the JSON interpretation now.")


def parse_gen(raw: str) -> Optional[Dict]:
    from .agent import _first_json
    o = _first_json(raw or "")
    if not o or "interpretation" not in o:
        return None
    try:
        grav = int(o.get("gravity", 0))
    except (TypeError, ValueError):
        grav = 0
    if not 1 <= grav <= 5:
        return None
    comps = o.get("components") or []
    if not isinstance(comps, list):
        comps = [str(comps)]
    return {"interpretation": str(o["interpretation"])[:2500],
            "gravity": grav, "components": [str(c)[:40] for c in comps][:5]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="dryrun",
                    choices=["dryrun", "ollama", "openai"])
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--units", default="train",
                    choices=["train", "test", "all"],
                    help="which units' anomalies to interpret; KB use is "
                         "train-only, test generation is for audits")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N new records (0 = all)")
    ap.add_argument("--device", default="orin_nano_8gb")
    a = ap.parse_args()

    ds = FD002(seed=42)
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    units = {"train": ds.train_units, "test": ds.test_units,
             "all": ds.train_units + ds.test_units}[a.units]

    an = ds.df[(ds.df["anomaly_label"] == -1)
               & (ds.df["unit_ID"].isin(units))]
    todo = [(int(r.unit_ID), int(r.cycle)) for r in an.itertuples()]

    done = set()
    for f in GEN_DIR.glob("unit_*.jsonl"):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((int(r["unit_ID"]), int(r["cycle"])))
    todo = [t for t in todo if t not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"[gen] {len(an)} anomalies in scope, {len(done)} already "
          f"generated, {len(todo)} to do")
    if not todo:
        return

    backend = get_backend(a.backend, a.model, device=a.device,
                          log_path=GEN_DIR / "_gen_calls.jsonl")
    fewshot = _fewshot(ds)
    t0, n_ok, n_fail = time.time(), 0, 0
    for i, (u, c) in enumerate(todo, 1):
        prompt = anomaly_prompt(ds, u, c, fewshot)
        raw = backend.generate(prompt, system=SYSTEM)
        rec = parse_gen(raw)
        if rec is None:
            raw = backend.generate(prompt + "\n\nReply with ONLY the JSON "
                                            "object.", system=SYSTEM)
            rec = parse_gen(raw)
        if rec is None:
            n_fail += 1
            continue
        n_ok += 1
        out = {"unit_ID": u, "cycle": c, "source": "generated",
               "gen_model": a.model, **rec}
        with open(GEN_DIR / f"unit_{u}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(out) + "\n")
        if i % 25 == 0:
            r = (time.time() - t0) / i
            print(f"[gen] {i}/{len(todo)} ok={n_ok} fail={n_fail} "
                  f"({r:.1f}s/rec, ETA {(len(todo)-i)*r/60:.0f} min)  "
                  f"{backend.totals()}")
    print(f"[gen] done: ok={n_ok} fail={n_fail}  totals={backend.totals()}")


if __name__ == "__main__":
    main()

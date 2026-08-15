"""Interpretation memory: the TIOT paper's enriched database as an agent KB.

Records = TRAIN-unit anomaly rows (IF label -1): each carries the rule text
(`Splits=(...)` + counters) and, where the unit is one of the SLM-interpreted
train units, an interpretation excerpt. Retrieval is feature-space kNN on the
14-sensor regime-referenced z-vector at the anomaly's (unit, cycle) -- the
same space the fault layer and the no-learning diagnosis twin use, so
"the agent's retrieval" and "the twin" are directly comparable and
embedder-free.

Leakage rules (guarded in smoke K-series):
  * KB units are TRAIN only. Interpreted units falling in the test split
    (58, 140 under seed 42) are EXCLUDED from the KB and kept aside as
    held-out qualitative references.
  * Queries carry no unit identity requirement; when the query unit happens
    to be a train unit (twin calibration runs), its own records are excluded
    from results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..data import FD002, SENSORS
from .faults import current_z

ROOT = Path(__file__).resolve().parent.parent
INTERP_DIR = ROOT / "data" / "interpretations"


@dataclass
class KBRecord:
    unit: int
    cycle: int
    rul_then: int              # known outcome of the precedent (train unit)
    splits: str
    interpretation: str        # "" when the unit has no SLM interpretation


class InterpretationKB:
    def __init__(self, ds: FD002, interp_dir: Path = INTERP_DIR,
                 excerpt_chars: int = 480, cache: Optional[Path] = None):
        if cache and cache.exists():
            import pickle
            with open(cache, "rb") as f:
                other = pickle.load(f)
            self.__dict__.update(other.__dict__)
            return
        self.records: List[KBRecord] = []
        interp_map: Dict[tuple, str] = {}
        self.held_out_interpreted: List[Dict] = []

        train = set(ds.train_units)
        for f in sorted(interp_dir.glob("unit_*.json")):
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                u = int(r["unit_ID"])
                cyc = int(r.get("cycle", r.get("cycles", 0)))
                text = str(r.get("interpretation", ""))[:excerpt_chars]
                if u in train:
                    interp_map[(u, cyc)] = text
                else:
                    self.held_out_interpreted.append(
                        {"unit": u, "cycle": cyc, "interpretation": text})

        an = ds.df[(ds.df["anomaly_label"] == -1)
                   & (ds.df["unit_ID"].isin(train))]
        zs = []
        for _, row in an.iterrows():
            u, cyc = int(row["unit_ID"]), int(row["cycle"])
            self.records.append(KBRecord(
                unit=u, cycle=cyc, rul_then=ds.rul(u, cyc),
                splits=str(row["text"])[:300],
                interpretation=interp_map.get((u, cyc), "")))
            zs.append(current_z(ds, u, cyc))
        self._Z = np.vstack(zs) if zs else np.zeros((0, len(SENSORS)))
        self.n_interpreted = sum(1 for r in self.records if r.interpretation)
        if cache:
            import pickle
            cache.parent.mkdir(parents=True, exist_ok=True)
            with open(cache, "wb") as f:
                pickle.dump(self, f)

    # ------------------------------------------------------------- search
    def search(self, zvec: np.ndarray, k: int = 4,
               exclude_unit: Optional[int] = None,
               prefer_interpreted: bool = True) -> List[KBRecord]:
        if not len(self.records):
            return []
        d = np.linalg.norm(self._Z - zvec[None, :], axis=1)
        order = np.argsort(d)
        out: List[KBRecord] = []
        # pass 1: nearest interpreted precedents (the TIOT premise under test)
        if prefer_interpreted:
            for i in order:
                r = self.records[int(i)]
                if exclude_unit is not None and r.unit == exclude_unit:
                    continue
                if r.interpretation:
                    out.append(r)
                if len(out) >= max(1, k // 2):
                    break
        # pass 2: nearest overall, fill to k
        for i in order:
            r = self.records[int(i)]
            if exclude_unit is not None and r.unit == exclude_unit:
                continue
            if r in out:
                continue
            out.append(r)
            if len(out) >= k:
                break
        return out

    def observation(self, zvec: np.ndarray, k: int = 4,
                    exclude_unit: Optional[int] = None) -> str:
        recs = self.search(zvec, k=k, exclude_unit=exclude_unit)
        payload = [{
            "unit": r.unit, "cycle": r.cycle, "rul_then": r.rul_then,
            "rule": r.splits,
            **({"slm_interpretation_excerpt": r.interpretation}
               if r.interpretation else {})} for r in recs]
        return json.dumps({
            "k": len(payload),
            "note": "training-fleet anomaly precedents; nearest in "
                    "regime-referenced z-space; rul_then is what actually "
                    "remained (clipped 125)",
            "precedents": payload})

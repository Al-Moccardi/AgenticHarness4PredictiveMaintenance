"""Data layer for the agentic-vs-ML PdM study. FD002 (CMAPSS), 260 units.

Protocol deliberately mirrors Moccardi et al. (ResPdM) so numbers are
directly comparable with their Table 1:
  window W = 20 cycles, RUL clipped at RMAX = 125, unit-level 80/20 split
  (208 train / 52 test), one-step horizon, S-score per Saxena et al. 2008.

Ground truth: RUL(unit, cycle) = clip(EoL_unit - cycle, 0, 125). Fully
computable from the run-to-failure trajectories -- no pseudo-labels anywhere
in the evaluation targets. The only derived signal used as INPUT is the
degradation-state layer: the ResPdM-style k=7 regime refinement (DBSCAN
lineage) whose 7th cluster captures the failure modality; it never appears as
a target. Isolation-Forest-era columns present in the CSV (anomaly_label,
anomaly_score, cumulative counters, split-rule text, SLM interpretations) are
IGNORED throughout -- that prior pipeline is out of scope for this study.
The healthy per-regime reference is defined purely from ground truth:
TRAIN-unit rows with true RUL > 100 (early life).

Leakage rule enforced here and tested in smoke_test.py:
  any view at (unit u, cycle c) may use rows of u with cycle <= c, plus any
  row of TRAIN units. Test-unit futures and test-unit cross-retrieval are
  structurally unreachable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SENSORS = ["T24", "T30", "T50", "P30", "Nf", "Nc", "Ps30", "phi",
           "NRf", "NRc", "BPR", "htBleed", "W31", "W32"]
W = 20          # history window (paper 1)
RMAX = 125      # RUL clip (paper 1)
DEGRADATION_CLUSTER = 6   # k=7 refinement: cluster 6 = degradation state


@dataclass(frozen=True)
class Snapshot:
    unit: int
    cycle: int
    rul: int            # clipped ground truth


class FD002:
    def __init__(self, data_dir: Path = ROOT / "data", seed: int = 42,
                 train_frac: float = 0.8):
        _csv = data_dir / "Dataset_with_interpretations.csv"
        if not _csv.exists():        # v7.2 export: original 31 cols + RUL
            _csv = data_dir / "Dataset_with_interpretations_RUL.csv"
        df = pd.read_csv(_csv)
        k7 = pd.read_csv(data_dir / "FD002_labeled_k7.csv")
        assert len(df) == len(k7), "k7 labels must be row-aligned"
        df = df.copy()
        df["k7"] = k7["cluster_label"].values
        df = df.sort_values(["unit_ID", "cycle"]).reset_index(drop=True)

        self.df = df
        self.eol = df.groupby("unit_ID")["cycle"].max().to_dict()

        units = sorted(df["unit_ID"].unique())
        rng = np.random.default_rng(seed)
        rng.shuffle(units)
        n_tr = int(round(train_frac * len(units)))
        self.train_units = sorted(int(u) for u in units[:n_tr])
        self.test_units = sorted(int(u) for u in units[n_tr:])
        self._by_unit: Dict[int, pd.DataFrame] = {
            int(u): g.reset_index(drop=True) for u, g in df.groupby("unit_ID")}

        # healthy per-regime reference for z-scoring: TRAIN units, early
        # life (true RUL > 100). Ground-truth-defined; no detector involved.
        tr = df[df["unit_ID"].isin(self.train_units)].copy()
        tr["_rul_raw"] = tr["unit_ID"].map(self.eol) - tr["cycle"]
        healthy = tr[tr["_rul_raw"] > 100]
        self.regime_ref: Dict[int, Dict[str, Tuple[float, float]]] = {}
        for c, g in healthy.groupby("h_clust"):
            self.regime_ref[int(c)] = {
                s: (float(g[s].mean()), float(g[s].std(ddof=0)) or 1.0)
                for s in SENSORS}

    # ------------------------------------------------------------- labels
    def rul(self, unit: int, cycle: int) -> int:
        return int(min(max(self.eol[unit] - cycle, 0), RMAX))

    # ------------------------------------------------------ snapshot lists
    def snapshots(self, units: List[int]) -> List[Snapshot]:
        """All cycles with a full W-history. Evaluation universe."""
        out: List[Snapshot] = []
        for u in units:
            for c in range(W, self.eol[u] + 1):
                out.append(Snapshot(u, c, self.rul(u, c)))
        return out

    def sample_snapshots(self, units: List[int], per_bucket: int,
                         seed: int, buckets=((0, 30), (31, 100), (101, RMAX)),
                         ) -> List[Snapshot]:
        """Stratified by true clipped RUL, for the (expensive) LLM arms.
        ML arms are evaluated on BOTH the full universe and this subset."""
        rng = np.random.default_rng(seed)
        pool = self.snapshots(units)
        out: List[Snapshot] = []
        for lo, hi in buckets:
            cand = [s for s in pool if lo <= s.rul <= hi]
            idx = rng.choice(len(cand), size=min(per_bucket, len(cand)),
                             replace=False)
            out.extend(cand[int(i)] for i in idx)
        rng.shuffle(out)
        return out

    # ------------------------------------------------- leakage-safe views
    def history(self, unit: int, cycle: int, n: int = W) -> pd.DataFrame:
        """Rows of `unit` with cycle in (cycle-n, cycle]. Never the future."""
        g = self._by_unit[unit]
        return g[(g["cycle"] > cycle - n) & (g["cycle"] <= cycle)]

    def row(self, unit: int, cycle: int) -> pd.Series:
        g = self._by_unit[unit]
        m = g[g["cycle"] == cycle]
        if m.empty:
            raise KeyError(f"unit {unit} cycle {cycle}")
        return m.iloc[0]

    def state_entered(self, unit: int, cycle: int) -> Optional[int]:
        """First cycle <= c at which the k7 degradation state was entered."""
        g = self._by_unit[unit]
        past = g[(g["cycle"] <= cycle) & (g["k7"] == DEGRADATION_CLUSTER)]
        return int(past["cycle"].min()) if len(past) else None

    def train_state_residuals(self) -> Dict[str, float]:
        """Residual life after first degradation-state entry, over TRAIN
        units that ever enter. A ResPdM-flavoured survival prior the agent
        may consult; leakage-safe by construction (train units only)."""
        res = []
        for u in self.train_units:
            g = self._by_unit[u]
            hit = g[g["k7"] == DEGRADATION_CLUSTER]
            if len(hit):
                res.append(self.eol[u] - int(hit["cycle"].min()))
        if not res:
            return {"n_units": 0}
        arr = np.array(res, float)
        return {"n_units": len(res),
                "median_residual_after_entry": float(np.median(arr)),
                "iqr_lo": float(np.quantile(arr, .25)),
                "iqr_hi": float(np.quantile(arr, .75))}

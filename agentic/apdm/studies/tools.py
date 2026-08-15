"""Tools for the agentic arms. Every tool is bound to ONE snapshot (unit,
cycle) at construction and is structurally incapable of leaking:

  - same-unit access is truncated at the bound cycle;
  - cross-unit access goes only through the TRAIN-snapshot kNN index;
  - `ml_predict` (P3 only) exposes the XGBoost estimate for THIS snapshot,
    with its full-test MAE attached so the agent knows how much to trust it.

Tool outputs are the same quantities the ML arms consume -- see features.py.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

import numpy as np

from ..data import FD002, SENSORS, W, Snapshot
from .features import snapshot_features, summary_text
from .ml_models import MLBundle


class ToolBox:
    def __init__(self, ds: FD002, bundle: MLBundle, snap: Snapshot,
                 allow_ml: bool = False, xgb_full_mae: float = 12.4,
                 fault_layer=None):
        self.ds, self.bundle, self.snap = ds, bundle, snap
        self.allow_ml = allow_ml
        self.xgb_full_mae = xgb_full_mae
        self.fault_layer = fault_layer
        self._x_std = None

    # ------------------------------------------------------------- registry
    def specs(self) -> str:
        s = [
            "- sensor_summary() -> per-sensor z-scores vs the healthy "
            "reference of the current regime, 20-cycle slopes, diagnostic "
            "anomaly counters, degradation-state flag.",
            "- sensor_window(sensor:str) -> the raw last-20-cycle values of "
            "one sensor (choose from: " + ", ".join(SENSORS) + ").",
            "- similar_cases(k:int=7) -> the k most similar TRAINING "
            "snapshots (feature-space nearest neighbours) with the remaining "
            "life each of them actually had, plus their median.",
            "- degradation_status() -> whether this unit has entered the "
            "degradation state (k=7 regime refinement, cluster 6), when, and "
            "the TRAINING fleet's residual-life statistics after entry.",
        ]
        if self.fault_layer is not None:
            s.append("- fault_library() -> the fleet's fault-phenotype "
                     "library: characteristic sensor signatures, implicated "
                     "subsystems, and state-onset / residual-life statistics "
                     "from the TRAINING fleet.")
        if self.allow_ml:
            s.append("- ml_predict() -> the XGBoost regressor's RUL estimate "
                     "for this exact snapshot, with its known average error.")
        return "\n".join(s)

    def names(self) -> List[str]:
        base = ["sensor_summary", "sensor_window", "similar_cases",
                "degradation_status"]
        if self.fault_layer is not None:
            base.append("fault_library")
        return base + (["ml_predict"] if self.allow_ml else [])

    def call(self, name: str, args: Optional[Dict] = None) -> str:
        args = args or {}
        try:
            if name == "sensor_summary":
                return summary_text(self.ds, self.snap, top_k=8)
            if name == "sensor_window":
                return self._sensor_window(str(args.get("sensor", "")))
            if name == "similar_cases":
                return self._similar(int(args.get("k", 7) or 7))
            if name == "degradation_status":
                return self._state()
            if name == "fault_library" and self.fault_layer is not None:
                return self.fault_layer.library_json()
            if name == "ml_predict" and self.allow_ml:
                return self._ml()
            return json.dumps({"error": f"unknown tool '{name}'",
                               "available": self.names()})
        except Exception as e:  # noqa: BLE001 -- degrade, never abort a run
            return json.dumps({"error": f"{type(e).__name__}: {e}"})

    # ---------------------------------------------------------------- impl
    def _sensor_window(self, sensor: str) -> str:
        if sensor not in SENSORS:
            return json.dumps({"error": f"'{sensor}' is not a sensor",
                               "sensors": SENSORS})
        h = self.ds.history(self.snap.unit, self.snap.cycle, n=W)
        vals = [round(float(v), 3) for v in h[sensor]]
        return json.dumps({"sensor": sensor,
                           "cycles": [int(c) for c in h["cycle"]],
                           "values": vals})

    def _similar(self, k: int) -> str:
        if self._x_std is None:
            x = snapshot_features(self.ds, self.snap)
            self._x_std = (x - self.bundle.scaler_mu) / self.bundle.scaler_sd
        med, neigh = self.bundle.knn_query(self._x_std, k=max(1, min(k, 15)))
        return json.dumps({"k": len(neigh),
                           "note": "training-fleet precedents; rul_then is "
                                   "the remaining life each neighbour "
                                   "actually had (clipped at 125)",
                           "median_rul_of_neighbours": round(med, 1),
                           "neighbours": neigh})

    def _state(self) -> str:
        entry = self.ds.state_entered(self.snap.unit, self.snap.cycle)
        prior = self.ds.train_state_residuals()
        return json.dumps({
            "unit": self.snap.unit, "current_cycle": self.snap.cycle,
            "degradation_state_entered": entry is not None,
            "state_entry_cycle": entry,
            "cycles_in_state": (self.snap.cycle - entry) if entry else 0,
            "training_fleet_prior_after_entry": prior,
            "note": "prior is over TRAINING units only; residuals are raw "
                    "cycles (uncapped), your answer must stay in 0-125"})

    def _ml(self) -> str:
        p = self.bundle.xgb_predict_one(self.ds, self.snap)
        return json.dumps({
            "model": "XGBoost (trained on the 208 training units)",
            "predicted_rul": round(p, 1),
            "known_mean_abs_error_full_test": round(self.xgb_full_mae, 1),
            "note": "you may accept, adjust, or override this estimate; "
                    "justify any adjustment from the other tools"})

"""ML baselines. Trained on TRAIN-unit snapshots, targets = clipped RUL.

The lineup is chosen so every agentic arm has a non-language twin:
  naive_mean / linear_deg : the RcM-style comparators of paper 1
  knn                     : the CRITICAL control -- identical retrieval to the
                            agent's `similar_cases` tool, minus the language
                            model. If the P2 agent ~= knn, the LLM added
                            nothing over its own retrieval tool.
  xgb / histgb / mlp      : learned tabular models on the same features
                            (xgb doubles as the `ml_predict` tool in P3).
Paper 1's BiLSTM (MSE 348.98, R2 0.83, S 99164 on this protocol) anchors the
deep end without retraining it here.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..data import FD002, RMAX, Snapshot
from .features import feature_matrix, feature_names


@dataclass
class MLBundle:
    names: List[str]
    scaler_mu: np.ndarray
    scaler_sd: np.ndarray
    train_X: np.ndarray          # standardised, for kNN
    train_y: np.ndarray
    train_meta: List[Snapshot]
    mean_eol_train: float
    models: Dict[str, object]

    # ------------------------------------------------------------ predict
    def _std(self, X: np.ndarray) -> np.ndarray:
        return (X - self.scaler_mu) / self.scaler_sd

    def predict(self, name: str, ds: FD002, snaps: List[Snapshot],
                X: Optional[np.ndarray] = None) -> np.ndarray:
        if X is None:
            X = feature_matrix(ds, snaps)
        if name == "naive_mean":
            return np.full(len(snaps), float(self.train_y.mean()))
        if name == "linear_deg":
            return np.clip([self.mean_eol_train - s.cycle for s in snaps],
                           0, RMAX)
        if name == "knn":
            return np.array([self.knn_query(x)[0] for x in self._std(X)])
        mdl = self.models[name]
        return np.clip(mdl.predict(self._std(X)), 0, RMAX)

    def knn_query(self, x_std: np.ndarray, k: int = 7):
        """Median RUL of the k nearest TRAIN snapshots + the neighbours.
        Exactly what the agent's `similar_cases` tool exposes."""
        d = np.linalg.norm(self.train_X - x_std, axis=1)
        idx = np.argsort(d)[:k]
        ruls = self.train_y[idx]
        neigh = [{"unit": self.train_meta[i].unit,
                  "cycle": self.train_meta[i].cycle,
                  "rul_then": int(self.train_y[i]),
                  "distance": round(float(d[i]), 3)} for i in idx]
        return float(np.median(ruls)), neigh

    def xgb_predict_one(self, ds: FD002, s: Snapshot) -> float:
        from .features import snapshot_features
        x = self._std(snapshot_features(ds, s)[None, :])
        return float(np.clip(self.models["xgb"].predict(x)[0], 0, RMAX))


def train_all(ds: FD002, cache: Optional[Path] = None,
              seed: int = 42) -> MLBundle:
    if cache and cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    snaps = ds.snapshots(ds.train_units)
    X = feature_matrix(ds, snaps)
    y = np.array([s.rul for s in snaps], float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd

    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor
    from xgboost import XGBRegressor

    models: Dict[str, object] = {}
    models["xgb"] = XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.06,
        subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
        objective="reg:squarederror", random_state=seed, n_jobs=4,
    ).fit(Xs, y)
    models["histgb"] = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, random_state=seed).fit(Xs, y)
    models["mlp"] = MLPRegressor(
        hidden_layer_sizes=(128, 64), max_iter=300, early_stopping=True,
        random_state=seed).fit(Xs, y)

    bundle = MLBundle(
        names=feature_names(), scaler_mu=mu, scaler_sd=sd,
        train_X=Xs, train_y=y, train_meta=snaps,
        mean_eol_train=float(np.mean([ds.eol[u] for u in ds.train_units])),
        models=models)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(bundle, f)
    return bundle


ML_ARMS = ["naive_mean", "linear_deg", "knn", "mlp", "histgb", "xgb"]

"""Bridge to the OFFICIAL CMAPSS FD002 test split (test_FD002.txt).

WHY: the internal 208/52 split is ours; the official split (259 truncated
trajectories, one prediction per unit at the truncation point, gold in
RUL_FD002.txt) is what the entire CMAPSS literature reports on. Adding it
makes the ML anchor externally comparable.

WHAT "MATCHING" MEANS HERE -- and what it cannot mean:
  * Train CSV and test txt are DIFFERENT engines; there is no row join.
    Matching = the train-fitted apparatus applied to test units: the regime
    reference, the feature pipeline, the models, and (for agents) retrieval
    into the train fleet memory.
  * RUL cannot be "extracted" from test_FD002.txt: trajectories stop BEFORE
    failure by design. Gold is the 259-line RUL_FD002.txt. Until that file
    is present at data/RUL_FD002.txt, this module produces predictions and
    validation diagnostics but NO scores, and says so loudly.
  * Diagnosis/RCA gold (future-derived terminal signatures) is UNAVAILABLE
    on the official split -- units never reach EoL. RCA evaluation stays on
    the internal split; prognosis evaluation moves here. Two-track design.

REGIME ASSIGNMENT: test rows get h_clust by nearest train-regime centroid in
globally standardised sensor space (FD002's six operating regimes are widely
separated blobs). Validated by reassigning train rows against their true
h_clust; accuracy is printed and guarded in smoke test O1.

STATE CHANNEL: k7 labels exist only for the train CSV. Official snapshots run
with the state features at their 'not entered' value -- a real, common value
in training, so the models are coherent; it is simply less information.
If a test-clustered k7 export is provided later, it plugs in here.

  python -m apdm.run_official            # cached 208-unit bundle
  python -m apdm.run_official --train-all  # retrain on all 260 (final paper)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..data import FD002, RMAX, SENSORS, W, Snapshot

ROOT = Path(__file__).resolve().parent.parent
RAW_COLS = ['unit_ID', 'cycles', 'setting_1', 'setting_2', 'setting_3',
            'T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr',
            'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd',
            'PCNfR_dmd', 'W31', 'W32']


# ------------------------------------------------------------------ loading
def load_official_test(path: Path, ds_train: FD002) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=RAW_COLS)
    df = df.rename(columns={"cycles": "cycle"})
    df = df[["unit_ID", "cycle"] + SENSORS].copy()
    df["h_clust"] = assign_regimes(df, ds_train)
    return df.sort_values(["unit_ID", "cycle"]).reset_index(drop=True)


def _regime_centroids(ds: FD002):
    tr = ds.df[ds.df["unit_ID"].isin(ds.train_units)]
    mu = tr[SENSORS].mean()
    sd = tr[SENSORS].std(ddof=0).replace(0, 1.0)
    cent = tr.groupby("h_clust")[SENSORS].mean()
    return mu, sd, ((cent - mu) / sd), cent.index.to_numpy()


def assign_regimes(df: pd.DataFrame, ds: FD002) -> np.ndarray:
    mu, sd, cz, labels = _regime_centroids(ds)
    Z = (df[SENSORS] - mu) / sd
    d = ((Z.values[:, None, :] - cz.values[None, :, :]) ** 2).sum(-1)
    return labels[np.argmin(d, axis=1)]


def regime_assignment_accuracy(ds: FD002, n_sample: int = 5000,
                               seed: int = 0) -> float:
    """Reassign TRAIN rows by nearest centroid; agreement with true h_clust."""
    rng = np.random.default_rng(seed)
    tr = ds.df[ds.df["unit_ID"].isin(ds.train_units)]
    idx = rng.choice(len(tr), size=min(n_sample, len(tr)), replace=False)
    sub = tr.iloc[idx]
    pred = assign_regimes(sub, ds)
    return float((pred == sub["h_clust"].to_numpy()).mean())


# --------------------------------------------------- FD002-compatible view
class OfficialData:
    """Duck-types the slice of FD002 that features.py needs, over official
    test units. Regime reference is inherited from the TRAIN fit; the k7
    state channel is absent by construction (see module docstring)."""

    def __init__(self, ds_train: FD002, test_df: pd.DataFrame):
        self.regime_ref = ds_train.regime_ref
        self._by_unit: Dict[int, pd.DataFrame] = {
            int(u): g.reset_index(drop=True)
            for u, g in test_df.groupby("unit_ID")}
        self.last_cycle = {u: int(g["cycle"].max())
                           for u, g in self._by_unit.items()}

    def history(self, unit: int, cycle: int, n: int = W) -> pd.DataFrame:
        g = self._by_unit[unit]
        return g[(g["cycle"] > cycle - n) & (g["cycle"] <= cycle)]

    def state_entered(self, unit: int, cycle: int) -> Optional[int]:
        return None                     # k7 unavailable on the official split

    def snapshots(self) -> List[Snapshot]:
        """One snapshot per unit at its truncation point. rul=-1 placeholder
        until RUL_FD002.txt supplies gold."""
        return [Snapshot(u, c, -1) for u, c in sorted(self.last_cycle.items())]


def per_row_rul(test_df: pd.DataFrame, gold: np.ndarray) -> pd.DataFrame:
    """Attach a decreasing per-row RUL to the official test rows:
    RUL(u, c) = R_u + (T_u - c), where R_u is the file value at truncation
    and T_u the unit's last recorded cycle. Also the clip-125 variant."""
    units = sorted(test_df["unit_ID"].unique())
    assert len(units) == len(gold), (len(units), len(gold))
    gmap = {u: int(g) for u, g in zip(units, gold)}
    tmap = test_df.groupby("unit_ID")["cycle"].max().to_dict()
    out = test_df.copy()
    out["RUL"] = [gmap[u] + (tmap[u] - c)
                  for u, c in zip(out["unit_ID"], out["cycle"])]
    out["RUL_clip125"] = out["RUL"].clip(upper=RMAX)
    return out


def load_gold(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    vals = [int(float(x.split()[0])) for x in path.read_text().split("\n")
            if x.strip()]
    return np.asarray(vals)


# ---------------------------------------------------------------- runner
def main() -> None:
    import argparse
    from .features import feature_matrix
    from .metrics import regression_metrics
    from .ml_models import ML_ARMS, MLBundle, train_all

    ap = argparse.ArgumentParser()
    ap.add_argument("--test-file", default=str(ROOT / "data" / "test_FD002.txt"))
    ap.add_argument("--rul-file", default=str(ROOT / "data" / "RUL_FD002.txt"))
    ap.add_argument("--train-all", action="store_true",
                    help="retrain the bundle on all 260 train units "
                         "(standard literature protocol; ~5 min)")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(exist_ok=True)

    ds = FD002(seed=42, train_frac=1.0 if a.train_all else 0.8)
    cache = ROOT / "cache" / ("ml_official_all260.pkl" if a.train_all
                              else "ml_v2.pkl")
    bundle: MLBundle = train_all(ds, cache=cache)

    acc = regime_assignment_accuracy(ds)
    print(f"[official] regime nearest-centroid self-consistency on train: "
          f"{acc:.4f}")

    test_df = load_official_test(Path(a.test_file), ds)
    od = OfficialData(ds, test_df)
    snaps = od.snapshots()
    print(f"[official] {len(snaps)} test units; history lengths "
          f"min={min(od.last_cycle.values())}, "
          f"max={max(od.last_cycle.values())}")

    X = feature_matrix(od, snaps)          # type: ignore[arg-type]
    preds = {arm: bundle.predict(arm, od, snaps, X=X)   # type: ignore
             for arm in ML_ARMS}
    dfp = pd.DataFrame({"unit": [s.unit for s in snaps],
                        "last_cycle": [s.cycle for s in snaps],
                        **{f"pred_{k}": v for k, v in preds.items()}})

    gold = load_gold(Path(a.rul_file))
    if gold is None:
        dfp.to_csv(out / "official_predictions_UNSCORED.csv", index=False)
        print("\n[official] RUL_FD002.txt NOT FOUND -> predictions written "
              "UNSCORED to official_predictions_UNSCORED.csv.")
        print("[official] Supply data/RUL_FD002.txt (259 lines, one integer "
              "per unit in unit order) to score.")
        print("[official] prediction sanity (xgb): "
              f"min {dfp.pred_xgb.min():.0f}, med {dfp.pred_xgb.median():.0f},"
              f" max {dfp.pred_xgb.max():.0f}, frac at clip "
              f"{(dfp.pred_xgb >= RMAX - 1).mean():.2f}")
        return

    assert len(gold) == len(dfp), (len(gold), len(dfp))
    merged = per_row_rul(test_df, gold)
    merged.to_csv(out / "test_FD002_with_RUL.csv", index=False)
    print(f"[official] wrote per-row test_FD002_with_RUL.csv "
          f"({len(merged)} rows, RUL decreasing to the file value at "
          f"truncation)")
    dfp["rul_true"] = gold
    dfp["rul_true_clip"] = np.clip(gold, 0, RMAX)
    rows = []
    print(f"\n[official] scored on {len(dfp)} units "
          f"(literature protocol: one prediction per unit)")
    for arm in ML_ARMS:
        for tgt, tag in (("rul_true_clip", "clip125"), ("rul_true", "raw")):
            m = regression_metrics(dfp[f"pred_{arm}"], dfp[tgt])
            rows.append({"arm": arm, "target": tag, **m})
        mc = regression_metrics(dfp[f"pred_{arm}"], dfp["rul_true_clip"])
        print(f"  {arm:>10}: RMSE {mc['rmse']:6.2f}  MAE {mc['mae']:6.2f}  "
              f"S {mc['s_score']:>10.1f}   (clipped targets)")
    pd.DataFrame(rows).to_csv(out / "official_metrics.csv", index=False)
    dfp.to_csv(out / "official_predictions.csv", index=False)
    print(f"[official] wrote official_metrics.csv and official_predictions.csv")


if __name__ == "__main__":
    main()

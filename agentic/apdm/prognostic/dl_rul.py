"""CNN-GRU RUL model — faithful port of CPdM_pipeline (create_cnn_gru_model).

Architecture, preprocessing and seed exactly as in your pipeline:
  MinMaxScaler over the feature frame, sequence_length=20 windows per unit,
  Conv1D(64, k=3, same, relu) -> GRU(100) -> Dense(50, relu) -> Dense(1),
  Adam(1e-3), MSE, seed 42. Features here are the 14 edge sensors (the
  columns observable at test time), group=Unit_ID, time=cycles, target=RUL.

Two commands:

  # train on the run-to-failure fleet (your laptop; ~minutes with GPU/CPU)
  python -m apdm.dl_rul --train --csv queries/anomalies_multimodal.csv \
      --epochs 100 --out models/cnn_gru

  # precompute one RUL hint per edge anomaly (decouples TF from the bench)
  python -m apdm.dl_rul --hints --queries queries/test_FD002_with_interpretations.csv \
      --test-txt data/test_FD002.txt --model models/cnn_gru \
      --out queries/dl_hints.csv

The bench reads dl_hints.csv; TensorFlow is only needed here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SENSORS = ["T24", "T30", "T50", "P30", "Nf", "Nc", "Ps30", "phi",
           "NRf", "NRc", "BPR", "htBleed", "W31", "W32"]
SEQ = 20
SEED = 42
RUL_CAP = 125.0   # piecewise-linear RUL, the CMAPSS convention
ROOT = Path(__file__).resolve().parents[2]

RAW_COLS = (["unit_ID", "cycles", "set1", "set2", "set3"]
            + [f"s{i}" for i in range(1, 22)])
RAW_MAP = {"T24": "s2", "T30": "s3", "T50": "s4", "P30": "s7", "Nf": "s8",
           "Nc": "s9", "Ps30": "s11", "phi": "s12", "NRf": "s13",
           "NRc": "s14", "BPR": "s15", "htBleed": "s17", "W31": "s20",
           "W32": "s21"}


def _model():
    import tensorflow as tf
    from tensorflow.keras.layers import GRU, Conv1D, Dense
    from tensorflow.keras.models import Sequential
    tf.random.set_seed(SEED)
    m = Sequential([
        Conv1D(filters=64, kernel_size=3, activation="relu", padding="same",
               input_shape=(SEQ, len(SENSORS))),
        GRU(100, return_sequences=False),
        Dense(50, activation="relu"),
        Dense(1)])
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss="mse")
    return m


def _sequences(df: pd.DataFrame):
    X, y, g = [], [], []
    for u, gd in df.groupby("Unit_ID"):
        gd = gd.sort_values("cycles")
        f = gd[SENSORS].values
        t = np.minimum(gd["RUL"].values.astype(float), RUL_CAP)
        if len(f) > SEQ:
            for i in range(len(f) - SEQ):
                X.append(f[i:i + SEQ])
                y.append(t[i + SEQ])
                g.append(int(u))
    return np.asarray(X, np.float32), np.asarray(y, np.float32), np.asarray(g)


def train(csv: Path, out: Path, epochs: int, val_units: int = 26) -> None:
    np.random.seed(SEED)
    df = pd.read_csv(csv)
    lo = df[SENSORS].min().values
    hi = df[SENSORS].max().values
    df[SENSORS] = (df[SENSORS] - lo) / np.where(hi - lo == 0, 1, hi - lo)
    units = sorted(df.Unit_ID.unique())
    rng = np.random.default_rng(SEED)
    val = set(rng.choice(units, size=val_units, replace=False).tolist())
    X, y, g = _sequences(df)
    tr = ~np.isin(g, list(val))
    import tensorflow as tf
    m = _model()
    cb = [tf.keras.callbacks.EarlyStopping(patience=8, monitor="val_loss",
                                           restore_best_weights=True)]
    m.fit(X[tr], y[tr], validation_data=(X[~tr], y[~tr]), epochs=epochs,
          batch_size=256, verbose=2, callbacks=cb)
    pred = m.predict(X[~tr], verbose=0).ravel()
    mae = float(np.abs(pred - y[~tr]).mean())
    rmse = float(np.sqrt(((pred - y[~tr]) ** 2).mean()))
    out.mkdir(parents=True, exist_ok=True)
    m.save(out / "cnn_gru.keras")
    (out / "scaler.json").write_text(json.dumps(
        {"sensors": SENSORS, "lo": lo.tolist(), "hi": hi.tolist(),
         "seq": SEQ, "val_units": sorted(val),
         "val_mae": round(mae, 2), "val_rmse": round(rmse, 2)}))
    print(f"[dl] saved -> {out}  held-out units MAE={mae:.1f} RMSE={rmse:.1f}")


def load_test_raw(test_txt: Path) -> pd.DataFrame:
    t = pd.read_csv(test_txt, sep=r"\s+", header=None)
    t = t.iloc[:, :26]
    t.columns = RAW_COLS
    d = t[["unit_ID", "cycles"]].copy()
    for k, v in RAW_MAP.items():
        d[k] = t[v]
    return d


def hints(queries: Path, test_txt: Path, model_dir: Path, out: Path) -> None:
    import tensorflow as tf
    sc = json.loads((model_dir / "scaler.json").read_text())
    lo = np.array(sc["lo"])
    hi = np.array(sc["hi"])
    m = tf.keras.models.load_model(model_dir / "cnn_gru.keras")
    raw = load_test_raw(test_txt)
    q = pd.read_csv(queries)
    u = "Unit_ID" if "Unit_ID" in q.columns else "unit_ID"
    c = "cycles" if "cycles" in q.columns else "cycle"
    an = q[q.anomaly_label == -1][[u, c]].drop_duplicates()
    rows = []
    for _, r in an.iterrows():
        g = raw[(raw.unit_ID == r[u]) & (raw.cycles <= r[c])].sort_values(
            "cycles")[SENSORS].values
        if len(g) == 0:
            continue
        if len(g) < SEQ:                      # pad-left with the first row
            g = np.vstack([np.repeat(g[:1], SEQ - len(g), 0), g])
        w = (g[-SEQ:] - lo) / np.where(hi - lo == 0, 1, hi - lo)
        rul = float(m.predict(w[None].astype(np.float32), verbose=0)[0, 0])
        rows.append({"unit_ID": int(r[u]), "cycle": int(r[c]),
                     "dl_rul_raw": round(min(max(rul, 0.0), RUL_CAP), 1)})
    h = pd.DataFrame(rows).sort_values(["unit_ID", "cycle"])
    # monotone post-processing (isotonic-style, literature standard):
    # non-increasing per unit, mono(t) = min(pred(t), mono(prev)).
    # NO elapsed subtraction: subtracting elapsed cycles compounds any
    # early error across sparse anomaly gaps and zeroes late-life hints
    # of long-lived units by construction.
    mono, prev_u = [], None
    for _, r in h.iterrows():
        v = float(r.dl_rul_raw)
        if prev_u == r.unit_ID and mono:
            v = min(v, mono[-1])
        mono.append(round(v, 1))
        prev_u = r.unit_ID
    h["dl_rul"] = mono
    h.to_csv(out, index=False)
    print(f"[dl] {len(rows)} hints -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--hints", action="store_true")
    ap.add_argument("--csv", default=str(ROOT / "queries/anomalies_multimodal.csv"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--queries", default=str(
        ROOT / "queries/test_FD002_with_interpretations.csv"))
    ap.add_argument("--test-txt", default=str(ROOT / "data/test_FD002.txt"))
    ap.add_argument("--model", default=str(ROOT / "models/cnn_gru"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.train:
        train(Path(a.csv), Path(a.out or a.model), a.epochs)
    elif a.hints:
        hints(Path(a.queries), Path(a.test_txt), Path(a.model),
              Path(a.out or (ROOT / "queries/dl_hints.csv")))
    else:
        raise SystemExit("use --train or --hints")


if __name__ == "__main__":
    main()

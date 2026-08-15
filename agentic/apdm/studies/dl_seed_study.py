"""dl_seed_study — seed robustness of the CNN-GRU RUL model.

Trains the SAME architecture (your CPdM pipeline: MinMax, window 20,
Conv1D(64,3,same)->GRU(100)->Dense(50)->Dense(1), Adam 1e-3, batch 256)
under N random seeds, holding the VALIDATION UNITS FIXED (the same 26
run-to-failure units chosen with meta-seed 42 that the production model
uses), so the seed variance isolates initialisation + SGD stochasticity.

Outputs (default evaluation/results/dl_seed_study/):
  histories.csv        per-seed, per-epoch train/val loss (learning curves)
  val_predictions.csv  per-seed predictions on every held-out sequence
  seed_summary.csv     per-seed val MAE/RMSE with 95% bootstrap CIs
                       resampled over UNITS (the honest unit-level CI)
  fig_dl_seeds.(pdf|png)  learning curves | per-seed MAE+CI | pooled
                       calibration | per-unit MAE spread across seeds

Run (laptop):
  python -m apdm.dl_seed_study --epochs 100
  python -m apdm.dl_seed_study --epochs 40 --seeds 0 1 2 3 42   # faster
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dl_rul import RUL_CAP, SENSORS, SEQ, _sequences

ROOT = Path(__file__).resolve().parent.parent


def build_model(seed: int):
    import tensorflow as tf
    from tensorflow.keras.layers import GRU, Conv1D, Dense, Input
    from tensorflow.keras.models import Sequential
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)
    m = Sequential([Input(shape=(SEQ, len(SENSORS))),
                    Conv1D(64, 3, activation="relu", padding="same"),
                    GRU(100), Dense(50, activation="relu"), Dense(1)])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return m


def unit_bootstrap(pred_df: pd.DataFrame, n_boot: int = 2000,
                   rng_seed: int = 42):
    """95% CI of MAE resampling held-out UNITS (dependent rows -> unit CI)."""
    rng = np.random.default_rng(rng_seed)
    units = pred_df.unit.unique()
    per_unit = pred_df.groupby("unit").apply(
        lambda g: float(np.abs(g.yhat - g.y).mean()))
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(units, size=len(units), replace=True)
        stats.append(float(np.mean([per_unit[u] for u in pick])))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "queries/anomalies_multimodal.csv"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 42])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "evaluation/results/dl_seed_study"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.csv)
    lo = df[SENSORS].min().values
    hi = df[SENSORS].max().values
    df[SENSORS] = (df[SENSORS] - lo) / np.where(hi - lo == 0, 1, hi - lo)
    units = sorted(df.Unit_ID.unique())
    val_units = set(np.random.default_rng(42)          # FIXED across seeds =
                    .choice(units, size=26, replace=False).tolist())  # prod split
    X, y, g = _sequences(df)
    tr = ~np.isin(g, list(val_units))
    print(f"[seed-study] sequences {X.shape} | fixed val units "
          f"{len(val_units)} ({(~tr).sum()} sequences)")

    import tensorflow as tf
    hist_rows, pred_rows, summ = [], [], []
    for sd in a.seeds:
        m = build_model(sd)
        cb = [tf.keras.callbacks.EarlyStopping(
            patience=a.patience, monitor="val_loss",
            restore_best_weights=True)]
        h = m.fit(X[tr], y[tr], validation_data=(X[~tr], y[~tr]),
                  epochs=a.epochs, batch_size=256, verbose=0, callbacks=cb)
        for ep, (l, vl) in enumerate(zip(h.history["loss"],
                                         h.history["val_loss"])):
            hist_rows.append({"seed": sd, "epoch": ep + 1,
                              "loss": l, "val_loss": vl})
        yhat = m.predict(X[~tr], verbose=0).ravel()
        pv = pd.DataFrame({"seed": sd, "unit": g[~tr],
                           "y": y[~tr], "yhat": np.clip(yhat, 0, RUL_CAP)})
        pred_rows.append(pv)
        mae = float(np.abs(pv.yhat - pv.y).mean())
        rmse = float(np.sqrt(((pv.yhat - pv.y) ** 2).mean()))
        lo_ci, hi_ci = unit_bootstrap(pv)
        summ.append({"seed": sd, "epochs_ran": len(h.history["loss"]),
                     "val_mae": round(mae, 2), "val_rmse": round(rmse, 2),
                     "unit_mae_ci_lo": round(lo_ci, 2),
                     "unit_mae_ci_hi": round(hi_ci, 2)})
        print(f"[seed {sd}] epochs={len(h.history['loss'])} "
              f"MAE={mae:.2f} [{lo_ci:.2f},{hi_ci:.2f}] RMSE={rmse:.2f}",
              flush=True)
        pd.DataFrame(hist_rows).to_csv(out / "histories.csv", index=False)
        pd.concat(pred_rows).to_csv(out / "val_predictions.csv", index=False)
        pd.DataFrame(summ).to_csv(out / "seed_summary.csv", index=False)

    S = pd.DataFrame(summ)
    print(f"\n[seed-study] MAE across seeds: {S.val_mae.mean():.2f} "
          f"± {S.val_mae.std():.2f} (range {S.val_mae.min():.2f}"
          f"-{S.val_mae.max():.2f})")
    render(out)


def render(out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MM = 1 / 25.4
    W2 = 190 * MM
    C = {"main": "#0072B2", "alt": "#D55E00", "third": "#009E73",
         "grey": "#666666", "accent": "#CC79A7", "light": "#BBBBBB"}
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 6.5, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
        "figure.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42})
    H = pd.read_csv(out / "histories.csv")
    P = pd.read_csv(out / "val_predictions.csv")
    S = pd.read_csv(out / "seed_summary.csv")
    seeds = S.seed.tolist()
    pal = [C["main"], C["third"], C["accent"], C["alt"], C["grey"]]

    fig, axes = plt.subplots(2, 2, figsize=(W2, 0.62 * W2))
    (a, b), (c, d) = axes
    for k, sd in enumerate(seeds):
        h = H[H.seed == sd]
        a.plot(h.epoch, h.loss, color=pal[k % 5], lw=0.9, alpha=0.85)
        a.plot(h.epoch, h.val_loss, color=pal[k % 5], lw=1.0, ls="--")
    a.set_yscale("log")
    a.set_xlabel("epoch")
    a.set_ylabel("MSE loss (log)")
    a.plot([], [], color=C["grey"], lw=0.9, label="train")
    a.plot([], [], color=C["grey"], lw=1.0, ls="--", label="validation")
    a.legend()
    a.text(-0.16, 1.06, "(a)", transform=a.transAxes, fontweight="bold",
           fontsize=9)
    a.set_title("Learning curves, 5 seeds (early stopping)", loc="left",
                pad=3)

    x = np.arange(len(seeds))
    b.errorbar(x, S.val_mae,
               yerr=[S.val_mae - S.unit_mae_ci_lo,
                     S.unit_mae_ci_hi - S.val_mae],
               fmt="o", ms=5, capsize=3, color=C["main"], lw=1.1)
    b.axhline(S.val_mae.mean(), color=C["light"], ls="--", lw=0.8)
    b.set_xticks(x)
    b.set_xticklabels([str(s) for s in seeds])
    b.set_xlabel("seed")
    b.set_ylabel("held-out MAE (cycles)")
    b.text(0.03, 0.05, f"mean {S.val_mae.mean():.1f} "
                       f"± {S.val_mae.std():.1f} across seeds",
           transform=b.transAxes, fontsize=6.8, color=C["grey"])
    b.text(-0.16, 1.06, "(b)", transform=b.transAxes, fontweight="bold",
           fontsize=9)
    b.set_title("Validation MAE (95% unit-bootstrap CI)", loc="left", pad=3)

    sub = P.sample(min(len(P), 4000), random_state=1)
    c.scatter(sub.y, sub.yhat, s=3, alpha=0.15, color=C["main"],
              rasterized=True)
    c.plot([0, 125], [0, 125], color=C["grey"], ls="--", lw=0.8)
    c.set_xlabel("true RUL (capped)")
    c.set_ylabel("predicted RUL")
    c.text(-0.16, 1.06, "(c)", transform=c.transAxes, fontweight="bold",
           fontsize=9)
    c.set_title("Held-out calibration (all seeds pooled)", loc="left", pad=3)

    pu = (P.assign(err=lambda t: np.abs(t.yhat - t.y))
          .groupby(["unit", "seed"]).err.mean().unstack("seed"))
    order = pu.mean(axis=1).sort_values().index
    pu = pu.loc[order]
    d.fill_between(range(len(pu)), pu.min(axis=1), pu.max(axis=1),
                   color=C["main"], alpha=0.25, label="seed range")
    d.plot(range(len(pu)), pu.mean(axis=1), color=C["main"], lw=1.1,
           label="seed mean")
    d.set_xlabel("held-out unit (sorted by MAE)")
    d.set_ylabel("unit MAE (cycles)")
    d.legend()
    d.text(-0.16, 1.06, "(d)", transform=d.transAxes, fontweight="bold",
           fontsize=9)
    d.set_title("Per-unit error: seed spread vs unit spread", loc="left",
                pad=3)

    fig.tight_layout(w_pad=2.2, h_pad=1.6)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_dl_seeds.{ext}")
    plt.close(fig)
    print(f"[seed-study] figure -> {out}/fig_dl_seeds.(pdf|png)")


if __name__ == "__main__":
    main()

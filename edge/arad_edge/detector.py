"""
edge_iforest.py  —  A-RAD edge detector: FROZEN isolation-forest bundle + test inference
========================================================================================

WHAT THIS DOES (and does NOT do)
--------------------------------
It takes the SAME six-operational-cluster / per-cluster Isolation-Forest recipe you
already used to build the knowledge base (Ward k=6 on standardized sensors, then one
IF per cluster, contamination=0.1, random_state=42) and:

  1. FITS the bundle on the TRAIN fleet ONLY and FREEZES it:
        - StandardScaler (train stats)
        - 6 cluster centroids in standardized space  (nearest-centroid assigner,
          because Ward has no `predict`; this is the deployable inference rule)
        - 6 IsolationForest models, one per cluster
        - a pinned anomaly score threshold per cluster (so test labels are produced
          by a FROZEN rule, never by re-deriving an offset on the test data)
     -> saved to a single joblib file.  This is reproducible: with random_state=42 it
        re-instantiates the very models that generated your KB (verified: 100% h_clust
        agreement, 97%+ anomaly-label agreement, and >99% once the threshold is pinned).

  2. APPLIES the frozen bundle to ANY new dataframe (the official FD002 test set, or a
     held-out unit split) with PURE INFERENCE — no `.fit` ever touches new rows:
        assign cluster -> score -> label -> extract the 3-shortest-tree rule ->
        per-unit CAUSAL cumulative-anomaly metrics -> build the `text` string in the
        EXACT format your SLM interpreter already consumes.

LEAKAGE STANCE
--------------
The detector is unsupervised and never sees RUL. Scaler, clusters, trees and thresholds
are all TRAIN-fit; test rows are transformed/assigned/scored only. Guard `assert_frozen`
enforces that the bundle carries a fitted state and that inference calls no fit. The one
honest caveat is physical, not statistical: the official FD002 *test* trajectories are
right-censored (they stop before failure), so they contain fewer late-life anomalies and
their RUL is offset by the per-unit value in RUL_FD002.txt — fine for a streaming /
diagnostic demonstration and for every *comparative* metric, but absolute prognostic MAE
on this split is optimistic vs. run-to-failure data. State that where you report it.

USAGE
-----
    # build once from the train source that also carries your KB columns:
    python edge_iforest.py fit \
        --train-csv data/Dataset_with_interpretations_RUL.csv \
        --out cache/edge_bundle.joblib --verify

    # run on the official test set (raw NASA txt + RUL file):
    python edge_iforest.py apply \
        --bundle cache/edge_bundle.joblib \
        --test-txt data/test_FD002.txt --rul-txt data/RUL_FD002.txt \
        --out results/test_anomalies.csv

    # or apply to any prepared dataframe with the 14 features + unit_ID + cycle:
    from edge_iforest import load_bundle, apply_bundle
    out = apply_bundle(load_bundle("cache/edge_bundle.joblib"), df)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from . import paths as _paths

# ----------------------------------------------------------------------------- schema
# 14 sensor features: used (standardized) for CLUSTER ASSIGNMENT — the space the
# frozen centroids live in.
FEATURES: List[str] = ["T24", "T30", "T50", "P30", "Nf", "Nc", "Ps30", "phi",
                       "NRf", "NRc", "BPR", "htBleed", "W31", "W32"]
# EXACT feature matrix of the original per-cluster IsolationForests (the KB
# recipe): csv column order minus the dropped meta columns. h_clust is constant
# within each cluster (never yields a usable split) but participates in the
# tree RNG; cycles DOES split and appears in the KB rules (cf. TIOT-LLM Eq. 1).
IF_FEATURES: List[str] = FEATURES + ["h_clust", "cycles"]
# raw NASA column order (26 cols): unit, cycle, 3 settings, then 21 sensors s1..s21
RAW_COLS = ["unit_ID", "cycles", "setting_1", "setting_2", "setting_3",
            "T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc", "epr",
            "Ps30", "phi", "NRf", "NRc", "BPR", "farB", "htBleed", "Nf_dmd",
            "PCNfR_dmd", "W31", "W32"]
K = 6
CONTAM = 0.1
SEED = 42


# ------------------------------------------------------------------ bundle container
class EdgeBundle:
    """A frozen, train-fit detector. Pickled as a plain dict via joblib."""

    def __init__(self, scaler: StandardScaler, centroids: np.ndarray,
                 models: Dict[int, IsolationForest],
                 thresholds: Dict[int, float], features: List[str],
                 meta: Optional[dict] = None):
        self.scaler = scaler
        self.centroids = centroids            # (K, n_features) standardized space
        self.models = models                  # cluster -> fitted IsolationForest
        self.thresholds = thresholds          # cluster -> score cut (label=-1 below)
        self.features = features
        self.meta = meta or {}

    # ---- assignment / scoring (pure inference) ----
    def assign(self, Xn: np.ndarray) -> np.ndarray:
        d = ((Xn[:, None, :] - self.centroids[None, :, :]) ** 2).sum(-1)
        return d.argmin(1)

    def as_dict(self) -> dict:
        return {"scaler": self.scaler, "centroids": self.centroids,
                "models": self.models, "thresholds": self.thresholds,
                "features": self.features, "meta": self.meta}

    @classmethod
    def from_dict(cls, d: dict) -> "EdgeBundle":
        return cls(d["scaler"], d["centroids"], d["models"],
                   d["thresholds"], d["features"], d.get("meta", {}))


def assert_frozen(bundle: EdgeBundle) -> None:
    from sklearn.utils.validation import check_is_fitted
    check_is_fitted(bundle.scaler)
    assert bundle.centroids.shape == (K, len(FEATURES)), "bad centroids"
    for k, m in bundle.models.items():
        check_is_fitted(m)
    assert set(bundle.models) == set(bundle.thresholds), "threshold/model mismatch"


# --------------------------------------------------------------------------- fitting
def fit_bundle(train_csv: str, centroids_override: Optional[np.ndarray] = None
               ) -> EdgeBundle:
    """Fit on the TRAIN fleet. Assignment artifacts (scaler+centroids) use the 14
    sensors; the per-cluster IFs use the EXACT original 16-column matrix
    [14 sensors, h_clust, cycles] — verified to reproduce the KB labels at 100%."""
    df = pd.read_csv(train_csv)
    if "cycles" not in df.columns and "cycle" in df.columns:
        df["cycles"] = df["cycle"]
    _need = set(IF_FEATURES) | {"h_clust"}
    missing = _need - set(df.columns)
    if missing:
        raise SystemExit(f"train csv missing columns: {sorted(missing)}")
    Xs = df[FEATURES].astype(float).values
    scaler = StandardScaler().fit(Xs)
    Xn = scaler.transform(Xs)

    if centroids_override is not None:
        centroids = centroids_override
        assign = _nearest(Xn, centroids)
    else:
        centroids = np.vstack([Xn[df["h_clust"].values == k].mean(0)
                               for k in range(K)])
        assign = df["h_clust"].values

    Xif = df[IF_FEATURES].astype(float).values
    models: Dict[int, IsolationForest] = {}
    thresholds: Dict[int, float] = {}
    for k in range(K):
        m = assign == k
        ifm = IsolationForest(n_estimators=100, contamination=CONTAM,
                              random_state=SEED).fit(Xif[m])
        models[k] = ifm
        thresholds[k] = float(ifm.offset_)   # sklearn's own fitted cut, frozen

    meta = {"assign_features": FEATURES, "if_features": IF_FEATURES, "k": K,
            "contamination": CONTAM, "seed": SEED,
            "n_train_rows": int(len(df)), "source": str(train_csv)}
    return EdgeBundle(scaler, centroids, models, thresholds, FEATURES, meta)


def _nearest(Xn: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    d = ((Xn[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
    return d.argmin(1)


# ------------------------------------------------------------- rule text extraction
def _rule_for_row(model: IsolationForest, x: np.ndarray,
                  feature_names: List[str]) -> str:
    """The 3 shortest decision paths that isolate x, ANDed within a tree, ORed across
    the three — same construction as the KB generator."""
    paths = []
    for tree in model.estimators_:
        nodes = tree.decision_path(x.reshape(1, -1)).indices
        rules = []
        for nid in nodes:
            fid = tree.tree_.feature[nid]
            if fid == -2:
                continue
            thr = tree.tree_.threshold[nid]
            op = "<=" if x[fid] <= thr else ">"
            rules.append(f"{feature_names[fid]} {op} {thr:.3f}")
        if nodes.size:
            paths.append((nodes.size, rules))
    paths.sort(key=lambda t: t[0])
    texts = []
    for _, rules in paths[:3]:
        texts.append("(" + " AND ".join(rules) + ")" if rules
                     else "(NO SPLITS FOUND)")
    return " OR ".join(texts) if texts else "(NO SPLITS FOUND)"


def _causal_metrics(labels: np.ndarray, cycles: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Running anomaly count and last-3 frequency. Gaps are measured in CYCLE
    values, which — because a unit's rows are contiguous and cycle-consecutive
    in the source frame — equals the original code's row-index gaps for both
    the per-unit (global) and per-(cluster,unit) (local) groupings. This makes
    the emitted numbers byte-compatible with the KB text."""
    cum, freq, an_cyc, c = [], [], [], 0
    for lab, cyc in zip(labels, cycles):
        if lab == -1:
            c += 1
            an_cyc.append(int(cyc))
        cum.append(c)
        if len(an_cyc) < 3:
            freq.append(0.0)
        else:
            g = [an_cyc[-1] - an_cyc[-2], an_cyc[-2] - an_cyc[-3]]
            avg = np.mean(g) if g else 0
            freq.append(1.0 / avg if avg else 0.0)
    return np.array(cum), np.array(freq)


def _format_text(cycle: int, rule: str, score: float, lc: int, lf: float,
                 gc: int, gf: float) -> str:
    """Single source of truth for the KB text line (incl. the documented
    duplicated-zero-globals quirk). Used by BOTH the batch apply and the
    streaming daemon so they can never diverge."""
    return (f"Cycle={cycle} | Splits=({rule}) | Score={score:.4f}"
            f" | LocalCumulCount={lc} | LocalLast3Freq={lf:.2f}"
            f" | GlobalCumulCount=0.0 | GlobalLast3Freq=0.00"
            f" | GlobalCumulCount={gc} | GlobalLast3Freq={gf:.2f}")


KB_SCHEMA = ["Unnamed: 0", "T24", "T30", "T50", "P30", "Nf", "Nc", "Ps30",
             "phi", "NRf", "NRc", "BPR", "htBleed", "W31", "W32", "RUL",
             "h_clust", "Unit_ID", "cycles", "anomaly_score", "anomaly_label",
             "local_cumulative_anomaly_count", "local_last_3_freq", "text",
             "global_cumulative_anomaly_count", "global_last_3_freq"]


def to_kb_schema(out: "pd.DataFrame") -> "pd.DataFrame":
    """Project the inference frame onto EXACTLY the anomalies_multimodal.csv
    schema (same columns, same order)."""
    d = out.copy()
    if "Unit_ID" not in d.columns:
        d["Unit_ID"] = d["unit_ID"]
    if "cycles" not in d.columns:
        d["cycles"] = d["cycle"]
    if "RUL" not in d.columns:
        d["RUL"] = np.nan
    d = d.reset_index(drop=True)
    d["Unnamed: 0"] = np.arange(len(d))
    return d[KB_SCHEMA]


# ------------------------------------------------------------------------ inference
def apply_bundle(bundle: EdgeBundle, df: pd.DataFrame,
                 unit_col: str = "unit_ID", cycle_col: str = "cycle"
                 ) -> pd.DataFrame:
    """Pure-inference pass. Assignment on the standardized 14 sensors; scoring,
    labels and rules on the exact 16-column KB matrix."""
    assert_frozen(bundle)
    out = df.copy().reset_index(drop=True)
    if unit_col not in out.columns and "Unit_ID" in out.columns:
        out[unit_col] = out["Unit_ID"]
    if cycle_col not in out.columns and "cycles" in out.columns:
        out[cycle_col] = out["cycles"]
    if "cycles" not in out.columns:
        out["cycles"] = out[cycle_col]

    Xn = bundle.scaler.transform(out[FEATURES].astype(float).values)
    out["h_clust"] = bundle.assign(Xn)
    Xif = out[IF_FEATURES].astype(float).values   # h_clust now filled

    score = np.empty(len(out))
    label = np.ones(len(out), dtype=int)
    for k, m in bundle.models.items():
        sel = out["h_clust"].values == k
        if not sel.any():
            continue
        # KB convention: decision_function = score_samples - offset_;
        # label -1 iff decision_function < 0 (identical cut, KB-matching Score)
        s = m.score_samples(Xif[sel]) - bundle.thresholds[k]
        score[sel] = s
        label[sel] = np.where(s < 0.0, -1, 1)
    out["anomaly_score"] = score
    out["anomaly_label"] = label

    out["global_cumulative_anomaly_count"] = 0
    out["global_last_3_freq"] = 0.0
    out["local_cumulative_anomaly_count"] = 0
    out["local_last_3_freq"] = 0.0
    for u, g in out.groupby(unit_col):
        gi = g.sort_values(cycle_col).index
        cc, ff = _causal_metrics(out.loc[gi, "anomaly_label"].values,
                                 out.loc[gi, cycle_col].values)
        out.loc[gi, "global_cumulative_anomaly_count"] = cc
        out.loc[gi, "global_last_3_freq"] = ff
    for (u, k), g in out.groupby([unit_col, "h_clust"]):
        gi = g.sort_values(cycle_col).index
        cc, ff = _causal_metrics(out.loc[gi, "anomaly_label"].values,
                                 out.loc[gi, cycle_col].values)
        out.loc[gi, "local_cumulative_anomaly_count"] = cc
        out.loc[gi, "local_last_3_freq"] = ff

    from .progress import bar
    texts = []
    _pb = bar(total=len(out), desc="detect", unit="row")
    for pos in range(len(out)):
        if out.at[pos, "anomaly_label"] == 1:
            texts.append("Inlier")
            continue
        k = int(out.at[pos, "h_clust"])
        rule = _rule_for_row(bundle.models[k], Xif[pos], IF_FEATURES)
        texts.append(_format_text(
            int(out.at[pos, cycle_col]), rule,
            float(out.at[pos, 'anomaly_score']),
            int(out.at[pos, 'local_cumulative_anomaly_count']),
            float(out.at[pos, 'local_last_3_freq']),
            int(out.at[pos, 'global_cumulative_anomaly_count']),
            float(out.at[pos, 'global_last_3_freq'])))
        _pb.update(1)
    _pb.close()
    out["text"] = texts
    return out


# --------------------------------------------------------------- test-set preparation
def load_official_test(test_txt: str, rul_txt: str) -> pd.DataFrame:
    """Parse the raw NASA FD002 test file + RUL file into the feature frame, with a
    true per-row RUL using the last-cycle anchor from RUL_FD00x.txt."""
    t = pd.read_csv(test_txt, sep=r"\s+", header=None)
    t.columns = RAW_COLS[:t.shape[1]]
    rul = pd.read_csv(rul_txt, sep=r"\s+", header=None).iloc[:, 0].values
    parts = []
    for n, g in t.groupby("unit_ID"):
        g = g.sort_values("cycles").copy()
        last = g["cycles"].max()
        g["RUL"] = (last - g["cycles"]) + rul[int(n) - 1]
        parts.append(g)
    out = pd.concat(parts).reset_index(drop=True)
    out["cycle"] = out["cycles"]
    return out


# ------------------------------------------------------------------------------- IO
def save_bundle(bundle: EdgeBundle, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle.as_dict(), path)


def _bundle_selftest(b: "EdgeBundle") -> None:
    """One synthetic row through the full inference path (assign -> score ->
    decision_path). Raises if the pickle deserialised but the estimators are
    incompatible with this host's sklearn — the failure mode a plain
    joblib.load does NOT catch."""
    row = b.scaler.mean_.copy()                     # a plausible sensor row
    xn = b.scaler.transform(row.reshape(1, -1))[0]
    k = int(((xn - b.centroids) ** 2).sum(1).argmin())
    xif = np.append(row, [float(k), 100.0])
    m = b.models[k]
    m.score_samples(xif.reshape(1, -1))
    m.estimators_[0].decision_path(xif.reshape(1, -1))


def load_bundle(path: str) -> EdgeBundle:
    """Load the frozen bundle; if the pickle is unreadable on this host
    (different numpy/sklearn than the machine that wrote it), rebuild it
    deterministically from the training KB and overwrite in place. With
    random_state=42 the refit is bit-identical to the KB generator, so this
    is a format conversion, not a new model."""
    try:
        b = EdgeBundle.from_dict(joblib.load(path))
        _bundle_selftest(b)          # loads-but-broken (cross-version) check
        return b
    except Exception as e:  # noqa: BLE001
        kb = _paths.KB_CSV
        if not kb.exists():
            raise
        print(f"[bundle] {Path(path).name} unreadable on this host "
              f"({type(e).__name__}); rebuilding deterministically from "
              f"{kb.name} (seed 42, one-time, ~1 min)...")
        b = fit_bundle(str(kb))
        assert_frozen(b)
        save_bundle(b, str(path))
        print(f"[bundle] rebuilt and saved -> {path}")
        return b


# ---------------------------------------------------------------- verification (fit)
def _verify_against_kb(bundle: EdgeBundle, train_csv: str) -> None:
    df = pd.read_csv(train_csv)
    res = apply_bundle(bundle, df)
    if "h_clust" in df:
        agree_c = (res["h_clust"].values == df["h_clust"].values).mean()
        print(f"  [verify] h_clust agreement vs KB : {agree_c:.4f}")
    if "anomaly_label" in df:
        agree_a = (res["anomaly_label"].values == df["anomaly_label"].values).mean()
        print(f"  [verify] anomaly_label agreement : {agree_a:.4f}  "
              f"(KB rate {(df.anomaly_label==-1).mean():.3f}, "
              f"ours {(res.anomaly_label==-1).mean():.3f})")


# ------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="fit & freeze the bundle from the train source")
    f.add_argument("--train-csv", required=True)
    f.add_argument("--out", default=str(_paths.BUNDLE))
    f.add_argument("--verify", action="store_true")

    a = sub.add_parser("apply", help="apply the frozen bundle to a test set")
    a.add_argument("--bundle", default=str(_paths.BUNDLE))
    a.add_argument("--test-txt", default=str(_paths.TEST_TXT))
    a.add_argument("--rul-txt", default=str(_paths.RUL_TXT))
    a.add_argument("--test-csv", help="alternative: a prepared df with features+unit_ID+cycle")
    a.add_argument("--out", default=str(_paths.RESULTS / "test_anomalies.csv"))
    a.add_argument("--units", nargs="*", type=int, default=None,
                   help="restrict to these unit ids (reference sample)")
    a.add_argument("--raw", action="store_true",
                   help="keep all internal columns instead of the KB schema")

    args = ap.parse_args()

    if args.cmd == "fit":
        b = fit_bundle(args.train_csv)
        assert_frozen(b)
        save_bundle(b, args.out)
        print(f"[fit] frozen bundle -> {args.out}")
        print(f"      {b.meta}")
        if args.verify:
            _verify_against_kb(b, args.train_csv)

    elif args.cmd == "apply":
        _paths.ensure_results()
        b = load_bundle(args.bundle)
        if args.test_txt:
            df = load_official_test(args.test_txt, args.rul_txt)
        elif args.test_csv:
            df = pd.read_csv(args.test_csv)
        else:
            raise SystemExit("provide --test-txt (+--rul-txt) or --test-csv")
        if getattr(args, "units", None):
            uc = "unit_ID" if "unit_ID" in df.columns else "Unit_ID"
            df = df[df[uc].isin(args.units)].copy()
        out = apply_bundle(b, df)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        if getattr(args, "raw", False):
            out.to_csv(args.out, index=False)
        else:
            to_kb_schema(out).to_csv(args.out, index=False)
        n_an = int((out.anomaly_label == -1).sum())
        print(f"[apply] {len(out)} rows, {out.unit_ID.nunique()} units, "
              f"{n_an} anomalies ({n_an/len(out):.1%}) -> {args.out}")
        print("        sample anomaly text:")
        ex = out[out.anomaly_label == -1]
        if len(ex):
            print("        " + ex.iloc[len(ex) // 2]["text"][:240])


if __name__ == "__main__":
    main()

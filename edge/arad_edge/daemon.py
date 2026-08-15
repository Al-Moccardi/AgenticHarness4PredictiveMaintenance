"""
edge_stream.py — online detection daemon (row-at-a-time, frozen bundle)
=======================================================================

Tails the collector's incoming.csv and, for EVERY new row, runs pure
inference with the frozen bundle: assign regime -> score -> label -> update
the causal counters incrementally -> (if anomaly) extract the rule and build
the KB-format text. Guaranteed identical to the batch `edge_iforest.py apply`
output (guard S2 checks label+text equality), because both share the same
formatter and the same cycle-gap counter definition.

Outputs (append-only, safe to tail):
    detections_stream.csv   KB schema (RUL empty until --finalize)
    queue_anomalies.jsonl   one event per anomaly: {unit,cycle,text,
                            t_arrival,t_detected} — the interpreter's queue
    stream_events.jsonl     per-row timing: {unit,cycle,t_arrival,t_detected,
                            detect_us,label}

Modes:
    python3 edge_stream.py --follow --max-idle 10     # live, exit after quiet
    python3 edge_stream.py                            # one pass over the file
    python3 edge_stream.py --finalize --rul-txt RUL_FD002.txt
                                                      # fill RUL post-hoc
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .detector import (KB_SCHEMA, IF_FEATURES, FEATURES, RAW_COLS,
                       _format_text, _rule_for_row, load_bundle)
from . import paths

HERE = paths.PROJECT_ROOT


class CausalState:
    """Incremental per-unit (global) and per-(unit,regime) (local) counters,
    cycle-gap last-3 frequency — the batch definition, maintained online."""

    def __init__(self) -> None:
        self.g: Dict[int, Tuple[int, List[int]]] = {}
        self.l: Dict[Tuple[int, int], Tuple[int, List[int]]] = {}

    @staticmethod
    def _upd(store, key, is_anom, cycle):
        cnt, last = store.get(key, (0, []))
        if is_anom:
            cnt += 1
            last = (last + [int(cycle)])[-3:]
        store[key] = (cnt, last)
        if len(last) < 3:
            f = 0.0
        else:
            avg = ((last[-1] - last[-2]) + (last[-2] - last[-3])) / 2.0
            f = 1.0 / avg if avg else 0.0
        return cnt, f

    def update(self, unit: int, regime: int, cycle: int, is_anom: bool):
        gc, gf = self._upd(self.g, unit, is_anom, cycle)
        lc, lf = self._upd(self.l, (unit, regime), is_anom, cycle)
        return lc, lf, gc, gf


def stream(a) -> None:
    paths.ensure_results()
    bundle = load_bundle(str(a.bundle))
    det_path = Path(a.out)
    q_path = Path(a.queue)
    ev_path = Path(a.events)
    if not a.resume:
        for p in (det_path, q_path, ev_path):
            p.unlink(missing_ok=True)
    if not det_path.exists():
        det_path.write_text(",".join(KB_SCHEMA) + "\n")

    state = CausalState()
    emit_ts: Dict[Tuple[int, int], float] = {}
    log = Path(a.collector_log)
    log_off = 0
    n_row = det_row = 0
    src = Path(a.incoming)
    offset = len(",".join(RAW_COLS)) + 1 if src.exists() else 0
    idle_since = time.time()
    idx = {c: i for i, c in enumerate(RAW_COLS)}
    fs_i = [idx[c] for c in FEATURES]

    with det_path.open("a") as fdet, q_path.open("a") as fq, \
            ev_path.open("a") as fev:
        while True:
            # pick up collector emission timestamps ------------------------
            if log.exists():
                txt = log.read_text()
                for line in txt[log_off:].splitlines():
                    if line.strip():
                        r = json.loads(line)
                        emit_ts[(r["unit"], r["cycle"])] = r["t_emit"]
                log_off = len(txt)
            # new raw rows -------------------------------------------------
            if not src.exists():
                time.sleep(0.05)
                if not a.follow:
                    break
                continue
            data = src.read_text()
            chunk = data[offset:]
            offset = len(data) - len(chunk) + len(chunk)  # = len(data)
            lines = [l for l in chunk.splitlines() if l.strip()]
            if not lines:
                if not a.follow:
                    break
                if time.time() - idle_since > a.max_idle:
                    break
                time.sleep(a.poll)
                continue
            idle_since = time.time()
            for line in lines:
                vals = line.split(",")
                if vals[0] == "unit_ID":
                    continue
                t_arr = time.time()
                unit = int(float(vals[idx["unit_ID"]]))
                cycle = int(float(vals[idx["cycles"]]))
                t0 = time.perf_counter()
                xs = np.array([float(vals[i]) for i in fs_i])
                xn = bundle.scaler.transform(xs.reshape(1, -1))[0]
                k = int(((xn - bundle.centroids) ** 2).sum(1).argmin())
                xif = np.append(xs, [float(k), float(cycle)])
                m = bundle.models[k]
                score = float(m.score_samples(xif.reshape(1, -1))[0]
                              - bundle.thresholds[k])
                is_anom = score < 0.0
                lc, lf, gc, gf = state.update(unit, k, cycle, is_anom)
                if is_anom:
                    rule = _rule_for_row(m, xif, IF_FEATURES)
                    text = _format_text(cycle, rule, score, lc, lf, gc, gf)
                else:
                    text = "Inlier"
                dt_us = (time.perf_counter() - t0) * 1e6
                row = {c: "" for c in KB_SCHEMA}
                row.update({"Unnamed: 0": n_row, "RUL": "",
                            "h_clust": k, "Unit_ID": unit, "cycles": cycle,
                            "anomaly_score": f"{score:.18g}",
                            "anomaly_label": -1 if is_anom else 1,
                            "local_cumulative_anomaly_count": lc,
                            "local_last_3_freq": f"{lf:.18g}",
                            "text": '"' + text.replace('"', '""') + '"',
                            "global_cumulative_anomaly_count": gc,
                            "global_last_3_freq": f"{gf:.18g}"})
                for c in FEATURES:
                    row[c] = vals[idx[c]]
                fdet.write(",".join(str(row[c]) for c in KB_SCHEMA) + "\n")
                ev = {"unit": unit, "cycle": cycle,
                      "t_arrival": emit_ts.get((unit, cycle), t_arr),
                      "t_detected": t_arr, "detect_us": round(dt_us, 1),
                      "label": -1 if is_anom else 1}
                fev.write(json.dumps(ev) + "\n")
                if is_anom:
                    fq.write(json.dumps({**ev, "h_clust": k, "text": text,
                                         "lc": lc, "lf": lf, "gc": gc,
                                         "gf": gf}) + "\n")
                n_row += 1
                det_row += is_anom
            fdet.flush()
            fq.flush()
            fev.flush()
    print(f"[stream] processed {n_row} rows, {det_row} anomalies "
          f"-> {a.out} / {a.queue}")


def finalize(a) -> None:
    """Fill the RUL column post-hoc (evaluation metadata; unknowable live)."""
    d = pd.read_csv(Path(a.out))
    rul = pd.read_csv(Path(a.rul_txt), sep=r"\s+", header=None
                      ).iloc[:, 0].values
    last = d.groupby("Unit_ID")["cycles"].max()
    d["RUL"] = [int(last[u] - c + rul[int(u) - 1])
                for u, c in zip(d.Unit_ID, d.cycles)]
    d.to_csv(Path(a.out), index=False)
    print(f"[stream] finalized RUL for {len(d)} rows in {a.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=str(paths.BUNDLE))
    ap.add_argument("--incoming", default=str(paths.RESULTS / "incoming.csv"))
    ap.add_argument("--collector-log", default=str(paths.RESULTS / "collector_log.jsonl"))
    ap.add_argument("--out", default=str(paths.RESULTS / "detections_stream.csv"))
    ap.add_argument("--queue", default=str(paths.RESULTS / "queue_anomalies.jsonl"))
    ap.add_argument("--events", default=str(paths.RESULTS / "stream_events.jsonl"))
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--max-idle", type=float, default=15.0)
    ap.add_argument("--poll", type=float, default=0.2)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--rul-txt", default=str(paths.RUL_TXT))
    a = ap.parse_args()
    if a.finalize:
        finalize(a)
    else:
        stream(a)


if __name__ == "__main__":
    main()

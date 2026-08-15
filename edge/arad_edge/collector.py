"""
stream_collect.py — the collection simulator (data "entering the Jetson")
=========================================================================

Replays the official test_FD002.txt as a gradually populating CSV, the way a
fleet would actually report: at tick t, every unit still alive emits its
cycle-t row (units interleaved). The tick duration is the time-compression
knob — the same knob later used as the offered-load axis in the statistics.

    incoming.csv        the growing raw stream (26 NASA columns, NO RUL)
    collector_log.jsonl one record per appended row: {unit, cycle, t_emit}

Usage:
    python3 stream_collect.py --tick 0.5                 # 2 fleet-cycles/s
    python3 stream_collect.py --tick 0 --max-ticks 40    # instant, first 40 cycles
    python3 stream_collect.py --units 7 15 16            # subset of units
The file is append-only and header-once, so the daemon (edge_stream.py) can
tail it concurrently.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from .detector import RAW_COLS
from . import paths

HERE = paths.PROJECT_ROOT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-txt", default=str(paths.TEST_TXT))
    ap.add_argument("--out", default=str(paths.RESULTS / "incoming.csv"))
    ap.add_argument("--log", default=str(paths.RESULTS / "collector_log.jsonl"))
    ap.add_argument("--tick", type=float, default=0.5,
                    help="seconds per fleet cycle (0 = as fast as possible)")
    ap.add_argument("--max-ticks", type=int, default=0,
                    help="stop after N cycles (0 = full trajectories)")
    ap.add_argument("--units", nargs="*", type=int, default=None)
    a = ap.parse_args()

    paths.ensure_results()
    t = pd.read_csv(Path(a.test_txt), sep=r"\s+", header=None)
    t.columns = RAW_COLS[: t.shape[1]]
    if a.units:
        t = t[t.unit_ID.isin(a.units)]
    t = t.sort_values(["cycles", "unit_ID"]).reset_index(drop=True)
    max_c = int(t.cycles.max())
    if a.max_ticks:
        max_c = min(max_c, a.max_ticks)

    out = Path(a.out)
    log = Path(a.log)
    out.unlink(missing_ok=True)
    log.unlink(missing_ok=True)
    out.write_text(",".join(RAW_COLS) + "\n")

    n = 0
    t0 = time.time()
    with out.open("a") as fo, log.open("a") as fl:
        from .progress import bar
        pb = bar(total=max_c, desc="collect", unit="cycle")
        for c in range(1, max_c + 1):
            batch = t[t.cycles == c]
            now = time.time()
            for _, r in batch.iterrows():
                fo.write(",".join(str(r[col]) for col in RAW_COLS) + "\n")
                fl.write(json.dumps({"unit": int(r.unit_ID), "cycle": c,
                                     "t_emit": now}) + "\n")
                n += 1
            fo.flush()
            fl.flush()
            pb.update(1)
            if a.tick:
                time.sleep(a.tick)
        pb.close()
    print(f"[collect] emitted {n} rows over {max_c} fleet cycles in "
          f"{time.time()-t0:.1f}s -> {a.out}")


if __name__ == "__main__":
    main()

"""Guard suite for the v8 prognostic decision layer (PA/PB/PC series).
Offline, no model needed; ~1 min. Complements apdm.smoke_test (L/F/O/K/D/
H/E/G/V), which must also stay green.

  PA  protocol invariants: frozen split membership, RUL column identity,
      outcome-derived action bands, scoring identities
  PB  leakage: shipped store meta, smoke-store retrieval, z-kNN retrieval,
      test-interpretation loader
  PC  ticket parser and deterministic verifier gates
  PD  dryrun end-to-end over all eight arms (subprocess, exactly the CLI),
      row-level resume, and report generation (tables, figures, HEADLINES)

  python -m apdm.smoke_prognosis
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SMOKE_STORE = ROOT / "cache" / "_smoke_store"
SMOKE_OUT = ROOT / "results_smoke"

FROZEN_TEST_UNITS = [7, 15, 16, 17, 20, 23, 24, 31, 32, 35, 36, 37, 42, 45,
                     48, 50, 58, 59, 61, 62, 65, 78, 98, 103, 110, 111, 114,
                     137, 138, 140, 141, 142, 161, 164, 165, 176, 177, 182,
                     183, 191, 195, 196, 203, 214, 225, 229, 230, 236, 238,
                     240, 243, 244]

OK = True
N = [0, 0]


def check(name: str, cond: bool, extra: str = "") -> None:
    global OK
    N[1] += 1
    N[0] += bool(cond)
    OK &= bool(cond)
    print(f"  [{'ok' if cond else 'FAIL'}] {name}"
          + (f"  ({extra})" if extra else ""))


def main() -> int:
    from ..data import FD002, RMAX
    from .events import severity_band
    from .faults import current_z
    from .interpret_kb import InterpretationKB
    from .prognosis import (extract_claims, load_test_interpretations,
                            parse_ticket, retrieve_semantic, retrieve_zknn,
                            run_b0, score_row, verify_ticket, winkler)
    from ..vector_store import HashEmbedder, VectorStore

    ds = FD002(seed=42)

    print("\n-- PA protocol invariants --")
    check("PA1 frozen 208/52 split (seed 42) membership",
          ds.test_units == FROZEN_TEST_UNITS
          and len(ds.train_units) == 208)
    rul_col = ds.df["RUL"].to_numpy()
    rec = np.minimum(ds.df["unit_ID"].map(ds.eol).to_numpy()
                     - ds.df["cycle"].to_numpy(), RMAX)
    check("PA2 shipped RUL column == clip(EoL - cycle, 125) row-wise",
          bool((rul_col == rec).all()), f"{len(rul_col)} rows")
    check("PA3 action bands are outcome-derived severity bands",
          [severity_band(x) for x in (120, 60, 40, 20, 5)] == [1, 2, 3, 4, 5])
    check("PA4 Winkler interval score identity",
          winkler(20, 40, 30) == 20 and winkler(20, 40, 10) == 20 + 100
          and winkler(20, 40, 50) == 20 + 100)
    over = score_row({"rul_estimate": 40, "rul_range": [30, 50],
                      "action": 3}, 30, 3)
    under = score_row({"rul_estimate": 20, "rul_range": [10, 30],
                       "action": 4}, 30, 3)
    check("PA5 S-score penalises overestimation harder (same |err|)",
          over["s_i"] > under["s_i"],
          f"{over['s_i']:.2f} vs {under['s_i']:.2f}")

    print("\n-- PB leakage --")
    train = set(ds.train_units)
    vs_meta = [json.loads(l) for l in
               (ROOT / "data" / "vector_store" / "meta.jsonl"
                ).read_text().splitlines() if l.strip()]
    check("PB1 SHIPPED vector store indexes TRAIN units only",
          all(m["unit"] in train for m in vs_meta),
          f"{len(vs_meta)} records")
    if not (SMOKE_STORE / "embeddings.npz").exists():
        vs = VectorStore.build(ds, HashEmbedder(),
                               sources=[ROOT / "data" / "interpretations"])
        vs.save(SMOKE_STORE)
    store = VectorStore.load(SMOKE_STORE)
    emb = HashEmbedder()
    u, c = FROZEN_TEST_UNITS[0], None
    an = ds.df[(ds.df["anomaly_label"] == -1) & (ds.df["unit_ID"] == u)]
    c = int(an["cycle"].iloc[len(an) // 2])
    ev = retrieve_semantic(ds, store, emb, "high T50 low NRc airflow", u, 4)
    check("PB2 semantic retrieval returns k train precedents with outcomes",
          ev["k"] == 4 and all(p["unit"] in train for p in ev["precedents"])
          and all(0 <= p["rul_then"] <= RMAX for p in ev["precedents"]))
    kb = InterpretationKB(ds, cache=ROOT / "cache" / "kb.pkl")
    ev2 = retrieve_zknn(ds, kb, u, c, 4)
    check("PB3 z-kNN twin returns k train precedents with outcomes",
          ev2["k"] == 4 and all(p["unit"] in train
                                for p in ev2["precedents"]))
    interp = load_test_interpretations(ds)
    check("PB4 test-interpretation loader: TEST units only, >=19 records",
          len(interp) >= 19
          and all(k[0] in set(FROZEN_TEST_UNITS) for k in interp),
          f"{len(interp)} records")

    print("\n-- PC ticket parser & verifier gates --")
    t = parse_ticket('noise {"rul_estimate": 41.6, "rul_range": {"lo": 28, '
                     '"hi": 60}, "action": "schedule inspection", '
                     '"cited_precedents": "u12c140 and u9c33", '
                     '"rationale": "ok"} tail')
    check("PC1 parser: dict-range, word action, ids from prose",
          t is not None and t["rul_estimate"] == 42
          and t["rul_range"] == [28, 60] and t["action"] == 3
          and "u12c140" in t["cited_precedents"])
    t2 = parse_ticket("rul_estimate: 240; rul_range: 30-55; action: 3")
    check("PC2 parser: regex fallback + clamping to [0,125]",
          t2 is not None and t2["rul_estimate"] == 125
          and t2["rul_range"] == [30, 55])
    check("PC3 parser: missing required field -> None",
          parse_ticket('{"rul_estimate": 40, "action": 2}') is None)
    zvec = current_z(ds, u, c)
    ev_c = {"k": 4, "stats": {"min": 30, "max": 60},
            "precedents": [{"id": "u1c10"}, {"id": "u2c20"}]}
    bad = {"rul_estimate": 100, "rul_range": [50, 40], "action": 4,
           "cited_precedents": ["nope"], "rationale": ""}
    v, _ = verify_ticket(bad, ev_c, zvec)
    kinds = {x.split(":")[0] for x in v}
    check("PC4 gates fire: support, range order, action, citation",
          {"support", "range", "action", "citation"} <= kinds,
          "; ".join(sorted(kinds)))
    sen = "T50"
    zfake = zvec.copy()
    from ..data import SENSORS
    zfake[SENSORS.index(sen)] = -2.5
    good = {"rul_estimate": 45, "rul_range": [32, 58], "action": 3,
            "cited_precedents": ["u1c10"],
            "rationale": f"{sen} is clearly high versus the reference."}
    v2, faith = verify_ticket(good, ev_c, zfake)
    check("PC5 contradicted claim caught; faithfulness accounts it",
          any(x.startswith("claims") for x in v2)
          and faith["contradicted"] == 1.0,
          f"claims={extract_claims(good['rationale'])}")
    b0 = run_b0(ev if ev["k"] else ev_c)
    v3, _ = verify_ticket(b0, ev, zvec)
    check("PC6 retrieval-only baseline passes its own gates",
          not v3, f"est={b0['rul_estimate']}")

    print("\n-- PD dryrun end-to-end, resume, report --")
    if SMOKE_OUT.exists():
        shutil.rmtree(SMOKE_OUT)
    cmd = [sys.executable, "-m", "apdm.prognosis", "--backend", "dryrun",
           "--per-bucket", "2", "--sample-seed", "99", "--k", "4",
           "--store-dir", str(SMOKE_STORE), "--hw-mode", "off",
           "--out", "results_smoke"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    check("PD1 runner exits cleanly over all eight arms",
          r.returncode == 0, (r.stderr or "").strip()[-160:])
    pred = SMOKE_OUT / "prog_predictions_dryrun_seed99.csv"
    df = pd.read_csv(pred) if pred.exists() else pd.DataFrame()
    check("PD2 8 arms x 6 stratified cases, all parsed, all in range",
          len(df) == 48 and not df.parse_failed.any()
          and df.pred_rul.between(0, RMAX).all()
          and df.gold_action.between(1, 5).all(),
          f"{len(df)} rows")
    if len(df):
        agent = df[df.arm.isin(["P1_direct", "P2_rag", "P3_react",
                                "P4_reflexion", "P5_verifier",
                                "P6_specialists"])]
        check("PD3 LLM arms call the backend; ReAct uses tools; "
              "no escalations on the deterministic stub",
              (agent.llm_calls >= 1).all()
              and (df[df.arm == "P3_react"].tool_calls >= 1).all()
              and not df[df.arm == "P5_verifier"].escalated.astype(
                  bool).any())
        check("PD4 baselines are model-free and cite their precedents",
              (df[df.arm.isin(["B0_retrieval", "B1_zknn"])
                  ].llm_calls == 0).all()
              and df[df.arm == "B0_retrieval"].cited_valid.astype(
                  bool).all())
    n_lines = sum(1 for _ in open(SMOKE_OUT / "prog_rows_dryrun_seed99"
                                              ".jsonl"))
    r2 = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    n_lines2 = sum(1 for _ in open(SMOKE_OUT / "prog_rows_dryrun_seed99"
                                               ".jsonl"))
    check("PD5 row-level resume: second run adds nothing",
          r2.returncode == 0 and n_lines2 == n_lines == 48
          and "resume: 48 rows" in r2.stdout)
    r3 = subprocess.run([sys.executable, "-m", "apdm.report_prognosis",
                         "--sample-seed", "99", "--out", "results_smoke"],
                        cwd=ROOT, capture_output=True, text=True)
    heads = SMOKE_OUT / "HEADLINES_seed99.md"
    check("PD6 report: summary + paired + HEADLINES generated",
          r3.returncode == 0
          and (SMOKE_OUT / "prognosis_summary_seed99.csv").exists()
          and (SMOKE_OUT / "prognosis_paired_seed99.csv").exists()
          and heads.exists() and "## H2" in heads.read_text(),
          (r3.stderr or "").strip()[-160:])
    figs = list((SMOKE_OUT / "figures").glob("*.png")) if \
        (SMOKE_OUT / "figures").exists() else []
    check("PD7 figures produced", len(figs) >= 2,
          ", ".join(f.name for f in figs))

    print(f"\n{'ALL CHECKS PASSED' if OK else 'SOME CHECKS FAILED'} "
          f"({N[0]}/{N[1]})")
    return 0 if OK else 1


if __name__ == "__main__":
    raise SystemExit(main())

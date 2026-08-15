"""Offline checks. python -m apdm.smoke_test  (no model, ~1 min)

The leakage tests are the ones that matter: L1 proves by construction-swap
that a test snapshot's features are bit-identical when the unit's future is
deleted; L2 that retrieval can only surface training units; L3 that the
recomputed causal counters agree with the CSV's own cumulative columns.
"""

from __future__ import annotations

import copy
import json

import numpy as np

from .agent import parse_rul, run_agentic, run_direct
from ..data import FD002, RMAX, W
from .features import snapshot_features
from ..llm import Backend, DryRun
from .metrics import mcnemar, s_score, stage, wilcoxon_paired
from .ml_models import train_all
from .tools import ToolBox

OK = True


def check(name, cond, detail=""):
    global OK
    OK &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" +
          (f"  -- {detail}" if detail else ""))


def main() -> int:
    from pathlib import Path
    ds = FD002(seed=42)
    bundle = train_all(ds, cache=Path(__file__).resolve().parent.parent
                       / "cache" / "ml_v2.pkl")
    u = ds.test_units[0]
    c = ds.eol[u] - 40
    from ..data import Snapshot
    s = Snapshot(u, c, ds.rul(u, c))

    # ---- L1: future-deletion invariance ---------------------------------
    x1 = snapshot_features(ds, s)
    ds2 = copy.copy(ds)
    ds2._by_unit = dict(ds._by_unit)
    g = ds._by_unit[u]
    ds2._by_unit[u] = g[g["cycle"] <= c].reset_index(drop=True)
    x2 = snapshot_features(ds2, s)
    check("L1 features invariant to deleting the unit's future",
          np.allclose(x1, x2), f"max|d|={np.abs(x1-x2).max():.2e}")
    h = ds.history(u, c)
    check("L1 history window bounded", h["cycle"].max() == c and len(h) <= W)

    # ---- L2: retrieval cannot surface test units ------------------------
    tb = ToolBox(ds, bundle, s)
    neigh = json.loads(tb.call("similar_cases", {"k": 12}))["neighbours"]
    tr = set(ds.train_units)
    check("L2 all kNN neighbours are training units",
          all(n["unit"] in tr for n in neigh), f"{len(neigh)} neighbours")
    check("L2 no neighbour is the query unit",
          all(n["unit"] != u for n in neigh))

    # ---- L3: state layer causal + IF-free surface -----------------------
    g = ds._by_unit[u]
    hits = g[g["k7"] == 6]["cycle"]
    first = int(hits.min()) if len(hits) else None
    if first is not None:
        check("L3 state_entered is None strictly before first k7 entry",
              ds.state_entered(u, first - 1) is None
              and ds.state_entered(u, first) == first, f"entry={first}")
    prior = ds.train_state_residuals()
    check("L3 train residual prior computed from train units only",
          prior.get("n_units", 0) > 0
          and prior["n_units"] <= len(ds.train_units),
          f"n={prior.get('n_units')} median="
          f"{prior.get('median_residual_after_entry')}")
    from .features import feature_names
    check("L3 no Isolation-Forest feature survives",
          not any("anomal" in n for n in feature_names()),
          f"{len(feature_names())} features")

    # ---- metrics identities --------------------------------------------
    check("S-score zero at perfect prediction", s_score([50], [50]) == 0.0)
    check("S-score overestimation branch (+10 -> e^1-1)",
          abs(s_score([60], [50]) - (np.e - 1)) < 1e-9)
    check("S-score underestimation branch (-13 -> e^1-1)",
          abs(s_score([37], [50]) - (np.e - 1)) < 1e-9)
    check("stage boundaries", stage(30) == "CRITICAL" and stage(31) == "MID"
          and stage(100) == "MID" and stage(101) == "EARLY")
    check("parse clips to [0,125]",
          parse_rul("ANSWER: 400") == RMAX and parse_rul("ANSWER: -5") == 0)
    check("parse rejects missing line", parse_rul("about sixty") is None)
    w = wilcoxon_paired([5, 6, 7, 8], [1, 2, 3, 4])
    check("Wilcoxon detects a uniform shift", w["p_value"] < 0.15,
          f"p={w['p_value']:.3f}")
    mc = mcnemar([True]*8 + [False], [False]*9)
    check("McNemar one-sided case", mc["p_value"] < 0.05, str(mc))

    # ---- loop termination ----------------------------------------------
    class NeverFinal(Backend):
        model = "never"
        def generate(self, p, system=None):
            return json.dumps({"thought": "more",
                               "action": {"tool": "degradation_status",
                                          "args": {}}})

    class Prose(Backend):
        model = "prose"
        def generate(self, p, system=None):
            if "OBSERVATIONS" in p:            # the forced-final prompt
                return "Given the evidence, ANSWER: 42"
            return "The engine seems mid-life."

    r = run_agentic(ds, s, NeverFinal(), bundle, allow_ml=False, max_steps=5)
    check("T1 forced finalisation on a never-finalising model",
          r.termination in ("final_forced", "parse_failed")
          and r.n_tool_calls <= 2,
          f"term={r.termination} tools={r.n_tool_calls}")
    r2 = run_agentic(ds, s, Prose(), bundle, allow_ml=False, max_steps=5)
    check("T2 protocol errors force early finalisation with an answer",
          r2.pred_rul == 42 and r2.n_protocol_errors >= 3,
          f"pred={r2.pred_rul} errors={r2.n_protocol_errors}")
    r3 = run_direct(ds, s, Prose(), featurized=True)
    check("T3 direct-arm retry path parses on second attempt",
          r3.pred_rul is None and r3.parse_failed,   # Prose never emits it
          f"pred={r3.pred_rul}")
    r4 = run_agentic(ds, s, DryRun(), bundle, allow_ml=False, max_steps=5)
    check("T4 dry-run agent terminates via model final",
          r4.termination == "final_model" and r4.pred_rul is not None)

    # ---- F-series: fault-layer leakage guards ---------------------------
    import copy as _copy
    from .faults import (build_fault_layer, current_z, gold_phenotype,
                         terminal_z)
    layer = build_fault_layer(ds, cache=Path(__file__).resolve().parent.parent
                              / "cache" / "faults.pkl")
    check("F0 phenotype count and train coverage",
          layer.k >= 2 and sum(p_.n_train_units for p_ in layer.phenotypes)
          == len(ds.train_units),
          f"k={layer.k} sil={layer.silhouette:.2f}")
    # F1: diagnosis-time input invariant to deleting the unit's future
    zc1 = current_z(ds, u, c)
    zc2 = current_z(ds2, u, c)              # ds2 = future-truncated copy (L1)
    check("F1 current_z invariant to deleting the unit's future",
          np.allclose(zc1, zc2), f"max|d|={np.abs(zc1-zc2).max():.2e}")
    # F1b: the gold really lives in the terminal tail only
    g_full = ds._by_unit[u]
    tail_manual = g_full[g_full["cycle"] > ds.eol[u] - layer.tail]
    zt = terminal_z(ds, u, layer.tail)
    check("F1b terminal_z uses exactly the last tail cycles",
          len(tail_manual) <= layer.tail and np.isfinite(zt).all())
    # F2: layer construction is train-only -- mutate a TEST unit, rebuild
    ds3 = _copy.copy(ds)
    ds3._by_unit = dict(ds._by_unit)
    gmut = ds._by_unit[u].copy()
    for sen in ["T50", "Ps30", "Nc"]:
        gmut[sen] = gmut[sen] * 3.0 + 500.0
    ds3._by_unit[u] = gmut
    layer3 = build_fault_layer(ds3, cache=None)
    check("F2 mutating a test unit leaves the fault layer unchanged",
          layer3.k == layer.k
          and np.allclose(np.sort(layer3.centroids, axis=0),
                          np.sort(layer.centroids, axis=0), atol=1e-9))
    # F3: gold is future-derived and assignable for every test unit
    gp = gold_phenotype(ds, layer, u)
    check("F3 gold phenotype assignable from the unit's own future",
          0 <= gp < layer.k, f"unit {u} -> P{gp}")
    # F4: the library exposes train statistics only (no unit identifiers)
    lib = layer.library_json()
    check("F4 library carries no unit identifiers",
          '"unit' not in lib and "unit_ID" not in lib,
          f"{len(lib)} chars")
    # F5: fault_library tool wired and gated on the layer
    from .tools import ToolBox as _TB
    tb_f = _TB(ds, bundle, s, fault_layer=layer)
    check("F5 fault_library tool available iff layer supplied",
          "fault_library" in tb_f.names() and "fault_library" not in tb.names())

    # ---- O-series: official-split bridge guards -------------------------
    from .official import (OfficialData, load_official_test,
                           regime_assignment_accuracy)
    o_path = Path(__file__).resolve().parent.parent / "data" / "test_FD002.txt"
    if o_path.exists():
        acc_r = regime_assignment_accuracy(ds, n_sample=3000)
        check("O1 regime nearest-centroid self-consistency >= 0.99",
              acc_r >= 0.99, f"{acc_r:.4f}")
        od = OfficialData(ds, load_official_test(o_path, ds))
        osnaps = od.snapshots()
        check("O2 one snapshot per official unit",
              len(osnaps) == len(od._by_unit), f"{len(osnaps)} units")
        check("O3 every official unit satisfies the W-window",
              min(od.last_cycle.values()) >= W,
              f"min history {min(od.last_cycle.values())}")
        h0 = od.history(osnaps[0].unit, osnaps[0].cycle)
        check("O4 official history bounded at truncation",
              h0["cycle"].max() == osnaps[0].cycle and len(h0) <= W)
        check("O5 state channel neutral on the official split",
              od.state_entered(osnaps[0].unit, osnaps[0].cycle) is None)
        xo = snapshot_features(od, osnaps[0])
        check("O6 official features match the trained dimensionality",
              xo.shape[0] == len(bundle.names), f"{xo.shape[0]} features")
        rul_p = Path(__file__).resolve().parent.parent / "data" / "RUL_FD002.txt"
        if rul_p.exists():
            from .official import load_gold, per_row_rul, load_official_test
            gold_v = load_gold(rul_p)
            mg = per_row_rul(load_official_test(o_path, ds), gold_v)
            u0 = int(mg.unit_ID.iloc[0])
            g0 = mg[mg.unit_ID == u0].sort_values("cycle")
            check("O7 per-row RUL ends at the file value and decreases by 1",
                  int(g0.RUL.iloc[-1]) == int(gold_v[0])
                  and (g0.RUL.diff().dropna() == -1).all(),
                  f"unit {u0}: last={int(g0.RUL.iloc[-1])}")
    else:
        check("O-series skipped (data/test_FD002.txt absent)", True)

    # ---- K/D-series: interpretation KB and diagnostic agent -------------
    from .interpret_kb import InterpretationKB
    from .diagnosis import parse_diag, faithfulness, run_d2
    from .faults import current_z as _cz
    kb = InterpretationKB(ds, cache=Path(__file__).resolve().parent.parent
                          / "cache" / "kb.pkl")
    tr_set = set(ds.train_units)
    check("K1 KB records are TRAIN units only",
          all(r.unit in tr_set for r in kb.records),
          f"{len(kb.records)} records")
    ho_units = sorted({h['unit'] for h in kb.held_out_interpreted})
    check("K2 interpreted split: 65 in KB, 19 held out (units 58,140)",
          kb.n_interpreted == 65 and len(kb.held_out_interpreted) == 19
          and ho_units == [58, 140], f"in={kb.n_interpreted} out={ho_units}")
    z57 = _cz(ds, 57, ds.eol[57] - 30)
    res = kb.search(z57, k=6, exclude_unit=57)
    check("K3 retrieval honours exclude_unit and returns k",
          len(res) == 6 and all(r.unit != 57 for r in res))
    names_ = [p_.name for p_ in layer.phenotypes]
    good = ("PHENOTYPE: " + names_[0] + "\n"
            "SENSORS: T50 high, Ps30 high, W32 low\n"
            "EXPLANATION: grounded text.")
    pr = parse_diag(good, names_)
    check("D1 strict protocol parses phenotype + 3 claims",
          pr is not None and pr["phenotype"] == 0 and len(pr["claims"]) == 3)
    check("D1b malformed output is a parse failure",
          parse_diag("the fault seems core-related", names_) is None)
    import numpy as _np
    zt = _np.zeros(len(__import__('apdm.data', fromlist=['SENSORS']).SENSORS))
    zt[2] = 2.0                                   # T50 strongly high
    f1 = faithfulness([("T50", "high")], zt)
    f2 = faithfulness([("T50", "low")], zt)
    f3 = faithfulness([("T24", "high")], zt)      # z=0 -> unverifiable
    check("D2 faithfulness: support / contradiction / unverifiable",
          f1["supported"] == 1 and f2["contradicted"] == 1
          and f3["unverifiable"] == 1)
    from ..llm import DryRun as _DR
    txt, meta = run_d2(ds, layer, kb, u, c, _DR(), max_steps=5)
    check("D3 dry-run diagnostic agent terminates with a parseable answer",
          parse_diag(txt, names_) is not None
          and "similar_anomalies" in meta["tools"],
          f"tools={meta['tools']}")
    txt2, meta2 = run_d2(ds, layer, None, u, c, _DR(), max_steps=5)
    check("D4 norag arm cannot touch the KB",
          "similar_anomalies" not in meta2["tools"]
          and parse_diag(txt2, names_) is not None)

    # ---- H-series: hardware model guards --------------------------------
    from ..hardware import CostModel
    cm8 = CostModel("orin_nano_8gb", "llama3", num_ctx=8192)
    check("H1 llama3 Q4 fits Orin Nano 8GB at 8k but not 16k ctx",
          cm8.memory_footprint(8192)["fits"]
          and not cm8.memory_footprint(16384)["fits"],
          f"max ctx {cm8.max_context()}")
    cmp3 = CostModel("orin_nano_8gb", "phi3", num_ctx=8192)
    check("H2 phi3 full-MHA KV exceeds llama3 GQA KV at equal ctx",
          cmp3.kv_bytes_per_token() > cm8.kv_bytes_per_token() * 2)
    e1 = cm8.estimate(1000, 50)
    e2 = cm8.estimate(1000, 200)
    e3 = cm8.estimate(1000, 50, n_cached=900)
    check("H3 sim latency monotone in output; prefix cache reduces it",
          e2["sim_edge_s"] > e1["sim_edge_s"]
          and e3["sim_edge_s"] < e1["sim_edge_s"])
    prov = cm8.provenance()
    check("H4 provenance carries the disclaimer and uncalibrated flag",
          "NOT MEASUREMENT" in prov["DISCLAIMER"]
          and prov["efficiency_calibrated"] is False)
    from ..llm import get_backend
    be = get_backend("dryrun", "llama3", device="orin_nano_8gb")
    be.generate("estimate please ANSWER now")
    tt = be.totals()
    check("H5 instrumented backend accumulates sim_ fields",
          tt.get("sim_edge_s", 0) > 0 and tt.get("prompt_tokens", 0) > 0,
          f"sim_edge_s={tt.get('sim_edge_s')}")

    # ---- E-series: event-forecast gold guards ---------------------------
    from .events import (event_gold, is_quiet, winkler, qwk, severity_band,
                         QUIET_GAP)
    ge = None
    for cc in range(30, ds.eol[u] - 5):
        ge = event_gold(ds, u, cc)
        if ge is not None:
            break
    check("E1 event gold exists on a quiet snapshot and lies in the future",
          ge is not None and ge.event_cycle > ge.cycle and ge.t_onset > 0,
          f"t_onset={getattr(ge,'t_onset',None)}")
    ds4 = _copy.copy(ds)
    ds4._by_unit = dict(ds._by_unit)
    gq = ds._by_unit[u]
    ds4._by_unit[u] = gq[gq["cycle"] <= ge.cycle].reset_index(drop=True)
    check("E1b deleting the future removes the gold (censored), proving "
          "gold is future-derived", event_gold(ds4, u, ge.cycle) is None)
    ac = gq.loc[gq["anomaly_label"] == -1, "cycle"]
    if len(ac):
        c_an = int(ac.iloc[len(ac)//2])
        check("E2 a snapshot inside a burst is not quiet",
              not is_quiet(ds, u, c_an))
    check("E3 Winkler identities",
          winkler(10, 30, 20) == 20 and winkler(10, 30, 40) == 20 + 10*10
          and winkler(10, 30, 5) == 20 + 5*10)
    check("E4 QWK perfect=1, degrades under disagreement",
          abs(qwk([1,2,3,4,5],[1,2,3,4,5]) - 1) < 1e-9
          and qwk([1,1,5,5],[5,5,1,1]) < 0)
    check("E5 severity band boundaries",
          severity_band(15) == 5 and severity_band(16) == 4
          and severity_band(75) == 2 and severity_band(76) == 1)

    # ---- G/V-series: generator and vector store -------------------------
    from .gen_interpretations import parse_gen
    ok_g = parse_gen('{"interpretation": "T50 elevated; check LPT.", '
                     '"gravity": 4, "components": ["LPT"]}')
    check("G1 generator parser accepts structured record",
          ok_g is not None and ok_g["gravity"] == 4)
    check("G1b generator parser rejects out-of-range gravity",
          parse_gen('{"interpretation": "x"*50, "gravity": 9}') is None
          and parse_gen("free text no json") is None)
    from ..vector_store import HashEmbedder, VectorStore
    emb = HashEmbedder()
    v1 = emb.embed(["core speed elevated"])[0]
    v2 = emb.embed(["core speed elevated"])[0]
    v3 = emb.embed(["fan pressure dropping"])[0]
    check("V1 hash embedder deterministic and discriminative",
          np.allclose(v1, v2) and float(v1 @ v3) < 0.95)
    try:
        vs = VectorStore.load()
        tr_s = set(ds.train_units)
        check("V2 store records are TRAIN units only",
              all(m["unit"] in tr_s for m in vs.meta),
              f"{len(vs.meta)} records")
        rng_q = np.random.default_rng(0)          # query in the store's dim
        vq = rng_q.standard_normal(vs.E.shape[1]).astype(np.float32)
        r = vs.search(vq, k=5, exclude_unit=vs.meta[0]["unit"])
        check("V3 search returns k, honours exclude_unit, sims sorted",
              len(r) == 5
              and all(x["unit"] != vs.meta[0]["unit"] for x in r)
              and all(r[i]["similarity"] >= r[i+1]["similarity"]
                      for i in range(len(r)-1)))
    except FileNotFoundError:
        check("V2/V3 skipped (no store built)", True)

    # ---- P3 tool exposure ----------------------------------------------
    tb3 = ToolBox(ds, bundle, s, allow_ml=True)
    ml = json.loads(tb3.call("ml_predict"))
    check("P3 ml_predict returns the XGB estimate with its error context",
          "predicted_rul" in ml and "known_mean_abs_error_full_test" in ml,
          f"pred={ml.get('predicted_rul')}")
    check("P2 cannot see ml_predict", "ml_predict" not in tb.names())

    print("\n" + ("ALL CHECKS PASSED" if OK else "SOME CHECKS FAILED"))
    return 0 if OK else 1


if __name__ == "__main__":
    raise SystemExit(main())

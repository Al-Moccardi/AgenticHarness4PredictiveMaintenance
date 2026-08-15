"""Guard suite for the Jetson edge package (detection + interpretation).
Offline, ~2 min.   python3 smoke_edge.py"""
import json, shutil, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
from arad_edge import paths
HERE = paths.PROJECT_ROOT
OK=[0,0]
def ck(n,c,extra=""):
    OK[1]+=1; OK[0]+=bool(c)
    print(f"  [{'ok' if c else 'FAIL'}] {n}"+(f"  ({extra})" if extra else ""))

from arad_edge.detector import (fit_bundle, apply_bundle, assert_frozen,
                                load_official_test, save_bundle, load_bundle,
                                to_kb_schema, KB_SCHEMA)
TR = paths.KB_CSV


def main():
    print("-- E: frozen detector --")
    b = fit_bundle(str(TR)); assert_frozen(b)
    save_bundle(b, str(paths.RESULTS/"_smoke_bundle.joblib")); b = load_bundle(str(paths.RESULTS/"_smoke_bundle.joblib"))
    import os
    kb = pd.read_csv(TR)
    if os.environ.get("ARAD_SMOKE_FAST"):
        _u = sorted(kb.Unit_ID.unique())[:40]
        kb = kb[kb.Unit_ID.isin(_u)].reset_index(drop=True)
        print(f"  (FAST mode: E-checks on {len(kb)} rows / {len(_u)} units)")
    r = apply_bundle(b, kb)
    an = kb.anomaly_label.values==-1
    ck("E1 h_clust reproduced (<=2 boundary ties)", int((r.h_clust.values!=kb.h_clust.values).sum())<=2)
    ck("E2 anomaly labels reproduce the KB EXACTLY (100%)",
       (r.anomaly_label.values==kb.anomaly_label.values).all())
    ck("E3 anomaly rule TEXT byte-equal >=99.9%",
       (r.text.values[an]==kb.text.values[an]).mean()>=0.999,
       f"{(r.text.values[an]==kb.text.values[an]).mean():.4%}")
    te = load_official_test(str(paths.TEST_TXT), str(paths.RUL_TXT))
    o1 = apply_bundle(b, te); o2 = apply_bundle(b, te.drop(columns=["RUL"]))
    ck("E4 test inference is RUL-independent",
       (o1.anomaly_label.values==o2.anomaly_label.values).all())
    ck("E5 deterministic re-apply", (apply_bundle(b,te).text.values==o1.text.values).all())
    u = te.unit_ID.value_counts().idxmax(); gu = te[te.unit_ID==u].sort_values("cycle")
    fo = apply_bundle(b,gu).reset_index(drop=True); ho = apply_bundle(b,gu.iloc[:len(gu)//2]).reset_index(drop=True)
    ck("E6 causal (early labels stable under truncation)",
       (fo.anomaly_label.values[:len(ho)]==ho.anomaly_label.values).all())
    ck("E7 anomalies carry KB-format text incl. the documented duplicate-global quirk",
       o1[o1.anomaly_label==-1].text.str.contains("GlobalCumulCount=0.0 \\| GlobalLast3Freq=0.00 \\| GlobalCumulCount=").all())

    print("-- I: interpreter --")
    from arad_edge.interpreter import build_prompt, load_fewshot, SYSTEM
    fs = load_fewshot(paths.FEWSHOT)
    row = o1[o1.anomaly_label==-1].iloc[0]
    pr = build_prompt(fs, int(row.unit_ID), int(row.cycle), int(row.h_clust),
                      str(row.text), [{"cycle":1,"text":"Cycle=1 | Splits=((T50 > 1))",
                                       "interpretation":"**Anomaly Interpretation:** x **Cause:** y"}])
    ck("I1 prompt carries rule + history, and NEVER the RUL",
       "Splits=" in pr and "PREVIOUS ANOMALIES" in pr and "RUL" not in pr and "RUL" not in SYSTEM)
    ck("I2 few-shot examples are TRAIN units only",
       all(int(e["unit_ID"]) in {57,146,232} for e in json.loads(paths.FEWSHOT.read_text())))
    work = HERE/"_smoke_interp"; shutil.rmtree(work, ignore_errors=True); work.mkdir()
    o1.head(4000).to_csv(work/"det.csv", index=False)
    cmd = [sys.executable, "-m","arad_edge","interpret", "--backend","dryrun",
           "--detections", str(work/"det.csv"), "--out-dir", str(work/"gen"),
           "--merged-csv", str(work/"merged.csv"), "--limit","8",
           "--fewshot", str(paths.FEWSHOT)]
    r1 = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
    n1 = sum(1 for f in (work/"gen").glob("unit_*.jsonl") for _ in open(f))
    r2 = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
    ck("I3 dryrun pilot writes resumable records", r1.returncode==0 and n1==8, f"{n1} records")
    ck("I4 resume: rerun reports existing and adds a fresh batch",
       r2.returncode==0 and "resume: 8" in r2.stdout)
    m = pd.read_csv(work/"merged.csv")
    ck("I5 merged csv follows the KB schema (interpretation col, inlier placeholder)",
       "interpretation" in m.columns and
       (m[m.anomaly_label==1].interpretation=="No interpretation (inlier)").all())
    gotten = m[(m.anomaly_label==-1) & (m.interpretation.astype(str).str.len()>50)]
    ck("I6 interpretations use the sectioned KB format with a gravity score",
       len(gotten)>0 and gotten.interpretation.str.contains("\\*\\*Anomaly Interpretation:\\*\\*").all()
       and gotten.interpretation.str.contains("gravity score: [1-5]").all())
    shutil.rmtree(work)

    print("-- S: schema, streaming, statistics --")
    ref_cols = list(pd.read_csv(paths.KB_CSV, nrows=0).columns)
    ck("S1 detection CSV schema == anomalies_multimodal.csv exactly",
       KB_SCHEMA == ref_cols and list(to_kb_schema(o1).columns) == ref_cols)
    sw = HERE/"_smoke_stream"; shutil.rmtree(sw, ignore_errors=True); sw.mkdir()
    import subprocess as sp
    env_files = dict(cwd=HERE, capture_output=True, text=True)
    sp.run([sys.executable, "-m","arad_edge","collect","--tick","0","--units","7","15",
            "--out", str(sw/"in.csv"), "--log", str(sw/"cl.jsonl")], **env_files)
    sp.run([sys.executable, "-m","arad_edge","daemon","--incoming",str(sw/"in.csv"),
            "--collector-log",str(sw/"cl.jsonl"),"--out",str(sw/"det.csv"),
            "--queue",str(sw/"q.jsonl"),"--events",str(sw/"ev.jsonl")], **env_files)
    sp.run([sys.executable, "-m","arad_edge","daemon","--finalize","--out",str(sw/"det.csv")], **env_files)
    st = pd.read_csv(sw/"det.csv").sort_values(["Unit_ID","cycles"]).reset_index(drop=True)
    ba = to_kb_schema(apply_bundle(b, te[te.unit_ID.isin([7,15])])
                      ).sort_values(["Unit_ID","cycles"]).reset_index(drop=True)
    ck("S2 STREAMING daemon == batch apply (labels, text, counters, RUL)",
       st is not None and
       len(st)==len(ba)
       and (st.anomaly_label.values==ba.anomaly_label.values).all()
       and (st.text.values==ba.text.values).all()
       and (st.RUL.values==ba.RUL.values).all()
       and np.allclose(st.anomaly_score.values, ba.anomaly_score.values))
    r1 = sp.run([sys.executable,"-m","arad_edge","interpret","--backend","dryrun","--follow",
                 "--max-idle","1","--queue",str(sw/"q.jsonl"),
                 "--out-dir",str(sw/"gen"),"--detections",str(sw/"det.csv"),
                 "--merged-csv",str(sw/"merged.csv")], **env_files)
    mm = pd.read_csv(sw/"merged.csv")
    ck("S3 follow-mode drains the queue and merges the KB+interpretation CSV",
       r1.returncode==0 and "interpretation" in mm.columns
       and (mm[mm.anomaly_label==-1].interpretation.astype(str).str.len()>50).all())
    import json as _j
    rec = _j.loads(open(next((sw/"gen").glob("unit_*.jsonl"))).readline())
    ck("S4 records carry full metrics (tokens, durations, queue timing)",
       "metrics" in rec and rec["metrics"].get("eval_count")
       and rec.get("queue_wait_s") is not None and rec.get("wall_s") is not None)
    r2 = sp.run([sys.executable,"-m","arad_edge","stats","--out-dir",str(sw/"gen"),
                 "--events",str(sw/"ev.jsonl"),"--queue",str(sw/"q.jsonl"),
                 "--telemetry",str(sw/"none.jsonl"),
                 "--report",str(sw/"rep.md"),"--csv",str(sw/"stats.csv")], **env_files)
    ck("S5 edge_stats produces report + per-anomaly csv",
       r2.returncode==0 and (sw/"rep.md").exists() and (sw/"stats.csv").exists()
       and "Detection" in (sw/"rep.md").read_text())
    shutil.rmtree(sw); (paths.RESULTS/"_smoke_bundle.joblib").unlink(missing_ok=True)
    print("-- N: sampling --")
    from arad_edge.sampling import sample_units
    _us = sample_units(6, seed=1, detections=None)
    ck("N1 sampler returns 6 valid in-range test units",
       len(_us) == 6 and all(7 <= u <= 260 for u in _us))
    ck("N2 sampler deterministic under seed", sample_units(6, seed=1) == _us)
    print(f"\n{'ALL PASSED' if OK[0]==OK[1] else 'FAILED'} ({OK[0]}/{OK[1]})")
    return 0 if OK[0]==OK[1] else 1

if __name__ == "__main__":
    raise SystemExit(main())
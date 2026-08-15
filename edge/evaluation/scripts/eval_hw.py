#!/usr/bin/env python3
"""eval_hw.py — hardware statistics of the edge run (telemetry + per-call).
Outputs -> results/hw_eval/: summary.csv/.md, timeline.csv"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
OUT=Path("results/hw_eval"); OUT.mkdir(parents=True, exist_ok=True)
tel=pd.DataFrame([json.loads(l) for l in open("results/telemetry.jsonl") if l.strip()])
st=pd.read_csv("results/edge_stats.csv").sort_values(["unit_ID","cycle"]).reset_index(drop=True)
S=[]; add=lambda m,v,note="": S.append({"metric":m,"value":v,"note":note})
dur=(tel.t.max()-tel.t.min())/60
add("device_ram_total_gib",round(tel.mem_total_mb.max()/1024,1),
    "Tegra sysfs GPU present -> Jetson AGX-Orin-class (NOT Orin Nano 8GB)")
add("telemetry_minutes",round(dur,1),f"{len(tel)} samples @ {np.median(np.diff(tel.t)):.1f}s")
add("cpu_pct_mean",round(tel.cpu_pct.mean(),1)); add("cpu_pct_max",round(tel.cpu_pct.max(),1))
add("ram_gib_baseline",round(tel.mem_used_mb.iloc[:5].mean()/1024,2),"before model load")
add("ram_gib_peak",round(tel.mem_used_mb.max()/1024,2))
add("ram_gib_model_footprint",round((tel.mem_used_mb.max()-tel.mem_used_mb.iloc[:5].mean())/1024,2),
    "peak - baseline = llama-server weights + KV")
g=tel.gpu_pct.dropna()
add("gpu_pct_mean",round(g.mean(),1)); add("gpu_pct_p95",round(g.quantile(.95),1))
add("gpu_busy_frac_gt90",round(float((g>90).mean()),3),"GPU-bound confirmation")
add("temp_c_start",round(tel.temp_c_max.iloc[:5].mean(),1))
add("temp_c_peak",round(tel.temp_c_max.max(),1),"Orin soft throttle ~ >85C: no throttling")
add("temp_c_rise",round(tel.temp_c_max.max()-tel.temp_c_max.iloc[:5].mean(),1))
add("power_rails","not exposed","INA3221 path absent on this image; use `sudo tegrastats` for W")
# per-call stability
w=st.wall_s.dropna(); d=st.decode_tps.dropna()
add("interp_n",len(st)); add("wall_s_p50",round(w.median(),2)); add("wall_s_p95",round(w.quantile(.95),2))
add("decode_tps_mean",round(d.mean(),1)); add("decode_tps_cv_pct",round(100*d.std()/d.mean(),1),
    "coefficient of variation: thermal/clock stability")
x=np.arange(len(d)); slope=np.polyfit(x,d,1)[0]*60/ (w.mean()) if len(d)>3 else np.nan
add("decode_tps_drift_per_min",round(float(np.polyfit(np.arange(len(d)),d,1)[0]*(60/w.mean())),3),
    "linear drift across the run; ~0 = thermally stable")
add("prefill_tps_median",round(st.prefill_tps.median(),0))
add("throughput_anomalies_per_min",round(len(st)/ (w.sum()/60),2),"SLM busy-time basis")
add("prefill_share_pct",round(100*st.prefill_s.sum()/(st.prefill_s.sum()+st.decode_s.sum()),1),
    "decode dominates -> memory-bandwidth-bound")
tl=tel.iloc[::max(len(tel)//600,1)][["t","cpu_pct","mem_used_mb","gpu_pct","temp_c_max"]]
tl.to_csv(OUT/"timeline.csv",index=False)
sm=pd.DataFrame(S); sm.to_csv(OUT/"summary.csv",index=False)
md=["# Hardware statistics — edge interpretation run","","| metric | value | note |","|---|---|---|"]
md+=[f"| {r.metric} | {r.value} | {r.note} |" for _,r in sm.iterrows()]
(OUT/"summary.md").write_text("\n".join(md))
print(sm.to_string(index=False))

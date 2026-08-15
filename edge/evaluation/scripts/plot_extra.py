#!/usr/bin/env python3
"""Figures 7-9: IF deep-dive, SLM behaviour, hardware. Same journal style."""
from pathlib import Path
import json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
MM=1/25.4; W2=190*MM
C={"main":"#0072B2","alt":"#D55E00","third":"#009E73","grey":"#666666",
   "light":"#BBBBBB","accent":"#CC79A7"}
plt.rcParams.update({"font.family":"serif","font.size":8,"axes.labelsize":8,
 "axes.titlesize":8.5,"xtick.labelsize":7,"ytick.labelsize":7,
 "legend.fontsize":7,"axes.linewidth":0.6,"lines.linewidth":1.2,
 "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,
 "grid.alpha":0.25,"legend.frameon":False,"figure.dpi":300,
 "savefig.bbox":"tight","pdf.fonttype":42})
def panel(ax,l,t=""):
    ax.text(-0.16,1.06,f"({l})",transform=ax.transAxes,fontweight="bold",
            fontsize=9,va="bottom")
    if t: ax.set_title(t,loc="left",pad=3)
def save(fig,name,out):
    out.mkdir(parents=True,exist_ok=True)
    for e in ("pdf","png"): fig.savefig(out/f"{name}.{e}")
    plt.close(fig); print(" ",name)
OUT=Path("results/figures/paper")

# ================= fig 7: IF deep dive =================
su=pd.read_csv("results/if_eval_extra/sensor_usage.csv")
uni=pd.read_csv("results/if_eval_extra/univariate_auc.csv")
tc=pd.read_csv("results/if_eval_extra/threshold_curve.csv")
sr=pd.read_csv("results/if_eval_extra/seed_robustness.csv")
sm=pd.read_csv("results/if_eval_extra/summary.csv").set_index("metric")
fig,axes=plt.subplots(2,2,figsize=(W2,0.62*W2)); (a,b),(c,d)=axes
# (a) sensor-usage heatmap
piv=su.pivot_table(index="sensor",columns="regime",values="share").fillna(0)
piv=piv.loc[piv.mean(1).sort_values(ascending=False).index]
im=a.imshow(piv.values,aspect="auto",cmap="Blues",vmin=0,vmax=piv.values.max())
a.set_xticks(range(piv.shape[1])); a.set_xticklabels([f"R{c_}" for c_ in piv.columns])
a.set_yticks(range(piv.shape[0])); a.set_yticklabels(piv.index,fontsize=6)
cb=fig.colorbar(im,ax=a,fraction=0.045,pad=0.02); cb.set_label("share of rules",fontsize=6.5)
cb.ax.tick_params(labelsize=6); a.grid(False)
panel(a,"a","Which sensors the rules split on")
# (b) univariate vs multivariate
top=uni.head(8)
b.barh(np.arange(len(top))[::-1],top.auc,0.62,color=C["third"],alpha=0.9)
b.set_yticks(np.arange(len(top))[::-1]); b.set_yticklabels(top.sensor,fontsize=6.5)
for v,lab,col in ((0.9036,"regime IF",C["main"]),(0.9291,"regime distance",C["accent"])):
    b.axvline(v,color=col,lw=1.1); b.text(v,len(top)-0.4,f" {lab} {v:.3f}",
        color=col,fontsize=6,rotation=90,va="top")
b.set_xlim(0.5,1.0); b.set_xlabel("AUC (degraded vs healthy)")
b.grid(axis="y",alpha=0)
panel(b,"b","Single sensors vs regime-conditional detectors")
# (c) operating curve
c.plot(tc.healthy_fa_rate*100,tc.lead_median,"o-",color=C["main"],ms=3.6)
for _,r in tc.iterrows():
    c.annotate(f"{r.alarm_frac:.0%}",(r.healthy_fa_rate*100,r.lead_median),
               textcoords="offset points",xytext=(4,3),fontsize=5.6,color=C["grey"])
dep=tc[tc.alarm_frac==0.10].iloc[0]
c.plot(dep.healthy_fa_rate*100,dep.lead_median,"s",color=C["alt"],ms=6)
c.annotate("deployed (10%)",xy=(dep.healthy_fa_rate*100,dep.lead_median),
           xytext=(5.5,12),fontsize=6.4,color=C["alt"],
           arrowprops=dict(arrowstyle="->",lw=0.6,color=C["alt"]))
c.set_xlabel("false-alarm rate in healthy zone (%)")
c.set_ylabel("median degradation lead time (cycles)")
panel(c,"c","Sensitivity is a dial, not a fact")
# (d) seed robustness
d.plot(sr.seed.astype(str),sr.auc,"o",color=C["main"],ms=5)
d.axhline(sr.auc.mean(),color=C["light"],ls="--",lw=0.8)
d.set_ylim(0.85,0.96); d.set_xlabel("random seed"); d.set_ylabel("AUC")
ag=sm.loc["seed_label_agreement_mean","value"]
d.text(0.03,0.06,f"pairwise label agreement {float(ag):.1%}",
       transform=d.transAxes,fontsize=6.6,color=C["grey"])
panel(d,"d","Stable under the RNG")
fig.tight_layout(w_pad=2.2,h_pad=1.6); save(fig,"fig7_if_deepdive",OUT)

# ================= fig 8: SLM behaviour =================
sec=pd.read_csv("results/slm_eval_extra/sections.csv")
rec=pd.read_csv("results/slm_eval_extra/recommendations.csv")
sim=pd.read_csv("results/slm_eval_extra/similarity_pairs.csv")
sme=pd.read_csv("results/slm_eval_extra/summary.csv").set_index("metric")
fig,axes=plt.subplots(2,2,figsize=(W2,0.62*W2)); (a,b),(c,d)=axes
# (a) section lengths
y=np.arange(len(sec))[::-1]
a.barh(y,sec.mean_words,0.6,xerr=sec.sd,color=C["main"],alpha=0.9,
       error_kw=dict(lw=0.8,ecolor=C["grey"]))
a.set_yticks(y); a.set_yticklabels(sec.section,fontsize=6.3)
a.set_xlabel("words per section (mean ± sd)"); a.grid(axis="y",alpha=0)
panel(a,"a","Where the words go")
# (b) action x gravity (the collapse)
ct=pd.crosstab(rec.gravity,rec.action)
order=["monitor","inspect","schedule maintenance","repair","replace","shutdown/stop","other"]
cols_map={"monitor":C["third"],"inspect":"#56B4E9","schedule maintenance":"#F0E442",
          "repair":C["accent"],"replace":"#999999","shutdown/stop":C["alt"],"other":C["light"]}
bot=np.zeros(len(ct))
for act in [o for o in order if o in ct.columns]:
    b.bar(ct.index.astype(int).astype(str),ct[act],0.62,bottom=bot,
          color=cols_map[act],label=act)
    bot+=ct[act].values
b.set_xlabel("gravity score"); b.set_ylabel("anomalies")
b.legend(fontsize=5.6,ncols=1,loc="center right")
share=(rec.action=="shutdown/stop").mean()
b.text(0.03,0.93,f"{share:.0%} of ALL recommendations\nare shutdown-class",
       transform=b.transAxes,fontsize=6.6,va="top",color=C["alt"])
panel(b,"b","Action policy collapses to 'stop'")
# (c) pairwise similarity
c.hist(sim.jaccard3,bins=40,color=C["main"],alpha=0.8,edgecolor="white",lw=0.2)
c.axvline(sim.jaccard3.mean(),color=C["alt"],lw=1.1)
c.text(sim.jaccard3.mean(),c.get_ylim()[1]*0.95,
       f" mean {sim.jaccard3.mean():.2f}",color=C["alt"],fontsize=6.6,va="top")
c.set_xlabel("pairwise word-3-gram Jaccard between interpretations")
c.set_ylabel("pairs")
c.text(0.97,0.8,f"near-duplicates (>0.5): {(sim.jaccard3>0.5).mean():.1%}",
       transform=c.transAxes,ha="right",fontsize=6.4,color=C["grey"])
panel(c,"c","No template collapse")
# (d) escalation + history usage summary bars
vals=[("history referenced\nwhen prior>0",float(sme.loc["trend_mentions_history_when_prior>0","value"])),
      ("gravity-urgency\nalignment (rho)",float(sme.loc["spearman_gravity_vs_action_urgency","value"])),
      ("hard action when\ngravity>=4",float(sme.loc["hard_action_rate_when_gravity_ge4","value"]))]
y=np.arange(len(vals))[::-1]
d.barh(y,[v for _,v in vals],0.58,color=[C["main"],C["accent"],C["alt"]])
for yy,(lab,v) in zip(y,vals):
    d.text(min(v+0.03,1.02),yy,f"{v:.2f}",va="center",fontsize=7)
d.set_yticks(y); d.set_yticklabels([l for l,_ in vals],fontsize=6.3)
d.set_xlim(0,1.15); d.grid(axis="y",alpha=0)
d.set_xlabel("rate / correlation")
panel(d,"d","Behavioural summary")
fig.tight_layout(w_pad=2.2,h_pad=1.6); save(fig,"fig8_slm_behaviour",OUT)

# ================= fig 9: hardware =================
tel=pd.read_csv("results/hw_eval/timeline.csv")
st=pd.read_csv("results/edge_stats.csv")
hw=pd.read_csv("results/hw_eval/summary.csv").set_index("metric")
fig,axes=plt.subplots(2,2,figsize=(W2,0.62*W2)); (a,b),(c,d)=axes
tt=(tel.t-tel.t.min())/60
# (a) GPU + temp
a.plot(tt,tel.gpu_pct,color=C["main"],lw=0.8)
a.set_ylabel("GPU (%)",color=C["main"]); a.tick_params(axis="y",colors=C["main"])
a.set_ylim(0,105)
a2=a.twinx(); a2.plot(tt,tel.temp_c_max,color=C["alt"],lw=1.0)
a2.set_ylabel("max temp (°C)",color=C["alt"]); a2.tick_params(axis="y",colors=C["alt"])
a2.grid(False); a2.spines["right"].set_visible(True); a2.set_ylim(50,90)
a2.axhline(85,color=C["alt"],ls=":",lw=0.8)
a2.text(tt.max()*0.99,85.5,"throttle",color=C["alt"],fontsize=5.8,ha="right")
a.set_xlabel("time (min)")
panel(a,"a",f"GPU-bound, thermally safe (peak {tel.temp_c_max.max():.0f}°C)")
# (b) RAM
b.plot(tt,tel.mem_used_mb/1024,color=C["main"],lw=1.1)
base=tel.mem_used_mb.iloc[:3].mean()/1024; peak=tel.mem_used_mb.max()/1024
b.axhline(base,color=C["light"],ls=":",lw=0.8); b.axhline(peak,color=C["light"],ls=":",lw=0.8)
b.annotate(f"model + KV\n+{peak-base:.1f} GiB",xy=(tt.iloc[len(tt)//5],peak-0.4),
           fontsize=6.4,color=C["grey"])
b.set_xlabel("time (min)"); b.set_ylabel("RAM used (GiB)")
panel(b,"b","Memory footprint of the SLM tier")
# (c) decode stability
dts=st.decode_tps.dropna().reset_index(drop=True)
c.plot(dts.index,dts,"o",ms=2.2,alpha=0.55,color=C["main"])
roll=dts.rolling(9,center=True,min_periods=3).mean()
c.plot(dts.index,roll,color=C["alt"],lw=1.2)
cv=float(hw.loc["decode_tps_cv_pct","value"])
c.text(0.03,0.08,f"CV {cv:.1f}% — no thermal drift",transform=c.transAxes,
       fontsize=6.6,color=C["grey"])
c.set_xlabel("interpretation # (chronological)"); c.set_ylabel("decode tok/s")
panel(c,"c","Sustained throughput is flat")
# (d) time budget
pf,dc=st.prefill_s.sum(),st.decode_s.sum()
other=max(st.wall_s.sum()-pf-dc,0)
d.barh([0],[pf],0.5,color=C["third"],label=f"prefill {pf/(pf+dc+other):.0%}")
d.barh([0],[dc],0.5,left=[pf],color=C["main"],label=f"decode {dc/(pf+dc+other):.0%}")
d.barh([0],[other],0.5,left=[pf+dc],color=C["light"],label="other")
d.set_yticks([]); d.set_xlabel("total seconds across the run")
d.legend(loc="lower center",ncols=3,fontsize=6.2)
thr=float(hw.loc["throughput_anomalies_per_min","value"])
d.text(0.5,0.86,f"{thr:.1f} anomalies/min sustained  ·  "
       f"1,062-event full test set ≈ {1062/thr/60:.1f} h",
       transform=d.transAxes,ha="center",fontsize=7,color=C["grey"])
d.set_ylim(-1.0,0.9); d.grid(False)
panel(d,"d","Decode-bound time budget")
fig.tight_layout(w_pad=2.4,h_pad=1.6); save(fig,"fig9_hardware",OUT)

#!/usr/bin/env python3
"""
eval_slm_extra.py — behavioural evaluation of the edge SLM (beyond grounding)
=============================================================================
  diversity        template-collapse check: pairwise word-3-gram Jaccard between
                   interpretations, distinct-token ratio, most-repeated phrases
  structure        words per section (the six required sections)
  recommendations  action taxonomy of the Recommendation section, crossed with
                   the gravity score (does urgency align with severity?)
  temporal use     does the Trend section actually reference history when the
                   unit HAS prior anomalies?
  escalation       within-unit: does gravity rise as the unit ages?
  readability      words/sentence, word length (technician-facing text)

Outputs -> results/slm_eval_extra/: summary.csv/.md, sections.csv,
recommendations.csv, gravity_action.csv, similarity_pairs.csv,
top_phrases.csv, per_unit_escalation.csv
"""
from __future__ import annotations
import json, re
from collections import Counter
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

MERGED = "results/test_FD002_with_interpretations.csv"
OUT = Path("results/slm_eval_extra"); OUT.mkdir(parents=True, exist_ok=True)
SEC = ["Anomaly Interpretation","Cause","Impact","Anomalous Trend",
       "Expected Future Failures","Recommendation"]
STOP = set("the a an of to and or in on for is are be with this that it as by "
           "at from may can could which".split())
ACTIONS = [("shutdown/stop", r"\b(shut\s?down|stop the (engine|machine)|halt|ground(ing)? the)\b"),
           ("replace", r"\breplace(ment)?\b"),
           ("repair", r"\brepair\b|\boverhaul\b"),
           ("inspect", r"\binspect(ion)?\b|\bexamine\b|\bcheck\b"),
           ("schedule maintenance", r"\bschedul\w+\b|\bmaintenance window\b|\bnext (opportunity|window)\b"),
           ("monitor", r"\bmonitor(ing)?\b|\bcontinue operation\b|\bobserve\b|\bwatch\b")]
HIST_KW = r"\b(previous|prior|earlier|past|recurring|again|pattern|cumulative|count(er)?s?)\b"

def sections(t):
    out={}
    for i,s in enumerate(SEC):
        m=re.search(rf"\*\*{re.escape(s)}:\*\*(.*?)(?=\*\*|$)", t, re.S)
        out[s]=m.group(1).strip() if m else ""
    return out

def words(t): return [w for w in re.findall(r"[a-zA-Z]{3,}", t.lower()) if w not in STOP]

d=pd.read_csv(MERGED); u="Unit_ID"; c="cycles"
an=d[(d.anomaly_label==-1)&d.interpretation.astype(str).str.len().gt(50)].copy()
an=an.sort_values([u,c]).reset_index(drop=True)
an["n_prior"]=an.groupby(u).cumcount()
texts=an.interpretation.astype(str).tolist()
secs=[sections(t) for t in texts]
bodies=[" ".join(x.values()) for x in secs]
S=[]; add=lambda m,v,n=np.nan,note="": S.append({"metric":m,"value":v,"n":n,"note":note})

# ---- diversity / template collapse -----------------------------------------
grams=[set(zip(w:=words(t), w[1:], w[2:])) for t in bodies]
sims=[len(a&b)/max(len(a|b),1) for a,b in combinations(grams,2)]
sims=np.array(sims)
pd.DataFrame({"jaccard3":sims}).to_csv(OUT/"similarity_pairs.csv",index=False)
add("pairwise_3gram_jaccard_mean",round(float(sims.mean()),4),len(sims))
add("pairwise_3gram_jaccard_p95",round(float(np.percentile(sims,95)),4),len(sims))
add("pairwise_3gram_jaccard_max",round(float(sims.max()),4),len(sims),
    "near-duplicate pair share:"
    f" {(sims>0.5).mean():.1%}")
allw=[w for t in bodies for w in words(t)]
add("distinct_token_ratio",round(len(set(allw))/len(allw),4),len(allw))
ph=Counter()
for t in bodies:
    w=words(t)
    ph.update(tuple(w[i:i+5]) for i in range(len(w)-4))
top=pd.DataFrame([{"phrase":" ".join(k),"count":v} for k,v in ph.most_common(15)])
top.to_csv(OUT/"top_phrases.csv",index=False)
add("top_phrase_share",round(top['count'].iloc[0]/len(texts),3),len(texts),
    f"'{top['phrase'].iloc[0]}'")

# ---- section structure ------------------------------------------------------
srows=[{"section":s,"mean_words":np.mean([len(words(x[s])) for x in secs]),
        "sd":np.std([len(words(x[s])) for x in secs]),
        "empty":sum(1 for x in secs if not x[s])} for s in SEC]
pd.DataFrame(srows).round(2).to_csv(OUT/"sections.csv",index=False)
add("total_words_mean",round(float(np.mean([len(words(t)) for t in texts])),1),len(texts))
sent=[max(len(re.findall(r"[.!?]+",t)),1) for t in texts]
add("words_per_sentence_mean",round(float(np.mean(
    [len(words(t))/s_ for t,s_ in zip(texts,sent)])),1),len(texts))

# ---- recommendations vs gravity --------------------------------------------
def act_of(rec):
    for name,pat in ACTIONS:
        if re.search(pat,rec,re.I): return name
    return "other"
an["action"]=[act_of(x["Recommendation"]) for x in secs]
grav=[int(m.group(1)) if (m:=re.search(r"gravity score:\s*([1-5])",t)) else np.nan
      for t in texts]
an["gravity"]=grav
an[["Unit_ID","cycles","RUL","gravity","action","n_prior"]].to_csv(
    OUT/"recommendations.csv",index=False)
ct=pd.crosstab(an.gravity,an.action)
ct.to_csv(OUT/"gravity_action.csv")
URG={"monitor":1,"schedule maintenance":2,"inspect":3,"repair":4,
     "replace":5,"shutdown/stop":6}
an["urgency"]=an.action.map(URG)
ok=an.dropna(subset=["gravity","urgency"])
if len(ok)>8:
    r,pv=stats.spearmanr(ok.gravity,ok.urgency)
    add("spearman_gravity_vs_action_urgency",round(float(r),3),len(ok),
        f"p={pv:.3g}; ranks monitor(1)..shutdown(6)")
hi=an[an.gravity>=4]
add("hard_action_rate_when_gravity_ge4",
    round(float(hi.action.isin({"shutdown/stop","replace","repair"}).mean()),3)
    if len(hi) else np.nan,len(hi),"shutdown/replace/repair")

# ---- temporal-context utilization ------------------------------------------
mention=[bool(re.search(HIST_KW,x["Anomalous Trend"],re.I)) for x in secs]
an["hist_mention"]=mention
w_h=an[an.n_prior>0]; wo_h=an[an.n_prior==0]
add("trend_mentions_history_when_prior>0",
    round(float(w_h.hist_mention.mean()),3),len(w_h))
add("trend_mentions_history_when_no_prior",
    round(float(wo_h.hist_mention.mean()),3),len(wo_h),
    "counters are always in the prompt, so nonzero is expected")

# ---- within-unit gravity escalation ----------------------------------------
esc=[]
for uu,g in an.dropna(subset=["gravity"]).groupby(u):
    if g.gravity.nunique()>=2 and len(g)>=3:
        r=stats.spearmanr(g[c],g.gravity)[0]
        esc.append({"unit":int(uu),"n":len(g),"rho_gravity_vs_cycle":r})
e=pd.DataFrame(esc); e.to_csv(OUT/"per_unit_escalation.csv",index=False)
if len(e):
    add("units_gravity_escalates_rho_median",round(float(e.rho_gravity_vs_cycle.median()),3),len(e))
    add("frac_units_gravity_nondecreasing_trend",
        round(float((e.rho_gravity_vs_cycle>=0).mean()),3),len(e))

sm=pd.DataFrame(S); sm.to_csv(OUT/"summary.csv",index=False)
md=["# SLM behavioural evaluation","",f"{len(an)} interpretations.","",
    "| metric | value | n | note |","|---|---|---|---|"]
md+= [f"| {r.metric} | {r.value} | {'' if r.n!=r.n else int(r.n)} | {r.note} |"
      for _,r in sm.iterrows()]
(OUT/"summary.md").write_text("\n".join(md))
print(sm.to_string(index=False))

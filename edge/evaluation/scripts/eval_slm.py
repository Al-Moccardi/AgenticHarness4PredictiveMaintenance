#!/usr/bin/env python3
"""
eval_slm.py — does the edge SLM say true things about the rule it was given?
============================================================================

The interpretation layer has one job: turn an isolation-forest rule into prose a
technician can act on, WITHOUT inventing anything. That is checkable
deterministically, because the rule is a set of (sensor, operator, threshold)
triples and the interpretation is text. Nothing here needs human annotation.

Metrics, per anomaly, all in [0, 1] unless noted:

  grounding
    sensor_recall        share of the rule's sensors that the text mentions
    sensor_precision     share of mentioned sensors that are in the rule
    hallucinated_sensor  1 if the text names a sensor absent from the rule
  faithfulness
    direction_agreement  share of mentioned rule constraints whose direction
                         word (high/low...) matches the operator (>/<=)
    direction_contradiction  share whose direction word is OPPOSITE
    threshold_anchoring  share of mentioned constraints whose numeric threshold
                         appears near the mention
  protocol
    format_complete      all six required sections present
    has_gravity          a gravity score 1-5 was emitted
    echo_contamination   1 if the model transcribed the input record
    truncated            generation hit the token cap
  severity validity (the interesting one)
    gravity              1-5, correlated against the TRUE RUL, which the model
                         never sees. A severity opinion that does not track
                         remaining life is decoration, not diagnosis.

Outputs (--outdir, default results/slm_eval/):
    per_anomaly.csv    one row per interpretation, every metric above
    summary.csv        aggregate rate + Wilson 95% CI + n
    summary.md         formatted
    gravity_vs_rul.csv  RUL distribution per gravity level (+ Spearman)
    cost.csv            latency / tokens / throughput joined from edge_stats.csv

Usage
-----
    python3 scripts/eval_slm.py
    python3 scripts/eval_slm.py --merged results/test_FD002_with_interpretations.csv \
        --stats results/edge_stats.csv --outdir results/slm_eval
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats

SENSORS = ["T24", "T30", "T50", "P30", "Nf", "Nc", "Ps30", "phi", "NRf",
           "NRc", "BPR", "htBleed", "W31", "W32", "cycles"]
# physical aliases the model legitimately uses instead of the token
ALIAS = {
    "T24": ["lpc outlet temp", "lpc temperature"],
    "T30": ["hpc outlet temp", "hpc temperature"],
    "T50": ["lpt outlet temp", "lpt temperature", "turbine outlet temp"],
    "P30": ["hpc outlet pressure"],
    "Ps30": ["static pressure"],
    "Nf": ["fan speed"], "Nc": ["core speed"],
    "NRf": ["corrected fan speed"], "NRc": ["corrected core speed"],
    "phi": ["fuel flow"], "BPR": ["bypass ratio"],
    "htBleed": ["bleed enthalpy", "bleed"],
    "W31": ["hpt coolant bleed"], "W32": ["lpt coolant bleed"],
    "cycles": ["cycle count", "operating cycle"],
}
UP = ["high", "higher", "elevated", "increase", "increased", "increasing",
      "excessive", "above", "rise", "rising", "greater", "exceed", "exceeds"]
DOWN = ["low", "lower", "reduced", "decrease", "decreased", "decreasing",
        "insufficient", "below", "drop", "dropping", "less", "fall", "falling",
        "diminished"]
SECTIONS = ["**Anomaly Interpretation:**", "**Cause:**", "**Impact:**",
            "**Anomalous Trend:**", "**Expected Future Failures:**",
            "**Recommendation:**"]
WIN = 90          # chars around a mention searched for direction / threshold


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(c - h, 0.0), min(c + h, 1.0))


def parse_rule(text: str) -> List[Tuple[str, str, float]]:
    """Extract (sensor, op, threshold) triples from the Splits=... field."""
    m = re.search(r"Splits=\((.*)\)\s*\|\s*Score", text, re.S)
    body = m.group(1) if m else text
    out = []
    for s, op, v in re.findall(
            r"\b(" + "|".join(SENSORS) + r")\s*(<=|>=|<|>)\s*(-?\d+\.?\d*)",
            body):
        out.append((s, op, float(v)))
    return out


def mentions(txt: str, sensor: str) -> List[int]:
    """Character positions where the sensor is referred to (token or alias)."""
    pos = [m.start() for m in re.finditer(rf"\b{re.escape(sensor)}\b", txt)]
    low = txt.lower()
    for al in ALIAS.get(sensor, []):
        pos += [m.start() for m in re.finditer(re.escape(al), low)]
    return sorted(pos)


def dir_near(txt: str, at: int) -> Set[str]:
    w = txt[max(0, at - WIN): at + WIN].lower()
    d = set()
    if any(re.search(rf"\b{k}\b", w) for k in UP):
        d.add("up")
    if any(re.search(rf"\b{k}\b", w) for k in DOWN):
        d.add("down")
    return d


def thr_near(txt: str, at: int, val: float) -> bool:
    w = txt[max(0, at - WIN): at + WIN]
    for num in re.findall(r"-?\d+\.?\d*", w):
        try:
            if abs(float(num) - val) <= max(abs(val) * 0.01, 0.5):
                return True
        except ValueError:
            continue
    return False


def score_one(rule_text: str, interp: str) -> Dict:
    rule = parse_rule(rule_text)
    rule_sens = {s for s, _, _ in rule}
    ment_sens = {s for s in SENSORS if mentions(interp, s)}
    inter = rule_sens & ment_sens
    d_ok = d_bad = thr_ok = n_cons = 0
    for s, op, v in rule:
        ps = mentions(interp, s)
        if not ps:
            continue
        n_cons += 1
        want = "up" if op in (">", ">=") else "down"
        other = "down" if want == "up" else "up"
        dirs = set()
        for p in ps:
            dirs |= dir_near(interp, p)
        if want in dirs:
            d_ok += 1
        elif other in dirs:
            d_bad += 1
        if any(thr_near(interp, p, v) for p in ps):
            thr_ok += 1
    grav = re.search(r"gravity score:\s*([1-5])", interp)
    return {
        "n_rule_constraints": len(rule), "n_rule_sensors": len(rule_sens),
        "n_mentioned_sensors": len(ment_sens),
        "sensor_recall": len(inter) / len(rule_sens) if rule_sens else np.nan,
        "sensor_precision": len(inter) / len(ment_sens) if ment_sens else np.nan,
        "extra_sensor_flag": int(bool(ment_sens - rule_sens)),
        "n_extra_sensors": len(ment_sens - rule_sens),
        "n_constraints_mentioned": n_cons,
        "direction_agreement": d_ok / n_cons if n_cons else np.nan,
        "direction_contradiction": d_bad / n_cons if n_cons else np.nan,
        "threshold_anchoring": thr_ok / n_cons if n_cons else np.nan,
        "format_complete": int(all(x in interp for x in SECTIONS)),
        "has_gravity": int(bool(grav)),
        "gravity": int(grav.group(1)) if grav else np.nan,
        "echo_contamination": int("ANOMALY RECORD" in interp[:400]),
        "n_chars": len(interp),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged",
                    default="results/test_FD002_with_interpretations.csv")
    ap.add_argument("--stats", default="results/edge_stats.csv")
    ap.add_argument("--outdir", default="results/slm_eval")
    a = ap.parse_args()

    d = pd.read_csv(a.merged)
    ucol = "Unit_ID" if "Unit_ID" in d.columns else "unit_ID"
    ccol = "cycles" if "cycles" in d.columns else "cycle"
    an = d[(d.anomaly_label == -1)
           & d.interpretation.astype(str).str.len().gt(50)].copy()
    if not len(an):
        raise SystemExit("no interpretations found in " + a.merged)
    print(f"[slm] scoring {len(an)} interpretations")

    rows = []
    for _, r in an.iterrows():
        m = score_one(str(r.text), str(r.interpretation))
        m.update({"unit_ID": int(r[ucol]), "cycle": int(r[ccol]),
                  "RUL": float(r.RUL), "anomaly_score": float(r.anomaly_score),
                  "h_clust": int(r.h_clust)})
        rows.append(m)
    pa = pd.DataFrame(rows)

    # join measured cost, if available
    if Path(a.stats).exists():
        st = pd.read_csv(a.stats)
        keep = [c for c in ("unit_ID", "cycle", "wall_s", "prompt_tokens",
                            "gen_tokens", "prefill_tps", "decode_tps",
                            "prefill_s", "decode_s", "stop_reason",
                            "prompt_truncated") if c in st.columns]
        pa = pa.merge(st[keep], on=["unit_ID", "cycle"], how="left")
        if "stop_reason" in pa:
            pa["truncated"] = (pa.stop_reason.astype(str)
                               .str.contains("length|limit", case=False)
                               .astype(int))

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    pa.to_csv(out / "per_anomaly.csv", index=False)

    # ---------------- aggregate ------------------------------------------
    S: List[Dict] = []

    def add_rate(name, col, higher_better=True):
        v = pa[col].dropna()
        if not len(v):
            return
        # binary columns get Wilson; ratio columns get the mean + bootstrap
        if set(v.unique()) <= {0, 1}:
            k, n = int(v.sum()), len(v)
            lo, hi = wilson(k, n)
            S.append({"metric": name, "value": round(k / n, 4),
                      "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "n": n,
                      "higher_better": higher_better})
        else:
            rng = np.random.default_rng(42)
            bs = [v.values[rng.integers(0, len(v), len(v))].mean()
                  for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            S.append({"metric": name, "value": round(float(v.mean()), 4),
                      "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
                      "n": len(v), "higher_better": higher_better})

    for c, hb in [("sensor_recall", True), ("sensor_precision", True),
                  ("direction_agreement", True),
                  ("direction_contradiction", False),
                  ("threshold_anchoring", True),
                  ("extra_sensor_flag", False), ("format_complete", True),
                  ("has_gravity", True), ("echo_contamination", False)]:
        add_rate(c, c, hb)
    if "truncated" in pa:
        add_rate("truncated", "truncated", False)

    # ---------------- gravity validity -----------------------------------
    g = pa.dropna(subset=["gravity"])
    if len(g) > 8:
        rho, p = stats.spearmanr(g.gravity, g.RUL)
        S.append({"metric": "spearman_gravity_vs_RUL", "value": round(rho, 4),
                  "ci_lo": np.nan, "ci_hi": np.nan, "n": len(g),
                  "higher_better": False})
        S.append({"metric": "spearman_gravity_vs_RUL_pvalue",
                  "value": float(f"{p:.3g}"), "ci_lo": np.nan, "ci_hi": np.nan,
                  "n": len(g), "higher_better": False})
        rho2, p2 = stats.spearmanr(g.gravity, g.anomaly_score)
        S.append({"metric": "spearman_gravity_vs_anomaly_score",
                  "value": round(rho2, 4), "ci_lo": np.nan, "ci_hi": np.nan,
                  "n": len(g), "higher_better": False})
        gv = (g.groupby("gravity")
              .agg(n=("RUL", "size"), rul_median=("RUL", "median"),
                   rul_q25=("RUL", lambda x: x.quantile(.25)),
                   rul_q75=("RUL", lambda x: x.quantile(.75)),
                   score_median=("anomaly_score", "median")).reset_index())
        gv.to_csv(out / "gravity_vs_rul.csv", index=False)

    # ---------------- cost ------------------------------------------------
    if "wall_s" in pa and pa.wall_s.notna().any():
        c = pa.dropna(subset=["wall_s"])
        cost = {"n": len(c), "wall_p50": c.wall_s.median(),
                "wall_p95": c.wall_s.quantile(.95),
                "gen_tokens_median": c.gen_tokens.median()
                if "gen_tokens" in c else np.nan,
                "decode_tps_median": c.decode_tps.median()
                if "decode_tps" in c else np.nan,
                "prefill_tps_median": c.prefill_tps.median()
                if "prefill_tps" in c else np.nan}
        pd.DataFrame([cost]).to_csv(out / "cost.csv", index=False)
        for k, v in cost.items():
            if k != "n":
                S.append({"metric": f"cost_{k}", "value": round(float(v), 3),
                          "ci_lo": np.nan, "ci_hi": np.nan, "n": len(c),
                          "higher_better": np.nan})

    sm = pd.DataFrame(S)
    sm.to_csv(out / "summary.csv", index=False)

    L = ["# Edge SLM interpretation — deterministic evaluation against the rule",
         "",
         f"{len(pa)} interpretations. Every metric is computed from the "
         f"isolation-forest rule the model was given; no human annotation. "
         f"Binary rates carry Wilson 95% intervals, ratio metrics a 2000-sample "
         f"bootstrap. The model never sees RUL, so `spearman_gravity_vs_RUL` "
         f"tests whether its severity opinion tracks reality.", "",
         "`sensor_precision` is the primary unsupported-content measure. "
         "`extra_sensor_flag` (>=1 sensor named that is not in the rule) is "
         "deliberately harsh: naming a sensor as a downstream CONSEQUENCE "
         "(\"may raise T50\") is legitimate engineering reasoning, not "
         "hallucination. Read the two together, and inspect "
         "`n_extra_sensors` before calling anything a failure.", "",
         "| metric | value | 95% CI | n |", "|---|---|---|---|"]
    for _, r in sm.iterrows():
        ci = "" if r.ci_lo != r.ci_lo else f"[{r.ci_lo:g}, {r.ci_hi:g}]"
        L.append(f"| {r.metric} | {r.value:g} | {ci} | {int(r.n)} |")
    (out / "summary.md").write_text("\n".join(L))
    (out / "config.json").write_text(json.dumps(vars(a), indent=1))

    print(f"[slm] wrote per_anomaly.csv, summary.csv/.md -> {out}\n")
    show = ["sensor_recall", "sensor_precision", "extra_sensor_flag",
            "direction_agreement", "direction_contradiction",
            "threshold_anchoring", "format_complete", "echo_contamination",
            "spearman_gravity_vs_RUL", "cost_wall_p50", "cost_decode_tps_median"]
    s = sm.set_index("metric")
    for k in show:
        if k in s.index:
            r = s.loc[k]
            ci = "" if r.ci_lo != r.ci_lo else f"  [{r.ci_lo:g}, {r.ci_hi:g}]"
            print(f"  {k:34s} {r.value:>8g}{ci}")


if __name__ == "__main__":
    main()

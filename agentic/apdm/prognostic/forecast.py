"""P7 — the prognostic agent, and its deterministic future evaluation.

WHAT THE AGENT SEES (all causal, no leakage):
  * the current case (edge interpretation, rule, score, counters)
  * THIS unit's own anomaly progression so far (its previous edge anomalies)
  * retrieved precedents WITH THEIR FUTURES: train units' subsequent
    anomalies and time-to-failure are known history, so each precedent is
    shown as "...and here is what happened to that engine afterwards"
  * optionally, the diagnostic ticket (diagnosis-first chaining)
  * optionally, the CNN-GRU hint: "a trained sequence model estimates RUL=X"

WHAT THE AGENT MUST EMIT (one JSON):
  {"progression_narrative": "<how degradation will unfold>",
   "expected_trends": [{"sensor": "T50", "direction": "up|down|stable"}, x3-5],
   "anomaly_outlook": "accelerating|steady|sporadic",
   "rul_estimate": <int>, "rul_range": [lo, hi],
   "cited_precedents": ["u<unit>c<cycle>", ...], "confidence": "low|med|high"}

HOW IT IS SCORED (evaluation-only use of the test future + official gold):
  * RUL: error vs truth anchored on RUL_FD002 (gold + last_cycle - c):
    MAE, bias, range coverage, and the CMAPSS S-score from your pipeline.
  * expected_trends: each named sensor's REALISED slope over the next
    H<=40 observed cycles of the test trajectory (robust z-slope, dead-band
    0.3 sigma -> "stable"); direction accuracy.
  * anomaly_outlook: realised future anomaly rate (detector output on
    cycles (c, c+H]) vs the unit's past rate; ratio>=1.5 accelerating,
    <=0.6 sporadic, else steady; outlook accuracy.
The future rows are never in any prompt; they exist only in the scorer.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..llm import Backend
from ..patterns import case_block

ROOT = Path(__file__).resolve().parents[2]

F_TICKET = ('{"progression_narrative": "<2-4 sentences on how degradation '
            'will unfold>", "expected_trends": [{"sensor": "<one of T24,T30,'
            'T50,P30,Nf,Nc,Ps30,phi,NRf,NRc,BPR,htBleed,W31,W32>", '
            '"direction": "up|down|stable"}, ...3 to 5 items...], '
            '"anomaly_outlook": "accelerating|steady|sporadic", '
            '"rul_estimate": <int cycles, 0-125>, "rul_range": [<lo>, <hi>], '
            '"cited_precedents": ["u<unit>c<cycle>", ...], '
            '"confidence": "low|med|high"}')

RUL_CAP = 125.0

SYSTEM_F = ("You are the fleet PROGNOSTIC agent. From the current case, the "
            "unit's own anomaly progression, and precedent engines whose "
            "subsequent history is known, forecast THIS SPECIFIC unit's "
            "degradation. RUL is on the capped scale: never above 125. RUL "
            "can only DECREASE as cycles pass; stay consistent with your own "
            "previous prognosis for this unit when one is shown. "
            "Ground every expectation in the precedents' futures and the "
            "unit's trend. OUTPUT PROTOCOL: ONE JSON object exactly:\n"
            + F_TICKET
            + "\nAlso include \"projected_progression\": <=50 words hypothesizing THIS unit's own future (timing of next anomalies, gravity trend, horizon to end of life), grounded in the cited futures.")


# ------------------------------------------------------------ context parts
def unit_history(queries: List[Dict], q: Dict, max_n: int = 6) -> str:
    prev = [p for p in queries if p["unit"] == q["unit"]
            and p["cycle"] < q["cycle"]][-max_n:]
    if not prev:
        return "UNIT PROGRESSION SO FAR: this is the unit's first detected anomaly."
    lines = [f"UNIT PROGRESSION SO FAR ({len(prev)} previous anomalies):"]
    for p in prev:
        head = p["interpretation"].split("**Cause")[0][-160:].replace("\n", " ")
        lines.append(f"  cycle {p['cycle']}: score={p['score']:.3f} "
                     f"local_count={p['counters']['local_count']} :: {head}")
    return "\n".join(lines)


class PrecedentFutures:
    """Index of the store meta grouped by train unit; gives each retrieved
    precedent its subsequent trajectory (known outcomes -- train data)."""

    def __init__(self, meta_path: Path):
        by_unit: Dict[int, List[Dict]] = defaultdict(list)
        for line in Path(meta_path).read_text().splitlines():
            if line.strip():
                m = json.loads(line)
                by_unit[int(m["unit"])].append(m)
        self.by_unit = {u: sorted(v, key=lambda r: r["cycle"])
                        for u, v in by_unit.items()}

    def history_of(self, unit: int, cycle: int, max_n: int = 4) -> str:
        """The precedent's PAST at its matched point - diagnosis territory.
        No rul_then, no futures: those belong to the prognostic agent."""
        rows = [r for r in self.by_unit.get(unit, []) if r["cycle"] < cycle]
        if not rows:
            return "    history: this was that unit's FIRST detected anomaly."
        seq = "; ".join(
            f"c{r['cycle']} (g{r.get('gravity')}): "
            f"{str(r.get('text',''))[:70]}" for r in rows[-max_n:])
        return (f"    history ({len(rows)} prior anomalies): ...{seq}"
                if len(rows) > max_n else
                f"    history ({len(rows)} prior anomalies): {seq}")

    def future_of(self, unit: int, cycle: int, max_n: int = 5) -> str:
        rows = [r for r in self.by_unit.get(unit, []) if r["cycle"] > cycle]
        rul_now = next((r["rul_then"] for r in self.by_unit.get(unit, [])
                        if r["cycle"] == cycle), None)
        eol = (f"engine failed {int(rul_now)} cycles after this point."
               if rul_now is not None else "end of life unknown.")
        if not rows:
            return f"    afterwards: no further anomalies recorded; {eol}"
        seq = "; ".join(f"c{r['cycle']} (rul {r['rul_then']}, "
                        f"g{r.get('gravity')})" for r in rows[:max_n])
        return (f"    afterwards: {len(rows)} more anomalies -> {seq}"
                f"{' ...' if len(rows) > max_n else ''}; {eol}")


def fmt_precedents_with_futures(recs: List[Dict], pf: PrecedentFutures) -> str:
    out = ["PRECEDENTS FROM FLEET MEMORY (with their known futures):"]
    for r in recs:
        pid = f"u{r['unit']}c{r['cycle']}"
        out.append(f"[{pid}] similarity={r['similarity']:.3f} "
                   f"rul_then={r['rul_then']} g={r.get('gravity')}\n"
                   f"    {r['text'][:240]}\n"
                   + pf.future_of(r["unit"], r["cycle"]))
    return "\n".join(out)


def parse_forecast(text: str) -> Optional[Dict]:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        j = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(j, dict):
        return None
    out = {"progression_narrative": str(j.get("progression_narrative",
                                              ""))[:900],
           "anomaly_outlook": str(j.get("anomaly_outlook", "")).strip(),
           "confidence": str(j.get("confidence", "")),
           "cited_precedents": [str(x) for x in
                                (j.get("cited_precedents") or [])][:8],
           "projected_progression": str(
               j.get("projected_progression", ""))[:400]}
    tr = []
    for t in (j.get("expected_trends") or [])[:6]:
        if isinstance(t, str):
            tr.append(t.strip()[:80])                 # keep verbatim
        elif isinstance(t, dict):
            sensor = (t.get("sensor") or t.get("parameter")
                      or t.get("name") or t.get("feature") or "")
            direction = (t.get("direction") or t.get("trend")
                         or t.get("expected") or t.get("change") or "")
            if sensor:
                tr.append({"sensor": str(sensor).strip()[:60],
                           "direction": str(direction).strip()[:30]})
    out["expected_trends"] = tr
    try:
        out["rul_estimate"] = float(j.get("rul_estimate"))
    except (TypeError, ValueError):
        out["rul_estimate"] = None
    rr = j.get("rul_range") or []
    try:
        out["rul_range"] = [float(rr[0]), float(rr[1])]
    except Exception:  # noqa: BLE001
        out["rul_range"] = None
    return out


# ------------------------------------------------------------------ the agent
def p7(be: Backend, q: Dict, recs: List[Dict], pf: PrecedentFutures,
       queries: List[Dict], diagnosis: Optional[Dict] = None,
       dl_rul: Optional[float] = None,
       prev_forecast: Optional[Dict] = None,
       signals: Optional[Dict] = None) -> Dict:
    parts = [case_block(q), unit_history(queries, q),
             fmt_precedents_with_futures(recs, pf)]
    if prev_forecast:
        el = q["cycle"] - prev_forecast["cycle"]
        parts.append(f"YOUR PREVIOUS PROGNOSIS for THIS unit, at cycle "
                     f"{prev_forecast['cycle']} ({el} cycles ago), estimated "
                     f"RUL = {prev_forecast['rul']:.0f}. RUL cannot rise: "
                     f"your new estimate must not exceed that value.")
    if diagnosis:
        parts.append("DIAGNOSTIC TICKET (from the diagnostic agent):\n"
                     + json.dumps(diagnosis))
    if dl_rul is not None:
        parts.append(f"MODEL HINT: a trained CNN-GRU sequence model over "
                     f"this unit's last {20} cycles estimates "
                     f"RUL = {dl_rul:.0f} cycles. Weigh it against the "
                     f"precedents; you may deviate, but justify.")
    if signals:
        from .future_progression import (fmt_reliability,
                                         fmt_future_progression)
        rb = fmt_reliability(signals.get("reliability"))
        fb = fmt_future_progression(signals.get("future_progression"))
        if rb:
            parts.append(rb)
        if fb:
            parts.append(fb)
    parts.append("Produce the prognosis JSON now.")
    out = be.generate("\n\n".join(parts), system=SYSTEM_F)
    fc = parse_forecast(out)
    return {"forecast": fc, "raw": out}


# ----------------------------------------------------- deterministic scoring
def s_score(err: float) -> float:
    """CMAPSS asymmetric score for ONE prediction (your compute_s_score)."""
    d = err  # pred - true
    return float(np.exp(-d / 13.0) - 1.0) if d < 0 else float(
        np.exp(d / 10.0) - 1.0)


def realised_trend(raw_unit: pd.DataFrame, sensor: str, c: int,
                   horizon: int = 40) -> Optional[str]:
    g = raw_unit.sort_values("cycles")
    past = g[g.cycles <= c][sensor].values
    fut = g[(g.cycles > c) & (g.cycles <= c + horizon)][sensor].values
    if len(fut) < 5 or len(past) < 5:
        return None
    sd = float(np.std(past)) or 1e-9
    x = np.arange(len(fut))
    slope = float(np.polyfit(x, (fut - np.mean(past)) / sd, 1)[0])
    delta = slope * len(fut)                      # total sigma-change
    if abs(delta) < 0.30:
        return "stable"
    return "up" if delta > 0 else "down"


def realised_outlook(det_unit: pd.DataFrame, c: int,
                     horizon: int = 40) -> Optional[str]:
    g = det_unit.sort_values("cycles")
    past = g[g.cycles <= c]
    fut = g[(g.cycles > c) & (g.cycles <= c + horizon)]
    if len(fut) < 5:
        return None
    r_past = float((past.anomaly_label == -1).mean()) if len(past) else 0.0
    r_fut = float((fut.anomaly_label == -1).mean())
    if r_fut >= max(1.5 * r_past, r_past + 0.05):
        return "accelerating"
    if r_fut <= max(0.6 * r_past, 1e-9) and r_fut < 0.05:
        return "sporadic"
    return "steady"


_DIR = {"up": "up", "increase": "up", "increasing": "up", "rise": "up",
        "rising": "up", "higher": "up", "grow": "up", "growing": "up",
        "upward": "up",
        "down": "down", "decrease": "down", "decreasing": "down",
        "drop": "down", "dropping": "down", "fall": "down",
        "falling": "down", "lower": "down", "decline": "down",
        "declining": "down", "downward": "down",
        "stable": "stable", "steady": "stable", "constant": "stable",
        "flat": "stable", "unchanged": "stable"}
_SENS = ["T24", "T30", "T50", "P30", "Nf", "Nc", "Ps30", "phi", "NRf",
         "NRc", "BPR", "htBleed", "W31", "W32"]
_SENS_LOW = {x.lower(): x for x in _SENS}
_SENS_ALIAS = {"lpt": "T50", "lpt outlet": "T50", "hpc outlet": "T30",
               "fan inlet": "T24", "static pressure": "Ps30",
               "fan speed": "Nf", "core speed": "Nc", "bypass": "BPR",
               "bypass ratio": "BPR", "bleed": "htBleed",
               "fuel flow": "phi", "coolant": "W31"}


def _norm_trends(fc: Optional[Dict]) -> List[Dict]:
    """Tolerant extraction: dict or 'T50: increasing' strings; synonym
    directions; sensor names cleaned of units/parentheses; case-insensitive."""
    out = []
    for t in (fc or {}).get("expected_trends", []) or []:
        if isinstance(t, str):
            m = re.match(r"\s*([A-Za-z]+\d*)\s*[:\-]?\s*(\w+)", t)
            if not m:
                continue
            sensor, direction = m.group(1), m.group(2)
        elif isinstance(t, dict):
            sensor = str(t.get("sensor", ""))
            direction = str(t.get("direction", ""))
        else:
            continue
        low = sensor.lower()
        sm = re.match(r"\s*([A-Za-z]+\d*)", sensor)
        sensor = (_SENS_LOW.get((sm.group(1) if sm else sensor).lower())
                  or next((v for k, v in _SENS_ALIAS.items() if k in low),
                          None))
        dm = re.match(r"\s*([A-Za-z]+)", direction)
        direction = _DIR.get((dm.group(1) if dm else direction).lower())
        if sensor and direction:
            out.append({"sensor": sensor, "direction": direction})
    return out


def evaluate_forecast(fc: Optional[Dict], q: Dict, raw_unit: pd.DataFrame,
                      det_unit: pd.DataFrame, horizon: int = 40) -> Dict:
    out: Dict[str, Optional[float]] = {"json_valid": float(bool(fc))}
    true_rul = q.get("true_rul")
    if fc and fc.get("rul_estimate") is not None and true_rul is not None:
        err = fc["rul_estimate"] - true_rul
        out["rul_err"] = round(err, 1)
        out["rul_abs_err"] = round(abs(err), 1)
        out["s_score"] = round(s_score(err), 2)
        rr = fc.get("rul_range")
        out["range_coverage"] = (float(rr[0] <= true_rul <= rr[1])
                                 if rr else np.nan)
        out["range_width"] = (round(rr[1] - rr[0], 1) if rr else np.nan)
    else:
        out.update({"rul_err": np.nan, "rul_abs_err": np.nan,
                    "s_score": np.nan, "range_coverage": np.nan,
                    "range_width": np.nan})
    # trends
    hits = tot = 0
    detail = []
    for t in _norm_trends(fc):
        real = realised_trend(raw_unit, t["sensor"], q["cycle"], horizon) \
            if t["sensor"] in raw_unit.columns else None
        if real is None:
            continue
        tot += 1
        ok = int(t["direction"] == real)
        hits += ok
        detail.append({"sensor": t["sensor"], "pred": t["direction"],
                       "real": real, "ok": ok})
    out["trend_n"] = float(tot) if tot else np.nan
    out["trend_direction_acc"] = (hits / tot) if tot else np.nan
    out["trend_detail"] = detail
    # outlook
    real_o = realised_outlook(det_unit, q["cycle"], horizon)
    pred_o = (fc or {}).get("anomaly_outlook")
    out["outlook_real"] = real_o
    out["outlook_pred"] = pred_o
    out["outlook_acc"] = (float(pred_o == real_o)
                          if real_o and pred_o in
                          ("accelerating", "steady", "sporadic") else np.nan)
    return out


# ------------------------------------------------------------ mock for smoke
class MockForecastBackend(Backend):
    name = "mockf"
    model = "mockf"

    def __init__(self, log_path=None):
        self.log_path = log_path

    def generate(self, prompt, system=None):
        ids = re.findall(r"\[(u\d+c\d+)\]", prompt)[:2]
        ruls = [float(x) for x in re.findall(r"rul_then=(\d+\.?\d*)", prompt)]
        hint = re.findall(r"RUL = (\d+)", prompt)
        est = int(hint[0]) if hint else (int(np.median(ruls)) if ruls else 70)
        out = json.dumps({
            "progression_horizon": 30,
            "projected_progression": "Expect further anomalies with rising "
                                     "gravity toward end of life.",
            "progression_narrative": "Degradation should follow the cited "
                                     "precedents toward end of life.",
            "expected_trends": [{"sensor": "T50", "direction": "up"},
                                {"sensor": "Ps30", "direction": "up"},
                                {"sensor": "phi", "direction": "down"}],
            "anomaly_outlook": "accelerating",
            "rul_estimate": est, "rul_range": [max(est - 20, 0), est + 20],
            "cited_precedents": ids, "confidence": "med"})
        self._log(system, prompt, out)
        return out

SYSTEM_P = (
    "You are the future-progression analyst of a predictive-maintenance "
    "system. The RUL number is handled by a deterministic model and is "
    "GIVEN to you; do NOT re-estimate it. Your ONLY job: commit to a "
    "hypothesis of THIS unit's future progression, grounded in the cited "
    "precedents' observed futures. OUTPUT PROTOCOL: ONE JSON object "
    "exactly:\n"
    "{\"projected_progression\": str (<=50 words: what will happen - "
    "timing of the next anomalies, gravity trend, path to end of life; "
    "MANDATORY, never empty),\n"
    " \"progression_horizon\": int (cycles from now to end of life "
    "implied by YOUR hypothesis; MANDATORY),\n"
    " \"confidence\": \"low\"|\"medium\"|\"high\",\n"
    " \"cited_precedents\": [ids you relied on]}\n"
    "No other keys. No prose outside the JSON.")


def parse_progression(txt: str) -> Optional[Dict]:
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        j = json.loads(m.group(0))
    except Exception:
        try:
            j = json.loads(m.group(0).replace("\n", " "))
        except Exception:
            return None
    pp = str(j.get("projected_progression", "") or "")[:400]
    try:
        hz = float(j.get("progression_horizon"))
    except (TypeError, ValueError):
        hz = None
    return {"projected_progression": pp,
            "progression_horizon": hz,
            "confidence": str(j.get("confidence", "") or "").lower()[:10],
            "cited_precedents": [str(x) for x in
                                 (j.get("cited_precedents") or [])][:8]}


def p7_progression(be: Backend, q: Dict, recs: List[Dict],
                   pf: PrecedentFutures, queries: List[Dict],
                   diagnosis: Optional[Dict] = None,
                   dl_rul: Optional[float] = None,
                   signals: Optional[Dict] = None) -> Dict:
    """Progression-only variant: the tool owns the number, the SLM owns
    the hypothesis. Returns {"forecast": ..., "raw": ...}."""
    from .future_progression import fmt_reliability, fmt_future_progression
    parts = [f"MONITORED UNIT {q['unit']}, anomaly at cycle {q['cycle']}.",
             f"EDGE INTERPRETATION: {q.get('interpretation') or 'n/a'}"]
    if diagnosis:
        parts.append(f"DIAGNOSIS: severity {diagnosis.get('severity')}, "
                     f"action {diagnosis.get('action')}.")
    if dl_rul is not None:
        parts.append(f"RUL (GIVEN, deterministic model): "
                     f"{float(dl_rul):.0f} cycles. Do not re-estimate it.")
    parts.append("SIMILAR PRECEDENT CASES:\n"
                 + fmt_precedents_with_futures(recs, pf))
    if signals:
        rb = fmt_reliability(signals.get("reliability"))
        fb = fmt_future_progression(signals.get("future_progression"))
        if rb:
            parts.append(rb)
        if fb:
            parts.append(fb)
    parts.append("Produce the progression JSON now.")
    raw = be.generate("\n\n".join(parts), system=SYSTEM_P)
    fc = parse_progression(raw)
    return {"forecast": fc, "raw": raw}


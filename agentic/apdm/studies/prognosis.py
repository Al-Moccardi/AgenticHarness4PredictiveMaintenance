"""v8 -- Retrieval-augmented prognostic DECISION layer over the
interpretation store: the user-specified pipeline, end to end.

  IF anomalies + rules  ->  natural-language interpretations (TIOT layer)
  -> frozen 208/52 unit split (seed 42, identical everywhere)
  -> vector store over TRAIN interpretations (nomic, n=4,458)
  -> for each TEST-unit anomaly: QUERY = its own interpretation (real or
     generated; grounded sensor summary as fallback), RETRIEVE similar train
     interpretations together with their raw-signal context and their known
     OUTCOMES (rul_then), and have agents produce a maintenance TICKET:
        {"rul_estimate", "rul_range" (central 80%), "action" 1-5,
         "cited_precedents", "rationale"}

The estimate is retrieval-grounded case-based reasoning: the system is
ML-free, so the only prognostic signal is what actually happened to the
retrieved precedents. Its honest reference is therefore the SAME retrieval
with the language model switched off (B0), plus the embedder-free z-space
kNN twin (B1) that prices the semantic representation itself.

Arms -- a ladder of increasing agentic structure, every rung answering one
question about what a small edge model can exploit:

  B0_retrieval    no LLM: median / [q10,q90] of the semantic neighbours'
                  outcomes; action = severity band of the estimate.
  B1_zknn         no LLM: same, neighbours from regime-referenced z-space
                  (the representation twin the store must beat).
  P1_direct       single shot, case evidence only (no retrieval).
  P2_rag          single shot + retrieved precedents (semantic RAG).
  P3_react        ReAct loop; the model gathers its own evidence via tools.
  P4_reflexion    draft (=P2) then one self-critique-and-revise pass.
  P5_verifier     draft (=P2) then DETERMINISTIC acceptance gates with one
                  violation-driven repair; unresolved -> escalated to human.
                  (The RAD guardrails promoted from evaluation to runtime.)
  P6_specialists  three cooperating role agents: retrieval analyst ->
                  prognostics estimator -> maintenance planner.

Scoring (internal test split only): MAE / bias / per-case S-score on
rul_estimate; coverage, mean width and Winkler (alpha=.2) on rul_range;
exact and +/-1 accuracy on the OUTCOME-derived action band (events.py
severity bands -- never the interpretations' own gravity opinions);
faithfulness of rationale claims vs current-window z-evidence; parse
failures first-class; cost per decision (calls, tokens, simulated Jetson
seconds/energy) on every arm.

  python -m apdm.smoke_prognosis                       # offline, dryrun
  python -m apdm.gen_interpretations --backend ollama --model llama3.2:3b \\
      --units test                                     # step 0 (~986 recs)
  python -m apdm.prognosis --backend ollama --model llama3.2:3b \\
      --sample-seed 1                                  # one model, 8 arms
  python -m apdm.bench_prognosis --sample-seed 1       # the model sweep
  python -m apdm.report_prognosis --sample-seed 1      # tables + HEADLINES
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .agent import _first_json
from ..data import FD002, RMAX, SENSORS, Snapshot
from .events import severity_band
from .faults import current_z
from .features import summary_text
from .interpret_kb import InterpretationKB
from ..llm import Backend, get_backend
from .metrics import BUCKETS, s_score, wilson

ROOT = Path(__file__).resolve().parent.parent
GEN_DIR = ROOT / "data" / "interpretations_generated"
REAL_DIR = ROOT / "data" / "interpretations"

ARMS = ["B0_retrieval", "B1_zknn", "P1_direct", "P2_rag", "P3_react",
        "P4_reflexion", "P5_verifier", "P6_specialists"]

ACTION_LABELS = {1: "continue normal operation", 2: "monitor closely",
                 3: "schedule inspection", 4: "plan maintenance",
                 5: "immediate intervention"}
ACTION_WORDS = {"continue": 1, "normal": 1, "monitor": 2, "watch": 2,
                "observe": 2, "inspect": 3, "inspection": 3, "schedule": 3,
                "plan": 4, "maintenance": 4, "maintain": 4, "repair": 4,
                "stop": 5, "immediate": 5, "halt": 5, "shutdown": 5}
ALPHA = 0.2                                  # central 80% interval target

TICKET_TEMPLATE = """{"rul_estimate": <integer 0-125>,
 "rul_range": [<integer lo>, <integer hi>],
 "action": <integer 1-5>,
 "cited_precedents": ["<id>", ...],
 "rationale": "<2-4 sentences grounded ONLY in the evidence shown; never invent sensor values>"}"""

TICKET_SYSTEM = f"""You are the fleet-level prognostic decision layer of an
IoT predictive-maintenance system for turbofan engines. For ONE detected
anomaly, estimate the remaining useful life and plan the maintenance
response, grounded ONLY in the evidence provided.

OUTPUT PROTOCOL - reply with ONE JSON object and NOTHING else:
{TICKET_TEMPLATE}

Action scale: 1=continue normal operation, 2=monitor closely,
3=schedule inspection, 4=plan maintenance within the interval,
5=immediate intervention / stop.
"rul_range" is your central 80% interval: the true remaining life should
fall inside it with 80% probability. When precedents are provided, anchor
the estimate in the remaining life they ACTUALLY had (rul_then) and cite
the ids you used.

WORKED EXAMPLE of a valid reply:
{{"rul_estimate": 41, "rul_range": [28, 60], "action": 3, "cited_precedents": ["u12c140", "u87c95"], "rationale": "The retrieved precedents with the same low-BPR / high-T50 pattern had 30-58 cycles of life remaining, and this unit entered the degradation state 12 cycles ago, so a mid-band inspection is appropriate."}}"""

AGENT_SYSTEM = f"""You are the fleet-level prognostic decision layer of an
IoT predictive-maintenance system for turbofan engines, with tool access.
Estimate the remaining useful life of the unit below and plan the
maintenance response.

OUTPUT PROTOCOL - reply with ONE JSON object and NOTHING else.
Tool call: {{"thought": "...", "action": {{"tool": "<name>", "args": {{...}}}}}}
Answer:    {{"thought": "...", "final": {TICKET_TEMPLATE}}}

Action scale: 1=continue, 2=monitor, 3=schedule inspection,
4=plan maintenance, 5=immediate intervention. "rul_range" is your central
80% interval. Anchor the estimate in the remaining life the retrieved
precedents ACTUALLY had and cite their ids.

WORKED EXAMPLE of an answer:
  {{"thought": "precedents cluster at 30-58 cycles", "final": {{"rul_estimate": 41, "rul_range": [28, 60], "action": 3, "cited_precedents": ["u12c140"], "rationale": "Precedents with the same pattern had 30-58 cycles remaining; the unit entered the degradation state 12 cycles ago."}}}}

Rules: gather only the evidence you need, never repeat a call, ground every
claim in observations.

TOOLS
__TOOLS__"""

FORCE = (f"You are out of tool steps. Using ONLY the observations below, "
         f"produce the maintenance ticket NOW as ONE JSON object:\n"
         f"{TICKET_TEMPLATE}\n\nOBSERVATIONS:\n{{obs}}")


# ------------------------------------------------------------------ queries
def load_test_interpretations(ds: FD002) -> Dict[Tuple[int, int], str]:
    """(unit, cycle) -> interpretation text, TEST units only, from both the
    real files (units 58/140 under seed 42) and anything generated with
    `gen_interpretations --units test`. Causal by construction of the
    generator (prompt uses rows <= cycle only)."""
    out: Dict[Tuple[int, int], str] = {}
    test = set(ds.test_units)
    for d in (REAL_DIR, GEN_DIR):
        if not d.exists():
            continue
        for f in sorted(list(d.glob("unit_*.json")) + list(d.glob("unit_*.jsonl"))):
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                u = int(r["unit_ID"])
                if u not in test:
                    continue
                c = int(r.get("cycle", r.get("cycles", 0)))
                txt = str(r.get("interpretation", "")).strip()
                if len(txt) >= 40:
                    out[(u, c)] = txt
    return out


def case_context(ds: FD002, unit: int, cycle: int) -> str:
    """The current anomaly's grounded evidence: the shared sensor summary
    plus the detector-layer record (rule, score, causal counters)."""
    row = ds.row(unit, cycle)
    lines = [summary_text(ds, Snapshot(unit, cycle, 0), top_k=8),
             f"Isolation-forest rule: {str(row['text'])[:260]}",
             f"Anomaly score: {float(row['anomaly_score']):.4f} "
             f"(more negative = stronger outlier)",
             f"Causal anomaly counters: local_count="
             f"{int(row['local_cumulative_anomaly_count'])}, "
             f"local_last3_freq={float(row['local_last_3_freq']):.2f}, "
             f"global_count={int(row['global_cumulative_anomaly_count'])}, "
             f"global_last3_freq={float(row['global_last_3_freq']):.2f}"]
    return "\n".join(lines)


# ----------------------------------------------------------------- evidence
def _pid(u: int, c: int) -> str:
    return f"u{u}c{c}"


def _enrich(ds: FD002, u: int, c: int) -> Dict:
    z = current_z(ds, u, c)
    order = np.argsort(-np.abs(z))[:3]
    row = ds.row(u, c)
    return {"rule": str(row["text"])[:180],
            "anomaly_score": round(float(row["anomaly_score"]), 4),
            "top_deviations": [f"{SENSORS[i]} z={z[i]:+.2f}" for i in order]}


def _stats(ruls: List[int]) -> Dict:
    a = np.asarray(ruls, float)
    return {"median_rul_then": float(np.median(a)),
            "q10": float(np.quantile(a, 0.10)),
            "q90": float(np.quantile(a, 0.90)),
            "iqr_lo": float(np.quantile(a, 0.25)),
            "iqr_hi": float(np.quantile(a, 0.75)),
            "min": float(a.min()), "max": float(a.max())}


def retrieve_semantic(ds: FD002, store, embedder, query_text: str,
                      unit: int, k: int) -> Dict:
    qv = embedder.embed([query_text[:2000]])[0]
    recs = store.search(qv, k=k, exclude_unit=unit)
    prec = []
    for r in recs:
        prec.append({"id": _pid(r["unit"], r["cycle"]),
                     "unit": r["unit"], "cycle": r["cycle"],
                     "rul_then": int(r["rul_then"]),
                     "similarity": r["similarity"],
                     **_enrich(ds, r["unit"], r["cycle"]),
                     "interpretation_excerpt": str(r["text"])[:380]})
    return {"k": len(prec), "retrieval": "semantic (interpretation store)",
            "note": "TRAIN-fleet precedents; rul_then is the remaining life "
                    "each of them ACTUALLY had (clipped at 125)",
            "stats": _stats([p["rul_then"] for p in prec]),
            "precedents": prec}


def retrieve_zknn(ds: FD002, kb: InterpretationKB, unit: int, cycle: int,
                  k: int) -> Dict:
    zq = current_z(ds, unit, cycle)
    recs = kb.search(zq, k=k, exclude_unit=unit, prefer_interpreted=False)
    prec = []
    for r in recs:
        prec.append({"id": _pid(r.unit, r.cycle), "unit": r.unit,
                     "cycle": r.cycle, "rul_then": int(r.rul_then),
                     **_enrich(ds, r.unit, r.cycle),
                     **({"interpretation_excerpt": r.interpretation[:380]}
                        if r.interpretation else {})})
    return {"k": len(prec), "retrieval": "z-space kNN (embedder-free twin)",
            "note": "TRAIN-fleet precedents; rul_then is the remaining life "
                    "each of them ACTUALLY had (clipped at 125)",
            "stats": _stats([p["rul_then"] for p in prec]),
            "precedents": prec}


def evidence_text(ev: Dict) -> str:
    return json.dumps(ev)


# ------------------------------------------------------------------- ticket
def _to_action(v) -> Optional[int]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        a = int(round(float(v)))
        return a if 1 <= a <= 5 else None
    if isinstance(v, str):
        m = re.search(r"[1-5]", v)
        if m:
            return int(m.group(0))
        low = v.lower()
        for w, a in ACTION_WORDS.items():
            if w in low:
                return a
    return None


def parse_ticket(text_or_obj) -> Optional[Dict]:
    """Strict ticket parser. Accepts the structured dict from the JSON
    protocol or raw text containing it; regex fallback for the numerics.
    Required: rul_estimate, rul_range, action. Values are clamped, never
    reordered -- inconsistencies are the verifier's job to see."""
    o = text_or_obj if isinstance(text_or_obj, dict) else \
        _first_json(str(text_or_obj or ""))
    est = lo = hi = act = None
    cited: List[str] = []
    rat = ""
    if o:
        try:
            est = int(round(float(o.get("rul_estimate"))))
        except (TypeError, ValueError):
            est = None
        rr = o.get("rul_range")
        if isinstance(rr, dict):
            rr = [rr.get("lo", rr.get("low")), rr.get("hi", rr.get("high"))]
        if isinstance(rr, (list, tuple)) and len(rr) >= 2:
            try:
                lo, hi = int(round(float(rr[0]))), int(round(float(rr[1])))
            except (TypeError, ValueError):
                lo = hi = None
        act = _to_action(o.get("action"))
        c = o.get("cited_precedents") or o.get("cited") or []
        if isinstance(c, str):
            c = re.findall(r"u\d+c\d+", c)
        cited = [str(x) for x in c][:8] if isinstance(c, list) else []
        rat = str(o.get("rationale", ""))[:900]
    if est is None or act is None or lo is None or hi is None:
        t = text_or_obj if isinstance(text_or_obj, str) else \
            json.dumps(text_or_obj, default=str)
        if est is None:
            m = re.search(r"rul[_\s]?estimate\D{0,8}(\d+)", t, re.IGNORECASE)
            est = int(m.group(1)) if m else None
        if lo is None or hi is None:
            m = re.search(r"rul[_\s]?range\D{0,8}(\d+)\D{1,6}(\d+)", t,
                          re.IGNORECASE)
            if m:
                lo, hi = int(m.group(1)), int(m.group(2))
        if act is None:
            m = re.search(r'"?action"?\D{0,6}([1-5])', t, re.IGNORECASE)
            act = int(m.group(1)) if m else None
        if not cited:
            cited = re.findall(r"u\d+c\d+", t)[:8]
        if not rat and isinstance(o, dict):
            rat = str(o.get("rationale", ""))[:900]
    if est is None or act is None or lo is None or hi is None:
        return None
    clamp = lambda x: int(max(0, min(RMAX, x)))          # noqa: E731
    return {"rul_estimate": clamp(est), "rul_range": [clamp(lo), clamp(hi)],
            "action": act, "cited_precedents": cited, "rationale": rat}


# ----------------------------------------------------------------- verifier
_DIR_HIGH = ("high", "elevated", "increased", "rising", "above")
_DIR_LOW = ("low", "reduced", "decreased", "falling", "below")


def extract_claims(rationale: str) -> List[Tuple[str, str]]:
    """(sensor, high|low) claims stated in free text: a known sensor token
    with a direction word within a +/-40-char window (ambiguous windows,
    containing both directions, are skipped)."""
    out, seen = [], set()
    txt = rationale or ""
    for sen in SENSORS:
        for m in re.finditer(rf"\b{re.escape(sen)}\b", txt, re.IGNORECASE):
            w = txt[max(0, m.start() - 40): m.end() + 40].lower()
            hi = any(d in w for d in _DIR_HIGH)
            lo = any(d in w for d in _DIR_LOW)
            if hi == lo:
                continue
            key = (sen, "high" if hi else "low")
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out


def claim_faithfulness(claims: List[Tuple[str, str]], zvec: np.ndarray,
                       tau: float = 1.0) -> Dict[str, float]:
    zmap = dict(zip(SENSORS, zvec))
    sup = con = unv = 0
    for sen, d in claims:
        z = zmap[sen]
        if abs(z) < tau:
            unv += 1
        elif (z > 0) == (d == "high"):
            sup += 1
        else:
            con += 1
    n = max(len(claims), 1)
    return {"n_claims": len(claims), "supported": sup / n,
            "contradicted": con / n, "unverifiable": unv / n}


def verify_ticket(ticket: Dict, ev: Optional[Dict],
                  zvec: np.ndarray) -> Tuple[List[str], Dict]:
    """The deterministic acceptance gates (RAD guardrails as runtime
    checks). Returns (violations, faithfulness-metrics). Gates:
      support   estimate traceable to precedent outcomes (+/-15 slack)
      range     lo <= estimate <= hi and a sane 80%-interval width
      action    action within one band of the estimate's severity band
      citation  at least one cited id is a really-retrieved precedent
      claims    no rationale claim CONTRADICTED by |z|>=1 evidence
    """
    v: List[str] = []
    est = ticket["rul_estimate"]
    lo, hi = ticket["rul_range"]
    if ev and ev.get("k", 0) >= 3:
        mn, mx = ev["stats"]["min"], ev["stats"]["max"]
        if not (mn - 15 <= est <= mx + 15):
            v.append(f"support: rul_estimate {est} is outside the retrieved "
                     f"outcomes' span [{mn:.0f},{mx:.0f}] (+/-15); anchor "
                     f"the estimate in the precedents' rul_then")
    if not (lo <= est <= hi):
        v.append(f"range: rul_range [{lo},{hi}] must contain rul_estimate "
                 f"{est} with lo <= estimate <= hi")
    if not (4 <= hi - lo <= 120):
        v.append(f"range: width {hi - lo} is implausible for a central 80% "
                 f"interval (want 4-120 cycles)")
    band = severity_band(est)
    if abs(ticket["action"] - band) > 1:
        v.append(f"action: {ticket['action']} is inconsistent with "
                 f"rul_estimate {est} (severity band {band}); stay within "
                 f"one band")
    if ev and ev.get("k", 0) > 0:
        valid = {p["id"] for p in ev["precedents"]}
        if not any(c in valid for c in ticket["cited_precedents"]):
            v.append("citation: cite at least one retrieved precedent id "
                     "you actually used")
    claims = extract_claims(ticket.get("rationale", ""))
    faith = claim_faithfulness(claims, zvec)
    bad = [f"{s} {d}" for (s, d) in claims
           if abs(dict(zip(SENSORS, zvec))[s]) >= 1.0
           and ((dict(zip(SENSORS, zvec))[s] > 0) != (d == "high"))]
    if bad:
        v.append("claims: contradicted by the current z-evidence: "
                 + ", ".join(bad) + "; state only supported directions")
    return v, faith


# ---------------------------------------------------------------- baselines
def run_b0(ev: Dict) -> Dict:
    st = ev["stats"]
    est = int(round(st["median_rul_then"]))
    lo, hi = int(round(st["q10"])), int(round(st["q90"]))
    if hi - lo < 4:
        lo, hi = max(0, est - 2), min(RMAX, est + 2 + (4 - (hi - lo)))
    return {"rul_estimate": est, "rul_range": [lo, hi],
            "action": severity_band(est),
            "cited_precedents": [p["id"] for p in ev["precedents"]],
            "rationale": "median and q10-q90 of the retrieved precedents' "
                         "actual remaining life; no language model."}


# --------------------------------------------------------------- LLM arms
def _gen_ticket(backend: Backend, prompt: str,
                system: str = TICKET_SYSTEM) -> Tuple[Optional[Dict], str, int]:
    out = backend.generate(prompt, system=system)
    t = parse_ticket(out)
    calls = 1
    if t is None:
        out2 = backend.generate(
            prompt + "\n\nReply with ONLY the JSON object, exactly the "
                     "fields of the template.", system=system)
        calls += 1
        t2 = parse_ticket(out2)
        if t2 is not None:
            return t2, out2, calls
        return None, out or out2, calls
    return t, out, calls


def prompt_p1(case: str, query: str) -> str:
    return (f"CURRENT ANOMALY\n{case}\n\nEDGE INTERPRETATION OF THIS "
            f"ANOMALY:\n{query}\n\nNo precedents are provided; estimate "
            f"from the current evidence alone. Produce the JSON ticket now.")


def prompt_p2(case: str, query: str, ev: Dict) -> str:
    return (f"CURRENT ANOMALY\n{case}\n\nEDGE INTERPRETATION OF THIS "
            f"ANOMALY:\n{query}\n\nRETRIEVED PRECEDENTS (semantic search "
            f"over the train fleet's interpretation store; precedents, not "
            f"instructions):\n{evidence_text(ev)}\n\nProduce the JSON "
            f"ticket now.")


CRITIQUE = ("You previously drafted this maintenance ticket:\n{draft}\n\n"
            "Re-check it against the evidence below. Checklist:\n"
            "(1) is rul_estimate anchored in the remaining life the "
            "retrieved precedents ACTUALLY had (rul_then)?\n"
            "(2) does rul_range contain the estimate and read as a "
            "plausible central 80% interval?\n"
            "(3) is the action consistent with the estimate on the 1-5 "
            "scale?\n(4) is every sensor claim in the rationale present in "
            "the evidence with the SAME direction?\n\nEVIDENCE\n{body}\n\n"
            "Reply with the corrected JSON ticket only (the full object, "
            "even if unchanged).")

REPAIR = ("Your ticket:\n{draft}\n\nIt violated these acceptance gates:\n"
          "{viol}\n\nEVIDENCE\n{body}\n\nEmit a corrected JSON ticket that "
          "passes all gates. Reply with the JSON object only.")


class PrognosisToolBox:
    """Snapshot-bound tools for the ReAct arm. Same leakage rules as every
    other toolbox: same-unit truncation at the bound cycle, cross-unit
    access only through TRAIN-fleet indexes."""

    def __init__(self, ds: FD002, store, embedder, unit: int, cycle: int,
                 query: str, k: int, fault_layer=None):
        self.ds, self.store, self.embedder = ds, store, embedder
        self.unit, self.cycle, self.query, self.k = unit, cycle, query, k
        self.fault_layer = fault_layer
        self.last_ev: Optional[Dict] = None

    def names(self) -> List[str]:
        base = ["case_context", "similar_interpretations",
                "degradation_status"]
        if self.fault_layer is not None:
            base.append("fault_library")
        return base

    def specs(self) -> str:
        s = ["- case_context() -> the current anomaly's grounded evidence: "
             "z-deviations vs the healthy regime reference, the extracted "
             "isolation-forest rule, score, causal counters, and this "
             "anomaly's own edge interpretation.",
             "- similar_interpretations(k:int=" + str(self.k) + ") -> the k "
             "most similar TRAIN-fleet interpreted anomalies (semantic "
             "search), each with its raw-signal context and the remaining "
             "life it ACTUALLY had (rul_then), plus outcome statistics.",
             "- degradation_status() -> whether this unit has entered the "
             "degradation state (k=7 cluster 6), when, and the train "
             "fleet's residual-life prior after entry."]
        if self.fault_layer is not None:
            s.append("- fault_library() -> the fleet's fault phenotypes: "
                     "signatures, implicated subsystems, onset and "
                     "residual-life statistics (train units only).")
        return "\n".join(s)

    def call(self, name: str, args: Dict) -> str:
        try:
            if name == "case_context":
                return (case_context(self.ds, self.unit, self.cycle)
                        + "\nEDGE INTERPRETATION OF THIS ANOMALY:\n"
                        + self.query[:900])
            if name == "similar_interpretations":
                k = int(args.get("k", self.k) or self.k)
                ev = retrieve_semantic(self.ds, self.store, self.embedder,
                                       self.query, self.unit,
                                       max(1, min(k, 10)))
                self.last_ev = ev
                return evidence_text(ev)
            if name == "degradation_status":
                e = self.ds.state_entered(self.unit, self.cycle)
                return json.dumps({
                    "entered": e is not None, "entry_cycle": e,
                    "cycles_in_state": (self.cycle - e) if e else 0,
                    "train_fleet_prior": self.ds.train_state_residuals()})
            if name == "fault_library" and self.fault_layer is not None:
                return self.fault_layer.library_json()
            return json.dumps({"error": f"unknown tool '{name}'",
                               "available": self.names()})
        except Exception as exc:  # noqa: BLE001 -- degrade, never abort
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


def run_react(ds, store, embedder, unit, cycle, query, backend, k,
              max_steps: int = 6, fault_layer=None
              ) -> Tuple[Optional[Dict], Dict, Optional[Dict]]:
    tb = PrognosisToolBox(ds, store, embedder, unit, cycle, query, k,
                          fault_layer)
    system = AGENT_SYSTEM.replace("__TOOLS__", tb.specs())
    scratch = [f"QUESTION: Estimate the remaining useful life of unit "
               f"{unit} at cycle {cycle} and plan the maintenance action."]
    seen, tools = set(), []
    llm = tool = perr = 0
    term = "step_budget"
    for step in range(1, max_steps + 1):
        if step == max_steps:
            obs = "\n".join(x for x in scratch if x.startswith("OBSERVATION"))
            out = backend.generate(FORCE.format(obs=obs[:6000] or "(none)"))
            llm += 1
            t = parse_ticket(out)
            return t, {"llm_calls": llm, "tool_calls": tool,
                       "protocol_errors": perr, "tools": tools,
                       "termination": "final_forced" if t else "parse_failed"
                       }, tb.last_ev
        hdr = (f"STEP {step}/{max_steps}."
               + (" Answer now unless a tool is essential."
                  if max_steps - step <= 2 else ""))
        raw = backend.generate("\n\n".join(scratch + [hdr]), system=system)
        llm += 1
        msg = _first_json(raw)
        if msg is None:
            perr += 1
            scratch.append("OBSERVATION: invalid protocol JSON; reply with "
                           'one object containing "action" or "final".')
            if perr >= 3:
                obs = "\n".join(x for x in scratch
                                if x.startswith("OBSERVATION"))
                out = backend.generate(FORCE.format(obs=obs[:6000] or "(none)"))
                llm += 1
                t = parse_ticket(out)
                return t, {"llm_calls": llm, "tool_calls": tool,
                           "protocol_errors": perr, "tools": tools,
                           "termination": ("final_forced" if t else
                                           "parse_failed")}, tb.last_ev
            continue
        if "final" in msg:
            t = parse_ticket(msg.get("final"))
            if t is None:
                scratch.append("OBSERVATION: your final was not a valid "
                               "ticket; emit final again with exactly the "
                               "template fields.")
                continue
            return t, {"llm_calls": llm, "tool_calls": tool,
                       "protocol_errors": perr, "tools": tools,
                       "termination": "final_model"}, tb.last_ev
        act = msg.get("action") or {}
        name = str(act.get("tool", "")) if isinstance(act, dict) else ""
        args = act.get("args") if isinstance(act, dict) else {}
        if not isinstance(args, dict):
            args = {}
        if name not in tb.names():
            perr += 1
            scratch.append(f"OBSERVATION: '{name}' is not a tool. "
                           f"Available: {', '.join(tb.names())}.")
            continue
        key = name + json.dumps(args, sort_keys=True, default=str)
        if key in seen:
            scratch.append(f"OBSERVATION: {name} already called; answer or "
                           f"pick another tool.")
            continue
        seen.add(key)
        obs = tb.call(name, args)
        tool += 1
        tools.append(name)
        scratch += [f"ACTION: {name}({json.dumps(args, default=str)})",
                    f"OBSERVATION: {obs[:2400]}"]
    return None, {"llm_calls": llm, "tool_calls": tool,
                  "protocol_errors": perr, "tools": tools,
                  "termination": "step_budget"}, tb.last_ev


# specialists ---------------------------------------------------------------
ANALYST_SYSTEM = ("You are the retrieval analyst of a fleet prognostics "
                  "team. Summarise the retrieved precedents for your "
                  "colleagues: the sensor pattern they share with the "
                  "current anomaly, the spread of the remaining life they "
                  "ACTUALLY had (rul_then), and any outlier precedent. "
                  "3-5 plain-text sentences. Do NOT give an estimate.")

ESTIMATOR_SYSTEM = """You are the prognostics estimator of a fleet
maintenance team. Combine the current evidence, your analyst's brief, and
the precedents' actual outcomes into a remaining-useful-life estimate.

OUTPUT PROTOCOL - reply with ONE JSON object and NOTHING else:
{"rul_estimate": <integer 0-125>, "rul_range": [<integer lo>, <integer hi>],
 "cited_precedents": ["<id>", ...]}
"rul_range" is your central 80% interval. Anchor the estimate in the
precedents' rul_then values and cite the ids you used."""

PLANNER_SYSTEM = """You are the maintenance planner of a fleet maintenance
team. Given the estimator's remaining-useful-life estimate and the current
evidence, choose the maintenance action and justify it.

OUTPUT PROTOCOL - reply with ONE JSON object and NOTHING else:
{"action": <integer 1-5>, "rationale": "<2-4 sentences grounded ONLY in the
evidence; never invent sensor values>"}
Action scale: 1=continue normal operation, 2=monitor closely, 3=schedule
inspection, 4=plan maintenance within the interval, 5=immediate
intervention / stop."""


def run_specialists(case: str, query: str, ev: Dict, backend: Backend
                    ) -> Tuple[Optional[Dict], Dict]:
    brief = backend.generate(
        f"CURRENT ANOMALY\n{case}\n\nRETRIEVED PRECEDENTS:\n"
        f"{evidence_text(ev)}\n\nWrite the analyst brief now.",
        system=ANALYST_SYSTEM)
    llm = 1
    est_prompt = (f"CURRENT ANOMALY\n{case}\n\nEDGE INTERPRETATION:\n{query}"
                  f"\n\nANALYST BRIEF:\n{(brief or '')[:900]}\n\nRETRIEVED "
                  f"PRECEDENTS:\n{evidence_text(ev)}\n\nProduce the "
                  f"estimator JSON now.")
    out = backend.generate(est_prompt, system=ESTIMATOR_SYSTEM)
    llm += 1
    e = _first_json(out or "")
    if e is None or "rul_estimate" not in e:
        out = backend.generate(est_prompt + "\n\nReply with ONLY the JSON "
                                            "object.",
                               system=ESTIMATOR_SYSTEM)
        llm += 1
        e = _first_json(out or "")
    stage = None
    if e is None or "rul_estimate" not in e:
        stage = "estimator"
    ticket = None
    if stage is None:
        partial = {"rul_estimate": e.get("rul_estimate"),
                   "rul_range": e.get("rul_range"),
                   "action": 3, "cited_precedents":
                       e.get("cited_precedents", []), "rationale": ""}
        base = parse_ticket(partial)
        if base is None:
            stage = "estimator"
        else:
            pl_prompt = (f"ESTIMATE: rul_estimate={base['rul_estimate']} "
                         f"cycles, rul_range={base['rul_range']} (central "
                         f"80%).\n\nCURRENT ANOMALY\n{case[:900]}\n\n"
                         f"Produce the planner JSON now.")
            pout = backend.generate(pl_prompt, system=PLANNER_SYSTEM)
            llm += 1
            p = _first_json(pout or "")
            act = _to_action((p or {}).get("action"))
            if act is None:
                pout = backend.generate(pl_prompt + "\n\nReply with ONLY "
                                                    "the JSON object.",
                                        system=PLANNER_SYSTEM)
                llm += 1
                p = _first_json(pout or "")
                act = _to_action((p or {}).get("action"))
            if act is None:
                stage = "planner"
            else:
                ticket = dict(base)
                ticket["action"] = act
                ticket["rationale"] = str((p or {}).get("rationale", ""))[:900]
    return ticket, {"llm_calls": llm, "tool_calls": 0,
                    "protocol_errors": 0, "tools": [],
                    "termination": ("composed" if ticket else
                                    f"parse_failed_{stage}"),
                    "analyst_brief": (brief or "")[:400]}


# ------------------------------------------------------------------ dryrun
class TicketDryRun(Backend):
    """Deterministic plumbing stub for THIS module (llm.py's DryRun serves
    the earlier arms). Reads the neighbour statistics straight out of the
    prompt and answers with the implied ticket; numbers are meaningless by
    design -- only the mechanics are under test."""
    name = "dryrun"
    model = "dryrun"

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path

    @staticmethod
    def _ticket_from(prompt: str, narrow: bool = False) -> str:
        m = re.search(r'"median_rul_then":\s*([\d.]+)', prompt)
        if m:
            est = int(round(float(m.group(1))))
            q1 = re.search(r'"q10":\s*([\d.]+)', prompt)
            q9 = re.search(r'"q90":\s*([\d.]+)', prompt)
            lo = int(round(float(q1.group(1)))) if q1 else max(0, est - 15)
            hi = int(round(float(q9.group(1)))) if q9 else min(RMAX, est + 15)
            if hi - lo < 4:
                lo, hi = max(0, est - 5), min(RMAX, est + 5)
            if narrow:                      # gate-compliant repair behaviour
                lo, hi = max(0, est - 40), min(RMAX, est + 40)
            ids = re.findall(r'"id":\s*"(u\d+c\d+)"', prompt)[:2]
        else:
            est, lo, hi, ids = 60, 40, 85, []
        return json.dumps({
            "rul_estimate": est, "rul_range": [lo, hi],
            "action": severity_band(est), "cited_precedents": ids,
            "rationale": "[dry-run] estimate anchored on the median outcome "
                         "of the retrieved precedents; range spans their "
                         "q10-q90."})

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        sysb = system or ""
        if "retrieval analyst" in sysb:
            out = ("[dry-run] The precedents share the dominant deviation "
                   "pattern of the current anomaly; their actual remaining "
                   "lives span the reported quantiles with no extreme "
                   "outlier.")
        elif '"action": <integer 1-5>' in sysb and '"rul_estimate"' not in sysb:
            m = re.search(r"rul_estimate=(\d+)", prompt)
            est = int(m.group(1)) if m else 60
            out = json.dumps({"action": severity_band(est),
                              "rationale": "[dry-run] action follows the "
                                           "estimate's severity band."})
        elif '"tool"' in sysb:                       # ReAct protocol
            if '"median_rul_then"' not in prompt:
                out = json.dumps({"thought": "retrieve precedents",
                                  "action": {"tool": "similar_interpretations",
                                             "args": {}}})
            else:
                out = json.dumps({"thought": "use the precedents",
                                  "final": json.loads(
                                      self._ticket_from(prompt))})
        else:                                        # single-shot / repair
            out = self._ticket_from(prompt,
                                    narrow="acceptance gates" in prompt)
        self._log(system, prompt, out)
        return out


# ------------------------------------------------------------------ scoring
def winkler(lo: float, hi: float, y: float, alpha: float = ALPHA) -> float:
    w = hi - lo
    if y < lo:
        w += (2.0 / alpha) * (lo - y)
    elif y > hi:
        w += (2.0 / alpha) * (y - hi)
    return w


def score_row(ticket: Optional[Dict], gold_rul: int, gold_action: int
              ) -> Dict:
    if ticket is None:
        return {"parse_failed": True}
    est = ticket["rul_estimate"]
    lo, hi = ticket["rul_range"]
    d = est - gold_rul
    return {"parse_failed": False, "pred_rul": est, "lo": lo, "hi": hi,
            "err": d, "abs_err": abs(d),
            "s_i": float(s_score([est], [gold_rul])),
            "in_range": bool(lo <= gold_rul <= hi), "width": hi - lo,
            "winkler": round(winkler(lo, hi, gold_rul), 2),
            "pred_action": ticket["action"],
            "action_exact": ticket["action"] == gold_action,
            "action_pm1": abs(ticket["action"] - gold_action) <= 1,
            "action_abs_diff": abs(ticket["action"] - gold_action)}


# ------------------------------------------------------------------- cases
@dataclass(frozen=True)
class Case:
    unit: int
    cycle: int
    rul: int
    action: int


def test_cases(ds: FD002) -> List[Case]:
    an = ds.df[(ds.df["anomaly_label"] == -1)
               & (ds.df["unit_ID"].isin(ds.test_units))]
    return [Case(int(r.unit_ID), int(r.cycle),
                 ds.rul(int(r.unit_ID), int(r.cycle)),
                 severity_band(ds.rul(int(r.unit_ID), int(r.cycle))))
            for r in an.itertuples()]


def sample_cases(ds: FD002, per_bucket: int, seed: int,
                 full: bool = False) -> List[Case]:
    pool = test_cases(ds)
    if full:
        return pool
    rng = np.random.default_rng(seed)
    out: List[Case] = []
    for lo, hi, _ in BUCKETS:
        cand = [c for c in pool if lo <= c.rul <= hi]
        idx = rng.choice(len(cand), size=min(per_bucket, len(cand)),
                         replace=False)
        out.extend(cand[int(i)] for i in idx)
    rng.shuffle(out)
    return out


# ------------------------------------------------------------------- runner
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="dryrun",
                    choices=["dryrun", "ollama", "openai"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--per-bucket", type=int, default=30)
    ap.add_argument("--full", action="store_true",
                    help="all 986 test anomalies instead of the sample")
    ap.add_argument("--sample-seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--query", default="auto",
                    choices=["auto", "summary"],
                    help="auto = the anomaly's own interpretation with "
                         "summary fallback; summary = force the grounded "
                         "summary (query-representation ablation)")
    ap.add_argument("--store-dir", default=None)
    ap.add_argument("--device", default="orin_nano_8gb")
    ap.add_argument("--hw-mode", default="model", choices=["model", "off"])
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    model = a.model or ("gpt-4o-mini" if a.backend == "openai"
                        else "dryrun" if a.backend == "dryrun" else "llama3")
    out = ROOT / a.out
    out.mkdir(exist_ok=True)
    tag = f"{model.replace(':', '_').replace('/', '_')}_seed{a.sample_seed}"

    ds = FD002(seed=42)

    # store + query embedder ------------------------------------------------
    from ..vector_store import STORE_DIR, VectorStore, get_embedder
    store_dir = Path(a.store_dir) if a.store_dir else STORE_DIR
    if not (store_dir / "embeddings.npz").exists():
        raise SystemExit("[prog] no vector store; run: python -m "
                         "apdm.vector_store --build --embedder ollama")
    store = VectorStore.load(store_dir)
    emb_kind = "ollama" if store.embedder_name.startswith("ollama") else "hash"
    embedder = get_embedder(emb_kind)
    print(f"[prog] store: {len(store.meta)} train interpretations, "
          f"embedder={store.embedder_name}"
          + (" [HASH FALLBACK - pipeline test only, never report]"
             if emb_kind == "hash" else ""))

    kb = InterpretationKB(ds, cache=ROOT / "cache" / "kb.pkl")
    fault_layer = None
    try:
        from .faults import build_fault_layer
        fault_layer = build_fault_layer(ds, cache=ROOT / "cache" /
                                        "faults.pkl")
    except Exception as exc:  # noqa: BLE001
        print(f"[prog] fault layer unavailable ({exc}); ReAct runs without "
              f"fault_library")

    interp = load_test_interpretations(ds)
    cases = sample_cases(ds, a.per_bucket, a.sample_seed, full=a.full)
    if a.limit:
        cases = cases[: a.limit]
    n_with = sum(1 for c in cases if (c.unit, c.cycle) in interp)
    print(f"[prog] {len(cases)} test-anomaly cases "
          f"({n_with} with their own interpretation, "
          f"{len(cases) - n_with} on summary fallback"
          + (", forced summary" if a.query == "summary" else "") + ")")
    if a.query == "auto" and n_with < len(cases):
        print("[prog] NOTE: generate the missing test interpretations with "
              "  python -m apdm.gen_interpretations --backend ollama "
              "--model llama3.2:3b --units test")

    backend: Backend
    if a.backend == "dryrun":
        backend = TicketDryRun(log_path=out / f"prog_calls_{tag}.jsonl")
        if a.device and a.hw_mode == "model":
            from ..hardware import CostModel
            try:
                backend.cost = CostModel(a.device, "llama3.2:3b",
                                         num_ctx=8192)
            except Exception:  # noqa: BLE001
                backend.cost = None
    else:
        backend = get_backend(a.backend, model,
                              log_path=out / f"prog_calls_{tag}.jsonl",
                              device=a.device if a.hw_mode == "model"
                              else None, hw_mode=a.hw_mode)
    if backend.cost is not None:
        (out / f"prog_hardware_{tag}.json").write_text(
            json.dumps(backend.cost.provenance(), indent=2))

    # resume ----------------------------------------------------------------
    rows_path = out / f"prog_rows_{tag}.jsonl"
    done_rows: List[Dict] = []
    done_keys = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done_rows.append(r)
                done_keys.add((r["arm"], r["unit"], r["cycle"]))
        if done_rows:
            print(f"[prog] resume: {len(done_rows)} rows already done")

    def emit(row: Dict) -> None:
        done_rows.append(row)
        with open(rows_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    for arm in a.arms:
        todo = [c for c in cases if (arm, c.unit, c.cycle) not in done_keys]
        print(f"[prog] === {arm} | {model} | {len(todo)}/{len(cases)} "
              f"cases ===")
        for i, c in enumerate(todo, 1):
            t0 = time.time()
            tb0 = backend.totals()
            if hasattr(backend, "reset_prefix"):
                backend.reset_prefix()
            qtext = (interp.get((c.unit, c.cycle)) if a.query == "auto"
                     else None)
            qsource = "interpretation" if qtext else "summary_fallback"
            if qtext is None:
                qtext = summary_text(ds, Snapshot(c.unit, c.cycle, 0),
                                     top_k=8)
            qtext = qtext[:1200]
            case = case_context(ds, c.unit, c.cycle)
            zc = current_z(ds, c.unit, c.cycle)
            meta: Dict = {"llm_calls": 0, "tool_calls": 0,
                          "protocol_errors": 0, "tools": [],
                          "termination": "deterministic"}
            ev: Optional[Dict] = None
            ticket: Optional[Dict] = None
            extra: Dict = {}

            if arm == "B0_retrieval":
                ev = retrieve_semantic(ds, store, embedder, qtext, c.unit,
                                       a.k)
                ticket = run_b0(ev)
            elif arm == "B1_zknn":
                ev = retrieve_zknn(ds, kb, c.unit, c.cycle, a.k)
                ticket = run_b0(ev)
            elif arm == "P1_direct":
                ticket, raw, n = _gen_ticket(backend,
                                             prompt_p1(case, qtext))
                meta.update(llm_calls=n, termination="direct")
            elif arm in ("P2_rag", "P4_reflexion", "P5_verifier"):
                ev = retrieve_semantic(ds, store, embedder, qtext, c.unit,
                                       a.k)
                body = prompt_p2(case, qtext, ev)
                ticket, raw, n = _gen_ticket(backend, body)
                meta.update(llm_calls=n, termination="direct")
                if arm == "P4_reflexion" and ticket is not None:
                    rev_raw = backend.generate(
                        CRITIQUE.format(draft=json.dumps(ticket),
                                        body=body[:7000]),
                        system=TICKET_SYSTEM)
                    meta["llm_calls"] += 1
                    rev = parse_ticket(rev_raw)
                    extra["revised"] = rev is not None
                    extra["revision_changed"] = (rev is not None
                                                 and rev != ticket)
                    if rev is not None:
                        ticket = rev
                    meta["termination"] = "reflexion"
                if arm == "P5_verifier" and ticket is not None:
                    viol, _ = verify_ticket(ticket, ev, zc)
                    extra["violations_pre"] = len(viol)
                    extra["violations_pre_list"] = "; ".join(
                        v.split(":")[0] for v in viol)
                    if viol:
                        rep_raw = backend.generate(
                            REPAIR.format(draft=json.dumps(ticket),
                                          viol="\n".join("- " + v
                                                         for v in viol),
                                          body=body[:7000]),
                            system=TICKET_SYSTEM)
                        meta["llm_calls"] += 1
                        rep = parse_ticket(rep_raw)
                        if rep is not None:
                            ticket = rep
                        viol2, _ = verify_ticket(ticket, ev, zc)
                        extra["violations_post"] = len(viol2)
                        extra["escalated"] = bool(viol2)
                    else:
                        extra["violations_post"] = 0
                        extra["escalated"] = False
                    meta["termination"] = ("escalated"
                                           if extra.get("escalated")
                                           else "verified")
            elif arm == "P3_react":
                ticket, meta, ev = run_react(ds, store, embedder, c.unit,
                                             c.cycle, qtext, backend, a.k,
                                             a.max_steps, fault_layer)
            elif arm == "P6_specialists":
                ev = retrieve_semantic(ds, store, embedder, qtext, c.unit,
                                       a.k)
                ticket, meta = run_specialists(case, qtext, ev, backend)

            row = {"arm": arm, "model": model, "unit": c.unit,
                   "cycle": c.cycle, "gold_rul": c.rul,
                   "gold_action": c.action, "query_source": qsource,
                   "k_retrieved": (ev or {}).get("k", 0),
                   "seconds": round(time.time() - t0, 2),
                   **{k_: meta.get(k_) for k_ in
                      ("llm_calls", "tool_calls", "protocol_errors",
                       "termination")},
                   "tools": "|".join(meta.get("tools", [])), **extra,
                   **score_row(ticket, c.rul, c.action)}
            if ticket is not None:
                viol_all, faith = verify_ticket(ticket, ev, zc)
                row.update(faith)
                row["post_hoc_violations"] = len(viol_all)
                row["cited_valid"] = bool(ev) and any(
                    p["id"] in ticket["cited_precedents"]
                    for p in (ev or {}).get("precedents", []))
                row["answer"] = json.dumps(ticket)[:700]
            tb1 = backend.totals()
            row.update({k_: round(tb1[k_] - tb0.get(k_, 0), 4)
                        for k_ in ("prompt_tokens", "completion_tokens",
                                   "sim_edge_s", "sim_energy_j")
                        if k_ in tb1})
            emit(row)
            if i % 20 == 0:
                print(f"[prog]   {i}/{len(todo)}")
        sub = pd.DataFrame([r for r in done_rows if r["arm"] == arm])
        ok = sub[~sub.parse_failed.astype(bool)]
        if len(ok):
            print(f"[prog] {arm}: parse_ok {len(ok)}/{len(sub)}, "
                  f"MAE {ok.abs_err.mean():.1f}, "
                  f"coverage {ok.in_range.mean():.2f}, "
                  f"width {ok.width.mean():.0f}, "
                  f"action +/-1 {ok.action_pm1.mean():.2f}")
        else:
            print(f"[prog] {arm}: parse_ok 0/{len(sub)}")

    df = pd.DataFrame(done_rows)
    df.to_csv(out / f"prog_predictions_{tag}.csv", index=False)
    print(f"[prog] wrote prog_predictions_{tag}.csv "
          f"({len(df)} rows)  totals={backend.totals()}")


if __name__ == "__main__":
    main()

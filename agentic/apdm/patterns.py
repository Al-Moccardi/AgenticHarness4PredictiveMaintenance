"""Agentic pattern grid over the fleet vector store — LLM-local, ML-free.

The experiment your RAD Table III started (prompting strategies x RAGAS),
lifted to the AGENTIC level: the same diagnostic case is answered by seven
coordination patterns, each under three prompting styles, always grounded in
the SAME frozen train-fleet vector store (nomic-embed-text, n=4,458). No ML
models are loaded, no OpenAI is called: the coordinator LLM and the RAGAS
judge are both local (Ollama).

Queries are the EDGE tier's own outputs — the interpretations your Jetson
wrote for official-test anomalies (test_FD002_with_interpretations.csv) — so
the run is the literal cloud-to-thing loop: edge writes, coordinator reads.
Query units are namespaced ("T65") and never touch the store (guard G1).

Patterns
  B0_retrieval     no LLM. Ticket assembled from the k neighbours (median
                   rul_then, span, band-mapped action). The floor every
                   agent must beat to justify its tokens.
  P1_direct        single shot, case evidence only, no retrieval.
  P2_rag           single shot, case + top-k precedents.
  P3_react         tool loop (<=4 steps): search_memory / read_precedent /
                   finish. The model decides what to retrieve.
  P4_reflexion     P2 draft -> self-critique -> revision.
  P5_verifier      P2 draft -> deterministic gates -> repair loop
                   (<=2 repairs) -> escalate flag if still failing.
  P6_specialists   analyst summarises precedents -> diagnostician drafts ->
                   reviewer checks citations and finalises.

Prompt styles (applied to every LLM pattern)
  plain      the protocol, nothing else
  cot        a mandatory "reasoning" field written BEFORE the conclusions
  fewshot    one worked example ticket in the prompt

Ticket protocol (all patterns emit the same JSON)
  {"diagnosis", "action" in {continue_monitoring, schedule_inspection,
   plan_maintenance, immediate_shutdown}, "rul_estimate", "rul_range",
   "cited_precedents" ["u<unit>c<cycle>", ...], "reasoning"}
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .llm import Backend
from .vector_store import VectorStore

EDGE_UNITS = (65, 103, 110, 131, 135, 209, 222, 245)   # the Jetson run's units

ACTIONS = ["continue_monitoring", "schedule_inspection",
           "plan_maintenance", "immediate_shutdown"]
# outcome-derived bands (train-fleet convention): RUL>120 monitor, 60-120
# inspect, 25-60 plan, <25 shutdown
BANDS = [(120, "continue_monitoring"), (60, "schedule_inspection"),
         (25, "plan_maintenance"), (-1, "immediate_shutdown")]

TICKET = ('{"diagnosis": "<one sentence>", '
          '"action": "continue_monitoring|schedule_inspection|'
          'plan_maintenance|immediate_shutdown", '
          '"rul_estimate": <int cycles>, "rul_range": [<lo>, <hi>], '
          '"cited_precedents": ["u<unit>c<cycle>", ...], '
          '"reasoning": "<why, grounded in the evidence>"}')

SYSTEM = ("You are the fleet maintenance coordinator for turbofan engines. "
          "You receive one anomaly case from an edge device and, when "
          "provided, precedent cases retrieved from the fleet memory. "
          "OUTPUT PROTOCOL: answer with ONE JSON object exactly matching:\n"
          + TICKET + "\nHARD RULES: (1) cite ONLY precedent ids that were "
          "shown to you; (2) rul_estimate (cycles) MUST lie inside the "
          "rul_then range of the precedents you cite; (3) action MUST match "
          "the band of rul_estimate: >120 continue_monitoring, 60-120 "
          "schedule_inspection, 25-60 plan_maintenance, <25 "
          "immediate_shutdown.")

FEWSHOT = ("EXAMPLE TICKET (for format only):\n"
           '{"diagnosis": "Progressive HPC efficiency loss consistent with '
           'three late-life precedents.", "action": "plan_maintenance", '
           '"rul_estimate": 42, "rul_range": [30, 55], '
           '"cited_precedents": ["u12c188", "u77c201"], '
           '"reasoning": "Neighbours at similar counters failed 35-60 '
           'cycles later; rule sensors T50/NRc match precedent u12c188."}')

COT = ("Write the \"reasoning\" field FIRST and think step by step inside "
       "it: (1) what the rule and counters say, (2) which precedents match "
       "and their rul_then, (3) how you map that to an action band. Then "
       "fill the remaining fields consistently with your reasoning.")


# ------------------------------------------------------------------ queries
def load_queries(csv_path: Path, units=EDGE_UNITS) -> List[Dict]:
    """Edge-written anomalies -> coordinator queries, RESTRICTED to the same
    stratified unit set the edge tier processed. Namespaced unit ids so
    official-test unit 65 can never collide with train unit 65."""
    d = pd.read_csv(csv_path)
    if units:
        u0 = "Unit_ID" if "Unit_ID" in d.columns else "unit_ID"
        d = d[d[u0].isin(units)]
    u = "Unit_ID" if "Unit_ID" in d.columns else "unit_ID"
    c = "cycles" if "cycles" in d.columns else "cycle"
    an = d[(d.anomaly_label == -1)
           & d.interpretation.astype(str).str.len().gt(50)].copy()
    out = []
    for _, r in an.sort_values([u, c]).iterrows():
        out.append({
            "qid": f"T{int(r[u])}c{int(r[c])}",
            "unit": int(r[u]), "cycle": int(r[c]),
            "interpretation": str(r.interpretation).strip(),
            "rule": str(r.text)[:400],
            "score": float(r.anomaly_score),
            "counters": {
                "local_count": int(r.local_cumulative_anomaly_count),
                "local_last3": float(r.local_last_3_freq),
                "global_count": int(r.global_cumulative_anomaly_count),
                "global_last3": float(r.global_last_3_freq)},
            "true_rul": float(r.RUL) if "RUL" in an.columns else None})
    return out


def case_block(q: Dict) -> str:
    ct = q["counters"]
    return (f"CURRENT CASE (unit {q['qid']})\n"
            f"Edge interpretation:\n{q['interpretation'][:1100]}\n"
            f"Isolation rule: {q['rule'][:260]}\n"
            f"Anomaly score: {q['score']:.4f} (more negative = stronger)\n"
            f"Causal counters: local={ct['local_count']} "
            f"(last3 freq {ct['local_last3']:.2f}), "
            f"global={ct['global_count']} "
            f"(last3 freq {ct['global_last3']:.2f})"
            + ("\n" + q["dl_hint_line"] if q.get("dl_hint_line") else ""))


def fmt_precedents(recs: List[Dict], full: bool = False) -> str:
    lines = []
    for r in recs:
        pid = f"u{r['unit']}c{r['cycle']}"
        head = (f"[{pid}] similarity={r['similarity']:.3f} "
                f"rul_then={r['rul_then']} gravity={r.get('gravity')}")
        body = r["text"] if full else r["text"][:280]
        lines.append(f"{head}\n{body}")
    ruls = [r["rul_then"] for r in recs if r["rul_then"] is not None]
    span = (f"OUTCOME SPAN of these precedents: rul_then {min(ruls):.0f}-"
            f"{max(ruls):.0f} cycles. Your rul_estimate must fall inside "
            f"the span of the ones you cite.\n" if ruls else "")
    return span + "PRECEDENTS FROM FLEET MEMORY:\n" + "\n---\n".join(lines)


def style_suffix(style: str) -> str:
    if style == "cot":
        return "\n\n" + COT
    if style == "fewshot":
        return "\n\n" + FEWSHOT
    return ""


# ------------------------------------------------------------------ parsing
def parse_ticket(text: str) -> Tuple[Optional[Dict], bool]:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None, False
    try:
        j = json.loads(m.group(0))
    except json.JSONDecodeError:
        try:
            j = json.loads(m.group(0).replace("'", '"'))
        except Exception:  # noqa: BLE001
            return None, False
    if not isinstance(j, dict):
        return None, False
    out = {"diagnosis": str(j.get("diagnosis", ""))[:400],
           "action": str(j.get("action", "")).strip(),
           "reasoning": str(j.get("reasoning", ""))[:1200],
           "cited_precedents": [str(x) for x in
                                (j.get("cited_precedents") or [])][:8]}
    try:
        out["rul_estimate"] = float(j.get("rul_estimate"))
    except (TypeError, ValueError):
        out["rul_estimate"] = None
    rr = j.get("rul_range") or []
    try:
        out["rul_range"] = [float(rr[0]), float(rr[1])]
    except Exception:  # noqa: BLE001
        out["rul_range"] = None
    return out, True


def band_action(rul: float) -> str:
    for thr, act in BANDS:
        if rul > thr:
            return act
    return "immediate_shutdown"


# ------------------------------------------------------------------- gates
def gates(ticket: Optional[Dict], shown_ids: List[str]) -> List[str]:
    v = []
    if ticket is None:
        return ["G0_json_invalid"]
    if ticket["action"] not in ACTIONS:
        v.append("G1_action_invalid")
    if ticket["rul_estimate"] is None:
        v.append("G2_rul_missing")
    cited = ticket["cited_precedents"]
    if not cited:
        v.append("G3_no_citations")
    elif any(c not in shown_ids for c in cited):
        v.append("G4_citation_not_shown")
    if (ticket["rul_estimate"] is not None and ticket["rul_range"]
            and not (ticket["rul_range"][0] - 1e-6 <= ticket["rul_estimate"]
                     <= ticket["rul_range"][1] + 1e-6)):
        v.append("G5_estimate_outside_range")
    if ticket["rul_estimate"] is not None and ticket["action"] in ACTIONS:
        if ticket["action"] != band_action(ticket["rul_estimate"]):
            v.append("G6_action_band_mismatch")
    return v


# ------------------------------------------------------------- the patterns
class Runner:
    def __init__(self, be: Optional[Backend], store: VectorStore,
                 embedder, k: int = 4, stage_aware: bool = False,
                 stage_lambda: float = 0.35):
        self.be, self.store, self.embedder, self.k = be, store, embedder, k
        self.stage_aware, self.stage_lambda = stage_aware, stage_lambda
        # per-precedent life-stage signature, computed once from the meta:
        # (anomaly ordinal within its unit, absolute cycle) both normalised.
        self._stage: Dict = {}
        if stage_aware:
            by_unit: Dict[int, List] = {}
            for m in store.meta:
                by_unit.setdefault(int(m["unit"]), []).append(m)
            for u, ms in by_unit.items():
                for i, m in enumerate(sorted(ms, key=lambda r: r["cycle"])):
                    self._stage[(int(m["unit"]), int(m["cycle"]))] = (
                        min(i, 30) / 30.0, min(int(m["cycle"]), 300) / 300.0)

    # retrieval used by every retrieval-bearing pattern
    def retrieve(self, q: Dict, k: Optional[int] = None) -> List[Dict]:
        kk = k or self.k
        qv = self.embedder.embed([q["interpretation"]])[0]
        if not self.stage_aware:
            return self.store.search(qv, k=kk)
        # LEAK-FREE query stage: only observable quantities (its own anomaly
        # count so far and its cycle) -- never RUL.
        qs = (min(q["counters"]["local_count"], 30) / 30.0,
              min(q["cycle"], 300) / 300.0)
        cand = self.store.search(qv, k=max(6 * kk, 24))
        for r in cand:
            ps = self._stage.get((int(r["unit"]), int(r["cycle"])), qs)
            r["stage_dist"] = abs(qs[0] - ps[0]) + abs(qs[1] - ps[1])
            r["rerank"] = r["similarity"] - self.stage_lambda * r["stage_dist"]
        cand.sort(key=lambda r: r["rerank"], reverse=True)
        return cand[:kk]

    def _ask(self, prompt: str, style: str) -> str:
        return self.be.generate(prompt + style_suffix(style), system=SYSTEM)

    # ---- B0: no LLM ------------------------------------------------------
    def b0(self, q: Dict) -> Dict:
        recs = self.retrieve(q)
        ruls = [r["rul_then"] for r in recs if r["rul_then"] is not None]
        est = float(np.median(ruls)) if ruls else None
        rng = ([float(np.quantile(ruls, .1)), float(np.quantile(ruls, .9))]
               if ruls else None)
        t = {"diagnosis": "nearest-precedent consensus (no LLM)",
             "action": band_action(est) if est is not None else "",
             "rul_estimate": est, "rul_range": rng,
             "cited_precedents": [f"u{r['unit']}c{r['cycle']}" for r in recs],
             "reasoning": ""}
        return {"ticket": t, "contexts": recs, "steps": 0}

    # ---- P1 / P2 ---------------------------------------------------------
    def p1(self, q, style):
        out = self._ask(case_block(q) + "\n\nProduce the ticket now.", style)
        t, _ = parse_ticket(out)
        return {"ticket": t, "contexts": [], "raw": out, "steps": 1}

    def p2(self, q, style):
        recs = self.retrieve(q)
        out = self._ask(case_block(q) + "\n\n" + fmt_precedents(recs)
                        + "\n\nProduce the ticket now.", style)
        t, _ = parse_ticket(out)
        return {"ticket": t, "contexts": recs, "raw": out, "steps": 1}

    # ---- P3: ReAct tool loop (v2: seeded, tolerant, self-recovering) ----
    def p3(self, q, style):
        # v2: seed the loop with an initial retrieval so the agent starts
        # from evidence and REFINES it, instead of having to bootstrap
        # tool-driving from nothing (the measured 3B failure mode).
        seen: Dict[str, Dict] = {}
        for r in self.retrieve(q):
            seen[f"u{r['unit']}c{r['cycle']}"] = r
        obs = fmt_precedents(list(seen.values()))
        trace = []
        proto = ('Respond with ONE JSON object: {"tool": "search_memory", '
                 '"query": "<text>"} or {"tool": "read_precedent", '
                 '"id": "u<unit>c<cycle>"} or {"tool": "finish", '
                 '"ticket": ' + TICKET + "}")
        for step in range(3):
            p = (case_block(q) + f"\n\nOBSERVATIONS SO FAR:\n{obs}\n\n"
                 f"Step {step+1}/3. Refine the evidence or finish.\n" + proto)
            out = self.be.generate(p + style_suffix(style), system=SYSTEM)
            m = re.search(r"\{.*\}", out or "", re.S)
            act = None
            if m:
                try:
                    act = json.loads(m.group(0))
                except Exception:  # noqa: BLE001
                    act = None
            if not isinstance(act, dict):
                # v2 tolerant fallback: recover the tool name from raw text
                low = (out or "").lower()
                if "finish" in low:
                    t, _ = parse_ticket(out)
                    if t:
                        trace.append({"step": step, "raw": (out or "")[:300]})
                        return {"ticket": t,
                                "contexts": list(seen.values()),
                                "trace": trace, "steps": step + 1}
                act = {"tool": "search_memory", "query": q["interpretation"][:120]}
            trace.append({"step": step, "raw": (out or "")[:300]})
            tool = str(act.get("tool", ""))
            if tool == "finish":
                t, _ = parse_ticket(json.dumps(act.get("ticket", {})))
                return {"ticket": t, "contexts": list(seen.values()),
                        "trace": trace, "steps": step + 1}
            if tool == "search_memory":
                qq = dict(q)
                qq["interpretation"] = str(act.get("query")
                                           or q["interpretation"])
                for r in self.retrieve(qq):
                    seen[f"u{r['unit']}c{r['cycle']}"] = r
                obs = fmt_precedents(list(seen.values()))
            elif tool == "read_precedent":
                pid = str(act.get("id", ""))
                r = seen.get(pid)
                obs = (fmt_precedents([r], full=True) if r
                       else obs + f"\n(id {pid} not in memory view)")
        out = self._ask(case_block(q) + "\n\n"
                        + fmt_precedents(list(seen.values()))
                        + "\n\nYou are out of tool steps. Produce the ticket "
                          "NOW, citing only the ids above.", style)
        t, _ = parse_ticket(out)
        return {"ticket": t, "contexts": list(seen.values()),
                "trace": trace, "steps": 4}

    # ---- P4: reflexion ---------------------------------------------------
    def p4(self, q, style):
        first = self.p2(q, style)
        recs = first["contexts"]
        critique = self.be.generate(
            case_block(q) + "\n\n" + fmt_precedents(recs)
            + "\n\nDRAFT TICKET:\n" + json.dumps(first["ticket"] or {})
            + "\n\nCriticise the draft in <=3 bullet points: unsupported "
              "claims, wrong citations, action/RUL inconsistency. "
              "Text only, no JSON.", system=None)
        out = self._ask(case_block(q) + "\n\n" + fmt_precedents(recs)
                        + "\n\nDRAFT:\n" + json.dumps(first["ticket"] or {})
                        + "\n\nCRITIQUE:\n" + (critique or "")[:800]
                        + "\n\nProduce the REVISED ticket now. Keep action equal to "
                          "the band of rul_estimate (>120 monitor, 60-120 inspect, "
                          "25-60 plan, <25 shutdown); change other fields only where "
                          "the critique found a violation.", style)
        t, _ = parse_ticket(out)
        return {"ticket": t, "contexts": recs, "critique": critique,
                "steps": 3}

    # ---- P5: verifier-gated ---------------------------------------------
    def p5(self, q, style):
        cur = self.p2(q, style)
        recs = cur["contexts"]
        shown = [f"u{r['unit']}c{r['cycle']}" for r in recs]

        def _span(t):
            cited = (t or {}).get("cited_precedents") or []
            cr = [r["rul_then"] for r in recs
                  if f"u{r['unit']}c{r['cycle']}" in cited
                  and r.get("rul_then") is not None]
            if not cr:
                cr = [r["rul_then"] for r in recs
                      if r.get("rul_then") is not None]
            if not cr:
                return None
            lo, hi = min(cr), max(cr)
            pad = 0.2 * max(hi - lo, 10)
            return lo - pad, hi + pad

        def _viol(t):
            v = gates(t, shown)
            sp = _span(t)
            est = (t or {}).get("rul_estimate")
            if sp and est is not None and not (sp[0] <= est <= sp[1]):
                v.append("G7_rul_outside_cited_span")
            return v

        repairs = 0
        viol = _viol(cur["ticket"])
        while viol and repairs < 2:
            out = self._ask(
                case_block(q) + "\n\n" + fmt_precedents(recs)
                + "\n\nYour previous ticket:\n"
                + json.dumps(cur["ticket"] or {})
                + "\n\nIt FAILED these deterministic checks: "
                + ", ".join(viol)
                + ". Fix ONLY what is violated (cite only shown ids; keep "
                  "rul_estimate inside rul_range; action must match the "
                  "band of rul_estimate: >120 monitor, 60-120 inspect, "
                  "25-60 plan, <25 shutdown). Produce the corrected ticket.",
                style)
            t, _ = parse_ticket(out)
            cur = {"ticket": t, "contexts": recs}
            repairs += 1
            viol = _viol(t)
        auto = False
        t_ = cur["ticket"]
        if (viol and set(viol) <= {"G6_action_band_mismatch"}
                and t_ and t_.get("rul_estimate") is not None):
            t_ = dict(t_)
            t_["action"] = band_action(t_["rul_estimate"])
            cur = {"ticket": t_, "contexts": recs}
            auto = True
            viol = _viol(t_)
        return {"ticket": cur["ticket"], "contexts": recs,
                "repairs": repairs, "violations_final": viol,
                "auto_corrected": auto,
                "escalated": bool(viol), "steps": 1 + repairs}

    # ---- P6: specialists -------------------------------------------------
    def p6(self, q, style):
        recs = self.retrieve(q)
        analysis = self.be.generate(
            case_block(q) + "\n\n" + fmt_precedents(recs, full=True)
            + "\n\nYou are the RETRIEVAL ANALYST. In <=6 bullet points: "
              "which precedents genuinely match this case (cite ids), their "
              "rul_then values, and which do not match. Text only.",
            system=None)
        draft_raw = self._ask(
            case_block(q) + "\n\nANALYST NOTES:\n" + (analysis or "")[:900]
            + "\n\nYou are the DIAGNOSTICIAN. Produce the ticket.", style)
        draft, _ = parse_ticket(draft_raw)
        shown = [f"u{r['unit']}c{r['cycle']}" for r in recs]
        review = self._ask(
            "You are the REVIEWER. Precedent ids that were actually shown: "
            + ", ".join(shown) + "\n\nTICKET UNDER REVIEW:\n"
            + json.dumps(draft or {})
            + "\n\nIf citations, action band (>120 monitor, 60-120 inspect, "
              "25-60 plan, <25 shutdown) and ranges are all valid, return "
              "the ticket unchanged; otherwise return the corrected ticket. "
              "Never change the action unless it mismatches the band of "
              "rul_estimate.",
            style)
        t, _ = parse_ticket(review)
        return {"ticket": t or draft, "contexts": recs,
                "analysis": analysis, "steps": 3}

    def run(self, pattern: str, q: Dict, style: str) -> Dict:
        if pattern == "B0_retrieval":
            return self.b0(q)
        return {"P1_direct": self.p1, "P2_rag": self.p2, "P3_react": self.p3,
                "P4_reflexion": self.p4, "P5_verifier": self.p5,
                "P6_specialists": self.p6}[pattern](q, style)


PATTERNS = ["B0_retrieval", "P1_direct", "P2_rag", "P3_react",
            "P4_reflexion", "P5_verifier", "P6_specialists"]
STYLES = ["plain", "cot", "fewshot"]


# ------------------------------------------------------- mock (smoke only)
class MockBackend(Backend):
    """Deterministic valid-JSON backend so the whole grid can be plumbing-
    tested offline. Never report its numbers."""
    name = "mock"
    model = "mock"

    def __init__(self, log_path=None):
        self.log_path = log_path

    def generate(self, prompt, system=None):
        ids = re.findall(r"\[(u\d+c\d+)\]", prompt)[:2]
        ruls = [float(x) for x in re.findall(r"rul_then=(\d+\.?\d*)", prompt)]
        est = int(np.median(ruls)) if ruls else 80
        if "compare_progression" in prompt and '"severity"' in prompt:
            ids = re.findall(r"\[(u\d+c\d+)\]", prompt)[:2]
            if "PROGRESSION COMPARISON" not in prompt and ids:
                out = json.dumps({"tool": "compare_progression",
                                  "id": ids[0]})
            else:
                gs = [int(x) for x in re.findall(r"gravity=(\d)", prompt)]
                sev = int(np.median(gs)) if gs else 3
                out = json.dumps({"tool": "finish", "ticket": {
                    "diagnosis": "mock fault", "matched_pattern":
                    "progression matches prior histories of precedents",
                    "severity": sev,
                    "action": {1: "continue_monitoring",
                               2: "continue_monitoring",
                               3: "schedule_inspection",
                               4: "plan_maintenance",
                               5: "immediate_shutdown"}[sev],
                    "cited_precedents": ids, "reasoning": "mock"}})
            self._log(system, prompt, out)
            return out
        if "DIAGNOSTIC agent" in (system or "") or '"severity"' in prompt:
            gs = [int(x) for x in re.findall(r"gravity=(\d)", prompt)]
            sev = int(np.median(gs)) if gs else 3
            out = json.dumps({"diagnosis": "mock fault characterization",
                              "matched_pattern": "progression matches the "
                              "prior history of cited precedents",
                              "severity": sev,
                              "action": {1: "continue_monitoring",
                                         2: "continue_monitoring",
                                         3: "schedule_inspection",
                                         4: "plan_maintenance",
                                         5: "immediate_shutdown"}[sev],
                              "cited_precedents": ids,
                              "reasoning": "mock"})
            self._log(system, prompt, out)
            return out
        if "Choose ONE tool" in prompt and "OBSERVATIONS SO FAR:\n(no" in prompt:
            out = json.dumps({"tool": "search_memory", "query": "similar"})
        elif "Choose ONE tool" in prompt:
            out = json.dumps({"tool": "finish", "ticket": {
                "diagnosis": "mock consensus", "action": band_action(est),
                "rul_estimate": est, "rul_range": [max(est - 15, 0), est + 15],
                "cited_precedents": ids, "reasoning": "mock"}})
        elif "Criticise" in prompt or "ANALYST" in prompt.upper():
            out = "- looks consistent\n- citations shown\n- band ok"
        else:
            out = json.dumps({"diagnosis": "mock diagnosis",
                              "action": band_action(est),
                              "rul_estimate": est,
                              "rul_range": [max(est - 15, 0), est + 15],
                              "cited_precedents": ids,
                              "reasoning": "mock reasoning grounded in "
                                           + ", ".join(ids)})
        self._log(system, prompt, out)
        return out


# ======================================================================
# DIAGNOSTIC PROTOCOL v3 - retrospective case-matching (no prognosis).
# The agent compares the monitored unit's anomaly PROGRESSION with the
# HISTORIES of the most similar precedents up to their matched point.
# No RUL, no futures, no rul_then anywhere in this protocol.
# ======================================================================
SEV2ACT = {1: "continue_monitoring", 2: "continue_monitoring",
           3: "schedule_inspection", 4: "plan_maintenance",
           5: "immediate_shutdown"}

TICKET_D = ('{"diagnosis": "<fault characterization, 1-2 sentences>", '
            '"matched_pattern": "<how THIS unit\'s progression matches the '
            'cited precedents\' histories>", "severity": <int 1-5>, '
            '"action": "continue_monitoring|schedule_inspection|'
            'plan_maintenance|immediate_shutdown", '
            '"cited_precedents": ["u<unit>c<cycle>", ...], '
            '"reasoning": "<grounded in rule sensors and progression>"}')

SYSTEM_D = ("You are the fleet DIAGNOSTIC agent. Your job is RETROSPECTIVE: "
            "characterise the fault by comparing the monitored unit's "
            "anomaly progression so far with the HISTORIES of similar "
            "precedent cases. Do NOT forecast and do NOT estimate remaining "
            "life - that is another agent's job. HARD RULES: cite only "
            "shown precedent ids; severity is 1 (benign) to 5 (critical), "
            "judged from the match between progressions and the precedents' "
            "gravity; action MUST follow severity: 1-2 continue_monitoring, "
            "3 schedule_inspection, 4 plan_maintenance, 5 "
            "immediate_shutdown. OUTPUT PROTOCOL: ONE JSON object:\n"
            + TICKET_D)

HIST_KW = re.compile(r"\b(previous|prior|progression|recurring|earlier|"
                     r"history|histories|count(er)?s?|past|first "
                     r"anomal|accelerat)\b", re.I)


def fmt_precedents_diag(recs: List[Dict], hist) -> str:
    lines = []
    for r in recs:
        pid = f"u{r['unit']}c{r['cycle']}"
        lines.append(
            f"[{pid}] similarity={r['similarity']:.3f} "
            f"gravity={r.get('gravity')}\n{str(r['text'])[:220]}\n"
            + hist.history_of(r["unit"], r["cycle"]))
    return ("PRECEDENTS FROM FLEET MEMORY (each with ITS OWN history up to "
            "the matched point):\n" + "\n---\n".join(lines))


def parse_ticket_d(text: str) -> Tuple[Optional[Dict], bool]:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None, False
    try:
        j = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        try:
            j = json.loads(m.group(0).replace("'", '"'))
        except Exception:  # noqa: BLE001
            return None, False
    if not isinstance(j, dict):
        return None, False
    out = {"diagnosis": str(j.get("diagnosis", ""))[:400],
           "matched_pattern": str(j.get("matched_pattern", ""))[:500],
           "action": str(j.get("action", "")).strip(),
           "reasoning": str(j.get("reasoning", ""))[:800],
           "cited_precedents": [str(x) for x in
                                (j.get("cited_precedents") or [])][:8]}
    try:
        out["severity"] = int(j.get("severity"))
    except (TypeError, ValueError):
        out["severity"] = None
    return out, True


def gates_d(t: Optional[Dict], shown: List[str], has_history: bool) -> List[str]:
    v = []
    if t is None:
        return ["D0_json_invalid"]
    if t["action"] not in ACTIONS:
        v.append("D1_action_invalid")
    if t["severity"] is None or not 1 <= t["severity"] <= 5:
        v.append("D2_severity_invalid")
    cited = t["cited_precedents"]
    if not cited:
        v.append("D3_no_citations")
    elif any(c not in shown for c in cited):
        v.append("D4_citation_not_shown")
    if (t["severity"] is not None and t["action"] in ACTIONS
            and t["action"] != SEV2ACT.get(t["severity"])):
        v.append("D5_severity_action_mismatch")
    if has_history and not HIST_KW.search(
            t["matched_pattern"] + " " + t["diagnosis"]):
        v.append("D6_progression_not_referenced")
    return v


DIAG_TOOLS = ('TOOLS (choose ONE per turn):\n'
              '{"tool": "search_memory", "query": "<text>"}\n'
              '{"tool": "read_precedent", "id": "u<unit>c<cycle>"}  '
              '(full interpretation text)\n'
              '{"tool": "read_history", "id": "u<unit>c<cycle>"}  '
              '(that unit\'s prior anomalies up to the matched point)\n'
              '{"tool": "compare_progression", "id": "u<unit>c<cycle>"}  '
              '(DETERMINISTIC side-by-side of the monitored unit\'s '
              'progression vs that precedent\'s history)\n'
              '{"tool": "finish", "ticket": ' + TICKET_D + "}")

_DSENS = ["T24", "T30", "T50", "P30", "Nf", "Nc", "Ps30", "phi", "NRf",
          "NRc", "BPR", "htBleed", "W31", "W32"]


class DiagRunner(Runner):
    """Retrospective diagnostic arms. Same floors, new protocol."""

    def __init__(self, *a, hist=None, unit_hist_fn=None, queries=None, **kw):
        super().__init__(*a, **kw)
        self.hist = hist
        self.unit_hist_fn = unit_hist_fn or (lambda q: "")
        self.queries = queries or []

    # ---- the deterministic comparison tool (no LLM involved) ----------
    def compare_progression(self, q: Dict, pid: str) -> str:
        m = re.match(r"u(\d+)c(\d+)", pid)
        if not m:
            return f"(compare_progression: id {pid} not understood)"
        pu, pc = int(m.group(1)), int(m.group(2))
        prior_q = [x for x in self.queries
                   if x["unit"] == q["unit"] and x["cycle"] < q["cycle"]]
        rate_q = (len(prior_q) / max(q["cycle"], 1))
        sens_q = set(x for x in _DSENS if x in q["rule"])
        rows = [r for r in (self.hist.by_unit.get(pu, []) if self.hist
                            else []) if r["cycle"] < pc]
        rate_p = len(rows) / max(pc, 1)
        gseq = [r.get("gravity") for r in rows[-6:]]
        prec = next((r for r in (self.hist.by_unit.get(pu, [])
                                 if self.hist else [])
                     if r["cycle"] == pc), None)
        sens_p = set(x for x in _DSENS
                     if prec and x in str(prec.get("text", "")))
        shared = sorted(sens_q & sens_p)
        return ("PROGRESSION COMPARISON (deterministic)\n"
                f"  monitored unit T{q['unit']}: {len(prior_q)} prior "
                f"anomalies in {q['cycle']} cycles "
                f"(rate {rate_q:.3f}/cycle), local_count="
                f"{q['counters']['local_count']}, last3_freq="
                f"{q['counters']['local_last3']:.2f}\n"
                f"  precedent u{pu}@c{pc}: {len(rows)} prior anomalies in "
                f"{pc} cycles (rate {rate_p:.3f}/cycle), recent gravities "
                f"{gseq}\n"
                f"  anomaly-rate ratio (unit/precedent): "
                f"{(rate_q / rate_p) if rate_p else float('inf'):.2f}\n"
                f"  shared rule sensors: {shared or 'none'}")

    # ---- the tool loop (retrospective toolbox) ------------------------
    def d_agent_gather(self, q: Dict, max_steps: int = 3):
        seen: Dict[str, Dict] = {}
        for r in self.retrieve(q):
            seen[f"u{r['unit']}c{r['cycle']}"] = r
        obs = fmt_precedents_diag(list(seen.values()), self.hist)
        extra: List[str] = []
        trace: List[Dict] = []
        base = [case_block(q), self.unit_hist_fn(q)]
        for step in range(max_steps):
            nudge = ("" if step < max_steps - 1
                     else " This is your LAST tool step - prefer finish.")
            pr = ("\n\n".join(x for x in base if x)
                  + f"\n\nOBSERVATIONS:\n{obs}"
                  + ("\n" + "\n".join(extra) if extra else "")
                  + f"\n\nStep {step+1}/{max_steps}.{nudge}\n"
                  + DIAG_TOOLS)
            out = self.be.generate(pr, system=SYSTEM_D)
            m = re.search(r"\{.*\}", out or "", re.S)
            act = None
            if m:
                try:
                    act = json.loads(m.group(0))
                except Exception:  # noqa: BLE001
                    act = None
            trace.append({"step": step, "raw": (out or "")[:260]})
            if isinstance(act, dict) and "severity" in act and \
                    "tool" not in act:
                t, _ = parse_ticket_d(json.dumps(act))
                if t:
                    return t, list(seen.values()), trace, step + 1
            if not isinstance(act, dict):
                continue
            tool = str(act.get("tool", ""))
            if tool == "finish":
                t, _ = parse_ticket_d(json.dumps(act.get("ticket", {})))
                return t, list(seen.values()), trace, step + 1
            if tool == "search_memory":
                qq = dict(q)
                qq["interpretation"] = str(act.get("query")
                                           or q["interpretation"])
                for r in self.retrieve(qq):
                    seen[f"u{r['unit']}c{r['cycle']}"] = r
                obs = fmt_precedents_diag(list(seen.values()), self.hist)
            elif tool == "read_precedent":
                pid = str(act.get("id", ""))
                r = seen.get(pid)
                extra.append(f"FULL TEXT of {pid}:\n"
                             + (str(r["text"])[:700] if r
                                else "(id not in memory view)"))
            elif tool == "read_history":
                pid = str(act.get("id", ""))
                mm = re.match(r"u(\d+)c(\d+)", pid)
                extra.append(f"HISTORY of {pid}:\n"
                             + (self.hist.history_of(int(mm.group(1)),
                                                     int(mm.group(2)),
                                                     max_n=8)
                                if mm else "(id not understood)"))
            elif tool == "compare_progression":
                extra.append(self.compare_progression(
                    q, str(act.get("id", ""))))
        # out of steps: forced grounded finish
        out = self.be.generate(
            "\n\n".join(x for x in base if x)
            + f"\n\nOBSERVATIONS:\n{obs}"
            + ("\n" + "\n".join(extra) if extra else "")
            + "\n\nOut of tool steps. OUTPUT the diagnostic ticket JSON "
              "now, citing only the ids above:\n" + TICKET_D,
            system=SYSTEM_D)
        t, _ = parse_ticket_d(out)
        return t, list(seen.values()), trace, max_steps + 1

    def _prompt(self, q, recs):
        parts = [case_block(q), self.unit_hist_fn(q)]
        if recs is not None:
            parts.append(fmt_precedents_diag(recs, self.hist))
        parts.append("Produce the diagnostic ticket now.")
        return "\n\n".join(x for x in parts if x)

    def d_b0(self, q):
        recs = self.retrieve(q)
        gs = [r.get("gravity") for r in recs
              if isinstance(r.get("gravity"), (int, float))]
        sev = int(round(float(np.median(gs)))) if gs else 3
        sev = min(max(sev, 1), 5)
        t = {"diagnosis": "nearest-precedent consensus (no LLM)",
             "matched_pattern": "median gravity of retrieved precedents",
             "severity": sev, "action": SEV2ACT[sev],
             "cited_precedents": [f"u{r['unit']}c{r['cycle']}" for r in recs],
             "reasoning": ""}
        return {"ticket": t, "contexts": recs, "steps": 0}

    def d_p1(self, q):
        out = self.be.generate(self._prompt(q, None), system=SYSTEM_D)
        t, _ = parse_ticket_d(out)
        return {"ticket": t, "contexts": [], "steps": 1}

    def d_p2(self, q):
        recs = self.retrieve(q)
        out = self.be.generate(self._prompt(q, recs), system=SYSTEM_D)
        t, _ = parse_ticket_d(out)
        return {"ticket": t, "contexts": recs, "steps": 1}

    def d_p5(self, q, has_history):
        t0, recs, trace, gsteps = self.d_agent_gather(q)
        cur = {"ticket": t0, "contexts": recs}
        shown = [f"u{r['unit']}c{r['cycle']}" for r in recs]
        repairs = 0
        viol = gates_d(cur["ticket"], shown, has_history)
        while viol and repairs < 2:
            out = self.be.generate(
                self._prompt(q, recs)
                + "\n\nYour previous ticket:\n"
                + json.dumps(cur["ticket"] or {})
                + "\n\nIt FAILED these checks: " + ", ".join(viol)
                + ". Fix ONLY what is violated: cite only shown ids; "
                  "severity 1-5; action must follow severity (1-2 monitor, "
                  "3 inspect, 4 plan, 5 shutdown); explicitly COMPARE the "
                  "unit's own progression with the cited precedents' "
                  "histories. Produce the corrected ticket.",
                system=SYSTEM_D)
            t, _ = parse_ticket_d(out)
            cur = {"ticket": t, "contexts": recs}
            repairs += 1
            viol = gates_d(t, shown, has_history)
        auto = False
        t_ = cur["ticket"]
        if (viol and set(viol) <= {"D5_severity_action_mismatch"}
                and t_ and t_.get("severity") is not None):
            t_ = dict(t_)
            t_["action"] = SEV2ACT[min(max(int(t_["severity"]), 1), 5)]
            cur = {"ticket": t_, "contexts": recs}
            auto = True
            viol = gates_d(t_, shown, has_history)
        return {"ticket": cur["ticket"], "contexts": recs,
                "repairs": repairs, "auto_corrected": auto,
                "escalated": bool(viol), "trace": trace,
                "steps": gsteps + repairs}

    def run_diag(self, pattern, q, has_history):
        if pattern == "B0_retrieval":
            return self.d_b0(q)
        if pattern == "P1_direct":
            return self.d_p1(q)
        if pattern == "P2_rag":
            return self.d_p2(q)
        return self.d_p5(q, has_history)

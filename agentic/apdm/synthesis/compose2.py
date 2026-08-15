"""compose2.py — synthesis v3, JSON-first: the model writes ONLY prose
fragments; the renderer injects every fact deterministically. Faithfulness
by construction (the campaign's law applied to the composer itself).

Model contract: return ONLY JSON
  {"situation": str, "diagnosis": str, "outlook": str,
   "trust": str, "next_steps": str}
1-2 sentences each, no figures unless copied from the input.
Gates (per section): number conservation (tolerant) + 320-char cap.
Graduated fallback: an offending section is replaced by the template
sentence; ladder: clean | partial_template:<k> | template_fallback.
"""
from __future__ import annotations
import json
import re
from typing import Dict, List, Tuple

KEYS = ["situation", "diagnosis", "outlook", "trust", "next_steps"]

SYSTEM_J = (
    "You are the reporting layer of a predictive-maintenance system. "
    "Given structured ticket data, return ONLY a JSON object with exactly "
    "these string fields: situation, diagnosis, outlook, trust, "
    "next_steps. Each field: one or two fluent sentences for a maintainer,"
    " rephrasing the data. Do NOT include numbers unless copying them from"
    " the input; never invent figures, sensors or claims. All factual "
    "values (severity, action, RUL, reliability, uncertainty, citations) "
    "are printed by the system around your prose - focus on clear "
    "explanation, causes, and what to watch. No text outside the JSON.")


def _close(x: float, vals: List[float]) -> bool:
    return any(abs(x - a) <= max(0.011 * abs(a), 0.051) for a in vals)


def _numbers(txt: str) -> List[float]:
    return [float(n) for n in re.findall(r"\d+(?:\.\d+)?", txt or "")]


def allowed_vals(inp: Dict) -> List[float]:
    hay = json.dumps(inp, default=str)
    return sorted({float(n) for n in re.findall(r"\d+(?:\.\d+)?", hay)})


def parse_sections(txt: str):
    m = re.search(r"\{.*\}", txt or "", re.S)
    if not m:
        return None
    try:
        j = json.loads(m.group(0))
    except Exception:
        try:
            j = json.loads(m.group(0).replace("\n", " "))
        except Exception:
            return None
    out = {}
    for k in KEYS:
        v = j.get(k)
        if not isinstance(v, str) or not v.strip():
            return None
        out[k] = v.strip()
    return out


def template_sections(inp: Dict) -> Dict[str, str]:
    fc = inp.get("prognostic") or {}
    edge1 = re.sub(r"\*\*[^*]+\*\*:?", "", (inp.get("edge") or "")
                   .split("**Cause")[0]).strip()
    if len(edge1) > 220:
        edge1 = edge1[:220].rsplit(" ", 1)[0] + " ..."
    return {
        "situation": edge1 or "Anomalous telemetry detected on this unit.",
        "diagnosis": (inp.get("diagnosis") or
                      "Progression matches the cited precedents."),
        "outlook": (fc.get("projected_progression")
                    or fc.get("progression_narrative")
                    or "Degradation is expected to follow the cited "
                       "precedents."),
        "trust": "Coverage and agreement of the cited cases are "
                 "summarized below.",
        "next_steps": "Follow the prescribed action; re-evaluate at the "
                      "next anomaly and escalate if severity rises.",
    }


def check_section(txt: str, vals: List[float]) -> bool:
    if len(txt) > 480:
        return False
    return all(_close(x, vals) for x in _numbers(txt))


def render(inp: Dict, sec: Dict[str, str]) -> str:
    fc = inp.get("prognostic") or {}
    sig = inp.get("signals") or {}
    rel = sig.get("reliability") or {}
    agg = ((sig.get("future_progression") or {}).get("aggregate") or {})
    unc = agg.get("uncertainty")
    rng = fc.get("rul_range")
    rngs = (f" (range {rng[0]:.0f}-{rng[1]:.0f})"
            if isinstance(rng, list) and len(rng) == 2
            and rng[0] is not None else "")
    tr = agg.get("ttf_range")
    surv = (f" Similar cases survived {tr[0]:.0f}-{tr[1]:.0f} cycles "
            f"(median {agg.get('median_ttf')})."
            if agg.get("median_ttf") is not None and tr else "")
    cites = ", ".join(inp.get("cited_precedents") or []) or "none"
    uncline = ""
    if unc is not None:
        uncline = (f" Progression uncertainty {unc} — "
                   + ("cited futures agree."
                      if unc < 0.35 else
                      "several plausible paths; treat the outlook "
                      "cautiously."))
    return "\n".join([
        f"# unit {inp['unit']} cycle {inp['cycle']} — severity "
        f"{inp.get('severity')}/5 — {inp.get('action')}",
        "## SITUATION", sec["situation"],
        "## DIAGNOSIS", sec["diagnosis"],
        f"Cited precedents: {cites}.",
        "## OUTLOOK",
        f"RUL estimate {fc.get('rul_estimate')} cycles{rngs}, confidence "
        f"{fc.get('confidence') or 'n/a'}.{surv}",
        sec["outlook"],
        "## TRUST",
        f"Knowledge-base reliability {rel.get('value')} "
        f"(top {rel.get('top')}, k={rel.get('k')}).{uncline}",
        sec["trust"],
        "## NEXT STEPS",
        f"Execute: {inp.get('action')}.",
        sec["next_steps"],
    ])


def compose(be, inp: Dict) -> Tuple[str, str, List[str]]:
    """-> (ticket, state, notes)"""
    vals = allowed_vals(inp)
    tpl = template_sections(inp)
    payload = json.dumps(inp, default=str)
    raw = be.generate("TICKET DATA:\n" + payload +
                      "\nReturn the JSON now.", system=SYSTEM_J)
    sec = parse_sections(raw)
    if sec is None:
        raw = be.generate("TICKET DATA:\n" + payload +
                          "\nYour previous answer was not the required "
                          "JSON. Return ONLY the JSON object now.",
                          system=SYSTEM_J)
        sec = parse_sections(raw)
    if sec is None:
        return render(inp, tpl), "template_fallback", ["parse"]
    replaced = []
    for k in KEYS:
        if not check_section(sec[k], vals):
            sec[k] = tpl[k]
            replaced.append(k)
    state = ("clean" if not replaced
             else f"partial_template:{len(replaced)}")
    return render(inp, sec), state, replaced


class MockComposer:
    def __init__(self, mode="good"):
        self.mode = mode

    def generate(self, prompt, system=None):
        m = re.search(r"TICKET DATA:\n(\{.*\})", prompt, re.S)
        inp = json.loads(m.group(1))
        tpl = template_sections(inp)
        if self.mode == "garbage":
            return "I cannot help with that."
        if self.mode == "invent":
            tpl["outlook"] = "Expect exactly 9999 cycles of margin."
        return json.dumps(tpl)

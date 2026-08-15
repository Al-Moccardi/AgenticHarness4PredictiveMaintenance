"""compose.py — the synthesis layer: one SLM call that turns the pipeline's
structured outputs (edge + diagnostic + prognostic + signals) into a single
well-built maintenance ticket.

Design law (from the whole campaign): the composer may REPHRASE and
ORGANIZE, never ALTER. Gates:
  S1 all required sections present
  S2 severity, action, RUL estimate, reliability, uncertainty verbatim
  S3 every cited precedent id present
  S4 number conservation: every number in the output must exist in the
     input (formatting-tolerant); the composer can not invent figures
One repair round; then the deterministic TEMPLATE renders the ticket
(guaranteed output). Ladder: clean | repaired | template_fallback.
The composer never sees ground truth.
"""
from __future__ import annotations
import json
import re
from typing import Dict, List, Optional, Tuple

SECTIONS = {"SITUATION": ["SITUATION"],
            "DIAGNOSIS": ["DIAGNOS"],
            "OUTLOOK": ["OUTLOOK", "PROGNOS", "FORECAST"],
            "TRUST": ["TRUST", "RELIABILIT", "CONFIDENCE"],
            "NEXT STEPS": ["NEXT STEP", "ACTION", "RECOMMEND"]}

SYSTEM_S = (
    "You are the reporting layer of a predictive-maintenance system. "
    "Given the structured ticket data (JSON), write THE FINAL MAINTENANCE "
    "TICKET in markdown with exactly these sections: "
    "## SITUATION, ## DIAGNOSIS, ## OUTLOOK, ## TRUST, ## NEXT STEPS, "
    "preceded by a header line '# unit U cycle C — severity S/5 — ACTION'. "
    "Rules: you may rephrase and organize but NEVER alter facts; copy every "
    "number verbatim from the input (severity, RUL estimate and range, "
    "reliability, uncertainty, precedent survival figures); include every "
    "cited precedent id; do not invent any number, sensor value or claim "
    "not present in the input; 170 words maximum; no preamble outside the "
    "ticket.")


def _fmt_variants(x) -> set:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return set()
    out = {f"{f}", f"{f:.0f}", f"{f:.1f}", f"{f:.2f}", f"{f:.3f}"}
    if f == int(f):
        out.add(str(int(f)))
    return out


def _close(x: float, allowed_vals) -> bool:
    for a in allowed_vals:
        if abs(x - a) <= max(0.011 * abs(a), 0.051):
            return True
    return False


def allowed_numbers(inp: Dict) -> set:
    keys = []
    keys += [inp.get("unit"), inp.get("cycle"), inp.get("severity")]
    fc = inp.get("prognostic") or {}
    keys += [fc.get("rul_estimate"), fc.get("dl_hint"),
             fc.get("progression_horizon")]
    rng = fc.get("rul_range") or []
    keys += list(rng) if isinstance(rng, list) else []
    sig = inp.get("signals") or {}
    rel = sig.get("reliability") or {}
    keys += [rel.get("value"), rel.get("top"), rel.get("k")]
    agg = (sig.get("future_progression") or {}).get("aggregate") or {}
    keys += [agg.get("median_ttf"), agg.get("uncertainty")]
    tr = agg.get("ttf_range") or []
    keys += list(tr) if isinstance(tr, list) else []
    for c in (sig.get("future_progression") or {}).get("cases") or []:
        keys += [c.get("rul_then"), c.get("n_more")]
        keys += [e.get("plus") for e in c.get("events", [])]
        keys += [e.get("gravity") for e in c.get("events", [])]
        m = re.match(r"u(\d+)c(\d+)", c.get("id", ""))
        if m:
            keys += [m.group(1), m.group(2)]
    esc = agg.get("escalating") or ""
    keys += list(esc.split("/")) if "/" in str(esc) else []
    allowed = set()
    for k in keys:
        allowed |= _fmt_variants(k)
    # numbers quoted in the input's own free text are legitimate sources
    for txt in (inp.get("edge"), inp.get("diagnosis"),
                fc.get("progression_narrative"),
                fc.get("projected_progression")):
        for n in re.findall(r"\d+(?:\.\d+)?", str(txt or "")):
            allowed |= _fmt_variants(n)
    allowed |= {"1", "5", "140"}  # scale anchors: severity x/5
    return allowed


def check(text: str, inp: Dict) -> List[str]:
    v = []
    up = text.upper()
    for name, variants in SECTIONS.items():
        if not any(x in up for x in variants):
            v.append(f"S1:{name}")
    t = inp
    fc = t.get("prognostic") or {}
    sig = t.get("signals") or {}
    rel = (sig.get("reliability") or {}).get("value")
    unc = ((sig.get("future_progression") or {}).get("aggregate")
           or {}).get("uncertainty")
    must = {"severity": t.get("severity"),
            "rul_estimate": fc.get("rul_estimate"),
            "reliability": rel, "uncertainty": unc}
    found = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]
    for name, val in must.items():
        if val is None:
            continue
        if not _close(float(val), found):
            v.append(f"S2:{name}")
    def _norm(x):
        return re.sub(r"[\s_\-]+", " ", str(x).lower())
    act = (t.get("action") or "").strip()
    if act and _norm(act) not in _norm(text):
        v.append("S2:action")
    for cid in t.get("cited_precedents") or []:
        if cid not in text:
            v.append(f"S3:{cid}")
    allow_vals = sorted({float(a) for a in allowed_numbers(t)})
    for n in set(re.findall(r"\d+(?:\.\d+)?", text)):
        if not _close(float(n), allow_vals):
            v.append(f"S4:{n}")
    return v


def template(inp: Dict) -> str:
    t = inp
    fc = t.get("prognostic") or {}
    sig = t.get("signals") or {}
    rel = sig.get("reliability") or {}
    fp = sig.get("future_progression") or {}
    agg = fp.get("aggregate") or {}
    unc = agg.get("uncertainty")
    edge1 = (t.get("edge") or "").split("**Cause")[0]
    edge1 = re.sub(r"\*\*[^*]+\*\*:?", "", edge1).strip()
    if len(edge1) > 220:
        edge1 = edge1[:220].rsplit(" ", 1)[0] + " ..."
    cites = ", ".join(t.get("cited_precedents") or []) or "none"
    rng = fc.get("rul_range")
    rngs = (f" (range {rng[0]:.0f}-{rng[1]:.0f})"
            if isinstance(rng, list) and len(rng) == 2 and rng[0] is not None
            else "")
    pp = fc.get("projected_progression")
    lines = [
        f"# unit {t['unit']} cycle {t['cycle']} — severity "
        f"{t.get('severity')}/5 — {t.get('action')}",
        "## SITUATION", edge1,
        "## DIAGNOSIS",
        f"{(t.get('diagnosis') or '').strip()} "
        f"Cited precedents: {cites}.",
        "## OUTLOOK",
        f"RUL estimate {fc.get('rul_estimate')} cycles{rngs}, confidence "
        f"{fc.get('confidence') or 'n/a'}."
        + ((lambda tr: f" Similar cases survived "
            f"{tr[0]:.0f}-{tr[1]:.0f} cycles "
            f"(median {agg.get('median_ttf')}).")(agg["ttf_range"])
           if agg.get("median_ttf") is not None
           and agg.get("ttf_range") else "")
        + (f" Progression commentary: {pp}" if pp else ""),
        "## TRUST",
        f"Knowledge-base reliability {rel.get('value')} "
        f"(top {rel.get('top')}, k={rel.get('k')}). "
        + (f"Progression uncertainty {unc} — "
           + ("cited futures agree." if (unc or 0) < 0.35
              else "several plausible paths; treat the horizon cautiously.")
           if unc is not None else ""),
        "## NEXT STEPS",
        f"Execute: {t.get('action')}. Re-evaluate at the next anomaly; "
        f"escalate if severity rises.",
    ]
    return "\n".join(lines)


def _must_line(inp: Dict) -> str:
    fc = inp.get("prognostic") or {}
    sig = inp.get("signals") or {}
    rel = (sig.get("reliability") or {}).get("value")
    unc = ((sig.get("future_progression") or {}).get("aggregate")
           or {}).get("uncertainty")
    cites = ", ".join(inp.get("cited_precedents") or []) or "none"
    return (f"REQUIRED FACTS - every one MUST appear verbatim in the "
            f"ticket: severity={inp.get('severity')}; "
            f"action={inp.get('action')}; "
            f"RUL estimate={fc.get('rul_estimate')}; "
            f"reliability={rel}; uncertainty={unc}; "
            f"cited precedents: {cites}.")


def compose(be, inp: Dict) -> Tuple[str, str, List[str]]:
    """-> (ticket_text, state, violations_of_first_attempt)"""
    payload = json.dumps(inp, default=str)
    base = (_must_line(inp) + "\nTICKET DATA:\n" + payload +
            "\nWrite the final ticket now.")
    txt = be.generate(base, system=SYSTEM_S)
    v1 = check(txt, inp)
    if not v1:
        return txt, "clean", []
    missing = "; ".join(v1[:6])
    txt2 = be.generate(base + f"\nYour previous ticket was rejected "
                       f"({missing}). Rewrite the full ticket and make "
                       "sure every REQUIRED FACT above appears verbatim.",
                       system=SYSTEM_S)
    if not check(txt2, inp):
        return txt2, "repaired", v1
    return template(inp), "template_fallback", v1


class MockComposer:
    """Offline backend: returns the deterministic template (gates pass)."""
    def __init__(self, break_first: bool = False):
        self.break_first = break_first
        self._n = 0

    def generate(self, prompt, system=None):
        m = re.search(r"TICKET DATA:\n(\{.*\})", prompt, re.S)
        inp = json.loads(m.group(1))
        self._n += 1
        if self.break_first and self._n == 1:
            return template(inp) + "\nBonus fact: expect 9999 cycles."
        return template(inp)

"""RAGAS-style metrics, computed fully locally (no OpenAI, no ragas dep).

Same three metrics RAD reported, same definitions, local judge:

  faithfulness       judge decomposes the ticket's claims, then verdicts
                     each claim against the retrieved contexts only.
                     score = supported / total  (RAGAS Es et al. 2023, Sec 3)
  answer_relevancy   judge writes 2 questions the ticket answers; score =
                     mean cosine (nomic embeddings) between them and the
                     actual case. (RAGAS reverse-question definition)
  context_precision  judge verdicts each retrieved chunk: was it useful for
                     this ticket? score = useful / retrieved.

Plus deterministic grounding (no judge, no randomness):
  json_valid, citation_validity (cited ⊆ shown), citation_coverage,
  rul_in_cited_span (estimate inside [min,max] of cited precedents'
  rul_then, ±20% slack), action_band_consistency.

Honesty note for the paper: these are RAGAS-definition metrics with a LOCAL
judge (llama3.2 by default), not OpenAI-judged RAGAS. Say so; the RAD paper's
absolute numbers are not directly comparable, the ranking claims are.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Dict, List, Optional

import numpy as np

from ..llm import Backend
from ..patterns import ACTIONS, band_action

J_CLAIMS = ('Decompose the TICKET into at most 6 short factual claims '
            '(diagnosis content, cited matches, RUL statement, action). '
            'OUTPUT PROTOCOL: {"claims": ["...", ...]}')
J_VERDICT = ('For EACH claim decide if it is supported by the CONTEXTS '
             '(the precedents shown) and the CASE. OUTPUT PROTOCOL: '
             '{"verdicts": [{"claim": "...", "supported": true|false}, ...]}')
J_QUESTIONS = ('Write 2 short questions that this TICKET is answering. '
               'OUTPUT PROTOCOL: {"questions": ["...", "..."]}')
J_CTX = ('For EACH precedent decide if it was useful evidence for this '
         'TICKET. OUTPUT PROTOCOL: '
         '{"verdicts": [{"id": "...", "useful": true|false}, ...]}')


def _json(be: Backend, prompt: str) -> Optional[Dict]:
    out = be.generate(prompt, system="OUTPUT PROTOCOL enforced. "
                                     "Answer with ONE JSON object only.")
    m = re.search(r"\{.*\}", out or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return None


class DryJudge(Backend):
    """Deterministic pseudo-judge for offline plumbing tests ONLY."""
    name = "dryjudge"
    model = "dryjudge"

    def __init__(self):
        pass

    def generate(self, prompt, system=None):
        h = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)
        if "claims" in prompt[-200:] or "Decompose" in prompt:
            return json.dumps({"claims": ["c1", "c2", "c3"]})
        if "questions" in prompt[-200:] or "2 short questions" in prompt:
            return json.dumps({"questions": ["what failed", "what to do"]})
        if '"id"' in prompt[-300:] or "useful" in prompt[-300:]:
            ids = re.findall(r"\[(u\d+c\d+)\]", prompt)
            return json.dumps({"verdicts": [
                {"id": i, "useful": (h + k) % 4 != 0}
                for k, i in enumerate(ids)]})
        return json.dumps({"verdicts": [
            {"claim": f"c{k+1}", "supported": (h + k) % 3 != 0}
            for k in range(3)]})


def evaluate_episode(ep: Dict, judge: Backend, embedder) -> Dict:
    t = ep.get("ticket") or {}
    ctxs = ep.get("contexts") or []
    shown = [f"u{c['unit']}c{c['cycle']}" for c in ctxs]
    case = ep["case_text"]
    ticket_txt = json.dumps(t) if t else "(no ticket)"
    ctx_txt = "\n---\n".join(
        f"[{i}] rul_then={c['rul_then']}: {c['text'][:400]}"
        for i, c in zip(shown, ctxs)) or "(none)"

    out: Dict[str, Optional[float]] = {}

    # ---------------- deterministic grounding ----------------------------
    out["json_valid"] = float(bool(t))
    cited = t.get("cited_precedents") or []
    out["citation_coverage"] = float(bool(cited))
    out["citation_validity"] = (float(all(c in shown for c in cited))
                                if cited else (1.0 if not shown else 0.0))
    est = t.get("rul_estimate")
    cr = [c["rul_then"] for c in ctxs
          if f"u{c['unit']}c{c['cycle']}" in cited
          and c["rul_then"] is not None]
    if est is not None and cr:
        lo, hi = min(cr), max(cr)
        pad = 0.2 * max(hi - lo, 10)
        out["rul_in_cited_span"] = float(lo - pad <= est <= hi + pad)
    else:
        out["rul_in_cited_span"] = np.nan
    out["action_band_consistency"] = (
        float(t.get("action") == band_action(est))
        if est is not None and t.get("action") in ACTIONS else 0.0)

    # ---------------- RAGAS-style (judged) --------------------------------
    j = _json(judge, f"TICKET:\n{ticket_txt}\n\n{J_CLAIMS}")
    claims = [str(c) for c in (j or {}).get("claims", [])][:6]
    if claims:
        v = _json(judge, f"CASE:\n{case[:900]}\n\nCONTEXTS:\n{ctx_txt}\n\n"
                         f"CLAIMS:\n" + json.dumps(claims) + f"\n\n{J_VERDICT}")
        vs = (v or {}).get("verdicts", [])
        sup = [bool(x.get("supported")) for x in vs if isinstance(x, dict)]
        out["faithfulness"] = (sum(sup) / len(sup)) if sup else np.nan
        out["n_claims"] = float(len(sup)) if sup else np.nan
    else:
        out["faithfulness"] = np.nan
        out["n_claims"] = np.nan

    jq = _json(judge, f"TICKET:\n{ticket_txt}\n\n{J_QUESTIONS}")
    qs = [str(x) for x in (jq or {}).get("questions", [])][:2]
    if qs:
        vq = embedder.embed(qs)
        vc = embedder.embed([case])[0]
        vc = vc / (np.linalg.norm(vc) or 1)
        sims = [float(v_ @ vc / (np.linalg.norm(v_) or 1)) for v_ in vq]
        out["answer_relevancy"] = float(np.mean(sims))
    else:
        out["answer_relevancy"] = np.nan

    if ctxs:
        jc = _json(judge, f"TICKET:\n{ticket_txt}\n\nPRECEDENTS:\n{ctx_txt}"
                          f"\n\n{J_CTX}")
        vs = (jc or {}).get("verdicts", [])
        u = [bool(x.get("useful")) for x in vs if isinstance(x, dict)]
        out["context_precision"] = (sum(u) / len(u)) if u else np.nan
    else:
        out["context_precision"] = np.nan
    return out

"""Diagnostic / root-cause agents over the interpretation layer.

One elicitation per snapshot; STRICT output protocol (parse failures are a
reported metric):
  PHENOTYPE: <name or id from the fault library>
  SENSORS: <sensor> <high|low>, <sensor> <high|low>, <sensor> <high|low>
  EXPLANATION: <free text, grounded in the observations>

Arms
  D1_featurized  single shot: sensor summary + fault library in the prompt
  D2_agentic     ReAct loop over sensor_summary, fault_library,
                 similar_anomalies (the interpretation KB), degradation_status
  D2_norag       D2 minus similar_anomalies -- the ablation that directly
                 tests the TIOT premise: does retrieving past
                 rule-texts/interpretations measurably improve diagnosis?

Scoring, all on the internal test split (the only place future-derived gold
exists):
  * phenotype accuracy vs gold, per RUL bucket, PAIRED (McNemar) against the
    two non-LLM twins from run_faults: nearest-centroid rule and logistic;
  * signature quality: Jaccard and precision@3 of the SENSORS claims against
    the unit's future-derived terminal signature;
  * FAITHFULNESS of the structured claims against current-window evidence:
    a claimed (sensor, direction) is supported if |z|>=1 and the sign
    matches, contradicted if |z|>=1 and the sign opposes, else unverifiable.
    This is the RAD-verifier idea reduced to its mechanical core.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .agent import _first_json
from ..data import FD002, RMAX, SENSORS
from .faults import build_fault_layer, current_z, gold_phenotype, gold_signature
from .features import summary_text
from .interpret_kb import InterpretationKB
from ..llm import Backend, get_backend
from .metrics import mcnemar, wilson

ROOT = Path(__file__).resolve().parent.parent
BUCKETS = [(0, 30, "critical"), (31, 100, "mid"), (101, RMAX, "early")]

TASK = """You are a turbofan diagnostic engineer performing root-cause
analysis. Identify which fault phenotype this unit is developing and which
sensor deviations characterise it. Reply with EXACTLY these three lines:
PHENOTYPE: <one phenotype name from the library>
SENSORS: <sensor> <high|low>, <sensor> <high|low>, <sensor> <high|low>
EXPLANATION: <2-4 sentences grounded ONLY in the evidence shown; never invent
sensor values or thresholds>"""

AGENT_SYSTEM = """You are a turbofan diagnostic engineer with tool access,
performing root-cause analysis of a developing fault.

OUTPUT PROTOCOL - reply with ONE JSON object and NOTHING else.
Tool call: {"thought": "...", "action": {"tool": "<name>", "args": {...}}}
Answer:    {"thought": "...", "final": {"phenotype": "<one phenotype name from the library>", "sensors": [["<sensor>", "high|low"], ["<sensor>", "high|low"], ["<sensor>", "high|low"]], "explanation": "<2-4 sentences grounded ONLY in the evidence shown>"}}

WORKED EXAMPLE of an answer:
  {"thought": "core-speed deviations dominate", "final": {"phenotype": "Nc-NRc", "sensors": [["Nc", "high"], ["NRc", "high"], ["Nf", "high"]], "explanation": "Core and corrected core speed are both far above the regime reference and rising."}}

Rules: gather only what you need, never repeat a call, ground every claim in
observations, and give exactly three sensors in the final.

TOOLS
__TOOLS__"""

# The forced-finalisation prompt reuses the EXACT D1 template: the vaguer
# earlier wording ("output the three labelled lines") parsed 4/90 with
# llama3 while the explicit template parses 90/90 in D1.
FORCE = (TASK + "\n\nYou are out of tool steps. Answer NOW from the "
         "observations below, in plain text.\n\nOBSERVATIONS:\n{obs}")


# ------------------------------------------------------------------ parsing
def _match_phenotype(raw: str, phen_names: List[str]) -> Optional[int]:
    for i, name in enumerate(phen_names):
        if name.lower() in raw.lower() or re.search(rf"\b{i}\b", raw):
            return i
    # tolerate a phenotype named by one of its sensors (e.g. "Nc" -> Nc-NRc)
    for i, name in enumerate(phen_names):
        if any(tok and tok.lower() in raw.lower() for tok in name.split("-")):
            return i
    return None


def _norm_dir(d: str) -> Optional[str]:
    d = (d or "").strip().lower()
    if d in ("high", "elevated", "increased", "rising", "above"):
        return "high"
    if d in ("low", "reduced", "decreased", "falling", "below"):
        return "low"
    return None


def parse_diag(text, phen_names: List[str]) -> Optional[Dict]:
    """Accepts (a) a structured dict from the JSON protocol, (b) the three
    labelled plain-text lines, or (c) either embedded in a JSON blob."""
    if isinstance(text, dict):
        pid = _match_phenotype(str(text.get("phenotype", "")), phen_names)
        claims = []
        for item in (text.get("sensors") or []):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                sen, d = str(item[0]), _norm_dir(str(item[1]))
            elif isinstance(item, dict):
                sen = str(item.get("sensor", ""))
                d = _norm_dir(str(item.get("direction", "")))
            elif isinstance(item, str):
                m = re.match(r"\s*(\w+)[\s:,]+(\w+)", item)
                sen, d = (m.group(1), _norm_dir(m.group(2))) if m else ("", None)
            else:
                continue
            sen = next((x for x in SENSORS if x.lower() == sen.lower()), None)
            if sen and d:
                claims.append((sen, d))
        if pid is None or not claims:
            return None
        return {"phenotype": pid, "claims": claims[:3],
                "explanation": str(text.get("explanation", ""))[:800]}
    if not text:
        return None
    mp = re.search(r"PHENOTYPE:\s*([^\n]+)", text, re.IGNORECASE)
    ms = re.search(r"SENSORS:\s*([^\n]+)", text, re.IGNORECASE)
    if not mp or not ms:
        return None
    raw_p = mp.group(1).strip()
    pid = None
    for i, name in enumerate(phen_names):
        if name.lower() in raw_p.lower() or re.search(rf"\b{i}\b", raw_p):
            pid = i
            break
    claims: List[Tuple[str, str]] = []
    for part in re.split(r"[,;]", ms.group(1)):
        m = re.search(r"([A-Za-z]\w{1,7})\s+(high|low|elevated|reduced|"
                      r"increased|decreased)", part.strip(), re.IGNORECASE)
        if not m:
            continue
        sen = next((s for s in SENSORS if s.lower() == m.group(1).lower()), None)
        if sen:
            d = m.group(2).lower()
            claims.append((sen, "high" if d in
                           ("high", "elevated", "increased") else "low"))
    if pid is None or not claims:
        return None
    me = re.search(r"EXPLANATION:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    return {"phenotype": pid, "claims": claims[:3],
            "explanation": (me.group(1).strip()[:800] if me else "")}


def faithfulness(claims: List[Tuple[str, str]], zvec: np.ndarray,
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


# ------------------------------------------------------------------- arms
class DiagToolBox:
    def __init__(self, ds: FD002, layer, kb: Optional[InterpretationKB],
                 unit: int, cycle: int):
        self.ds, self.layer, self.kb = ds, layer, kb
        self.unit, self.cycle = unit, cycle
        self._z = None

    def z(self) -> np.ndarray:
        if self._z is None:
            self._z = current_z(self.ds, self.unit, self.cycle)
        return self._z

    def names(self) -> List[str]:
        base = ["sensor_summary", "fault_library", "degradation_status"]
        if self.kb is not None:
            base.insert(2, "similar_anomalies")
        return base

    def specs(self) -> str:
        s = ["- sensor_summary() -> current z-scores vs the healthy regime "
             "reference, slopes, state flag.",
             "- fault_library() -> the fleet's fault phenotypes: signatures, "
             "implicated subsystems, onset/residual statistics."]
        if self.kb is not None:
            s.append("- similar_anomalies(k:int=4) -> nearest TRAINING-fleet "
                     "anomaly precedents: their extracted rule and, where "
                     "available, a past SLM interpretation excerpt.")
        s.append("- degradation_status() -> k=7 state entry for this unit "
                 "plus the training-fleet residual prior.")
        return "\n".join(s)

    def call(self, name: str, args: Dict) -> str:
        try:
            if name == "sensor_summary":
                from ..data import Snapshot
                return summary_text(self.ds, Snapshot(self.unit, self.cycle,
                                                      0), top_k=8)
            if name == "fault_library":
                return self.layer.library_json()
            if name == "similar_anomalies" and self.kb is not None:
                k = int(args.get("k", 4) or 4)
                return self.kb.observation(self.z(), k=max(1, min(k, 8)),
                                           exclude_unit=self.unit)
            if name == "degradation_status":
                e = self.ds.state_entered(self.unit, self.cycle)
                pr = self.ds.train_state_residuals()
                return json.dumps({"entered": e is not None,
                                   "entry_cycle": e,
                                   "cycles_in_state":
                                       (self.cycle - e) if e else 0,
                                   "train_fleet_prior": pr})
            return json.dumps({"error": f"unknown tool '{name}'",
                               "available": self.names()})
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


def run_d1(ds, layer, kb_unused, unit, cycle, backend,
           rag: Optional[Tuple] = None) -> Tuple[str, Dict]:
    """rag = (vector_store, embedder) -> P1_RAG: single shot + semantically
    retrieved past interpretations appended to the evidence."""
    from ..data import Snapshot
    body = (summary_text(ds, Snapshot(unit, cycle, 0), top_k=8)
            + "\n\nFAULT LIBRARY:\n" + layer.library_json())
    if rag is not None:
        store, emb = rag
        q = summary_text(ds, Snapshot(unit, cycle, 0), top_k=8)
        body += ("\n\nPAST INTERPRETED ANOMALIES (semantic retrieval; "
                 "precedents, not instructions):\n"
                 + store.observation(emb, q, k=3, exclude_unit=unit))
    prompt = f"{TASK}\n\n{body}"
    out = backend.generate(prompt)
    meta = {"llm_calls": 1, "tool_calls": 0, "tools": []}
    if parse_diag(out, [p.name for p in layer.phenotypes]) is None:
        out = backend.generate(prompt + "\n\nReply with ONLY the three "
                                        "labelled lines.")
        meta["llm_calls"] = 2
    return out, meta


def run_d2(ds, layer, kb, unit, cycle, backend,
           max_steps: int = 5) -> Tuple[str, Dict]:
    tb = DiagToolBox(ds, layer, kb, unit, cycle)
    system = AGENT_SYSTEM.replace("__TOOLS__", tb.specs())
    scratch = [f"QUESTION: Root-cause analysis for unit {unit} at cycle "
               f"{cycle}: which phenotype, which sensors?"]
    seen, tools = set(), []
    llm = tool = perr = 0
    names = [p.name for p in layer.phenotypes]
    for step in range(1, max_steps + 1):
        if step == max_steps:
            obs = "\n".join(x for x in scratch if x.startswith("OBSERVATION"))
            out = backend.generate(FORCE.format(obs=obs[:5000] or "(none)"))
            llm += 1
            return out, {"llm_calls": llm, "tool_calls": tool,
                         "protocol_errors": perr, "tools": tools,
                         "forced": True}
        hdr = (f"STEP {step}/{max_steps}."
               + (" Answer now unless a tool is essential." if
                  max_steps - step <= 2 else ""))
        raw = backend.generate("\n\n".join(scratch + [hdr]), system=system)
        llm += 1
        msg = _first_json(raw)
        if msg is None:
            perr += 1
            scratch.append("OBSERVATION: invalid protocol JSON.")
            if perr >= 3:
                obs = "\n".join(x for x in scratch
                                if x.startswith("OBSERVATION"))
                out = backend.generate(FORCE.format(obs=obs[:5000] or "(none)"))
                llm += 1
                return out, {"llm_calls": llm, "tool_calls": tool,
                             "protocol_errors": perr, "tools": tools,
                             "forced": True}
            continue
        if "final" in msg:
            fin = msg.get("final", "")
            if parse_diag(fin, names) is None:
                scratch.append("OBSERVATION: your final lacked the three "
                               "labelled lines; emit final again with "
                               "PHENOTYPE / SENSORS / EXPLANATION.")
                continue
            return fin, {"llm_calls": llm, "tool_calls": tool,
                         "protocol_errors": perr, "tools": tools,
                         "forced": False}
        act = msg.get("action") or {}
        name = str(act.get("tool", "")) if isinstance(act, dict) else ""
        args = act.get("args") if isinstance(act, dict) else {}
        if not isinstance(args, dict):
            args = {}
        if name not in tb.names():
            perr += 1
            scratch.append(f"OBSERVATION: '{name}' is not a tool. Available: "
                           f"{', '.join(tb.names())}.")
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
                    f"OBSERVATION: {obs[:1800]}"]
    return "", {"llm_calls": llm, "tool_calls": tool, "protocol_errors": perr,
                "tools": tools, "forced": True}


ARMS = ["D1_featurized", "D1_rag", "D2_agentic", "D2_norag"]


# ------------------------------------------------------------------ runner
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="dryrun",
                    choices=["dryrun", "ollama", "openai"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--per-bucket", type=int, default=30)
    ap.add_argument("--sample-seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=5)
    ap.add_argument("--device", default="orin_nano_8gb")
    ap.add_argument("--hw-mode", default="model", choices=["model", "off"])
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    model = a.model or ("gpt-4o-mini" if a.backend == "openai" else "llama3")
    out = ROOT / a.out
    out.mkdir(exist_ok=True)

    ds = FD002(seed=42)
    layer = build_fault_layer(ds, cache=ROOT / "cache" / "faults.pkl")
    kb = InterpretationKB(ds, cache=ROOT / "cache" / "kb.pkl")
    rag = None
    if "D1_rag" in a.arms:
        from ..vector_store import STORE_DIR, VectorStore, get_embedder
        if (STORE_DIR / "embeddings.npz").exists():
            store = VectorStore.load()
            emb_kind = ("ollama" if store.embedder_name.startswith("ollama")
                        else "hash")
            rag = (store, get_embedder(emb_kind))
            print(f"[diag] semantic store: {len(store.meta)} records, "
                  f"embedder={store.embedder_name}"
                  + (" [HASH FALLBACK - pipeline test only]"
                     if emb_kind == "hash" else ""))
        else:
            raise SystemExit("[diag] D1_rag requested but no vector store; "
                             "run: python -m apdm.vector_store --build "
                             "--embedder ollama")
    print(f"[diag] KB: {len(kb.records)} train anomaly precedents, "
          f"{kb.n_interpreted} with SLM interpretations; "
          f"{len(kb.held_out_interpreted)} interpreted records held out "
          f"(test units)")

    snaps = ds.sample_snapshots(ds.test_units, a.per_bucket, seed=a.sample_seed)
    if a.limit:
        snaps = snaps[: a.limit]
    gold_ph = {u: gold_phenotype(ds, layer, u) for u in ds.test_units}
    gold_sig = {u: set(gold_signature(ds, layer, u)) for u in ds.test_units}

    # twins, computed once on the same snapshots (pairing)
    from sklearn.linear_model import LogisticRegression
    from .features import feature_matrix
    tr = ds.snapshots(ds.train_units)
    Xtr = feature_matrix(ds, tr)
    ytr = np.array([layer.train_unit_phenotype[s.unit] for s in tr])
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1
    lr = LogisticRegression(max_iter=2000).fit((Xtr - mu) / sd, ytr)
    Xte = feature_matrix(ds, snaps)
    twin_lr = lr.predict((Xte - mu) / sd)
    twin_rule = np.array([layer.assign_z(current_z(ds, s.unit, s.cycle))[0]
                          for s in snaps])

    tag = f"{model.replace(':', '_')}_seed{a.sample_seed}"
    backend = get_backend(a.backend, model,
                          log_path=out / f"diag_calls_{tag}.jsonl",
                          device=a.device if a.hw_mode == "model" else None,
                          hw_mode=a.hw_mode)
    if backend.cost is not None:
        (out / f"hardware_provenance_{tag}.json").write_text(
            json.dumps(backend.cost.provenance(), indent=2))
    names = [p.name for p in layer.phenotypes]

    rows = []
    for arm in a.arms:
        print(f"[diag] === {arm} | {model} | {len(snaps)} snapshots ===")
        for i, s in enumerate(snaps):
            t0 = time.time()
            tb0 = backend.totals()
            if hasattr(backend, "reset_prefix"):
                backend.reset_prefix()
            if arm == "D1_featurized":
                txt, meta = run_d1(ds, layer, None, s.unit, s.cycle, backend)
            elif arm == "D1_rag":
                txt, meta = run_d1(ds, layer, None, s.unit, s.cycle, backend,
                                   rag=rag)
            else:
                txt, meta = run_d2(ds, layer,
                                   kb if arm == "D2_agentic" else None,
                                   s.unit, s.cycle, backend, a.max_steps)
            parsed = parse_diag(txt, names)
            txt_s = (json.dumps(txt, default=str)
                     if isinstance(txt, dict) else str(txt))
            zc = current_z(ds, s.unit, s.cycle)
            row = {"arm": arm, "model": model, "unit": s.unit,
                   "cycle": s.cycle, "rul": s.rul,
                   "gold_phenotype": gold_ph[s.unit],
                   "twin_rule": int(twin_rule[i]),
                   "twin_lr": int(twin_lr[i]),
                   "parse_failed": parsed is None,
                   "seconds": round(time.time() - t0, 2), **meta,
                   "tools": "|".join(meta.get("tools", []))}
            tb1 = backend.totals()
            row.update({k: round(tb1[k] - tb0.get(k, 0), 4)
                        for k in ("prompt_tokens", "completion_tokens",
                                  "sim_edge_s", "sim_energy_j")
                        if k in tb1})
            if parsed:
                claims = set(parsed["claims"])
                gs = gold_sig[s.unit]
                row.update({"pred_phenotype": parsed["phenotype"],
                            "correct": parsed["phenotype"] == gold_ph[s.unit],
                            "sig_jaccard": len(claims & gs) /
                            len(claims | gs) if claims | gs else 0.0,
                            "sig_p_at_3": len(claims & gs) /
                            max(len(claims), 1),
                            **faithfulness(parsed["claims"], zc),
                            "answer": txt_s[:700]})
            rows.append(row)
        sub = pd.DataFrame([r for r in rows if r["arm"] == arm])
        ok = sub[~sub.parse_failed]
        print(f"[diag] {arm}: parse_ok {len(ok)}/{len(sub)}"
              + (f", phenotype acc {ok.correct.mean():.3f}, "
                 f"sig P@3 {ok.sig_p_at_3.mean():.3f}, "
                 f"faithfulness sup {ok.supported.mean():.3f}"
                 if len(ok) else ""))

    df = pd.DataFrame(rows)
    df.to_csv(out / f"diag_predictions_{tag}.csv", index=False)

    print("\n=== phenotype accuracy by bucket (twins on identical snapshots) ===")
    print(f"{'bucket':<10}{'n':>5}" + "".join(f"{a_:>16}" for a_ in a.arms)
          + f"{'rule':>10}{'logistic':>10}")
    base = df[df.arm == a.arms[0]]
    for lo, hi, name in BUCKETS + [(0, RMAX, "overall")]:
        mask = (base.rul >= lo) & (base.rul <= hi)
        if not mask.any():
            continue
        line = f"{name:<10}{int(mask.sum()):>5}"
        for arm in a.arms:
            g = df[(df.arm == arm)].loc[mask.values]
            okg = g[~g.parse_failed]
            line += f"{(okg.correct.mean() if len(okg) else float('nan')):>16.3f}"
        line += f"{(base.loc[mask, 'twin_rule'] == base.loc[mask, 'gold_phenotype']).mean():>10.3f}"
        line += f"{(base.loc[mask, 'twin_lr'] == base.loc[mask, 'gold_phenotype']).mean():>10.3f}"
        print(line)

    print("\n=== McNemar: agent vs logistic twin (paired) ===")
    for arm in a.arms:
        g = df[df.arm == arm]
        okg = g[~g.parse_failed]
        if not len(okg):
            continue
        mc = mcnemar(list(okg.correct),
                     list(okg.twin_lr == okg.gold_phenotype))
        pf_lo, pf_hi = wilson(int(g.parse_failed.sum()), len(g))
        print(f"  {arm:<14} n={len(okg):<4} agent_wins={mc['n10']:<3} "
              f"twin_wins={mc['n01']:<3} p={mc['p_value']:.4f}  "
              f"parse_fail={g.parse_failed.mean():.2f} "
              f"[{pf_lo:.2f},{pf_hi:.2f}]")
    print(f"\n[diag] wrote diag_predictions_{tag}.csv")


if __name__ == "__main__":
    main()

"""The four LLM arms. One elicitation per snapshot: an integer RUL in
[0, 125], parsed from a mandatory 'ANSWER: <int>' line (one strict retry,
then the arm records a parse failure -- reported, never silently imputed).

  P0_raw        raw 20-cycle window of all 14 sensors in the prompt
  P1_featurized the engineered summary (same quantities as the ML features)
  P2_agentic    ReAct loop over the leakage-safe tools (no ML access)
  P3_hybrid     P2 + the ml_predict tool: the agent as supervisor/adjuster
                of the ML estimate -- the dynamic version of paper 1's
                diagnostic-based RUL adjustment (their eq. 3), with the
                hand-tuned delta replaced by contextual judgement.

Loop design carries the termination lessons forward (rewritten lean):
step budget in every header, forced plain-text finalisation on the last
step, repeat-call suppression, protocol errors distinct from tool calls,
raw output logged every step. `termination` is recorded per snapshot.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..data import FD002, RMAX, SENSORS, Snapshot
from .features import summary_text
from ..llm import Backend
from .ml_models import MLBundle
from .tools import ToolBox

TASK = (f"You are a turbofan prognostics engineer. Estimate the REMAINING "
        f"USEFUL LIFE (RUL) of the engine unit below, in flight cycles, as "
        f"an integer between 0 and {RMAX} (values above {RMAX} are reported "
        f"as {RMAX}). End your reply with a final line exactly of the form "
        f"'ANSWER: <integer>'.")

AGENT_SYSTEM = """You are a turbofan prognostics engineer with tool access.

OUTPUT PROTOCOL - reply with ONE JSON object and NOTHING else.
To call a tool:  {"thought": "...", "action": {"tool": "<name>", "args": {...}}}
To answer:       {"thought": "...", "final": "<short reasoning> ANSWER: <integer 0-125>"}

Rules:
- Estimate the unit's remaining useful life in cycles (0-125).
- Gather only the evidence you need, then answer; never repeat a call.
- Your "final" MUST end with 'ANSWER: <integer>'.

TOOLS
__TOOLS__"""

FORCE = ("You are out of tool steps. Using ONLY the observations below, "
         "state the RUL estimate now in plain text, ending with "
         "'ANSWER: <integer 0-125>'.\n\nOBSERVATIONS:\n{obs}")


@dataclass
class ArmResult:
    unit: int
    cycle: int
    true_rul: int
    pred_rul: Optional[int]
    parse_failed: bool
    n_llm_calls: int
    n_tool_calls: int
    n_protocol_errors: int
    termination: str
    seconds: float
    tools_used: List[str] = field(default_factory=list)
    answer_text: str = ""
    trace: List[Dict] = field(default_factory=list)


# ------------------------------------------------------------------ parsing
def parse_rul(text: str) -> Optional[int]:
    m = re.search(r"ANSWER:\s*(-?\d+(?:\.\d+)?)", text or "", re.IGNORECASE)
    if not m:
        return None
    return int(max(0, min(RMAX, round(float(m.group(1))))))


def _first_json(raw: str) -> Optional[Dict]:
    depth, start, instr, esc = 0, None, False, False
    for i, ch in enumerate(raw or ""):
        if instr:
            esc = (ch == "\\") and not esc
            if ch == '"' and not esc:
                instr = False
            continue
        if ch == '"':
            instr = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                try:
                    o = json.loads(raw[start:i + 1])
                    return o if isinstance(o, dict) else None
                except json.JSONDecodeError:
                    start = None
    return None


# ------------------------------------------------------------- direct arms
def _raw_window_text(ds: FD002, s: Snapshot) -> str:
    h = ds.history(s.unit, s.cycle)
    lines = [f"Unit {s.unit}, current cycle {s.cycle}, operating regime "
             f"{int(h.iloc[-1]['h_clust'])} (regimes 0-5 are different "
             f"flight conditions; compare like with like).",
             "Last 20 cycles, one line per sensor (oldest -> newest):"]
    for sen in SENSORS:
        vals = " ".join(f"{v:.2f}" for v in h[sen])
        lines.append(f"{sen}: {vals}")
    return "\n".join(lines)


def run_direct(ds: FD002, s: Snapshot, backend: Backend,
               featurized: bool) -> ArmResult:
    t0 = time.time()
    body = summary_text(ds, s) if featurized else _raw_window_text(ds, s)
    prompt = f"{TASK}\n\n{body}"
    out = backend.generate(prompt)
    pred = parse_rul(out)
    calls = 1
    if pred is None:                              # one strict retry
        out2 = backend.generate(prompt + "\n\nReply with ONLY the line "
                                         "'ANSWER: <integer 0-125>'.")
        calls += 1
        pred = parse_rul(out2)
        out = out2 if pred is not None else out
    return ArmResult(s.unit, s.cycle, s.rul, pred, pred is None, calls, 0, 0,
                     "direct" if pred is not None else "parse_failed",
                     time.time() - t0, [], (out or "")[:600])


# ------------------------------------------------------------- agentic arms
def run_agentic(ds: FD002, s: Snapshot, backend: Backend, bundle: MLBundle,
                allow_ml: bool, max_steps: int = 6,
                xgb_full_mae: float = 12.4) -> ArmResult:
    t0 = time.time()
    tb = ToolBox(ds, bundle, s, allow_ml=allow_ml, xgb_full_mae=xgb_full_mae)
    system = AGENT_SYSTEM.replace("__TOOLS__", tb.specs())
    scratch = [f"QUESTION: Estimate the remaining useful life of unit "
               f"{s.unit} at cycle {s.cycle}."]
    seen, trace, tools_used = set(), [], []
    llm_calls = tool_calls = proto_err = 0
    pred, answer, term = None, "", "step_budget"

    for step in range(1, max_steps + 1):
        left = max_steps - step
        if left == 0:                                  # forced finalisation
            obs = "\n".join(x for x in scratch if x.startswith("OBSERVATION"))
            out = backend.generate(FORCE.format(obs=obs[:6000] or "(none)"))
            llm_calls += 1
            trace.append({"step": step, "kind": "forced_final",
                          "raw": (out or "")[:800]})
            pred = parse_rul(out)
            answer, term = (out or "")[:600], \
                ("final_forced" if pred is not None else "parse_failed")
            break
        header = (f"STEP {step}/{max_steps}. "
                  + ("Prefer answering now unless a tool is essential."
                     if left <= 2 else ""))
        raw = backend.generate("\n\n".join(scratch + [header]), system=system)
        llm_calls += 1
        msg = _first_json(raw)
        if msg is None:
            proto_err += 1
            trace.append({"step": step, "kind": "protocol_error",
                          "raw": (raw or "")[:800]})
            scratch.append("OBSERVATION: not valid protocol JSON; reply with "
                           'one object containing "action" or "final".')
            if proto_err >= 3:
                obs = "\n".join(x for x in scratch
                                if x.startswith("OBSERVATION"))
                out = backend.generate(FORCE.format(obs=obs[:6000] or "(none)"))
                llm_calls += 1
                pred = parse_rul(out)
                answer = (out or "")[:600]
                term = "final_forced" if pred is not None else "parse_failed"
                trace.append({"step": step, "kind": "forced_final",
                              "raw": (out or "")[:800]})
                break
            continue
        if "final" in msg:
            answer = str(msg.get("final", ""))[:600]
            pred = parse_rul(answer)
            trace.append({"step": step, "kind": "final", "raw": raw[:800]})
            if pred is None:
                scratch.append("OBSERVATION: your final did not contain "
                               "'ANSWER: <integer 0-125>'. Emit final again "
                               "with that exact last line.")
                continue
            term = "final_model"
            break
        act = msg.get("action") or {}
        name = str(act.get("tool", "")) if isinstance(act, dict) else ""
        args = act.get("args") if isinstance(act, dict) else {}
        if not isinstance(args, dict):
            args = {}
        if name not in tb.names():
            proto_err += 1
            trace.append({"step": step, "kind": "unknown_tool", "tool": name,
                          "raw": raw[:800]})
            scratch.append(f"OBSERVATION: '{name}' is not a tool. Available: "
                           f"{', '.join(tb.names())}.")
            continue
        key = name + json.dumps(args, sort_keys=True, default=str)
        if key in seen:
            trace.append({"step": step, "kind": "repeat", "tool": name})
            scratch.append(f"OBSERVATION: you already called {name}; its "
                           f"result is above. Answer or call a different "
                           f"tool.")
            continue
        seen.add(key)
        obs = tb.call(name, args)
        tool_calls += 1
        tools_used.append(name)
        if len(obs) > 2200:
            obs = obs[:2200] + " ...[clipped]"
        scratch += [f"ACTION: {name}({json.dumps(args, default=str)})",
                    f"OBSERVATION: {obs}"]
        trace.append({"step": step, "kind": "action", "tool": name,
                      "args": args, "raw": raw[:800],
                      "observation": obs[:400]})

    return ArmResult(s.unit, s.cycle, s.rul, pred, pred is None,
                     llm_calls, tool_calls, proto_err, term,
                     time.time() - t0, tools_used, answer, trace)


ARMS = ["P0_raw", "P1_featurized", "P2_agentic", "P3_hybrid"]


def run_arm(arm: str, ds: FD002, s: Snapshot, backend: Backend,
            bundle: MLBundle, max_steps: int = 6,
            xgb_full_mae: float = 12.4) -> ArmResult:
    if arm == "P0_raw":
        return run_direct(ds, s, backend, featurized=False)
    if arm == "P1_featurized":
        return run_direct(ds, s, backend, featurized=True)
    if arm == "P2_agentic":
        return run_agentic(ds, s, backend, bundle, False, max_steps,
                           xgb_full_mae)
    if arm == "P3_hybrid":
        return run_agentic(ds, s, backend, bundle, True, max_steps,
                           xgb_full_mae)
    raise ValueError(arm)

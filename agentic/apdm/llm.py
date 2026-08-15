"""LLM backends. Lean rewrite; the termination lessons are in agent.py.

ollama : explicit options (num_ctx, temperature=0, seed) -- the silent 2048
         head-truncation default is what killed the previous project's runs.
openai : parity backend for a strong-coordinator reference point.
dryrun : deterministic protocol-follower. NOT a model; its answers carry no
         scientific value. It exists so the whole pipeline (tools, loop,
         parsing, scoring) runs and is testable with no model installed.
Every call is appended to a JSONL log (prompt tail + full response), so any
future failure is visible in the artifact, not reconstructed forensically.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Optional


class Backend:
    name = "base"
    model = "?"
    cost = None            # optional apdm.hardware.CostModel (sim_ metrics)

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        raise NotImplementedError

    # -- shared logging ----------------------------------------------------
    log_path: Optional[Path] = None
    n_calls = 0
    prompt_chars = 0
    completion_chars = 0
    prompt_tokens = 0
    completion_tokens = 0
    sim_edge_s = 0.0
    sim_energy_j = 0.0
    _prev_prompt = ""

    def _log(self, system, prompt, out, meta=None):
        meta = dict(meta or {})
        self.n_calls += 1
        full = (system or "") + (prompt or "")
        self.prompt_chars += len(full)
        self.completion_chars += len(out or "")
        # REAL token counts when the runtime reports them; heuristic fallback.
        from .hardware import count_tokens
        n_in = meta.get("prompt_eval_count") or count_tokens(full)
        n_out = meta.get("eval_count") or count_tokens(out or "")
        self.prompt_tokens += int(n_in)
        self.completion_tokens += int(n_out)
        if self.cost is not None:
            i = 0
            m = min(len(self._prev_prompt), len(full))
            while i < m and self._prev_prompt[i] == full[i]:
                i += 1
            n_cached = count_tokens(full[:i])   # llama.cpp prefix-cache reuse
            est = self.cost.estimate(int(n_in), int(n_out), n_cached=n_cached)
            self.sim_edge_s += est["sim_edge_s"]
            self.sim_energy_j += est["sim_energy_j"]
            meta.update({k: round(v, 4) if isinstance(v, float) else v
                         for k, v in est.items()})
            meta["n_cached_est"] = n_cached
        self._prev_prompt = full
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": time.time(), "model": self.model,
                                    "prompt_tail": prompt[-1200:],
                                    "response": out, **meta},
                                   default=str) + "\n")

    def totals(self) -> Dict[str, float]:
        return {"n_calls": self.n_calls,
                "prompt_chars": self.prompt_chars,
                "completion_chars": self.completion_chars,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "sim_edge_s": round(self.sim_edge_s, 3),
                "sim_energy_j": round(self.sim_energy_j, 2)}

    def reset_prefix(self):
        self._prev_prompt = ""


class Ollama(Backend):
    name = "ollama"

    def __init__(self, model="llama3", num_ctx=8192, seed=0,
                 num_predict=400, log_path: Optional[Path] = None):
        import ollama
        self._c = ollama
        self.model = model
        self.log_path = log_path
        self.options = {"num_ctx": num_ctx, "temperature": 0.0, "seed": seed,
                        "num_predict": num_predict, "top_p": 1.0}

    def generate(self, prompt, system=None):
        full = f"{system}\n\n{prompt}" if system else prompt
        want_json = "OUTPUT PROTOCOL" in (system or "")
        kw = {"model": self.model, "prompt": full, "options": dict(self.options)}
        if want_json:
            kw["format"] = "json"
        for attempt in range(3):
            try:
                r = self._c.generate(**kw)
                out = (r.get("response") or "").strip()
                self._log(system, prompt, out,
                          {"attempt": attempt,
                           "prompt_eval_count": r.get("prompt_eval_count"),
                           "eval_count": r.get("eval_count"),
                           "total_duration_ns": r.get("total_duration")})
                if out:
                    return out
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    self._log(system, prompt, "", {"error": str(e)})
                    raise
                time.sleep(1.5 * (attempt + 1))
        return ""


class OpenAI(Backend):
    name = "openai"

    def __init__(self, model="gpt-4o-mini", seed=0, max_tokens=400,
                 log_path: Optional[Path] = None):
        from openai import OpenAI as _O
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")
        self._c = _O()
        self.model = model
        self.seed = seed
        self.max_tokens = max_tokens
        self.log_path = log_path

    def generate(self, prompt, system=None):
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        kw = {"model": self.model, "messages": msgs, "temperature": 0.0,
              "seed": self.seed, "max_tokens": self.max_tokens}
        if "OUTPUT PROTOCOL" in (system or ""):
            kw["response_format"] = {"type": "json_object"}
        r = self._c.chat.completions.create(**kw)
        out = (r.choices[0].message.content or "").strip()
        self._log(system, prompt, out)
        return out


class DryRun(Backend):
    """Deterministic plumbing stub. Direct prompts -> a fixed guess; agentic
    prompts -> one similar_cases call, then the retrieved median. Numbers are
    meaningless by design; only the mechanics are under test."""
    name = "dryrun"
    model = "dryrun"

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path

    def _gen_interp(self, prompt):
        devs = re.findall(r"(\w{2,7}) \([^)]*\): z=([+-]?\d+\.\d+)", prompt)
        devs = devs[:3] or [("Ps30", "1.0")]
        parts = ", ".join(f"{s_} {'high' if float(z) >= 0 else 'low'} "
                          f"(z={z})" for s_, z in devs)
        grav = min(5, max(1, int(1 + sum(abs(float(z)) for _, z in devs) / 3)))
        return json.dumps({
            "interpretation": f"[dry-run] Deviations in {parts} relative to "
                              f"the regime-healthy reference suggest a "
                              f"developing core-side inefficiency; verify "
                              f"sensor calibration and compressor health "
                              f"at next opportunity.",
            "gravity": grav,
            "components": ["HPC", "core"]})

    def _diag(self, prompt, system):
        if "OUTPUT PROTOCOL" in (system or ""):
            if "similar_anomalies" in (system or "") and \
                    "precedents" not in prompt:
                return json.dumps({"thought": "consult precedents",
                                   "action": {"tool": "similar_anomalies",
                                              "args": {"k": 4}}})
            if '"name"' not in prompt:
                return json.dumps({"thought": "need the library",
                                   "action": {"tool": "fault_library",
                                              "args": {}}})
            if "z=" not in prompt:
                return json.dumps({"thought": "need the evidence",
                                   "action": {"tool": "sensor_summary",
                                              "args": {}}})
            devs = re.findall(r"(\w{2,7}):\s*z=([+-]?\d+\.\d+)", prompt)
            devs = sorted(devs, key=lambda t: -abs(float(t[1])))[:3] or [
                ("Ps30", "1.0"), ("Nc", "1.0"), ("T50", "1.0")]
            names = re.findall(r'"name":\s*"([^"]+)"', prompt)
            return json.dumps({"thought": "compose", "final": {
                "phenotype": names[0] if names else "P0",
                "sensors": [[s_, "high" if float(z) >= 0 else "low"]
                            for s_, z in devs],
                "explanation": "[dry-run] deviations taken from the evidence."}})
        return _dry_diag_answer(prompt)

    def generate(self, prompt, system=None):
        _blob = (system or "") + prompt
        if '"gravity"' in (system or "") and "interpretation" in (system or ""):
            out = self._gen_interp(prompt)
            self._log(system, prompt, out)
            return out
        if ("PHENOTYPE:" in _blob or "diagnostic engineer" in _blob
                or '"phenotype"' in _blob):
            out = self._diag(prompt, system)
            self._log(system, prompt, out)
            return out
        if "OUTPUT PROTOCOL" in (system or ""):
            if "similar_cases" not in prompt or "OBSERVATION" not in prompt:
                out = json.dumps({"thought": "retrieve precedents",
                                  "action": {"tool": "similar_cases",
                                             "args": {"k": 7}}})
            else:
                m = re.search(r'"median_rul_of_neighbours":\s*([\d.]+)', prompt)
                v = int(float(m.group(1))) if m else 60
                out = json.dumps({"thought": "use retrieved median",
                                  "final": f"Neighbours suggest ~{v} cycles. "
                                           f"ANSWER: {v}"})
        else:
            m = re.search(r'"median_rul_of_neighbours":\s*([\d.]+)', prompt)
            v = int(float(m.group(1))) if m else 60
            out = f"[dry-run] ANSWER: {v}"
        self._log(system, prompt, out)
        return out


def _dry_diag_answer(prompt: str) -> str:
    """Compose a grounded three-line diagnosis from whatever the prompt or
    observations contain: first phenotype name in the library JSON, top
    z-deviations from the sensor summary."""
    names = re.findall(r'"name":\s*"([^"]+)"', prompt)
    ph = names[0] if names else "P0"
    devs = re.findall(r"(\w{2,7}):\s*z=([+-]?\d+\.\d+)", prompt)
    devs = sorted(devs, key=lambda t: -abs(float(t[1])))[:3]
    if not devs:
        devs = [("Ps30", "1.0"), ("Nc", "1.0"), ("T50", "1.0")]
    sens = ", ".join(f"{s_} {'high' if float(z) >= 0 else 'low'}"
                     for s_, z in devs)
    return (f"PHENOTYPE: {ph}\nSENSORS: {sens}\nEXPLANATION: [dry-run] "
            f"deviations taken verbatim from the evidence shown.")


def get_backend(name: str, model: str, num_ctx=8192, seed=0,
                log_path: Optional[Path] = None,
                device: Optional[str] = "orin_nano_8gb",
                hw_mode: str = "model") -> Backend:
    name = (name or "dryrun").lower()
    if name == "ollama":
        be = Ollama(model, num_ctx=num_ctx, seed=seed, log_path=log_path)
    elif name == "openai":
        be = OpenAI(model, seed=seed, log_path=log_path)
    elif name == "dryrun":
        be = DryRun(log_path=log_path)
    else:
        raise ValueError(name)
    if device and hw_mode == "model" and name != "openai":
        from .hardware import CostModel
        be.cost = CostModel(device, model, num_ctx=num_ctx)
        mem = be.cost.memory_footprint()
        tag = "fits" if mem["fits"] else "DOES NOT FIT"
        print(f"[hardware] {model} on {be.cost.d.name}: "
              f"{mem['total_gb']:.2f}/{mem['usable_gb']:.2f} GB at "
              f"num_ctx={num_ctx} -> {tag} (max ctx {be.cost.max_context()}); "
              f"sim_* are MODEL ESTIMATES, not measurements")
    return be

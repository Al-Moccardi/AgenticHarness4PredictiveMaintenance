"""
interpret_edge.py — SLM natural-language interpretation of test anomalies (Jetson tier)
=======================================================================================

Consumes the frozen detector's output (test_anomalies.csv) and produces, for every
anomaly event, a natural-language interpretation IN THE SAME SECTIONED FORMAT as the
original knowledge-base interpretations:

    **Anomaly Interpretation:** ... **Cause:** ... **Impact:** ...
    **Anomalous Trend:** ... **Expected Future Failures:** ... (gravity score: N)
    **Recommendation:** ...

Faithful to the original pipeline (TIOT-LLM Fig. 8):
  * input   = the anomaly's rule `text` (Splits / Score / cumulative counters)
  * context = up to the 3 PREVIOUS anomalies of the SAME unit (their rule text and,
              when already produced in this run, their interpretation) — strictly
              causal: only earlier cycles, never the future
  * few-shot = real interpretations from TRAIN units only (bundled)
  * model   = llama3.2:3b served locally by Ollama (temperature 0)

Operational properties for the Jetson Orin Nano:
  * resumable: one JSONL per unit under --out-dir; existing (unit,cycle) records are
    skipped, so an interrupted overnight run continues where it stopped
  * bounded memory: num_ctx default 4096 (KV cache and the 3B Q4 weights fit
    comfortably inside 8 GB unified memory)
  * per-call log (durations, token counts) for the paper's edge-cost table
  * --backend dryrun: deterministic offline templates for pipeline testing only

Leakage stance: the prompt NEVER contains RUL or any outcome information — only the
rule text, counters, and prior in-unit anomaly context. Enforced by construction and
by the smoke guard.

Usage on the Jetson (after `ollama pull llama3.2:3b`):
    python3 interpret_edge.py --backend ollama --detections test_anomalies.csv
    # pilot first:
    python3 interpret_edge.py --backend ollama --detections test_anomalies.csv --limit 10
Output:
    interpretations_test/unit_*.jsonl          (resumable store)
    interp_calls.jsonl                         (per-call log)
    test_FD002_with_interpretations.csv        (merged, KB-schema)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from . import paths
HERE = paths.PROJECT_ROOT

SYSTEM = """You are the diagnostic interpretation layer of an IoT predictive
maintenance system for turbofan engines, running on an edge device. You receive
one detected anomaly: its isolation-forest rule (feature/threshold splits), its
anomaly score, cumulative anomaly counters, and up to three previous anomalies
of the same engine unit.

Write a concise engineering interpretation with EXACTLY these sections:

**Anomaly Interpretation:** what combination of sensor conditions the rule
describes and what it suggests (2-3 sentences).
**Cause:** the most plausible physical cause, grounded ONLY in the sensors
named by the rule.
**Impact:** the expected consequence for engine performance if unaddressed.
**Anomalous Trend:** what the previous anomalies of this unit (if any) and the
cumulative counters suggest about the direction of this unit's condition.
**Expected Future Failures:** the risk outlook, ending with a gravity score in
the exact form: (gravity score: N)  where N is an integer 1-5
(1 = trivial, 5 = severe/imminent).
**Recommendation:** the concrete maintenance action.

Rules: mention only sensors that appear in the given splits or history; never
invent numeric values; keep the whole answer under 250 words."""

SENSOR_HINTS = {
    "T24": "LPC outlet temperature", "T30": "HPC outlet temperature",
    "T50": "LPT outlet temperature", "P30": "HPC outlet pressure",
    "Nf": "physical fan speed", "Nc": "physical core speed",
    "Ps30": "HPC outlet static pressure", "phi": "fuel flow / Ps30 ratio",
    "NRf": "corrected fan speed", "NRc": "corrected core speed",
    "BPR": "bypass ratio", "htBleed": "bleed enthalpy",
    "W31": "HPT coolant bleed", "W32": "LPT coolant bleed",
    "cycles": "operating cycle count"}


# ------------------------------------------------------------------ prompt build
def load_fewshot(path: Path, k: int = 2) -> str:
    """Bundled real TRAIN-unit examples (rule text -> interpretation)."""
    if not path.exists():
        return ""
    ex = json.loads(path.read_text())[:k]
    parts = []
    for e in ex:
        parts.append("EXAMPLE\nANOMALY RECORD:\n" + e["text"][:400]
                     + "\nINTERPRETATION:\n" + e["interpretation"][:900])
    return "\n\n".join(parts)


def build_prompt(fewshot: str, unit: int, cycle: int, regime: int, text: str,
                 history: List[Dict]) -> str:
    hints = ", ".join(f"{s}={d}" for s, d in SENSOR_HINTS.items())
    hist = ""
    if history:
        lines = []
        for h in history[-3:]:
            line = f"- cycle {h['cycle']}: {h['text'][:220]}"
            if h.get("interpretation"):
                first = str(h["interpretation"]).split("**Cause:**")[0]
                line += f"\n  earlier interpretation (excerpt): {first[-260:]}"
            lines.append(line)
        hist = ("PREVIOUS ANOMALIES OF THIS UNIT (most recent last):\n"
                + "\n".join(lines) + "\n\n")
    return (f"{fewshot}\n\nSENSOR LEGEND: {hints}\n\n{hist}"
            f"ANOMALY RECORD\nUnit {unit}, cycle {cycle}, operating regime "
            f"{regime}.\n{text[:500]}\n\n"
            f"Write the interpretation now, using exactly the required "
            f"sections.")


# ------------------------------------------------------------------- backends
class OllamaBackend:
    def __init__(self, model: str, host: str, num_ctx: int, log: Path):
        self.model, self.host, self.num_ctx = model, host, num_ctx
        self.log = log

    def generate(self, system: str, prompt: str, timeout: int = 300) -> str:
        import requests
        body = {"model": self.model, "system": system, "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "num_ctx": self.num_ctx}}
        t0 = time.time()
        r = requests.post(f"{self.host}/api/generate", json=body,
                          timeout=timeout)
        r.raise_for_status()
        j = r.json()
        self.last_metrics = {
            "model": self.model, "backend": "ollama",
            "seconds": round(time.time() - t0, 3),
            "total_duration_ns": j.get("total_duration"),
            "load_duration_ns": j.get("load_duration"),
            "prompt_eval_count": j.get("prompt_eval_count"),
            "prompt_eval_duration_ns": j.get("prompt_eval_duration"),
            "eval_count": j.get("eval_count"),
            "eval_duration_ns": j.get("eval_duration")}
        pe, pd_ = j.get("prompt_eval_count"), j.get("prompt_eval_duration")
        ec, ed = j.get("eval_count"), j.get("eval_duration")
        if pe and pd_:
            self.last_metrics["prefill_tps"] = round(pe / (pd_ / 1e9), 1)
        if ec and ed:
            self.last_metrics["decode_tps"] = round(ec / (ed / 1e9), 1)
        with self.log.open("a") as f:
            f.write(json.dumps({"t": time.time(), **self.last_metrics}) + "\n")
        return j.get("response", "")


class LlamaCppBackend:
    """llama.cpp `llama-server` backend via the OpenAI-compatible endpoint
    (POST /v1/chat/completions). The server applies the model's own chat
    template from the GGUF metadata, so system+user roles work exactly like
    Ollama. Metrics come from llama.cpp's `timings` object when present
    (prompt_n/prompt_ms/predicted_n/predicted_ms and per-second rates) and
    fall back to `usage` + wall clock, so edge_stats keeps working unchanged.
    """

    def __init__(self, model: str, host: str, num_ctx: int, log: Path):
        # num_ctx is a SERVER-side setting for llama.cpp (-c 4096); kept here
        # only so the CLI surface stays identical across backends.
        self.model, self.host, self.num_ctx = model, host, num_ctx
        self.log = log

    def generate(self, system: str, prompt: str, timeout: int = 600) -> str:
        import requests
        body = {"model": self.model,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": prompt}],
                "temperature": 0, "max_tokens": 512, "stream": False}
        t0 = time.time()
        r = requests.post(f"{self.host}/v1/chat/completions", json=body,
                          timeout=timeout)
        r.raise_for_status()
        j = r.json()
        content = j["choices"][0]["message"]["content"]
        wall = time.time() - t0
        usage = j.get("usage", {}) or {}
        tim = j.get("timings", {}) or {}
        p_n = tim.get("prompt_n", usage.get("prompt_tokens"))
        g_n = tim.get("predicted_n", usage.get("completion_tokens"))
        p_ms = tim.get("prompt_ms")
        g_ms = tim.get("predicted_ms")
        self.last_metrics = {
            "model": self.model, "backend": "llamacpp",
            "seconds": round(wall, 3),
            "total_duration_ns": int(wall * 1e9),
            "prompt_eval_count": p_n, "eval_count": g_n,
            "prompt_eval_duration_ns": int(p_ms * 1e6) if p_ms else None,
            "eval_duration_ns": int(g_ms * 1e6) if g_ms else None}
        pf = tim.get("prompt_per_second")
        dc = tim.get("predicted_per_second")
        if pf is None and p_n and p_ms:
            pf = p_n / (p_ms / 1000.0)
        if dc is None and g_n:
            dc = g_n / ((g_ms / 1000.0) if g_ms else max(wall, 1e-6))
        if pf:
            self.last_metrics["prefill_tps"] = round(float(pf), 1)
        if dc:
            self.last_metrics["decode_tps"] = round(float(dc), 1)
        with self.log.open("a") as f:
            f.write(json.dumps({"t": time.time(), **self.last_metrics}) + "\n")
        return content


class DryRunBackend:
    """Offline deterministic template — pipeline testing ONLY, never for the
    paper. Mentions the first two rule sensors so downstream grounding checks
    have something real to verify."""
    def __init__(self, log: Path):
        self.log = log

    def generate(self, system: str, prompt: str, timeout: int = 0) -> str:
        import re
        feats = re.findall(r"([A-Za-z]\w{0,7})\s*(?:<=|>)", prompt)
        a = feats[0] if feats else "T50"
        b = feats[1] if len(feats) > 1 else "Nc"
        n_hist = prompt.count("- cycle ")
        self.last_metrics = {"model": "dryrun", "backend": "dryrun",
                             "seconds": 0.01,
                             "prompt_eval_count": max(len(prompt)//4, 1),
                             "eval_count": 180, "prefill_tps": 900.0,
                             "decode_tps": 20.0,
                             "total_duration_ns": int(1e7)}
        with self.log.open("a") as f:
            f.write(json.dumps({"t": time.time(), **self.last_metrics}) + "\n")
        return (f"**Anomaly Interpretation:** The rule isolates this cycle by "
                f"an unusual combination of {a} and {b}, indicating a "
                f"deviation from the regime's normal envelope.\n"
                f"**Cause:** A plausible cause is degraded efficiency in the "
                f"subsystem monitored by {a}.\n"
                f"**Impact:** If unaddressed, performance and fuel "
                f"consumption will deteriorate.\n"
                f"**Anomalous Trend:** {n_hist} previous anomalies were "
                f"considered; the counters suggest a developing pattern.\n"
                f"**Expected Future Failures:** Moderate risk in the coming "
                f"cycles. (gravity score: 3)\n"
                f"**Recommendation:** Inspect the affected subsystem at the "
                f"next opportunity.")


# ---------------------------------------------------------------------- runner
def _cols(det: pd.DataFrame):
    u = "Unit_ID" if "Unit_ID" in det.columns else "unit_ID"
    c = "cycles" if "cycles" in det.columns else "cycle"
    return u, c


def _load_done(out_dir: Path):
    done: Dict[int, Dict[int, Dict]] = {}
    for f in out_dir.glob("unit_*.jsonl"):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.setdefault(int(r["unit_ID"]), {})[int(r["cycle"])] = r
    return done


def _record(out_dir, be, a, unit, cycle, rec_extra, prompt):
    t_start = time.time()
    try:
        resp = be.generate(SYSTEM, prompt)
    except Exception as e:  # noqa: BLE001
        print(f"[interp] u{unit}c{cycle}: {type(e).__name__}: {e} -> retry")
        time.sleep(2.0)
        resp = be.generate(SYSTEM, prompt)
    t_end = time.time()
    m = getattr(be, "last_metrics", {})
    rec = {"unit_ID": int(unit), "cycle": int(cycle),
           "source": "edge_generated",
           "gen_model": m.get("model", "?"),
           "interpretation": resp.strip(),
           "t_start": round(t_start, 3), "t_end": round(t_end, 3),
           "wall_s": round(t_end - t_start, 3), **rec_extra,
           "metrics": m}
    with (out_dir / f"unit_{int(unit)}.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def run_batch(a, be, fewshot, out_dir) -> None:
    det = pd.read_csv(a.detections)
    u, c = _cols(det)
    det = det.sort_values([u, c]).reset_index(drop=True)
    an = det[det.anomaly_label == -1]
    if a.units:
        an = an[an[u].isin(a.units)]
    done = _load_done(out_dir)
    n_done = sum(len(v) for v in done.values())
    print(f"[interp] batch: {len(an)} anomalies over {an[u].nunique()} units"
          f" | resume: {n_done} already on disk")
    from .progress import bar
    target = a.limit if a.limit else (len(an) - n_done)
    pb = bar(total=max(target, 0), desc="interpret", unit="anom")
    new, t0 = 0, time.time()
    try:
        for unit, g in an.groupby(u):
            hist: List[Dict] = []
            for _, row in g.sort_values(c).iterrows():
                cyc = int(row[c])
                rec = done.get(int(unit), {}).get(cyc)
                if rec is None:
                    prompt = build_prompt(fewshot, int(unit), cyc,
                                          int(row["h_clust"]),
                                          str(row["text"]), hist)
                    rec = _record(out_dir, be, a, unit, cyc, {}, prompt)
                    new += 1
                    pb.update(1)
                    m = rec.get("metrics", {})
                    if m.get("decode_tps"):
                        pb.set_postfix_str(f"u{unit} {m.get('eval_count','?')}tok "
                                           f"{m['decode_tps']:.0f}tok/s")
                    if a.limit and new >= a.limit:
                        raise KeyboardInterrupt
                hist.append({"cycle": cyc, "text": str(row["text"]),
                             "interpretation": rec.get("interpretation")})
    except KeyboardInterrupt:
        print(f"\n[interp] stopped after {new} new (resume by re-running)")
    finally:
        pb.close()


def run_follow(a, be, fewshot, out_dir) -> None:
    """Consume the streaming daemon's anomaly queue FIFO, live."""
    qp = Path(a.queue)
    done = _load_done(out_dir)
    hist: Dict[int, List[Dict]] = {}
    for uu, recs in done.items():
        hist[uu] = [{"cycle": cc, "text": "",
                     "interpretation": r.get("interpretation")}
                    for cc, r in sorted(recs.items())]
    from .progress import bar
    off, new = 0, 0
    idle_since = time.time()
    pb = bar(total=None, desc="interpret(stream)", unit="anom")
    print(f"[interp] follow mode on {Path(a.queue).name} "
          f"(max-idle {a.max_idle}s)")
    while True:
        if not qp.exists():
            time.sleep(0.2)
            if time.time() - idle_since > a.max_idle:
                break
            continue
        txt = qp.read_text()
        lines = [l for l in txt[off:].splitlines() if l.strip()]
        off = len(txt)
        if not lines:
            if time.time() - idle_since > a.max_idle:
                break
            time.sleep(a.poll)
            continue
        idle_since = time.time()
        for line in lines:
            ev = json.loads(line)
            unit, cyc = int(ev["unit"]), int(ev["cycle"])
            if done.get(unit, {}).get(cyc):
                continue
            h = hist.setdefault(unit, [])
            prompt = build_prompt(fewshot, unit, cyc, int(ev["h_clust"]),
                                  str(ev["text"]), h)
            extra = {"t_arrival": ev.get("t_arrival"),
                     "t_detected": ev.get("t_detected"),
                     "queue_wait_s": round(time.time()
                                           - ev.get("t_detected",
                                                    time.time()), 3)}
            rec = _record(out_dir, be, a, unit, cyc, extra, prompt)
            rec["staleness_s"] = round(rec["t_end"]
                                       - ev.get("t_arrival", rec["t_end"]), 3)
            h.append({"cycle": cyc, "text": str(ev["text"]),
                      "interpretation": rec["interpretation"]})
            done.setdefault(unit, {})[cyc] = rec
            new += 1
            pb.update(1)
            if a.limit and new >= a.limit:
                pb.close()
                print(f"[interp] follow: limit {a.limit} reached")
                return
    pb.close()
    print(f"[interp] follow: {new} interpretations, queue drained")


def merge(a, out_dir) -> None:
    det = pd.read_csv(a.detections)
    u, c = _cols(det)
    interp = {}
    for f in out_dir.glob("unit_*.jsonl"):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                interp[(int(r["unit_ID"]), int(r["cycle"]))] = \
                    r["interpretation"]
    det["interpretation"] = [
        interp.get((int(r[u]), int(r[c])),
                   "No interpretation (inlier)" if r["anomaly_label"] == 1
                   else "")
        for _, r in det.iterrows()]
    covered = sum(1 for _, r in det[det.anomaly_label == -1].iterrows()
                  if interp.get((int(r[u]), int(r[c]))))
    det.to_csv(Path(a.merged_csv), index=False)
    print(f"[interp] merged -> {a.merged_csv}  ({covered}/"
          f"{len(det[det.anomaly_label == -1])} anomalies interpreted)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", default=str(paths.RESULTS / "test_anomalies.csv"))
    ap.add_argument("--backend", default="llamacpp",
                    choices=["llamacpp", "ollama", "dryrun"])
    ap.add_argument("--model", default="llama-3.2-3b-instruct",
                    help="recorded as gen_model; for ollama use e.g. llama3.2:3b")
    ap.add_argument("--host", default=None,
                    help="server URL; default per backend: llamacpp http://localhost:8080, ollama http://localhost:11434")
    ap.add_argument("--num-ctx", type=int, default=4096)
    ap.add_argument("--out-dir", default=str(paths.RESULTS / "interpretations"))
    ap.add_argument("--merged-csv",
                    default=str(paths.RESULTS / "test_FD002_with_interpretations.csv"))
    ap.add_argument("--fewshot", default=str(paths.FEWSHOT))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--units", nargs="*", type=int, default=None)
    ap.add_argument("--follow", action="store_true",
                    help="consume the live anomaly queue instead of a csv")
    ap.add_argument("--queue", default=str(paths.RESULTS / "queue_anomalies.jsonl"))
    ap.add_argument("--max-idle", type=float, default=20.0)
    ap.add_argument("--poll", type=float, default=0.3)
    ap.add_argument("--no-merge", action="store_true")
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(exist_ok=True)
    log = paths.RESULTS / "interp_calls.jsonl"; paths.ensure_results()
    host = a.host or {"llamacpp": "http://localhost:8080",
                      "ollama": "http://localhost:11434"}.get(a.backend)
    if a.backend == "llamacpp":
        be = LlamaCppBackend(a.model, host, a.num_ctx, log)
    elif a.backend == "ollama":
        be = OllamaBackend(a.model, host, a.num_ctx, log)
    else:
        be = DryRunBackend(log)
    fewshot = load_fewshot(Path(a.fewshot))
    if a.follow:
        run_follow(a, be, fewshot, out_dir)
        if not a.no_merge and Path(a.detections).exists():
            merge(a, out_dir)
    else:
        run_batch(a, be, fewshot, out_dir)
        if not a.no_merge:
            merge(a, out_dir)


if __name__ == "__main__":
    main()

"""
edge_stats.py — full hardware & pipeline statistics for the Jetson edge tier
============================================================================
Consumes everything the pipeline logs and produces the paper-ready numbers:

  inputs   stream_events.jsonl      (per-row detection timing + arrival ts)
           interp_calls.jsonl       (per-call Ollama metrics)
           interpretations_test/    (per-anomaly records incl. queue_wait,
                                     staleness in follow mode)
           telemetry.jsonl          (cpu/ram/gpu/temp/power samples)
  outputs  edge_stats.csv           (one row per interpreted anomaly)
           edge_stats_report.md     (aggregate tables, ready to cite)
           figures/*.png            (latency hist, tokens-vs-latency, queue
                                     depth, telemetry timeline)  [matplotlib]

    python3 edge_stats.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths
HERE = paths.RESULTS


def _load_jsonl(p: Path):
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def pct(x, q):
    x = np.asarray(list(x), float)
    return float(np.percentile(x, q)) if len(x) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(paths.RESULTS / "interpretations"))  # abs or relative
    ap.add_argument("--report", default=str(paths.RESULTS / "edge_stats_report.md"))
    ap.add_argument("--csv", default=str(paths.RESULTS / "edge_stats.csv"))
    ap.add_argument("--events", default="stream_events.jsonl")
    ap.add_argument("--queue", default="queue_anomalies.jsonl")
    ap.add_argument("--telemetry", default="telemetry.jsonl")
    a = ap.parse_args()

    L = []
    A = L.append
    A("# Edge tier — full statistics report\n")

    # ---------------- detection ------------------------------------------
    ev = _load_jsonl(Path(a.events) if Path(a.events).is_absolute() else HERE / a.events)
    if ev:
        d_us = np.array([e["detect_us"] for e in ev], float)
        span = max(e["t_detected"] for e in ev) - min(e["t_detected"]
                                                     for e in ev)
        thru = len(ev) / span if span > 0 else float("inf")
        n_an = sum(1 for e in ev if e["label"] == -1)
        A("## Detection (frozen Isolation-Forest, online)\n")
        A(f"- rows processed: **{len(ev)}** ({n_an} anomalies, "
          f"{n_an/len(ev):.1%})")
        A(f"- per-row inference: median **{pct(d_us,50)/1000:.2f} ms**, "
          f"p95 {pct(d_us,95)/1000:.2f} ms, p99 {pct(d_us,99)/1000:.2f} ms")
        A(f"- sustained throughput during replay: **{thru:.0f} rows/s** "
          f"(compute-bound capacity ≈ {1e6/np.mean(d_us):.0f} rows/s)\n")

    # ---------------- interpretation -------------------------------------
    recs = []
    for f in sorted(Path(a.out_dir).glob("unit_*.jsonl")):
        recs += _load_jsonl(f)
    rows = []
    for r in recs:
        m = r.get("metrics", {}) or {}
        rows.append({
            "unit_ID": r["unit_ID"], "cycle": r["cycle"],
            "backend": m.get("backend", "?"), "model": r.get("gen_model"),
            "wall_s": r.get("wall_s"),
            "queue_wait_s": r.get("queue_wait_s"),
            "staleness_s": r.get("staleness_s"),
            "prompt_tokens": m.get("prompt_eval_count"),
            "gen_tokens": m.get("eval_count"),
            "load_s": (m.get("load_duration_ns") or 0) / 1e9 or None,
            "prefill_s": (m.get("prompt_eval_duration_ns") or 0) / 1e9
            or None,
            "decode_s": (m.get("eval_duration_ns") or 0) / 1e9 or None,
            "prefill_tps": m.get("prefill_tps"),
            "decode_tps": m.get("decode_tps"),
            "interp_chars": len(str(r.get("interpretation", "")))})
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(Path(a.csv), index=False)
        w = df.wall_s.dropna()
        A("## SLM interpretation (per anomaly)\n")
        A(f"- interpretations: **{len(df)}** | model(s): "
          f"{', '.join(sorted(set(df.model.dropna().astype(str))))}")
        A(f"- wall latency: median **{pct(w,50):.1f} s**, p95 "
          f"{pct(w,95):.1f} s, p99 {pct(w,99):.1f} s")
        if df.prompt_tokens.notna().any():
            A(f"- tokens: prompt median {df.prompt_tokens.median():.0f}, "
              f"generated median {df.gen_tokens.median():.0f}")
        if df.prefill_tps.notna().any():
            A(f"- token rates: prefill median "
              f"{df.prefill_tps.median():.0f} tok/s, decode median "
              f"**{df.decode_tps.median():.1f} tok/s**")
        if df.queue_wait_s.notna().any():
            A(f"- queue wait: median {df.queue_wait_s.median():.1f} s, "
              f"p95 {pct(df.queue_wait_s.dropna(),95):.1f} s")
        if df.staleness_s.notna().any():
            A(f"- end-to-end staleness (arrival→interpretation done): "
              f"median **{df.staleness_s.median():.1f} s**, p95 "
              f"{pct(df.staleness_s.dropna(),95):.1f} s")
        A(f"- total SLM busy time: {w.sum()/3600:.2f} h | mean per anomaly "
          f"{w.mean():.1f} s\n")

    # ---------------- telemetry ------------------------------------------
    tel = _load_jsonl(Path(a.telemetry) if Path(a.telemetry).is_absolute() else HERE / a.telemetry)
    if tel:
        t = pd.DataFrame(tel)
        A("## Hardware telemetry\n")
        A(f"- samples: {len(t)} over {(t.t.max()-t.t.min())/60:.1f} min")
        A(f"- CPU: mean {t.cpu_pct.mean():.0f}%, max {t.cpu_pct.max():.0f}%")
        if t.mem_used_mb.notna().any():
            A(f"- RAM: mean {t.mem_used_mb.mean()/1024:.2f} GiB, peak "
              f"**{t.mem_used_mb.max()/1024:.2f} GiB** of "
              f"{t.mem_total_mb.max()/1024:.1f}")
        if t.gpu_pct.notna().any():
            A(f"- GPU: mean {t.gpu_pct.mean():.0f}%, max "
              f"{t.gpu_pct.max():.0f}%")
        if t.temp_c_max.notna().any():
            A(f"- max temperature: {t.temp_c_max.max():.1f} °C")
        if t.power_w.notna().any():
            dt = np.diff(t.t, prepend=t.t.iloc[0])
            energy_wh = float(np.nansum(t.power_w.values * dt) / 3600)
            A(f"- power: mean {t.power_w.mean():.1f} W, peak "
              f"{t.power_w.max():.1f} W | integrated energy "
              f"**{energy_wh:.2f} Wh**")
            if len(df):
                A(f"- energy per interpretation ≈ "
                  f"{energy_wh*3600/len(df):.0f} J")
        A("")

    # ---------------- figures --------------------------------------------
    figs = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        (paths.RESULTS / "figures").mkdir(exist_ok=True)
        if len(df) and df.wall_s.notna().any():
            fig, ax = plt.subplots(figsize=(6, 3.4))
            ax.hist(df.wall_s.dropna(), bins=30)
            ax.set_xlabel("interpretation wall latency (s)")
            ax.set_ylabel("count")
            ax.set_title("SLM interpretation latency")
            fig.tight_layout()
            fig.savefig(paths.RESULTS / "figures/fig_interp_latency.png", dpi=160)
            plt.close(fig)
            figs.append("figures/fig_interp_latency.png")
        if len(df) and df.gen_tokens.notna().any():
            fig, ax = plt.subplots(figsize=(6, 3.4))
            ax.scatter(df.gen_tokens, df.wall_s, s=10, alpha=.5)
            ax.set_xlabel("generated tokens")
            ax.set_ylabel("wall s")
            ax.set_title("Latency vs generated tokens")
            fig.tight_layout()
            fig.savefig(paths.RESULTS / "figures/fig_tokens_vs_latency.png", dpi=160)
            plt.close(fig)
            figs.append("figures/fig_tokens_vs_latency.png")
        q = _load_jsonl(Path(a.queue) if Path(a.queue).is_absolute() else HERE / a.queue)
        fin = sorted(r["t_end"] for r in recs if "t_end" in r)
        if q and fin:
            enq = sorted(e["t_detected"] for e in q)
            t0 = min(enq)
            ts = np.linspace(0, max(max(enq), max(fin)) - t0, 300)
            depth = [sum(1 for x in enq if x - t0 <= s)
                     - sum(1 for x in fin if x - t0 <= s) for s in ts]
            fig, ax = plt.subplots(figsize=(6, 3.4))
            ax.plot(ts, depth)
            ax.set_xlabel("time since first anomaly (s)")
            ax.set_ylabel("queue depth")
            ax.set_title("Anomaly queue depth over the replay")
            fig.tight_layout()
            fig.savefig(paths.RESULTS / "figures/fig_queue_depth.png", dpi=160)
            plt.close(fig)
            figs.append("figures/fig_queue_depth.png")
        if tel:
            t = pd.DataFrame(tel)
            fig, ax = plt.subplots(figsize=(7, 3.4))
            tt = t.t - t.t.min()
            ax.plot(tt, t.mem_used_mb / 1024, label="RAM GiB")
            if t.gpu_pct.notna().any():
                ax2 = ax.twinx()
                ax2.plot(tt, t.gpu_pct, "C1", alpha=.6)
                ax2.set_ylabel("GPU %")
            elif t.power_w.notna().any():
                ax2 = ax.twinx()
                ax2.plot(tt, t.power_w, "C2", alpha=.6)
                ax2.set_ylabel("W")
            ax.set_xlabel("s")
            ax.set_ylabel("RAM GiB")
            ax.set_title("Telemetry timeline")
            fig.tight_layout()
            fig.savefig(paths.RESULTS / "figures/fig_telemetry.png", dpi=160)
            plt.close(fig)
            figs.append("figures/fig_telemetry.png")
    except ImportError:
        A("_matplotlib not installed — figures skipped "
          "(pip3 install matplotlib)_\n")
    if figs:
        A("## Figures\n")
        for f in figs:
            A(f"![]({f})")
        A("")

    Path(a.report).write_text("\n".join(L))
    print(f"[stats] wrote {a.report}"
          + (f", {a.csv}" if len(df) else "")
          + (f", {len(figs)} figures" if figs else ""))
    print("\n".join(L[:26]))


if __name__ == "__main__":
    main()

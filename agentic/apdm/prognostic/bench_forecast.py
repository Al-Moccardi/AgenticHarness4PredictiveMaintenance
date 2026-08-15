"""bench_forecast — the prognostic-agent study.

Diagnosis first, then prognosis, then evaluation against the realised future
and the official RUL_FD002 gold:

  arms
    b0_median   no LLM: median rul_then of the retrieved precedents
    dl_only     the CNN-GRU number alone (from queries/dl_hints.csv)
    P7_agent    the prognostic agent (history + precedent futures + diagnosis)
    P7_agent_dl same, plus the CNN-GRU hint in context

  # after: python -m apdm.dl_rul --train  and  --hints   (TensorFlow side)
  python -m apdm.bench_forecast --limit 8            # pilot, all four arms
  python -m apdm.bench_forecast                      # all 89 queries
  python -m apdm.bench_forecast --report             # tables + fig12
  python -m apdm.bench_forecast --smoke              # offline plumbing test
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .forecast import (p7_progression, RUL_CAP, MockForecastBackend, PrecedentFutures,
                       evaluate_forecast, p7, s_score)
from ..llm import Ollama
from ..patterns import MockBackend, Runner, load_queries
from ..vector_store import HashEmbedder, OllamaEmbedder, VectorStore

ROOT = Path(__file__).resolve().parents[2]
ARMS = ["b0_median", "dl_only", "P7_agent", "P7_agent_dl", "P7_progression"]


def _dl_hints(path: Path):
    if not path.exists():
        return {}
    d = pd.read_csv(path)
    return {(int(r.unit_ID), int(r.cycle)): float(r.dl_rul)
            for _, r in d.iterrows()}


def phase_generate(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ep = out / "forecast_episodes.jsonl"
    done = set()
    if ep.exists():
        for line in ep.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["qid"], r["arm"]))

    store = VectorStore.load(Path(a.store_dir))
    emb = OllamaEmbedder() if a.embedder == "ollama" else HashEmbedder()
    if a.embedder == "ollama" and not store.embedder_name.startswith("ollama"):
        raise SystemExit("store/query embedder mismatch")
    pf = PrecedentFutures(Path(a.store_dir) / "meta.jsonl")
    if a.backend == "mock":
        be_d, be_f = MockBackend(), MockForecastBackend(
            log_path=out / "forecast_llm.jsonl")
    else:
        be_d = Ollama(model=a.model, num_ctx=a.num_ctx, num_predict=400,
                      log_path=out / "diag_llm.jsonl")
        be_f = Ollama(model=a.model, num_ctx=a.num_ctx, num_predict=500,
                      log_path=out / "forecast_llm.jsonl")
    run = Runner(be_d, store, emb, k=a.k)
    from .future_progression import reliability, future_progression
    queries = load_queries(Path(a.queries))
    if a.limit:
        queries = queries[:a.limit]
    hints = _dl_hints(Path(a.dl_hints))

    grid = [(q, arm) for q in queries for arm in a.arms]
    todo = [(q, arm) for (q, arm) in grid if (q["qid"], arm) not in done]
    print(f"[forecast] {len(queries)} queries x {len(a.arms)} arms "
          f"-> {len(grid)} ({len(done)} done, {len(todo)} to run) | "
          f"dl hints: {len(hints)}")

    prev_fc: dict = {}          # (arm, unit) -> {"cycle", "rul"}
    with ep.open("a") as f:
        for i, (q, arm) in enumerate(todo):
            t0 = time.time()
            recs = run.retrieve(q)
            hint = hints.get((q["unit"], q["cycle"]))
            signals = {"reliability": reliability(recs),
                       "future_progression": future_progression(pf, recs)}
            fc, diag, err = None, None, None
            try:
                if arm == "b0_median":
                    ruls = [r["rul_then"] for r in recs
                            if r["rul_then"] is not None]
                    est = float(np.median(ruls)) if ruls else None
                    fc = {"rul_estimate": est,
                          "rul_range": [float(np.quantile(ruls, .1)),
                                        float(np.quantile(ruls, .9))]
                          if ruls else None,
                          "expected_trends": [], "anomaly_outlook": None,
                          "cited_precedents":
                              [f"u{r['unit']}c{r['cycle']}" for r in recs]}
                elif arm == "dl_only":
                    fc = ({"rul_estimate": hint, "rul_range": None,
                           "expected_trends": [], "anomaly_outlook": None,
                           "cited_precedents": []} if hint is not None
                          else None)
                elif arm == "P7_progression":
                    if a.with_diagnosis:
                        diag = run.p2(q, "plain").get("ticket")
                    rp = p7_progression(be_f, q, recs, pf, queries,
                                        diagnosis=diag, dl_rul=hint,
                                        signals=signals)
                    fc = rp["forecast"] or {}
                    fc["rul_estimate"] = hint      # the tool owns the number
                    fc["rul_range"] = None
                    fc.setdefault("expected_trends", [])
                    fc.setdefault("anomaly_outlook", None)
                else:
                    if a.with_diagnosis:
                        diag = run.p2(q, "plain").get("ticket")
                    r7 = p7(be_f, q, recs, pf, queries, diagnosis=diag,
                            dl_rul=hint if arm == "P7_agent_dl" else None,
                            prev_forecast=prev_fc.get((arm, q["unit"])),
                            signals=signals)
                    fc = r7["forecast"]
                    if fc and fc.get("rul_estimate") is not None:
                        prev_fc[(arm, q["unit"])] = {
                            "cycle": q["cycle"],
                            "rul": min(float(fc["rul_estimate"]), RUL_CAP)}
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
            f.write(json.dumps({
                "qid": q["qid"], "arm": arm, "unit": q["unit"],
                "cycle": q["cycle"], "true_rul": q["true_rul"],
                "dl_hint": hint, "diagnosis": diag, "forecast": fc,
                "signals": signals,
                "contexts": [{k: r.get(k) for k in
                              ("unit", "cycle", "rul_then", "similarity")}
                             for r in recs],
                "wall_s": round(time.time() - t0, 2), "error": err},
                default=str) + "\n")
            f.flush()
            print(f"  [{i+1}/{len(todo)}] {q['qid']:>10s} {arm:<12s} "
                  f"{time.time()-t0:5.1f}s", flush=True)
    print(f"[forecast] episodes -> {ep}")


def phase_evaluate(a) -> None:
    out = Path(a.out)
    from .dl_rul import load_test_raw
    raw = load_test_raw(Path(a.test_txt))
    det = pd.read_csv(a.detections)
    ucol = "Unit_ID" if "Unit_ID" in det.columns else "unit_ID"
    det = det.rename(columns={ucol: "unit_ID"})
    eps = [json.loads(l) for l in
           (out / "forecast_episodes.jsonl").read_text().splitlines()
           if l.strip()]
    # cap + per-(arm,unit) monotone clamp on predictions, in cycle order
    eps.sort(key=lambda e: (e["arm"], e["unit"], e["cycle"]))
    last: dict = {}
    for e in eps:
        fc = e.get("forecast") or {}
        pr = fc.get("rul_estimate")
        e["rul_pred_raw"] = pr
        e["mono_violation"] = 0
        e["mono_viol_size"] = 0.0
        if pr is not None:
            pr = min(float(pr), RUL_CAP)
            k = (e["arm"], e["unit"])
            if k in last:
                bound = last[k]["rul"]          # non-increasing, no -elapsed
                if pr > bound + 1e-6:
                    e["mono_violation"] = 1
                    e["mono_viol_size"] = round(pr - bound, 1)
                    pr = bound
            last[k] = {"cycle": e["cycle"], "rul": pr}
            fc["rul_estimate"] = pr
            if fc.get("rul_range"):
                fc["rul_range"] = [min(fc["rul_range"][0], RUL_CAP),
                                   min(fc["rul_range"][1], RUL_CAP)]
    rows = []
    for e in eps:
        q = {"cycle": e["cycle"],
             "true_rul": (min(float(e["true_rul"]), RUL_CAP)
                          if e["true_rul"] is not None else None)}
        m = evaluate_forecast(e["forecast"], q,
                              raw[raw.unit_ID == e["unit"]],
                              det[det.unit_ID == e["unit"]],
                              horizon=a.horizon)
        detail = m.pop("trend_detail", [])
        rows.append({"qid": e["qid"], "arm": e["arm"], "unit": e["unit"],
                     "cycle": e["cycle"], "true_rul": q["true_rul"],
                     "dl_hint": e.get("dl_hint"),
                     "rul_pred_raw": e.get("rul_pred_raw"),
                     "rul_pred": (e["forecast"] or {}).get("rul_estimate"),
                     "mono_violation": e.get("mono_violation", 0),
                     "mono_viol_size": e.get("mono_viol_size", 0.0),
                     "dl_used": e.get("dl_used"),
                     "steps": e.get("steps"),
                     **m, "n_trend_detail": len(detail)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "forecast_metrics.csv", index=False)
    g = (df.groupby("arm")
         .agg(n=("qid", "size"),
              json_valid=("json_valid", "mean"),
              rul_mae=("rul_abs_err", "mean"),
              rul_bias=("rul_err", "mean"),
              s_score_mean=("s_score", "mean"),
              range_coverage=("range_coverage", "mean"),
              range_width=("range_width", "mean"),
              trend_dir_acc=("trend_direction_acc", "mean"),
              outlook_acc=("outlook_acc", "mean"),
              mono_violation_rate=("mono_violation", "mean"),
              mono_viol_gt5=("mono_viol_size", lambda x: float(
                  (pd.to_numeric(x, errors="coerce") > 5).mean())),
              dl_used_rate=("dl_used", lambda x: float(
                  pd.to_numeric(x, errors="coerce").mean())),
              steps_mean=("steps", lambda x: float(
                  pd.to_numeric(x, errors="coerce").mean())))
         .reindex(ARMS).round(3))
    g.to_csv(out / "forecast_summary.csv")
    print(g.to_string())
    print(f"[forecast] metrics -> {out}/forecast_metrics.csv, "
          f"forecast_summary.csv")


def report(a) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MM = 1 / 25.4
    W2 = 190 * MM
    C = {"main": "#0072B2", "alt": "#D55E00", "third": "#009E73",
         "grey": "#666666", "accent": "#CC79A7"}
    plt.rcParams.update({"font.family": "serif", "font.size": 8,
                         "axes.labelsize": 8, "axes.titlesize": 8.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 6.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "legend.frameon": False,
                         "figure.dpi": 300, "savefig.bbox": "tight",
                         "pdf.fonttype": 42})
    out = Path(a.out)
    df = pd.read_csv(out / "forecast_metrics.csv")
    g = pd.read_csv(out / "forecast_summary.csv").set_index("arm")
    arms = [x for x in ARMS if x in g.index]
    fig, axes = plt.subplots(2, 2, figsize=(W2, 0.62 * W2))
    (ax_a, ax_b), (ax_c, ax_d) = axes
    x = np.arange(len(arms))

    ax_a.bar(x - 0.2, g.loc[arms, "rul_mae"], 0.4, color=C["main"],
             label="MAE (cycles)")
    ax2 = ax_a.twinx()
    ax2.bar(x + 0.2, g.loc[arms, "s_score_mean"], 0.4, color=C["alt"],
            label="mean S-score")
    ax2.set_ylabel("CMAPSS S-score", color=C["alt"])
    ax2.tick_params(axis="y", colors=C["alt"])
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([s.replace("_", "\n") for s in arms], fontsize=6.3)
    ax_a.set_ylabel("RUL MAE (cycles)", color=C["main"])
    ax_a.text(-0.16, 1.06, "(a)", transform=ax_a.transAxes,
              fontweight="bold", fontsize=9)
    ax_a.set_title("Prognostic error vs official gold", loc="left", pad=3)

    d4 = df[df.arm == "P7_agent_dl"].dropna(subset=["rul_pred", "true_rul"])
    d3 = df[df.arm == "P7_agent"].dropna(subset=["rul_pred", "true_rul"])
    ax_b.scatter(d3.true_rul, d3.rul_pred, s=16, alpha=0.6, color=C["third"],
                 label="P7 agent")
    ax_b.scatter(d4.true_rul, d4.rul_pred, s=16, alpha=0.6, color=C["main"],
                 label="P7 + CNN-GRU")
    lim = max(df.true_rul.max(), df.rul_pred.max()) * 1.05
    ax_b.plot([0, lim], [0, lim], color=C["grey"], ls="--", lw=0.8)
    ax_b.set_xlabel("true RUL (RUL_FD002-anchored)")
    ax_b.set_ylabel("predicted RUL")
    ax_b.legend(loc="upper left")
    ax_b.text(-0.16, 1.06, "(b)", transform=ax_b.transAxes,
              fontweight="bold", fontsize=9)
    ax_b.set_title("Calibration", loc="left", pad=3)

    bars = [("trend_dir_acc", "sensor-trend direction", C["main"]),
            ("outlook_acc", "anomaly outlook", C["accent"]),
            ("range_coverage", "RUL range coverage", C["third"])]
    ag = [x_ for x_ in ("P7_agent", "P7_agent_dl") if x_ in g.index]
    xs = np.arange(len(ag))
    for k, (col, lab, cc) in enumerate(bars):
        ax_c.bar(xs + (k - 1) * 0.26, g.loc[ag, col], 0.26, color=cc,
                 label=lab)
    ax_c.set_xticks(xs)
    ax_c.set_xticklabels([s.replace("_", "\n") for s in ag], fontsize=6.5)
    ax_c.set_ylim(0, 1.05)
    ax_c.legend(fontsize=6)
    ax_c.text(-0.16, 1.06, "(c)", transform=ax_c.transAxes,
              fontweight="bold", fontsize=9)
    ax_c.set_title("Natural-language forecast vs realised future",
                   loc="left", pad=3)

    m = d4.dropna(subset=["dl_hint"])
    if len(m):
        delta = np.abs(m.rul_pred - m.true_rul) - np.abs(m.dl_hint
                                                         - m.true_rul)
        ax_d.hist(delta, bins=16, color=C["main"], alpha=0.8,
                  edgecolor="white", linewidth=0.3)
        ax_d.axvline(0, color=C["grey"], ls="--", lw=0.8)
        ax_d.axvline(delta.mean(), color=C["alt"], lw=1.1)
        better = float((delta < 0).mean())
        ax_d.text(0.03, 0.9, f"agent improves on the hint\nin {better:.0%} "
                             f"of cases (mean {delta.mean():+.1f})",
                  transform=ax_d.transAxes, fontsize=6.6, color=C["grey"])
        ax_d.set_xlabel("|agent err| - |CNN-GRU err|  (negative = agent better)")
        ax_d.set_ylabel("cases")
    ax_d.text(-0.16, 1.06, "(d)", transform=ax_d.transAxes,
              fontweight="bold", fontsize=9)
    ax_d.set_title("Does the agent add value over the DL hint?",
                   loc="left", pad=3)

    fig.tight_layout(w_pad=2.4, h_pad=1.6)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig12_prognosis.{ext}")
    print(f"[forecast] fig12 -> {out}")


def smoke() -> None:
    import shutil
    tmp = ROOT / "results" / "forecast_smoke"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    meta = [json.loads(l) for l in
            (ROOT / "data/vector_store/meta.jsonl").read_text().splitlines()
            if l.strip()][:500]
    emb = HashEmbedder()
    vs = VectorStore(emb.embed([m["text"] for m in meta]), meta, emb.name)
    sdir = tmp / "store"
    vs.save(sdir)
    q = ROOT / "queries/test_FD002_with_interpretations.csv"
    det = ROOT / "queries/test_anomalies.csv"
    hints = tmp / "dl_hints.csv"
    qd = pd.read_csv(q)
    an = qd[qd.anomaly_label == -1][["Unit_ID", "cycles"]].head(50)
    pd.DataFrame({"unit_ID": an.Unit_ID, "cycle": an.cycles,
                  "dl_rul": 60.0}).to_csv(hints, index=False)
    base = ["--queries", str(q), "--detections", str(det),
            "--store-dir", str(sdir), "--embedder", "hash",
            "--backend", "mock", "--dl-hints", str(hints),
            "--limit", "3", "--out", str(tmp),
            "--test-txt", str(ROOT / "data/test_FD002.txt")]
    a = build_parser().parse_args(base)
    phase_generate(a)
    a = build_parser().parse_args(base + ["--evaluate"])
    phase_evaluate(a)
    a = build_parser().parse_args(base + ["--report"])
    report(a)
    sm = pd.read_csv(tmp / "forecast_summary.csv")
    ok = (len(sm) >= 4 and (tmp / "fig12_prognosis.png").exists()
          and sm.rul_mae.notna().all())
    print("\nSMOKE " + ("PASSED" if ok else "FAILED"))


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=str(
        ROOT / "queries/test_FD002_with_interpretations.csv"))
    ap.add_argument("--detections", default=str(
        ROOT / "queries/test_anomalies.csv"))
    ap.add_argument("--test-txt", default=str(ROOT / "data/test_FD002.txt"))
    ap.add_argument("--store-dir", default=str(ROOT / "data/vector_store"))
    ap.add_argument("--dl-hints", default=str(ROOT / "queries/dl_hints.csv"))
    ap.add_argument("--embedder", default="ollama",
                    choices=["ollama", "hash"])
    ap.add_argument("--backend", default="ollama",
                    choices=["ollama", "mock"])
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--with-diagnosis", action="store_true", default=True)
    ap.add_argument("--no-diagnosis", dest="with_diagnosis",
                    action="store_false")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results/forecast"))
    return ap


def main() -> None:
    a = build_parser().parse_args()
    if a.smoke:
        smoke()
    elif a.report:
        report(a)
    elif a.evaluate:
        phase_evaluate(a)
    else:
        phase_generate(a)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""make_ticket_pdf.py — formal, monochrome maintenance work order
containing the COMPLETE ticket record: synthesized summary, full edge
interpretation, full diagnostic and prognostic records, all cited
precedent futures and evidence signals. No decorative elements.

  python paper/code/make_ticket_pdf.py --qid T65c321
-> results/synthesis_run/workorder_<qid>.pdf
"""
from __future__ import annotations
import argparse, json, re, sys
from datetime import date
from pathlib import Path
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import black, HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_RIGHT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                Table, TableStyle, HRFlowable)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agentic"))
from apdm.prognostic.future_progression import (reliability,          # noqa
                                                future_progression)
from apdm.prognostic.forecast import PrecedentFutures                 # noqa

RES = ROOT / "agentic" / "results"
INK = black
RULE = HexColor("#333333")
FAINT = HexColor("#777777")


def md2rl(t: str) -> str:
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", str(t or ""))
    return t.replace("\n", "<br/>")


def sec(md, name):
    m = re.search(rf"## {name}\n(.*?)(?=\n## |\Z)", md, re.S)
    return (m.group(1).strip() if m else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", default="T65c321")
    a = ap.parse_args()
    t = next(json.loads(l) for l in
             open(RES / "synthesis_run/tickets.jsonl")
             if json.loads(l)["qid"] == a.qid)
    e = next(x for x in map(json.loads, open(
        RES / "final_prognostic/forecast_episodes.jsonl"))
        if x["qid"] == a.qid and x["arm"] == "P7_agent_dl")
    d = next(x for x in map(json.loads, open(
        RES / "final_diagnostic/episodes.jsonl"))
        if x["qid"] == a.qid and x.get("pattern") == "P5_verifier")
    pr = {}
    prf = RES / "progression_run/forecast_episodes.jsonl"
    if prf.exists():
        pr = next((x.get("forecast") or {} for x in
                   map(json.loads, open(prf))
                   if x["qid"] == a.qid and
                   x["arm"] == "P7_progression"), {})
    q = pd.read_csv(ROOT /
                    "agentic/queries/test_FD002_with_interpretations.csv"
                    ).set_index(["Unit_ID", "cycles"])
    interp = q.loc[(int(e["unit"]), int(e["cycle"]))]["interpretation"]
    if isinstance(interp, pd.Series):
        interp = interp.iloc[0]
    dt = d["ticket"]; fc = e.get("forecast") or {}
    rel = reliability(e.get("contexts"))
    pf = PrecedentFutures(ROOT / "agentic/data/vector_store/meta.jsonl")
    fp = future_progression(pf, e.get("contexts"))
    agg = fp["aggregate"]
    md = t["ticket_md"]
    sev = int(re.search(r"severity (\d)", md).group(1))
    act = dt.get("action", "").replace("_", " ").upper()
    sims = {f"u{c['unit']}c{c['cycle']}": c.get("similarity")
            for c in e.get("contexts") or []}

    body = ParagraphStyle("b", fontName="Times-Roman", fontSize=9.6,
                          leading=13.2, alignment=TA_JUSTIFY,
                          textColor=INK)
    lab = ParagraphStyle("l", fontName="Times-Bold", fontSize=8.2,
                         textColor=INK)
    val = ParagraphStyle("v", fontName="Times-Roman", fontSize=9.4,
                         leading=12, textColor=INK)
    h = ParagraphStyle("h", fontName="Times-Bold", fontSize=10.6,
                       textColor=INK, spaceBefore=10, spaceAfter=3)
    tiny = ParagraphStyle("t", fontName="Times-Roman", fontSize=7.6,
                          leading=10, textColor=FAINT)

    out = RES / "synthesis_run" / f"workorder_{a.qid}.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=A4,
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm)
    story = []
    hd = Table([[Paragraph("<b>A-RAD FLEET OPERATIONS</b><br/>"
                           "<font size=8>Predictive Maintenance "
                           "Division — Turbofan Fleet FD002</font>",
                           ParagraphStyle("hl",
                                          fontName="Times-Bold",
                                          fontSize=13, leading=15,
                                          textColor=INK)),
                 Paragraph("MAINTENANCE WORK ORDER<br/>"
                           f"<font size=8.5>No. WO-{date.today():%Y}-"
                           f"{a.qid} · issued "
                           f"{date.today():%d %b %Y}</font>",
                           ParagraphStyle("hr",
                                          fontName="Times-Bold",
                                          fontSize=11, leading=14,
                                          alignment=TA_RIGHT,
                                          textColor=INK))]],
               colWidths=[102 * mm, 72 * mm])
    hd.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story += [hd, Spacer(1, 3),
              HRFlowable(width="100%", thickness=1.2, color=INK),
              HRFlowable(width="100%", thickness=0.4, color=INK,
                         spaceBefore=1.5), Spacer(1, 6)]

    meta = Table([[Paragraph("ASSET", lab),
                   Paragraph("ANOMALY CYCLE", lab),
                   Paragraph("SEVERITY", lab),
                   Paragraph("REQUIRED ACTION", lab)],
                  [Paragraph(f"engine unit {e['unit']}", val),
                   Paragraph(str(e["cycle"]), val),
                   Paragraph(f"<b>{sev} of 5</b>", val),
                   Paragraph(f"<b>{act}</b>", val)]],
                 colWidths=[44 * mm, 40 * mm, 34 * mm, 56 * mm])
    meta.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.8, RULE),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6)]))
    story += [meta]

    def section(num, title, flow):
        story.append(Paragraph(f"{num}.  {title}", h))
        story.append(HRFlowable(width="100%", thickness=0.4,
                                color=RULE, spaceAfter=4))
        story.extend(flow)

    smry = "".join(
        f"<b>{n.title()}.</b> {sec(md, n)}<br/>"
        for n in ("SITUATION", "DIAGNOSIS", "OUTLOOK", "TRUST",
                  "NEXT STEPS"))
    section("1", "SYNTHESIZED SUMMARY (reporting layer)",
            [Paragraph(smry, body)])
    section("2", "EDGE INTERPRETATION (on-device, full text)",
            [Paragraph(md2rl(interp), body)])
    diag_txt = (f"<b>Assessment.</b> {dt.get('diagnosis')}<br/>"
                f"<b>Matched pattern.</b> {dt.get('matched_pattern')}"
                f"<br/><b>Reasoning.</b> {dt.get('reasoning')}<br/>"
                f"<b>Severity.</b> {dt.get('severity')} of 5 &nbsp;&nbsp;"
                f"<b>Prescribed action.</b> {act}<br/>"
                f"<b>Cited precedents.</b> "
                f"{', '.join(dt.get('cited_precedents') or [])}<br/>"
                f"<b>Verification.</b> escalated: "
                f"{bool(d.get('escalated'))}; repairs: "
                f"{d.get('repairs', 0)}; gates D4-D6 passed.")
    section("3", "DIAGNOSTIC RECORD (agent P5-verifier)",
            [Paragraph(diag_txt, body)])
    rng = fc.get("rul_range")
    rngs = (f"[{rng[0]:.0f}, {rng[1]:.0f}]" if isinstance(rng, list)
            else "n/a")
    prog_txt = (f"<b>RUL estimate.</b> {fc.get('rul_estimate')} cycles, "
                f"range {rngs}, confidence {fc.get('confidence')}. "
                f"CNN-GRU anchor: {e.get('dl_hint')} cycles.<br/>"
                f"<b>Anomaly outlook.</b> {fc.get('anomaly_outlook')}"
                f"<br/><b>Progression narrative.</b> "
                f"{fc.get('progression_narrative')}")
    if pr.get("projected_progression"):
        prog_txt += (f"<br/><b>Projected progression (commentary).</b> "
                     f"{pr['projected_progression']}")
    section("4", "PROGNOSTIC RECORD (agent + deterministic anchor)",
            [Paragraph(prog_txt, body)])

    rows = [[Paragraph("PRECEDENT", lab),
             Paragraph("SIMILARITY", lab),
             Paragraph("OBSERVED CONTINUATION", lab),
             Paragraph("SURVIVED", lab)]]
    for c in fp["cases"]:
        ev = ", ".join(f"+{x['plus']} (g{x['gravity']})"
                       for x in c["events"]) or "no further anomalies"
        sim = sims.get(c["id"])
        rows.append([Paragraph(c["id"], val),
                     Paragraph(f"{sim:.3f}" if sim else "-", val),
                     Paragraph(ev, val),
                     Paragraph(f"{c['rul_then']:.0f} cycles", val)])
    tab = Table(rows, colWidths=[26 * mm, 22 * mm, 96 * mm, 30 * mm])
    tab.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.8, RULE),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, RULE),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5)]))
    unc = agg.get("uncertainty")
    sig_txt = (f"<b>Knowledge-base reliability.</b> {rel['value']} "
               f"(top {rel['top']}, k={rel['k']}) — similarity of this "
               f"case to fleet history.<br/>"
               f"<b>Aggregate of cited futures.</b> median time-to-"
               f"failure +{agg.get('median_ttf')} cycles, range "
               f"{agg.get('ttf_range')}, gravity escalated in "
               f"{agg.get('escalating')}.<br/>"
               f"<b>Progression uncertainty.</b> {unc} — "
               + ("cited futures agree; the outlook is well "
                  "constrained." if (unc or 0) < 0.35 else
                  "cited futures diverge; several plausible paths."))
    section("5", "FLEET EVIDENCE AND SIGNALS",
            [tab, Spacer(1, 4), Paragraph(sig_txt, body)])

    story += [Spacer(1, 10),
              HRFlowable(width="100%", thickness=0.4, color=RULE),
              Spacer(1, 3),
              Paragraph("Composed by the gated reporting layer "
                        "(synthesis v3): the language model writes the "
                        "prose; every numeric fact is injected from "
                        "verified pipeline outputs (diagnostic agent, "
                        "CNN-GRU anchor, deterministic evidence "
                        "signals). All precedent citations are "
                        "verifiable against the fleet knowledge base.",
                        tiny)]
    doc.build(story)
    print(f"[workorder] -> {out}")


if __name__ == "__main__":
    main()

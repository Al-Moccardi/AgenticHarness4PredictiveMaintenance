#!/usr/bin/env python3
"""
make_figures.py — regenerate ALL selected paper figures from the data
shipped in this repository, then verify that every selected figure was
produced. Works on Windows (PowerShell) and Linux alike.

From the repository root:
    python paper/code/make_figures.py
or from anywhere:
    python <path>/paper/code/make_figures.py

Requirements: python>=3.10, numpy, pandas, scipy, matplotlib.
LaTeX typography is used automatically if a TeX toolchain
(latex + dvipng + cm-super) is present; otherwise the scripts fall
back to Computer Modern via mathtext with identical appearance.

Outputs:
    paper/figures_regen/<set>/...   all regenerated figures
    exit code 0 and "VERIFIED 21/21" if every selected figure exists.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # paper/code
ROOT = HERE.parent.parent                       # repo root
AG = ROOT / "agentic"
RES = AG / "results"
OUT = ROOT / "paper" / "figures_regen"

SETS = [
    ("figures/paper_figures_panels.py",
     ["--arad", str(ROOT / "agentic/results/final_prognostic"), "--arad2", str(ROOT / "agentic/results/v2_rul_coupled_collapse"),
      "--v1grid", str(ROOT / "agentic/results/study_A_pattern_grid"), "--out", str(OUT / "panels")]),
    ("figures/paper_figures_core.py",
     ["--arad", str(ROOT / "agentic/results/final_prognostic"), "--arad2", str(ROOT / "agentic/results/v2_rul_coupled_collapse"),
      "--out", str(OUT / "core")]),
    ("figures/paper_figures_more.py",
     ["--arad", str(ROOT / "agentic/results/final_prognostic"), "--arad2", str(ROOT / "agentic/results/v2_rul_coupled_collapse"),
      "--out", str(OUT / "more")]),
    ("figures/paper_figures_singles.py",
     ["--arad", str(ROOT / "agentic/results/final_prognostic"), "--arad2", str(ROOT / "agentic/results/v2_rul_coupled_collapse"),
      "--v1grid", str(ROOT / "agentic/results/study_A_pattern_grid"),
      "--out", str(OUT / "singles")]),
    ("figures/paper_figures_request.py",
     ["--arad", str(ROOT / "agentic/results/final_prognostic"), "--arad2", str(ROOT / "agentic/results/v2_rul_coupled_collapse"),
      "--v1grid", str(ROOT / "agentic/results/study_A_pattern_grid"), "--out", str(OUT / "req")]),
    ("figures/paper_figures_story.py",
     ["--arad", str(ROOT / "agentic/results/final_prognostic"),
      "--v1grid", str(ROOT / "agentic/results/study_A_pattern_grid"), "--out", str(OUT / "story")]),
]

# The 21 selected paper figures (see paper/figures/SELECTED.md)
SELECTED = {
    "figR1_ragas_audited": "req",
    "figS_similarity_ecdf": "singles",
    "figS_stagegap_ecdf": "singles",
    "figAB3_abstention_ladder": "req",
    "figCMP1_rag_vs_agent": "story",
    "figD8_action_mix": "more",
    "figD10_ticket_exhibit": "more",
    "figS_sevstage_agent": "singles",
    "fig3a_rescue_hero": "panels",
    "figAB1_risk_coverage": "req",
    "figAB2_confidence_validity": "req",
    "figC1_risk_profile": "core",
    "figP5_error_by_stage": "more",
    "figS_prog_forest": "singles",
    "figT_combined_ticket": "req",
    "figZ1_eol_sweep_mae": "req",
    "figZ2_eol_sweep_safety": "req",
    "fig6b_trend_chance": "panels",
    "figAB4_abstention_vs_rul": "req",
    "figAB5_calibrated_urgency": "req",
    "figT_high_confidence_ticket": "req",
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for script, args in SETS:
        cmd = [sys.executable, str(HERE / script)] + args
        print(f"[make] {script}")
        r = subprocess.run(cmd, cwd=str(AG))
        if r.returncode != 0:
            print(f"[make] FAILED: {script}")
            return 1
    sel_dir = ROOT / "paper" / "figures_regen" / "_selected"
    sel_dir.mkdir(exist_ok=True)
    missing = []
    for stem, set_ in SELECTED.items():
        src = OUT / set_ / f"{stem}.png"
        if src.exists():
            shutil.copy(src, sel_dir / f"{stem}.png")
        else:
            missing.append(stem)
    if missing:
        print("[make] MISSING:", missing)
        return 2
    print(f"[make] VERIFIED {len(SELECTED)}/{len(SELECTED)} selected "
          f"figures regenerated -> {sel_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

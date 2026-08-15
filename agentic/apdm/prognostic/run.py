#!/usr/bin/env python3
"""Entry point: python -m apdm.prognostic.run [flags]  (forwards to bench_forecast)."""
import os, sys
from pathlib import Path
AG = Path(__file__).resolve().parents[2]
os.chdir(AG); sys.path.insert(0, str(AG))
argv = sys.argv[1:]
if "--out" not in argv:
    argv += ["--out", str(AG / "results" / 'progression_run')]
if "--arms" not in argv and not any(x in argv for x in ("--evaluate","--report","--smoke","--help","-h")):
    argv += ["--arms", 'P7_progression']
sys.argv = ["bench_forecast"] + argv
from apdm.prognostic import bench_forecast as _m
_m.main()

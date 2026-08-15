#!/usr/bin/env python3
"""Entry point: python -m apdm.diagnostic.run [flags]  (forwards to bench_arad)."""
import os, sys
from pathlib import Path
AG = Path(__file__).resolve().parents[2]
os.chdir(AG); sys.path.insert(0, str(AG))
argv = sys.argv[1:]
if "--out" not in argv:
    argv += ["--out", str(AG / "results" / 'diagnostic_rerun')]
sys.argv = ["bench_arad"] + argv
from apdm.diagnostic import bench_arad as _m
_m.main()

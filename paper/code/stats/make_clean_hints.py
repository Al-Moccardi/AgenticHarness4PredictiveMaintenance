"""make_clean_hints.py — build queries/dl_hints_clean.csv for the P3 clean-tool arm.

The dl_only arm in forecast_metrics.csv IS the clean, monotone-corrected
CNN-GRU prediction per case. This writes it in the dl_hints.csv schema so
bench_forecast / bench_arad can consume it via --dl-hints.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
m = pd.read_csv(ROOT / "agentic/results/final_prognostic/forecast_metrics.csv")
d = m[m.arm == "dl_only"][["unit", "cycle", "rul_pred"]].rename(
    columns={"unit": "unit_ID", "rul_pred": "dl_rul"})
d["dl_rul_raw"] = d["dl_rul"]
out = ROOT / "agentic/queries/dl_hints_clean.csv"
d[["unit_ID", "cycle", "dl_rul_raw", "dl_rul"]].to_csv(out, index=False)
print(f"wrote {out} ({len(d)} rows, zeros: {(d.dl_rul == 0).sum()})")

"""Project layout — every module resolves files through here, so the package
works from any working directory once PROJECT_ROOT is found."""
from __future__ import annotations

import os
from pathlib import Path

# arad_edge/paths.py -> project root is the parent of the package dir
PROJECT_ROOT = Path(
    os.environ.get("ARAD_ROOT", Path(__file__).resolve().parent.parent))

PKG = PROJECT_ROOT / "arad_edge"
DATA = PROJECT_ROOT / "data"
MODELS = PROJECT_ROOT / "models"
CONFIG = PROJECT_ROOT / "config"
RESULTS = PROJECT_ROOT / "results"
EXPECTED = PROJECT_ROOT / "expected"

# canonical artifact locations
BUNDLE = MODELS / "edge_bundle.joblib"
CENTROIDS = MODELS / "centroids.json"
TEST_TXT = DATA / "test_FD002.txt"
RUL_TXT = DATA / "RUL_FD002.txt"
KB_CSV = DATA / "anomalies_multimodal.csv"
FEWSHOT = CONFIG / "fewshot_examples.json"


def ensure_results() -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    return RESULTS


def rel(p: Path) -> str:
    try:
        return str(Path(p).relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)

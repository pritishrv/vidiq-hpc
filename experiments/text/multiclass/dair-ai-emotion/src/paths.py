from __future__ import annotations

from pathlib import Path

from io_utils import ensure_dir


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = EXPERIMENT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = EXPERIMENT_ROOT / "data" / "processed"
CONFIDENCE_DIR = DATA_PROCESSED_DIR / "confidence_scores"
ARTIFACTS_DIR = EXPERIMENT_ROOT / "artifacts"
EMBEDDINGS_DIR = ARTIFACTS_DIR / "embeddings"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
LOGS_DIR = ARTIFACTS_DIR / "logs"
RUNS_DIR = EXPERIMENT_ROOT / "runs"


def ensure_base_dirs() -> None:
    for path in [
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        CONFIDENCE_DIR,
        EMBEDDINGS_DIR,
        METRICS_DIR,
        PLOTS_DIR,
        LOGS_DIR,
        RUNS_DIR,
    ]:
        ensure_dir(path)

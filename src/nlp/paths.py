"""Rutas del proyecto para notebooks y scripts reproducibles."""

import os
from pathlib import Path

RANDOM_STATE = 42

# Activar con NLP_DEV_MODE=1: grilla reducida y muestras más chicas en transformers.
DEV_MODE = os.environ.get("NLP_DEV_MODE", "0") == "1"

# Subconjunto político principal
POLITICS_REAL_SUBJECTS = ["politicsNews"]
POLITICS_FAKE_SUBJECTS = ["politics"]
POLITICS_FAKE_OPTIONAL = ["Government News", "left-news"]


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    msg = "No se encontró pyproject.toml"
    raise FileNotFoundError(msg)


PROJECT_ROOT = _find_project_root()
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EMBEDDINGS = PROJECT_ROOT / "data" / "embeddings"
RESULTS_FIGURES = PROJECT_ROOT / "results" / "figures"
RESULTS_METRICS = PROJECT_ROOT / "results" / "metrics"
RESULTS_MODELS = PROJECT_ROOT / "results" / "models"
RESULTS_ERROR = PROJECT_ROOT / "results" / "error_analysis"

# Ablación de fuente (Experimento 1): caída de F2 en val que activa [SOURCE].
SOURCE_ABLATION_F2_DROP_THRESHOLD = 0.03
SOURCE_ABLATION_DECISION = RESULTS_METRICS / "source_ablation_decision.json"


def linguistic_features_cache_path(
    prefix: str,
    field: str,
    split: str,
    *,
    normalize_source: bool = False,
) -> Path:
    """Ruta Parquet para cache de features lingüísticas (Experimento 2)."""
    suffix = "_norm" if normalize_source else ""
    return (
        DATA_PROCESSED / f"linguistic_features_{prefix}_{field}_{split}{suffix}.parquet"
    )


def ensure_output_dirs() -> None:
    """Crea directorios de salida si no existen."""
    for path in (
        DATA_PROCESSED,
        DATA_EMBEDDINGS,
        RESULTS_FIGURES,
        RESULTS_METRICS,
        RESULTS_MODELS,
        RESULTS_ERROR,
    ):
        path.mkdir(parents=True, exist_ok=True)

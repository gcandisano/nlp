"""Rutas del proyecto para notebooks y scripts reproducibles."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_FIGURES = PROJECT_ROOT / "results" / "figures"
RESULTS_METRICS = PROJECT_ROOT / "results" / "metrics"
RESULTS_MODELS = PROJECT_ROOT / "results" / "models"
RESULTS_ERROR = PROJECT_ROOT / "results" / "error_analysis"

RANDOM_STATE = 42

# Subconjunto político principal
POLITICS_REAL_SUBJECTS = ["politicsNews"]
POLITICS_FAKE_SUBJECTS = ["politics"]
POLITICS_FAKE_OPTIONAL = ["Government News", "left-news"]

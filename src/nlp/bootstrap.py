"""Preparación inicial del proyecto."""
from nlp.paths import DATA_RAW, ensure_output_dirs
from nlp.setup import prepare_environment

KAGGLE_URL = (
    "https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/"
)


def main() -> None:
    missing = [name for name in ("Fake.csv", "True.csv") if not (DATA_RAW / name).exists()]
    if missing:
        names = ", ".join(missing)
        msg = (
            f"Faltan archivos en data/raw/: {names}. "
            f"Descargar dataset de Kaggle: {KAGGLE_URL}"
        )
        raise FileNotFoundError(msg)

    ensure_output_dirs()
    prepare_environment()
    print("Listo. Ejecutar notebooks 01 → 06 con: uv run jupyter lab")

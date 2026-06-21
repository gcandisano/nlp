"""Utilidades para el análisis manual de errores (Experimento 5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MIN_ERROR_SAMPLE = 30

ERROR_CATEGORIES = [
    "lenguaje_neutral_en_fake",
    "titulo_ambiguo",
    "ironia_sarcasmo",
    "informacion_parcialmente_verdadera",
    "sesgo_fuente",
    "tema_politico_cargado",
    "otro",
]

ANNOTATION_COLUMNS = [
    "news_id",
    "error_type",
    "title",
    "text_fragment",
    "true_label",
    "prediction",
    "confidence",
    "category",
    "comment",
]


def model_confidence(pipe, X) -> np.ndarray:
    """Confianza del modelo en su predicción (mayor = más seguro)."""
    if hasattr(pipe, "decision_function"):
        return np.abs(pipe.decision_function(X))

    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)
        pred = pipe.predict(X)
        return np.array([proba[i, pred[i]] for i in range(len(pred))])

    return np.full(len(X), np.nan)


def confidence_from_proba(y_pred: np.ndarray, proba_fake: np.ndarray) -> np.ndarray:
    """Confianza a partir de probabilidad de la clase fake (1)."""
    return np.where(y_pred == 1, proba_fake, 1.0 - proba_fake)


def build_error_frame(
    df: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    *,
    text_fragment_len: int = 300,
) -> pd.DataFrame:
    """Arma DataFrame de FP/FN con fragmento de texto y confianza."""
    work = df.copy()
    work["true_label"] = np.asarray(y_true)
    work["prediction"] = y_pred
    work["confidence"] = scores

    work["error_type"] = np.where(
        (work["true_label"] == 0) & (work["prediction"] == 1),
        "FP",
        np.where(
            (work["true_label"] == 1) & (work["prediction"] == 0),
            "FN",
            "correct",
        ),
    )

    errors = work[work["error_type"].isin(["FP", "FN"])].copy()
    errors["text_fragment"] = errors["text"].astype(str).str[:text_fragment_len]
    return errors.sort_values("confidence", ascending=False)


def write_blank_annotation_template(errors: pd.DataFrame, path: Path) -> Path:
    """Escribe plantilla CSV para anotación manual. No sobrescribe si ya existe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f"Plantilla existente conservada: {path}")
        return path

    template = errors.reset_index().rename(columns={"index": "news_id"})
    export = template[
        [
            "news_id",
            "error_type",
            "title",
            "text_fragment",
            "true_label",
            "prediction",
            "confidence",
        ]
    ].copy()
    export["category"] = ""
    export["comment"] = ""
    export = export[ANNOTATION_COLUMNS]
    export.to_csv(path, index=False)
    print(f"Plantilla creada: {path} ({len(export)} filas)")
    return path


def load_annotations(path: Path) -> pd.DataFrame:
    """Lee anotaciones manuales y valida categorías no vacías."""
    if not path.exists():
        msg = f"No se encontró el archivo de anotaciones: {path}"
        raise FileNotFoundError(msg)

    df = pd.read_csv(path)
    missing_cols = set(ANNOTATION_COLUMNS) - set(df.columns)
    if missing_cols:
        msg = f"Columnas faltantes en {path}: {sorted(missing_cols)}"
        raise ValueError(msg)

    filled = df["category"].fillna("").astype(str).str.strip()
    if (filled != "").any():
        invalid = filled[(filled != "") & ~filled.isin(ERROR_CATEGORIES)]
        if not invalid.empty:
            bad = sorted(invalid.unique())
            msg = f"Categorías inválidas en {path}: {bad}"
            raise ValueError(msg)

    return df


def annotations_complete(df: pd.DataFrame) -> bool:
    """Indica si todas las filas tienen categoría válida de la taxonomía."""
    if df.empty:
        return False

    cat = df["category"].fillna("").astype(str).str.strip()
    if (cat == "").any():
        return False
    return cat.isin(ERROR_CATEGORIES).all()


def category_distribution(annotations: pd.DataFrame) -> pd.DataFrame:
    """Cuenta categorías en el orden definido por la taxonomía."""
    counts = annotations["category"].value_counts()
    rows = [
        {"category": cat, "count": int(counts.get(cat, 0))}
        for cat in ERROR_CATEGORIES
        if counts.get(cat, 0) > 0
    ]
    return pd.DataFrame(rows)


def warn_if_small_sample(n_errors: int, *, min_sample: int = MIN_ERROR_SAMPLE) -> None:
    """Advierte si la muestra disponible es menor que el mínimo metodológico."""
    n_fp_fn = n_errors
    print(f"Total errores en test: {n_fp_fn}")
    if n_fp_fn < min_sample:
        print(
            f"ADVERTENCIA: la metodología sugiere ≥{min_sample} errores, pero el modelo solo "
            f"comete {n_fp_fn} en test. Se analizan todos los casos disponibles; "
            "esto limita la generalización cualitativa."
        )

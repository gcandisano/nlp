"""Métricas de evaluación para clasificación fake/real."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from nlp.paths import RESULTS_METRICS


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Calcula métricas obligatorias del TP. Clase positiva = fake (1)."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_fake": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_fake": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_fake": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f2_fake": fbeta_score(y_true, y_pred, beta=2, pos_label=1, zero_division=0),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["roc_auc"] = np.nan
    else:
        metrics["roc_auc"] = np.nan
    return metrics


def metrics_to_row(
    metrics: dict,
    extra: dict | None = None,
) -> dict:
    row = dict(metrics)
    if extra:
        row.update(extra)
    return row


def _select_test_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Se queda con las filas de test cuando existe la columna split."""
    if "split" in df.columns:
        test_rows = df[df["split"] == "test"]
        return test_rows if not test_rows.empty else df
    return df


def _drop_dev_rows(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Descarta filas de corridas DEV (sample_frac < 1.0) para no contaminar la tabla.

    El Experimento 3 puede entrenar DistilBERT con el 10% del train (NLP_DEV_MODE);
    esas filas no son comparables con baselines de datos completos y se excluyen.
    """
    if "sample_frac" in df.columns:
        dev_mask = df["sample_frac"].fillna(1.0) < 1.0
        if dev_mask.any():
            print(
                f"[consolidate_results] Aviso: se descartan {int(dev_mask.sum())} "
                f"fila(s) de '{source}' con sample_frac<1.0 (corrida DEV); no entran "
                "a la tabla comparativa final."
            )
        return df[~dev_mask]
    return df


def consolidate_results(
    baseline_path=None,
    embedding_path=None,
    transformer_path=None,
    linguistic_path=None,
    output_path=None,
) -> pd.DataFrame:
    """Combina resultados de todos los experimentos en una tabla comparativa.

    Solo entran filas de test y de corridas con datos completos (sample_frac=1.0).
    El subconjunto político es el corpus de referencia y se ordena primero;
    full_dataset (control de sensibilidad) queda debajo y no compite por "mejor modelo".
    """
    specs = [
        (baseline_path, RESULTS_METRICS / "baseline_results.csv", "baseline"),
        (
            linguistic_path,
            RESULTS_METRICS / "linguistic_features_results.csv",
            "linguistic",
        ),
        (embedding_path, RESULTS_METRICS / "embedding_results.csv", "embedding"),
        (transformer_path, RESULTS_METRICS / "transformer_results.csv", "transformer"),
    ]

    frames = []
    for given, default, name in specs:
        path = given or default
        if path.exists():
            df = _select_test_rows(_drop_dev_rows(pd.read_csv(path), name))
            if not df.empty:
                frames.append(df)

    if not frames:
        return pd.DataFrame()

    all_results = pd.concat(frames, ignore_index=True)
    if "dataset_scope" not in all_results.columns:
        all_results["dataset_scope"] = "politics"

    # politics (referencia) primero; cualquier otro scope (full_dataset) después.
    all_results["_scope_rank"] = (all_results["dataset_scope"] != "politics").astype(
        int
    )
    all_results = (
        all_results.sort_values(["_scope_rank", "f2_fake"], ascending=[True, False])
        .drop(columns="_scope_rank")
        .reset_index(drop=True)
    )

    out = output_path or (RESULTS_METRICS / "all_model_results.csv")
    all_results.to_csv(out, index=False)
    return all_results

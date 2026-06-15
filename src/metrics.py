"""Métricas de evaluación para clasificación fake/real."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


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


def consolidate_results(
    baseline_path,
    embedding_path=None,
    transformer_path=None,
    output_path=None,
) -> pd.DataFrame:
    """Combina resultados de todos los experimentos."""
    from src.paths import RESULTS_METRICS

    frames = []
    baseline_path = baseline_path or (RESULTS_METRICS / "baseline_results.csv")
    if baseline_path.exists():
        baseline_df = pd.read_csv(baseline_path)
        if "split" in baseline_df.columns:
            test_rows = baseline_df[baseline_df["split"] == "test"]
            frames.append(test_rows if not test_rows.empty else baseline_df)
        else:
            frames.append(baseline_df)

    if embedding_path is None:
        embedding_path = RESULTS_METRICS / "embedding_results.csv"
    if embedding_path.exists():
        frames.append(pd.read_csv(embedding_path))

    if transformer_path is None:
        transformer_path = RESULTS_METRICS / "transformer_results.csv"
    if transformer_path.exists():
        frames.append(pd.read_csv(transformer_path))

    if not frames:
        return pd.DataFrame()

    all_results = pd.concat(frames, ignore_index=True)
    if "dataset_scope" not in all_results.columns:
        all_results["dataset_scope"] = "politics"

    sort_cols = ["f2_fake"]
    if "dataset_scope" in all_results.columns:
        sort_cols = ["dataset_scope", "f2_fake"]
    all_results = all_results.sort_values(sort_cols, ascending=[True, False]).reset_index(drop=True)

    out = output_path or (RESULTS_METRICS / "all_model_results.csv")
    all_results.to_csv(out, index=False)
    return all_results

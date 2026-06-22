"""Carga y guardado eficiente de splits procesados."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nlp.paths import DATA_PROCESSED


def _split_path(prefix: str, split: str, ext: str) -> Path:
    return DATA_PROCESSED / f"{prefix}_{split}.{ext}"


def save_split(df: pd.DataFrame, prefix: str, split: str) -> None:
    """Persiste un split en Parquet y CSV."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_split_path(prefix, split, "parquet"), index=False)
    df.to_csv(_split_path(prefix, split, "csv"), index=False)


def load_split(
    prefix: str, split: str, columns: list[str] | None = None
) -> pd.DataFrame:
    """Lee un split; prefiere Parquet si existe.

    Si se piden columnas, se devuelven en el orden solicitado en ambos backends
    (Parquet y CSV), para que el resultado no dependa del formato disponible.
    """
    parquet_path = _split_path(prefix, split, "parquet")
    csv_path = _split_path(prefix, split, "csv")

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path, columns=columns)
    elif csv_path.exists():
        df = pd.read_csv(csv_path, usecols=columns)
    else:
        msg = f"No se encontró split: {prefix}_{split}"
        raise FileNotFoundError(msg)

    if columns is not None:
        df = df[columns]
    return df


def load_splits(
    prefix: str,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga train, val y test con columnas opcionales."""
    train = load_split(prefix, "train", columns)
    val = load_split(prefix, "val", columns)
    test = load_split(prefix, "test", columns)
    return train, val, test

"""Carga, entrenamiento y promediado de embeddings con cache en disco."""

from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors, Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm.auto import tqdm

from nlp.metrics import compute_metrics, metrics_to_row
from nlp.paths import DATA_EMBEDDINGS, RANDOM_STATE
from nlp.preprocessing import normalize_source_markers

GLOVE_URL = "http://nlp.stanford.edu/data/glove.6B.zip"
DEFAULT_C_GRID: tuple[float, ...] = (0.1, 1.0, 10.0)


def embedding_text_series(
    df: pd.DataFrame,
    text_col: str,
    *,
    normalize_source: bool = False,
) -> pd.Series:
    """Serie de texto para embeddings, con normalización de fuente opcional."""
    series = df[text_col].fillna("")
    if normalize_source:
        return series.map(normalize_source_markers)
    return series


def _glove_txt_path(dim: int) -> Path:
    return DATA_EMBEDDINGS / f"glove.6B.{dim}d.txt"


def _glove_kv_path(dim: int) -> Path:
    return DATA_EMBEDDINGS / f"glove.6B.{dim}d.kv"


def _ensure_glove_txt(dim: int) -> Path:
    DATA_EMBEDDINGS.mkdir(parents=True, exist_ok=True)
    glove_file = _glove_txt_path(dim)
    if glove_file.exists():
        return glove_file

    zip_path = DATA_EMBEDDINGS / "glove.6B.zip"
    if not zip_path.exists():
        print("Descargando GloVe (puede tardar varios minutos)...")
        urllib.request.urlretrieve(GLOVE_URL, zip_path)
    print("Extrayendo GloVe...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract(f"glove.6B.{dim}d.txt", DATA_EMBEDDINGS)
    return glove_file


def load_glove_vectors(dim: int = 100) -> KeyedVectors:
    """Carga GloVe desde cache gensim o lo crea desde el archivo de texto."""
    kv_path = _glove_kv_path(dim)
    if kv_path.exists():
        print(f"Cargando GloVe cacheado: {kv_path.name}")
        return KeyedVectors.load(str(kv_path), mmap="r")

    glove_file = _ensure_glove_txt(dim)
    print(f"Parseando GloVe {dim}d (solo la primera vez)...")
    vectors = KeyedVectors.load_word2vec_format(
        str(glove_file), binary=False, no_header=True
    )
    vectors.save(str(kv_path))
    print(f"GloVe cacheado en {kv_path.name}")
    return vectors


def train_or_load_word2vec(
    texts,
    cache_path: Path,
    *,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 5,
    workers: int = 4,
    epochs: int = 10,
    seed: int = RANDOM_STATE,
) -> Word2Vec:
    """Entrena Word2Vec sobre el corpus o carga el modelo cacheado."""
    if cache_path.exists():
        print(f"Cargando Word2Vec cacheado: {cache_path.name}")
        return Word2Vec.load(str(cache_path))

    print(f"Entrenando Word2Vec -> {cache_path.name}")
    tokenized = [str(t).lower().split() for t in texts]
    model = Word2Vec(
        sentences=tokenized,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        seed=seed,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(cache_path))
    return model


def text_to_embedding(text: str, vectors: KeyedVectors, dim: int = 100) -> np.ndarray:
    tokens = str(text).lower().split()
    vecs = [vectors[t] for t in tokens if t in vectors]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)


def embed_corpus(texts, vectors: KeyedVectors, dim: int = 100) -> np.ndarray:
    return np.vstack(
        [text_to_embedding(t, vectors, dim) for t in tqdm(texts, desc="Embedding")]
    )


def _texts_fingerprint(texts, dim: int, *, tag: str = "") -> str:
    """Huella estable del contenido a embeber (texto + dim + etiqueta)."""
    h = hashlib.sha256()
    h.update(tag.encode("utf-8"))
    h.update(str(dim).encode("utf-8"))
    for t in texts:
        h.update(b"\x00")
        h.update(str(t).encode("utf-8"))
    return h.hexdigest()


def load_or_compute_document_embeddings(
    texts,
    cache_path: Path,
    vectors: KeyedVectors,
    dim: int = 100,
    *,
    tag: str = "",
) -> np.ndarray:
    """Devuelve embeddings de documento desde cache .npz o los calcula."""
    fingerprint = _texts_fingerprint(texts, dim, tag=tag)
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        cached_fp = cached["fingerprint"].item() if "fingerprint" in cached else None
        if (
            cached_fp == fingerprint
            and cached["embeddings"].shape[0] == len(texts)
            and cached["embeddings"].shape[1] == dim
        ):
            print(f"Embeddings cargados desde cache: {cache_path.name}")
            return cached["embeddings"]

    print(f"Calculando embeddings -> {cache_path.name}")
    embeddings = embed_corpus(texts, vectors, dim)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path, embeddings=embeddings, fingerprint=fingerprint, dim=dim
    )
    return embeddings


def load_or_compute_glove_embeddings(
    texts,
    cache_path: Path,
    vectors: KeyedVectors,
    dim: int = 100,
) -> np.ndarray:
    """Alias retrocompatible para embeddings GloVe."""
    return load_or_compute_document_embeddings(
        texts, cache_path, vectors, dim, tag="glove"
    )


def _proba_from_pipeline(pipe: Pipeline, X: np.ndarray) -> np.ndarray | None:
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(X)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    return None


def tune_and_evaluate_dense_classifiers(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_val: np.ndarray,
    y_val: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series,
    *,
    representation: str,
    dataset_scope: str = "politics",
    use_source_normalization: bool = False,
    c_grid: tuple[float, ...] = DEFAULT_C_GRID,
) -> list[dict]:
    """Ajusta LR y LinearSVC con StandardScaler; selecciona C por F2 en val."""
    results: list[dict] = []
    classifiers = [
        ("logistic_regression", LogisticRegression),
        ("linear_svm", LinearSVC),
    ]

    for model_name, model_cls in classifiers:
        best_f2, best_pipe, best_param = -1.0, None, None
        for c in c_grid:
            if model_name == "logistic_regression":
                clf = model_cls(C=c, random_state=RANDOM_STATE, max_iter=2000)
            else:
                clf = model_cls(C=c, random_state=RANDOM_STATE)
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
            pipe.fit(X_train, y_train)
            y_val_pred = pipe.predict(X_val)
            val_metrics = compute_metrics(y_val, y_val_pred)
            if val_metrics["f2_fake"] > best_f2:
                best_f2 = val_metrics["f2_fake"]
                best_pipe = pipe
                best_param = c

        y_test_pred = best_pipe.predict(X_test)
        y_proba = _proba_from_pipeline(best_pipe, X_test)
        test_metrics = compute_metrics(y_test, y_test_pred, y_proba)
        results.append(
            metrics_to_row(
                test_metrics,
                {
                    "model": model_name,
                    "representation": representation,
                    "best_param": best_param,
                    "dataset_scope": dataset_scope,
                    "split": "test",
                    "use_source_normalization": use_source_normalization,
                },
            )
        )

    return results

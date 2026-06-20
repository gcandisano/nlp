"""Carga y promediado de embeddings GloVe con cache en disco."""

from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors
from tqdm.auto import tqdm

from nlp.paths import DATA_EMBEDDINGS

GLOVE_URL = "http://nlp.stanford.edu/data/glove.6B.zip"


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


def _texts_fingerprint(texts, dim: int) -> str:
    """Huella estable del contenido a embeber (texto + dim)."""
    h = hashlib.sha256()
    h.update(str(dim).encode("utf-8"))
    for t in texts:
        h.update(b"\x00")
        h.update(str(t).encode("utf-8"))
    return h.hexdigest()


def load_or_compute_glove_embeddings(
    texts,
    cache_path: Path,
    vectors: KeyedVectors,
    dim: int = 100,
) -> np.ndarray:
    """Devuelve embeddings desde cache .npz o los calcula y persiste.

    La cache se valida por huella del contenido (texto + dim), no solo por
    cantidad de filas, para no reutilizar vectores de un preprocesamiento
    anterior o de otro campo de texto cuando el conteo de filas coincide.
    """
    fingerprint = _texts_fingerprint(texts, dim)
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

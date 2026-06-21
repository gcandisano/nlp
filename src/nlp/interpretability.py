"""Análisis de importancia de atributos (Experimento 4)."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from nlp.modeling import (
    MODEL_CONFIGS,
    config_from_row,
    fit_pipeline_from_config,
    get_linear_feature_weights,
    get_text_col,
)
from nlp.paths import RANDOM_STATE
from nlp.preprocessing import normalize_source_markers

LINEAR_MODELS = tuple(MODEL_CONFIGS.keys())


def load_baseline_config(config_path: Path) -> dict:
    """Lee la configuración del mejor baseline guardada en JSON."""
    cfg = pd.read_json(config_path, typ="series")
    return {
        "model": cfg["model"],
        "vectorizer": cfg["vectorizer"],
        "text_field": cfg["text_field"],
        "stopwords": cfg["stopwords"],
        "ngram_range": eval(cfg["ngram_range"]),
        "max_features": int(cfg["max_features"]),
        "best_param": cfg["best_param"],
    }


def load_best_baseline_pipeline(
    model_path: Path,
    config_path: Path,
    baseline_results: pd.DataFrame,
    train: pd.DataFrame,
    *,
    dataset_scope: str = "politics",
    normalize_source: bool = False,
) -> tuple[Pipeline, dict, str]:
    """Carga el pipeline guardado o refitea la mejor config del subconjunto político."""
    config: dict | None = None
    if config_path.is_file():
        config = load_baseline_config(config_path)

    pipe: Pipeline | None = None
    if model_path.is_file():
        loaded = joblib.load(model_path)
        clf = loaded.named_steps["clf"]
        if hasattr(clf, "coef_") or hasattr(clf, "feature_log_prob_"):
            pipe = loaded

    if pipe is None:
        best_row = _best_overall_row(baseline_results, dataset_scope)
        config = config_from_row(best_row)
        return fit_pipeline_from_config(
            train, config, normalize_source=normalize_source
        )

    if config is None:
        msg = f"No se encontró configuración en {config_path}"
        raise FileNotFoundError(msg)

    text_col = get_text_col(config["text_field"], config["stopwords"])
    return pipe, config, text_col


def _validation_results(
    baseline_results: pd.DataFrame,
    dataset_scope: str,
) -> pd.DataFrame:
    return baseline_results[
        (baseline_results["dataset_scope"] == dataset_scope)
        & (baseline_results["split"] == "val")
    ]


def _best_overall_row(baseline_results: pd.DataFrame, dataset_scope: str) -> pd.Series:
    val = _validation_results(baseline_results, dataset_scope)
    return val.nlargest(1, "f2_fake").iloc[0]


def best_config_rows_per_model(
    baseline_results: pd.DataFrame,
    dataset_scope: str = "politics",
    *,
    force_text_field: str | None = None,
    force_vectorizer: str | None = None,
) -> pd.DataFrame:
    """Mejor fila de validación por tipo de modelo lineal.

    ``force_text_field`` / ``force_vectorizer`` fijan el campo de texto y/o el
    vectorizador comunes a todos los modelos. Comparar la consistencia de términos
    entre LR/SVM/NB exige el MISMO vocabulario: sin fijarlo, NB puede ganar en
    'title' y LR/SVM en 'body', y el solapamiento Jaccard da 0.0 por comparar campos
    distintos, no por inconsistencia algorítmica real.
    """
    val = _validation_results(baseline_results, dataset_scope)
    if force_text_field is not None:
        val = val[val["text_field"] == force_text_field]
    if force_vectorizer is not None:
        val = val[val["vectorizer"] == force_vectorizer]
    idx = val.groupby("model")["f2_fake"].idxmax()
    return val.loc[idx].sort_values("f2_fake", ascending=False)


def feature_importance_dataframe(pipe: Pipeline) -> pd.DataFrame:
    """DataFrame ordenado por |coeficiente| con un término por fila."""
    names = pipe.named_steps["vec"].get_feature_names_out()
    coefs = get_linear_feature_weights(pipe)
    importance = pd.DataFrame({"term": names, "coefficient": coefs})
    importance["abs_coef"] = importance["coefficient"].abs()
    return importance.sort_values("abs_coef", ascending=False)


def top_terms_table(
    importance: pd.DataFrame,
    *,
    n: int = 30,
    model: str | None = None,
) -> pd.DataFrame:
    """Top términos positivos (fake) y negativos (real)."""
    top_fake = importance.nlargest(n, "coefficient")[["term", "coefficient"]]
    top_real = importance.nsmallest(n, "coefficient")[["term", "coefficient"]]
    out = pd.concat(
        [
            top_fake.assign(direction="fake"),
            top_real.assign(direction="real"),
        ],
        ignore_index=True,
    )
    if model is not None:
        out["model"] = model
    return out


def term_overlap_summary(
    terms_by_model: Mapping[str, set[str]],
) -> pd.DataFrame:
    """Solapamiento Jaccard entre conjuntos de términos por modelo."""
    models = list(terms_by_model)
    rows: list[dict] = []
    for i, model_a in enumerate(models):
        for model_b in models[i:]:
            left = terms_by_model[model_a]
            right = terms_by_model[model_b]
            union = left | right
            overlap = len(left & right)
            jaccard = overlap / len(union) if union else 0.0
            rows.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "overlap": overlap,
                    "jaccard": jaccard,
                }
            )
    return pd.DataFrame(rows)


def prepare_text_series(
    series: pd.Series,
    *,
    normalize_source: bool = False,
) -> pd.Series:
    """Texto listo para análisis descriptivo (adjetivos, bigramas frecuentes)."""
    prepared = series.fillna("").astype(str)
    if normalize_source:
        return prepared.map(normalize_source_markers)
    return prepared


def fit_bigram_interpretation_pipeline(
    train: pd.DataFrame,
    config: dict,
    *,
    normalize_source: bool = False,
) -> tuple[Pipeline, str]:
    """Refitea con ngram_range=(1, 2) si hace falta para leer bigramas predictivos."""
    bigram_config = dict(config)
    if config["ngram_range"] != (1, 2):
        bigram_config["ngram_range"] = (1, 2)
    return fit_pipeline_from_config(
        train, bigram_config, normalize_source=normalize_source
    )


def top_bigrams_table(
    importance: pd.DataFrame,
    *,
    n: int = 20,
    model: str | None = None,
) -> pd.DataFrame:
    """Bigramas con mayor peso (términos que contienen espacio)."""
    bigrams = importance[importance["term"].str.contains(" ", regex=False)].copy()
    top_fake = bigrams.nlargest(n, "coefficient")[["term", "coefficient"]]
    top_real = bigrams.nsmallest(n, "coefficient")[["term", "coefficient"]]
    out = pd.concat(
        [
            top_fake.assign(direction="fake"),
            top_real.assign(direction="real"),
        ],
        ignore_index=True,
    )
    if model is not None:
        out["model"] = model
    return out


def top_ngrams_by_class_frequency(
    df: pd.DataFrame,
    text_col: str,
    label: int,
    *,
    n: int = 20,
    ngram_range: tuple[int, int] = (2, 2),
    normalize_source: bool = False,
) -> list[tuple[str, int]]:
    """Bigramas más frecuentes por clase (análisis descriptivo, no predictivo)."""
    texts = prepare_text_series(
        df.loc[df["label"] == label, text_col],
        normalize_source=normalize_source,
    )
    vec = CountVectorizer(ngram_range=ngram_range, token_pattern=r"(?u)\b\w+\b")
    matrix = vec.fit_transform(texts)
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    features = vec.get_feature_names_out()
    ranked = sorted(
        zip(features, counts, strict=False),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked[:n]


def extract_adjective_counts(
    nlp: spacy.Language,
    texts: pd.Series,
    *,
    sample_size: int = 3000,
) -> Counter[str]:
    """Cuenta lemas adjetivos en una muestra aleatoria de textos."""
    adj_counter: Counter[str] = Counter()
    # Muestreo aleatorio con semilla fija. Con .head() se tomaban los textos más
    # antiguos (el split es temporal y train va primero), sesgando la muestra a
    # una época; .sample(random_state=...) la hace representativa y reproducible.
    if len(texts) > sample_size:
        sample = texts.sample(n=sample_size, random_state=RANDOM_STATE).astype(str)
    else:
        sample = texts.astype(str)
    for doc in nlp.pipe(sample, batch_size=256):
        for token in doc:
            if token.pos_ == "ADJ":
                adj_counter[token.lemma_.lower()] += 1
    return adj_counter


def adjective_sentiment_table(
    adj_counter: Counter[str],
    *,
    class_label: str,
    top_n: int = 50,
    vader: SentimentIntensityAnalyzer | None = None,
) -> pd.DataFrame:
    """Tabla de adjetivos frecuentes con su valencia léxica por clase.

    Usa el score por palabra del lexicon de VADER (``vader.lexicon``, rango ~-4..+4),
    NO ``polarity_scores(adj)``: VADER es un analizador a nivel oración cuyo compound
    depende de mayúsculas, signos y modificadores — todos ausentes en un adjetivo
    aislado—, por lo que devolvía 0.0 para la mayoría de los lemas y la métrica de
    "carga emocional" quedaba dominada por unas pocas palabras. La columna
    ``in_lexicon`` indica si el adjetivo tiene entrada en el lexicon (cobertura).
    """
    vader = vader or SentimentIntensityAnalyzer()
    rows: list[dict] = []
    for adj, count in adj_counter.most_common(top_n):
        valence = float(vader.lexicon.get(adj, 0.0))
        rows.append(
            {
                "adjective": adj,
                "count": count,
                "valence": valence,
                "valence_abs": abs(valence),
                "in_lexicon": adj in vader.lexicon,
                "class": class_label,
            }
        )
    return pd.DataFrame(rows)


def adjective_sentiment_summary(adj_df: pd.DataFrame) -> pd.DataFrame:
    """Resumen de carga emocional por clase: valencia ponderada y cobertura léxica."""
    rows: list[dict] = []
    for class_label, group in adj_df.groupby("class"):
        total = group["count"].sum()
        weighted_abs = (group["valence_abs"] * group["count"]).sum() / total
        weighted_valence = (group["valence"] * group["count"]).sum() / total
        with_valence = group[group["valence_abs"] > 0]
        rows.append(
            {
                "class": class_label,
                "n_adjectives": len(group),
                "n_con_valencia": int(len(with_valence)),
                "total_count": int(total),
                "mean_valence_abs_weighted": weighted_abs,
                "mean_valence_weighted": weighted_valence,
            }
        )
    return pd.DataFrame(rows)


def load_source_normalization_flag(path: Path) -> bool:
    """Indica si los análisis descriptivos deben normalizar marcadores de fuente."""
    with path.open(encoding="utf-8") as handle:
        decision = json.load(handle)
    return bool(decision.get("use_source_normalization", False))

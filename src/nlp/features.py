"""Extracción de features lingüísticas interpretables (Experimento 2)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import spacy
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from nlp.metrics import compute_metrics
from nlp.paths import (
    RANDOM_STATE,
    SOURCE_ABLATION_DECISION,
    linguistic_features_cache_path,
)
from nlp.preprocessing import normalize_source_markers, replace_urls

FEATURE_NAMES = [
    "ratio_exclamacion",
    "ratio_mayusculas",
    "long_oracion_prom",
    "ratio_adj_sust",
    "sentimiento_vader",
    "densidad_ner",
    "freq_url",
    "freq_pronombres",
]

TEXT_FIELDS = ["title_text", "body_text", "full_text"]

DEFAULT_C_GRID = (0.1, 1.0, 10.0)


def prepare_text(text: str, *, normalize_source: bool = False) -> str:
    """Prepara texto crudo: URLs a [URL] y opcional normalización de fuente."""
    if pd.isna(text) or text is None:
        return ""
    prepared = replace_urls(str(text))
    if normalize_source:
        prepared = normalize_source_markers(prepared)
    return prepared


def _empty_features() -> dict[str, float]:
    return dict.fromkeys(FEATURE_NAMES, 0.0)


def _count_allcaps_words(text: str) -> tuple[int, int]:
    words = text.split()
    if not words:
        return 0, 0
    allcaps = sum(1 for w in words if w.isalpha() and w.isupper() and len(w) > 1)
    return allcaps, len(words)


def _is_first_or_second_person_pronoun(token) -> bool:
    if token.pos_ != "PRON":
        return False
    person = token.morph.get("Person")
    return "1" in person or "2" in person


def extract_features_from_doc(
    prepared_text: str,
    doc,
    vader: SentimentIntensityAnalyzer,
) -> dict[str, float]:
    """Calcula las 8 features a partir de un doc spaCy ya procesado."""
    if not prepared_text.strip():
        return _empty_features()

    n_sentences = max(len(list(doc.sents)), 1)
    n_exclam = prepared_text.count("!")
    allcaps, n_words = _count_allcaps_words(prepared_text)

    n_tokens = len(doc)
    n_adj = sum(1 for t in doc if t.pos_ == "ADJ")
    n_nouns = sum(1 for t in doc if t.pos_ in {"NOUN", "PROPN"})
    n_entities = len(doc.ents)
    n_pronouns = sum(1 for t in doc if _is_first_or_second_person_pronoun(t))
    n_urls = prepared_text.count("[URL]")

    sent_lengths = [len(sent) for sent in doc.sents]
    long_oracion_prom = sum(sent_lengths) / n_sentences if sent_lengths else 0.0

    return {
        "ratio_exclamacion": n_exclam / n_sentences,
        "ratio_mayusculas": allcaps / max(n_words, 1),
        "long_oracion_prom": long_oracion_prom,
        "ratio_adj_sust": n_adj / max(n_nouns, 1),
        "sentimiento_vader": vader.polarity_scores(prepared_text)["compound"],
        "densidad_ner": n_entities / n_sentences,
        "freq_url": float(n_urls),
        "freq_pronombres": n_pronouns / max(n_tokens, 1),
    }


def extract_features_dataframe(
    series: pd.Series,
    *,
    normalize_source: bool = False,
    nlp=None,
    vader: SentimentIntensityAnalyzer | None = None,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Extrae features para una Serie de textos usando spaCy pipe."""
    nlp = nlp or spacy.load("en_core_web_sm")
    vader = vader or SentimentIntensityAnalyzer()

    texts = series.fillna("").astype(str).tolist()
    prepared_texts = [prepare_text(t, normalize_source=normalize_source) for t in texts]

    rows: list[dict[str, float]] = []
    for doc, prepared_text in tqdm(
        zip(
            nlp.pipe(prepared_texts, batch_size=batch_size),
            prepared_texts,
            strict=True,
        ),
        total=len(texts),
        desc="Features lingüísticas",
    ):
        rows.append(extract_features_from_doc(prepared_text, doc, vader))

    return pd.DataFrame(rows, columns=FEATURE_NAMES, index=series.index)


def load_or_extract_features(
    df: pd.DataFrame,
    text_col: str,
    prefix: str,
    split: str,
    *,
    normalize_source: bool = False,
    force: bool = False,
    nlp=None,
    vader: SentimentIntensityAnalyzer | None = None,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Lee features cacheadas o las extrae y persiste en Parquet."""
    cache_path = linguistic_features_cache_path(
        prefix, text_col, split, normalize_source=normalize_source
    )
    if cache_path.exists() and not force:
        cached = pd.read_parquet(cache_path)
        if len(cached) == len(df):
            return cached
        # El cache no coincide con el split actual (re-preprocesamiento,
        # NLP_DEV_MODE 10% vs. corrida completa, etc.): re-extraer para evitar
        # alinear features con artículos equivocados.
        print(
            f"Cache desalineado en {cache_path.name} "
            f"({len(cached)} filas vs. {len(df)} esperadas); re-extrayendo."
        )

    features = extract_features_dataframe(
        df[text_col],
        normalize_source=normalize_source,
        nlp=nlp,
        vader=vader,
        batch_size=batch_size,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(cache_path, index=False)
    return features


def load_source_normalization_decision(path: Path | None = None) -> dict:
    """Lee la decisión de ablación de fuente del Experimento 1."""
    path = path or SOURCE_ABLATION_DECISION
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def tune_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    C_grid: tuple[float, ...] = DEFAULT_C_GRID,
) -> tuple[LogisticRegression, float, dict]:
    """Selecciona C por F2 fake en validación."""
    best_c = C_grid[0]
    best_f2 = -1.0
    best_metrics: dict = {}

    for c in C_grid:
        clf = LogisticRegression(
            C=c,
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        y_proba = clf.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val, y_pred, y_proba)
        if metrics["f2_fake"] > best_f2:
            best_f2 = metrics["f2_fake"]
            best_c = c
            best_metrics = metrics

    best_clf = LogisticRegression(
        C=best_c,
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    best_clf.fit(X_train, y_train)
    return best_clf, best_c, best_metrics


def coefficients_dataframe(
    clf: LogisticRegression,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Tabla de coeficientes ordenada por magnitud absoluta."""
    names = feature_names or FEATURE_NAMES
    coefs = clf.coef_[0]
    out = pd.DataFrame({"feature": names, "coefficient": coefs})
    out["abs_coefficient"] = out["coefficient"].abs()
    return out.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def evaluate_linguistic_model(
    clf: LogisticRegression,
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[dict, pd.Series, pd.Series]:
    """Predice y devuelve métricas, etiquetas y probabilidades fake."""
    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)[:, 1]
    return compute_metrics(y, y_pred, y_proba), y_pred, y_proba


def select_best_text_field(
    field_results: pd.DataFrame,
) -> str:
    """Elige el campo textual con mayor F2 fake en validación."""
    best_row = field_results.loc[field_results["f2_fake"].idxmax()]
    return str(best_row["text_field"])

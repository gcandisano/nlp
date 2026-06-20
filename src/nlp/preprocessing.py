"""Funciones reutilizables de preprocesamiento de texto."""
import re

import pandas as pd

from nlp.paths import (
    DATA_PROCESSED,
    DATA_RAW,
    POLITICS_FAKE_OPTIONAL,
    POLITICS_FAKE_SUBJECTS,
    POLITICS_REAL_SUBJECTS,
)

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False

URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+",
    flags=re.IGNORECASE,
)
PUNCTUATION_PATTERN = re.compile(r"[^\w\s!?]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def ensure_nltk_resources() -> None:
    """Descarga recursos NLTK si no están disponibles."""
    if not _NLTK_AVAILABLE:
        return
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


def _get_stopwords() -> set:
    ensure_nltk_resources()
    if _NLTK_AVAILABLE:
        return set(stopwords.words("english"))
    return {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "shall", "can", "this",
        "that", "these", "those", "it", "its", "they", "them", "their", "we",
        "our", "you", "your", "he", "she", "his", "her", "as", "not", "no",
    }


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def replace_urls(text: str, token: str = "[URL]") -> str:
    return URL_PATTERN.sub(token, text)


def remove_punctuation(text: str, keep_exclamation_question: bool = True) -> str:
    if keep_exclamation_question:
        return PUNCTUATION_PATTERN.sub(" ", text)
    return re.sub(r"[^\w\s]", " ", text)


def remove_stopwords_from_tokens(tokens: list[str], stop_set: set | None = None) -> list[str]:
    stop_set = stop_set or _get_stopwords()
    return [t for t in tokens if t not in stop_set]


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    ensure_nltk_resources()
    if not _NLTK_AVAILABLE:
        return tokens
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(t) for t in tokens]


def clean_text(
    text: str,
    remove_stopwords: bool = False,
    lemmatize: bool = False,
    lowercase: bool = True,
    remove_punct: bool = True,
) -> str:
    """Pipeline de limpieza de una cadena de texto."""
    if pd.isna(text) or text is None:
        return ""

    text = str(text)
    if lowercase:
        text = text.lower()
    text = replace_urls(text)
    if remove_punct:
        text = remove_punctuation(text, keep_exclamation_question=True)
    text = normalize_whitespace(text)

    tokens = text.split()
    if lemmatize:
        tokens = lemmatize_tokens(tokens)
    if remove_stopwords:
        tokens = remove_stopwords_from_tokens(tokens)

    return normalize_whitespace(" ".join(tokens))


def add_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Crea columnas title_text, body_text y full_text."""
    out = df.copy()
    out["title_text"] = out["title"].fillna("").astype(str)
    out["body_text"] = out["text"].fillna("").astype(str)
    out["full_text"] = (
        out["title_text"].str.strip() + " " + out["body_text"].str.strip()
    ).str.strip()
    return out


def add_clean_text_columns(
    df: pd.DataFrame,
    lemmatize: bool = False,
) -> pd.DataFrame:
    """Agrega variantes con y sin stopwords para title, body y full."""
    out = add_text_columns(df)
    for col in ["title_text", "body_text", "full_text"]:
        out[f"clean_{col}_with_stopwords"] = out[col].apply(
            lambda x: clean_text(x, remove_stopwords=False, lemmatize=lemmatize)
        )
        out[f"clean_{col}_without_stopwords"] = out[col].apply(
            lambda x: clean_text(x, remove_stopwords=True, lemmatize=lemmatize)
        )
    return out


def drop_content_duplicates(
    df: pd.DataFrame,
    subset: list[str] | None = None,
    keep: str = "first",
) -> tuple[pd.DataFrame, dict]:
    """Elimina duplicados por contenido (title+text por defecto).

    Se aplica antes del split temporal para evitar que el mismo artículo
    aparezca en train y test, lo que inflaría métricas o introduciría leakage.
    """
    subset = subset or ["title", "text"]
    before = len(df)

    dup_mask = df.duplicated(subset=subset, keep=False)
    label_conflicts = 0
    if dup_mask.any():
        conflict_groups = (
            df[dup_mask]
            .groupby(subset, dropna=False)["label"]
            .nunique()
        )
        label_conflicts = int((conflict_groups > 1).sum())

    out = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    stats = {
        "rows_before": before,
        "rows_after": len(out),
        "removed": before - len(out),
        "label_conflicts": label_conflicts,
    }
    return out, stats


def parse_dates(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Parsea fechas con múltiples formatos presentes en el dataset."""
    out = df.copy()
    out[date_col] = out[date_col].astype(str).str.strip()

    parsed = pd.to_datetime(out[date_col], format="%B %d, %Y", errors="coerce")

    mask = parsed.isna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(
            out.loc[mask, date_col], format="%d-%b-%y", errors="coerce"
        )

    mask = parsed.isna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(out.loc[mask, date_col], errors="coerce")

    out["parsed_date"] = parsed
    return out


def temporal_split(
    df: pd.DataFrame,
    date_col: str = "parsed_date",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split temporal 70/15/15 ordenando por fecha."""
    sorted_df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    n = len(sorted_df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train = sorted_df.iloc[:train_end].copy()
    val = sorted_df.iloc[train_end:val_end].copy()
    test = sorted_df.iloc[val_end:].copy()
    return train, val, test


def class_distribution_report(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    counts = df[label_col].value_counts().sort_index()
    total = len(df)
    report = pd.DataFrame({
        "class": ["real (0)", "fake (1)"],
        "count": [counts.get(0, 0), counts.get(1, 0)],
    })
    report["percentage"] = (report["count"] / total * 100).round(2)
    report["split_total"] = total
    return report


def filter_politics_subset(df: pd.DataFrame, include_optional: bool = False) -> pd.DataFrame:
    fake_subjects = list(POLITICS_FAKE_SUBJECTS)
    if include_optional:
        fake_subjects.extend(POLITICS_FAKE_OPTIONAL)

    mask_real = (df["label"] == 0) & (df["subject"].isin(POLITICS_REAL_SUBJECTS))
    mask_fake = (df["label"] == 1) & (df["subject"].isin(fake_subjects))
    return df[mask_real | mask_fake].copy()


def _save_splits(dataset: pd.DataFrame, prefix: str) -> None:
    train, val, test = temporal_split(dataset)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = DATA_PROCESSED / f"{prefix}_{name}.csv"
        split.to_csv(path, index=False)
        print(f"\n{prefix}_{name}: {len(split):,} rows")
        print(class_distribution_report(split))


def run_preprocessing_pipeline() -> None:
    """Lee datos raw, preprocesa y escribe splits en data/processed/."""
    fake_df = pd.read_csv(DATA_RAW / "Fake.csv")
    true_df = pd.read_csv(DATA_RAW / "True.csv")
    fake_df["label"] = 1
    true_df["label"] = 0
    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = parse_dates(df)
    df = df.dropna(subset=["parsed_date"]).reset_index(drop=True)
    df, dedup_stats = drop_content_duplicates(df)
    print(
        f"Deduplicación title+text: {dedup_stats['removed']:,} filas eliminadas "
        f"({dedup_stats['rows_before']:,} -> {dedup_stats['rows_after']:,})"
    )
    if dedup_stats["label_conflicts"]:
        print(f"  Grupos duplicados con etiquetas distintas: {dedup_stats['label_conflicts']}")
    df = add_clean_text_columns(df)

    politics_df = filter_politics_subset(df, include_optional=False)
    _save_splits(politics_df, "politics")
    _save_splits(df, "full")

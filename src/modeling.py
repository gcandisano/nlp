"""Configuración y utilidades para modelos baseline."""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.paths import RANDOM_STATE

TEXT_FIELDS = {
    "title": "clean_title_text",
    "body": "clean_body_text",
    "full": "clean_full_text",
}

STOPWORD_OPTS = {
    "with_stopwords": "_with_stopwords",
    "without_stopwords": "_without_stopwords",
}

MODEL_CONFIGS = {
    "logistic_regression": {
        "model": LogisticRegression,
        "params": {"C": [0.1, 1, 10], "max_iter": [1000], "random_state": [RANDOM_STATE]},
        "param_name": "C",
    },
    "multinomial_nb": {
        "model": MultinomialNB,
        "params": {"alpha": [0.1, 1]},
        "param_name": "alpha",
    },
    "linear_svm": {
        "model": LinearSVC,
        "params": {"C": [0.1, 1, 10], "random_state": [RANDOM_STATE]},
        "param_name": "C",
    },
}


def get_text_col(field_key: str, stop_key: str) -> str:
    return TEXT_FIELDS[field_key] + STOPWORD_OPTS[stop_key]


def build_vectorizer(vtype: str, ngram_range, max_features: int):
    if vtype == "bow":
        return CountVectorizer(ngram_range=ngram_range, max_features=max_features, min_df=2)
    return TfidfVectorizer(ngram_range=ngram_range, max_features=max_features, min_df=2)


def build_model(model_name: str, param_value):
    cfg = MODEL_CONFIGS[model_name]
    param_name = cfg["param_name"]
    model_kwargs = {param_name: param_value}
    if model_name == "logistic_regression":
        model_kwargs.update(max_iter=1000, random_state=RANDOM_STATE)
    elif model_name == "linear_svm":
        model_kwargs["random_state"] = RANDOM_STATE
    return cfg["model"](**model_kwargs)

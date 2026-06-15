"""Configuración y utilidades para modelos baseline."""
from __future__ import annotations

from itertools import product

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from tqdm.auto import tqdm

from src.metrics import compute_metrics, metrics_to_row
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


def build_pipeline(vectorizer_type: str, ngram_range, max_features: int, model_name: str, param_value):
    return Pipeline([
        ("vec", build_vectorizer(vectorizer_type, ngram_range, max_features)),
        ("clf", build_model(model_name, param_value)),
    ])


def config_from_row(row) -> dict:
    return {
        "model": row["model"],
        "vectorizer": row["vectorizer"],
        "text_field": row["text_field"],
        "stopwords": row["stopwords"],
        "ngram_range": eval(row["ngram_range"]),
        "max_features": int(row["max_features"]),
        "best_param": row["best_param"],
    }


def fit_pipeline_from_config(train: pd.DataFrame, config: dict) -> tuple[Pipeline, str]:
    text_col = get_text_col(config["text_field"], config["stopwords"])
    pipe = build_pipeline(
        config["vectorizer"],
        config["ngram_range"],
        config["max_features"],
        config["model"],
        config["best_param"],
    )
    pipe.fit(train[text_col].fillna(""), train["label"])
    return pipe, text_col


def predict_proba_scores(pipe: Pipeline, X):
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    if hasattr(clf, "decision_function"):
        scores = pipe.decision_function(X)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    return None


def evaluate_pipeline_on_split(pipe: Pipeline, df: pd.DataFrame, text_col: str, split_name: str, extra: dict) -> dict:
    X = df[text_col].fillna("")
    y = df["label"]
    y_pred = pipe.predict(X)
    y_proba = predict_proba_scores(pipe, X)
    metrics = compute_metrics(y, y_pred, y_proba)
    row = metrics_to_row(metrics, {**extra, "split": split_name})
    return row


def run_baseline_grid(
    train,
    val,
    dataset_scope: str,
    ngram_opts=None,
    max_features_opts=None,
    vectorizer_types=None,
    max_combos: int | None = None,
) -> pd.DataFrame:
    """Entrena la grilla y selecciona hiperparámetros solo con validación."""
    ngram_opts = ngram_opts or [(1, 1), (1, 2)]
    max_features_opts = max_features_opts or [10000, 30000, 50000]
    vectorizer_types = vectorizer_types or ["bow", "tfidf"]

    results = []
    combos = list(product(
        TEXT_FIELDS.keys(),
        STOPWORD_OPTS.keys(),
        ngram_opts,
        max_features_opts,
        vectorizer_types,
        MODEL_CONFIGS.keys(),
    ))
    if max_combos:
        combos = combos[:max_combos]

    for field, stop, ngram, max_feat, vtype, model_name in tqdm(combos, desc=dataset_scope):
        text_col = get_text_col(field, stop)
        X_train = train[text_col].fillna("")
        X_val = val[text_col].fillna("")
        y_train, y_val = train["label"], val["label"]

        cfg = MODEL_CONFIGS[model_name]
        param_name = cfg["param_name"]
        best_f2, best_pipe, best_param = -1, None, None

        for param_val in cfg["params"][param_name]:
            pipe = build_pipeline(vtype, ngram, max_feat, model_name, param_val)
            pipe.fit(X_train, y_train)
            y_val_pred = pipe.predict(X_val)
            m = compute_metrics(y_val, y_val_pred)
            if m["f2_fake"] > best_f2:
                best_f2 = m["f2_fake"]
                best_pipe = pipe
                best_param = param_val

        y_proba = predict_proba_scores(best_pipe, X_val)
        val_m = compute_metrics(y_val, best_pipe.predict(X_val), y_proba)
        results.append(metrics_to_row(val_m, {
            "model": model_name,
            "vectorizer": vtype,
            "text_field": field,
            "stopwords": stop,
            "ngram_range": str(ngram),
            "max_features": max_feat,
            "best_param": best_param,
            "dataset_scope": dataset_scope,
            "split": "val",
        }))

    return pd.DataFrame(results)


def evaluate_best_configs_on_test(val_results: pd.DataFrame, train, test, dataset_scope: str) -> tuple[pd.DataFrame, pd.Series, Pipeline, str]:
    """Elige la mejor config por F2 en val y evalúa una sola vez en test."""
    scope_results = val_results[val_results["dataset_scope"] == dataset_scope]
    best_row = scope_results.nlargest(1, "f2_fake").iloc[0]
    config = config_from_row(best_row)
    pipe, text_col = fit_pipeline_from_config(train, config)
    test_row = evaluate_pipeline_on_split(
        pipe,
        test,
        text_col,
        "test",
        {
            "model": best_row["model"],
            "vectorizer": best_row["vectorizer"],
            "text_field": best_row["text_field"],
            "stopwords": best_row["stopwords"],
            "ngram_range": best_row["ngram_range"],
            "max_features": int(best_row["max_features"]),
            "best_param": best_row["best_param"],
            "dataset_scope": dataset_scope,
        },
    )
    return pd.DataFrame([test_row]), best_row, pipe, text_col


def get_linear_feature_weights(pipe: Pipeline):
    """Devuelve pesos por término para modelos lineales del pipeline baseline."""
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "coef_"):
        return clf.coef_[0]
    if hasattr(clf, "feature_log_prob_"):
        return clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    raise ValueError("El modelo guardado no expone coeficientes interpretables.")

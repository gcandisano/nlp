"""Genera los notebooks del TP NLP Fake News."""
import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"


def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(source: str):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def save(name, cells):
    path = NOTEBOOKS_DIR / name
    path.write_text(json.dumps(nb(cells), indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Created {path}")


# Common setup cell
SETUP = """
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / 'src').exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.paths import *
from src.plotting import setup_style, save_figure

setup_style()
""".strip()


def notebook_01():
    cells = [
        md("""# Notebook 01 — Análisis Exploratorio de Datos (EDA)

## Objetivo

Explorar el dataset de noticias fake y real para entender su estructura, distribución temporal, longitudes de texto y posibles sesgos temáticos.

> **Aclaración metodológica:** Este análisis describe patrones en los datos. No implica que el modelo posterior verifique hechos contra la realidad; solo aprenderá correlaciones lingüísticas del dataset."""),
        code(SETUP),
        md("## 1. Carga de datos y etiquetado"),
        code("""
fake_df = pd.read_csv(DATA_RAW / 'Fake.csv')
true_df = pd.read_csv(DATA_RAW / 'True.csv')

fake_df['label'] = 1
true_df['label'] = 0

df = pd.concat([fake_df, true_df], ignore_index=True)
print(f'Total registros: {len(df):,}')
df.head()
"""),
        md("## 2. Columnas, nulos y duplicados"),
        code("""
print('Columnas:', df.columns.tolist())
print('\\nNulos por columna:')
print(df.isnull().sum())
print('\\nDuplicados exactos:', df.duplicated().sum())
print('\\nDuplicados en title+text:', df.duplicated(subset=['title', 'text']).sum())
"""),
        md("## 3. Parsing de fechas"),
        code("""
from src.preprocessing import parse_dates, add_text_columns

df = parse_dates(df)
invalid_dates = df['parsed_date'].isna().sum()
print(f'Fechas inválidas tras parsing multi-formato: {invalid_dates}')
print('Formatos detectados: "%B %d, %Y" (ej. May 31, 2017) y "%d-%b-%y" (ej. 19-Feb-18)')
df = df.dropna(subset=['parsed_date']).reset_index(drop=True)
df = add_text_columns(df)
df[['date', 'parsed_date']].head()
"""),
        md("## 4. Distribución de clases"),
        code("""
class_counts = df['label'].value_counts().sort_index()
total = len(df)
print('Distribución de clases:')
for label, name in [(0, 'real'), (1, 'fake')]:
    count = class_counts.get(label, 0)
    pct = count / total * 100
    print(f'  {name}: {count:,} ({pct:.2f}%)')
"""),
        md("## 5. Distribución temporal"),
        code("""
for label, name in [(0, 'real'), (1, 'fake')]:
    subset = df[df['label'] == label]
    print(f'{name}: {subset["parsed_date"].min().date()} -> {subset["parsed_date"].max().date()}')

df['year_month'] = df['parsed_date'].dt.to_period('M').astype(str)
monthly = df.groupby(['year_month', 'label']).size().unstack(fill_value=0)
monthly.columns = ['real', 'fake']
monthly.tail(10)
"""),
        md("## 6. Longitud de textos (palabras)"),
        code("""
for col in ['title', 'text', 'full_text']:
    df[f'{col}_word_count'] = df[col].fillna('').astype(str).str.split().str.len()

length_stats = df.groupby('label')[['title_word_count', 'text_word_count', 'full_text_word_count']].agg(['mean', 'median', 'std'])
length_stats.index = ['real', 'fake']
length_stats.round(2)
"""),
        md("## 7. Gráficos exploratorios"),
        code("""
# Distribución de clases
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(data=df, x='label', hue='label', palette=['#2ecc71', '#e74c3c'], legend=False, ax=ax)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Real (0)', 'Fake (1)'])
ax.set_title('Distribución de clases')
ax.set_ylabel('Cantidad')
save_figure(fig, RESULTS_FIGURES / 'eda_class_distribution.png')
plt.show()

# Distribución temporal
fig, ax = plt.subplots(figsize=(12, 4))
monthly_plot = df.groupby(df['parsed_date'].dt.to_period('M')).size()
monthly_plot.index = monthly_plot.index.astype(str)
monthly_plot.plot(ax=ax)
ax.set_title('Cantidad de noticias por mes')
ax.set_xlabel('Mes')
ax.set_ylabel('Cantidad')
plt.xticks(rotation=45)
save_figure(fig, RESULTS_FIGURES / 'eda_temporal_distribution.png')
plt.show()
"""),
        code("""
# Longitud de títulos y cuerpos
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col, title in zip(axes, ['title_word_count', 'text_word_count'], ['Títulos', 'Cuerpos']):
    sns.histplot(data=df, x=col, hue='label', bins=50, element='step', stat='density', common_norm=False, ax=ax)
    ax.set_title(f'Distribución de longitud — {title}')
    ax.legend(['Real', 'Fake'])
save_figure(fig, RESULTS_FIGURES / 'eda_text_length_distribution.png')
plt.show()
"""),
        md("""## 8. Análisis de `subject`

La columna `subject` **no se usará como feature** para entrenar modelos. Solo sirve para análisis exploratorio y para definir el subconjunto político.

**Advertencia:** `subject` puede funcionar como predictor demasiado fuerte y sesgado, ya que la distribución temática difiere mucho entre fake y real."""),
        code("""
subject_class = pd.crosstab(df['subject'], df['label'], margins=True)
subject_class.columns = ['real', 'fake', 'total']
subject_class['pct_fake'] = (subject_class['fake'] / subject_class['total'] * 100).round(1)
subject_class.sort_values('total', ascending=False)
"""),
        code("""
fig, ax = plt.subplots(figsize=(10, 6))
plot_df = df.groupby(['subject', 'label']).size().reset_index(name='count')
sns.barplot(data=plot_df, y='subject', x='count', hue='label', ax=ax)
ax.set_title('Distribución por subject y clase')
ax.legend(['Real', 'Fake'])
save_figure(fig, RESULTS_FIGURES / 'eda_subject_distribution.png')
plt.show()

print('\\n--- Conteos por subject y clase (para definir subconjunto político) ---')
politics_related = ['politicsNews', 'politics', 'Government News', 'left-news']
for subj in politics_related:
    sub = df[df['subject'] == subj]
    if len(sub) > 0:
        real_c = (sub['label'] == 0).sum()
        fake_c = (sub['label'] == 1).sum()
        print(f'{subj}: real={real_c:,}, fake={fake_c:,}, total={len(sub):,}')
"""),
        md("""### Decisión del subconjunto político

- **Reales:** `politicsNews` (único subject político en noticias reales)
- **Falsas:** `politics` (principal subject político en fake news)
- `Government News` y `left-news` son opcionales; se evalúan arriba pero el experimento principal usa solo `politics` para mantener comparabilidad temática."""),
        md("## 9. Palabras frecuentes por clase"),
        code("""
from collections import Counter
import re

def top_words(texts, n=20):
    words = []
    for t in texts:
        tokens = re.findall(r'\\b[a-z]{3,}\\b', str(t).lower())
        words.extend(tokens)
    return Counter(words).most_common(n)

fake_words = top_words(df[df['label'] == 1]['full_text'])
real_words = top_words(df[df['label'] == 0]['full_text'])

print('Top términos FAKE:')
for w, c in fake_words:
    print(f'  {w}: {c:,}')
print('\\nTop términos REAL:')
for w, c in real_words:
    print(f'  {w}: {c:,}')
"""),
        code("""
# Word clouds opcionales
try:
    from wordcloud import WordCloud

    for label, name, color in [(1, 'fake', '#e74c3c'), (0, 'real', '#2ecc71')]:
        text = ' '.join(df[df['label'] == label]['full_text'].astype(str))
        wc = WordCloud(width=800, height=400, background_color='white', colormap='Reds' if label else 'Greens').generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'Word cloud — {name}')
        save_figure(fig, RESULTS_FIGURES / f'eda_wordcloud_{name}.png')
        plt.show()
except ImportError:
    print('wordcloud no instalado; se omite word cloud')
"""),
        md("""## 10. Conclusiones preliminares del EDA

1. El dataset está relativamente balanceado entre fake y real.
2. Hay **fuerte asimetría temática**: las reales concentran `politicsNews` y `worldnews`; las falsas tienen múltiples subjects (`News`, `politics`, `left-news`, etc.).
3. Esto justifica el **experimento principal sobre el subconjunto político** para reducir sesgo temático.
4. Las noticias reales (Reuters) pueden tener marcas de estilo periodístico formal que el modelo podría aprender como proxy de veracidad.
5. El modelo posterior aprenderá **patrones del dataset**, no verificará hechos.

> Los duplicados detectados aquí se eliminan en el **Notebook 02** (criterio: `title` + `text`) antes del split temporal."""),
    ]
    save("01_eda.ipynb", cells)


def notebook_02():
    cells = [
        md("""# Notebook 02 — Preprocesamiento y Partición Temporal

## Objetivo

Implementar funciones reutilizables de limpieza de texto y generar splits temporales (70/15/15) para:
- **Subconjunto político** (experimento principal)
- **Dataset completo** (experimento complementario)

> Los modelos aprenden patrones lingüísticos del dataset, no verifican hechos."""),
        code(SETUP),
        md("## 1. Carga y preparación inicial"),
        code("""
from src.preprocessing import (
    add_clean_text_columns, parse_dates, temporal_split,
    class_distribution_report, filter_politics_subset, drop_content_duplicates,
)

fake_df = pd.read_csv(DATA_RAW / 'Fake.csv')
true_df = pd.read_csv(DATA_RAW / 'True.csv')
fake_df['label'] = 1
true_df['label'] = 0
df = pd.concat([fake_df, true_df], ignore_index=True)
df = parse_dates(df)
df = df.dropna(subset=['parsed_date']).reset_index(drop=True)
print(f'Registros con fecha válida: {len(df):,}')
"""),
        md("""## 2. Eliminación de duplicados

En el EDA se detectaron duplicados por `title` + `text`. Se eliminan **antes del split temporal** para:
- evitar que el mismo artículo aparezca en train y test (data leakage);
- no inflar métricas por filas repetidas.

Criterio: conservar la **primera** ocurrencia de cada par `(title, text)`."""),
        code("""
df, dedup_stats = drop_content_duplicates(df)
print(f"Filas antes:  {dedup_stats['rows_before']:,}")
print(f"Filas después: {dedup_stats['rows_after']:,}")
print(f"Eliminadas:    {dedup_stats['removed']:,}")
if dedup_stats['label_conflicts']:
    print(f"Grupos duplicados con etiquetas distintas (se conservó la primera): {dedup_stats['label_conflicts']}")
"""),
        md("## 3. Funciones de preprocesamiento de texto"),

        code("""
# Demostración de las funciones de limpieza
from src.preprocessing import clean_text

sample = "Check this OUT!!! https://example.com/news  Trump said WHAT???  Visit www.fake.com"
print('Original:', sample)
print('Con stopwords:', clean_text(sample, remove_stopwords=False))
print('Sin stopwords:', clean_text(sample, remove_stopwords=True))
"""),
        md("""### Variantes de texto generadas

Para cada uno de `title_text`, `body_text`, `full_text`:
- `clean_*_with_stopwords`
- `clean_*_without_stopwords`

Las URLs se reemplazan por `[URL]` (no se eliminan). Se preservan `!` y `?` para análisis posterior."""),
        code("""
df = add_clean_text_columns(df, lemmatize=False)
politics_df = filter_politics_subset(df, include_optional=False)
print(f'Subconjunto político: {len(politics_df):,} registros')
print(politics_df.groupby('label').size())
"""),
        md("## 4. Split temporal (70% train / 15% val / 15% test)"),
        code("""
def process_and_save(dataset, prefix):
    train, val, test = temporal_split(dataset)
    for name, split in [('train', train), ('val', val), ('test', test)]:
        print(f'\\n=== {prefix}_{name} ===')
        print(class_distribution_report(split))
        path = DATA_PROCESSED / f'{prefix}_{name}.csv'
        split.to_csv(path, index=False)
        print(f'Guardado: {path}')
    return train, val, test

pol_train, pol_val, pol_test = process_and_save(politics_df, 'politics')
full_train, full_val, full_test = process_and_save(df, 'full')
"""),
        md("""### Verificación de balance en splits

Si alguna partición queda muy desbalanceada, se reporta aquí. **No se aplica oversampling/undersampling** ni se rompe el criterio temporal salvo necesidad documentada."""),
        code("""
def check_imbalance(train, val, test, threshold=0.65):
    for name, split in [('train', train), ('val', val), ('test', test)]:
        fake_pct = split['label'].mean()
        status = 'OK' if 0.35 <= fake_pct <= threshold else 'DESBALANCEADO'
        print(f'{name}: {fake_pct:.1%} fake — {status}')

print('--- Subconjunto político ---')
check_imbalance(pol_train, pol_val, pol_test)
print('\\n--- Dataset completo ---')
check_imbalance(full_train, full_val, full_test)
"""),
        md("""## 5. Análisis de signos `!` y `?` y URLs

Evaluamos si estos patrones aparecen más en una clase que en otra (información potencial para el modelo)."""),
        code("""
for label, name in [(0, 'real'), (1, 'fake')]:
    sub = politics_df[politics_df['label'] == label]
    excl_pct = sub['full_text'].str.contains('!', regex=False).mean() * 100
    quest_pct = sub['full_text'].str.contains('?', regex=False).mean() * 100
    url_pct = sub['full_text'].str.contains(r'https?://|www\\.', regex=True).mean() * 100
    print(f'{name}: ! en {excl_pct:.1f}%, ? en {quest_pct:.1f}%, URLs en {url_pct:.1f}%')
"""),
        md("""## Conclusiones

- Se generaron 6 archivos en `data/processed/`: `politics_{train,val,test}.csv` y `full_{train,val,test}.csv`.
- Cada archivo incluye columnas de texto crudo y preprocesado (con/sin stopwords).
- El split es **temporal**: train = más antiguo, test = más reciente, evitando leakage temporal.
- `subject` no se usa como feature; solo definió el filtro político."""),
    ]
    save("02_preprocessing_and_splits.ipynb", cells)


def notebook_03():
    cells = [
        md("""# Notebook 03 — Modelos Baseline

## Objetivo

Entrenar modelos clásicos de clasificación supervisada sobre representaciones BoW y TF-IDF.

**Experimento principal:** subconjunto político.  
**Experimento complementario:** dataset completo (solo baselines).

> El modelo aprende patrones lingüísticos del dataset; no verifica hechos.

**Métrica principal:** F2-score de la clase fake (prioriza recall de fake sin ignorar precisión)."""),
        code(SETUP + """

from itertools import product
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, roc_curve, auc
import joblib
from tqdm.auto import tqdm

from src.metrics import compute_metrics, metrics_to_row
from src.modeling import MODEL_CONFIGS, TEXT_FIELDS, STOPWORD_OPTS, build_model, build_vectorizer, get_text_col
from src.plotting import plot_confusion_matrix
"""),
        md("## 1. Carga de splits"),
        code("""
def load_splits(prefix):
    train = pd.read_csv(DATA_PROCESSED / f'{prefix}_train.csv')
    val = pd.read_csv(DATA_PROCESSED / f'{prefix}_val.csv')
    test = pd.read_csv(DATA_PROCESSED / f'{prefix}_test.csv')
    return train, val, test

pol_train, pol_val, pol_test = load_splits('politics')
full_train, full_val, full_test = load_splits('full')
"""),
        md("## 2. Configuración de la grilla de experimentos"),
        code("""
NGRAM_OPTS = [(1, 1), (1, 2)]
MAX_FEATURES_OPTS = [10000, 30000, 50000]
VECTORIZER_TYPES = ['bow', 'tfidf']
"""),
        md("## 3. Entrenamiento y selección por validación (F2 fake)"),
        code("""
def run_baseline_grid(train, val, dataset_scope, max_combos=None):
    """Entrena la grilla y elige hiperparámetros solo con validación."""
    results = []
    combos = list(product(
        TEXT_FIELDS.keys(), STOPWORD_OPTS.keys(), NGRAM_OPTS,
        MAX_FEATURES_OPTS, VECTORIZER_TYPES, MODEL_CONFIGS.keys()
    ))
    if max_combos:
        combos = combos[:max_combos]

    for field, stop, ngram, max_feat, vtype, model_name in tqdm(combos, desc=dataset_scope):
        text_col = get_text_col(field, stop)
        X_train = train[text_col].fillna('')
        X_val = val[text_col].fillna('')
        y_train, y_val = train['label'], val['label']

        cfg = MODEL_CONFIGS[model_name]
        param_name = cfg['param_name']
        best_f2, best_param, best_metrics = -1, None, None

        for param_val in cfg['params'][param_name]:
            pipe = Pipeline([
                ('vec', build_vectorizer(vtype, ngram, max_feat)),
                ('clf', build_model(model_name, param_val)),
            ])
            pipe.fit(X_train, y_train)
            y_val_pred = pipe.predict(X_val)
            m = compute_metrics(y_val, y_val_pred)
            if m['f2_fake'] > best_f2:
                best_f2 = m['f2_fake']
                best_param = param_val
                best_metrics = m

        row = metrics_to_row(best_metrics, {
            'model': model_name,
            'vectorizer': vtype,
            'text_field': field,
            'stopwords': stop,
            'ngram_range': str(ngram),
            'max_features': max_feat,
            'best_param': best_param,
            'dataset_scope': dataset_scope,
            'split': 'val',
        })
        results.append(row)

    return pd.DataFrame(results)


def build_pipeline_from_config(config_row):
    ngram = eval(config_row['ngram_range'])
    return Pipeline([
        ('vec', build_vectorizer(
            config_row['vectorizer'], ngram, int(config_row['max_features'])
        )),
        ('clf', build_model(config_row['model'], config_row['best_param'])),
    ])


def evaluate_on_test(config_row, train, test):
    """Entrena con train y evalúa una sola vez en test."""
    text_col = get_text_col(config_row['text_field'], config_row['stopwords'])
    pipe = build_pipeline_from_config(config_row)
    pipe.fit(train[text_col].fillna(''), train['label'])

    y_test_pred = pipe.predict(test[text_col].fillna(''))
    y_proba = None
    clf = pipe.named_steps['clf']
    if hasattr(clf, 'predict_proba'):
        y_proba = clf.predict_proba(test[text_col].fillna(''))[:, 1]
    elif hasattr(clf, 'decision_function'):
        scores = clf.decision_function(test[text_col].fillna(''))
        y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    test_m = compute_metrics(test['label'], y_test_pred, y_proba)
    result_row = metrics_to_row(test_m, {
        'model': config_row['model'],
        'vectorizer': config_row['vectorizer'],
        'text_field': config_row['text_field'],
        'stopwords': config_row['stopwords'],
        'ngram_range': config_row['ngram_range'],
        'max_features': int(config_row['max_features']),
        'best_param': config_row['best_param'],
        'dataset_scope': config_row['dataset_scope'],
        'split': 'test',
    })
    return pipe, result_row

# Experimento principal: subconjunto político (solo métricas de validación)
politics_results = run_baseline_grid(pol_train, pol_val, 'politics')
"""),
        code("""
# Experimento complementario: dataset completo (solo métricas de validación)
full_results = run_baseline_grid(full_train, full_val, 'full_dataset')

val_results = pd.concat([politics_results, full_results], ignore_index=True)
print('Mejores 10 configuraciones según validación:')
print(val_results.sort_values('f2_fake', ascending=False).head(10).to_string(index=False))
"""),
        md("## 4. Mejor modelo (selección en val) y evaluación final en test"),
        code("""
# Selección final por validación; test solo para el mejor modelo de cada alcance
pol_best_val = politics_results.sort_values('f2_fake', ascending=False).iloc[0]
full_best_val = full_results.sort_values('f2_fake', ascending=False).iloc[0]

pipe_pol, pol_best_test = evaluate_on_test(pol_best_val, pol_train, pol_test)
_, full_best_test = evaluate_on_test(full_best_val, full_train, full_test)

baseline_results = pd.concat([
    val_results,
    pd.DataFrame([pol_best_test, full_best_test]),
], ignore_index=True)
baseline_results.to_csv(RESULTS_METRICS / 'baseline_results.csv', index=False)

print('Mejor configuración politics (val):')
print(pol_best_val.to_string())
print('\nEvaluación en test del mejor modelo politics:')
print(pd.Series(pol_best_test).to_string())

text_col = get_text_col(pol_best_test['text_field'], pol_best_test['stopwords'])
y_pred = pipe_pol.predict(pol_test[text_col].fillna(''))
cm = confusion_matrix(pol_test['label'], y_pred)

plot_confusion_matrix(
    cm, ['Real', 'Fake'],
    f'Matriz de confusión — {pol_best_test["model"]}',
    RESULTS_FIGURES / 'baseline_best_confusion_matrix.png',
)

clf = pipe_pol.named_steps['clf']
if hasattr(clf, 'predict_proba'):
    y_proba = clf.predict_proba(pol_test[text_col].fillna(''))[:, 1]
elif hasattr(clf, 'decision_function'):
    scores = clf.decision_function(pol_test[text_col].fillna(''))
    y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
else:
    y_proba = None

if y_proba is not None:
    fpr, tpr, _ = roc_curve(pol_test['label'], y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title('Curva ROC — mejor baseline (politics, test)')
    ax.legend()
    save_figure(fig, RESULTS_FIGURES / 'baseline_best_roc_curve.png')
    plt.show()

joblib.dump(pipe_pol, RESULTS_MODELS / 'best_baseline_politics.joblib')
pd.Series(pol_best_test).to_json(RESULTS_MODELS / 'best_baseline_politics_config.json')
joblib.dump(pipe_pol, RESULTS_MODELS / 'best_baseline_model.joblib')
print('Modelos guardados en results/models/')
"""),
        md("""## 5. Comparación politics vs full_dataset

Si el rendimiento en el dataset completo es mucho mayor que en el subconjunto político, probablemente parte del rendimiento se explica por sesgos de tema, fuente o estructura del dataset — no solo por patrones lingüísticos de fake news."""),
        code("""
compare_val = val_results.groupby('dataset_scope')['f2_fake'].max()
compare_test = baseline_results[baseline_results['split'] == 'test'].set_index('dataset_scope')['f2_fake']
print('Mejor F2 en validación por alcance:')
print(compare_val)
print('\nF2 en test del mejor modelo seleccionado en val:')
print(compare_test)

fig, ax = plt.subplots(figsize=(8, 4))
top_pol = politics_results.nlargest(5, 'f2_fake')['f2_fake'].mean()
top_full = full_results.nlargest(5, 'f2_fake')['f2_fake'].mean()
sns.barplot(
    x=['politics (top-5 avg)', 'full_dataset (top-5 avg)'],
    y=[top_pol, top_full],
    ax=ax,
)
ax.set_ylabel('F2 fake en validación (promedio top-5)')
ax.set_title('Comparación subconjunto político vs dataset completo (val)')
save_figure(fig, RESULTS_FIGURES / 'baseline_politics_vs_full.png')
plt.show()
"""),
        md("""## Conclusiones

- Se compararon LR, MNB y Linear SVM con BoW y TF-IDF.
- Se evaluaron título, cuerpo y título+cuerpo; con y sin stopwords.
- La métrica principal F2 prioriza detectar fake news (minimizar falsos negativos).
- El modelo aprende patrones del dataset, no verifica hechos."""),
    ]
    save("03_baseline_models.ipynb", cells)


# Continue with notebooks 04, 05, 06
def notebook_04():
    cells = [
        md("""# Notebook 04 — Embeddings y Transformers

## Objetivo

Comparar modelos más complejos sobre el **subconjunto político**:
- **Parte A:** Embeddings preentrenados (GloVe) + LR/SVM
- **Parte B:** Fine-tuning de DistilBERT

> Los modelos aprenden patrones lingüísticos del dataset; no verifican hechos."""),
        code(SETUP + """

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
import joblib

from src.metrics import compute_metrics, metrics_to_row
from src.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
"""),
        md("## Parte A — Embeddings (GloVe)"),
        code("""
pol_train = pd.read_csv(DATA_PROCESSED / 'politics_train.csv')
pol_val = pd.read_csv(DATA_PROCESSED / 'politics_val.csv')
pol_test = pd.read_csv(DATA_PROCESSED / 'politics_test.csv')

GLOVE_DIM = 100
GLOVE_URL = 'http://nlp.stanford.edu/data/glove.6B.zip'
GLOVE_DIR = PROJECT_ROOT / 'data' / 'embeddings'
GLOVE_DIR.mkdir(parents=True, exist_ok=True)
"""),
        code("""
import urllib.request
import zipfile

def load_glove_vectors(dim=100):
    glove_file = GLOVE_DIR / f'glove.6B.{dim}d.txt'
    if not glove_file.exists():
        zip_path = GLOVE_DIR / 'glove.6B.zip'
        if not zip_path.exists():
            print('Descargando GloVe (puede tardar varios minutos)...')
            urllib.request.urlretrieve(GLOVE_URL, zip_path)
        print('Extrayendo GloVe...')
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extract(f'glove.6B.{dim}d.txt', GLOVE_DIR)

    vectors = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='Cargando GloVe'):
            parts = line.strip().split()
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            vectors[word] = vec
    return vectors

glove = load_glove_vectors(GLOVE_DIM)
print(f'Vectores cargados: {len(glove):,}')
"""),
        code("""
def text_to_embedding(text, vectors, dim=100):
    tokens = str(text).lower().split()
    vecs = [vectors[t] for t in tokens if t in vectors]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0)

def embed_corpus(texts, vectors, dim=100):
    return np.vstack([text_to_embedding(t, vectors, dim) for t in tqdm(texts, desc='Embedding')])

TEXT_COL = 'clean_full_text_without_stopwords'
X_train_emb = embed_corpus(pol_train[TEXT_COL].fillna(''), glove, GLOVE_DIM)
X_val_emb = embed_corpus(pol_val[TEXT_COL].fillna(''), glove, GLOVE_DIM)
X_test_emb = embed_corpus(pol_test[TEXT_COL].fillna(''), glove, GLOVE_DIM)
y_train, y_val, y_test = pol_train['label'], pol_val['label'], pol_test['label']
"""),
        code("""
embedding_results = []
for model_name, ModelClass, params in [
    ('logistic_regression', LogisticRegression, {'C': [0.1, 1, 10]}),
    ('linear_svm', LinearSVC, {'C': [0.1, 1, 10]}),
]:
    best_f2, best_clf, best_param = -1, None, None
    for C in params['C']:
        clf = ModelClass(C=C, random_state=RANDOM_STATE, max_iter=2000) if model_name == 'logistic_regression' else ModelClass(C=C, random_state=RANDOM_STATE)
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
        pipe.fit(X_train_emb, y_train)
        y_val_pred = pipe.predict(X_val_emb)
        m = compute_metrics(y_val, y_val_pred)
        if m['f2_fake'] > best_f2:
            best_f2, best_clf, best_param = m['f2_fake'], pipe, C

    y_test_pred = best_clf.predict(X_test_emb)
    y_proba = None
    if hasattr(best_clf.named_steps['clf'], 'predict_proba'):
        y_proba = best_clf.predict_proba(X_test_emb)[:, 1]
    elif hasattr(best_clf.named_steps['clf'], 'decision_function'):
        s = best_clf.decision_function(X_test_emb)
        y_proba = (s - s.min()) / (s.max() - s.min() + 1e-9)

    test_m = compute_metrics(y_test, y_test_pred, y_proba)
    embedding_results.append(metrics_to_row(test_m, {
        'model': model_name, 'representation': 'glove_avg',
        'best_param': best_param, 'dataset_scope': 'politics', 'split': 'test',
    }))

embedding_df = pd.DataFrame(embedding_results)
embedding_df.to_csv(RESULTS_METRICS / 'embedding_results.csv', index=False)
embedding_df
"""),
        md("""## Parte B — Fine-tuning DistilBERT

> **Alcance:** solo subconjunto **politics** (no se entrena sobre el dataset completo)."""),
        code("""
# Parámetros (config única; ajustar si hay limitaciones computacionales)
SAMPLE_FRAC = 1.0
LEARNING_RATES = [2e-5]
BATCH_SIZES = [8]
EPOCHS_LIST = [2]
MAX_LENGTHS = [256]

# Reutilizar checkpoint ya entrenado (evita re-entrenar ~13h en CPU)
REUSE_EXISTING_CHECKPOINT = True
CHECKPOINT_DIR = RESULTS_MODELS / 'distilbert_checkpoints'

print(f'SAMPLE_FRAC={SAMPLE_FRAC}')
print('Alcance DistilBERT: politics solamente')
"""),
        code("""
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
)
from sklearn.metrics import fbeta_score, accuracy_score, precision_recall_fscore_support

def prepare_transformer_input(row, max_chars=3000):
    title = str(row.get('title', ''))
    body = str(row.get('text', ''))
    full = f\"{title} {body}\".strip()
    if len(full) > max_chars:
        paragraphs = body.split('\\n\\n')
        short_body = paragraphs[0] if paragraphs else body[:2000]
        full = f\"{title} {short_body}\".strip()
    return full[:max_chars]

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=max_length)
        self.labels = list(labels)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def compute_transformer_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    f2 = fbeta_score(labels, preds, beta=2, pos_label=1, zero_division=0)
    return {'f2_fake': f2, 'f1_fake': f1[1] if len(f1) > 1 else 0, 'accuracy': accuracy_score(labels, preds)}
"""),
        code("""
# Preparar datos
tr = pol_train.copy()
va = pol_val.copy()
te = pol_test.copy()

if SAMPLE_FRAC < 1.0:
    tr = tr.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE)
    va = va.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE)
    print(f'Muestra reducida: train={len(tr)}, val={len(va)}')

tr['transformer_text'] = tr.apply(prepare_transformer_input, axis=1)
va['transformer_text'] = va.apply(prepare_transformer_input, axis=1)
te['transformer_text'] = te.apply(prepare_transformer_input, axis=1)

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
"""),
        code("""
def find_best_checkpoint(checkpoint_dir):
    \"\"\"Busca el mejor checkpoint guardado por Trainer (load_best_model_at_end).\"\"\"
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    best_path, best_metric = None, -1.0
    for state_file in checkpoint_dir.glob('checkpoint-*/trainer_state.json'):
        import json
        state = json.loads(state_file.read_text(encoding='utf-8'))
        metric = state.get('best_metric')
        path = state.get('best_model_checkpoint')
        if metric is not None and metric >= best_metric and path:
            best_metric = metric
            best_path = Path(path)
    return best_path

transformer_results = []
best_trainer, best_row, best_model = None, None, None
best_f2 = -1

for lr in LEARNING_RATES:
    for bs in BATCH_SIZES:
        for epochs in EPOCHS_LIST:
            for max_len in MAX_LENGTHS:
                print(f'\\n=== lr={lr}, bs={bs}, epochs={epochs}, max_len={max_len} ===')
                train_ds = NewsDataset(tr['transformer_text'], tr['label'], tokenizer, max_len)
                val_ds = NewsDataset(va['transformer_text'], va['label'], tokenizer, max_len)
                test_ds = NewsDataset(te['transformer_text'], te['label'], tokenizer, max_len)

                checkpoint_path = find_best_checkpoint(CHECKPOINT_DIR)
                can_reuse = (
                    REUSE_EXISTING_CHECKPOINT
                    and checkpoint_path is not None
                    and checkpoint_path.exists()
                    and lr == 2e-5 and bs == 8 and epochs == 2 and max_len == 256
                )

                if can_reuse:
                    print(f'Reutilizando checkpoint existente: {checkpoint_path}')
                    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_path))
                    trainer = None
                else:
                    model = AutoModelForSequenceClassification.from_pretrained(
                        'distilbert-base-uncased', num_labels=2
                    )
                    args = TrainingArguments(
                        output_dir=str(CHECKPOINT_DIR),
                        learning_rate=lr,
                        per_device_train_batch_size=bs,
                        per_device_eval_batch_size=bs,
                        num_train_epochs=epochs,
                        eval_strategy='epoch',
                        save_strategy='epoch',
                        load_best_model_at_end=True,
                        metric_for_best_model='f2_fake',
                        greater_is_better=True,
                        logging_steps=50,
                        seed=RANDOM_STATE,
                        report_to='none',
                    )
                    trainer = Trainer(
                        model=model, args=args,
                        train_dataset=train_ds, eval_dataset=val_ds,
                        compute_metrics=compute_transformer_metrics,
                        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
                    )
                    trainer.train()
                    model = trainer.model

                pred = Trainer(model=model).predict(test_ds)
                y_pred = np.argmax(pred.predictions, axis=-1)
                y_proba = torch.softmax(torch.tensor(pred.predictions), dim=-1)[:, 1].numpy()
                test_m = compute_metrics(te['label'], y_pred, y_proba)
                row = metrics_to_row(test_m, {
                    'model': 'distilbert-base-uncased',
                    'learning_rate': lr, 'batch_size': bs,
                    'epochs': epochs, 'max_length': max_len,
                    'sample_frac': SAMPLE_FRAC,
                    'dataset_scope': 'politics', 'split': 'test',
                    'reused_checkpoint': can_reuse,
                })
                transformer_results.append(row)
                print('Métricas test:', {k: round(v, 4) for k, v in test_m.items() if k != 'roc_auc'})
                if test_m['f2_fake'] > best_f2:
                    best_f2 = test_m['f2_fake']
                    best_trainer = trainer
                    best_row = row
                best_model = model

transformer_df = pd.DataFrame(transformer_results)
transformer_df.to_csv(RESULTS_METRICS / 'transformer_results.csv', index=False)
if best_model is not None:
    best_model.save_pretrained(str(RESULTS_MODELS / 'best_distilbert'))
    tokenizer.save_pretrained(str(RESULTS_MODELS / 'best_distilbert'))
    print('Modelo guardado en results/models/best_distilbert')
transformer_df.sort_values('f2_fake', ascending=False)
"""),
        md("""## Conclusiones

- Los embeddings GloVe capturan semántica pero pueden perder señales de estilo.
- DistilBERT puede capturar contexto más rico; usar `SAMPLE_FRAC < 1.0` si hay limitaciones de hardware.
- Ningún modelo verifica hechos; aprenden patrones del dataset."""),
    ]
    save("04_embeddings_and_transformers.ipynb", cells)


def notebook_05():
    cells = [
        md("""# Notebook 05 — Importancia de Atributos y Análisis de Adjetivos

## Objetivo

Interpretar qué términos lingüísticos asocia el mejor modelo lineal con fake vs real, y validar hipótesis sobre adjetivos.

> Los coeficientes reflejan patrones del dataset, no veracidad factual."""),
        code(SETUP + """

import joblib
from collections import Counter

from src.modeling import get_text_col
"""),
        md("## 1. Cargar mejor modelo baseline (politics)"),
        code("""
import json

from src.modeling import get_text_col

pol_train = pd.read_csv(DATA_PROCESSED / 'politics_train.csv')

pipe = joblib.load(RESULTS_MODELS / 'best_baseline_politics.joblib')
with open(RESULTS_MODELS / 'best_baseline_politics_config.json', encoding='utf-8') as f:
    best_cfg = json.load(f)

TEXT_COL = get_text_col(best_cfg['text_field'], best_cfg['stopwords'])
clf = pipe.named_steps['clf']
if not hasattr(clf, 'coef_'):
    raise ValueError(
        f"El mejor modelo ({best_cfg['model']}) no expone coeficientes lineales."
    )

print(
    'Modelo cargado para análisis de coeficientes:',
    f"{best_cfg['model']} + {best_cfg['vectorizer']} ({TEXT_COL})",
)
"""),
        md("## 2. Coeficientes más importantes"),
        code("""
feature_names = pipe.named_steps['vec'].get_feature_names_out()
coefs = pipe.named_steps['clf'].coef_[0]
importance = pd.DataFrame({'term': feature_names, 'coefficient': coefs})
importance['abs_coef'] = importance['coefficient'].abs()
importance = importance.sort_values('abs_coef', ascending=False)

top_fake = importance.nlargest(30, 'coefficient')[['term', 'coefficient']]
top_real = importance.nsmallest(30, 'coefficient')[['term', 'coefficient']]

feature_importance = pd.concat([
    top_fake.assign(direction='fake'),
    top_real.assign(direction='real'),
])
feature_importance.to_csv(RESULTS_METRICS / 'feature_importance.csv', index=False)
feature_importance.head(10)
"""),
        code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, data, title, color in [
    (axes[0], top_fake.head(15), 'Términos asociados a FAKE', '#e74c3c'),
    (axes[1], top_real.head(15).assign(coefficient=lambda x: x['coefficient'].abs()), 'Términos asociados a REAL', '#2ecc71'),
]:
    sns.barplot(data=data.iloc[::-1], y='term', x='coefficient', color=color, ax=ax)
    ax.set_title(title)
save_figure(fig, RESULTS_FIGURES / 'feature_importance_top_terms.png')
plt.show()
"""),
        md("""## 3. Términos frecuentes y n-gramas por clase

Buscamos expresiones sensacionalistas/emocionales (fake) vs estilo periodístico formal (real, ej. marcas Reuters)."""),
        code("""
def top_ngrams_by_class(df, col, label, n=20):
    texts = df[df['label'] == label][col].fillna('').astype(str)
    all_ngrams = []
    for t in texts:
        tokens = t.split()
        all_ngrams.extend([' '.join(tokens[i:i+2]) for i in range(len(tokens)-1)])
    return Counter(all_ngrams).most_common(n)

fake_bi = top_ngrams_by_class(pol_train, TEXT_COL, 1)
real_bi = top_ngrams_by_class(pol_train, TEXT_COL, 0)
print('Bigramas frecuentes FAKE:', fake_bi[:10])
print('Bigramas frecuentes REAL:', real_bi[:10])
"""),
        md("## 4. Análisis de adjetivos con spaCy"),
        code("""
import spacy
import subprocess
import sys

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    subprocess.run([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'], check=True)
    nlp = spacy.load('en_core_web_sm')

def extract_adjectives(texts, sample_size=3000):
    adj_counter = Counter()
    for doc in nlp.pipe(texts.head(sample_size).astype(str), batch_size=100):
        for token in doc:
            if token.pos_ == 'ADJ':
                adj_counter[token.lemma_.lower()] += 1
    return adj_counter

fake_adj = extract_adjectives(pol_train[pol_train['label']==1][TEXT_COL])
real_adj = extract_adjectives(pol_train[pol_train['label']==0][TEXT_COL])

adj_rows = []
for adj, cnt in fake_adj.most_common(50):
    adj_rows.append({'adjective': adj, 'count': cnt, 'class': 'fake'})
for adj, cnt in real_adj.most_common(50):
    adj_rows.append({'adjective': adj, 'count': cnt, 'class': 'real'})

adj_df = pd.DataFrame(adj_rows)
adj_df.to_csv(RESULTS_METRICS / 'adjectives_by_class.csv', index=False)
adj_df.head(10)
"""),
        code("""
# Gráfico comparativo de adjetivos
fake_top = pd.DataFrame(fake_adj.most_common(15), columns=['adjective', 'fake_count'])
real_top = pd.DataFrame(real_adj.most_common(15), columns=['adjective', 'real_count'])
adj_plot = fake_top.merge(real_top, on='adjective', how='outer').fillna(0)

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(adj_plot))
width = 0.35
ax.barh(x - width/2, adj_plot['fake_count'], width, label='Fake', color='#e74c3c')
ax.barh(x + width/2, adj_plot['real_count'], width, label='Real', color='#2ecc71')
ax.set_yticks(x)
ax.set_yticklabels(adj_plot['adjective'])
ax.set_xlabel('Frecuencia')
ax.set_title('Adjetivos más frecuentes por clase')
ax.legend()
save_figure(fig, RESULTS_FIGURES / 'adjectives_by_class.png')
plt.show()
"""),
        md("""## Conclusiones

- Los términos con coeficiente positivo se asocian estadísticamente con fake en este dataset.
- Los adjetivos pueden diferir en tono emocional/sensacionalista vs formal.
- Estas asociaciones son específicas del corpus y no implican verificación factual."""),
    ]
    save("05_feature_importance.ipynb", cells)


def notebook_06():
    cells = [
        md("""# Notebook 06 — Análisis de Errores

## Objetivo

Analizar manualmente errores del mejor modelo clásico (y transformer si está disponible) sobre el subconjunto político.

- **FP (Falso Positivo):** noticia real clasificada como fake
- **FN (Falso Negativo):** noticia fake clasificada como real"""),
        code(SETUP + """

import joblib
from sklearn.metrics import confusion_matrix

ERROR_CATEGORIES = [
    'lenguaje_neutral_en_fake',
    'estilo_sensacionalista_en_real',
    'texto_corto_o_ambiguo',
    'nombres_politicos_frecuentes',
    'sesgo_de_fuente',
    'ironia_o_sarcasmo',
    'informacion_parcialmente_verdadera',
    'fuente_ambigua',
    'otro',
]
"""),
        md("## 1. Cargar modelo y generar predicciones"),
        code("""
pol_test = pd.read_csv(DATA_PROCESSED / 'politics_test.csv')
pipe = joblib.load(RESULTS_MODELS / 'best_baseline_politics.joblib')

from src.modeling import get_text_col

config_path = RESULTS_MODELS / 'best_baseline_politics_config.json'
if config_path.exists():
    best_cfg = pd.read_json(config_path, typ='series')
    TEXT_COL = get_text_col(best_cfg['text_field'], best_cfg['stopwords'])
else:
    TEXT_COL = 'clean_full_text_without_stopwords'

X_test = pol_test[TEXT_COL].fillna('')
y_true = pol_test['label']
y_pred = pipe.predict(X_test)

pol_test = pol_test.copy()
pol_test['prediction'] = y_pred
pol_test['error_type'] = np.where(
    (y_true == 0) & (y_pred == 1), 'FP',
    np.where((y_true == 1) & (y_pred == 0), 'FN', 'correct')
)
errors = pol_test[pol_test['error_type'].isin(['FP', 'FN'])].copy()
print(f'Total errores: {len(errors)} (FP={len(errors[errors.error_type==\"FP\"])}, FN={len(errors[errors.error_type==\"FN\"])})')
"""),
        md("## 2. Selección de muestra (≥30 errores)"),
        code("""
def assign_category(row):
    \"\"\"Heurística inicial para categorizar errores; revisar manualmente.\"\"\"
    text = (str(row['title']) + ' ' + str(row['text'])).lower()
    wc = len(text.split())
    if wc < 80:
        return 'texto_corto_o_ambiguo'
    sensational = any(w in text for w in ['shocking', 'disturbing', 'bombshell', 'slams', 'destroyed', 'embarrassing'])
    formal = any(w in text for w in ['reuters', 'according to', 'officials said', 'spokesman'])
    political = any(w in text for w in ['trump', 'clinton', 'obama', 'congress', 'senate'])
    if row['error_type'] == 'FP' and sensational:
        return 'estilo_sensacionalista_en_real'
    if row['error_type'] == 'FN' and formal:
        return 'lenguaje_neutral_en_fake'
    if formal:
        return 'sesgo_de_fuente'
    if political:
        return 'nombres_politicos_frecuentes'
    if sensational:
        return 'estilo_sensacionalista_en_real'
    return 'otro'

n_fp = min(15, len(errors[errors['error_type'] == 'FP']))
n_fn = min(15, len(errors[errors['error_type'] == 'FN']))
sample_fp = errors[errors['error_type'] == 'FP'].head(n_fp)
sample_fn = errors[errors['error_type'] == 'FN'].head(n_fn)
sample = pd.concat([sample_fp, sample_fn])

sample['category'] = sample.apply(assign_category, axis=1)
sample['comment'] = sample.apply(
    lambda r: f\"Error {r['error_type']}: posible causa relacionada con {r['category'].replace('_', ' ')}.\",
    axis=1
)
sample['text_fragment'] = sample['text'].astype(str).str[:300]
"""),
        code("""
error_export = sample[[
    'title', 'text_fragment', 'label', 'prediction', 'error_type', 'category', 'comment'
]].reset_index().rename(columns={'index': 'news_id', 'label': 'true_label'})
error_export.to_csv(RESULTS_ERROR / 'error_examples.csv', index=False)
error_export.head(10)
"""),
        md("## 3. Distribución de categorías de error"),
        code("""
cat_dist = sample['category'].value_counts().reset_index()
cat_dist.columns = ['category', 'count']
cat_dist.to_csv(RESULTS_ERROR / 'error_category_distribution.csv', index=False)

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=cat_dist, x='count', y='category', ax=ax, color='#3498db')
ax.set_title('Distribución de categorías de error (muestra)')
ax.set_xlabel('Cantidad')
save_figure(fig, RESULTS_FIGURES / 'error_category_distribution.png')
plt.show()
"""),
        md("## 4. Comparación con transformer (si disponible)"),
        code("""
transformer_path = RESULTS_METRICS / 'transformer_results.csv'
if transformer_path.exists():
    tr_res = pd.read_csv(transformer_path)
    best_tr = tr_res.sort_values('f2_fake', ascending=False).iloc[0]
    print('Mejor transformer (test):')
    print(best_tr[['model', 'f2_fake', 'accuracy', 'recall_fake']].to_string())

    baseline_df = pd.read_csv(RESULTS_METRICS / 'baseline_results.csv')
    if 'split' in baseline_df.columns:
        baseline_df = baseline_df[baseline_df['split'] == 'test']
    baseline_best = baseline_df[baseline_df['dataset_scope'] == 'politics'].iloc[0]
    compare_df = pd.DataFrame([
        {'model_type': 'baseline', 'f2_fake': baseline_best['f2_fake'], 'accuracy': baseline_best['accuracy']},
        {'model_type': 'transformer', 'f2_fake': best_tr['f2_fake'], 'accuracy': best_tr['accuracy']},
    ])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=compare_df, x='model_type', y='f2_fake', ax=ax)
    ax.set_title('F2 fake: baseline vs transformer')
    save_figure(fig, RESULTS_FIGURES / 'error_analysis_baseline_vs_transformer.png')
    plt.show()
else:
    print('Resultados de transformer no disponibles; ejecutar notebook 04 primero.')
"""),
        md("""## Conclusiones

- Los FP suelen involucrar noticias reales con lenguaje llamativo o nombres políticos frecuentes.
- Los FN pueden deberse a fake news con tono neutral o imitación de estilo periodístico.
- El modelo aprende patrones del dataset; los errores revelan casos donde esos patrones no aplican.
- Revisar `results/error_analysis/error_examples.csv` para el análisis cualitativo detallado."""),
        md("## 5. Consolidación de resultados"),
        code("""
from src.metrics import consolidate_results

all_results = consolidate_results(
    baseline_path=RESULTS_METRICS / 'baseline_results.csv',
    output_path=RESULTS_METRICS / 'all_model_results.csv',
)
print(f'Resultados consolidados: {len(all_results)} filas')
all_results.head(10)
"""),
    ]
    save("06_error_analysis.ipynb", cells)


def notebook_consolidate_cell():
    """Add consolidation to nb06 or separate - we'll add to nb06 via extra cell in script"""
    pass


if __name__ == "__main__":
    notebook_01()
    notebook_02()
    notebook_03()
    notebook_04()
    notebook_05()
    notebook_06()
    print("All notebooks generated")


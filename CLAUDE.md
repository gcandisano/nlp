# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

University NLP assignment (TP) that trains and compares supervised classifiers to separate **fake** vs **real** news by *linguistic patterns*. The models learn statistical correlations in the corpus — they do **not** verify factual truth. The research emphasis is interpretability: *which* text features discriminate, not just which model scores best.

Working language is **Spanish** — docs, code comments, and printed output are in Spanish. Match this when editing.

Full experimental design: `docs/Informe.md`. Per-experiment methodology decisions: `docs/adr/`.

## Commands

```bash
uv sync                       # install deps (Python 3.12)
uv run setup                  # download NLTK + spaCy resources, create output dirs (requires data/raw CSVs)
uv run jupyter lab            # run the pipeline notebooks

uv sync --group dev           # add dev tools (ruff)
uv run ruff check src/nlp     # lint (line-length 88; notebooks/ and .venv excluded)
uv run ruff format src/nlp    # format (double quotes, space indent)
```

There is **no test suite**. Validation happens by running notebooks 01→07 in order and inspecting `results/`.

The dataset is not in the repo: download [Fake and Real News](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/) and place `Fake.csv` / `True.csv` in `data/raw/` before `uv run setup`.

## Architecture

Two layers: a reusable Python package (`src/nlp/`) holds all logic; the notebooks (`notebooks/`) are thin orchestration that import from it. **Put reusable logic in `src/nlp/`, not in notebook cells.** ruff does not lint notebooks.

**`src/nlp/` modules:**
- `paths.py` — single source of truth for all filesystem paths, `RANDOM_STATE = 42`, and the politics-subset subject lists. Import paths from here; never hardcode directories.
- `preprocessing.py` — text cleaning (`clean_text`, URL→`[URL]`), date parsing (multi-format), deduplication, and `run_preprocessing_pipeline()` which produces the temporal splits. Generates `clean_*` columns in both `_with_stopwords` / `_without_stopwords` variants.
- `modeling.py` — baseline grid engine: vectorizer + sklearn pipeline builders, `run_baseline_grid` (selects hyperparams on val), `evaluate_best_configs_on_test` (test once), `get_linear_feature_weights` (coefficients for interpretability).
- `metrics.py` — `compute_metrics` (the canonical metric dict) and `consolidate_results` which merges baseline/embedding/transformer CSVs into `results/metrics/all_model_results.csv`.
- `plotting.py` — matplotlib/seaborn helpers; `save_figure` writes to `results/figures/`.
- `bootstrap.py` / `setup.py` — the `setup` console script (validates raw data present, downloads NLP resources).

**Pipeline (run in order; later notebooks consume earlier outputs).** Notebook numbers ≠ experiment numbers: notebooks 01–02 are EDA/preprocessing, then notebook `0(N+3)` implements Experimento N.
1. `01_eda.ipynb` — exploratory analysis → `results/figures/`
2. `02_preprocessing_and_splits.ipynb` — runs the preprocessing pipeline → `data/processed/{politics,full}_{train,val,test}.csv`
3. `03_baseline_models.ipynb` (Exp 1) — BoW/TF-IDF × LR/NB/LinearSVC grid + source ablation → `results/metrics/baseline_results.csv`, `results/models/`
4. `04_linguistic_features.ipynb` (Exp 2) — **scaffold only**: 8 interpretable features (spaCy POS/NER + VADER) → LogisticRegression, plus title/body/combined sub-experiment. Code cells are TODO; `vaderSentiment` and an optional `src/nlp/features.py` are **not created yet**.
5. `05_embeddings_and_transformers.ipynb` (Exp 3) — GloVe/Word2Vec + DistilBERT/BERT → `embedding_results.csv`, `transformer_results.csv`
6. `06_feature_importance.ipynb` (Exp 4) — linear coefficients, adjectives per class
7. `07_error_analysis.ipynb` (Exp 5) — manual FP/FN taxonomy, model comparison

Data flow: `data/raw/` → `data/processed/` (splits) → `results/` (figures, metrics, models, error_analysis). `data/embeddings/` caches downloaded GloVe vectors.

## Conventions that the whole codebase depends on

- **Label encoding: `fake = 1` (positive class), `real = 0`.** All precision/recall/F-beta use `pos_label=1`.
- **Primary metric is `f2_fake`** (F2-score of the fake class, β=2). It is the selection criterion in the grid and the comparison metric across every experiment — a false negative (fake passed as real) is treated as costlier than a false positive. Don't switch to F1/accuracy for model selection.
- **Temporal split, not random**: 70/15/15 ordered by publication date (train = oldest, test = newest). Validation/test may be class-imbalanced relative to train; this is expected. Hyperparameters are selected on **validation only**; test is evaluated exactly once.
- **Two dataset scopes**: `politics` (real=`politicsNews`, fake=`politics`) is the main experiment — it controls for the strong topical bias in `subject`. `full` is a sensitivity control for baselines only. The `subject` column is **never** used as a feature.
- **Source ablation** (notebook 03): retrain best model with source tokens (`reuters`, `ap`, `afp`) normalized to `[SOURCE]`. A large F2 drop means the dataset encodes source identity, in which case later experiments should run on source-normalized text.
- Map experiment → notebook → ADR: each `docs/adr/experimento-0N-*.md` documents the *why* behind notebook `0(N+2)`. Read the relevant ADR before changing methodology.

## Gotchas

- Notebook **03** trains a large grid (3 models × 2 vectorizers × 3 text fields × 2 stopword settings × 2 n-gram settings × 3 `max_features`) — it is slow. `run_baseline_grid` accepts `max_combos` to cap it during development.
- Notebook **05** downloads GloVe (~850 MB) on first run. With limited GPU/RAM, lower `SAMPLE_FRAC` for DistilBERT (e.g. `0.1`).
- spaCy model is **`en_core_web_sm`** everywhere (pinned in `pyproject.toml`, loaded in `setup.py` and notebook 06; documented in Informe/ADRs). This is **deliberate, not a gap**: the features use POS tagging, NER, and sentence segmentation (≈identical accuracy across `sm`/`lg`), not word vectors — don't "upgrade" to `lg`, it's a ~560 MB download for no benefit here.
- GloVe is **`glove.6B.100d`** by design (not `840B/300d`): 100d averaged document vectors suffice for this lexical binary task. ADR-03 documents the choice.
- After installing/updating deps, restart the Jupyter kernel if it was already open.
- `config_from_row` in `modeling.py` uses `eval()` to parse the `ngram_range` string round-tripped through CSV — keep that column as a tuple-literal string (e.g. `"(1, 2)"`).

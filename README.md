# TP NLP: Clasificación de Fake News por Patrones Lingüísticos

Trabajo práctico que entrena y evalúa modelos de clasificación supervisada para distinguir noticias **fake** y **real** según patrones lingüísticos del corpus. Los modelos **no verifican la verdad factual**; aprenden correlaciones estadísticas del dataset.

El análisis completo (datos, experimentos, métricas y limitaciones) está en [docs/Informe.md](docs/Informe.md). Las decisiones metodológicas de cada experimento —qué modelo y representación se eligieron, con qué protocolo de evaluación y por qué— están registradas como **ADRs** en [docs/adr/](docs/adr/): hay un ADR por experimento.

## Estructura del proyecto

```text
nlp/
├── src/nlp/              # Paquete Python (preprocesamiento, modelos, métricas)
├── notebooks/            # Pipeline del TP (ejecutar 01 → 07 en orden)
├── data/
│   ├── raw/              # CSV originales (Fake.csv, True.csv)
│   └── processed/        # Splits temporales preprocesados
├── results/              # Figuras, métricas, modelos, análisis de errores
├── docs/                 # Informe y ADRs
├── pyproject.toml
└── uv.lock
```

## Requisitos

- Python 3.12
- [uv](https://docs.astral.sh/uv/) (recomendado) o pip
- Dataset [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/) en `data/raw/Fake.csv` y `data/raw/True.csv`

## Cómo ejecutar

1. Descargar el dataset y colocar `Fake.csv` y `True.csv` en `data/raw/`.

2. Instalar y preparar el entorno.

**Con uv:**

```bash
uv sync
uv run setup
```

**Con pip:**

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
setup
```

3. Ejecutar el pipeline en orden:

```bash
jupyter lab
```

Abrir y correr los notebooks en orden. Los notebooks **01** y **02** preparan los datos; del **03** al **07** cada uno implementa un experimento documentado en su ADR (ver [docs/adr/](docs/adr/)):

| Notebook | Contenido | ADR del experimento |
| :------- | :-------- | :------------------ |
| `01_eda.ipynb` | **Análisis exploratorio**: longitudes, distribución temática, léxico por clase y detección de duplicados | — |
| `02_preprocessing_and_splits.ipynb` | **Preprocesamiento y partición temporal** 70/15/15 → `data/processed/` | — |
| `03_baseline_models.ipynb` | Modelos tradicionales BoW/TF-IDF + ablación de marcadores de fuente | [Experimento 1](docs/adr/experimento-01-baseline-modelos-tradicionales.md) |
| `04_linguistic_features.ipynb` | Features lingüísticas interpretables (spaCy + VADER) — _scaffold, pendiente de implementar_ | [Experimento 2](docs/adr/experimento-02-features-linguisticas.md) |
| `05_embeddings_and_transformers.ipynb` | Embeddings (GloVe / Word2Vec) y Transformers (DistilBERT / BERT) | [Experimento 3](docs/adr/experimento-03-embeddings-transformers.md) |
| `06_feature_importance.ipynb` | Importancia de atributos y análisis de adjetivos por clase | [Experimento 4](docs/adr/experimento-04-importancia-atributos.md) |
| `07_error_analysis.ipynb` | Análisis manual de errores (falsos positivos / falsos negativos) | [Experimento 5](docs/adr/experimento-05-analisis-errores.md) |

Los resultados quedan en `results/`. El resumen final: `results/metrics/all_model_results.csv`.

## Notas

- El notebook **03** entrena una grilla amplia de baselines; puede tardar bastante.
- El notebook **05** descarga GloVe (~850 MB) la primera vez. Si hay poca GPU/RAM, reducir `SAMPLE_FRAC` en DistilBERT (ej. `0.1`).
- El split es **temporal** (no aleatorio); validación y test pueden quedar levemente desbalanceados respecto a train.
- Tras instalar dependencias, reiniciar el kernel de Jupyter si ya estaba abierto.
- Para desarrollo con uv: `uv sync --group dev` instala ruff (`uv run ruff check src/nlp`).

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
│   ├── processed/        # Splits temporales (.parquet + .csv)
│   └── embeddings/       # Cache GloVe (.kv) y embeddings por split (.npz)
├── results/              # Figuras, métricas, modelos, análisis de errores
├── docs/                 # Informe y ADRs
├── pyproject.toml
└── uv.lock
```

## Requisitos

- Python 3.12
- [uv](https://docs.astral.sh/uv/) (recomendado) o pip
- Dataset [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/) en `data/raw/Fake.csv` y `data/raw/True.csv`

## Cómo ejecutar (de principio a fin)

### 1. Datos

Descargar el [dataset de Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/) y colocar los archivos en `data/raw/`:

```text
data/raw/Fake.csv
data/raw/True.csv
```

### 2. Entorno

**Con uv (recomendado):**

```bash
uv sync
uv run setup
```

`setup` valida que existan los CSV, descarga recursos NLTK/spaCy y crea las carpetas de salida.

**Con pip:**

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
setup
```

### 3. Pipeline de notebooks

```bash
uv run jupyter lab
```

Abrir `notebooks/` y ejecutar **en orden**, reiniciando el kernel si acabás de instalar dependencias:

| Paso | Notebook | Qué produce |
| :--- | :------- | :---------- |
| 1 | `01_eda.ipynb` | Figuras exploratorias en `results/figures/` |
| 2 | `02_preprocessing_and_splits.ipynb` | Splits en `data/processed/{politics,full}_{train,val,test}.{parquet,csv}` |
| 3 | `03_baseline_models.ipynb` | `results/metrics/baseline_results.csv`, modelos en `results/models/` |
| 4 | `04_linguistic_features.ipynb` | `linguistic_features_results.csv`, cache en `data/processed/linguistic_features_*.parquet` |
| 5 | `05_embeddings_and_transformers.ipynb` | `embedding_results.csv`, `transformer_results.csv`, `transformer_val_search.csv`; cache GloVe/Word2Vec en `data/embeddings/` |
| 6 | `06_feature_importance.ipynb` | `feature_importance.csv`, `adjectives_by_class.csv` |
| 7 | `07_error_analysis.ipynb` | `results/metrics/all_model_results.csv`, análisis de errores |

Cada notebook del 03 al 07 corresponde a un experimento documentado en [docs/adr/](docs/adr/).

**Salida final:** `results/metrics/all_model_results.csv` (tabla comparativa de modelos).

### Modo desarrollo (`NLP_DEV_MODE`)

Para iterar más rápido **sin** correr la grilla completa ni entrenar DistilBERT sobre todo el train:

```bash
NLP_DEV_MODE=1 uv run jupyter lab
```

Con `NLP_DEV_MODE=1` (lee la variable al importar `nlp.paths`):

| Notebook | Efecto |
| :------- | :----- |
| **03** | Grilla `politics` limitada a 20 combinaciones; se omite `full_dataset` |
| **04** | Muestra al 10% por split (extracción spaCy más rápida) |
| **05** | `SAMPLE_FRAC=0.1` en DistilBERT (10% solo de **train**); grilla HP acotada a 4 combos |

No usar este modo para los **resultados finales del TP**: la metodología y las métricas reportadas deben salir de una corrida completa sin `NLP_DEV_MODE`.

### Corrida completa (recomendada para entregar)

```bash
uv sync
uv run setup
uv run jupyter lab
```

Luego ejecutar notebooks **01 → 07** en orden, sin `NLP_DEV_MODE`. Tiempos orientativos:

- **03** — grilla de baselines (la parte más lenta en CPU).
- **05** — primera vez descarga GloVe (~850 MB) y entrena Word2Vec en train; corridas siguientes usan cache. DistilBERT en CPU puede tardar muchas horas; para entrega completa conviene GPU (local o Colab). Con poca RAM podés usar `NLP_DEV_MODE=1` para iterar.

## Detalle por notebook

| Notebook | Contenido | ADR del experimento |
| :------- | :-------- | :------------------ |
| `01_eda.ipynb` | **Análisis exploratorio**: longitudes, distribución temática, léxico por clase y detección de duplicados | — |
| `02_preprocessing_and_splits.ipynb` | **Preprocesamiento y partición temporal** 70/15/15 → `data/processed/` | — |
| `03_baseline_models.ipynb` | Modelos tradicionales BoW/TF-IDF | [Experimento 1](docs/adr/experimento-01-baseline-modelos-tradicionales.md) |
| `04_linguistic_features.ipynb` | Features lingüísticas interpretables (spaCy + VADER, LR sin scaler) | [Experimento 2](docs/adr/experimento-02-features-linguisticas.md) |
| `05_embeddings_and_transformers.ipynb` | Embeddings (GloVe + Word2Vec) y DistilBERT con grilla HP | [Experimento 3](docs/adr/experimento-03-embeddings-transformers.md) |
| `06_feature_importance.ipynb` | Importancia de atributos y análisis de adjetivos por clase | [Experimento 4](docs/adr/experimento-04-importancia-atributos.md) |
| `07_error_analysis.ipynb` | Análisis manual de errores (falsos positivos / falsos negativos) | [Experimento 5](docs/adr/experimento-05-analisis-errores.md) |

## Notas

- El notebook **03** entrena una grilla amplia de baselines; puede tardar bastante. En desarrollo usar `NLP_DEV_MODE=1`.
- El notebook **05** guarda checkpoints por combinación de hiperparámetros en `results/models/distilbert_checkpoints/` y exporta el mejor modelo a `results/models/best_distilbert`.
- El split es **temporal** (no aleatorio); validación y test pueden quedar levemente desbalanceados respecto a train.
- Tras instalar o actualizar dependencias, reiniciar el kernel de Jupyter si ya estaba abierto.
- Desarrollo con uv: `uv sync --group dev` instala ruff (`uv run ruff check src/nlp`).

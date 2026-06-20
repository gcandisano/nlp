# TP NLP: Clasificación de Fake News por Patrones Lingüísticos

Trabajo práctico que entrena y evalúa modelos de clasificación supervisada para distinguir noticias **fake** y **real** según patrones lingüísticos del corpus. Los modelos **no verifican la verdad factual**; aprenden correlaciones estadísticas del dataset.

El análisis completo (datos, experimentos, métricas y limitaciones) está en [docs/Informe.md](docs/Informe.md).

## Estructura del proyecto

```text
nlp/
├── src/nlp/              # Paquete Python (preprocesamiento, modelos, métricas)
├── notebooks/            # Pipeline del TP (ejecutar 01 → 06 en orden)
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

Abrir y correr `notebooks/01_eda.ipynb` → `06_error_analysis.ipynb`.

Los resultados quedan en `results/`. El resumen final: `results/metrics/all_model_results.csv`.

## Notas

- El notebook **03** entrena una grilla amplia de baselines; puede tardar bastante.
- El notebook **04** descarga GloVe (~850 MB) la primera vez. Si hay poca GPU/RAM, reducir `SAMPLE_FRAC` en DistilBERT (ej. `0.1`).
- El split es **temporal** (no aleatorio); validación y test pueden quedar levemente desbalanceados respecto a train.
- Tras instalar dependencias, reiniciar el kernel de Jupyter si ya estaba abierto.
- Para desarrollo con uv: `uv sync --group dev` instala ruff (`uv run ruff check src/nlp`).

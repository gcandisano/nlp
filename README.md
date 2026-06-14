# TP NLP: Clasificación de Fake News por Patrones Lingüísticos

## Objetivo

Este trabajo práctico entrena y evalúa modelos de clasificación supervisada que distinguen noticias **fake** y **real** según **patrones lingüísticos aprendidos del dataset**.

> **Importante:** Los modelos **no verifican la verdad factual** de una noticia. Aprenden correlaciones estadísticas presentes en los datos de entrenamiento, que pueden reflejar estilo, fuente, tema o período temporal.

## Dataset

Se utiliza el dataset público de Kaggle **"Fake and Real News Dataset"**:

- `data/raw/Fake.csv` — noticias falsas (`label = 1`)
- `data/raw/True.csv` — noticias reales (`label = 0`)

Columnas: `title`, `text`, `subject`, `date`.

## Estructura del proyecto

```text
nlp/
├── data/
│   ├── raw/           # CSV originales
│   └── processed/     # Splits temporales preprocesados
├── notebooks/         # Notebooks del TP (ejecutar en orden)
├── results/
│   ├── figures/       # Gráficos
│   ├── metrics/       # CSVs de métricas
│   ├── models/        # Mejor modelo clásico (joblib)
│   └── error_analysis/
├── src/               # Utilidades compartidas
├── requirements.txt
└── README.md
```

## Cómo ejecutar

1. Crear entorno virtual e instalar dependencias:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger_eng')"
```

2. Verificar que existan `data/raw/Fake.csv` y `data/raw/True.csv`.

3. Ejecutar los notebooks **en orden**:

| Notebook | Contenido |
|----------|-----------|
| `01_eda.ipynb` | Análisis exploratorio |
| `02_preprocessing_and_splits.ipynb` | Preprocesamiento y split temporal |
| `03_baseline_models.ipynb` | Modelos clásicos (BoW, TF-IDF) |
| `04_embeddings_and_transformers.ipynb` | Embeddings + DistilBERT |
| `05_feature_importance.ipynb` | Importancia de atributos y adjetivos |
| `06_error_analysis.ipynb` | Análisis manual de errores + consolidación |

**Alternativa:** el script `scripts/run_preprocessing.py` genera los splits en `data/processed/` sin ejecutar el notebook 02.

4. Los artefactos se guardan automáticamente en `data/processed/` y `results/`.

5. Tras ejecutar los notebooks 03-06, consolidar resultados (también se hace al final del notebook 06):

```bash
python scripts/consolidate_results.py
```

### Notas de ejecución

- El notebook **03** ejecuta una grilla completa de baselines (~216 configuraciones × 2 alcances). Puede tardar bastante según el hardware.
- El notebook **04** descarga GloVe (~850 MB) la primera vez. Para DistilBERT, ajustar `SAMPLE_FRAC` (ej. `0.1`) si hay limitaciones de GPU/RAM.
- El split temporal puede producir **ligero desbalance** en validación/test respecto a train (documentado en notebook 02). No se aplica resampling para preservar el criterio temporal.

## Experimentos implementados

### Experimento principal (subconjunto político)

El análisis principal se realiza sobre noticias de temática política para reducir el sesgo por diferencias temáticas entre clases:

- Reales: `politicsNews`
- Falsas: `politics`

Sobre este subconjunto se entrenan y evalúan:

1. Modelos baseline (Logistic Regression, Multinomial Naive Bayes, Linear SVM)
2. Comparación título vs cuerpo vs título+cuerpo
3. Con stopwords vs sin stopwords
4. Embeddings preentrenados (GloVe)
5. Fine-tuning de DistilBERT
6. Análisis de importancia de atributos
7. Análisis de adjetivos por clase
8. Análisis de errores

### Experimento complementario (dataset completo)

Se repiten los modelos baseline sobre el dataset completo y se comparan métricas contra el subconjunto político.

## Métrica principal: F2-score (clase fake)

Se utiliza **F2-score de la clase fake** como métrica principal para seleccionar el mejor modelo porque:

- Queremos **minimizar falsos negativos** (noticias falsas clasificadas como reales).
- No usamos solo recall porque un modelo que predice todo como fake tendría recall alto pero baja precisión.
- F2 prioriza recall pero sigue penalizando falsos positivos.

## Limitaciones del enfoque

- El modelo **no verifica hechos** contra la realidad.
- El modelo **aprende patrones del dataset**, no reglas de veracidad universal.
- Puede haber **sesgos por fuente, tema y período temporal**.
- Las noticias reales parecen provenir mayormente de **Reuters**, lo que puede introducir marcas de estilo/fuente.
- El **split temporal** mejora la evaluación pero no elimina todos los sesgos.
- Los resultados **no necesariamente generalizan** a noticias actuales, otros idiomas u otros medios.
- Usar `subject` como predictor podría **inflar artificialmente** el rendimiento; por eso no se usa como feature.

## Resultados consolidados

Tras ejecutar todos los notebooks, `results/metrics/all_model_results.csv` combina resultados de baseline, embeddings y transformers, con columna `dataset_scope` (`politics` / `full_dataset`), ordenados por F2-score de la clase fake.

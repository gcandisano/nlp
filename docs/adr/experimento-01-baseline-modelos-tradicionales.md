# ADR — Experimento 1: Clasificación con modelos tradicionales (Baseline)

**Estado:** Aceptado  
**Fecha:** 2025-06  
**Relacionado con:** [Informe.md](../Informe.md) § Experimento 1

## Contexto

El trabajo busca detectar patrones lingüísticos de desinformación en un corpus de ~45.000 noticias en inglés (fake vs. real). Antes de modelos complejos, necesitamos una línea base interpretable y reproducible.

Hallazgos del EDA que condicionan este experimento:

- Las noticias reales concentran `politicsNews` y `worldnews`; las falsas tienen subjects más diversos (`News`, `politics`, `left-news`, etc.). Usar el dataset completo introduce un sesgo temático fuerte.
- El término `"reuters"` aparece con mucha frecuencia en noticias reales: los modelos podrían aprender identidad de fuente en lugar de estilo lingüístico.
- Un falso negativo (fake clasificada como real) es socialmente más costoso que un falso positivo.

## Decisión

### Alcance de datos

- **Experimento principal:** subconjunto político — reales con `subject = politicsNews`, falsas con `subject = politics`. Mantiene comparabilidad temática entre clases.
- **Experimento complementario:** dataset completo, solo para baselines, como control de sensibilidad.
- **Excluir** la columna `subject` como feature en todos los modelos.
- **Split temporal** 70/15/15 (train/val/test) ordenado por fecha de publicación, no aleatorio.

### Representaciones

| Representación | Rol |
| :------------- | :-- |
| **Bag of Words (BoW)** | Punto de referencia mínimo |
| **TF-IDF** | Representación principal: penaliza términos frecuentes en todo el corpus y resalta términos discriminativos por documento |
| **Unigramas + bigramas** | Capturan expresiones compuestas (`"breaking news"`, `"white house"`) que unigramas aislados no representan |

Grilla de vectorización: `max_features` ∈ {10.000, 30.000, 50.000}, `min_df = 2`, n-gramas (1,1) y (1,2).

### Modelos

| Modelo | Hiperparámetros explorados | Motivo |
| :----- | :------------------------- | :----- |
| **Regresión Logística** | `C` ∈ {0.1, 1, 10} | Interpretable (coeficientes por término), rápido, baseline estándar en NLP |
| **Naive Bayes Multinomial** | `alpha` ∈ {0.1, 1} | Clásico para conteos de palabras; complementa enfoques lineales |
| **LinearSVC** | `C` ∈ {0.1, 1, 10} | SVM lineal escalable; `LinearSVC` en lugar de kernel RBF por tamaño del corpus y velocidad |

Se usa `LinearSVC` (no `SVC` con kernel no lineal) por eficiencia en matrices sparse de alta dimensionalidad.

### Campos de texto y preprocesamiento

Se entrena sobre tres campos: `title`, `body` y `full` (título + cuerpo), en dos condiciones de stopwords (con y sin eliminación), para cuantificar el impacto de las stopwords.

Preprocesamiento previo: minúsculas, eliminación de puntuación/números, URLs reemplazadas por `[URL]`.

### Selección de hiperparámetros y evaluación

- Selección **solo con validación** (F2 de la clase fake).
- Evaluación final **una sola vez** en test con la mejor configuración.
- **Métrica principal: F2-score** (β=2, prioriza recall sobre precision).
- Métricas secundarias: accuracy, precision, recall, F1, matriz de confusión, ROC-AUC.

### Sub-experimento: ablación de marcadores de fuente

Tras identificar el mejor modelo, se reentrena en dos condiciones:

- **A:** texto original (incluye `"reuters"`, `"ap"`, `"afp"`, etc.).
- **B:** tokens de fuente normalizados a `[SOURCE]`.

Si el F2 cae significativamente en B, el dataset codifica identidad de fuente y los experimentos siguientes deben usar textos con fuentes normalizadas.

## Alternativas consideradas

| Alternativa | Por qué se descartó |
| :---------- | :------------------ |
| Split aleatorio estratificado | No simula despliegue real (entrenar en pasado, evaluar en futuro) |
| Usar `subject` como feature | Predictor demasiado fuerte; confunde tema con veracidad |
| Solo TF-IDF sin BoW | BoW sirve como referencia para medir ganancia de TF-IDF |
| SVM con kernel RBF | Costoso en ~45k documentos con vocabularios de 50k features |
| F1 como métrica principal | No refleja el costo asimétrico de falsos negativos en detección de fake news |
| Random Forest sobre BoW | Menos interpretable que modelos lineales para el análisis posterior de coeficientes |

## Consecuencias

- Los resultados de este experimento son el **baseline obligatorio** para comparar Experimento 2 (features lingüísticas) y Experimento 3 (embeddings/Transformers).
- La ablación de fuente determina si el preprocesamiento de los experimentos siguientes incluye normalización `[SOURCE]`.
- La grilla amplia (3 modelos × 2 vectorizadores × 3 campos × 2 stopwords × 2 n-gramas × 3 max_features) implica tiempos de entrenamiento largos; es el costo de no asumir hiperparámetros a priori.
- El subconjunto político reduce sesgo temático pero acota la generalización a noticias de ese dominio.

# ADR — Experimento 4: Análisis de importancia de atributos

**Estado:** Aceptado  
**Fecha:** 2025-06  
**Relacionado con:** [Informe.md](../Informe.md) § Experimento 4

## Contexto

Los Experimentos 1–3 reportan métricas agregadas (F2, AUC, etc.) pero el TP prioriza el **análisis lingüístico interpretable**. Necesitamos responder:

- ¿Qué palabras y n-gramas asocia el modelo con fake vs. real?
- ¿Los términos de mayor peso reflejan lenguaje emocional/sensacionalista (Hipótesis 1)?
- ¿Los atributos relevantes son consistentes entre modelos lineales distintos?

Este experimento no entrena modelos nuevos desde cero: **interpreta** los ya entrenados en Exp. 1 (y coeficientes de features en Exp. 2 si están disponibles).

## Decisión

### Modelos analizados

| Tipo | Fuente de importancia | Qué extraemos |
| :--- | :-------------------- | :------------ |
| **Regresión Logística** (pipeline BoW/TF-IDF) | `coef_[0]` del clasificador | Top términos positivos (fake) y negativos (real) |
| **LinearSVC** | `coef_[0]` | Misma lectura que LR |
| **Multinomial Naive Bayes** | `feature_log_prob_[1] - feature_log_prob_[0]` | Log-odds por término; comparable en espíritu a coeficientes |
| **Regresión Logística sobre features lingüísticas** (Exp. 2) | Coeficientes por feature | Valida hipótesis sobre exclamaciones, sentimiento, NER, etc. |

Se usa el **mejor modelo lineal** del subconjunto político según F2 en validación (artefacto guardado en `results/models/`).

### Análisis de adjetivos (validación Hipótesis 1)

Con **spaCy** (`en_core_web_sm`):

1. Extraer adjetivos por clase (fake vs. real) del subconjunto político.
2. Tabla de adjetivos más frecuentes por clase.
3. Comparar carga semántica / sentimiento (VADER o análisis cualitativo de polaridad).

La hipótesis es que fake news usan adjetivos con mayor carga emocional y sensacionalista.

### Análisis de n-gramas

Además de unigramas, se revisan **bigramas** con mayor peso en el modelo lineal seleccionado refiteado con `ngram_range=(1,2)` (el mejor baseline del Exp. 1 resultó **BoW**, no TF-IDF), buscando expresiones sensacionalistas (`"breaking news"`, `"shocking video"`, etc.).

### Visualización

- Tablas top-N términos por signo del coeficiente.
- Gráficos de barras comparativos (fake vs. real).
- Word clouds o gráficos de frecuencia de adjetivos por clase.

### Datos y preprocesamiento

Subconjunto político, split de test reservado. El análisis se hace sobre el vocabulario aprendido en train (sin leakage).

Si la ablación de fuente (Exp. 1) indicó sesgo por `"reuters"`, el modelo interpretado debe ser el entrenado con fuentes normalizadas; de lo contrario, `"reuters"` aparecerá como predictor de "real" y distorsionará la lectura lingüística.

## Alternativas consideradas

| Alternativa | Por qué se descartó |
| :---------- | :------------------ |
| SHAP / LIME sobre Transformers | Costoso y fuera del alcance; Exp. 5 cubre errores cualitativos para modelos de caja negra |
| Permutation importance | Menos directo que coeficientes para modelos lineales ya entrenados |
| Análisis solo en train | Riesgo de sobreajuste; se prioriza vocabulario del pipeline ajustado pero ejemplos del test para adjetivos |
| TF-IDF weights sin modelo | Mide frecuencia, no poder predictivo condicionado al clasificador |
| Análisis en dataset completo | Sesgo temático contamina la interpretación; se usa subconjunto político |

## Consecuencias

- Conecta resultados cuantitativos (F2) con narrativa lingüística del informe.
- Si `"reuters"` o `"ap"` dominan coeficientes negativos, confirma el diagnóstico de sesgo de fuente del EDA.
- La consistencia de términos entre LR, SVM y NB refuerza que los patrones no son artefacto de un solo algoritmo.
- Los hallazgos de adjetivos alimentan la discusión de limitaciones: correlación estilística ≠ verificación factual.
- No aplica a DistilBERT; para ese modelo la interpretabilidad queda limitada al análisis de errores (Exp. 5).

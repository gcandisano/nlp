# ADR — Experimento 3: Clasificación con embeddings y Transformers

**Estado:** Aceptado  
**Fecha:** 2025-06  
**Relacionado con:** [Informe.md](../Informe.md) § Experimento 3

## Contexto

Los Experimentos 1 y 2 usan representaciones bag-of-words o features manuales que ignoran el **contexto** dentro de la oración ("bank" como institución vs. ribera). Queremos medir si modelos con representaciones distribuidas o arquitecturas Transformer mejoran la detección respecto del baseline.

Restricciones del proyecto:

- Recursos computacionales limitados (TP de curso, posible ausencia de GPU).
- El foco del trabajo sigue siendo interpretabilidad lingüística; modelos de caja negra deben justificar su ganancia en F2.
- Se trabaja sobre el **subconjunto político** (misma decisión que Exp. 1).

## Decisión

### Parte A — Embeddings estáticos + clasificador lineal

| Embedding | Configuración | Motivo |
| :-------- | :------------ | :----- |
| **GloVe** | Preentrenado en Wikipedia + Gigaword; en implementación: `glove.6B.100d` (100 dimensiones) | Corpus incluye texto periodístico; buena cobertura de vocabulario del dominio |
| **Word2Vec** | Entrenado sobre el corpus del TP | Evalúa si embeddings específicos del dominio (2015–2017) capturan vocabulario de fake news mejor que vectores genéricos |

**Representación de documento:** promedio de embeddings de palabras del artículo (mean pooling).

**Clasificador:** Regresión Logística o LinearSVC sobre vectores densos, con `StandardScaler` previo.

Se prioriza GloVe preentrenado; Word2Vec propio sirve como contraste de dominio.

### Parte B — Transformers con fine-tuning

| Modelo | Rol | Motivo |
| :----- | :-- | :----- |
| **DistilBERT** (`distilbert-base-uncased`) | Modelo principal | ~97% del rendimiento de BERT-base con 60% de parámetros y 2× velocidad de inferencia |

**Protocolo de entrenamiento:**

- Tokenizer oficial de Hugging Face.
- Texto de entrada: `clean_full_text_without_stopwords` (misma limpieza que embeddings); se respeta la decisión de ablación de fuente del Exp. 1.
- Fine-tuning supervisado de clasificación binaria.
- Grilla de hiperparámetros en **validación** (F2 fake); early stopping por época.
- Hiperparámetros a ajustar: learning rate, batch size, épocas, warmup ratio y scheduler lineal.
- **Test se evalúa una sola vez** con la mejor configuración de validación.

Si hay poca GPU/RAM, `NLP_DEV_MODE=1` submuestrea solo el **train** (ej. 10%); la validación permanece completa.

### Modelos descartados explícitamente

No se usan RoBERTa-large, GPT, LLaMA ni otros modelos de gran escala: el fine-tuning supervisado requiere recursos que exceden el alcance del TP.

**BERT-base** (`bert-base-uncased`) se descartó en la implementación local: el fine-tuning en CPU es impracticable para el alcance del TP (~horas por corrida vs. minutos con DistilBERT). DistilBERT ya cubre la comparación contextual; BERT-base queda como referencia bibliográfica. Opcionalmente, la Parte B puede ejecutarse en **Google Colab con GPU** si se desea extender el experimento.

FastText se menciona en el informe como alternativa teórica; en la implementación actual se prioriza GloVe + Word2Vec por cobertura y simplicidad.

### Evaluación

Mismas métricas que Experimentos 1 y 2. F2 como métrica principal para tabla comparativa unificada (`results/metrics/all_model_results.csv`).

## Alternativas consideradas

| Alternativa | Por qué se descartó |
| :---------- | :------------------ |
| GloVe 840B × 300d | Archivo ~2 GB; `glove.6B.100d` (~850 MB comprimido) es suficiente y más manejable |
| Embeddings + LSTM/GRU | Más complejo de entrenar y tunear sin ganancia clara vs. Transformers en clasificación de texto |
| Solo Transformers sin embeddings estáticos | Los embeddings + LR/SVM aíslan la ganancia del contexto vs. la de representaciones densas preentrenadas |
| RoBERTa-large | Requiere GPU con mucha VRAM; fuera del alcance del proyecto |
| ULMFiT / modelos recurrentes | Transformers son el estándar actual y están disponibles vía Hugging Face |
| Max pooling en lugar de mean pooling | Mean pooling es convención establecida para document embeddings; menos sensible a outliers |
| BERT-base en CPU local | Fine-tuning impracticable sin GPU; DistilBERT cubre el rol contextual. Extensión opcional vía Colab |

## Consecuencias

- GloVe se descarga la primera vez (~850 MB); requiere conectividad y espacio en disco.
- DistilBERT es el candidato a "mejor modelo de caja negra" para comparar en Experimento 5 (análisis de errores).
- Si Transformers no superan significativamente al baseline lineal, se refuerza la conclusión de que los patrones discriminativos en este corpus son léxicos/estilísticos y accesibles con métodos interpretables.
- La pérdida de interpretabilidad se documenta explícitamente: los resultados de Exp. 4 (coeficientes) no aplican directamente a DistilBERT; el análisis de errores (Exp. 5) compensa parcialmente esa limitación.

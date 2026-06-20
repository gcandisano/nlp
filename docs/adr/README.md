# Architecture Decision Records (ADR)

Registro de las decisiones metodológicas del TP: por qué elegimos ciertos modelos, representaciones y protocolos de evaluación en cada experimento.

Cada ADR sigue esta estructura:

| Sección | Contenido |
| :------ | :-------- |
| **Estado** | Propuesto / Aceptado / Supersedido |
| **Contexto** | Problema, restricciones y hallazgos del EDA que motivan la decisión |
| **Decisión** | Qué hacemos y con qué configuración |
| **Alternativas consideradas** | Opciones descartadas y por qué |
| **Consecuencias** | Impacto en resultados, interpretabilidad y experimentos posteriores |

## Índice

| ADR | Experimento |
| :-- | :---------- |
| [experimento-01-baseline-modelos-tradicionales.md](./experimento-01-baseline-modelos-tradicionales.md) | Clasificación con modelos tradicionales (BoW / TF-IDF) |
| [experimento-02-features-linguisticas.md](./experimento-02-features-linguisticas.md) | Clasificación con features lingüísticas interpretables |
| [experimento-03-embeddings-transformers.md](./experimento-03-embeddings-transformers.md) | Embeddings preentrenados y Transformers |
| [experimento-04-importancia-atributos.md](./experimento-04-importancia-atributos.md) | Análisis de importancia de atributos |
| [experimento-05-analisis-errores.md](./experimento-05-analisis-errores.md) | Análisis de errores |

El diseño experimental completo está en [Informe.md](../Informe.md).

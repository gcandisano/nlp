# ADR — Experimento 5: Análisis de errores

**Estado:** Aceptado  
**Fecha:** 2025-06  
**Relacionado con:** [Informe.md](../Informe.md) § Experimento 5

## Contexto

Un buen F2 no explica **por qué** el modelo falla. Para un sistema de apoyo a verificación (no verdad absoluta), entender los modos de error es tan importante como la métrica agregada.

Preguntas que guían este experimento:

- ¿Los falsos negativos (fake → real) comparten estilo periodístico neutral?
- ¿Los falsos positivos (real → fake) tienen titulares ambiguos o sensacionalistas?
- ¿Transformers cometen errores distintos a los modelos tradicionales?

## Decisión

### Modelos analizados

Se seleccionan los **mejores modelos por familia** del subconjunto político (según F2 en validación/test):

| Familia | Candidato típico |
| :------ | :--------------- |
| Tradicional (Exp. 1) | Mejor pipeline BoW/TF-IDF + LR/SVM/NB |
| Transformer (Exp. 3) | DistilBERT fine-tuned (si está disponible) |

No se analizan todos los modelos de la grilla: solo los representativos de cada enfoque para comparación cualitativa.

### Muestra de errores

- Objetivo: **≥30 ejemplos** mal clasificados en test **cuando los haya**. Con F2 ≈ 0,99 el mejor baseline comete solo **17** errores (6 FP, 11 FN); en ese caso se analizan **todos** los disponibles y se documenta la limitación (la muestra es ilustrativa, no exhaustiva).
- Balance entre **falsos positivos (FP)** y **falsos negativos (FN)** en la medida de lo posible.
- Criterio de selección: representatividad (mezcla de scores de confianza alta y baja), no solo los casos más extremos.

Definiciones:

- **FP:** noticia real clasificada como fake (`label=0`, `pred=1`).
- **FN:** noticia fake clasificada como real (`label=1`, `pred=0`).

### Taxonomía de errores

Cada caso se categoriza manualmente en una taxonomía predefinida:

| Categoría | Descripción |
| :-------- | :---------- |
| `lenguaje_neutral_en_fake` | Fake redactada con estilo formal, sin marcadores sensacionalistas |
| `titulo_ambiguo` | El titular no alinea con el cuerpo o induce confusión |
| `ironía_sarcasmo` | Tono no literal que el modelo no captura |
| `informacion_parcialmente_verdadera` | Mezcla de hechos y desinformación |
| `sesgo_fuente` | Marcadores de agencia periodística o ausencia de ellos |
| `tema_politico_cargado` | Vocabulario político compartido entre clases |
| `otro` | Casos que no encajan; requiere nota explicativa |

La taxonomía es fija antes del análisis para evitar sesgo retrospectivo.

### Metodología

1. Generar predicciones del mejor modelo tradicional (y DistilBERT) sobre `politics_test.csv`.
2. Filtrar FP y FN.
3. Muestrear ≥30 casos balanceados cuando existan; si el modelo comete menos (caso del baseline, 17), analizar todos los errores disponibles.
4. Revisión manual: leer título + extracto del cuerpo, asignar categoría, anotar observación breve.
5. Reportar distribución de categorías por modelo y comparar entre tradicional vs. Transformer.

### Artefactos

Resultados en `results/error_analysis/` (CSV con predicciones erróneas, categorías y notas).

## Alternativas consideradas

| Alternativa | Por qué se descartó |
| :---------- | :------------------ |
| Solo análisis automático (confusion matrix) | No explica causas lingüísticas |
| Muestra aleatoria sin balance FP/FN | Una clase de error puede dominar y ocultar el otro |
| Analizar todos los errores (~miles) | Inviable manualmente; 30+ es suficiente para patrones cualitativos en un TP |
| Errores solo del mejor Transformer | Pierde comparación con enfoque interpretable |
| Análisis en dataset completo | Mezcla dominios temáticos; dificulta atribuir causas |
| Clustering automático de errores | Menos interpretable que taxonomía definida a priori |

## Consecuencias

- Complementa Exp. 4: donde los coeficientes muestran *qué* aprende el modelo, este experimento muestra *dónde* falla en casos concretos.
- Los FN de alta confianza son los más preocupantes para un despliegue real (fake que "parece Reuters").
- Si Transformers y modelos tradicionales comparten las mismas categorías de error, sugiere que el límite está en el dataset/señal, no en la arquitectura.
- Las limitaciones documentadas (ironía, neutralidad, veracidad parcial) justifican por qué el sistema es **apoyo** a verificación humana, no reemplazo.
- El análisis manual no es reproducible al 100%; se mitiga con taxonomía fija y muestra documentada en CSV.

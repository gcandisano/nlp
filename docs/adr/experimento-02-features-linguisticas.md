# ADR — Experimento 2: Clasificación con features lingüísticas interpretables

**Estado:** Aceptado  
**Fecha:** 2025-06  
**Relacionado con:** [Informe.md](../Informe.md) § Experimento 2

## Contexto

El objetivo central del TP no es solo maximizar accuracy, sino entender **qué propiedades del lenguaje** distinguen fake de real en este corpus. El EDA mostró:

- Fake news ligeramente más extensas y con mayor variabilidad en longitud.
- Mayor uso de mayúsculas, exclamaciones y referencias a URLs en contenido de origen social.
- Fake news con mayor densidad de figuras políticas nombradas.
- El periodismo formal (Reuters) tiende a oraciones más largas y estructuradas.

Los modelos de representación densa (BoW, embeddings) no responden directamente preguntas del tipo "¿los signos de exclamación predicen fake news?".

## Decisión

### Extracción de features

Se construye un vector de **8 features explícitas** por artículo:

| Feature | Herramienta | Justificación |
| :------ | :---------- | :------------ |
| `ratio_exclamacion` | regex / conteo | Contenido sensacionalista usa `!` con más frecuencia que periodismo formal |
| `ratio_mayusculas` | regex | Énfasis emocional y urgencia artificial típicos de fake news |
| `long_oracion_prom` | spaCy (segmentación) | Reuters tiende a oraciones más largas y estructuradas |
| `ratio_adj_sust` | spaCy POS tagging | Ratio alto → lenguaje más evaluativo y emocional |
| `sentimiento_vader` | VADER | Polarización extrema asociada a contenido sensacionalista |
| `densidad_ner` | spaCy NER | Fake news mencionan más figuras políticas por oración en este corpus |
| `freq_url` | conteo de `[URL]` | Fake de redes sociales incluyen más enlaces externos |
| `freq_pronombres` | regex / POS | Periodismo formal evita 1.ª/2.ª persona; contenido informal las usa |

### Herramientas de NLP

| Herramienta | Modelo / versión | Motivo de elección |
| :---------- | :--------------- | :----------------- |
| **spaCy** | `en_core_web_lg` | Entrenado sobre OntoNotes (texto periodístico); mejor balance precisión/velocidad que NLTK; más integrable que Stanford CoreNLP |
| **VADER** | `vaderSentiment` | Diseñado para texto informal; maneja MAYÚSCULAS, `!!!` e hiperbole; score numérico interpretable y liviano frente a Transformers de sentimiento |

### Clasificador

**Regresión Logística** sobre el vector de 8 features.

No se usa Random Forest, SVM ni redes neuronales: el objetivo es **interpretabilidad de coeficientes**, no máximo rendimiento. Un coeficiente positivo en `ratio_exclamacion` se lee directamente como "más exclamaciones → más probable fake".

### Sub-experimento: título vs. cuerpo vs. combinado

Se extraen features y se entrena el clasificador en tres condiciones:

1. Solo `title`
2. Solo `text` (cuerpo)
3. `title + text` concatenados

Esto valida la **Hipótesis 2**: el cuerpo completo aporta más patrones lingüísticos que el titular solo.

### Evaluación

- Coeficientes del LR ordenados por magnitud (positivo → asociado a fake, negativo → real).
- Tabla comparativa de F2 entre las tres condiciones de campo textual.
- Mismas métricas generales que Experimento 1 (F2 como principal).

### Datos

Mismo subconjunto político y split temporal que Experimento 1. Si la ablación de fuente del Exp. 1 muestra caída grande de F2, las features se extraen sobre texto con fuentes normalizadas a `[SOURCE]`.

## Alternativas consideradas

| Alternativa | Por qué se descartó |
| :---------- | :------------------ |
| Random Forest sobre features | Importancia de atributos menos directa que coeficientes lineales |
| Transformers para sentimiento (BERT-based) | Menos interpretable y más costoso; VADER cubre el patrón buscado |
| NLTK para POS/NER | Menor precisión y más lento que spaCy en este dominio |
| Features léxicas adicionales (TF-IDF de adjetivos) | Se reserva el análisis de adjetivos al Experimento 4 |
| spaCy `en_core_web_sm` | Modelo pequeño sin word vectors; `lg` mejora POS y NER |
| TextBlob para sentimiento | No maneja tan bien énfasis tipográfico y exclamaciones repetidas |

## Consecuencias

- Si un modelo con 8 features alcanza F2 competitivo con el baseline de Exp. 1, ese es el **resultado lingüístico central** del trabajo.
- Los resultados contextualizan si la ganancia de Transformers (Exp. 3) justifica la pérdida de interpretabilidad.
- La extracción con spaCy `lg` es costosa en tiempo; conviene cachear features en `data/processed/`.
- El sub-experimento título/cuerpo informa qué campo priorizar en análisis de errores y en despliegue hipotético (solo titulares vs. artículo completo).

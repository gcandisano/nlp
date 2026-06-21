const pptxgen = require("pptxgenjs");
const path = require("path");

const FIG = path.resolve(__dirname, "..", "..", "results", "figures");
const fig = (name) => path.join(FIG, name);

// ---- Design system ---------------------------------------------------------
const C = {
  navy: "14213D", // primary dark
  navy2: "1B2A4A",
  paper: "F5F4EF", // warm newspaper paper
  white: "FFFFFF",
  ink: "1E293B", // body text
  muted: "64748B",
  fake: "D7263D", // red — fake
  real: "1B998B", // teal — real
  gold: "E0A458", // accent
  line: "D7D2C4",
};

const F = { head: "Georgia", body: "Calibri" };

const W = 13.333;
const H = 7.5;

const pres = new pptxgen();
pres.defineLayout({ name: "WIDE", width: W, height: H });
pres.layout = "WIDE";
pres.author = "Grupo NLP - ITBA";
pres.title = "Clasificación de noticias por patrones lingüísticos de fake news";

const makeShadow = () => ({
  type: "outer",
  color: "000000",
  blur: 7,
  offset: 3,
  angle: 135,
  opacity: 0.18,
});

// place an image into a box (x,y,w,h) preserving aspect ratio, centered
function fitImage(slide, file, ow, oh, box, opts = {}) {
  const ar = ow / oh;
  let w = box.w;
  let h = w / ar;
  if (h > box.h) {
    h = box.h;
    w = h * ar;
  }
  const x = box.x + (box.w - w) / 2;
  const y = box.y + (box.h - h) / 2;
  slide.addImage({ path: file, x, y, w, h, ...opts });
}

// Section header bar for content slides
function header(slide, kicker, title) {
  slide.background = { color: C.paper };
  // top band
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0,
    y: 0,
    w: W,
    h: 1.18,
    fill: { color: C.navy },
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0,
    y: 1.18,
    w: W,
    h: 0.06,
    fill: { color: C.gold },
  });
  if (kicker) {
    slide.addText(kicker.toUpperCase(), {
      x: 0.55,
      y: 0.16,
      w: 11,
      h: 0.3,
      fontFace: F.body,
      fontSize: 12,
      bold: true,
      color: C.gold,
      charSpacing: 3,
      margin: 0,
    });
  }
  slide.addText(title, {
    x: 0.55,
    y: 0.42,
    w: 12.2,
    h: 0.68,
    fontFace: F.head,
    fontSize: 27,
    bold: true,
    color: C.white,
    margin: 0,
    valign: "middle",
  });
}

function footer(slide, n) {
  slide.addText("Fake News NLP · ITBA 2026", {
    x: 0.55,
    y: H - 0.42,
    w: 6,
    h: 0.3,
    fontFace: F.body,
    fontSize: 9,
    color: C.muted,
    margin: 0,
  });
  slide.addText(String(n), {
    x: W - 1.0,
    y: H - 0.42,
    w: 0.5,
    h: 0.3,
    fontFace: F.body,
    fontSize: 9,
    color: C.muted,
    align: "right",
    margin: 0,
  });
}

// a stat callout card
function statCard(slide, x, y, w, h, value, label, color) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x,
    y,
    w,
    h,
    fill: { color: C.white },
    line: { color: C.line, width: 1 },
    shadow: makeShadow(),
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x,
    y,
    w: 0.09,
    h,
    fill: { color: color },
  });
  slide.addText(value, {
    x: x + 0.15,
    y: y + 0.12,
    w: w - 0.25,
    h: h * 0.55,
    fontFace: F.head,
    fontSize: 30,
    bold: true,
    color: C.navy,
    align: "left",
    valign: "middle",
    margin: 0,
  });
  slide.addText(label, {
    x: x + 0.18,
    y: y + h * 0.6,
    w: w - 0.3,
    h: h * 0.36,
    fontFace: F.body,
    fontSize: 11.5,
    color: C.muted,
    align: "left",
    valign: "top",
    margin: 0,
  });
}

// bullet list helper
function bullets(slide, items, box, opts = {}) {
  const runs = items.map((it, i) => {
    const isObj = typeof it === "object";
    const txt = isObj ? it.text : it;
    return {
      text: txt,
      options: {
        bullet: { code: "2022", indent: 14 },
        breakLine: true,
        fontFace: F.body,
        fontSize: opts.fontSize || 15,
        color: isObj && it.color ? it.color : C.ink,
        bold: isObj ? !!it.bold : false,
        paraSpaceAfter: opts.gap != null ? opts.gap : 9,
        indentLevel: isObj && it.indent ? it.indent : 0,
      },
    };
  });
  slide.addText(runs, { x: box.x, y: box.y, w: box.w, h: box.h, valign: "top", margin: 0 });
}

// caption under an image
function caption(slide, text, x, y, w) {
  slide.addText(text, {
    x,
    y,
    w,
    h: 0.32,
    fontFace: F.body,
    fontSize: 10,
    italic: true,
    color: C.muted,
    align: "center",
    margin: 0,
  });
}

let pageNo = 0;
function newContent(kicker, title) {
  const s = pres.addSlide();
  header(s, kicker, title);
  pageNo += 1;
  footer(s, pageNo);
  return s;
}

// Full dark section-divider slide
function divider(num, title, desc) {
  const s = pres.addSlide();
  s.background = { color: C.navy };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.32, h: H, fill: { color: C.gold } });
  s.addText(num, {
    x: 1.1,
    y: 1.5,
    w: 4,
    h: 2.2,
    fontFace: F.head,
    fontSize: 150,
    bold: true,
    color: C.navy2,
    margin: 0,
    valign: "middle",
  });
  s.addText(title, {
    x: 1.15,
    y: 3.55,
    w: 11,
    h: 1.1,
    fontFace: F.head,
    fontSize: 44,
    bold: true,
    color: C.white,
    margin: 0,
  });
  s.addShape(pres.shapes.RECTANGLE, { x: 1.2, y: 4.7, w: 1.6, h: 0.07, fill: { color: C.gold } });
  if (desc) {
    s.addText(desc, {
      x: 1.2,
      y: 4.95,
      w: 10.2,
      h: 1.0,
      fontFace: F.body,
      fontSize: 18,
      color: "C9D2E3",
      margin: 0,
    });
  }
  return s;
}

// concept card with icon-circle, title and body bullets
function conceptCard(slide, x, y, w, h, glyph, glyphColor, title, lines, opts = {}) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x,
    y,
    w,
    h,
    fill: { color: C.white },
    line: { color: C.line, width: 1 },
    shadow: makeShadow(),
  });
  slide.addShape(pres.shapes.OVAL, { x: x + 0.28, y: y + 0.28, w: 0.62, h: 0.62, fill: { color: glyphColor } });
  const glyphSize = glyph.length >= 3 ? 14 : glyph.length === 2 ? 19 : 22;
  slide.addText(glyph, {
    x: x + 0.28,
    y: y + 0.28,
    w: 0.62,
    h: 0.62,
    fontFace: F.head,
    fontSize: glyphSize,
    bold: true,
    color: C.white,
    align: "center",
    valign: "middle",
    margin: 0,
  });
  slide.addText(title, {
    x: x + 1.05,
    y: y + 0.3,
    w: w - 1.3,
    h: 0.6,
    fontFace: F.body,
    fontSize: 18,
    bold: true,
    color: C.navy,
    valign: "middle",
    margin: 0,
  });
  bullets(slide, lines, { x: x + 0.32, y: y + 1.08, w: w - 0.62, h: h - 1.3 }, {
    fontSize: opts.fontSize || 13.5,
    gap: opts.gap != null ? opts.gap : 8,
  });
}

// =====================================================================
// SLIDE 1 — Title
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: W, h: 0.18, fill: { color: C.fake } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0.18, w: W, h: 0.06, fill: { color: C.gold } });

  s.addText("PROCESAMIENTO DE LENGUAJE NATURAL · ITBA", {
    x: 0.9,
    y: 1.25,
    w: 11,
    h: 0.4,
    fontFace: F.body,
    fontSize: 14,
    bold: true,
    color: C.gold,
    charSpacing: 4,
    margin: 0,
  });
  s.addText("Clasificación automática de noticias\nsegún patrones lingüísticos de las fake news", {
    x: 0.9,
    y: 1.75,
    w: 11.6,
    h: 1.9,
    fontFace: F.head,
    fontSize: 38,
    bold: true,
    color: C.white,
    lineSpacingMultiple: 1.02,
    margin: 0,
  });
  s.addText(
    "Modelos de NLP que separan noticias falsas de reales por su estilo lingüístico — con foco en la interpretabilidad: no qué modelo gana, sino qué rasgos del texto discriminan.",
    {
      x: 0.9,
      y: 3.75,
      w: 10.8,
      h: 1.0,
      fontFace: F.body,
      fontSize: 16,
      color: "C9D2E3",
      margin: 0,
    }
  );

  // chips
  const chips = ["Baselines BoW/TF-IDF", "Features lingüísticas", "GloVe · Word2Vec", "DistilBERT"];
  let cx = 0.9;
  chips.forEach((c) => {
    const w = 0.35 + c.length * 0.105;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: cx,
      y: 5.05,
      w,
      h: 0.5,
      fill: { color: C.navy2 },
      line: { color: C.gold, width: 1 },
      rectRadius: 0.08,
    });
    s.addText(c, {
      x: cx,
      y: 5.05,
      w,
      h: 0.5,
      fontFace: F.body,
      fontSize: 12,
      color: C.white,
      align: "center",
      valign: "middle",
      margin: 0,
    });
    cx += w + 0.25;
  });

  s.addText(
    [
      { text: "Integrantes: ", options: { bold: true, color: C.gold } },
      { text: "[Nombre 1] · [Nombre 2] · [Nombre 3]", options: { color: "C9D2E3" } },
    ],
    { x: 0.9, y: 6.25, w: 11, h: 0.4, fontFace: F.body, fontSize: 14, margin: 0 }
  );
  s.addText("Dataset: Fake and Real News (Kaggle) · 44.898 artículos en inglés (2015–2018)", {
    x: 0.9,
    y: 6.7,
    w: 11,
    h: 0.4,
    fontFace: F.body,
    fontSize: 12,
    italic: true,
    color: C.muted,
    margin: 0,
  });
}

// =====================================================================
// DIVIDER 01 — Objetivos
// =====================================================================
divider("01", "Objetivos y motivación", "Qué problema resolvemos y por qué importa la interpretabilidad.");

// =====================================================================
// SLIDE 2 — Objetivos y motivación
// =====================================================================
{
  const s = newContent("Objetivos y motivación", "¿Qué queremos hacer y por qué?");
  bullets(
    s,
    [
      { text: "Problema: la desinformación se propaga más rápido que su verificación manual.", bold: true },
      "Objetivo: entrenar y comparar clasificadores supervisados que separen noticias falsas de reales por sus patrones lingüísticos.",
      { text: "Importante: los modelos detectan correlaciones de estilo en el corpus — NO verifican la veracidad factual.", color: C.fake, bold: true },
      "Énfasis en interpretabilidad: qué rasgos discriminan (longitud de oración, carga emocional, densidad de entidades, URLs), no solo qué modelo puntúa mejor.",
      "Herramienta de apoyo a la verificación humana, consciente de los sesgos del dataset.",
    ],
    { x: 0.6, y: 1.55, w: 7.0, h: 4.0 },
    { fontSize: 15.5, gap: 13 }
  );

  // right: hypotheses card
  const hx = 7.95;
  s.addShape(pres.shapes.RECTANGLE, {
    x: hx,
    y: 1.55,
    w: 4.78,
    h: 5.2,
    fill: { color: C.navy },
    shadow: makeShadow(),
  });
  s.addText("HIPÓTESIS DE INVESTIGACIÓN", {
    x: hx + 0.3,
    y: 1.8,
    w: 4.2,
    h: 0.4,
    fontFace: F.body,
    fontSize: 13,
    bold: true,
    color: C.gold,
    charSpacing: 2,
    margin: 0,
  });
  const hyp = [
    ["H1", "Las fake usan adjetivos con más carga emocional que el periodismo formal."],
    ["H2", "El cuerpo completo clasifica mejor que el titular solo."],
    ["H3", "El rendimiento cae al borrar marcadores de fuente (\u201creuters\u201d)."],
  ];
  let hy = 2.35;
  hyp.forEach(([k, v]) => {
    s.addText(k, {
      x: hx + 0.3,
      y: hy,
      w: 0.8,
      h: 0.9,
      fontFace: F.head,
      fontSize: 26,
      bold: true,
      color: C.gold,
      margin: 0,
      valign: "top",
    });
    s.addText(v, {
      x: hx + 1.05,
      y: hy + 0.03,
      w: 3.45,
      h: 1.25,
      fontFace: F.body,
      fontSize: 13.5,
      color: C.white,
      margin: 0,
      valign: "top",
    });
    hy += 1.45;
  });
}

// =====================================================================
// DIVIDER 02 — Datos y metodología
// =====================================================================
divider("02", "Datos y metodología", "El corpus, el análisis exploratorio y el diseño experimental.");

// =====================================================================
// SLIDE 3 — Datos
// =====================================================================
{
  const s = newContent("Datos", "El corpus: Fake and Real News (Kaggle)");
  statCard(s, 0.6, 1.55, 2.85, 1.5, "44.898", "artículos en inglés (True + Fake)", C.navy);
  statCard(s, 3.6, 1.55, 2.85, 1.5, "21.417", "noticias reales · fuente: Reuters (2016-17)", C.real);
  statCard(s, 6.6, 1.55, 2.85, 1.5, "23.481", "noticias falsas (2015 - feb 2018)", C.fake);

  bullets(
    s,
    [
      "Columnas: title, text, subject, date. La columna subject NUNCA se usa como feature (es un sesgo del dataset).",
      "Clases balanceadas (~48% reales / 52% falsas) → facilita el entrenamiento supervisado.",
      { text: "Split TEMPORAL 70/15/15 (no aleatorio): se entrena con el pasado y se evalúa con artículos más recientes.", bold: true },
      "Tras parsear fechas y deduplicar (title, text): 44.898 → 39.099 registros para los splits.",
      { text: "Foco: subconjunto POLÍTICO (real=politicsNews, fake=politics) para controlar el sesgo temático; dataset completo solo como control.", color: C.navy, bold: true },
    ],
    { x: 0.6, y: 3.3, w: 6.5, h: 3.5 },
    { fontSize: 14, gap: 11 }
  );

  s.addShape(pres.shapes.RECTANGLE, {
    x: 7.35,
    y: 3.35,
    w: 5.4,
    h: 3.45,
    fill: { color: C.white },
    line: { color: C.line, width: 1 },
    shadow: makeShadow(),
  });
  fitImage(s, fig("01_eda_class_distribution.png"), 844, 596, {
    x: 7.5,
    y: 3.45,
    w: 5.1,
    h: 3.0,
  });
  caption(s, "Distribución de clases (balanceada)", 7.35, 6.45, 5.4);
}

// =====================================================================
// SLIDE 4 — EDA: sesgo temático + nubes de palabras
// =====================================================================
{
  const s = newContent("Análisis exploratorio", "Sesgo temático y vocabulario por clase");
  // left: subject distribution
  s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 1.5, w: 5.3, h: 3.5, fill: { color: C.white }, line: { color: C.line, width: 1 }, shadow: makeShadow() });
  fitImage(s, fig("01_eda_subject_distribution.png"), 1437, 827, { x: 0.7, y: 1.6, w: 5.0, h: 3.3 });
  caption(s, "Reales → politicsNews/worldnews · Fake → temas diversos", 0.55, 5.02, 5.3);

  bullets(
    s,
    [
      { text: "El subject separa casi perfectamente las clases → sesgo, se excluye como feature.", color: C.fake, bold: true },
      "Reales: término más frecuente \u201csaid\u201d (99.076) y presencia constante de \u201creuters\u201d.",
      "Fake: \u201ctrump\u201d (88.636), figuras políticas (clinton, obama) y \u201cvia\u201d/\u201cvideo\u201d (origen en redes).",
    ],
    { x: 0.55, y: 5.4, w: 5.3, h: 1.9 },
    { fontSize: 12.5, gap: 7 }
  );

  // right: two wordclouds stacked
  s.addShape(pres.shapes.RECTANGLE, { x: 6.2, y: 1.5, w: 6.55, h: 2.55, fill: { color: C.white }, line: { color: C.line, width: 1 }, shadow: makeShadow() });
  fitImage(s, fig("01_eda_wordcloud_fake.png"), 1185, 638, { x: 6.35, y: 1.6, w: 6.25, h: 2.2 });
  caption(s, "Word cloud — FAKE", 6.2, 3.78, 6.55);

  s.addShape(pres.shapes.RECTANGLE, { x: 6.2, y: 4.35, w: 6.55, h: 2.55, fill: { color: C.white }, line: { color: C.line, width: 1 }, shadow: makeShadow() });
  fitImage(s, fig("01_eda_wordcloud_real.png"), 1185, 638, { x: 6.35, y: 4.45, w: 6.25, h: 2.2 });
  caption(s, "Word cloud — REAL", 6.2, 6.63, 6.55);
}

// =====================================================================
// SLIDE 5 — EDA: longitud + preprocesamiento
// =====================================================================
{
  const s = newContent("Análisis exploratorio · Preprocesamiento", "Longitud de textos y pipeline de limpieza");
  s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 1.5, w: 6.3, h: 3.4, fill: { color: C.white }, line: { color: C.line, width: 1 }, shadow: makeShadow() });
  fitImage(s, fig("01_eda_text_length_distribution.png"), 1521, 596, { x: 0.7, y: 1.65, w: 6.0, h: 3.1 });
  caption(s, "Las fake tienen mayor variabilidad y cola más larga (423 vs 386 palabras prom.)", 0.55, 4.92, 6.3);

  s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 5.35, w: 6.3, h: 1.55, fill: { color: C.white }, line: { color: C.line, width: 1 }, shadow: makeShadow() });
  fitImage(s, fig("02_split_class_balance.png"), 1270, 596, { x: 0.7, y: 5.4, w: 6.0, h: 1.45 });

  s.addText("PIPELINE DE PREPROCESAMIENTO", {
    x: 7.2, y: 1.55, w: 5.5, h: 0.35, fontFace: F.body, fontSize: 13, bold: true, color: C.navy, charSpacing: 1, margin: 0,
  });
  bullets(
    s,
    [
      "Minúsculas; eliminación de números y caracteres especiales.",
      { text: "Se conservan ! y ? (señales de estilo) y se reemplazan URLs por [URL].", bold: true },
      "Tokenización; variantes con y sin stopwords (se mide el impacto).",
      "Sin lematización en baselines: se preservan formas superficiales discriminativas.",
      "Parseo de fechas multi-formato + deduplicación exacta antes de particionar (evita leakage).",
      { text: "Se conserva el texto crudo para medir mayúsculas, exclamaciones y sentimiento (Exp 2).", color: C.muted },
    ],
    { x: 7.2, y: 1.95, w: 5.55, h: 5.0 },
    { fontSize: 14, gap: 11 }
  );
}

// =====================================================================
// SLIDE 6 — Metodología
// =====================================================================
{
  const s = newContent("Metodología", "Cómo lo hicimos: 5 experimentos, una métrica");
  // metric callout
  s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 1.5, w: 5.6, h: 1.75, fill: { color: C.navy }, shadow: makeShadow() });
  s.addText("MÉTRICA PRINCIPAL", { x: 0.8, y: 1.65, w: 5, h: 0.3, fontFace: F.body, fontSize: 12, bold: true, color: C.gold, charSpacing: 2, margin: 0 });
  s.addText([
    { text: "F2-score", options: { fontFace: F.head, fontSize: 32, bold: true, color: C.white } },
    { text: "  (clase fake)", options: { fontSize: 16, color: "C9D2E3" } },
  ], { x: 0.8, y: 2.0, w: 5.1, h: 0.6, margin: 0 });
  s.addText("Prioriza recall: un falso negativo (fake pasada como real) es más costoso que un falso positivo.", {
    x: 0.8, y: 2.62, w: 5.1, h: 0.55, fontFace: F.body, fontSize: 12, color: "C9D2E3", margin: 0,
  });

  bullets(
    s,
    [
      { text: "Etiquetas: fake = 1 (clase positiva), real = 0.", bold: true },
      "Hiperparámetros elegidos SOLO en validación; test se evalúa una única vez.",
      "Split temporal → val/test pueden tener desbalance de clases (es esperado).",
    ],
    { x: 0.6, y: 3.45, w: 5.6, h: 1.8 },
    { fontSize: 13.5, gap: 9 }
  );

  // right: experiment map
  const ex = [
    ["1", "Baselines tradicionales", "BoW/TF-IDF × LR/NB/SVM + ablación de fuente", C.navy],
    ["2", "Features lingüísticas", "8 rasgos interpretables (spaCy + VADER) → LR", C.real],
    ["3", "Embeddings & Transformers", "GloVe · Word2Vec · DistilBERT (fine-tuning)", C.gold],
    ["4", "Importancia de atributos", "Coeficientes lineales + adjetivos por clase", C.fake],
    ["5", "Análisis de errores", "Taxonomía manual de FP/FN; baseline vs transformer", C.navy],
  ];
  let ey = 1.5;
  const eh = 1.0;
  ex.forEach(([n, t, d, col]) => {
    s.addShape(pres.shapes.RECTANGLE, { x: 6.5, y: ey, w: 6.25, h: eh - 0.12, fill: { color: C.white }, line: { color: C.line, width: 1 }, shadow: makeShadow() });
    s.addShape(pres.shapes.OVAL, { x: 6.66, y: ey + 0.18, w: 0.52, h: 0.52, fill: { color: col } });
    s.addText(n, { x: 6.66, y: ey + 0.18, w: 0.52, h: 0.52, fontFace: F.head, fontSize: 20, bold: true, color: C.white, align: "center", valign: "middle", margin: 0 });
    s.addText(t, { x: 7.35, y: ey + 0.08, w: 5.3, h: 0.38, fontFace: F.body, fontSize: 15, bold: true, color: C.navy, margin: 0, valign: "middle" });
    s.addText(d, { x: 7.35, y: ey + 0.44, w: 5.3, h: 0.36, fontFace: F.body, fontSize: 11.5, color: C.muted, margin: 0, valign: "middle" });
    ey += eh;
  });
}

// =====================================================================
// CONCEPT — Partición temporal de los datos
// =====================================================================
{
  const s = newContent("Concepto · Metodología", "¿Por qué partición temporal y no aleatoria?");
  s.addText(
    "Como las noticias tienen fecha, ordenamos por fecha de publicación y cortamos en bloques: entrenamos con el pasado y evaluamos con artículos más recientes. Simula el escenario real de detección.",
    { x: 0.6, y: 1.45, w: 12.1, h: 0.7, fontFace: F.body, fontSize: 15, italic: true, color: C.muted, margin: 0 }
  );

  // timeline bar
  const tx = 0.8;
  const tw = 11.7;
  const ty = 2.55;
  const th = 0.95;
  const segs = [
    { f: 0.7, label: "TRAIN · 70%", sub: "artículos más antiguos", col: C.navy },
    { f: 0.15, label: "VAL · 15%", sub: "hiperparámetros", col: C.real },
    { f: 0.15, label: "TEST · 15%", sub: "más recientes", col: C.fake },
  ];
  let cx = tx;
  segs.forEach((sg) => {
    const w = tw * sg.f;
    s.addShape(pres.shapes.RECTANGLE, { x: cx, y: ty, w, h: th, fill: { color: sg.col } });
    s.addText(sg.label, { x: cx, y: ty + 0.16, w, h: 0.4, fontFace: F.body, fontSize: sg.f < 0.2 ? 12 : 16, bold: true, color: C.white, align: "center", valign: "middle", margin: 0 });
    s.addText(sg.sub, { x: cx, y: ty + 0.52, w, h: 0.3, fontFace: F.body, fontSize: sg.f < 0.2 ? 9.5 : 11, color: "E8ECF5", align: "center", valign: "middle", margin: 0 });
    cx += w;
  });
  s.addText("más antiguo", { x: tx, y: ty + th + 0.05, w: 3, h: 0.3, fontFace: F.body, fontSize: 11, italic: true, color: C.muted, margin: 0 });
  s.addText("más reciente  →", { x: tx + tw - 3, y: ty + th + 0.05, w: 3, h: 0.3, fontFace: F.body, fontSize: 11, italic: true, color: C.muted, align: "right", margin: 0 });

  bullets(
    s,
    [
      { text: "Validación: ajuste de hiperparámetros y detección de overfitting.", bold: true },
      { text: "Test: se evalúa una sola vez, con la mejor configuración.", bold: true },
      "Más realista que una partición aleatoria: el modelo no ve el futuro durante el entrenamiento.",
      { text: "Consecuencia: val/test pueden quedar desbalanceadas en clases. Es esperado — no reordenamos para no romper el criterio temporal.", color: C.fake },
      "También evita leakage: deduplicamos por (title, text) antes de particionar.",
    ],
    { x: 0.7, y: 4.05, w: 12.0, h: 2.7 },
    { fontSize: 14.5, gap: 11 }
  );
}

// =====================================================================
// CONCEPT — Métricas y por qué F2
// =====================================================================
{
  const s = newContent("Concepto · Metodología", "Métricas: precisión, recall y por qué F2");
  // confusion matrix mini (2x2)
  const mx = 0.7;
  const my = 1.9;
  const cell = 1.35;
  s.addText("Predicción", { x: mx + 1.0, y: my - 0.45, w: cell * 2, h: 0.3, fontFace: F.body, fontSize: 11, bold: true, color: C.muted, align: "center", margin: 0 });
  s.addText("Real (0)", { x: mx + 1.0, y: my - 0.05, w: cell, h: 0.3, fontFace: F.body, fontSize: 11, color: C.muted, align: "center", margin: 0 });
  s.addText("Fake (1)", { x: mx + 1.0 + cell, y: my - 0.05, w: cell, h: 0.3, fontFace: F.body, fontSize: 11, color: C.muted, align: "center", margin: 0 });
  s.addText("Real", { x: mx, y: my + 0.3, w: 0.95, h: cell, fontFace: F.body, fontSize: 11, color: C.muted, align: "right", valign: "middle", margin: 0 });
  s.addText("Fake", { x: mx, y: my + 0.3 + cell, w: 0.95, h: cell, fontFace: F.body, fontSize: 11, color: C.muted, align: "right", valign: "middle", margin: 0 });
  const cm = [
    { x: mx + 1.0, y: my + 0.3, t: "VN", col: "DDE3EE", fg: C.ink },
    { x: mx + 1.0 + cell, y: my + 0.3, t: "FP", col: C.gold, fg: C.white },
    { x: mx + 1.0, y: my + 0.3 + cell, t: "FN", col: C.fake, fg: C.white },
    { x: mx + 1.0 + cell, y: my + 0.3 + cell, t: "VP", col: C.real, fg: C.white },
  ];
  cm.forEach((c) => {
    s.addShape(pres.shapes.RECTANGLE, { x: c.x, y: c.y, w: cell, h: cell, fill: { color: c.col }, line: { color: C.white, width: 2 } });
    s.addText(c.t, { x: c.x, y: c.y, w: cell, h: cell, fontFace: F.head, fontSize: 24, bold: true, color: c.fg, align: "center", valign: "middle", margin: 0 });
  });
  s.addText("Convención: fake = 1 (clase positiva), real = 0.", { x: mx, y: my + 0.3 + cell * 2 + 0.15, w: cell * 2 + 1, h: 0.4, fontFace: F.body, fontSize: 11, italic: true, color: C.muted, margin: 0 });

  // formula cards
  const fx = 5.3;
  const fcards = [
    ["Precisión", "VP / (VP + FP)", "De lo marcado como fake, ¿cuánto era fake?"],
    ["Recall", "VP / (VP + FN)", "De las fake reales, ¿cuántas detectamos?"],
    ["F2-score", "(1+2²)·P·R / (2²·P + R)", "Media F con β=2: el recall pesa 4× más que la precisión."],
  ];
  let fy = 1.55;
  fcards.forEach(([t, f, d], i) => {
    const col = i === 2 ? C.navy : C.white;
    s.addShape(pres.shapes.RECTANGLE, { x: fx, y: fy, w: 7.45, h: 1.45, fill: { color: col }, line: { color: C.line, width: 1 }, shadow: makeShadow() });
    s.addText(t, { x: fx + 0.25, y: fy + 0.15, w: 2.4, h: 1.15, fontFace: F.head, fontSize: 19, bold: true, color: i === 2 ? C.white : C.navy, valign: "middle", margin: 0 });
    s.addText(f, { x: fx + 2.5, y: fy + 0.12, w: 4.8, h: 0.5, fontFace: "Consolas", fontSize: 14, bold: true, color: i === 2 ? C.gold : C.real, valign: "middle", margin: 0 });
    s.addText(d, { x: fx + 2.5, y: fy + 0.62, w: 4.8, h: 0.7, fontFace: F.body, fontSize: 12, color: i === 2 ? "C9D2E3" : C.ink, valign: "top", margin: 0 });
    fy += 1.6;
  });

  s.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: 6.0, w: 12.05, h: 0.85, fill: { color: C.fake }, shadow: makeShadow() });
  s.addText(
    [
      { text: "¿Por qué F2 y no F1/accuracy?  ", options: { bold: true, color: C.white } },
      { text: "Un falso negativo (una fake que pasa como real) es más dañino que un falso positivo. F2 prioriza no dejar pasar fakes.", options: { color: "FDEAEC" } },
    ],
    { x: 0.95, y: 6.0, w: 11.6, h: 0.85, fontFace: F.body, fontSize: 14, valign: "middle", margin: 0 }
  );
}

// =====================================================================
// DIVIDER 03 — Resultados
// =====================================================================
divider("03", "Resultados", "Cinco experimentos: de los baselines clásicos a los Transformers.");

// =====================================================================
// CONCEPT — BoW vs TF-IDF
// =====================================================================
{
  const s = newContent("Concepto · Representación de texto", "¿Cómo convertimos texto en números?");
  s.addText(
    "Los modelos no leen palabras: necesitan vectores. Las dos representaciones clásicas cuentan palabras, sin entender contexto ni orden.",
    { x: 0.6, y: 1.45, w: 12.1, h: 0.55, fontFace: F.body, fontSize: 15, italic: true, color: C.muted, margin: 0 }
  );
  conceptCard(
    s,
    0.6,
    2.15,
    5.9,
    3.55,
    "B",
    C.navy,
    "Bag of Words (BoW)",
    [
      "Cuenta la frecuencia bruta de cada palabra en el documento.",
      "Simple e interpretable, pero da peso alto a palabras frecuentes aunque no discriminen.",
      { text: "Vector disperso, sin semántica ni orden.", color: C.muted },
    ],
    { fontSize: 14, gap: 10 }
  );
  conceptCard(
    s,
    6.85,
    2.15,
    5.9,
    3.55,
    "T",
    C.real,
    "TF-IDF",
    [
      { text: "Pondera: frecuencia en el documento × rareza en el corpus.", bold: true },
      "Penaliza términos comunes (the, is) y resalta los discriminativos.",
      { text: "Nuestra representación principal sobre BoW simple.", color: C.muted },
    ],
    { fontSize: 14, gap: 10 }
  );
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 5.95, w: 12.15, h: 0.85, fill: { color: C.navy }, shadow: makeShadow() });
  s.addText(
    [
      { text: "Bigramas: ", options: { bold: true, color: C.gold } },
      { text: "además de palabras sueltas (unigramas) probamos pares — ", options: { color: C.white } },
      { text: "\u201cwhite house\u201d, \u201cbreaking news\u201d", options: { italic: true, color: C.white } },
      { text: " — que capturan expresiones con significado propio.", options: { color: C.white } },
    ],
    { x: 0.85, y: 5.95, w: 11.7, h: 0.85, fontFace: F.body, fontSize: 14, valign: "middle", margin: 0 }
  );
}

// =====================================================================
// CONCEPT — Modelos de clasificación clásicos
// =====================================================================
{
  const s = newContent("Concepto · Experimento 1", "Los tres clasificadores clásicos");
  s.addText(
    "Sobre las representaciones BoW/TF-IDF entrenamos tres modelos supervisados estándar. Todos son rápidos e interpretables; sirven como línea de base.",
    { x: 0.6, y: 1.45, w: 12.1, h: 0.6, fontFace: F.body, fontSize: 15, italic: true, color: C.muted, margin: 0 }
  );
  const mw = 3.95;
  conceptCard(
    s,
    0.6,
    2.25,
    mw,
    4.3,
    "LR",
    C.navy,
    "Regresión Logística",
    [
      "Modela la probabilidad de que una noticia sea fake.",
      "Frontera de decisión lineal sobre las palabras.",
      { text: "Hiperparámetro C: regula la fuerza de la regularización (sobreajuste vs. simplicidad).", bold: true },
      { text: "Coeficientes interpretables por término.", color: C.muted },
    ],
    { fontSize: 13, gap: 9 }
  );
  conceptCard(
    s,
    4.7,
    2.25,
    mw,
    4.3,
    "NB",
    C.gold,
    "Naive Bayes",
    [
      "Modelo probabilístico (multinomial) sobre conteos de palabras.",
      "Asume independencia entre palabras: ingenuo pero efectivo en texto.",
      { text: "Hiperparámetro alpha: suavizado de Laplace para palabras no vistas.", bold: true },
      { text: "Muy rápido de entrenar.", color: C.muted },
    ],
    { fontSize: 13, gap: 9 }
  );
  conceptCard(
    s,
    8.8,
    2.25,
    mw,
    4.3,
    "SVM",
    C.real,
    "Linear SVM",
    [
      "Busca el hiperplano de máximo margen entre clases.",
      "Robusto en espacios de alta dimensión (muchas palabras).",
      { text: "Hiperparámetro C: equilibra margen amplio vs. errores de clasificación.", bold: true },
      { text: "Fue nuestro mejor baseline.", color: C.fake, bold: true },
    ],
    { fontSize: 13, gap: 9 }
  );
}

// =====================================================================
// SLIDE 7 — Exp 1 Baseline
// =====================================================================
{
  const s = newContent("Resultados · Experimento 1", "Baselines tradicionales (BoW / TF-IDF)");
  statCard(s, 0.55, 1.5, 3.0, 1.45, "0,989", "F2 fake (test) — LinearSVM + BoW(1,2)", C.navy);
  statCard(s, 0.55, 3.1, 3.0, 1.45, "0,998", "ROC-AUC en test", C.real);
  s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 4.7, w: 3.0, h: 2.0, fill: { color: C.white }, line: { color: C.line, width: 1 }, shadow: makeShadow() });
  bullets(
    s,
    [
      "Mejor: LinearSVM, BoW, cuerpo, sin stopwords, bigramas.",
      "Solo 17 errores en test (6 FP, 11 FN).",
      { text: "F2≈0,99: el dataset se separa casi trivialmente.", color: C.fake, bold: true },
    ],
    { x: 0.72, y: 4.82, w: 2.7, h: 1.8 },
    { fontSize: 11.5, gap: 6 }
  );

  s.addShape(pres.shapes.RECTANGLE, { x: 3.75, y: 1.5, w: 4.45, h: 5.25, fill: { color: C.white }, line: { color: C.line, width: 1 }, shadow: makeShadow() });
  fitImage(s, fig("03_baseline_best_confusion_matrix.png"), 692, 596, { x: 3.9, y: 1.65, w: 4.15, h: 5.0 });
  caption(s, "Matriz de confusión (test)", 3.75, 6.5, 4.45);

  s.addShape(pres.shapes.RECTANGLE, { x: 8.4, y: 1.5, w: 4.35, h: 5.25, fill: { color: C.white }, line: { color: C.line, width: 1 }, shadow: makeShadow() });
  fitImage(s, fig("03_baseline_best_roc_curve.png"), 811, 711, { x: 8.55, y: 1.65, w: 4.05, h: 5.0 });
  caption(s, "Curva ROC (test)", 8.4, 6.5, 4.35);
}

// =====================================================================
// SLIDE 8 — Exp 1 ablación de fuente (H3)
// =====================================================================
{
  const s = newContent("Resultados · Experimento 1 (ablación)", "H3 — ¿El modelo se apoya en la fuente?");
  s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 1.5, w: 6.5, h: 5.3, fill: { color: C.white }, line: { color: C.line, width: 1 }, shadow: makeShadow() });
  fitImage(s, fig("03_baseline_source_ablation.png"), 927, 596, { x: 0.7, y: 1.7, w: 6.2, h: 4.9 });
  caption(s, "F2 con texto original vs. marcadores de fuente normalizados a [SOURCE]", 0.55, 6.45, 6.5);

  s.addShape(pres.shapes.RECTANGLE, { x: 7.4, y: 1.55, w: 5.35, h: 1.5, fill: { color: C.real }, shadow: makeShadow() });
  s.addText([
    { text: "H3 NO confirmada\n", options: { fontFace: F.head, fontSize: 22, bold: true, color: C.white } },
    { text: "El modelo aprende patrones lingüísticos, no identidad de fuente.", options: { fontSize: 13, color: "EAFBF6" } },
  ], { x: 7.6, y: 1.7, w: 5.0, h: 1.2, margin: 0, valign: "middle" });

  bullets(
    s,
    [
      "Normalizar \u201creuters/ap/afp\u201d → [SOURCE] baja el F2 en validación solo ~0,020 (0,992 → 0,972).",
      { text: "Por debajo del umbral de 0,03 → no se activa la normalización de fuente.", bold: true },
      "Hallazgo positivo de validez: la señal aprendida es mayormente de estilo lingüístico.",
      { text: "Matiz: el dataset sigue teniendo fuente única (Reuters) en la clase real → ver Limitaciones.", color: C.muted },
    ],
    { x: 7.4, y: 3.3, w: 5.35, h: 3.4 },
    { fontSize: 14, gap: 12 }
  );
}

// =====================================================================
// CONCEPT — spaCy + VADER
// =====================================================================
{
  const s = newContent("Concepto · Experimento 2", "Las herramientas: spaCy y VADER");
  s.addText(
    "Para extraer los 8 rasgos de estilo combinamos dos librerías especializadas en lenguaje natural.",
    { x: 0.6, y: 1.45, w: 12.1, h: 0.5, fontFace: F.body, fontSize: 15, italic: true, color: C.muted, margin: 0 }
  );
  conceptCard(
    s,
    0.6,
    2.15,
    5.95,
    4.45,
    "sp",
    C.navy,
    "spaCy (en_core_web_sm)",
    [
      { text: "POS tagging: etiqueta cada palabra (adjetivo, sustantivo, pronombre…).", bold: true },
      { text: "NER: reconoce entidades nombradas (personas, lugares, organizaciones).", bold: true },
      "Segmentación de oraciones (para longitud y densidad).",
      "Entrenado en OntoNotes (texto periodístico) → muy preciso en noticias.",
      { text: "Usamos sm y no lg: POS/NER/segmentación son casi idénticos y evitamos ~560 MB de word vectors que no usamos.", color: C.muted },
    ],
    { fontSize: 13, gap: 8 }
  );
  conceptCard(
    s,
    6.8,
    2.15,
    5.95,
    4.45,
    "VA",
    C.real,
    "VADER",
    [
      { text: "Analizador de sentimiento basado en lexicón, diseñado para texto informal.", bold: true },
      "Maneja nativamente MAYÚSCULAS, signos de exclamación e hipérbole — justo los patrones sensacionalistas.",
      "Devuelve un score compuesto en [-1, +1] (muy negativo → muy positivo).",
      "Liviano e interpretable: ideal como feature de entrada al clasificador.",
      { text: "Alternativa a modelos de sentimiento basados en Transformers, más pesados y opacos.", color: C.muted },
    ],
    { fontSize: 13, gap: 8 }
  );
}

// =====================================================================
// CONCEPT — Las 8 features lingüísticas
// =====================================================================
{
  const s = newContent("Concepto · Experimento 2", "Las 8 features lingüísticas interpretables");
  s.addText(
    "En vez de miles de palabras, describimos cada noticia con 8 rasgos de estilo medibles (spaCy + VADER). Un LR sobre ellos dice qué discrimina, en unidades interpretables.",
    { x: 0.6, y: 1.45, w: 12.1, h: 0.6, fontFace: F.body, fontSize: 14, italic: true, color: C.muted, margin: 0 }
  );
  const hdr = (t) => ({ text: t, options: { bold: true, color: C.white, fill: { color: C.navy }, fontSize: 13, valign: "middle" } });
  const rows = [
    [hdr("Feature"), hdr("Qué mide"), hdr("Señal esperada")],
    ["ratio_exclamacion", "Signos ! por oración", "+ fake (sensacionalismo)"],
    ["ratio_mayusculas", "Palabras en MAYÚSCULAS", "+ fake (énfasis/urgencia)"],
    ["long_oracion_prom", "Tokens por oración", "+ real (estilo formal)"],
    ["ratio_adj_sust", "Adjetivos / sustantivos", "+ fake (lenguaje evaluativo)"],
    ["sentimiento_vader", "Score compuesto VADER", "polarización del contenido"],
    ["densidad_ner", "Entidades por oración", "+ fake (figuras políticas)"],
    ["freq_url", "Frecuencia del token [URL]", "+ fake (origen en redes)"],
    ["freq_pronombres", "Pronombres 1.ª/2.ª persona", "+ fake (informalidad)"],
  ];
  const tbl = rows.map((r, i) =>
    r.map((cell) => {
      const isObj = typeof cell === "object";
      const base = {
        fontFace: F.body,
        fontSize: 13,
        color: C.ink,
        valign: "middle",
        margin: 4,
        fill: { color: i === 0 ? C.navy : i % 2 ? "F0EEE6" : C.white },
      };
      return isObj ? { text: cell.text, options: { ...base, ...cell.options } } : { text: cell, options: { ...base, bold: false } };
    })
  );
  s.addTable(tbl, { x: 0.6, y: 2.2, w: 12.15, colW: [3.35, 4.4, 4.4], rowH: 0.5, border: { type: "solid", pt: 0.5, color: C.line } });
  s.addText("spaCy en_core_web_sm (POS, NER, segmentación) · VADER (sentimiento informal) · Clasificador: Regresión Logística sin scaler → coeficientes interpretables.", {
    x: 0.6, y: 6.75, w: 12.15, h: 0.4, fontFace: F.body, fontSize: 11, italic: true, color: C.muted, margin: 0,
  });
}

// =====================================================================
// SLIDE 9 — Exp 2 features lingüísticas (H2)
// =====================================================================
{
  const s = newContent("Resultados · Experimento 2", "Features lingüísticas interpretables (H2)");
  s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 1.5, w: 5.9, h: 3.05, fill: { color: C.white }, line: { color: C.line, width: 1 }, shadow: makeShadow() });
  fitImage(s, fig("04_h2_f2_by_text_field.png"), 927, 596, { x: 0.7, y: 1.6, w: 5.6, h: 2.85 });
  caption(s, "F2 (val) por campo de texto: título / cuerpo / combinado", 0.55, 4.5, 5.9);

  s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 4.95, w: 5.9, h: 1.85, fill: { color: C.fake }, shadow: makeShadow() });
  s.addText([
    { text: "H2 refutada\n", options: { fontFace: F.head, fontSize: 22, bold: true, color: C.white } },
    { text: "El TÍTULO solo clasifica mejor (F2 val 0,92) que el cuerpo (0,77) y el combinado (0,83). Las señales de estilo se concentran en titulares sensacionalistas.", options: { fontSize: 13, color: "FDEAEC" } },
  ], { x: 0.78, y: 5.1, w: 5.5, h: 1.6, margin: 0, valign: "middle" });

  s.addShape(pres.shapes.RECTANGLE, { x: 6.75, y: 1.5, w: 6.0, h: 5.3, fill: { color: C.white }, line: { color: C.line, width: 1 }, shadow: makeShadow() });
  fitImage(s, fig("04_linguistic_coefficients.png"), 1179, 711, { x: 6.9, y: 1.65, w: 5.7, h: 4.9 });
  caption(s, "Coeficientes del LR (8 features) · + = fake, − = real", 6.75, 6.5, 6.0);
}

// =====================================================================
// CONCEPT — ¿Qué es un embedding?
// =====================================================================
{
  const s = newContent("Concepto · Experimento 3", "¿Qué es un embedding?");
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.5, w: 12.15, h: 1.55, fill: { color: C.navy }, shadow: makeShadow() });
  s.addText(
    [
      { text: "Un embedding es un vector denso que representa cada palabra según su significado. ", options: { bold: true, color: C.white } },
      { text: "Palabras con uso similar quedan cerca en el espacio vectorial ", options: { color: "C9D2E3" } },
      { text: "(rey − hombre + mujer ≈ reina)", options: { italic: true, color: C.gold } },
      { text: ". Cada noticia = promedio de los vectores de sus palabras → entrada a un LR / SVM.", options: { color: "C9D2E3" } },
    ],
    { x: 0.85, y: 1.5, w: 11.65, h: 1.55, fontFace: F.body, fontSize: 15, valign: "middle", margin: 0 }
  );
  conceptCard(
    s,
    0.6,
    3.3,
    5.9,
    3.4,
    "G",
    C.gold,
    "GloVe (preentrenado)",
    [
      { text: "glove.6B.100d: 6.000 millones de tokens, 100 dimensiones.", bold: true },
      "Entrenado sobre Wikipedia + Gigaword (texto periodístico).",
      "Amplia cobertura de vocabulario, pero genérico.",
      { text: "Elegimos 6B/100d sobre 840B/300d: ganancia marginal, 3× memoria.", color: C.muted },
    ],
    { fontSize: 13.5, gap: 9 }
  );
  conceptCard(
    s,
    6.85,
    3.3,
    5.9,
    3.4,
    "W",
    C.real,
    "Word2Vec (de dominio)",
    [
      { text: "Entrenado por nosotros sobre ~12,6k docs del train político.", bold: true },
      "Captura la jerga específica del corpus 2015-2017.",
      "Permite comparar embeddings de dominio vs. genéricos.",
      { text: "Limitación: mezcla efecto de dominio con tamaño del corpus.", color: C.muted },
    ],
    { fontSize: 13.5, gap: 9 }
  );
}

// =====================================================================
// SLIDE 10a — Exp 3 embeddings (resultados)
// =====================================================================
{
  const s = newContent("Resultados · Experimento 3 (a)", "Embeddings: GloVe vs. Word2Vec");
  s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 1.5, w: 7.6, h: 5.3, fill: { color: C.white }, line: { color: C.line, width: 1 }, shadow: makeShadow() });
  fitImage(s, fig("05_embedding_comparison.png"), 1377, 598, { x: 0.7, y: 2.1, w: 7.3, h: 4.1 });
  caption(s, "F2 fake (test) por representación y clasificador", 0.55, 6.45, 7.6);

  statCard(s, 8.45, 1.5, 4.3, 1.35, "0,945", "Word2Vec dominio + LinearSVM (test)", C.real);
  statCard(s, 8.45, 3.0, 4.3, 1.35, "0,905", "GloVe genérico + LR (test)", C.gold);
  s.addShape(pres.shapes.RECTANGLE, { x: 8.45, y: 4.5, w: 4.3, h: 2.3, fill: { color: C.navy }, shadow: makeShadow() });
  bullets(
    s,
    [
      { text: "El Word2Vec de dominio supera al GloVe genérico.", color: C.white, bold: true },
      { text: "El vocabulario específico del corpus importa más que el tamaño del preentrenamiento.", color: "C9D2E3" },
      { text: "Pero ninguno alcanza el baseline BoW (0,989): la señal es mayormente léxica, no semántica.", color: "C9D2E3" },
    ],
    { x: 8.67, y: 4.65, w: 3.95, h: 2.0 },
    { fontSize: 12.5, gap: 8 }
  );
}

// =====================================================================
// CONCEPT — Transformers / DistilBERT
// =====================================================================
{
  const s = newContent("Concepto · Experimento 3", "Transformers, BERT y fine-tuning");
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.5, w: 12.15, h: 1.5, fill: { color: C.navy }, shadow: makeShadow() });
  s.addText(
    [
      { text: "A diferencia del promedio de embeddings, un Transformer usa auto-atención: ", options: { bold: true, color: C.white } },
      { text: "interpreta cada palabra según TODO su contexto en la oración (\u201cbanco\u201d de plaza vs. de dinero).", options: { color: "C9D2E3" } },
    ],
    { x: 0.85, y: 1.5, w: 11.65, h: 1.5, fontFace: F.body, fontSize: 15, valign: "middle", margin: 0 }
  );
  conceptCard(
    s,
    0.6,
    3.25,
    6.0,
    3.45,
    "F",
    C.navy,
    "Fine-tuning",
    [
      { text: "Partimos de BERT preentrenado en enormes corpus de texto.", bold: true },
      "Ajustamos sus pesos a nuestra tarea (fake vs real) con los datos etiquetados.",
      "Grilla de hiperparámetros en validación con early stopping sobre F2.",
      { text: "Aprovecha conocimiento del lenguaje ya aprendido.", color: C.muted },
    ],
    { fontSize: 13.5, gap: 9 }
  );
  s.addShape(pres.shapes.RECTANGLE, { x: 6.75, y: 3.25, w: 6.0, h: 3.45, fill: { color: C.white }, line: { color: C.line, width: 1 }, shadow: makeShadow() });
  s.addShape(pres.shapes.OVAL, { x: 7.03, y: 3.53, w: 0.62, h: 0.62, fill: { color: C.real } });
  s.addText("D", { x: 7.03, y: 3.53, w: 0.62, h: 0.62, fontFace: F.head, fontSize: 22, bold: true, color: C.white, align: "center", valign: "middle", margin: 0 });
  s.addText("¿Por qué DistilBERT?", { x: 7.8, y: 3.55, w: 4.7, h: 0.6, fontFace: F.body, fontSize: 18, bold: true, color: C.navy, valign: "middle", margin: 0 });
  statCard(s, 7.03, 4.45, 2.7, 1.0, "~97%", "del rendimiento de BERT-base", C.real);
  statCard(s, 9.85, 4.45, 2.7, 1.0, "60%", "de los parámetros", C.navy);
  s.addText(
    "2× más rápido. El fine-tuning de BERT-base es inviable en CPU local; DistilBERT cubre el rol contextual dentro del alcance del TP (extensión opcional en Colab con GPU).",
    { x: 7.03, y: 5.6, w: 5.5, h: 1.0, fontFace: F.body, fontSize: 12.5, color: C.ink, margin: 0, valign: "top" }
  );
}

// =====================================================================
// SLIDE 10b — Exp 3 DistilBERT (resultados)
// =====================================================================
{
  const s = newContent("Resultados · Experimento 3 (b)", "Transformers: DistilBERT");
  s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 1.5, w: 7.6, h: 5.3, fill: { color: C.white }, line: { color: C.line, width: 1 }, shadow: makeShadow() });
  fitImage(s, fig("05_transformer_hp_search.png"), 1157, 600, { x: 0.7, y: 2.1, w: 7.3, h: 4.1 });
  caption(s, "Búsqueda de hiperparámetros en validación (learning rate, batch, warmup)", 0.55, 6.45, 7.6);

  statCard(s, 8.45, 1.5, 4.3, 1.45, "0,999", "F2 fake (test) — el mejor modelo", C.real);
  s.addShape(pres.shapes.RECTANGLE, { x: 8.45, y: 3.1, w: 4.3, h: 3.7, fill: { color: C.navy }, shadow: makeShadow() });
  s.addText("LECTURA", { x: 8.67, y: 3.25, w: 3.9, h: 0.3, fontFace: F.body, fontSize: 12, bold: true, color: C.gold, charSpacing: 2, margin: 0 });
  bullets(
    s,
    [
      { text: "DistilBERT logra el F2 más alto de todos los enfoques.", color: C.white, bold: true },
      { text: "La ventaja sobre el baseline BoW (0,989) es marginal: el dataset ya era casi separable.", color: "C9D2E3" },
      { text: "Costo: GPU y horas de entrenamiento.", color: "C9D2E3" },
      { text: "Pierde la interpretabilidad de coeficientes de los modelos lineales.", color: "C9D2E3" },
    ],
    { x: 8.67, y: 3.65, w: 3.9, h: 3.0 },
    { fontSize: 12.5, gap: 9 }
  );
}

// =====================================================================
// SLIDE 11 — Exp 4 importancia + H1
// =====================================================================
{
  const s = newContent("Resultados · Experimento 4", "Importancia de atributos y carga emocional (H1)");
  s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 1.5, w: 7.05, h: 5.3, fill: { color: C.white }, line: { color: C.line, width: 1 }, shadow: makeShadow() });
  fitImage(s, fig("06_feature_importance_top_terms.png"), 2082, 876, { x: 0.7, y: 2.35, w: 6.75, h: 3.6 });
  caption(s, "Términos con mayor peso lineal por clase (LinearSVM)", 0.55, 6.45, 7.05);

  s.addShape(pres.shapes.RECTANGLE, { x: 7.85, y: 1.5, w: 4.9, h: 3.2, fill: { color: C.white }, line: { color: C.line, width: 1 }, shadow: makeShadow() });
  fitImage(s, fig("06_h1_adjective_valence.png"), 824, 564, { x: 8.0, y: 1.6, w: 4.6, h: 3.0 });
  caption(s, "Carga emocional (VADER) de adjetivos por clase", 7.85, 4.6, 4.9);

  s.addShape(pres.shapes.RECTANGLE, { x: 7.85, y: 5.05, w: 4.9, h: 1.75, fill: { color: C.real }, shadow: makeShadow() });
  s.addText([
    { text: "H1 sostenida en dirección\n", options: { fontFace: F.head, fontSize: 18, bold: true, color: C.white } },
    { text: "Carga emocional de adjetivos fake (~0,30) duplica a la de los reales (~0,12). La diferencia la marcan pocos adjetivos cargados (great, illegal, criminal).", options: { fontSize: 12, color: "EAFBF6" } },
  ], { x: 8.05, y: 5.15, w: 4.55, h: 1.55, margin: 0, valign: "middle" });
}

// =====================================================================
// SLIDE 12 — Comparación consolidada de modelos
// =====================================================================
{
  const s = newContent("Resultados", "Comparación de todos los modelos (F2 test)");
  s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 1.45, w: 7.4, h: 5.4, fill: { color: C.white }, line: { color: C.line, width: 1 }, shadow: makeShadow() });
  fitImage(s, fig("07_model_comparison_f2.png"), 1460, 619, { x: 0.7, y: 2.45, w: 7.1, h: 3.3 });
  caption(s, "F2 de la clase fake en test — todos los enfoques", 0.55, 6.5, 7.4);

  // ranking table on right
  const rows = [
    [{ text: "Modelo", options: { bold: true, color: C.white, fill: { color: C.navy }, fontSize: 13 } }, { text: "F2", options: { bold: true, color: C.white, fill: { color: C.navy }, align: "center", fontSize: 13 } }],
    ["DistilBERT", "0,999"],
    ["LinearSVM · BoW(1,2)", "0,989"],
    ["W2V dominio · SVM", "0,945"],
    ["GloVe · LR", "0,905"],
    ["Features ling. (título)", "0,904"],
  ];
  const tbl = rows.map((r, i) =>
    r.map((cell) => {
      const isObj = typeof cell === "object";
      const base = {
        fontFace: F.body,
        fontSize: 13,
        color: C.ink,
        valign: "middle",
        margin: 3,
        fill: { color: i === 0 ? C.navy : i % 2 ? "F0EEE6" : C.white },
      };
      return isObj ? { text: cell.text, options: { ...base, ...cell.options } } : { text: cell, options: { ...base, bold: i === 1 } };
    })
  );
  s.addTable(tbl, { x: 8.3, y: 1.95, w: 4.45, colW: [3.25, 1.2], rowH: 0.52, border: { type: "solid", pt: 0.5, color: C.line }, align: "left" });
  s.addText("DistilBERT y el baseline BoW dominan; los enfoques interpretables quedan ~5-8 pts por debajo pero explican QUÉ discrimina.", {
    x: 8.3, y: 5.5, w: 4.45, h: 1.3, fontFace: F.body, fontSize: 12.5, italic: true, color: C.muted, margin: 0, valign: "top",
  });
}

// =====================================================================
// SLIDE 13 — Exp 5 análisis de errores
// =====================================================================
{
  const s = newContent("Resultados · Experimento 5", "Análisis de errores: baseline vs. DistilBERT");
  s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 1.5, w: 6.5, h: 5.3, fill: { color: C.white }, line: { color: C.line, width: 1 }, shadow: makeShadow() });
  fitImage(s, fig("07_error_category_comparison.png"), 1629, 711, { x: 0.7, y: 2.1, w: 6.2, h: 4.1 });
  caption(s, "Distribución de categorías de error por tipo de modelo", 0.55, 6.45, 6.5);

  bullets(
    s,
    [
      { text: "Baseline: 17 errores totales (6 FP, 11 FN).", bold: true },
      "Categorías más frecuentes: tema político cargado (5), sesgo de fuente (4), información parcialmente verdadera (3).",
      { text: "DistilBERT comete muchísimos menos errores (≈2): información parcial y sesgo de fuente.", color: C.navy, bold: true },
      "Errores típicos: lenguaje neutral en fake, títulos ambiguos, ironía/sarcasmo.",
      { text: "Limitación: <30 errores → se analizan todos los disponibles y se documenta.", color: C.muted },
    ],
    { x: 7.3, y: 1.7, w: 5.45, h: 5.0 },
    { fontSize: 14, gap: 12 }
  );
}

// =====================================================================
// DIVIDER 04 — Conclusiones
// =====================================================================
divider("04", "Conclusiones", "Qué encontramos, qué limita el trabajo y cómo seguir.");

// =====================================================================
// SLIDE 14 — Conclusiones
// =====================================================================
{
  const s = newContent("Conclusiones", "Qué encontramos");
  const cards = [
    ["H1", "Sostenida en dirección", "Adjetivos fake con doble carga emocional que los reales — pero impulsada por pocos términos cargados.", C.real],
    ["H2", "Refutada", "El título solo clasifica mejor que el cuerpo: el estilo sensacionalista vive en los titulares.", C.fake],
    ["H3", "No confirmada", "El modelo no depende de marcadores de fuente; aprende patrones lingüísticos (validez del dataset).", C.real],
  ];
  let cx = 0.55;
  const cw = 4.0;
  cards.forEach(([k, verd, d, col]) => {
    s.addShape(pres.shapes.RECTANGLE, { x: cx, y: 1.55, w: cw, h: 2.6, fill: { color: C.white }, line: { color: C.line, width: 1 }, shadow: makeShadow() });
    s.addShape(pres.shapes.RECTANGLE, { x: cx, y: 1.55, w: cw, h: 0.7, fill: { color: col } });
    s.addText(k, { x: cx + 0.2, y: 1.6, w: 1, h: 0.6, fontFace: F.head, fontSize: 26, bold: true, color: C.white, valign: "middle", margin: 0 });
    s.addText(verd, { x: cx + 1.0, y: 1.6, w: cw - 1.2, h: 0.6, fontFace: F.body, fontSize: 15, bold: true, color: C.white, valign: "middle", align: "right", margin: 0 });
    s.addText(d, { x: cx + 0.22, y: 2.45, w: cw - 0.44, h: 1.55, fontFace: F.body, fontSize: 13.5, color: C.ink, valign: "top", margin: 0 });
    cx += cw + 0.32;
  });

  s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 4.45, w: 12.2, h: 2.35, fill: { color: C.navy }, shadow: makeShadow() });
  s.addText("MENSAJES CLAVE", { x: 0.85, y: 4.6, w: 11, h: 0.35, fontFace: F.body, fontSize: 13, bold: true, color: C.gold, charSpacing: 2, margin: 0 });
  bullets(
    s,
    [
      { text: "El F2≈0,99 NO es evidencia de detección de desinformación: el dataset se separa por artefactos de estilo/fuente.", color: C.white },
      { text: "El valor real es lingüístico: 8 features interpretables alcanzan F2≈0,90 y explican qué rasgos discriminan.", color: C.white },
      { text: "Transformers (DistilBERT) ganan en métricas, pero pierden interpretabilidad frente a los modelos lineales.", color: C.white },
    ],
    { x: 0.85, y: 5.0, w: 11.6, h: 1.7 },
    { fontSize: 14, gap: 8 }
  );
}

// =====================================================================
// SLIDE 15 — Limitaciones y mejoras
// =====================================================================
{
  const s = newContent("Limitaciones y posibles mejoras", "Tan importantes como los resultados");
  s.addText("LIMITACIONES", { x: 0.6, y: 1.5, w: 6, h: 0.35, fontFace: F.body, fontSize: 14, bold: true, color: C.fake, charSpacing: 2, margin: 0 });
  bullets(
    s,
    [
      { text: "Fuente única en la clase real: todas las reales son de Reuters → \u201creal\u201d ≈ estilo editorial Reuters, no ley general.", bold: true },
      "F2≈0,99 como huella de artefacto: el dataset separa casi trivialmente por boilerplate/estructura.",
      "Reutilización del test en varios experimentos → comparación cross-experimento es exploratoria.",
      "Split temporal del dataset completo invierte la prevalencia de clases (confound temporal).",
      "Word2Vec dominio vs. GloVe mezcla efecto de dominio con tamaño del corpus.",
      "No se entrenó BERT-base ni modelos mayores (costo computacional).",
    ],
    { x: 0.6, y: 1.9, w: 6.15, h: 4.9 },
    { fontSize: 13, gap: 9 }
  );

  s.addShape(pres.shapes.RECTANGLE, { x: 7.05, y: 1.5, w: 5.7, h: 5.3, fill: { color: C.white }, line: { color: C.line, width: 1 }, shadow: makeShadow() });
  s.addText("POSIBLES MEJORAS", { x: 7.3, y: 1.7, w: 5, h: 0.35, fontFace: F.body, fontSize: 14, bold: true, color: C.real, charSpacing: 2, margin: 0 });
  bullets(
    s,
    [
      { text: "Incorporar noticias reales de múltiples fuentes (romper el sesgo Reuters).", bold: true },
      "Reportar también balanced-accuracy y AUC-PR para el desbalance.",
      "Deduplicación difusa (MinHash/LSH) para near-duplicates de boilerplate.",
      "Evaluar transfer a otros corpus/dominios de noticias.",
      "Probar modelos contextuales mayores con interpretabilidad (atención, SHAP).",
    ],
    { x: 7.3, y: 2.15, w: 5.25, h: 4.5 },
    { fontSize: 13.5, gap: 12 }
  );
}

// =====================================================================
// SLIDE 16 — Anexo / Gracias
// =====================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: W, h: 0.18, fill: { color: C.gold } });
  s.addText("GRACIAS", { x: 0.9, y: 1.6, w: 11, h: 1.0, fontFace: F.head, fontSize: 52, bold: true, color: C.white, margin: 0 });
  s.addText("¿Preguntas?", { x: 0.9, y: 2.75, w: 11, h: 0.6, fontFace: F.body, fontSize: 22, color: C.gold, margin: 0 });

  s.addText("ANEXO — material de respaldo disponible", { x: 0.9, y: 3.7, w: 11, h: 0.4, fontFace: F.body, fontSize: 13, bold: true, color: "C9D2E3", charSpacing: 2, margin: 0 });
  bullets(
    s,
    [
      { text: "EDA: distribución temporal por clase, puntuación/URLs por clase.", color: "DCE3F0" },
      { text: "Exp 1: baselines politics vs. dataset completo (control de sensibilidad).", color: "DCE3F0" },
      { text: "Exp 4: bigramas relevantes y nubes de adjetivos por clase.", color: "DCE3F0" },
      { text: "Código y notebooks reproducibles (01→07) + informe completo.", color: "DCE3F0" },
    ],
    { x: 0.9, y: 4.15, w: 11.5, h: 2.4 },
    { fontSize: 14.5, gap: 9 }
  );
  s.addText("Dataset: Fake and Real News (Kaggle) · spaCy en_core_web_sm · GloVe 6B/100d · DistilBERT-base-uncased", {
    x: 0.9, y: 6.8, w: 11.5, h: 0.4, fontFace: F.body, fontSize: 11, italic: true, color: C.muted, margin: 0,
  });
}

const out = path.resolve(__dirname, "Presentacion_FakeNews_NLP.pptx");
pres.writeFile({ fileName: out }).then(() => console.log("WROTE", out));

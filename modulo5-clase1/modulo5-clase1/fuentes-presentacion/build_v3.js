// ════════════════════════════════════════════════════════════════════════════
//  Módulo 5 · Clase 1 — Pronosticar la producción de un pozo
//  Reconstrucción v3: orden correcto, cada herramienta explicada antes de usarse
//  16:9 wide (13.33 × 7.5 in)
// ════════════════════════════════════════════════════════════════════════════
const pptxgen = require("pptxgenjs");
const { execSync } = require("child_process");

const DIM = JSON.parse(execSync(
  `python3 -c "
import json,glob;from PIL import Image
d={}
for f in glob.glob('../fig_v3/*.png')+glob.glob('../logos/*_trim.png'):
    im=Image.open(f); d[f.split('/')[-1][:-4]]=im.width/im.height
print(json.dumps(d))"`).toString());

// ── sistema visual ──────────────────────────────────────────────────────────
const RED = "C82B40", DARK = "6B1525", INK = "23272E", MUTED = "6B7280";
const LGRAY = "E7E9EC", GRAY = "F4F5F7", PINK = "FBEBED", TERM = "111827",
      TGREEN = "86EFAC", WHITE = "FFFFFF", GOLD = "B45309", BLUE = "1D4ED8";
const F = "Calibri", FM = "Consolas";
const W = 13.33, H = 7.5, MX = 0.8;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "Carlos Enrique Mosquera Trujillo";
pres.title = "Módulo 5 · Clase 1 — Pronosticar la producción de un pozo";

let N = 0;

// ── primitivas ──────────────────────────────────────────────────────────────
function img(s, name, o) {
  const r = DIM[name];
  let { x, y, w, h, cx } = o;
  if (w && !h) h = w / r;
  if (h && !w) w = h * r;
  if (o.maxW && w > o.maxW) { w = o.maxW; h = w / r; }
  if (o.maxH && h > o.maxH) { h = o.maxH; w = h * r; }
  if (cx !== undefined) x = cx - w / 2;
  const dir = name.endsWith("_trim") ? "logos" : "fig_v3";
  s.addImage({ path: `../${dir}/${name}.png`, x, y, w, h });
  return { w, h, y2: y + h };
}

function slide() {
  N += 1;
  const s = pres.addSlide();
  s.background = { color: WHITE };
  img(s, "logo_udla_trim", { x: 11.15, y: 0.36, h: 0.32 });
  img(s, "SLB_trim", { x: 12.12, y: 0.32, h: 0.40 });
  s.addText("Machine Learning for Petroleum Engineers Using Python", {
    x: MX, y: 7.08, w: 7, h: 0.3, fontFace: F, fontSize: 9.5, color: "9AA0A6", margin: 0 });
  s.addText(String(N), { x: W - MX - 0.5, y: 7.08, w: 0.5, h: 0.3, fontFace: F,
    fontSize: 11, color: RED, bold: true, align: "right", margin: 0 });
  return s;
}

// título + línea de "por qué" (el subtítulo siempre responde: ¿para qué esto?)
function T(s, t, why) {
  s.addText(t, { x: MX, y: 0.36, w: 10.1, h: 0.72, fontFace: F, fontSize: 29,
    bold: true, color: RED, margin: 0, valign: "middle" });
  if (why) s.addText(why, { x: MX, y: 1.10, w: W - 2 * MX - 0.3, h: 0.44,
    fontFace: F, fontSize: 15, color: MUTED, margin: 0 });
}

// frase de cierre al pie de la lámina
function cierre(s, txt, color = INK) {
  s.addText(txt, { x: MX, y: 6.55, w: W - 2 * MX, h: 0.42, fontFace: F,
    fontSize: 15, color, align: "center", margin: 0, italic: true });
}

function divisor(kicker, titulo, sub, min) {
  N += 1;
  const s = pres.addSlide();
  s.background = { color: DARK };
  img(s, "logo_udla_white_trim", { x: MX, y: 0.5, h: 0.34 });
  img(s, "SLB_white_trim", { x: 11.95, y: 0.42, h: 0.48 });
  s.addText(kicker, { x: 0, y: 2.35, w: W, h: 0.45, fontFace: F, fontSize: 15,
    color: "D8969F", bold: true, align: "center", charSpacing: 7, margin: 0 });
  s.addText(titulo, { x: 0.8, y: 2.9, w: W - 1.6, h: 1.25, fontFace: F, fontSize: 44,
    bold: true, color: WHITE, align: "center", margin: 0 });
  s.addText(sub, { x: 2.0, y: 4.25, w: W - 4.0, h: 0.75, fontFace: F, fontSize: 18,
    color: "EBC9CE", align: "center", margin: 0 });
  if (min) s.addText(min, { x: 0, y: 6.72, w: W, h: 0.4, fontFace: F, fontSize: 13,
    color: "AE7B84", align: "center", charSpacing: 2, margin: 0 });
  return s;
}

// tarjeta: encabezado rojo + cuerpo
function card(s, x, y, w, h, head, body, tone = "gray", fs = 13.5) {
  const bg = tone === "pink" ? PINK : tone === "dark" ? DARK : GRAY;
  s.addShape("roundRect", { x, y, w, h, rectRadius: 0.08, fill: { color: bg } });
  s.addText(head, { x: x + 0.26, y: y + 0.18, w: w - 0.52, h: 0.38, fontFace: F,
    fontSize: 16, bold: true, color: tone === "dark" ? WHITE : RED, margin: 0 });
  s.addText(body, { x: x + 0.26, y: y + 0.62, w: w - 0.52, h: h - 0.82, fontFace: F,
    fontSize: fs, color: tone === "dark" ? "F2DDE0" : INK, margin: 0, valign: "top",
    lineSpacing: fs * 1.42 });
}

// nota al margen sin encabezado
function nota(s, x, y, w, h, body, tone = "gray", fs = 14) {
  const bg = tone === "pink" ? PINK : tone === "dark" ? DARK : GRAY;
  s.addShape("roundRect", { x, y, w, h, rectRadius: 0.08, fill: { color: bg } });
  s.addText(body, { x: x + 0.26, y: y + 0.2, w: w - 0.52, h: h - 0.4, fontFace: F,
    fontSize: fs, color: tone === "dark" ? "F2DDE0" : INK, margin: 0, valign: "middle",
    lineSpacing: fs * 1.44 });
}

// fórmula sobre lienzo claro
function formula(s, name, x, y, w, h, pie) {
  s.addShape("roundRect", { x, y, w, h, rectRadius: 0.09, fill: { color: GRAY } });
  const fh = Math.min((h - (pie ? 0.75 : 0.5)), (w - 0.7) / DIM[name]);
  img(s, name, { cx: x + w / 2, y: y + (h - fh - (pie ? 0.42 : 0)) / 2, h: fh });
  if (pie) s.addText(pie, { x: x + 0.3, y: y + h - 0.62, w: w - 0.6, h: 0.48,
    fontFace: F, fontSize: 12.5, color: MUTED, align: "center", margin: 0, lineSpacing: 15.5 });
}

function code(s, x, y, w, txt, fs = 13.5) {
  const h = txt.split("\n").length * fs * 1.46 / 72 + 0.4;
  s.addShape("roundRect", { x, y, w, h, rectRadius: 0.07, fill: { color: GRAY },
    line: { color: LGRAY, width: 0.75 } });
  s.addText("PYTHON", { x: x + 0.2, y: y - 0.13, w: 1.02, h: 0.26, fontFace: F,
    fontSize: 8.5, bold: true, color: WHITE, align: "center", margin: 0,
    fill: { color: RED }, charSpacing: 1 });
  s.addText(txt, { x: x + 0.26, y: y + 0.2, w: w - 0.5, h: h - 0.34, fontFace: FM,
    fontSize: fs, color: "111827", margin: 0, lineSpacing: fs * 1.46, valign: "top" });
  return y + h;
}

function salida(s, x, y, w, txt, fs = 13) {
  const h = txt.split("\n").length * fs * 1.46 / 72 + 0.42;
  s.addShape("roundRect", { x, y, w, h, rectRadius: 0.07, fill: { color: TERM } });
  s.addText(txt, { x: x + 0.26, y: y + 0.21, w: w - 0.5, h: h - 0.34, fontFace: FM,
    fontSize: fs, color: TGREEN, margin: 0, lineSpacing: fs * 1.46, valign: "top" });
  return y + h;
}

function vinetas(s, x, y, w, h, items, fs = 15.5, gap = 12) {
  s.addText(items.map(i => ({ text: i, options: { bullet: { code: "25B8", color: RED },
    color: INK, fontSize: fs, fontFace: F, paraSpaceAfter: gap, breakLine: true } })),
    { x, y, w, h, margin: 0, valign: "top" });
}

function pasos(s, x, y, w, items, fs = 15, dy = 0.72) {
  items.forEach((t, i) => {
    const yy = y + i * dy;
    s.addShape("ellipse", { x, y: yy, w: 0.42, h: 0.42, fill: { color: RED } });
    s.addText(String(i + 1), { x, y: yy, w: 0.42, h: 0.42, fontFace: F, fontSize: 14,
      bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
    s.addText(t, { x: x + 0.62, y: yy - 0.02, w: w - 0.62, h: 0.55, fontFace: F,
      fontSize: fs, color: INK, margin: 0, valign: "top", lineSpacing: fs * 1.4 });
  });
}

const TH = { fill: { color: RED }, color: WHITE, bold: true, fontFace: F,
  fontSize: 14, valign: "middle" };
const td = (i, j, e = {}) => ({ fontFace: F, fontSize: 14, color: INK, valign: "middle",
  bold: j === 0, fill: { color: i % 2 ? WHITE : "FAFAFB" }, ...e });
const BORDE = { type: "solid", pt: 0.75, color: LGRAY };

// ════════════════════════════════════════════════════════════════════════════
//  PORTADA
// ════════════════════════════════════════════════════════════════════════════
{
  N += 1;
  const s = pres.addSlide();
  s.background = { color: WHITE };
  s.addShape("rect", { x: 0, y: 0, w: W, h: 0.18, fill: { color: RED } });
  s.addShape("rect", { x: 0, y: 7.32, w: W, h: 0.18, fill: { color: DARK } });
  img(s, "logo_udla_trim", { x: 1.3, y: 0.95, h: 1.05 });
  img(s, "SLB_trim", { x: 10.7, y: 1.05, h: 0.92 });
  s.addText("Pronosticar la producción de un pozo", {
    x: 0.9, y: 2.85, w: W - 1.8, h: 1.0, fontFace: F, fontSize: 42, bold: true,
    color: DARK, align: "center", margin: 0 });
  s.addShape("rect", { x: W / 2 - 1.5, y: 4.05, w: 3.0, h: 0.04, fill: { color: RED } });
  s.addText("Módulo 5 · Clase 1", { x: 1, y: 4.35, w: W - 2, h: 0.42, fontFace: F,
    fontSize: 19, bold: true, color: RED, align: "center", margin: 0 });
  s.addText("Series de tiempo, exploración de datos y el primer pronóstico", {
    x: 1, y: 4.8, w: W - 2, h: 0.45, fontFace: F, fontSize: 18, color: INK,
    align: "center", margin: 0 });
  s.addText("Carlos Enrique Mosquera Trujillo     ·     SLB Ecuador     ·     2026", {
    x: 1, y: 6.25, w: W - 2, h: 0.4, fontFace: F, fontSize: 13.5, color: MUTED,
    align: "center", margin: 0 });
}

// ════════════════════════════════════════════════════════════════════════════
//  PARTE 0 — EL ENCARGO
// ════════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  T(s, "Empecemos por el problema, no por la herramienta");
  s.addText("Es 31 de enero de 2014. Gerencia arma el presupuesto del año y necesita una sola cosa:",
    { x: MX, y: 1.25, w: W - 2 * MX, h: 0.4, fontFace: F, fontSize: 16, color: INK, margin: 0 });
  s.addShape("roundRect", { x: 2.4, y: 1.8, w: 8.5, h: 1.0, rectRadius: 0.1,
    fill: { color: PINK } });
  s.addText("«¿Cuánto va a producir el pozo 15/9-F-14 los próximos 6 meses?»",
    { x: 2.4, y: 1.8, w: 8.5, h: 1.0, fontFace: F, fontSize: 21, bold: true,
      color: DARK, align: "center", valign: "middle", margin: 0 });
  img(s, "a1_encargo", { cx: W / 2, y: 3.0, w: 9.4 });
  cierre(s, "Su nombre va en ese número. Toda la clase de hoy existe para producirlo con fundamento.", DARK);
}

{
  const s = slide();
  T(s, "Qué es, exactamente, pronosticar",
    "Estimar el valor futuro de algo usando solo lo que se sabe hasta hoy. Tres decisiones lo definen.");
  const c = [
    ["Qué variable", "El número que hay que estimar, con sus unidades.",
     "La producción diaria del pozo, en barriles por día."],
    ["Desde cuándo", "El punto de partida. Todo lo posterior es desconocido.",
     "31 de enero de 2014 — el cierre del último mes."],
    ["Hasta cuándo", "Cuánto hacia adelante. Cuanto más lejos, más error.",
     "6 meses, mes a mes."],
  ];
  c.forEach((v, i) => {
    const x = MX + i * 4.05;
    s.addShape("roundRect", { x, y: 1.85, w: 3.85, h: 2.75, rectRadius: 0.09,
      fill: { color: GRAY } });
    s.addText(v[0], { x: x + 0.26, y: 2.05, w: 3.35, h: 0.4, fontFace: F, fontSize: 18,
      bold: true, color: RED, margin: 0 });
    s.addText(v[1], { x: x + 0.26, y: 2.5, w: 3.35, h: 0.85, fontFace: F, fontSize: 13.5,
      color: INK, margin: 0, valign: "top", lineSpacing: 18.5 });
    s.addText("En nuestro caso", { x: x + 0.26, y: 3.35, w: 3.35, h: 0.28, fontFace: F,
      fontSize: 10.5, bold: true, color: MUTED, charSpacing: 1, margin: 0 });
    s.addText(v[2], { x: x + 0.26, y: 3.62, w: 3.35, h: 0.85, fontFace: F, fontSize: 13.5,
      color: DARK, bold: true, margin: 0, valign: "top", lineSpacing: 18.5 });
  });
  nota(s, MX, 5.0, W - 2 * MX, 1.25,
    "Y una regla que gobierna todo lo demás:  ningún cálculo puede usar información que no existía el 31 de enero de 2014. " +
    "Si el modelo mira aunque sea un día del futuro, el resultado es una ilusión.", "dark", 15);
}

{
  const s = slide();
  T(s, "Hoy no hay herramientas nuevas: hay usos nuevos",
    "Todo lo que necesita el encargo ya lo aprendieron en los módulos anteriores.");
  const rows = [
    [{ text: "Ya lo vieron en", options: TH }, { text: "Aprendieron", options: TH },
     { text: "Hoy lo usamos para", options: TH }],
    ["Módulo 1 · Clase 3", "Este mismo archivo de producción de Volve", "Hacerle la pregunta del presupuesto"],
    ["Módulo 3 · Clase 1", "Regresión lineal — predecir un número", "El modelo que responde el encargo"],
    ["Módulo 3 · Clase 2", "Medir el error con costo de negocio", "Traducir el error a dólares"],
    ["Módulo 3 · Clase 5", "Validar sin hacer trampa", "No dejar que el modelo vea el futuro"],
    ["Módulo 4", "Agrupar sin etiquetas (K-means, DBSCAN)", "Decidir con qué historia entrenar"],
  ].map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: c,
      options: td(i, j, { fontSize: 14, color: j === 2 ? DARK : INK,
        fill: { color: j === 2 ? PINK : (i % 2 ? WHITE : "FAFAFB") } }) })));
  s.addTable(rows, { x: MX, y: 1.85, w: W - 2 * MX, border: BORDE,
    colW: [2.7, 4.9, 4.13], rowH: 0.68, margin: 0.12 });
  cierre(s, "Lo único genuinamente nuevo de hoy es el tiempo: qué le hace a los datos y al modelo.", DARK);
}

{
  const s = slide();
  T(s, "El camino de hoy", "Cuatro preguntas, en orden. Cada una nace de la anterior.");
  const rows = [
    [{ text: "", options: TH }, { text: "La pregunta", options: TH },
     { text: "Qué haremos", options: TH }, { text: "Min", options: { ...TH, align: "right" } }],
    ["1", "¿Con qué datos contamos?", "Conocer el archivo columna por columna", "0 – 20"],
    ["2", "¿Qué tipo de dato es este?", "Entender qué es una serie de tiempo", "20 – 35"],
    ["3", "¿Qué nos dicen estos datos?", "Explorarlos (EDA) con seis herramientas", "35 – 65"],
    ["4", "¿Cómo sabremos si acertamos?", "Medir un pronóstico — en barriles y en dólares", "65 – 80"],
    ["5", "El pronóstico", "Construirlo y entregarlo", "80 – 95"],
    ["", "Su turno", "El mismo encargo, otro pozo", "95 – 120"],
  ].map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: c,
      options: td(i, j, { fontSize: 15, align: j === 3 ? "right" : "left",
        color: j === 0 ? RED : INK,
        fill: { color: i === 6 ? PINK : (i % 2 ? WHITE : "FAFAFB") } }) })));
  s.addTable(rows, { x: MX, y: 1.9, w: W - 2 * MX, border: BORDE,
    colW: [0.7, 4.4, 6.13, 0.9], rowH: 0.63, margin: 0.12 });
}

// ════════════════════════════════════════════════════════════════════════════
divisor("PREGUNTA 1", "¿Con qué datos contamos?",
  "Conocer el archivo antes de tocarlo — columna por columna", "0 – 20 min");
// ════════════════════════════════════════════════════════════════════════════

{
  const s = slide();
  T(s, "El campo Volve",
    "Un campo real del Mar del Norte, operado por Equinor entre 2008 y 2016.");
  const f2 = [["7", "pozos en el campo"], ["15 634", "días de registro"],
              ["2008 – 2016", "nueve años de historia"]];
  f2.forEach((v, i) => {
    const y = 1.9 + i * 1.35;
    s.addShape("roundRect", { x: MX, y, w: 5.5, h: 1.15, rectRadius: 0.09,
      fill: { color: GRAY } });
    s.addText(v[0], { x: MX + 0.35, y: y + 0.12, w: 2.6, h: 0.9, fontFace: F,
      fontSize: 30, bold: true, color: RED, margin: 0, valign: "middle" });
    s.addText(v[1], { x: MX + 2.9, y: y + 0.12, w: 3.0, h: 0.9, fontFace: F,
      fontSize: 15, color: INK, margin: 0, valign: "middle" });
  });
  card(s, 7.0, 1.9, 5.55, 2.4, "Por qué existe este archivo",
    "Cuando Equinor cerró el campo en 2016, liberó todos sus datos al público. " +
    "Es de los poquísimos campos del mundo con historia completa abierta — por eso " +
    "todo el diplomado se apoya en él.", "pink", 14);
  card(s, 7.0, 4.45, 5.55, 1.75, "Ya lo conocen",
    "Es el mismo archivo con el que armaron su reporte de producción en el Módulo 1. " +
    "Hoy le vamos a hacer una pregunta mucho más exigente.", "gray", 14);
}

{
  const s = slide();
  T(s, "Las seis columnas del archivo",
    "Cada fila es un día de un pozo. Seis columnas, y una de ellas cambia todo lo demás.");
  const rows = [
    [{ text: "Columna", options: TH }, { text: "Qué contiene", options: TH },
     { text: "Unidad", options: TH }],
    ["fecha", "El día al que corresponde el registro", "—"],
    ["pozo", "Cuál de los 7 pozos del campo", "—"],
    ["horas", "Cuántas horas de ese día el pozo estuvo abierto y produciendo", "horas (0 a 24)"],
    ["oil", "Los barriles de petróleo que salieron ese día", "barriles"],
    ["gas", "El gas producido ese día", "pies cúbicos"],
    ["agua", "El agua producida ese día — sube con los años", "barriles"],
  ].map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: c,
      options: td(i, j, { fontSize: 14.5, fontFace: j === 0 ? FM : F,
        color: i === 3 ? RED : INK, bold: j === 0 || i === 3,
        fill: { color: i === 3 ? PINK : (i % 2 ? WHITE : "FAFAFB") } }) })));
  s.addTable(rows, { x: MX, y: 1.85, w: W - 2 * MX, border: BORDE,
    colW: [1.8, 7.43, 2.5], rowH: 0.6, margin: 0.12 });
  cierre(s, "La columna resaltada — horas — es la que casi nadie mira. Y sin ella, todo lo demás se malinterpreta.", DARK);
}

{
  const s = slide();
  T(s, "Cinco días reales del pozo 15/9-F-14",
    "Antes de cualquier análisis: miremos filas de verdad, tal como están en el archivo.");
  const rows = [
    [{ text: "fecha", options: TH }, { text: "horas", options: { ...TH, align: "right" } },
     { text: "oil", options: { ...TH, align: "right" } },
     { text: "agua", options: { ...TH, align: "right" } }],
    ["2010-09-28", "24.0", "2 825", "2 527"],
    ["2010-09-29", "24.0", "2 847", "2 917"],
    ["2010-09-30", "9.5", "1 122", "3 530"],
    ["2010-10-01", "24.0", "2 653", "2 996"],
    ["2010-10-02", "24.0", "2 857", "2 860"],
  ].map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: c,
      options: td(i, j, { fontSize: 15, fontFace: FM, align: j ? "right" : "left",
        color: i === 3 ? RED : INK, bold: i === 3,
        fill: { color: i === 3 ? PINK : (i % 2 ? WHITE : "FAFAFB") } }) })));
  s.addTable(rows, { x: MX, y: 1.9, w: 6.0, border: BORDE,
    colW: [2.1, 1.2, 1.4, 1.3], rowH: 0.6, margin: 0.12 });
  s.addText("El 30 de septiembre produjo 1 122 barriles: menos de la mitad que el día anterior.",
    { x: 7.2, y: 2.1, w: 5.35, h: 0.9, fontFace: F, fontSize: 17, bold: true,
      color: INK, margin: 0, lineSpacing: 24 });
  s.addText("¿Se dañó el pozo?", { x: 7.2, y: 3.15, w: 5.35, h: 0.45, fontFace: F,
    fontSize: 21, bold: true, color: RED, margin: 0 });
  nota(s, 7.2, 3.8, 5.35, 1.6,
    "No. Miren la columna horas: ese día el pozo solo estuvo abierto 9.5 horas de las 24. " +
    "Produjo menos porque operó menos tiempo.", "pink", 14.5);
}

{
  const s = slide();
  T(s, "La misma información, en gráfico",
    "Las dos columnas del 28 de septiembre al 2 de octubre. La forma es idéntica: no es casualidad.");
  img(s, "c1_ventana", { cx: W / 2, y: 1.85, w: 10.6 });
  cierre(s, "Cuando el pozo opera menos horas, produce menos barriles. Obvio — y sin embargo el archivo no lo dice.", DARK);
}

{
  const s = slide();
  T(s, "La corrección: de «barriles del día» a «capacidad del pozo»",
    "Si queremos pronosticar el yacimiento, necesitamos un número que no dependa de cuántas horas abrieron la válvula.");
  formula(s, "f_tasa", MX, 1.85, 5.5, 1.5);
  img(s, "c2_normalizado", { x: 6.6, y: 1.75, w: 5.95 });
  card(s, MX, 3.6, 5.5, 2.6, "Qué acaba de pasar",
    "El 30 de septiembre produjo 1 122 barriles en 9.5 horas. A ese ritmo, en 24 horas " +
    "habría producido 2 834 — exactamente lo mismo que sus días vecinos.\n\n" +
    "El pozo nunca cayó. Solo estuvo cerrado parte del día.", "pink", 14);
  cierre(s, "A este número corregido lo llamaremos la tasa del pozo. Es la variable que vamos a pronosticar.", DARK);
}

{
  const s = slide();
  T(s, "¿Y cuando horas vale cero?",
    "En 333 días del pozo 15/9-F-14 la columna oil marca cero. Vale la pena preguntarse por qué.");
  img(s, "a3_ceros", { x: MX, y: 1.8, w: 8.1 });
  card(s, 9.1, 1.95, 3.45, 2.1, "El cruce que lo resuelve",
    "333 días con oil = 0.\n332 días con horas = 0.\n\nCoinciden casi uno a uno.", "gray", 14);
  card(s, 9.1, 4.25, 3.45, 2.0, "La conclusión",
    "Los ceros no son errores de medición: son el pozo cerrado. Mantenimiento, " +
    "intervención, parada de plataforma.", "pink", 14);
  cierre(s, "Un cero se marca y se excluye — jamás se «rellena»: rellenarlo inventa producción que nunca existió.", DARK);
}

{
  const s = slide();
  T(s, "Nueve años de operación, de un vistazo",
    "Cada celda es un día del pozo 15/9-F-14. Así se ve la diferencia entre «el yacimiento» y «la operación».");
  img(s, "a4_disponibilidad", { cx: W / 2, y: 1.9, w: 11.4 });
  s.addText("rojo intenso = operó las 24 h      ·      rosa = operó parte del día      ·      blanco = cerrado      ·      gris = el día no está en el archivo",
    { x: MX, y: 6.4, w: W - 2 * MX, h: 0.35, fontFace: F, fontSize: 13,
      color: MUTED, align: "center", margin: 0 });
  cierre(s, "Las franjas blancas y rosas no son yacimiento: son decisiones de operación.", DARK);
}

{
  const s = slide();
  T(s, "Última verificación: ¿los 7 pozos son comparables?",
    "Antes de agrupar o promediar, conviene contar cuántos días tiene cada pozo — y cuántos sin petróleo.");
  const rows = [
    [{ text: "pozo", options: TH }, { text: "días", options: { ...TH, align: "right" } },
     { text: "días sin dato de petróleo", options: { ...TH, align: "right" } }],
    ["15/9-F-11", "1 165", "0"],
    ["15/9-F-12", "3 056", "0"],
    ["15/9-F-14", "3 056", "0"],
    ["15/9-F-1 C", "746", "0"],
    ["15/9-F-4", "3 327", "3 327"],
    ["15/9-F-5", "3 306", "3 146"],
  ].map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: c,
      options: td(i, j, { fontSize: 14.5, fontFace: FM, align: j ? "right" : "left",
        color: i >= 5 ? RED : INK, bold: i >= 5,
        fill: { color: i >= 5 ? PINK : (i % 2 ? WHITE : "FAFAFB") } }) })));
  s.addTable(rows, { x: MX, y: 1.9, w: 5.6, border: BORDE,
    colW: [2.2, 1.5, 1.9], rowH: 0.55, margin: 0.1 });
  s.addText("F-4 y F-5 no producen petróleo ni un solo día en nueve años.",
    { x: 6.9, y: 2.15, w: 5.65, h: 1.0, fontFace: F, fontSize: 19, bold: true,
      color: INK, margin: 0, lineSpacing: 26 });
  card(s, 6.9, 3.4, 5.65, 1.5, "No es un error del archivo",
    "Son pozos inyectores: meten agua al yacimiento para mantener la presión. " +
    "Nunca van a producir.", "pink", 14);
  card(s, 6.9, 5.05, 5.65, 1.25, "Lo que enseña",
    "Antes de promediar «los pozos del campo», hay que saber qué es cada fila.", "gray", 14);
}

{
  const s = slide();
  T(s, "Cerrando la pregunta 1: qué vamos a pronosticar",
    "Después de conocer el archivo, la variable del encargo queda definida con precisión.");
  s.addShape("roundRect", { x: 2.1, y: 1.75, w: 9.1, h: 1.35, rectRadius: 0.1,
    fill: { color: PINK } });
  s.addText("La tasa mensual del pozo 15/9-F-14, en barriles por día",
    { x: 2.1, y: 1.75, w: 9.1, h: 1.35, fontFace: F, fontSize: 21, bold: true,
      color: DARK, align: "center", valign: "middle", margin: 0 });
  pasos(s, MX, 3.5, 11.8, [
    "Corregida por horas de operación — mide el pozo, no el calendario de la plataforma",
    "Excluyendo los días cerrados — un cero no es producción baja, es ausencia de producción",
    "Solo del pozo 15/9-F-14 — es un productor, y es el que pidió gerencia",
    "Resumida mes a mes — porque el presupuesto se arma mensual",
  ], 15, 0.78);
  cierre(s, "Nada de esto lo dice el archivo. Lo decidimos nosotros, y por eso hay que poder defenderlo.", DARK);
}

// ════════════════════════════════════════════════════════════════════════════
divisor("PREGUNTA 2", "¿Qué tipo de dato es este?",
  "Por qué una serie de tiempo no se trata como las tablas de los módulos anteriores", "20 – 35 min");
// ════════════════════════════════════════════════════════════════════════════

{
  const s = slide();
  T(s, "Una prueba de un segundo: barajar las filas",
    "En los módulos 3 y 4 barajábamos las filas sin pensarlo — es lo que hace train_test_split.");
  img(s, "c6_barajar", { cx: W / 2, y: 1.8, w: 10.4 });
  card(s, MX, 5.15, 5.85, 1.15, "En Hugoton (Módulo 3)",
    "Cada fila era una profundidad y se explicaba sola. Barajar no destruía nada.", "gray", 14);
  card(s, 6.65, 5.15, 5.9, 1.15, "Aquí",
    "Barajar destruye la información. El orden ERA el dato.", "pink", 14);
}

{
  const s = slide();
  T(s, "Eso es una serie de tiempo",
    "Una tabla donde el orden de las filas es información. Tres propiedades la definen.");
  const c = [
    ["Está ordenada", "Cada fila es un instante, y «antes» y «después» significan algo real."],
    ["Tiene un ritmo", "Se observa cada cierto tiempo: cada día, cada mes. Y puede faltar."],
    ["Se acuerda de sí misma", "Lo de hoy se parece a lo de ayer. Esta es la propiedad clave."],
  ];
  c.forEach((v, i) => {
    const x = MX + i * 4.05;
    s.addShape("roundRect", { x, y: 1.85, w: 3.85, h: 2.0, rectRadius: 0.09,
      fill: { color: i === 2 ? PINK : GRAY } });
    s.addText(v[0], { x: x + 0.26, y: 2.08, w: 3.35, h: 0.4, fontFace: F, fontSize: 17,
      bold: true, color: RED, margin: 0 });
    s.addText(v[1], { x: x + 0.26, y: 2.55, w: 3.35, h: 1.15, fontFace: F, fontSize: 14,
      color: INK, margin: 0, valign: "top", lineSpacing: 19 });
  });
  s.addText("La tercera es la que hace posible pronosticar", { x: MX, y: 4.15, w: 11.8,
    h: 0.45, fontFace: F, fontSize: 19, bold: true, color: DARK, margin: 0 });
  s.addText("Si lo de hoy no dijera nada sobre lo de mañana, no habría nada que aprender: " +
    "el mejor pronóstico posible sería el promedio histórico, y nos podríamos ir a casa. " +
    "Más adelante lo vamos a comprobar con una herramienta específica.",
    { x: MX, y: 4.65, w: 11.8, h: 1.1, fontFace: F, fontSize: 15.5, color: INK,
      margin: 0, lineSpacing: 22 });
  cierre(s, "Y trae una consecuencia incómoda: la forma de validar del Módulo 3 aquí no sirve.", DARK);
}

{
  const s = slide();
  T(s, "De qué está hecha una serie de producción",
    "Separar lo que un modelo puede aprender de lo que no. En un pozo, cuatro cosas ocurren a la vez.");
  img(s, "a2_cruda", { x: 6.5, y: 1.75, w: 6.1 });
  const comp = [
    ["Tendencia", "La declinación: el yacimiento pierde presión y produce cada vez menos.", true],
    ["Estacionalidad", "Ciclos que se repiten. En un pozo, casi no existen: no le importa el mes.", false],
    ["Eventos", "Paradas, intervenciones, cambios de válvula. Son decisiones humanas.", true],
    ["Ruido", "Variación diaria: medición, cabeceo, condiciones del separador.", false],
  ];
  comp.forEach((v, i) => {
    const y = 1.85 + i * 1.12;
    s.addText(v[0], { x: MX, y, w: 5.4, h: 0.34, fontFace: F, fontSize: 16.5, bold: true,
      color: v[2] ? RED : MUTED, margin: 0 });
    s.addText(v[1], { x: MX, y: y + 0.36, w: 5.4, h: 0.7, fontFace: F, fontSize: 13.5,
      color: INK, margin: 0, valign: "top", lineSpacing: 18 });
  });
  nota(s, 6.5, 4.9, 6.1, 1.35,
    "En un pozo mandan la tendencia y los eventos. La estacionalidad — el corazón de " +
    "cualquier curso estándar de series de tiempo — aquí casi no aparece.", "pink", 14);
}

{
  const s = slide();
  T(s, "La consecuencia práctica: cómo se parte el tiempo",
    "Esto contradice algo que aprendieron bien en el Módulo 3 — y hay que decirlo explícitamente.");
  card(s, MX, 1.85, 5.85, 2.1, "Lo que hacían antes",
    "train_test_split reparte las filas al azar: unas para entrenar, otras para evaluar. " +
    "Correcto cuando las filas son independientes.", "gray", 14.5);
  card(s, 6.65, 1.85, 5.9, 2.1, "Lo que hay que hacer ahora",
    "Cortar por fecha: todo lo anterior al 31-ene-2014 entrena, lo posterior evalúa. " +
    "Nunca al azar.", "pink", 14.5);
  nota(s, MX, 4.3, W - 2 * MX, 1.9,
    "¿Por qué? Porque al repartir al azar, el modelo entrena con días de junio y lo evalúan " +
    "con días de marzo — es decir, ya vio el futuro. El resultado se ve espectacular y no " +
    "sirve para nada.\n\nEn la Clase 2 lo veremos con números: el mismo modelo pasa de " +
    "«excelente» a «inservible» solo por cambiar cómo se parten los datos.", "dark", 15);
}

// ════════════════════════════════════════════════════════════════════════════
divisor("PREGUNTA 3", "¿Qué nos dicen estos datos?",
  "Explorarlos antes de modelar — seis herramientas, cada una con su para qué", "35 – 65 min");
// ════════════════════════════════════════════════════════════════════════════

{
  const s = slide();
  T(s, "Explorar los datos: qué es y por qué no se salta",
    "EDA — Análisis Exploratorio de Datos. Mirar los datos con preguntas concretas antes de modelar.");
  const q = [
    ["¿Cómo se comporta el pozo?", "Su forma, su ritmo de caída, si cambió de comportamiento."],
    ["¿Qué le pasó en el camino?", "Los eventos que un modelo nunca podría predecir."],
    ["¿Hay algo que pronosticar?", "Si la serie no tiene memoria, ningún modelo va a ayudar."],
    ["¿Con qué parte entrenamos?", "No toda la historia sirve. Hay que decidir cuál."],
  ];
  q.forEach((v, i) => {
    const x = MX + (i % 2) * 6.1, y = 1.9 + Math.floor(i / 2) * 1.85;
    s.addShape("roundRect", { x, y, w: 5.85, h: 1.65, rectRadius: 0.09,
      fill: { color: GRAY } });
    s.addText(v[0], { x: x + 0.28, y: y + 0.18, w: 5.3, h: 0.4, fontFace: F,
      fontSize: 16.5, bold: true, color: RED, margin: 0 });
    s.addText(v[1], { x: x + 0.28, y: y + 0.62, w: 5.3, h: 0.85, fontFace: F,
      fontSize: 14, color: INK, margin: 0, valign: "top", lineSpacing: 19 });
  });
  nota(s, MX, 5.7, W - 2 * MX, 1.0,
    "Si nos saltamos esto, entregamos un número que nadie puede defender en el comité de presupuesto.",
    "dark", 15);
}

// ── HERRAMIENTA 1 ────────────────────────────────────────────────────────────
{
  const s = slide();
  T(s, "Herramienta 1 — La escala logarítmica: qué es",
    "Antes de usarla: en una escala normal, los espacios iguales son restas iguales. En una logarítmica, son multiplicaciones iguales.");
  img(s, "c3_que_es_log", { cx: W / 2, y: 1.95, w: 10.3 });
  cierre(s, "Es solo otra forma de estirar el eje vertical. Nada del dato cambia — cambia lo que se vuelve visible.", DARK);
}

{
  const s = slide();
  T(s, "Herramienta 1 — Para qué sirve en un pozo",
    "Una declinación pierde un porcentaje fijo cada mes. Eso, en escala logarítmica, se ve como una línea recta.");
  img(s, "a5_escalas", { cx: W / 2, y: 1.85, w: 11.0 });
  card(s, MX, 5.6, 5.85, 1.1, "Qué acabamos de aprender",
    "La caída del F-14 es casi una recta: pierde un porcentaje parecido todos los meses.", "gray", 13.5);
  card(s, 6.65, 5.6, 5.9, 1.1, "Y esto lo vamos a usar",
    "Si en esta escala es una recta, entonces una recta será nuestro modelo.", "pink", 13.5);
}

// ── HERRAMIENTA 2 ────────────────────────────────────────────────────────────
{
  const s = slide();
  T(s, "Herramienta 2 — La media móvil: qué es",
    "El dato diario salta demasiado para ver hacia dónde va el pozo. La media móvil promedia una ventana que se va deslizando.");
  img(s, "c4_que_es_ma", { cx: W / 2, y: 1.85, w: 8.9 });
  formula(s, "f_ma", 3.6, 5.35, 6.1, 1.05);
  s.addText("Cada punto rojo es el promedio de los últimos k días. Nada más que eso.",
    { x: MX, y: 6.62, w: W - 2 * MX, h: 0.36, fontFace: F, fontSize: 14.5,
      color: DARK, align: "center", margin: 0, italic: true });
}

{
  const s = slide();
  T(s, "Herramienta 2 — Elegir la ventana es una decisión",
    "El mismo pozo con tres ventanas distintas. No hay una correcta: hay una adecuada a la pregunta.");
  img(s, "a6_ventanas", { x: MX, y: 1.85, w: 8.35 });
  card(s, 9.4, 1.95, 3.15, 2.0, "Ventana corta", "Sigue de cerca la serie, pero también sigue el ruido.", "gray", 13.5);
  card(s, 9.4, 4.15, 3.15, 2.1, "Ventana larga", "Muy suave, pero llega tarde: en la caída sigue mostrando valores viejos.", "pink", 13.5);
  cierre(s, "Toda media móvil llega tarde: describe el pasado reciente, no el presente.", DARK);
}

{
  const s = slide();
  T(s, "Herramienta 2 — Un ajuste necesario: la mediana",
    "Los días cerrados valen cero. Un solo cero dentro de la ventana arrastra el promedio hacia abajo.");
  img(s, "a7_mediana", { x: MX, y: 1.85, w: 8.35 });
  card(s, 9.4, 2.1, 3.15, 1.9, "La media", "Baja cada vez que hay un cierre — aunque el pozo esté bien.", "gray", 13.5);
  card(s, 9.4, 4.2, 3.15, 2.05, "La mediana", "Toma el valor del medio: mientras menos de la mitad sean ceros, ni se entera.", "pink", 13.5);
  cierre(s, "Por eso en producción la mediana móvil es el resumen por defecto.", DARK);
}

// ── HERRAMIENTA 3 ────────────────────────────────────────────────────────────
{
  const s = slide();
  T(s, "Herramienta 3 — La autocorrelación: qué es",
    "Responde la pregunta que dejamos abierta: ¿se acuerda la serie de sí misma? Se comprueba corriéndola y comparándola consigo misma.");
  img(s, "c5_que_es_acf", { cx: W / 2, y: 1.75, w: 9.2 });
  formula(s, "f_corr", 3.9, 5.42, 5.5, 0.98);
  s.addText("0.93 significa: sabiendo la producción de hace tres meses, ya sé casi toda la de este mes.",
    { x: MX, y: 6.6, w: W - 2 * MX, h: 0.36, fontFace: F, fontSize: 14.5,
      color: DARK, align: "center", margin: 0, italic: true });
}

{
  const s = slide();
  T(s, "Herramienta 3 — El resultado para nuestro pozo",
    "Repetimos esa correlación para cada retraso posible, de 1 a 180 días. Eso es un correlograma.");
  img(s, "a8_acf", { x: MX, y: 1.85, w: 8.3 });
  card(s, 9.35, 2.0, 3.2, 2.05, "Cómo se lee", "Cada barra es una correlación. Alta y bajando despacio = mucha memoria.", "gray", 13.5);
  card(s, 9.35, 4.25, 3.2, 2.0, "El veredicto", "Sí hay algo que pronosticar. Podemos seguir.", "pink", 14);
}

// ── HALLAZGOS ────────────────────────────────────────────────────────────────
{
  const s = slide();
  T(s, "Un hallazgo mientras explorábamos: 20 filas imposibles",
    "Filtrando la columna horas aparecen 20 días con más de 24 horas. Parecen errores de captura.");
  img(s, "a9_dst", { x: MX, y: 1.8, w: 8.2 });
  card(s, 9.2, 1.95, 3.35, 2.3, "El patrón", "Los 20 caen el último domingo de octubre. Y los últimos domingos de marzo marcan 23 h.", "gray", 13.5);
  card(s, 9.2, 4.45, 3.35, 1.8, "La explicación", "Es el cambio de horario noruego: hay días de 23 y de 25 horas.", "pink", 14);
  cierre(s, "El medidor no se equivocó: el calendario es así. El eje del tiempo también es un dato que se audita.", DARK);
}

{
  const s = slide();
  T(s, "Otro hallazgo: el pozo 15/9-F-12 en diciembre de 2014",
    "Este pozo estuvo cerrado un mes entero. Cuando reabrió, produjo cinco veces más y con mucha menos agua.");
  img(s, "a10_evento", { x: MX, y: 1.8, w: 7.1 });
  card(s, 8.2, 2.0, 4.35, 1.85, "¿Qué pasó aquí?", "Eso no lo hace un yacimiento solo. El dato dice dónde; la experiencia de ustedes dice qué.", "gray", 14);
  card(s, 8.2, 4.05, 4.35, 2.2, "Lo que significa para el modelo",
    "Un modelo pronostica el yacimiento, no las decisiones de la gente. Por eso el pronóstico se entrega " +
    "con una condición: «válido mientras no haya intervención».", "pink", 13.5);
}

// ── HERRAMIENTA 4 ────────────────────────────────────────────────────────────
{
  const s = slide();
  T(s, "Herramienta 4 — Describir cada mes con dos números",
    "Para agrupar meses parecidos primero hay que decir en qué se parecen. Elegimos dos características de cada mes.");
  img(s, "c7_atributos", { x: MX, y: 1.8, w: 6.5 });
  card(s, 7.7, 1.95, 4.85, 1.75, "Característica 1 — la tasa",
    "Cuántos barriles por día produce el pozo ese mes (ya corregida por horas).", "gray", 14);
  card(s, 7.7, 3.9, 4.85, 1.75, "Característica 2 — el corte de agua",
    "Qué fracción de todo el líquido que sale es agua. Sube con los años.", "gray", 14);
  nota(s, 7.7, 5.85, 4.85, 0.85,
    "Cada mes queda convertido en un punto del gráfico.", "pink", 14);
}

{
  const s = slide();
  T(s, "Herramienta 4 — K-means agrupa los puntos vecinos",
    "El mismo K-means del Módulo 4: le pedimos tres grupos y él encuentra qué meses se parecen entre sí. Nadie le dice cuáles.");
  img(s, "a11_regimenes", { cx: W / 2, y: 1.9, w: 11.0 });
  s.addText("Encontró las tres etapas de la vida del pozo — y ninguna estaba etiquetada en el archivo.",
    { x: MX, y: 6.6, w: W - 2 * MX, h: 0.36, fontFace: F, fontSize: 15,
      color: DARK, align: "center", margin: 0, italic: true });
}

{
  const s = slide();
  T(s, "Herramienta 5 — DBSCAN señala los días que no encajan",
    "El otro algoritmo del Módulo 4. A diferencia de K-means, este puede decir «este punto no pertenece a ningún grupo».");
  img(s, "a12_anomalias", { cx: W / 2, y: 1.85, w: 11.0 });
  card(s, MX, 5.55, 5.85, 1.15, "Qué son esos días",
    "Caídas bruscas, picos de medición, arranques después de un cierre.", "gray", 13.5);
  card(s, 6.65, 5.55, 5.9, 1.15, "Para qué nos sirve",
    "Es la lista de días a revisar antes de entrenar cualquier modelo.", "pink", 13.5);
}

{
  const s = slide();
  T(s, "Lo que la exploración nos dejó decidido",
    "El EDA no era un trámite: de aquí salieron seis decisiones concretas para el pronóstico.");
  const rows = [
    [{ text: "Lo que vimos", options: TH }, { text: "Lo que decidimos", options: TH }],
    ["La columna horas explica caídas que no son del pozo", "Trabajar con la tasa corregida, no con oil"],
    ["Los ceros son cierres operativos", "Excluir esos días del entrenamiento"],
    ["F-4 y F-5 son inyectores", "Quedarnos solo con el pozo del encargo"],
    ["En escala logarítmica la caída es una recta", "Nuestro modelo será una recta sobre el logaritmo"],
    ["La serie tiene mucha memoria", "Sí hay algo que pronosticar: seguimos"],
    ["El pozo vivió tres etapas distintas", "Entrenar solo con la etapa actual, no con todo"],
  ].map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: c,
      options: td(i, j, { fontSize: 14, bold: j === 1, color: j === 1 ? DARK : INK,
        fill: { color: j === 1 ? PINK : (i % 2 ? WHITE : "FAFAFB") } }) })));
  s.addTable(rows, { x: MX, y: 1.85, w: W - 2 * MX, border: BORDE,
    colW: [6.0, 5.83], rowH: 0.72, margin: 0.12 });
  cierre(s, "La última decisión es la que más va a valer. Ya veremos cuánto.", DARK);
}

// ════════════════════════════════════════════════════════════════════════════
divisor("PREGUNTA 4", "¿Cómo sabremos si acertamos?",
  "Antes de construir el pronóstico: definir cómo se juzga", "65 – 80 min");
// ════════════════════════════════════════════════════════════════════════════

{
  const s = slide();
  T(s, "Primero: contra qué nos comparamos",
    "Un error de 96 barriles por día, ¿es bueno o malo? La pregunta no tiene respuesta sin una referencia.");
  s.addShape("roundRect", { x: 2.2, y: 1.8, w: 8.9, h: 1.35, rectRadius: 0.1,
    fill: { color: PINK } });
  s.addText("«El mes que viene producirá lo mismo que este mes»",
    { x: 2.2, y: 1.8, w: 8.9, h: 1.35, fontFace: F, fontSize: 21, bold: true,
      color: DARK, align: "center", valign: "middle", margin: 0 });
  s.addText("Ese es el pronóstico ingenuo: el más tonto posible.",
    { x: MX, y: 3.35, w: 11.8, h: 0.4, fontFace: F, fontSize: 16, color: INK,
      align: "center", margin: 0 });
  vinetas(s, 2.2, 4.0, 9.0, 2.1, [
    "No cuesta nada: no tiene parámetros ni requiere entrenamiento",
    "Es sorprendentemente difícil de vencer cuando la serie tiene mucha memoria",
    "Define el mínimo aceptable: un modelo que no le gana, no se presenta",
  ], 15.5, 14);
  cierre(s, "En la Clase 2 este pronóstico «tonto» le va a ganar 13 a 1 a un Random Forest. Conviene llegar con humildad.", DARK);
}

{
  const s = slide();
  T(s, "La medida principal: el error típico (MAE)",
    "Para cada mes, cuánto nos equivocamos. Después promediamos. Eso es todo.");
  formula(s, "f_mae", MX, 1.8, 6.0, 1.5);
  const rows = [
    [{ text: "Mes", options: TH }, { text: "Real", options: { ...TH, align: "right" } },
     { text: "Pronóstico", options: { ...TH, align: "right" } },
     { text: "Error", options: { ...TH, align: "right" } }],
    ["feb", "599", "560", "39"],
    ["mar", "539", "540", "1"],
    ["abr", "576", "521", "55"],
    ["may", "458", "503", "45"],
  ].map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: c,
      options: td(i, j, { fontSize: 14, fontFace: FM, align: j ? "right" : "left" }) })));
  s.addTable(rows, { x: 7.2, y: 1.8, w: 5.35, border: BORDE,
    colW: [1.15, 1.35, 1.6, 1.25], rowH: 0.48, margin: 0.1 });
  s.addText("MAE  =  (39 + 1 + 55 + 45) / 4  =  35 barriles por día",
    { x: 7.2, y: 4.3, w: 5.35, h: 0.7, fontFace: F, fontSize: 15, bold: true,
      color: DARK, margin: 0, lineSpacing: 21 });
  nota(s, MX, 3.6, 6.0, 1.4,
    "Se lee directo: «en un mes cualquiera, este pronóstico se equivoca por unos 35 barriles al día».",
    "pink", 14.5);
  cierre(s, "Se llama MAE — error absoluto medio — pero lo importante es que se entiende sin traducción.", DARK);
}

{
  const s = slide();
  T(s, "Otras tres medidas, para otras tres preguntas",
    "Ninguna es «la buena». Cada una responde algo distinto, y conviene reportar más de una.");
  const m = [["f_rmse", "¿Qué tan graves son mis peores meses?", "Castiga los errores grandes mucho más que los chicos."],
             ["f_mape", "¿En qué porcentaje me equivoco?", "Cuidado: se dispara cuando la producción es casi cero."],
             ["f_sesgo", "¿Me equivoco siempre para el mismo lado?", "Si sale positivo, me quedo corto sistemáticamente."]];
  m.forEach((v, i) => {
    const x = MX + i * 4.05;
    s.addShape("roundRect", { x, y: 1.85, w: 3.85, h: 3.9, rectRadius: 0.09,
      fill: { color: GRAY } });
    img(s, v[0], { cx: x + 1.93, y: 2.15, w: 3.2, maxH: 0.85 });
    s.addText(v[1], { x: x + 0.26, y: 3.55, w: 3.35, h: 0.75, fontFace: F, fontSize: 15,
      bold: true, color: RED, margin: 0, valign: "top", lineSpacing: 20 });
    s.addText(v[2], { x: x + 0.26, y: 4.35, w: 3.35, h: 1.15, fontFace: F, fontSize: 13,
      color: INK, margin: 0, valign: "top", lineSpacing: 18 });
  });
  cierre(s, "En la práctica: se reportan siempre el error típico y el sesgo. Los otros, según el caso.", DARK);
}

{
  const s = slide();
  T(s, "Por qué conviene mirar más de una",
    "Dos pronósticos del mismo pozo. Uno gana con una medida y pierde con la otra. ¿Cuál prefiere la operación?");
  img(s, "a14_metricas", { cx: W / 2, y: 1.85, w: 11.0 });
  cierre(s, "Si las facilidades no toleran un mes con 480 barriles de error, la respuesta es A — aunque «en promedio» B parezca mejor.", DARK);
}

{
  const s = slide();
  T(s, "El error no es un número: crece con la distancia",
    "Cuánto se equivoca el pronóstico ingenuo según cuántos meses adelante le pidamos.");
  img(s, "a13_horizonte", { x: MX, y: 1.85, w: 8.2 });
  card(s, 9.2, 2.05, 3.35, 2.0, "Lo que implica",
    "Un solo número de error no describe nada. El pronóstico se entrega como una tabla: a un mes, a tres, a seis.", "gray", 13.5);
  card(s, 9.2, 4.25, 3.35, 2.0, "Y por eso",
    "El horizonte se acuerda con quien va a usar el pronóstico, antes de modelar.", "pink", 14);
}

{
  const s = slide();
  T(s, "El error, traducido al idioma de gerencia",
    "Un error en barriles por día se convierte en dinero con una multiplicación.");
  img(s, "a15_costo", { x: MX, y: 1.9, w: 7.9 });
  formula(s, "f_costo", 8.9, 2.05, 3.65, 1.3);
  card(s, 8.9, 3.6, 3.65, 1.5, "Ejemplo",
    "Equivocarse en 100 barriles al día durante un mes ≈ 210 mil dólares.", "pink", 14);
  card(s, 8.9, 5.25, 3.65, 1.35, "Y no es simétrico",
    "Pasarse suele costar más que quedarse corto.", "gray", 14);
}

// ════════════════════════════════════════════════════════════════════════════
divisor("PREGUNTA 5", "El pronóstico",
  "Con todo lo anterior decidido, construirlo toma tres líneas", "80 – 95 min");
// ════════════════════════════════════════════════════════════════════════════

{
  const s = slide();
  T(s, "La idea del modelo ya la vimos en la exploración",
    "En escala logarítmica, la caída del pozo era casi una línea recta. Entonces ajustemos una recta.");
  formula(s, "f_modelo", 2.6, 1.85, 8.1, 1.4,
    "el logaritmo de la tasa baja de forma aproximadamente constante mes a mes");
  card(s, MX, 3.6, 5.85, 2.6, "Es la regresión del Módulo 3",
    "Misma clase, misma sintaxis, mismo .fit(). Lo único distinto: la variable de entrada " +
    "es el tiempo, y la de salida es el logaritmo de la tasa.", "gray", 14.5);
  card(s, 6.65, 3.6, 5.9, 2.6, "Por qué una recta y no un árbol",
    "Un árbol de decisión nunca predice un valor fuera del rango que vio entrenando. " +
    "Un pozo en declinación siempre sale de ese rango. Una recta sí puede seguir bajando.", "pink", 14.5);
}

{
  const s = slide();
  T(s, "El código completo",
    "Tres líneas. Todo el trabajo difícil ya lo hicimos en la exploración.");
  let y = code(s, MX, 1.9, 7.0,
"from sklearn.linear_model import LinearRegression\n\n" +
"# el corte honesto: solo lo anterior al origen\n" +
"tr = mensual[mensual.index <= '2014-01-31']\n\n" +
"t  = np.arange(len(tr)).reshape(-1, 1)\n" +
"modelo = LinearRegression().fit(t, np.log(tr))", 13.5);
  card(s, 8.2, 1.9, 4.35, 2.1, "La línea que importa",
    "El filtro por fecha. Sin él, el modelo entrena con datos que el 31 de enero de 2014 " +
    "todavía no existían.", "pink", 14);
  card(s, 8.2, 4.2, 4.35, 2.05, "Y para pronosticar",
    "Se evalúa la recta en los meses futuros y se deshace el logaritmo con np.exp().", "gray", 14);
}

{
  const s = slide();
  T(s, "La decisión que multiplica por tres la precisión",
    "El modelo es el mismo en los tres casos. Lo único que cambia es con qué parte de la historia lo entrenamos.");
  img(s, "a16_modelo", { cx: W / 2, y: 1.85, w: 10.6 });
  cierre(s, "Con toda la historia, la recta PIERDE contra el pronóstico ingenuo. Con solo el último año, le gana tres veces.", DARK);
}

{
  const s = slide();
  T(s, "Por qué pasa eso — y de dónde salió la respuesta",
    "La recta ajustada con nueve años mezcla tres pozos distintos: el de 2008, el de 2011 y el de 2013.");
  img(s, "a11_regimenes", { x: 6.0, y: 1.9, w: 6.55 });
  s.addText("K-means ya nos lo había dicho", { x: MX, y: 2.0, w: 5.0, h: 0.45,
    fontFace: F, fontSize: 19, bold: true, color: DARK, margin: 0 });
  s.addText("En la exploración encontramos que el pozo vivió tres etapas. " +
    "Entrenar con las tres es pedirle a una sola recta que describa tres comportamientos distintos.\n\n" +
    "Entrenar solo con la etapa actual — el último año — es darle al modelo el pozo que " +
    "tiene enfrente hoy.",
    { x: MX, y: 2.55, w: 5.0, h: 2.6, fontFace: F, fontSize: 14.5, color: INK,
      margin: 0, valign: "top", lineSpacing: 21 });
  nota(s, MX, 5.35, 5.0, 1.15,
    "El EDA no era decoración: fue el que eligió el tramo de entrenamiento.", "pink", 14.5);
}

{
  const s = slide();
  T(s, "El encargo, respondido", "Lo mismo de la lámina anterior, en el idioma del presupuesto.");
  img(s, "a17_resultado", { cx: W / 2, y: 1.8, w: 10.9 });
  nota(s, MX, 5.1, W - 2 * MX, 1.5,
    "Lo que se entrega a gerencia:  pronóstico mensual de febrero a julio de 2014, con un error " +
    "esperado de unos 30 barriles por día — unos 62 mil dólares al mes — entrenado con el " +
    "comportamiento actual del pozo y válido mientras no haya intervención.", "dark", 15);
}

// ════════════════════════════════════════════════════════════════════════════
divisor("SU TURNO", "El pozo 15/9-F-11",
  "El mismo encargo, otro pozo — y una sorpresa", "95 – 120 min");
// ════════════════════════════════════════════════════════════════════════════

{
  const s = slide();
  T(s, "El encargo",
    "Es 30 de junio de 2015. Gerencia ahora pide el 15/9-F-11, el pozo más joven del campo. Mismos 6 meses.");
  pasos(s, MX, 1.95, 11.8, [
    "Conozcan el pozo: días registrados, huecos, días cerrados, días parciales",
    "Construyan la tasa corregida y resúmanla mes a mes",
    "Explórenlo: escala logarítmica, media y mediana móvil, autocorrelación",
    "Identifiquen los tres eventos más importantes de su historia",
    "Calculen el pronóstico ingenuo — ese es el número a vencer",
    "Ajusten la recta: con toda la historia y con el último año. Comparen",
  ], 15, 0.72);
  nota(s, MX, 6.35, W - 2 * MX, 0.85,
    "Entregan una sola cosa: el número que firmarían, y una frase explicando con qué historia lo entrenaron.",
    "pink", 15);
}

{
  const s = slide();
  T(s, "Lo que se llevan hoy");
  vinetas(s, MX, 1.8, 6.3, 4.7, [
    "Resolvieron un encargo real, de la pregunta de gerencia al número entregado",
    "Con la regresión lineal que ya sabían — solo cambió qué se pone en el eje X",
    "Aprendieron a leer un archivo de producción columna por columna",
    "Y a corregir la variable antes de modelar nada",
    "Seis herramientas de exploración, cada una con su para qué",
    "Y a medir el error en barriles y en dólares, contra una referencia",
  ], 14.5, 13);
  nota(s, 7.5, 1.8, 5.05, 1.6,
    "La regla que gobierna todo el módulo:  ningún cálculo puede usar información que no " +
    "existía en el momento que se está describiendo.", "dark", 14.5);
  card(s, 7.5, 3.6, 5.05, 1.75, "La próxima clase",
    "Curvas de declinación de Arps, y por qué el mismo modelo puede parecer excelente " +
    "y ser inservible según cómo se partan los datos.", "pink", 14);
  s.addText("Para pensar: si el pronóstico ingenuo le gana a su modelo, ¿qué entrega usted?",
    { x: 7.5, y: 5.6, w: 5.05, h: 0.9, fontFace: F, fontSize: 14.5, italic: true,
      color: INK, margin: 0, lineSpacing: 20 });
}

{
  const s = slide();
  T(s, "Datos y referencias");
  vinetas(s, MX, 1.9, 11.8, 3.6, [
    "Campo Volve — Equinor ASA, dataset abierto (2018). Producción diaria de 7 pozos, 2007–2016",
    "Hyndman & Athanasopoulos — Forecasting: Principles and Practice, 3.ª edición, gratuito en otexts.com/fpp3",
    "pandas — guía de series de tiempo: pandas.pydata.org/docs/user_guide/timeseries.html",
    "scikit-learn — LinearRegression, KMeans y DBSCAN: scikit-learn.org",
    "Arps, J. J. (1945) — Analysis of Decline Curves. La referencia de la próxima clase",
  ], 15, 15);
  nota(s, MX, 5.6, W - 2 * MX, 1.1,
    "Todo el material está en el repositorio del diplomado: esta presentación, el cuaderno de " +
    "Google Colab (los datos se cargan solos desde internet) y una chuleta de una página.", "gray", 14.5);
}

pres.writeFile({ fileName: "presentacion.pptx" }).then(() =>
  console.log(`OK — ${N} láminas`));

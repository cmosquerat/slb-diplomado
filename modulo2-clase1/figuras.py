"""
Machine Learning for Petroleum Engineers Using Python
Modulo 2 - Clase 1: explorar datos. ¿Cuánto cuesta un cliente?

Genera todas las figuras de la presentación (fig_*.png) y, al final,
IMPRIME todas las cifras que aparecen en las láminas. Ninguna cifra de la
presentación se escribe a mano: se copia de esta salida. El cuaderno de la
clase tiene que dar exactamente los mismos números.

Dataset: datos/insurance.csv (Lantz, Machine Learning with R; datos
simulados con estadísticas del censo de EE. UU.).

Uso:  python3 figuras.py
Fuente de las figuras: Fira Sans, desde la variable de entorno FIRA_DIR o
desde ./_fuentes/. Si no la encuentra, usa la sans por defecto y avisa.
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
DATOS = AQUI.parent / "datos" / "insurance.csv"

# ---------------------------------------------------------------- estilo ---
AZUL = "#2a78d6"      # serie principal (paleta de referencia, slot 1)
NARANJA = "#eb6834"   # media (slot 2)
VERDE = "#1baf7a"     # mediana (slot 3)
TINTA = "#2D2D2D"
GRIS = "#6B7280"
REJILLA = "#E5E7EB"

fira = Path(os.environ.get("FIRA_DIR", AQUI / "_fuentes"))
ttfs = list(fira.glob("FiraSans-*.ttf")) if fira.exists() else []
if ttfs:
    for f in ttfs:
        font_manager.fontManager.addfont(str(f))
    plt.rcParams["font.family"] = "Fira Sans"
else:
    print("AVISO: no encontré Fira Sans; uso la sans por defecto.")

plt.rcParams.update({
    "font.size": 13,
    "axes.edgecolor": GRIS,
    "axes.labelcolor": TINTA,
    "axes.titlesize": 15,
    "axes.titleweight": "medium",
    "axes.titlecolor": TINTA,
    "axes.titlelocation": "left",
    "axes.titlepad": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": REJILLA,
    "grid.linewidth": 1,
    "axes.axisbelow": True,
    "xtick.color": GRIS,
    "ytick.color": GRIS,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})


def miles(x, dec=0):
    """Número con formato castellano: 13.270 o 30,7."""
    s = f"{x:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def guardar(fig, nombre):
    fig.savefig(AQUI / nombre)
    plt.close(fig)
    print(f"  escrita {nombre}")


def eje_dolares(ax, eje="x"):
    f = matplotlib.ticker.FuncFormatter(
        lambda v, _: "0" if v == 0 else f"{miles(v / 1000)} mil")
    (ax.xaxis if eje == "x" else ax.yaxis).set_major_formatter(f)


def linea_vertical(ax, x, color, texto, xy_texto):
    """Línea vertical con su etiqueta en un hueco del gráfico y una flecha."""
    ax.axvline(x, color=color, lw=2.2)
    y_flecha = xy_texto[1]
    ax.annotate(texto, xy=(x, y_flecha), xytext=xy_texto, color=TINTA,
                ha="left" if xy_texto[0] > x else "right", va="center",
                fontsize=13, fontweight="medium",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color,
                          lw=1.5),
                arrowprops=dict(arrowstyle="-", color=color, lw=1.5))


# ----------------------------------------------------------------- datos ---
df = pd.read_csv(DATOS)
c = df["charges"]
media, mediana = c.mean(), c.median()
p10, p25, p75, p90 = c.quantile([0.10, 0.25, 0.75, 0.90])
iqr = p75 - p25
bigote_sup = p75 + 1.5 * iqr

print("Escribiendo figuras en", AQUI)

# 1 · edad: histograma de a un año -------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.hist(df["age"], bins=np.arange(17.5, 65.5, 1), color=AZUL,
        edgecolor="white", linewidth=1.2)
ax.set_xlabel("Edad (años)")
ax.set_ylabel("Número de clientes")
ax.set_title("Edad: casi pareja, salvo los de 18 y 19 años")
guardar(fig, "fig_hist_edad.png")

# 2 · IMC: simétrico, media y mediana casi iguales ----------------------------
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.hist(df["bmi"], bins=np.arange(15, 54.5, 1.5), color=AZUL,
        edgecolor="white", linewidth=1.2)
ax.set_xlabel("Índice de masa corporal (kg/m²)")
ax.set_ylabel("Número de clientes")
ax.set_title("IMC: forma de campana, centrada cerca de 30")
ax.set_ylim(0, 150)
linea_vertical(ax, df["bmi"].mean(), NARANJA,
               f"media {miles(df['bmi'].mean(), 1)}", (40, 138))
linea_vertical(ax, df["bmi"].median(), VERDE,
               f"mediana {miles(df['bmi'].median(), 1)}", (21, 138))
guardar(fig, "fig_hist_imc.png")

# 3 · costo: sesgado, la media se va a la derecha ----------------------------
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.hist(c, bins=np.arange(0, 66000, 2000), color=AZUL,
        edgecolor="white", linewidth=1.2)
eje_dolares(ax)
ax.set_xlabel("Costo médico del año (USD)")
ax.set_ylabel("Número de clientes")
ax.set_title("Costo: muchos clientes baratos y una cola larga de caros")
ax.set_ylim(0, 200)
linea_vertical(ax, mediana, VERDE, f"mediana {miles(mediana)}", (8300, 190))
linea_vertical(ax, media, NARANJA, f"media {miles(media)}", (14300, 190))
guardar(fig, "fig_hist_costo.png")

# 3b · versiones sin líneas, para la parte de formas (antes de media/mediana)
for col, nom, xl, binsx, tit in [
        ("charges", "fig_hist_costo_simple.png", "Costo médico del año (USD)",
         np.arange(0, 66000, 2000), "Costo: un montón a la izquierda y una cola larga a la derecha"),
        ("bmi", "fig_hist_imc_simple.png", "Índice de masa corporal (kg/m²)",
         np.arange(15, 54.5, 1.5), "IMC: forma de campana, centrada cerca de 30")]:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.hist(df[col], bins=binsx, color=AZUL, edgecolor="white", linewidth=1.2)
    if col == "charges":
        eje_dolares(ax)
    ax.set_xlabel(xl)
    ax.set_ylabel("Número de clientes")
    ax.set_title(tit)
    guardar(fig, nom)

# 4 · histograma + boxplot del costo, mismo eje -------------------------------
fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.3],
                                          "hspace": 0.08})
a1.hist(c, bins=np.arange(0, 66000, 2000), color=AZUL, edgecolor="white",
        linewidth=1.2)
a1.set_ylabel("Clientes")
a1.set_title("El boxplot resume el histograma en una línea")
a2.boxplot(c, orientation="horizontal", widths=0.55, patch_artist=True,
           boxprops=dict(facecolor="#d6e6f8", edgecolor=AZUL, lw=1.8),
           medianprops=dict(color=VERDE, lw=3),
           whiskerprops=dict(color=AZUL, lw=1.8),
           capprops=dict(color=AZUL, lw=1.8),
           flierprops=dict(marker="o", markersize=5, markerfacecolor="white",
                           markeredgecolor=AZUL, alpha=0.8))
a2.set_yticks([])
a2.grid(False)
a2.spines["left"].set_visible(False)
eje_dolares(a2)
a2.set_xlabel("Costo médico del año (USD)")
for x, t, y in [(p25, "P25", 1.42), (mediana, "mediana", 1.68),
                (p75, "P75", 1.42), (bigote_sup, "fin del bigote", 1.42)]:
    a2.annotate(t, (x, 1.28), (x, y), ha="center", va="bottom", fontsize=11,
                color=TINTA, arrowprops=dict(arrowstyle="-", color=GRIS))
a2.text(52000, 1.68, "puntos sueltos: «atípicos» según la regla",
        ha="center", va="bottom", fontsize=11, color=TINTA)
a2.set_ylim(0.55, 1.95)
guardar(fig, "fig_hist_box_costo.png")

# 5 · percentiles: los 1.338 clientes ordenados -----------------------------
orden = np.sort(c.values)
pos = np.arange(1, len(orden) + 1) / len(orden) * 100
fig, ax = plt.subplots(figsize=(10, 4.4))
ax.fill_between(pos, orden, color="#d6e6f8", step="mid")
ax.plot(pos, orden, color=AZUL, lw=2)
eje_dolares(ax, "y")
ax.set_xlabel("Clientes ordenados del más barato al más caro (%)")
ax.set_ylabel("Costo del año (USD)")
ax.set_xlim(0, 100)
ax.set_title("Un percentil es una posición en la fila ordenada")
for q, val in [(10, p10), (50, mediana), (90, p90)]:
    ax.plot([q, q], [0, val], color=TINTA, lw=1.2, ls="--")
    ax.plot(q, val, "o", color=TINTA, ms=7)
    izq = q > 20
    ax.text(q - 1.2 if izq else q + 1.2, val + 2500, f"P{q} = {miles(val)}",
            ha="right" if izq else "left", va="bottom", fontsize=12,
            color=TINTA, fontweight="medium")
guardar(fig, "fig_percentiles_costo.png")

# 6 · concentración: qué parte del costo pone cada décimo de clientes --------
dec = pd.qcut(c.rank(method="first"), 10, labels=False) + 1
parte = c.groupby(dec).sum() / c.sum() * 100
fig, ax = plt.subplots(figsize=(10, 4.2))
colores = [AZUL] * 9 + [NARANJA]
ax.bar(parte.index, parte.values, color=colores, width=0.72)
for i, v in parte.items():
    ax.text(i, v + 0.6, f"{miles(v, 0)} %", ha="center", fontsize=11,
            color=TINTA)
ax.set_xticks(range(1, 11))
ax.set_xticklabels([f"{i}.º" for i in range(1, 11)])
ax.set_xlabel("Décimo de clientes, del más barato (1.º) al más caro (10.º)")
ax.set_ylabel("% del costo total")
ax.set_title("El 10 % más caro de los clientes pone casi un tercio del costo")
guardar(fig, "fig_concentracion.png")

# 7 · categóricas: conteos en barras ---------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(12, 3.8),
                        gridspec_kw={"width_ratios": [2, 2, 4]})
nombres = {"sex": "Sexo", "smoker": "¿Fuma?", "region": "Región"}
for ax, col in zip(axs, ["sex", "smoker", "region"]):
    vc = df[col].value_counts()
    ax.bar(vc.index, vc.values, color=AZUL, width=0.62)
    for k, v in vc.items():
        ax.text(k, v + 15, miles(v), ha="center", fontsize=11, color=TINTA)
    ax.set_title(nombres[col])
    ax.set_ylim(0, 1200)
    ax.tick_params(axis="x", labelsize=11)
axs[0].set_ylabel("Clientes")
guardar(fig, "fig_categoricas.png")

# 8 · hijos: variable discreta, barras separadas -----------------------------
vc = df["children"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(10, 3.8))
ax.bar(vc.index, vc.values, color=AZUL, width=0.6)
for k, v in vc.items():
    ax.text(k, v + 8, miles(v), ha="center", fontsize=11, color=TINTA)
ax.set_xlabel("Número de hijos cubiertos por el seguro")
ax.set_ylabel("Clientes")
ax.set_title("Hijos: un conteo, solo toma valores enteros")
guardar(fig, "fig_hijos.png")

# 9 · IMC en categorías con orden (ordinal) ---------------------------------
cortes = [0, 18.5, 25, 30, np.inf]
etiquetas = ["Bajo peso", "Normal", "Sobrepeso", "Obesidad"]
cat_imc = pd.cut(df["bmi"], bins=cortes, labels=etiquetas, right=False)
vc_imc = cat_imc.value_counts().reindex(etiquetas)
fig, ax = plt.subplots(figsize=(10, 3.8))
ax.bar(etiquetas, vc_imc.values, color=AZUL, width=0.6)
for k, v in zip(etiquetas, vc_imc.values):
    ax.text(k, v + 8, miles(v), ha="center", fontsize=11, color=TINTA)
ax.set_ylabel("Clientes")
ax.set_title("IMC en categorías: las barras van en su orden, no por tamaño")
guardar(fig, "fig_imc_categorias.png")

# 10 · el ejemplo de cinco clientes: media contra mediana --------------------
toy = np.array([2, 3, 3, 4, 50])
fig, ax = plt.subplots(figsize=(10, 2.6))
ax.scatter(toy, np.zeros_like(toy), s=160, color=AZUL, zorder=3,
           edgecolor="white", linewidth=1.5)
ax.scatter([3], [0.32], s=160, color=AZUL, zorder=3, edgecolor="white",
           linewidth=1.5)
ax.axvline(toy.mean(), color=NARANJA, lw=2.2)
ax.axvline(np.median(toy), color=VERDE, lw=2.2)
ax.text(toy.mean() + 0.6, 0.85, f"media {miles(toy.mean(), 1)}", color=TINTA,
        fontsize=13, fontweight="medium", va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=NARANJA, lw=1.5))
ax.text(np.median(toy) + 0.6, -0.75, "mediana 3", color=TINTA, fontsize=13,
        fontweight="medium", va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=VERDE, lw=1.5))
ax.set_ylim(-1.2, 1.3)
ax.set_yticks([])
ax.grid(False)
ax.spines["left"].set_visible(False)
ax.set_xlim(0, 53)
ax.set_xlabel("Costo del año (miles de USD)")
ax.set_title("Cinco clientes: uno solo arrastra la media")
guardar(fig, "fig_ejemplo_media.png")

# 11 · las jorobas del costo ------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.hist(c, bins=np.arange(0, 66000, 1000), color=AZUL, edgecolor="white",
        linewidth=0.8)
eje_dolares(ax)
ax.set_xlabel("Costo médico del año (USD)")
ax.set_ylabel("Número de clientes")
ax.set_title("Con barras más finas aparecen varias jorobas. ¿Por qué?")
ax.set_ylim(0, 122)
for x, y, t in [(7500, 112, "1"), (20000, 32, "2"), (40000, 24, "3")]:
    ax.text(x, y, t, ha="center", fontsize=15, fontweight="semibold",
            color="white", bbox=dict(boxstyle="circle,pad=0.35", fc=NARANJA,
                                     ec="none"))
guardar(fig, "fig_jorobas.png")

# 12 · cómo se arma un histograma: 20 clientes, punto por punto -------------
edades20 = df["age"].head(20).values
bordes = [15, 25, 35, 45, 55, 65]
fig, ax = plt.subplots(figsize=(10, 4.4))
for i in range(len(bordes) - 1):
    a, b = bordes[i], bordes[i + 1]
    dentro = sorted([e for e in edades20 if a <= e < b])
    ax.bar((a + b) / 2, len(dentro), width=9.2, color="#d6e6f8",
           edgecolor=AZUL, lw=1.5, zorder=1)
    for j, e in enumerate(dentro):
        ax.scatter((a + b) / 2, j + 0.5, s=330, color=AZUL, zorder=3,
                   edgecolor="white", linewidth=1.5)
        ax.text((a + b) / 2, j + 0.5, str(e), color="white", fontsize=9,
                ha="center", va="center", zorder=4, fontweight="medium")
    ax.text((a + b) / 2, len(dentro) + 0.25, f"{len(dentro)} clientes",
            ha="center", va="bottom", fontsize=12, color=TINTA)
ax.set_xticks(bordes)
ax.set_xlim(14, 66)
ax.set_ylim(0, 10.5)
ax.set_xlabel("Edad (años), en intervalos de 10 años")
ax.set_ylabel("Clientes en el intervalo")
ax.set_title("Los primeros 20 clientes: cada punto es una persona con su edad")
guardar(fig, "fig_como_se_arma.png")

# 13 · el ancho de las barras cambia lo que se ve --------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 3.9), sharey=False)
for ax, n, t in [(axs[0], 5, "5 barras: se pierde el detalle"),
                 (axs[1], 60, "60 barras: aparecen detalles y algo de ruido")]:
    ax.hist(c, bins=n, color=AZUL, edgecolor="white", linewidth=0.8)
    eje_dolares(ax)
    ax.set_title(t)
    ax.set_xlabel("Costo médico del año (USD)")
axs[0].set_ylabel("Clientes")
guardar(fig, "fig_bins.png")

# 14 · las formas de una distribución (datos inventados, solo para ver) -----
rng = np.random.default_rng(7)
formas = [
    ("Simétrica", rng.normal(50, 10, 4000)),
    ("Sesgada a la derecha", rng.lognormal(3.2, 0.55, 4000)),
    ("Sesgada a la izquierda", 100 - rng.lognormal(3.2, 0.55, 4000)),
    ("Uniforme", rng.uniform(10, 90, 4000)),
    ("Bimodal", np.r_[rng.normal(30, 7, 2000), rng.normal(70, 7, 2000)]),
]
fig, axs = plt.subplots(1, 5, figsize=(15, 3.4))
for ax, (t, v) in zip(axs, formas):
    ax.hist(v, bins=35, color=AZUL, edgecolor="white", linewidth=0.5)
    ax.axvline(v.mean(), color=NARANJA, lw=2.2)
    ax.axvline(np.median(v), color=VERDE, lw=2.2, ls=(0, (4, 2)))
    ax.set_title(t, fontsize=14)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.grid(False)
fig.legend(handles=[plt.Line2D([], [], color=NARANJA, lw=2.2, label="media"),
                    plt.Line2D([], [], color=VERDE, lw=2.2, ls=(0, (4, 2)),
                               label="mediana")],
           loc="lower center", ncol=2, frameon=False, fontsize=13,
           bbox_to_anchor=(0.5, -0.1))
guardar(fig, "fig_formas.png")

# 15 · dispersión: misma media, distinta separación (datos inventados) -----
gA = rng.normal(10000, 1000, 3000)
gB = rng.normal(10000, 3000, 3000)
fig, axs = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True,
                        gridspec_kw={"hspace": 0.45})
for ax, g, t in [(axs[0], gA, "Grupo A: desviación estándar de 1.000"),
                 (axs[1], gB, "Grupo B: desviación estándar de 3.000")]:
    ax.hist(g, bins=np.arange(0, 20500, 500), color=AZUL,
            edgecolor="white", linewidth=0.6)
    ax.axvline(10000, color=NARANJA, lw=2.2)
    ax.text(10600, ax.get_ylim()[1] * 0.8, "media 10.000", fontsize=12,
            color=TINTA, fontweight="medium",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=NARANJA,
                      lw=1.5))
    ax.set_title(t, fontsize=14)
    ax.set_yticks([])
    ax.grid(False)
eje_dolares(axs[1])
axs[1].set_xlabel("Costo del año (USD), ejemplo inventado")
guardar(fig, "fig_dispersion.png")

# ------------------------------------------------------------------ CIFRAS --
pd.set_option("display.width", 120)
print("\n" + "=" * 70)
print("CIFRAS DE LAS LÁMINAS")
print("=" * 70)
print("\n-- df.head()\n", df.head())
print("\n-- df.shape", df.shape)
print("\n-- df.info()")
df.info()
print("\n-- df.dtypes\n", df.dtypes)
print("\n-- df.describe()\n", df.describe().round(2))
print("\n-- vacíos por columna\n", df.isna().sum())
print("\n-- duplicados", df.duplicated().sum())
print(df[df.duplicated(keep=False)])
for col in ["sex", "smoker", "region", "children"]:
    print(f"\n-- value_counts({col})\n", df[col].value_counts())
    print(df[col].value_counts(normalize=True).round(3))
print("\n-- edad 18 y 19:", (df.age == 18).sum(), (df.age == 19).sum(),
      "| promedio de clientes por cada otra edad:",
      round(df[df.age > 19].age.value_counts().mean(), 1))
print("\n-- smoker como 0/1: media =",
      round((df.smoker == "yes").astype(int).mean(), 3))
print("\n-- IMC: media", round(df.bmi.mean(), 2), "mediana",
      round(df.bmi.median(), 2), "| >50:", (df.bmi > 50).sum(),
      "| max", df.bmi.max())
print(df[df.bmi > 50])
print("\n-- IMC en categorías\n", vc_imc)
print("\n-- COSTO")
print("media", round(media, 2), "| mediana", round(mediana, 2),
      "| moda (no sirve, casi no se repite):", c.round(0).mode().tolist()[:3],
      "| valores distintos:", c.nunique())
print("min", round(c.min(), 2), "max", round(c.max(), 2), "rango",
      round(c.max() - c.min(), 2))
print("std", round(c.std(), 2), "| P10", round(p10, 2), "P25", round(p25, 2),
      "P75", round(p75, 2), "P90", round(p90, 2), "| IQR", round(iqr, 2))
print("bigote superior (P75 + 1,5·IQR)", round(bigote_sup, 2),
      "| atípicos por la regla:", (c > bigote_sup).sum(),
      "| de ellos fumadores:", ((c > bigote_sup) & (df.smoker == "yes")).sum())
print("costo total", round(c.sum(), 2), "| 1.338 × media",
      round(len(df) * media, 2))
print("parte del costo por décimo (%):", parte.round(1).tolist())
print("10 % más caro pone", round(parte.iloc[-1], 1), "% | 50 % más barato pone",
      round(parte.iloc[:5].sum(), 1), "%")
print("sesgo del costo", round(c.skew(), 2), "| sesgo IMC",
      round(df.bmi.skew(), 2), "| sesgo edad", round(df.age.skew(), 2))
print("\n-- cinco clientes", toy, "media", toy.mean(), "mediana",
      np.median(toy))
print("\n-- edad: media", round(df.age.mean(), 2), "mediana", df.age.median(),
      "| hijos: media", round(df.children.mean(), 2), "mediana",
      df.children.median(), "moda", df.children.mode()[0])
print("\n-- cliente más caro\n", df.loc[c.idxmax()])
print("\n-- más de 3 hijos:", (df.children > 3).sum(),
      "| fumadores mayores de 50:",
      ((df.smoker == "yes") & (df.age > 50)).sum(),
      "| fumadores en southeast:",
      ((df.smoker == "yes") & (df.region == "southeast")).sum())

print("\n-- concentración, como en el cuaderno (>= P90)")
caros = df[df["charges"] >= df["charges"].quantile(0.9)]
print("clientes caros", len(caros), "| parte del costo",
      round(caros["charges"].sum() / df["charges"].sum() * 100, 1), "%")

print("\n-- IMC para el ejercicio 4")
b = df["bmi"]
bq25, bq75 = b.quantile(0.25), b.quantile(0.75)
biqr = bq75 - bq25
print("media", round(b.mean(), 2), "mediana", round(b.median(), 2),
      "std", round(b.std(), 2), "P10", round(b.quantile(0.1), 2),
      "P90", round(b.quantile(0.9), 2), "P25", round(bq25, 2), "P75",
      round(bq75, 2), "IQR", round(biqr, 2))
print("límite superior", round(bq75 + 1.5 * biqr, 2), "| inferior",
      round(bq25 - 1.5 * biqr, 2), "| por encima:",
      (b > bq75 + 1.5 * biqr).sum(), "| por debajo:",
      (b < bq25 - 1.5 * biqr).sum())
print(df[b > bq75 + 1.5 * biqr].sort_values("bmi"))

print("\n-- edades de los primeros 20:", df.age.head(20).tolist())
print("\n-- salidas para las láminas de pandas")
print(df["charges"].head())
print(df[["age", "charges"]].head())
print(df.iloc[0])
print(df.iloc[0:3])
print(df.loc[0:3])
print("loc[543,'charges'] =", df.loc[543, "charges"])
print((df["smoker"] == "yes").head())
print(df.sort_values("charges", ascending=False).head())
print(df.head(10)[["age", "charges"]])
print("edad: P25", df.age.quantile(.25), "P75", df.age.quantile(.75),
      "min", df.age.min(), "max", df.age.max())

print("\n-- desviación estándar: ejemplo de 4 valores")
ga, gb = pd.Series([4, 5, 5, 6]), pd.Series([1, 3, 7, 9])
print("A media", ga.mean(), "std", round(ga.std(), 2), "| B media", gb.mean(), "std", round(gb.std(), 2))
dist = gb - gb.mean()
print("B distancias", dist.tolist(), "cuadrados", (dist ** 2).tolist(), "suma", (dist ** 2).sum(),
      "/(n-1)", round((dist ** 2).sum() / 3, 2), "raíz", round(((dist ** 2).sum() / 3) ** .5, 2))
for col in ["bmi", "charges"]:
    m, sd_ = df[col].mean(), df[col].std()
    for k in [1, 2]:
        lo, hi = m - k * sd_, m + k * sd_
        dentro = ((df[col] >= lo) & (df[col] <= hi)).mean()
        print(f"{col}: media ± {k} desv = [{lo:.2f}, {hi:.2f}] contiene {dentro:.1%}")

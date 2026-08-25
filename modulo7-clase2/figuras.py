"""
Machine Learning for Petroleum Engineers Using Python
Modulo 7 - Clase 2: genera todas las figuras de la presentacion.

La DEMO de la clase es el tasador de diamantes, de punta a punta: todas las
figuras y cifras salen de ese caso. Del RETO (Titanic) no se genera NINGUNA
figura ni cifra: la regla de la clase es que los retos no tienen soluciones
escritas en ningun archivo.

Lee datos/diamantes.csv del repositorio (local o por URL) y escribe los
fig_*.png de esta carpeta. Al final imprime TODAS las cifras que aparecen
en las laminas: ninguna se escribe a mano, y el cuaderno tiene que
reproducirlas.

Uso:  python3 figuras.py
"""

import os
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- estilo ----
RED, DARK = "#C82B40", "#6B1525"
BLUE, GREEN, ORANGE = "#2563EB", "#16A34A", "#EA580C"
AMBER = "#D97706"
INK, MUTED = "#2D2D2D", "#6B7280"
GRAY, LGRAY = "#9CA3AF", "#E5E7EB"


def _fuente():
    for d in (os.environ.get("FIRA_DIR"), "_fuentes"):
        if d and os.path.isdir(d):
            for f in glob.glob(os.path.join(d, "*.ttf")):
                try:
                    matplotlib.font_manager.fontManager.addfont(f)
                except Exception:
                    pass
    if "Fira Sans" in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        return "Fira Sans"
    print("  (aviso: Fira Sans no encontrada; se usa la sans por defecto)")
    return matplotlib.rcParams["font.sans-serif"][0]


plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": [_fuente()], "font.size": 11,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRAY, "axes.labelcolor": INK, "axes.titlecolor": DARK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": LGRAY, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
})


def guardar(fig, nombre):
    fig.savefig(nombre, facecolor="white")
    plt.close(fig)
    print(f"  {nombre}")


# ------------------------------------------------------------- el modelo ----
# Exactamente la misma receta que el cuaderno: si algo difiere, el deck miente.
SEMILLA = 0
ARBOLES = 100
TEST = 0.2
DIM_MAX = 20.0        # mm: por encima de esto, la "piedra" no es creible


def cargar():
    ruta = "../datos/diamantes.csv"
    if os.path.exists(ruta):
        return pd.read_csv(ruta)
    return pd.read_csv("https://raw.githubusercontent.com/cmosquerat/"
                       "slb-diplomado/main/datos/diamantes.csv")


def certificar(di):
    """La revision de calidad: marca las piedras fisicamente imposibles."""
    imposibles = (((di[["x", "y", "z"]] == 0).any(axis=1))
                  | (di.y > DIM_MAX) | (di.z > DIM_MAX))
    return imposibles


def entrenar(d):
    """El tasador: la misma receta del cuaderno, con la misma semilla."""
    X = pd.get_dummies(
        d[["carat", "cut", "color", "clarity", "x", "y", "z",
           "depth", "table"]],
        columns=["cut", "color", "clarity"])
    Xtr, Xte, ytr, yte = train_test_split(
        X, d.price, test_size=TEST, random_state=SEMILLA)
    rf = RandomForestRegressor(ARBOLES, n_jobs=-1, random_state=SEMILLA)
    rf.fit(Xtr, ytr)
    return rf, X, Xtr, Xte, ytr, yte


def tasar_con_banda(rf, fila):
    """La respuesta honesta: cada arbol opina, y la banda sale de ahi."""
    votos = np.array([a.predict(fila.values)[0] for a in rf.estimators_])
    return (float(np.median(votos)), float(np.percentile(votos, 10)),
            float(np.percentile(votos, 90)))


# =========================================================== FIGURAS ========
def fig_certificar(di, imposibles):
    """Antes de entrenar: las 23 piedras que no existen."""
    fig, ax = plt.subplots(figsize=(10.2, 4.3))
    muestra = di[~imposibles].sample(4000, random_state=SEMILLA)
    ax.scatter(muestra.carat, muestra.price, s=6, color=LGRAY, alpha=.6,
               label="piedras normales (muestra)")
    malas = di[imposibles]
    ax.scatter(malas.carat, malas.price, s=55, color=RED, zorder=5,
               marker="X", label=f"las {imposibles.sum()} imposibles")
    ax.annotate("piedras con 0 mm de ancho,\no de 3 a 6 cm — no existen",
                xy=(malas.carat.iloc[0], malas.price.iloc[0]),
                xytext=(2.6, 4000), fontsize=10.5, color=RED,
                fontweight="bold", linespacing=1.3,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.set_xlabel("quilates", fontsize=9.5)
    ax.set_ylabel("precio [USD]", fontsize=9.5)
    ax.legend(fontsize=9.5, loc="upper left")
    ax.set_title("Primero se certifica el dato: 23 filas que la física rechaza",
                 fontsize=12.5, fontweight="bold", loc="left", pad=8)
    guardar(fig, "fig_certificar.png")


def fig_ajuste(yte, pred, r2, mae):
    """Lo que el cliente entiende: tasado contra pagado."""
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.grid(True)
    idx = np.random.RandomState(SEMILLA).choice(len(yte), 3000, replace=False)
    ax.scatter(np.array(yte)[idx] / 1000, pred[idx] / 1000, s=6,
               color=BLUE, alpha=.35)
    lim = max(yte.max(), pred.max()) / 1000
    ax.plot([0, lim], [0, lim], color=INK, lw=1.8, ls="--")
    ax.text(lim * .72, lim * .64, "tasación perfecta", fontsize=9.5,
            color=INK, rotation=38)
    ax.set_xlabel("lo que de verdad se pagó  [miles de USD]", fontsize=9.5)
    ax.set_ylabel("lo que tasó el modelo  [miles de USD]", fontsize=9.5)
    ax.text(0.03, 0.95,
            f"error típico: USD {mae:.0f}\nsobre un precio mediano de "
            f"USD 2.401", transform=ax.transAxes, fontsize=11.5,
            fontweight="bold", color=DARK, va="top", linespacing=1.4)
    ax.set_title("El tasador, examinado con 10.784 piedras que nunca vio",
                 fontsize=12.5, fontweight="bold", loc="left", pad=8)
    guardar(fig, "fig_ajuste.png")


def fig_importancias(rf, X):
    """En que se fija el tasador -- y la lectura honesta."""
    imp = 100 * pd.Series(rf.feature_importances_, index=X.columns)
    imp = imp.sort_values(ascending=False).head(6)
    nombres = {"y": "ancho (y) [mm]", "carat": "quilates", "x": "largo (x) [mm]",
               "z": "profundidad (z) [mm]", "clarity_SI2": "claridad SI2",
               "clarity_I1": "claridad I1", "depth": "proporción",
               "table": "tabla"}
    imp.index = [nombres.get(i, i) for i in imp.index]
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    ax.grid(axis="x")
    ax.barh(imp.index[::-1], imp.values[::-1],
            color=[GRAY] * (len(imp) - 2) + [GREEN, GREEN], alpha=.9)
    for i, v in enumerate(imp.values[::-1]):
        ax.text(v + 0.8, i, f"{v:.0f} %", va="center", fontsize=10.5,
                fontweight="bold", color=DARK)
    ax.set_xlabel("cuánto usa el tasador cada dato  [%]", fontsize=9.5)
    ax.set_title("En qué se fija: el tamaño manda (por dos caminos)",
                 fontsize=12.5, fontweight="bold", loc="left", pad=8)
    fig.text(0.5, -0.04,
             "el ancho y los quilates miden casi lo mismo — el tamaño de la "
             "piedra — repartido entre dos columnas. No son dos causas: son "
             "una, contada dos veces.",
             ha="center", fontsize=10, color=DARK, style="italic")
    guardar(fig, "fig_importancias.png")


def fig_cero(tasacion_cero):
    """La demo negativa: el tasador sin contrato aprecia lo imposible."""
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.6),
                             gridspec_kw=dict(width_ratios=[1, 1.1]))
    ax = axes[0]
    ax.set_axis_off(); ax.grid(False)
    ax.text(0.5, 0.9, "LO QUE LE PEDIMOS", ha="center", fontsize=11,
            fontweight="bold", color=MUTED)
    for i, (k, v) in enumerate([("quilates", "0.0"), ("ancho", "0 mm"),
                                ("largo", "0 mm"), ("profundidad", "0 mm")]):
        ax.add_patch(FancyBboxPatch((0.14, 0.62 - i * 0.17), 0.72, 0.115,
                                    boxstyle="round,pad=0.01",
                                    fc="white", ec=GRAY, lw=1.2))
        ax.text(0.2, 0.677 - i * 0.17, k, fontsize=9.5, color=MUTED)
        ax.text(0.8, 0.677 - i * 0.17, v, fontsize=10, fontweight="bold",
                color=INK, ha="right")
    ax = axes[1]
    ax.set_axis_off(); ax.grid(False)
    ax.text(0.5, 0.9, "LO QUE RESPONDIÓ", ha="center", fontsize=11,
            fontweight="bold", color=MUTED)
    ax.text(0.5, 0.47, f"USD {tasacion_cero:,.0f}".replace(",", "."),
            ha="center", va="center", fontsize=44, fontweight="bold",
            color=RED)
    ax.text(0.5, 0.13, "por una piedra que no existe.\nSin dudar. Sin avisar.",
            ha="center", fontsize=11, color=DARK, linespacing=1.4,
            style="italic")
    guardar(fig, "fig_cero.png")


def fig_banda(rf, Xte, yte):
    """La respuesta honesta: no un numero, una banda -- la de los arboles."""
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.6), sharey=False)
    rs = np.random.RandomState(SEMILLA)
    elegidos = rs.choice(len(Xte), 3, replace=False)
    for ax, i in zip(axes, elegidos):
        fila = Xte.iloc[[i]]
        votos = np.array([a.predict(fila.values)[0] for a in rf.estimators_])
        med, lo, hi = (np.median(votos), np.percentile(votos, 10),
                       np.percentile(votos, 90))
        real = np.array(yte)[i]
        ax.hist(votos / 1000, bins=18, color=BLUE, alpha=.65)
        ax.axvline(med / 1000, color=DARK, lw=2.2)
        ax.axvline(real / 1000, color=GREEN, lw=2.2, ls="--")
        ax.axvspan(lo / 1000, hi / 1000, color=BLUE, alpha=.12)
        ax.set_xlabel("tasación [miles de USD]", fontsize=8.5)
        ax.set_title(f"P10–P90: {lo/1000:,.1f} a {hi/1000:,.1f} mil"
                     .replace(",", "."), fontsize=10, fontweight="bold",
                     loc="left")
    axes[0].set_ylabel("cuántos árboles opinaron eso", fontsize=8.5)
    fig.legend(handles=[
        plt.Line2D([], [], color=DARK, lw=2.2, label="la tasación (mediana)"),
        plt.Line2D([], [], color=GREEN, lw=2.2, ls="--",
                   label="lo que de verdad se pagó")],
        loc="upper right", fontsize=9, ncol=2, frameon=False)
    fig.suptitle("Los 100 árboles no opinan lo mismo — y esa discrepancia ES la banda",
                 fontsize=12.5, fontweight="bold", color=DARK, x=0.005,
                 ha="left")
    fig.tight_layout(w_pad=1.8)
    guardar(fig, "fig_banda.png")


# =============================================================== MAIN =======
def main():
    print("cargando diamantes ...")
    di = cargar()
    imposibles = certificar(di)
    d = di[~imposibles].copy()
    print(f"  {len(di)} piedras | imposibles: {imposibles.sum()} | "
          f"quedan {len(d)}")

    print("\nentrenando el tasador ...")
    import time
    t0 = time.time()
    rf, X, Xtr, Xte, ytr, yte = entrenar(d)
    seg = time.time() - t0
    pred = rf.predict(Xte)
    r2 = r2_score(yte, pred)
    mae = mean_absolute_error(yte, pred)
    print(f"  {ARBOLES} arboles en {seg:.0f} s | R2 {r2:.3f} | "
          f"MAE USD {mae:.0f}")

    # la tasacion del imposible: la MISMA fila que arma la app del
    # cuaderno (todo en cero, depth/table en la mediana, Ideal/G/SI1)
    fila_cero = pd.DataFrame(0, index=[0], columns=X.columns, dtype=float)
    fila_cero["depth"] = d.depth.median()
    fila_cero["table"] = d.table.median()
    for c in ["cut_Ideal", "color_G", "clarity_SI1"]:
        if c in fila_cero.columns:
            fila_cero[c] = 1.0
    tas_cero = float(rf.predict(fila_cero)[0])

    # una piedra de referencia para la lamina del contrato
    fila_ref = Xte.iloc[[0]]
    med, lo, hi = tasar_con_banda(rf, fila_ref)
    real_ref = float(np.array(yte)[0])

    print("\ngenerando figuras ...")
    fig_certificar(di, imposibles)
    fig_ajuste(yte, pred, r2, mae)
    fig_importancias(rf, X)
    fig_cero(tas_cero)
    fig_banda(rf, Xte, yte)

    print("\n" + "=" * 74)
    print("NUMEROS PARA LAS LAMINAS  (no escribir ninguno a mano)")
    print("=" * 74)
    print(f"\nEL DATO:")
    print(f"  piedras: {len(di)} | imposibles rechazadas: {imposibles.sum()} "
          f"(dimension 0 mm, o mas de {DIM_MAX:.0f} mm)")
    print(f"  quedan para trabajar: {len(d)}")
    print(f"\nEL TASADOR (RandomForest, {ARBOLES} arboles, semilla {SEMILLA}):")
    print(f"  entrena en ~{seg:.0f} segundos")
    print(f"  examen con {len(yte)} piedras que nunca vio "
          f"({100*TEST:.0f} % reservado)")
    print(f"  R2 {r2:.3f} | error tipico USD {mae:.0f} | "
          f"precio mediano USD {d.price.median():.0f}")
    imp = pd.Series(rf.feature_importances_, index=X.columns)
    top = imp.sort_values(ascending=False).head(3)
    print("  en que se fija: " + " | ".join(
        f"{k} {100*v:.0f} %" for k, v in top.items()))
    print(f"\nLA DEMO NEGATIVA:")
    print(f"  tasacion de una piedra de 0 quilates y 0 mm: "
          f"USD {tas_cero:,.0f}".replace(",", "."))
    print(f"\nLA RESPUESTA CON BANDA (piedra de referencia del examen):")
    print(f"  mediana USD {med:,.0f} | banda P10-P90 USD {lo:,.0f} a "
          f"{hi:,.0f} | real USD {real_ref:,.0f}".replace(",", "."))
    print(f"\nDEL RETO (Titanic) NO SE IMPRIME NADA: sin soluciones escritas.")


if __name__ == "__main__":
    main()

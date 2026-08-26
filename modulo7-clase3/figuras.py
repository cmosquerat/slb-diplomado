"""
Machine Learning for Petroleum Engineers Using Python
Modulo 7 - Clase 3: genera todas las figuras de la presentacion.

Acto 1: el Titanic de punta a punta (la demo; reproduce exactamente el
cuaderno Modulo7_Clase3_Titanic_De_Punta_A_Punta). Acto 2: las figuras de
apoyo (moons y MNIST); lo que la clase construye en vivo con el agente NO
se precalcula aca.

Las funciones del tasador de diamantes (la Clase 2) quedan abajo como
respaldo y no se ejecutan.

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


# (main de diamantes desactivado: las funciones quedan como respaldo)

# ================================================= ACTO 1 · TITANIC =========
# La demo de la clase paso a ser el Titanic de punta a punta. Estas figuras
# reproducen EXACTAMENTE el cuaderno Modulo7_Clase3_Titanic_De_Punta_A_Punta.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

NOMBRES_ES = {"pclass": "clase del boleto", "age": "edad",
              "sibsp": "hermanos/cónyuge", "parch": "padres/hijos",
              "fare": "tarifa pagada", "sex_female": "es mujer",
              "sex_male": "es hombre", "embarked_C": "embarcó en Cherburgo",
              "embarked_Q": "embarcó en Queenstown",
              "embarked_S": "embarcó en Southampton"}


def titanic_pipeline():
    """El mismo flujo del cuaderno, con las mismas semillas."""
    ruta = "../datos/pasajeros_titanic.csv"
    p = (pd.read_csv(ruta) if os.path.exists(ruta) else
         pd.read_csv("https://raw.githubusercontent.com/cmosquerat/"
                     "slb-diplomado/main/datos/pasajeros_titanic.csv"))
    t = p[["survived", "pclass", "sex", "age",
           "sibsp", "parch", "fare", "embarked"]].copy()
    t["age"] = t.age.fillna(t.age.median())
    t["embarked"] = t.embarked.fillna(t.embarked.mode()[0])
    X = pd.get_dummies(t.drop(columns="survived"), columns=["sex", "embarked"])
    Xe, Xx, ye, yx = train_test_split(X, t.survived, test_size=0.2,
                                      random_state=SEMILLA,
                                      stratify=t.survived)
    cands = {
        "Regresión\nlogística": (
            Pipeline([("esc", StandardScaler()),
                      ("m", LogisticRegression(max_iter=3000,
                                               random_state=SEMILLA))]),
            {"m__C": [0.1, 1, 10]}),
        "Bosque\naleatorio": (
            RandomForestClassifier(300, random_state=SEMILLA, n_jobs=-1),
            {"max_depth": [4, 7, None], "min_samples_leaf": [1, 4]}),
        "Gradient\nboosting": (
            HistGradientBoostingClassifier(random_state=SEMILLA),
            {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31]}),
    }
    cv = {}
    for nom, (m, g) in cands.items():
        cv[nom] = GridSearchCV(m, g, cv=5, scoring="accuracy",
                               n_jobs=-1).fit(Xe, ye)
    ganador = max(cv, key=lambda k: cv[k].best_score_)
    return p, t, X, Xe, Xx, ye, yx, cv, cv[ganador].best_estimator_, ganador


def fig_t_eda(p):
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.5))
    ax = axes[0]
    ax.grid(axis="y")
    ps = p.groupby("sex").survived.mean() * 100
    ax.bar(["hombres", "mujeres"], [ps["male"], ps["female"]],
           color=[GRAY, RED], width=.55)
    for i, v in enumerate([ps["male"], ps["female"]]):
        ax.text(i, v + 2, f"{v:.0f} %", ha="center", fontsize=12,
                fontweight="bold", color=DARK)
    ax.set_ylim(0, 90)
    ax.set_ylabel("supervivencia [%]", fontsize=9)
    ax.set_title("Por sexo", fontsize=11, fontweight="bold", loc="left")

    ax = axes[1]
    ax.grid(axis="y")
    pc = p.groupby("pclass").survived.mean() * 100
    ax.bar(["1ª", "2ª", "3ª"], pc.values, color=BLUE, width=.55)
    for i, v in enumerate(pc.values):
        ax.text(i, v + 2, f"{v:.0f} %", ha="center", fontsize=12,
                fontweight="bold", color=DARK)
    ax.set_ylim(0, 90)
    ax.set_title("Por clase del boleto", fontsize=11, fontweight="bold",
                 loc="left")

    ax = axes[2]
    ax.grid(axis="y")
    ax.hist([p[p.survived == 1].age.dropna(), p[p.survived == 0].age.dropna()],
            bins=18, label=["sobrevivió", "no"], color=[GREEN, GRAY])
    ax.set_xlabel("edad [años]", fontsize=9)
    ax.legend(fontsize=8.5)
    ax.set_title("Edades", fontsize=11, fontweight="bold", loc="left")
    fig.suptitle("El EDA en tres hechos: sexo, clase y edad",
                 fontsize=12.5, fontweight="bold", color=DARK, x=0.005,
                 ha="left")
    fig.tight_layout(w_pad=1.8)
    guardar(fig, "fig_t_eda.png")


def fig_t_grid(cv):
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    ax.grid(axis="y")
    noms = list(cv)
    vals = [100 * cv[n].best_score_ for n in noms]
    cols = [GRAY, GRAY, GREEN]
    orden = np.argsort(vals)
    cols = [GREEN if i == orden[-1] else GRAY for i in range(3)]
    ax.bar(noms, vals, 0.55, color=cols, alpha=.9)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.35, f"{v:.1f} %", ha="center", fontsize=12.5,
                fontweight="bold", color=DARK)
    ax.set_ylim(70, 88)
    ax.set_ylabel("acierto en validación cruzada [%]", fontsize=9.5)
    ax.set_title("Tres modelos, sus mejores recetas — decide la validación cruzada",
                 fontsize=12, fontweight="bold", loc="left", pad=8)
    guardar(fig, "fig_t_grid.png")


def fig_t_examen(yx, pred, proba):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.8),
                             gridspec_kw=dict(width_ratios=[0.8, 1.2]))
    ax = axes[0]
    ax.grid(False)
    mc = confusion_matrix(yx, pred)
    ax.imshow(mc, cmap="Reds", alpha=.75)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, mc[i, j], ha="center", va="center",
                    fontsize=17, fontweight="bold", color=INK)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["predijo: no", "predijo: sí"], fontsize=9)
    ax.set_yticklabels(["real: no", "real: sí"], fontsize=9)
    ax.set_title("La matriz de confusión", fontsize=11.5,
                 fontweight="bold", loc="left")

    ax = axes[1]
    ax.grid(axis="y")
    ax.hist([proba[np.array(yx) == 0], proba[np.array(yx) == 1]], bins=15,
            label=["no sobrevivió", "sobrevivió"], color=[GRAY, GREEN])
    ax.set_xlabel("probabilidad que dio el modelo", fontsize=9.5)
    ax.set_ylabel("personas", fontsize=9.5)
    ax.legend(fontsize=9)
    ax.set_title("¿Las probabilidades separan? — coloreado por la verdad",
                 fontsize=11.5, fontweight="bold", loc="left")
    fig.tight_layout(w_pad=2.2)
    guardar(fig, "fig_t_examen.png")


def fig_t_shap(modelo, Xx):
    import shap
    ex = shap.TreeExplainer(modelo)
    sv = ex.shap_values(Xx)
    Xes = Xx.rename(columns=NOMBRES_ES)
    shap.summary_plot(sv, Xes, show=False, plot_size=(9.6, 4.2))
    fig = plt.gcf()
    fig.axes[0].set_xlabel("empuje  (derecha: hacia sobrevivir)", fontsize=9.5)
    fig.suptitle("Qué mueve la predicción, y hacia dónde — cada punto es un pasajero",
                 fontsize=12, fontweight="bold", color=DARK, x=0.005, ha="left")
    fig.tight_layout()
    guardar(fig, "fig_t_shap.png")
    return ex


def fig_t_persona(modelo, ex, Xx, yx, columnas, i=3):
    fila = Xx.iloc[[i]]
    p = modelo.predict_proba(fila)[0, 1]
    contrib = pd.Series(ex.shap_values(fila)[0], index=columnas)
    top = contrib.sort_values(key=abs, ascending=False).head(4)
    fig, ax = plt.subplots(figsize=(8.8, 3.2))
    ax.grid(axis="x")
    cols = [GREEN if v > 0 else RED for v in top.values[::-1]]
    ax.barh([NOMBRES_ES.get(k, k) for k in top.index[::-1]],
            top.values[::-1], color=cols, alpha=.9)
    ax.axvline(0, color=INK, lw=1.4)
    ax.set_xlabel("empuje  (derecha: hacia sobrevivir)", fontsize=9.5)
    real = "sobrevivió" if yx.iloc[i] else "no sobrevivió"
    ax.set_title(f"Un pasajero del examen: probabilidad {p:.2f} — real: {real}",
                 fontsize=12, fontweight="bold", loc="left", pad=8)
    guardar(fig, "fig_t_persona.png")
    return p, real


# =============================================== ACTO 2 · LA RED QUE VE =====
def fig_moons():
    """La no linealidad, vista: la recta que no puede y el MLP que si.
    Es la figura del INSTRUCTOR para leer el resultado juntos despues de
    que la clase lo construya con el agente."""
    from sklearn.datasets import make_moons
    from sklearn.neural_network import MLPClassifier
    puntos, et = make_moons(n_samples=250, noise=0.25, random_state=SEMILLA)
    recta = LogisticRegression().fit(puntos, et)
    red = MLPClassifier((16, 16), max_iter=3000,
                        random_state=SEMILLA).fit(puntos, et)

    xx, yy = np.meshgrid(np.linspace(-1.8, 2.8, 300),
                         np.linspace(-1.3, 1.8, 300))
    malla = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))
    for ax, m, nom, acc_col in [
            (axes[0], recta, "La RECTA (regresión logística)", RED),
            (axes[1], red, "La RED (dos capas de 16 neuronas)", GREEN)]:
        z = m.predict(malla).reshape(xx.shape)
        ax.contourf(xx, yy, z, alpha=.18, levels=1,
                    colors=["#2563EB", "#C82B40"])
        ax.contour(xx, yy, z, levels=1, colors=[INK], linewidths=1.6)
        ax.scatter(puntos[et == 0, 0], puntos[et == 0, 1], c=BLUE, s=16)
        ax.scatter(puntos[et == 1, 0], puntos[et == 1, 1], c=RED, s=16)
        acc = 100 * m.score(puntos, et)
        ax.set_title(f"{nom} · acierta {acc:.0f} %", fontsize=11.5,
                     fontweight="bold", loc="left", color=acc_col)
        ax.grid(False)
    fig.suptitle("La no linealidad, vista: la frontera que cada modelo PUEDE trazar",
                 fontsize=12.5, fontweight="bold", color=DARK, x=0.005,
                 ha="left")
    fig.tight_layout(w_pad=2.0)
    guardar(fig, "fig_moons.png")
    return 100 * recta.score(puntos, et), 100 * red.score(puntos, et)


def fig_mnist():
    """El dato de la Parte 3, presentado. El .npz es exactamente el que
    descarga keras.datasets.mnist."""
    ruta = ("/private/tmp/claude-501/-Users-cmosquerat-Documents-GitHub-"
            "slb-diplomado/17ff0dd6-d936-43c0-bd10-f0fc4e571d53/"
            "scratchpad/mnist.npz")
    if not os.path.exists(ruta):
        import urllib.request
        ruta = "mnist.npz"
        if not os.path.exists(ruta):
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/tensorflow/"
                "tf-keras-datasets/mnist.npz", ruta)
    dat = np.load(ruta)
    x, yv = dat["x_train"], dat["y_train"]
    fig, axes = plt.subplots(2, 8, figsize=(11.4, 3.2))
    for ax, i in zip(axes.ravel(), range(16)):
        ax.imshow(x[i], cmap="gray_r")
        ax.set_title(f"«{yv[i]}»", fontsize=10, fontweight="bold",
                     color=DARK)
        ax.axis("off")
    fig.suptitle("MNIST: 70.000 dígitos escritos a mano — acá no hay tabla: hay 28×28 píxeles",
                 fontsize=12.5, fontweight="bold", color=DARK, x=0.005,
                 ha="left")
    fig.tight_layout()
    guardar(fig, "fig_mnist.png")
    return x.shape


# ====================================================== MAIN v2 =============
def main2():
    print("ACTO 1 · Titanic (reproduce el cuaderno de la demo) ...")
    p, t, X, Xe, Xx, ye, yx, cv, modelo, ganador = titanic_pipeline()
    pred = modelo.predict(Xx)
    proba = modelo.predict_proba(Xx)[:, 1]
    acc = accuracy_score(yx, pred)

    # la fuga, medida igual que en el cuaderno
    Xtr = pd.get_dummies(p[["alive"]])
    a, b, c, d2 = train_test_split(Xtr, p.survived, test_size=0.2,
                                   random_state=SEMILLA)
    fuga = accuracy_score(d2, RandomForestClassifier(
        50, random_state=SEMILLA).fit(a, c).predict(b))

    fig_t_eda(p)
    fig_t_grid(cv)
    fig_t_examen(yx, pred, proba)
    ex = fig_t_shap(modelo, Xx)
    p_persona, real_persona = fig_t_persona(modelo, ex, Xx, yx,
                                            list(X.columns))

    print("\nACTO 2 · figuras de apoyo ...")
    acc_recta, acc_red = fig_moons()
    forma = fig_mnist()

    alta = proba > 0.8
    print("\n" + "=" * 74)
    print("NUMEROS PARA LAS LAMINAS  (no escribir ninguno a mano)")
    print("=" * 74)
    print(f"\nEL DATO: {len(p)} pasajeros | supervivencia global "
          f"{100*p.survived.mean():.1f} %")
    print(f"  faltantes: age {int(p.age.isna().sum())} | deck "
          f"{int(p.deck.isna().sum())} | embarked "
          f"{int(p.embarked.isna().sum())}")
    print(f"  supervivencia por sexo: hombres "
          f"{100*p[p.sex=='male'].survived.mean():.0f} % | mujeres "
          f"{100*p[p.sex=='female'].survived.mean():.0f} %")
    print(f"\nLA FUGA: accuracy del 'modelo perfecto' con alive: {fuga:.3f}")
    print(f"\nGRIDSEARCH (CV=5):")
    for n in cv:
        print(f"  {n.replace(chr(10),' '):22s} {cv[n].best_score_:.3f}  "
              f"{cv[n].best_params_}")
    print(f"  GANADOR: {ganador.replace(chr(10),' ')}")
    print(f"\nEXAMEN FINAL: {acc:.3f} ({(np.array(yx)==pred).sum()} de "
          f"{len(yx)})")
    print(f"  a {alta.sum()} personas les dio >80 %: sobrevivieron "
          f"{int(yx[alta].sum())} ({100*yx[alta].mean():.0f} %)")
    print(f"\nLA PERSONA DE LA LAMINA: probabilidad {p_persona:.2f} | "
          f"real: {real_persona}")
    print(f"\nMOONS: recta acierta {acc_recta:.0f} % | red {acc_red:.0f} %")
    print(f"MNIST: {forma[0]} imagenes de {forma[1]}x{forma[2]} pixeles")
    print("\nDEL ACTO 2 EN VIVO (red keras, curvas, YOLO) no se imprime")
    print("nada: se construye en clase dirigiendo al agente.")


if __name__ == "__main__":
    main2()

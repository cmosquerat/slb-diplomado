"""
Machine Learning for Petroleum Engineers Using Python
Modulo 5 - Clase 5: genera todas las figuras de la presentacion.

Lee  ../datos/flota_turbomaquinas_nasa.csv  y escribe los fig_*.png de esta
carpeta. Al final imprime TODAS las cifras que aparecen en las laminas:
ninguna se escribe a mano, y el cuaderno de la clase tiene que reproducirlas.

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
from matplotlib.patches import Rectangle, FancyArrow, Circle
from matplotlib.lines import Line2D
from sklearn.ensemble import (HistGradientBoostingRegressor,
                              GradientBoostingRegressor)
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, GroupKFold
import shap

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
                matplotlib.font_manager.fontManager.addfont(f)
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

AQUI = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(AQUI, "..", "datos", "flota_turbomaquinas_nasa.csv")


def guardar(fig, nombre):
    fig.savefig(os.path.join(AQUI, nombre))
    plt.close(fig)
    print(f"  {nombre}")


# ------------------------------------------------------- datos y features ----
SENSORES = ["T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc",
            "epr", "Ps30", "phi", "NRf", "NRc", "BPR", "farB", "htBleed",
            "Nf_dmd", "PCNfR_dmd", "W31", "W32"]

# traduccion corta para ejes y laminas (tag industrial -> que mide)
NOMBRE_ES = {
    "T24": "temp. salida compresor de baja",
    "T30": "temp. salida compresor de alta",
    "T50": "temp. salida de la turbina",
    "P15": "presión del anular (bypass)",
    "P30": "presión salida del compresor",
    "Nf": "velocidad del eje del fan",
    "Nc": "velocidad del eje del núcleo",
    "Ps30": "presión estática del compresor",
    "phi": "combustible por presión",
    "NRf": "velocidad corregida del fan",
    "NRc": "velocidad corregida del núcleo",
    "BPR": "relación de derivación",
    "farB": "relación combustible-aire",
    "htBleed": "entalpía de sangrado",
    "W31": "refrigerante turbina alta",
    "W32": "refrigerante turbina baja",
}

VENTANA, MITAD = 20, 10          # ciclos de ventana y desfase de pendiente
N_TEST, SEMILLA = 25, 0          # unidades apartadas y semilla del sorteo
UMBRAL, MOVIL = 20, 15           # aviso cuando P10 < UMBRAL; el equipo
                                 # llega MOVIL ciclos despues del aviso
COSTO_PLAN, COSTO_EMERG = 0.3, 1.5   # MUSD, supuestos a la vista

# la malla del afinado: la celda (0.1, None) es el default de sklearn
MALLA = {"learning_rate": [0.03, 0.1, 0.3],
         "max_depth": [2, 3, None]}


def cargar():
    df = pd.read_csv(CSV)
    vida = df.groupby("unidad")["ciclo"].max().rename("vida")
    df = df.join(vida, on="unidad")
    df["rul"] = df["vida"] - df["ciclo"]
    return df, vida


def sensores_vivos(df):
    """Los sensores que si se mueven. Los 6 planos se descubren en el EDA."""
    return [c for c in SENSORES if df[c].std() > 1e-6]


def hacer_features(df, vivos):
    """Valor actual + media y pendiente de ventana de 20 ciclos, por unidad.
    La misma receta de la Clase 2: los modelos no comen señales, comen tablas."""
    g = df.groupby("unidad")
    feats = {}
    for c in vivos:
        m = g[c].transform(
            lambda s: s.rolling(VENTANA, min_periods=5).mean())
        feats[c] = df[c]
        feats[c + "_m"] = m
        feats[c + "_d"] = m - m.groupby(df["unidad"]).shift(MITAD)
    feats["edad"] = df["ciclo"]
    return pd.DataFrame(feats).fillna(0)


def particion(df, vida):
    rng = np.random.RandomState(SEMILLA)
    test_u = np.sort(rng.choice(vida.index.to_numpy(), N_TEST,
                                replace=False))
    tr = ~df["unidad"].isin(test_u)
    te = df["unidad"].isin(test_u)
    return tr, te, test_u


def entrenar(df, vida):
    """Todo lo que las laminas necesitan, en un solo lugar.

    El afinado se hace con GridSearchCV + GroupKFold POR UNIDAD y SOLO con
    las 75 unidades de entrenamiento: el test no vota. Es la misma trampa
    de siempre (C2, C3): si la unidad se parte entre folds, la malla elige
    con informacion contaminada."""
    vivos = sensores_vivos(df)
    X = hacer_features(df, vivos)
    tr, te, test_u = particion(df, vida)
    y = df["rul"]

    vida_med = vida[~vida.index.isin(test_u)].median()
    tonto = np.clip(vida_med - df["ciclo"], 0, None)

    base = HistGradientBoostingRegressor(max_iter=300,
                                         random_state=SEMILLA)
    folds = list(GroupKFold(n_splits=5).split(
        X[tr], y[tr], groups=df.loc[tr, "unidad"]))
    busqueda = GridSearchCV(base, MALLA, cv=folds,
                            scoring="neg_mean_absolute_error")
    busqueda.fit(X[tr], y[tr])
    modelo = busqueda.best_estimator_
    pred = pd.Series(np.clip(modelo.predict(X), 0, None), index=df.index)

    p10 = HistGradientBoostingRegressor(loss="quantile", quantile=0.10,
                                        max_iter=300, random_state=SEMILLA,
                                        **busqueda.best_params_)
    p10.fit(X[tr], y[tr])
    pred10 = pd.Series(np.clip(p10.predict(X), 0, None), index=df.index)

    # ablacion: el mismo modelo sin saber la edad de la unidad
    X_se = X.drop(columns=["edad"])
    sin_edad = HistGradientBoostingRegressor(max_iter=300,
                                             random_state=SEMILLA,
                                             **busqueda.best_params_)
    sin_edad.fit(X_se[tr], y[tr])
    pred_se = pd.Series(np.clip(sin_edad.predict(X_se), 0, None),
                        index=df.index)

    return dict(X=X, vivos=vivos, tr=tr, te=te, test_u=test_u,
                vida_med=vida_med, tonto=tonto, modelo=modelo,
                pred=pred, p10=p10, pred10=pred10,
                busqueda=busqueda, pred_se=pred_se)


def maes(df, s):
    """MAE global y por zona de vida restante, tonto vs modelo (solo test)."""
    y, te = df["rul"], s["te"]
    filas = {}
    zonas = [("global", y >= 0), ("z100", y >= 100),
             ("z50", (y >= 50) & (y < 100)), ("z0", y < 50)]
    for nombre, mascara in zonas:
        m = te & mascara
        filas[nombre] = (mean_absolute_error(y[m], s["tonto"][m]),
                         mean_absolute_error(y[m], s["pred"][m]))
    return filas


# ------------------------------------------------- politicas y su costo ----
def simular_politicas(df, vida, s):
    """Recorre las unidades de test con cuatro politicas de mantenimiento.
    Devuelve por politica: emergencias, vida desperdiciada mediana y costo
    por cada 1.000 ciclos de operacion."""
    test_u = s["test_u"]
    vidas_tr = vida[~vida.index.isin(test_u)]
    calendario = {"mediana": vidas_tr.median(),
                  "seguro": vidas_tr.quantile(0.10)}

    resultados = {}

    def anotar(nombre, cortes):
        emerg, desperdicio, ciclos, costo = 0, [], 0, 0.0
        for u in test_u:
            v = vida.loc[u]
            c = cortes[u]
            if c >= v:                    # no llego a tiempo: revienta
                emerg += 1
                ciclos += v
                costo += COSTO_EMERG
                desperdicio.append(0)
            else:                         # cambio programado
                ciclos += c
                costo += COSTO_PLAN
                desperdicio.append(v - c)
        resultados[nombre] = dict(
            emergencias=emerg,
            desperdicio=float(np.median(desperdicio)),
            costo_kciclo=1000.0 * costo / ciclos)

    # 1) correr a la falla: nunca se interviene
    anotar("falla", {u: np.inf for u in test_u})
    # 2) calendario a la mediana de la flota
    anotar("mediana", {u: calendario["mediana"] for u in test_u})
    # 3) calendario conservador (P10 de las vidas)
    anotar("seguro", {u: calendario["seguro"] for u in test_u})
    # 4) modelo: aviso cuando el P10 del RUL baja de UMBRAL ciclos;
    #    el cambio ocurre MOVIL ciclos despues (la movilizacion tarda)
    cortes = {}
    for u in test_u:
        m = df["unidad"] == u
        aviso = df.loc[m & (s["pred10"] < UMBRAL), "ciclo"]
        cortes[u] = aviso.iloc[0] + MOVIL if len(aviso) else np.inf
    anotar("modelo", cortes)

    return resultados, calendario


# ------------------------------------------------------------- figuras ----
def fig_esp():
    """Esquema: la ESP en el pozo, de que se compone y que la mata."""
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.axis("off"); ax.grid(False)

    # el pozo
    ax.add_patch(Rectangle((1.1, 0.4), 1.6, 8.4, fc="white", ec=INK, lw=1.6))
    ax.add_patch(Rectangle((0.4, 8.4), 3.0, 0.9, fc=LGRAY, ec=INK, lw=1.2))
    ax.text(1.9, 8.85, "superficie", ha="center", va="center",
            fontsize=9, color=INK)
    # componentes de la ESP, de abajo hacia arriba
    piezas = [("motor eléctrico", 1.1, ORANGE),
              ("protector (sello)", 0.9, GRAY),
              ("succión (intake)", 0.8, BLUE),
              ("etapas de la bomba\n(impulsores en serie)", 3.3, RED)]
    y0 = 0.8
    for nombre, alto, color in piezas:
        ax.add_patch(Rectangle((1.35, y0), 1.1, alto, fc=color, ec=INK,
                               lw=1.0, alpha=0.85))
        ax.annotate(nombre, xy=(2.55, y0 + alto / 2),
                    xytext=(3.6, y0 + alto / 2), fontsize=9.5, color=INK,
                    va="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9))
        y0 += alto + 0.15
    ax.annotate("cable de potencia\n(baja por fuera de la tubería)",
                xy=(1.3, 7.6), xytext=(3.6, 8.1), fontsize=9.5,
                color=INK, va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9))
    ax.add_patch(FancyArrow(1.9, 9.35, 0, 0.3, width=0.10, fc=GREEN,
                            ec=GREEN))
    ax.text(2.2, 9.5, "fluido a superficie", fontsize=9, color=GREEN)

    # que la degrada
    ax.text(6.6, 9.3, "Lo que la va matando", fontsize=12, color=DARK,
            weight="bold")
    causas = [
        ("desgaste y abrasión (arena)", "los impulsores pierden filo"),
        ("incrustación (escala)", "se angosta el paso del fluido"),
        ("gas libre en la succión", "la bomba cavita y vibra"),
        ("calor en el motor", "el aislamiento envejece"),
    ]
    y = 8.4
    for causa, efecto in causas:
        ax.add_patch(Circle((6.35, y + 0.13), 0.09, fc=RED, ec=RED))
        ax.text(6.6, y, causa, fontsize=10.5, color=INK, weight="bold")
        ax.text(6.6, y - 0.62, efecto, fontsize=9.5, color=MUTED)
        y -= 1.55
    ax.text(6.6, 1.15, "Nada de esto se ve desde superficie.\n"
            "Lo que sí se ve: presiones, temperaturas\n"
            "y velocidades — la huella indirecta.",
            fontsize=10, color=DARK, style="italic")
    return fig


def fig_espagueti(df, vida):
    """Cien vidas del sensor T50: toda la flota, cada linea una unidad."""
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    for u, g in df.groupby("unidad"):
        ax.plot(g["ciclo"], g["T50"], lw=0.7, color=GRAY, alpha=0.45)
    for u, color in [(vida.idxmin(), RED), (vida.idxmax(), BLUE)]:
        g = df[df["unidad"] == u]
        ax.plot(g["ciclo"], g["T50"], lw=2.2, color=color,
                label=f"unidad {u}: vivió {vida.loc[u]} ciclos")
    ax.set_xlabel("ciclo de operación")
    ax.set_ylabel("T50 · temp. salida de turbina (°R)")
    ax.set_title("Las 100 unidades de la flota, desde sanas hasta la falla")
    ax.legend(loc="upper left", fontsize=10)
    return fig


def fig_vidas(vida):
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    ax.hist(vida, bins=24, color=RED, edgecolor="white", alpha=0.85)
    med = vida.median()
    ax.axvline(med, color=INK, lw=1.8, ls="--")
    ax.text(med + 4, ax.get_ylim()[1] * 0.92, f"mediana: {med:.0f}",
            color=INK, fontsize=10.5)
    ax.annotate(f"la más corta: {vida.min()}",
                xy=(vida.min(), 0.4), xytext=(vida.min() - 4, 6.5),
                ha="right", fontsize=10.5, color=DARK,
                arrowprops=dict(arrowstyle="->", color=DARK))
    ax.annotate(f"la más larga: {vida.max()}",
                xy=(vida.max(), 0.4), xytext=(vida.max() - 30, 6.5),
                fontsize=10.5, color=DARK,
                arrowprops=dict(arrowstyle="->", color=DARK))
    ax.set_xlabel("vida total (ciclos)")
    ax.set_ylabel("unidades")
    ax.set_title("Misma máquina, mismo plano de fábrica: vidas de "
                 f"{vida.min()} a {vida.max()} ciclos")
    return fig


def fig_sensores_muertos(df):
    """Cuanto se mueve cada sensor a lo largo de la vida (desv. estandar)."""
    sd = df[SENSORES].std().sort_values()
    colores = [RED if v < 1e-6 else GRAY for v in sd]
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.barh(range(len(sd)), np.where(sd < 1e-6, 0.04, sd), color=colores,
            edgecolor="white", log=True)
    ax.set_yticks(range(len(sd)))
    ax.set_yticklabels(sd.index, fontsize=9)
    ax.set_xlabel("desviación estándar (escala log)")
    ax.set_title("Seis sensores no se mueven nunca: están muertos "
                 "(o clavados en consigna)")
    muertos = [c for c in SENSORES if df[c].std() < 1e-6]
    ax.text(0.985, 0.06, "los rojos: ni un solo cambio\nen 20.631 filas",
            transform=ax.transAxes, ha="right", fontsize=10.5, color=RED)
    return fig, muertos


def fig_degradacion(df, vida):
    """La huella: el mismo sensor alineado al momento de la muerte."""
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.9), sharex=True)
    rng = np.random.RandomState(3)
    unidades = rng.choice(vida.index, 12, replace=False)
    for ax, c in zip(axs, ["T50", "Ps30"]):
        for u in unidades:
            g = df[df["unidad"] == u]
            ax.plot(g["ciclo"] - vida.loc[u], g[c], lw=0.9, alpha=0.7,
                    color=RED if c == "T50" else BLUE)
        ax.axvline(0, color=INK, lw=1.4, ls="--")
        ax.text(0.97, 0.05, "falla", transform=ax.transAxes, ha="right",
                fontsize=10, color=INK)
        ax.set_xlim(-220, 8)
        ax.set_xlabel("ciclos antes de la falla")
        ax.set_title(f"{c} · {NOMBRE_ES[c]}", fontsize=11)
    axs[0].set_ylabel("valor del sensor")
    fig.suptitle("Doce unidades alineadas por su muerte: "
                 "la degradación tiene una huella común", y=1.02,
                 fontsize=12.5, color=DARK)
    return fig


def fig_reloj(df, vida):
    """Definir el RUL: el reloj que corre hacia atras."""
    u = 1
    g = df[df["unidad"] == u]
    v = vida.loc[u]
    fig, axs = plt.subplots(2, 1, figsize=(9.0, 4.4), sharex=True,
                            gridspec_kw={"height_ratios": [1.15, 1]})
    axs[0].plot(g["ciclo"], g["T50"], color=RED, lw=1.4)
    axs[0].set_ylabel("T50 (°R)")
    axs[0].set_title(f"Unidad {u}: lo que el sensor ve "
                     f"(vivió {v} ciclos)")
    axs[1].plot(g["ciclo"], g["rul"], color=INK, lw=1.8)
    axs[1].set_ylabel("RUL (ciclos)")
    axs[1].set_xlabel("ciclo de operación")
    axs[1].set_title("Lo que queremos predecir: cuánta vida le queda "
                     "(el reloj corre hacia atrás)")
    for c0, txt in [(60, f"en el ciclo 60\nquedan {v-60}"),
                    (v - 30, "quedan 30")]:
        axs[1].annotate(txt, xy=(c0, v - c0), xytext=(c0 + 12, v - c0 + 42),
                        fontsize=9.5, color=DARK,
                        arrowprops=dict(arrowstyle="->", color=DARK))
    return fig


def fig_rival(df, vida, s):
    """El calendario, aplicado a tres unidades de vidas distintas."""
    test_u = s["test_u"]
    vidas_te = vida.loc[test_u].sort_values()
    elegidas = [vidas_te.index[0], vidas_te.index[len(vidas_te) // 2],
                vidas_te.index[-1]]
    fig, axs = plt.subplots(1, 3, figsize=(9.8, 3.4), sharey=True)
    for ax, u in zip(axs, elegidas):
        g = df[df["unidad"] == u]
        ax.plot(g["ciclo"], g["rul"], color=INK, lw=1.8, label="RUL real")
        ax.plot(g["ciclo"], np.clip(s["vida_med"] - g["ciclo"], 0, None),
                color=AMBER, lw=1.8, ls="--",
                label=f"calendario:\n{s['vida_med']:.0f} − edad")
        ax.set_title(f"unidad {u} · vivió {vida.loc[u]}", fontsize=10.5)
        ax.set_xlabel("ciclo")
    axs[0].set_ylabel("vida restante (ciclos)")
    axs[0].legend(fontsize=8.5, loc="upper right")
    fig.suptitle("El rival: «todas duran lo mismo». Acierta en la del medio "
                 "— por suerte", y=1.03, fontsize=12.5, color=DARK)
    return fig


def fig_boost_paso(df, vida):
    """Un paso de la cadena, en camara lenta, sobre datos de una unidad."""
    u = 1
    g = df[df["unidad"] == u]
    x = g["ciclo"].to_numpy().reshape(-1, 1)
    ysen = g["T50"].to_numpy()
    y = (ysen - ysen.mean())            # centrado para que se vea la resta

    a1 = DecisionTreeRegressor(max_depth=2, random_state=0).fit(x, y)
    p1 = a1.predict(x)
    r1 = y - p1
    a2 = DecisionTreeRegressor(max_depth=2, random_state=0).fit(x, r1)
    p2 = a2.predict(x)

    fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.3), sharex=True)
    axs[0].scatter(x, y, s=6, color=GRAY, alpha=0.6)
    axs[0].plot(x, p1, color=RED, lw=2.2)
    axs[0].set_title("1) un árbol chico ajusta\nlo que puede", fontsize=10.5)
    axs[1].scatter(x, r1, s=6, color=GRAY, alpha=0.6)
    axs[1].plot(x, p2, color=BLUE, lw=2.2)
    axs[1].set_title("2) el siguiente árbol aprende\nSOLO el error que quedó",
                     fontsize=10.5)
    axs[2].scatter(x, y, s=6, color=GRAY, alpha=0.6)
    axs[2].plot(x, p1 + p2, color=GREEN, lw=2.2)
    axs[2].set_title("3) la suma de los dos\nya sigue mejor la curva",
                     fontsize=10.5)
    for ax in axs:
        ax.set_xlabel("ciclo (unidad 1)")
    axs[0].set_ylabel("T50, centrada")
    return fig


def fig_boost_arboles(df, vida, s, gb_lento):
    """El mismo modelo detenido en 1, 10 y 300 arboles, sobre una unidad."""
    test_u = s["test_u"]
    u = vida.loc[test_u].sort_values().index[len(test_u) // 2]
    m = df["unidad"] == u
    Xu = s["X"][m]
    g = df[m]

    etapas = {1: None, 10: None, 300: None}
    for i, pred in enumerate(gb_lento.staged_predict(Xu), start=1):
        if i in etapas:
            etapas[i] = np.clip(pred, 0, None)

    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    ax.plot(g["ciclo"], g["rul"], color=INK, lw=2.4, label="RUL real")
    for (n, pred), color in zip(etapas.items(), [LGRAY, GRAY, RED]):
        ax.plot(g["ciclo"], pred, lw=1.8, color=color,
                label=f"{n} árbol{'es' if n > 1 else ''}")
    ax.set_xlabel("ciclo de operación")
    ax.set_ylabel("vida restante (ciclos)")
    ax.set_title(f"Unidad {u} (nunca vista): la cadena se acerca "
                 "corrección tras corrección")
    ax.legend(fontsize=10)
    return fig, u


def fig_perilla(df, s, gb_lento, gb_rapido):
    """MAE de validacion vs numero de arboles, dos tamanos de paso."""
    te = s["te"]
    Xte, yte = s["X"][te], df.loc[te, "rul"]
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    for gb, color, nombre in [(gb_lento, RED, "paso 0,1 (con calma)"),
                              (gb_rapido, AMBER, "paso 0,9 (a lo bruto)")]:
        errores = [mean_absolute_error(yte, np.clip(p, 0, None))
                   for p in gb.staged_predict(Xte)]
        ax.plot(range(1, len(errores) + 1), errores, lw=1.8, color=color,
                label=nombre)
    ax.set_xlabel("número de árboles en la cadena")
    ax.set_ylabel("MAE en unidades nunca vistas (ciclos)")
    ax.set_title("La perilla del boosting: pasos chicos y muchos árboles "
                 "le ganan a pasos grandes")
    ax.legend(fontsize=10.5)
    return fig


def fig_malla(busqueda):
    """El resultado de la malla: MAE de validacion cruzada por celda."""
    r = pd.DataFrame(busqueda.cv_results_)
    lrs = MALLA["learning_rate"]
    ds = MALLA["max_depth"]
    M = np.zeros((len(ds), len(lrs)))
    for _, fila in r.iterrows():
        i = ds.index(fila["param_max_depth"])
        j = lrs.index(fila["param_learning_rate"])
        M[i, j] = -fila["mean_test_score"]

    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    im = ax.imshow(M, cmap="RdYlGn_r", aspect="auto")
    for i in range(len(ds)):
        for j in range(len(lrs)):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center",
                    va="center", fontsize=12, weight="bold", color=INK)
    mi = ds.index(busqueda.best_params_["max_depth"])
    mj = lrs.index(busqueda.best_params_["learning_rate"])
    ax.add_patch(Rectangle((mj - 0.48, mi - 0.48), 0.96, 0.96,
                           fill=False, ec=INK, lw=2.5))
    ax.text(mj, mi + 0.32, "la elegida", ha="center", fontsize=9.5,
            color=INK)
    ax.set_xticks(range(len(lrs)))
    ax.set_xticklabels([f"paso {v}" for v in lrs], fontsize=10.5)
    ax.set_yticks(range(len(ds)))
    ax.set_yticklabels([f"profundidad {v}" if v else "profundidad libre"
                        for v in ds], fontsize=10.5)
    ax.set_title("MAE de validación cruzada (por unidad, solo train) "
                 "en cada celda de la malla")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="MAE (ciclos)")
    return fig, M


def fig_marcador(filas):
    fig, ax = plt.subplots(figsize=(8.6, 3.7))
    etiquetas = ["toda la vida", "zona de decisión\n(quedan < 50 ciclos)"]
    tonto = [filas["global"][0], filas["z0"][0]]
    modelo = [filas["global"][1], filas["z0"][1]]
    xs = np.arange(2)
    ax.bar(xs - 0.17, tonto, 0.34, color=AMBER, label="calendario")
    ax.bar(xs + 0.17, modelo, 0.34, color=RED, label="gradient boosting")
    for x, v in zip(xs - 0.17, tonto):
        ax.text(x, v + 0.6, f"{v:.1f}", ha="center", fontsize=11.5,
                color=INK, weight="bold")
    for x, v in zip(xs + 0.17, modelo):
        ax.text(x, v + 0.6, f"{v:.1f}", ha="center", fontsize=11.5,
                color=DARK, weight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(etiquetas, fontsize=11)
    ax.set_ylabel("error medio (ciclos)")
    ax.set_title("El marcador, en las 25 unidades apartadas")
    ax.legend(fontsize=10.5)
    return fig


def fig_zona(df, s, filas):
    """El error por tramo de vida restante: donde el modelo cobra."""
    fig, ax = plt.subplots(figsize=(9.0, 3.9))
    zonas = ["z100", "z50", "z0"]
    etiquetas = ["quedan 100+\n(unidad joven)", "quedan 50–100",
                 "quedan < 50\n(zona de decisión)"]
    xs = np.arange(3)
    ax.bar(xs - 0.17, [filas[z][0] for z in zonas], 0.34, color=AMBER,
           label="calendario")
    ax.bar(xs + 0.17, [filas[z][1] for z in zonas], 0.34, color=RED,
           label="gradient boosting")
    for dx, serie, color in [(-0.17, [filas[z][0] for z in zonas], INK),
                             (0.17, [filas[z][1] for z in zonas], DARK)]:
        for x, v in zip(xs + dx, serie):
            ax.text(x, v + 0.8, f"{v:.1f}", ha="center", fontsize=11,
                    color=color, weight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(etiquetas, fontsize=10.5)
    ax.set_ylabel("error medio (ciclos)")
    ax.set_title("Con la unidad joven casi empatan; cerca de la falla, "
                 "el modelo aplasta")
    ax.legend(fontsize=10.5)
    return fig


def fig_unidad(df, vida, s):
    """Una unidad apartada, vista de cerca: real vs calendario vs modelo."""
    test_u = s["test_u"]
    u = vida.loc[test_u].sort_values().index[2]    # una de vida corta
    m = df["unidad"] == u
    g = df[m]
    fig, ax = plt.subplots(figsize=(9.2, 4.1))
    ax.plot(g["ciclo"], g["rul"], color=INK, lw=2.4, label="RUL real")
    ax.plot(g["ciclo"], np.clip(s["vida_med"] - g["ciclo"], 0, None),
            color=AMBER, lw=1.8, ls="--", label="calendario")
    ax.plot(g["ciclo"], s["pred"][m], color=RED, lw=1.9,
            label="gradient boosting")
    ax.axhline(0, color=GRAY, lw=0.8)
    v = vida.loc[u]
    ax.axvline(v, color=RED, lw=1.0, ls=":")
    ax.text(v - 2, ax.get_ylim()[1] * 0.9, "falla real", ha="right",
            fontsize=10, color=RED)
    ax.set_xlabel("ciclo de operación")
    ax.set_ylabel("vida restante (ciclos)")
    ax.set_title(f"Unidad {u}, vivió {v} ciclos: el calendario la creía "
                 "joven cuando ya se moría")
    ax.legend(fontsize=10)
    return fig, u


def fig_importancias(df, s):
    """Permutation importance en las unidades apartadas."""
    te = s["te"]
    rng = np.random.RandomState(1)
    idx = rng.choice(np.where(te)[0], 3000, replace=False)
    r = permutation_importance(s["modelo"], s["X"].iloc[idx],
                               df["rul"].iloc[idx], n_repeats=5,
                               random_state=SEMILLA, scoring=
                               "neg_mean_absolute_error")
    imp = pd.Series(r.importances_mean, index=s["X"].columns)
    top = imp.sort_values().tail(10)

    def etiqueta(c):
        base = c.rstrip("_md").rstrip("_") if c[-2:] in ("_m", "_d") else c
        base = c[:-2] if c.endswith(("_m", "_d")) else c
        sufijo = {"_m": " (media 20c)", "_d": " (pendiente)"}.get(c[-2:], "")
        return f"{base}{sufijo}"

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.barh([etiqueta(c) for c in top.index], top.values, color=RED,
            edgecolor="white")
    ax.set_xlabel("cuántos ciclos de error agrega desordenar la columna")
    ax.set_title("En qué se fija el modelo (importancia por permutación, "
                 "unidades nunca vistas)")
    return fig, top


def fig_asimetria():
    """La penalizacion del PHM08: tarde castiga mas que temprano."""
    err = np.linspace(-45, 45, 400)
    castigo = np.where(err < 0, np.exp(-err / 13) - 1, np.exp(err / 10) - 1)
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    ax.plot(err, castigo, color=INK, lw=2.0)
    ax.fill_between(err, castigo, 0, where=err > 0, color=RED, alpha=0.18)
    ax.fill_between(err, castigo, 0, where=err < 0, color=BLUE, alpha=0.14)
    ax.text(28, 12, "dije que quedaba MÁS\nde lo que quedaba:\nrevienta "
            "en operación", fontsize=10.5, color=DARK, ha="center")
    ax.text(-28, 12, "dije que quedaba MENOS:\ncambio prematuro,\nvida "
            "desperdiciada", fontsize=10.5, color=BLUE, ha="center")
    ax.set_xlabel("error del pronóstico de RUL (ciclos) · "
                  "positivo = optimista")
    ax.set_ylabel("castigo (score PHM08)")
    ax.set_title("Hasta el benchmark de NASA califica asimétrico: "
                 "llegar tarde cuesta más que llegar temprano")
    return fig


def fig_cuantiles(df, vida, s, u):
    """P50 y P10 del RUL sobre la misma unidad: la banda que se opera."""
    m = df["unidad"] == u
    g = df[m]
    fig, ax = plt.subplots(figsize=(9.2, 4.1))
    ax.plot(g["ciclo"], g["rul"], color=INK, lw=2.2, label="RUL real")
    ax.plot(g["ciclo"], s["pred"][m], color=RED, lw=1.8,
            label="P50 (lo esperado)")
    ax.plot(g["ciclo"], s["pred10"][m], color=BLUE, lw=1.8, ls="--",
            label="P10 (el prudente)")
    ax.axhline(UMBRAL, color=GREEN, lw=1.4, ls=":")
    ax.text(g["ciclo"].iloc[5], UMBRAL + 3,
            f"umbral de aviso: {UMBRAL} ciclos", fontsize=9.5, color=GREEN)
    aviso = g.loc[s["pred10"][m] < UMBRAL, "ciclo"]
    if len(aviso):
        ax.axvline(aviso.iloc[0], color=GREEN, lw=1.4, ls=":")
        ax.axvline(aviso.iloc[0] + MOVIL, color=GREEN, lw=1.4)
        ax.text(aviso.iloc[0] - 3, ax.get_ylim()[1] * 0.75,
                "aviso: se pide\nel equipo", ha="right", fontsize=9.5,
                color=GREEN)
        ax.text(aviso.iloc[0] + MOVIL + 3, ax.get_ylim()[1] * 0.75,
                f"cambio, {MOVIL}\nciclos después", fontsize=9.5,
                color=GREEN)
    ax.set_xlabel("ciclo de operación")
    ax.set_ylabel("vida restante (ciclos)")
    ax.set_title(f"Unidad {u}: no se opera con el P50 — se opera con el "
                 "P10, como la banda de la C3")
    ax.legend(fontsize=10, loc="upper right")
    return fig


def fig_politicas(res):
    """El costo de cada politica, en MUSD por 1.000 ciclos de operacion."""
    orden = ["falla", "mediana", "seguro", "modelo"]
    nombres = ["correr\na la falla", "calendario\n(mediana)",
               "calendario\nconservador", "modelo\n(P10 < margen)"]
    colores = [DARK, AMBER, GRAY, RED]
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    vals = [res[k]["costo_kciclo"] for k in orden]
    bars = ax.bar(nombres, vals, color=colores, width=0.62)
    for b, k in zip(bars, orden):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05,
                f"{res[k]['costo_kciclo']:.2f}", ha="center", fontsize=11.5,
                weight="bold", color=INK)
        ax.text(b.get_x() + b.get_width() / 2, -0.28,
                f"{res[k]['emergencias']} emerg.", ha="center",
                fontsize=9.5, color=MUTED)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("MUSD por 1.000 ciclos corridos")
    ax.set_title("Las cuatro políticas sobre las 25 unidades apartadas "
                 "(supuestos: 0,3 / 1,5 MUSD)")
    ax.margins(y=0.15)
    return fig


def fig_lista(df, vida, s):
    """La lista del mes: foto de la flota de test a mitad de su vida."""
    filas = []
    for u in s["test_u"]:
        m = df["unidad"] == u
        g = df[m]
        corte = int(vida.loc[u] * 0.72)     # la foto: hoy
        f = g[g["ciclo"] == corte]
        if len(f) == 0:
            continue
        filas.append(dict(unidad=u, p10=float(s["pred10"][f.index[0]]),
                          rul_real=int(f["rul"].iloc[0]),
                          idx=f.index[0]))
    tabla = pd.DataFrame(filas).sort_values("p10").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    colores = [RED if i < 5 else GRAY for i in range(len(tabla))]
    ax.barh(range(len(tabla)), tabla["p10"], color=colores,
            edgecolor="white")
    ax.set_yticks(range(len(tabla)))
    ax.set_yticklabels([f"unidad {u}" for u in tabla["unidad"]], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(UMBRAL, color=GREEN, lw=1.5, ls=":")
    ax.text(UMBRAL + 1, len(tabla) - 1.2, f"umbral de aviso: {UMBRAL}",
            fontsize=9.5, color=GREEN)
    ax.set_xlabel("P10 de la vida restante (ciclos)")
    ax.set_title("La lista del mes: mismas 25 unidades, foto al 72 % de su "
                 "vida — los 5 rojos van primero")
    return fig, tabla


def fig_shap_local(df, s, lista):
    """El porque de la primera unidad de la lista: waterfall de Shapley."""
    fila = lista.iloc[0]
    idx = int(fila["idx"])
    explicador = shap.TreeExplainer(s["modelo"])
    sv = explicador(s["X"].iloc[[idx]])
    plt.close("all")
    shap.plots.waterfall(sv[0], max_display=9, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9.0, 4.4)
    fig.suptitle(f"Unidad {int(fila['unidad'])}: de la media de la flota "
                 f"({sv.base_values[0]:.0f}) al pronóstico "
                 f"({sv.base_values[0] + sv.values[0].sum():.0f} ciclos)",
                 fontsize=12, color=DARK, y=1.02)
    contribs = pd.Series(sv.values[0], index=s["X"].columns)
    return fig, contribs, float(sv.base_values[0]), fila


def fig_shap_global(df, s):
    """La vista global: que sensores mandan, y en que direccion."""
    te_idx = np.where(s["te"])[0]
    rng = np.random.RandomState(2)
    muestra = np.sort(rng.choice(te_idx, 2000, replace=False))
    explicador = shap.TreeExplainer(s["modelo"])
    sv = explicador(s["X"].iloc[muestra])
    plt.close("all")
    shap.plots.beeswarm(sv, max_display=10, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9.2, 4.4)
    ax = fig.gca()
    ax.set_xlabel("empuje al pronóstico de vida restante (ciclos)")
    fig.suptitle("2.000 pronósticos de las unidades apartadas, "
                 "repartidos por Shapley", fontsize=12, color=DARK, y=1.0)
    return fig


# --------------------------------------------------------------- main ----
def main():
    print("leyendo datos ...")
    df, vida = cargar()
    print(f"  {len(df):,} filas, {df.unidad.nunique()} unidades")

    print("\nfiguras de EDA y fisica ...")
    guardar(fig_esp(), "fig_esp.png")
    guardar(fig_espagueti(df, vida), "fig_espagueti.png")
    guardar(fig_vidas(vida), "fig_vidas.png")
    f, muertos = fig_sensores_muertos(df)
    guardar(f, "fig_sensores_muertos.png")
    guardar(fig_degradacion(df, vida), "fig_degradacion.png")
    guardar(fig_reloj(df, vida), "fig_reloj.png")

    print("\nentrenando (malla de afinado + tonto + P50 + P10) ...")
    s = entrenar(df, vida)
    filas = maes(df, s)

    print("boosting en camara lenta (staged) ...")
    gb_lento = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.1,
        random_state=SEMILLA)
    gb_lento.fit(s["X"][s["tr"]], df.loc[s["tr"], "rul"])
    gb_rapido = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.9,
        random_state=SEMILLA)
    gb_rapido.fit(s["X"][s["tr"]], df.loc[s["tr"], "rul"])

    print("\nfiguras del modelo ...")
    guardar(fig_rival(df, vida, s), "fig_rival.png")
    guardar(fig_boost_paso(df, vida), "fig_boost_paso.png")
    f, u_media = fig_boost_arboles(df, vida, s, gb_lento)
    guardar(f, "fig_boost_arboles.png")
    guardar(fig_perilla(df, s, gb_lento, gb_rapido), "fig_perilla.png")
    f, M_malla = fig_malla(s["busqueda"])
    guardar(f, "fig_malla.png")
    guardar(fig_marcador(filas), "fig_marcador.png")
    guardar(fig_zona(df, s, filas), "fig_zona.png")
    f, u_corta = fig_unidad(df, vida, s)
    guardar(f, "fig_unidad.png")
    f, top = fig_importancias(df, s)
    guardar(f, "fig_importancias.png")

    print("\npoliticas y dinero ...")
    res, calendario = simular_politicas(df, vida, s)
    guardar(fig_asimetria(), "fig_asimetria.png")
    guardar(fig_cuantiles(df, vida, s, u_corta), "fig_cuantiles.png")
    guardar(fig_politicas(res), "fig_politicas.png")
    f, lista = fig_lista(df, vida, s)
    guardar(f, "fig_lista.png")

    print("\nexplicabilidad (Shapley) ...")
    f, contribs, base, fila_top = fig_shap_local(df, s, lista)
    guardar(f, "fig_shap_local.png")
    guardar(fig_shap_global(df, s), "fig_shap_global.png")

    # ------------------------------------------------------------ cifras ----
    te, y = s["te"], df["rul"]
    cobertura = float((y[te] >= s["pred10"][te]).mean())

    print("\n" + "=" * 64)
    print("CIFRAS PARA LAS LAMINAS (todas salen de aca, ninguna a mano)")
    print("=" * 64)
    print(f"flota: {df.unidad.nunique()} unidades, {len(df):,} filas, "
          f"{len(SENSORES)} sensores")
    print(f"vidas: min {vida.min()}, mediana {vida.median():.0f}, "
          f"max {vida.max()}  (relación {vida.max()/vida.min():.1f} a 1)")
    print(f"sensores muertos ({len(muertos)}): {', '.join(muertos)}")
    print(f"features: {s['X'].shape[1]} columnas "
          f"({len(s['vivos'])} sensores x 3 + edad)")
    print(f"particion: {100-N_TEST} unidades entrenan, {N_TEST} apartadas "
          f"(semilla {SEMILLA})")
    print(f"calendario del rival: mediana de vidas train = "
          f"{s['vida_med']:.0f} ciclos")
    print("-" * 64)
    b = s["busqueda"]
    print(f"afinado (GridSearchCV, GroupKFold de 5 por unidad, solo train):")
    print(f"  mejor celda: paso {b.best_params_['learning_rate']}, "
          f"profundidad {b.best_params_['max_depth']}  "
          f"(MAE CV {-b.best_score_:.1f})")
    r = pd.DataFrame(b.cv_results_)
    fila_def = r[(r.param_learning_rate == 0.1) &
                 (r.param_max_depth.isna())]
    print(f"  celda default (0.1, libre):          "
          f"(MAE CV {-fila_def.mean_test_score.iloc[0]:.1f})")
    print(f"  la malla completa va de {(-r.mean_test_score).min():.1f} a "
          f"{(-r.mean_test_score).max():.1f} ciclos")
    print("-" * 64)
    print("MAE (ciclos)             calendario   boosting")
    for z, nombre in [("global", "toda la vida    "),
                      ("z100", "quedan 100+     "),
                      ("z50", "quedan 50-100   "),
                      ("z0", "quedan <50      ")]:
        print(f"  {nombre}       {filas[z][0]:6.1f}     {filas[z][1]:6.1f}")
    print(f"mejora en la zona de decision: "
          f"{filas['z0'][0]/filas['z0'][1]:.1f}x")
    print("-" * 64)
    y_te, se_te = y[te], s["pred_se"][te]
    z = y_te < 50
    print(f"ablacion sin edad: MAE global "
          f"{mean_absolute_error(y_te, se_te):.1f} "
          f"(con edad {filas['global'][1]:.1f}), zona <50 "
          f"{mean_absolute_error(y_te[z], se_te[z]):.1f} "
          f"(con edad {filas['z0'][1]:.1f})")
    print("-" * 64)
    print(f"P10: el RUL real fue >= P10 el {cobertura*100:.0f} % del tiempo "
          f"(nominal: 90 %)")
    print(f"politica: aviso si P10 < {UMBRAL} ciclos; cambio {MOVIL} "
          f"ciclos despues | costos: "
          f"{COSTO_PLAN} / {COSTO_EMERG} MUSD (supuestos a la vista)")
    print("-" * 64)
    print("politicas (25 unidades de test):")
    print("                        emergencias  vida desp.  MUSD/1000 ciclos")
    nombres = {"falla": "correr a la falla ", "mediana": "calendario mediana",
               "seguro": "calendario conserv", "modelo": "modelo P10<margen "}
    for k in ["falla", "mediana", "seguro", "modelo"]:
        r = res[k]
        print(f"  {nombres[k]}      {r['emergencias']:3d} de 25   "
              f"{r['desperdicio']:6.0f}      {r['costo_kciclo']:6.2f}")
    ahorro = (1 - res["modelo"]["costo_kciclo"] /
              res["seguro"]["costo_kciclo"]) * 100
    print(f"modelo vs mejor calendario: {ahorro:.0f} % mas barato por ciclo")
    print(f"calendario conservador cambia a los "
          f"{calendario['seguro']:.0f} ciclos")
    print("-" * 64)
    print("importancias (top 5, de mayor a menor):")
    for c, v in top.sort_values(ascending=False).head(5).items():
        print(f"  {c:14s} {v:6.2f} ciclos de error si se desordena")
    print("-" * 64)
    print(f"shapley de la unidad {int(fila_top['unidad'])} "
          f"(primera de la lista):")
    print(f"  media de la flota: {base:.0f} ciclos -> pronostico "
          f"{base + contribs.sum():.0f}  (RUL real: "
          f"{int(fila_top['rul_real'])})")
    for c, v in contribs.abs().sort_values(ascending=False).head(4).items():
        print(f"  {c:14s} empuja {contribs[c]:+6.1f} ciclos")
    print("-" * 64)
    print(f"unidad de la figura de cerca: {u_corta} "
          f"(vivio {vida.loc[u_corta]} ciclos)")
    print(f"unidad de la figura 1/10/300: {u_media} "
          f"(vivio {vida.loc[u_media]} ciclos)")


if __name__ == "__main__":
    main()

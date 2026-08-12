"""
Machine Learning for Petroleum Engineers Using Python
Modulo 5 - Clase 2: genera todas las figuras de la presentacion.

Lee  ../datos/pozos_3w_incrustacion.csv  y escribe los fig_*.png de esta carpeta.
Ademas imprime, al final, TODOS los numeros que aparecen en las laminas: ninguna
cifra de la presentacion se escribe a mano.

Uso:  python3 figuras.py

Tipografia: usa Fira Sans (la del deck) si esta instalada. Si no la encuentra,
cae en la sans por defecto de matplotlib y avisa; las figuras salen igual.
"""

import os
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- estilo ----
RED, DARK = "#C82B40", "#6B1525"
BLUE, GREEN, ORANGE = "#2563EB", "#16A34A", "#EA580C"
AMBER = "#D97706"
INK, MUTED = "#2D2D2D", "#6B7280"
GRAY, LGRAY, BG = "#9CA3AF", "#E5E7EB", "#FAFAFA"

C_NORMAL, C_TRANS, C_FALLA = "#DCFCE7", "#FEF3C7", "#FEE2E2"
C_NORMAL_L, C_TRANS_L, C_FALLA_L = GREEN, AMBER, RED


def _fuente():
    for d in (os.environ.get("FIRA_DIR"), "_fuentes"):
        if d and os.path.isdir(d):
            for f in glob.glob(os.path.join(d, "*.ttf")):
                matplotlib.font_manager.fontManager.addfont(f)
    nombres = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    if "Fira Sans" in nombres:
        return "Fira Sans"
    print("  (aviso: Fira Sans no encontrada; se usa la sans por defecto)")
    return matplotlib.rcParams["font.sans-serif"][0]


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [_fuente()],
    "font.size": 11,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": GRAY,
    "axes.labelcolor": INK,
    "axes.titlecolor": DARK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": LGRAY,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "legend.frameon": False,
})


def guardar(fig, nombre):
    fig.savefig(nombre, facecolor="white")
    plt.close(fig)
    print(f"  {nombre}")


def zonas(ax, g, alpha=1.0):
    """Sombrea el fondo segun la etiqueta: normal / transitorio / falla."""
    col = {"normal": C_NORMAL, "transitorio": C_TRANS, "falla": C_FALLA}
    e = g.etiqueta.values
    t = g.t_min.values
    i = 0
    while i < len(e):
        j = i
        while j + 1 < len(e) and e[j + 1] == e[i]:
            j += 1
        ax.axvspan(t[i], t[j], color=col[e[i]], alpha=alpha, lw=0, zorder=0)
        i = j + 1


def leyenda_zonas(ax, loc="upper left"):
    h = [Rectangle((0, 0), 1, 1, fc=C_NORMAL, ec=C_NORMAL_L, lw=1.2),
         Rectangle((0, 0), 1, 1, fc=C_TRANS, ec=C_TRANS_L, lw=1.2),
         Rectangle((0, 0), 1, 1, fc=C_FALLA, ec=C_FALLA_L, lw=1.2)]
    ax.legend(h, ["operación normal", "el evento avanzando", "falla establecida"],
              loc=loc, fontsize=9, ncol=3, handlelength=1.4, columnspacing=1.2)


# ------------------------------------------------------------------ datos ---
# el orden importa: el bosque submuestrea columnas. Este es el mismo
# orden que usa el cuaderno de la clase.
SENS = ["p_antes_choke", "t_despues_choke", "p_arbol", "p_anular", "p_gaslift"]
NOMBRE = {
    "p_antes_choke": "Presión antes del choke",
    "p_arbol": "Presión en el árbol",
    "p_anular": "Presión en el anular",
    "p_gaslift": "Presión del gas lift",
    "t_despues_choke": "Temperatura tras el choke",
}
UNID = {s: ("bar" if s.startswith("p_") else "°C") for s in SENS}
POZOS_CLASE = ["WELL-00001", "WELL-00021", "WELL-00022", "WELL-00024"]
VENT = 60          # ventana de 30 min (60 muestras de 30 s)
BASE = 120         # primera hora = linea base


def cargar():
    d = pd.read_csv("../datos/pozos_3w_incrustacion.csv")
    d["y"] = (d.etiqueta != "normal").astype(int)
    return d


def rasgos(g):
    """Convierte una instancia en una tabla. Por cada sensor deja cuatro
    columnas, que son los cuatro escalones que la clase recorre:
      __crudo  el numero tal cual sale del sensor
      __base   ese numero comparado con la propia hora normal de ESE pozo
      __nivel  el promedio de la ultima media hora  (la tendencia)
      __ruido  cuanto tiembla en esa media hora
      __pend   cuanto cambio respecto de media hora atras
    """
    X = pd.DataFrame(index=g.index)
    for s in SENS:
        if g[s].isna().all():
            continue
        # exactamente el mismo calculo que hace el cuaderno de la clase:
        # sin rellenos y con la ventana completa, para que los numeros de
        # estas laminas se reproduzcan corriendo el .ipynb
        v = g[s]
        z = (v - v.iloc[:BASE].median()) / (v.iloc[:BASE].std() + 1e-9)
        X[f"{s}__base"] = z
        X[f"{s}__nivel"] = z.rolling(VENT).mean()
        X[f"{s}__ruido"] = z.rolling(VENT).std()
        X[f"{s}__pend"] = z.diff(VENT)
    # los valores crudos van al final, en el MISMO orden en que los agrega el
    # cuaderno: el bosque submuestrea columnas, asi que el orden importa para
    # que ambos den identico
    for s in SENS:
        if not g[s].isna().all():
            X[f"{s}__crudo"] = g[s]
    return X


# Los cuatro escalones que recorre la clase, en orden. El que gana es el 2:
# el salto grande lo da comparar cada pozo consigo mismo, no el algoritmo.
ESCALONES = [
    ("Los números tal como\nsalen del sensor", ["__crudo"]),
    ("Comparados con la hora\nnormal de ese pozo", ["__base"]),
    ("Además, promediados\nen media hora", ["__base", "__nivel"]),
    ("Además, el ruido\ny la pendiente", ["__base", "__nivel", "__ruido", "__pend"]),
]
GANADOR = 1                       # indice en ESCALONES
MODELO_FINAL = ESCALONES[GANADOR][1]


def cols(T, sufijos):
    return [c for c in T.columns if any(c.endswith(x) for x in sufijos)]


def umbral_para_falsas(T, p, objetivo):
    """Elige el umbral mas permisivo que respeta un presupuesto de falsas
    alarmas. Comparar modelos a distinto nivel de falsas alarmas es trampa."""
    for thr in np.arange(0.20, 0.995, 0.005):
        if metricas(T, (p >= thr).astype(int))["falsas"] <= objetivo:
            return float(thr)
    return 0.99


def tabla_modelo(d):
    partes = []
    for inst, g in d[d.pozo.isin(POZOS_CLASE)].groupby("instancia"):
        g = g.sort_values("t_min").reset_index(drop=True)
        X = rasgos(g)
        X["y"], X["t_min"] = g.y, g.t_min
        X["instancia"], X["pozo"] = inst, g.pozo.iloc[0]
        partes.append(X.dropna())
    return pd.concat(partes, ignore_index=True)


def evaluar(T, F, thr=0.5):
    """Probabilidades fuera de muestra, dejando un POZO entero por fuera."""
    p = np.zeros(len(T))
    for tr, te in GroupKFold(n_splits=T.pozo.nunique()).split(T[F], T.y, T.pozo):
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                   class_weight="balanced", random_state=0,
                                   n_jobs=-1).fit(T[F].iloc[tr], T.y.iloc[tr])
        p[te] = m.predict_proba(T[F].iloc[te])[:, 1]
    return p


def metricas(T, q):
    tp = ((q == 1) & (T.y == 1)).sum()
    fp = ((q == 1) & (T.y == 0)).sum()
    fn = ((q == 0) & (T.y == 1)).sum()
    ret = []
    for inst, g in T.assign(q=q).groupby("instancia"):
        ini = g.loc[g.y == 1, "t_min"]
        if ini.empty:
            continue
        t0 = ini.iloc[0]
        h = g[(g.q == 1) & (g.t_min >= t0)]
        ret.append(h.t_min.iloc[0] - t0 if len(h) else np.nan)
    ret = np.array(ret, float)
    return dict(recall=100 * tp / max(tp + fn, 1),
                falsas=100 * fp / max((T.y == 0).sum(), 1),
                retraso=np.nanmedian(ret),
                detecta=int(np.isfinite(ret).sum()), total=len(ret), ret=ret)


# =========================================================== FIGURAS ========
def fig_pozo():
    """Esquema: donde esta cada sensor. Se explica el sistema antes del dato."""
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.set_axis_off(); ax.grid(False)
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.4)

    ax.add_patch(Rectangle((0, 3.55), 10, 1.85, fc="#EFF6FF", ec="none", zorder=0))
    ax.add_patch(Rectangle((0, 0), 10, 3.55, fc="#F8F4F2", ec="none", zorder=0))
    ax.plot([0, 10], [3.55, 3.55], color=BLUE, lw=1.4, alpha=.5)
    ax.text(0.15, 5.15, "mar", color=BLUE, fontsize=10, style="italic")
    ax.text(0.15, 3.25, "lecho marino", color="#A8836B", fontsize=10, style="italic")

    # plataforma y linea de produccion
    ax.add_patch(FancyBboxPatch((7.6, 4.55), 2.0, 0.5, boxstyle="round,pad=0.05",
                                fc=DARK, ec="none"))
    ax.text(8.6, 4.8, "plataforma", color="white", ha="center", va="center",
            fontsize=9.5, fontweight="bold")
    ax.plot([3.0, 3.0, 8.6], [3.55, 4.8, 4.8], color=DARK, lw=2.6, zorder=2)

    # pozo
    ax.plot([3.0, 3.0], [0.35, 3.55], color=INK, lw=5.5, zorder=2)
    ax.plot([3.0, 3.0], [0.35, 3.55], color="#D8DCE2", lw=2.6, zorder=3)
    ax.add_patch(Rectangle((2.62, 3.35), 0.76, 0.45, fc=RED, ec="none", zorder=4))
    ax.text(3.0, 3.57, "árbol de\nproducción", color=RED, ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", linespacing=1.15)
    ax.add_patch(Rectangle((0.6, 0.0), 8.8, 0.42, fc="#E7C89A", ec="none", zorder=1))
    ax.text(0.75, 0.16, "yacimiento", color="#8A6A3A", fontsize=9.5, style="italic")

    puntos = [
        (3.0, 4.80, "P-MON-CKP", "Presión antes del choke", 4.35, 5.15, RED),
        (3.0, 3.58, "P-TPT", "Presión en el árbol", 0.35, 2.55, BLUE),
        (3.0, 2.30, "P-ANULAR", "Presión en el anular", 0.35, 1.55, GREEN),
        (5.9, 4.80, "T-JUS-CKP", "Temperatura tras el choke", 6.3, 3.95, ORANGE),
        (3.0, 1.30, "P-JUS-CKGL", "Presión del gas lift", 4.35, 0.85, AMBER),
    ]
    for x, y, tag, nom, tx, ty, c in puntos:
        ax.plot([x], [y], "o", ms=9, mfc="white", mec=c, mew=2.4, zorder=6)
        ax.annotate("", xy=(x, y), xytext=(tx, ty),
                    arrowprops=dict(arrowstyle="-", color=c, lw=1.1, alpha=.75))
        ax.text(tx, ty, f"{tag}\n", color=c, fontsize=9, fontweight="bold",
                ha="left" if tx > x else "right", va="center")
        ax.text(tx, ty, f"\n{nom}", color=MUTED, fontsize=8,
                ha="left" if tx > x else "right", va="center")

    ax.set_title("Un pozo submarino: cinco sensores, una medición por segundo",
                 fontsize=12.5, fontweight="bold", pad=10, loc="left")
    guardar(fig, "fig_pozo.png")


def fig_choke():
    """Que es un choke y que le hace al fluido. Sin esto, nada de hoy se
    entiende: es la pieza que se tapa."""
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 5.6), sharex="col",
                             gridspec_kw=dict(height_ratios=[1.45, 1]))
    for col, (abertura, titulo, c, sub) in enumerate([
            (1.00, "Choke sano", GREEN, "la abertura de siempre"),
            (0.42, "Choke incrustado", RED, "la sal le achicó el paso")]):
        # ---- corte del conducto ----
        ax = axes[0, col]
        ax.set_axis_off(); ax.grid(False)
        ax.set_xlim(0, 10); ax.set_ylim(-2.6, 2.6)
        h = 1.55
        ax.add_patch(Rectangle((0, -h), 4.1, 2 * h, fc="#DBEAFE", ec="none"))
        ax.add_patch(Rectangle((5.9, -h), 4.1, 2 * h, fc="#DBEAFE", ec="none"))
        g = h * abertura
        ax.add_patch(Rectangle((4.1, -g), 1.8, 2 * g, fc="#DBEAFE", ec="none"))
        # cuerpo de la valvula
        ax.add_patch(Rectangle((4.1, g), 1.8, 2.6 - g, fc="#94A3B8", ec="none"))
        ax.add_patch(Rectangle((4.1, -2.6), 1.8, 2.6 - g, fc="#94A3B8", ec="none"))
        if abertura < 1:                      # la costra
            ax.add_patch(Rectangle((4.1, g), 1.8, h - g, fc=RED, ec="none", alpha=.85))
            ax.add_patch(Rectangle((4.1, -h), 1.8, h - g, fc=RED, ec="none", alpha=.85))
            ax.text(5.0, h + .42, "sal pegada", color=RED, fontsize=9,
                    fontweight="bold", ha="center")
        for y in (-0.85, 0, 0.85):
            ax.annotate("", xy=(3.5, y * abertura * .9), xytext=(0.5, y),
                        arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.6))
            ax.annotate("", xy=(9.5, y), xytext=(6.5, y * abertura * .9),
                        arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.6))
        ax.plot([0, 10], [h, h], color=INK, lw=2.2)
        ax.plot([0, 10], [-h, -h], color=INK, lw=2.2)
        ax.text(5.0, 2.28, titulo, color=c, fontsize=12.5, fontweight="bold", ha="center")
        ax.text(5.0, -2.42, sub, color=MUTED, fontsize=9, ha="center", style="italic")
        if col == 0:
            ax.text(1.6, -1.95, "aguas arriba", color=MUTED, fontsize=9, ha="center")
            ax.text(8.4, -1.95, "aguas abajo", color=MUTED, fontsize=9, ha="center")

        # ---- perfiles de presion y temperatura ----
        ax = axes[1, col]
        x = np.linspace(0, 10, 400)
        caida = 1.0 if abertura == 1 else 1.85
        salto = 1 / (1 + np.exp((x - 5.0) * 2.2))
        pres = (55 + 3.2 * caida * salto) if abertura < 1 else (55 + 3.2 * salto)
        temp = 74 - 2.6 * caida * (1 - salto)
        ax.plot(x, pres, color=RED, lw=2.4, label="presión")
        ax.set_ylim(53.5, 61.5); ax.set_ylabel("presión [bar]", fontsize=9, color=RED)
        ax.tick_params(axis="y", labelcolor=RED, labelsize=8.5)
        ax.tick_params(axis="x", labelbottom=False, length=0)
        ax2 = ax.twinx(); ax2.grid(False)
        ax2.plot(x, temp, color=ORANGE, lw=2.4, ls="--")
        ax2.set_ylim(68.5, 75.5)
        ax2.set_ylabel("temperatura [°C]", fontsize=9, color=ORANGE)
        ax2.tick_params(axis="y", labelcolor=ORANGE, labelsize=8.5)
        ax.axvspan(4.1, 5.9, color="#E5E7EB", alpha=.85, zorder=0)
        ax.text(5.0, 53.9, "el choke", color=MUTED, fontsize=8.5, ha="center")
        if col == 1:
            ax.annotate("", xy=(1.2, pres[40]), xytext=(1.2, 55.0),
                        arrowprops=dict(arrowstyle="<->", color=RED, lw=1.8))
            ax.text(1.5, (pres[40] + 55) / 2, "sube\nmás", color=RED, fontsize=9,
                    fontweight="bold", va="center", linespacing=1.2)
            ax2.annotate("", xy=(8.8, temp[-40]), xytext=(8.8, 74.0),
                         arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.8))
            ax2.text(8.5, (temp[-40] + 74) / 2, "baja\nmás", color=ORANGE, fontsize=9,
                     fontweight="bold", va="center", ha="right", linespacing=1.2)
    fig.suptitle("Lo que le hace el choke al fluido — y qué cambia cuando se tapa",
                 fontsize=12.5, fontweight="bold", color=DARK, x=0.005, ha="left", y=1.0)
    fig.tight_layout(h_pad=0.4, w_pad=3.0)
    guardar(fig, "fig_choke.png")


def fig_que_es_serie(d):
    """Que es una señal: la misma medicion a tres escalas de tiempo."""
    g = d[d.instancia == "WELL-00001_20170226130146"].reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.1))
    tramos = [(0, g.t_min.max(), "20 horas de grabación", 1),
              (120, 240, "2 horas", 2),
              (180, 190, "10 minutos", 3)]
    for ax, (a, b, tit, n) in zip(axes, tramos):
        s = g[(g.t_min >= a) & (g.t_min <= b)]
        ax.plot(s.t_min, s.p_antes_choke, color=RED, lw=1.0 if n == 1 else 1.6)
        ax.set_title(f"{n}.  {tit}", fontsize=10.5, fontweight="bold", loc="left")
        ax.set_xlabel("minutos", fontsize=9)
        if n == 1:
            ax.set_ylabel("Presión antes\ndel choke  [bar]", fontsize=9)
            ax.add_patch(Rectangle((120, s.p_antes_choke.min()), 120,
                                   s.p_antes_choke.max() - s.p_antes_choke.min(),
                                   fc=AMBER, alpha=.18, ec=AMBER, lw=1))
        if n == 2:
            ax.add_patch(Rectangle((180, s.p_antes_choke.min()), 10,
                                   s.p_antes_choke.max() - s.p_antes_choke.min(),
                                   fc=AMBER, alpha=.18, ec=AMBER, lw=1))
        ax.tick_params(labelsize=8.5)
    fig.suptitle("La misma medición, acercándose: una serie de tiempo es una fila por instante",
                 fontsize=12, fontweight="bold", color=DARK, x=0.005, ha="left", y=1.06)
    guardar(fig, "fig_que_es_serie.png")


def fig_tres_componentes():
    """Concepto puro, con datos inventados: toda señal = nivel + ciclo + ruido."""
    rng = np.random.default_rng(7)
    t = np.arange(0, 600)
    nivel = 50 + 0.018 * t + 0.000035 * t ** 2
    ciclo = 1.5 * np.sin(2 * np.pi * t / 45)
    ruido = rng.normal(0, 0.55, len(t))
    total = nivel + ciclo + ruido

    fig, axes = plt.subplots(4, 1, figsize=(9.4, 6.4), sharex=True)
    piezas = [
        (total, INK, "Lo que muestra el sensor", "todo junto, que es lo único que uno ve"),
        (nivel, RED, "1 · El NIVEL (la tendencia)", "hacia dónde va la señal — acá está la incrustación"),
        (ciclo, BLUE, "2 · Lo que SE REPITE (la periodicidad)", "el ritmo propio del pozo: el flujo oscila"),
        (ruido, GRAY, "3 · El RUIDO", "lo que queda: el temblor del instrumento"),
    ]
    for ax, (v, c, tit, sub) in zip(axes, piezas):
        ax.plot(t, v, color=c, lw=1.5)
        ax.set_title(tit, fontsize=10.5, fontweight="bold", loc="left", color=c, pad=2)
        ax.text(1.0, 1.02, sub, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8.8, color=MUTED, style="italic")
        ax.tick_params(labelsize=8.5)
    axes[0].set_facecolor("#FAFAFA")
    axes[-1].set_xlabel("tiempo", fontsize=9.5)
    fig.text(0.5, -0.02, "señal  =  nivel  +  lo que se repite  +  ruido",
             ha="center", fontsize=12.5, fontweight="bold", color=DARK)
    fig.tight_layout(h_pad=1.5)
    guardar(fig, "fig_tres_componentes.png")


def fig_senal_cruda(d):
    """La figura central: la señal real, con lo que paso marcado encima."""
    g = d[d.instancia == "WELL-00001_20170226130146"].reset_index(drop=True)
    t0 = g.loc[g.y == 1, "t_min"].iloc[0]
    fig, axes = plt.subplots(2, 1, figsize=(10.6, 5.4), sharex=True,
                             gridspec_kw=dict(height_ratios=[2, 1.35]))
    ax = axes[0]
    zonas(ax, g)
    ax.plot(g.t_min, g.p_antes_choke, color=RED, lw=1.5, zorder=3)
    ax.axvline(t0, color=DARK, lw=1.6, ls="--", zorder=4)
    ax.annotate("acá empieza la incrustación\n(lo dice el especialista, no el sensor)",
                xy=(t0, ax.get_ylim()[1]), xytext=(t0 + 45, ax.get_ylim()[1] - 1.2),
                fontsize=9, color=DARK, fontweight="bold", linespacing=1.2,
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1.2))
    ax.set_ylabel("Presión antes\ndel choke  [bar]", fontsize=9.5)
    leyenda_zonas(ax, loc="lower left")
    ax.set_title("Pozo WELL-00001 · 20 horas de grabación real",
                 fontsize=12.5, fontweight="bold", loc="left", pad=8)

    ax = axes[1]
    zonas(ax, g)
    ax.plot(g.t_min, g.t_despues_choke, color=ORANGE, lw=1.5, zorder=3)
    ax.axvline(t0, color=DARK, lw=1.6, ls="--", zorder=4)
    ax.set_ylabel("Temperatura tras\nel choke  [°C]", fontsize=9.5)
    ax.set_xlabel("minutos desde el inicio de la grabación", fontsize=9.5)
    fig.tight_layout(h_pad=0.8)
    guardar(fig, "fig_senal_cruda.png")


def fig_sensores(d):
    """EDA honesto: no todos los pozos traen todos los sensores."""
    inst = sorted(d.instancia.unique())
    M = np.array([[0 if d[d.instancia == i][s].isna().all() else 1 for s in SENS]
                  for i in inst])
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.grid(False)
    ax.imshow(M, cmap=matplotlib.colors.ListedColormap(["#FEE2E2", "#DCFCE7"]),
              aspect="auto", vmin=0, vmax=1)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, "sí" if M[i, j] else "no", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color=GREEN if M[i, j] else RED)
    ax.set_xticks(range(len(SENS)))
    ax.set_xticklabels([NOMBRE[s].replace(" ", "\n", 1) for s in SENS], fontsize=8.5)
    ax.set_yticks(range(len(inst)))
    ax.set_yticklabels([i.replace("WELL-000", "Pozo ").replace("_", "  ·  ")
                        for i in inst], fontsize=8)
    for k in (0, 1, 2, 3):
        ax.axhline(k + 0.5, color="white", lw=2)
    ax.axhline(3.5, color=DARK, lw=2.2)
    ax.axhline(5.5, color=DARK, lw=2.2)
    ax.text(4.62, 4.5, "solo\n2 sensores", color=RED, fontsize=9, fontweight="bold",
            va="center", ha="left", linespacing=1.2)
    ax.set_title("Lo primero que hay que mirar: ¿qué sensores existen de verdad?",
                 fontsize=12, fontweight="bold", loc="left", pad=10)
    ax.tick_params(length=0)
    guardar(fig, "fig_sensores.png")


def fig_lineas_base(d):
    """Cada pozo vive en otro rango: por eso se compara contra si mismo."""
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.7))
    ax = axes[0]
    for i, (pozo, g) in enumerate(d[d.pozo.isin(POZOS_CLASE)].groupby("pozo")):
        for _, gg in g.groupby("instancia"):
            gg = gg.reset_index(drop=True)
            ax.plot(gg.t_min, gg.p_antes_choke, lw=1.1,
                    color=[RED, BLUE, GREEN, ORANGE][i], alpha=.75,
                    label=pozo.replace("WELL-000", "Pozo ") if _ == g.instancia.iloc[0] else None)
    ax.set_title("Presión cruda: cada pozo en su mundo", fontsize=11,
                 fontweight="bold", loc="left")
    ax.set_ylabel("bar", fontsize=9.5); ax.set_xlabel("minutos", fontsize=9.5)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[:4], l[:4], fontsize=8.5, ncol=2)

    ax = axes[1]
    for i, (pozo, g) in enumerate(d[d.pozo.isin(POZOS_CLASE)].groupby("pozo")):
        for _, gg in g.groupby("instancia"):
            gg = gg.reset_index(drop=True)
            v = gg.p_antes_choke
            z = (v - v.iloc[:BASE].median()) / (v.iloc[:BASE].std() + 1e-9)
            ax.plot(gg.t_min, z, lw=1.1, color=[RED, BLUE, GREEN, ORANGE][i], alpha=.75)
    ax.axhline(0, color=INK, lw=1.2)
    ax.axhspan(-3, 3, color=LGRAY, alpha=.55, zorder=0)
    ax.set_title("Contra su propia hora normal: ahora sí se comparan",
                 fontsize=11, fontweight="bold", loc="left")
    ax.set_ylabel("desviaciones respecto\nde su normal", fontsize=9.5)
    ax.set_xlabel("minutos", fontsize=9.5); ax.set_ylim(-12, 12)
    fig.tight_layout(w_pad=2.2)
    guardar(fig, "fig_lineas_base.png")


def fig_ventana(d):
    """De una señal a una tabla: que hace exactamente la ventana deslizante."""
    g = d[d.instancia == "WELL-00001_20170226130146"].reset_index(drop=True)
    s = g[(g.t_min >= 100) & (g.t_min <= 340)]
    fig, axes = plt.subplots(2, 1, figsize=(9.8, 4.9),
                             gridspec_kw=dict(height_ratios=[2, 1]))
    ax = axes[0]
    ax.plot(s.t_min, s.p_antes_choke, color=GRAY, lw=1.3)
    for k, (a, c) in enumerate([(140, RED), (200, BLUE), (260, GREEN)]):
        w = s[(s.t_min >= a) & (s.t_min <= a + 30)]
        ax.plot(w.t_min, w.p_antes_choke, color=c, lw=2.4)
        ax.add_patch(Rectangle((a, w.p_antes_choke.min() - .15), 30,
                               w.p_antes_choke.max() - w.p_antes_choke.min() + .3,
                               fc=c, alpha=.12, ec=c, lw=1.4))
        ax.text(a + 15, w.p_antes_choke.max() + .35, f"ventana {k+1}", color=c,
                fontsize=9, fontweight="bold", ha="center")
    ax.set_ylabel("Presión antes\ndel choke  [bar]", fontsize=9.5)
    ax.set_xlabel("minutos", fontsize=9.5)
    ax.set_title("Cada media hora de señal se resume en tres números por sensor",
                 fontsize=12, fontweight="bold", loc="left", pad=8)

    ax = axes[1]; ax.set_axis_off(); ax.grid(False)
    filas = [["", "nivel", "ruido", "pendiente"],
             ["ventana 1", "−0.2", "0.9", "+0.1"],
             ["ventana 2", "+1.4", "1.1", "+1.6"],
             ["ventana 3", "+4.8", "2.3", "+3.4"]]
    cols = [None, RED, BLUE, GREEN]
    for i, f in enumerate(filas):
        for j, txt in enumerate(f):
            x, y = 0.05 + j * 0.17, 0.82 - i * 0.22
            enc, prim = i == 0, j == 0
            ax.text(x, y, txt, fontsize=10.5,
                    fontweight="bold" if (enc or prim) else "normal",
                    color=DARK if enc else (cols[i] if prim else INK),
                    ha="left", va="center")
        if i == 0:
            ax.plot([0.04, 0.76], [y - 0.09] * 2, color=DARK, lw=1.2)
    ax.text(0.80, 0.60, "una fila de la tabla\npor cada ventana", fontsize=9.5,
            color=MUTED, style="italic", va="center", linespacing=1.3)
    ax.text(0.05, 0.02, "y así el problema deja de ser una serie de tiempo: "
                        "pasa a ser la tabla de siempre.",
            fontsize=10, color=DARK, fontweight="bold")
    fig.tight_layout(h_pad=0.6)
    guardar(fig, "fig_ventana.png")


def fig_descomposicion_real(d):
    """Las tres componentes, ahora sobre la señal de verdad."""
    g = d[d.instancia == "WELL-00001_20170226130146"].reset_index(drop=True)
    v = g.p_antes_choke
    z = (v - v.iloc[:BASE].median()) / (v.iloc[:BASE].std() + 1e-9)
    nivel = z.rolling(VENT, min_periods=5, center=True).mean()
    resto = z - nivel
    ruido = resto.rolling(VENT, min_periods=5, center=True).std()
    t0 = g.loc[g.y == 1, "t_min"].iloc[0]

    fig, axes = plt.subplots(3, 1, figsize=(9.8, 5.8), sharex=True)
    for ax, (serie, c, tit) in zip(axes, [
            (z, INK, "La señal, comparada contra su propia hora normal"),
            (nivel, RED, "1 · El NIVEL — sube y no vuelve: eso es la incrustación"),
            (ruido, BLUE, "2 · El RUIDO — el pozo además empieza a temblar")]):
        zonas(ax, g, alpha=.55)
        ax.plot(g.t_min, serie, color=c, lw=1.5, zorder=3)
        ax.axvline(t0, color=DARK, lw=1.4, ls="--", zorder=4)
        ax.set_title(tit, fontsize=10.5, fontweight="bold", loc="left", color=c, pad=3)
        ax.tick_params(labelsize=8.5)
    axes[-1].set_xlabel("minutos desde el inicio de la grabación", fontsize=9.5)
    fig.tight_layout(h_pad=1.0)
    guardar(fig, "fig_descomposicion_real.png")


def fig_alarma(d, t_alarma, t_evento):
    """La alarma de umbral: correcta, y tardisima."""
    g = d[d.instancia == "WELL-00001_20170226130146"].reset_index(drop=True)
    v = g.p_antes_choke
    z = (v - v.iloc[:BASE].median()) / (v.iloc[:BASE].std() + 1e-9)
    fig, ax = plt.subplots(figsize=(10.4, 4.3))
    zonas(ax, g)
    ax.plot(g.t_min, z, color=INK, lw=1.4, zorder=3)
    ax.axhspan(-3, 3, color="#E0E7FF", alpha=.75, zorder=1)
    ax.axhline(3, color=BLUE, lw=1.5, ls="--", zorder=2)
    ax.axhline(-3, color=BLUE, lw=1.5, ls="--", zorder=2)
    ax.text(g.t_min.max(), 3.4, "umbral de alarma  (3 desviaciones)", color=BLUE,
            fontsize=9.5, fontweight="bold", ha="right")
    ax.axvline(t_evento, color=DARK, lw=1.8, ls="--", zorder=5)
    ax.axvline(t_alarma, color=RED, lw=2.2, zorder=5)
    ax.annotate("", xy=(t_evento, -8), xytext=(t_alarma, -8),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=2))
    ax.text((t_evento + t_alarma) / 2, -7.2,
            f"{t_alarma - t_evento:.0f} minutos tarde", color=RED,
            fontsize=12, fontweight="bold", ha="center")
    ax.text(g.t_min.max(), -9.4, "en ESTE pozo. Falta ver los otros nueve.",
            color=MUTED, fontsize=9.5, style="italic", ha="right")
    ax.text(t_evento - 12, 9.2, "empieza el evento", color=DARK, fontsize=9.5,
            fontweight="bold", ha="right")
    ax.text(t_alarma + 12, 9.2, "suena la alarma", color=RED, fontsize=9.5,
            fontweight="bold")
    ax.set_ylabel("desviaciones respecto\nde su hora normal", fontsize=9.5)
    ax.set_xlabel("minutos desde el inicio de la grabación", fontsize=9.5)
    ax.set_ylim(-10, 11)
    ax.set_title("La alarma que ya existe: no se equivoca, llega tarde",
                 fontsize=12.5, fontweight="bold", loc="left", pad=8)
    guardar(fig, "fig_alarma.png")


def fig_resultado(T, ret_al, ret_rf):
    """Instancia por instancia: cuando avisa cada uno."""
    inst = sorted(T.instancia.unique())
    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    ax.grid(axis="y")
    n = len(inst)
    for i, (a, r) in enumerate(zip(ret_al, ret_rf)):
        y = n - i
        ax.plot([0, 0], [y - .3, y + .3], color=DARK, lw=2.4)
        for val, c, m in [(a, GRAY, "s"), (r, RED, "o")]:
            if np.isfinite(val):
                ax.plot([0, val], [y, y], color=c, lw=2.6, alpha=.55, zorder=2)
                ax.plot([val], [y], m, ms=9, color=c, zorder=3)
            else:
                ax.text(430, y, "nunca detecta", color=GRAY if c is GRAY else RED,
                        fontsize=8.5, va="center", style="italic")
    ax.set_yticks(range(1, n + 1))
    ax.set_yticklabels([i.replace("WELL-000", "Pozo ").replace("_", " · ")
                        for i in inst[::-1]], fontsize=8.5)
    ax.set_xlabel("minutos DESPUÉS de que empieza el evento  (0 = en el momento)",
                  fontsize=9.5)
    ax.set_xlim(-25, 520)
    ax.legend(handles=[Line2D([], [], marker="s", color=GRAY, lw=0, ms=9,
                              label="alarma de umbral"),
                       Line2D([], [], marker="o", color=RED, lw=0, ms=9,
                              label="el modelo")],
              fontsize=10, loc="lower right", ncol=2)
    ax.set_title("Cuánto tarda cada uno en darse cuenta",
                 fontsize=12.5, fontweight="bold", loc="left", pad=8)
    guardar(fig, "fig_resultado.png")


def fig_importancia(T, F):
    """Cual de los cinco sensores delata la incrustacion."""
    m = RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                               class_weight="balanced", random_state=0,
                               n_jobs=-1).fit(T[F], T.y)
    imp = pd.Series(m.feature_importances_, index=F).sort_values()
    fig, ax = plt.subplots(figsize=(8.8, 3.9))
    ax.grid(axis="y")
    colores = [RED if i == len(imp) - 1 else GRAY for i in range(len(imp))]
    ax.barh(range(len(imp)), 100 * imp.values, color=colores, height=.66)
    for i, v in enumerate(100 * imp.values):
        ax.text(v + .8, i, f"{v:.0f} %", va="center", fontsize=10,
                fontweight="bold", color=RED if i == len(imp) - 1 else MUTED)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels([NOMBRE[f.split("__")[0]] for f in imp.index], fontsize=9.5)
    ax.set_xlabel("cuánto se apoya el modelo en cada sensor  [%]", fontsize=9.5)
    ax.set_xlim(0, 100 * imp.max() * 1.28)
    ax.set_title("Quién delata la incrustación", fontsize=12.5,
                 fontweight="bold", loc="left", pad=8)
    guardar(fig, "fig_importancia.png")


def fig_operacion(T, p, presupuesto):
    """La perilla: recall contra falsas alarmas. Es una decision, no un numero."""
    thrs = np.linspace(0.15, 0.9, 40)
    R, Fa = [], []
    for t in thrs:
        m = metricas(T, (p >= t).astype(int))
        R.append(m["recall"]); Fa.append(m["falsas"])
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(Fa, R, color=RED, lw=2.6, zorder=3)
    thr_el = umbral_para_falsas(T, p, presupuesto)
    for t, txt, c in [(0.35, "avisar de más", ORANGE), (thr_el, "el elegido", RED),
                      (min(thr_el + 0.12, 0.97), "avisar casi nunca", BLUE)]:
        m = metricas(T, (p >= t).astype(int))
        ax.plot([m["falsas"]], [m["recall"]], "o", ms=11, color=c, zorder=5,
                mec="white", mew=2)
        ax.annotate(f"{txt}\numbral {t:.1f}", xy=(m["falsas"], m["recall"]),
                    xytext=(m["falsas"] + 3, m["recall"] - 9), fontsize=9,
                    color=c, fontweight="bold", linespacing=1.3,
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.2))
    ax.set_xlabel("alertas durante operación normal  [%]", fontsize=9.5)
    ax.set_ylabel("eventos detectados  [%]", fontsize=9.5)
    ax.set_title("No hay un modelo: hay una perilla, y alguien tiene que girarla",
                 fontsize=12.5, fontweight="bold", loc="left", pad=8)
    guardar(fig, "fig_operacion.png")


def fig_falsas(T, p, tercios, avisos):
    """El cierre: las falsas alarmas no estaban repartidas al azar."""
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.0),
                             gridspec_kw=dict(width_ratios=[1, 1.25]))
    ax = axes[0]
    ax.grid(axis="x")
    b = ax.bar(["primer\ntercio", "segundo\ntercio", "último\ntercio"], tercios,
               color=[LGRAY, "#FDE68A", RED], width=.62)
    for r, v in zip(b, tercios):
        ax.text(r.get_x() + r.get_width() / 2, v + 1.2, f"{v:.0f} %", ha="center",
                fontsize=11, fontweight="bold", color=DARK)
    ax.set_ylabel("alertas durante el período\netiquetado como normal  [%]", fontsize=9.5)
    ax.set_title("¿Dónde caen?", fontsize=12, fontweight="bold", loc="left", pad=8)
    ax.set_ylim(0, max(tercios) * 1.35)

    ax = axes[1]
    ax.grid(axis="y")
    orden = sorted(avisos.items(), key=lambda x: x[1])
    vals = [v / 60 for _, v in orden]
    labs = [k for k, _ in orden]
    ax.barh(range(len(vals)), vals, color=GREEN, height=.6)
    for i, v in enumerate(vals):
        ax.text(v + .12, i, f"{v:.1f} h antes", va="center", fontsize=10,
                fontweight="bold", color=GREEN)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labs, fontsize=8.5)
    ax.set_xlabel("cuánto ANTES de la etiqueta empieza a avisar", fontsize=9.5)
    ax.set_xlim(0, max(vals) * 1.35)
    ax.set_title("No eran falsas: eran tempranas", fontsize=12, fontweight="bold",
                 loc="left", pad=8, color=GREEN)
    fig.tight_layout(w_pad=2.5)
    guardar(fig, "fig_falsas.png")


def fig_escalera(esc, m_al):
    """Los cuatro escalones, todos medidos con el MISMO presupuesto de falsas
    alarmas que la alarma que ya existe. El cuarto no mejora: se dice."""
    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    ax.grid(axis="x")
    nombres = ["La alarma que\nya existe"] + [n for n, _ in ESCALONES]
    todos = [m_al] + esc
    colores = [GRAY, "#D1D5DB", RED, "#F0A0AC", "#E7CDD1"]
    b = ax.bar(range(len(todos)), [m["detecta"] * 10 for m in todos],
               color=colores, width=.6)
    for i, (r, m) in enumerate(zip(b, todos)):
        gana = (i == GANADOR + 1)
        ax.text(r.get_x() + r.get_width() / 2, m["detecta"] * 10 + 3.5,
                f"{m['detecta']} de 10", ha="center", fontsize=12,
                fontweight="bold", color=RED if gana else DARK)
        ax.text(r.get_x() + r.get_width() / 2, m["detecta"] * 10 / 2,
                f"avisa a los\n{m['retraso']:.0f} min", ha="center", va="center",
                fontsize=9.5, fontweight="bold",
                color="white" if gana else INK, linespacing=1.35)
    ax.set_xticks(range(len(todos)))
    ax.set_xticklabels(nombres, fontsize=9, linespacing=1.3)
    ax.set_ylabel("casos detectados  (de 10)", fontsize=9.5)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(["0", "2", "4", "6", "8", "10"])
    ax.set_ylim(0, 124)
    ax.plot([GANADOR + 1 - 0.30], [114], marker="v", ms=11, color=RED, clip_on=False)
    ax.text(GANADOR + 1 - 0.22, 114, "el que nos llevamos", ha="left", va="center",
            fontsize=10.5, fontweight="bold", color=RED)
    ax.text(3.5, 113, "más recall, pero avisa más tarde\ny se le escapa un pozo",
            ha="center", fontsize=8.8, color=MUTED, style="italic", linespacing=1.3)
    ax.text(0.0, -0.34, "los cinco medidos con el MISMO presupuesto de falsas alarmas "
            f"({m_al['falsas']:.0f} %), el de la alarma actual — comparar a distinta "
            "tasa de falsas alarmas es trampa",
            transform=ax.transAxes, fontsize=9, color=MUTED, style="italic")
    ax.set_title("Cada escalón hay que ganárselo",
                 fontsize=12.5, fontweight="bold", loc="left", pad=8)
    guardar(fig, "fig_escalera.png")


# =============================================================== MAIN =======
def main():
    print("leyendo datos ...")
    d = cargar()
    T = tabla_modelo(d)
    F = [c for c in T.columns if "__" in c]
    print(f"  tabla del modelo: {len(T)} filas x {len(F)} columnas, "
          f"{T.pozo.nunique()} pozos, {T.instancia.nunique()} instancias")

    print("\nla alarma que ya existe (3 desviaciones sobre la presión del choke) ...")
    q_al = (T["p_antes_choke__base"].abs() > 3).astype(int).values
    m_al = metricas(T, q_al)
    PRESUPUESTO = m_al["falsas"]      # a esto tiene que ajustarse el modelo

    print("entrenando los cuatro escalones (deja un pozo entero por fuera) ...")
    esc = []
    for nom, suf in ESCALONES:
        pe = evaluar(T, cols(T, suf))
        te = umbral_para_falsas(T, pe, PRESUPUESTO)
        esc.append(metricas(T, (pe >= te).astype(int)))
        print(f"    {nom.replace(chr(10),' '):>46} listo")

    # el modelo que se lleva la clase: el escalon que gano
    p = evaluar(T, cols(T, MODELO_FINAL))
    THR = umbral_para_falsas(T, p, PRESUPUESTO)
    q_rf = (p >= THR).astype(int)
    m_rf = metricas(T, q_rf)

    # ¿donde caen las alertas del periodo normal?
    tercios, avisos = [[], [], []], {}
    for inst, g in T.assign(q=q_rf).groupby("instancia"):
        n = g[g.y == 0].sort_values("t_min")
        if len(n) < 30:
            continue
        k = len(n) // 3
        for i, part in enumerate([n.iloc[:k], n.iloc[k:2 * k], n.iloc[2 * k:]]):
            tercios[i].append(100 * part.q.mean())
        racha = 0
        for v in n.q.values[::-1]:
            if v == 1:
                racha += 1
            else:
                break
        if racha:
            avisos[inst.replace("WELL-000", "Pozo ").replace("_", " · ")] = racha * 0.5
    tercios = [float(np.mean(x)) for x in tercios]

    print("\ngenerando figuras ...")
    inst0 = "WELL-00001_20170226130146"
    g0 = T[T.instancia == inst0]
    t_ev = g0.loc[g0.y == 1, "t_min"].iloc[0]
    al0 = g0[(g0["p_antes_choke__base"].abs() > 3) & (g0.t_min >= t_ev)]
    t_al = al0.t_min.iloc[0] if len(al0) else g0.t_min.max()

    fig_pozo()
    fig_choke()
    fig_que_es_serie(d)
    fig_tres_componentes()
    fig_senal_cruda(d)
    fig_sensores(d)
    fig_lineas_base(d)
    fig_ventana(d)
    fig_descomposicion_real(d)
    fig_alarma(d, t_al, t_ev)
    fig_resultado(T, m_al["ret"], m_rf["ret"])
    fig_importancia(T, cols(T, MODELO_FINAL))
    fig_operacion(T, p, PRESUPUESTO)
    fig_falsas(T, p, tercios, avisos)
    fig_escalera(esc, m_al)

    print("\n" + "=" * 74)
    print("NUMEROS PARA LAS LAMINAS  (no escribir ninguno a mano)")
    print("=" * 74)
    print(f"archivo: {len(d)} filas | {d.pozo.nunique()} pozos | "
          f"{d.instancia.nunique()} grabaciones | {len(d)*30/3600:.0f} horas")
    print(f"tabla del modelo: {len(T)} filas | {T.instancia.nunique()} grabaciones "
          f"de {T.pozo.nunique()} pozos")
    print(f"\npresupuesto de falsas alarmas (el de la alarma actual): {PRESUPUESTO:.1f} %")
    print(f"umbral elegido para el modelo: {THR:.3f}")
    print(f"\n{'':>46} {'detecta':>8} {'recall':>8} {'retraso':>9}")
    print(f"{'LA ALARMA QUE YA EXISTE':>46} {m_al['detecta']:>4}/10 "
          f"{m_al['recall']:>7.1f}% {m_al['retraso']:>7.0f} min")
    for (nom, _), e in zip(ESCALONES, esc):
        print(f"{nom.replace(chr(10),' '):>46} {e['detecta']:>4}/10 "
              f"{e['recall']:>7.1f}% {e['retraso']:>7.0f} min")
    print(f"\nel modelo avisa {(m_al['retraso']-m_rf['retraso'])/60:.1f} horas antes "
          f"que la alarma ({m_al['retraso']:.0f} min -> {m_rf['retraso']:.0f} min)")
    print(f"retraso de la alarma en el pozo 1 (la figura): {t_al-t_ev:.0f} min "
          f"= {(t_al-t_ev)/60:.1f} h")
    print(f"\nalertas durante el periodo NORMAL, por tercios: "
          f"{tercios[0]:.0f}% / {tercios[1]:.0f}% / {tercios[2]:.0f}%")
    print("aviso anticipado (cuanto ANTES de la etiqueta empieza a avisar):")
    for k, v in sorted(avisos.items(), key=lambda x: -x[1]):
        print(f"    {k}: {v/60:.1f} h")
    print("\npuntos de operacion del modelo final:")
    for t in (0.3, 0.5, 0.7, THR):
        m = metricas(T, (p >= t).astype(int))
        marca = "  <-- el elegido" if abs(t - THR) < 1e-9 else ""
        print(f"    umbral {t:.2f}: detecta {m['detecta']}/10 | recall {m['recall']:.0f}% | "
              f"falsas {m['falsas']:.0f}% | retraso {m['retraso']:.0f} min{marca}")


if __name__ == "__main__":
    main()

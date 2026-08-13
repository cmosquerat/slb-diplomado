"""
Machine Learning for Petroleum Engineers Using Python
Modulo 5 - Clase 3: genera todas las figuras de la presentacion.

Lee  ../datos/campos_noruega_declinacion.csv  y escribe los fig_*.png de esta
carpeta. Al final imprime TODAS las cifras que aparecen en las laminas: ninguna
se escribe a mano, y el cuaderno de la clase tiene que reproducirlas.

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
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit

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


def guardar(fig, nombre):
    fig.savefig(nombre, facecolor="white")
    plt.close(fig)
    print(f"  {nombre}")


# ------------------------------------------------------------- el modelo ----
AJUSTE = 5 * 12        # con cuanta historia ajustamos: 5 anios
OBJETIVO = 12 * 12     # que predecimos: el acumulado a 12 anios
EJEMPLO = "DRAUGEN"    # el campo que sigue toda la clase
DIAS_MES = 30.4        # dias promedio de un mes: pasa de bbl/dia a barriles

# --- supuestos economicos. Estos numeros los pone finanzas, no el ingeniero:
# se declaran a la vista para que se puedan discutir y cambiar.
PRECIO = 70.0          # USD por barril
OPEX = 15.0            # USD por barril producido
COSTO_FIJO = 8e6       # USD por mes que cuesta tener el campo abierto
D_MIN = 0.05           # declinacion terminal: 5 %/anio
MARGEN = PRECIO - OPEX
Q_LIMITE = COSTO_FIJO / (MARGEN * DIAS_MES)    # bbl/dia por debajo de los cuales
                                               # el campo pierde plata


def hiperbolica(t, qi, Di, b):
    """Arps (1945) hiperbolica. Con b -> 0 se vuelve la exponencial."""
    return qi / np.power(1.0 + b * Di * t, 1.0 / b)


def cargar():
    d = pd.read_csv("../datos/campos_noruega_declinacion.csv")
    campos, reales = {}, {}
    for c, g in d.groupby("campo"):
        g = g.sort_values("mes_desde_pico")
        campos[c] = g.oil_bpd.reset_index(drop=True)
        # el acumulado REAL usa los dias de cada mes, no un promedio
        reales[c] = (g.oil_bpd * g.dias).reset_index(drop=True)
    return d, campos, reales


def ajustar(q):
    """Devuelve los dos ajustes de Arps sobre los primeros 5 anios."""
    y = q.iloc[:AJUSTE].values
    t = np.arange(AJUSTE)
    m = y > 0
    if m.sum() < 24:
        return None
    be = np.polyfit(t[m], np.log(y[m]), 1)
    if be[0] >= 0:
        return None
    D_anual = 100 * (1 - np.exp(be[0] * 12))
    try:
        p, _ = curve_fit(hiperbolica, t[m], y[m], p0=[y[m][0], 0.02, 0.5],
                         bounds=([0, 1e-6, 1e-3], [np.inf, 1.0, 2.0]), maxfev=20000)
    except Exception:
        return None
    return dict(exp_coef=be, hip_par=p, D_anual=D_anual, b=p[2])


def curva_exp(be, n):
    return np.exp(np.polyval(be, np.arange(n)))


def curva_hip(p, n):
    return hiperbolica(np.arange(n), *p)


def curva_hip_terminal(p, n, d_min=D_MIN):
    """La hiperbolica con freno: cuando su declinacion instantanea baja de
    d_min, se cambia a exponencial. Es la practica estandar de la industria.
    Sin esto, un b alto predice que el campo no se muere nunca."""
    qi, Di, b = p
    t = np.arange(n)
    q = hiperbolica(t, *p)
    D_inst = Di / (1.0 + b * Di * t)               # declinacion nominal mensual
    d_min_mes = -np.log(1 - d_min) / 12.0
    sw = np.where(D_inst <= d_min_mes)[0]
    if len(sw):
        i = sw[0]
        q[i:] = q[i] * np.exp(-d_min_mes * (t[i:] - t[i]))
    return q


def mes_de_abandono(q):
    """Primer mes en que la curva cae por debajo del limite economico."""
    i = np.where(q < Q_LIMITE)[0]
    return int(i[0]) if len(i) else None


def resultados(campos, reales):
    """Una fila por campo: lo que predijo cada metodo y lo que de verdad paso."""
    filas = []
    for c, q in campos.items():
        a = ajustar(q)
        if a is None:
            continue
        # DIA es el promedio de dias de un mes: oil_bpd es un CAUDAL, y para
        # acumular barriles hay que multiplicarlo por los dias de cada mes
        filas.append(dict(
            campo=c,
            real=reales[c].iloc[:OBJETIVO].sum(),
            acum5=reales[c].iloc[:AJUSTE].sum(),
            exp=curva_exp(a["exp_coef"], OBJETIVO).sum() * DIAS_MES,
            hip=curva_hip(a["hip_par"], OBJETIVO).sum() * DIAS_MES,
            D_anual=a["D_anual"], b=a["b"],
        ))
    r = pd.DataFrame(filas)
    r["k"] = r.real / r.acum5                      # el multiplicador de verdad
    r["e_exp"] = 100 * (r.exp / r.real - 1)
    r["e_hip"] = 100 * (r.hip / r.real - 1)
    # analogos: dejando el campo de lado, ¿que multiplicador dicen los otros?
    an, dentro = [], []
    for i, row in r.iterrows():
        o = r.drop(i)
        an.append(100 * (row.acum5 * o.k.median() / row.real - 1))
        lo, hi = o.k.quantile([0.10, 0.90])
        dentro.append(bool(row.acum5 * lo <= row.real <= row.acum5 * hi))
    r["e_an"] = an
    r["en_banda"] = dentro
    return r


def calibracion(r):
    """Para cada nivel nominal, que fraccion cayo realmente adentro."""
    niveles, reales = [], []
    for nivel in np.arange(0.1, 0.96, 0.05):
        a = (1 - nivel) / 2
        ok = 0
        for i, row in r.iterrows():
            o = r.drop(i)
            lo, hi = o.k.quantile([a, 1 - a])
            ok += row.acum5 * lo <= row.real <= row.acum5 * hi
        niveles.append(100 * nivel)
        reales.append(100 * ok / len(r))
    return np.array(niveles), np.array(reales)


# =========================================================== FIGURAS ========
def fig_que_es_declinacion(campos):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.0))
    ax = axes[0]
    for c in ["STATFJORD", "GULLFAKS", "EKOFISK", "DRAUGEN", "NORNE"]:
        if c in campos:
            q = campos[c].iloc[:25 * 12]
            ax.plot(np.arange(len(q)) / 12, q / 1000, lw=1.6, label=c.title())
    ax.set_xlabel("años desde el pico", fontsize=9.5)
    ax.set_ylabel("producción de petróleo\n[miles de barriles por día]", fontsize=9.5)
    ax.set_title("Cinco campos del Mar del Norte", fontsize=11.5,
                 fontweight="bold", loc="left")
    ax.legend(fontsize=8.5, ncol=2)

    ax = axes[1]
    for c in ["STATFJORD", "GULLFAKS", "EKOFISK", "DRAUGEN", "NORNE"]:
        if c in campos:
            q = campos[c].iloc[:25 * 12]
            ax.plot(np.arange(len(q)) / 12, 100 * q / q.iloc[:6].mean(), lw=1.6)
    ax.set_yscale("log")
    ax.set_xlabel("años desde el pico", fontsize=9.5)
    ax.set_ylabel("% de su propio pico\n(escala logarítmica)", fontsize=9.5)
    ax.set_title("Los mismos cinco, en % de su pico: casi la misma curva",
                 fontsize=11.5, fontweight="bold", loc="left")
    fig.tight_layout(w_pad=2.4)
    guardar(fig, "fig_que_es_declinacion.png")


def fig_el_encargo(campos):
    q = campos[EJEMPLO]
    fig, ax = plt.subplots(figsize=(10.4, 4.3))
    t = np.arange(len(q)) / 12
    ax.plot(t[:AJUSTE], q.iloc[:AJUSTE] / 1000, color=RED, lw=2.2)
    ax.add_patch(Rectangle((AJUSTE / 12, 0), 12 - AJUSTE / 12, q.max() / 1000 * 1.1,
                           fc=LGRAY, alpha=.75, ec="none"))
    ax.text((AJUSTE / 12 + 12) / 2, q.max() / 1000 * 0.55,
            "esto todavía\nno pasó", ha="center", va="center", fontsize=13,
            color=MUTED, fontweight="bold", linespacing=1.3)
    ax.axvline(AJUSTE / 12, color=DARK, lw=1.8, ls="--")
    ax.text(AJUSTE / 12 - 0.2, q.max() / 1000 * 1.02, "hoy", ha="right",
            fontsize=11, fontweight="bold", color=DARK)
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, q.max() / 1000 * 1.12)
    ax.set_xlabel("años desde el pico", fontsize=9.5)
    ax.set_ylabel("producción\n[miles de bbl/día]", fontsize=9.5)
    ax.set_title(f"Campo {EJEMPLO.title()}: cinco años de historia. "
                 "¿Cuánto produce en los próximos siete?",
                 fontsize=12.5, fontweight="bold", loc="left", pad=8)
    guardar(fig, "fig_el_encargo.png")


def fig_que_es_arps(campos):
    """Concepto puro: la exponencial es una recta cuando se dobla el eje."""
    q = campos[EJEMPLO].iloc[:AJUSTE]
    t = np.arange(len(q))
    b = np.polyfit(t, np.log(q.values), 1)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.9))
    for ax, log in zip(axes, [False, True]):
        ax.plot(t / 12, q / 1000, "o", ms=2.6, color=GRAY, label="lo que midió el campo")
        ax.plot(t / 12, np.exp(np.polyval(b, t)) / 1000, color=RED, lw=2.4,
                label="la recta de Arps")
        if log:
            ax.set_yscale("log")
            ax.set_title("En escala logarítmica: es una RECTA", fontsize=11.5,
                         fontweight="bold", loc="left", color=RED)
        else:
            ax.set_title("En escala normal: es una curva que cae", fontsize=11.5,
                         fontweight="bold", loc="left")
        ax.set_xlabel("años desde el pico", fontsize=9.5)
        ax.set_ylabel("producción [miles de bbl/día]", fontsize=9.5)
    axes[0].legend(fontsize=9)
    fig.text(0.5, -0.03,
             "por eso «declinación exponencial» y «recta sobre el logaritmo» "
             "son la misma frase — la de la Clase 1, ahora con nombre",
             ha="center", fontsize=10, color=DARK, style="italic")
    fig.tight_layout(w_pad=2.4)
    guardar(fig, "fig_que_es_arps.png")


def fig_doblar(campos):
    """La version estatica de la animacion del cuaderno: se dobla el eje y se
    mira lo que le sobra a la recta."""
    y = campos[EJEMPLO].iloc[:AJUSTE].values
    t = np.arange(len(y))

    def doblar(y, lam):
        return np.log(y) if abs(lam) < 1e-9 else (y ** lam - 1) / lam

    fig, axes = plt.subplots(2, 3, figsize=(11.6, 4.6), sharex=True,
                             gridspec_kw=dict(height_ratios=[2, 1]))
    for j, (lam, nom) in enumerate([(1.0, "escala normal"),
                                    (0.5, "a medio doblar"),
                                    (0.0, "escala logarítmica")]):
        z = doblar(y, lam)
        z = (z - z.mean()) / z.std()
        tt = np.linspace(-1, 1, len(z))
        recta = np.polyval(np.polyfit(tt, z, 1), tt)
        resto = z - recta
        arco = np.polyval(np.polyfit(tt, resto, 2), tt)
        curv = abs(np.polyfit(tt, resto, 2)[0])

        a = axes[0, j]
        a.plot(t / 12, z, "o", ms=2.6, color=GRAY)
        a.plot(t / 12, recta, color=RED, lw=2.2)
        a.set_ylim(-2.6, 2.6)
        a.set_title(f"λ = {lam:.1f}   ·   {nom}", fontsize=10.5,
                    fontweight="bold", loc="left", color=DARK)
        if j == 0:
            a.set_ylabel("producción\n(eje doblado)", fontsize=9)

        b = axes[1, j]
        b.axhline(0, color=INK, lw=1.2)
        b.plot(t / 12, resto, "o", ms=2.2, color=GRAY)
        b.fill_between(t / 12, 0, arco, color=RED, alpha=.35)
        b.plot(t / 12, arco, color=RED, lw=1.8)
        b.set_ylim(-1.5, 1.5)
        b.set_xlabel("años desde el pico", fontsize=9)
        b.text(0.97, 0.08, f"{curv:.2f}", transform=b.transAxes, ha="right",
               fontsize=13, fontweight="bold",
               color=GREEN if curv < 0.2 else RED)
        if j == 0:
            b.set_ylabel("lo que le sobra\na la recta", fontsize=9)
    fig.suptitle("Doblando el eje: la panza del residuo se aplana",
                 fontsize=12.5, fontweight="bold", color=DARK, x=0.005, ha="left")
    fig.tight_layout(h_pad=0.5, w_pad=1.6)
    guardar(fig, "fig_doblar.png")


def fig_que_es_b():
    """Que hace el exponente b: la cola."""
    t = np.arange(0, 12 * 12)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.9))
    cols = [INK, BLUE, GREEN]
    for ax, log in zip(axes, [False, True]):
        for (b, c, et) in zip([0.001, 0.5, 1.0], cols,
                              ["b = 0   (exponencial)", "b = 0,5  (lo habitual)",
                               "b = 1   (armónica)"]):
            ax.plot(t / 12, hiperbolica(t, 100, 0.02, b), color=c, lw=2.2, label=et)
        if log:
            ax.set_yscale("log")
            ax.set_title("En logaritmo: solo b = 0 es recta", fontsize=11.5,
                         fontweight="bold", loc="left")
        else:
            ax.set_title("Todas arrancan igual; se separan en la cola",
                         fontsize=11.5, fontweight="bold", loc="left")
        ax.set_xlabel("años", fontsize=9.5)
        ax.set_ylabel("producción [% del inicio]", fontsize=9.5)
    axes[0].legend(fontsize=9)
    fig.text(0.5, -0.03, "b mide cuánta cola tiene el campo. Y la cola es "
             "justamente lo que uno no ve en los primeros cinco años.",
             ha="center", fontsize=10, color=DARK, style="italic")
    fig.tight_layout(w_pad=2.4)
    guardar(fig, "fig_que_es_b.png")


def fig_ajuste_comparado(campos):
    q = campos[EJEMPLO]
    a = ajustar(q)
    n = OBJETIVO
    fig, ax = plt.subplots(figsize=(10.4, 4.4))
    t = np.arange(n) / 12
    ax.axvspan(0, AJUSTE / 12, color="#EEF2FF", zorder=0)
    ax.plot(np.arange(len(q[:n])) / 12, q.iloc[:n] / 1000, color=INK, lw=1.8,
            label="lo que de verdad produjo")
    ax.plot(t, curva_exp(a["exp_coef"], n) / 1000, color=RED, lw=2.2, ls="--",
            label="Arps exponencial")
    ax.plot(t, curva_hip(a["hip_par"], n) / 1000, color=GREEN, lw=2.2, ls="-.",
            label=f"Arps hiperbólica  (b = {a['b']:.2f})")
    ax.axvline(AJUSTE / 12, color=DARK, lw=1.6, ls=":")
    ax.text(AJUSTE / 24, ax.get_ylim()[1] * .93, "con esto se ajustó",
            ha="center", fontsize=9.5, color=MUTED, style="italic")
    ax.set_xlabel("años desde el pico", fontsize=9.5)
    ax.set_ylabel("producción\n[miles de bbl/día]", fontsize=9.5)
    ax.legend(fontsize=9.5)
    ax.set_title(f"Campo {EJEMPLO.title()}: las dos curvas, y lo que pasó",
                 fontsize=12.5, fontweight="bold", loc="left", pad=8)
    guardar(fig, "fig_ajuste_comparado.png")


def fig_sesgo(r):
    fig, ax = plt.subplots(figsize=(10.2, 4.3))
    bins = np.arange(-60, 61, 6)
    ax.hist(r.e_exp, bins=bins, color=RED, alpha=.62, label="Arps exponencial")
    ax.hist(r.e_hip, bins=bins, color=GREEN, alpha=.62, label="Arps hiperbólica")
    ax.axvline(0, color=INK, lw=2)
    ax.axvline(r.e_exp.median(), color=RED, lw=2.4, ls="--")
    ax.axvline(r.e_hip.median(), color=GREEN, lw=2.4, ls="--")
    ax.annotate(f"la exponencial se queda\ncorta {abs(r.e_exp.median()):.0f} % "
                "en la mitad de los campos",
                xy=(r.e_exp.median(), ax.get_ylim()[1] * .72),
                xytext=(-52, ax.get_ylim()[1] * .78), fontsize=9.5, color=RED,
                fontweight="bold", linespacing=1.3,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.text(0.6, ax.get_ylim()[1] * .93, "acertar\nes acá", fontsize=9.5,
            color=INK, fontweight="bold", linespacing=1.2)
    ax.set_xlabel("error del acumulado a 12 años  [%]   —   negativo = predijo de menos",
                  fontsize=9.5)
    ax.set_ylabel("cantidad de campos", fontsize=9.5)
    ax.legend(fontsize=10)
    ax.set_title("El error no está centrado: la exponencial subestima siempre",
                 fontsize=12.5, fontweight="bold", loc="left", pad=8)
    guardar(fig, "fig_sesgo.png")


def fig_analogos(r, campos):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.0),
                             gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axes[0]
    for c, q in campos.items():
        n = min(len(q), OBJETIVO)
        ax.plot(np.arange(n) / 12, 100 * q.iloc[:n] / q.iloc[:6].mean(),
                color=GRAY, lw=0.7, alpha=.45)
    q = campos[EJEMPLO]
    ax.plot(np.arange(OBJETIVO) / 12, 100 * q.iloc[:OBJETIVO] / q.iloc[:6].mean(),
            color=RED, lw=2.4)
    ax.set_yscale("log")
    ax.set_xlabel("años desde el pico", fontsize=9.5)
    ax.set_ylabel("% de su propio pico", fontsize=9.5)
    ax.set_title(f"Los {len(campos)} campos, y el nuestro en rojo",
                 fontsize=11.5, fontweight="bold", loc="left")

    ax = axes[1]
    ax.hist(r.k, bins=14, color=BLUE, alpha=.72)
    for qq, c, et in [(0.10, ORANGE, "P10"), (0.50, DARK, "P50"), (0.90, ORANGE, "P90")]:
        v = r.k.quantile(qq)
        ax.axvline(v, color=c, lw=2.2, ls="--" if qq != .5 else "-")
        ax.text(v, ax.get_ylim()[1] * (.97 if qq != .5 else .86), f" {et}\n {v:.1f}×",
                color=c, fontsize=9.5, fontweight="bold", va="top", linespacing=1.2)
    ax.set_xlabel("cuántas veces el acumulado de los primeros 5 años\n"
                  "terminó siendo el acumulado a 12 años", fontsize=9.5)
    ax.set_ylabel("cantidad de campos", fontsize=9.5)
    ax.set_title("Lo que dice la experiencia ajena", fontsize=11.5,
                 fontweight="bold", loc="left")
    fig.tight_layout(w_pad=2.4)
    guardar(fig, "fig_analogos.png")


def fig_banda(r, campos):
    row = r[r.campo == EJEMPLO].iloc[0]
    o = r[r.campo != EJEMPLO]
    lo, med, hi = (row.acum5 * o.k.quantile(q) for q in (0.10, 0.50, 0.90))
    fig, ax = plt.subplots(figsize=(9.6, 4.3))
    ax.grid(axis="x")
    y = [0, 1, 2, 3]
    vals = [row.exp, row.hip, med, row.real]
    cols = [RED, GREEN, BLUE, INK]
    nom = ["Arps exponencial", "Arps hiperbólica", "Análogos (P50)",
           "LO QUE DE VERDAD PASÓ"]
    ax.barh(y[:3], [v / 1e6 for v in vals[:3]], color=cols[:3], height=.55)
    ax.plot([vals[3] / 1e6], [3], "D", ms=13, color=INK)
    ax.hlines(2, lo / 1e6, hi / 1e6, color=BLUE, lw=3.4, alpha=.45)
    ax.plot([lo / 1e6, hi / 1e6], [2, 2], "|", ms=16, color=BLUE)
    ax.text((lo + hi) / 2e6, 2.42, f"banda P10–P90: {lo/1e6:.0f} a {hi/1e6:.0f} MMbbl",
            ha="center", fontsize=9.5, color=BLUE, fontweight="bold")
    ax.axvline(vals[3] / 1e6, color=INK, lw=1.4, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels(nom, fontsize=10)
    ax.set_xlabel("acumulado de petróleo en 12 años  [millones de barriles]", fontsize=9.5)
    ax.set_title(f"Campo {EJEMPLO.title()}: las tres respuestas y el resultado",
                 fontsize=12.5, fontweight="bold", loc="left", pad=8)
    guardar(fig, "fig_banda.png")


def fig_calibracion(r):
    niv, real = calibracion(r)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot([0, 100], [0, 100], color=GRAY, lw=1.6, ls="--",
            label="una banda perfectamente honesta")
    ax.plot(niv, real, "o-", color=RED, lw=2.6, ms=6,
            label="nuestra banda de análogos")
    i = int(np.argmin(np.abs(niv - 80)))
    ax.plot([niv[i]], [real[i]], "o", ms=13, mfc="none", mec=DARK, mew=2.4)
    ax.annotate(f"a nivel 80 % contuvo\nla realidad el {real[i]:.0f} % de las veces",
                xy=(niv[i], real[i]), xytext=(28, 88), fontsize=10, color=DARK,
                fontweight="bold", linespacing=1.3,
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1.4))
    ax.set_xlabel("lo que la banda PROMETE  [%]", fontsize=9.5)
    ax.set_ylabel("lo que la banda CUMPLE  [%]", fontsize=9.5)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_title("La única pregunta que importa de un intervalo: ¿cumple lo que promete?",
                 fontsize=12, fontweight="bold", loc="left", pad=8)
    guardar(fig, "fig_calibracion.png")


def fig_limite_economico(campos):
    """Cuando deja de pagar el campo, y como la curva elegida mueve esa fecha."""
    q = campos[EJEMPLO]
    a = ajustar(q)
    N = 60 * 12
    ce = curva_exp(a["exp_coef"], N)
    ch = curva_hip(a["hip_par"], N)
    ct = curva_hip_terminal(a["hip_par"], N)
    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    t = np.arange(N) / 12
    ax.plot(t, ce, color=RED, lw=2.2, ls="--", label="Arps exponencial")
    ax.plot(t, ch, color=GRAY, lw=2.0, ls=":", label="hiperbólica sin freno")
    ax.plot(t, ct, color=GREEN, lw=2.4, label="hiperbólica + declinación terminal")
    ax.axhline(Q_LIMITE, color=INK, lw=2)
    ax.axhspan(0, Q_LIMITE, color="#FEE2E2", alpha=.8, zorder=0)
    ax.text(0.4, Q_LIMITE * 0.45, "acá el campo PIERDE plata", fontsize=10,
            color=RED, fontweight="bold")
    ax.text(0.4, Q_LIMITE * 1.12, f"límite económico: {Q_LIMITE:,.0f} bbl/día"
            .replace(",", "."), fontsize=9.5, color=INK, fontweight="bold")
    for c, cur, nom in [(RED, ce, "exp"), (GREEN, ct, "terminal")]:
        m = mes_de_abandono(cur)
        if m:
            ax.plot([m / 12], [Q_LIMITE], "v", ms=11, color=c, zorder=5)
            ax.text(m / 12, Q_LIMITE * 1.7, f"{m/12:.0f} años", ha="center",
                    fontsize=10, color=c, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(Q_LIMITE * 0.25, q.max() * 1.4)
    ax.set_xlim(0, 60)
    ax.set_xlabel("años desde el pico", fontsize=9.5)
    ax.set_ylabel("producción  [bbl/día]", fontsize=9.5)
    ax.legend(fontsize=9.5, loc="upper right")
    ax.set_title("La curva que se elige decide en qué año se cierra el campo",
                 fontsize=12.5, fontweight="bold", loc="left", pad=8)
    guardar(fig, "fig_limite_economico.png")


def fig_dolares(r):
    """El sesgo del -7 %, traducido al idioma de gerencia."""
    r = r.copy()
    r["musd"] = (r.exp - r.real) * MARGEN / 1e6
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.0),
                             gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axes[0]
    ax.hist(r.musd, bins=16, color=RED, alpha=.72)
    ax.axvline(0, color=INK, lw=2)
    ax.axvline(r.musd.median(), color=DARK, lw=2.4, ls="--")
    ax.text(r.musd.median(), ax.get_ylim()[1] * .88,
            f" mediana:\n USD {r.musd.median():,.0f} M".replace(",", "."),
            fontsize=10, color=DARK, fontweight="bold", linespacing=1.25)
    ax.set_xlabel("valor mal declarado por campo  [millones de USD]", fontsize=9.5)
    ax.set_ylabel("cantidad de campos", fontsize=9.5)
    ax.set_title("Lo que cuesta el sesgo, campo por campo", fontsize=11.5,
                 fontweight="bold", loc="left")

    ax = axes[1]
    ax.set_axis_off(); ax.grid(False)
    ax.text(0.5, 0.86, "En los 54 campos juntos", ha="center", fontsize=11.5,
            fontweight="bold", color=DARK)
    ax.text(0.5, 0.50, f"USD {abs(r.musd.sum())/1000:,.1f}".replace(",", ",") +
            "\nmil millones", ha="center", va="center", fontsize=30,
            fontweight="bold", color=RED, linespacing=1.05)
    ax.text(0.5, 0.16, "de reservas subvaluadas por usar\nla curva equivocada",
            ha="center", fontsize=10.5, color=MUTED, linespacing=1.35)
    fig.tight_layout(w_pad=2.0)
    guardar(fig, "fig_dolares.png")


def fig_marcador(r):
    fig, ax = plt.subplots(figsize=(10.4, 4.4))
    met = [("Arps\nexponencial", r.e_exp, RED),
           ("Arps\nhiperbólica", r.e_hip, GREEN),
           ("Análogos\n(P50)", r.e_an, BLUE)]
    x = np.arange(len(met))
    ax.bar(x - .19, [m[1].abs().median() for m in met], .36, color=[m[2] for m in met],
           label="error típico")
    ax.bar(x + .19, [abs(m[1].median()) for m in met], .36,
           color=[m[2] for m in met], alpha=.38, label="sesgo (error sistemático)")
    for i, m in enumerate(met):
        ax.text(i - .19, m[1].abs().median() + .35, f"{m[1].abs().median():.0f} %",
                ha="center", fontsize=10.5, fontweight="bold", color=DARK)
        ax.text(i + .19, abs(m[1].median()) + .35, f"{m[1].median():+.0f} %",
                ha="center", fontsize=10.5, fontweight="bold", color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels([m[0] for m in met], fontsize=10)
    ax.set_ylabel("error del acumulado a 12 años  [%]", fontsize=9.5)
    ax.set_ylim(0, max(m[1].abs().median() for m in met) * 1.45)
    ax.legend(fontsize=9.5, ncol=2)
    ax.set_title("El marcador: ninguno acierta el número, y eso es el resultado",
                 fontsize=12.5, fontweight="bold", loc="left", pad=8)
    guardar(fig, "fig_marcador.png")


# =============================================================== MAIN =======
def main():
    print("leyendo datos ...")
    d, campos, reales = cargar()
    print(f"  {len(campos)} campos, {len(d)} filas")

    print("\najustando Arps a cada campo ...")
    r = resultados(campos, reales)
    print(f"  {len(r)} campos evaluables")

    print("\ngenerando figuras ...")
    fig_que_es_declinacion(campos)
    fig_el_encargo(campos)
    fig_que_es_arps(campos)
    fig_doblar(campos)
    fig_que_es_b()
    fig_ajuste_comparado(campos)
    fig_sesgo(r)
    fig_analogos(r, campos)
    fig_banda(r, campos)
    fig_calibracion(r)
    fig_marcador(r)
    fig_limite_economico(campos)
    fig_dolares(r)

    a = ajustar(campos[EJEMPLO])
    row = r[r.campo == EJEMPLO].iloc[0]
    o = r[r.campo != EJEMPLO]
    niv, cal = calibracion(r)
    i80 = int(np.argmin(np.abs(niv - 80)))

    print("\n" + "=" * 74)
    print("NUMEROS PARA LAS LAMINAS  (no escribir ninguno a mano)")
    print("=" * 74)
    print(f"campos {len(campos)} | filas {len(d)} | evaluables {len(r)}")
    print(f"anios de historia despues del pico: mediana "
          f"{d.groupby('campo').size().median()/12:.0f}, maximo "
          f"{d.groupby('campo').size().max()/12:.0f}")
    print(f"\najuste con {AJUSTE//12} anios, se predice el acumulado a {OBJETIVO//12} anios\n")
    print(f"{'metodo':>22} {'|error| mediano':>16} {'sesgo':>8} {'peor':>7} {'>50%':>7}")
    for nom, col in [("Arps exponencial", "e_exp"), ("Arps hiperbolica", "e_hip"),
                     ("Analogos (P50)", "e_an")]:
        print(f"{nom:>22} {r[col].abs().median():14.0f} % {r[col].median():+7.0f} % "
              f"{r[col].abs().max():6.0f} % {(r[col].abs()>50).sum():4d}/{len(r)}")
    print(f"\nexponente b de la hiperbolica: mediana {r.b.median():.2f} | "
          f"campos con b>0.1: {(r.b>0.1).sum()} de {len(r)}")
    print(f"declinacion anual D: mediana {r.D_anual.median():.0f} %/anio")
    print(f"\nmultiplicador k (acum 12a / acum 5a): P10 {r.k.quantile(.1):.2f}x  "
          f"P50 {r.k.quantile(.5):.2f}x  P90 {r.k.quantile(.9):.2f}x")
    N = 60 * 12
    ab = []
    for c, q in campos.items():
        a2 = ajustar(q)
        if a2 is None:
            continue
        ab.append((mes_de_abandono(curva_exp(a2["exp_coef"], N)),
                   mes_de_abandono(curva_hip(a2["hip_par"], N)),
                   mes_de_abandono(curva_hip_terminal(a2["hip_par"], N))))
    ab = pd.DataFrame(ab, columns=["exp", "hip", "ter"])
    print(f"\nECONOMIA (precio {PRECIO:.0f}, opex {OPEX:.0f}, fijo "
          f"{COSTO_FIJO/1e6:.0f} MUSD/mes) -> limite {Q_LIMITE:,.0f} bbl/dia"
          .replace(",", "."))
    print(f"  campos que la hiperbolica SIN FRENO dice que nunca mueren: "
          f"{ab.hip.isna().sum()} de {len(ab)}")
    print(f"  con declinacion terminal ({100*D_MIN:.0f} %/anio): "
          f"{ab.ter.isna().sum()} de {len(ab)}")
    print(f"  abandono mediano: exponencial {ab.exp.median()/12:.0f} anios | "
          f"con terminal {ab.ter.median()/12:.0f} anios")
    musd = (r.exp - r.real) * MARGEN / 1e6
    print(f"  el sesgo en dinero: mediana USD {musd.median():,.0f} M por campo | "
          f"USD {abs(musd.sum())/1000:,.1f} mil millones en los 54".replace(",", "."))
    print(f"\nBANDA P10-P90: contuvo la realidad en {r.en_banda.sum()}/{len(r)} campos "
          f"({100*r.en_banda.mean():.0f} %)   <-- promete 80 %")
    print(f"calibracion al nivel 80: {cal[i80]:.0f} %")
    print(f"\nel campo de la clase ({EJEMPLO}):")
    print(f"    D anual {a['D_anual']:.0f} %/anio | b {a['b']:.2f}")
    print(f"    real                 {row.real/1e6:7.1f} MMbbl")
    print(f"    Arps exponencial     {row.exp/1e6:7.1f} MMbbl  ({row.e_exp:+.0f} %)")
    print(f"    Arps hiperbolica     {row.hip/1e6:7.1f} MMbbl  ({row.e_hip:+.0f} %)")
    print(f"    Analogos P50         {row.acum5*o.k.median()/1e6:7.1f} MMbbl  "
          f"({row.e_an:+.0f} %)")
    print(f"    banda P10-P90        {row.acum5*o.k.quantile(.1)/1e6:.0f} a "
          f"{row.acum5*o.k.quantile(.9)/1e6:.0f} MMbbl  "
          f"-> {'CONTIENE' if row.en_banda else 'NO contiene'} la realidad")


if __name__ == "__main__":
    main()

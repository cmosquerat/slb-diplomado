"""
Machine Learning for Petroleum Engineers Using Python
Modulo 7 - Clase 1: genera todas las figuras de la presentacion.

No necesita datos externos: usa los datasets que trae seaborn (se bajan
solos la primera vez) y las cifras publicas de adopcion de IA en 2026, que
estan declaradas abajo con su fuente.

Al final imprime TODAS las cifras que aparecen en las laminas: ninguna se
escribe a mano, y el cuaderno de la clase tiene que reproducirlas.

Uso:  python3 figuras.py
"""

import os
import glob
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

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


FUENTE = _fuente()
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": [FUENTE], "font.size": 11,
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


# ------------------------------------------------------- cifras publicas ----
# Todas de 2026. La fuente va en la lamina de referencias del deck.
ADOPCION = 93          # % de desarrolladores que usa alguna herramienta de IA
AGENTES = 55           # % que usa AGENTES con regularidad
AGENTES_SENIOR = 63.5  # % entre ingenieros staff+ (los mas senior)
CONFIA_PLENO = 3       # % que confia plenamente en la salida
DESCONFIA = 46         # % que desconfia de su precision
CONFIA = 33            # % que confia
CASI_CORRECTO = 66     # % cuya mayor frustracion es "casi correcto, pero no"
DEPURAR_MAS = 45       # % que dice que depurar IA tarda mas que escribir
GANANCIA_MIN = 30      # % mas rapido en tareas acotadas (rango bajo)
GANANCIA_MAX = 55      # % mas rapido en tareas acotadas (rango alto)
METR_REAL = -19        # % de velocidad REAL en el ensayo controlado de METR
METR_PERCIBIDA = +20   # % de velocidad PERCIBIDA por esos mismos ingenieros
APPS_CON_AGENTES = 40  # % de apps empresariales con agentes a fin de 2026
APPS_ANTES = 5         # % el anio anterior


def cargar():
    """Los tres datasets de la clase. seaborn los baja y los cachea."""
    taxis = sns.load_dataset("taxis")
    taxis["pickup"] = pd.to_datetime(taxis.pickup)
    taxis["dropoff"] = pd.to_datetime(taxis.dropoff)
    taxis["hora"] = taxis.pickup.dt.hour
    taxis["dia"] = taxis.pickup.dt.dayofweek
    taxis["minutos"] = (taxis.dropoff - taxis.pickup).dt.total_seconds() / 60
    return taxis, sns.load_dataset("healthexp"), sns.load_dataset("flights")


# =========================================================== FIGURAS ========
def fig_adopcion():
    """Esto ya paso: la adopcion en 2026, y quien la lidera."""
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.0),
                             gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axes[0]
    ax.grid(axis="y")
    etiquetas = ["Usa alguna\nherramienta de IA", "Usa AGENTES\ncon regularidad",
                 "Agentes entre los\ningenieros senior"]
    vals = [ADOPCION, AGENTES, AGENTES_SENIOR]
    cols = [BLUE, GREEN, DARK]
    ax.bar(range(3), vals, 0.55, color=cols, alpha=.9)
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"{v:.0f} %", ha="center", fontsize=15,
                fontweight="bold", color=cols[i])
    ax.set_xticks(range(3))
    ax.set_xticklabels(etiquetas, fontsize=9.5)
    ax.set_ylim(0, 108)
    ax.set_ylabel("% de desarrolladores", fontsize=9.5)
    ax.set_title("Dónde estamos en 2026", fontsize=12, fontweight="bold",
                 loc="left")

    ax = axes[1]
    ax.set_axis_off(); ax.grid(False)
    ax.text(0.5, 0.93, "Y el dato que sorprende", ha="center", fontsize=12,
            fontweight="bold", color=DARK)
    ax.text(0.5, 0.60,
            "Los ingenieros\nMÁS SENIOR\nson los que más\nusan agentes",
            ha="center", va="center", fontsize=15, fontweight="bold",
            color=GREEN, linespacing=1.35)
    ax.text(0.5, 0.16,
            "No es una herramienta para principiantes.\n"
            "Es una herramienta que rinde más\ncuanto más criterio tiene quien la usa.",
            ha="center", va="center", fontsize=10, color=MUTED, linespacing=1.4,
            style="italic")
    fig.tight_layout(w_pad=2.0)
    guardar(fig, "fig_adopcion.png")


def fig_cuello_botella():
    """El cuello de botella se movio: de escribir a verificar."""
    fig, axes = plt.subplots(2, 1, figsize=(10.4, 4.4))
    etapas = ["Entender\nel problema", "Escribir\nel código",
              "Verificar\nque sirve", "Usarlo para\ndecidir"]

    for ax, titulo, anchos, cuello in [
            (axes[0], "ANTES", [1.0, 3.2, 0.9, 0.9], 1),
            (axes[1], "AHORA", [1.0, 0.5, 2.6, 0.9], 2)]:
        ax.set_axis_off(); ax.grid(False)
        x = 0.0
        total = sum(anchos)
        for i, (et, w) in enumerate(zip(etapas, anchos)):
            es_cuello = (i == cuello)
            col = RED if es_cuello else LGRAY
            ax.add_patch(FancyBboxPatch((x, 0.25), w - 0.06, 0.5,
                                        boxstyle="round,pad=0.02",
                                        fc=col, ec="none",
                                        alpha=1.0 if es_cuello else .85))
            ax.text(x + (w - 0.06) / 2, 0.5, et, ha="center", va="center",
                    fontsize=9.5, color="white" if es_cuello else INK,
                    fontweight="bold" if es_cuello else "normal",
                    linespacing=1.2)
            x += w
        ax.set_xlim(-0.15, total + 0.05)
        ax.set_ylim(0, 1)
        ax.text(-0.12, 0.5, titulo, ha="right", va="center", fontsize=11,
                fontweight="bold", color=DARK)
    fig.text(0.5, -0.02,
             "El ancho es el tiempo. Lo rojo es donde se atasca el trabajo.",
             ha="center", fontsize=10.5, color=DARK, style="italic")
    fig.tight_layout(h_pad=1.2)
    guardar(fig, "fig_cuello_botella.png")


def fig_fea_vs_bonita(taxis):
    """La primera victoria: los MISMOS datos, dos veces."""
    d = taxis.groupby("hora").size()

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))

    # ---- izquierda: lo que sale por defecto ----
    ax = axes[0]
    ax.plot(d.index, d.values, color="#1f77b4")
    ax.grid(False)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.set_title("Lo que sale por defecto", fontsize=12, fontweight="bold",
                 loc="left", color=MUTED)

    # ---- derecha: la misma informacion, comunicando ----
    ax = axes[1]
    ax.grid(axis="y")
    pico = int(d.idxmax())
    colores = [RED if h == pico else LGRAY for h in d.index]
    ax.bar(d.index, d.values, color=colores, width=0.82)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([f"{h}h" for h in range(0, 24, 3)], fontsize=9.5)
    ax.set_xlabel("hora del día", fontsize=9.5)
    ax.set_ylabel("viajes", fontsize=9.5)
    ax.annotate(f"la hora pico es a las {pico}h\n({d.max()} viajes)",
                xy=(pico, d.max()), xytext=(pico - 9.5, d.max() * 0.94),
                fontsize=10.5, color=RED, fontweight="bold", linespacing=1.3,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.set_title("La misma información, comunicando", fontsize=12,
                 fontweight="bold", loc="left", color=DARK)
    fig.suptitle("Los mismos datos, los mismos números. Solo cambia si se entiende.",
                 fontsize=12.5, fontweight="bold", color=DARK, x=0.005, ha="left")
    fig.tight_layout(w_pad=2.4)
    guardar(fig, "fig_fea_vs_bonita.png")
    return pico, int(d.max())


def fig_tres_dominios(taxis, health, flights):
    """La misma tecnica sirve en cualquier dominio."""
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7))

    # transporte
    ax = axes[0]
    ax.grid(axis="y")
    d = taxis.groupby("hora").size()
    ax.fill_between(d.index, d.values, color=BLUE, alpha=.75)
    ax.set_xlabel("hora del día", fontsize=9)
    ax.set_ylabel("viajes", fontsize=9)
    ax.set_title("TRANSPORTE\n¿a qué hora hay demanda?", fontsize=10.5,
                 fontweight="bold", loc="left", color=BLUE)

    # presupuesto / gestion
    ax = axes[1]
    ult = health[health.Year == health.Year.max()]
    ax.scatter(ult.Spending_USD / 1000, ult.Life_Expectancy, s=60,
               color=GREEN, alpha=.85)
    for _, r in ult.iterrows():
        ax.annotate(r.Country[:3].upper(), (r.Spending_USD / 1000,
                    r.Life_Expectancy), fontsize=7, color=MUTED,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("gasto por persona [miles USD]", fontsize=9)
    ax.set_ylabel("expectativa de vida [años]", fontsize=9)
    ax.set_title("GESTIÓN\n¿el gasto se traduce en resultado?", fontsize=10.5,
                 fontweight="bold", loc="left", color=GREEN)

    # planeacion de demanda
    ax = axes[2]
    piv = flights.pivot(index="month", columns="year", values="passengers")
    ax.plot(piv.columns, piv.sum(), color=ORANGE, lw=2.4)
    ax.set_xlabel("año", fontsize=9)
    ax.set_ylabel("pasajeros al año", fontsize=9)
    ax.set_title("PLANEACIÓN\n¿cuánto vamos a necesitar?", fontsize=10.5,
                 fontweight="bold", loc="left", color=ORANGE)

    fig.suptitle("La misma habilidad, tres problemas que no se parecen en nada",
                 fontsize=12.5, fontweight="bold", color=DARK, x=0.005, ha="left")
    fig.tight_layout(w_pad=2.0)
    guardar(fig, "fig_tres_dominios.png")


def fig_escalera_prompt(taxis):
    """Los tres niveles de especificacion, y lo que sale de cada uno."""
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9))
    d = taxis.groupby("hora").size()

    # nivel 1: "hazme un grafico"
    ax = axes[0]
    ax.plot(taxis.distance.values[:300], color="#1f77b4")
    ax.grid(False)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.set_title("«hazme un gráfico»", fontsize=11, fontweight="bold",
                 loc="left", color=MUTED)

    # nivel 2: dice que variable
    ax = axes[1]
    ax.grid(axis="y")
    ax.bar(d.index, d.values, color=GRAY, width=0.82)
    ax.set_xlabel("hora", fontsize=9)
    ax.set_ylabel("viajes", fontsize=9)
    ax.set_title("«grafica los viajes por hora»", fontsize=11,
                 fontweight="bold", loc="left", color=AMBER)

    # nivel 3: objetivo, contexto, restricciones y criterio de aceptacion
    ax = axes[2]
    ax.grid(axis="y")
    pico = int(d.idxmax())
    ax.bar(d.index, d.values, color=[RED if h == pico else LGRAY for h in d.index],
           width=0.82)
    ax.axhline(d.mean(), color=INK, ls="--", lw=1.3)
    ax.text(0.4, d.mean() * 1.06, "promedio", fontsize=8, color=INK)
    ax.annotate(f"pico {pico}h", xy=(pico, d.max()),
                xytext=(pico - 8.5, d.max() * 0.88), fontsize=9.5, color=RED,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    ax.set_xticks(range(0, 24, 4))
    ax.set_xlabel("hora del día", fontsize=9)
    ax.set_ylabel("viajes", fontsize=9)
    ax.set_title("«...marcando el pico y el promedio»", fontsize=11,
                 fontweight="bold", loc="left", color=GREEN)

    fig.suptitle("La calidad de la salida es la calidad de la especificación",
                 fontsize=12.5, fontweight="bold", color=DARK, x=0.005, ha="left")
    fig.tight_layout(w_pad=2.0)
    guardar(fig, "fig_escalera_prompt.png")


def fig_mapa_calor(taxis):
    """El ejercicio guiado: cuando conviene tener gente en la calle."""
    dias = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
    piv = (taxis.groupby(["dia", "hora"]).size()
           .unstack(fill_value=0).reindex(range(7), fill_value=0))
    fig, ax = plt.subplots(figsize=(11.0, 3.6))
    ax.grid(False)
    sns.heatmap(piv, cmap="rocket_r", ax=ax, cbar_kws={"label": "viajes"},
                linewidths=.4, linecolor="white")
    ax.set_yticklabels(dias, rotation=0, fontsize=9.5)
    ax.set_xlabel("hora del día", fontsize=9.5)
    ax.set_ylabel("")
    ax.set_title("¿Cuándo conviene tener gente en la calle?",
                 fontsize=12.5, fontweight="bold", loc="left", pad=10)
    guardar(fig, "fig_mapa_calor.png")
    d, h = np.unravel_index(np.argmax(piv.values), piv.shape)
    return dias[d], int(piv.columns[h]), int(piv.values.max())


def fig_trampa(taxis):
    """LA figura de la clase: un analisis impecable sobre datos que mienten."""
    g = taxis.groupby("payment").tip.agg(["mean", "size"])
    g = g.reindex(["cash", "credit card"])
    nombres = ["Efectivo", "Tarjeta"]
    cero = taxis[taxis.payment == "cash"]
    n_cero = int((cero.tip > 0).sum())

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.1),
                             gridspec_kw=dict(width_ratios=[1, 1.12]))
    ax = axes[0]
    ax.grid(axis="y")
    ax.bar(nombres, g["mean"].values, 0.5, color=[GRAY, GREEN], alpha=.9)
    for i, v in enumerate(g["mean"].values):
        ax.text(i, v + 0.08, f"USD {v:.2f}", ha="center", fontsize=13,
                fontweight="bold", color=DARK)
    ax.set_ylabel("propina promedio [USD]", fontsize=9.5)
    ax.set_ylim(0, g["mean"].max() * 1.28)
    ax.set_title("Lo que el agente va a reportar", fontsize=11.5,
                 fontweight="bold", loc="left")
    ax.text(0.5, -0.30, "«Quien paga en efectivo nunca deja propina»",
            transform=ax.transAxes, ha="center", fontsize=11,
            color=RED, fontweight="bold", style="italic")

    ax = axes[1]
    ax.set_axis_off(); ax.grid(False)
    ax.text(0.5, 0.95, "Y es falso", ha="center", fontsize=13,
            fontweight="bold", color=RED)
    ax.text(0.5, 0.55,
            f"De los {int(g.loc['cash','size'])} viajes en efectivo,\n"
            f"exactamente {n_cero} tienen propina.\n\n"
            "No es que no la dejen: el taxímetro solo\n"
            "registra la propina cuando va en la tarjeta.\n"
            "La de efectivo va al bolsillo del conductor\n"
            "y nunca entra al sistema.",
            ha="center", va="center", fontsize=10.5, color=INK, linespacing=1.5)
    ax.text(0.5, 0.08, "Un cero que no es un cero: es una AUSENCIA",
            ha="center", fontsize=11.5, color=DARK, fontweight="bold")
    fig.tight_layout(w_pad=2.0)
    guardar(fig, "fig_trampa.png")
    return int(g.loc["cash", "size"]), n_cero, float(g.loc["credit card", "mean"])


def fig_metr():
    """El ensayo controlado: la sensacion enganio a los expertos."""
    fig, ax = plt.subplots(figsize=(9.8, 4.2))
    ax.grid(axis="x")
    y = [1, 0]
    vals = [METR_PERCIBIDA, METR_REAL]
    cols = [BLUE, RED]
    nombres = ["Lo que CREYERON\nque pasó", "Lo que de verdad\npasó"]
    ax.barh(y, vals, 0.5, color=cols, alpha=.9)
    for i, v in enumerate(vals):
        ax.text(v + (1.2 if v > 0 else -1.2), y[i],
                f"{v:+.0f} %", va="center", ha="left" if v > 0 else "right",
                fontsize=15, fontweight="bold", color=cols[i])
    ax.axvline(0, color=INK, lw=1.8)
    ax.set_yticks(y)
    ax.set_yticklabels(nombres, fontsize=10.5)
    ax.set_xlim(-30, 30)
    ax.set_xlabel("velocidad con la herramienta, contra trabajar sin ella  [%]",
                  fontsize=9.5)
    ax.set_title("Ensayo controlado con desarrolladores experimentados",
                 fontsize=12.5, fontweight="bold", loc="left", pad=8)
    fig.text(0.5, -0.04,
             "Se sintieron un 20 % más rápidos. Fueron un 19 % más lentos. "
             "La sensación no es una medición.",
             ha="center", fontsize=10.5, color=DARK, style="italic")
    guardar(fig, "fig_metr.png")


def fig_ciclo():
    """El ciclo de trabajo real con un agente."""
    pasos = [("ESPECIFICAR", "qué quiero, con qué\ndatos y qué restricciones", DARK),
             ("GENERAR", "el agente escribe\ny ejecuta", BLUE),
             ("VERIFICAR", "¿los números tienen\nsentido? ¿el dato miente?", RED),
             ("DECIDIR", "sirve, o vuelvo\na especificar", GREEN)]
    fig, ax = plt.subplots(figsize=(11.2, 3.4))
    ax.set_axis_off(); ax.grid(False)
    ax.set_xlim(0, 10); ax.set_ylim(0, 3)
    for i, (tit, sub, col) in enumerate(pasos):
        x = 0.35 + i * 2.45
        ax.add_patch(FancyBboxPatch((x, 1.0), 1.95, 1.25,
                                    boxstyle="round,pad=0.05",
                                    fc="white", ec=col, lw=2.2))
        ax.text(x + 0.97, 1.88, tit, ha="center", fontsize=11.5,
                fontweight="bold", color=col)
        ax.text(x + 0.97, 1.38, sub, ha="center", fontsize=8.5,
                color=MUTED, linespacing=1.3)
        if i < 3:
            ax.add_patch(FancyArrowPatch((x + 2.02, 1.62), (x + 2.38, 1.62),
                                         arrowstyle="-|>", mutation_scale=16,
                                         color=GRAY, lw=1.8))
    ax.add_patch(FancyArrowPatch((9.1, 0.98), (1.3, 0.98),
                                 connectionstyle="arc3,rad=0.18",
                                 arrowstyle="-|>", mutation_scale=16,
                                 color=GRAY, lw=1.6, ls="--"))
    ax.text(5.2, 0.28, "y si no sirve, se vuelve a especificar — eso es el trabajo",
            ha="center", fontsize=9.5, color=MUTED, style="italic")
    ax.text(0.35, 2.55, "El trabajo de ustedes está en las cajas de los extremos",
            fontsize=12, fontweight="bold", color=DARK)
    guardar(fig, "fig_ciclo.png")


def fig_de_grafica_a_app():
    """El salto de la clase: de una celda a una herramienta que otro usa."""
    fig, ax = plt.subplots(figsize=(11.0, 3.5))
    ax.set_axis_off(); ax.grid(False)
    ax.set_xlim(0, 10); ax.set_ylim(0, 3)

    cajas = [
        (0.3, "UNA CELDA", "la gráfica existe\nmientras usted mire\nla pantalla", GRAY),
        (3.6, "UNA APP", "tiene controles:\nel otro elige\nqué mirar", BLUE),
        (6.9, "UN LINK", "se lo manda\na quien decide\ny lo abre solo", GREEN),
    ]
    for x, tit, sub, col in cajas:
        ax.add_patch(FancyBboxPatch((x, 0.8), 2.6, 1.5,
                                    boxstyle="round,pad=0.05",
                                    fc="white", ec=col, lw=2.4))
        ax.text(x + 1.3, 1.92, tit, ha="center", fontsize=12.5,
                fontweight="bold", color=col)
        ax.text(x + 1.3, 1.28, sub, ha="center", fontsize=9,
                color=MUTED, linespacing=1.35)
    for x in (3.0, 6.3):
        ax.add_patch(FancyArrowPatch((x, 1.55), (x + 0.5, 1.55),
                                     arrowstyle="-|>", mutation_scale=18,
                                     color=GRAY, lw=2.0))
    ax.text(5.0, 0.25,
            "El salto que separa un análisis de una herramienta",
            ha="center", fontsize=11, color=DARK, fontweight="bold")
    guardar(fig, "fig_de_grafica_a_app.png")


# =============================================================== MAIN =======
def main():
    print("cargando datasets de seaborn ...")
    taxis, health, flights = cargar()
    print(f"  taxis {len(taxis)} filas | healthexp {len(health)} | "
          f"flights {len(flights)}")

    print("\ngenerando figuras ...")
    fig_adopcion()
    fig_cuello_botella()
    pico, viajes_pico = fig_fea_vs_bonita(taxis)
    fig_tres_dominios(taxis, health, flights)
    fig_escalera_prompt(taxis)
    dia_top, hora_top, n_top = fig_mapa_calor(taxis)
    n_efec, n_prop_efec, prop_tarjeta = fig_trampa(taxis)
    fig_metr()
    fig_ciclo()
    fig_de_grafica_a_app()

    print("\n" + "=" * 74)
    print("NUMEROS PARA LAS LAMINAS  (no escribir ninguno a mano)")
    print("=" * 74)
    print("\nADOPCION EN 2026:")
    print(f"  usa alguna herramienta de IA:        {ADOPCION} %")
    print(f"  usa agentes con regularidad:         {AGENTES} %")
    print(f"  agentes entre ingenieros senior:     {AGENTES_SENIOR} %")
    print(f"  confia PLENAMENTE en la salida:      {CONFIA_PLENO} %")
    print(f"  desconfia de su precision:           {DESCONFIA} %  "
          f"(vs {CONFIA} % que confia)")
    print(f"  frustracion #1 'casi correcto':      {CASI_CORRECTO} %")
    print(f"  depurar IA tarda mas que escribir:   {DEPURAR_MAS} %")
    print(f"  ganancia en tareas acotadas:         {GANANCIA_MIN}-{GANANCIA_MAX} %")
    print(f"  apps empresariales con agentes:      {APPS_ANTES} % -> "
          f"{APPS_CON_AGENTES} %")
    print(f"  ensayo METR: percibida {METR_PERCIBIDA:+d} % | real {METR_REAL:+d} %")

    print("\nDATASETS DE LA CLASE:")
    print(f"  taxis:     {len(taxis)} viajes, {taxis.pickup.min().date()} a "
          f"{taxis.pickup.max().date()}")
    print(f"  healthexp: {len(health)} filas, {health.Country.nunique()} paises, "
          f"{health.Year.min()}-{health.Year.max()}")
    print(f"  flights:   {len(flights)} meses, {flights.year.min()}-"
          f"{flights.year.max()}")

    print("\nHALLAZGOS EN LOS DATOS (los que salen en las laminas):")
    print(f"  hora pico de taxis:            {pico}h con {viajes_pico} viajes")
    print(f"  momento de mas demanda:        {dia_top} a las {hora_top}h "
          f"({n_top} viajes)")
    print(f"  duracion mediana de un viaje:  {taxis.minutos.median():.0f} min")
    print(f"  tarifa mediana:                USD {taxis.total.median():.2f}")

    print("\nLA TRAMPA DEL DATASET (el resultado negativo de la clase):")
    print(f"  viajes pagados en efectivo:              {n_efec}")
    print(f"  de esos, con propina registrada:         {n_prop_efec}")
    print(f"  propina promedio con tarjeta:            USD {prop_tarjeta:.2f}")
    print("  -> la conclusion facil es FALSA: la propina en efectivo")
    print("     no se registra, no es que no exista")


if __name__ == "__main__":
    main()

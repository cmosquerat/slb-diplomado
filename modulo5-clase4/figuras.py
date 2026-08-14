"""
Machine Learning for Petroleum Engineers Using Python
Modulo 5 - Clase 4: genera todas las figuras de la presentacion.

Lee  ../datos/campos_noruega_agua.csv  y escribe los fig_*.png de esta
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
from matplotlib.patches import Rectangle, Polygon, Circle
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- estilo ----
RED, DARK = "#C82B40", "#6B1525"
BLUE, GREEN, ORANGE = "#2563EB", "#16A34A", "#EA580C"
AMBER = "#D97706"
INK, MUTED = "#2D2D2D", "#6B7280"
GRAY, LGRAY = "#9CA3AF", "#E5E7EB"
AQUA = "#0E7490"                       # el color del agua en toda la clase


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
BBL = 6.2898
HIST = 4 * 12          # meses de historia que exigimos antes de examinar
H = 5 * 12             # el horizonte de la clase: 5 anios
UMBRAL_AGUA = 0.02     # 2 % de corte sostenido: "llego el agua"
CRUCE = 0.50           # la linea que decide la clase: mitad agua, mitad crudo
EJEMPLO = "OSEBERG"    # el campo que sigue toda la clase
ANIO_EJEMPLO = 2013    # nos paramos en enero de 2013 y tapamos el futuro
SENAL = "ALVHEIM"      # el campo con el que se presenta la curva en S

# EL TELON: el agua solo se reporta desde enero de 2000 (verificado en el
# EDA: en 1998 producian 35 campos y solo 1 reportaba agua). Antes de esa
# fecha los ceros de la columna agua son mentira. Ningun examen puede usar
# meses anteriores a esta fecha.
TELON = pd.Timestamp(2000, 1, 1)

# --- supuesto economico. Este numero lo pone finanzas, no el ingeniero:
# se declara a la vista para que se pueda discutir y cambiar.
COSTO_AGUA = 1.5       # USD por barril de agua: levantarla, separarla,
                       # tratarla y devolverla al subsuelo


def cargar():
    d = pd.read_csv("../datos/campos_noruega_agua.csv")
    d["fecha"] = pd.to_datetime(d.fecha)
    campos = {c: g.reset_index(drop=True) for c, g in d.groupby("campo")}
    return d, campos


def corte(g, i0, i1):
    """Corte de agua AGREGADO del tramo [i0, i1): barriles de agua sobre
    barriles de liquido. Se suman barriles, no se promedian porcentajes."""
    if i0 < 0 or i1 > len(g):
        return np.nan
    o = (g.oil_bpd * g.dias).iloc[i0:i1].sum()
    a = (g.agua_bpd * g.dias).iloc[i0:i1].sum()
    return a / (a + o) if (a + o) > 0 else np.nan


def gor(g, i0, i1):
    """GOR agregado del tramo, en Sm3 de gas por Sm3 de petroleo."""
    if i0 < 0 or i1 > len(g):
        return np.nan
    o = (g.oil_bpd / BBL * g.dias).iloc[i0:i1].sum()
    ga = (g.gas_ksm3d * 1e3 * g.dias).iloc[i0:i1].sum()
    return ga / o if o > 0 else np.nan


def llegada_agua(g):
    """El mes en que 'llego el agua': corte mensual > 2 % sostenido 3 meses.
    Solo se busca desde el TELON (antes de 2000 el agua no se reportaba)."""
    wc = g.agua_bpd / (g.agua_bpd + g.oil_bpd).replace(0, np.nan)
    wc[g.fecha < TELON] = np.nan
    con = (wc > UMBRAL_AGUA).rolling(3).sum() == 3
    return int(con.idxmax()) if con.any() else None


def censurado(g):
    """True si al campo NO le vimos llegar el agua: ya producia antes del
    2000 y, cuando el regulador empezo a contar el agua, ya venia con ella.
    Su 'llegada' seria un invento del reporte, no del yacimiento."""
    if g.fecha.min() >= TELON - pd.DateOffset(months=6):
        return False
    ia = llegada_agua(g)
    if ia is None:
        return False
    return g.fecha.iloc[ia] < TELON + pd.DateOffset(months=18)


def censurado(g):
    """True si al campo NO le vimos llegar el agua: ya producia antes del
    2000 y, cuando el regulador empezo a contar, ya venia con agua. Su
    'llegada' seria un invento del reporte, no del yacimiento."""
    if g.fecha.min() >= TELON - pd.DateOffset(months=6):
        return False
    ia = llegada_agua(g)
    return ia is None or g.fecha.iloc[ia] < TELON + pd.DateOffset(months=18)


def examenes(campos, h=H):
    """Un examen por campo y por anio: nos paramos en un enero con 4 anios de
    historia, miramos solo el pasado, y el futuro a h meses queda para
    calificar. Devuelve una fila por examen."""
    filas = []
    for c, g in campos.items():
        oil_acum = (g.oil_bpd * g.dias).cumsum()
        i_agua = llegada_agua(g)
        for t in range(HIST, len(g) - h):
            if g.fecha.iloc[t].month != 1:
                continue
            # el telon: los dos anios de historia que mira el examen tienen
            # que estar despues de que el regulador empezara a contar el agua
            if g.fecha.iloc[t - 24] < TELON:
                continue
            wc_hoy = corte(g, t - 12, t)
            wc_ayer = corte(g, t - 24, t - 12)
            wc_fut = corte(g, t + h - 12, t + h)
            if any(np.isnan(v) for v in (wc_hoy, wc_ayer, wc_fut)):
                continue
            pico = g.oil_bpd.iloc[:t].max()
            if pico <= 0:
                continue
            anuales = [corte(g, t + k - 12, t + k) for k in range(12, h + 1, 12)]
            anuales = [v for v in anuales if not np.isnan(v)]
            oil_hoy = g.oil_bpd.iloc[t - 12:t].mean()
            filas.append(dict(
                campo=c, t=t, anio=g.fecha.iloc[t].year,
                wc_hoy=wc_hoy, pend=wc_hoy - wc_ayer,
                edad_agua=(t - i_agua) / 12 if i_agua is not None and t > i_agua
                          else 0.0,
                frac_decl=oil_hoy / pico, log_pico=np.log10(pico),
                log_np=np.log10(max(oil_acum.iloc[t - 1], 1.0)), meses=t,
                oil_hoy=oil_hoy,
                gor_hoy=gor(g, t - 12, t), gor_ayer=gor(g, t - 24, t - 12),
                gor_fut=gor(g, t + h - 12, t + h),
                wc_fut=wc_fut,
                wc_max=max(anuales) if anuales else np.nan))
    return pd.DataFrame(filas)


def rivales(s, h=H):
    """Los cuatro rivales, sobre la tabla de examenes. Todos honestos:
    ninguno ve el futuro ni el campo que esta prediciendo."""
    s = s.copy()
    s["persistencia"] = s.wc_hoy
    s["tendencia"] = (s.wc_hoy + s.pend * h / 12).clip(0, 0.99)

    # analogos: entre los examenes de OTROS campos con corte parecido hoy
    # (± 5 puntos), ¿en que corte terminaron a 5 anios? mediana y banda
    med, lo, hi = [], [], []
    wc_o, y_o, c_o = s.wc_hoy.values, s.wc_fut.values, s.campo.values
    for i in range(len(s)):
        m = (np.abs(wc_o - wc_o[i]) < 0.05) & (c_o != c_o[i])
        if m.sum() >= 8:
            med.append(np.median(y_o[m]))
            lo.append(np.quantile(y_o[m], 0.10))
            hi.append(np.quantile(y_o[m], 0.90))
        else:
            med.append(wc_o[i]); lo.append(np.nan); hi.append(np.nan)
    s["analogos"], s["banda_lo"], s["banda_hi"] = med, lo, hi

    # Random Forest de flota: aprende el INCREMENTO de corte, nunca examina
    # un campo que vio en el entrenamiento (se dejan campos enteros afuera)
    feats = ["wc_hoy", "pend", "edad_agua", "frac_decl", "log_pico",
             "log_np", "meses", "analogos"]
    X = s[feats].values
    y = (s.wc_fut - s.wc_hoy).values
    oof = np.full(len(s), np.nan)
    for tr, te in GroupKFold(5).split(X, y, s.campo.values):
        rf = RandomForestRegressor(400, min_samples_leaf=20,
                                   random_state=0, n_jobs=-1)
        rf.fit(X[tr], y[tr])
        oof[te] = rf.predict(X[te])
    s["bosque"] = np.clip(s.wc_hoy.values + oof, 0, 0.99)
    return s, feats


MODELOS = [("El agua sigue igual", "persistencia", GRAY),
           ("Su propia tendencia", "tendencia", ORANGE),
           ("Análogos de la flota", "analogos", BLUE),
           ("Bosque de flota", "bosque", GREEN)]


def curvas_familia(campos):
    """Para cada campo con >= 6 anios de agua: su corte anual desde que el
    agua llego (anios 0..15). Los anios que aun no vivio se rellenan con el
    ultimo valor observado."""
    curvas = {}
    for c, g in campos.items():
        if censurado(g):        # no le vimos llegar el agua: no tiene curva
            continue
        ia = llegada_agua(g)
        if ia is None:
            continue
        ys = np.array([corte(g, ia + 12 * a, ia + 12 * (a + 1))
                       for a in range(16)], float)
        if np.isfinite(ys).sum() < 6:
            continue
        ult = np.nanmax(np.where(np.isfinite(ys))[0])
        for j in range(len(ys)):
            if not np.isfinite(ys[j]):
                ys[j] = ys[ult] if j > ult else 0.0
        curvas[c] = ys
    return curvas


def familias(curvas, k=3):
    nombres = sorted(curvas)
    M = np.array([curvas[c] for c in nombres])
    km = KMeans(k, n_init=20, random_state=0).fit(M)
    orden = np.argsort(km.cluster_centers_[:, 8])      # por corte a 8 anios
    etiqueta = {int(f): r for r, f in enumerate(orden)}
    asig = {nombres[i]: etiqueta[int(km.labels_[i])] for i in range(len(nombres))}
    centros = km.cluster_centers_[orden]
    return nombres, M, asig, centros


NOMBRES_FAM = ["LENTA", "MEDIA", "BRAVA"]
COL_FAM = [GREEN, AMBER, RED]


def examen_familias(campos, curvas, s):
    """El examen honesto del no supervisado: ¿los analogos DE TU FAMILIA
    pronostican mejor que los analogos planos? Para cada campo, las familias
    se reconstruyen SIN ese campo (nadie se examina con su propia historia)."""
    res = []
    nombres = sorted(curvas)
    for cx in s.campo.unique():
        otros = [c for c in nombres if c != cx]
        if len(otros) < 10:
            continue
        Mo = np.array([curvas[c] for c in otros])
        km = KMeans(3, n_init=10, random_state=0).fit(Mo)
        lab = dict(zip(otros, km.labels_))
        g = campos[cx]
        ia = llegada_agua(g)
        for _, row in s[s.campo == cx].iterrows():
            t = int(row.t)
            fam = None
            if ia is not None and t > ia + 12:
                span = min(int((t - ia) / 12), 15)
                parcial = np.array([corte(g, ia + 12 * a, ia + 12 * (a + 1))
                                    for a in range(span)])
                ok = np.isfinite(parcial)
                if ok.sum() >= 2:
                    dist = [np.nanmean((parcial[ok]
                            - km.cluster_centers_[f][:span][ok]) ** 2)
                            for f in range(3)]
                    fam = int(np.argmin(dist))
            pool = s[s.campo != cx]
            cerca = pool[np.abs(pool.wc_hoy - row.wc_hoy) < 0.05]
            cerca_f = (cerca[cerca.campo.isin([c for c in otros
                                               if lab[c] == fam])]
                       if fam is not None else cerca)
            fila = dict(campo=cx, wc_fut=row.wc_fut, fam=fam)
            for et, cc in [("plano", cerca), ("familia", cerca_f)]:
                if len(cc) >= 8:
                    fila[f"med_{et}"] = cc.wc_fut.median()
                    fila[f"lo_{et}"] = cc.wc_fut.quantile(0.10)
                    fila[f"hi_{et}"] = cc.wc_fut.quantile(0.90)
            res.append(fila)
    r = pd.DataFrame(res).dropna(subset=["med_plano", "med_familia"])
    return r[r.fam.notna()]


def hoy_features(campos, s):
    """La foto de HOY de cada campo activo: los mismos rasgos que en los
    examenes, calculados sobre los ultimos meses del dato."""
    filas = []
    for c, g in campos.items():
        if len(g) < HIST + 12:
            continue
        t = len(g)
        activo = (g.oil_bpd + g.agua_bpd).iloc[-12:].mean() > 100
        if not activo:
            continue
        wc_hoy = corte(g, t - 12, t)
        wc_ayer = corte(g, t - 24, t - 12)
        if np.isnan(wc_hoy) or np.isnan(wc_ayer):
            continue
        ia = llegada_agua(g)
        oil_acum = (g.oil_bpd * g.dias).sum()
        oil_hoy = g.oil_bpd.iloc[-12:].mean()
        pico = g.oil_bpd.max()
        filas.append(dict(
            campo=c, wc_hoy=wc_hoy, pend=wc_hoy - wc_ayer,
            edad_agua=(t - ia) / 12 if ia is not None and t > ia else 0.0,
            frac_decl=oil_hoy / pico, log_pico=np.log10(pico),
            log_np=np.log10(max(oil_acum, 1.0)), meses=t,
            oil_hoy=oil_hoy, agua_hoy=g.agua_bpd.iloc[-12:].mean()))
    hoy = pd.DataFrame(filas)

    # analogos para la foto de hoy, del pool de examenes historicos
    med, lo, hi = [], [], []
    for _, row in hoy.iterrows():
        m = s[(np.abs(s.wc_hoy - row.wc_hoy) < 0.05) & (s.campo != row.campo)]
        if len(m) >= 8:
            med.append(m.wc_fut.median())
            lo.append(m.wc_fut.quantile(0.10)); hi.append(m.wc_fut.quantile(0.90))
        else:
            med.append(row.wc_hoy); lo.append(np.nan); hi.append(np.nan)
    hoy["analogos"], hoy["banda_lo"], hoy["banda_hi"] = med, lo, hi
    return hoy


def lista_de_hoy(campos, s, feats):
    """El entregable: el bosque, entrenado con TODOS los examenes historicos,
    califica la foto de hoy de los campos que aun estan debajo del cruce."""
    hoy = hoy_features(campos, s)
    rf = RandomForestRegressor(400, min_samples_leaf=20, random_state=0,
                               n_jobs=-1)
    rf.fit(s[feats].values, (s.wc_fut - s.wc_hoy).values)
    hoy["bosque"] = np.clip(hoy.wc_hoy + rf.predict(hoy[feats].values), 0, 0.99)
    lista = hoy[(hoy.wc_hoy < CRUCE) & (hoy.oil_hoy > 5000)].copy()
    return lista.sort_values("bosque", ascending=False), hoy


# =========================================================== FIGURAS ========
def fig_draugen(campos):
    """El gancho: el campo de la Clase 3 tiene una segunda cara."""
    g = campos["DRAUGEN"]
    t = np.arange(len(g)) / 12
    wc_hoy = 100 * corte(g, len(g) - 12, len(g))
    wor_hoy = g.agua_bpd.iloc[-12:].mean() / g.oil_bpd.iloc[-12:].mean()

    vis = (g.fecha >= TELON).values          # el agua solo existe desde 2000
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
    ax = axes[0]
    ax.plot(t, g.oil_bpd / 1000, color=INK, lw=1.8, label="petróleo")
    ax.set_xlabel("años desde el arranque", fontsize=9.5)
    ax.set_ylabel("miles de barriles por día", fontsize=9.5)
    ax.set_title("El Draugen que vimos en la Clase 3", fontsize=11.5,
                 fontweight="bold", loc="left")
    ax.legend(fontsize=9.5)

    ax = axes[1]
    ax.plot(t, g.oil_bpd / 1000, color=INK, lw=1.8, label="petróleo")
    ax.plot(t[vis], g.agua_bpd[vis] / 1000, color=AQUA, lw=1.8, label="agua")
    ax.fill_between(t[vis], 0, g.agua_bpd[vis] / 1000, color=AQUA, alpha=.18)
    ax.axvspan(0, t[vis][0], color=LGRAY, alpha=.45)
    ax.text(t[vis][0] / 2, g.agua_bpd.max() / 1000 * 0.45,
            "antes del 2000\nel agua no se\nreportaba", ha="center", fontsize=8.5,
            color=MUTED, linespacing=1.3)
    ax.set_xlabel("años desde el arranque", fontsize=9.5)
    ax.set_title("El Draugen completo: lo que también salía del pozo",
                 fontsize=11.5, fontweight="bold", loc="left", color=DARK)
    ax.legend(fontsize=9.5, loc="upper left")
    ax.annotate(f"hoy: {wor_hoy:.0f} barriles de agua\npor cada uno de petróleo",
                xy=(t[-1] - 1, g.agua_bpd.iloc[-12:].mean() / 1000),
                xytext=(t[-1] - 17, g.agua_bpd.max() / 1000 * 0.9),
                fontsize=10, color=AQUA, fontweight="bold", linespacing=1.3,
                arrowprops=dict(arrowstyle="->", color=AQUA, lw=1.5))
    fig.tight_layout(w_pad=2.4)
    guardar(fig, "fig_draugen.png")
    return wc_hoy, wor_hoy


def fig_flota(d, campos):
    """La flota entera: cuanto petroleo y cuanta agua, y cuanto cuesta."""
    m = d.groupby("fecha")[["oil_bpd", "agua_bpd"]].sum()
    u = d[d.fecha > d.fecha.max() - pd.DateOffset(months=12)]
    f = u.groupby("campo")[["oil_bpd", "agua_bpd"]].mean()
    f = f[f.sum(axis=1) > 100]
    oil_hoy, agua_hoy = f.oil_bpd.sum(), f.agua_bpd.sum()
    usd_anio = agua_hoy * 365 * COSTO_AGUA

    fig, ax = plt.subplots(figsize=(10.8, 4.4))
    anios = np.array(m.index.year + (m.index.month - 1) / 12)
    vis = np.asarray(m.index >= TELON)
    ax.plot(anios, m.oil_bpd / 1e6, color=INK, lw=2.0, label="petróleo")
    ax.plot(anios[vis], m.agua_bpd.values[vis] / 1e6, color=AQUA, lw=2.0,
            label="agua")
    ax.fill_between(anios[vis], m.oil_bpd.values[vis] / 1e6,
                    m.agua_bpd.values[vis] / 1e6,
                    where=(m.agua_bpd.values > m.oil_bpd.values)[vis],
                    color=AQUA, alpha=.15)
    ax.axvspan(anios[0], TELON.year, color=LGRAY, alpha=.4)
    ax.text((anios[0] + TELON.year) / 2, m.oil_bpd.max() / 1e6 * 0.72,
            "aquí el agua también\nsalía — pero nadie\nla estaba contando",
            ha="center", fontsize=9, color=MUTED, linespacing=1.35)
    mm = m[vis]
    i_cruce = int(np.argmax(mm.agua_bpd.values > mm.oil_bpd.values))
    x_cruce = mm.index[i_cruce].year + (mm.index[i_cruce].month - 1) / 12
    ax.axvline(x_cruce, color=DARK, lw=1.4, ls="--")
    ax.text(x_cruce + 0.6, m.oil_bpd.max() / 1e6 * 0.97,
            f"{mm.index[i_cruce].year}: la plataforma empezó\na producir más agua "
            "que petróleo", fontsize=9.5, color=DARK, linespacing=1.3)
    ax.set_xlabel("año", fontsize=9.5)
    ax.set_ylabel("millones de barriles por día", fontsize=9.5)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_title("Toda la plataforma noruega, mes a mes, desde 1971",
                 fontsize=12.5, fontweight="bold", loc="left", pad=8)
    guardar(fig, "fig_flota.png")
    return oil_hoy, agua_hoy, usd_anio, mm.index[i_cruce].year, len(f)


def fig_como_llega():
    """Esquema conceptual: de donde sale el agua y por que llega al pozo.
    Sin datos: es el dibujo que se explica hablando."""
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3))
    for ax, modo in zip(axes, ["cono", "canal"]):
        ax.set_xlim(0, 10); ax.set_ylim(0, 6)
        ax.set_axis_off(); ax.grid(False)
        # roca con petroleo arriba, agua abajo
        ax.add_patch(Rectangle((0, 3.1), 10, 1.9, fc="#F3E8D8", ec="none"))
        ax.add_patch(Rectangle((0, 0.6), 10, 2.5, fc="#CFE8F3", ec="none"))
        ax.text(0.3, 4.55, "roca con petróleo", fontsize=9.5, color="#8A6D3B")
        ax.text(0.3, 1.0, "agua de abajo (acuífero)", fontsize=9.5, color=AQUA)
        # el pozo
        ax.add_patch(Rectangle((4.9, 3.4), 0.2, 2.6, fc=INK, ec="none"))
        ax.add_patch(Rectangle((4.7, 5.8), 0.6, 0.25, fc=INK, ec="none"))
        ax.plot([4.9, 4.9], [3.4, 4.4], color="white", lw=0)
        for yy in (3.6, 3.9, 4.2):
            ax.plot([4.78, 5.22], [yy, yy], color="white", lw=1.1)
        if modo == "cono":
            xs = np.linspace(0, 10, 200)
            contacto = 3.1 + 1.15 * np.exp(-((xs - 5.0) ** 2) / 0.9)
            ax.fill_between(xs, 0.6, contacto, color="#CFE8F3")
            ax.plot(xs, contacto, color=AQUA, lw=2.2)
            ax.annotate("el agua se LEVANTA\nhacia los disparos",
                        xy=(5.0, 4.15), xytext=(6.9, 5.0), fontsize=10,
                        color=AQUA, fontweight="bold", linespacing=1.3,
                        arrowprops=dict(arrowstyle="->", color=AQUA, lw=1.6))
            ax.set_title("CONIFICACIÓN: el agua sube por debajo del pozo",
                         fontsize=11.5, fontweight="bold", loc="left")
        else:
            ax.add_patch(Rectangle((0, 3.55), 10, 0.5, fc="#CFE8F3", ec=AQUA,
                                   lw=1.4))
            for x0 in (1.2, 2.6, 4.0):
                ax.annotate("", xy=(x0 + 1.0, 3.8), xytext=(x0, 3.8),
                            arrowprops=dict(arrowstyle="->", color=AQUA, lw=1.8))
            ax.text(0.3, 3.62, "capa muy permeable", fontsize=9, color=AQUA,
                    fontweight="bold")
            ax.annotate("el agua CORRE por la capa\nfácil y llega de costado",
                        xy=(4.6, 3.8), xytext=(6.6, 5.0), fontsize=10,
                        color=AQUA, fontweight="bold", linespacing=1.3,
                        arrowprops=dict(arrowstyle="->", color=AQUA, lw=1.6))
            ax.set_title("CANALIZACIÓN: el agua encuentra una autopista",
                         fontsize=11.5, fontweight="bold", loc="left")
    fig.text(0.5, -0.03, "dos caminos distintos, un mismo final: un día el agua "
             "llega a los disparos, y de ahí en adelante viene con el petróleo",
             ha="center", fontsize=10.5, color=DARK, style="italic")
    fig.tight_layout(w_pad=2.0)
    guardar(fig, "fig_como_llega.png")


def fig_corte_wor(campos):
    """De barriles a senal: que es el corte de agua, y su curva en S.
    Se usa un campo que arranco DESPUES del 2000, para que se le vea la
    historia del agua completa."""
    g = campos[SENAL]
    t = np.arange(len(g)) / 12
    wc = 100 * (g.agua_bpd * g.dias).rolling(12).sum() / \
        ((g.agua_bpd + g.oil_bpd) * g.dias).rolling(12).sum()

    fig, axes = plt.subplots(2, 1, figsize=(10.6, 5.4), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.1, 1]))
    ax = axes[0]
    ax.plot(t, g.oil_bpd / 1000, color=INK, lw=1.6, label="petróleo")
    ax.plot(t, g.agua_bpd / 1000, color=AQUA, lw=1.6, label="agua")
    ax.set_ylabel("miles de bbl/día", fontsize=9.5)
    ax.legend(fontsize=9.5, loc="upper right")
    ax.set_title(f"{SENAL.title()}: los barriles que se miden cada mes",
                 fontsize=11.5, fontweight="bold", loc="left")

    ax = axes[1]
    ax.plot(t, wc, color=RED, lw=2.2)
    ax.axhline(50, color=DARK, lw=1.4, ls="--")
    ax.text(1.0, 53, "la línea de la mitad: más agua que petróleo",
            fontsize=9.5, color=DARK)
    ax.set_ylim(0, 100)
    ax.set_xlabel("años desde el arranque", fontsize=9.5)
    ax.set_ylabel("corte de agua  [%]", fontsize=9.5)
    ax.set_title("La señal: corte de agua = agua ÷ (agua + petróleo)",
                 fontsize=11.5, fontweight="bold", loc="left", color=RED)
    ax.annotate("arranca en cero,\nun día despega...", xy=(t[np.argmax(wc.values > 5)], 6),
                xytext=(3.5, 30), fontsize=9.5, color=MUTED, linespacing=1.3,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    ax.annotate("...y arriba se aplana:\nla curva en S", xy=(t[-1] - 1, wc.iloc[-1] - 3),
                xytext=(t[-1] - 12, 22), fontsize=9.5, color=MUTED, linespacing=1.3,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    fig.tight_layout(h_pad=1.0)
    guardar(fig, "fig_corte_wor.png")


def fig_gas_senal(campos):
    """La otra senal: el GOR. Dos campos reales, uno tranquilo y uno que grita."""
    fig, ax = plt.subplots(figsize=(10.4, 4.2))
    for c, col in [("GRANE", GREEN), ("OSEBERG", ORANGE)]:
        g = campos[c]
        n = len(g)
        anios, valores = [], []
        for a in range(2, n // 12):
            v = gor(g, 12 * a, 12 * (a + 1))
            if v and v > 0:
                anios.append(a); valores.append(v)
        ax.plot(anios, valores, color=col, lw=2.0, label=c.title())
    ax.set_yscale("log")
    ax.set_xlabel("años desde el arranque", fontsize=9.5)
    ax.set_ylabel("GOR  [Sm³ de gas / Sm³ de petróleo]", fontsize=9.5)
    ax.legend(fontsize=10)
    ax.set_title("La otra señal: cuánto gas viene con cada barril",
                 fontsize=12.5, fontweight="bold", loc="left", pad=8)
    ax.text(0.99, 0.05, "si el GOR sube, el yacimiento está perdiendo presión\n"
            "(o alguien está inyectando gas: por eso hay que conocer el campo)",
            transform=ax.transAxes, ha="right", fontsize=9.5, color=MUTED,
            style="italic", linespacing=1.35)
    guardar(fig, "fig_gas_senal.png")


def fig_espagueti(campos, curvas):
    """Todas las trayectorias juntas: caos por calendario, estructura por
    edad del agua. El mismo truco de alinear de la C3, con otro reloj."""
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
    ax = axes[0]
    for c, g in campos.items():
        if c not in curvas:
            continue
        v = g[g.fecha >= TELON]
        wc = 100 * (v.agua_bpd * v.dias).rolling(12).sum() / \
            ((v.agua_bpd + v.oil_bpd) * v.dias).rolling(12).sum()
        anios = v.fecha.dt.year + (v.fecha.dt.month - 1) / 12
        ax.plot(anios, wc, color=GRAY, lw=0.7, alpha=.5)
    ax.set_ylim(0, 100)
    ax.set_xlabel("año de calendario", fontsize=9.5)
    ax.set_ylabel("corte de agua  [%]", fontsize=9.5)
    ax.set_title("Por calendario: un nido imposible de leer",
                 fontsize=11.5, fontweight="bold", loc="left")

    ax = axes[1]
    for c, ys in curvas.items():
        ax.plot(np.arange(16), 100 * ys, color=GRAY, lw=0.8, alpha=.55)
    ax.axhline(50, color=DARK, lw=1.2, ls="--")
    ax.set_xlabel("años desde que LLEGÓ el agua (el reloj de cada campo)",
                  fontsize=9.5)
    ax.set_title("Con el reloj del agua: ahora se puede comparar",
                 fontsize=11.5, fontweight="bold", loc="left", color=GREEN)
    fig.tight_layout(w_pad=2.4)
    guardar(fig, "fig_espagueti.png")


def fig_familias_kmeans(curvas, nombres, M, asig, centros):
    """Izquierda: la idea de agrupar curvas. Derecha: las tres familias."""
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3))
    ax = axes[0]
    for ys in M:
        ax.plot(np.arange(16), 100 * ys, color=GRAY, lw=0.8, alpha=.4)
    for f in range(3):
        ax.plot(np.arange(16), 100 * centros[f], color=COL_FAM[f], lw=3.2,
                label=f"familia {NOMBRES_FAM[f].title()}")
    ax.axhline(50, color=DARK, lw=1.2, ls="--")
    ax.set_xlabel("años desde que llegó el agua", fontsize=9.5)
    ax.set_ylabel("corte de agua  [%]", fontsize=9.5)
    ax.legend(fontsize=9)
    ax.set_title(f"K-means agrupó las {len(M)} curvas en 3 moldes",
                 fontsize=11.5, fontweight="bold", loc="left")

    ax = axes[1]
    ax.grid(axis="y")
    conteo = [sum(1 for c in asig if asig[c] == f) for f in range(3)]
    a50 = []
    for f in range(3):
        idx = np.where(100 * centros[f] >= 50)[0]
        a50.append(str(int(idx[0])) + " años" if len(idx) else "no llega\nen 15 años")
    x = np.arange(3)
    ax.bar(x, [100 * centros[f][5] for f in range(3)], 0.55,
           color=COL_FAM, alpha=.85)
    for f in range(3):
        ax.text(f, 100 * centros[f][5] + 2.5,
                f"{100*centros[f][5]:.0f} %", ha="center", fontsize=12,
                fontweight="bold", color=COL_FAM[f])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{NOMBRES_FAM[f].title()}\n{conteo[f]} campos · "
                        f"cruza 50 %: {a50[f]}" for f in range(3)], fontsize=9)
    ax.set_ylabel("corte a los 5 años de llegar el agua  [%]", fontsize=9.5)
    ax.set_ylim(0, 100)
    ax.set_title("Tres maneras de envejecer con el agua", fontsize=11.5,
                 fontweight="bold", loc="left")
    fig.tight_layout(w_pad=2.4)
    guardar(fig, "fig_familias_kmeans.png")
    return conteo, a50


def fig_familias_miembros(M, nombres, asig, centros):
    """Las tres familias, cada una con sus campos y algunos nombres."""
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.0), sharey=True)
    for f, ax in enumerate(axes):
        miembros = [c for c in nombres if asig[c] == f]
        for c in miembros:
            ax.plot(np.arange(16), 100 * np.array(M[nombres.index(c)]),
                    color=COL_FAM[f], lw=1.0, alpha=.35)
        ax.plot(np.arange(16), 100 * centros[f], color=COL_FAM[f], lw=3.0)
        ax.axhline(50, color=DARK, lw=1.0, ls="--")
        ax.set_ylim(0, 100)
        ax.set_xlabel("años desde que llegó el agua", fontsize=9)
        ax.set_title(f"{NOMBRES_FAM[f].title()}  ·  {len(miembros)} campos",
                     fontsize=11, fontweight="bold", loc="left",
                     color=COL_FAM[f])
        ejemplos = ", ".join(sorted(miembros)[:3]).title()
        ax.text(0.03, 0.93, ejemplos + ", ...", transform=ax.transAxes,
                fontsize=8, color=MUTED, va="top")
    axes[0].set_ylabel("corte de agua  [%]", fontsize=9.5)
    fig.tight_layout(w_pad=1.6)
    guardar(fig, "fig_familias_miembros.png")


def fig_familia_examen(rfam):
    """El examen honesto: ¿la familia mejora el pronostico? No."""
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))
    ax = axes[0]
    ax.grid(axis="y")
    maes = [100 * (rfam[f"med_{et}"] - rfam.wc_fut).abs().mean()
            for et in ("plano", "familia")]
    x = np.arange(2)
    ax.bar(x, maes, 0.5, color=[BLUE, AMBER], alpha=.85)
    for i, v in enumerate(maes):
        ax.text(i, v + 0.25, f"{v:.1f} pp", ha="center", fontsize=13,
                fontweight="bold", color=DARK)
    ax.set_xticks(x)
    ax.set_xticklabels(["análogos\nde toda la flota", "análogos\nsolo de tu familia"],
                       fontsize=10)
    ax.set_ylim(0, max(maes) * 1.22)
    ax.set_ylabel("error típico a 5 años  [puntos de corte]", fontsize=9.5)
    ax.set_title("¿Pronostica mejor tu familia? No.", fontsize=11.5,
                 fontweight="bold", loc="left", color=DARK)

    ax = axes[1]
    ax.grid(axis="x")
    datos = []
    for et in ("plano", "familia"):
        ancho = 100 * (rfam[f"hi_{et}"] - rfam[f"lo_{et}"]).mean()
        cob = 100 * ((rfam.wc_fut >= rfam[f"lo_{et}"]) &
                     (rfam.wc_fut <= rfam[f"hi_{et}"])).mean()
        datos.append((ancho, cob))
    y = [1, 0]
    for i, (et, col) in enumerate([("toda la flota", BLUE), ("tu familia", AMBER)]):
        ax.barh(y[i], datos[i][0], 0.45, color=col, alpha=.8)
        ax.text(datos[i][0] + 0.8, y[i],
                f"cumple el {datos[i][1]:.0f} %", va="center",
                fontsize=10.5, fontweight="bold",
                color=GREEN if datos[i][1] >= 78 else RED)
    ax.set_yticks(y)
    ax.set_yticklabels(["análogos de\ntoda la flota", "análogos solo\nde tu familia"],
                       fontsize=10)
    ax.set_xlim(0, 50)
    ax.set_xlabel("ancho de la banda P10–P90  [puntos de corte]", fontsize=9.5)
    ax.set_title("La banda angosta se paga: deja de cumplir", fontsize=11.5,
                 fontweight="bold", loc="left", color=DARK)
    fig.text(0.5, -0.04, "la familia no mejora el número — el corte de hoy ya trae "
             "tu historia adentro. Para lo que sí sirve: saber con quién compararse.",
             ha="center", fontsize=10.5, color=DARK, style="italic")
    fig.tight_layout(w_pad=2.4)
    guardar(fig, "fig_familia_examen.png")


def fig_encargo(campos):
    """Oseberg, enero de 2013: el futuro tapado. El dispositivo de la C3."""
    g = campos[EJEMPLO][campos[EJEMPLO].fecha >= TELON].reset_index(drop=True)
    idx = int(np.argmax(g.fecha >= pd.Timestamp(ANIO_EJEMPLO, 1, 1)))
    t = np.array(g.fecha.dt.year + (g.fecha.dt.month - 1) / 12)
    wc = 100 * (g.agua_bpd * g.dias).rolling(12).sum() / \
        ((g.agua_bpd + g.oil_bpd) * g.dias).rolling(12).sum()
    fin = idx + H

    fig, ax = plt.subplots(figsize=(10.6, 4.3))
    ax.plot(t[:idx], wc.iloc[:idx], color=RED, lw=2.4)
    ax.add_patch(Rectangle((t[idx], 0), t[fin] - t[idx], 100, fc=LGRAY,
                           alpha=.75, ec="none"))
    ax.text((t[idx] + t[fin]) / 2, 78, "esto todavía\nno pasó", ha="center",
            va="center", fontsize=13, color=MUTED, fontweight="bold",
            linespacing=1.3)
    ax.axvline(t[idx], color=DARK, lw=1.8, ls="--")
    ax.text(t[idx] - 0.25, 92, f"enero de {ANIO_EJEMPLO}", ha="right", fontsize=11,
            fontweight="bold", color=DARK)
    ax.axhline(50, color=DARK, lw=1.2, ls=":")
    ax.text(t[0] + 0.3, 52, "la línea de la mitad", fontsize=9, color=DARK)
    ax.set_xlim(t[0], t[fin] + 0.3)
    ax.set_ylim(0, 100)
    ax.set_xlabel("año", fontsize=9.5)
    ax.set_ylabel("corte de agua  [%]", fontsize=9.5)
    wc_hoy = corte(g, idx - 12, idx)
    ax.set_title(f"{EJEMPLO.title()}, enero de 2013: va en {100*wc_hoy:.0f} %. "
                 "¿Dónde está en 2018?", fontsize=12.5, fontweight="bold",
                 loc="left", pad=8)
    guardar(fig, "fig_encargo.png")


def fig_horizonte(campos):
    """LA figura de la clase: a que distancia empieza a cobrar la flota."""
    hs = [12, 24, 36, 48, 60]
    res = {"persistencia": [], "tendencia": [], "analogos": [], "bosque": []}
    for h in hs:
        s_h = examenes(campos, h)
        s_h, _ = rivales(s_h, h)
        for _, col, _ in MODELOS:
            res[col].append(100 * (s_h[col] - s_h.wc_fut).abs().mean())

    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    for nom, col, c in MODELOS:
        ax.plot(np.array(hs) / 12, res[col], "o-", color=c, lw=2.4, ms=6,
                label=nom)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("¿a cuántos años estamos mirando?", fontsize=9.5)
    ax.set_ylabel("error típico  [puntos de corte de agua]", fontsize=9.5)
    ax.legend(fontsize=9.5, loc="upper left")
    ax.axvspan(0.8, 1.5, color=LGRAY, alpha=.45)
    ax.annotate("a 1 año, el más simple empata:\naquí NO hace falta aprender de nadie",
                xy=(1.06, res["persistencia"][0] + 0.3), xytext=(2.25, 5.1),
                fontsize=9.5, color=MUTED, va="bottom", linespacing=1.35,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    ax.annotate("a 5 años, la experiencia\nde la flota es la que menos\nse equivoca",
                xy=(5, res["analogos"][-1]), xytext=(3.6, res["analogos"][-1] - 4.5),
                fontsize=9.5, color=BLUE, fontweight="bold", linespacing=1.3,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4))
    ax.set_title("El horizonte decide el método: cerca gana lo simple, lejos gana la flota",
                 fontsize=12, fontweight="bold", loc="left", pad=8)
    guardar(fig, "fig_horizonte.png")
    return dict((k, v) for k, v in res.items()), hs


def fig_sesgo_agua(s):
    """El agua nunca se queda quieta: el error de la persistencia es sesgado."""
    e = 100 * (s.persistencia - s.wc_fut)
    fig, ax = plt.subplots(figsize=(10.2, 4.2))
    bins = np.arange(-75, 40, 4)
    ax.hist(e, bins=bins, color=GRAY, alpha=.8)
    ax.axvline(0, color=INK, lw=2)
    ax.axvline(e.median(), color=RED, lw=2.4, ls="--")
    subio = 100 * (s.wc_fut > s.wc_hoy).mean()
    ax.annotate(f"la mitad de las veces se quedó\ncorta {abs(e.median()):.0f} puntos o más",
                xy=(e.median(), ax.get_ylim()[1] * .62),
                xytext=(-62, ax.get_ylim()[1] * .74), fontsize=10, color=RED,
                fontweight="bold", linespacing=1.3,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.text(0.98, 0.9, f"en el {subio:.0f} % de los exámenes\nel corte real de 5 años\n"
            "después fue MÁS ALTO que hoy", transform=ax.transAxes, ha="right",
            va="top", fontsize=10.5, color=DARK, fontweight="bold", linespacing=1.35)
    ax.set_xlabel("error de «el agua sigue igual» a 5 años  [puntos]   —   "
                  "negativo = el agua vino peor", fontsize=9.5)
    ax.set_ylabel("cantidad de exámenes", fontsize=9.5)
    ax.set_title("El agua nunca se queda quieta: suponer que sigue igual siempre queda corto",
                 fontsize=12, fontweight="bold", loc="left", pad=8)
    guardar(fig, "fig_sesgo_agua.png")
    return e.median(), subio


def fig_marcador(s):
    """El marcador a 5 anios: los cuatro rivales."""
    fig, ax = plt.subplots(figsize=(10.6, 4.4))
    x = np.arange(len(MODELOS))
    maes = [100 * (s[col] - s.wc_fut).abs().mean() for _, col, _ in MODELOS]
    sesgos = [100 * (s[col] - s.wc_fut).mean() for _, col, _ in MODELOS]
    ax.bar(x - .19, maes, .36, color=[c for _, _, c in MODELOS], label="error típico")
    ax.bar(x + .19, np.abs(sesgos), .36, color=[c for _, _, c in MODELOS],
           alpha=.38, label="sesgo (error sistemático)")
    for i in range(len(MODELOS)):
        ax.text(i - .19, maes[i] + .3, f"{maes[i]:.1f}", ha="center",
                fontsize=10.5, fontweight="bold", color=DARK)
        ax.text(i + .19, abs(sesgos[i]) + .3, f"{sesgos[i]:+.1f}", ha="center",
                fontsize=10.5, fontweight="bold", color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _, _ in MODELOS], fontsize=9.5)
    ax.set_ylabel("puntos de corte de agua, a 5 años", fontsize=9.5)
    ax.legend(fontsize=9.5, ncol=2)
    ax.set_title("El marcador a 5 años: la experiencia ajena da el mejor número",
                 fontsize=12.5, fontweight="bold", loc="left", pad=8)
    guardar(fig, "fig_marcador.png")
    return maes, sesgos


def fig_oseberg(s, campos):
    """El veredicto del campo-ejemplo: cuatro respuestas y lo que paso."""
    row = s[(s.campo == EJEMPLO) & (s.anio == ANIO_EJEMPLO)].iloc[0]
    fig, ax = plt.subplots(figsize=(9.8, 4.3))
    ax.grid(axis="x")
    y = np.arange(4)
    vals = [row.persistencia, row.tendencia, row.analogos, row.wc_fut]
    cols = [GRAY, ORANGE, BLUE, INK]
    noms = ["«El agua sigue igual»", "Su propia tendencia",
            "Análogos de la flota", "LO QUE DE VERDAD PASÓ"]
    ax.barh(y[:3], [100 * v for v in vals[:3]], color=cols[:3], height=.5)
    ax.plot([100 * vals[3]], [3], "D", ms=13, color=INK)
    ax.text(100 * vals[3] + 1.5, 3, f"{100*vals[3]:.0f} %", va="center",
            fontsize=11, fontweight="bold", color=INK)
    ax.hlines(2.44, 100 * row.banda_lo, 100 * row.banda_hi, color=BLUE, lw=3.0,
              alpha=.45)
    ax.plot([100 * row.banda_lo, 100 * row.banda_hi], [2.44, 2.44], "|", ms=13,
            color=BLUE)
    ax.text(100 * row.banda_hi + 1.5, 2.44,
            f"banda: {100*row.banda_lo:.0f} a {100*row.banda_hi:.0f} %",
            va="center", ha="left", fontsize=9.5, color=BLUE, fontweight="bold")
    for i in range(3):
        ax.text(100 * vals[i] - 1.5, y[i], f"{100*vals[i]:.0f} %", va="center",
                ha="right", fontsize=10.5, fontweight="bold", color="white")
    ax.axvline(100 * vals[3], color=INK, lw=1.4, ls=":")
    ax.axvline(50, color=DARK, lw=1.2, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(noms, fontsize=10)
    ax.set_xlim(0, 88)
    ax.set_xlabel("corte de agua de Oseberg en 2018  [%]", fontsize=9.5)
    ax.set_title(f"{EJEMPLO.title()} 2013 → 2018: la experiencia ajena clavó la respuesta",
                 fontsize=12.5, fontweight="bold", loc="left", pad=8)
    guardar(fig, "fig_oseberg.png")
    return row


def fig_vigilancia(s):
    """La lista de vigilancia: arriba casi empate; en los lejanos, solo la
    flota conserva la vista. AUC contado como el juego de las parejas."""
    ev = s[s.wc_hoy < CRUCE].copy()
    ev["evento"] = ev.wc_max >= CRUCE
    lejos = ev[ev.wc_hoy < 0.30]

    # aciertos por anio con presupuesto de 10
    aciertos = {"persistencia": [], "bosque": []}
    ncoh = 0
    for a, ga in ev.groupby("anio"):
        if len(ga) < 15 or ga.evento.sum() < 2:
            continue
        ncoh += 1
        for m in aciertos:
            aciertos[m].append(int(ga.nlargest(10, m).evento.sum()))
    parejas = {}
    for nom, col in [("cerca", "persistencia"), ("bosque", "bosque")]:
        parejas[nom] = (100 * roc_auc_score(ev.evento, ev[col]),
                        100 * roc_auc_score(lejos.evento, lejos[col]))

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))
    ax = axes[0]
    ax.grid(axis="y")
    x = np.arange(2)
    vals = [np.mean(aciertos["persistencia"]), np.mean(aciertos["bosque"])]
    ax.bar(x, vals, 0.5, color=[GRAY, GREEN], alpha=.9)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.12, f"{v:.1f}", ha="center", fontsize=14,
                fontweight="bold", color=DARK)
    ax.set_xticks(x)
    ax.set_xticklabels(["vigilar a los que van\nmás cerca del cruce",
                        "vigilar a los que\nseñala el bosque"], fontsize=10)
    ax.set_ylim(0, 10)
    ax.set_ylabel("aciertos con presupuesto para 10 campos", fontsize=9.5)
    ax.set_title(f"Para ORDENAR la lista: empate ({ncoh} años examinados)",
                 fontsize=11.5, fontweight="bold", loc="left")

    ax = axes[1]
    ax.grid(axis="y")
    x = np.arange(2)
    w = 0.34
    ax.bar(x - w / 2, [parejas["cerca"][0], parejas["cerca"][1]], w,
           color=GRAY, alpha=.9, label="cercanía al cruce")
    ax.bar(x + w / 2, [parejas["bosque"][0], parejas["bosque"][1]], w,
           color=GREEN, alpha=.9, label="bosque de flota")
    ax.axhline(50, color=DARK, lw=1.6, ls="--")
    ax.text(1.44, 50.5, "una moneda\nal aire", fontsize=9, color=DARK,
            ha="right", linespacing=1.2)
    for i, m in enumerate(["cerca", "bosque"]):
        for j in range(2):
            v = parejas[m][j]
            ax.text(j + (i - 0.5) * w, v + 0.8, f"{v:.0f}", ha="center",
                    fontsize=11, fontweight="bold",
                    color=GRAY if m == "cerca" else GREEN)
    ax.set_xticks(x)
    ax.set_xticklabels(["entre todos los campos", "solo los tranquilos hoy\n(corte < 30 %)"],
                       fontsize=10)
    ax.set_ylim(40, 95)
    ax.set_ylabel("parejas bien ordenadas, de cada 100", fontsize=9.5)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("Y también empatan al mirar lejos", fontsize=11.5,
                 fontweight="bold", loc="left", color=DARK)
    fig.tight_layout(w_pad=2.4)
    guardar(fig, "fig_vigilancia.png")
    return ev, lejos, aciertos, parejas, ncoh


def fig_importancias(s, feats):
    """En que se fija el bosque para adivinar el incremento."""
    rf = RandomForestRegressor(400, min_samples_leaf=20, random_state=0,
                               n_jobs=-1)
    rf.fit(s[feats].values, (s.wc_fut - s.wc_hoy).values)
    imp = 100 * pd.Series(rf.feature_importances_, index=feats)
    etiquetas = {
        "wc_hoy": "corte de agua de hoy", "pend": "cuánto subió este año",
        "edad_agua": "años desde que llegó el agua",
        "frac_decl": "cuánto declinó el petróleo",
        "log_pico": "tamaño del campo (pico)",
        "log_np": "petróleo ya producido", "meses": "edad del campo",
        "analogos": "lo que dicen los análogos"}
    imp.index = [etiquetas[f] for f in feats]
    imp = imp.sort_values()
    fig, ax = plt.subplots(figsize=(9.8, 4.2))
    ax.grid(axis="x")
    ax.barh(imp.index, imp.values, color=[GREEN if v == imp.max() else GRAY
                                          for v in imp.values], alpha=.9)
    for i, v in enumerate(imp.values):
        ax.text(v + 0.4, i, f"{v:.0f} %", va="center", fontsize=10,
                fontweight="bold", color=DARK)
    ax.set_xlabel("cuánto usa el bosque cada dato  [%]", fontsize=9.5)
    ax.set_title("En qué se fija el bosque (y en qué casi no)",
                 fontsize=12.5, fontweight="bold", loc="left", pad=8)
    guardar(fig, "fig_importancias.png")
    return imp


def fig_gor_veredicto(s):
    """La misma receta sobre el gas: empata con el tonto. Se dice."""
    gg = s.dropna(subset=["gor_hoy", "gor_ayer", "gor_fut"])
    gg = gg[(gg.gor_hoy > 0) & (gg.gor_ayer > 0) & (gg.gor_fut > 0)].copy()
    lg_hoy, lg_fut = np.log10(gg.gor_hoy), np.log10(gg.gor_fut)
    t3g = []
    lg_o, yg_o, cg_o = lg_hoy.values, lg_fut.values, gg.campo.values
    for i in range(len(gg)):
        m = (np.abs(lg_o - lg_o[i]) < 0.15) & (cg_o != cg_o[i])
        t3g.append(np.median(yg_o[m]) if m.sum() >= 8 else lg_o[i])
    f_pers = (lg_hoy - lg_fut).abs().median()
    f_anal = pd.Series(np.abs(np.array(t3g) - lg_fut.values)).median()
    sube = 100 * (lg_fut > lg_hoy + np.log10(1.5)).mean()

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0),
                             gridspec_kw=dict(width_ratios=[1, 1.1]))
    ax = axes[0]
    ax.grid(axis="y")
    vals = [10 ** f_pers, 10 ** f_anal]
    x = np.arange(2)
    ax.bar(x, vals, 0.5, color=[GRAY, BLUE], alpha=.9)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f"×{v:.2f}", ha="center", fontsize=13,
                fontweight="bold", color=DARK)
    ax.set_xticks(x)
    ax.set_xticklabels(["«el GOR sigue igual»", "análogos de la flota"],
                       fontsize=10)
    ax.set_ylim(1, max(vals) * 1.15)
    ax.set_ylabel("error típico como factor  (×1 = perfecto)", fontsize=9.5)
    ax.set_title("GOR a 5 años: la flota NO le gana al tonto",
                 fontsize=11.5, fontweight="bold", loc="left", color=DARK)

    ax = axes[1]
    ax.set_axis_off(); ax.grid(False)
    ax.text(0.5, 0.80, "¿Por qué el agua sí y el gas no?", ha="center",
            fontsize=12.5, fontweight="bold", color=DARK)
    ax.text(0.5, 0.42,
            "El agua obedece la física del yacimiento,\n"
            "que es parecida entre campos: por eso la\n"
            "experiencia ajena sirve.\n\n"
            "El GOR obedece además a DECISIONES del\n"
            f"operador (inyectar gas, venderlo, quemarlo).\n"
            f"Y aun así: en el {sube:.0f} % de los exámenes el GOR\n"
            "subió más de 1,5 veces en 5 años — vigilarlo\n"
            "sigue siendo obligatorio.",
            ha="center", va="center", fontsize=10.5, color=INK, linespacing=1.45)
    fig.tight_layout(w_pad=1.6)
    guardar(fig, "fig_gor_veredicto.png")
    return 10 ** f_pers, 10 ** f_anal, sube, len(gg)


def fig_lista_hoy(lista):
    """El entregable: la lista de vigilancia de hoy, con banda."""
    top = lista.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    ax.grid(axis="x")
    y = np.arange(len(top))
    for i, (_, row) in enumerate(top.iterrows()):
        if np.isfinite(row.banda_lo):
            ax.hlines(i, 100 * row.banda_lo, 100 * row.banda_hi, color=BLUE,
                      lw=3.0, alpha=.30)
    ax.scatter(100 * top.wc_hoy, y, s=52, color=GRAY, zorder=3,
               label="corte de hoy")
    ax.scatter(100 * top.bosque, y, s=70, color=RED, zorder=4, marker="D",
               label="corte esperado en 5 años (bosque)")
    for i, (_, row) in enumerate(top.iterrows()):
        ax.annotate("", xy=(100 * row.bosque - 1.2, i), xytext=(100 * row.wc_hoy + 1.2, i),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.1))
    ax.axvline(50, color=DARK, lw=1.6, ls="--")
    ax.text(51.0, 0.0, "la línea de la mitad", fontsize=9, color=DARK,
            rotation=90, va="bottom", ha="left")
    ax.set_yticks(y)
    ax.set_yticklabels([c.title() for c in top.campo], fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_xlabel("corte de agua  [%]", fontsize=9.5)
    ax.legend(fontsize=9.5, loc="lower right")
    ax.set_title("La lista de vigilancia de HOY: dato de mayo de 2026, banda de análogos",
                 fontsize=12.5, fontweight="bold", loc="left", pad=8)
    guardar(fig, "fig_lista_hoy.png")


# =============================================================== MAIN =======
def main():
    print("leyendo datos ...")
    d, campos = cargar()
    print(f"  {len(campos)} campos, {len(d)} filas, hasta "
          f"{d.fecha.max().date()}")

    print("\nexamenes anuales y rivales ...")
    s = examenes(campos)
    s, feats = rivales(s)
    print(f"  {len(s)} examenes de {s.campo.nunique()} campos "
          f"({s.anio.min()}-{s.anio.max()})")

    print("\nfamilias ...")
    n_cens = sum(1 for g in campos.values() if censurado(g))
    curvas = curvas_familia(campos)
    nombres, M, asig, centros = familias(curvas)
    rfam = examen_familias(campos, curvas, s)

    print("\ngenerando figuras ...")
    wc_draugen, wor_draugen = fig_draugen(campos)
    oil_hoy, agua_hoy, usd_anio, anio_cruce_flota, n_activos = fig_flota(d, campos)
    fig_como_llega()
    fig_corte_wor(campos)
    fig_gas_senal(campos)
    fig_espagueti(campos, curvas)
    conteo_fam, a50_fam = fig_familias_kmeans(curvas, nombres, M, asig, centros)
    fig_familias_miembros(M, nombres, asig, centros)
    fig_familia_examen(rfam)
    fig_encargo(campos)
    res_h, hs = fig_horizonte(campos)
    sesgo_med, pct_subio = fig_sesgo_agua(s)
    maes, sesgos = fig_marcador(s)
    row_ej = fig_oseberg(s, campos)
    ev, lejos, aciertos, parejas, ncoh = fig_vigilancia(s)
    imp = fig_importancias(s, feats)
    f_pers, f_anal, gor_sube, n_gor = fig_gor_veredicto(s)
    lista, hoy = lista_de_hoy(campos, s, feats)
    fig_lista_hoy(lista)

    b = s.dropna(subset=["banda_lo", "banda_hi"])
    cobertura = 100 * ((b.wc_fut >= b.banda_lo) & (b.wc_fut <= b.banda_hi)).mean()
    ancho = 100 * (b.banda_hi - b.banda_lo).mean()

    print("\n" + "=" * 74)
    print("NUMEROS PARA LAS LAMINAS  (no escribir ninguno a mano)")
    print("=" * 74)
    print(f"campos {len(campos)} | filas {len(d)} | hasta {d.fecha.max().date()}")
    print(f"activos ultimo anio: {n_activos}")
    print(f"EL TELON: el agua solo se reporta desde {TELON.date()}. "
          f"campos censurados (ya venian con agua en 2000): {n_cens}")
    print(f"\nLA FLOTA HOY: petroleo {oil_hoy/1e6:.2f} M bbl/d | "
          f"agua {agua_hoy/1e6:.2f} M bbl/d | {agua_hoy/oil_hoy:.1f} de agua "
          f"por 1 de petroleo")
    print(f"  agua por anio: {agua_hoy*365/1e9:.2f} mil millones de bbl")
    print(f"  a {COSTO_AGUA} USD/bbl: USD {usd_anio/1e9:.1f} mil millones/anio")
    print(f"  el anio en que la plataforma cruzo (mas agua que petroleo): "
          f"{anio_cruce_flota}")
    print(f"\nDRAUGEN hoy: corte {wc_draugen:.0f} % | {wor_draugen:.0f} barriles "
          f"de agua por 1 de petroleo")
    print(f"\nexamenes: {len(s)} ({s.campo.nunique()} campos, "
          f"{s.anio.min()}-{s.anio.max()}) | historia minima {HIST//12} anios, "
          f"horizonte {H//12} anios")
    print(f"\nMARCADOR a 5 anios (error tipico | sesgo):")
    for (nom, col, _), mae_, se_ in zip(MODELOS, maes, sesgos):
        print(f"  {nom:22s} {mae_:5.1f} pp | {se_:+5.1f} pp")
    print(f"  banda P10-P90 analogos: cumple {cobertura:.0f} % (promete 80) | "
          f"ancho medio {ancho:.0f} pp")
    print(f"  la persistencia se quedo corta en el {pct_subio:.0f} % de los "
          f"examenes (mediana {sesgo_med:+.0f} pp)")
    print(f"\nHORIZONTE (error tipico en pp):")
    print(f"  {'anios':>8s} " + " ".join(f"{h//12:>6d}" for h in hs))
    for nom, col, _ in MODELOS:
        print(f"  {col:>12s} " + " ".join(f"{v:6.1f}" for v in res_h[col]))
    print(f"\nFAMILIAS (k=3, {len(nombres)} campos con >= 6 anios de agua):")
    for f in range(3):
        print(f"  {NOMBRES_FAM[f]:6s} n={conteo_fam[f]:2d} | corte a 5a de agua "
              f"{100*centros[f][5]:3.0f} % | cruza 50 %: {a50_fam[f]}")
    print(f"  examen honesto (n={len(rfam)}): error analogos planos "
          f"{100*(rfam.med_plano-rfam.wc_fut).abs().mean():.1f} pp vs por familia "
          f"{100*(rfam.med_familia-rfam.wc_fut).abs().mean():.1f} pp")
    print(f"  banda por familia: ancho "
          f"{100*(rfam.hi_familia-rfam.lo_familia).mean():.0f} pp vs "
          f"{100*(rfam.hi_plano-rfam.lo_plano).mean():.0f} pp plano | cumple "
          f"{100*((rfam.wc_fut>=rfam.lo_familia)&(rfam.wc_fut<=rfam.hi_familia)).mean():.0f} % vs "
          f"{100*((rfam.wc_fut>=rfam.lo_plano)&(rfam.wc_fut<=rfam.hi_plano)).mean():.0f} %")
    print(f"\nVIGILANCIA (cruza 50 % en 5 anios):")
    print(f"  candidatos {len(ev)} | cruzaron {int(ev.evento.sum())} "
          f"({100*ev.evento.mean():.0f} %)")
    print(f"  presupuesto 10 por anio ({ncoh} anios): cercania "
          f"{np.mean(aciertos['persistencia']):.1f} vs bosque "
          f"{np.mean(aciertos['bosque']):.1f}")
    print(f"  parejas de 100 (todos): cercania {parejas['cerca'][0]:.0f} vs "
          f"bosque {parejas['bosque'][0]:.0f}")
    print(f"  parejas de 100 (tranquilos <30 %): cercania "
          f"{parejas['cerca'][1]:.0f} vs bosque {parejas['bosque'][1]:.0f}  "
          f"[{int(lejos.evento.sum())} de {len(lejos)} cruzaron]")
    print(f"\nEJEMPLO {EJEMPLO} enero {ANIO_EJEMPLO}: hoy {100*row_ej.wc_hoy:.0f} % ->")
    print(f"  sigue igual {100*row_ej.persistencia:.0f} % | tendencia "
          f"{100*row_ej.tendencia:.0f} % | analogos {100*row_ej.analogos:.0f} % | "
          f"bosque {100*row_ej.bosque:.0f} %")
    print(f"  banda {100*row_ej.banda_lo:.0f}-{100*row_ej.banda_hi:.0f} % | "
          f"REAL 2018: {100*row_ej.wc_fut:.0f} %")
    print(f"\nGOR a 5 anios (n={n_gor}): tonto x{f_pers:.2f} vs analogos "
          f"x{f_anal:.2f} | GOR subio >1.5x en {gor_sube:.0f} % de examenes")
    print(f"\nEN QUE SE FIJA EL BOSQUE (top):")
    for nom, v in imp.sort_values(ascending=False).head(4).items():
        print(f"  {nom:32s} {v:4.0f} %")
    print(f"\nLA LISTA DE HOY (corte < 50 %, petroleo > 5.000 bbl/d):")
    print(f"  candidatos {len(lista)}")
    cols = ["campo", "wc_hoy", "pend", "bosque", "banda_lo", "banda_hi", "oil_hoy"]
    t = lista[cols].head(12).copy()
    for c in ["wc_hoy", "pend", "bosque", "banda_lo", "banda_hi"]:
        t[c] = (100 * t[c]).round(0)
    t["oil_hoy"] = (t.oil_hoy / 1000).round(0)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()

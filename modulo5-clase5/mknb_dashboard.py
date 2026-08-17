"""
Machine Learning for Petroleum Engineers Using Python
Modulo 5 - Clase 5: genera el cuaderno PARALELO del tablero de
explicabilidad (explainerdashboard), pensado para Google Colab.

Es el companero "para llevar" del cuaderno principal: mismo dato, mismo
modelo (los hiperparametros ganadores de la malla de la clase), pero en
vez de laminas estaticas monta un tablero interactivo.

No se ejecuta con nbconvert: su ultima celda levanta un servidor Dash y se
queda sirviendo. La construccion del explainer y del dashboard (la parte
pesada) esta verificada localmente.

Uso:  python3 mknb_dashboard.py
"""

import json

NOMBRE = "Modulo5_Clase5_Tablero_Explicabilidad.ipynb"

celdas = []


def _lineas(texto):
    ls = texto.strip("\n").split("\n")
    return [l + "\n" for l in ls[:-1]] + ls[-1:]


def md(texto):
    celdas.append({"cell_type": "markdown", "metadata": {},
                   "source": _lineas(texto)})


def code(texto):
    celdas.append({"cell_type": "code", "execution_count": None,
                   "metadata": {}, "outputs": [],
                   "source": _lineas(texto)})


md(r"""
# Módulo 5 · Clase 5 — Tablero de Explicabilidad (para llevar)

**Machine Learning for Petroleum Engineers Using Python** · SLB Ecuador / UDLA

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmosquerat/slb-diplomado/blob/main/modulo5-clase5/Modulo5_Clase5_Tablero_Explicabilidad.ipynb)

Este es el compañero interactivo del cuaderno de la clase. En la clase
explicamos los pronósticos de vida útil remanente (RUL) con valores de
Shapley, una figura a la vez. Aquí montamos **todo eso como un tablero web
navegable** con la librería `explainerdashboard`: importancias, el porqué de
cada unidad, dependencias sensor a sensor y un simulador *qué-pasa-si* —
sin escribir una línea de Dash.

**Instrucciones (Colab):** `Entorno de ejecución → Ejecutar todo`. La
instalación y el cálculo de Shapley tardan unos 3–5 minutos; el tablero
aparece incrustado al final.
""")

code(r"""
%pip install -q explainerdashboard shap
""")

md(r"""
## El dato y el modelo de la clase

El mismo CSV curado del repositorio (la flota C-MAPSS de NASA: 100
turbomáquinas corridas hasta la falla) y el mismo modelo ganador: gradient
boosting con los hiperparámetros que eligió la malla de la clase
(`learning_rate=0.03`, profundidad libre — ver el cuaderno principal, Paso 7).
""")

code(r"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

URL = ("https://raw.githubusercontent.com/cmosquerat/slb-diplomado/"
       "main/datos/flota_turbomaquinas_nasa.csv")
d = pd.read_csv(URL)
vida = d.groupby("unidad")["ciclo"].max().rename("vida")
d = d.join(vida, on="unidad")
d["rul"] = d["vida"] - d["ciclo"]

SENSORES = ["T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc",
            "epr", "Ps30", "phi", "NRf", "NRc", "BPR", "farB", "htBleed",
            "Nf_dmd", "PCNfR_dmd", "W31", "W32"]
vivos = [c for c in SENSORES if d[c].std() > 1e-6]

g = d.groupby("unidad")
feats = {}
for c in vivos:
    m = g[c].transform(lambda s: s.rolling(20, min_periods=5).mean())
    feats[c] = d[c]
    feats[c + "_m"] = m
    feats[c + "_d"] = m - m.groupby(d["unidad"]).shift(10)
feats["edad"] = d["ciclo"]
X = pd.DataFrame(feats).fillna(0)
y = d["rul"]

rng = np.random.RandomState(0)
test_u = np.sort(rng.choice(vida.index.to_numpy(), 25, replace=False))
tr = ~d["unidad"].isin(test_u)
te = d["unidad"].isin(test_u)

modelo = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03,
                                       max_depth=None, random_state=0)
modelo.fit(X[tr], y[tr])
print("modelo entrenado con", tr.sum(), "filas de 75 unidades")
""")

md(r"""
## El explainer

`RegressionExplainer` calcula de una vez todo lo que el tablero necesita:
valores de Shapley por fila, importancias por permutación, dependencias.
Le damos una **muestra de 800 fotos de las 25 unidades apartadas** (el
modelo jamás las vio), cada una etiquetada como `unidad · ciclo` para poder
buscarlas por nombre en el tablero.
""")

code(r"""
from explainerdashboard import RegressionExplainer, ExplainerDashboard

rng2 = np.random.RandomState(2)
sub = np.sort(rng2.choice(np.where(te)[0], 800, replace=False))
etiquetas = [f"unidad {u} · ciclo {c}"
             for u, c in zip(d.unidad.iloc[sub], d.ciclo.iloc[sub])]

DESCRIPCIONES = {"edad": "ciclos corridos desde la instalación "
                         "(la única arma del calendario)"}
for c in vivos:
    DESCRIPCIONES[c] = f"sensor {c}: valor del ciclo actual"
    DESCRIPCIONES[c + "_m"] = (f"sensor {c}: media de los últimos 20 "
                               "ciclos (la foto sin temblor)")
    DESCRIPCIONES[c + "_d"] = (f"sensor {c}: pendiente de esa media "
                               "(¿se está torciendo?)")

explainer = RegressionExplainer(
    modelo,
    X.iloc[sub].reset_index(drop=True),
    y.iloc[sub].reset_index(drop=True),
    idxs=etiquetas,
    units="ciclos",
    descriptions=DESCRIPCIONES,
)
print("explainer listo:", len(sub), "pronósticos explicados")
""")

md(r"""
## Cómo leer el tablero (antes de abrirlo)

| pestaña | qué muestra | en el idioma de la clase |
|---|---|---|
| **Feature Importances** | qué columnas mueven más los pronósticos | la vista *global* (el beeswarm de la clase) |
| **Regression Stats** | el examen del modelo en las unidades apartadas | el marcador — miren el MAE |
| **Individual Predictions** | el porqué de UNA foto, pieza por pieza | el *waterfall* de Shapley — el párrafo de la orden de trabajo |
| **What if...** | mover una variable y ver el pronóstico cambiar | el simulador — con letra chica, ver abajo |
| **Feature Dependence** | cómo responde el pronóstico a cada sensor | la física redescubierta (T50 sube → vida baja) |

**La letra chica, que no se les olvide:**

- Shapley reparte el pronóstico **del modelo**, no la culpa física. Sensores
  correlacionados se roban crédito entre sí (la ablación de la clase lo
  demostró con la edad).
- El *What if* mueve una variable **con las demás quietas** — cosa que la
  física real no permite: en una máquina degradándose todo se mueve junto.
  Úsenlo para entender al modelo, no para simular la máquina.
""")

code(r"""
db = ExplainerDashboard(
    explainer,
    title="Flota RUL — ¿Cuánta vida le queda a la bomba?",
    shap_interaction=False,     # lo pesado, fuera: no lo necesitamos
    decision_trees=False,       # HistGBR no expone árboles individuales
    show_metrics=["mean-absolute-error", "root-mean-squared-error",
                  "R-squared"],  # el MAPE explota cuando el RUL real es 0
    mode="inline",              # incrustado en el cuaderno (Colab)
)
db.run(port=8050)
""")

md(r"""
---
**Para correrlo fuera de Colab** (máquina propia): cambien
`mode="inline"` por `mode="external"` y abran `http://localhost:8050`.

**Referencia:** Dijk, O. *explainerdashboard* —
[explainerdashboard.readthedocs.io](https://explainerdashboard.readthedocs.io).
El modelo, el dato y los números de este tablero son exactamente los del
cuaderno principal de la Clase 5; si algo no coincide, es un error nuestro
y queremos saberlo.
""")

nb = {
    "cells": celdas,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"provenance": []},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
with open(NOMBRE, "w") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"escrito {NOMBRE} ({len(celdas)} celdas)")

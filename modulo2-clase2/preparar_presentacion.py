"""
Machine Learning for Petroleum Engineers Using Python
Modulo 2 - Clase 2: inserta los datos en presentacion.html.

La presentación dibuja sus gráficos en vivo con los datos reales de
datos/volve_pozos_diario.csv. Este script los escribe dentro del HTML, entre
los marcadores /*DATOS*/ y /*FIN_DATOS*/. Se puede correr las veces que haga falta.

Uso:  python3 preparar_presentacion.py
"""

import json
import re
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
HTML = AQUI / "presentacion.html"
df = pd.read_csv(AQUI.parent / "datos" / "volve_pozos_diario.csv")

# Veinte días abiertos de un pozo, en miles de barriles, para armar el histograma paso a paso
abiertos = df[(df["horas"] > 0) & (df["pozo"] == "Pozo 2") & (df["petroleo"] < 25000)]
hb20 = (abiertos["petroleo"].sample(20, random_state=4) / 1000).round(0).astype(int).tolist()

datos = {
    "pz": (df["pozo"].str[-1].astype(int) - 1).tolist(),
    "d": df["dias"].tolist(),
    "h": df["horas"].tolist(),
    "p": df["petroleo"].astype(int).tolist(),
    "g": df["gas"].astype(int).tolist(),
    "a": df["agua"].astype(int).tolist(),
    "hb20": hb20,
}

texto = HTML.read_text(encoding="utf-8")
nuevo, n = re.subn(r"/\*DATOS\*/.*?/\*FIN_DATOS\*/",
                   lambda _: "/*DATOS*/" + json.dumps(datos, separators=(",", ":")) + "/*FIN_DATOS*/",
                   texto, flags=re.S)
assert n == 1, "no encontré los marcadores /*DATOS*/ ... /*FIN_DATOS*/"
HTML.write_text(nuevo, encoding="utf-8")
print(f"datos insertados: {len(df)} filas; hb20 = {hb20}")
print(f"tamaño de presentacion.html: {HTML.stat().st_size / 1024:.0f} KB")

"""
Machine Learning for Petroleum Engineers Using Python
Modulo 2 - Clase 1: inserta los datos en presentacion.html.

La presentación dibuja sus gráficos en vivo con los datos reales. Este script
los lee de datos/insurance.csv (y una muestra del registro del pozo 15/9-19 de
Volve para la portada) y los escribe dentro del HTML, entre los marcadores
/*DATOS*/ y /*FIN_DATOS*/. Se puede correr las veces que haga falta.

Uso:  python3 preparar_presentacion.py
"""

import json
import re
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
DATOS = AQUI.parent / "datos"
HTML = AQUI / "presentacion.html"

df = pd.read_csv(DATOS / "insurance.csv")
reg = pd.read_csv(DATOS / "volve_registros.csv")

# Parte del costo total que pone cada décimo de clientes (igual que figuras.py)
dec = pd.qcut(df["charges"].rank(method="first"), 10, labels=False) + 1
deciles = (df["charges"].groupby(dec).sum() / df["charges"].sum() * 100).round(1)

muestra = reg.iloc[:: max(1, len(reg) // 160)].head(160)
codigo_region = {"northeast": 0, "northwest": 1, "southwest": 2, "southeast": 3}

datos = {
    "charges": [round(v, 2) for v in df["charges"]],
    "age": df["age"].tolist(),
    "bmi": [round(v, 3) for v in df["bmi"]],
    "children": df["children"].tolist(),
    "smoker": (df["smoker"] == "yes").astype(int).tolist(),
    "sex": (df["sex"] == "male").astype(int).tolist(),
    "region": df["region"].map(codigo_region).tolist(),
    "cmin": round(df["charges"].min(), 2),
    "cmax": round(df["charges"].max(), 2),
    "age20": df["age"].head(20).tolist(),
    "deciles": deciles.tolist(),
    "GR": [round(v, 1) for v in muestra["GR"]],
    "NEU": [round(v, 1) for v in muestra["NEU"]],
    "DEN": [round(v, 3) for v in muestra["DEN"]],
}

texto = HTML.read_text(encoding="utf-8")
nuevo, n = re.subn(r"/\*DATOS\*/.*?/\*FIN_DATOS\*/",
                   lambda _: "/*DATOS*/" + json.dumps(datos, separators=(",", ":")) + "/*FIN_DATOS*/",
                   texto, flags=re.S)
assert n == 1, "no encontré los marcadores /*DATOS*/ ... /*FIN_DATOS*/"
HTML.write_text(nuevo, encoding="utf-8")
print(f"datos insertados: {len(df)} filas, deciles {deciles.tolist()}")
print(f"tamaño de presentacion.html: {HTML.stat().st_size / 1024:.0f} KB")

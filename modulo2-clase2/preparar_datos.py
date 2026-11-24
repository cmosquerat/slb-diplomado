"""Arma la tabla de la clase 2: un pozo de petróleo en un día por fila.

Entrada: datos/volve_produccion.csv (producción diaria real del campo Volve,
Equinor, datos abiertos). Salida: datos/volve_pozos_diario.csv

- Solo los 5 pozos que producen petróleo. Se llaman Pozo 1 a Pozo 5, por la
  fecha en que empezaron a producir.
- Petróleo y agua en barriles por día; gas en miles de pies cúbicos por día.
- dias: días desde que ese pozo empezó a producir.
- La tabla NO se limpia: trae días con el pozo cerrado, agua negativa y días
  de 25 horas. Se limpia en la clase 3.
"""
from pathlib import Path
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
BARRILES = 6.2898            # barriles en un metro cúbico
MIL_PIES3 = 35.3147 / 1000   # miles de pies cúbicos en un metro cúbico

d = pd.read_csv(RAIZ / "datos" / "volve_produccion.csv", parse_dates=["fecha"])
d = d.dropna(subset=["oil"])                    # sin los pozos que solo inyectan agua
d = d[d["pozo"] != "15/9-F-5"]                  # inyector con pocos días de petróleo
inicio = d.groupby("pozo")["fecha"].min().sort_values()
nombre = {p: f"Pozo {i + 1}" for i, p in enumerate(inicio.index)}

t = pd.DataFrame({
    "fecha": d["fecha"].dt.strftime("%Y-%m-%d"),
    "pozo": d["pozo"].map(nombre),
    "dias": (d["fecha"] - d["pozo"].map(inicio)).dt.days,
    "horas": d["horas"].round(1),
    "petroleo": (d["oil"] * BARRILES).round(0),
    "gas": (d["gas"] * MIL_PIES3).round(0),
    "agua": (d["agua"] * BARRILES).round(0),
})
t = t.sort_values(["pozo", "fecha"]).reset_index(drop=True)
t.to_csv(RAIZ / "datos" / "volve_pozos_diario.csv", index=False)
print(t.shape, nombre)
print(t.head())
print("vacíos:", t.isna().sum().to_dict())

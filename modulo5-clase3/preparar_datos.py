"""
Machine Learning for Petroleum Engineers Using Python
Modulo 5 - Clase 3: prepara datos/campos_noruega_declinacion.csv

Fuente: Sokkeldirektoratet (Norwegian Offshore Directorate), factpages,
tabla "Production figures - monthly by field". Datos abiertos.
Es la misma fuente del campo Volve que se uso en el Modulo 1 y en la Clase 1.

Que hace:
  - baja la produccion mensual de TODOS los campos de la plataforma noruega
  - convierte de millones de Sm3 a barriles por dia
  - alinea cada campo desde su PICO de produccion, porque la declinacion
    empieza ahi y no en el primer barril: asi los campos se pueden comparar
  - se queda con los campos que tienen al menos 15 anios despues del pico,
    que son los unicos donde se puede verificar un pronostico a 12 anios

El archivo que sale tiene una fila por campo y por mes desde el pico.

Uso:  python3 preparar_datos.py
"""

import io
import urllib.request

import numpy as np
import pandas as pd

URL = (
    "https://factpages.sodir.no/public?/Factpages/external/tableview/"
    "field_production_monthly&rs:Command=Render&rc:Toolbar=false"
    "&rc:Parameters=f&IpAddress=not_used&CultureCode=en"
    "&rs:Format=CSV&Top100=false"
)

BBL = 6.2898          # barriles en un metro cubico estandar
MIN_ANIOS = 15        # anios de historia despues del pico que exigimos
SALIDA = "../datos/campos_noruega_declinacion.csv"


def main():
    print("descargando de sodir.no ...")
    with urllib.request.urlopen(URL, timeout=300) as r:
        crudo = r.read().decode("utf-8-sig")
    d = pd.read_csv(io.StringIO(crudo))
    d.columns = [c.strip() for c in d.columns]

    d = d.rename(columns={
        "prfInformationCarrier": "campo",
        "prfYear": "anio",
        "prfMonth": "mes",
        "prfPrdOilNetMillSm3": "oil_msm3",
        "prfPrdGasNetBillSm3": "gas_gsm3",
        "prfPrdProducedWaterInFieldMillSm3": "agua_msm3",
    })
    d["fecha"] = pd.to_datetime(dict(year=d.anio, month=d.mes, day=1))
    d["dias"] = d.fecha.dt.days_in_month
    d["oil_bpd"] = d.oil_msm3 * 1e6 * BBL / d.dias
    d["agua_bpd"] = d.agua_msm3 * 1e6 * BBL / d.dias

    partes = []
    for campo, g in d.groupby("campo"):
        g = g.sort_values("fecha").reset_index(drop=True)
        if g.oil_bpd.sum() <= 0:
            continue

        # el pico: el mes con mayor produccion, suavizada medio anio para que
        # un mes bueno suelto no se lo lleve
        suave = g.oil_bpd.rolling(6, center=True, min_periods=1).mean()
        if suave.notna().sum() == 0:
            continue
        g = g.iloc[int(suave.idxmax()):].reset_index(drop=True)

        if len(g) < MIN_ANIOS * 12 or g.oil_bpd.iloc[:12].sum() <= 0:
            continue

        g["mes_desde_pico"] = np.arange(len(g))
        partes.append(g[["campo", "fecha", "mes_desde_pico", "dias",
                         "oil_bpd", "agua_bpd"]])

    d = pd.concat(partes, ignore_index=True)
    d["oil_bpd"] = d.oil_bpd.round(1)
    d["agua_bpd"] = d.agua_bpd.round(1)
    d["fecha"] = d.fecha.dt.strftime("%Y-%m-%d")
    d.to_csv(SALIDA, index=False)

    print(f"\nescrito {SALIDA}: {len(d)} filas, {d.campo.nunique()} campos")
    res = d.groupby("campo").agg(
        meses=("mes_desde_pico", "size"),
        pico_bpd=("oil_bpd", "max"),
        desde=("fecha", "min"),
    ).sort_values("pico_bpd", ascending=False)
    print(f"\nlos 10 campos mas grandes:\n{res.head(10).round(0).to_string()}")
    print(f"\naños de historia despues del pico: "
          f"mediana {res.meses.median()/12:.0f}, maximo {res.meses.max()/12:.0f}")


if __name__ == "__main__":
    main()

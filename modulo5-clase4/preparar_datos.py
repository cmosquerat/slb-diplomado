"""
Machine Learning for Petroleum Engineers Using Python
Modulo 5 - Clase 4: prepara datos/campos_noruega_agua.csv

Fuente: Sokkeldirektoratet (Norwegian Offshore Directorate), factpages,
tabla "Production figures - monthly by field". Datos abiertos.
Es la misma fuente de la Clase 3 -- pero esta vez nos quedamos con las TRES
corrientes que reporta cada campo: petroleo, gas y agua. La Clase 3 uso solo
el petroleo; esta clase mira lo que el agua y el gas vienen avisando.

Que hace:
  - baja la produccion mensual de TODOS los campos de la plataforma noruega
  - convierte petroleo y agua de millones de Sm3/mes a barriles por dia,
    y el gas de miles de millones de Sm3/mes a miles de Sm3 por dia
  - pone en cero los meses con valores NEGATIVOS (son ajustes contables del
    regulador: ~30 meses en todo el archivo)
  - se queda con los campos de petroleo de verdad (pico >= 5.000 bbl/dia)
    con al menos un anio de historia -- los muy nuevos tambien entran,
    porque la lista de vigilancia de hoy los necesita
  - NO alinea ni recorta nada mas: el corte de agua, el WOR y el GOR se
    calculan en el cuaderno, porque calcularlos es parte de la clase

ADVERTENCIA que la clase usa como leccion de EDA: el agua producida solo se
reporta desde ENERO DE 2000. Antes de esa fecha la columna esta en cero para
casi todos los campos (verificado: en 1998 producian 35 campos y solo 1
reportaba agua; en 2000 la reportan todos). Un campo viejo no "estrena" agua
en 2000: simplemente ahi empezaron a contarla. Todo analisis del agua tiene
que arrancar en 2000.

El archivo que sale tiene una fila por campo y por mes de calendario.

Uso:  python3 preparar_datos.py
"""

import io
import urllib.request

import pandas as pd

URL = (
    "https://factpages.sodir.no/public?/Factpages/external/tableview/"
    "field_production_monthly&rs:Command=Render&rc:Toolbar=false"
    "&rc:Parameters=f&IpAddress=not_used&CultureCode=en"
    "&rs:Format=CSV&Top100=false"
)

BBL = 6.2898          # barriles en un metro cubico estandar
PICO_MIN = 5000       # bbl/dia: por debajo de esto no es un campo de petroleo
MESES_MIN = 12        # historia minima para entrar al archivo
SALIDA = "../datos/campos_noruega_agua.csv"


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
    d["oil_bpd"] = (d.oil_msm3 * 1e6 * BBL / d.dias).clip(lower=0)
    d["agua_bpd"] = (d.agua_msm3 * 1e6 * BBL / d.dias).clip(lower=0)
    d["gas_ksm3d"] = (d.gas_gsm3 * 1e9 / 1e3 / d.dias).clip(lower=0)
    negativos = int((d.oil_msm3 < 0).sum() + (d.agua_msm3 < 0).sum()
                    + (d.gas_gsm3 < 0).sum())
    print(f"meses con valores negativos puestos en cero: {negativos}")

    partes = []
    for campo, g in d.groupby("campo"):
        g = g.sort_values("fecha").reset_index(drop=True)
        if g.oil_bpd.max() < PICO_MIN or len(g) < MESES_MIN:
            continue
        partes.append(g[["campo", "fecha", "dias",
                         "oil_bpd", "agua_bpd", "gas_ksm3d"]])

    d = pd.concat(partes, ignore_index=True)
    d["oil_bpd"] = d.oil_bpd.round(1)
    d["agua_bpd"] = d.agua_bpd.round(1)
    d["gas_ksm3d"] = d.gas_ksm3d.round(2)
    d["fecha"] = d.fecha.dt.strftime("%Y-%m-%d")
    d.to_csv(SALIDA, index=False)

    print(f"\nescrito {SALIDA}: {len(d)} filas, {d.campo.nunique()} campos, "
          f"hasta {d.fecha.max()}")

    u = d[pd.to_datetime(d.fecha) > pd.to_datetime(d.fecha.max())
          - pd.DateOffset(months=12)]
    f = u.groupby("campo")[["oil_bpd", "agua_bpd"]].mean()
    f = f[f.sum(axis=1) > 100]
    print(f"\nla flota en el ultimo anio ({len(f)} campos activos):")
    print(f"  petroleo {f.oil_bpd.sum()/1e6:.2f} millones de bbl/dia")
    print(f"  agua     {f.agua_bpd.sum()/1e6:.2f} millones de bbl/dia")
    top = f.assign(corte=100 * f.agua_bpd / (f.agua_bpd + f.oil_bpd)) \
           .sort_values("agua_bpd", ascending=False)
    print("\nlos 10 que mas agua levantan hoy:")
    print(top.head(10).round(0).to_string())


if __name__ == "__main__":
    main()

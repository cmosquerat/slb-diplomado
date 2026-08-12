"""
Machine Learning for Petroleum Engineers Using Python
Modulo 5 - Clase 2: prepara datos/pozos_3w_incrustacion.csv

Fuente: 3W Dataset v2.0.0 - Petrobras
  https://github.com/petrobras/3W
  Datos bajo licencia CC BY 4.0. Codigo del toolkit bajo Apache-2.0.
  Referencia: Vargas et al. (2019), "A realistic and public dataset with rare
  undesirable real events in oil wells", Journal of Petroleum Science and
  Engineering, 181, 106223.

Que trae: instancias REALES (no simuladas ni dibujadas) de la clase 7 del 3W,
"Scaling in PCK" -- incrustacion en el choke de produccion. Cada instancia es
una grabacion continua de sensores de un pozo submarino a 1 Hz, etiquetada
segundo a segundo por especialistas de Petrobras:

    class = 0     operacion normal
    class = 107   transitorio del evento 7 (la incrustacion progresando)
    class = 7     estado de falla ya establecido

Que hace este script:
  - baja 12 instancias reales de la clase 7
  - remuestrea de 1 Hz a una muestra cada 30 s (la incrustacion tarda horas;
    a 1 Hz solo se agrega ruido y peso)
  - convierte presiones de Pa a bar y deja las temperaturas en grados C
  - traduce nombres de columna, conservando la etiqueta del tag industrial
    en el diccionario de abajo (y en la lamina "Las variables, una a una")

Los pozos 1, 21, 22 y 24 tienen los cinco sensores y son el material de clase.
El pozo 23 tiene solo dos sensores vivos: es la practica.

Uso:  python3 preparar_datos.py
"""

import io
import urllib.request

import pandas as pd

RAW = "https://raw.githubusercontent.com/petrobras/3W/main/dataset/7/"

# Instancias reales de la clase 7. El nombre codifica pozo y fecha de inicio.
INSTANCIAS = [
    "WELL-00001_20170226130146.parquet",
    "WELL-00021_20180611011218.parquet",
    "WELL-00021_20190403013307.parquet",
    "WELL-00022_20181101193049.parquet",
    "WELL-00023_20181014121654.parquet",
    "WELL-00023_20181030045054.parquet",
    "WELL-00024_20160825100303.parquet",
    "WELL-00024_20160828150330.parquet",
    "WELL-00024_20160830190958.parquet",
    "WELL-00024_20160911063934.parquet",
    "WELL-00024_20160920153331.parquet",
    "WELL-00024_20160923141645.parquet",
]

# tag industrial -> (nombre en el CSV, factor de conversion)
SENSORES = {
    "P-MON-CKP": ("p_antes_choke", 1e-5),   # Pa -> bar. Presion aguas arriba del choke
    "P-TPT": ("p_arbol", 1e-5),             # Pa -> bar. Presion en el arbol submarino
    "P-ANULAR": ("p_anular", 1e-5),         # Pa -> bar. Presion en el espacio anular
    "P-JUS-CKGL": ("p_gaslift", 1e-5),      # Pa -> bar. Presion del gas de levantamiento
    "T-JUS-CKP": ("t_despues_choke", 1.0),  # grados C. Temperatura aguas abajo del choke
}

ETIQUETAS = {0: "normal", 107: "transitorio", 7: "falla"}

PASO = 30          # segundos entre muestras
SALIDA = "../datos/pozos_3w_incrustacion.csv"


def main():
    partes = []
    for nombre in INSTANCIAS:
        print(f"  bajando {nombre} ...", flush=True)
        with urllib.request.urlopen(RAW + nombre, timeout=300) as r:
            d = pd.read_parquet(io.BytesIO(r.read()))

        d = d.iloc[::PASO].reset_index(drop=True)
        out = pd.DataFrame()
        out["pozo"] = [nombre.split("_")[0]] * len(d)
        out["instancia"] = nombre.replace(".parquet", "")
        out["t_min"] = (d.index * PASO / 60.0).round(1)
        for tag, (col, k) in SENSORES.items():
            # 5 decimales, no 3: la presion del gas lift varia ~0.01 bar y con
            # menos resolucion el redondeo se come justo la senal que interesa
            out[col] = (d[tag] * k).round(5) if tag in d.columns else pd.NA
        out["etiqueta"] = d["class"].map(ETIQUETAS)
        out = out[out.etiqueta.notna()]
        partes.append(out)

    d = pd.concat(partes, ignore_index=True)
    # el arranque de cada grabacion puede venir sin etiqueta: se descarta y se
    # vuelve a poner el reloj en cero, para que t_min sea "minutos de grabacion"
    d["t_min"] = (d.t_min - d.groupby("instancia").t_min.transform("min")).round(1)
    d.to_csv(SALIDA, index=False)

    print(f"\nescrito {SALIDA}: {len(d)} filas, {d.pozo.nunique()} pozos")
    print(d.groupby("pozo").agg(
        instancias=("instancia", "nunique"),
        horas=("t_min", lambda s: round(len(s) * PASO / 3600, 1)),
        sensores=("p_arbol", lambda s: 5 - int(s.isna().all()) * 3),
    ).to_string())
    print("\netiquetas:")
    print(d.etiqueta.value_counts().to_string())


if __name__ == "__main__":
    main()

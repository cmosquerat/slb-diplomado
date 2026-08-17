"""
Machine Learning for Petroleum Engineers Using Python
Modulo 5 - Clase 5: prepara datos/flota_turbomaquinas_nasa.csv

Fuente: NASA C-MAPSS, "Turbofan Engine Degradation Simulation Data Set"
  Saxena, A., Goebel, K., Simon, D., Eklund, N. (2008). "Damage Propagation
  Modeling for Aircraft Engine Run-to-Failure Simulation", PHM 2008.
  Publicado por el Prognostics Center of Excellence (PCoE) de NASA Ames.
  Dominio publico (obra del gobierno de EE.UU.).

  Descarga oficial (S3 del PCoE):
    https://phm-datasets.s3.amazonaws.com/NASA/
      6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip
  Espejo verificado (mismo archivo train_FD001.txt, byte a byte):
    https://raw.githubusercontent.com/jiaxiang-cheng/
      PyTorch-LSTM-for-RUL-Prediction/main/CMAPSSData/train_FD001.txt

Que trae: el subconjunto FD001 -- una flota de 100 turbomaquinas "gemelas"
(mismo diseno, distinto desgaste inicial de fabrica), cada una operada en
condicion constante DESDE SANA HASTA LA FALLA. 21 sensores termodinamicos
por ciclo de operacion. Es el dataset de mantenimiento predictivo mas usado
del mundo desde el PHM Data Challenge 2008.

Por que lo usamos para hablar de bombas ESP: no existe dato publico de
fallas de ESP (se verificaron ocho fuentes; el RIFTS del consorcio es
cerrado). Una ESP es una turbomaquina centrifuga multietapa: ejes que giran,
degradacion progresiva, sensores de presion y temperatura. La sustitucion
se declara en la lamina de acotacion de la clase, igual que la Clase 2
declaro "un choke, no una bomba".

Que hace este script:
  - baja el zip oficial de NASA (con espejo de respaldo)
  - extrae train_FD001.txt (20.631 filas, 100 unidades)
  - nombra las columnas con el tag industrial de cada sensor (T24, Ps30, ...)
  - NO calcula el RUL: construir el reloj es material de la clase
  - escribe ../datos/flota_turbomaquinas_nasa.csv

Uso:  python3 preparar_datos.py
"""

import io
import os
import zipfile
import urllib.request

import pandas as pd

URL_NASA = ("https://phm-datasets.s3.amazonaws.com/NASA/"
            "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip")
URL_ESPEJO = ("https://raw.githubusercontent.com/jiaxiang-cheng/"
              "PyTorch-LSTM-for-RUL-Prediction/main/CMAPSSData/"
              "train_FD001.txt")

# Columnas del archivo original, en orden. Los tags son los del modelo
# C-MAPSS (Saxena et al. 2008, tabla 2); la traduccion al espanol vive en
# figuras.py y en la lamina "Las Variables, Una a Una".
COLUMNAS = ["unidad", "ciclo", "op_1", "op_2", "op_3",
            "T2", "T24", "T30", "T50",          # temperaturas por estacion
            "P2", "P15", "P30",                 # presiones por estacion
            "Nf", "Nc",                         # velocidades de eje
            "epr", "Ps30", "phi",               # relacion de presion, etc.
            "NRf", "NRc", "BPR", "farB",
            "htBleed", "Nf_dmd", "PCNfR_dmd",
            "W31", "W32"]                       # flujos de refrigeracion


def bajar_fd001():
    """Devuelve el contenido de train_FD001.txt como bytes."""
    try:
        print("bajando el zip oficial de NASA (~12 MB) ...")
        with urllib.request.urlopen(URL_NASA, timeout=120) as r:
            zip_ext = zipfile.ZipFile(io.BytesIO(r.read()))
        # el zip de NASA trae adentro CMAPSSData.zip; a veces directo
        for nombre in zip_ext.namelist():
            if nombre.endswith("train_FD001.txt"):
                return zip_ext.read(nombre)
            if nombre.endswith(".zip"):
                interno = zipfile.ZipFile(io.BytesIO(zip_ext.read(nombre)))
                for n2 in interno.namelist():
                    if n2.endswith("train_FD001.txt"):
                        return interno.read(n2)
        raise FileNotFoundError("train_FD001.txt no aparece en el zip")
    except Exception as e:
        print(f"  fuente oficial fallo ({e}); usando el espejo ...")
        with urllib.request.urlopen(URL_ESPEJO, timeout=120) as r:
            return r.read()


def main():
    crudo = bajar_fd001()
    df = pd.read_csv(io.BytesIO(crudo), sep=r"\s+", header=None,
                     names=COLUMNAS)

    # sanidad: la flota completa, cada unidad desde el ciclo 1 hasta morir
    assert df["unidad"].nunique() == 100, "deberian ser 100 unidades"
    assert len(df) == 20631, "deberian ser 20.631 filas"
    assert (df.groupby("unidad")["ciclo"].min() == 1).all()

    destino = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "datos", "flota_turbomaquinas_nasa.csv")
    df.to_csv(destino, index=False)

    vidas = df.groupby("unidad")["ciclo"].max()
    print(f"escrito {os.path.normpath(destino)}")
    print(f"  {len(df):,} filas, {df.unidad.nunique()} unidades, "
          f"{len(COLUMNAS)} columnas")
    print(f"  vidas: min {vidas.min()}, mediana {vidas.median():.0f}, "
          f"max {vidas.max()} ciclos")


if __name__ == "__main__":
    main()

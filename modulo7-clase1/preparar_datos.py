"""
Machine Learning for Petroleum Engineers Using Python
Modulo 7 - Clase 1: prepara los cuatro datasets de la clase en datos/

Fuente: los conjuntos de practica de seaborn (github.com/mwaskom/seaborn-data,
licencia del proyecto seaborn, BSD-3). Son datos publicos clasicos:

  taxis       viajes de taxi de Nueva York (TLC Trip Record Data)
  mpg         consumo de vehiculos 1970-82 (UCI / StatLib)
  car_crashes accidentes viales por estado (NHTSA via FiveThirtyEight)
  diamonds    precios y medidas de diamantes (ggplot2)

Por que se copian al repositorio del curso en vez de cargarlos de seaborn
en vivo: para que la clase no dependa de un tercero en el momento de
dictarla. El cuaderno los carga por URL raw de ESTE repositorio.

Que hace: los baja de la fuente y los escribe TAL CUAL (sin transformar)
en ../datos/. Se corre una vez.

Uso:  python3 preparar_datos.py
"""

import hashlib
import urllib.request

BASE = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/"

ARCHIVOS = {
    "taxis.csv": "../datos/taxis_nyc.csv",
    "mpg.csv": "../datos/vehiculos_mpg.csv",
    "car_crashes.csv": "../datos/accidentes_viales_eeuu.csv",
    "diamonds.csv": "../datos/diamantes.csv",
}


def main():
    for origen, destino in ARCHIVOS.items():
        with urllib.request.urlopen(BASE + origen, timeout=120) as r:
            crudo = r.read()
        open(destino, "wb").write(crudo)
        h = hashlib.sha256(crudo).hexdigest()[:12]
        filas = crudo.count(b"\n")
        print(f"{destino}: {filas} filas, {len(crudo):,} bytes, sha {h}")


if __name__ == "__main__":
    main()

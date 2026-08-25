"""
Machine Learning for Petroleum Engineers Using Python
Modulo 7 - Clase 2: prepara el dataset del reto en datos/

Fuente: el conjunto de practica 'titanic' de seaborn
(github.com/mwaskom/seaborn-data), que a su vez viene del dataset clasico
de Kaggle/OpenML: los 891 pasajeros del Titanic con edad, clase, tarifa y
si sobrevivieron.

El dataset de la DEMO de esta clase (diamantes) ya esta publicado en
datos/diamantes.csv desde la Clase 1.

Se copia tal cual, sin transformar: encontrar sus problemas es parte del
reto. Se corre una vez.

Uso:  python3 preparar_datos.py
"""

import hashlib
import urllib.request

URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
SALIDA = "../datos/pasajeros_titanic.csv"


def main():
    with urllib.request.urlopen(URL, timeout=120) as r:
        crudo = r.read()
    open(SALIDA, "wb").write(crudo)
    h = hashlib.sha256(crudo).hexdigest()[:12]
    filas = crudo.count(b"\n")
    print(f"{SALIDA}: {filas} filas, {len(crudo):,} bytes, sha {h}")


if __name__ == "__main__":
    main()

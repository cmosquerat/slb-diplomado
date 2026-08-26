"""
Machine Learning for Petroleum Engineers Using Python
Modulo 7 - Clase 3: los datos de la clase, documentados.

Esta clase no publica datos nuevos. Usa:

  Acto 1 (demo Titanic):
    datos/pasajeros_titanic.csv -- ya publicado por la Clase 2 (copia
    exacta del conjunto de practica de seaborn).

  Acto 2 (guiado):
    make_moons        -- lo genera scikit-learn en el momento (semilla 0)
    MNIST             -- lo descarga keras.datasets.mnist en Colab
                         (mismo .npz de storage.googleapis.com/tensorflow)
    imagen de prueba  -- https://ultralytics.com/images/bus.jpg
    las fotos propias -- cada estudiante sube la suya (sin instalaciones
                         de la empresa, documentos ni personas sin permiso)

Este script solo VERIFICA que lo publicado responde. Se corre antes de
clase, como chequeo.

Uso:  python3 preparar_datos.py
"""

import urllib.request

URLS = {
    "pasajeros_titanic": ("https://raw.githubusercontent.com/cmosquerat/"
                          "slb-diplomado/main/datos/pasajeros_titanic.csv"),
    "mnist (fuente keras)": ("https://storage.googleapis.com/tensorflow/"
                             "tf-keras-datasets/mnist.npz"),
    "imagen de prueba yolo": "https://ultralytics.com/images/bus.jpg",
}


def main():
    for nombre, url in URLS.items():
        try:
            req = urllib.request.Request(url, method="HEAD")
            estado = urllib.request.urlopen(req, timeout=60).status
        except Exception as e:            # noqa: BLE001
            estado = f"ERROR: {e}"
        print(f"{nombre}: {estado}")


if __name__ == "__main__":
    main()

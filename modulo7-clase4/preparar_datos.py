"""
Machine Learning for Petroleum Engineers Using Python
Modulo 7 - Clase 4 (la ultima): los datos de la clase, documentados.

Esta clase no publica datos nuevos. Usa:

  Fashion-MNIST  -- keras.datasets.fashion_mnist en Colab (los cuatro .gz
                    de storage.googleapis.com/tensorflow, de Zalando
                    Research, licencia MIT)
  foto de prueba -- bus.jpg de los assets de Ultralytics
  modelos        -- yolo11n cls/det/seg/pose (los baja ultralytics) y
                    gemini-2.5-flash via la llave de la clase
  sus fotos      -- cada estudiante sube la suya (sin instalaciones de la
                    empresa, documentos ni personas sin permiso)

La LLAVE DE LA CLASE: se crea en aistudio.google.com antes de clase, se
comparte por el chat del curso, y SE REVOCA al terminar. Ese ciclo es
parte de la leccion de gestion de secretos.

Este script solo VERIFICA que lo publicado responde. Se corre antes de
clase, como chequeo.

Uso:  python3 preparar_datos.py
"""

import urllib.request

BASE_GZ = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
URLS = {
    "fashion train imgs": BASE_GZ + "train-images-idx3-ubyte.gz",
    "fashion train lbls": BASE_GZ + "train-labels-idx1-ubyte.gz",
    "fashion test imgs": BASE_GZ + "t10k-images-idx3-ubyte.gz",
    "fashion test lbls": BASE_GZ + "t10k-labels-idx1-ubyte.gz",
    "foto de prueba yolo": ("https://raw.githubusercontent.com/ultralytics/"
                            "ultralytics/main/ultralytics/assets/bus.jpg"),
}


def main():
    for nombre, url in URLS.items():
        try:
            req = urllib.request.Request(url, method="HEAD")
            estado = urllib.request.urlopen(req, timeout=60).status
        except Exception as e:            # noqa: BLE001
            estado = f"ERROR: {e}"
        print(f"{nombre}: {estado}")
    print("\nrecordatorio: crear la llave de la clase en aistudio.google.com")
    print("y probar una llamada multimodal desde Colab ANTES de la clase.")


if __name__ == "__main__":
    main()

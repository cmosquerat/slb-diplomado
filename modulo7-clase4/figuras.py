"""
Machine Learning for Petroleum Engineers Using Python
Modulo 7 - Clase 4: genera todas las figuras de la presentacion.

Tres fuentes, todas reproducibles:

  1. La SOLUCION del reto de Fashion-MNIST: una corrida de referencia con
     un MLP equivalente al de la clase (mismas capas, misma normalizacion,
     8 epocas, semilla 0). En clase la solucion se corre en vivo con Keras;
     los numeros varian en decimales, las lecciones no.
  2. Las CUATRO TAREAS de vision: las imagenes anotadas por los cuatro
     modelos de Ultralytics (yolo11n cls/det/seg/pose) sobre la foto de
     prueba bus.jpg. Se generan una vez con generar_yolo() (requiere
     ultralytics instalado) y quedan como _yolo_*.jpg en esta carpeta.
  3. Cifras publicas menores, declaradas abajo.

Al final imprime TODAS las cifras que aparecen en las laminas.

Uso:  python3 figuras.py
"""

import gzip
import os
import glob
import struct
import urllib.request
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- estilo ----
RED, DARK = "#C82B40", "#6B1525"
BLUE, GREEN, ORANGE = "#2563EB", "#16A34A", "#EA580C"
INK, MUTED = "#2D2D2D", "#6B7280"
GRAY, LGRAY = "#9CA3AF", "#E5E7EB"


def _fuente():
    for d in (os.environ.get("FIRA_DIR"), "_fuentes"):
        if d and os.path.isdir(d):
            for f in glob.glob(os.path.join(d, "*.ttf")):
                try:
                    matplotlib.font_manager.fontManager.addfont(f)
                except Exception:
                    pass
    if "Fira Sans" in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        return "Fira Sans"
    print("  (aviso: Fira Sans no encontrada; se usa la sans por defecto)")
    return matplotlib.rcParams["font.sans-serif"][0]


plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": [_fuente()], "font.size": 11,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRAY, "axes.labelcolor": INK, "axes.titlecolor": DARK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": LGRAY, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
})


def guardar(fig, nombre):
    fig.savefig(nombre, facecolor="white")
    plt.close(fig)
    print(f"  {nombre}")


SEMILLA = 0
EPOCAS = 8
PRENDAS = ["camiseta/top", "pantalón", "suéter", "vestido", "abrigo",
           "sandalia", "camisa", "zapatilla", "bolso", "botín"]
MNIST_REF = 98        # % aproximado que dio la misma red con digitos (en vivo)

BASE_GZ = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
ARCHIVOS_GZ = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
               "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]


# ------------------------------------------- solucion de referencia ---------
def _leer_imgs(p):
    with gzip.open(p) as f:
        _, n, r, c = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), np.uint8).reshape(n, r, c)


def _leer_lbls(p):
    with gzip.open(p) as f:
        f.read(8)
        return np.frombuffer(f.read(), np.uint8)


def cargar_fashion():
    os.makedirs("_fashion", exist_ok=True)
    for a in ARCHIVOS_GZ:
        p = f"_fashion/{a}"
        if not os.path.exists(p):
            print(f"  bajando {a} ...")
            urllib.request.urlretrieve(BASE_GZ + a, p)
    xe = _leer_imgs("_fashion/train-images-idx3-ubyte.gz")
    ye = _leer_lbls("_fashion/train-labels-idx1-ubyte.gz")
    xx = _leer_imgs("_fashion/t10k-images-idx3-ubyte.gz")
    yx = _leer_lbls("_fashion/t10k-labels-idx1-ubyte.gz")
    return xe, ye, xx, yx


def resolver_referencia(xe, ye, xx, yx):
    """La corrida de referencia: MLP equivalente al de la clase.
    (128 neuronas relu, entradas /255, 10% de validacion, 8 epocas.)"""
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score
    Xe = xe.reshape(len(xe), -1) / 255.0
    Xx = xx.reshape(len(xx), -1) / 255.0
    nv = 6000
    Xt, yt, Xv, yv = Xe[:-nv], ye[:-nv], Xe[-nv:], ye[-nv:]
    red = MLPClassifier((128,), max_iter=1, warm_start=True,
                        random_state=SEMILLA)
    tr, va = [], []
    for _ in range(EPOCAS):
        red.fit(Xt, yt)
        tr.append(accuracy_score(yt, red.predict(Xt)))
        va.append(accuracy_score(yv, red.predict(Xv)))
    pred = red.predict(Xx)
    proba = red.predict_proba(Xx)
    return np.array(tr), np.array(va), pred, proba


def fig_sol_curvas(tr, va):
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    ep = np.arange(1, len(tr) + 1)
    ax.plot(ep, 100 * tr, "o-", color=INK, lw=2.2, label="entrenamiento")
    ax.plot(ep, 100 * va, "o-", color=RED, lw=2.2, label="validación")
    ax.fill_between(ep, 100 * va, 100 * tr, color=RED, alpha=.08)
    ax.annotate("la brecha que crece:\nla red empieza a memorizar",
                xy=(ep[-1] - 0.1, 100 * (tr[-1] + va[-1]) / 2),
                xytext=(4.6, 84.5), fontsize=10, color=RED,
                fontweight="bold", linespacing=1.3,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.set_xlabel("época", fontsize=9.5)
    ax.set_ylabel("acierto [%]", fontsize=9.5)
    ax.legend(fontsize=9.5, loc="lower right")
    ax.set_title("La misma red que sacó ~98 % en dígitos, ahora con ropa",
                 fontsize=12.5, fontweight="bold", loc="left", pad=8)
    guardar(fig, "fig_sol_curvas.png")


def fig_sol_confusion(yx, pred):
    from sklearn.metrics import confusion_matrix
    mc = confusion_matrix(yx, pred)
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    ax.grid(False)
    ax.imshow(mc, cmap="Reds", vmax=300)
    for i in range(10):
        for j in range(10):
            if mc[i, j] > 30 and i != j:
                ax.text(j, i, mc[i, j], ha="center", va="center",
                        fontsize=8, fontweight="bold", color=INK)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(PRENDAS, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(PRENDAS, fontsize=8)
    ax.set_xlabel("lo que predijo", fontsize=9.5)
    ax.set_ylabel("lo que era", fontsize=9.5)
    ax.set_title("Dónde se equivoca: casi todo entre prendas parecidas",
                 fontsize=12, fontweight="bold", loc="left", pad=8)
    guardar(fig, "fig_sol_confusion.png")
    m2 = mc.copy(); np.fill_diagonal(m2, 0)
    i, j = np.unravel_index(np.argsort(m2.ravel())[-3:][::-1], m2.shape)
    return [(PRENDAS[a], PRENDAS[b], int(m2[a, b])) for a, b in zip(i, j)]


def fig_sol_errores(xx, yx, pred, proba):
    mal = np.where(pred != yx)[0]
    conf = proba[mal, pred[mal]]
    peores = mal[np.argsort(conf)[::-1][:9]]     # los errores mas confiados
    fig, axes = plt.subplots(3, 3, figsize=(7.6, 7.2))
    for ax, i in zip(axes.ravel(), peores):
        ax.imshow(xx[i], cmap="gray_r")
        ax.set_title(f"creyó {PRENDAS[pred[i]]} ({100*proba[i, pred[i]]:.0f} %)\n"
                     f"era {PRENDAS[yx[i]]}", fontsize=8.5, linespacing=1.2)
        ax.axis("off")
    fig.suptitle("Los nueve errores MÁS confiados de la corrida de referencia",
                 fontsize=12, fontweight="bold", color=DARK, x=0.02, ha="left")
    fig.tight_layout()
    guardar(fig, "fig_sol_errores.png")


# --------------------------------------------- las cuatro tareas (yolo) -----
RUTA_YOLO = ("/private/tmp/claude-501/-Users-cmosquerat-Documents-GitHub-"
             "slb-diplomado/17ff0dd6-d936-43c0-bd10-f0fc4e571d53/"
             "scratchpad/yolo_out")


def generar_yolo():
    """Corre los cuatro modelos sobre bus.jpg y guarda _yolo_*.jpg aca.
    Requiere ultralytics; se corre UNA vez y las imagenes quedan."""
    from ultralytics import YOLO
    if not os.path.exists("_yolo_bus.jpg"):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/ultralytics/ultralytics/"
            "main/ultralytics/assets/bus.jpg", "_yolo_bus.jpg")
    for peso, et in [("yolo11n-cls.pt", "cls"), ("yolo11n.pt", "det"),
                     ("yolo11n-seg.pt", "seg"), ("yolo11n-pose.pt", "pose")]:
        r = YOLO(peso)("_yolo_bus.jpg", verbose=False)[0]
        r.save(filename=f"_yolo_{et}.jpg")


def fig_cuatro_tareas():
    """El panel: la MISMA foto por las cuatro tareas."""
    faltan = [f"_yolo_{e}.jpg" for e in ("cls", "det", "seg", "pose")
              if not os.path.exists(f"_yolo_{e}.jpg")]
    if faltan:
        # intenta copiarlas de la corrida del scratchpad, o generarlas
        for e in ("cls", "det", "seg", "pose"):
            org = os.path.join(RUTA_YOLO, f"anotada_{e}.jpg")
            if os.path.exists(org) and not os.path.exists(f"_yolo_{e}.jpg"):
                import shutil
                shutil.copy(org, f"_yolo_{e}.jpg")
        if any(not os.path.exists(f"_yolo_{e}.jpg")
               for e in ("cls", "det", "seg", "pose")):
            generar_yolo()

    titulos = {
        "cls": "CLASIFICACIÓN · ¿qué es? — «minibus» 57 % (y duda)",
        "det": "DETECCIÓN · ¿qué y dónde? — 1 bus, 4 personas",
        "seg": "SEGMENTACIÓN · ¿qué píxeles? — máscaras exactas",
        "pose": "POSE · ¿en qué posición? — 17 puntos por persona",
    }
    fig, axes = plt.subplots(1, 4, figsize=(12.6, 4.1))
    for ax, e in zip(axes, ("cls", "det", "seg", "pose")):
        ax.imshow(plt.imread(f"_yolo_{e}.jpg"))
        ax.set_title(titulos[e], fontsize=8.6, fontweight="bold", loc="left")
        ax.axis("off")
    fig.suptitle("La misma foto, cuatro preguntas — cuatro modelos, la misma línea de código",
                 fontsize=12.5, fontweight="bold", color=DARK, x=0.005,
                 ha="left")
    fig.tight_layout()
    guardar(fig, "fig_cuatro_tareas.png")


# =============================================================== MAIN =======
def main():
    print("solucion de referencia de Fashion-MNIST ...")
    xe, ye, xx, yx = cargar_fashion()
    tr, va, pred, proba = resolver_referencia(xe, ye, xx, yx)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(yx, pred)

    print("\ngenerando figuras ...")
    fig_sol_curvas(tr, va)
    confusiones = fig_sol_confusion(yx, pred)
    fig_sol_errores(xx, yx, pred, proba)
    fig_cuatro_tareas()

    print("\n" + "=" * 74)
    print("NUMEROS PARA LAS LAMINAS  (no escribir ninguno a mano)")
    print("=" * 74)
    print(f"\nSOLUCION DE REFERENCIA (MLP 128, /255, {EPOCAS} epocas, "
          f"semilla {SEMILLA}):")
    print(f"  con digitos, la misma red dio ~{MNIST_REF} % (en vivo, Keras)")
    print(f"  con ropa: examen {100*acc:.1f} % | epoca final: train "
          f"{100*tr[-1]:.1f} % vs val {100*va[-1]:.1f} %")
    print("  confusiones mas grandes del examen:")
    for a, b, n in confusiones:
        print(f"    era {a:14s} -> creyo {b:10s} ({n} veces)")
    print("\nLAS CUATRO TAREAS (bus.jpg, yolo11n, corrida real):")
    print("  clasificacion: minibus 57 % | police_van 34 %  <- DUDA")
    print("  deteccion:     1 bus + 4 personas | confianzas 0.94, 0.89, ...")
    print("  segmentacion:  + stop sign | masks (6, 640, 480) pixeles")
    print("  pose:          4 personas x 17 puntos (x, y, confianza)")
    print("\nDEL TRABAJO EN VIVO (llave de la clase, llamadas multimodales)")
    print("no se imprime nada: se hace en clase y la llave se revoca al final.")


if __name__ == "__main__":
    main()

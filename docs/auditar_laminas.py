"""Detecta laminas de Beamer cuyo contenido se sale del area de texto.

LaTeX NO avisa de esto: el contenido verbatim de los bloques de codigo
(minted) no se encoge con [shrink], y una figura demasiado alta mas los
bullets de abajo tampoco disparan ningun warning. El unico juez es el PDF.

Uso:   python3 docs/auditar_laminas.py [carpeta ...]
       python3 docs/auditar_laminas.py modulo5-clase3

Nota: la portada (pagina 1) sale marcada siempre --- su barra inferior de
diseno ocupa esa zona a proposito. Es el unico falso positivo conocido.


Calibrado contra el PDF: una lamina sana no tiene NI UN PIXEL de tinta en
x in [2%, 28%] con y in [90%, 99.5%]. El pie va centrado y el numero de
pagina a la derecha, asi que ese rectangulo esta siempre vacio.
Las laminas divisorias (fondo lleno) se excluyen por su falta de blanco.
"""
import glob, os, subprocess, sys
from PIL import Image

RAIZ = "/Users/cmosquerat/Documents/GitHub/slb-diplomado"
DPI = "150"


def auditar(carpeta, rehacer=True):
    base, tmp = f"{RAIZ}/{carpeta}", f"/tmp/aud_{carpeta}"
    if rehacer:
        os.makedirs(tmp, exist_ok=True)
        for f in glob.glob(f"{tmp}/*.png"):
            os.remove(f)
        subprocess.run(["pdftoppm", "-png", "-r", DPI,
                        f"{base}/presentacion.pdf", f"{tmp}/p"], check=True)
    malas = []
    for f in sorted(glob.glob(f"{tmp}/*.png")):
        im = Image.open(f).convert("L")
        W, H = im.size
        px = im.load()
        blanco = sum(1 for y in range(0, H, 6) for x in range(0, W, 6) if px[x, y] > 230)
        if blanco / ((H // 6) * (W // 6)) < 0.55:
            continue                                   # divisor / portada / cierre
        tinta = sum(1 for y in range(int(0.90 * H), int(0.995 * H))
                    for x in range(int(0.02 * W), int(0.28 * W)) if px[x, y] < 150)
        if tinta > 15:
            malas.append((int(os.path.basename(f).split("-")[-1][:-4]), tinta))
    return malas


for carpeta in (sys.argv[1:] or ["modulo5-clase2", "modulo5-clase3"]):
    m = auditar(carpeta)
    print(f"\n=== {carpeta} ===  {'TODO OK' if not m else str(len(m))+' laminas se desbordan'}")
    for pg, n in m:
        print(f"  pagina {pg:>3}   ({n} px de tinta invadiendo el pie)")

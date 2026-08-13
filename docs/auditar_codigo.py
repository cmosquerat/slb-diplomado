"""Audita los bloques de codigo: lineas demasiado largas y bloques demasiado altos.

minted parte las lineas largas ('breaklines'), y cada quiebre suma altura al
bloque. Un bloque que se pasa de alto se monta sobre el pie de la lamina.
"""
import re, sys

MAX_CHARS = 40      # lo que entra sin partirse en una columna de 0.58\textwidth
MAX_LINEAS = 12     # lineas que entran en la altura util de una lamina

for carpeta in ["modulo5-clase2", "modulo5-clase3"]:
    p = f"/Users/cmosquerat/Documents/GitHub/slb-diplomado/{carpeta}/presentacion.tex"
    s = open(p).read()
    print(f"\n{'='*72}\n{carpeta}\n{'='*72}")
    for entorno in ("pyblock", "outputblock"):
        for m in re.finditer(rf"\\begin{{{entorno}}}(.*?)\\end{{{entorno}}}", s, re.S):
            cuerpo = m.group(1).strip("\n").split("\n")
            linea_tex = s[:m.start()].count("\n") + 1
            # de que lamina es
            ant = s[:m.start()]
            tit = re.findall(r"\\begin\{frame\}[^\{]*\{([^}]*)", ant)
            tit = tit[-1] if tit else "?"
            largas = [(i + 1, len(l), l.strip()) for i, l in enumerate(cuerpo)
                      if len(l) > MAX_CHARS]
            if largas or len(cuerpo) > MAX_LINEAS:
                print(f"\n  L{linea_tex}  [{entorno}]  «{tit[:52]}»")
                print(f"     {len(cuerpo)} lineas" +
                      ("  <-- DEMASIADO ALTO" if len(cuerpo) > MAX_LINEAS else ""))
                for n, ancho, txt in largas:
                    print(f"     linea {n:>2}: {ancho:>3} car.  {txt[:58]}")

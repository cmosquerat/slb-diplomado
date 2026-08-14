# Handoff — Módulo 5

*Estado al commit `7f0a36e`. Árbol limpio, local y remoto sincronizados.*

Este documento describe **qué hay y cómo está hecho**. No propone contenido para las
clases que faltan: para el reparto acordado de temas ver
[`modulo5-retrospectiva-y-ruta.md`](modulo5-retrospectiva-y-ruta.md), Parte D.

---

## 1 · Qué está publicado

| Clase | Título | Láminas | Celdas | Figuras | Estado |
|---|---|---|---|---|---|
| **C1** | Pronosticar la producción de un pozo (Volve) | 52 | 71 | — | Dictada. **No** cumple las reglas de diseño vigentes |
| **C2** | El pozo avisa antes de romperse (3W, Petrobras) | 49 | 118 | 15 | Lista, no dictada |
| **C3** | ¿Cuánto queda? — Arps y DCA probabilístico (Mar del Norte) | 56 | 118 | 17 | Lista, no dictada |

Datasets en `datos/`:

| archivo | tamaño | fuente | usado en |
|---|---|---|---|
| `pozos_3w_incrustacion.csv` | 3,2 MB | 3W Dataset v2.0.0, Petrobras (CC BY 4.0) | C2 |
| `campos_noruega_declinacion.csv` | 631 KB | Sokkeldirektoratet, datos abiertos | C3 |

Ambos se cargan en los cuadernos por **URL raw de GitHub** — verificado: los dos
cuadernos corren de punta a punta sin modificar, y los CSV publicados son byte a byte
idénticos a los locales.

---

## 2 · Cómo está construido cada clase

Cada carpeta `modulo5-claseN/` contiene cuatro piezas, y el orden en que se tocan importa:

1. **`preparar_datos.py`** — baja el dato de la fuente original y escribe el CSV curado en
   `datos/`. Documenta la URL y las transformaciones. Se corre una vez.
2. **`figuras.py`** — lee el CSV, genera **todos** los `fig_*.png` y, al final, **imprime
   todas las cifras que aparecen en las láminas**. Ninguna cifra del deck se escribe a
   mano: se copia de esta salida.
3. **`presentacion.tex`** — Beamer. El preámbulo son las líneas 1–245 de
   `modulo3-clase5/presentacion.tex` copiadas literales; solo cambian el comentario de
   cabecera y el subtítulo de portada.
4. **`ModuloN_ClaseM_*.ipynb`** — el cuaderno, que **tiene que reproducir exactamente los
   mismos números** que imprime `figuras.py`.

> **Regla que ya costó cara una vez:** si `figuras.py` y el cuaderno calculan distinto,
> el deck miente. En la C3 hubo que alinear el orden de las columnas y de la lista de
> sensores porque el Random Forest submuestrea columnas y el orden cambiaba el resultado.

---

## 3 · Entorno de trabajo

Nada de esto está en el repo; hay que rearmarlo en una máquina nueva.

**LaTeX** — TeX Live instalado en `~/texlive` (sin sudo, con `install-tl`):

```bash
export PATH="$HOME/texlive/current/bin/universal-darwin:$PATH"
tlmgr install beamer pgf tcolorbox minted fvextra fontawesome5 fira mweights \
              fontaxes babel-spanish booktabs collection-latexrecommended \
              tikzfill xkeyval cm-super
```

`minted` necesita `pygmentize` en el `PATH`. Está resuelto con un symlink al binario del
venv de Python.

**Python** — venv con `pandas numpy matplotlib scikit-learn statsmodels scipy pyarrow
jupyter nbconvert`.

**Tipografía de las figuras** — Fira Sans, la misma del deck. `figuras.py` la busca en la
variable de entorno `FIRA_DIR` o en `./_fuentes/`; si no la encuentra usa la sans por
defecto y avisa. Los `.ttf` se bajan de `github.com/google/fonts/ofl/firasans`.

**Compilar un deck:**

```bash
cd modulo5-claseN
pdflatex -shell-escape presentacion.tex     # dos pasadas
pdflatex -shell-escape presentacion.tex
```

---

## 4 · Herramientas de control de calidad

Están en `docs/` y hay que correrlas **antes de dar por terminada** una clase.

**`auditar_laminas.py`** — renderiza el PDF y marca las láminas cuyo contenido se sale del
área de texto. **LaTeX no avisa de esto**: el contenido verbatim de los bloques de código
no se encoge con `[shrink]`, y una figura alta con bullets debajo tampoco dispara ningún
warning. Está calibrado contra el PDF: una lámina sana no tiene **ni un píxel** de tinta en
el margen inferior izquierdo.

```bash
python3 docs/auditar_laminas.py modulo5-clase3
```

Único falso positivo conocido: la página 1 (portada), cuya barra inferior ocupa esa zona a
propósito.

**`auditar_codigo.py`** — marca, antes de compilar, las líneas de código demasiado anchas
y los bloques demasiado altos. Los límites medidos empíricamente son **40 caracteres por
línea** y **12 líneas por bloque**; pasarse de ahí hace que `minted` parta las líneas y el
bloque se monte sobre el pie.

---

## 5 · Hallazgos empíricos que condicionan lo que se haga después

No son opiniones: se midieron, y ya hicieron descartar dos diseños completos.

**El ML no cobra en una serie de un solo activo.** Se probó de cinco formas distintas
sobre producción mensual; el modelo tonto («el valor de hoy») ganó o empató en casi
todas. Una serie de producción es suave y muy autocorrelacionada. El ML cobra cuando hay
**muchas unidades y muchas variables**. El detalle está en la Parte B de
`modulo5-retrospectiva-y-ruta.md`.

> **Consecuencia operativa:** antes de comprometer una clase con un dataset, medir si el
> modelo le gana al modelo tonto. Si no le gana, el problema está mal elegido.

**No hay dato público bulk a nivel de pozo.** Se verificaron ocho fuentes (NDIC, Colorado
ECMC, Wyoming, Texas RRC, Kansas, Louisiana, Nuevo México, Utah): todas están detrás de
JavaScript, suscripción o portales no descargables. Lo que sí funciona
programáticamente es **Sokkeldirektoratet** (Noruega, nivel campo) y el **3W de
Petrobras** (sensores de pozo, GitHub).

**El punto estimado no es entregable; la banda sí.** Medido en las dos clases con datos
distintos: los estimadores puntuales fallan en ambas direcciones y ningún método es
establemente mejor, pero un intervalo bien construido cumple lo que promete (en la C3, el
76 % contra un 80 % nominal, en 41 de 54 campos).

---

## 6 · Lo que quedó pendiente

**a) Las métricas de la Clase 2 están en la unidad equivocada.** Es una crítica del
instructor, aceptada y no corregida. Se le calculó *recall* y *% de falsas alarmas* **por
muestra** a una alarma de sala de control, que es un dispositivo **de evento**: no está
diseñada para permanecer encendida durante todo el transitorio, y la métrica la castiga
por eso. Las dos métricas que sí están bien son *cuántos casos detecta* y *cuántos
minutos tarda*. El arreglo es sacar el recall y el porcentaje de falsas por muestra, y
reemplazarlos por disparos falsos **por día de operación normal**. Toca `figuras.py`, el
cuaderno y tres láminas de la C2.

**b) Dos láminas de la Clase 3 pueden estar de más.** Quedaron de un intento anterior de
explicar el ajuste: *«Antes de Ajustar Nada: ¿Qué Quiere Decir Ajustar?»* (la de las
barras de error, que se sostiene sola) y *«Por Eso `curve_fit` Pide Cosas que `polyfit`
No»* (que quizá sobra ahora que la figura de los dos caminos lo dice todo). Decisión
pendiente del instructor.

**c) Los generadores de los cuadernos no están en el repo.** Los `.ipynb` se produjeron
con scripts `mknb*.py` que quedaron fuera del control de versiones. Los cuadernos
publicados están completos y corren, pero **para editarlos hay que hacerlo a mano sobre el
JSON** o volver a escribir el generador. Conviene resolverlo antes de la próxima clase.

**d) La Clase 1 sigue sin cumplir las reglas.** Está publicada tal como se dictó, en
`pptxgenjs` en vez de Beamer, y con los problemas documentados en la Parte A de la
retrospectiva. La decisión de reconstruirla o no quedó abierta.

---

## 7 · Dónde está cada cosa

```
docs/
  modulo5-retrospectiva-y-ruta.md   por qué falló la C1, las 12 reglas de diseño,
                                    y el reparto de temas de las clases que faltan
  handoff-modulo5.md                este documento
  auditar_laminas.py                detecta laminas desbordadas (correr SIEMPRE)
  auditar_codigo.py                 detecta bloques de codigo demasiado anchos/altos

modulo5-clase2/   3W Petrobras · deteccion temprana
modulo5-clase3/   Mar del Norte · curvas de declinacion
datos/            los CSV curados, que los cuadernos bajan por URL raw
```

Las **12 reglas de diseño** de la Parte C de la retrospectiva son el contrato de calidad
del módulo. Las dos que más se olvidan: *el dinero y la decisión van en los primeros 10
minutos*, y *al menos un resultado negativo por clase, dicho en voz alta*.

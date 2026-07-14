# Diplomado en Data Science Aplicada con Python

**Cliente:** SLB Ecuador
**Institución:** Universidad de las Américas (UDLA)
**Instructor:** Carlos Enrique Mosquera Trujillo — cmosquerat@unal.edu.co

Diplomado corporativo enfocado en aplicaciones de análisis de datos para la industria petrolera. Todo el material está preparado para ejecutarse en **Google Colab** (sin instalación local).

## Contenido

| Clase | Tema | Estado |
|-------|------|--------|
| [Clase 1](clase-01/) | Fundamentos de Python (variables, tipos, `input`, `if` básico) | ✅ Publicada |
| [Clase 2](clase-02/) | Control de flujo — condicionales, listas y ciclos `for` | ✅ Publicada |
| Clase 3 | pandas, numpy, matplotlib/seaborn; CSV/Excel/JSON; well logs (LAS); limpieza y normalización (dataset de 30 años) | 🔜 En preparación |

## Clase 1: Fundamentos de Python + Control de Flujo

- [`clase-01/presentacion.pdf`](clase-01/presentacion.pdf) — slides de la sesión.
- [`clase-01/Clase_01_Fundamentos_de_Python.ipynb`](clase-01/Clase_01_Fundamentos_de_Python.ipynb) — cuaderno Colab con ejercicios y caso integrador.
- [`clase-01/presentacion.tex`](clase-01/presentacion.tex) — fuente LaTeX (Beamer).

**Contenidos cubiertos:**

1. `print()`, operaciones aritméticas
2. Variables, tipos (`int`, `float`, `str`, `bool`), conversiones
3. `input()` — leer datos del usuario
4. Comparaciones, booleanos, `if / elif / else`
5. Listas, indexing, slicing
6. Ciclos `for`, patrones acumulador/contador
7. **Caso integrador:** Reporte diario del pozo Sacha-042 (BOPD netos, márgenes, proyección mensual)

## Clase 2: Control de flujo — condicionales, listas y ciclos

- [`clase-02/presentacion.pdf`](clase-02/presentacion.pdf) — slides de la sesión.
- [`clase-02/Clase_02_Control_de_Flujo.ipynb`](clase-02/Clase_02_Control_de_Flujo.ipynb) — cuaderno Colab (los ejercicios se dejan en blanco para resolver en clase).
- [`clase-02/presentacion.tex`](clase-02/presentacion.tex) — fuente LaTeX (Beamer).

**Contenidos cubiertos:**

1. Condicionales a fondo: cuándo usar `if` solo, `if/else`, `if/elif/else`
2. La trampa `if / if / if` vs `if / elif / elif`; `else` como catch-all
3. Operadores lógicos `and`, `or`, `not` y paréntesis
4. Listas: qué son, sintaxis `[ ]`, índice desde 0, slicing
5. Métodos de lista (`.append`, `.insert`, `.remove`, `.pop`), operador `in`, listas paralelas
6. Ciclos `for`: `range()`, `enumerate()`, `zip()`
7. Patrones **acumulador**, **contador** y **filtro**
8. **Práctica integradora:** reporte de una batería de 5 pozos

### Abrir el notebook en Colab

Dentro de Colab: `File → Open notebook → GitHub → cmosquerat/slb-diplomado` y selecciona el `.ipynb` de la clase.

### Compilar la presentación localmente

Requiere TeX Live con `minted`, `tcolorbox`, `FiraSans`, `fontawesome5`:

```bash
cd clase-01
pdflatex -shell-escape presentacion.tex
pdflatex -shell-escape presentacion.tex  # segunda pasada para referencias
```

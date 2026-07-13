# Diplomado en Data Science Aplicada con Python

**Cliente:** SLB Ecuador
**Institución:** Universidad de las Américas (UDLA)
**Instructor:** Carlos Enrique Mosquera Trujillo — cmosquerat@unal.edu.co

Diplomado corporativo enfocado en aplicaciones de análisis de datos para la industria petrolera. Todo el material está preparado para ejecutarse en **Google Colab** (sin instalación local).

## Contenido

| Clase | Tema | Estado |
|-------|------|--------|
| [Clase 1](clase-01/) | Fundamentos de Python + Control de Flujo | ✅ Publicada |
| Clase 2 | numpy, pandas, matplotlib, seaborn; lectura de CSV/Excel/JSON | 🔜 En preparación |
| Clase 3 | Well logs (LAS), datos petrofísicos y de producción, limpieza y normalización | 🔜 En preparación |

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

### Abrir el notebook en Colab

Dentro de Colab: `File → Open notebook → GitHub → cmosquerat/slb-diplomado` y selecciona el `.ipynb` de la clase.

### Compilar la presentación localmente

Requiere TeX Live con `minted`, `tcolorbox`, `FiraSans`, `fontawesome5`:

```bash
cd clase-01
pdflatex -shell-escape presentacion.tex
pdflatex -shell-escape presentacion.tex  # segunda pasada para referencias
```

# Machine Learning for Petroleum Engineers Using Python

**Cliente:** SLB Ecuador
**Institución:** Universidad de las Américas (UDLA)
**Instructor:** Carlos Enrique Mosquera Trujillo — cmosquerat@unal.edu.co

Diplomado corporativo enfocado en aplicaciones de análisis de datos para la industria petrolera. Todo el material está preparado para ejecutarse en **Google Colab** (sin instalación local).

## Contenido

| Clase | Tema | Estado |
|-------|------|--------|
| [Clase 1](clase-01/) | Fundamentos de Python (variables, tipos, `input`, `if` básico) | ✅ Publicada |
| [Clase 2](clase-02/) | Control de flujo — condicionales, listas y ciclos `for` | ✅ Publicada |
| [Clase 3](clase-03/) | pandas y carga de datos: librerías/pip, CSV, well logs (LAS) y limpieza — campo Volve | ✅ Publicada |
| [Módulo 3 · Clase 1](modulo3-clase1/) | ML supervisado — Regresión lineal: predecir el registro sónico (Volve) | ✅ Publicada |
| [Módulo 3 · Clase 2](modulo3-clase2/) | Clasificación — Regresión logística, métricas y **costo de negocio** (FORCE 2020) | ✅ Publicada |

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

## Clase 3: pandas y carga de datos — campo Volve

- [`clase-03/presentacion.pdf`](clase-03/presentacion.pdf) — slides de la sesión.
- [`clase-03/Clase_03_Pandas_Carga_Datos.ipynb`](clase-03/Clase_03_Pandas_Carga_Datos.ipynb) — cuaderno Colab (ejercicios en blanco).
- [`clase-03/presentacion.tex`](clase-03/presentacion.tex) — fuente LaTeX (Beamer).
- [`datos/volve_produccion.csv`](datos/volve_produccion.csv) — producción diaria real del campo Volve (Equinor, 15 634 filas, 7 pozos, 2007–2016).
- [`datos/volve_15-9-19.LAS`](datos/volve_15-9-19.LAS) — well log real del pozo 15/9-19 (29 754 profundidades, 7 curvas).

**Contenidos cubiertos:**

1. Qué es un dato / dataset / datos tabulares; qué es un CSV
2. Librerías: qué son, `pip` (instalar) e `import` (usar)
3. pandas: DataFrame y Series; paréntesis de diccionarios; primer DataFrame
4. Cargar archivos en Colab: URL, subida manual y Google Drive
5. Métodos básicos: `head/tail/shape/info`, `describe`, seleccionar, filtrar, columnas calculadas, `groupby`
6. Gráficos con pandas: `.plot()` (línea, barras)
7. Well logs: qué son, el formato LAS (dimensiones, curvas GR/DEN/NEU/AC/RDEP), `lasio`
8. Limpieza breve: `NaN`, `isna`, `dropna`, `fillna`
9. **Práctica integradora:** reporte de producción del campo Volve

*Datos: campo Volve, Equinor (dataset abierto, 2018).*

## Módulo 3 · Clase 1: Regresión lineal — predecir el sónico

- [`modulo3-clase1/presentacion.pdf`](modulo3-clase1/presentacion.pdf) — slides (47 págs).
- [`modulo3-clase1/Modulo3_Clase1_Regresion_Lineal.ipynb`](modulo3-clase1/Modulo3_Clase1_Regresion_Lineal.ipynb) — cuaderno Colab (prácticas en blanco).
- [`datos/volve_registros.csv`](datos/volve_registros.csv) — registros del pozo 15/9-19 listos para ML (6 893 profundidades × 6 columnas, sin NaN).

**Contenidos:** qué es ML (supervisado/regresión) · relaciones entre variables · correlación (dirección + firmeza) · linealidad y sus límites · regresión simple y múltiple (sklearn) · métricas RMSE/MAE/R² con criterios y valores aceptables · estandarización y data leakage · train/test split y overfitting · caso: reconstruir el registro sónico cuando el sensor falla.

## Módulo 3 · Clase 2: Clasificación — métricas y costo de negocio

- [`modulo3-clase2/presentacion.pdf`](modulo3-clase2/presentacion.pdf) — slides (42 págs).
- [`modulo3-clase2/Modulo3_Clase2_Clasificacion.ipynb`](modulo3-clase2/Modulo3_Clase2_Clasificacion.ipynb) — cuaderno Colab (prácticas en blanco).
- [`datos/litologia_force2020.csv`](datos/litologia_force2020.csv) — registros de 11 pozos del Mar del Norte con litología interpretada por geólogos (62 792 filas).

**Contenidos:** el mapa del ML (supervisado vs no supervisado; regresión vs clasificación) · el problema: ¿roca reservorio o sello? · por qué la recta falla con sí/no · sigmoide y **regresión logística** (qué es, para qué sirve, cómo leerla) · desbalance de clases · **métricas a fondo**: la analogía del detector de gas, matriz de confusión con nombres de negocio, precision vs recall (de dónde sale cada una), F1, criterios y valores aceptables · **el costo de negocio**: matriz de costos, el umbral como perilla de decisión y el modelo con peor accuracy que le conviene a la empresa.

*Datos: FORCE 2020 Machine Learning Contest (Noruega) — dataset abierto.*

### Abrir el notebook en Colab

Dentro de Colab: `File → Open notebook → GitHub → cmosquerat/slb-diplomado` y selecciona el `.ipynb` de la clase.

### Compilar la presentación localmente

Requiere TeX Live con `minted`, `tcolorbox`, `FiraSans`, `fontawesome5`:

```bash
cd clase-01
pdflatex -shell-escape presentacion.tex
pdflatex -shell-escape presentacion.tex  # segunda pasada para referencias
```

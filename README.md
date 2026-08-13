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
| [Módulo 3 · Clase 3](modulo3-clase3/) | No linealidad — Árboles de decisión y Random Forest (litología + medidor virtual de flujo) | ✅ Publicada |
| [Módulo 3 · Clase 4](modulo3-clase4/) | Clasificación multiclase — codificación, métricas macro/weighted y matriz de penalización (facies Hugoton, SEG 2016) | ✅ Publicada |
| [Módulo 3 · Clase 5](modulo3-clase5/) | SVM y kernel trick, pipelines, validación honesta (CV, GroupKFold, fuga de información) y SMOTE — integridad de gasoductos (PHMSA) | ✅ Publicada |
| [Módulo 5 · Clase 1](modulo5-clase1/) | Pronosticar la producción de un pozo — series de tiempo, exploración y el primer pronóstico (campo Volve) | ✅ Publicada |
| [Módulo 5 · Clase 2](modulo5-clase2/) | El pozo avisa antes de romperse — detección temprana de incrustación en el choke con señales de sensores (3W, Petrobras) | ✅ Publicada |
| [Módulo 5 · Clase 3](modulo5-clase3/) | ¿Cuánto queda? — curvas de declinación de Arps, sesgo medido en 54 campos y DCA probabilístico auditado | ✅ Publicada |

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

- [`modulo3-clase2/presentacion.pdf`](modulo3-clase2/presentacion.pdf) — slides (49 págs).
- [`modulo3-clase2/Modulo3_Clase2_Clasificacion.ipynb`](modulo3-clase2/Modulo3_Clase2_Clasificacion.ipynb) — cuaderno Colab (prácticas en blanco).
- [`datos/litologia_force2020.csv`](datos/litologia_force2020.csv) — registros de 11 pozos del Mar del Norte con litología interpretada por geólogos (62 792 filas). La etiqueta viene **como texto** (`Sandstone`/`Shale`): binarizarla es parte del ejercicio.

**Contenidos:** el mapa del ML (supervisado vs no supervisado; regresión vs clasificación) · el problema: ¿roca reservorio o sello? · por qué la recta falla con sí/no · sigmoide y **regresión logística** (qué es, para qué sirve, cómo leerla) · exploración a fondo (crossplot neutrón–densidad, matriz de correlación, boxplots por clase, perfil de pozo, balance por pozo) · desbalance de clases · **binarización del target** (preprocesamiento: por qué el modelo no entiende texto y por qué importa cuál clase es el `1`) · **métricas a fondo**: la analogía del detector de gas, matriz de confusión con nombres de negocio, precision vs recall (de dónde sale cada una), F1, criterios y valores aceptables · **el costo de negocio**: matriz de costos, el umbral como perilla de decisión y el modelo con peor accuracy que le conviene a la empresa.

*Datos: FORCE 2020 Machine Learning Contest (Noruega) — dataset abierto.*

## Módulo 3 · Clase 3: No linealidad — Árboles y Random Forest

- [`modulo3-clase3/presentacion.pdf`](modulo3-clase3/presentacion.pdf) — slides (42 págs).
- [`modulo3-clase3/Modulo3_Clase3_Arboles_RandomForest.ipynb`](modulo3-clase3/Modulo3_Clase3_Arboles_RandomForest.ipynb) — cuaderno Colab (prácticas en blanco).
- [`datos/operacion_pozos_volve.csv`](datos/operacion_pozos_volve.csv) — operación diaria de 5 pozos productores de Volve (7 862 días: presiones, temperatura, choke → oil medido).

**Contenidos:** qué es la no linealidad (el paso deja de ser parejo; ejemplos de campo) · las dos lunas y el fracaso de la frontera recta · árboles de decisión desde cero (ejemplo cotidiano de operación, anatomía raíz/hojas/profundidad, cómo la máquina elige cada corte, el primer corte real GR=49) · overfitting visible (train 1.000) · Random Forest (diversidad + voto) · **métricas a fondo del bosque** (matriz de confusión, precision/recall, marcador completo, umbral por costo: logística 5 953 → RF+umbral 1 150, −81 %) · cómo reportarlo al que decide · bonus regresión: medidor virtual de flujo (MAE 583 → 78 Sm³/día).

*Datos: FORCE 2020 y campo Volve (Equinor) — datasets abiertos.*

## Módulo 3 · Clase 4: Clasificación multiclase — el mapa de facies

- [`modulo3-clase4/presentacion.pdf`](modulo3-clase4/presentacion.pdf) — slides (54 págs).
- [`modulo3-clase4/Modulo3_Clase4_Multiclase_Facies.ipynb`](modulo3-clase4/Modulo3_Clase4_Multiclase_Facies.ipynb) — cuaderno Colab (mini-ejercicios y prácticas en blanco).
- [`datos/hugoton_facies.csv`](datos/hugoton_facies.csv) — facies del campo Hugoton, Kansas (concurso SEG 2016: 4 149 intervalos, 10 pozos, 5 registros + contexto, 9 facies).

**Contenidos:** qué es una facies y por qué es un mapa de calidad de roca · exploración a fondo (registros uno a uno, perfil de pozo, balance 7:1) · datos faltantes e **imputación** (menú de métodos y cómo la distribución decide media vs mediana) · **codificación** completa: binarizar / LabelEncoder / one-hot, la trampa del orden falso (la facies 4.37 no existe) · RF multiclase (mismo código, acc 0.764 vs tonto 0.229) · **métricas multiclase**: matriz 9×9 leída geológicamente (67 % de errores en facies vecinas), precision/recall por clase, **macro vs weighted** y el modelo perezoso ciego a clases raras · **matriz de penalización** (costo 1 718 → 379) y cómo reportarlo al que decide · parámetros vs hiperparámetros y **GridSearchCV** básico.

*Datos: SEG 2016 Machine Learning Contest (Hall, 2016) — dataset abierto.*

## Módulo 3 · Clase 5: SVM, pipelines y validación honesta — cierre del módulo

- [`modulo3-clase5/presentacion.pdf`](modulo3-clase5/presentacion.pdf) — slides (47 págs).
- [`modulo3-clase5/Modulo3_Clase5_SVM_Pipelines_Validacion.ipynb`](modulo3-clase5/Modulo3_Clase5_SVM_Pipelines_Validacion.ipynb) — cuaderno Colab (mini-ejercicios y prácticas en blanco).
- [`datos/phmsa_gasoductos.csv`](datos/phmsa_gasoductos.csv) — 638 incidentes reales en gasoductos de transmisión (PHMSA/DOT EE. UU., 2010–hoy): características de la línea, causa investigada, ignición y costo en dólares.

**Contenidos:** problema nuevo de integridad de ductos (¿fue corrosión? ¿la fuga se enciende?) con costos reales · **SVM**: margen máximo y vectores de soporte (la carretera más ancha) · el desastre de no escalar (0.776) y **pipelines** como honestidad automatizada (0.823, el tubo con `make_pipeline`) · **kernel trick** gráfico: círculos 58 %→100 % con la dimensión creada a mano y el plano 3D, las lunas por última vez, perillas `gamma` y `C` · la lección honesta (el lineal empata en datos tabulares reales) · **validación honesta**: `cross_val_score`, `GridSearchCV` sobre el tubo (`svc__C`), la **fuga de información medida en vivo** (0.823→0.906 con una columna post-incidente) y **GroupKFold** — la deuda del módulo pagada: facies con pozos por fuera **0.78→0.53** · **SMOTE** para la ignición (12 %): recall 0.00→0.35 con el trade-off en dólares · cierre del módulo: marcador y mapa de modelos.

*Datos: PHMSA (U.S. DOT) — datos públicos del gobierno de EE. UU.*

### Abrir el notebook en Colab

Dentro de Colab: `File → Open notebook → GitHub → cmosquerat/slb-diplomado` y selecciona el `.ipynb` de la clase.


---

## Módulo 5 · Clase 2: el pozo avisa antes de romperse

- [`modulo5-clase2/presentacion.pdf`](modulo5-clase2/presentacion.pdf) — slides (49 láminas).
- [`modulo5-clase2/Modulo5_Clase2_Deteccion_En_Senales.ipynb`](modulo5-clase2/Modulo5_Clase2_Deteccion_En_Senales.ipynb) — cuaderno Colab (118 celdas, EDA de 15–20 min al inicio).
- [`modulo5-clase2/figuras.py`](modulo5-clase2/figuras.py) — genera las 15 figuras **y todas las cifras** de las láminas.
- [`modulo5-clase2/preparar_datos.py`](modulo5-clase2/preparar_datos.py) — arma el CSV desde la fuente original.
- [`datos/pozos_3w_incrustacion.csv`](datos/pozos_3w_incrustacion.csv) — 315 horas de sensores de 5 pozos submarinos, etiquetadas segundo a segundo.

Un pozo submarino a 1 200 m de profundidad. El choke de producción se incrusta con sal y
se va cerrando solo. Cuando la alarma de la sala de control suena, el pozo ya está en
falla. **¿Cuánto antes se puede saber?**

La clase explica desde cero qué es un choke, por qué al taparse la presión de arriba sube
y la temperatura de abajo baja (Joule–Thomson), y qué es exactamente una serie de tiempo.
Después mide la alarma que ya existe —**8 de 10 casos, 101 minutos tarde**— y la usa como
récord a batir, obligando a todo lo demás al mismo presupuesto de falsas alarmas.

**Contenidos cubiertos:**

1. El choke: qué es, por qué existe y por qué no se deja abierto · el efecto Joule–Thomson
2. Qué es una serie de tiempo, y qué es una etiqueta puesta por una persona
3. EDA completo: qué sensores existen de verdad (un pozo tiene solo dos), en qué rango
   trabaja cada pozo, cuánto dura la ventana de aviso
4. La alarma de umbral de 3 desviaciones, medida sobre las 10 grabaciones
5. De la señal a una tabla: nivel, ruido y pendiente sobre ventanas de 30 minutos
6. Random Forest validado con `GroupKFold` **por pozo** — un pozo que el modelo nunca vio
7. **La escalera:** los números crudos detectan 2 de 10; compararlos con la propia hora
   normal del pozo detecta **10 de 10 y avisa 74 minutos antes**. Promediar y agregar
   ruido/pendiente **no mejoran**, y se dice
8. El umbral como decisión de negocio, no técnica
9. Las «falsas alarmas» que no lo eran: se concentran justo antes de la etiqueta
10. **Práctica:** el pozo con solo dos sensores, con el primer paso resuelto

*Datos: 3W Dataset v2.0.0, Petrobras — CC BY 4.0. Vargas et al. (2019), JPSE 181, 106223.*

Ver también [`docs/modulo5-retrospectiva-y-ruta.md`](docs/modulo5-retrospectiva-y-ruta.md):
retrospectiva de la Clase 1, reglas de diseño del módulo y hoja de ruta de las 10 h.


---

## Módulo 5 · Clase 3: ¿cuánto queda?

- [`modulo5-clase3/presentacion.pdf`](modulo5-clase3/presentacion.pdf) — slides (56 láminas).
- [`modulo5-clase3/Modulo5_Clase3_Curvas_De_Declinacion.ipynb`](modulo5-clase3/Modulo5_Clase3_Curvas_De_Declinacion.ipynb) — cuaderno Colab (121 celdas, EDA de 15–20 min al inicio y una **animación** que dobla el eje en vivo).
- [`modulo5-clase3/figuras.py`](modulo5-clase3/figuras.py) — genera las 17 figuras **y todas las cifras** de las láminas.
- [`modulo5-clase3/preparar_datos.py`](modulo5-clase3/preparar_datos.py) — arma el CSV desde la fuente original.
- [`datos/campos_noruega_declinacion.csv`](datos/campos_noruega_declinacion.csv) — 54 campos del Mar del Norte alineados desde su pico, hasta 49 años de historia.

Un campo pasó su pico hace cinco años. Gerencia decide si sigue invirtiendo o lo prepara
para abandono. **¿Cuánto petróleo le queda en los próximos siete años?**

Lo que hace especial a esta clase: los 54 campos **ya produjeron** esos doce años. Se
puede tapar la mitad, pronosticar, destapar y ver quién tenía razón. Es una auditoría,
no una simulación.

**Contenidos cubiertos:**

1. Por qué declina un campo: presión, agua y gas liberado — las tres causas, desde cero
2. Alinear desde el **pico** y no por calendario · comparar cada campo consigo mismo
3. **Arps (1945)**: la exponencial es la recta sobre el logaritmo de la Clase 1, ahora
   con nombre y autor. Qué es **D**, la tasa de declinación
4. **Cómo se encuentra el pico**, que no es «el mes de mayor producción»: el máximo crudo
   y el suavizado coinciden en solo **2 de 66 campos**, y se separan hasta 58 meses.
   Cuando hay meseta o redesarrollo, el pico **no es un dato: es una decisión**
5. **Qué quiere decir «ajustar»**: minimizar la suma de errores al cuadrado, mostrado con
   barras. Y la diferencia real entre `polyfit` y `curve_fit` — el primero **despeja** una
   fórmula (se resuelve a mano en el cuaderno y da idéntico), el segundo **camina** cuesta
   abajo por un valle largo donde `Di` y `b` se compensan entre sí
6. **Qué es b**, por donde se entiende: no cambia cuánto declina el campo hoy, cambia
   cuánto va a declinar mañana
7. **Por qué sirve doblar el eje**, mostrado en movimiento: una animación deforma el eje
   de forma continua (λ de 1 a 0) y se ve cómo la «panza» del residuo se aplana, de
   **0,50 a 0,06**. Y el resultado honesto: el logaritmo endereza en **33 de 53 campos**
   — el tercio que falla es el que tiene cola, y es la razón de que exista la hiperbólica
4. **El defecto, medido en 54 campos:** la exponencial no falla al azar, falla siempre
   para el mismo lado — sesgo de **−7 %**. Error y sesgo no son lo mismo
5. La **hiperbólica** y el exponente **b**, que es la cola del campo. Se gana su lugar:
   mata el sesgo (−7 % → +1 %) y baja el peor caso de 41 % a **29 %**
6. **El límite económico**: un campo no produce hasta cero, produce hasta que deja de
   pagar sus costos fijos. La curva elegida decide **el año de cierre** — en Draugen,
   19 años con la exponencial contra 54 con la hiperbólica frenada
7. **La trampa de la cola**: sin freno, la hiperbólica dice que **12 de 54 campos no se
   mueren nunca**. La corrección estándar (declinación terminal del 5 %/año) los baja a 4
8. **El sesgo, en dólares**: ese −7 % son **USD 264 millones** por campo y
   **USD 11.400 millones** en los 54 juntos — y como va siempre para el mismo lado,
   en una cartera no se compensa, se suma
9. **P10 / P50 / P90** explicados como lo que son: una promesa verificable
10. La banda construida con **análogos**, dejando siempre el propio campo fuera del grupo
11. **Calibración** — la única prueba de que un intervalo es honesto. La banda prometía
    80 % y cumplió **76 %** en 41 de 54 campos
12. Lo que el método **no** autoriza a decir
13. **Práctica:** el campo Gullfaks de punta a punta, con el primer paso resuelto

*Datos: Sokkeldirektoratet (Norwegian Offshore Directorate), datos abiertos. Arps, J. J.
(1945), Analysis of Decline Curves, Trans. AIME 160(01), 228–247.*

### Compilar la presentación localmente

Requiere TeX Live con `minted`, `tcolorbox`, `FiraSans`, `fontawesome5`:

```bash
cd clase-01
pdflatex -shell-escape presentacion.tex
pdflatex -shell-escape presentacion.tex  # segunda pasada para referencias
```

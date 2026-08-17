# Propuesta — Módulo 5, Clase 5 (v2, construida)

*Documento de diseño. La v1 proponía una clase de pura economía de la decisión sin dato
nuevo; el instructor pidió otra cosa: material nuevo, dataset nuevo, un modelo
interesante pero fácil de explicar, y el puente a negocio que estas clases suelen omitir.
Esta versión responde a eso. La viabilidad ya está **medida**, no supuesta (ver §5).*

> **Estado: CONSTRUIDA.** La clase quedó en `modulo5-clase5/` (deck de 59 láminas
> auditado, cuaderno generado por `mknb.py`, `preparar_datos.py` y `figuras.py`).
> Sobre esta propuesta se sumaron, a pedido del instructor: **hyperparameter tuning
> con método** (malla + GroupKFold, con su resultado negativo: compra 0,2 ciclos),
> **valores de Shapley** como explicabilidad local (waterfall por unidad + beeswarm
> de flota + la lámina de la letra chica, auditada con la ablación sin edad), y
> `explainerdashboard` como celda opcional «para llevar» en el cuaderno. La política
> final quedó: aviso cuando P10 < 20 ciclos, cambio 15 después → 0 emergencias,
> 1,49 MUSD/1.000 ciclos, 34 % más barato que el mejor calendario.

---

## 1 · Lo que el PDF obliga, y lo que eso permite

El temario contratado deja para la C5: **Tema 8 — Fallas de bombas ESP mediante ML**,
**Tema 9 — Optimización de levantamiento artificial**, y el cierre de la **Práctica 2 —
ML para fallas usando series de tiempo**. Además la C4 prometió que la C5 cierra la
cuenta del módulo (cubierto vs diferido).

La C2 ya entregó el *método* de detección («¿está fallando?»). Repetirlo sobre otra
flota sería parafraseo. Lo que el temario permite —y ningún curso da— es la **otra**
pregunta del mantenimiento, la que gerencia presupuesta:

> **No «¿está fallando?», sino «¿cuánta vida le queda?»**

---

## 2 · La clase

> **C5 — «¿Cuánta vida le queda a la bomba?»**
> *Vida útil remanente (RUL), el modelo que la industria usa de verdad, y el precio de
> equivocarse tarde.*

Tres piezas de material **nuevo**, cada una fácil de explicar y con sentido de ML:

### a) El concepto que se omite: RUL y la curva de vida

En ESP la industria vive de la *run life*: cuántos días corre una bomba antes del
workover. La decisión de negocio no es una alarma: es **cuándo programo el equipo de
intervención**, con semanas de anticipación. Eso exige un número en el futuro: «a esta
bomba le quedan ~40 días». Ese número se llama vida útil remanente (RUL) y los cursos de
ML casi nunca lo enseñan porque no es ni clasificación pura ni serie de tiempo pura — es
exactamente el hueco entre la C2 (detectar) y lo que el mantenimiento necesita
(planear).

Se explica con una sola figura: el reloj que corre hacia atrás. Cada unidad de la flota
nace, se degrada y muere; en cada instante su RUL es «cuánto le falta». Convertir una
flota corrida hasta la falla en una tabla de (sensores hoy → vida restante) es una
transformación de datos que un ingeniero entiende en cinco minutos y que le sirve toda
la carrera.

### b) El modelo que se omite: Gradient Boosting

El diplomado enseñó regresión lineal, logística, árboles, Random Forest, SVM y K-means.
Falta **el modelo que gana en la industria con datos tabulares**: gradient boosting
(XGBoost/LightGBM — en clase, el `HistGradientBoostingRegressor` de scikit-learn, cero
dependencias nuevas).

Y es *más* fácil de explicar que la SVM:

> El Random Forest es un comité: mil árboles opinan **en paralelo** y se promedia.
> El gradient boosting es una cadena: cada árbol nuevo se entrena **sobre los errores
> del anterior** y aporta una corrección chiquita. Sumar corrección tras corrección.

Figura pedagógica: la predicción de una unidad con 1 árbol, 10 árboles, 300 árboles —
se ve la curva acercándose a la verdad. Ninguna matemática nueva: es el árbol de la C3
del Módulo 3, puesto en serie en vez de en paralelo. Nada de pares, vecinos ni análogos.

### c) El puente a negocio que se omite: el error tiene dirección

Con RUL, equivocarse no cuesta simétrico:

- **Predecir de más** (dije 60 días, quedaban 20): la bomba revienta en operación →
  workover no programado, equipo de emergencia, producción diferida larga. Carísimo.
- **Predecir de menos** (dije 20, quedaban 60): saqué una bomba con vida útil adentro →
  intervención prematura, vida desperdiciada. Caro, pero mucho menos.

Hasta el *benchmark académico* lo sabe: el desafío PHM08 de NASA califica con una
penalización **asimétrica** (llegar tarde castiga exponencialmente más que llegar
temprano). La lámina de negocio sale sola: el número que se opera no es el P50 del RUL,
es el percentil que tus costos aguantan — y con eso se conecta con la banda de la C3 y
la matriz de costos del Módulo 3, citadas y re-explicadas (regla 4).

---

## 3 · El dataset: C-MAPSS, y la sustitución declarada

**NASA C-MAPSS (FD001)** — 100 turbomáquinas gemelas corridas **desde sanas hasta la
falla**, 21 sensores por unidad (presiones, temperaturas, velocidades de eje), 20.631
registros. Dominio público (gobierno de EE. UU.), archivo de texto plano, sin registro
ni API.

Por qué este y no otro:

1. **Es el dataset de mantenimiento predictivo más probado andragógicamente que
   existe.** Salió del PHM Data Challenge 2008 y lleva quince años siendo el estándar
   con miles de papers y tutoriales. No hay dataset de fallas con más kilometraje
   docente en el mundo.
2. **Es una flota con muchas variables** — exactamente donde el módulo ya midió que el
   ML cobra (retrospectiva, Parte B). 100 unidades, 15 sensores útiles.
3. **Tiene la física correcta para la sustitución.** Una ESP *es* una turbomáquina
   centrífuga multietapa: ejes que giran, degradación progresiva, sensores de presión y
   temperatura. No existe dato público de fallas de ESP (ya se verificaron 8 fuentes —
   handoff §5; el RIFTS del consorcio ESP es cerrado). Se declara en la lámina de
   acotación, con el mismo precedente de la C2: *«un choke, no una bomba»* → *«una
   turbomáquina de NASA, no una ESP; los costos que le colgamos sí son de ESP»*.
   (Regla 12: la promesa se paga o se retira explícitamente.)
4. **El EDA regala hallazgos**: 6 de los 21 sensores son planos (no miden nada — hay que
   descubrirlo mirando, no leyendo la documentación), y las vidas van de **128 a 362
   ciclos, mediana 199**. Ese rango de casi 3 a 1 es la lámina que mata al
   mantenimiento por calendario antes de entrenar nada.

## 4 · El rival tonto es la política real de la industria

Aquí hay una alineación que ninguna clase anterior tuvo: el modelo tonto de RUL es
`vida mediana de la flota − edad de la unidad`. **Eso no es un hombre de paja: es
literalmente el mantenimiento por calendario** («las bombas duran ~199 ciclos, cámbiala
cuando se acerque»). O sea que el duelo modelo-vs-tonto de la casa *es* el debate real
de negocio: mantenimiento por condición vs mantenimiento por calendario. Ganarle al
tonto = justificar el proyecto de ML frente a la política vigente, en la misma lámina.

## 5 · La viabilidad, medida antes de comprometer nada

Regla de la Parte B: no se compromete una clase sin medir que el modelo le gana al
tonto. Hecho (2026-08-17, script de ~40 líneas, features nivel curso: valor actual +
media y pendiente de ventana de 20 ciclos, 25 unidades apartadas por grupo):

| | MAE (toda la vida) | MAE cuando quedan <50 ciclos |
|---|---|---|
| Calendario (mediana flota − edad) | 35,2 ciclos | 21,0 ciclos |
| Gradient boosting | **24,8 ciclos** | **4,6 ciclos** |

La lectura pedagógica es mejor que el marcador global: en la **zona de la decisión**
(cuando de verdad hay que programar el equipo), el calendario se equivoca por 21 ciclos
y el modelo por menos de 5 — **4,5 veces mejor justo donde vive la plata**. Y el
resultado negativo honesto (regla 11) también está adentro: con la unidad **joven**, el
modelo no sabe más que el calendario — los sensores aún no muestran degradación y no hay
señal que aprender. *El modelo no adivina el futuro lejano; reconoce la degradación
presente.* Esa lámina es la vacuna contra el vendedor de humo, y es de las cosas más
valiosas que se pueden decir en un curso de ML aplicado.

## 6 · Estructura (120 min, 4 bloques, ~45 láminas)

| Min | Bloque | Contenido |
|---|---|---|
| 0–12 | **El encargo y el dinero** | La flota de ESP; workover programado vs falla en operación (costos como *supuestos a la vista*, estilo «La Cuenta» de C4); la política vigente es el calendario; el encargo: «dime cuánta vida le queda a cada una, y a cuáles les programo equipo este mes». Acotación: la sustitución C-MAPSS↔ESP, declarada. |
| 12–40 | **La flota y el reloj** | Qué es una ESP y por qué se degrada; el dataset; EDA: los 6 sensores muertos, las vidas de 128 a 362 (adiós calendario); construir el RUL — el reloj que corre hacia atrás; la trampa de validar sin separar por unidad (cita a C2/C3, GroupKFold re-explicado). |
| 40–70 | **El modelo nuevo** | Gradient boosting desde cero: la cadena que corrige vs el comité que vota; la figura 1→10→300 árboles; el marcador contra el calendario; la zona de la decisión (21 vs 4,6); en qué se fija (importancias con sentido físico); el resultado negativo: la unidad joven. |
| 70–95 | **El precio de equivocarse tarde** | El error con dirección; la penalización asimétrica (hasta NASA califica así); del P50 al percentil que tus costos aguantan; la lista del mes con presupuesto de k intervenciones, en USD — cierre del arco C2 (detectar) → C5 (planear). |
| 95–118 | **Su Turno** | Práctica con andamiaje: 25 unidades apartadas; tabla de features ya construida y primer paso resuelto; predicen RUL, arman la lista del mes bajo presupuesto y escriben el párrafo «Así Sí» con el número en USD. |
| 118–120 | **Cierre** | Lo aprendido; temario: Temas 8 y 9 + Práctica 2 cerrada; **la cuenta del módulo**: 5 de 9 temas + 3 prácticas, y los temas 4–7 diferidos al Módulo 7, en voz alta, como se prometió en C4. |

Sobre el Tema 9 (optimización de levantamiento): se cubre como lo que la decisión de RUL
*es* — maximizar la producción levantada por dólar de intervención, eligiendo cuándo y a
cuál unidad del sistema de levantamiento se le mete mano. Lo que no entra
(dimensionamiento, curvas de gas lift) se nombra como diferido, no se esconde.

## 7 · Piezas a construir

Las cuatro de siempre (handoff §2), en este orden:

1. `preparar_datos.py` — baja FD001 de la fuente NASA (o espejo verificado), documenta
   licencia y transformaciones, escribe `datos/flota_turbomaquinas_rul.csv`.
2. `figuras.py` — figuras + **todas** las cifras impresas: espagueti de 100 vidas,
   histograma 128–362, sensores muertos, degradación de un sensor vs RUL, 1→10→300
   árboles, marcador, error por zona de vida, curva de costo asimétrico, la lista del mes.
3. `presentacion.tex` — preámbulo literal de `modulo3-clase5`.
4. Cuaderno — reproduce exactamente las cifras de `figuras.py`. (Pendiente previo: los
   generadores `mknb*.py` no están en el repo — resolver antes de esta clase.)

Auditorías `auditar_codigo.py` y `auditar_laminas.py` antes de dar por terminada.

## 8 · Chequeo contra las 12 reglas

| Regla | Cómo se cumple |
|---|---|
| 1 · Pregunta, no técnica | «¿Cuánta vida le queda a la bomba?» |
| 2 · Nada sin encargo | El boosting entra porque el encargo pide un número fino en la zona de decisión y el calendario falla ahí por 21 ciclos |
| 3 · Dinero en 10 min | Workover programado vs emergencia abre la clase |
| 4 · Nada por sabido | Árboles (M3C3), banda (M5C3) y matriz de costos (M3C4) citados y re-explicados |
| 5 · Temario textual | Temas 8 y 9 + Práctica 2, en la lámina de cierre |
| 6 · Beamer M3 | Preámbulo literal |
| 7 · 4 bloques, un dataset, una pregunta | Sí |
| 8 · EDA primero | Sensores muertos y vidas 3:1 antes de cualquier modelo |
| 9 · Práctica con andamiaje | Tabla construida, primer paso resuelto, 25 unidades vírgenes |
| 10 · Números generados | Todo sale de `figuras.py`; las cifras de §5 ya salen de un script |
| 11 · Resultado negativo | La unidad joven: sin degradación visible, el modelo no sabe más que el calendario |
| 12 · Promesas | Sustitución ESP declarada; la cuenta del módulo prometida en C4 se paga |

## 9 · Qué se recicla de la v1 y qué se descarta

De la v1 sobrevive el bloque de negocio (costo asimétrico, percentil que se firma, lista
con presupuesto) — comprimido en el bloque 70–95, colgado del RUL en vez de flotar solo.
Se descarta como clase entera: el instructor tiene razón en que sin dato nuevo ni modelo
nuevo era un cierre, no una clase. La deuda de métricas de la C2 (falsas por día) queda
donde estaba: corrección mecánica de la C2, no material de la C5.

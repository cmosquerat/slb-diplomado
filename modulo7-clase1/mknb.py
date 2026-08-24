"""
Machine Learning for Petroleum Engineers Using Python
Modulo 7 - Clase 1: genera el cuaderno de la clase.

Este script ESCRIBE el .ipynb y queda versionado para poder regenerarlo.

El cuaderno tiene TRES DEMOS resueltas (con el prompt que se uso y el codigo
que salio) y TRES EJERCICIOS RETO en los que el alumno arma su propio
prompt y dirige al agente. REGLA DE LA CLASE: las soluciones de los retos
NO existen en ningun archivo -- se construyen en vivo. Este generador la
respeta: no hay ninguna respuesta de los retos aca adentro.

Uso:  python3 mknb.py
"""

import json

NOMBRE = "Modulo7_Clase1_Dirigir_Un_Agente.ipynb"

celdas = []


def _lineas(texto):
    """Cada linea del source lleva su salto al final, menos la ultima."""
    ls = texto.strip("\n").split("\n")
    return [l + "\n" for l in ls[:-1]] + ls[-1:]


def md(texto):
    celdas.append({"cell_type": "markdown", "metadata": {},
                   "source": _lineas(texto)})


def code(texto):
    celdas.append({"cell_type": "code", "execution_count": None,
                   "metadata": {}, "outputs": [],
                   "source": _lineas(texto)})


# =============================================================== PORTADA ====
md(r"""
# Módulo 7 · Clase 1 — Lo que antes necesitaba un programador

**Machine Learning for Petroleum Engineers Using Python** · SLB Ecuador / UDLA

---

### Cómo se usa este cuaderno

La clase alterna dos tipos de sección:

- 🎬 **DEMO** — un mini-proyecto resuelto, con el **prompt exacto** que se le
  dio al agente y lo que salió. Ustedes lo corren y lo discutimos.
- 🛠️ **RETO** — el mismo tipo de problema, en **otro dataset**, y lo
  resuelven **ustedes** dirigiendo al agente. Solo hay pistas y preguntas.

> ⚠️ **Los retos no tienen solución escrita en ninguna parte** — ni en este
> cuaderno, ni en el repositorio. Se construyen en clase, en vivo. Así que si
> se atascan, la jugada correcta es preguntar, no buscar.

**El agente:** el panel de **Gemini** de Colab (el ícono ✨). A él le van a
copiar los prompts.

**No hace falta haber programado nunca.**

### Hoy, ustedes son la consultora

Así llega el trabajo de verdad: **nadie llega con un dataset**. Llegan con un
problema, una decisión que no pueden aplazar, y un costo si se equivocan.
Hoy pasan **cuatro clientes** por su mesa — una empresa de movilidad (mis
demos), un gerente de flota, un área HSE y un comprador de diamantes (sus
retos).

> **Regla de la consultora:** no se abre ningún dato sin poder decir antes el
> **problema**, la **decisión** y el **costo del error**. Con un agente, esa
> regla vale doble.

---

### El proyecto del módulo — desde ya

Este módulo se aprueba con **una aplicación que resuelva un problema real de
su trabajo**, construida dirigiendo a un agente, con su link funcionando.

- **Equipos de hasta 3 personas** (solo también vale).
- **Entrega: hasta el jueves de la próxima semana.**
- **Esta semana** se introducen las herramientas **reales** con las que se
  produce ML en la industria hoy: agentes de código, servir un modelo, apps
  que otros usan. Cada clase le suma una pieza a su proyecto.
- Se entrega con los **prompts** que usaron, **un error del agente que
  ustedes cazaron**, y qué **decisión** habilita la herramienta.

**Tarea de hoy mismo:** elijan equipo y problema. El problema correcto es
uno donde alguien pierde tiempo con una hoja de cálculo.
""")

md(r"""
## 0 · Preparación

Una sola celda. Todo viene instalado en Colab.
""")

code(r"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ¿estamos en Colab? la parte de la app lo usa al final
try:
    import google.colab          # noqa: F401
    EN_COLAB = True
except ImportError:
    EN_COLAB = False

print("listo. ¿en Colab?", EN_COLAB)
""")

md(r"""
## 0.1 · Las herramientas de hoy, en una frase cada una

| herramienta | qué es |
|---|---|
| **Colab** | el taller: un cuaderno de Python en el navegador, gratis |
| **Gemini** | el contratista: el agente que vive dentro de Colab |
| **pandas** | la mesa de trabajo: deja los datos en *tablas* manejables con código |
| **seaborn** | el dibujante: gráficas con las decisiones de diseño ya tomadas — y trae los datos de práctica de hoy |
| **Gradio** | la vitrina: convierte una función en una página web con un link |

## 0.2 · Los cuatro datos de hoy

Todos vienen **dentro de seaborn**: una línea y están cargados.

| dataset | el cliente | quién lo trabaja |
|---|---|---|
| `taxis` | la empresa de movilidad que reparte mal su flota | 🎬 las demos |
| `mpg` | el gerente de flota que defiende la renovación ante finanzas | 🛠️ reto 1 |
| `car_crashes` | el área HSE con presupuesto para UNA campaña | 🛠️ reto 2 |
| `diamonds` | el comprador al que le exigen el modelo de precios *ya* | 🛠️ reto 3 |

A propósito ninguno es de petróleo: la habilidad es la misma en cualquier
dominio, y el proyecto del módulo es con **su** dato.
""")

code(r"""
taxis = sns.load_dataset("taxis")

# preparamos las columnas de tiempo que van a usar las demos
taxis["pickup"] = pd.to_datetime(taxis.pickup)
taxis["dropoff"] = pd.to_datetime(taxis.dropoff)
taxis["hora"] = taxis.pickup.dt.hour          # 0 a 23
taxis["dia"] = taxis.pickup.dt.dayofweek      # 0 = lunes

print(f"{len(taxis)} viajes, de {taxis.pickup.min().date()} "
      f"a {taxis.pickup.max().date()}")
taxis.head(3)
""")

# ================================================== DEMO 1 ==================
md(r"""
---

# 🎬 Demo 1 · Una gráfica que no hay que explicar

**El caso.** Nuestro primer cliente: una empresa de movilidad con cientos de
carros. En las horas muertas los carros están **parados quemando costo
fijo**; en las pico **no alcanzan** y los viajes se los lleva la
competencia. El gerente de operaciones arma los **turnos del mes** la
próxima semana, y el error duele en las dos direcciones: turno de más es
nómina parada, turno de menos es ingreso regalado.

**Lo que nos pidió:** *«díganme cuándo hay demanda, en algo que yo pueda
mostrar en el comité sin que ustedes estén al lado explicando».*

Recién ahora — con problema, decisión y costo claros — abrimos el dato.
Primero, lo que sale sin pensar:
""")

code(r"""
por_hora = taxis.groupby("hora").size()

plt.plot(por_hora.index, por_hora.values)
plt.show()
""")

md(r"""
Correcta y muda: no dice qué son los ejes, usa una línea para categorías, y
**no dice cuál es la respuesta**.

### El encargo que le di al agente

```
Tengo un DataFrame `taxis` de seaborn con viajes de taxi de Nueva York.
Columnas: pickup y dropoff (fecha y hora, ya convertidas), distance, fare,
tip, total, payment, pickup_zone, pickup_borough. Ya calculé hora (0-23)
y dia (0=lunes).

OBJETIVO: quiero saber a qué hora del día hay más demanda de taxis.

RESTRICCIONES:
- usa seaborn
- va en un reporte para gerencia: tiene que entenderse sin que yo lo explique
- ejes con nombre y unidades, en español

CRITERIO DE ACEPTACIÓN:
- la hora pico marcada y legible de un vistazo
- avísame si hay horas sin ningún viaje registrado
```

Fíjense dónde está cada uno de los **cuatro fundamentos**: objetivo,
contexto, restricciones, criterio de aceptación. Y lo que salió:
""")

code(r"""
def grafico_demanda(datos, titulo="Demanda de taxis por hora del día"):
    '''Lo que devolvio el agente para la demo 1 (ligeramente ordenado).'''
    por_hora = datos.groupby("hora").size().reindex(range(24), fill_value=0)

    fig, ax = plt.subplots(figsize=(9, 4))
    pico = int(por_hora.idxmax())

    colores = ["#C82B40" if h == pico else "#D8DAE0" for h in por_hora.index]
    ax.bar(por_hora.index, por_hora.values, color=colores)
    ax.set_xlabel("hora del día")
    ax.set_ylabel("cantidad de viajes")
    ax.set_xticks(range(0, 24, 2))
    ax.set_title(titulo, loc="left", fontweight="bold")
    ax.annotate(f"hora pico: {pico}h  ({por_hora.max()} viajes)",
                xy=(pico, por_hora.max()),
                xytext=(pico - 9, por_hora.max() * 0.92),
                color="#C82B40", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#C82B40"))
    sns.despine(ax=ax)

    sin_viajes = int((por_hora == 0).sum())
    if sin_viajes:
        print(f"aviso: {sin_viajes} horas sin ningún viaje registrado")
    return fig


fig = grafico_demanda(taxis)
plt.show()
""")

md(r"""
### Las cinco decisiones que separan las dos gráficas

1. **Barras en vez de línea** — las horas son categorías, no un continuo.
2. **Un solo color con sentido** — gris todo, rojo lo que importa.
3. **Ejes con nombre**, en el idioma del que lee.
4. **El hallazgo escrito sobre la figura** — no en la boca del presentador.
5. **Se quitó lo que sobra.**

Ninguna es programación. Las cinco son **criterio de comunicación** — por eso
las deciden ustedes, no el agente.
""")

# ================================================== RETO 1 ================
md(r"""
---

# 🛠️ Reto 1 · La flota  *(12 min)*

**EL CASO.** Su cliente administra una **flota de vehículos** y la próxima
semana defiende ante finanzas el plan de **renovación**. La duda que lo
frena: *¿los modelos más nuevos de verdad consumen menos, o es un mito
caro?* De la respuesta depende **aprobar o aplazar** una compra grande.

> **SU ENCARGO:** respondan con el dato — **¿el consumo mejoró con los años,
> y cuánto?**

Columnas que les sirven: `mpg` (millas por galón — **más es mejor**),
`model_year`, `origin`, `weight`.

**Armen su propio prompt** con los cuatro fundamentos y páselo al panel de
Gemini. Esta plantilla es la que se llevan del curso:

```
CONTEXTO: tengo [qué datos], con columnas [cuáles] en [qué unidades].
Es para [quién].

OBJETIVO: quiero responder [qué pregunta].

RESTRICCIONES:
- usa [qué librerías]
- [qué NO hacer]
- en español, ejes con unidades

CRITERIO DE ACEPTACIÓN:
- [qué tiene que verse o cumplirse]
- avísame si [qué problema podría tener el dato]
```
""")

code(r"""
# los datos de SU ejercicio
mpg = sns.load_dataset("mpg")
print(f"{len(mpg)} vehículos")
mpg.head(3)
""")

code(r"""
# ✍️ peguen aquí lo que les devuelva el agente, y córranlo
""")

md(r"""
**Criterio de aceptación, en el idioma del cliente:**

- Finanzas mira la lámina **30 segundos**. ¿Se defiende sola, con el número
  de mejora escrito encima?
- Este dataset tiene **valores faltantes** en una columna. Si finanzas
  pregunta *«¿con cuántos vehículos calculó esto?»*, ¿tienen la respuesta —
  o el agente se los tragó sin avisar? *(si no avisó: ¿qué le faltó a su
  criterio de aceptación?)*

⏸️ **Aquí paramos y lo armamos juntos antes de seguir.**
""")

# ================================================== DEMO 2 ==================
md(r"""
---

# 🎬 Demo 2 · Una pregunta de operación

**El mismo cliente, una semana después.** Ya sabe a qué hora hay demanda.
Pero los turnos no se arman por hora suelta: se arman **por día y hora** —
el viernes a las 17h no es el martes a las 17h. Su decisión necesita el
cruce de las dos cosas.

### El encargo que le di

```
CONTEXTO: el mismo DataFrame `taxis`. Ya tiene hora (0-23) y
dia (0=lunes ... 6=domingo).

OBJETIVO: saber en qué combinaciones de día y hora se concentra la
demanda, para decidir turnos.

RESTRICCIONES:
- un mapa de calor con seaborn: días en filas, horas en columnas
- días en español, en orden de lunes a domingo
- nada de escalas rojo-verde (hay daltónicos en la sala)

CRITERIO DE ACEPTACIÓN:
- que se pueda señalar con el dedo el peor momento
- dime cuál es y cuántos viajes tiene
```

La restricción del rojo-verde no es estética: es que el gráfico lo pueda leer
**toda** la sala. Ese tipo de restricción solo la pone quien conoce a su
público. Lo que salió:
""")

code(r"""
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]

tabla = (taxis.groupby(["dia", "hora"]).size()
         .unstack(fill_value=0)
         .reindex(index=range(7), columns=range(24), fill_value=0))

fig, ax = plt.subplots(figsize=(11, 3.4))
sns.heatmap(tabla, cmap="rocket_r", ax=ax,
            cbar_kws={"label": "viajes"}, linewidths=.4, linecolor="white")
ax.set_yticklabels(DIAS, rotation=0)
ax.set_xlabel("hora del día")
ax.set_ylabel("")
ax.set_title("¿Cuándo conviene tener más gente en la calle?",
             loc="left", fontweight="bold")
plt.show()

d, h = np.unravel_index(np.argmax(tabla.values), tabla.shape)
print(f"momento de más demanda: {DIAS[d]} a las {tabla.columns[h]}h "
      f"({tabla.values.max()} viajes)")
""")

# ================================================== RETO 2 ================
md(r"""
---

# 🛠️ Reto 2 · Seguridad vial  *(12 min)*

**EL CASO.** Su cliente mueve carga por carretera en todo EE. UU. El área
**HSE** tiene presupuesto para **una sola campaña** de seguridad vial este
año, y el comité discute dos cosas: *¿el alcohol es de verdad el factor que
acompaña los accidentes?* y *¿en qué estados invertir?* Una campaña en el
lugar equivocado es plata quemada — y un año perdido.

> **SU ENCARGO:** muestren la **relación alcohol–accidentes** y propongan
> **tres estados** para la campaña.

Columnas que les sirven: `total` (accidentados por cada 100 millones de
millas conducidas), `alcohol`, `speeding`, `abbrev` (el estado).

Pista de diseño: comparar dos variables pide un tipo de gráfico; señalar
estados prioritarios pide otra cosa encima — y ambas se le pueden pedir al
agente en el mismo encargo.
""")

code(r"""
# los datos de SU ejercicio
crashes = sns.load_dataset("car_crashes")
print(f"{len(crashes)} estados")
crashes.head(3)
""")

code(r"""
# ✍️ peguen aquí lo que les devuelva el agente, y córranlo
""")

md(r"""
**Criterio de aceptación, en el idioma del cliente:**

- ¿El comité *ve* la relación en la figura, o tiene que creerles de palabra?
- Sus tres estados: ¿pueden defender **por qué esos y no otros**?
- Si el gerente concluye *«el alcohol causa los accidentes, listo»* — ¿se lo
  dejan pasar? ¿Qué le contestan?

⏸️ **Paramos y lo discutimos.**
""")

# ================================================== DEMO 3 ==================
md(r"""
---

# 🎬 Demo 3 · Donde el agente falla con confianza

**El mismo cliente, tercera visita — ahora con plata en juego.** La gerencia
comercial quiere **empujar el pago con tarjeta**, convencida de que así «se
recuperan» propinas que hoy se pierden. Nos piden el dato que respalde la
política:

> **¿Los pasajeros que pagan en efectivo dejan menos propina que los que
> pagan con tarjeta?**

De la respuesta depende una decisión con dinero: comisiones de tarjeta,
incentivos, terminales nuevas. El agente responde en veinte segundos, con un
gráfico impecable:
""")

code(r"""
propinas = taxis.groupby("payment").tip.mean()

fig, ax = plt.subplots(figsize=(6, 3.6))
ax.bar(["Efectivo", "Tarjeta"],
       [propinas["cash"], propinas["credit card"]],
       color=["#9CA3AF", "#16A34A"], width=.5)
for i, v in enumerate([propinas["cash"], propinas["credit card"]]):
    ax.text(i, v + .06, f"USD {v:.2f}", ha="center", fontweight="bold")
ax.set_ylabel("propina promedio [USD]")
ax.set_title("Propina promedio según forma de pago", loc="left",
             fontweight="bold")
sns.despine(ax=ax)
plt.show()
""")

md(r"""
La conclusión obvia — *«quien paga en efectivo nunca deja propina»* — es
**falsa**. Cuenten antes de creer:
""")

code(r"""
efectivo = taxis[taxis.payment == "cash"]

print(f"viajes en efectivo:                  {len(efectivo)}")
print(f"de esos, con propina mayor que cero: {(efectivo.tip > 0).sum()}")
""")

md(r"""
**Exactamente cero.** Ese número perfecto es la pista: en datos reales, nada
es exactamente cero por casualidad.

> El taxímetro **solo registra la propina cuando va en la tarjeta**. La de
> efectivo va al bolsillo del conductor y nunca entra al sistema. La columna
> no dice «no dejó propina»: dice **«no tengo ese dato»**.

**Esto ya lo vivieron:** es el mismo error del módulo del agua — campos con
cero agua producida durante veinticinco años, porque el regulador no la
exigía hasta el año 2000. Distinto dominio, distinto país, **el mismo
error**. Y lo que lo detectó no fue una técnica: fue alguien que conocía el
negocio y dijo *«esto no puede ser»*.

### El encargo que convierte esa alarma en método

```
Antes de concluir nada de `taxis`: revisión de calidad. Filas totales y
faltantes; mínimo/máximo/ceros por columna numérica; y DIME si algún grupo
tiene SIEMPRE el mismo valor (suele ser artefacto del registro). Lista lo
que te parezca raro, aunque no estés seguro.
```

Guárdenlo: se corre con **cada dataset nuevo, antes de cualquier análisis**.

### La lista de verificación (para todo lo que devuelva un agente)

1. ¿Cuántas filas entraron y cuántas salieron?
2. ¿Hay ceros, vacíos o números redondos sospechosos?
3. ¿El resultado tiene sentido físico o de negocio?
4. ¿Qué decisión se toma con esto, y qué pasa si está mal?

Ninguna exige saber programar. Las cuatro exigen saber del negocio.
""")

# ================================================== RETO 3 ================
md(r"""
---

# 🛠️ Reto 3 · Los diamantes  *(12 min)*

**EL CASO.** Su cliente va a comprar un **lote grande de piedras** y el área
comercial exige el modelo de precios *para ayer*. Pero si el modelo se
entrena con datos sucios, **tasa mal** — y comprar caro no se nota hasta que
toca vender. Ustedes son el último filtro: deciden si este dato se
**certifica** o se **devuelve**.

> **SU ENCARGO:** antes de que nadie entrene nada, **revisión de calidad**.
> Este dataset esconde **al menos un valor físicamente imposible** — y más
> de uno.

Columnas: `carat` (quilates), `cut` (calidad del corte), `price` [USD], y
`x`, `y`, `z` — las **dimensiones físicas de la piedra, en milímetros**.
""")

code(r"""
# los datos de SU ejercicio
diamonds = sns.load_dataset("diamonds")
print(f"{len(diamonds)} diamantes")
diamonds.head(3)
""")

code(r"""
# ✍️ peguen aquí la revisión de calidad que les devuelva el agente
""")

md(r"""
**Criterio de aceptación, en el idioma del cliente:**

- ¿Cuántas filas **rechazan**, y con qué argumento físico? *(¿qué significa
  que una dimensión valga cero? ¿y el máximo de cada una es creíble? — un
  quilate típico mide unos 6 mm)*
- El correo al proveedor del dato, en **dos frases**: ¿qué dice?
- La firma: ¿dejarían entrenar el modelo **hoy** — sí o no?

⏸️ **Paramos: cada grupo dice qué encontró y qué haría.**
""")

# ================================================== PARTE 4 · LA APP ========
md(r"""
---

# 🎬 Demo 4 · De la gráfica a una herramienta

Mientras el análisis viva en una celda, cada vez que alguien quiera ver otra
cosa **tiene que llamarlos**. Una app se la mandan y se acabó.

**Gradio** convierte una función de Python en una página web con controles, y
en Colab genera un **link público** para compartir.

> ⚠️ El link vive **mientras el cuaderno esté corriendo**. Perfecto para una
> reunión; no es un sistema permanente.

La app más chica que sirve — la función de la Demo 1, con un control:
""")

code(r"""
taxis["dia_nombre"] = taxis.dia.map(dict(enumerate(DIAS)))


def ver_demanda(dia_elegido):
    datos = taxis[taxis.dia_nombre == dia_elegido]
    return grafico_demanda(datos, titulo=f"Demanda de taxis · {dia_elegido}")


# probamos la funcion sola, antes de montar nada
_ = ver_demanda("viernes")
plt.show()
""")

code(r"""
# En Colab esta celda publica el link. Fuera de Colab solo construye la
# interfaz (para poder verificar que el cuaderno corre entero).
try:
    import gradio as gr

    demo = gr.Interface(
        fn=ver_demanda,
        inputs=gr.Dropdown(DIAS, value="viernes", label="Día de la semana"),
        outputs=gr.Plot(label="Demanda por hora"),
        title="Demanda de taxis por día",
        description="Elija un día para ver en qué horas se concentran los viajes.")

    if EN_COLAB:
        demo.launch(share=True)      # <- aquí aparece el link público
    else:
        print("interfaz construida; en Colab esta celda abre el link")
except ImportError:
    print("gradio no está en este entorno. En Colab ya viene listo;")
    print("si hiciera falta:  !pip install gradio")
""")

# ================================================== SU TURNO FINAL ==========
md(r"""
---

# 🛠️ Su turno final · La app de SU dato  *(18 min)*

**EL CASO.** Su cliente — el de la flota, el de HSE o el de los diamantes,
**elijan uno** — quedó contento con el análisis. Y pidió lo que piden todos
los clientes contentos: *«déjenme la herramienta a mí, que no quiero
llamarlos cada vez»*.

> **SU ENCARGO:** conviertan su análisis en una **app de Gradio con link**.
>
> La app debe tener **al menos dos controles** (qué filtrar o comparar — lo
> deciden ustedes), mostrar **una gráfica que se entienda sola**, y **no
> caerse** si la combinación elegida no tiene datos.

Armen el prompt con los cuatro fundamentos. El caso «sin datos» va en el
**criterio de aceptación** — ese renglón es el que evita la llamada del jefe
diciendo «se me rompió».
""")

code(r"""
# ✍️ peguen aquí su app, y córranla
""")

md(r"""
**Antes de mandar el link, rómpanlo ustedes primero:**

- Elijan la combinación más rara que permitan sus controles. ¿Sobrevive?
- ¿El texto que muestra coincide con lo que muestra la gráfica?
- ¿Se lo mandarían a su jefe tal como está? Si dudan — ¿qué le falta?

⏸️ **Cierre: tres voluntarios abren su link en el proyector.**
""")

# ================================================== CIERRE ==================
md(r"""
---

# Cierre

## Lo que vimos → lo que decidimos

| Lo que vimos | Lo que decidimos |
|---|---|
| La gráfica por defecto no comunica | Cinco decisiones de diseño, siempre las mismas |
| El agente responde según cómo se le pida | Los **cuatro fundamentos** en cada encargo |
| El criterio de aceptación casi nunca se escribe | Escribirlo siempre — es lo que más cambia el resultado |
| 1.812 viajes en efectivo y **cero** propinas | Desconfiar de todo cero perfecto |
| El agente no dio ninguna señal de alarma | **Verificar siempre**: las cuatro preguntas |
| Un análisis en una celda no lo usa nadie más | Convertirlo en app con link |

## Los dos resultados negativos de hoy

1. **El agente entregó un análisis correcto con una conclusión falsa**, sin
   ninguna señal de alarma. El código estaba bien; el dato mentía.
2. En un ensayo controlado, desarrolladores experimentados fueron **19% más
   lentos** con estas herramientas — sintiéndose 20% más rápidos.

Ninguno es razón para no usarlas. Los dos son razón para **medir en vez de
confiar en la sensación**.

## Si se llevan una sola cosa

> **El agente escribe el código. Ustedes responden por el resultado.**

Escribir se volvió barato; saber si el resultado tiene sentido, no. La
habilidad de hoy no es «usar Gemini»: es **especificar bien y verificar
siempre** — y sirve con cualquier herramienta que salga el año que viene.

---

## Tarea para la próxima clase

Equipo (**máximo 3**) y problema elegidos, escritos en dos frases: **qué
duele**, y **quién decidiría distinto** con una herramienta.

Recuerden el plazo: la entrega del proyecto es **hasta el jueves de la
próxima semana** — su app con link funcionando, los prompts que usaron,
**un error del agente que ustedes cazaron**, y qué decisión se toma con la
herramienta.

---

*Machine Learning for Petroleum Engineers Using Python* · SLB Ecuador / UDLA
· 2026
""")

# ================================================================ ESCRIBIR ==
nb = {
    "cells": celdas,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "colab": {"provenance": [], "toc_visible": True},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(NOMBRE, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

n_md = sum(1 for c in celdas if c["cell_type"] == "markdown")
n_code = sum(1 for c in celdas if c["cell_type"] == "code")
texto = " ".join("".join(c["source"]) for c in celdas)
demos = texto.count("🎬")
retos = texto.count("🛠️")
print(f"escrito {NOMBRE}: {len(celdas)} celdas ({n_md} md, {n_code} codigo)")
print(f"  demos: {demos} menciones | retos: {retos} menciones")
print("  verificacion anti-spoiler: no debe haber respuestas de retos")
for palabra in ["31.7", "31,7", "0.85", "0,85", "20 filas", "58.9", "58,9"]:
    if palabra in texto:
        raise SystemExit(f"  SPOILER DETECTADO: '{palabra}' — revisar")
print("  sin spoilers: OK")

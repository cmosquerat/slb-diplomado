"""
Machine Learning for Petroleum Engineers Using Python
Modulo 7 - Clase 1: genera el cuaderno de la clase.

Este script ESCRIBE el .ipynb. Se versiona en el repo a proposito: asi el
cuaderno se puede editar aca y regenerar, en vez de tocar el JSON a mano.

El cuaderno es GUIADO: cada ejercicio trae el prompt literal que el alumno
copia y pega en el panel de Gemini de Colab, y una celda de referencia con
el resultado esperado.

Uso:  python3 mknb.py
"""

import json

NOMBRE = "Modulo7_Clase1_Dirigir_Un_Agente.ipynb"

celdas = []


def _lineas(texto):
    """En el .ipynb cada linea del source lleva su salto al final, menos la
    ultima. Si se omite, todo el codigo queda pegado en un renglon."""
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

Este cuaderno es **guiado**. Cada ejercicio tiene tres partes:

1. 🤖 **Un prompt para copiar y pegar** en el panel de Gemini de Colab
   (el ícono ✨ abajo a la derecha, o el botón *Gemini* arriba).
2. ✍️ **Una celda vacía** donde pegan lo que el agente les devuelva.
3. ✅ **Una celda de referencia** con una solución posible, para comparar.

> **No copien la referencia sin intentarlo antes.** La habilidad que se
> llevan de esta clase no es tener el código: es saber pedirlo y saber
> revisarlo.

### Lo que van a lograr hoy

- Que sus gráficas se entiendan sin que ustedes las expliquen.
- Dirigir a un agente con la precisión con la que le encargan un trabajo a
  un contratista.
- Detectar un error que el agente **no** va a detectar.
- Y terminar con una aplicación funcionando, con su link.

**No hace falta haber programado nunca.**
""")

md(r"""
## 0 · Preparación

Una sola celda. Todo esto ya viene instalado en Colab; solo lo estamos
trayendo al cuaderno.
""")

code(r"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ¿estamos en Colab? algunas celdas del final lo necesitan
try:
    import google.colab          # noqa: F401
    EN_COLAB = True
except ImportError:
    EN_COLAB = False

print("listo. ¿en Colab?", EN_COLAB)
""")

# ================================================== PARTE 1 · GRAFICAS ======
md(r"""
---

# Parte 1 · Que sus datos se vean como lo que valen

## 1.1 · Los datos de hoy

Vamos a trabajar con **viajes de taxi de Nueva York**. No son datos de
petróleo a propósito: quiero que se vea que la habilidad no depende del tema.
En el proyecto final cada uno trae los suyos.

Seaborn trae varios conjuntos de práctica listos. Se cargan con una línea y
no hay que descargar nada.
""")

code(r"""
taxis = sns.load_dataset("taxis")

print(f"{len(taxis)} viajes")
print(f"columnas: {list(taxis.columns)}")
taxis.head()
""")

md(r"""
Cada fila es un viaje. Las columnas que nos importan hoy:

| columna | qué es |
|---|---|
| `pickup` / `dropoff` | fecha y hora de inicio y fin |
| `distance` | distancia en millas |
| `fare` / `tip` / `total` | tarifa, propina y total en dólares |
| `payment` | `cash` (efectivo) o `credit card` (tarjeta) |
| `pickup_borough` | barrio donde subió el pasajero |

Lo primero, siempre: preparar las columnas de fecha para poder usarlas.
""")

code(r"""
# pandas las lee como texto; hay que decirle que son fechas
taxis["pickup"] = pd.to_datetime(taxis.pickup)
taxis["dropoff"] = pd.to_datetime(taxis.dropoff)

# y de ahí sacamos las piezas que vamos a usar
taxis["hora"] = taxis.pickup.dt.hour           # 0 a 23
taxis["dia"] = taxis.pickup.dt.dayofweek       # 0 = lunes
taxis["minutos"] = (taxis.dropoff - taxis.pickup).dt.total_seconds() / 60

print(f"desde {taxis.pickup.min()} hasta {taxis.pickup.max()}")
print(f"duración mediana de un viaje: {taxis.minutos.median():.0f} minutos")
print(f"tarifa mediana: USD {taxis.total.median():.2f}")
""")

md(r"""
## 1.2 · La gráfica que sale por defecto

Preguntemos algo simple: **¿a qué hora hay más demanda de taxis?**
""")

code(r"""
por_hora = taxis.groupby("hora").size()

plt.plot(por_hora.index, por_hora.values)
plt.show()
""")

md(r"""
Funciona. Los números están bien. Y sin embargo, si esto aparece en una
reunión, alguien va a preguntar «¿qué estoy viendo?».

**Miren todo lo que le falta:**

- No dice qué son los ejes ni en qué unidades.
- No tiene título.
- Usa una línea, que sugiere continuidad — pero las horas del día son
  categorías separadas, no una serie continua.
- Y sobre todo: **no dice cuál es la respuesta**. Hay que buscarla con el ojo.

> Un gráfico que hay que explicar hablando es un gráfico que falló. Es lo
> único que la mayoría de la gente va a ver de todo su trabajo.

## 1.3 · 🤖 Su primer encargo al agente

Abran el panel de Gemini y peguen esto **tal cual**. Fíjense en que no dice
«hazme un gráfico bonito»: dice qué quiero saber, con qué datos, con qué
restricciones y cómo sabremos que quedó bien.

```
Tengo un DataFrame `taxis` de seaborn con viajes de taxi de Nueva York.
Columnas: pickup y dropoff (fecha y hora, ya convertidas), distance, fare,
tip, total, payment, pickup_zone, pickup_borough. Ya calculé las columnas
hora (0-23) y dia (0=lunes).

OBJETIVO: quiero saber a qué hora del día hay más demanda de taxis.

RESTRICCIONES:
- usa seaborn
- el gráfico va en un reporte para gerencia: tiene que entenderse sin que
  yo lo explique
- ejes con nombre y unidades, en español

CRITERIO DE ACEPTACIÓN:
- la hora pico tiene que estar marcada y ser legible de un vistazo
- avísame si hay horas sin ningún viaje registrado
```
""")

md(r"""
### ✍️ Peguen aquí lo que les devolvió el agente
""")

code(r"""
# pega aquí el código del agente y ejecútalo
""")

md(r"""
### ✅ Una solución posible

No es *la* respuesta correcta — es una de muchas. Compárenla con la suya y
fíjense en las **cinco decisiones** que la separan de la primera gráfica.
""")

code(r"""
def grafico_demanda(datos, titulo="Demanda de taxis por hora del día"):
    '''Grafica cuantos viajes empiezan en cada hora del dia.

    Devuelve la figura, para poder reutilizarla despues en la app.
    '''
    por_hora = datos.groupby("hora").size()
    # 1. las horas sin viajes existen aunque no aparezcan en el groupby
    por_hora = por_hora.reindex(range(24), fill_value=0)

    fig, ax = plt.subplots(figsize=(9, 4))
    pico = int(por_hora.idxmax())

    # 2. barras, no linea: las horas son categorias
    # 3. un solo color con sentido: gris todo, rojo lo que importa
    colores = ["#C82B40" if h == pico else "#D8DAE0" for h in por_hora.index]
    ax.bar(por_hora.index, por_hora.values, color=colores)

    # 4. los ejes dicen que son, en el idioma del que lee
    ax.set_xlabel("hora del día")
    ax.set_ylabel("cantidad de viajes")
    ax.set_xticks(range(0, 24, 2))
    ax.set_title(titulo, loc="left", fontweight="bold")

    # 5. el hallazgo, escrito sobre la figura
    ax.annotate(f"hora pico: {pico}h  ({por_hora.max()} viajes)",
                xy=(pico, por_hora.max()),
                xytext=(pico - 9, por_hora.max() * 0.92),
                color="#C82B40", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#C82B40"))

    sns.despine(ax=ax)
    return fig, pico, int(por_hora.max()), int((por_hora == 0).sum())


fig, pico, n_pico, horas_vacias = grafico_demanda(taxis)
plt.show()

print(f"hora pico: {pico}h con {n_pico} viajes")
print(f"horas sin ningún viaje: {horas_vacias}")
""")

md(r"""
### Las cinco decisiones

Ninguna es programación. Las cinco son **criterio de comunicación**, y por
eso las tienen que decidir ustedes:

1. **Barras en vez de línea** — las horas son categorías, no un continuo.
2. **Un solo color con sentido** — gris todo, rojo lo que importa. El color
   señala, no decora.
3. **Ejes con nombre** en el idioma del que lee.
4. **El hallazgo escrito sobre la figura** — si la conclusión es «el pico es
   a las 18h», eso va en la imagen, no en la boca del que presenta.
5. **Se quitó lo que sobra** — sin marco, sin cuadrícula de más.

## 1.4 · La misma habilidad, otro problema

Para que quede claro que esto no es «un truco para datos de taxis», la misma
receta sobre otro dominio completamente distinto: **gasto en salud contra
expectativa de vida**, por país.
""")

code(r"""
salud = sns.load_dataset("healthexp")
print(f"{len(salud)} filas | {salud.Country.nunique()} países | "
      f"{salud.Year.min()}-{salud.Year.max()}")

ultimo = salud[salud.Year == salud.Year.max()]

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.scatter(ultimo.Spending_USD / 1000, ultimo.Life_Expectancy,
           s=90, color="#16A34A", alpha=.85)
for _, r in ultimo.iterrows():
    ax.annotate(r.Country, (r.Spending_USD / 1000, r.Life_Expectancy),
                xytext=(5, 4), textcoords="offset points", fontsize=9)
ax.set_xlabel("gasto en salud por persona [miles de USD al año]")
ax.set_ylabel("expectativa de vida [años]")
ax.set_title(f"¿El gasto se traduce en resultado?  ({salud.Year.max()})",
             loc="left", fontweight="bold")
sns.despine(ax=ax)
plt.show()
""")

md(r"""
Misma técnica, pregunta de gestión presupuestal. Un país gasta bastante más
que los demás y no vive más — y esa conclusión salta sola de la figura.

**Ese es el punto de todo el módulo:** la habilidad de construir herramientas
sobre datos es la misma, sea producción de un campo, viajes de un taxi, o
presupuesto de un área.
""")

# ============================================ PARTE 2 · LOS FUNDAMENTOS =====
md(r"""
---

# Parte 2 · Cómo se le pide bien a un agente

## 2.1 · Tres cosas distintas que la gente llama «la IA»

| | qué hace | quién ve el resultado |
|---|---|---|
| **Autocompletado** | sugiere el final de la línea que escribes | tú |
| **Asistente** | le preguntas, te contesta con código para copiar | tú lo corres |
| **Agente** | escribe, **ejecuta**, **mira el error** y **corrige solo** | él mismo, y repite |

> La diferencia no es que el agente adivine mejor: es que **itera**. Y por
> eso, cuando se equivoca, se equivoca con más convicción — ya probó, ya
> corrigió, y te entrega algo que *corre*.

## 2.2 · Los cuatro fundamentos

No son trucos de *prompt*. Son las cuatro cosas que uno le dice a cualquiera
a quien le encarga un trabajo:

| | qué es | ejemplo malo | ejemplo bueno |
|---|---|---|---|
| **Objetivo** | qué quiero *saber* (no qué gráfico quiero) | «hazme un gráfico» | «quiero saber a qué hora hay más demanda» |
| **Contexto** | qué datos hay, cómo se llaman, en qué unidades, para quién es | — | «DataFrame `taxis`, columna `hora` de 0 a 23, es para gerencia» |
| **Restricciones** | qué usar, qué no hacer, en qué idioma | — | «usa seaborn, en español, sin colores rojo-verde» |
| **Criterio de aceptación** | **cómo sabemos que quedó bien** | — | «la hora pico marcada, y avísame si faltan datos» |

El cuarto es el que casi nadie escribe y el que más cambia el resultado. Es
el que convierte un pedido en un **encargo de ingeniería**.

## 2.3 · La plantilla que se llevan

Guárdenla. Sirve para pedir una gráfica, un análisis o una aplicación entera:

```
CONTEXTO: tengo [qué datos], con columnas [cuáles] en [qué unidades].
Es para [quién / qué reunión].

OBJETIVO: quiero responder [qué pregunta].

RESTRICCIONES:
- usa [qué librerías]
- [qué NO hacer]
- en español, ejes con unidades

CRITERIO DE ACEPTACIÓN:
- [qué tiene que verse o cumplirse]
- avísame si [qué problema podría tener el dato]
```

Ese último renglón —«avísame si...»— es el que convierte al agente en un
aliado para **encontrar** problemas, en vez de en alguien que los tapa.

## 2.4 · 🤖 Segundo encargo: una pregunta de operación

Ahora una pregunta que un jefe de operaciones hace de verdad:
**¿cuándo conviene tener más gente en la calle?**

```
CONTEXTO: el mismo DataFrame `taxis`. Ya tiene las columnas hora (0-23) y
dia (0=lunes ... 6=domingo).

OBJETIVO: saber en qué combinaciones de día de la semana y hora se concentra
la demanda, para decidir turnos de conductores.

RESTRICCIONES:
- un mapa de calor con seaborn: días en las filas, horas en las columnas
- los días en español y en orden de lunes a domingo
- no uses una escala de color rojo-verde (hay daltónicos en la sala)

CRITERIO DE ACEPTACIÓN:
- que yo pueda señalar con el dedo el momento de más demanda
- dime cuál es ese momento y cuántos viajes tiene
```
""")

md(r"""
### ✍️ Peguen aquí lo que les devolvió el agente
""")

code(r"""
# pega aquí el código del agente y ejecútalo
""")

md(r"""
### ✅ Una solución posible
""")

code(r"""
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]

# una fila por dia, una columna por hora, el valor es cuantos viajes hubo
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

# el momento de mas demanda
d, h = np.unravel_index(np.argmax(tabla.values), tabla.shape)
print(f"momento de más demanda: {DIAS[d]} a las {tabla.columns[h]}h "
      f"con {tabla.values.max()} viajes")
""")

md(r"""
Con eso ya se puede decidir un turno. Y noten quién tomó la decisión: **el
agente hizo el gráfico, pero la pregunta la hicieron ustedes** — y la pregunta
es la que vale.
""")

# ============================================== PARTE 3 · VERIFICAR =========
md(r"""
---

# Parte 3 · La parte incómoda: dónde falla

Hasta acá todo salió bien, y esa es justamente la trampa. Un agente que
acierta noventa veces genera confianza — y a la noventa y una hace algo que
*parece* igual de bueno y está mal.

## 3.1 · Una pregunta perfectamente razonable

> **¿Los pasajeros que pagan en efectivo dejan menos propina que los que
> pagan con tarjeta?**

Es el tipo de pregunta que se hace para decidir una política de medios de
pago, y tiene consecuencias en dinero. El agente la responde en veinte
segundos, con un gráfico impecable. Hagámoslo nosotros:
""")

code(r"""
propinas = taxis.groupby("payment").tip.mean()
print(propinas.round(2).to_string())

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
El cálculo está bien. El gráfico está bien. Y la conclusión obvia —**«quien
paga en efectivo nunca deja propina»**— es **falsa**.

## 3.2 · Miren el número exacto

Antes de seguir, cuenten. En datos reales, nada es exactamente cero por
casualidad.
""")

code(r"""
efectivo = taxis[taxis.payment == "cash"]
tarjeta = taxis[taxis.payment == "credit card"]

print(f"viajes en efectivo:                    {len(efectivo)}")
print(f"de esos, con propina mayor que cero:   {(efectivo.tip > 0).sum()}")
print()
print(f"viajes con tarjeta:                    {len(tarjeta)}")
print(f"de esos, con propina mayor que cero:   {(tarjeta.tip > 0).sum()}")
""")

md(r"""
## 3.3 · Un cero que no es un cero

De **1.812** viajes en efectivo, los que tienen propina son **exactamente
cero**. No tres, no cincuenta: cero. Ese número perfecto es la pista.

> **El taxímetro solo registra la propina cuando va en la tarjeta.** La
> propina en efectivo va directo al bolsillo del conductor y nunca entra al
> sistema.

La columna no dice «no dejó propina». Dice **«no tengo ese dato»**. Y si
alguien decide con eso —por ejemplo, empujar el pago con tarjeta para
«recuperar» propinas— está tomando una decisión de negocio sobre una
**ausencia de datos**.

### Esto ya lo vivieron

Es el mismo problema del módulo del agua: campos que aparecían con **cero
agua producida** durante veinticinco años. No es que no produjeran agua — es
que el regulador no la empezó a exigir hasta el año 2000.

Distinto dominio, distinto dato, distinto país, **el mismo error**. Y en los
dos casos lo que lo detectó no fue una técnica: fue alguien que conocía el
negocio y dijo *«esto no puede ser»*.

**El agente no tiene esa alarma. Ustedes sí.**

## 3.4 · 🤖 Tercer encargo: pónganle trampas al dato

Ahora usen al agente para lo contrario: para **buscar** problemas en vez de
taparlos. Este prompt vale por toda la clase — guárdenlo y córranlo con cada
dataset nuevo, **antes** de cualquier análisis.

```
CONTEXTO: DataFrame `taxis` de seaborn, con viajes de taxi.

OBJETIVO: quiero saber si estos datos tienen algún problema ANTES de sacar
conclusiones.

Hazme una revisión de calidad que incluya:
- filas totales, y cuántas tienen algún dato faltante
- para cada columna numérica: mínimo, máximo, y cuántos valores son
  exactamente cero
- si alguna combinación de categorías tiene SIEMPRE el mismo valor (eso
  suele ser un artefacto del sistema que registra, no un hecho real)

CRITERIO DE ACEPTACIÓN:
- una tabla que yo pueda leer en 30 segundos
- y una lista de las cosas que te parecen raras, aunque no estés seguro
```
""")

md(r"""
### ✍️ Peguen aquí lo que les devolvió el agente
""")

code(r"""
# pega aquí el código del agente y ejecútalo
""")

md(r"""
### ✅ Una revisión de calidad mínima
""")

code(r"""
def revisar(df):
    '''Revision de calidad rapida: lo minimo antes de analizar nada.'''
    print(f"filas: {len(df)}   columnas: {df.shape[1]}")

    faltantes = df.isna().sum()
    faltantes = faltantes[faltantes > 0]
    if len(faltantes):
        print(f"\ncolumnas con datos faltantes:\n{faltantes.to_string()}")
    else:
        print("\nsin datos faltantes")

    print("\nnuméricas — mínimo, máximo y cuántos ceros exactos:")
    num = df.select_dtypes("number")
    resumen = pd.DataFrame({
        "min": num.min(), "max": num.max(),
        "ceros": (num == 0).sum(),
        "% ceros": (100 * (num == 0).mean()).round(1)})
    print(resumen.to_string())

    # la prueba que encuentra el artefacto: ¿algun grupo es SIEMPRE cero?
    print("\nsospechas:")
    hubo = False
    for cat in df.select_dtypes(["object", "category"]).columns:
        if df[cat].nunique() > 12:
            continue
        for n in num.columns:
            g = df.groupby(cat, observed=True)[n].apply(lambda s: (s == 0).mean())
            for valor, frac in g.items():
                if frac == 1.0:
                    print(f"  · TODOS los '{valor}' tienen {n} = 0 "
                          f"({(df[cat] == valor).sum()} filas). "
                          f"¿de verdad es cero, o no se registra?")
                    hubo = True
    if not hubo:
        print("  ninguna")


revisar(taxis)
""")

md(r"""
Eso es lo que el agente debería haber hecho **antes** de responder sobre las
propinas. Fíjense en la última sección: la prueba de «¿algún grupo es siempre
cero?» es la que destapa el artefacto sin que uno sepa de antemano qué buscar.

## 3.5 · La lista de verificación

Cuatro preguntas para cualquier resultado que devuelva un agente. Toman dos
minutos y evitan la mayoría de los desastres:

1. **¿Cuántas filas entraron y cuántas salieron?** Si descartó la mitad de
   los datos sin avisar, todo lo demás sobra.
2. **¿Hay ceros, vacíos o números redondos sospechosos?** Un cero perfecto
   casi siempre es una ausencia disfrazada.
3. **¿El resultado tiene sentido físico o de negocio?** Si dice que un pozo
   produce más que el campo entero, no hace falta revisar el código.
4. **¿Qué decisión se tomaría con esto, y qué pasa si está mal?** Si la
   respuesta es «nada grave», sigan. Si no, verifiquen a mano.

> Ninguna de las cuatro exige saber programar. Las cuatro exigen saber del
> negocio — que es exactamente lo que ustedes traen.
""")

# ================================================ PARTE 4 · LA APP ==========
md(r"""
---

# Parte 4 · De la gráfica a una herramienta

Mientras el análisis viva en una celda, cada vez que alguien quiera ver otra
cosa **tiene que llamarlos**. Una app se la mandan y se acabó.

## 4.1 · Qué es Gradio

Una librería que convierte una función de Python en una página web con
controles. Uno le dice *«esta función recibe un día y devuelve un gráfico»* y
Gradio arma la interfaz sola. No hay que saber nada de páginas web.

En Colab hace además algo muy útil: genera un **link público** que se le
puede mandar a cualquiera.

> ⚠️ **Lo que hay que saber del link:** funciona **mientras el cuaderno esté
> corriendo**. Si cierran Colab o se desconecta la sesión, el link deja de
> responder. Sirve perfecto para mostrar algo en una reunión; no sirve como
> sistema permanente. Cómo se hace permanente lo vemos más adelante.

## 4.2 · La app más chica que sirve

Tres piezas: la **función** que hace el trabajo, los **controles** de entrada
y salida, y `launch(share=True)` que publica el link.
""")

code(r"""
# preparamos una columna con el nombre del día, para el control
taxis["dia_nombre"] = taxis.dia.map(dict(enumerate(DIAS)))


def ver_demanda(dia_elegido):
    '''Recibe el nombre de un dia y devuelve el grafico de ese dia.'''
    datos = taxis[taxis.dia_nombre == dia_elegido]
    fig, pico, n, _ = grafico_demanda(
        datos, titulo=f"Demanda de taxis · {dia_elegido}")
    return fig


# probamos la función sola, antes de montar la app
_ = ver_demanda("viernes")
plt.show()
""")

code(r"""
# En Colab esta celda abre el link publico. Fuera de Colab solo construye
# la interfaz, para poder verificar que el cuaderno corre entero.
try:
    import gradio as gr

    demo = gr.Interface(
        fn=ver_demanda,
        inputs=gr.Dropdown(DIAS, value="viernes", label="Día de la semana"),
        outputs=gr.Plot(label="Demanda por hora"),
        title="Demanda de taxis por día",
        description="Elija un día para ver en qué horas se concentran los viajes.")

    if EN_COLAB:
        demo.launch(share=True)      # <- acá aparece el link público
    else:
        print("interfaz construida. En Colab, esta celda abre el link.")
except ImportError:
    print("gradio no está instalado en este entorno.")
    print("En Colab ya viene listo; si hiciera falta: !pip install gradio")
""")

md(r"""
## 4.3 · 🤖 Su turno: construyan la herramienta

Ahora ustedes. La app de arriba solo deja elegir el día; falta que sirva de
verdad.

```
CONTEXTO: en este cuaderno ya tengo el DataFrame `taxis` cargado, la función
grafico_demanda(datos, titulo) que devuelve la figura, y una app de Gradio
mínima que solo deja elegir el día.

OBJETIVO: convertirla en una app que pueda mandarle por link al jefe de
operaciones.

La app tiene que dejarle elegir:
- el día de la semana
- el barrio de origen (columna pickup_borough)

y mostrarle el gráfico de demanda por hora para esa combinación, más una
línea de texto que diga cuál es la hora pico y cuántos viajes tiene.

RESTRICCIONES:
- usa gradio, con launch(share=True)
- todo en español
- si la combinación elegida no tiene datos, que lo diga con un mensaje
  claro en vez de fallar

CRITERIO DE ACEPTACIÓN:
- me tiene que dar un link que yo pueda abrir
- y no se puede caer si elijo un barrio sin viajes
```

**Antes de pegarlo:** fíjense en el último renglón del criterio de
aceptación. Ese es el que evita la llamada del jefe diciendo «se me rompió».
""")

md(r"""
### ✍️ Peguen aquí su app
""")

code(r"""
# pega aquí el código del agente y ejecútalo
""")

md(r"""
### Cómo revisarla antes de mandar el link

Prueben ustedes mismos lo que va a hacer el que la reciba:

- Elijan un barrio con pocos viajes. ¿Se cae o avisa?
- ¿El texto de la hora pico coincide con lo que muestra el gráfico?
- ¿Se entiende sin que ustedes estén al lado explicando?

> Es la misma revisión que le harían a un contratista antes de firmarle el
> acta. Si sale mal, la firma es de ustedes — el agente no va a la reunión.
""")

# ================================================ CIERRE Y PROYECTO =========
md(r"""
---

# Cierre

## Lo que vimos → lo que decidimos

| Lo que vimos | Lo que decidimos |
|---|---|
| La gráfica por defecto no comunica | Cinco decisiones de diseño, siempre las mismas |
| El agente responde según cómo se le pida | Usar los **cuatro fundamentos** en cada encargo |
| El criterio de aceptación casi nunca se escribe | Escribirlo siempre — es lo que más cambia el resultado |
| 1.812 viajes en efectivo y **cero** propinas | Desconfiar de todo cero perfecto: revisar antes de concluir |
| El agente no dio ninguna señal de alarma | **Verificar siempre**, con la lista de cuatro preguntas |
| Un análisis en una celda no lo puede usar nadie más | Convertirlo en app con link |

## Los dos resultados negativos de hoy

1. **El agente entregó un análisis correcto con una conclusión falsa**, y no
   dio ninguna señal de alarma. El código estaba bien; el dato mentía.
2. En un ensayo controlado, desarrolladores experimentados fueron **19% más
   lentos** con estas herramientas, mientras se sentían **20% más rápidos**.

Ninguno de los dos es razón para no usarlas. Los dos son razón para **medir
en vez de confiar en la sensación**.

## Si se llevan una sola cosa

> **El agente escribe el código. Ustedes responden por el resultado.**

Y eso no es una carga: es la razón por la que su criterio vale más ahora que
hace tres años. Escribir se volvió barato; saber si el resultado tiene
sentido, no.

Nadie escribió una línea de código hoy. Y todos se van con una aplicación
funcionando.

---

## Su proyecto del módulo

Empiecen a pensarlo desde hoy. En la última clase cada uno presenta:

**Una aplicación que resuelva un problema real de su trabajo**, construida
dirigiendo a un agente, con su link funcionando.

Y tres cosas más, que valen tanto como la app:

- **Los prompts que usaron** — queremos ver cómo dirigieron, no solo qué
  salió.
- **Un error que el agente cometió**, y cómo lo detectaron.
- **Qué decisión se toma** con la herramienta, y quién la toma.

### Tarea para la próxima clase

Piensen en **un** problema de su trabajo donde hoy alguien pierde tiempo con
una hoja de cálculo. Ese es su proyecto. Tráiganlo escrito en dos frases.

### Ideas, por si ayuda

| Idea | Datos que necesita |
|---|---|
| Conversor de unidades de campo | ninguno |
| Priorizador de órdenes de trabajo | lista de pendientes con fecha y criticidad |
| Mapa de demanda por turno | registros con fecha y hora |
| Seguimiento de gasto contra presupuesto | ejecución mensual |
| Calculadora de corte de agua y WOR | producción por pozo |
| Detector de lecturas raras de un sensor | serie de tiempo del instrumento |

---

*Machine Learning for Petroleum Engineers Using Python* · SLB Ecuador / UDLA ·
2026
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
n_prompts = sum(1 for c in celdas
                if c["cell_type"] == "markdown" and "🤖" in "".join(c["source"]))
print(f"escrito {NOMBRE}: {len(celdas)} celdas "
      f"({n_md} markdown, {n_code} codigo, {n_prompts} prompts guiados)")

"""
Machine Learning for Petroleum Engineers Using Python
Modulo 2 - Clase 2: genera el cuaderno de la clase.

Este script ESCRIBE el .ipynb y queda versionado para poder regenerarlo.

Los ejercicios van con espacios en blanco (___). La presentación no trae
las soluciones. Para comprobar que el cuaderno
corre, el script puede escribir además una copia con las soluciones
puestas, en la ruta que se le indique:

Uso:  python3 mknb.py                    # solo el cuaderno de la clase
      python3 mknb.py /ruta/copia.ipynb  # además, la copia resuelta
"""

import json
import sys

NOMBRE = "Modulo2_Clase2_Distribuciones_y_Relaciones.ipynb"
URL = ("https://raw.githubusercontent.com/cmosquerat/slb-diplomado/"
       "main/datos/volve_pozos_diario.csv")
URL_MOTORES = ("https://raw.githubusercontent.com/cmosquerat/slb-diplomado/"
               "main/datos/flota_turbomaquinas_nasa.csv")

clase = []      # celdas del cuaderno de la clase
resuelto = []   # mismas celdas, con los ejercicios resueltos


def _lineas(texto):
    ls = texto.strip("\n").split("\n")
    return [l + "\n" for l in ls[:-1]] + ls[-1:]


def _celda(tipo, texto):
    if tipo == "markdown":
        return {"cell_type": "markdown", "metadata": {},
                "source": _lineas(texto)}
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": _lineas(texto)}


def md(texto):
    clase.append(_celda("markdown", texto))
    resuelto.append(_celda("markdown", texto))


def code(texto):
    clase.append(_celda("code", texto))
    resuelto.append(_celda("code", texto))


def ejercicio(con_blancos, solucion):
    """Celda de ejercicio: en la clase va con ___, en la copia, resuelta."""
    assert "___" in con_blancos, "un ejercicio tiene que tener blancos"
    assert "___" not in solucion
    clase.append(_celda("code", con_blancos))
    resuelto.append(_celda("code", solucion))




# ================================================================ PORTADA ===
md(r"""
# Módulo 2 · Clase 2 — Distribuciones y relaciones entre variables

**Machine Learning for Petroleum Engineers Using Python** · SLB Ecuador / UDLA

---

Este cuaderno va a la par de la presentación: cada sección lleva el mismo
número que la lámina que dice *"En el cuaderno"*.

1. **Análisis univariado:** cómo se distribuyen los valores de una variable.
2. **Análisis bivariado:** cómo se relacionan dos variables entre sí.

**Human in the loop.** Cada cálculo de este cuaderno es una línea de Python, y
cualquier asistente de inteligencia artificial la puede escribir. Lo que no se
automatiza es el criterio: qué preguntar, qué estadístico usar y si el
resultado tiene sentido. Por eso, después de cada resultado, la pregunta
importante es *¿qué significa?*

**Cómo usar este cuaderno:** cada celda se ejecuta con ▶ o con `Shift + Enter`,
en orden. En las prácticas hay que reemplazar cada `___` antes de ejecutar.
""")

# ===================================================================== 0 ===
md(r"""
## 0 · Cargar los datos

Producción diaria de cinco pozos del campo Volve (Mar del Norte, Equinor, datos
abiertos), de 2008 a 2016. **Cada fila es un pozo en un día.**

| Columna | Qué es |
|---|---|
| `fecha` | el día |
| `pozo` | Pozo 1 a Pozo 5 |
| `dias` | días desde que ese pozo empezó a producir |
| `horas` | horas que el pozo estuvo abierto ese día |
| `petroleo` | barriles de petróleo por día |
| `gas` | miles de pies cúbicos de gas por día |
| `agua` | barriles de agua por día |

**Variable objetivo:** `petroleo`. **Variables predictoras:** `dias`, `horas`, `gas`, `agua`, `pozo`.
""")

code(f"""
# pandas: tablas. matplotlib: gráficos
import pandas as pd
import matplotlib.pyplot as plt

# Leer el archivo desde internet
url = "{URL}"
df = pd.read_csv(url)

# Las primeras filas y el tamaño de la tabla
print(df.shape)
df.head()
""")

code(r"""
# Los días con el pozo cerrado tienen 0 horas
print('Días con el pozo cerrado:', (df['horas'] == 0).sum())

# Para estudiar la producción se usan solo los días con el pozo abierto
abiertos = df[df['horas'] > 0]
print('Días con el pozo abierto:', len(abiertos))

# Un nombre corto para la columna que más vamos a usar
p = abiertos['petroleo']
""")

# ================================================================= 1.1 ===
md(r"""
---
# Parte 1 · Análisis univariado

## 1.1 · Distribución e histograma

La **distribución** de una variable describe cómo se reparten sus valores. Se
dibuja con un **histograma**: el rango se divide en intervalos del mismo ancho
(*bins*) y la altura de cada barra es la cantidad de datos en ese intervalo.
""")

code(r"""
# Histograma del petróleo con 38 intervalos
plt.hist(p, bins=38)
plt.xlabel('Petróleo (barriles por día)')
plt.ylabel('Días')
plt.show()
""")

code(r"""
# El número de intervalos cambia lo que se ve: 5 contra 60
fig, ejes = plt.subplots(1, 2, figsize=(12, 4))
ejes[0].hist(p, bins=5)
ejes[0].set_title('5 intervalos')
ejes[1].hist(p, bins=60)
ejes[1].set_title('60 intervalos')
plt.show()
""")

# ================================================================= 1.2 ===
md(r"""
## 1.2 · Tendencia central: media, mediana y moda

- **Media:** la suma de los valores dividida para cuántos son.
- **Mediana:** el valor central de los datos ordenados.
- **Moda:** el valor más frecuente.

La media es **sensible a los valores extremos**; la mediana es **robusta**.
""")

code(r"""
# Nueve empleados ganan alrededor de 1.000 y el gerente gana 50.000
sueldos = pd.Series([850, 900, 950, 1000, 1000, 1000, 1050, 1100, 1150, 50000])

print('Media:  ', sueldos.mean())
print('Mediana:', sueldos.median())
print('Moda:   ', sueldos.mode()[0])
""")

md(r"""
Nadie en la empresa gana 5.900 dólares. Como científicos de datos elegimos qué
número mostrar: con datos así, se reporta la mediana o se muestra la
distribución.
""")

code(r"""
# Media y mediana del petróleo
print('Media:  ', round(p.mean()))
print('Mediana:', round(p.median()))
""")

# ================================================================= 1.3 ===
md(r"""
## 1.3 · Forma: simetría y asimetría

- La **cola** es el extremo de la distribución donde los valores son poco frecuentes.
- **Asimetría positiva** (sesgo a la derecha): la cola va hacia los valores altos; la mayoría de los valores son bajos.
- **Asimetría negativa** (sesgo a la izquierda): la cola va hacia los valores bajos; la mayoría de los valores son altos.
- La media se desplaza hacia la cola.

El **coeficiente de asimetría** lo resume: cerca de 0, simétrica; positivo, cola
a la derecha; negativo, cola a la izquierda; más de 1 en valor absoluto, muy
asimétrica.
""")

code(r"""
# Coeficiente de asimetría del petróleo y de las horas
print('Petróleo:', round(p.skew(), 2))
print('Horas:   ', round(abiertos['horas'].skew(), 2))

# ¿Qué parte de los días produjo menos de 5.000 barriles?
print('Días con menos de 5.000 barriles:', round(100 * (p < 5000).mean(), 1), '%')
""")

# ================================================================= 1.4 ===
md(r"""
## 1.4 · Dispersión: varianza, desviación estándar y coeficiente de variación

- **Desviación estándar:** la distancia típica de los valores a la media. Dice qué tan representativa es la media.
- **Varianza:** el cuadrado de la desviación estándar.
- **Coeficiente de variación:** desviación / media. Compara la dispersión de variables con escalas distintas.

En una distribución en campana, el 68 % de los datos está a menos de 1
desviación de la media, el 95 % a menos de 2 y el 99,7 % a menos de 3.
""")

code(r"""
# Dos empresas con la misma media y distinta dispersión
empresa_a = pd.Series([960, 970, 980, 990, 1000, 1000, 1010, 1020, 1030, 1040])
empresa_b = pd.Series([200, 400, 600, 800, 1000, 1000, 1200, 1400, 1600, 1800])

print('Medias:     ', empresa_a.mean(), empresa_b.mean())
print('Desviación: ', round(empresa_a.std()), round(empresa_b.std()))
""")

code(r"""
# Varianza, desviación estándar y coeficiente de variación del petróleo
print('Varianza:  ', round(p.var()))
print('Desviación:', round(p.std()))
print('CV:        ', round(p.std() / p.mean(), 2))

# ¿Se cumple la regla del 68 %?
dentro = (p > p.mean() - p.std()) & (p < p.mean() + p.std())
print('Días a menos de 1 desviación de la media:', round(100 * dentro.mean(), 1), '%')
print('Media - 1 desviación:', round(p.mean() - p.std()))
""")

md(r"""
La desviación es mayor que la media, media − 1 desviación da una producción
negativa y la regla del 68 % no se cumple (84,5 %). La distribución no es una
campana: es muy asimétrica.
""")

# ================================================================= 1.5 ===
md(r"""
## 1.5 · El puntaje z

**z = (valor − media) / desviación**: cuántas desviaciones estándar se aleja un
valor de la media. No tiene unidades, así que permite comparar valores de
escalas distintas.
""")

code(r"""
# Carla vendió 120 mil en una región con media 100 mil y desviación 10 mil
# Luis vendió 150 mil en una región con media 140 mil y desviación 20 mil
z_carla = (120 - 100) / 10
z_luis = (150 - 140) / 20
print('z de Carla:', z_carla)
print('z de Luis: ', z_luis)
""")

md(r"""
### Un caso donde el puntaje z funciona bien: motores de turbina

Otro conjunto de datos: 100 motores de turbina, simulados por la NASA. En cada
vuelo (`ciclo`) se registra la temperatura de salida de la turbina (`T50`, en
grados Rankine). Cada motor vuela hasta que falla.
""")

code(f"""
# Leer los datos de los motores
url_motores = "{URL_MOTORES}"
motores = pd.read_csv(url_motores)

# Pasar la temperatura de grados Rankine a grados Celsius
motores['temp'] = (motores['T50'] - 491.67) * 5 / 9

# Histograma: tiene forma de campana
plt.hist(motores['temp'], bins=34)
plt.xlabel('Temperatura de salida de la turbina (°C)')
plt.ylabel('Lecturas')
plt.show()
""")

code(r"""
# Puntaje z de cada lectura
zt = (motores['temp'] - motores['temp'].mean()) / motores['temp'].std()

# ¿Se cumple la regla 68 - 95 - 99,7?
for k in [1, 2, 3]:
    print('A menos de', k, 'desviaciones:', round(100 * (zt.abs() < k).mean(), 1), '%')

# Lecturas atípicas: |z| > 3
atipicas = motores[zt.abs() > 3]
print('Lecturas atípicas:', len(atipicas))
""")

code(r"""
# ¿Son errores del sensor? Veamos cuántos vuelos le quedaban a cada motor
# 'vida' es el último vuelo de cada motor; 'restan', los vuelos que le faltaban
motores['vida'] = motores.groupby('unidad')['ciclo'].transform('max')
motores['restan'] = motores['vida'] - motores['ciclo']

# Vuelos que le quedaban al motor en cada lectura atípica
motores[zt.abs() > 3]['restan'].value_counts().sort_index()
""")

md(r"""
Las lecturas atípicas no son errores: vienen de motores a punto de fallar. Un
motor desgastado se calienta más, y el puntaje z encontró esa señal. Entender
qué significaba requirió conocer el equipo: eso es *human in the loop*.

Con el petróleo, que es asimétrico, el puntaje z funciona peor (sección 1.7).
""")

code(r"""
# El puntaje z de cada día de producción
z = (p - p.mean()) / p.std()

# Estandarizar deja la variable con media 0 y desviación 1
print('Media de z:     ', abs(round(z.mean(), 2)))
print('Desviación de z:', round(z.std(), 2))
""")

# ================================================================= 1.6 ===
md(r"""
## 1.6 · Percentiles, cuartiles y diagrama de caja

El percentil p deja por debajo al p % de los datos ordenados. Los **cuartiles**
son P25, P50 (la mediana) y P75. El **rango intercuartílico** (RIC) es P75 − P25:
el ancho de la caja, donde está el 50 % central de los datos.
""")

code(r"""
# Percentiles 10, 25, 50, 75 y 90
p.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
""")

code(r"""
# Diagrama de caja del petróleo, un pozo al lado del otro
abiertos.boxplot(column='petroleo', by='pozo')
plt.ylabel('Petróleo (barriles por día)')
plt.show()

# El rango intercuartílico de cada pozo
abiertos.groupby('pozo')['petroleo'].quantile(0.75) - abiertos.groupby('pozo')['petroleo'].quantile(0.25)
""")

# ================================================================= 1.7 ===
md(r"""
## 1.7 · Valores atípicos

- **Puntaje z:** atípico si |z| > 3. Funciona bien en distribuciones simétricas.
- **Criterio de Tukey:** atípico si está más allá de P75 + 1,5 × RIC o de P25 − 1,5 × RIC.

Un **valor atípico** es raro pero posible. Un **error** es imposible o está mal
registrado.
""")

code(r"""
# Atípicos según el puntaje z
print('Según z:    ', (z.abs() > 3).sum())

# Atípicos según Tukey
p25 = p.quantile(0.25)
p75 = p.quantile(0.75)
ric = p75 - p25
print('Según Tukey:', (p > p75 + 1.5 * ric).sum())
""")

code(r"""
# Valores imposibles: errores, no atípicos
print('Días con agua negativa:  ', (df['agua'] < 0).sum())
print('Días con más de 24 horas:', (df['horas'] > 24).sum())
""")

# ================================================================= 1.8 ===
md(r"""
## 1.8 · Poblaciones mezcladas

Cuando una variable mezcla situaciones distintas, como días abiertos y días
cerrados, se separan antes de resumirla.
""")

code(r"""
print('Todos los días:     media', round(df['petroleo'].mean()), ' mediana', round(df['petroleo'].median()))
print('Solo días abiertos: media', round(p.mean()), ' mediana', round(p.median()))
""")

md(r"""
---
### ✏️ Práctica 1 · La distribución de un pozo

Elija un pozo, reemplace cada `___` y ejecute. Después comente: ¿qué forma
tiene la distribución? ¿Qué estadístico usaría para describirlo? ¿Por qué los
dos criterios dan cantidades distintas de atípicos?
""")

ejercicio(r"""
# a) Elija su pozo: 'Pozo 1', 'Pozo 2', ... 'Pozo 5'
mi_pozo = abiertos[abiertos['pozo'] == '___']
x = mi_pozo['petroleo']

# b) Histograma con 30 intervalos
plt.hist(x, bins=___)
plt.show()

# c) Media, mediana, desviación estándar y asimetría
print('Media:     ', round(x.___()))
print('Mediana:   ', round(x.___()))
print('Desviación:', round(x.___()))
print('Asimetría: ', round(x.___(), 2))
""", r"""
# a) Elija su pozo: 'Pozo 1', 'Pozo 2', ... 'Pozo 5'
mi_pozo = abiertos[abiertos['pozo'] == 'Pozo 2']
x = mi_pozo['petroleo']

# b) Histograma con 30 intervalos
plt.hist(x, bins=30)
plt.show()

# c) Media, mediana, desviación estándar y asimetría
print('Media:     ', round(x.mean()))
print('Mediana:   ', round(x.median()))
print('Desviación:', round(x.std()))
print('Asimetría: ', round(x.skew(), 2))
""")

ejercicio(r"""
# d) Atípicos según el puntaje z
zx = (x - x.mean()) / x.std()
print('Según z:    ', (zx.abs() > ___).sum())

# e) Atípicos según Tukey
q1 = x.quantile(___)
q3 = x.quantile(___)
r = q3 - q1
print('Según Tukey:', ((x > q3 + 1.5 * r) | (x < q1 - 1.5 * r)).sum())
""", r"""
# d) Atípicos según el puntaje z
zx = (x - x.mean()) / x.std()
print('Según z:    ', (zx.abs() > 3).sum())

# e) Atípicos según Tukey
q1 = x.quantile(0.25)
q3 = x.quantile(0.75)
r = q3 - q1
print('Según Tukey:', ((x > q3 + 1.5 * r) | (x < q1 - 1.5 * r)).sum())
""")

# ================================================================= 2.1 ===
md(r"""
---
# Parte 2 · Análisis bivariado

Se estudia la relación entre variables para saber qué predictoras contienen
información sobre la variable objetivo, cuáles son redundantes y qué forma
tienen las relaciones.

## 2.1 · Diagrama de dispersión

Una relación se describe por su **dirección** (positiva o negativa), su
**forma** (lineal, curva o sin patrón) y su **fuerza**.
""")

code(r"""
# Días contra petróleo en el Pozo 2
p2 = abiertos[abiertos['pozo'] == 'Pozo 2']
plt.scatter(p2['dias'], p2['petroleo'], alpha=0.3)
plt.xlabel('Días desde que el pozo empezó')
plt.ylabel('Petróleo (barriles por día)')
plt.show()
""")

# ================================================================= 2.2 ===
md(r"""
## 2.2 · Linealidad y correlación de Pearson

Una relación es **lineal** cuando cada unidad que aumenta x produce siempre el
mismo cambio en y (una recta, y = a + b·x). En la realidad, pocas lo son.

La **correlación de Pearson** (r) va de −1 a 1 y mide qué tan cerca están los
puntos de una recta: menos de 0,3 en valor absoluto, débil; de 0,3 a 0,7,
moderada; más de 0,7, fuerte. **r = 0 no significa que no haya relación**:
significa que no hay relación lineal.
""")

code(r"""
# El gas viene disuelto en el petróleo: una relación casi lineal
plt.scatter(abiertos['petroleo'], abiertos['gas'], alpha=0.2)
plt.xlabel('Petróleo (barriles por día)')
plt.ylabel('Gas (miles de pies cúbicos)')
plt.show()

print('r =', round(abiertos['gas'].corr(abiertos['petroleo']), 3))
""")

# ================================================================= 2.3 ===
md(r"""
## 2.3 · Una tercera variable

¿Más horas abiertas dan más petróleo? Una **variable de confusión** es una
tercera variable que influye en las dos que se comparan. **Correlación no
implica causalidad.**
""")

code(r"""
# Todos los pozos juntos
print('Horas y petróleo, todos los pozos:', round(abiertos['horas'].corr(abiertos['petroleo']), 2))

# Dentro de cada pozo
for nombre in ['Pozo 1', 'Pozo 2', 'Pozo 3', 'Pozo 4', 'Pozo 5']:
    c = abiertos[abiertos['pozo'] == nombre]
    print(nombre, round(c['horas'].corr(c['petroleo']), 2))
""")

# ================================================================= 2.4 ===
md(r"""
## 2.4 · Correlación de Spearman

Una relación es **monótona** si, cuando una variable aumenta, la otra siempre
aumenta o siempre disminuye, aunque no sea en línea recta. **Spearman** calcula
la correlación sobre las posiciones de los datos ordenados.
""")

code(r"""
# La declinación del Pozo 1: curva, pero siempre hacia abajo
p1 = abiertos[abiertos['pozo'] == 'Pozo 1']
print('Pearson: ', round(p1['dias'].corr(p1['petroleo']), 2))
print('Spearman:', round(p1['dias'].corr(p1['petroleo'], method='spearman'), 2))
""")

# ================================================================= 2.5 ===
md(r"""
## 2.5 · Matriz de correlación y multicolinealidad

La **multicolinealidad** ocurre cuando dos predictoras están muy
correlacionadas entre sí: aportan la misma información. Una correlación casi
perfecta entre una predictora y la variable objetivo es una señal para revisar:
el gas se mide el mismo día que el petróleo, así que no sirve para explicarlo
de antemano (**fuga de información**).
""")

code(r"""
cols = ['horas', 'dias', 'petroleo', 'gas', 'agua']
abiertos[cols].corr().round(2)
""")

md(r"""
---
### ✏️ Práctica 2 · La relación de un pozo con el tiempo

Con el pozo de la práctica 1. Después comente: dirección, forma y fuerza de la
relación; cuál correlación es mayor en valor absoluto y por qué; qué
predictora parece aportar más información.
""")

ejercicio(r"""
# a) Dispersión de días contra petróleo
plt.scatter(mi_pozo['___'], mi_pozo['___'], alpha=0.3)
plt.show()

# b) Pearson y Spearman
print('Pearson: ', round(mi_pozo['dias'].corr(mi_pozo['petroleo']), 2))
print('Spearman:', round(mi_pozo['dias'].corr(mi_pozo['petroleo'], method='___'), 2))

# c) Matriz de correlación de su pozo
mi_pozo[cols].___().round(2)
""", r"""
# a) Dispersión de días contra petróleo
plt.scatter(mi_pozo['dias'], mi_pozo['petroleo'], alpha=0.3)
plt.show()

# b) Pearson y Spearman
print('Pearson: ', round(mi_pozo['dias'].corr(mi_pozo['petroleo']), 2))
print('Spearman:', round(mi_pozo['dias'].corr(mi_pozo['petroleo'], method='spearman'), 2))

# c) Matriz de correlación de su pozo
mi_pozo[cols].corr().round(2)
""")

# ============================================================ SÍNTESIS ===
md(r"""
---
## Síntesis · Guía de revisión de un análisis exploratorio

1. Definir la pregunta del análisis.
2. Identificar la variable objetivo y las predictoras, y cuándo se conoce cada predictora.
3. Definir qué representa una fila.
4. Revisar los tipos de datos.
5. Analizar la variable objetivo: forma, tendencia central, dispersión, atípicos.
6. Analizar cada predictora con los mismos criterios.
7. Relacionar cada predictora con la objetivo: dirección, forma, fuerza, terceras variables.
8. Registrar hallazgos: errores para limpiar y preguntas para el dueño de los datos.

| Problema | Qué se revisa además |
|---|---|
| **Regresión** | asimetría de la objetivo · forma de cada relación · multicolinealidad · fuga de información |
| **Clasificación** | proporción de cada clase · predictoras comparadas por clase |
| **Agrupamiento** | escalas distintas (estandarizar con puntaje z) · atípicos · variables redundantes |
| **Series de tiempo** | fechas faltantes · tendencia · cambios de nivel · no mezclar datos futuros con pasados |

**Human in the loop:** Python o un asistente de IA calculan; la persona elige el
estadístico, decide si un atípico es error, separa poblaciones y juzga si una
relación tiene sentido.

**Para la clase 3 (limpieza de datos):** agua negativa, días con más de 24
horas, días con el pozo cerrado y fechas faltantes.
""")

# ============================================================== ESCRIBIR ===
def _cuaderno(celdas):
    return {"cells": celdas,
            "metadata": {"kernelspec": {"display_name": "Python 3",
                                        "language": "python",
                                        "name": "python3"},
                         "language_info": {"name": "python"},
                         "colab": {"provenance": []}},
            "nbformat": 4, "nbformat_minor": 5}


def _escribir(celdas, ruta):
    for i, c in enumerate(celdas):
        c["id"] = f"c{i:03d}"
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(_cuaderno(celdas), f, ensure_ascii=False, indent=1)
        f.write("\n")


_escribir(clase, NOMBRE)
print(f"escrito {NOMBRE}: {len(clase)} celdas")
if len(sys.argv) > 1:
    _escribir(resuelto, sys.argv[1])
    print(f"escrita la copia resuelta en {sys.argv[1]}")

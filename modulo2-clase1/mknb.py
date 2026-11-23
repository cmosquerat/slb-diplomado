"""
Machine Learning for Petroleum Engineers Using Python
Modulo 2 - Clase 1: genera el cuaderno de la clase.

Este script ESCRIBE el .ipynb y queda versionado para poder regenerarlo.

Los ejercicios van con espacios en blanco (___). Sus soluciones están en
las láminas que siguen a cada ejercicio. Para comprobar que el cuaderno
corre, el script puede escribir además una copia con las soluciones
puestas, en la ruta que se le indique:

Uso:  python3 mknb.py                    # solo el cuaderno de la clase
      python3 mknb.py /ruta/copia.ipynb  # además, la copia resuelta
"""

import json
import sys

NOMBRE = "Modulo2_Clase1_Explorar_Datos.ipynb"
URL = ("https://raw.githubusercontent.com/cmosquerat/slb-diplomado/"
       "main/datos/insurance.csv")

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
# Módulo 2 · Clase 1 — Exploración de datos

**Machine Learning for Petroleum Engineers Using Python** · SLB Ecuador / UDLA

---

### Qué es explorar datos

Explorar datos es **mirarlos con orden, antes de usarlos**, para saber qué
hay, si se puede confiar en ellos y qué dicen. En inglés se llama
*Exploratory Data Analysis* (EDA).

John Tukey, el estadístico que le puso nombre en 1977, decía que es trabajo
de detective: buscar pistas con números, conteos y gráficos.

**Para qué sirve:**

1. Entender qué hay: qué mide cada columna, en qué unidades, de quién son los datos.
2. Encontrar problemas: datos vacíos, repetidos, errores. (Se tratan en la clase 2.)
3. Hacer mejores preguntas: algo raro aparece y uno pregunta por qué.
4. Elegir cómo seguir: qué número resume mejor, qué variables usar, qué modelo probar.

**Cuándo se hace:** al principio de todo proyecto con datos, antes de limpiar
y mucho antes de hacer un modelo. Y otra vez cada vez que llegan datos nuevos.

### El dataset de hoy

Costos médicos de **1.338 personas** con seguro de salud en Estados Unidos.
Lo usamos porque se entiende sin explicación y tiene de todo para practicar.
Una pregunta de fondo nos acompaña: **¿cuánto cuesta una persona asegurada?**

### Cómo usar este cuaderno

- Cada celda gris es código. Para correrla: clic en la celda y **Shift + Enter**.
- Córralas **en orden**, de arriba hacia abajo. Si algo falla, vuelva a correr
  desde la primera celda.
- Las líneas que empiezan con `#` son **comentarios**: Python no las ejecuta,
  están para explicar.
- Los ejercicios están marcados con ✏️. Donde vea `___` (tres guiones bajos),
  tiene que escribir usted.

### El plan

1. Los datos
2. Tipos de datos
3. Filas y columnas en pandas
4. Revisar los datos antes de usarlos
5. La forma de los datos
6. Resumir con números
""")

# ========================================================= 0 · PREPARACIÓN ===
md(r"""
## 0 · Preparación

### Qué es una librería

Python, solo, sabe hacer cosas básicas: sumar, guardar valores, repetir pasos.
Una **librería** es un paquete de herramientas que otras personas ya
programaron y probaron, para que uno no empiece de cero. Es como una caja de
herramientas: uno no fabrica el martillo, lo saca de la caja y lo usa.

Hoy usamos dos:

- **pandas**: para trabajar con tablas, como un Excel dentro de Python.
- **matplotlib**: para hacer gráficos.

Para usar una librería hay dos pasos:

1. **Instalarla**: se hace una sola vez. En Google Colab ya vienen instaladas.
2. **Importarla**: se hace cada vez que se abre el cuaderno. Es sacar la
   herramienta de la caja.

`import pandas as pd` la importa y le pone un apodo corto, `pd`. Desde ahí se
escribe `pd` en lugar de `pandas`. Todo el mundo usa ese mismo apodo.
""")

code(r"""
# Cargamos pandas (tablas) y le decimos pd
import pandas as pd

# Cargamos la parte de matplotlib que hace gráficos y le decimos plt
import matplotlib.pyplot as plt
""")

md(r"""
Ahora leemos el archivo. Es un **CSV** (*comma separated values*): un archivo
de texto donde cada línea es una fila y las columnas se separan con comas.
""")

code(rf"""
# La dirección de internet donde está el archivo
url = "{URL}"

# read_csv lee el archivo y lo guarda como una tabla.
# A la tabla le ponemos el nombre df (de "data frame").
df = pd.read_csv(url)

# Un mensaje para saber que todo salió bien
print("Datos cargados")
""")

# ===================================================== 1 · LA PREGUNTA ======
md(r"""
## 1 · Los datos

Antes de tocar el código hay que saber **qué significa cada columna**. Esto
se llama el *diccionario de datos*.

| Columna | Qué es | Unidad o valores |
|---|---|---|
| `age` | edad del cliente | años |
| `sex` | sexo | `male` (hombre), `female` (mujer) |
| `bmi` | índice de masa corporal (IMC) | kg/m² |
| `children` | hijos cubiertos por el seguro | número |
| `smoker` | si fuma | `yes`, `no` |
| `region` | región de Estados Unidos donde vive | `northeast`, `northwest`, `southeast`, `southwest` |
| `charges` | lo que costó en gastos médicos ese año | dólares (USD) |

**Qué es el IMC.** Es el peso dividido para la estatura al cuadrado. Una
persona de 80 kg y 1,70 m tiene un IMC de 80 / (1,70 × 1,70) = 27,7.
La Organización Mundial de la Salud usa estas categorías:
menos de 18,5 bajo peso; de 18,5 a 25 normal; de 25 a 30 sobrepeso;
30 o más obesidad.

**De dónde salen los datos.** Es un dataset clásico de enseñanza, del libro
*Machine Learning with R* de Brett Lantz. Son datos **simulados** a partir
de estadísticas del censo de Estados Unidos. Se parecen a la realidad, pero
no son personas reales. Eso lo vamos a notar en la exploración.
""")

# ===================================================== 2 · TIPOS ============
md(r"""
## 2 · Tipos de datos

El tipo de un dato decide **qué operaciones tienen sentido**. La edad
promedio de los clientes es 39 años, y eso significa algo. La región
promedio no significa nada.

Hay dos grandes grupos:

- **Numéricos**: son cantidades. Se miden o se cuentan.
- **Categóricos**: son etiquetas. Dicen a qué grupo pertenece algo.

Una pregunta sencilla ayuda a separarlos: **¿tiene sentido sumarlo o
restarlo?** Si la respuesta es sí, es numérico.

| Tipo | Qué es | En este dataset | Otros ejemplos |
|---|---|---|---|
| Numérico continuo | se mide, puede tener decimales | `bmi`, `charges` | estatura, peso, precio, temperatura |
| Numérico discreto | se cuenta, solo enteros | `children` | personas en una familia, fallas de una bomba en el año |
| Categórico nominal | grupos sin orden | `region` | ciudad, marca de carro |
| Categórico binario | solo dos valores | `sex`, `smoker` | ¿tiene carro?, ¿el pozo está produciendo? |
| Categórico ordinal | grupos con un orden | categorías de IMC | nivel de estudios: primaria, secundaria, universidad |

**Un caso intermedio: la edad.** La edad real es continua (uno tiene 39 años
y unos meses), pero aquí está registrada en años cumplidos. En esta tabla
se comporta como discreta.

**La trampa:** no todo número es una cantidad. Una cédula, un código postal
o un número de factura son números, pero son etiquetas. Si la región viniera
como 1, 2, 3 y 4, el promedio daría 2,5, y eso no significa nada.

### Cómo ve pandas los tipos

pandas mira el contenido de cada columna y le asigna un tipo:

- `int64`: número entero
- `float64`: número con decimales
- `object` (o `str` en versiones nuevas de pandas): texto

pandas no sabe si un número es una cantidad o un código. Eso lo decide
usted.
""")

code(r"""
# dtypes muestra el tipo que pandas le asignó a cada columna
df.dtypes
""")

md(r"""
**Un truco con las binarias.** Si convertimos "sí fuma" en 1 y "no fuma"
en 0, el promedio de esa columna es la **proporción** de fumadores.
""")

code(r"""
# df['smoker'] == 'yes' da True (verdadero) o False (falso) para cada cliente.
# Python cuenta True como 1 y False como 0.
# El promedio de unos y ceros es la proporción de unos.
proporcion = (df['smoker'] == 'yes').mean()

# round(numero, 3) redondea a 3 decimales
print('Proporción de fumadores:', round(proporcion, 3))
""")

md(r"""
### ✏️ Ejercicio 1 · Clasificar variables (5 minutos)

Doble clic en esta celda para editarla. Escriba el tipo de cada variable:
**continuo, discreto, nominal, binario u ordinal**. Si es un número que en
realidad es una etiqueta, dígalo.

**a) Las columnas de este dataset**

| Columna | Tipo |
|---|---|
| `age` | |
| `sex` | |
| `bmi` | |
| `children` | |
| `smoker` | |
| `region` | |
| `charges` | |

**b) Datos de una persona**

| Variable | Tipo |
|---|---|
| Número de cédula | |
| Estatura (m) | |
| Fecha de nacimiento | |
| Ciudad donde vive | |
| Número de hermanos | |
| Precio de la gasolina que paga (USD por galón) | |
| Satisfacción con su trabajo (mala, regular, buena) | |
| ¿Tiene carro? (sí, no) | |
""")

# ========================================== 3 · FILAS Y COLUMNAS EN PANDAS ==
md(r"""
## 3 · Filas y columnas en pandas

Una tabla tiene:

- **Filas**: cada una es una **observación**. Aquí, un cliente.
- **Columnas**: cada una es una **variable**, algo que se midió para todos.
- **Celdas**: el valor de una variable para una observación.

**Una fila es una observación**: una cosa que se midió, con todos sus datos.
Qué es esa "cosa" depende de la tabla:

| Tabla | Cada fila es |
|---|---|
| una encuesta | una persona |
| las ventas de una tienda | una venta |
| la producción de un campo | un pozo en un día |
| esta tabla | una persona y lo que costó en un año |

Antes de contar o promediar, pregunte **¿qué es una fila?** Contar filas es
contar esas cosas.

En pandas:

- La tabla completa se llama **DataFrame**.
- Una sola columna se llama **Series**.
- Cada fila tiene un número que la identifica, el **índice**. Empieza en 0.
""")

md(r"""
### El primer vistazo
""")

code(r"""
# head() muestra las primeras 5 filas
df.head()
""")

code(r"""
# tail() muestra las últimas 5 filas
df.tail()
""")

code(r"""
# shape dice el tamaño: (número de filas, número de columnas)
df.shape
""")

code(r"""
# info() es un resumen de la tabla:
# - cuántas filas hay (RangeIndex)
# - cada columna, cuántos valores NO vacíos tiene (Non-Null Count)
# - y de qué tipo es (Dtype)
df.info()
""")

md(r"""
### Seleccionar columnas

Se escribe el nombre de la tabla, corchetes y el nombre de la columna entre
comillas. El nombre tiene que ser exacto: `Charges` con mayúscula no existe.
""")

code(r"""
# Una sola columna: el resultado es una Series
df['charges']
""")

code(r"""
# Varias columnas: doble corchete.
# El corchete de afuera selecciona; el de adentro es la lista de nombres.
df[['age', 'charges']]
""")

md(r"""
### Seleccionar filas

Hay dos formas, y conviene no confundirlas:

| | `iloc` | `loc` |
|---|---|---|
| Busca por | **posición** (primera, segunda...) | **etiqueta** del índice |
| `[0:3]` incluye | filas 0, 1 y 2 (el 3 **no**) | filas 0, 1, 2 **y 3** |

En esta tabla la posición y la etiqueta coinciden, porque el índice va de 0
en adelante. Cuando filtremos o reordenemos, ya no van a coincidir.
""")

code(r"""
# iloc[0]: la fila que está en la primera posición
df.iloc[0]
""")

code(r"""
# iloc[0:3]: de la posición 0 a la 2. El 3 no se incluye.
df.iloc[0:3]
""")

code(r"""
# loc[0:3]: las filas con etiqueta 0, 1, 2 y 3. Con loc, el 3 sí se incluye.
df.loc[0:3]
""")

code(r"""
# loc también puede pedir una fila y una columna a la vez:
# el costo del cliente con etiqueta 543
print(df.loc[543, 'charges'])
""")

md(r"""
### Filtrar filas con una condición

Filtrar es quedarse solo con las filas que cumplen algo. Funciona en dos
pasos.

**Paso 1.** La condición produce una columna de `True` (verdadero) y `False`
(falso), una por cliente.
""")

code(r"""
# Para cada cliente: ¿fuma? Mostramos solo los primeros 5 resultados.
(df['smoker'] == 'yes').head()
""")

md(r"""
**Paso 2.** Se pone esa condición dentro de `df[ ]`, y pandas se queda con
las filas donde dice `True`.
""")

code(r"""
# Guardamos solo los clientes que fuman
fumadores = df[df['smoker'] == 'yes']

# len() cuenta cuántas filas quedaron
len(fumadores)
""")

md(r"""
Los símbolos para comparar:

| Símbolo | Significa |
|---|---|
| `==` | igual a (son dos signos igual) |
| `!=` | distinto de |
| `>` , `<` | mayor que, menor que |
| `>=` , `<=` | mayor o igual, menor o igual |

Para **combinar** condiciones: `&` significa "y", `|` significa "o".
Cada condición va **entre paréntesis**.
""")

code(r"""
# Clientes que fuman Y tienen más de 50 años
fuman_y_mayores = df[(df['smoker'] == 'yes') & (df['age'] > 50)]

# Cuántos son
len(fuman_y_mayores)
""")

md(r"""
### Ordenar y contar
""")

code(r"""
# sort_values ordena por una columna.
# ascending=False quiere decir de mayor a menor.
# head() muestra los 5 primeros: los 5 clientes más caros.
df.sort_values('charges', ascending=False).head()
""")

code(r"""
# value_counts cuenta cuántas veces aparece cada valor de una columna
df['region'].value_counts()
""")

md(r"""
### ✏️ Ejercicio 2 · Moverse por la tabla (8 minutos)

Reemplace cada `___` y corra la celda.
""")

ejercicio(r"""
# a) ¿Cuántos clientes tienen más de 3 hijos?
muchos_hijos = df[df['children'] > ___]
len(muchos_hijos)
""", r"""
# a) ¿Cuántos clientes tienen más de 3 hijos?
muchos_hijos = df[df['children'] > 3]
len(muchos_hijos)
""")

ejercicio(r"""
# b) Muestre solo la edad y el costo de los primeros 10 clientes
df[['___', '___']].head(10)
""", r"""
# b) Muestre solo la edad y el costo de los primeros 10 clientes
df[['age', 'charges']].head(10)
""")

ejercicio(r"""
# c) ¿Quién es el cliente más caro? Ordene de mayor a menor y mire el primero
df.sort_values('___', ascending=False).head(1)
""", r"""
# c) ¿Quién es el cliente más caro? Ordene de mayor a menor y mire el primero
df.sort_values('charges', ascending=False).head(1)
""")

ejercicio(r"""
# d) ¿Cuántos fumadores viven en la región southeast?
fuman_southeast = df[(df['smoker'] == '___') & (df['region'] == '___')]
len(fuman_southeast)
""", r"""
# d) ¿Cuántos fumadores viven en la región southeast?
fuman_southeast = df[(df['smoker'] == 'yes') & (df['region'] == 'southeast')]
len(fuman_southeast)
""")

# ================================================== 4 · REVISAR LOS DATOS ===
md(r"""
## 4 · Revisar los datos antes de usarlos

Antes de sacar cualquier promedio, se revisa la tabla con una lista fija.
No es graficar todo: es hacerle a la tabla preguntas con un propósito.

| Pregunta | Cómo se responde |
|---|---|
| ¿Qué tamaño tiene? | `df.shape` |
| ¿De qué tipo es cada columna? | `df.dtypes`, `df.info()` |
| ¿Hay datos vacíos? | `df.isna().sum()` |
| ¿Los mínimos y máximos tienen sentido? | `df.describe()` |
| ¿Las categorías son las que esperamos? | `df['columna'].value_counts()` |
| ¿Hay filas repetidas? | `df.duplicated().sum()` |

### ¿Hay datos vacíos?

Un dato vacío es una celda sin valor. pandas lo muestra como `NaN`
(*Not a Number*).
""")

code(r"""
# isna() marca con True cada celda vacía.
# sum() suma por columna: cuenta cuántas celdas vacías hay en cada una.
df.isna().sum()
""")

md(r"""
Ninguna columna tiene vacíos. En datos reales eso casi nunca pasa. Aquí
pasa porque los datos son simulados.

### ¿Los mínimos y máximos tienen sentido?

`describe()` resume las columnas numéricas:

- `count`: cuántos valores no vacíos
- `mean`: la media (el promedio)
- `std`: la desviación estándar (cuánto se separan los valores; la vemos más adelante)
- `min` y `max`: el valor más chico y el más grande
- `25%`, `50%`, `75%`: percentiles (también los vemos más adelante)
""")

code(r"""
# Resumen de las columnas numéricas
df.describe()
""")

md(r"""
**Leerlo con sentido:**

- **Edad de 18 a 64 años.** No hay menores de edad ni mayores de 64. En
  Estados Unidos, a los 65 años la mayoría de personas pasa al seguro del
  gobierno (Medicare). Estos datos describen adultos en edad de trabajar.
- **IMC de 16 a 53.** Valores extremos, pero posibles.
- **Hijos de 0 a 5.** Normal.
- **Costo de 1.122 a 63.770 dólares.** Una diferencia enorme entre el más
  barato y el más caro. Hay que mirarla con cuidado.

Ojo: Python usa **punto** para los decimales. Donde dice `30.66`, nosotros
leemos 30,66.

### ¿Las categorías son las esperadas?
""")

code(r"""
# Cuántos clientes hay de cada sexo
df['sex'].value_counts()
""")

code(r"""
# Cuántos fuman y cuántos no
df['smoker'].value_counts()
""")

code(r"""
# Cuántos hay en cada región
df['region'].value_counts()
""")

md(r"""
Las categorías son las del diccionario y no hay ninguna mal escrita. Pero
fíjese en las regiones: 364, 325, 325 y 324. Son casi iguales. En datos
reales eso sería raro. Es otra pista de que el dataset es simulado.

**Explorar también sirve para saber de dónde vienen los datos.**

### ¿Hay filas repetidas?
""")

code(r"""
# duplicated() marca con True cada fila que es copia exacta de otra anterior.
# sum() cuenta cuántas hay.
print('Filas repetidas:', df.duplicated().sum())
""")

code(r"""
# keep=False marca las dos copias, para verlas juntas
df[df.duplicated(keep=False)]
""")

md(r"""
Dos hombres de 19 años, con el mismo IMC, de la misma región, sin hijos,
que no fuman y que costaron lo mismo **hasta el centavo**.

¿Es la misma persona registrada dos veces, o dos personas distintas? Por
ahora no la borramos: **la anotamos**. Qué hacer con ella es tema de la
próxima clase.

### ¿A quién representan estos datos?

Adultos de 18 a 64 años, de cuatro regiones de Estados Unidos, de los
cuales el 20 % fuma. Lo que concluyamos vale para esa población, no para
cualquier persona.
""")

# ================================================ 5 · LA FORMA DE LOS DATOS =
md(r"""
## 5 · La forma de los datos

La **distribución** de una variable es cómo se reparten sus valores: cuáles
son comunes, cuáles son raros y dónde se amontonan.

Un solo número, como el promedio, esconde la forma. Por eso primero se mira
la forma y después se elige qué número usar.

### El histograma

Es el gráfico que muestra la forma de una variable numérica. Se arma así:

1. Se divide el rango de valores en **intervalos** del mismo ancho (en
   inglés, *bins*). Por ejemplo, edades de 15 a 25, de 25 a 35, etc.
2. Se cuenta cuántos valores caen en cada intervalo.
3. Se dibuja una barra por intervalo. La altura es ese conteo.

El eje horizontal muestra los valores. El eje vertical muestra cuántos
clientes hay.
""")

code(r"""
# plt.hist dibuja un histograma.
# bins=30 quiere decir: dividir el rango en 30 intervalos.
plt.hist(df['charges'], bins=30)

# Los nombres de los ejes y el título
plt.xlabel('Costo médico del año (USD)')
plt.ylabel('Número de clientes')
plt.title('Distribución del costo')

# Mostrar el gráfico
plt.show()
""")

md(r"""
### El número de barras cambia lo que se ve

Con pocas barras se pierde el detalle. Con demasiadas aparece ruido. No hay
un número correcto: se prueban varios.
""")

code(r"""
# Con 5 barras
plt.hist(df['charges'], bins=5)
plt.title('5 barras')
plt.show()

# Con 60 barras
plt.hist(df['charges'], bins=60)
plt.title('60 barras')
plt.show()
""")

md(r"""
### Las formas que hay que saber reconocer

| Forma | Qué se ve | Qué significa | Qué implica |
|---|---|---|---|
| **Simétrica** | una campana | casi todo cerca del centro; lo muy alto y lo muy bajo son igual de raros | el promedio describe bien lo típico |
| **Sesgada a la derecha** | un montón de valores bajos y una **cola larga hacia los altos** | la mayoría es chica, unos pocos son muy grandes | el promedio queda por encima de lo típico; los de la cola no son errores por ser grandes |
| **Sesgada a la izquierda** | un montón de valores altos y una cola hacia los bajos | la mayoría es alta, unos pocos son muy bajos | el promedio queda por debajo de lo típico; los bajos merecen revisión |
| **Uniforme** | todas las barras parecidas | todos los valores son igual de comunes | no hay un valor típico |
| **Bimodal** | dos jorobas | dos grupos mezclados en la misma columna | el promedio puede caer donde no hay nadie; hay que separar los grupos |

Ejemplos: estaturas (simétrica), sueldos y costos médicos (sesgada a la
derecha), notas de un examen fácil (sesgada a la izquierda), un dado
(uniforme), estatura de niños y adultos juntos (bimodal).

**El sesgo se nombra por el lado de la cola, no por el lado del montón.**
Una distribución sesgada a la derecha tiene el montón a la izquierda.

pandas mide el sesgo con `skew()`:

- cerca de 0: simétrica
- positivo: cola a la derecha
- negativo: cola a la izquierda
""")

code(r"""
# El sesgo del costo, redondeado a 2 decimales
print('Sesgo del costo:', round(df['charges'].skew(), 2))
""")

md(r"""
El costo está **sesgado a la derecha**: muchos clientes baratos y una cola
larga de clientes caros.

### Variables discretas y categóricas: gráfico de barras

Para una variable que solo toma unos pocos valores (como los hijos) o para
una categórica, no se usa el histograma. Se cuentan los valores y se dibuja
una barra por cada uno.
""")

code(r"""
# Contamos cuántos clientes tienen 0, 1, 2... hijos.
# sort_index() los pone en orden: 0, 1, 2, 3, 4, 5
conteo_hijos = df['children'].value_counts().sort_index()

# kind='bar' dibuja barras
conteo_hijos.plot(kind='bar')
plt.xlabel('Número de hijos')
plt.ylabel('Número de clientes')
plt.show()
""")

code(r"""
# Lo mismo para una categórica: la región
df['region'].value_counts().plot(kind='bar')
plt.ylabel('Número de clientes')
plt.show()
""")

md(r"""
### Una variable ordinal: el IMC en categorías

Convertimos el IMC (un número) en sus cuatro categorías. El resultado es
una variable **ordinal**: las categorías tienen orden.

`pd.cut` corta una columna numérica en tramos. Le damos los bordes de cada
tramo y el nombre de cada uno. `right=False` quiere decir que cada tramo
incluye su borde izquierdo: 25 cae en "Sobrepeso", no en "Normal".
""")

code(r"""
# Los bordes de los tramos: 0, 18.5, 25, 30 y "infinito" (no hay límite)
bordes = [0, 18.5, 25, 30, float('inf')]

# El nombre de cada tramo, en orden
nombres = ['Bajo peso', 'Normal', 'Sobrepeso', 'Obesidad']

# Creamos una columna nueva con la categoría de cada cliente
df['imc_categoria'] = pd.cut(df['bmi'], bins=bordes, labels=nombres, right=False)

# Contamos. sort_index() respeta el orden de las categorías.
df['imc_categoria'].value_counts().sort_index()
""")

md(r"""
En un gráfico de una variable ordinal, las barras van **en su orden**, no
de la más alta a la más baja.

### ✏️ Ejercicio 3 · Describir la forma (7 minutos)

Reemplace cada `___`, corra las celdas y responda abajo.
""")

ejercicio(r"""
# a) Histograma de la edad, con una barra por cada año.
#    Las edades van de 18 a 64: son 47 años distintos.
plt.hist(df['___'], bins=___)
plt.xlabel('Edad (años)')
plt.ylabel('Número de clientes')
plt.show()
""", r"""
# a) Histograma de la edad, con una barra por cada año.
#    Las edades van de 18 a 64: son 47 años distintos.
plt.hist(df['age'], bins=47)
plt.xlabel('Edad (años)')
plt.ylabel('Número de clientes')
plt.show()
""")

ejercicio(r"""
# b) Histograma del IMC, con 25 barras
plt.hist(df['___'], bins=25)
plt.xlabel('IMC (kg/m²)')
plt.ylabel('Número de clientes')
plt.show()
""", r"""
# b) Histograma del IMC, con 25 barras
plt.hist(df['bmi'], bins=25)
plt.xlabel('IMC (kg/m²)')
plt.ylabel('Número de clientes')
plt.show()
""")

ejercicio(r"""
# c) El sesgo de la edad y el del IMC
print('Sesgo de la edad:', round(df['age'].___(), 2))
print('Sesgo del IMC:', round(df['bmi'].___(), 2))
""", r"""
# c) El sesgo de la edad y el del IMC
print('Sesgo de la edad:', round(df['age'].skew(), 2))
print('Sesgo del IMC:', round(df['bmi'].skew(), 2))
""")

md(r"""
**Responda (doble clic para editar):**

1. ¿Qué forma tiene la edad? ¿Y el IMC?
2. En la edad hay dos barras que sobresalen. ¿Qué edades son? ¿Se le ocurre
   por qué?
3. ¿Los valores de `skew()` confirman lo que se ve en los gráficos?
""")

# ================================================= 6 · RESUMIR CON NÚMEROS ==
md(r"""
## 6 · Resumir con números

Ya vimos la forma. Ahora sí, los números que la resumen.

### El centro: media, mediana y moda

- **Media** (el promedio): se suman todos los valores y se divide para
  cuántos son.
- **Mediana**: se ordenan los valores de menor a mayor y se toma el del
  medio. La mitad queda por debajo y la otra mitad por encima.
- **Moda**: el valor que más se repite.

Un ejemplo con cinco clientes que costaron 2, 3, 3, 4 y 50 mil dólares:

- Media: (2 + 3 + 3 + 4 + 50) / 5 = 62 / 5 = **12,4**
- Mediana: ordenados, el del medio es **3**
- Moda: el que más se repite es **3**

Un solo cliente caro jala la media hasta 12,4, aunque cuatro de los cinco
cuestan 4 o menos. Su costo se reparte entre los cinco: si sube 5 mil, la
media sube 1 mil.

La mediana no se mueve mientras él siga siendo el más caro, porque el del
medio sigue siendo el mismo. Por eso se dice que la mediana es **robusta**:
los valores extremos casi no la afectan.

Pruebe en la celda de abajo: cambie el 50 por 20, o por 100, y vuelva a
correrla.
""")

code(r"""
# Los cinco clientes del ejemplo, como una Series de pandas
ejemplo = pd.Series([2, 3, 3, 4, 50])

print('Media:', ejemplo.mean())
print('Mediana:', ejemplo.median())
# mode() puede devolver varios valores si empatan; [0] toma el primero
print('Moda:', ejemplo.mode()[0])
""")

code(r"""
# Ahora con el costo real
# round(numero, 2) redondea a 2 decimales
print('Media del costo:', round(df['charges'].mean(), 2))
print('Mediana del costo:', round(df['charges'].median(), 2))
""")

md(r"""
La media (13.270) es mayor que la mediana (9.382). ¿Por qué?

- La **media** es el punto de equilibrio: si el histograma fuera una tabla
  apoyada en un punto, la media es donde queda balanceada. Lo que está lejos
  pesa más.
- La **mediana** solo cuenta: deja la mitad de las personas a cada lado, sin
  importar qué tan lejos estén.

En una distribución simétrica los dos lados pesan parecido y las dos
coinciden. En una sesgada a la derecha, la cola está lejos y corre la media
hacia ella.

**¿Cuál se usa? Depende de la pregunta.**

- Si la aseguradora quiere saber **cuánto va a pagar en total**, necesita la
  media, porque media × número de clientes = total.
- Si quiere saber **cuánto cuesta un cliente típico**, la respuesta es la
  mediana.

¿Y la moda? Sirve en variables discretas o categóricas. En el costo no
sirve: casi ningún valor se repite.
""")

code(r"""
# El total que pagó la aseguradora
print('Total:', round(df['charges'].sum(), 2))

# Es lo mismo que la media por el número de clientes
print('Media x clientes:', round(df['charges'].mean() * len(df), 2))
""")

code(r"""
# La moda sí sirve para los hijos: lo más común es no tener hijos
print('Moda de los hijos:', df['children'].mode()[0])
""")

code(r"""
# nunique() cuenta cuántos valores distintos hay.
# En el costo, casi todos son distintos: por eso la moda no sirve ahí.
print('Valores distintos del costo:', df['charges'].nunique(), 'de', len(df))
""")

md(r"""
### La dispersión: qué tan separados están los valores

Dos grupos pueden tener la misma media y ser muy distintos: en uno todos
cuestan parecido; en el otro hay de todo.

- **Rango**: máximo menos mínimo. Es fácil, pero depende de solo dos
  valores, los más extremos.
- **Desviación estándar**: la distancia típica de los valores a la media.
  Si es chica, los valores están cerca de la media. Si es grande, están
  repartidos. Está en las mismas unidades que los datos (aquí, dólares).

**Cómo se calcula**, con cuatro valores: 1, 3, 7 y 9 (media 5).

1. La distancia de cada valor a la media: −4, −2, 2, 4.
2. Al cuadrado, para que las negativas no cancelen a las positivas: 16, 4, 4, 16.
3. Se suman (40) y se divide para el número de valores menos uno: 40 / 3 = 13,33.
4. La raíz, para volver a las unidades de los datos: √13,33 = **3,65**.

Con 4, 5, 5 y 6 (también media 5) da 0,82: los valores están mucho más cerca.
""")

code(r"""
# Dos grupos con la misma media (5) y distinta separación
grupo_a = pd.Series([4, 5, 5, 6])
grupo_b = pd.Series([1, 3, 7, 9])

# std() hace toda la cuenta de la desviación estándar
print('Desviación del grupo A:', round(grupo_a.std(), 2))
print('Desviación del grupo B:', round(grupo_b.std(), 2))
""")

code(r"""
# El rango: el más caro menos el más barato
print('Rango:', round(df['charges'].max() - df['charges'].min(), 2))

# La desviación estándar
print('Desviación estándar:', round(df['charges'].std(), 2))
""")

md(r"""
La desviación estándar (12.110) es casi tan grande como la media (13.270).
Eso dice que los costos varían muchísimo de un cliente a otro.

**Cómo se lee.** En una campana, entre media − 1 desviación y media + 1
desviación queda cerca de 2 de cada 3 valores. En el IMC queda el 67 %.

**Cuándo no sirve.** En el costo, que está sesgado, en esa misma franja queda
el 84,5 %, y media − 2 desviaciones da −10.950 dólares: un costo que no
existe. En datos sesgados es mejor usar percentiles.

### Percentiles

Imagine a los 1.338 clientes en fila, ordenados del más barato al más caro.

- El **percentil 10 (P10)** es el valor que deja al 10 % de los clientes por
  debajo.
- El **P50** deja a la mitad por debajo: es la **mediana**.
- El **P90** deja al 90 % por debajo.

En pandas se piden con `quantile()`, escribiendo el porcentaje como decimal:
0.1 para el P10, 0.9 para el P90.

> **Ojo con la convención de reservas.** En reservas de petróleo, el "P90" es
> la estimación baja: el volumen que hay con 90 % de probabilidad de que haya
> al menos eso. En estadística, el P90 es el valor alto, el que deja al 90 %
> por debajo. El P90 de reservas corresponde al P10 estadístico.
""")

code(r"""
# Varios percentiles a la vez: P10, P25, P50, P75 y P90
df['charges'].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
""")

md(r"""
Lectura: **9 de cada 10 clientes cuestan menos de 34.832 dólares**. Y la
mitad cuesta menos de 9.382.

### Cuartiles y rango intercuartílico

Los percentiles 25, 50 y 75 se llaman **cuartiles**, porque parten la fila
en cuatro partes iguales.

El **rango intercuartílico** (en inglés, *IQR*) es P75 − P25: el ancho del
50 % de clientes del centro. Igual que la mediana, no se deja mover por los
extremos.
""")

code(r"""
# Los cuartiles 1 y 3
p25 = df['charges'].quantile(0.25)
p75 = df['charges'].quantile(0.75)

# El rango intercuartílico
iqr = p75 - p25
print('P25:', round(p25, 2))
print('P75:', round(p75, 2))
print('IQR:', round(iqr, 2))
""")

md(r"""
### El boxplot (diagrama de caja)

Resume la distribución en un solo dibujo:

- La **caja** va del P25 al P75. Adentro está el 50 % central.
- La **línea dentro de la caja** es la mediana.
- Los **bigotes** son las líneas que salen de la caja. Llegan hasta el último
  dato que está a menos de 1,5 × IQR del borde de la caja.
- Los **puntos sueltos** son los datos que quedan más allá de los bigotes.
  La regla los llama "atípicos".

Se puede dibujar vertical (como aquí) u horizontal.
""")

code(r"""
# plt.boxplot dibuja el diagrama de caja
plt.boxplot(df['charges'])
plt.ylabel('Costo médico del año (USD)')
plt.title('Boxplot del costo')
plt.show()
""")

md(r"""
### Atípico no quiere decir error

La regla del boxplot marca como atípico todo lo que pase de
P75 + 1,5 × IQR. Contemos cuántos clientes son.
""")

code(r"""
# El límite de la regla
limite = p75 + 1.5 * iqr
print('Límite:', round(limite, 2))

# Cuántos clientes quedan por encima del límite
atipicos = df[df['charges'] > limite]
print('Atípicos:', len(atipicos))

# Cuántos de esos atípicos fuman
print('De ellos fuman:', len(atipicos[atipicos['smoker'] == 'yes']))
""")

md(r"""
Son 139 clientes, y 136 de ellos fuman. **No son errores: son los clientes
caros, y son reales.** Si los borráramos, la aseguradora dejaría de ver
justo a los que más le cuestan.

La regla **marca** valores para que alguien los revise. **No decide** si
están bien o mal. Eso lo decide quien conoce el problema.

### ¿Cuánto del costo ponen los clientes más caros?
""")

code(r"""
# El P90: el 10 % de clientes más caros está por encima de este valor
p90 = df['charges'].quantile(0.9)

# Los clientes del 10 % más caro
caros = df[df['charges'] >= p90]
print('Clientes en el 10 % más caro:', len(caros))

# Qué parte del costo total ponen ellos
parte = caros['charges'].sum() / df['charges'].sum()

# Lo multiplicamos por 100 para verlo como porcentaje
print('Parte del costo total:', round(parte * 100, 1), '%')
""")

md(r"""
El 10 % de clientes más caro pone el **32 %** del costo total. Para la
aseguradora, el riesgo está en la cola.

### ✏️ Ejercicio 4 · El resumen completo del IMC (8 minutos)

Reemplace cada `___`, corra las celdas y responda abajo.
""")

ejercicio(r"""
# a) Media, mediana y desviación estándar del IMC
print('Media:', round(df['bmi'].___(), 2))
print('Mediana:', round(df['bmi'].___(), 2))
print('Desviación estándar:', round(df['bmi'].___(), 2))
""", r"""
# a) Media, mediana y desviación estándar del IMC
print('Media:', round(df['bmi'].mean(), 2))
print('Mediana:', round(df['bmi'].median(), 2))
print('Desviación estándar:', round(df['bmi'].std(), 2))
""")

ejercicio(r"""
# b) El P10 y el P90 del IMC
df['bmi'].quantile([___, ___])
""", r"""
# b) El P10 y el P90 del IMC
df['bmi'].quantile([0.1, 0.9])
""")

ejercicio(r"""
# c) Boxplot del IMC
plt.boxplot(df['___'])
plt.ylabel('IMC (kg/m²)')
plt.show()
""", r"""
# c) Boxplot del IMC
plt.boxplot(df['bmi'])
plt.ylabel('IMC (kg/m²)')
plt.show()
""")

ejercicio(r"""
# d) ¿Cuántos clientes quedan por encima del límite de la regla?
b25 = df['bmi'].quantile(0.25)
b75 = df['bmi'].quantile(___)
limite_imc = b75 + 1.5 * (b75 - b25)
print('Límite:', round(limite_imc, 2))
df[df['bmi'] > limite_imc]
""", r"""
# d) ¿Cuántos clientes quedan por encima del límite de la regla?
b25 = df['bmi'].quantile(0.25)
b75 = df['bmi'].quantile(0.75)
limite_imc = b75 + 1.5 * (b75 - b25)
print('Límite:', round(limite_imc, 2))
df[df['bmi'] > limite_imc]
""")

md(r"""
**Responda (doble clic para editar):**

1. ¿La media y la mediana del IMC se parecen? ¿Por qué, según su forma?
2. ¿Los atípicos del IMC son errores o personas reales? ¿Cómo lo sabe?
3. Si tuviera que dar un solo número para el IMC de un cliente típico, ¿cuál
   daría?
""")

# ================================================================ CIERRE ===
md(r"""
## Cierre

### ¿Cuánto cuesta una persona asegurada?

| Si la pregunta es... | El número | Valor |
|---|---|---|
| ¿Cuánto vamos a pagar en total? | la media × clientes | 13.270 por cliente |
| ¿Cuánto cuesta una persona típica? | la mediana | 9.382 |
| ¿Hasta cuánto cuestan 9 de cada 10? | el P90 | 34.832 |
| ¿Dónde está el riesgo? | el 10 % más caro | 32 % del costo |

**El mismo dato da respuestas distintas según la pregunta.** Por eso la
pregunta va primero.

### Lo que vimos y lo que significa

| Lo que vimos | Lo que significa |
|---|---|
| No hay vacíos y las regiones están casi empatadas | los datos son simulados; los reales no se ven así |
| Edades de 18 a 64 | la población son adultos en edad de trabajar |
| Una fila repetida | hay que decidir qué hacer con ella (próxima clase) |
| El costo está sesgado a la derecha | la media y la mediana dicen cosas distintas |
| 139 atípicos por la regla, 136 fuman | atípico no es error |

### Una pregunta que queda abierta
""")

code(r"""
# El costo con 60 barras: aparecen varias jorobas
plt.hist(df['charges'], bins=60)
plt.xlabel('Costo médico del año (USD)')
plt.ylabel('Número de clientes')
plt.title('¿Por qué hay varias jorobas?')
plt.show()
""")

md(r"""
Varias jorobas suelen ser **grupos distintos mezclados en la misma columna**.
¿Qué grupos son? Lo vamos a ver en la clase 3, cuando miremos cómo se
relacionan las variables entre sí.

### Lo que sigue

**Clase 2 · Limpiar datos.** Estos datos venían limpios. Los datos reales no:
traen vacíos, filas repetidas, valores imposibles y errores de digitación.
Vamos a ver qué implica cada problema y cómo decidir qué hacer con él.
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

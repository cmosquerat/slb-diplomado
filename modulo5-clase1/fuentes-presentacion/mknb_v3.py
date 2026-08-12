import json, pathlib
C=[]
def md(s): C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n").split("\n")})
def py(s): C.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],
                     "source":s.strip("\n").split("\n")})

md("""
# Pronosticar la producción de un pozo
### Módulo 5 · Clase 1 — *Machine Learning for Petroleum Engineers Using Python*
SLB Ecuador / UDLA · Carlos Enrique Mosquera Trujillo

---

## El encargo

> **Es 31 de enero de 2014.** Gerencia arma el presupuesto del año y necesita una sola cosa:
>
> ### «¿Cuánto va a producir el pozo 15/9-F-14 los próximos 6 meses?»

Su nombre va en ese número. Todo este cuaderno existe para producirlo con fundamento.

**El camino:**

| | La pregunta | Qué haremos |
|---|---|---|
| 1 | ¿Con qué datos contamos? | Conocer el archivo columna por columna |
| 2 | ¿Qué tipo de dato es este? | Entender qué es una serie de tiempo |
| 3 | ¿Qué nos dicen estos datos? | Explorarlos con seis herramientas |
| 4 | ¿Cómo sabremos si acertamos? | Medir un pronóstico, en barriles y en dólares |
| 5 | El pronóstico | Construirlo y entregarlo |

> **La regla que gobierna todo:** ningún cálculo puede usar información que no existía
> el 31 de enero de 2014.
""")

py("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (11, 3.8)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
pd.set_option('display.width', 120)

# los datos se cargan solos desde GitHub — no hay que subir nada
URL = ('https://raw.githubusercontent.com/cmosquerat/slb-diplomado/'
       'main/datos/volve_produccion.csv')

df = pd.read_csv(URL, parse_dates=['fecha'])
print(df.shape)
df.head()
""")

# ══════════════════════════════════════════════════════════════════════════
md("""
---
# 1 · ¿Con qué datos contamos?

## El campo Volve

Un campo real del Mar del Norte, operado por Equinor entre 2008 y 2016. Cuando lo
cerraron, liberaron **todos** sus datos al público — por eso podemos trabajar con
historia real y completa.

### Las seis columnas

| Columna | Qué contiene | Unidad |
|---|---|---|
| `fecha` | el día al que corresponde el registro | — |
| `pozo` | cuál de los 7 pozos del campo | — |
| **`horas`** | **cuántas horas de ese día el pozo estuvo abierto y produciendo** | **0 a 24** |
| `oil` | los barriles de petróleo que salieron ese día | barriles |
| `gas` | el gas producido ese día | pies cúbicos |
| `agua` | el agua producida ese día — sube con los años | barriles |

La columna resaltada es la que casi nadie mira. Y sin ella, todo lo demás se malinterpreta.
"""),
py("""
print(df['pozo'].value_counts(), '\\n')
print('rango de fechas:', df['fecha'].min().date(), '→', df['fecha'].max().date())
""")

md("""
## Cinco días reales del pozo 15/9-F-14

Antes de cualquier análisis: miremos filas de verdad.
""")
py("""
f14 = (df[df['pozo'] == '15/9-F-14']
         .sort_values('fecha')
         .set_index('fecha'))

f14.loc['2010-09-28':'2010-10-02', ['horas', 'oil', 'agua']]
""")

md("""
El **30 de septiembre** produjo 1 122 barriles: menos de la mitad que el día anterior.

**¿Se dañó el pozo?**

No. Miren la columna `horas`: ese día solo estuvo abierto **9.5 horas de las 24**.
Produjo menos porque operó menos tiempo.

## La corrección: de «barriles del día» a «capacidad del pozo»

$$\\text{tasa} \;=\; \\frac{\\text{barriles del día}}{\\text{horas que operó}} \\times 24$$
""")
py("""
w = f14.loc['2010-09-28':'2010-10-02'].copy()
w['tasa'] = w['oil'] * 24 / w['horas']
w[['horas', 'oil', 'tasa']].round(0)
""")

md("""
El 30 de septiembre, a ese ritmo, en 24 horas habría producido **2 834** — exactamente
lo mismo que sus días vecinos. **El pozo nunca cayó.**

A este número corregido lo llamaremos **la tasa del pozo**. Es la variable que vamos
a pronosticar.

## ¿Y cuando `horas` vale cero?
""")
py("""
print('días con oil = 0   :', (f14['oil'] == 0).sum())
print('días con horas = 0 :', (f14['horas'] == 0).sum())
print('días con oil > 0 y horas = 0 :', ((f14['oil'] > 0) & (f14['horas'] == 0)).sum())
""")

md("""
Coinciden casi uno a uno, y **nunca** hay producción con cero horas.

> **Los ceros no son errores de medición: son el pozo cerrado.** Mantenimiento,
> intervención, parada de plataforma.
>
> Un cero se marca y se excluye — jamás se «rellena»: rellenarlo inventa producción
> que nunca existió.

## Última verificación: ¿los 7 pozos son comparables?
""")
py("""
df.groupby('pozo').agg(
    dias=('fecha', 'size'),
    dias_sin_petroleo=('oil', lambda s: s.isna().sum()),
)
""")

md("""
`15/9-F-4` y `15/9-F-5` no producen petróleo **ni un solo día en nueve años**: son
**pozos inyectores** — meten agua al yacimiento para mantener la presión.

Antes de promediar «los pozos del campo», hay que saber qué es cada fila.

---
## Cerrando la pregunta 1: qué vamos a pronosticar

> ### La tasa mensual del pozo 15/9-F-14, en barriles por día

1. Corregida por horas de operación — mide el pozo, no el calendario de la plataforma
2. Excluyendo los días cerrados
3. Solo del pozo del encargo
4. Resumida mes a mes — porque el presupuesto se arma mensual

Nada de esto lo dice el archivo. **Lo decidimos nosotros**, y por eso hay que poder defenderlo.
""")
py("""
# construimos la variable del encargo
d = f14[(f14['horas'] > 0) & (f14['oil'] > 0)].copy()
d['tasa'] = d['oil'] * 24 / d['horas']
d['wc']   = d['agua'] / (d['agua'] + d['oil'])      # corte de agua

mensual = d['tasa'].resample('ME').median().dropna()
print(f'{len(mensual)} meses, de {mensual.index[0].date()} a {mensual.index[-1].date()}')
mensual.tail()
""")

# ══════════════════════════════════════════════════════════════════════════
md("""
---
# 2 · ¿Qué tipo de dato es este?

## Una prueba de un segundo: barajar las filas
""")
py("""
rng = np.random.default_rng(7)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
ax[0].plot(mensual.values, color='#C82B40', lw=2)
ax[0].set_title('los datos en su orden')
ax[1].plot(rng.permutation(mensual.values), color='#9CA3AF', lw=2)
ax[1].set_title('las mismas filas, barajadas')
plt.tight_layout()
""")

md("""
En Hugoton (Módulo 3) cada fila era una profundidad y se explicaba sola: barajar no
destruía nada. **Aquí barajar destruye la información. El orden ERA el dato.**

## Eso es una serie de tiempo

| Propiedad | Qué significa |
|---|---|
| **Está ordenada** | cada fila es un instante, y «antes» y «después» significan algo real |
| **Tiene un ritmo** | se observa cada cierto tiempo — y puede faltar |
| **Se acuerda de sí misma** | lo de hoy se parece a lo de ayer · **esta es la clave** |

Si lo de hoy no dijera nada sobre lo de mañana, el mejor pronóstico posible sería el
promedio histórico. Más adelante lo comprobamos.

### El ritmo: ¿están todos los días?
""")
py("""
completo = f14.asfreq('D')
print('días de calendario:', len(completo))
print('registros         :', len(f14))
print('días ausentes     :', completo['oil'].isna().sum())

huecos = f14.index.to_series().diff().dt.days
print('\\nhuecos de más de un día:', (huecos > 1).sum(), '| el más largo:', int(huecos.max()), 'días')
""")

md("""
**Tres cosas distintas — no se tratan igual:**

| | Qué significa |
|---|---|
| **Día ausente** | la fila no existe: el archivo no dice nada de ese día |
| **`NaN`** | la fila existe y el valor no se midió |
| **Cero** | se midió, y la respuesta fue cero (aquí: el pozo cerrado) |

## La consecuencia práctica: cómo se parte el tiempo

- **Antes:** `train_test_split` repartía las filas al azar. Correcto con filas independientes.
- **Ahora:** se corta **por fecha**. Todo lo anterior al 31-ene-2014 entrena, lo posterior evalúa.

Al repartir al azar, el modelo entrenaría con días de junio y lo evaluaríamos con días
de marzo: **ya vio el futuro**. El resultado se ve espectacular y no sirve para nada.

---
# 3 · ¿Qué nos dicen estos datos?

**EDA** — Análisis Exploratorio de Datos. Mirar los datos con preguntas concretas antes
de modelar:

1. ¿Cómo se comporta el pozo?
2. ¿Qué le pasó en el camino?
3. ¿Hay algo que pronosticar?
4. ¿Con qué parte de la historia entrenamos?

## Herramienta 1 — La escala logarítmica

**Qué es:** en una escala normal, los espacios iguales son *restas* iguales. En una
logarítmica, son *multiplicaciones* iguales — cada «×2» ocupa lo mismo.

**Para qué sirve:** una declinación pierde un porcentaje fijo cada mes. Eso, en escala
logarítmica, se ve como una **línea recta**.
""")
py("""
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for a, log in zip(ax, [False, True]):
    a.plot(mensual.index, mensual.values, color='#C82B40', lw=2.2)
    if log:
        a.set_yscale('log')
    a.set_title('escala logarítmica' if log else 'escala normal')
ax[0].set_ylabel('tasa [bbl/d]')
plt.tight_layout()
""")

md("""
La caída del F-14 es **casi una recta**: pierde un porcentaje parecido todos los meses.

> Si en esta escala es una recta, entonces **una recta será nuestro modelo**.

## Herramienta 2 — La media móvil

**Qué es:** el dato diario salta demasiado. La media móvil promedia una ventana que se
va deslizando.

$$\\text{MA}_k(t) \;=\; \\frac{y_t + y_{t-1} + \\cdots + y_{t-k+1}}{k}$$
""")
py("""
ejemplo = pd.Series([820, 760, 910, 700, 845, 780, 690, 800, 720, 760, 830, 700])
pd.DataFrame({'valor': ejemplo, 'media_movil_3': ejemplo.rolling(3).mean().round(0)})
""")

md("""
**Elegir la ventana es una decisión.** Ventana corta: sigue de cerca la serie, pero
también sigue el ruido. Ventana larga: muy suave, pero **llega tarde**.
""")
py("""
w = d['tasa'].asfreq('D').loc['2012-06':'2014-06']
w = w.clip(upper=w.quantile(0.995))

w.plot(color='#E5E7EB', lw=0.9, label='día a día')
w.rolling(7).mean().plot(color='#E69F00', lw=2, label='media móvil 7 días')
w.rolling(30).mean().plot(color='#2563EB', lw=2, label='media móvil 30 días')
w.rolling(90).mean().plot(color='#C82B40', lw=2.4, label='media móvil 90 días')
plt.legend(); plt.ylabel('tasa [bbl/d]'); plt.show()
""")

md("""
### Un ajuste necesario: la mediana

Los días cerrados valen cero. Un solo cero dentro de la ventana **arrastra el promedio
hacia abajo**. La mediana toma el valor del medio: mientras menos de la mitad sean
ceros, ni se entera.
""")
py("""
w2 = f14['oil'].loc['2013-01':'2014-06']

w2.plot(color='#E5E7EB', lw=0.9, label='día a día (con cierres)')
w2.rolling(15).mean().plot(color='#C82B40', lw=2.2, label='media móvil 15 d')
w2.rolling(15).median().plot(color='#2563EB', lw=2.2, label='mediana móvil 15 d')
plt.legend(); plt.ylabel('barriles por día'); plt.show()
""")

md("""
> Por eso en producción **la mediana móvil es el resumen por defecto**.
>
> Y una trampa: `rolling(30, center=True)` usa 15 días del **futuro** para describir hoy.
> Se ve precioso en un gráfico y es fuga de información en un modelo.

## Herramienta 3 — La autocorrelación

**Qué es:** responde la pregunta que dejamos abierta — *¿se acuerda la serie de sí misma?*
Se corre la serie k períodos y se compara consigo misma.

$$\\rho_k \;=\; \\text{correlación}\\left(y_t,\; y_{t-k}\\right)$$
""")
py("""
k = 3   # meses

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].plot(mensual.index, mensual.values, color='#23272E', lw=2, label='la serie')
ax[0].plot(mensual.index, mensual.shift(k).values, color='#C82B40', lw=2, ls='--',
           label=f'la misma, corrida {k} meses')
ax[0].legend(); ax[0].set_ylabel('tasa [bbl/d]')
ax[1].scatter(mensual.shift(k), mensual, s=25, color='#C82B40', alpha=.6)
ax[1].set_xlabel(f'tasa hace {k} meses'); ax[1].set_ylabel('tasa de este mes')
ax[1].set_title(f'correlación = {mensual.corr(mensual.shift(k)):.2f}')
plt.tight_layout()
""")

md("""
**0.93** significa: sabiendo la producción de hace tres meses, ya sé casi toda la de
este mes.

Repitiendo esa correlación para cada retraso posible se obtiene un **correlograma**.
""")
py("""
from statsmodels.graphics.tsaplots import plot_acf

serie_diaria = d['tasa'].asfreq('D').interpolate(limit=5).dropna()
plot_acf(serie_diaria, lags=180)
plt.title('correlograma — cada barra es una correlación'); plt.show()
""")

md("""
Alta y bajando despacio = **mucha memoria**. El veredicto: sí hay algo que pronosticar.
Podemos seguir.

## Un hallazgo mientras explorábamos: 20 filas imposibles
""")
py("""
sospechosas = df[df['horas'] > 24]
print('filas con horas > 24:', len(sospechosas))
print('días de la semana   :', sospechosas['fecha'].dt.day_name().unique())
print('meses               :', sospechosas['fecha'].dt.month.unique())
""")

md("""
Los 20 caen el **último domingo de octubre**. Y buscando el caso contrario:
""")
py("""
cortos = df[(df['horas'] > 22.9) & (df['horas'] < 23.1)]
cortos = cortos[(cortos['fecha'].dt.month == 3) &
                (cortos['fecha'].dt.day_name() == 'Sunday')]
print(np.sort(cortos['fecha'].dt.date.unique()))
""")

md("""
> **Es el cambio de horario noruego:** hay días de 23 y de 25 horas.
> El medidor no se equivocó — el calendario es así.
>
> **El eje del tiempo también es un dato que se audita.**

## Otro hallazgo: el pozo 15/9-F-12 en diciembre de 2014
""")
py("""
f12 = df[df['pozo'] == '15/9-F-12'].sort_values('fecha').set_index('fecha')
g = f12.loc['2014-09':'2015-06'].resample('ME').agg(
        {'oil': 'mean', 'agua': 'mean', 'horas': 'sum'})
g['corte_agua'] = (g['agua'] / (g['agua'] + g['oil'])).round(2)
g.round(0)
""")

md("""
Un mes entero cerrado. Al reabrir: **cinco veces más petróleo y mucha menos agua**.

> Un modelo pronostica el yacimiento, **no las decisiones de la gente**. Por eso el
> pronóstico se entrega con una condición: *«válido mientras no haya intervención»*.

## Herramienta 4 — Describir cada mes con dos números, y agruparlos

Para agrupar meses parecidos, primero hay que decir **en qué se parecen**. Elegimos dos
características de cada mes:

- **la tasa** — cuántos barriles por día produce ese mes
- **el corte de agua** — qué fracción del líquido que sale es agua

Cada mes queda convertido en un **punto**. Y ahí entra el K-means del Módulo 4.
""")
py("""
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

atributos = d[['tasa', 'wc']].resample('ME').median().dropna()
X = StandardScaler().fit_transform(atributos)
atributos['grupo'] = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(X)

fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.8))
colores = ['#1B7F4B', '#E69F00', '#C82B40']
orden = atributos.groupby('grupo')['tasa'].mean().sort_values(ascending=False).index
for c, col in zip(orden, colores):
    g2 = atributos[atributos['grupo'] == c]
    ax[0].scatter(g2['wc'] * 100, g2['tasa'], s=40, color=col)
    ax[1].scatter(g2.index, g2['tasa'], s=32, color=col)
ax[0].set_xlabel('corte de agua [%]'); ax[0].set_ylabel('tasa [bbl/d]')
ax[0].set_title('los meses agrupados'); ax[1].set_title('los mismos grupos, en el tiempo')
plt.tight_layout()
""")

md("""
K-means encontró **las tres etapas de la vida del pozo** — plateau, transición y cola —
y ninguna estaba etiquetada en el archivo.

## Herramienta 5 — DBSCAN señala los días que no encajan

A diferencia de K-means, DBSCAN puede decir «este punto no pertenece a ningún grupo».
Esos son los **días anómalos**.
""")
py("""
from sklearn.cluster import DBSCAN

dd = d[d['horas'] >= 20].copy()
dd['dlog'] = np.log(dd['tasa']).diff()
feats = dd[['tasa', 'wc', 'dlog']].dropna()

etiqueta = DBSCAN(eps=0.6, min_samples=12).fit_predict(
               StandardScaler().fit_transform(feats))
raros = feats[etiqueta == -1]

plt.plot(feats.index, feats['tasa'], lw=0.8, color='#9CA3AF')
plt.scatter(raros.index, raros['tasa'], s=28, color='#C82B40',
            label=f'{len(raros)} días raros ({100*len(raros)/len(feats):.1f} %)')
plt.legend(); plt.ylabel('tasa [bbl/d]'); plt.show()
""")

md("""
Caídas bruscas, picos de medición, arranques después de un cierre.
**Es la lista de días a revisar antes de entrenar cualquier modelo.**

---
## Lo que la exploración nos dejó decidido

| Lo que vimos | Lo que decidimos |
|---|---|
| `horas` explica caídas que no son del pozo | trabajar con la **tasa corregida** |
| los ceros son cierres operativos | **excluir** esos días |
| F-4 y F-5 son inyectores | quedarnos solo con el pozo del encargo |
| en escala logarítmica la caída es una recta | el modelo será una **recta sobre el logaritmo** |
| la serie tiene mucha memoria | sí hay algo que pronosticar: seguimos |
| el pozo vivió **tres etapas** | entrenar solo con la **etapa actual** |

La última decisión es la que más va a valer.
""")

# ══════════════════════════════════════════════════════════════════════════
md("""
---
# 4 · ¿Cómo sabremos si acertamos?

## Primero: contra qué nos comparamos

> ### «El mes que viene producirá lo mismo que este mes»

Ese es el **pronóstico ingenuo**: el más tonto posible. No cuesta nada, es
sorprendentemente difícil de vencer, y **define el mínimo aceptable**: un modelo que no
le gana, no se presenta.

## La medida principal: el error típico (MAE)

Para cada mes, cuánto nos equivocamos. Después promediamos.

$$\\text{MAE} \;=\; \\frac{1}{n}\\sum \\left|\\,\\text{real} - \\text{pronóstico}\\,\\right|$$
""")
py("""
ORIGEN = pd.Timestamp('2014-01-31')

tr = mensual[mensual.index <= ORIGEN]                                  # historia
te = mensual[(mensual.index > ORIGEN) & (mensual.index <= '2014-07-31')]  # los 6 meses

ingenuo = np.repeat(tr.iloc[-1], len(te))

def metricas(real, pron):
    e = real - pron
    return pd.Series({
        'MAE'  : np.mean(np.abs(e)),
        'RMSE' : np.sqrt(np.mean(e ** 2)),
        'MAPE' : 100 * np.mean(np.abs(e / real)),
        'sesgo': np.mean(e),
    }).round(1)

pd.DataFrame({'real': te.round(0).values,
              'ingenuo': ingenuo.round(0),
              'error': (te.values - ingenuo).round(0)},
             index=te.index.strftime('%Y-%m'))
""")
py("""
metricas(te.values, ingenuo)
""")

md("""
**Otras tres medidas, para otras tres preguntas:**

| | Qué pregunta responde | Cuidado con |
|---|---|---|
| **RMSE** | ¿qué tan graves son mis peores meses? | castiga mucho los errores grandes |
| **MAPE** | ¿en qué porcentaje me equivoco? | se dispara si la producción es casi cero |
| **sesgo** | ¿me equivoco siempre para el mismo lado? | puede dar cero con errores enormes que se cancelan |

## El error no es un número: crece con la distancia
""")
py("""
horizontes = range(1, 13)
err = [(mensual - mensual.shift(h)).abs().dropna().mean() for h in horizontes]

plt.plot(list(horizontes), err, 'o-', color='#C82B40', lw=2.5, ms=8)
plt.xlabel('¿cuántos meses adelante pronosticamos?')
plt.ylabel('error típico [bbl/d]'); plt.xticks(list(horizontes)); plt.show()

print(f'a 1 mes: {err[0]:.0f} bbl/d     a 12 meses: {err[-1]:.0f} bbl/d')
""")

md("""
Un solo número de error no describe nada: **el pronóstico se entrega como una tabla**
—a un mes, a tres, a seis— y el horizonte se acuerda antes de modelar.

## El error, traducido al idioma de gerencia
""")
py("""
PRECIO = 70    # USD por barril

mae_ingenuo = np.mean(np.abs(te.values - ingenuo))
print(f'MAE del ingenuo : {mae_ingenuo:.0f} bbl/d')
print(f'Costo al mes    : {mae_ingenuo * 30 * PRECIO / 1000:.0f} mil USD')
""")

# ══════════════════════════════════════════════════════════════════════════
md("""
---
# 5 · El pronóstico

## La idea ya la vimos en la exploración

En escala logarítmica la caída era casi una recta. **Entonces ajustemos una recta:**

$$\\log(\\text{tasa}) \;=\; a + b\\cdot t$$

Es la **regresión lineal del Módulo 3**: misma clase, misma sintaxis, mismo `.fit()`.
Lo único distinto es que la variable de entrada es el tiempo.

> **¿Por qué una recta y no un árbol?** Un árbol nunca predice un valor fuera del rango
> que vio entrenando. Un pozo en declinación **siempre** sale de ese rango.
""")
py("""
from sklearn.linear_model import LinearRegression

def recta_log(serie, n_futuro):
    \"\"\"Ajusta log(y) = a + b·t y devuelve el pronóstico de los próximos n meses.\"\"\"
    t  = np.arange(len(serie)).reshape(-1, 1)
    lr = LinearRegression().fit(t, np.log(serie.values))
    t_fut = np.arange(len(serie), len(serie) + n_futuro).reshape(-1, 1)
    return np.exp(lr.predict(t_fut))

candidatos = {
    'ingenuo (último mes)'      : ingenuo,
    'recta con toda la historia': recta_log(tr, len(te)),
    'recta con el último año'   : recta_log(tr.iloc[-12:], len(te)),
}

for nombre, pron in candidatos.items():
    mae = np.mean(np.abs(te.values - pron))
    print(f'{nombre:28s}  MAE {mae:6.1f} bbl/d   ≈ {mae*30*PRECIO/1000:5.0f} mil USD/mes')
""")
py("""
plt.plot(tr.loc['2012':].index, tr.loc['2012':].values, color='#9CA3AF', lw=2,
         label='historia conocida')
plt.plot(te.index, te.values, 'o', color='#23272E', ms=9, label='lo que realmente pasó')
for (nombre, pron), col, ls in zip(candidatos.items(),
                                   ['#E69F00', '#2563EB', '#C82B40'], ['--', ':', '-']):
    plt.plot(te.index, pron, ls, color=col, lw=2.4, label=nombre)
plt.axvline(ORIGEN, color='#23272E', lw=1.2, ls='--')
plt.legend(fontsize=9); plt.ylabel('tasa [bbl/d]'); plt.show()
""")

md("""
## Por qué pasa eso

Con **toda la historia**, la recta PIERDE contra el pronóstico ingenuo. Con solo el
**último año**, le gana tres veces.

K-means ya nos lo había dicho: el pozo vivió **tres etapas**. Ajustar una sola recta a
las tres es pedirle que describa tres comportamientos distintos. Entrenar con la etapa
actual es darle al modelo **el pozo que tiene enfrente hoy**.

> **El EDA no era decoración: fue el que eligió el tramo de entrenamiento.**

---
## El encargo, respondido
""")
py("""
pron_final = recta_log(tr.iloc[-12:], len(te))
mae_final  = np.mean(np.abs(te.values - pron_final))

entrega = pd.DataFrame({'pronostico_bbl_d': pron_final.round(0)},
                       index=te.index.strftime('%Y-%m'))
print(entrega.to_string())
print(f'\\nerror esperado: ±{mae_final:.0f} bbl/d  ({mae_final*30*PRECIO/1000:.0f} mil USD/mes)')
""")

md("""
> **Lo que se entrega a gerencia:** pronóstico mensual de febrero a julio de 2014, con
> un error esperado de unos 30 barriles por día — unos 62 mil dólares al mes —
> entrenado con el comportamiento actual del pozo y **válido mientras no haya intervención**.

Un número, su incertidumbre y su condición. Eso es un pronóstico profesional.
""")

# ══════════════════════════════════════════════════════════════════════════
md("""
---
# Su turno: el pozo 15/9-F-11

> **Es 30 de junio de 2015.** Gerencia ahora pide el **15/9-F-11**, el pozo más joven
> del campo (arrancó en julio de 2013). Mismos 6 meses.

Repitan el camino completo. *Pista: en este pozo el resultado los va a sorprender — y la
decisión de qué entregar es exactamente el trabajo del ingeniero.*
""")
py("""
# punto de partida
f11 = (df[df['pozo'] == '15/9-F-11']
         .sort_values('fecha')
         .set_index('fecha'))

ORIGEN_11 = pd.Timestamp('2015-06-30')
f11.head()
""")

md("""
### 1 · Conozcan el pozo
Días registrados, días de calendario, huecos, días cerrados, días parciales.
""")
py("# TODO 1\n\n")

md("""
### 2 · Construyan la variable
La tasa corregida por horas, sin días cerrados, resumida mes a mes (mediana).
""")
py("# TODO 2\n\n")

md("""
### 3 · Explórenlo
Escala logarítmica, media y mediana móvil, autocorrelación. ¿Hay memoria?
""")
py("# TODO 3\n\n")

md("""
### 4 · Los tres eventos
Identifiquen los tres eventos más importantes de la historia del pozo y propongan qué
fue cada uno. El dato dice *dónde*; su experiencia de campo dice *qué*.
""")
py("# TODO 4\n\n")

md("""
### 5 · El número a vencer
Calculen el pronóstico ingenuo a 6 meses desde `ORIGEN_11` y su MAE.
""")
py("# TODO 5\n\n")

md("""
### 6 · La recta
Ajusten `recta_log` con toda la historia y con el último año. Comparen contra el ingenuo.
""")
py("# TODO 6\n\n")

md("""
### 7 · La entrega

**¿Qué número firma usted, con qué historia lo entrenó — y le gana al ingenuo?**

Una frase, defendible.
""")
py("# TODO 7 — su entrega\n\n")

md("""
---
## Lo que se llevan hoy

- Resolvieron un encargo real: de la pregunta de gerencia al número entregado
- Con la **regresión lineal que ya sabían** — solo cambió qué se pone en el eje X
- Aprendieron a leer un archivo de producción columna por columna, y a **corregir la
  variable** antes de modelar nada
- Seis herramientas de exploración, cada una con su para qué
- Y a medir el error **en barriles y en dólares**, contra una referencia

> **La regla del módulo:** ningún cálculo puede usar información que no existía en el
> momento que se está describiendo.

**Próxima clase — Curvas de declinación de Arps**, y por qué el mismo modelo puede
parecer excelente y ser inservible según cómo se partan los datos.

---
*Datos: campo Volve, Equinor ASA (dataset abierto, 2018).*
""")

nb={"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11"},"colab":{"provenance":[],"toc_visible":True}},
    "nbformat":4,"nbformat_minor":5}
p=pathlib.Path('Modulo5_Clase1_Pronosticar_Produccion.ipynb')
p.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
print('✓',p.name,'—',len(C),'celdas')

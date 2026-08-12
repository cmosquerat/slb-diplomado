# Chuleta — Pronosticar la producción de un pozo
### Módulo 5 · Clase 1 · *Machine Learning for Petroleum Engineers Using Python*

> **La regla del módulo**
> Ningún cálculo puede usar información que no existía en el momento que se está describiendo.

---

## 1 · Conocer el archivo

| Columna | Qué contiene | Unidad |
|---|---|---|
| `fecha` | el día del registro | — |
| `pozo` | cuál de los 7 pozos | — |
| **`horas`** | **horas que el pozo estuvo abierto ese día** | **0 a 24** |
| `oil` | barriles de petróleo que salieron ese día | barriles |
| `gas` | gas producido ese día | pies cúbicos |
| `agua` | agua producida ese día | barriles |

> `oil` es un **volumen del día**, no una tasa. Si el pozo operó 9.5 h, produjo menos
> barriles — pero su capacidad no cambió.

```python
df  = pd.read_csv(URL, parse_dates=['fecha'])

f14 = (df[df['pozo'] == '15/9-F-14']
         .sort_values('fecha')       # ← imprescindible antes de indexar
         .set_index('fecha'))
```

---

## 2 · Corregir la variable

```python
d = f14[(f14['horas'] > 0) & (f14['oil'] > 0)].copy()   # fuera los cierres
d['tasa'] = d['oil'] * 24 / d['horas']                  # capacidad del pozo
d['wc']   = d['agua'] / (d['agua'] + d['oil'])          # corte de agua

mensual = d['tasa'].resample('ME').median().dropna()    # mediana: robusta
```

| Situación | Qué significa | Qué hacer |
|---|---|---|
| `horas == 0` | pozo cerrado | excluir — **nunca** rellenar |
| `0 < horas < 24` | día parcial | normalizar a 24 h |
| `horas == 25` o `== 23` | cambio de horario | es correcto, no es un error |
| `oil` siempre nulo | es un **inyector** | no es un productor |

---

## 3 · Frecuencia y huecos

```python
completo = f14.asfreq('D')                      # impone el calendario diario
completo['oil'].isna().sum()                    # días ausentes

huecos = f14.index.to_series().diff().dt.days
huecos[huecos > 1]                              # dónde y qué tan largos
```

**Tres cosas distintas:** día **ausente** (la fila no existe) · **`NaN`** (existe, no se
midió) · **cero** (se midió y dio cero — aquí: cerrado).

---

## 4 · Explorar (EDA)

```python
# escala logarítmica: una declinación constante se ve como una recta
mensual.plot(logy=True)

# media móvil: promedia una ventana que se desliza
s.rolling(30).mean()       # nivel reciente
s.rolling(30).median()     # robusto a cierres ← por defecto en producción
s.rolling(30).std()        # estabilidad
s.ewm(span=30).mean()      # memoria que se desvanece

# NUNCA en un modelo:
s.rolling(30, center=True).mean()   # usa 15 días del futuro

# autocorrelación: ¿se acuerda la serie de sí misma?
mensual.corr(mensual.shift(3))
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(serie_diaria, lags=180)

# mirar hacia atrás (los predictores honestos)
s.shift(1)        # el valor de ayer
s.diff()          # hoy − ayer
np.log(s).diff()  # tasa de declinación
```

### Sin etiquetas (Módulo 4 aplicado al tiempo)

```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

# 1) describir cada mes con dos características → cada mes es un punto
atributos = d[['tasa', 'wc']].resample('ME').median().dropna()

# 2) agrupar los meses parecidos → las etapas de vida del pozo
X = StandardScaler().fit_transform(atributos)
atributos['grupo'] = KMeans(n_clusters=3, n_init=10).fit_predict(X)

# 3) señalar los días que no encajan → la lista a auditar
etiqueta = DBSCAN(eps=0.6, min_samples=12).fit_predict(
               StandardScaler().fit_transform(diario[['tasa', 'wc', 'dlog']]))
raros = diario[etiqueta == -1]
```

---

## 5 · Medir un pronóstico

```python
ORIGEN = pd.Timestamp('2014-01-31')
tr = mensual[mensual.index <= ORIGEN]          # el corte honesto
te = mensual[mensual.index >  ORIGEN][:6]

ingenuo = np.repeat(tr.iloc[-1], len(te))      # «igual que este mes» ← la vara

def metricas(real, pron):
    e = real - pron
    return {'MAE'  : np.mean(np.abs(e)),       # error típico
            'RMSE' : np.sqrt(np.mean(e**2)),   # castiga los peores meses
            'MAPE' : 100 * np.mean(np.abs(e / real)),
            'sesgo': np.mean(e)}               # ¿siempre para el mismo lado?
```

- El error **crece con el horizonte**: se reporta por horizonte, no como un solo número.
- En dólares: `MAE × 30 × precio`. Un error de 100 bbl/d durante un mes ≈ 210 mil USD.
- **Un modelo que no le gana al ingenuo, no se presenta.**

---

## 6 · El modelo

```python
from sklearn.linear_model import LinearRegression   # el mismo del Módulo 3

def recta_log(serie, n_futuro):
    t  = np.arange(len(serie)).reshape(-1, 1)
    lr = LinearRegression().fit(t, np.log(serie.values))     # log(tasa) = a + b·t
    t_fut = np.arange(len(serie), len(serie) + n_futuro).reshape(-1, 1)
    return np.exp(lr.predict(t_fut))

recta_log(tr, 6)              # con toda la historia  → MAE 132  (pierde)
recta_log(tr.iloc[-12:], 6)   # con el último año     → MAE  30  (gana 3×)
```

> El modelo es trivial: **lo que vale es qué historia le das de comer.**
> El EDA (K-means) fue el que eligió el tramo de entrenamiento.
>
> **Una recta extrapola; un árbol no.** Un pozo en declinación siempre sale del rango visto.

---

## 7 · Vocabulario

| Término | Qué significa |
|---|---|
| **Origen** | el instante desde el cual se pronostica; lo posterior es desconocido |
| **Horizonte** | cuánto hacia adelante — el error se reporta **por horizonte** |
| **In-sample** | error sobre datos ya vistos: siempre optimista, no es un resultado |
| **Out-of-sample** | error sobre datos posteriores al origen: el único que se muestra |

> En una serie de tiempo la partición **no se sortea: se corta por fecha**.

---

*Datos: campo Volve, Equinor ASA (dataset abierto, 2018).*

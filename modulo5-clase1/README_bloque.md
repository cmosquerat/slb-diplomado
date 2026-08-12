<!-- ═══════════════════════════════════════════════════════════════════════
     Bloque para el README.md del repositorio  cmosquerat/slb-diplomado
     1) añadir la fila a la tabla "Contenido"
     2) pegar la sección al final, después del Módulo 3 · Clase 5
     ═══════════════════════════════════════════════════════════════════════ -->

<!-- ① FILA PARA LA TABLA DE CONTENIDO -->

| [Módulo 5 · Clase 1](modulo5-clase1/) | Pronosticar la producción de un pozo — series de tiempo, exploración de datos y el primer pronóstico (campo Volve) | ✅ Publicada |

<!-- ② SECCIÓN COMPLETA -->

## Módulo 5 · Clase 1: Pronosticar la producción de un pozo

- [`modulo5-clase1/presentacion.pdf`](modulo5-clase1/presentacion.pdf) — slides (52 láminas).
- [`modulo5-clase1/presentacion.pptx`](modulo5-clase1/presentacion.pptx) — la misma presentación, editable en PowerPoint.
- [`modulo5-clase1/Modulo5_Clase1_Pronosticar_Produccion.ipynb`](modulo5-clase1/Modulo5_Clase1_Pronosticar_Produccion.ipynb) — cuaderno Colab; los datos se cargan solos desde GitHub (la práctica se deja en blanco).
- [`modulo5-clase1/chuleta_series_tiempo.md`](modulo5-clase1/chuleta_series_tiempo.md) — referencia rápida de una página.
- [`modulo5-clase1/fuentes-presentacion/`](modulo5-clase1/fuentes-presentacion/) — generador del deck (`build_v3.js`, pptxgenjs), scripts de figuras y logotipos.
- [`datos/volve_produccion.csv`](datos/volve_produccion.csv) — producción diaria real del campo Volve (Equinor, 15 634 filas, 7 pozos, 2007–2016).

Primera clase del Módulo 5. Toda la sesión gira alrededor de un encargo con fecha:
*es 31 de enero de 2014, gerencia arma el presupuesto y necesita saber cuánto producirá
el pozo 15/9-F-14 los próximos 6 meses*. No hay herramientas nuevas — hay usos nuevos:
el archivo es el del Módulo 1, el modelo es el `LinearRegression` del Módulo 3 · Clase 1
(con el tiempo como variable de entrada, sobre el logaritmo de la tasa), el costo en
dólares viene de la Clase 2, la validación honesta de la Clase 5, y K-means y DBSCAN del
Módulo 4 deciden con qué historia entrenar.

**Contenidos cubiertos:**

1. **¿Con qué datos contamos?** — el campo Volve, las seis columnas una por una, cinco días
   reales del pozo, y la columna `horas`: el 30-sep-2010 el pozo produjo la mitad *porque
   operó 9.5 h de 24*. De ahí sale la tasa corregida, la variable del encargo
2. Los ceros son cierres operativos, no errores · los pozos F-4 y F-5 son inyectores
3. **¿Qué tipo de dato es este?** — barajar las filas, las tres propiedades de una serie de
   tiempo, sus cuatro componentes, y por qué la partición ya no puede ser aleatoria
4. **¿Qué nos dicen estos datos?** — el EDA con seis herramientas, cada una explicada antes
   de aplicarse: la escala logarítmica, la media móvil y su ventana, la mediana frente a los
   cierres, la autocorrelación, K-means para las etapas del pozo y DBSCAN para los días raros
5. Dos hallazgos reales: el cambio de horario noruego escondido en la columna `horas`, y el
   cierre del pozo F-12 en diciembre de 2014
6. **¿Cómo sabremos si acertamos?** — el pronóstico ingenuo como referencia, MAE, RMSE, MAPE
   y sesgo, el error por horizonte, y la traducción a dólares
7. **El pronóstico** — la recta sobre el logaritmo de la tasa. Con toda la historia pierde
   contra el ingenuo (277 vs 201 mil USD/mes de error); entrenada con el último año — la
   etapa que K-means encontró — gana 3× (62 mil USD/mes)
8. **Práctica integradora:** el mismo encargo sobre el pozo 15/9-F-11, donde el modelo *no*
   le gana al ingenuo y la decisión de qué entregar es el aprendizaje

*Datos: campo Volve, Equinor ASA (dataset abierto, 2018).*

<!-- ③ NOTA SOBRE LA FUENTE DE LA PRESENTACIÓN

A diferencia de las clases anteriores (Beamer/LaTeX), esta presentación se genera con
pptxgenjs. Para regenerarla:

    cd modulo5-clase1/fuentes-presentacion
    python3 figs_v3_conceptos.py     # figuras explicativas
    python3 figs_v3_datos.py         # figuras sobre los datos reales + fórmulas
    node build_v3.js                 # produce presentacion.pptx

El PDF se obtiene exportando desde PowerPoint, o con:
    soffice --headless --convert-to pdf presentacion.pptx
-->

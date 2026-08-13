# Módulo 5 — retrospectiva de la Clase 1, reglas de diseño y hoja de ruta

*Documento interno del curso. Escrito después de dictar la Clase 1 del Módulo 5.*

---

## Parte A · Qué falló en la Clase 1

La Clase 1 se dictó completa. El diagnóstico en sala fue:

> *«Herramientas sin propósito, cosas dadas por sentadas y nada desde una perspectiva
> de negocio, como hacer un EDA. Se fue mucho a la serie de tiempo en específico pero
> no a lo que realmente se necesitaba.»*

La auditoría de los archivos confirma esa lectura y le pone causa. La propuesta comercial
(`Propuesta SCHLUMBERGER Machine Learning 1125.pdf`) define el módulo así:

> **MÓDULO 5 — Machine Learning aplicado a ingeniería de yacimientos & producción · 10 h**
>
> *Temas:* ML aplicado a curvas de declinación (DCA) probabilístico · Predicción de GOR y
> WOR · Predicción de breakthrough de agua · Feature selection para análisis de EOR ·
> Predicción de presión promedio de yacimiento con ML · ML para análisis de pruebas de
> presión (DST, RFT, Buildup) · Modelos para identificar daño de formación · Fallas de
> bombas ESP mediante ML · Optimización de levantamiento artificial.
>
> *Prácticas:* Modelo ML para predicción de WOR · Modelo ML para fallas en ESP usando
> series de tiempo · Integración volumétrica + ML para forecasting.

**La Clase 1 no nombró ninguno de esos nueve temas.** Dictó una introducción a series de
tiempo. Series de tiempo es el vehículo del módulo, no su destino, y esa confusión es la
causa raíz de todo lo demás.

### Las cinco fallas

| # | Falla | Evidencia |
|---|---|---|
| **F1** | **El módulo se confundió con su vehículo.** Se contrató «ML aplicado a yacimientos y producción»; se dictó «introducción a series de tiempo». | Cero menciones a DCA, WOR, GOR, breakthrough, BHP, skin, ESP o levantamiento artificial en las 52 láminas. Ni siquiera se nombró **Arps** para la recta sobre `log(q)`, que *es* declinación exponencial. |
| **F2** | **Catálogo de herramientas en vez de encargo.** El orden lo dictó el catálogo, no el problema. | **Doce** láminas tituladas literalmente *«Herramienta 1 — La escala logarítmica»*, *«Herramienta 2 — La media móvil»*, … *«Herramienta 5 — DBSCAN»* (`build_v3.js:561–690`). Son 30 de los 120 minutos. Cada una explica «qué es / para qué sirve» en abstracto; ninguna la exige el encargo. |
| **F3** | **Se re-enseñó lo ya cobrado, y rápido.** Escala logarítmica, media móvil, mediana vs media, K-means y DBSCAN pertenecen al Módulo 2 (EDA) y al Módulo 4 (no supervisado), ya dictados. | De ahí la sensación contradictoria del grupo: *«es repaso»* y *«lo dieron por sentado»* al mismo tiempo. Se pasó por encima de material con dueño, sin profundizarlo ni usarlo. |
| **F4** | **El negocio llegó tarde y como conversión, no como decisión.** | El dinero aparece en la lámina **42 de 52**, y solo como aritmética (`MAE × 30 × 70 USD/bbl`). Ninguna decisión del encargo dependía del número. Contraste: en el Módulo 3 el costo enmarcaba la clase desde el arranque (matriz de costos en C2, matriz de penalización en C4, «récord a batir» en las cinco). |
| **F5** | **Ruptura del sistema del curso.** Se abandonó Beamer/LaTeX por `pptxgenjs`. | Otra tipografía (Calibri/Consolas vs Fira Sans/Fira Mono), otra retícula, otra paleta. Se perdieron los dispositivos que el grupo ya sabía leer: *Dónde Estamos*, *Ustedes Ya…*, *Acotar el Problema*, *récord a batir*, *Su Turno · 15 min*, *Lo Que Aprendimos Hoy*, *Lo Que Sigue*. Además quedó fuera de git, con la carpeta anidada dos veces y con diez `f_*.png` sin script generador. |

Falla complementaria: la práctica quedó en **siete celdas `# TODO` sin andamiaje**.

---

## Parte B · Un hallazgo del rediseño que conviene dejar escrito

Al diseñar la Clase 2 se intentó primero un encargo de **pronóstico de agua** sobre
producción mensual de campos del Mar del Norte. Se descartó después de medirlo. Vale la
pena registrar por qué, porque es una trampa fácil de repetir:

| Prueba | Resultado |
|---|---|
| Cruce de umbral a 6 años vista | Mal condicionado: el mes falla hasta ±44 meses |
| Pronóstico a 12 meses con Random Forest y rezagos | Ingenuo 4 932 bbl/d · RF 5 766 bbl/d — **el modelo pierde** |
| Diagnóstico WOR vs Np (Eldfisk) | R² = 0,55 en la última década — no sirve |
| Corte de agua de Volve con presiones | RF 19,5 pp vs tonto 31,0 pp: gana, pero 19 puntos es un mal modelo |
| WOR vs Np (Draugen, 36 meses) | 13,2 % vs ingenuo 18,0 % — gana, pero poco |

**La conclusión:** una serie de producción de **un solo activo** no premia al ML. Es suave
y muy autocorrelacionada, así que *«el valor de hoy»* es un rival casi imbatible. El ML
cobra cuando hay **muchas unidades y muchas variables**, y producción mensual por campo no
tiene ni lo uno ni lo otro.

> **Regla que sale de acá:** antes de comprometer una clase con un dataset, medir si el
> modelo le gana al modelo tonto. Si no le gana, el problema está mal elegido, y ninguna
> cantidad de afinado lo arregla.

---

## Parte C · Reglas de diseño para el resto del Módulo 5

Verificables antes de dictar. Cada clase nueva se revisa contra esta lista.

1. **La clase se nombra por una pregunta de ingeniería, no por una técnica.**
   Prohibido: *«Series de tiempo II»*. Válido: *«¿Cuánto antes se puede saber?»*
2. **Ninguna herramienta entra sin un encargo que la exija.** Prueba: si se quita la
   herramienta y el encargo se sigue respondiendo, se quita la herramienta.
3. **El dinero y la decisión van en los primeros 10 minutos**, como motivo de que la
   clase exista — no como conversión al final.
4. **Nada se da por sabido.** Cada concepto que se usa se vuelve a explicar desde cero
   cuando aparece, aunque tenga dueño en otro módulo. Se cita el origen *y* se explica.
5. **Cada clase nombra al menos un tema textual del temario contratado**, y lo deja
   escrito en la lámina de cierre.
6. **Beamer/LaTeX con el sistema del Módulo 3, sin excepción.** Preámbulo copiado literal
   de `modulo3-clase5/presentacion.tex:1-245`.
7. **Presupuesto de 2 h = 4 bloques, ~45 láminas, un dataset, una pregunta.** Cada
   `\sectionframe` lleva su marca de minutos (eso la Clase 1 lo hizo bien; se conserva).
8. **El cuaderno abre con un EDA de 15–20 minutos**, antes de cualquier modelo, y cierra
   con una tabla *«lo que vimos → lo que decidimos»*.
9. **La práctica lleva andamiaje**: enunciado, datos ya cargados y el primer paso resuelto.
10. **Los números de las láminas se generan, no se escriben.** Un script produce figuras
    y cifras; el cuaderno tiene que reproducirlas exactamente.
11. **Al menos un resultado negativo por clase, dicho en voz alta.** Si una idea no gana,
    se muestra que no gana. Es lo que separa una clase honesta de una demostración.
12. **Toda promesa se paga o se retira explícitamente.**

---

## Parte D · Hoja de ruta de las 10 h

### La aritmética, sin maquillaje

El temario contratado tiene **9 temas avanzados y 3 prácticas** para **10 horas**. Las
clases C1 y C2 ya consumieron 4 h y entre las dos cubrieron **un** compromiso (el método
de la práctica de series de tiempo). Quedan **6 h para 9 temas**.

A la profundidad a la que este curso enseña —donde nada se da por sabido— eso no entra.
Son ~2 h por tema real. **Criterio adoptado: garantizar las tres prácticas comprometidas**,
que son la parte contractualmente verificable, y declarar por escrito lo que se difiere.

### Lo que se cubre

| Clase | Pregunta de ingeniería | Temas y prácticas contratadas | Estado |
|---|---|---|---|
| **C1** | ¿Cuánto producirá este pozo? | — *(ninguno; deuda: Arps prometido)* | Dictada |
| **C2** | ¿Cuánto antes se puede saber que el pozo se está tapando? | El **método** de la práctica *«ML para fallas usando series de tiempo»* — aplicado a un choke, no a una bomba | **Lista** |
| **C3** | **¿Cuánto queda, y con cuánta confianza?** | Tema 1 · **DCA probabilístico** · Práctica 3 · **Integración volumétrica + ML para forecasting** · **paga la deuda de Arps** | **Lista** |
| **C4** | **¿Qué me está avisando el agua y el gas?** | Tema 2 · **GOR y WOR** · Tema 3 · **breakthrough de agua** · Práctica 1 · **Modelo ML para predicción de WOR** | Pendiente |
| **C5** | **¿Cuándo se rompe la bomba, y cómo la hago rendir?** | Tema 8 · **Fallas de bombas ESP** · Tema 9 · **Optimización de levantamiento artificial** · Práctica 2 · cierra sobre ESP real | Pendiente |

Resultado: **5 de los 9 temas** y **las 3 prácticas comprometidas**.

### Lo que se difiere, explícitamente

Estos cuatro temas **no entran en 10 horas** y se declaran diferidos al **Módulo 7
(proyecto integrador)**, donde el alumno los aplica sobre su propio dataset:

- Tema 4 · Feature selection para análisis de **EOR**
- Tema 5 · Predicción de **presión promedio de yacimiento** con ML
- Tema 6 · ML para análisis de **pruebas de presión** (DST, RFT, Buildup)
- Tema 7 · Modelos para identificar **daño de formación**

Se dice en la lámina de cierre de la C5, no se deja sin mencionar.

### Principio de diseño heredado de la C2

La lección estructural de la Clase 2 no fue sobre incrustación: fue que **el ML cobra
cuando hay muchas unidades y muchas variables**. Con una sola serie, el modelo tonto
empata (ver Parte B). Por eso las tres clases que faltan se plantean **a nivel de flota**,
no de un solo pozo:

- **C3** — Arps ajustado al propio pozo es el **modelo tonto**; el modelo aprende de miles
  de pozos análogos, y el P10/P50/P90 sale de la dispersión de esos análogos. Así el DCA
  probabilístico deja de ser una fórmula y pasa a ser una medición.
- **C4** — «de estos N pozos, ¿cuáles hacen agua el año que viene?», no «cuánta agua hace
  este pozo».
- **C5** — el método de la C2 aplicado a una flota de bombas.

---

## Parte E · Higiene pendiente

- `modulo5-clase1/` está **fuera de git** y con la carpeta anidada dos veces
  (`modulo5-clase1/modulo5-clase1/`). Hay que aplanarla y commitearla.
- Los diez `f_*.png` de esa clase **no tienen script generador**; los dos scripts de
  figuras que sí existen apuntan a rutas de sandbox (`/mnt/user-data/…`) y no corren
  fuera del entorno donde se hicieron.
- Queda **recomendado, no decidido**: reconstruir la Clase 1 en Beamer y recortarla a
  un solo encargo, aplicando las reglas de la Parte C.

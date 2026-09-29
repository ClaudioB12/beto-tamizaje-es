# Resultados — sintetico-humo

> **PRUEBA DE HUMO (1 época, 64 tokens): estas cifras solo verifican el pipeline. NO REPORTAR.**
> **Datos sintéticos: no miden el desempeño del tamizaje. NO REPORTAR.**

F1 macro: media del F1 binario de ansiedad y de depresión, cada uno sobre sus filas con etiqueta conocida (propuesta D3, pendiente de aprobación).

## PE1 — F1 macro por arquitectura

**[NO CITABLE — prueba de humo · datos sintéticos]**

| Dominio | Arquitectura | F1 media ± DE entre semillas | Rango entre semillas | IC 95 % bootstrap sobre fragmentos | F1 ansiedad | F1 depresión |
|---|---|---|---|---|---|---|
| sintetico | beto_ajustado | 0.592 ± 0.033 | 0.539–0.624 | 0.592 [0.528, 0.644] | 0.664 | 0.519 |
| sintetico | beto_sonda | 0.556 ± 0.096 | 0.428–0.638 | 0.556 [0.503, 0.600] | 0.581 | 0.530 |
| sintetico | tfidf_lr (determinista) | 0.491 ± 0.000 | 0.491–0.491 | 0.491 [0.411, 0.560] | 0.561 | 0.421 |

Las dos medidas de variabilidad no son intercambiables. La **DE y el rango entre semillas** describen cuánto cambia el F1 al cambiar la inicialización y el orden de los lotes, con los fragmentos fijos. El **IC bootstrap sobre fragmentos** describe la incertidumbre por muestreo del conjunto de prueba, con las semillas fijas, y por tanto NO incorpora la variabilidad entre semillas. Un IC más estrecho que ±1 DE no es un error de cálculo.

«Determinista»: todas sus semillas dan el mismo F1, así que sus corridas equivalen a una sola observación. En H1 se contrasta contra ese valor con Wilcoxon de una muestra.

## H1 — Diferencias entre arquitecturas

**[NO CITABLE — prueba de humo · datos sintéticos]**

| Dominio | Comparación | Diferencia [IC 95 % bootstrap sobre fragmentos] | IC excluye 0 | Prueba sobre semillas (unilateral) |
|---|---|---|---|---|
| sintetico | beto_ajustado − beto_sonda | 0.036 [-0.016, 0.085] | no | Mann–Whitney p = 0.5794 (r = -0.04) |
| sintetico | beto_ajustado − tfidf_lr | 0.101 [0.020, 0.181] | sí | Wilcoxon 1 muestra p = 0.0312 (b determinista) |

Con 5 contra 5 semillas, el p mínimo de Mann–Whitney es 1/252 ≈ 0,004 y con una arquitectura determinista el de Wilcoxon de una muestra es 1/32 ≈ 0,031: son pisos que la prueba no puede superar por grande que sea la diferencia real.

## PE2 — Brecha de transferencia de dominio

No disponible: requiere predicciones de MentalRiskES y del conjunto clínico.

## PE3 — Concordancia de severidad (kappa ponderado)

No disponible: requiere bandas congeladas (D4) y puntajes PHQ-9/GAD-7 del conjunto clínico.

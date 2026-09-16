# Resultados — sintetico-humo

> **PRUEBA DE HUMO (1 época, 64 tokens): estas cifras solo verifican el pipeline. NO REPORTAR.**
> **Datos sintéticos: no miden el desempeño del tamizaje. NO REPORTAR.**

F1 macro: media del F1 binario de ansiedad y de depresión, cada uno sobre sus filas con etiqueta conocida (propuesta D3, pendiente de aprobación).

## PE1 — F1 macro por arquitectura

| Dominio | Arquitectura | Media ± DE | Rango | IC 95 % bootstrap | F1 ansiedad | F1 depresión |
|---|---|---|---|---|---|---|
| sintetico | beto_ajustado | 0.592 ± 0.033 | 0.539–0.624 | 0.592 [0.528, 0.644] | 0.664 | 0.519 |
| sintetico | beto_sonda | 0.556 ± 0.096 | 0.428–0.638 | 0.556 [0.503, 0.600] | 0.581 | 0.530 |
| sintetico | tfidf_lr | 0.491 ± 0.000 | 0.491–0.491 | 0.491 [0.411, 0.560] | 0.561 | 0.421 |

## H1 — Diferencias entre arquitecturas

| Dominio | Comparación | Diferencia [IC 95 %] | IC excluye 0 | Mann–Whitney p (unilateral) | r |
|---|---|---|---|---|---|
| sintetico | beto_ajustado − beto_sonda | 0.036 [-0.016, 0.085] | no | 0.5794 | -0.04 |
| sintetico | beto_ajustado − tfidf_lr | 0.101 [0.020, 0.181] | sí | 0.0040 | 1.00 |

## PE2 — Brecha de transferencia de dominio

No disponible: requiere predicciones de MentalRiskES y del conjunto clínico.

## PE3 — Concordancia de severidad (kappa ponderado)

No disponible: requiere bandas congeladas (D4) y puntajes PHQ-9/GAD-7 del conjunto clínico.

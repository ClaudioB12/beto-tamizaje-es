"""Contraste de H1 sobre las semillas (decisión D6).

El §2.5 del PPI propone Wilcoxon pareada. Con n = 5, su p mínimo alcanzable es
2/2⁵ = 0,0625 bilateral: nunca rechaza H₀ con α = 0,05. Se usa Mann–Whitney
exacta unilateral (H1 es direccional), cuyo p mínimo con 5 contra 5 es
1/C(10,5) ≈ 0,004. Ese mínimo también es un techo: por grande que sea la
diferencia real, esta prueba no puede aportar más evidencia. Por eso el IC
bootstrap es el contraste principal. La prueba del PPI se reporta igual, con su
límite explícito.

Arquitecturas deterministas: si todas las semillas de una arquitectura dan el
mismo F1 (TF–IDF con lbfgs), sus corridas son copias de una sola observación.
Mann–Whitney las contaría como independientes y exageraría la evidencia (con 5
copias, p = 0,004 en vez de 0,031). En ese caso se contrasta la otra
arquitectura contra ese valor constante con Wilcoxon de una muestra.
"""
from math import comb

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon


def prueba_semillas(f1_a, f1_b) -> dict:
    """¿Supera la arquitectura a a la b? Entradas: F1 macro por semilla."""
    a, b = np.asarray(f1_a, dtype=float), np.asarray(f1_b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        raise ValueError("Se necesitan al menos 2 semillas por arquitectura")

    a_constante, b_constante = bool(np.ptp(a) == 0), bool(np.ptp(b) == 0)
    resultado = {
        "a_determinista": a_constante,
        "b_determinista": b_constante,
        "prueba_usada": None,
        "mann_whitney_unilateral": None,
        "wilcoxon_una_muestra_unilateral": None,
        "wilcoxon_pareada_bilateral_ppi": None,
        "nota": None,
    }

    if a_constante and b_constante:
        resultado["nota"] = (
            "Ambas arquitecturas dan el mismo F1 en todas sus semillas: no hay variabilidad "
            "que contrastar. Usar el IC bootstrap."
        )
        return resultado

    if a_constante or b_constante:
        # a > b  <=>  (a − b) > 0. Si b es constante se contrastan las semillas de a
        # contra b; si lo es a, las de b contra a con la hipótesis invertida.
        variable, constante, alternativa = (a, b[0], "greater") if b_constante else (b, a[0], "less")
        diferencias = variable - constante
        no_nulas = int(np.count_nonzero(diferencias))
        p = 1.0 if no_nulas == 0 else float(wilcoxon(diferencias, alternative=alternativa).pvalue)
        cual = "b" if b_constante else "a"
        resultado.update(
            prueba_usada="wilcoxon_una_muestra_unilateral",
            wilcoxon_una_muestra_unilateral={
                "p": p,
                "valor_constante": float(constante),
                "semillas_contrastadas": len(variable),
                "p_minimo_alcanzable": 1 / 2**no_nulas if no_nulas else 1.0,
            },
            nota=(
                f"La arquitectura {cual} es determinista: sus {len(b if b_constante else a)} corridas "
                "equivalen a una sola observación. Se contrasta la otra contra ese valor; "
                "Mann–Whitney exageraría la evidencia."
            ),
        )
        return resultado

    mw = mannwhitneyu(a, b, alternative="greater", method="exact")
    resultado["prueba_usada"] = "mann_whitney_unilateral"
    resultado["mann_whitney_unilateral"] = {
        "U": float(mw.statistic),
        "p": float(mw.pvalue),
        "p_minimo_alcanzable": 1 / comb(len(a) + len(b), len(a)),
        # r = 2U/(n₁n₂) − 1: 1 si todas las semillas de a superan a todas las de b.
        "r_rango_biserial": float(2 * mw.statistic / (len(a) * len(b)) - 1),
    }

    if len(a) == len(b):
        diferencias = a - b
        p_minimo = 2 / 2 ** len(a)
        nota = f"con {len(a)} semillas su p mínimo es {p_minimo:.4f}" + (
            ": no puede ser < 0,05." if p_minimo >= 0.05 else "."
        )
        p = 1.0 if np.all(diferencias == 0) else float(wilcoxon(a, b, alternative="two-sided").pvalue)
        resultado["wilcoxon_pareada_bilateral_ppi"] = {"p": p, "p_minimo_alcanzable": p_minimo, "nota": nota}
    return resultado

"""Contraste de H1 sobre las cinco semillas (decisión D6).

El §2.5 del PPI propone Wilcoxon pareada. Con n = 5, su p mínimo alcanzable es
2/2⁵ = 0,0625 bilateral: nunca rechaza H₀ con α = 0,05. Se usa Mann–Whitney
exacta unilateral (H1 es direccional), cuyo p mínimo con 5 contra 5 es
1/C(10,5) ≈ 0,004. La prueba del PPI se reporta igual, con su límite explícito.
"""
from math import comb

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon


def prueba_semillas(f1_a, f1_b) -> dict:
    """¿Supera la arquitectura a a la b? Entradas: F1 macro por semilla."""
    a, b = np.asarray(f1_a, dtype=float), np.asarray(f1_b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        raise ValueError("Se necesitan al menos 2 semillas por arquitectura")

    mw = mannwhitneyu(a, b, alternative="greater", method="exact")
    resultado = {
        "mann_whitney_unilateral": {
            "U": float(mw.statistic),
            "p": float(mw.pvalue),
            "p_minimo_alcanzable": 1 / comb(len(a) + len(b), len(a)),
            # r = 2U/(n₁n₂) − 1: 1 si todas las semillas de a superan a todas las de b.
            "r_rango_biserial": float(2 * mw.statistic / (len(a) * len(b)) - 1),
        },
        "wilcoxon_pareada_bilateral_ppi": None,
    }

    if len(a) == len(b):
        diferencias = a - b
        p_minimo = 2 / 2 ** len(a)
        nota = (
            f"con {len(a)} semillas su p mínimo es {p_minimo:.4f}"
            + (": no puede ser < 0,05." if p_minimo >= 0.05 else ".")
        )
        if np.all(diferencias == 0):
            resultado["wilcoxon_pareada_bilateral_ppi"] = {"p": 1.0, "p_minimo_alcanzable": p_minimo, "nota": nota}
        else:
            w = wilcoxon(a, b, alternative="two-sided")
            resultado["wilcoxon_pareada_bilateral_ppi"] = {
                "p": float(w.pvalue),
                "p_minimo_alcanzable": p_minimo,
                "nota": nota,
            }
    return resultado

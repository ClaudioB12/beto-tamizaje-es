"""Acuerdo entre categorías: kappa ponderado (VD2) y kappa de Cohen (OE1)."""
import warnings

import numpy as np
from sklearn.metrics import cohen_kappa_score

N_BOOTSTRAP_PPI = 10_000


def _codificar(valores, orden) -> np.ndarray:
    posicion = {categoria: i for i, categoria in enumerate(orden)}
    desconocidas = sorted({v for v in valores if v not in posicion})
    if desconocidas:
        raise ValueError(f"Categorías fuera del orden declarado {list(orden)}: {desconocidas}")
    return np.array([posicion[v] for v in valores])


def _kappa_con_ic(a, b, etiquetas, pesos, n_bootstrap, alfa, semilla) -> dict:
    kappa = float(cohen_kappa_score(a, b, labels=etiquetas, weights=pesos))
    rng = np.random.default_rng(semilla)
    n = len(a)
    muestras = []
    with warnings.catch_warnings():
        # Un remuestreo con una sola categoría en ambos lados deja kappa sin
        # definir (acuerdo esperado = 1); esos remuestreos se descartan y se cuentan.
        warnings.simplefilter("ignore")
        for _ in range(n_bootstrap):
            i = rng.integers(0, n, size=n)
            valor = cohen_kappa_score(a[i], b[i], labels=etiquetas, weights=pesos)
            if np.isfinite(valor):
                muestras.append(valor)
    inferior, superior = (
        np.percentile(muestras, [100 * alfa / 2, 100 * (1 - alfa / 2)]) if muestras else (np.nan, np.nan)
    )
    return {
        "kappa": kappa,
        "ic_inferior": float(inferior),
        "ic_superior": float(superior),
        "nivel": 1 - alfa,
        "casos": n,
        "bootstrap_validos": len(muestras),
        "n_bootstrap": n_bootstrap,
    }


def kappa_ponderado(referencia, inferida, orden, n_bootstrap=N_BOOTSTRAP_PPI, alfa=0.05, semilla=2026) -> dict:
    """Kappa con pesos cuadráticos entre categorías ordinales (PPI §2.5, VD2)."""
    if len(referencia) != len(inferida):
        raise ValueError("referencia e inferida deben tener la misma longitud")
    a, b = _codificar(referencia, orden), _codificar(inferida, orden)
    return _kappa_con_ic(a, b, list(range(len(orden))), "quadratic", n_bootstrap, alfa, semilla)


def kappa_cohen(anotador_1, anotador_2, n_bootstrap=N_BOOTSTRAP_PPI, alfa=0.05, semilla=2026) -> dict:
    """Kappa de Cohen sin ponderar entre los dos anotadores (OE1)."""
    if len(anotador_1) != len(anotador_2):
        raise ValueError("Ambos anotadores deben etiquetar los mismos fragmentos")
    orden = sorted(set(anotador_1) | set(anotador_2))
    a, b = _codificar(anotador_1, orden), _codificar(anotador_2, orden)
    return _kappa_con_ic(a, b, list(range(len(orden))), None, n_bootstrap, alfa, semilla)

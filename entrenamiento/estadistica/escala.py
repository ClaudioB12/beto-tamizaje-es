"""Escala de Utilidad Clínica: validez de contenido y confiabilidad (OE5, VD3)."""
import numpy as np
from sklearn.decomposition import FactorAnalysis

UMBRAL_V_AIKEN_PPI = 0.80


def v_aiken(calificaciones, minimo: int = 1, maximo: int = 5) -> np.ndarray:
    """V de Aiken por ítem. Entrada: matriz jueces × ítems en una escala [minimo, maximo].

    V = Σ(x − mínimo) / (n · (máximo − mínimo)). El PPI acepta ítems con V ≥ 0,80.
    """
    matriz = np.atleast_2d(np.asarray(calificaciones, dtype=float))
    if matriz.min() < minimo or matriz.max() > maximo:
        raise ValueError(f"Calificaciones fuera de la escala {minimo}–{maximo}")
    jueces = matriz.shape[0]
    return (matriz - minimo).sum(axis=0) / (jueces * (maximo - minimo))


def alfa_cronbach(respuestas) -> float:
    """Matriz evaluaciones × ítems. α = k/(k−1) · (1 − Σ var(ítem) / var(total))."""
    matriz = np.asarray(respuestas, dtype=float)
    k = matriz.shape[1]
    if k < 2:
        raise ValueError("El alfa de Cronbach necesita al menos 2 ítems")
    varianza_total = matriz.sum(axis=1).var(ddof=1)
    if varianza_total == 0:
        raise ValueError("La puntuación total no varía: el alfa no está definido")
    return float(k / (k - 1) * (1 - matriz.var(axis=0, ddof=1).sum() / varianza_total))


def omega_mcdonald(respuestas, semilla: int = 0) -> dict:
    """Omega total a partir de un modelo de un factor sobre los ítems estandarizados.

    ω = (Σλ)² / ((Σλ)² + Σψ), con λ las cargas y ψ las varianzas únicas. Asume
    unidimensionalidad: si la escala tiene cuatro dimensiones, calcúlalo por
    dimensión y contrástalo con un paquete de referencia antes de reportar.
    """
    matriz = np.asarray(respuestas, dtype=float)
    desviaciones = matriz.std(axis=0, ddof=1)
    if np.any(desviaciones == 0):
        raise ValueError("Algún ítem no varía: no se puede estandarizar")
    estandarizada = (matriz - matriz.mean(axis=0)) / desviaciones
    modelo = FactorAnalysis(n_components=1, random_state=semilla).fit(estandarizada)
    cargas = modelo.components_[0]
    if cargas.sum() < 0:  # el signo del factor es arbitrario
        cargas = -cargas
    unicas = modelo.noise_variance_
    comun = cargas.sum() ** 2
    return {"omega": float(comun / (comun + unicas.sum())), "cargas": cargas.tolist()}

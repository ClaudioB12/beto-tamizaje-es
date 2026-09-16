"""Categorías de severidad de PHQ-9 y GAD-7.

Rangos estándar de cada instrumento. Esto está fijado por la literatura; la
traducción de la salida del clasificador a estas categorías es la decisión D4 y
vive en ``evaluacion.severidad``.

- PHQ-9 (Kroenke, Spitzer y Williams, 2001): 0–27.
  0–4 mínima · 5–9 leve · 10–14 moderada · 15–19 moderadamente severa · 20–27 severa
- GAD-7 (Spitzer, Kroenke, Williams y Löwe, 2006): 0–21.
  0–4 mínima · 5–9 leve · 10–14 moderada · 15–21 severa
"""
import numpy as np

ORDEN_PHQ9 = ("minima", "leve", "moderada", "moderadamente_severa", "severa")
LIMITES_PHQ9 = (5, 10, 15, 20)
MAXIMO_PHQ9 = 27

ORDEN_GAD7 = ("minima", "leve", "moderada", "severa")
LIMITES_GAD7 = (5, 10, 15)
MAXIMO_GAD7 = 21

# Qué instrumento ancla la severidad de cada etiqueta del clasificador.
INSTRUMENTO_POR_ETIQUETA = {"ansiedad": "gad7", "depresion": "phq9"}
ORDEN_POR_INSTRUMENTO = {"phq9": ORDEN_PHQ9, "gad7": ORDEN_GAD7}
_LIMITES = {"phq9": (LIMITES_PHQ9, MAXIMO_PHQ9), "gad7": (LIMITES_GAD7, MAXIMO_GAD7)}


def categoria(instrumento: str, puntajes) -> np.ndarray:
    """Puntajes enteros -> categoría del instrumento. Rechaza valores fuera de rango."""
    limites, maximo = _LIMITES[instrumento]
    valores = np.asarray(puntajes)
    if valores.size and (
        not np.all(np.equal(np.mod(valores, 1), 0)) or valores.min() < 0 or valores.max() > maximo
    ):
        raise ValueError(f"Puntajes de {instrumento} fuera del rango entero 0–{maximo}")
    indices = np.searchsorted(limites, valores, side="right")
    return np.asarray(ORDEN_POR_INSTRUMENTO[instrumento], dtype=object)[indices]

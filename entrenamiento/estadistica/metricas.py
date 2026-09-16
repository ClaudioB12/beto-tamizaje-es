"""Métricas de clasificación sobre etiquetas conocidas (VD1).

Convenciones:
- ``y`` y ``pred`` son matrices (n, 2) en el orden de ``ETIQUETAS``; ``y`` puede
  valer -1 (desconocida) y esas celdas no cuentan.
- F1 con denominador 0 (ningún positivo real ni predicho) vale 0, como
  ``zero_division=0`` en scikit-learn.
- F1 macro = media del F1 de ansiedad y del de depresión. Es la definición
  propuesta en la decisión D3, PENDIENTE de aprobación: si el equipo elige otra,
  este es el único lugar que cambia.
"""
import numpy as np

from datos.esquema import ETIQUETAS

DEFINICION_F1_MACRO = (
    "media del F1 binario de ansiedad y de depresión, cada uno sobre sus filas "
    "con etiqueta conocida (propuesta D3, pendiente de aprobación)"
)


def _conteos(y: np.ndarray, pred: np.ndarray) -> dict[str, np.ndarray]:
    """Conteos por etiqueta sobre el último eje de filas. Admite lotes (B, n, 2)."""
    conocidas = y >= 0
    positivo_real = (y == 1) & conocidas
    negativo_real = (y == 0) & conocidas
    positivo_pred = pred == 1
    eje = y.ndim - 2
    return {
        "tp": (positivo_real & positivo_pred).sum(axis=eje),
        "fp": (negativo_real & positivo_pred).sum(axis=eje),
        "fn": (positivo_real & ~positivo_pred).sum(axis=eje),
        "tn": (negativo_real & ~positivo_pred).sum(axis=eje),
    }


def _f1(tp, fp, fn):
    denominador = 2 * tp + fp + fn
    return np.divide(2 * tp, denominador, out=np.zeros(np.shape(tp), dtype=float), where=denominador > 0)


def f1_macro(y: np.ndarray, pred: np.ndarray) -> float:
    c = _conteos(np.asarray(y), np.asarray(pred))
    return float(_f1(c["tp"], c["fp"], c["fn"]).mean())


def f1_macro_lote(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """F1 macro de cada remuestreo: entradas (B, n, 2), salida (B,)."""
    c = _conteos(y, pred)
    return _f1(c["tp"], c["fp"], c["fn"]).mean(axis=-1)


def metricas_por_etiqueta(y: np.ndarray, pred: np.ndarray) -> dict[str, dict]:
    y, pred = np.asarray(y), np.asarray(pred)
    c = _conteos(y, pred)
    salida = {}
    for j, etiqueta in enumerate(ETIQUETAS):
        tp, fp, fn, tn = (int(c[k][j]) for k in ("tp", "fp", "fn", "tn"))
        salida[etiqueta] = {
            "f1": float(_f1(tp, fp, fn)),
            "sensibilidad": tp / (tp + fn) if tp + fn else 0.0,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "exactitud": (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn else 0.0,
            "matriz_confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "filas_conocidas": tp + fp + fn + tn,
        }
    return salida


def f1_control(y: np.ndarray, pred: np.ndarray) -> float | None:
    """F1 de la clase control (ambas etiquetas negativas).

    Solo se define en filas con las dos etiquetas conocidas; devuelve None si no
    hay ninguna, como en MentalRiskES si cada sujeto se anotó en un subconjunto.
    """
    y, pred = np.asarray(y), np.asarray(pred)
    ambas = (y >= 0).all(axis=1)
    if not ambas.any():
        return None
    control_real = (y[ambas] == 0).all(axis=1)
    control_pred = (pred[ambas] == 0).all(axis=1)
    tp = int((control_real & control_pred).sum())
    fp = int((~control_real & control_pred).sum())
    fn = int((control_real & ~control_pred).sum())
    return float(_f1(tp, fp, fn))

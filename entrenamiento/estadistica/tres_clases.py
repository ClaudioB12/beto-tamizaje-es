"""Métrica principal del short paper (§IV-D): macro-F1 de tres clases.

Definición (la misma en ambos dominios):
- Clase real: depresión, ansiedad o control. En Reddit sale de la comunidad
  (y = 1 en una etiqueta, -1 en la otra); el control tiene 0 y 0. Un caso con
  ambas etiquetas positivas (comorbilidad, solo en el conjunto clínico) queda
  fuera de esta métrica: se cuenta y se evalúa aparte con el F1 multietiqueta.
- Clase predicha: entre las etiquetas que superan su umbral, la de mayor
  probabilidad; control si ninguna lo supera.
- Macro-F1 = media del F1 de las tres clases (F1 = 0 si su denominador es 0).

Bootstrap: se remuestrean SUJETOS con reemplazo (en Reddit un autor aporta
hasta 5 publicaciones; en el conjunto clínico, sujeto = caso). En cada réplica
se calcula el macro-F1 de cada semilla sobre la misma muestra y se promedia
entre semillas. Las comparaciones entre arquitecturas usan las mismas réplicas
(pareadas). Intervalo por percentiles.

Este módulo no sustituye a ``metricas.f1_macro`` (decisión D3, macro-F1
enmascarado): ese sigue siendo el de la Tabla I y el de ``analizar.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from datos.esquema import ETIQUETAS

CLASES = ("ansiedad", "depresion", "control")
DOBLE = -1  # código de un caso con ambas etiquetas reales positivas
DEFINICION = (
    "macro-F1 de tres clases (ansiedad, depresión, control): clase predicha = "
    "etiqueta de mayor probabilidad entre las que superan su umbral, control si "
    "ninguna; casos con doble etiqueta real excluidos"
)


class EtiquetaAmbiguaError(ValueError):
    pass


def clase_real(tabla: pd.DataFrame) -> np.ndarray:
    """0 = ansiedad, 1 = depresión, 2 = control, DOBLE = ambas positivas."""
    y = tabla[[f"y_{e}" for e in ETIQUETAS]].to_numpy()
    ans, dep = y[:, 0], y[:, 1]
    clase = np.full(len(y), -9, dtype=int)
    clase[(ans == 1) & (dep != 1)] = 0
    clase[(dep == 1) & (ans != 1)] = 1
    clase[(ans == 0) & (dep == 0)] = 2
    clase[(ans == 1) & (dep == 1)] = DOBLE
    if (clase == -9).any():
        filas = tabla.loc[clase == -9, "fragmento_id"].head(5).tolist()
        raise EtiquetaAmbiguaError(
            "Filas sin clase única (p. ej., 0 en una etiqueta y -1 en la otra): " + ", ".join(map(str, filas))
        )
    return clase


def clase_predicha(tabla: pd.DataFrame) -> np.ndarray:
    p = tabla[[f"p_{e}" for e in ETIQUETAS]].to_numpy(dtype=float)
    supera = tabla[[f"pred_{e}" for e in ETIQUETAS]].to_numpy() == 1
    candidata = np.where(supera, p, -np.inf)
    clase = np.argmax(candidata, axis=1)
    clase[~supera.any(axis=1)] = 2
    return clase


def _f1(tp, fp, fn):
    den = 2 * tp + fp + fn
    return np.divide(2 * tp, den, out=np.zeros(np.shape(tp), dtype=float), where=den > 0)


def _conteos_por_sujeto(real, pred, sujeto_idx, n_sujetos) -> np.ndarray:
    """(n_sujetos, 3 clases, 3 conteos tp/fp/fn)."""
    conteos = np.zeros((n_sujetos, len(CLASES), 3))
    for k in range(len(CLASES)):
        np.add.at(conteos[:, k, 0], sujeto_idx, (real == k) & (pred == k))
        np.add.at(conteos[:, k, 1], sujeto_idx, (real != k) & (pred == k))
        np.add.at(conteos[:, k, 2], sujeto_idx, (real == k) & (pred != k))
    return conteos


def _macro_desde_conteos(c) -> np.ndarray:
    """c: (..., 3 clases, 3 conteos) -> macro-F1 (...)."""
    return _f1(c[..., 0], c[..., 1], c[..., 2]).mean(axis=-1)


def macro_f1(tabla: pd.DataFrame) -> float:
    real, pred = clase_real(tabla), clase_predicha(tabla)
    validas = real != DOBLE
    c = _conteos_por_sujeto(real[validas], pred[validas], np.zeros(validas.sum(), dtype=int), 1)[0]
    return float(_macro_desde_conteos(c))


class Remuestreo:
    """Réplicas bootstrap por sujeto compartidas por todas las arquitecturas de un dominio."""

    def __init__(self, referencia: pd.DataFrame, n_bootstrap: int, semilla: int):
        real = clase_real(referencia)
        self.validas = real != DOBLE
        self.real = real[self.validas]
        sujetos = referencia["sujeto_id"].astype(str).to_numpy()[self.validas]
        self.ids, self.sujeto_idx = np.unique(sujetos, return_inverse=True)
        self.n_sujetos = len(self.ids)
        rng = np.random.default_rng(semilla)
        # Pesos multinomiales: cuántas veces entra cada sujeto en cada réplica.
        self.pesos = rng.multinomial(self.n_sujetos, np.full(self.n_sujetos, 1 / self.n_sujetos), size=n_bootstrap)
        self.n_bootstrap = n_bootstrap
        self.n_dobles = int((~self.validas).sum())
        self.fragmento_id = referencia["fragmento_id"].to_numpy()

    def distribucion(self, tablas_por_semilla: dict[int, pd.DataFrame]) -> tuple[np.ndarray, list[float]]:
        """Macro-F1 promedio entre semillas en cada réplica, y macro-F1 de cada semilla."""
        por_replica, por_semilla = [], []
        for semilla in sorted(tablas_por_semilla):
            tabla = tablas_por_semilla[semilla]
            if not np.array_equal(tabla["fragmento_id"].to_numpy(), self.fragmento_id):
                raise ValueError(f"La semilla {semilla} no describe los mismos fragmentos en el mismo orden.")
            pred = clase_predicha(tabla)[self.validas]
            c = _conteos_por_sujeto(self.real, pred, self.sujeto_idx, self.n_sujetos)
            por_semilla.append(float(_macro_desde_conteos(c.sum(axis=0))))
            replicas = np.einsum("bs,skc->bkc", self.pesos, c)
            por_replica.append(_macro_desde_conteos(replicas))
        return np.mean(por_replica, axis=0), por_semilla


def intervalo(distribucion: np.ndarray, alfa: float = 0.05) -> tuple[float, float]:
    inf, sup = np.percentile(distribucion, [100 * alfa / 2, 100 * (1 - alfa / 2)])
    return float(inf), float(sup)

"""Intervalos bootstrap del F1 macro (PPI §2.5, contraste principal de H1 y PE2).

Diseño:
- Se remuestrean FRAGMENTOS del conjunto de prueba con reemplazo.
- En cada remuestreo se calcula el F1 macro de las cinco semillas sobre los
  mismos fragmentos y se promedia: el intervalo es de la media entre semillas.
- Para comparar arquitecturas en un dominio se usan los MISMOS índices para
  ambas (pareado por fragmento).
- Para la brecha entre dominios los remuestreos son independientes: los
  fragmentos de MentalRiskES y del conjunto clínico no se emparejan.
- Intervalo por percentiles.
"""
import numpy as np

from estadistica.metricas import f1_macro, f1_macro_lote

N_BOOTSTRAP_PPI = 10_000
_TAMANO_LOTE = 200


def _distribuciones_pareadas(y, grupos_de_predicciones, n_bootstrap, rng) -> list[np.ndarray]:
    """Una distribución por grupo, todas sobre los mismos índices remuestreados."""
    y = np.asarray(y, dtype=np.int8)
    grupos = [np.stack([np.asarray(p, dtype=np.int8) for p in preds]) for preds in grupos_de_predicciones]
    n = len(y)
    for grupo in grupos:
        if grupo.shape[1:] != y.shape:
            raise ValueError(f"Predicciones de forma {grupo.shape[1:]} y etiquetas de forma {y.shape}")

    salidas = [[] for _ in grupos]
    restantes = n_bootstrap
    while restantes:
        b = min(_TAMANO_LOTE, restantes)
        indices = rng.integers(0, n, size=(b, n))
        y_b = y[indices]
        for k, grupo in enumerate(grupos):
            por_semilla = [f1_macro_lote(y_b, semilla[indices]) for semilla in grupo]
            salidas[k].append(np.mean(por_semilla, axis=0))
        restantes -= b
    return [np.concatenate(s) for s in salidas]


def _intervalo(distribucion: np.ndarray, alfa: float) -> tuple[float, float]:
    inferior, superior = np.percentile(distribucion, [100 * alfa / 2, 100 * (1 - alfa / 2)])
    return float(inferior), float(superior)


def _media_semillas(y, preds) -> float:
    return float(np.mean([f1_macro(y, p) for p in preds]))


def ic_media_semillas(y, preds, n_bootstrap=N_BOOTSTRAP_PPI, alfa=0.05, semilla=2026) -> dict:
    rng = np.random.default_rng(semilla)
    (distribucion,) = _distribuciones_pareadas(y, [preds], n_bootstrap, rng)
    inferior, superior = _intervalo(distribucion, alfa)
    return {
        "estimacion": _media_semillas(y, preds),
        "ic_inferior": inferior,
        "ic_superior": superior,
        "nivel": 1 - alfa,
        "n_bootstrap": n_bootstrap,
        "fragmentos": len(y),
        "semillas": len(preds),
    }


def ic_diferencia(y, preds_a, preds_b, n_bootstrap=N_BOOTSTRAP_PPI, alfa=0.05, semilla=2026) -> dict:
    """F1(a) − F1(b) en el mismo conjunto, pareado por fragmento."""
    rng = np.random.default_rng(semilla)
    dist_a, dist_b = _distribuciones_pareadas(y, [preds_a, preds_b], n_bootstrap, rng)
    diferencias = dist_a - dist_b
    inferior, superior = _intervalo(diferencias, alfa)
    return {
        "estimacion": _media_semillas(y, preds_a) - _media_semillas(y, preds_b),
        "ic_inferior": inferior,
        "ic_superior": superior,
        "nivel": 1 - alfa,
        "excluye_cero": inferior > 0 or superior < 0,
        "proporcion_bootstrap_a_mayor": float((diferencias > 0).mean()),
        "n_bootstrap": n_bootstrap,
    }


def ic_brecha_dominio(y_origen, preds_origen, y_destino, preds_destino, n_bootstrap=N_BOOTSTRAP_PPI, alfa=0.05, semilla=2026) -> dict:
    """F1(origen) − F1(destino) con remuestreos independientes (PE2)."""
    rng_origen, rng_destino = np.random.default_rng(semilla).spawn(2)
    (dist_origen,) = _distribuciones_pareadas(y_origen, [preds_origen], n_bootstrap, rng_origen)
    (dist_destino,) = _distribuciones_pareadas(y_destino, [preds_destino], n_bootstrap, rng_destino)
    inferior, superior = _intervalo(dist_origen - dist_destino, alfa)
    return {
        "estimacion": _media_semillas(y_origen, preds_origen) - _media_semillas(y_destino, preds_destino),
        "ic_inferior": inferior,
        "ic_superior": superior,
        "nivel": 1 - alfa,
        "n_bootstrap": n_bootstrap,
    }

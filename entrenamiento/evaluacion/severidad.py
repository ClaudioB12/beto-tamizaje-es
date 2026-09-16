"""Bandas de severidad ancladas a PHQ-9 y GAD-7 (decisión D4, PENDIENTE).

Qué está fijado y qué no:

- Las categorías y su orden son las de cada instrumento
  (``evaluacion.instrumentos``).
- Los CORTES de probabilidad que traducen la salida del clasificador a esas
  categorías son la decisión D4. Este módulo no los inventa: los lee de un
  archivo que el equipo congela, con fecha, antes de abrir el conjunto clínico.

Formato de ``bandas_severidad.json``:

    {
      "congelado_en": "2026-10-01",
      "referencia": "acta o decisión que aprueba los cortes",
      "bandas": {
        "ansiedad":  {"cortes": [c1, c2, c3]},        -> GAD-7: 4 categorías
        "depresion": {"cortes": [c1, c2, c3, c4]}     -> PHQ-9: 5 categorías
      }
    }

Una probabilidad igual o mayor que un corte sube a la categoría siguiente.
"""
import json
import re
from pathlib import Path

import numpy as np

from datos.esquema import ETIQUETAS
from evaluacion.instrumentos import INSTRUMENTO_POR_ETIQUETA, ORDEN_POR_INSTRUMENTO


class BandasInvalidasError(ValueError):
    """El archivo de bandas no cumple el formato o sus cortes no son válidos."""


def cargar_bandas(ruta: Path) -> dict:
    datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
    fecha = datos.get("congelado_en")
    if not isinstance(fecha, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", fecha):
        raise BandasInvalidasError(f"{ruta}: congelado_en debe ser una fecha AAAA-MM-DD")
    if not datos.get("referencia"):
        raise BandasInvalidasError(f"{ruta}: falta referencia (acta o decisión que aprueba los cortes)")

    bandas = datos.get("bandas") or {}
    if set(bandas) != set(ETIQUETAS):
        raise BandasInvalidasError(f"{ruta}: bandas debe tener exactamente {list(ETIQUETAS)}")
    for etiqueta in ETIQUETAS:
        instrumento = INSTRUMENTO_POR_ETIQUETA[etiqueta]
        esperados = len(ORDEN_POR_INSTRUMENTO[instrumento]) - 1
        cortes = bandas[etiqueta].get("cortes")
        if (
            not isinstance(cortes, list)
            or len(cortes) != esperados
            or not all(isinstance(c, (int, float)) and 0 < c < 1 for c in cortes)
            or any(b <= a for a, b in zip(cortes, cortes[1:]))
        ):
            raise BandasInvalidasError(
                f"{ruta}: {etiqueta} ({instrumento}) necesita {esperados} cortes estrictamente "
                f"crecientes en (0, 1); llegó {cortes!r}"
            )
    return datos


def asignar_severidad(probabilidades: np.ndarray, etiqueta: str, bandas: dict) -> np.ndarray:
    orden = ORDEN_POR_INSTRUMENTO[INSTRUMENTO_POR_ETIQUETA[etiqueta]]
    cortes = bandas["bandas"][etiqueta]["cortes"]
    indices = np.searchsorted(cortes, probabilidades, side="right")
    return np.asarray(orden, dtype=object)[indices]

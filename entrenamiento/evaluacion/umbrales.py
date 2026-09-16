"""Umbral de decisión por etiqueta, calibrado SOLO en validación (PPI §2.2)."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from datos.esquema import DESCONOCIDA, ETIQUETAS

# De 0,05 a 0,95 en pasos de 0,01, como declara la guía de entrenamiento.
REJILLA = np.round(np.arange(0.05, 0.9501, 0.01), 2)


def calibrar_umbrales(val: pd.DataFrame, probabilidades: np.ndarray) -> dict[str, dict]:
    """Para cada etiqueta, el umbral de la rejilla que maximiza su F1 en validación.

    Solo cuentan las filas donde la etiqueta es conocida. Ante empate gana el
    umbral más bajo: en tamizaje, a igual F1 se prefiere la mayor sensibilidad.
    """
    if probabilidades.shape != (len(val), len(ETIQUETAS)):
        raise ValueError(
            f"Se esperaban probabilidades de forma {(len(val), len(ETIQUETAS))}; llegó {probabilidades.shape}"
        )
    umbrales = {}
    for j, etiqueta in enumerate(ETIQUETAS):
        conocidas = (val[etiqueta] != DESCONOCIDA).to_numpy()
        y = val.loc[conocidas, etiqueta].to_numpy()
        p = probabilidades[conocidas, j]
        if len(np.unique(y)) < 2:
            raise ValueError(
                f"{etiqueta}: la validación tiene una sola clase entre sus filas conocidas; "
                "el F1 no permite elegir umbral"
            )
        f1s = np.array([f1_score(y, p >= u, zero_division=0) for u in REJILLA])
        mejor = int(np.argmax(f1s))  # argmax devuelve el primero: el umbral más bajo
        umbrales[etiqueta] = {
            "umbral": float(REJILLA[mejor]),
            "f1_validacion": round(float(f1s[mejor]), 6),
            "filas_validacion": int(conocidas.sum()),
            # Un umbral en el borde suele significar que el modelo casi no separa
            # las clases y el F1 se maximiza prediciendo (casi) todo positivo o
            # todo negativo. No es un error, pero hay que mirarlo.
            "en_borde_de_rejilla": mejor in (0, len(REJILLA) - 1),
        }
    return umbrales

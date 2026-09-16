"""Tabla de predicciones por fragmento: el instrumento de VD1 (PPI §2.4)."""
import numpy as np
import pandas as pd

from datos.esquema import ETIQUETAS
from evaluacion.severidad import asignar_severidad


def construir_predicciones(
    df: pd.DataFrame,
    probabilidades: np.ndarray,
    umbrales: dict[str, dict],
    bandas: dict | None,
) -> pd.DataFrame:
    tabla = pd.DataFrame(
        {
            "fragmento_id": df["fragmento_id"].to_numpy(),
            "sujeto_id": df["sujeto_id"].to_numpy(),
            "dominio": df["dominio"].to_numpy(),
        }
    )
    for etiqueta in ETIQUETAS:
        tabla[f"y_{etiqueta}"] = df[etiqueta].to_numpy()
    for j, etiqueta in enumerate(ETIQUETAS):
        tabla[f"p_{etiqueta}"] = np.round(probabilidades[:, j], 6)
    for j, etiqueta in enumerate(ETIQUETAS):
        tabla[f"pred_{etiqueta}"] = (probabilidades[:, j] >= umbrales[etiqueta]["umbral"]).astype(int)
    for j, etiqueta in enumerate(ETIQUETAS):
        # Vacío mientras D4 no fije las bandas.
        tabla[f"sev_{etiqueta}"] = (
            asignar_severidad(probabilidades[:, j], etiqueta, bandas) if bandas else None
        )
    for puntaje in ("phq9", "gad7"):
        tabla[puntaje] = df[puntaje].to_numpy() if puntaje in df.columns else None
    return tabla.sort_values("fragmento_id").reset_index(drop=True)

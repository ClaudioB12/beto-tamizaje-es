"""Deduplicación previa a la partición."""
from dataclasses import asdict, dataclass

import pandas as pd

from datos.esquema import ETIQUETAS


@dataclass(frozen=True)
class InformeDeduplicacion:
    filas_entrada: int
    duplicados_eliminados: int
    textos_en_conflicto: int
    filas_en_conflicto_eliminadas: int
    filas_salida: int

    def como_dict(self) -> dict[str, int]:
        return asdict(self)


def deduplicar(df: pd.DataFrame) -> tuple[pd.DataFrame, InformeDeduplicacion]:
    """Deja una sola fila por ``texto_normalizado``.

    Si el mismo texto aparece con etiquetas distintas no hay forma de saber
    cuál es la correcta, así que se descartan todas sus filas. En MentalRiskES
    esto ocurrirá con mensajes genéricos ("gracias", "jajaja") escritos por
    sujetos de clases distintas. Una etiqueta conocida frente a una
    desconocida (0 frente a -1) también cuenta como conflicto: se prefiere
    perder la fila a inventar la etiqueta.
    """
    etiquetas = list(ETIQUETAS)
    variantes = df.groupby("texto_normalizado", sort=False)[etiquetas].nunique().max(axis=1)
    textos_en_conflicto = variantes[variantes > 1].index
    en_conflicto = df["texto_normalizado"].isin(textos_en_conflicto)

    sin_conflicto = df[~en_conflicto]
    limpio = sin_conflicto.drop_duplicates(subset="texto_normalizado", keep="first")

    informe = InformeDeduplicacion(
        filas_entrada=len(df),
        duplicados_eliminados=len(sin_conflicto) - len(limpio),
        textos_en_conflicto=len(textos_en_conflicto),
        filas_en_conflicto_eliminadas=int(en_conflicto.sum()),
        filas_salida=len(limpio),
    )
    return limpio.reset_index(drop=True), informe

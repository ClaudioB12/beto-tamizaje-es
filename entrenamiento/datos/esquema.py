"""Esquema normalizado que todo cargador de datos debe producir.

La partición, los modelos y la evaluación solo conocen este formato. Así se
desarrollan y prueban hoy con el dataset sintético, y cuando llegue
MentalRiskES basta con escribir su cargador.
"""
import pandas as pd

ETIQUETAS = ("ansiedad", "depresion")

# Una etiqueta desconocida NO es negativa (decisión D2 de la guía): un sujeto
# anotado solo en el subconjunto de depresión no dice nada sobre su ansiedad.
POSITIVA, NEGATIVA, DESCONOCIDA = 1, 0, -1
VALORES_ETIQUETA = {POSITIVA, NEGATIVA, DESCONOCIDA}

DOMINIOS = {"mentalriskes", "clinico", "sintetico"}

COLUMNAS_OBLIGATORIAS = (
    "fragmento_id",
    "sujeto_id",
    "texto",
    "texto_normalizado",
    "dominio",
    *ETIQUETAS,
)

_MAX_EJEMPLOS = 5


class EsquemaInvalidoError(ValueError):
    """El conjunto no cumple el contrato; el mensaje indica qué filas fallan."""


def validar(df: pd.DataFrame, *, etiquetas_por_sujeto: bool) -> None:
    """Falla con un mensaje accionable si ``df`` no cumple el esquema.

    ``etiquetas_por_sujeto`` exige que todas las filas de un sujeto compartan
    etiquetas. Aplica a MentalRiskES (la etiqueta es del usuario) y al
    sintético; no al conjunto clínico, donde los psicólogos etiquetan cada
    fragmento por separado.
    """
    faltantes = [c for c in COLUMNAS_OBLIGATORIAS if c not in df.columns]
    if faltantes:
        raise EsquemaInvalidoError(f"Faltan columnas obligatorias: {faltantes}")
    if df.empty:
        raise EsquemaInvalidoError("El conjunto está vacío")

    _exigir(df, ~df["fragmento_id"].duplicated(keep=False), "fragmento_id repetido")
    for columna in ("fragmento_id", "sujeto_id", "texto", "texto_normalizado"):
        vacia = df[columna].isna() | (df[columna].astype(str).str.strip() == "")
        _exigir(df, ~vacia, f"{columna} vacío")
    _exigir(df, df["dominio"].isin(DOMINIOS), f"dominio fuera de {sorted(DOMINIOS)}")

    for etiqueta in ETIQUETAS:
        if not pd.api.types.is_integer_dtype(df[etiqueta]):
            raise EsquemaInvalidoError(
                f"La columna {etiqueta} debe ser entera (1, 0 o -1) y es "
                f"{df[etiqueta].dtype}. ¿Quedaron nulos sin codificar como -1?"
            )
        _exigir(df, df[etiqueta].isin(VALORES_ETIQUETA), f"{etiqueta} fuera de {{1, 0, -1}}")

    alguna_conocida = (df[list(ETIQUETAS)] != DESCONOCIDA).any(axis=1)
    _exigir(df, alguna_conocida, "ambas etiquetas desconocidas (no aportan a la pérdida)")

    if etiquetas_por_sujeto:
        variantes = df.groupby("sujeto_id")[list(ETIQUETAS)].nunique().max(axis=1)
        inconsistentes = variantes[variantes > 1].index.tolist()
        if inconsistentes:
            raise EsquemaInvalidoError(
                f"{len(inconsistentes)} sujetos con etiquetas distintas entre sus "
                f"filas; ejemplos: {inconsistentes[:_MAX_EJEMPLOS]}"
            )


def _exigir(df: pd.DataFrame, validas: pd.Series, motivo: str) -> None:
    invalidas = df.loc[~validas, "fragmento_id"]
    if len(invalidas):
        raise EsquemaInvalidoError(
            f"{len(invalidas)} filas con {motivo}; ejemplos de fragmento_id: "
            f"{invalidas.head(_MAX_EJEMPLOS).tolist()}"
        )

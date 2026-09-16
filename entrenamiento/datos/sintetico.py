"""Adaptador del dataset sintético al esquema normalizado.

Sirve SOLO para probar el pipeline de punta a punta. Ninguna métrica obtenida
sobre estos datos debe reportarse.

Por qué ``sujeto_id`` es la frase base: ``generar_dataset.py`` escribe cada
frase base cinco veces (nueve en control) con un prefijo y un sufijo al azar
("Doctor, " + frase + " todo el tiempo."). Las variantes son casi idénticas.
Con el split aleatorio por fila de ``train.py``, el 100 % de la validación
tenía su frase base en entrenamiento. Agrupar por frase base obliga a que
todas sus variantes caigan en la misma partición, igual que agrupar
MentalRiskES por usuario.
"""
from pathlib import Path

import pandas as pd

import generar_dataset as generador
from datos.esquema import validar
from datos.normalizacion import normalizar_texto

RUTA_POR_DEFECTO = Path(__file__).resolve().parents[1] / "dataset_clasificacion.csv"

# El sintético es de etiqueta única por construcción: quien etiqueta "ansiedad"
# está afirmando que no es depresión, así que la otra etiqueta es 0, no -1.
_BINARIAS_POR_EMOCION = {
    "ansiedad": (1, 0),
    "depresion": (0, 1),
    "control": (0, 0),
}


def cargar_clasificacion(ruta: Path | str = RUTA_POR_DEFECTO) -> pd.DataFrame:
    crudo = pd.read_csv(ruta, encoding="utf-8-sig")

    emociones_desconocidas = set(crudo["emocion"]) - set(_BINARIAS_POR_EMOCION)
    if emociones_desconocidas:
        raise ValueError(f"Emociones no reconocidas en {ruta}: {sorted(emociones_desconocidas)}")

    bases = _frases_base()
    normalizados = crudo["texto"].map(normalizar_texto)
    binarias = crudo["emocion"].map(_BINARIAS_POR_EMOCION)

    df = pd.DataFrame(
        {
            "fragmento_id": [f"sint-{i:04d}" for i in range(len(crudo))],
            "sujeto_id": normalizados.map(lambda t: _grupo_de(t, bases)),
            "texto": crudo["texto"],
            "texto_normalizado": normalizados,
            "dominio": "sintetico",
            "ansiedad": binarias.map(lambda par: par[0]).astype("int64"),
            "depresion": binarias.map(lambda par: par[1]).astype("int64"),
            "etiqueta_original": crudo["etiqueta_unificada"],
        }
    )
    validar(df, etiquetas_por_sujeto=True)
    return df


def _frases_base() -> dict[str, str]:
    """Frase base normalizada -> identificador estable de su grupo."""
    bases: dict[str, str] = {}
    for emocion, por_severidad in (
        ("ansiedad", generador.FRASES_ANSIEDAD),
        ("depresion", generador.FRASES_DEPRESION),
    ):
        for severidad, frases in por_severidad.items():
            for i, frase in enumerate(frases):
                bases[normalizar_texto(frase)] = f"base-{emocion}-{severidad}-{i:02d}"
    for i, frase in enumerate(generador.FRASES_CONTROL):
        bases[normalizar_texto(frase)] = f"base-control-{i:02d}"
    return bases


def _grupo_de(texto_normalizado: str, bases: dict[str, str]) -> str:
    coincidencias = [grupo for base, grupo in bases.items() if base in texto_normalizado]
    if len(coincidencias) != 1:
        raise ValueError(
            f"Se esperaba exactamente una frase base para {texto_normalizado!r} y hay "
            f"{len(coincidencias)}. ¿Se regeneró el CSV con otras frases?"
        )
    return coincidencias[0]

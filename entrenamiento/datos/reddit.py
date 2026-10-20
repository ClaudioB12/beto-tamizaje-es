"""Adaptador del Reddit Mental Health Dataset (Low et al., 2020) al esquema.

Sustituye a MentalRiskES como fuente del dominio "redes sociales" del PPI
(informe de revisión del dataset, 14/10/2026). El corpus es abierto
(Zenodo 10.17605/OSF.IO/7PEYQ, licencia PDDL 1.0; su reutilización está
sujeta a los términos de la API de Reddit) y está en inglés: este cargador
lee el CSV ya traducido por ``traducir_reddit.py``.

Flujo de datos (desde entrenamiento/):
    preparar_reddit.py  -> datos_reddit/reddit_en.csv   (inglés, anonimizado)
    traducir_reddit.py  -> datos_reddit/reddit_es.csv   (añade texto_es)
    particionar.py --fuente reddit                      (congela la partición)

Etiquetas (decisión D2): la etiqueta viene de la comunidad donde se publicó.
Publicar en r/depression afirma depresión pero no dice nada sobre ansiedad,
así que esa otra etiqueta es desconocida (-1), no negativa. Los subreddits de
control (r/relationships, r/parenting) no son comunidades de salud mental y
se codifican como negativos en ambas.
"""
import re
from pathlib import Path

import pandas as pd

from datos.esquema import DESCONOCIDA, NEGATIVA, POSITIVA, validar
from datos.normalizacion import normalizar_texto

RAIZ = Path(__file__).resolve().parents[1]
RUTA_POR_DEFECTO = RAIZ / "datos_reddit" / "reddit_es.csv"

DOMINIO = "reddit"

# etiqueta de preparación -> (ansiedad, depresion)
BINARIAS_POR_ETIQUETA = {
    "ansiedad": (POSITIVA, DESCONOCIDA),
    "depresion": (DESCONOCIDA, POSITIVA),
    "control": (NEGATIVA, NEGATIVA),
}

COLUMNAS_TRADUCIDAS = ("fragmento_id", "sujeto_id", "subreddit", "fecha", "etiqueta", "texto_en", "texto_es")

# Términos que nombran la etiqueta en el texto traducido (fuga léxica). La
# variante enmascarada permite la ablación con y sin ellos.
_TERMINOS_ETIQUETA = re.compile(
    r"\b(depresi\w*|deprimid\w*|ansiedad\w*|ansios\w*|p[aá]nico\w*|r/\w+)",
    flags=re.IGNORECASE,
)
MASCARA = "[MASK]"

# Unas pocas publicaciones de los subreddits están en otro idioma (p. ej. chino).
# Traducirlas EN→ES produce basura, así que el cargador las descarta.
UMBRAL_LETRAS_LATINAS = 0.9


def sin_letras(texto: str) -> bool:
    """Traducción degenerada: el modelo devolvió solo símbolos (p. ej. "¡¡¡¡" o "• • •")."""
    return not any(c.isalpha() for c in texto)


def es_origen_no_ingles(texto: str) -> bool:
    letras = [c for c in texto if c.isalpha()]
    return bool(letras) and sum(c.isascii() for c in letras) / len(letras) < UMBRAL_LETRAS_LATINAS


def cargar_clasificacion(ruta: Path | str = RUTA_POR_DEFECTO) -> pd.DataFrame:
    return _cargar(ruta, enmascarar=False)


def cargar_clasificacion_enmascarada(ruta: Path | str = RUTA_POR_DEFECTO) -> pd.DataFrame:
    """Igual que ``cargar_clasificacion`` pero sin los términos de la etiqueta."""
    return _cargar(ruta, enmascarar=True)


def enmascarar(texto: str) -> str:
    return _TERMINOS_ETIQUETA.sub(MASCARA, texto)


def _cargar(ruta: Path | str, *, enmascarar: bool) -> pd.DataFrame:
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecuta antes preparar_reddit.py y traducir_reddit.py."
        )
    crudo = pd.read_csv(ruta, dtype=str, keep_default_na=False, encoding="utf-8")

    faltantes = [c for c in COLUMNAS_TRADUCIDAS if c not in crudo.columns]
    if faltantes:
        raise ValueError(f"{ruta.name} no tiene las columnas {faltantes}. ¿Terminó la traducción?")
    descartar = crudo["texto_en"].map(es_origen_no_ingles) | (
        (crudo["texto_es"].str.strip() != "") & crudo["texto_es"].map(sin_letras)
    )
    crudo = crudo[~descartar].reset_index(drop=True)
    sin_traducir = crudo["texto_es"].str.strip() == ""
    if sin_traducir.any():
        raise ValueError(
            f"{int(sin_traducir.sum())} filas sin texto_es en {ruta.name}. "
            "Vuelve a lanzar traducir_reddit.py: retoma donde quedó."
        )
    desconocidas = set(crudo["etiqueta"]) - set(BINARIAS_POR_ETIQUETA)
    if desconocidas:
        raise ValueError(f"Etiquetas no reconocidas en {ruta.name}: {sorted(desconocidas)}")

    texto = crudo["texto_es"].map(_enmascarar_si(enmascarar))
    binarias = crudo["etiqueta"].map(BINARIAS_POR_ETIQUETA)

    df = pd.DataFrame(
        {
            "fragmento_id": crudo["fragmento_id"],
            "sujeto_id": crudo["sujeto_id"],
            "texto": texto,
            "texto_normalizado": texto.map(normalizar_texto),
            "dominio": DOMINIO,
            "ansiedad": binarias.map(lambda par: par[0]).astype("int64"),
            "depresion": binarias.map(lambda par: par[1]).astype("int64"),
            "etiqueta_original": crudo["etiqueta"],
            "subreddit": crudo["subreddit"],
            "fecha": crudo["fecha"],
        }
    )
    # La etiqueta es de la comunidad, y preparar_reddit.py ya excluyó a los
    # autores que publicaron en más de una clase: todas las filas de un sujeto
    # comparten etiqueta.
    validar(df, etiquetas_por_sujeto=True)
    return df


def _enmascarar_si(activo: bool):
    return enmascarar if activo else (lambda t: t)

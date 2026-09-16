"""Clave de comparación de textos.

``texto_normalizado`` solo sirve para detectar duplicados y fugas entre
particiones; nunca es la entrada del modelo, que recibe ``texto`` tal cual.
"""
import re
import unicodedata

_NO_ALFANUMERICO = re.compile(r"[^0-9a-z ]+")


def normalizar_texto(texto: str) -> str:
    """Minúsculas, sin tildes ni signos, espacios colapsados.

    "¡Hola,   Doctór!" y "hola doctor" deben contar como el mismo texto: una
    diferencia de puntuación no impide que el modelo memorice la frase.
    """
    descompuesto = unicodedata.normalize("NFD", texto.lower())
    sin_tildes = "".join(c for c in descompuesto if unicodedata.category(c) != "Mn")
    return " ".join(_NO_ALFANUMERICO.sub(" ", sin_tildes).split())

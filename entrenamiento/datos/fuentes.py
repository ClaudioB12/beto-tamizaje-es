"""Registro de fuentes de datos: cómo se carga cada una y de qué archivo sale.

Lo comparten particionar, entrenar y evaluar. El conjunto clínico no: nunca
se particiona.

"reddit" sustituye a MentalRiskES como dominio de redes sociales (acceso
abierto, traducido al español). "reddit_enmascarado" lee el mismo archivo sin
los términos que nombran la etiqueta: sirve para la ablación de fuga léxica.
"""
from datos import reddit, sintetico

FUENTES = {
    "sintetico": (sintetico.cargar_clasificacion, sintetico.RUTA_POR_DEFECTO),
    "reddit": (reddit.cargar_clasificacion, reddit.RUTA_POR_DEFECTO),
    "reddit_enmascarado": (reddit.cargar_clasificacion_enmascarada, reddit.RUTA_POR_DEFECTO),
}

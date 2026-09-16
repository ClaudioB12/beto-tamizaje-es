"""Registro de fuentes de datos: cómo se carga cada una y de qué archivo sale.

Lo comparten particionar, entrenar y evaluar. MentalRiskES se añade aquí
cuando exista su cargador. El conjunto clínico no: nunca se particiona.
"""
from datos import sintetico

FUENTES = {
    "sintetico": (sintetico.cargar_clasificacion, sintetico.RUTA_POR_DEFECTO),
}

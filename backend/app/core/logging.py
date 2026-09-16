"""Configuración de logging con regla de anonimización.

Política: los logs del backend registran identificadores y metadatos, nunca
texto libre del paciente. Si excepcionalmente hay que loguear contenido
textual, debe pasar antes por ``texto_seguro_para_log``.
"""
import logging

FORMATO_LOG = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def configurar_logging(nivel: int = logging.INFO) -> None:
    logging.basicConfig(level=nivel, format=FORMATO_LOG)


def obtener_logger(nombre: str) -> logging.Logger:
    return logging.getLogger(f"agente_beto.{nombre}")


def texto_seguro_para_log(texto: str, max_largo: int = 60) -> str:
    """Des-identifica y trunca un texto antes de incluirlo en un log."""
    from app.services.desidentificacion import desidentificar

    seguro = desidentificar(texto)
    if len(seguro) > max_largo:
        return seguro[:max_largo] + "…"
    return seguro

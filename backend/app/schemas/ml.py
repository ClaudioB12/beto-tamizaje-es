"""Contrato HTTP con el servicio ML (BETO + recuperación semántica).

Estos esquemas validan las respuestas del servicio ML: si el servicio
devuelve algo fuera de contrato, el backend falla con 502 explícito en vez
de propagar datos malformados.
"""
from typing import Literal

from pydantic import BaseModel

Emocion = Literal["ansiedad", "depresion", "control"]
Severidad = Literal["leve", "moderado", "severo"]


class CrisisMLResponse(BaseModel):
    riesgo: bool
    motivo: str | None = None


class ClasificacionMLResponse(BaseModel):
    emocion: Emocion
    severidad: Severidad


class FragmentoML(BaseModel):
    id: str
    texto: str
    categoria: str


class RecuperacionMLResponse(BaseModel):
    fragmentos: list[FragmentoML]

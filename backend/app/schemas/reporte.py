from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class EventoCrisis(BaseModel):
    mensaje_id: str
    sesion_id: str
    fecha: datetime
    motivo: str | None = None


class ReporteContenido(BaseModel):
    """Estructura versionada del reporte clínico de Capa 2.

    Todo el contenido se calcula por agregación determinista sobre las
    clasificaciones ya persistidas; el resumen es una frase parametrizada,
    nunca generación libre de un modelo.
    """

    rango_inicio: datetime | None = None
    rango_fin: datetime | None = None
    total_sesiones: int
    total_mensajes_paciente: int
    distribucion_emociones: dict[str, int]
    distribucion_severidad: dict[str, int]
    eventos_crisis: list[EventoCrisis]
    tendencia_severidad: Literal["mejora", "estable", "empeoramiento", "sin_datos"]
    resumen_automatico: str
    advertencias: list[str]


class ReporteResponse(BaseModel):
    id: str
    paciente_id: str
    version: int
    generado_por_id: str
    creado_en: datetime
    contenido: ReporteContenido

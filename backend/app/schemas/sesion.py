from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.mensaje import RemitenteMensaje


class MensajeHistorial(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    remitente: RemitenteMensaje
    texto: str
    crisis_detectada: bool
    creado_en: datetime


class HistorialSesionResponse(BaseModel):
    sesion_id: str
    paciente_id: str
    estado: str
    iniciada_en: datetime
    mensajes: list[MensajeHistorial]

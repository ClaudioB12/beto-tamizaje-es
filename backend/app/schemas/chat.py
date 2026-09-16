from pydantic import BaseModel, Field


class MensajeChatRequest(BaseModel):
    texto: str = Field(min_length=1, max_length=2000)
    # Si es None se abre una sesión de conversación nueva.
    sesion_id: str | None = None


class MensajeChatResponse(BaseModel):
    """Respuesta de Capa 1 hacia el paciente.

    Deliberadamente NO expone emoción ni severidad: la clasificación es
    información clínica reservada a la Capa 2 (reporte para el psicólogo).
    """

    sesion_id: str
    mensaje_id: str
    respuesta: str
    crisis_detectada: bool
    recursos_ayuda: list[str] = []

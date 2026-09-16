from fastapi import APIRouter, Depends

from app.auth.dependencies import require_paciente
from app.models import Usuario
from app.schemas.chat import MensajeChatRequest, MensajeChatResponse
from app.services.orquestador import Orquestador, get_orquestador

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/mensaje", response_model=MensajeChatResponse)
async def enviar_mensaje(
    datos: MensajeChatRequest,
    paciente: Usuario = Depends(require_paciente),
    orquestador: Orquestador = Depends(get_orquestador),
) -> MensajeChatResponse:
    """Flujo completo de Capa 1: gate de crisis -> clasificación ->
    recuperación -> instanciación de plantilla -> respuesta."""
    return await orquestador.procesar_mensaje(paciente, datos)

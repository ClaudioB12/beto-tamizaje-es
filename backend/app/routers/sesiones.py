from fastapi import APIRouter, Depends

from app.auth.dependencies import get_current_user
from app.models import Usuario
from app.schemas.sesion import HistorialSesionResponse, MensajeHistorial
from app.services.orquestador import Orquestador, get_orquestador

router = APIRouter(prefix="/sesiones", tags=["sesiones"])


@router.get("/{sesion_id}/historial", response_model=HistorialSesionResponse)
def obtener_historial(
    sesion_id: str,
    usuario: Usuario = Depends(get_current_user),
    orquestador: Orquestador = Depends(get_orquestador),
) -> HistorialSesionResponse:
    """Historial de una sesión. Accesible por el paciente dueño de la sesión
    o por un psicólogo con vínculo activo con ese paciente."""
    sesion = orquestador.obtener_historial(sesion_id, usuario)
    return HistorialSesionResponse(
        sesion_id=sesion.id,
        paciente_id=sesion.paciente_id,
        estado=sesion.estado.value,
        iniciada_en=sesion.iniciada_en,
        mensajes=[MensajeHistorial.model_validate(m) for m in sesion.mensajes],
    )

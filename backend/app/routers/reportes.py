from fastapi import APIRouter, Depends

from app.auth.dependencies import require_psicologo
from app.models import Usuario
from app.schemas.reporte import ReporteContenido, ReporteResponse
from app.services.orquestador import Orquestador, get_orquestador

router = APIRouter(prefix="/reportes", tags=["reportes"])


@router.get("/{paciente_id}", response_model=ReporteResponse)
def obtener_reporte(
    paciente_id: str,
    regenerar: bool = False,
    psicologo: Usuario = Depends(require_psicologo),
    orquestador: Orquestador = Depends(get_orquestador),
) -> ReporteResponse:
    """Capa 2: devuelve el último reporte del paciente (lo genera si no
    existe). Con ?regenerar=true crea una versión nueva sin borrar las
    anteriores. Solo accesible por el psicólogo con vínculo activo."""
    reporte = orquestador.obtener_o_generar_reporte(paciente_id, psicologo, regenerar)
    return ReporteResponse(
        id=reporte.id,
        paciente_id=reporte.paciente_id,
        version=reporte.version,
        generado_por_id=reporte.generado_por_id,
        creado_en=reporte.creado_en,
        contenido=ReporteContenido.model_validate(reporte.contenido),
    )

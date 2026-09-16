"""Cliente HTTP hacia el servicio ML (contenedor local, sin APIs externas).

Toda falla de red o de contrato se convierte en una excepción de dominio
explícita: el backend nunca degrada en silencio si el servicio ML no está
disponible (503) o responde fuera de contrato (502).
"""
from typing import Any

import httpx
from pydantic import ValidationError

from app.config import get_settings
from app.core.exceptions import RespuestaMLInvalidaError, ServicioMLNoDisponibleError
from app.schemas.ml import (
    ClasificacionMLResponse,
    CrisisMLResponse,
    RecuperacionMLResponse,
)


class MLClient:
    def __init__(self, base_url: str, timeout_segundos: float):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_segundos

    async def _post(self, ruta: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url, timeout=self._timeout
            ) as cliente:
                respuesta = await cliente.post(ruta, json=payload)
                respuesta.raise_for_status()
                return respuesta.json()
        except httpx.HTTPStatusError as exc:
            raise RespuestaMLInvalidaError(
                f"El servicio ML respondió {exc.response.status_code} en {ruta}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise ServicioMLNoDisponibleError(
                f"Timeout de {self._timeout}s al llamar {ruta}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ServicioMLNoDisponibleError(
                f"No se pudo conectar con el servicio ML en {ruta}"
            ) from exc
        except ValueError as exc:
            raise RespuestaMLInvalidaError(
                f"El servicio ML devolvió un cuerpo no-JSON en {ruta}"
            ) from exc

    async def detectar_crisis(self, texto: str, paciente_id: str) -> CrisisMLResponse:
        datos = await self._post(
            "/ml/detectar-crisis", {"texto": texto, "paciente_id": paciente_id}
        )
        try:
            return CrisisMLResponse.model_validate(datos)
        except ValidationError as exc:
            raise RespuestaMLInvalidaError(
                "Respuesta fuera de contrato en /ml/detectar-crisis"
            ) from exc

    async def clasificar(self, texto: str) -> ClasificacionMLResponse:
        datos = await self._post("/ml/clasificar", {"texto": texto})
        try:
            return ClasificacionMLResponse.model_validate(datos)
        except ValidationError as exc:
            raise RespuestaMLInvalidaError(
                "Respuesta fuera de contrato en /ml/clasificar"
            ) from exc

    async def recuperar(self, emocion: str, severidad: str) -> RecuperacionMLResponse:
        datos = await self._post(
            "/ml/recuperar", {"emocion": emocion, "severidad": severidad}
        )
        try:
            return RecuperacionMLResponse.model_validate(datos)
        except ValidationError as exc:
            raise RespuestaMLInvalidaError(
                "Respuesta fuera de contrato en /ml/recuperar"
            ) from exc


def get_ml_client() -> MLClient:
    """Dependencia de FastAPI; en tests se sustituye por un doble de prueba."""
    settings = get_settings()
    return MLClient(
        base_url=settings.ml_service_url,
        timeout_segundos=settings.ml_timeout_segundos,
    )

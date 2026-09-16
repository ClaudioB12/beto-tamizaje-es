"""Excepciones de dominio y manejadores globales de FastAPI.

Regla de privacidad: los mensajes de estas excepciones NUNCA deben incluir
texto libre del paciente; solo identificadores y metadatos.
"""
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("agente_beto.errores")


class AppError(Exception):
    status_code: int = 500
    codigo: str = "error_interno"

    def __init__(self, mensaje: str = "Error interno del servidor"):
        self.mensaje = mensaje
        super().__init__(mensaje)


class CredencialesInvalidasError(AppError):
    status_code = 401
    codigo = "credenciales_invalidas"


class AccesoDenegadoError(AppError):
    status_code = 403
    codigo = "acceso_denegado"


class RecursoNoEncontradoError(AppError):
    status_code = 404
    codigo = "recurso_no_encontrado"


class ConflictoError(AppError):
    status_code = 409
    codigo = "conflicto"


class RespuestaMLInvalidaError(AppError):
    status_code = 502
    codigo = "respuesta_ml_invalida"


class ServicioMLNoDisponibleError(AppError):
    status_code = 503
    codigo = "servicio_ml_no_disponible"


class PlantillaNoDisponibleError(AppError):
    status_code = 500
    codigo = "plantilla_no_disponible"


def registrar_manejadores(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def manejar_app_error(request: Request, exc: AppError) -> JSONResponse:
        logger.warning(
            "%s (%s) en %s %s: %s",
            exc.codigo,
            exc.status_code,
            request.method,
            request.url.path,
            exc.mensaje,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"codigo": exc.codigo, "mensaje": exc.mensaje},
        )

    @app.exception_handler(Exception)
    async def manejar_error_no_controlado(request: Request, exc: Exception) -> JSONResponse:
        # Se loguea solo el tipo de excepción y la ruta: el detalle podría
        # contener texto del paciente y no debe llegar al log ni al cliente.
        logger.error(
            "Error no controlado %s en %s %s",
            type(exc).__name__,
            request.method,
            request.url.path,
        )
        return JSONResponse(
            status_code=500,
            content={"codigo": "error_interno", "mensaje": "Error interno del servidor"},
        )

"""Dependencias de FastAPI para autenticación y autorización por rol."""
from collections.abc import Callable

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.auth.jwt import decodificar_token
from app.core.exceptions import AccesoDenegadoError, CredencialesInvalidasError
from app.db.session import get_db
from app.models import RolUsuario, Usuario

# HTTPBearer: en /docs el botón "Authorize" pide pegar el token emitido por
# POST /auth/login. auto_error=False para responder 401 de dominio (no 403).
bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credenciales: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> Usuario:
    if credenciales is None:
        raise CredencialesInvalidasError("Falta el token de acceso")
    payload = decodificar_token(credenciales.credentials)
    usuario_id = payload.get("sub")
    if not usuario_id:
        raise CredencialesInvalidasError("Token sin identidad")
    usuario = db.get(Usuario, usuario_id)
    if usuario is None or not usuario.activo:
        raise CredencialesInvalidasError("Usuario no encontrado o inactivo")
    return usuario


def require_rol(rol: RolUsuario) -> Callable[..., Usuario]:
    def verificador(usuario: Usuario = Depends(get_current_user)) -> Usuario:
        if usuario.rol != rol:
            raise AccesoDenegadoError(f"Este recurso requiere rol '{rol.value}'")
        return usuario

    return verificador


require_paciente = require_rol(RolUsuario.PACIENTE)
require_psicologo = require_rol(RolUsuario.PSICOLOGO)

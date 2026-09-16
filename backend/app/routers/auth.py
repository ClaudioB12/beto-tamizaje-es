from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.jwt import crear_token, verificar_password
from app.config import get_settings
from app.core.exceptions import CredencialesInvalidasError
from app.db.session import get_db
from app.models import Usuario
from app.schemas.auth import LoginRequest, TokenResponse

router = APIRouter(prefix="/auth", tags=["autenticacion"])


@router.post("/login", response_model=TokenResponse)
def login(datos: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    usuario = db.scalar(select(Usuario).where(Usuario.email == datos.email))
    if (
        usuario is None
        or not usuario.activo
        or not verificar_password(datos.password, usuario.hashed_password)
    ):
        # Mensaje único para email inexistente y contraseña incorrecta:
        # no se revela cuál de los dos falló.
        raise CredencialesInvalidasError("Email o contraseña incorrectos")
    token = crear_token(usuario.id, usuario.rol.value)
    return TokenResponse(
        access_token=token,
        expira_en_minutos=get_settings().jwt_expiracion_minutos,
    )

"""Creación/validación de JWT y hashing de contraseñas (bcrypt)."""
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt

from app.config import get_settings
from app.core.exceptions import CredencialesInvalidasError


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verificar_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        # Hash malformado en base de datos: se trata como credencial inválida.
        return False


def crear_token(usuario_id: str, rol: str) -> str:
    settings = get_settings()
    ahora = datetime.now(timezone.utc)
    payload = {
        "sub": usuario_id,
        "rol": rol,
        "iat": ahora,
        "exp": ahora + timedelta(minutes=settings.jwt_expiracion_minutos),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algoritmo)


def decodificar_token(token: str) -> dict[str, Any]:
    settings = get_settings()
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algoritmo])
    except jwt.ExpiredSignatureError as exc:
        raise CredencialesInvalidasError("El token ha expirado") from exc
    except jwt.InvalidTokenError as exc:
        raise CredencialesInvalidasError("Token inválido") from exc

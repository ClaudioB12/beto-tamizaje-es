"""Configuración central del backend.

Todas las variables se leen del entorno con prefijo ``BETO_`` (o de un
archivo ``.env`` local). Los valores por defecto son válidos SOLO para el
prototipo local; en despliegue real ``jwt_secret`` debe inyectarse por
entorno y ``database_url`` apuntar a PostgreSQL.
"""
from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Valor centinela: útil en local, inaceptable en despliegue. Se compara por
# identidad exacta en la validación de abajo, así que no debe duplicarse.
JWT_SECRET_PROTOTIPO = "solo-para-prototipo-local-reemplazar-en-produccion"
LONGITUD_MINIMA_SECRETO = 32
ENTORNOS_PRODUCTIVOS = {"produccion", "production", "prod"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BETO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_nombre: str = "Agente BETO - Backend"
    entorno: str = "desarrollo"

    # SQLite para el prototipo; migración a PostgreSQL solo cambia esta URL.
    database_url: str = "sqlite:///./prototipo.db"

    jwt_secret: str = JWT_SECRET_PROTOTIPO
    jwt_algoritmo: str = "HS256"
    jwt_expiracion_minutos: int = 60

    ml_service_url: str = "http://ml-service:8001"
    ml_timeout_segundos: float = 10.0

    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    @model_validator(mode="after")
    def _rechazar_secreto_debil_en_produccion(self) -> "Settings":
        """Falla al arrancar, no en la primera petición.

        Sin esta guardia un despliegue que olvide inyectar BETO_JWT_SECRET
        arranca en silencio con el secreto de desarrollo, que es público: los
        tokens de cualquiera serían falsificables.
        """
        if self.entorno.strip().lower() not in ENTORNOS_PRODUCTIVOS:
            return self
        if self.jwt_secret == JWT_SECRET_PROTOTIPO:
            raise ValueError(
                "BETO_JWT_SECRET conserva el valor de prototipo con "
                f"BETO_ENTORNO={self.entorno!r}. Inyecta un secreto real."
            )
        if len(self.jwt_secret) < LONGITUD_MINIMA_SECRETO:
            raise ValueError(
                f"BETO_JWT_SECRET debe tener al menos {LONGITUD_MINIMA_SECRETO} "
                f"caracteres en producción (tiene {len(self.jwt_secret)})."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()

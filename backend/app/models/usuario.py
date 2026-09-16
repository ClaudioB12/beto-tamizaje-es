import enum
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.comun import ahora_utc, generar_uuid


class RolUsuario(str, enum.Enum):
    PACIENTE = "paciente"
    PSICOLOGO = "psicologo"


class Usuario(Base):
    __tablename__ = "usuarios"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generar_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    nombre_completo: Mapped[str] = mapped_column(String(255), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    # native_enum=False genera VARCHAR + CHECK: idéntico en SQLite y PostgreSQL,
    # sin tipos ENUM nativos que compliquen las migraciones Alembic.
    rol: Mapped[RolUsuario] = mapped_column(
        Enum(RolUsuario, native_enum=False, length=20, values_callable=lambda e: [m.value for m in e]),
        nullable=False,
        index=True,
    )
    activo: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    creado_en: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=ahora_utc, nullable=False
    )

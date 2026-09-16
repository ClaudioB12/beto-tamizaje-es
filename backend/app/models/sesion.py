import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.comun import ahora_utc, generar_uuid

if TYPE_CHECKING:
    from app.models.mensaje import Mensaje


class EstadoSesion(str, enum.Enum):
    ACTIVA = "activa"
    CERRADA = "cerrada"
    ESCALADA_CRISIS = "escalada_crisis"


class SesionConversacion(Base):
    __tablename__ = "sesiones_conversacion"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generar_uuid)
    paciente_id: Mapped[str] = mapped_column(
        ForeignKey("usuarios.id"), index=True, nullable=False
    )
    estado: Mapped[EstadoSesion] = mapped_column(
        Enum(EstadoSesion, native_enum=False, length=20, values_callable=lambda e: [m.value for m in e]),
        default=EstadoSesion.ACTIVA,
        nullable=False,
    )
    iniciada_en: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=ahora_utc, nullable=False
    )
    cerrada_en: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    mensajes: Mapped[list["Mensaje"]] = relationship(
        back_populates="sesion", order_by="Mensaje.creado_en"
    )

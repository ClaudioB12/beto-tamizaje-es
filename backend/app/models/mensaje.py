import enum
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.comun import ahora_utc, generar_uuid
from app.models.sesion import SesionConversacion


class RemitenteMensaje(str, enum.Enum):
    PACIENTE = "paciente"
    AGENTE = "agente"


class Mensaje(Base):
    __tablename__ = "mensajes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generar_uuid)
    sesion_id: Mapped[str] = mapped_column(
        ForeignKey("sesiones_conversacion.id"), index=True, nullable=False
    )
    remitente: Mapped[RemitenteMensaje] = mapped_column(
        Enum(RemitenteMensaje, native_enum=False, length=20, values_callable=lambda e: [m.value for m in e]),
        nullable=False,
    )
    # INVARIANTE DE PRIVACIDAD: este campo se persiste SIEMPRE des-identificado
    # (ver services/desidentificacion.py). Nunca guardar texto crudo aquí.
    texto: Mapped[str] = mapped_column(Text, nullable=False)
    # Clasificación del servicio ML (solo mensajes del paciente en flujo normal).
    emocion: Mapped[str | None] = mapped_column(String(30), nullable=True)
    severidad: Mapped[str | None] = mapped_column(String(30), nullable=True)
    crisis_detectada: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    crisis_motivo: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # Solo mensajes del agente: plantilla del banco versionado que se instanció.
    plantilla_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    creado_en: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=ahora_utc, nullable=False, index=True
    )

    sesion: Mapped[SesionConversacion] = relationship(back_populates="mensajes")

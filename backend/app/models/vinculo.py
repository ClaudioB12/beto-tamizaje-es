from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.comun import ahora_utc, generar_uuid
from app.models.usuario import Usuario


class VinculoPacientePsicologo(Base):
    """Vínculo clínico paciente-psicólogo.

    Es una entidad propia (no una M2M plana) porque tiene ciclo de vida:
    puede desactivarse sin borrarse (trazabilidad) y es la base de
    autorización de la Capa 2: un psicólogo solo accede a reportes de
    pacientes con vínculo ACTIVO.
    """

    __tablename__ = "vinculos_paciente_psicologo"
    __table_args__ = (
        UniqueConstraint("paciente_id", "psicologo_id", name="uq_vinculo_paciente_psicologo"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generar_uuid)
    paciente_id: Mapped[str] = mapped_column(
        ForeignKey("usuarios.id"), index=True, nullable=False
    )
    psicologo_id: Mapped[str] = mapped_column(
        ForeignKey("usuarios.id"), index=True, nullable=False
    )
    activo: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    creado_en: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=ahora_utc, nullable=False
    )

    paciente: Mapped[Usuario] = relationship("Usuario", foreign_keys=[paciente_id])
    psicologo: Mapped[Usuario] = relationship("Usuario", foreign_keys=[psicologo_id])

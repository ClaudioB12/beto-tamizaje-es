from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.comun import ahora_utc, generar_uuid

# JSON estándar en SQLite; JSONB automáticamente al migrar a PostgreSQL.
TipoJSON = JSON().with_variant(JSONB(), "postgresql")


class ReporteClinico(Base):
    """Reporte de Capa 2, versionado por filas inmutables.

    Cada regeneración inserta una fila nueva con version = max + 1; nunca se
    sobreescribe un reporte ya generado (trazabilidad clínica). El contenido
    es JSON validado por el esquema Pydantic ReporteContenido.
    """

    __tablename__ = "reportes_clinicos"
    __table_args__ = (
        UniqueConstraint("paciente_id", "version", name="uq_reporte_paciente_version"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generar_uuid)
    paciente_id: Mapped[str] = mapped_column(
        ForeignKey("usuarios.id"), index=True, nullable=False
    )
    generado_por_id: Mapped[str] = mapped_column(ForeignKey("usuarios.id"), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    contenido: Mapped[dict] = mapped_column(TipoJSON, nullable=False)
    creado_en: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=ahora_utc, nullable=False
    )

"""Utilidades compartidas por los modelos."""
import uuid
from datetime import datetime, timezone


def generar_uuid() -> str:
    """UUID v4 como texto: portable entre SQLite y PostgreSQL."""
    return str(uuid.uuid4())


def ahora_utc() -> datetime:
    return datetime.now(timezone.utc)

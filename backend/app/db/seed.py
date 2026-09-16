"""Datos de demostración para el prototipo.

Uso (desde backend/):  python -m app.db.seed

Crea un psicólogo, un paciente y su vínculo activo. Es idempotente: si los
usuarios ya existen, no duplica nada.
"""
from sqlalchemy import select

from app import models  # noqa: F401 — registra los modelos en Base.metadata
from app.auth.jwt import hash_password
from app.db.base import Base
from app.db.session import SessionLocal, engine
from app.models import RolUsuario, Usuario, VinculoPacientePsicologo

EMAIL_PSICOLOGO = "psicologo@demo.pe"
EMAIL_PACIENTE = "paciente@demo.pe"
PASSWORD_DEMO = "Demo1234!segura"


def ejecutar_seed() -> None:
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        psicologo = db.scalar(select(Usuario).where(Usuario.email == EMAIL_PSICOLOGO))
        if psicologo is None:
            psicologo = Usuario(
                email=EMAIL_PSICOLOGO,
                nombre_completo="Psicóloga Demo",
                hashed_password=hash_password(PASSWORD_DEMO),
                rol=RolUsuario.PSICOLOGO,
            )
            db.add(psicologo)

        paciente = db.scalar(select(Usuario).where(Usuario.email == EMAIL_PACIENTE))
        if paciente is None:
            paciente = Usuario(
                email=EMAIL_PACIENTE,
                nombre_completo="Paciente Demo",
                hashed_password=hash_password(PASSWORD_DEMO),
                rol=RolUsuario.PACIENTE,
            )
            db.add(paciente)

        db.flush()

        vinculo = db.scalar(
            select(VinculoPacientePsicologo).where(
                VinculoPacientePsicologo.paciente_id == paciente.id,
                VinculoPacientePsicologo.psicologo_id == psicologo.id,
            )
        )
        if vinculo is None:
            db.add(
                VinculoPacientePsicologo(
                    paciente_id=paciente.id, psicologo_id=psicologo.id, activo=True
                )
            )

        db.commit()
        print(f"Seed listo. Usuarios: {EMAIL_PSICOLOGO} / {EMAIL_PACIENTE}")
        print(f"Contraseña (solo demo): {PASSWORD_DEMO}")
    finally:
        db.close()


if __name__ == "__main__":
    ejecutar_seed()

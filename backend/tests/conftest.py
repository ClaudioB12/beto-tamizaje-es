import os

# Debe fijarse ANTES de importar la app: el engine global del backend se
# construye al importar, y para tests debe ser SQLite en memoria.
os.environ.setdefault("BETO_DATABASE_URL", "sqlite://")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.auth.jwt import crear_token, hash_password
from app.core.exceptions import ServicioMLNoDisponibleError
from app.db.base import Base
from app.db.session import get_db
from app.main import app
from app.models import RolUsuario, Usuario, VinculoPacientePsicologo
from app.schemas.ml import (
    ClasificacionMLResponse,
    CrisisMLResponse,
    FragmentoML,
    RecuperacionMLResponse,
)
from app.services.ml_client import get_ml_client


class MLClientFalso:
    """Doble de prueba del cliente ML: comportamiento configurable por test."""

    def __init__(self) -> None:
        self.disponible = True
        self.riesgo = False
        self.motivo: str | None = None
        self.emocion = "ansiedad"
        self.severidad = "moderado"
        self.fragmentos = [
            FragmentoML(
                id="frag-1",
                texto="La respiración diafragmática ayuda a reducir la activación fisiológica.",
                categoria="psicoeducacion",
            )
        ]
        self.llamadas: list[str] = []

    def _verificar_disponible(self) -> None:
        if not self.disponible:
            raise ServicioMLNoDisponibleError("Servicio ML no disponible (simulado)")

    async def detectar_crisis(self, texto: str, paciente_id: str) -> CrisisMLResponse:
        self.llamadas.append("detectar_crisis")
        self._verificar_disponible()
        return CrisisMLResponse(riesgo=self.riesgo, motivo=self.motivo)

    async def clasificar(self, texto: str) -> ClasificacionMLResponse:
        self.llamadas.append("clasificar")
        self._verificar_disponible()
        return ClasificacionMLResponse(emocion=self.emocion, severidad=self.severidad)

    async def recuperar(self, emocion: str, severidad: str) -> RecuperacionMLResponse:
        self.llamadas.append("recuperar")
        self._verificar_disponible()
        return RecuperacionMLResponse(fragmentos=self.fragmentos)


@pytest.fixture
def factoria_sesiones():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    engine.dispose()


@pytest.fixture
def db(factoria_sesiones):
    sesion = factoria_sesiones()
    yield sesion
    sesion.close()


@pytest.fixture
def ml_falso():
    return MLClientFalso()


@pytest.fixture
def client(factoria_sesiones, ml_falso):
    def override_get_db():
        db = factoria_sesiones()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_ml_client] = lambda: ml_falso
    with TestClient(app) as cliente:
        yield cliente
    app.dependency_overrides.clear()


PASSWORD_TEST = "Password123!"


def _crear_usuario(db, email: str, rol: RolUsuario) -> Usuario:
    usuario = Usuario(
        email=email,
        nombre_completo="Usuario Prueba",
        hashed_password=hash_password(PASSWORD_TEST),
        rol=rol,
    )
    db.add(usuario)
    db.commit()
    db.refresh(usuario)
    return usuario


@pytest.fixture
def paciente(db) -> Usuario:
    return _crear_usuario(db, "paciente@test.pe", RolUsuario.PACIENTE)


@pytest.fixture
def otro_paciente(db) -> Usuario:
    return _crear_usuario(db, "otro.paciente@test.pe", RolUsuario.PACIENTE)


@pytest.fixture
def psicologo(db) -> Usuario:
    return _crear_usuario(db, "psicologo@test.pe", RolUsuario.PSICOLOGO)


@pytest.fixture
def vinculo(db, paciente, psicologo) -> VinculoPacientePsicologo:
    v = VinculoPacientePsicologo(
        paciente_id=paciente.id, psicologo_id=psicologo.id, activo=True
    )
    db.add(v)
    db.commit()
    db.refresh(v)
    return v


def _headers(usuario: Usuario) -> dict[str, str]:
    token = crear_token(usuario.id, usuario.rol.value)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def headers_paciente(paciente):
    return _headers(paciente)


@pytest.fixture
def headers_otro_paciente(otro_paciente):
    return _headers(otro_paciente)


@pytest.fixture
def headers_psicologo(psicologo):
    return _headers(psicologo)

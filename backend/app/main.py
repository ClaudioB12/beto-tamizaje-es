from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import models  # noqa: F401 — registra todos los modelos en Base.metadata
from app.config import get_settings
from app.core.exceptions import registrar_manejadores
from app.core.logging import configurar_logging
from app.db.base import Base
from app.db.session import engine
from app.routers import auth, chat, reportes, sesiones

settings = get_settings()
configurar_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Prototipo (SQLite): las tablas se crean directamente al arrancar.
    # En producción (PostgreSQL) el esquema se gestiona SOLO con Alembic.
    if settings.database_url.startswith("sqlite"):
        Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title=settings.app_nombre,
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

registrar_manejadores(app)

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(reportes.router)
app.include_router(sesiones.router)


@app.get("/health", tags=["salud"])
def health() -> dict[str, str]:
    return {"estado": "ok"}

"""Banco de plantillas de Capa 1 (respuestas empáticas parametrizadas).

El banco vive en un archivo JSON versionado en el repositorio
(app/plantillas/banco_plantillas.json): la revisión y aprobación por los
psicólogos del centro se hace vía pull request, con diff legible e historial
git como registro de aprobación. El agente NUNCA genera texto libre: solo
instancia estas plantillas con fragmentos psicoeducativos recuperados.
"""
import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ValidationError

from app.core.exceptions import PlantillaNoDisponibleError

RUTA_BANCO_POR_DEFECTO = (
    Path(__file__).resolve().parents[1] / "plantillas" / "banco_plantillas.json"
)


class Plantilla(BaseModel):
    id: str
    emocion: str
    severidad: str | None = None
    texto: str

    def instanciar(self, fragmento: str | None) -> str:
        cuerpo = self.texto.replace("{fragmento}", fragmento or "")
        # Colapsa espacios dobles si no hubo fragmento que insertar.
        return " ".join(cuerpo.split())


class BancoPlantillas(BaseModel):
    version: str
    estado: str
    plantillas: list[Plantilla]
    respuesta_crisis: Plantilla
    respuesta_fallback: Plantilla

    def seleccionar(self, emocion: str, severidad: str, indice_rotacion: int = 0) -> Plantilla:
        candidatas = [
            p for p in self.plantillas if p.emocion == emocion and p.severidad == severidad
        ]
        if not candidatas:
            candidatas = [
                p for p in self.plantillas if p.emocion == emocion and p.severidad is None
            ]
        if not candidatas:
            return self.respuesta_fallback
        # Rotación determinista según el número de mensajes previos de la
        # sesión: variedad sin aleatoriedad (reproducible en auditoría).
        return candidatas[indice_rotacion % len(candidatas)]


@lru_cache
def cargar_banco_plantillas(ruta: str = str(RUTA_BANCO_POR_DEFECTO)) -> BancoPlantillas:
    ruta_archivo = Path(ruta)
    if not ruta_archivo.exists():
        raise PlantillaNoDisponibleError(
            f"No se encontró el banco de plantillas en {ruta_archivo}"
        )
    try:
        datos = json.loads(ruta_archivo.read_text(encoding="utf-8"))
        return BancoPlantillas.model_validate(datos)
    except (ValueError, ValidationError) as exc:
        raise PlantillaNoDisponibleError("El banco de plantillas es inválido") from exc


def get_banco_plantillas() -> BancoPlantillas:
    """Dependencia de FastAPI (cacheada: el banco es inmutable en runtime)."""
    return cargar_banco_plantillas()

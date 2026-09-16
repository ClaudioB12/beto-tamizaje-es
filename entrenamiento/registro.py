"""Trazabilidad compartida por entrenar, evaluar, analizar y exportar.

Toda corrida y todo resultado queda atado, por hash, a la partición congelada y
a la entrada de la que salió. Si alguno de los dos cambia, los comandos se
niegan a seguir en vez de mezclar resultados de experimentos distintos.
"""
import hashlib
import importlib.metadata
import io
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from datos.fuentes import FUENTES
from datos.particion import Particion, verificar_sin_fugas
from datos.preparacion import deduplicar

RAIZ = Path(__file__).resolve().parent
CARPETA_PARTICIONES = RAIZ / "particiones"
CARPETA_CORRIDAS = RAIZ / "corridas"
CARPETA_PREDICCIONES = RAIZ / "predicciones"
CARPETA_RESULTADOS = RAIZ / "resultados"
CARPETA_EXPORTADOS = RAIZ / "exportados"

# Una prueba de humo (1 época, textos cortos) nunca comparte carpeta con el
# experimento real: sus resultados no deben poder mezclarse por accidente.
SUFIJO_HUMO = "-humo"


class ParticionInconsistenteError(RuntimeError):
    """La partición congelada no corresponde con los datos o fue modificada."""


def nombre_experimento(fuente: str, humo: bool) -> str:
    return f"{fuente}{SUFIJO_HUMO if humo else ''}"


def ahora_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(contenido: bytes) -> str:
    return hashlib.sha256(contenido).hexdigest()


def sha256_archivo(ruta: Path) -> str:
    return sha256_bytes(Path(ruta).read_bytes())


def sha256_json(objeto) -> str:
    """Hash estable de una estructura: el orden de las claves no lo altera."""
    canonico = json.dumps(objeto, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return sha256_bytes(canonico.encode("utf-8"))


def escribir_json(ruta: Path, objeto) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(objeto, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def leer_json(ruta: Path):
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def cargar_particion_congelada(
    fuente: str, carpeta: Path = CARPETA_PARTICIONES
) -> tuple[Particion, dict]:
    """Reconstruye la partición desde ``asignaciones.csv``; nunca la recalcula.

    Verifica que ni las asignaciones ni el archivo de entrada hayan cambiado
    desde que se congelaron.
    """
    if fuente not in FUENTES:
        raise ValueError(f"Fuente desconocida {fuente!r}; disponibles: {sorted(FUENTES)}")
    base = Path(carpeta) / fuente
    ruta_asignaciones = base / "asignaciones.csv"
    if not ruta_asignaciones.exists():
        raise ParticionInconsistenteError(
            f"No hay partición congelada para {fuente}. Ejecuta: python particionar.py --fuente {fuente}"
        )
    informe = leer_json(base / "informe.json")

    contenido = ruta_asignaciones.read_bytes()
    if sha256_bytes(contenido) != informe["sha256_asignaciones"]:
        raise ParticionInconsistenteError(
            f"{ruta_asignaciones} no coincide con el hash de informe.json: se modificó después de congelarse."
        )

    cargar, entrada = FUENTES[fuente]
    if sha256_archivo(entrada) != informe["entrada"]["sha256"]:
        raise ParticionInconsistenteError(
            f"{Path(entrada).name} cambió desde que se congeló la partición: los fragmento_id ya no "
            "corresponden. Restaura el archivo original o congela una partición nueva con --sobrescribir."
        )

    datos, _ = deduplicar(cargar(entrada))
    asignaciones = pd.read_csv(io.BytesIO(contenido), dtype=str)
    if len(asignaciones) != len(datos) or set(asignaciones["fragmento_id"]) != set(datos["fragmento_id"]):
        raise ParticionInconsistenteError(
            "Los fragmentos de la fuente no coinciden con los de la partición congelada."
        )

    unido = datos.merge(asignaciones, on="fragmento_id", validate="one_to_one", suffixes=("", "_congelado"))
    if (unido["sujeto_id"] != unido["sujeto_id_congelado"]).any():
        raise ParticionInconsistenteError("Algún fragmento cambió de sujeto respecto de la partición congelada.")

    partes = {
        nombre: unido[unido["particion"] == nombre]
        .drop(columns=["particion", "sujeto_id_congelado"])
        .reset_index(drop=True)
        for nombre in ("train", "val", "test")
    }
    particion = Particion(**partes)
    verificar_sin_fugas(particion)
    return particion, informe


def describir_entorno() -> dict:
    """Versiones y dispositivo: lo que el §2.2 pide para reproducir una corrida."""
    paquetes = {}
    for distribucion in importlib.metadata.distributions():
        nombre = distribucion.metadata["Name"]
        if nombre:
            paquetes[nombre.lower()] = distribucion.version
    entorno = {
        "python": platform.python_version(),
        "sistema": platform.platform(),
        "paquetes": dict(sorted(paquetes.items())),
    }
    try:
        import torch

        entorno["torch"] = {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        entorno["torch"] = None
    return entorno

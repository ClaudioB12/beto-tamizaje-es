"""Gate determinista de crisis (primera barrera de seguridad de Capa 1).

Se ejecuta SIEMPRE, de forma síncrona y bloqueante, antes de cualquier
clasificación o generación de respuesta, y no depende de red ni de modelo:
es un léxico auditable de patrones de ideación suicida y autolesión que los
psicólogos del centro pueden revisar y ampliar directamente en este archivo.

El detector del servicio ML (/ml/detectar-crisis) es la segunda barrera;
esta capa garantiza que la detección nunca dependa únicamente del modelo.
"""
import unicodedata

# Los patrones se escriben ya normalizados: minúsculas y sin tildes,
# porque la entrada se normaliza igual antes de comparar.
_PATRONES_CRISIS: dict[str, list[str]] = {
    "ideacion_suicida": [
        "suicid",
        "quitarme la vida",
        "quitar la vida",
        "no quiero vivir",
        "no quiero seguir viviendo",
        "no vale la pena vivir",
        "quiero morir",
        "quisiera morir",
        "quiero morirme",
        "matarme",
        "mejor estar muerto",
        "mejor estar muerta",
        "acabar con mi vida",
        "terminar con mi vida",
        "acabar con todo",
        "desaparecer para siempre",
        "todos estarian mejor sin mi",
    ],
    "autolesion": [
        "autolesion",
        "hacerme dano",
        "lastimarme",
        "cortarme",
        "herirme",
        "golpearme",
    ],
}


def _normalizar(texto: str) -> str:
    descompuesto = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in descompuesto if unicodedata.category(c) != "Mn")


def evaluar_crisis_lexica(texto: str) -> tuple[bool, str | None]:
    """Devuelve (hay_crisis, motivo). El motivo identifica la categoría del
    patrón, nunca el texto del paciente."""
    normalizado = _normalizar(texto)
    for categoria, patrones in _PATRONES_CRISIS.items():
        for patron in patrones:
            if patron in normalizado:
                return True, f"lexico:{categoria}"
    return False, None

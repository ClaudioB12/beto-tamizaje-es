"""Des-identificación determinista de texto libre del paciente.

Barrera mínima obligatoria antes de PERSISTIR o LOGUEAR cualquier texto del
paciente: elimina PII estructurada (correos, URLs, teléfonos, DNI) y nombres
introducidos con fórmulas de auto-presentación.

Limitación conocida: nombres arbitrarios sin fórmula de presentación
("hablé con Carlos") requieren NER; esa capa corresponde al servicio ML y
complementa —no reemplaza— estas reglas.
"""
import re

_PATRONES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"https?://\S+"), "[URL]"),
    # Celulares peruanos: 9 dígitos empezando en 9, con o sin +51 y separadores.
    (re.compile(r"\b(?:\+51[\s-]?)?9\d{2}[\s-]?\d{3}[\s-]?\d{3}\b"), "[TELEFONO]"),
    (re.compile(r"\b\d{8}\b"), "[DNI]"),
    # "me llamo Ana María" / "mi nombre es Juan Pérez" -> conserva la fórmula,
    # reemplaza el nombre (exige mayúscula inicial para no tocar texto clínico).
    (
        re.compile(
            r"\b((?i:me llamo|mi nombre es))\s+"
            r"(?:[A-ZÁÉÍÓÚÑ][a-záéíóúñü]+)(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñü]+){0,3}"
        ),
        r"\1 [NOMBRE]",
    ),
]


def desidentificar(texto: str) -> str:
    resultado = texto
    for patron, reemplazo in _PATRONES:
        resultado = patron.sub(reemplazo, resultado)
    return resultado

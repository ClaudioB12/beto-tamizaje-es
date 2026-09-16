"""Tests unitarios de los servicios puros (sin API ni base de datos)."""
import pytest
from pydantic import ValidationError

from app.config import (
    JWT_SECRET_PROTOTIPO,
    LONGITUD_MINIMA_SECRETO,
    Settings,
)
from app.services.crisis_gate import evaluar_crisis_lexica
from app.services.desidentificacion import desidentificar


class TestDesidentificacion:
    def test_email(self):
        assert desidentificar("escríbeme a ana.q@mail.com") == "escríbeme a [EMAIL]"

    def test_dni(self):
        assert "[DNI]" in desidentificar("mi DNI es 45678123")

    def test_celular_peruano(self):
        assert "[TELEFONO]" in desidentificar("mi número es 987 654 321")

    def test_nombre_con_formula_de_presentacion(self):
        resultado = desidentificar("Hola, me llamo Ana María Quispe y estoy triste")
        assert "Ana" not in resultado
        assert "[NOMBRE]" in resultado
        assert "estoy triste" in resultado

    def test_no_altera_texto_clinico(self):
        texto = "me siento muy ansioso y no puedo dormir"
        assert desidentificar(texto) == texto


class TestCrisisGate:
    def test_ideacion_suicida_con_tildes_y_mayusculas(self):
        hay_crisis, motivo = evaluar_crisis_lexica("Ya NO quiero VIVIR más")
        assert hay_crisis is True
        assert motivo == "lexico:ideacion_suicida"

    def test_autolesion(self):
        hay_crisis, motivo = evaluar_crisis_lexica("a veces pienso en cortarme")
        assert hay_crisis is True
        assert motivo == "lexico:autolesion"

    def test_texto_sin_crisis(self):
        hay_crisis, motivo = evaluar_crisis_lexica("hoy me fue bien en el trabajo")
        assert hay_crisis is False
        assert motivo is None


class TestGuardiaSecretoProduccion:
    """El secreto de prototipo es público: en producción debe impedir el arranque."""

    def test_produccion_rechaza_secreto_de_prototipo(self):
        with pytest.raises(ValidationError):
            Settings(entorno="produccion", jwt_secret=JWT_SECRET_PROTOTIPO)

    def test_produccion_rechaza_secreto_corto(self):
        with pytest.raises(ValidationError):
            Settings(entorno="produccion", jwt_secret="corto")

    def test_produccion_acepta_secreto_fuerte(self):
        secreto = "k" * LONGITUD_MINIMA_SECRETO
        assert Settings(entorno="produccion", jwt_secret=secreto).jwt_secret == secreto

    def test_desarrollo_conserva_el_valor_por_defecto(self):
        # El flujo local no debe exigir configuración: la guardia solo aplica
        # a los entornos productivos.
        assert Settings(entorno="desarrollo").jwt_secret == JWT_SECRET_PROTOTIPO

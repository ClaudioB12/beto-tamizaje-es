import pytest

from datos.esquema import EsquemaInvalidoError, validar
from datos.normalizacion import normalizar_texto
from tests.conftest import construir_df


def _valido():
    return construir_df(
        [
            ("f1", "s1", "me siento nervioso", 1, 0),
            ("f2", "s1", "no puedo dormir", 1, 0),
            ("f3", "s2", "todo me cuesta", 0, 1),
            ("f4", "s3", "hoy salí a caminar", -1, 0),
        ]
    )


class TestNormalizacion:
    def test_tildes_mayusculas_y_signos(self):
        assert normalizar_texto("¡Hola,   DOCTÓR!") == "hola doctor"

    def test_variantes_de_una_frase_comparten_la_base(self):
        base = normalizar_texto("Siento un poco de nervios.")
        variante = normalizar_texto("Doctor, siento un poco de nervios todo el tiempo.")
        assert base in variante


class TestValidar:
    def test_acepta_un_conjunto_valido(self):
        validar(_valido(), etiquetas_por_sujeto=True)

    def test_rechaza_columna_faltante(self):
        with pytest.raises(EsquemaInvalidoError, match="sujeto_id"):
            validar(_valido().drop(columns="sujeto_id"), etiquetas_por_sujeto=False)

    def test_rechaza_valor_de_etiqueta_fuera_de_rango(self):
        df = _valido()
        df.loc[0, "ansiedad"] = 2
        with pytest.raises(EsquemaInvalidoError, match="ansiedad fuera"):
            validar(df, etiquetas_por_sujeto=False)

    def test_rechaza_etiquetas_no_enteras(self):
        # Un nulo convierte la columna en float: es el síntoma de haber olvidado
        # codificar lo desconocido como -1.
        df = _valido()
        df["depresion"] = df["depresion"].astype("float64")
        with pytest.raises(EsquemaInvalidoError, match="entera"):
            validar(df, etiquetas_por_sujeto=False)

    def test_rechaza_fila_sin_ninguna_etiqueta_conocida(self):
        df = _valido()
        df.loc[3, "depresion"] = -1
        with pytest.raises(EsquemaInvalidoError, match="f4"):
            validar(df, etiquetas_por_sujeto=False)

    def test_rechaza_fragmento_id_repetido(self):
        df = _valido()
        df.loc[1, "fragmento_id"] = "f1"
        with pytest.raises(EsquemaInvalidoError, match="repetido"):
            validar(df, etiquetas_por_sujeto=False)

    def test_etiquetas_inconsistentes_por_sujeto_solo_fallan_si_se_exige(self):
        df = _valido()
        df.loc[1, "ansiedad"] = 0  # s1 queda con ansiedad 1 y 0
        validar(df, etiquetas_por_sujeto=False)  # válido en el conjunto clínico
        with pytest.raises(EsquemaInvalidoError, match="s1"):
            validar(df, etiquetas_por_sujeto=True)

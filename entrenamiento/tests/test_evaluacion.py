import json

import numpy as np
import pytest

from evaluacion.instrumentos import categoria
from evaluacion.predicciones import construir_predicciones
from evaluacion.severidad import BandasInvalidasError, asignar_severidad, cargar_bandas
from evaluacion.umbrales import calibrar_umbrales
from tests.conftest import construir_df


class TestInstrumentos:
    @pytest.mark.parametrize(
        "puntaje, esperada",
        [(0, "minima"), (4, "minima"), (5, "leve"), (9, "leve"), (10, "moderada"), (14, "moderada"),
         (15, "moderadamente_severa"), (19, "moderadamente_severa"), (20, "severa"), (27, "severa")],
    )
    def test_limites_de_phq9(self, puntaje, esperada):
        assert categoria("phq9", [puntaje])[0] == esperada

    @pytest.mark.parametrize(
        "puntaje, esperada",
        [(0, "minima"), (4, "minima"), (5, "leve"), (9, "leve"), (10, "moderada"), (14, "moderada"),
         (15, "severa"), (21, "severa")],
    )
    def test_limites_de_gad7(self, puntaje, esperada):
        assert categoria("gad7", [puntaje])[0] == esperada

    @pytest.mark.parametrize("instrumento, invalido", [("phq9", 28), ("gad7", 22), ("gad7", -1), ("phq9", 4.5)])
    def test_rechaza_puntajes_fuera_de_rango(self, instrumento, invalido):
        with pytest.raises(ValueError):
            categoria(instrumento, [invalido])


class TestUmbrales:
    def _val(self):
        filas = [(f"f{i}", f"s{i}", f"t{i}", i % 2, (i // 2) % 2) for i in range(40)]
        return construir_df(filas)

    def test_elige_el_umbral_que_separa_perfectamente(self):
        val = self._val()
        probabilidades = np.column_stack(
            [np.where(val["ansiedad"] == 1, 0.72, 0.30), np.where(val["depresion"] == 1, 0.55, 0.10)]
        )
        umbrales = calibrar_umbrales(val, probabilidades)
        # Cualquier umbral en (0,30, 0,72] separa; ante empate gana el más bajo.
        assert umbrales["ansiedad"]["umbral"] == pytest.approx(0.31)
        assert umbrales["ansiedad"]["f1_validacion"] == 1.0
        assert umbrales["depresion"]["umbral"] == pytest.approx(0.11)

    def test_las_filas_desconocidas_no_influyen(self):
        val = self._val()
        probabilidades = np.column_stack([np.where(val["ansiedad"] == 1, 0.8, 0.2), np.full(len(val), 0.5)])
        base = calibrar_umbrales(val, probabilidades)["ansiedad"]
        con_ruido = val.copy()
        con_ruido.loc[:9, "ansiedad"] = -1
        probabilidades_ruido = probabilidades.copy()
        probabilidades_ruido[:10, 0] = 0.99  # predicciones absurdas en filas desconocidas
        ruido = calibrar_umbrales(con_ruido, probabilidades_ruido)["ansiedad"]
        assert ruido["umbral"] == base["umbral"] and ruido["filas_validacion"] == 30

    def test_marca_un_umbral_en_el_borde_de_la_rejilla(self):
        val = self._val()
        # Probabilidades sin relación con la etiqueta: lo mejor es predecir todo positivo.
        probabilidades = np.full((len(val), 2), 0.5)
        probabilidades[:, 0] = np.linspace(0.9, 0.95, len(val))
        umbrales = calibrar_umbrales(val, probabilidades)
        assert umbrales["ansiedad"]["umbral"] == pytest.approx(0.05)
        assert umbrales["ansiedad"]["en_borde_de_rejilla"] is True

    def test_un_umbral_interior_no_se_marca(self):
        val = self._val()
        probabilidades = np.column_stack(
            [np.where(val["ansiedad"] == 1, 0.72, 0.30), np.where(val["depresion"] == 1, 0.55, 0.10)]
        )
        assert calibrar_umbrales(val, probabilidades)["ansiedad"]["en_borde_de_rejilla"] is False

    def test_rechaza_validacion_de_una_sola_clase(self):
        val = construir_df([(f"f{i}", f"s{i}", f"t{i}", 0, i % 2) for i in range(10)])
        with pytest.raises(ValueError, match="una sola clase"):
            calibrar_umbrales(val, np.full((10, 2), 0.5))


def _bandas(tmp_path, **cambios):
    datos = {
        "congelado_en": "2026-10-01",
        "referencia": "acta de prueba",
        "bandas": {"ansiedad": {"cortes": [0.3, 0.6, 0.85]}, "depresion": {"cortes": [0.3, 0.5, 0.7, 0.9]}},
    }
    datos.update(cambios)
    ruta = tmp_path / "bandas.json"
    ruta.write_text(json.dumps(datos), encoding="utf-8")
    return ruta


class TestSeveridad:
    def test_asigna_categorias_del_instrumento_de_cada_etiqueta(self, tmp_path):
        bandas = cargar_bandas(_bandas(tmp_path))
        assert asignar_severidad(np.array([0.1, 0.3, 0.7, 0.95]), "ansiedad", bandas).tolist() == [
            "minima", "leve", "moderada", "severa"
        ]
        assert asignar_severidad(np.array([0.2, 0.75]), "depresion", bandas).tolist() == [
            "minima", "moderadamente_severa"
        ]

    def test_exige_tantos_cortes_como_categorias_menos_uno(self, tmp_path):
        with pytest.raises(BandasInvalidasError, match="gad7"):
            cargar_bandas(_bandas(tmp_path, bandas={"ansiedad": {"cortes": [0.5]}, "depresion": {"cortes": [0.3, 0.5, 0.7, 0.9]}}))

    def test_exige_cortes_crecientes(self, tmp_path):
        with pytest.raises(BandasInvalidasError, match="crecientes"):
            cargar_bandas(_bandas(tmp_path, bandas={"ansiedad": {"cortes": [0.6, 0.3, 0.85]}, "depresion": {"cortes": [0.3, 0.5, 0.7, 0.9]}}))

    def test_exige_fecha_y_referencia_de_congelamiento(self, tmp_path):
        with pytest.raises(BandasInvalidasError, match="congelado_en"):
            cargar_bandas(_bandas(tmp_path, congelado_en="pronto"))
        with pytest.raises(BandasInvalidasError, match="referencia"):
            cargar_bandas(_bandas(tmp_path, referencia=""))


def test_tabla_de_predicciones_tiene_el_formato_de_la_guia(tmp_path):
    df = construir_df([("f2", "s1", "a", 1, -1), ("f1", "s2", "b", 0, 1)])
    probabilidades = np.array([[0.9, 0.2], [0.1, 0.7]])
    umbrales = {"ansiedad": {"umbral": 0.5}, "depresion": {"umbral": 0.5}}
    tabla = construir_predicciones(df, probabilidades, umbrales, cargar_bandas(_bandas(tmp_path)))
    assert list(tabla.columns) == [
        "fragmento_id", "sujeto_id", "dominio", "y_ansiedad", "y_depresion", "p_ansiedad", "p_depresion",
        "pred_ansiedad", "pred_depresion", "sev_ansiedad", "sev_depresion", "phq9", "gad7",
    ]
    assert tabla["fragmento_id"].tolist() == ["f1", "f2"]  # ordenada: todas las corridas alinean
    assert tabla.loc[1, "y_depresion"] == -1
    assert tabla[["pred_ansiedad", "pred_depresion"]].to_numpy().tolist() == [[0, 1], [1, 0]]
    assert tabla.loc[1, "sev_ansiedad"] == "severa"

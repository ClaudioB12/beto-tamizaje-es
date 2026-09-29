"""Módulo estadístico, validado con datos simulados de respuesta conocida."""
import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score, f1_score

from estadistica.bootstrap import ic_brecha_dominio, ic_diferencia, ic_media_semillas
from estadistica.concordancia import kappa_cohen, kappa_ponderado
from estadistica.escala import alfa_cronbach, omega_mcdonald, v_aiken
from estadistica.metricas import f1_control, f1_macro, f1_macro_lote, metricas_por_etiqueta
from estadistica.pruebas import prueba_semillas


def _simular(rng, n, prevalencia=0.4, error=0.2):
    y = (rng.random((n, 2)) < prevalencia).astype(int)
    voltear = rng.random((n, 2)) < error
    return y, np.where(voltear, 1 - y, y)


class TestMetricas:
    def test_f1_por_etiqueta_coincide_con_sklearn(self):
        rng = np.random.default_rng(0)
        y, pred = _simular(rng, 300)
        m = metricas_por_etiqueta(y, pred)
        for j, etiqueta in enumerate(("ansiedad", "depresion")):
            assert m[etiqueta]["f1"] == pytest.approx(f1_score(y[:, j], pred[:, j]))
        assert f1_macro(y, pred) == pytest.approx(np.mean([f1_score(y[:, j], pred[:, j]) for j in (0, 1)]))

    def test_ignora_las_etiquetas_desconocidas(self):
        rng = np.random.default_rng(1)
        y, pred = _simular(rng, 200)
        con_desconocidas = y.copy()
        con_desconocidas[:50, 0] = -1
        esperado = f1_score(y[50:, 0], pred[50:, 0])
        assert metricas_por_etiqueta(con_desconocidas, pred)["ansiedad"]["f1"] == pytest.approx(esperado)
        assert metricas_por_etiqueta(con_desconocidas, pred)["ansiedad"]["filas_conocidas"] == 150

    def test_la_version_por_lotes_coincide(self):
        rng = np.random.default_rng(2)
        y, pred = _simular(rng, 100)
        lote = f1_macro_lote(np.stack([y, y[::-1]]), np.stack([pred, pred[::-1]]))
        assert lote == pytest.approx([f1_macro(y, pred)] * 2)

    def test_sin_positivos_el_f1_vale_cero(self):
        y = np.zeros((10, 2), dtype=int)
        assert f1_macro(y, y) == 0.0

    def test_f1_control_solo_con_ambas_etiquetas_conocidas(self):
        y = np.array([[0, 0], [1, 0], [0, 0], [0, 1]])
        pred = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
        assert f1_control(y, pred) == pytest.approx(f1_score([1, 0, 1, 0], [1, 1, 0, 0]))
        assert f1_control(np.array([[0, -1], [-1, 1]]), pred[:2]) is None


class TestBootstrap:
    def test_el_intervalo_contiene_la_estimacion(self):
        rng = np.random.default_rng(3)
        y, _ = _simular(rng, 400)
        preds = [_simular(np.random.default_rng(s), 400)[1] for s in range(5)]
        ic = ic_media_semillas(y, preds, n_bootstrap=500)
        assert ic["ic_inferior"] <= ic["estimacion"] <= ic["ic_superior"]
        assert ic["semillas"] == 5

    def test_cobertura_cercana_al_95(self):
        # Con un proceso generador conocido, el IC debe contener el F1 poblacional
        # en torno al 95 % de las repeticiones.
        rng = np.random.default_rng(4)
        y_grande, pred_grande = _simular(rng, 400_000)
        verdadero = f1_macro(y_grande, pred_grande)
        cubiertos, repeticiones = 0, 120
        for r in range(repeticiones):
            y, pred = _simular(np.random.default_rng(1000 + r), 300)
            ic = ic_media_semillas(y, [pred], n_bootstrap=400, semilla=r)
            cubiertos += ic["ic_inferior"] <= verdadero <= ic["ic_superior"]
        assert 0.88 <= cubiertos / repeticiones <= 0.99

    def test_diferencia_detecta_un_modelo_claramente_mejor(self):
        rng = np.random.default_rng(5)
        y, _ = _simular(rng, 500)
        mejor = [np.where(np.random.default_rng(s).random((500, 2)) < 0.05, 1 - y, y) for s in range(5)]
        peor = [np.where(np.random.default_rng(50 + s).random((500, 2)) < 0.35, 1 - y, y) for s in range(5)]
        d = ic_diferencia(y, mejor, peor, n_bootstrap=500)
        assert d["estimacion"] > 0 and d["excluye_cero"]

    def test_diferencia_de_un_modelo_consigo_mismo_es_cero(self):
        rng = np.random.default_rng(6)
        y, pred = _simular(rng, 300)
        d = ic_diferencia(y, [pred], [pred], n_bootstrap=200)
        assert d["estimacion"] == 0 and d["ic_inferior"] == 0 and d["ic_superior"] == 0

    def test_brecha_de_dominio_con_remuestreos_independientes(self):
        rng = np.random.default_rng(7)
        y_o, p_o = _simular(rng, 400, error=0.05)
        y_d, p_d = _simular(rng, 200, error=0.3)
        brecha = ic_brecha_dominio(y_o, [p_o], y_d, [p_d], n_bootstrap=500)
        assert brecha["estimacion"] > 0 and brecha["ic_inferior"] > 0

    def test_es_reproducible_con_la_misma_semilla(self):
        rng = np.random.default_rng(8)
        y, pred = _simular(rng, 200)
        assert ic_media_semillas(y, [pred], 300, semilla=1) == ic_media_semillas(y, [pred], 300, semilla=1)


class TestPruebaSemillas:
    def test_cinco_victorias_dan_el_p_minimo_de_mann_whitney(self):
        r = prueba_semillas([0.81, 0.82, 0.80, 0.83, 0.815], [0.70, 0.71, 0.69, 0.72, 0.705])
        mw = r["mann_whitney_unilateral"]
        assert mw["p"] == pytest.approx(1 / 252)
        assert mw["r_rango_biserial"] == 1.0
        assert mw["p"] < 0.05

    def test_documenta_que_la_prueba_del_ppi_no_alcanza_significancia(self):
        r = prueba_semillas([0.81, 0.82, 0.80, 0.83, 0.815], [0.70, 0.71, 0.69, 0.72, 0.705])
        wilcoxon = r["wilcoxon_pareada_bilateral_ppi"]
        assert wilcoxon["p"] == pytest.approx(0.0625)
        assert wilcoxon["p_minimo_alcanzable"] == pytest.approx(0.0625)
        assert "no puede ser < 0,05" in wilcoxon["nota"]

    def test_sin_diferencia_no_rechaza(self):
        r = prueba_semillas([0.70, 0.71, 0.69, 0.72, 0.705], [0.71, 0.70, 0.72, 0.69, 0.705])
        assert r["prueba_usada"] == "mann_whitney_unilateral"
        assert r["mann_whitney_unilateral"]["p"] > 0.05

    def test_un_grupo_determinista_no_cuenta_como_cinco_observaciones(self):
        # TF–IDF: cinco corridas idénticas. Mann–Whitney daría p = 1/252 ≈ 0,004
        # como si fueran cinco observaciones; la prueba correcta da 1/32.
        r = prueba_semillas([0.81, 0.82, 0.80, 0.83, 0.815], [0.49] * 5)
        assert r["b_determinista"] and r["prueba_usada"] == "wilcoxon_una_muestra_unilateral"
        assert r["mann_whitney_unilateral"] is None
        assert r["wilcoxon_una_muestra_unilateral"]["p"] == pytest.approx(1 / 32)
        assert "determinista" in r["nota"]

    def test_determinista_en_el_lado_a_invierte_la_hipotesis(self):
        r = prueba_semillas([0.9] * 5, [0.70, 0.71, 0.69, 0.72, 0.705])
        assert r["a_determinista"]
        assert r["wilcoxon_una_muestra_unilateral"]["p"] == pytest.approx(1 / 32)

    def test_ambos_deterministas_no_se_contrastan(self):
        r = prueba_semillas([0.7] * 5, [0.6] * 5)
        assert r["prueba_usada"] is None and "IC bootstrap" in r["nota"]


class TestConcordancia:
    ORDEN = ("minima", "leve", "moderada", "severa")

    def test_kappa_ponderado_coincide_con_sklearn(self):
        rng = np.random.default_rng(9)
        referencia = rng.choice(self.ORDEN, 150)
        inferida = np.where(rng.random(150) < 0.7, referencia, rng.choice(self.ORDEN, 150))
        k = kappa_ponderado(referencia, inferida, self.ORDEN, n_bootstrap=300)
        codigos = {c: i for i, c in enumerate(self.ORDEN)}
        esperado = cohen_kappa_score(
            [codigos[c] for c in referencia], [codigos[c] for c in inferida], weights="quadratic"
        )
        assert k["kappa"] == pytest.approx(esperado)
        assert k["ic_inferior"] <= k["kappa"] <= k["ic_superior"]

    def test_acuerdo_perfecto_vale_uno(self):
        valores = list(self.ORDEN) * 10
        assert kappa_ponderado(valores, valores, self.ORDEN, n_bootstrap=100)["kappa"] == pytest.approx(1.0)

    def test_rechaza_categorias_fuera_del_orden(self):
        with pytest.raises(ValueError, match="grave"):
            kappa_ponderado(["leve", "grave"], ["leve", "leve"], self.ORDEN, n_bootstrap=10)

    def test_kappa_de_cohen_entre_anotadores(self):
        a = ["ansiedad", "depresion", "control", "ansiedad"] * 20
        b = ["ansiedad", "depresion", "control", "control"] * 20
        k = kappa_cohen(a, b, n_bootstrap=200)
        assert k["kappa"] == pytest.approx(cohen_kappa_score(a, b))


class TestEscala:
    def test_v_aiken_ejemplo_calculado_a_mano(self):
        # 5 jueces, escala 1–5: (4+3+4+4+3) / (5·4) = 18/20 = 0,90 y (0+1+0+1+0)/20 = 0,10
        jueces = [[5, 1], [4, 2], [5, 1], [5, 2], [4, 1]]
        assert v_aiken(jueces).tolist() == pytest.approx([0.90, 0.10])

    def test_v_aiken_rechaza_valores_fuera_de_escala(self):
        with pytest.raises(ValueError):
            v_aiken([[6, 3]])

    def test_alfa_de_items_identicos_es_uno(self):
        rng = np.random.default_rng(10)
        base = rng.integers(1, 6, size=(40, 1))
        assert alfa_cronbach(np.hstack([base] * 4)) == pytest.approx(1.0)

    def test_alfa_de_items_independientes_es_cercano_a_cero(self):
        rng = np.random.default_rng(11)
        assert abs(alfa_cronbach(rng.normal(size=(5000, 6)))) < 0.05

    def test_omega_recupera_el_valor_de_un_modelo_congenérico(self):
        rng = np.random.default_rng(12)
        cargas = np.array([0.8, 0.7, 0.6, 0.5, 0.75])
        factor = rng.normal(size=(8000, 1))
        items = factor * cargas + rng.normal(size=(8000, 5)) * np.sqrt(1 - cargas**2)
        verdadero = cargas.sum() ** 2 / (cargas.sum() ** 2 + (1 - cargas**2).sum())
        assert omega_mcdonald(items)["omega"] == pytest.approx(verdadero, abs=0.02)

"""Elección de w, desglose por calidad de traducción y control de calidad (C1, M4)."""
import numpy as np
import pandas as pd
import pytest

import analizar_traduccion
import calidad_traduccion
import elegir_peso
from tests.test_tres_clases import _escribir_carpeta, _tabla


def _tabla_w(filas):
    cols = ["arquitectura", "w", "f1_macro", "ansiedad→depresion", "depresion→ansiedad", "control→alguna", "umbrales_en_borde"]
    return pd.DataFrame(filas, columns=cols)


def test_regla_de_w_elige_menor_doble_asignacion_que_cumple(tmp_path):
    rutas = []
    for i, ruido in enumerate((0.0, 0.002, -0.002)):
        t = _tabla_w([
            ["beto_ajustado", 0, 0.976 + ruido, 0.99, 0.98, 0.04, ""],
            ["beto_ajustado", 0.3, 0.966 + ruido, 0.52, 0.51, 0.07, "depresion"],
            ["beto_ajustado", 0.5, 0.962 + ruido, 0.33, 0.41, 0.07, ""],   # pierde 0,014: no cumple
            ["tfidf_lr", 0, 0.939, 0.83, 0.83, 0.05, ""],
            ["tfidf_lr", 0.5, 0.935, 0.42, 0.48, 0.107, ""],               # controles > 10 %: no cumple
        ])
        ruta = tmp_path / f"s{i}.csv"; t.to_csv(ruta, index=False); rutas.append(ruta)
    resumen = elegir_peso.resumir(elegir_peso.cargar(rutas))
    eleccion = elegir_peso.elegir(resumen)
    assert eleccion["beto_ajustado"]["w"] == pytest.approx(0.3)
    assert eleccion["tfidf_lr"]["w"] == 0 and eleccion["tfidf_lr"]["solo_cumple_w0"] is True
    assert "w = 0.3" in elegir_peso.resumen_markdown(resumen, eleccion)
    # Cumplimiento por semilla: descriptivo, con la pérdida frente a w = 0 de la misma semilla.
    semillas = elegir_peso.por_semilla(elegir_peso.cargar(rutas))
    fila = semillas[(semillas.arquitectura == "beto_ajustado") & (semillas.w == 0.5)].iloc[0]
    assert fila["perdida_vs_w0"] == pytest.approx(-0.014) and not fila["cumple_perdida"]
    assert "Cumplimiento por semilla" in elegir_peso.markdown_por_semilla(semillas, eleccion)


def test_desglose_por_calidad_de_traduccion(tmp_path):
    rng = np.random.default_rng(3)
    n = 240
    real = rng.integers(0, 3, n)
    y_ans = np.where(real == 0, 1, np.where(real == 2, 0, -1))
    y_dep = np.where(real == 1, 1, np.where(real == 2, 0, -1))
    suj = [f"s{i}" for i in range(n)]
    tablas = {s: _tabla(y_ans, y_dep, np.clip((real == 0) + rng.normal(0, .3, n), 0, 1),
                        np.clip((real == 1) + rng.normal(0, .3, n), 0, 1), sujetos=suj) for s in (13, 42)}
    _escribir_carpeta(tmp_path / "pred", {"beto_ajustado": tablas})
    ids = tablas[13]["fragmento_id"]
    largo = rng.integers(20, 120, n)
    datos = pd.DataFrame({"fragmento_id": ids, "texto_en": ["word " * k for k in largo],
                          "texto_es": ["palabra " * int(k * rng.uniform(.7, 1.0)) for k in largo]})
    diag = pd.DataFrame({"fragmento_id": ids[:12], "cobertura": 0.5, "motivos": "cobertura"})
    r = analizar_traduccion.analizar([tmp_path / "pred"], datos, diag, 300, 2026)
    assert r["estratos"]["marcada"]["publicaciones"] == 12
    assert sum(e["publicaciones"] for e in r["estratos"].values()) == n
    assert "razón baja" in analizar_traduccion.resumen_markdown(r)


def test_calidad_traduccion_kappa_y_desacuerdos():
    base = pd.DataFrame({"fragmento_id": [f"r{i}" for i in range(6)],
                         "adecuacion_1a5": [5, 4, 3, 5, 2, 4], "fluidez_1a5": [5, 4, 4, 5, 3, 4],
                         "senal_salud_mental_1a5": [5, 5, 4, 5, 2, 4],
                         "aceptable_si_no": ["sí", "sí", "sí", "sí", "no", "sí"]})
    base["evaluador"] = "Ana"
    otro = base.copy()
    otro["evaluador"] = "Luis"
    otro.loc[2, "aceptable_si_no"] = "no"
    otro["inversion_polaridad_si_no"] = ["no", "no", "no", "no", "sí", "no"]
    base["inversion_polaridad_si_no"] = "no"
    r = calidad_traduccion.comparar(base, otro)
    assert r["kappa"]["adecuacion_1a5"] == pytest.approx(1.0)
    assert r["desacuerdos_aceptable"] == ["r2"] and r["inversiones_polaridad"] == ["r4"]
    assert r["aceptables"]["ambos"] == pytest.approx(4 / 6)


def test_calidad_rechaza_revisar_ia_y_mismo_evaluador():
    base = pd.DataFrame({"fragmento_id": ["r0", "r1"], "adecuacion_1a5": [5, 4], "fluidez_1a5": [5, 4],
                         "senal_salud_mental_1a5": [5, 4], "aceptable_si_no": ["sí", "sí"],
                         "inversion_polaridad_si_no": ["no", "no"], "evaluador": "Ana"})
    otro = base.assign(evaluador="Luis")
    with pytest.raises(ValueError, match="fuera de escala"):
        calidad_traduccion.comparar(base.assign(aceptable_si_no=["sí", "revisar"]), otro)
    with pytest.raises(ValueError, match="pre-revisión"):
        calidad_traduccion.comparar(base.assign(IA_sugerencia_aceptable="sí"), otro)
    with pytest.raises(ValueError, match="mismo evaluador"):
        calidad_traduccion.comparar(base, base.copy())
    r = calidad_traduccion.comparar(base, otro)
    assert r["evaluadores"][1]["evaluador"] == "Luis" and r["momento"] == "acuerdo previo al consenso"

"""Piezas del Componente A que se preparan sin GPU: términos, bootstrap jerárquico,
probabilidades de validación, ablación solo en prueba y congelación."""
import numpy as np
import pandas as pd
import yaml

import ablacion_prueba
import analizar_pi1
import congelar_componente_A
import registro
from datos.reddit import enmascarar
from estadistica import tres_clases as tc
from tests.conftest import PARTICIONES
from tests.test_pipeline import experimento  # noqa: F401  (fixture de sesión)
from tests.test_tres_clases import _tabla


def test_lista_de_terminos_versionada():
    assert enmascarar("Tengo ansiedad, tomo antidepresivos y leo r/depression") == "Tengo [MASK], tomo [MASK] y leo [MASK]"
    assert enmascarar("mi GAD") == "mi [MASK]" and enmascarar("un tag") == "un tag"   # siglas solo en mayúsculas
    assert enmascarar("Me siento triste") == "Me siento triste"


def _tablas(ruido, n=240, semillas=(13, 42, 77)):
    rng = np.random.default_rng(5)
    real = rng.integers(0, 3, n)
    y_a = np.where(real == 0, 1, np.where(real == 2, 0, -1))
    y_d = np.where(real == 1, 1, np.where(real == 2, 0, -1))
    suj = [f"s{i // 2}" for i in range(n)]
    out = {}
    for s in semillas:
        r = np.random.default_rng(s + int(100 * ruido))
        out[s] = _tabla(y_a, y_d, np.clip((real == 0) + r.normal(0, ruido, n), 0, 1),
                        np.clip((real == 1) + r.normal(0, ruido, n), 0, 1), sujetos=suj)
    return out


def test_bootstrap_jerarquico_pareado_y_mas_ancho_con_semillas_variables():
    a, b = _tablas(0.25), _tablas(0.6)
    rem = tc.Remuestreo(a[13], 2000, 2026)
    # Misma arquitectura contra sí misma: diferencia exactamente 0 (mismos índices de semilla y sujeto).
    assert np.all(rem.distribucion_jerarquica(a) - rem.distribucion_jerarquica(a) == 0)
    d_prot = rem.distribucion(a)[0] - rem.distribucion(b)[0]
    d_jer = rem.distribucion_jerarquica(a) - rem.distribucion_jerarquica(b)
    estimacion = np.mean(rem.distribucion(a)[1]) - np.mean(rem.distribucion(b)[1])
    inf, sup = tc.intervalo(d_jer)
    assert inf <= estimacion <= sup
    assert np.ptp(tc.intervalo(d_jer)) >= np.ptp(tc.intervalo(d_prot)) * 0.9   # no más estrecho en lo sustancial


def test_analizar_pi1_reporta_ambos_intervalos(tmp_path):
    from tests.test_tres_clases import _escribir_carpeta
    _escribir_carpeta(tmp_path / "p", {"beto_ajustado": _tablas(0.2), "beto_sonda": _tablas(0.6), "tfidf_lr": _tablas(0.5)})
    r = analizar_pi1.analizar(tmp_path / "p", None, 500, 2026)
    c = r["comparaciones_h1"][0]
    assert {"ic95", "ic95_jerarquico", "limite_inferior_positivo_jerarquico"} <= set(c)
    assert "jerárquico" in analizar_pi1.resumen_markdown(r)


def test_validacion_guardada_y_ablacion_solo_en_prueba(experimento, tmp_path):  # noqa: F811
    carpetas, _ = experimento
    pred = carpetas["predicciones"] / "sintetico"
    indice = registro.leer_json(pred / "indice.json")
    for e in indice["archivos"]:
        v = e["validacion"]
        assert registro.sha256_archivo(pred / v["archivo"]) == v["sha256"]
    abl = tmp_path / "abl"
    assert ablacion_prueba.main(["--fuente", "sintetico", "--corridas", str(carpetas["corridas"]),
                                 "--particiones", str(PARTICIONES), "--predicciones", str(abl)]) == 0
    ind_abl = registro.leer_json(abl / "sintetico" / "indice.json")
    assert ind_abl["ablacion"]["tipo"].startswith("solo prueba")
    # Mismos fragmentos y mismas corridas que la evaluación original: la comparación es pareada.
    assert {(e["arquitectura"], e["semilla"]) for e in ind_abl["archivos"]} == \
           {(e["arquitectura"], e["semilla"]) for e in indice["archivos"]}
    original = pd.read_csv(pred / indice["archivos"][0]["archivo"])
    ablada = pd.read_csv(abl / "sintetico" / ind_abl["archivos"][0]["archivo"])
    assert original["fragmento_id"].tolist() == ablada["fragmento_id"].tolist()


def test_congelar_escribe_yaml_con_hashes(tmp_path):
    decision = tmp_path / "d.json"
    registro.escribir_json(decision, {"provisional": True, "w": {"beto_ajustado": 0.5, "beto_sonda": 0.5, "tfidf_lr": 0.1}})
    assert congelar_componente_A.main(["--decision", str(decision), "--fuente", "sintetico",
                                       "--particiones", str(PARTICIONES), "--salida", str(tmp_path / "o")]) == 0
    texto = (tmp_path / "o" / "componente_A_frozen.yaml").read_text(encoding="utf-8")
    y = yaml.safe_load(texto)
    assert y["estado"].startswith("provisional")
    assert y["arquitecturas"]["tfidf_lr"]["w"] == 0.1 and y["semillas"] == [13, 42, 77, 123, 2026]
    assert all(len(y["particion"][p]["sha256_ids"]) == 64 for p in ("train", "val", "test"))
    sha = (tmp_path / "o" / "componente_A_frozen.sha256").read_text().split()[0]
    assert sha == registro.sha256_bytes(texto.encode("utf-8"))

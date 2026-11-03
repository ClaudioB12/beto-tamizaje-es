"""Métrica de tres clases y análisis de PI1 (short paper, §IV-D)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

import analizar_pi1
import registro
from estadistica import tres_clases as tc


def _tabla(y_ans, y_dep, p_ans, p_dep, umbral=0.5, sujetos=None):
    n = len(y_ans)
    p_ans, p_dep = np.asarray(p_ans, float), np.asarray(p_dep, float)
    return pd.DataFrame(
        {
            "fragmento_id": [f"f{i:04d}" for i in range(n)],
            "sujeto_id": sujetos if sujetos is not None else [f"s{i:04d}" for i in range(n)],
            "dominio": "reddit",
            "y_ansiedad": y_ans,
            "y_depresion": y_dep,
            "p_ansiedad": p_ans,
            "p_depresion": p_dep,
            "pred_ansiedad": (p_ans >= umbral).astype(int),
            "pred_depresion": (p_dep >= umbral).astype(int),
        }
    )


def test_clase_predicha_elige_la_mayor_entre_las_que_superan_el_umbral():
    t = _tabla([1, 1, 0, -1], [-1, -1, 0, 1], [0.9, 0.6, 0.2, 0.7], [0.95, 0.4, 0.3, 0.8])
    assert tc.clase_predicha(t).tolist() == [1, 0, 2, 1]


def test_clase_real_reddit_y_comorbilidad():
    t = _tabla([1, -1, 0, 1], [-1, 1, 0, 1], [0] * 4, [0] * 4)
    assert tc.clase_real(t).tolist() == [0, 1, 2, tc.DOBLE]


def test_clase_real_ambigua_falla():
    with pytest.raises(tc.EtiquetaAmbiguaError):
        tc.clase_real(_tabla([0], [-1], [0], [0]))


def test_macro_f1_coincide_con_sklearn_y_excluye_dobles():
    rng = np.random.default_rng(0)
    real = rng.integers(0, 3, 300)
    y_ans = np.where(real == 0, 1, np.where(real == 2, 0, -1))
    y_dep = np.where(real == 1, 1, np.where(real == 2, 0, -1))
    p_ans, p_dep = rng.random(300), rng.random(300)
    t = _tabla(y_ans, y_dep, p_ans, p_dep)
    esperado = f1_score(real, tc.clase_predicha(t), average="macro", labels=[0, 1, 2], zero_division=0)
    assert tc.macro_f1(t) == pytest.approx(esperado)
    # Un caso con doble etiqueta real no cambia la métrica.
    doble = _tabla([1], [1], [0.9], [0.1])
    con_doble = pd.concat([t, doble.assign(fragmento_id="zz")], ignore_index=True)
    assert tc.macro_f1(con_doble) == pytest.approx(esperado)


def test_bootstrap_por_sujeto_contiene_la_estimacion_y_es_pareado():
    rng = np.random.default_rng(1)
    n = 400
    real = rng.integers(0, 3, n)
    y_ans = np.where(real == 0, 1, np.where(real == 2, 0, -1))
    y_dep = np.where(real == 1, 1, np.where(real == 2, 0, -1))
    sujetos = [f"s{i // 3}" for i in range(n)]  # 3 publicaciones por sujeto
    def modelo(ruido, s):
        r = np.random.default_rng(s)
        return (np.clip((real == 0) + r.normal(0, ruido, n), 0, 1), np.clip((real == 1) + r.normal(0, ruido, n), 0, 1))
    bueno = {s: _tabla(y_ans, y_dep, *modelo(0.3, s), sujetos=sujetos) for s in (13, 42, 77)}
    rem = tc.Remuestreo(bueno[13], 2000, 2026)
    assert rem.n_sujetos == len(set(sujetos))
    dist, por_semilla = rem.distribucion(bueno)
    inf, sup = tc.intervalo(dist)
    assert inf <= np.mean(por_semilla) <= sup
    # Misma arquitectura contra sí misma: diferencia exactamente 0 en todas las réplicas.
    dist2, _ = rem.distribucion(bueno)
    assert np.all(dist - dist2 == 0)


def _escribir_carpeta(carpeta, tablas_por_arq):
    carpeta.mkdir(parents=True)
    archivos = []
    for arq, por_semilla in tablas_por_arq.items():
        for semilla, tabla in por_semilla.items():
            nombre = f"{arq}-s{semilla}-reddit.csv"
            tabla.to_csv(carpeta / nombre, index=False)
            archivos.append(
                {
                    "archivo": nombre,
                    "arquitectura": arq,
                    "semilla": semilla,
                    "dominio": "reddit",
                    "sha256": registro.sha256_bytes((carpeta / nombre).read_bytes()),
                }
            )
    registro.escribir_json(carpeta / "indice.json", {"archivos": archivos})


def test_analisis_completo_con_ablacion(tmp_path):
    rng = np.random.default_rng(2)
    n = 300
    real = rng.integers(0, 3, n)
    y_ans = np.where(real == 0, 1, np.where(real == 2, 0, -1))
    y_dep = np.where(real == 1, 1, np.where(real == 2, 0, -1))
    sujetos = [f"s{i // 2}" for i in range(n)]

    def tablas(ruido):
        out = {}
        for s in (13, 42, 77):
            r = np.random.default_rng(s + int(ruido * 100))
            out[s] = _tabla(
                y_ans, y_dep,
                np.clip((real == 0) + r.normal(0, ruido, n), 0, 1),
                np.clip((real == 1) + r.normal(0, ruido, n), 0, 1),
                sujetos=sujetos,
            )
        return out

    _escribir_carpeta(tmp_path / "reddit", {"beto_ajustado": tablas(0.2), "beto_sonda": tablas(0.6), "tfidf_lr": tablas(0.5)})
    _escribir_carpeta(tmp_path / "abl", {"beto_ajustado": tablas(0.4)})
    r = analizar_pi1.analizar(tmp_path / "reddit", tmp_path / "abl", 1000, 2026)
    ft = r["por_arquitectura"]["beto_ajustado"]["macro_f1_tres_clases"]
    lp = r["por_arquitectura"]["beto_sonda"]["macro_f1_tres_clases"]
    assert ft["media_semillas"] > lp["media_semillas"]
    assert r["h1_parte_reddit"] is True
    assert r["ablacion"][0]["diferencia_original_menos_ablacion"] > 0
    assert "BETO-FT" in analizar_pi1.resumen_markdown(r)
    assert "\\begin{tabular}" in analizar_pi1.tabla_latex(r)

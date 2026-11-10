"""Integridad de la búsqueda de w, partición de la ablación, polaridad y reanudación (auditoría C1)."""
import pandas as pd
import pytest

import auditar_busqueda
import calidad_traduccion
import comparar_particiones
import entrenar
import registro
from tests.test_pipeline import PARTICIONES


def _busqueda(raiz, semillas=(13, 42), pesos=(0.0, 0.5), archs=("beto_ajustado", "tfidf_lr"), sha="a" * 64):
    for s in semillas:
        c = raiz / f"s{s}" / "reddit"
        (c / "por_corrida").mkdir(parents=True)
        filas = []
        for a in archs:
            for w in pesos:
                f1 = 0.9 + w / 100 + s / 1e4
                registro.escribir_json(c / "por_corrida" / f"{a}-w{w:g}.json", {
                    "arquitectura": a, "peso_negativo_debil": w, "semilla": s, "humo": False,
                    "sha256_asignaciones": sha, "rejilla_umbrales": "0.01-0.99 paso 0.01", "f1_macro": f1})
                filas.append({"arquitectura": a, "w": w, "f1_macro": f1})
        pd.DataFrame(filas).to_csv(c / "tabla.csv", index=False)


def test_auditoria_acepta_una_busqueda_completa(tmp_path):
    _busqueda(tmp_path)
    r = auditar_busqueda.auditar(tmp_path, "reddit", [13, 42], [0, 0.5], ["beto_ajustado", "tfidf_lr"])
    assert r["ok"] and r["combinaciones"] == 8 and len(r["sha256_tablas"]) == 2


def test_auditoria_detecta_faltantes_humo_y_otra_particion(tmp_path):
    _busqueda(tmp_path)
    (tmp_path / "s42/reddit/por_corrida/tfidf_lr-w0.5.json").unlink()
    j = tmp_path / "s13/reddit/por_corrida/beto_ajustado-w0.json"
    registro.escribir_json(j, {**registro.leer_json(j), "humo": True, "sha256_asignaciones": "b" * 64})
    r = auditar_busqueda.auditar(tmp_path, "reddit", [13, 42], [0, 0.5], ["beto_ajustado", "tfidf_lr"])
    texto = " ".join(r["problemas"])
    assert not r["ok"]
    assert "falta la combinación ('tfidf_lr', 0.5, 42)" in texto and "humo" in texto and "particiones distintas" in texto


def test_auditoria_detecta_tabla_que_no_coincide_con_su_json(tmp_path):
    _busqueda(tmp_path)
    t = tmp_path / "s13/reddit/tabla.csv"
    pd.read_csv(t).assign(f1_macro=0.5).to_csv(t, index=False)
    assert not auditar_busqueda.auditar(tmp_path, "reddit", [13, 42], [0, 0.5], ["beto_ajustado", "tfidf_lr"])["ok"]


def test_comparar_particiones_iguales_y_distintas(tmp_path):
    a = PARTICIONES / "sintetico" / "asignaciones.csv"
    assert comparar_particiones.comparar(a, a)["iguales"]
    b = tmp_path / "b.csv"
    d = pd.read_csv(a, dtype=str)
    d.loc[0, "particion"] = "test" if d.loc[0, "particion"] != "test" else "train"
    d.drop(index=2).to_csv(b, index=False)
    r = comparar_particiones.comparar(a, b)
    assert not r["iguales"] and r["n_asignacion_distinta"] == 1 and r["n_solo_en_a"] == 1


def test_calidad_exige_la_columna_de_polaridad():
    base = pd.DataFrame({"fragmento_id": ["r0", "r1"], "adecuacion_1a5": [5, 4], "fluidez_1a5": [5, 4],
                         "senal_salud_mental_1a5": [5, 4], "aceptable_si_no": ["sí", "sí"]})
    with pytest.raises(ValueError, match="inversion_polaridad_si_no"):
        calidad_traduccion.comparar(base, base.copy())


def test_reanudar_rechaza_una_corrida_completa_con_otra_configuracion(tmp_path, config_diminuta, capsys):
    corr = tmp_path / "corridas"
    args = ["--fuente", "sintetico", "--config", str(config_diminuta), "--particiones", str(PARTICIONES),
            "--corridas", str(corr), "--arquitecturas", "tfidf_lr", "--semillas", "13", "--humo"]
    assert entrenar.main(args) == 0
    manifiesto = next(corr.rglob("manifiesto.json"))
    registro.escribir_json(manifiesto, {**registro.leer_json(manifiesto), "config_sha256": "0" * 64})
    assert entrenar.main(args) == 2
    assert "otra configuración" in capsys.readouterr().err

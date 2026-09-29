"""De punta a punta: entrenar -> evaluar -> analizar -> exportar.

Usa la partición congelada real del sintético, TF–IDF y un BERT diminuto, con
carpetas temporales para corridas, predicciones y resultados.
"""
import json
import shutil

import pandas as pd
import pytest

import analizar
import entrenar
import evaluar
import exportar
import registro
from tests.conftest import PARTICIONES


@pytest.fixture(scope="session")
def experimento(tmp_path_factory, config_diminuta):
    """Entrena y evalúa las tres arquitecturas con dos semillas, una sola vez."""
    raiz = tmp_path_factory.mktemp("experimento")
    carpetas = {nombre: raiz / nombre for nombre in ("corridas", "predicciones", "resultados", "exportados")}
    comunes = ["--fuente", "sintetico"]
    assert entrenar.main(
        [*comunes, "--config", str(config_diminuta), "--corridas", str(carpetas["corridas"]),
         "--particiones", str(PARTICIONES)]
    ) == 0
    assert evaluar.main(
        [*comunes, "--corridas", str(carpetas["corridas"]), "--particiones", str(PARTICIONES),
         "--predicciones", str(carpetas["predicciones"])]
    ) == 0
    return carpetas, config_diminuta


class TestCargaDeParticionCongelada:
    def test_reconstruye_la_particion_sin_fugas(self):
        particion, informe = registro.cargar_particion_congelada("sintetico", PARTICIONES)
        resumen = informe["particiones"]
        assert {n: len(df) for n, df in particion.por_nombre().items()} == {
            n: resumen[n]["fragmentos"] for n in ("train", "val", "test")
        }

    def test_detecta_asignaciones_modificadas(self, tmp_path):
        copia = tmp_path / "sintetico"
        shutil.copytree(PARTICIONES / "sintetico", copia)
        ruta = copia / "asignaciones.csv"
        ruta.write_text(ruta.read_text(encoding="utf-8").replace(",test", ",train", 1), encoding="utf-8")
        with pytest.raises(registro.ParticionInconsistenteError, match="se modificó"):
            registro.cargar_particion_congelada("sintetico", tmp_path)

    def test_detecta_que_la_entrada_cambio(self, tmp_path):
        copia = tmp_path / "sintetico"
        shutil.copytree(PARTICIONES / "sintetico", copia)
        informe = json.loads((copia / "informe.json").read_text(encoding="utf-8"))
        informe["entrada"]["sha256"] = "0" * 64
        (copia / "informe.json").write_text(json.dumps(informe), encoding="utf-8")
        with pytest.raises(registro.ParticionInconsistenteError, match="cambió"):
            registro.cargar_particion_congelada("sintetico", tmp_path)


class TestEntrenar:
    def test_crea_una_corrida_completa_por_arquitectura_y_semilla(self, experimento):
        carpetas, _ = experimento
        manifiestos = sorted((carpetas["corridas"] / "sintetico").glob("*/manifiesto.json"))
        nombres = sorted(m.parent.name for m in manifiestos)
        assert nombres == sorted(
            f"{a}-s{s}" for a in ("beto_ajustado", "beto_sonda", "tfidf_lr") for s in (13, 42)
        )
        for ruta in manifiestos:
            m = json.loads(ruta.read_text(encoding="utf-8"))
            assert m["estado"] == "completa" and not m["humo"]
            assert (ruta.parent / "modelo" / "clasificador.json").exists()
            assert (ruta.parent / "entorno.json").exists()
            assert not (ruta.parent / "trabajo").exists()

    def test_la_sonda_registra_su_barrido_y_elige_la_menor_perdida(self, experimento):
        carpetas, _ = experimento
        m = json.loads((carpetas["corridas"] / "sintetico" / "beto_sonda-s13" / "manifiesto.json").read_text(encoding="utf-8"))
        resultados = m["barrido_sonda"]["resultados"]
        assert [r["learning_rate"] for r in resultados] == [1.0e-2, 1.0e-4]
        mejor = min(resultados, key=lambda r: r["mejor_eval_loss"])
        assert m["barrido_sonda"]["learning_rate_elegida"] == mejor["learning_rate"]
        assert m["hiperparametros"]["learning_rate"] == mejor["learning_rate"]
        assert m["mejor_eval_loss"] == mejor["mejor_eval_loss"]

    def test_retoma_sin_repetir_lo_completo(self, tmp_path, config_diminuta):
        argumentos = ["--fuente", "sintetico", "--config", str(config_diminuta), "--arquitecturas", "tfidf_lr",
                      "--corridas", str(tmp_path), "--particiones", str(PARTICIONES)]
        assert entrenar.main(argumentos) == 0
        completa = tmp_path / "sintetico" / "tfidf_lr-s13" / "manifiesto.json"
        interrumpida = tmp_path / "sintetico" / "tfidf_lr-s42"
        marca_completa = completa.stat().st_mtime_ns
        (interrumpida / "manifiesto.json").unlink()  # simula un corte antes de terminar

        assert entrenar.main(argumentos) == 0
        assert completa.stat().st_mtime_ns == marca_completa
        assert (interrumpida / "manifiesto.json").exists()

    def test_rechaza_semillas_no_declaradas(self, tmp_path, config_diminuta):
        assert entrenar.main(
            ["--fuente", "sintetico", "--config", str(config_diminuta), "--semillas", "7",
             "--corridas", str(tmp_path), "--particiones", str(PARTICIONES)]
        ) == 2

    def test_la_prueba_de_humo_usa_su_propia_carpeta(self, tmp_path, config_diminuta):
        assert entrenar.main(
            ["--fuente", "sintetico", "--config", str(config_diminuta), "--humo", "--arquitecturas", "beto_sonda",
             "--semillas", "13", "--corridas", str(tmp_path), "--particiones", str(PARTICIONES)]
        ) == 0
        m = json.loads((tmp_path / "sintetico-humo" / "beto_sonda-s13" / "manifiesto.json").read_text(encoding="utf-8"))
        assert m["humo"] and m["hiperparametros"]["max_epochs"] == 1
        assert len(m["barrido_sonda"]["resultados"]) == 1
        assert not (tmp_path / "sintetico").exists()


class TestEvaluar:
    def test_umbrales_en_cada_corrida_y_predicciones_de_prueba(self, experimento):
        carpetas, _ = experimento
        indice = json.loads((carpetas["predicciones"] / "sintetico" / "indice.json").read_text(encoding="utf-8"))
        assert len(indice["archivos"]) == 6
        particion, _ = registro.cargar_particion_congelada("sintetico", PARTICIONES)
        for entrada in indice["archivos"]:
            tabla = pd.read_csv(carpetas["predicciones"] / "sintetico" / entrada["archivo"])
            assert set(tabla["fragmento_id"]) == set(particion.test["fragmento_id"])
            corrida = carpetas["corridas"] / "sintetico" / f"{entrada['arquitectura']}-s{entrada['semilla']}"
            assert (corrida / "umbrales.json").exists()

    def test_sin_corridas_falla_con_mensaje(self, tmp_path, capsys):
        assert evaluar.main(["--fuente", "sintetico", "--corridas", str(tmp_path), "--particiones", str(PARTICIONES),
                             "--predicciones", str(tmp_path / "p")]) == 1
        assert "entrenar.py" in capsys.readouterr().err


class TestAnalizar:
    def test_produce_resultados_de_las_tres_arquitecturas_y_h1(self, experimento):
        carpetas, _ = experimento
        assert analizar.main(["--fuente", "sintetico", "--bootstrap", "200", "--predicciones",
                              str(carpetas["predicciones"]), "--resultados", str(carpetas["resultados"])]) == 0
        r = json.loads((carpetas["resultados"] / "sintetico" / "resultados.json").read_text(encoding="utf-8"))
        assert [f["arquitectura"] for f in r["por_arquitectura"]] == ["beto_ajustado", "beto_sonda", "tfidf_lr"]
        assert all(f["semillas"] == [13, 42] for f in r["por_arquitectura"])
        assert [(c["a"], c["b"]) for c in r["comparaciones_h1"]] == [
            ("beto_ajustado", "beto_sonda"), ("beto_ajustado", "tfidf_lr")
        ]
        assert r["brechas_dominio"] == [] and r["concordancia_severidad"] == []
        assert any("NO REPORTAR" in a for a in r["advertencias"])
        resumen = (carpetas["resultados"] / "sintetico" / "resumen.md").read_text(encoding="utf-8")
        assert "NO REPORTAR" in resumen and "PE1" in resumen
        # El rótulo va pegado a cada tabla, no solo en la cabecera del documento.
        assert resumen.count("[NO CITABLE") == 2
        for bloque in resumen.split("[NO CITABLE")[1:]:
            assert bloque.lstrip(" —").split("\n", 2)[2].lstrip().startswith("|")
        assert "DE entre semillas" in resumen and "bootstrap sobre fragmentos" in resumen
        # TF–IDF es determinista: no puede contrastarse con Mann–Whitney.
        tfidf = next(f for f in r["por_arquitectura"] if f["arquitectura"] == "tfidf_lr")
        assert tfidf["determinista_entre_semillas"]
        contra_tfidf = next(c for c in r["comparaciones_h1"] if c["b"] == "tfidf_lr")
        assert contra_tfidf["prueba_semillas"]["mann_whitney_unilateral"] is None

    def test_se_niega_a_usar_predicciones_modificadas(self, experimento, tmp_path, capsys):
        carpetas, _ = experimento
        copia = tmp_path / "sintetico"
        shutil.copytree(carpetas["predicciones"] / "sintetico", copia)
        archivo = next(copia.glob("tfidf_lr-s13-*.csv"))
        archivo.write_text(archivo.read_text(encoding="utf-8").replace(",1,", ",0,", 1), encoding="utf-8")
        assert analizar.main(["--fuente", "sintetico", "--bootstrap", "50", "--predicciones", str(tmp_path),
                              "--resultados", str(tmp_path / "r")]) == 1
        assert "hash" in capsys.readouterr().err


class TestExportar:
    def test_elige_la_semilla_de_menor_perdida_de_validacion(self, experimento):
        carpetas, _ = experimento
        assert exportar.main(["--fuente", "sintetico", "--corridas", str(carpetas["corridas"]),
                              "--exportados", str(carpetas["exportados"])]) == 0
        perdidas = {
            s: json.loads((carpetas["corridas"] / "sintetico" / f"beto_ajustado-s{s}" / "manifiesto.json").read_text(encoding="utf-8"))["mejor_eval_loss"]
            for s in (13, 42)
        }
        elegida = min(perdidas, key=lambda s: (perdidas[s], s))
        destino = carpetas["exportados"] / "sintetico" / f"beto_ajustado-s{elegida}"
        exportacion = json.loads((destino / "exportacion.json").read_text(encoding="utf-8"))
        assert exportacion["semilla_elegida"] == elegida
        assert exportacion["solo_pruebas"] is True
        assert "modelo/clasificador.json" in exportacion["archivos"]
        assert (destino / "umbrales.json").exists()

    def test_no_sobrescribe_sin_permiso(self, experimento):
        carpetas, _ = experimento
        argumentos = ["--fuente", "sintetico", "--corridas", str(carpetas["corridas"]),
                      "--exportados", str(carpetas["exportados"] / "otra")]
        assert exportar.main(argumentos) == 0
        assert exportar.main(argumentos) == 1

    def test_desempate_determinista(self, tmp_path):
        candidatos = [(tmp_path / "b", {"semilla": 42, "mejor_eval_loss": 0.5}),
                      (tmp_path / "a", {"semilla": 13, "mejor_eval_loss": 0.5})]
        assert exportar.seleccionar_por_validacion(candidatos)[1]["semilla"] == 13

"""De punta a punta: entrenar -> evaluar -> analizar -> exportar.

Usa la partición congelada real del sintético, TF–IDF y un BERT diminuto, con
carpetas temporales para corridas, predicciones y resultados.
"""
import json
import shutil

import pandas as pd
import pytest

import analizar
import buscar_peso
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


    def test_no_mezcla_corridas_de_distinto_peso(self, tmp_path, config_diminuta, capsys):
        comunes = ["--fuente", "sintetico", "--arquitecturas", "tfidf_lr", "--semillas", "13",
                   "--corridas", str(tmp_path / "corridas"), "--particiones", str(PARTICIONES)]
        assert entrenar.main([*comunes, "--config", str(config_diminuta)]) == 0
        otra = tmp_path / "w05.yaml"
        otra.write_text(config_diminuta.read_text(encoding="utf-8") + "etiquetas: {peso_negativo_debil: 0.5}\n",
                        encoding="utf-8")
        assert entrenar.main([*comunes, "--config", str(otra)]) == 2
        assert "--corridas" in capsys.readouterr().err
        # En una carpeta propia sí entrena, y el manifiesto registra su w.
        assert entrenar.main([*comunes[:-4], "--corridas", str(tmp_path / "w05"), "--particiones", str(PARTICIONES),
                              "--config", str(otra)]) == 0
        m = json.loads((tmp_path / "w05" / "sintetico" / "tfidf_lr-s13" / "manifiesto.json").read_text(encoding="utf-8"))
        assert m["hiperparametros"]["peso_negativo_debil"] == 0.5


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

    def test_registra_w_y_la_confusion_cruzada_de_validacion(self, experimento):
        carpetas, _ = experimento
        indice = json.loads((carpetas["predicciones"] / "sintetico" / "indice.json").read_text(encoding="utf-8"))
        particion, _ = registro.cargar_particion_congelada("sintetico", PARTICIONES)
        controles_val = int(((particion.val["ansiedad"] == 0) & (particion.val["depresion"] == 0)).sum())
        for entrada in indice["archivos"]:
            assert entrada["peso_negativo_debil"] == 0.0
            # Se mide en validación, no en prueba: el conteo de controles lo delata.
            assert entrada["confusion_cruzada_validacion"]["control→alguna"]["filas"] == controles_val

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
        assert resumen.count("[NO CITABLE") == 3
        for bloque in resumen.split("[NO CITABLE")[1:]:
            assert bloque.lstrip(" —").split("\n", 2)[2].lstrip().startswith("|")
        assert "DE entre semillas" in resumen and "bootstrap sobre fragmentos" in resumen
        assert "Confusión cruzada" in resumen and "peso_negativo_debil (w): 0.0" in resumen
        assert r["peso_negativo_debil"] == 0.0
        for f in r["por_arquitectura"]:
            assert set(f["confusion_cruzada_media"]) == {"ansiedad→depresion", "depresion→ansiedad", "control→alguna"}
            assert set(f["confusion_cruzada_por_semilla"]) == {"13", "42"}  # JSON: claves texto
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


class TestBuscarPeso:
    def _argumentos(self, tmp_path, config, *extra):
        return ["--fuente", "sintetico", "--config", str(config), "--particiones", str(PARTICIONES),
                "--resultados", str(tmp_path / "res"), "--trabajo", str(tmp_path / "trabajo"), *extra]

    def test_una_fila_por_arquitectura_y_peso_medida_en_validacion(self, tmp_path, config_diminuta, monkeypatch):
        particion, _ = registro.cargar_particion_congelada("sintetico", PARTICIONES)
        medidos = []
        original = buscar_peso.medir_en_validacion

        def espia(carpeta, df):
            medidos.append(set(df["fragmento_id"]))
            return original(carpeta, df)

        monkeypatch.setattr(buscar_peso, "medir_en_validacion", espia)
        argumentos = self._argumentos(tmp_path, config_diminuta, "--pesos", "0", "0.5",
                                      "--arquitecturas", "beto_ajustado", "tfidf_lr", "--humo")
        assert buscar_peso.main(argumentos) == 0

        assert medidos and all(m == set(particion.val["fragmento_id"]) for m in medidos)
        salida = tmp_path / "res" / "sintetico-humo"
        tabla = pd.read_csv(salida / "tabla.csv")
        assert sorted(zip(tabla["arquitectura"], tabla["w"])) == [
            ("beto_ajustado", 0.0), ("beto_ajustado", 0.5), ("tfidf_lr", 0.0), ("tfidf_lr", 0.5)
        ]
        assert {"ansiedad→depresion", "depresion→ansiedad", "control→alguna", "f1_macro"} <= set(tabla.columns)
        resumen = (salida / "resumen.md").read_text(encoding="utf-8")
        assert "VALIDACIÓN" in resumen and "NO REPORTAR" in resumen
        # Los modelos no se conservan: solo sus métricas.
        assert not any((tmp_path / "trabajo").rglob("clasificador.json"))

    def test_retoma_sin_repetir_lo_medido(self, tmp_path, config_diminuta, capsys):
        argumentos = self._argumentos(tmp_path, config_diminuta, "--pesos", "0", "1", "--arquitecturas", "tfidf_lr")
        assert buscar_peso.main(argumentos) == 0
        # El sintético no tiene -1: el script debe advertir que w no cambia nada.
        assert "w no cambia nada" in capsys.readouterr().out
        assert buscar_peso.main(argumentos) == 0
        assert capsys.readouterr().out.count("ya medida") == 2

    def test_repite_si_cambian_los_hiperparametros(self, tmp_path, config_diminuta, capsys):
        argumentos = self._argumentos(tmp_path, config_diminuta, "--pesos", "0", "--arquitecturas", "tfidf_lr")
        assert buscar_peso.main(argumentos) == 0
        archivo = tmp_path / "res" / "sintetico" / "por_corrida" / "tfidf_lr-w0.json"
        previa = registro.leer_json(archivo)
        assert len(previa["sha256_config"]) == 64
        # Huella distinta (otros hiperparámetros): no se reutiliza.
        registro.escribir_json(archivo, {**previa, "sha256_config": "0" * 64})
        capsys.readouterr()
        assert buscar_peso.main(argumentos) == 0
        assert "otra configuración, se repite" in capsys.readouterr().out
        # Registro anterior a la huella: se reutiliza con aviso.
        previa = registro.leer_json(archivo)
        registro.escribir_json(archivo, {k: v for k, v in previa.items() if k != "sha256_config"})
        assert buscar_peso.main(argumentos) == 0
        assert "sin huella de configuración" in capsys.readouterr().out

    def test_rechaza_pesos_fuera_de_0_1_y_semillas_no_declaradas(self, tmp_path, config_diminuta):
        assert buscar_peso.main(self._argumentos(tmp_path, config_diminuta, "--pesos", "1.5")) == 2
        assert buscar_peso.main(self._argumentos(tmp_path, config_diminuta, "--semilla", "7")) == 2

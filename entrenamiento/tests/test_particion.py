import json

import pandas as pd
import pytest

import particionar as cli
from configuracion import ConfiguracionInvalidaError, cargar_config_particion
from datos.particion import FugaDeDatosError, estrato, particionar
from tests.conftest import construir_df


class TestSinFugas:
    def test_ningun_sujeto_cruza_particiones(self, particion):
        tr, va, te = (set(p["sujeto_id"]) for p in particion.por_nombre().values())
        assert not (tr & va) and not (tr & te) and not (va & te)

    def test_ningun_texto_cruza_particiones(self, particion):
        # Este test habría detectado la fuga del train.py actual.
        tr, va, te = (set(p["texto_normalizado"]) for p in particion.por_nombre().values())
        assert not (tr & va) and not (tr & te) and not (va & te)

    def test_ninguna_variante_de_frase_base_cruza_hacia_prueba(self, particion):
        # En el sintético, "sujeto" = frase base: la garantía que falló en train.py,
        # donde el 100 % de la validación tenía su frase base en entrenamiento.
        vistas = set(particion.train["sujeto_id"]) | set(particion.val["sujeto_id"])
        assert not vistas & set(particion.test["sujeto_id"])

    def test_detecta_un_texto_compartido_entre_sujetos(self):
        # Sin deduplicar, un mensaje genérico escrito por muchos sujetos acaba en
        # varias particiones: debe fallar en vez de devolver la partición.
        filas = []
        for i in range(40):
            etiquetas = [(1, 0), (0, 1), (0, 0)][i % 3]
            filas.append((f"a{i}", f"s{i}", "gracias", *etiquetas))
            filas.append((f"b{i}", f"s{i}", f"mensaje propio {i}", *etiquetas))
        with pytest.raises(FugaDeDatosError, match="texto_normalizado"):
            particionar(construir_df(filas))


class TestReparto:
    def test_cada_fragmento_queda_en_exactamente_una_particion(self, particion, df_deduplicado):
        asignaciones = particion.asignaciones()
        assert len(asignaciones) == len(df_deduplicado)
        assert asignaciones["fragmento_id"].is_unique
        assert set(asignaciones["fragmento_id"]) == set(df_deduplicado["fragmento_id"])

    def test_proporciones_aproximadas_al_ppi(self, particion):
        total = sum(len(p) for p in particion.por_nombre().values())
        for p, esperado in zip(particion.por_nombre().values(), (0.70, 0.10, 0.20)):
            assert abs(len(p) / total - esperado) < 0.05

    def test_todas_las_particiones_contienen_todos_los_estratos(self, particion, df_deduplicado):
        esperados = set(estrato(df_deduplicado))
        for nombre, p in particion.por_nombre().items():
            assert set(estrato(p)) == esperados, nombre


class TestReproducibilidad:
    def test_misma_semilla_misma_particion(self, df_deduplicado):
        a = particionar(df_deduplicado, semilla=42).asignaciones()
        b = particionar(df_deduplicado, semilla=42).asignaciones()
        pd.testing.assert_frame_equal(a, b)

    def test_otra_semilla_otra_particion(self, df_deduplicado):
        a = particionar(df_deduplicado, semilla=42).asignaciones()
        b = particionar(df_deduplicado, semilla=13).asignaciones()
        assert not a.equals(b)


class TestValidacionDeParametros:
    @pytest.mark.parametrize("proporciones", [(0.6, 0.1, 0.2), (0.7, 0.2, 0.1, 0.0), (0.8, 0.0, 0.2)])
    def test_rechaza_proporciones_mal_formadas(self, df_deduplicado, proporciones):
        with pytest.raises(ValueError):
            particionar(df_deduplicado, proporciones)

    def test_rechaza_fracciones_que_no_son_un_pliegue(self, df_deduplicado):
        with pytest.raises(ValueError, match="1/k"):
            particionar(df_deduplicado, (0.70, 0.15, 0.15))

    def test_rechaza_pocos_sujetos(self):
        df = construir_df([(f"f{i}", f"s{i % 3}", f"texto {i}", 0, 0) for i in range(9)])
        with pytest.raises(ValueError, match="sujetos"):
            particionar(df)


class TestConfiguracion:
    def test_lee_el_archivo_del_experimento(self):
        config = cargar_config_particion()
        assert config.proporciones == (0.70, 0.10, 0.20)
        assert config.semilla == 42

    def test_rechaza_semilla_booleana(self, tmp_path):
        ruta = tmp_path / "exp.yaml"
        ruta.write_text("particion:\n  proporciones: [0.7, 0.1, 0.2]\n  semilla: true\n", encoding="utf-8")
        with pytest.raises(ConfiguracionInvalidaError, match="semilla"):
            cargar_config_particion(ruta)


class TestComando:
    def test_congela_y_no_sobrescribe_sin_permiso(self, tmp_path, capsys):
        salida = tmp_path / "sintetico"
        assert cli.main(["--fuente", "sintetico", "--salida", str(salida)]) == 0

        asignaciones = pd.read_csv(salida / "asignaciones.csv")
        informe = json.loads((salida / "informe.json").read_text(encoding="utf-8"))
        assert set(asignaciones["particion"]) == {"train", "val", "test"}
        assert informe["configuracion"] == {"proporciones": [0.7, 0.1, 0.2], "semilla": 42}
        assert len(informe["entrada"]["sha256"]) == 64

        assert cli.main(["--fuente", "sintetico", "--salida", str(salida)]) == 1
        assert "--sobrescribir" in capsys.readouterr().err

    def test_el_hash_de_las_asignaciones_es_estable(self, tmp_path):
        hashes = []
        for nombre in ("a", "b"):
            salida = tmp_path / nombre
            cli.main(["--fuente", "sintetico", "--salida", str(salida)])
            informe = json.loads((salida / "informe.json").read_text(encoding="utf-8"))
            hashes.append(informe["sha256_asignaciones"])
        assert hashes[0] == hashes[1]

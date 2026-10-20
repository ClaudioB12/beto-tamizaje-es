"""Fuente Reddit: preparación, troceo para la traducción y cargador."""
import pandas as pd
import pytest

import preparar_reddit as prep
from datos import reddit
from datos.esquema import DESCONOCIDA, NEGATIVA, POSITIVA
from datos.particion import particionar
from datos.preparacion import deduplicar
from traducir_reddit import diagnosticar, muestra_estratificada, oraciones_en_ingles, recortar_palabras, trocear


def _crudo(filas: list[tuple]) -> pd.DataFrame:
    """Filas (author, etiqueta, post) con el formato de los CSV de Zenodo."""
    df = pd.DataFrame(filas, columns=["author", "etiqueta", "post"])
    df["subreddit"] = df["etiqueta"].map({"depresion": "depression", "ansiedad": "Anxiety", "control": "parenting"})
    df["date"] = "2018/01/01"
    return df


def _texto(n: int, semilla: str = "") -> str:
    return " ".join(f"palabra{semilla}{i}" for i in range(n))


def _preparar(crudo, **cambios):
    parametros = dict(por_clase=1000, max_por_autor=5, min_palabras=3, semilla=42, sal="test")
    parametros.update(cambios)
    return prep.preparar(crudo, **parametros)


class TestLimpieza:
    def test_quita_enlaces_y_menciones(self):
        limpio = prep.limpiar("Mira https://x.com/a y www.y.org, gracias u/pepe_1 &amp; /u/ana")
        assert "http" not in limpio and "www" not in limpio
        assert "pepe" not in limpio and "ana" not in limpio
        assert "&" in limpio

    def test_anonimiza_de_forma_estable_y_sin_el_nombre(self):
        a, b = prep.anonimizar("usuario1", "sal"), prep.anonimizar("usuario1", "sal")
        assert a == b and "usuario1" not in a
        assert prep.anonimizar("usuario1", "otra") != a


class TestPreparar:
    def test_descarta_eliminados_cortos_y_duplicados(self):
        crudo = _crudo(
            [
                ("a1", "depresion", _texto(12)),
                ("a2", "depresion", _texto(12).upper()),  # duplicado sin distinguir mayúsculas
                ("a3", "depresion", "[removed]"),
                ("a4", "depresion", "muy corto"),
                ("[deleted]", "control", _texto(12, "c")),
            ]
        )
        salida, traza = _preparar(crudo)
        assert len(salida) == 1
        assert traza["tras_duplicados"] == 1

    def test_excluye_autores_que_publican_en_dos_clases(self):
        crudo = _crudo(
            [
                ("ambos", "depresion", _texto(12, "a")),
                ("ambos", "ansiedad", _texto(12, "b")),
                ("solo", "control", _texto(12, "c")),
            ]
        )
        salida, traza = _preparar(crudo)
        assert traza["autores_en_varias_clases_excluidos"] == 1
        assert set(salida["etiqueta"]) == {"control"}

    def test_limita_publicaciones_por_autor_y_balancea(self):
        filas = [("prolifico", "depresion", _texto(12, f"p{i}")) for i in range(10)]
        filas += [(f"c{i}", "control", _texto(12, f"c{i}")) for i in range(10)]
        salida, traza = _preparar(_crudo(filas), max_por_autor=3, por_clase=4)
        assert traza["seleccionadas_por_clase"] == {"control": 4, "depresion": 3}
        assert salida.groupby("sujeto_id").size().max() <= 3

    def test_no_guarda_el_nombre_de_usuario(self):
        salida, _ = _preparar(_crudo([("nombre_real", "control", _texto(12))]))
        assert "author" not in salida.columns
        assert not salida.astype(str).apply(lambda c: c.str.contains("nombre_real")).any().any()


class TestVerificarArchivos:
    def test_falla_si_falta_un_archivo(self, tmp_path):
        with pytest.raises(prep.ArchivoInvalidoError, match="Faltan"):
            prep.verificar_archivos(tmp_path)

    def test_falla_si_el_md5_no_coincide(self, tmp_path):
        for nombre in prep.ARCHIVOS:
            (tmp_path / nombre).write_text("subreddit,author,date,post\n", encoding="utf-8")
        with pytest.raises(prep.ArchivoInvalidoError, match="MD5"):
            prep.verificar_archivos(tmp_path)


class TestTroceo:
    contar = staticmethod(lambda t: len(t.split()))

    def test_respeta_el_limite_y_conserva_las_palabras(self):
        texto = "Una dos tres. Cuatro cinco seis siete. Ocho nueve. " + _texto(25) + "."
        trozos = trocear(texto, 10, self.contar)
        assert all(self.contar(t) <= 10 for t in trozos)
        assert " ".join(trozos).split() == texto.split()

    def test_texto_corto_queda_en_un_trozo(self):
        assert trocear("Hola. Estoy bien.", 10, self.contar) == ["Hola. Estoy bien."]

    def test_recorta_por_palabras(self):
        assert recortar_palabras(_texto(10), 4).split() == _texto(4).split()


def _traducido(tmp_path, filas: list[tuple]) -> "Path":
    """Filas (sujeto, etiqueta, texto_es) -> CSV con el formato de traducir_reddit."""
    df = pd.DataFrame(filas, columns=["sujeto_id", "etiqueta", "texto_es"])
    df.insert(0, "fragmento_id", [f"rd-{i:06d}" for i in range(len(df))])
    df["subreddit"] = "x"
    df["fecha"] = "2018/01/01"
    df["texto_en"] = "english"
    ruta = tmp_path / "reddit_es.csv"
    df.to_csv(ruta, index=False)
    return ruta


class TestCargador:
    def test_codifica_las_etiquetas_con_la_decision_d2(self, tmp_path):
        ruta = _traducido(
            tmp_path,
            [
                ("s1", "depresion", "Me siento vacío desde hace semanas."),
                ("s2", "ansiedad", "No puedo dejar de preocuparme por todo."),
                ("s3", "control", "Mi hijo empezó el colegio hoy."),
            ],
        )
        df = reddit.cargar_clasificacion(ruta).set_index("etiqueta_original")
        assert tuple(df.loc["depresion", ["ansiedad", "depresion"]]) == (DESCONOCIDA, POSITIVA)
        assert tuple(df.loc["ansiedad", ["ansiedad", "depresion"]]) == (POSITIVA, DESCONOCIDA)
        assert tuple(df.loc["control", ["ansiedad", "depresion"]]) == (NEGATIVA, NEGATIVA)
        assert set(df["dominio"]) == {"reddit"}

    def test_falla_si_quedan_filas_sin_traducir(self, tmp_path):
        ruta = _traducido(tmp_path, [("s1", "control", ""), ("s2", "control", "hola que tal")])
        with pytest.raises(ValueError, match="sin texto_es"):
            reddit.cargar_clasificacion(ruta)

    def test_la_variante_enmascarada_oculta_la_etiqueta(self, tmp_path):
        ruta = _traducido(tmp_path, [("s1", "depresion", "Mi depresión y mi ansiedad empeoran, estoy deprimido.")])
        texto = reddit.cargar_clasificacion_enmascarada(ruta)["texto"].iloc[0]
        assert "depres" not in texto.lower() and "ansiedad" not in texto.lower()
        assert "deprimido" not in texto.lower()
        assert texto.count(reddit.MASCARA) == 3

    def test_se_particiona_por_sujeto_sin_fugas(self, tmp_path):
        filas = []
        for clase in ("depresion", "ansiedad", "control"):
            for s in range(20):
                for k in range(2):
                    filas.append((f"{clase}-{s}", clase, f"texto {clase} {s} {k} " + _texto(5, f"{clase}{s}{k}")))
        df, _ = deduplicar(reddit.cargar_clasificacion(_traducido(tmp_path, filas)))
        particion = particionar(df, semilla=42)
        partes = particion.por_nombre()
        assert sum(len(p) for p in partes.values()) == len(df)
        assert not set(partes["train"]["sujeto_id"]) & set(partes["test"]["sujeto_id"])


class TestMuestraCalidad:
    def test_incluye_columnas_para_los_evaluadores(self):
        df = pd.DataFrame(
            {
                "fragmento_id": [f"f{i}" for i in range(60)],
                "etiqueta": ["depresion", "ansiedad", "control"] * 20,
                "texto_en": [_texto(5 + i) for i in range(60)],
                "texto_es": ["es"] * 60,
            }
        )
        muestra = muestra_estratificada(df, n=18, semilla=1)
        assert {"adecuacion_1a5", "fluidez_1a5", "senal_salud_mental_1a5", "aceptable_si_no"} <= set(muestra.columns)
        assert set(muestra["etiqueta"]) == {"depresion", "ansiedad", "control"}


class TestDiagnostico:
    @staticmethod
    def _df(filas):
        df = pd.DataFrame(filas, columns=["texto_en", "texto_es"])
        df.insert(0, "fragmento_id", [f"f{i}" for i in range(len(df))])
        return df

    def test_marca_la_traduccion_que_omite_oraciones(self):
        en = _texto(100)
        diag = diagnosticar(self._df([(en, _texto(60)), (en, _texto(98))]), max_palabras=400)
        assert diag["fragmento_id"].tolist() == ["f0"]
        assert diag["motivos"].iloc[0] == "cobertura"

    def test_la_cobertura_usa_solo_las_palabras_traducidas(self):
        diag = diagnosticar(self._df([(_texto(500), _texto(390))]), max_palabras=400)
        assert diag.empty

    def test_marca_bucles_que_no_estan_en_el_original(self):
        diag = diagnosticar(self._df([("Help me now please friends", "¡¡¡¡¡¡¡¡¡¡¡¡ ayuda ahora amigos")]), 400)
        assert diag["motivos"].iloc[0] == "repeticion"

    def test_detecta_titulos_sin_traducir(self):
        es = "New to Corporate America and I have no idea what to do. He tenido ansiedad toda mi vida."
        assert oraciones_en_ingles(es) == ["New to Corporate America and I have no idea what to do."]
        assert oraciones_en_ingles("Me he perdido a mí mismo. No sé qué hacer con mi vida.") == []

    def test_las_filas_sin_traducir_no_se_diagnostican(self):
        assert diagnosticar(self._df([(_texto(50), "")]), 400).empty

    def test_origen_no_ingles(self):
        diag = diagnosticar(self._df([("20190413 發病日，整日睡睡醒醒四次，進入連續的夢境", "20190413 ,,")]), 400)
        assert diag["motivos"].iloc[0] == "origen_no_ingles"


class TestTraduccionSinLetras:
    def test_el_diagnostico_no_cuenta_simbolos_como_palabras(self):
        df = TestDiagnostico._df([(_texto(60), " ".join(["•"] * 60))])
        assert diagnosticar(df, 400)["motivos"].iloc[0].startswith("cobertura")

    def test_el_cargador_descarta_traducciones_sin_letras(self, tmp_path):
        ruta = _traducido(tmp_path, [("s1", "control", "Mi hijo empezó el colegio."), ("s2", "control", "¡¡¡¡¡¡¡¡¡¡¡¡")])
        assert reddit.cargar_clasificacion(ruta)["sujeto_id"].tolist() == ["s1"]


class TestOrigenNoIngles:
    def test_detecta_otros_alfabetos(self):
        assert reddit.es_origen_no_ingles("發病日，整日睡睡醒醒四次")
        assert not reddit.es_origen_no_ingles("I feel so tired, café and naïve 😂")

    def test_el_cargador_los_excluye(self, tmp_path):
        ruta = _traducido(tmp_path, [("s1", "control", "Mi hijo empezó el colegio."), ("s2", "control", "20190413 ,,")])
        df = pd.read_csv(ruta, dtype=str)
        df.loc[1, "texto_en"] = "發病日，整日睡睡醒醒四次"
        df.to_csv(ruta, index=False)
        assert reddit.cargar_clasificacion(ruta)["sujeto_id"].tolist() == ["s1"]


class TestMuestraRecortada:
    def test_el_original_se_recorta_a_lo_traducido(self):
        df = pd.DataFrame(
            {
                "fragmento_id": [f"f{i}" for i in range(30)],
                "etiqueta": ["depresion", "ansiedad", "control"] * 10,
                "texto_en": [_texto(10 + 50 * i) for i in range(30)],
                "texto_es": ["es"] * 30,
            }
        )
        muestra = muestra_estratificada(df, n=9, semilla=1, max_palabras=400)
        assert muestra["texto_en"].str.split().str.len().max() <= 400
        largos = muestra[muestra["palabras_original"] > 400]
        assert (largos["recortado"] == "sí").all()
        assert (muestra.loc[muestra["palabras_original"] <= 400, "recortado"] == "no").all()

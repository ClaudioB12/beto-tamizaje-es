"""Cargador sintético y deduplicación."""
from datos.preparacion import deduplicar
from tests.conftest import construir_df


class TestCargadorSintetico:
    def test_conserva_todas_las_filas(self, df_sintetico):
        assert len(df_sintetico) == 535

    def test_agrupa_las_variantes_por_frase_base(self, df_sintetico):
        # 95 frases base; cada una aparece 5 veces (9 en control).
        tamanos = df_sintetico.groupby("sujeto_id").size()
        assert len(tamanos) == 95
        assert set(tamanos) == {5, 9}

    def test_traduce_la_etiqueta_unica_a_dos_binarias(self, df_sintetico):
        esperado = {"ansiedad": (1, 0), "depresion": (0, 1), "control": (0, 0)}
        for original, grupo in df_sintetico.groupby("etiqueta_original"):
            emocion = original.split("_")[0]
            pares = set(zip(grupo["ansiedad"], grupo["depresion"]))
            assert pares == {esperado[emocion]}


class TestDeduplicar:
    def test_deja_una_fila_por_texto_normalizado(self):
        df = construir_df(
            [
                ("f1", "s1", "No puedo dormir.", 1, 0),
                ("f2", "s2", "no puedo DORMIR", 1, 0),
                ("f3", "s3", "estoy bien", 0, 0),
            ]
        )
        limpio, informe = deduplicar(df)
        assert list(limpio["fragmento_id"]) == ["f1", "f3"]
        assert informe.duplicados_eliminados == 1
        assert informe.filas_en_conflicto_eliminadas == 0

    def test_descarta_todas_las_copias_si_las_etiquetas_no_coinciden(self):
        df = construir_df(
            [
                ("f1", "s1", "gracias", 1, 0),
                ("f2", "s2", "Gracias!", 0, 1),
                ("f3", "s3", "estoy bien", 0, 0),
            ]
        )
        limpio, informe = deduplicar(df)
        assert list(limpio["fragmento_id"]) == ["f3"]
        assert informe.textos_en_conflicto == 1
        assert informe.filas_en_conflicto_eliminadas == 2

    def test_desconocida_frente_a_conocida_cuenta_como_conflicto(self):
        df = construir_df(
            [
                ("f1", "s1", "me cuesta todo", 1, -1),
                ("f2", "s2", "me cuesta todo", 1, 0),
                ("f3", "s3", "estoy bien", 0, 0),
            ]
        )
        limpio, _ = deduplicar(df)
        assert list(limpio["fragmento_id"]) == ["f3"]

    def test_el_sintetico_tiene_duplicados_pero_ningun_conflicto(self, df_sintetico):
        _, informe = deduplicar(df_sintetico)
        assert informe.duplicados_eliminados > 0
        assert informe.textos_en_conflicto == 0

"""Las tres arquitecturas.

BETO se prueba con un BERT diminuto (2 capas, 32 dimensiones) que usa el
tokenizador real de BETO: los tests no descargan nada y terminan en segundos,
pero ejercitan el mismo código que entrenará el modelo completo.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from configuracion import (
    ConfigEntrenamiento,
    ConfigTfidf,
    ConfiguracionInvalidaError,
    HiperparametrosBeto,
    cargar_config_entrenamiento,
)
from modelos import construir_clasificador
from modelos.tfidf import ClasificadorTfidf
from tests.conftest import construir_df

# ---------------------------------------------------------------------------
# Fixtures (beto_diminuto vive en conftest.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def hiper(beto_diminuto) -> HiperparametrosBeto:
    return HiperparametrosBeto(
        modelo_base=str(beto_diminuto),
        revision=None,
        max_length=32,
        batch_size=8,
        batch_por_dispositivo=4,
        learning_rate=5.0e-3,
        weight_decay=0.0,
        dropout=0.1,
        max_epochs=2,
        paciencia=1,
    )


@pytest.fixture
def datos_pequenos(particion):
    return particion.train.head(40), particion.val.head(16)


# ---------------------------------------------------------------------------
# Pérdida enmascarada y colador
# ---------------------------------------------------------------------------


class TestPerdidaEnmascarada:
    def test_coincide_con_bce_sobre_las_etiquetas_conocidas(self):
        from torch.nn.functional import binary_cross_entropy_with_logits

        from modelos.beto import perdida_enmascarada

        logits = torch.tensor([[2.0, -1.0], [0.5, 3.0]])
        etiquetas = torch.tensor([[1.0, -1.0], [0.0, 1.0]])
        esperado = binary_cross_entropy_with_logits(
            torch.tensor([2.0, 0.5, 3.0]), torch.tensor([1.0, 0.0, 1.0])
        )
        assert torch.isclose(perdida_enmascarada(logits, etiquetas), esperado)

    def test_una_etiqueta_desconocida_no_recibe_gradiente(self):
        from modelos.beto import perdida_enmascarada

        logits = torch.tensor([[2.0, -1.0], [0.5, 3.0]], requires_grad=True)
        etiquetas = torch.tensor([[1.0, -1.0], [0.0, 1.0]])
        perdida_enmascarada(logits, etiquetas).backward()
        assert logits.grad[0, 1] == 0
        assert (logits.grad[[0, 1, 1], [0, 0, 1]] != 0).all()

    def test_rechaza_un_lote_sin_etiquetas_conocidas(self):
        from modelos.beto import perdida_enmascarada

        with pytest.raises(ValueError, match="conocida"):
            perdida_enmascarada(torch.zeros(2, 2), torch.full((2, 2), -1.0))


def test_colador_rellena_y_conserva_el_menos_uno(beto_diminuto):
    from transformers import AutoTokenizer

    from modelos.beto import ColadorEtiquetas

    tokenizer = AutoTokenizer.from_pretrained(beto_diminuto)
    ejemplos = [
        {**tokenizer("hola"), "labels": [1.0, -1.0]},
        {**tokenizer("me siento muy mal hoy"), "labels": [0.0, 1.0]},
    ]
    lote = ColadorEtiquetas(tokenizer)(ejemplos)
    assert lote["input_ids"].shape[0] == 2
    assert lote["input_ids"].shape[1] == len(ejemplos[1]["input_ids"])
    assert lote["labels"].dtype == torch.float32
    assert lote["labels"].tolist() == [[1.0, -1.0], [0.0, 1.0]]


# ---------------------------------------------------------------------------
# Acumulación de gradiente
# ---------------------------------------------------------------------------


def _pesos_tras_un_paso(clase_trainer, hiper, datos, batch_por_dispositivo, acumulacion, carpeta):
    """Un paso de SGD sin recorte: la actualización es proporcional al gradiente.

    Con Adam este test no serviría: su primer paso normaliza el gradiente y
    oculta un error de escala.
    """
    from transformers import TrainingArguments

    from modelos.beto import ClasificadorBeto, ColadorEtiquetas, _DatasetTextos

    clasificador = ClasificadorBeto("beto_ajustado", 7, hiper.con(dropout=0.0)).construir()
    argumentos = TrainingArguments(
        output_dir=str(carpeta),
        per_device_train_batch_size=batch_por_dispositivo,
        gradient_accumulation_steps=acumulacion,
        max_steps=1,
        learning_rate=0.5,
        optim="sgd",
        weight_decay=0.0,
        max_grad_norm=0.0,
        lr_scheduler_type="constant",
        save_strategy="no",
        report_to="none",
        seed=7,
    )
    trainer = clase_trainer(
        model=clasificador.modelo,
        args=argumentos,
        train_dataset=_DatasetTextos(datos, clasificador.tokenizer, hiper.max_length, con_etiquetas=True),
        data_collator=ColadorEtiquetas(clasificador.tokenizer),
    )
    trainer.train()
    return trainer.model.classifier.weight.detach().clone()


class TestAcumulacionDeGradiente:
    def test_dos_microlotes_de_4_equivalen_a_un_lote_de_8(self, hiper, particion, tmp_path):
        from modelos.beto import TrainerEnmascarado

        datos = particion.train.head(8)
        un_lote = _pesos_tras_un_paso(TrainerEnmascarado, hiper, datos, 8, 1, tmp_path / "a")
        acumulado = _pesos_tras_un_paso(TrainerEnmascarado, hiper, datos, 4, 2, tmp_path / "b")
        assert torch.allclose(un_lote, acumulado, atol=1e-5)

    def test_control_sin_la_correccion_el_paso_sale_distinto(self, hiper, particion, tmp_path):
        # Demuestra que el test anterior detecta el error: sin forzar
        # model_accepts_loss_kwargs = False, el gradiente acumulado se duplica.
        from modelos.beto import TrainerEnmascarado

        class TrainerSinCorreccion(TrainerEnmascarado):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.model_accepts_loss_kwargs = True

        datos = particion.train.head(8)
        un_lote = _pesos_tras_un_paso(TrainerSinCorreccion, hiper, datos, 8, 1, tmp_path / "a")
        acumulado = _pesos_tras_un_paso(TrainerSinCorreccion, hiper, datos, 4, 2, tmp_path / "b")
        assert not torch.allclose(un_lote, acumulado, atol=1e-5)


# ---------------------------------------------------------------------------
# Construcción y entrenamiento de BETO
# ---------------------------------------------------------------------------


class TestConstruccionBeto:
    def test_la_sonda_congela_todo_el_codificador(self, hiper):
        from modelos.beto import ClasificadorBeto

        modelo = ClasificadorBeto("beto_sonda", 13, hiper).construir().modelo
        assert not any(p.requires_grad for p in modelo.base_model.parameters())
        assert all(p.requires_grad for p in modelo.classifier.parameters())

    def test_la_sonda_solo_entrena_la_capa_lineal(self, hiper):
        from modelos.beto import ClasificadorBeto

        modelo = ClasificadorBeto("beto_sonda", 13, hiper).construir().modelo
        entrenables = {nombre for nombre, p in modelo.named_parameters() if p.requires_grad}
        assert entrenables == {"classifier.weight", "classifier.bias"}
        # Ningún parámetro congelado viene de un pooler recién inicializado.
        assert not [nombre for nombre, _ in modelo.named_parameters() if "pooler" in nombre]

    def test_la_sonda_lee_el_vector_cls_sin_transformar(self, hiper, particion):
        from modelos.beto import ClasificadorBeto

        clasificador = ClasificadorBeto("beto_sonda", 13, hiper).construir()
        modelo = clasificador.modelo.eval()
        entradas = clasificador.tokenizer(
            particion.val["texto"].head(4).tolist(), padding=True, return_tensors="pt"
        )
        with torch.no_grad():
            cls = modelo.base_model(**entradas).last_hidden_state[:, 0]
            esperado = modelo.classifier(cls)
            obtenido = modelo(**entradas).logits
        assert torch.allclose(obtenido, esperado, atol=1e-6)

    def test_el_ajuste_fino_entrena_todo(self, hiper):
        from modelos.beto import ClasificadorBeto

        modelo = ClasificadorBeto("beto_ajustado", 13, hiper).construir().modelo
        assert all(p.requires_grad for p in modelo.parameters())

    def test_la_semilla_fija_la_inicializacion_de_la_cabeza(self, hiper):
        from modelos.beto import ClasificadorBeto

        def cabeza(semilla):
            return ClasificadorBeto("beto_ajustado", semilla, hiper).construir().modelo.classifier.weight

        assert torch.equal(cabeza(13), cabeza(13))
        assert not torch.equal(cabeza(13), cabeza(42))

    def test_dos_salidas_multietiqueta(self, hiper):
        from modelos.beto import ClasificadorBeto

        config = ClasificadorBeto("beto_ajustado", 13, hiper).construir().modelo.config
        assert config.num_labels == 2
        assert config.problem_type == "multi_label_classification"
        assert config.id2label == {0: "ansiedad", 1: "depresion"}


class TestEntrenamientoBeto:
    def test_la_sonda_no_modifica_el_codificador_al_entrenar(self, hiper, datos_pequenos, tmp_path):
        from modelos.beto import ClasificadorBeto

        clasificador = ClasificadorBeto("beto_sonda", 13, hiper).construir()
        codificador_antes = clasificador.modelo.base_model.embeddings.word_embeddings.weight.clone()
        cabeza_antes = clasificador.modelo.classifier.weight.clone()

        clasificador.entrenar(*datos_pequenos, carpeta_trabajo=tmp_path)

        assert torch.equal(clasificador.modelo.base_model.embeddings.word_embeddings.weight, codificador_antes)
        assert not torch.equal(clasificador.modelo.classifier.weight, cabeza_antes)

    def test_el_ajuste_fino_si_modifica_el_codificador(self, hiper, datos_pequenos, tmp_path):
        from modelos.beto import ClasificadorBeto

        clasificador = ClasificadorBeto("beto_ajustado", 13, hiper).construir()
        antes = clasificador.modelo.base_model.encoder.layer[0].attention.self.query.weight.clone()
        clasificador.entrenar(*datos_pequenos, carpeta_trabajo=tmp_path)
        assert not torch.equal(clasificador.modelo.base_model.encoder.layer[0].attention.self.query.weight, antes)

    @pytest.mark.parametrize("arquitectura", ["beto_ajustado", "beto_sonda"])
    def test_entrena_predice_guarda_y_recarga(self, hiper, datos_pequenos, tmp_path, arquitectura):
        # En la sonda, la recarga debe volver a quitar el pooler que
        # from_pretrained recrea al azar; si no, las predicciones cambiarían.
        from modelos.beto import ClasificadorBeto

        train, val = datos_pequenos
        clasificador = ClasificadorBeto(arquitectura, 13, hiper).construir()
        informe = clasificador.entrenar(train, val, carpeta_trabajo=tmp_path / "trabajo")

        assert informe["mejor_eval_loss"] is not None
        probabilidades = clasificador.predecir_probabilidades(val)
        assert probabilidades.shape == (len(val), 2)
        assert ((probabilidades >= 0) & (probabilidades <= 1)).all()

        clasificador.guardar(tmp_path / "exportado")
        recargado = ClasificadorBeto.cargar(tmp_path / "exportado")
        assert recargado.arquitectura == arquitectura and recargado.semilla == 13
        np.testing.assert_allclose(recargado.predecir_probabilidades(val), probabilidades, atol=1e-6)


# ---------------------------------------------------------------------------
# TF–IDF
# ---------------------------------------------------------------------------


class TestTfidf:
    CONFIG = ConfigTfidf(ngram_max=2, C=1.0, max_iter=1000)

    def test_predice_una_probabilidad_por_etiqueta(self, particion):
        clasificador = ClasificadorTfidf(42, self.CONFIG)
        clasificador.entrenar(particion.train)
        probabilidades = clasificador.predecir_probabilidades(particion.test)
        assert probabilidades.shape == (len(particion.test), 2)
        assert ((probabilidades >= 0) & (probabilidades <= 1)).all()

    def test_ignora_las_filas_con_etiqueta_desconocida(self, particion):
        train = particion.train.copy()
        train.loc[train.index[:30], "ansiedad"] = -1
        clasificador = ClasificadorTfidf(42, self.CONFIG)
        informe = clasificador.entrenar(train)
        # Si el -1 se colara como clase, la regresión tendría tres clases.
        assert list(clasificador.modelos["ansiedad"].classes_) == [0, 1]
        assert informe["filas_entrenamiento_por_etiqueta"]["ansiedad"] == len(train) - 30
        assert informe["filas_entrenamiento_por_etiqueta"]["depresion"] == len(train)

    def test_rechaza_una_etiqueta_con_una_sola_clase(self):
        df = construir_df([(f"f{i}", f"s{i}", f"texto {i}", 0, i % 2) for i in range(10)])
        with pytest.raises(ValueError, match="ansiedad"):
            ClasificadorTfidf(42, self.CONFIG).entrenar(df)

    def test_guarda_y_recarga(self, particion, tmp_path):
        clasificador = ClasificadorTfidf(42, self.CONFIG)
        clasificador.entrenar(particion.train)
        clasificador.guardar(tmp_path)
        recargado = ClasificadorTfidf.cargar(tmp_path)
        np.testing.assert_array_equal(
            recargado.predecir_probabilidades(particion.test),
            clasificador.predecir_probabilidades(particion.test),
        )


# ---------------------------------------------------------------------------
# Configuración y fábrica
# ---------------------------------------------------------------------------


def _yaml(tmp_path, **cambios) -> Path:
    base = {
        "modelo_base": "dccuchile/bert-base-spanish-wwm-cased",
        "revision_modelo_base": "c4d86612f51b4f46759c8390d1798c2febe71b93",
        "max_length": "256",
        "batch_size": "16",
        "batch_por_dispositivo": "4",
        "learning_rate": "2.0e-5",
        "weight_decay": "0.01",
        "dropout": "0.1",
        "max_epochs": "5",
        "parada_temprana": "{metrica: eval_loss, paciencia: 2}",
        "semillas": "[13, 42, 77, 123, 2026]",
        "arquitecturas": "[beto_ajustado, beto_sonda, tfidf_lr]",
        "sonda": "{learning_rate: [1.0e-3, 5.0e-4], max_epochs: 20}",
        "tfidf": "{ngram_max: 2, C: 1.0, max_iter: 1000}",
    }
    base.update(cambios)
    ruta = tmp_path / "exp.yaml"
    ruta.write_text("\n".join(f"{k}: {v}" for k, v in base.items() if v is not None), encoding="utf-8")
    return ruta


class TestConfigEntrenamiento:
    def test_lee_el_archivo_del_experimento(self):
        config = cargar_config_entrenamiento()
        assert config.beto.revision == "c4d86612f51b4f46759c8390d1798c2febe71b93"
        assert config.beto.batch_size == 16 and config.beto.acumulacion == 1
        assert config.semillas == (13, 42, 77, 123, 2026)

    def test_la_acumulacion_mantiene_el_lote_efectivo(self, tmp_path):
        assert cargar_config_entrenamiento(_yaml(tmp_path)).beto.acumulacion == 4

    def test_exige_revision_para_un_modelo_del_hub(self, tmp_path):
        with pytest.raises(ConfiguracionInvalidaError, match="revision"):
            cargar_config_entrenamiento(_yaml(tmp_path, revision_modelo_base=None))

    def test_rechaza_lote_no_multiplo(self, tmp_path):
        with pytest.raises(ConfiguracionInvalidaError, match="múltiplo"):
            cargar_config_entrenamiento(_yaml(tmp_path, batch_por_dispositivo="5"))

    def test_rechaza_lote_por_dispositivo_cero(self, tmp_path):
        with pytest.raises(ConfiguracionInvalidaError, match="batch_por_dispositivo"):
            cargar_config_entrenamiento(_yaml(tmp_path, batch_por_dispositivo="0"))

    def test_avisa_de_la_notacion_cientifica_que_yaml_lee_como_texto(self, tmp_path):
        with pytest.raises(ConfiguracionInvalidaError, match="2.0e-5"):
            cargar_config_entrenamiento(_yaml(tmp_path, learning_rate="2e-5"))

    def test_rechaza_una_metrica_de_parada_distinta(self, tmp_path):
        with pytest.raises(ConfiguracionInvalidaError, match="eval_loss"):
            cargar_config_entrenamiento(_yaml(tmp_path, parada_temprana="{metrica: f1, paciencia: 2}"))


class TestFabrica:
    def test_la_sonda_exige_una_tasa_del_barrido(self):
        config = cargar_config_entrenamiento()
        with pytest.raises(ValueError, match="learning_rate_sonda"):
            construir_clasificador("beto_sonda", 13, config)
        with pytest.raises(ValueError, match="barrido"):
            construir_clasificador("beto_sonda", 13, config, learning_rate_sonda=0.3)

    def test_la_sonda_hereda_todo_salvo_tasa_y_epocas(self):
        config: ConfigEntrenamiento = cargar_config_entrenamiento()
        sonda = construir_clasificador("beto_sonda", 13, config, learning_rate_sonda=1.0e-3)
        assert sonda.hiper.learning_rate == 1.0e-3
        assert sonda.hiper.max_epochs == config.sonda_max_epochs
        assert sonda.hiper.con(learning_rate=config.beto.learning_rate, max_epochs=config.beto.max_epochs) == config.beto

    def test_cada_arquitectura_devuelve_su_clase(self):
        config = cargar_config_entrenamiento()
        assert construir_clasificador("tfidf_lr", 13, config).arquitectura == "tfidf_lr"
        assert construir_clasificador("beto_ajustado", 13, config).arquitectura == "beto_ajustado"
        with pytest.raises(ValueError, match="desconocida"):
            construir_clasificador("svm", 13, config)

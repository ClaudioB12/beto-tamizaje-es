from pathlib import Path

import pandas as pd
import pytest

from datos import sintetico
from datos.normalizacion import normalizar_texto
from datos.particion import particionar
from datos.preparacion import deduplicar

RAIZ = Path(__file__).resolve().parents[1]
TOKENIZADOR_BETO = RAIZ.parent / "ml-service" / "modelos" / "clasificacion"
PARTICIONES = RAIZ / "particiones"


@pytest.fixture(scope="session")
def df_sintetico() -> pd.DataFrame:
    return sintetico.cargar_clasificacion()


@pytest.fixture(scope="session")
def df_deduplicado(df_sintetico) -> pd.DataFrame:
    limpio, _ = deduplicar(df_sintetico)
    return limpio


@pytest.fixture(scope="session")
def particion(df_deduplicado):
    return particionar(df_deduplicado, semilla=42)


@pytest.fixture(scope="session")
def beto_diminuto(tmp_path_factory) -> Path:
    """BERT de 2 capas con el tokenizador real de BETO: sin descargas, en segundos."""
    import torch
    from transformers import AutoTokenizer, BertConfig, BertModel

    carpeta = tmp_path_factory.mktemp("beto_diminuto")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZADOR_BETO)
    torch.manual_seed(0)
    # Sin pooler, igual que el checkpoint real de BETO (preentrenado solo con
    # MLM). Con pooler, los tests no habrían revelado que la sonda quedaba
    # leyendo una proyección aleatoria congelada.
    BertModel(
        BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
        ),
        add_pooling_layer=False,
    ).save_pretrained(carpeta)
    tokenizer.save_pretrained(carpeta)
    return carpeta


@pytest.fixture(scope="session")
def config_diminuta(tmp_path_factory, beto_diminuto) -> Path:
    """Configuración de experimento válida que apunta al BERT diminuto."""
    ruta = tmp_path_factory.mktemp("config") / "experimento.yaml"
    ruta.write_text(
        f"""
particion: {{proporciones: [0.70, 0.10, 0.20], semilla: 42}}
modelo_base: "{beto_diminuto.as_posix()}"
revision_modelo_base: null
max_length: 32
batch_size: 8
batch_por_dispositivo: 4
learning_rate: 5.0e-3
weight_decay: 0.0
dropout: 0.1
max_epochs: 2
parada_temprana: {{metrica: eval_loss, paciencia: 1}}
semillas: [13, 42]
arquitecturas: [beto_ajustado, beto_sonda, tfidf_lr]
sonda: {{learning_rate: [1.0e-2, 1.0e-4], max_epochs: 2}}
tfidf: {{ngram_max: 2, C: 1.0, max_iter: 1000}}
""",
        encoding="utf-8",
    )
    return ruta


def construir_df(filas: list[tuple]) -> pd.DataFrame:
    """Filas (fragmento_id, sujeto_id, texto, ansiedad, depresion) -> esquema válido."""
    df = pd.DataFrame(filas, columns=["fragmento_id", "sujeto_id", "texto", "ansiedad", "depresion"])
    df["texto_normalizado"] = df["texto"].map(normalizar_texto)
    df["dominio"] = "sintetico"
    df["ansiedad"] = df["ansiedad"].astype("int64")
    df["depresion"] = df["depresion"].astype("int64")
    return df

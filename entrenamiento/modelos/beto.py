"""BETO con ajuste fino y BETO con sonda lineal (PPI §2.2, niveles a y b de la VI).

Ambos comparten modelo base, cabeza de clasificación y pérdida. La única
diferencia es que la sonda congela el codificador y solo entrena la capa de
clasificación.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from configuracion import HiperparametrosBeto
from datos.esquema import ETIQUETAS

ARQUITECTURAS_BETO = ("beto_ajustado", "beto_sonda")


class _PrimerToken(torch.nn.Module):
    """Sustituye al pooler en la sonda: devuelve el vector [CLS] sin transformar.

    El checkpoint de BETO se preentrenó solo con MLM y NO trae pesos del pooler
    (verificado: 205 tensores, ninguno ``bert.pooler``). transformers lo crea
    con pesos aleatorios. En el ajuste fino eso no importa porque se entrena;
    en la sonda quedaría congelado, y la capa lineal leería una proyección
    aleatoria de BETO en vez de sus representaciones. Esto debilitaría
    artificialmente al rival de H1.
    """

    def forward(self, estados_ocultos: torch.Tensor) -> torch.Tensor:
        return estados_ocultos[:, 0]


def perdida_enmascarada(logits: torch.Tensor, etiquetas: torch.Tensor) -> torch.Tensor:
    """BCE solo sobre las etiquetas conocidas; -1 no aporta gradiente (D2).

    Promedia sobre las etiquetas conocidas del lote, no sobre filas: una fila
    con sus dos etiquetas conocidas pesa el doble que una con una sola.
    """
    conocidas = etiquetas >= 0
    if not conocidas.any():
        raise ValueError(
            "Lote sin ninguna etiqueta conocida. El esquema lo impide fila a fila; "
            "revisa el cargador que produjo estos datos."
        )
    return binary_cross_entropy_with_logits(logits[conocidas], etiquetas[conocidas].float())


class TrainerEnmascarado(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # El forward de BERT acepta **kwargs, así que el Trainer supone que la
        # pérdida ya viene normalizada por lote efectivo y NO la divide entre los
        # pasos de acumulación. Con una pérdida propia eso multiplicaría el
        # gradiente por `gradient_accumulation_steps`: con batch_por_dispositivo 4
        # (GPU de 4 GB) el experimento no sería el mismo que en Colab.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        entradas = {clave: valor for clave, valor in inputs.items() if clave != "labels"}
        salidas = model(**entradas)
        perdida = perdida_enmascarada(salidas.logits, inputs["labels"])
        return (perdida, salidas) if return_outputs else perdida


class _DatasetTextos(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int, con_etiquetas: bool):
        self._codificado = tokenizer(df["texto"].tolist(), truncation=True, max_length=max_length)
        self._etiquetas = (
            df[list(ETIQUETAS)].to_numpy(dtype="float32") if con_etiquetas else None
        )

    def __len__(self) -> int:
        return len(self._codificado["input_ids"])

    def __getitem__(self, i: int) -> dict:
        ejemplo = {clave: valores[i] for clave, valores in self._codificado.items()}
        if self._etiquetas is not None:
            ejemplo["labels"] = self._etiquetas[i].tolist()
        return ejemplo


class ColadorEtiquetas:
    """Rellena las entradas al largo del lote y apila las etiquetas como float.

    Se hace a mano para que el -1 de las etiquetas desconocidas llegue intacto
    a la pérdida, sin depender de cómo cada versión de transformers trate la
    clave "labels" al rellenar.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, ejemplos: list[dict]) -> dict:
        sin_etiquetas = [{k: v for k, v in e.items() if k != "labels"} for e in ejemplos]
        lote = dict(self.tokenizer.pad(sin_etiquetas, return_tensors="pt"))
        if "labels" in ejemplos[0]:
            lote["labels"] = torch.tensor([e["labels"] for e in ejemplos], dtype=torch.float32)
        return lote


class ClasificadorBeto:
    def __init__(self, arquitectura: str, semilla: int, hiper: HiperparametrosBeto):
        if arquitectura not in ARQUITECTURAS_BETO:
            raise ValueError(f"Arquitectura {arquitectura!r} no es BETO: {ARQUITECTURAS_BETO}")
        self.arquitectura = arquitectura
        self.semilla = semilla
        self.hiper = hiper
        self.modelo = None
        self.tokenizer = None

    def construir(self) -> "ClasificadorBeto":
        # La semilla se fija ANTES de crear el modelo: la cabeza de clasificación
        # se inicializa al azar aquí, y esa inicialización es parte de lo que
        # varía entre las cinco semillas.
        set_seed(self.semilla)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hiper.modelo_base, revision=self.hiper.revision
        )
        self.modelo = AutoModelForSequenceClassification.from_pretrained(
            self.hiper.modelo_base,
            revision=self.hiper.revision,
            num_labels=len(ETIQUETAS),
            problem_type="multi_label_classification",
            id2label=dict(enumerate(ETIQUETAS)),
            label2id={etiqueta: i for i, etiqueta in enumerate(ETIQUETAS)},
            hidden_dropout_prob=self.hiper.dropout,
            attention_probs_dropout_prob=self.hiper.dropout,
        )
        self._preparar_arquitectura()
        return self

    def _preparar_arquitectura(self) -> None:
        """Convierte el modelo en sonda lineal: [CLS] congelado -> capa lineal.

        Se aplica también al cargar: el pooler sustituido no tiene pesos, así
        que ``from_pretrained`` vuelve a crearlo al azar y hay que reemplazarlo
        de nuevo. Por eso un modelo de sonda debe cargarse siempre con
        ``ClasificadorBeto.cargar``, nunca con ``from_pretrained`` a secas.
        """
        if self.arquitectura != "beto_sonda":
            return
        self.modelo.base_model.pooler = _PrimerToken()
        for parametro in self.modelo.base_model.parameters():
            parametro.requires_grad = False

    def entrenar(self, train: pd.DataFrame, val: pd.DataFrame, carpeta_trabajo: Path) -> dict:
        """Entrena con parada temprana sobre la pérdida de validación.

        Al terminar, ``self.modelo`` es el punto de control con menor
        ``eval_loss``, no el de la última época.
        """
        if self.modelo is None:
            self.construir()
        h = self.hiper
        argumentos = TrainingArguments(
            output_dir=str(carpeta_trabajo),
            per_device_train_batch_size=h.batch_por_dispositivo,
            per_device_eval_batch_size=h.batch_por_dispositivo * 2,
            gradient_accumulation_steps=h.acumulacion,
            learning_rate=h.learning_rate,
            weight_decay=h.weight_decay,
            num_train_epochs=h.max_epochs,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=1,
            seed=self.semilla,
            data_seed=self.semilla,
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=torch.cuda.is_available(),
            report_to="none",
        )
        trainer = TrainerEnmascarado(
            model=self.modelo,
            args=argumentos,
            train_dataset=_DatasetTextos(train, self.tokenizer, h.max_length, con_etiquetas=True),
            eval_dataset=_DatasetTextos(val, self.tokenizer, h.max_length, con_etiquetas=True),
            data_collator=ColadorEtiquetas(self.tokenizer),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=h.paciencia)],
        )
        trainer.train()
        self.modelo = trainer.model

        estado = trainer.state
        return {
            "mejor_eval_loss": estado.best_metric,
            "mejor_punto_de_control": estado.best_model_checkpoint,
            "epocas_completadas": estado.epoch,
            "pasos": estado.global_step,
            "historial": estado.log_history,
        }

    @torch.no_grad()
    def predecir_probabilidades(self, df: pd.DataFrame) -> np.ndarray:
        """Matriz (n, 2): probabilidad de positivo, en el orden de ``ETIQUETAS``."""
        if self.modelo is None:
            raise RuntimeError("El clasificador no está construido, entrenado ni cargado")
        self.modelo.eval()
        dispositivo = next(self.modelo.parameters()).device
        cargador = DataLoader(
            _DatasetTextos(df, self.tokenizer, self.hiper.max_length, con_etiquetas=False),
            batch_size=self.hiper.batch_por_dispositivo * 2,
            collate_fn=ColadorEtiquetas(self.tokenizer),
        )
        partes = []
        for lote in cargador:
            lote = {clave: valor.to(dispositivo) for clave, valor in lote.items()}
            partes.append(torch.sigmoid(self.modelo(**lote).logits).float().cpu().numpy())
        return np.concatenate(partes) if partes else np.empty((0, len(ETIQUETAS)))

    def guardar(self, carpeta: Path) -> None:
        carpeta.mkdir(parents=True, exist_ok=True)
        self.modelo.save_pretrained(carpeta)
        self.tokenizer.save_pretrained(carpeta)
        (carpeta / "clasificador.json").write_text(
            json.dumps(
                {
                    "arquitectura": self.arquitectura,
                    "semilla": self.semilla,
                    "hiperparametros": self.hiper.como_dict(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def cargar(cls, carpeta: Path) -> "ClasificadorBeto":
        meta = json.loads((carpeta / "clasificador.json").read_text(encoding="utf-8"))
        clasificador = cls(meta["arquitectura"], meta["semilla"], HiperparametrosBeto(**meta["hiperparametros"]))
        clasificador.tokenizer = AutoTokenizer.from_pretrained(carpeta)
        clasificador.modelo = AutoModelForSequenceClassification.from_pretrained(carpeta)
        clasificador._preparar_arquitectura()
        return clasificador

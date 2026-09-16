"""Pipeline de entrenamiento y fine-tuning de BETO para Agente BETO.

Diseñado y optimizado para GPUs locales de 4GB VRAM (como GTX 1050 Ti) o CPU.

Uso:
  python train.py --task clasificacion --epochs 4
  python train.py --task crisis --epochs 4
"""
import argparse
import csv
import json
import os
import random
import sys
from typing import Dict, List, Tuple

import torch
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# Fijar semillas
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODELO_BASE = "dccuchile/bert-base-spanish-wwm-cased"
MAX_LENGTH = 128  # Optimizado para mensajes conversacionales y 4GB VRAM

# ------------------------------------------------------------------------------
# Clases Dataset PyTorch
# ------------------------------------------------------------------------------

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


# ------------------------------------------------------------------------------
# Carga de datos y preparación
# ------------------------------------------------------------------------------

def cargar_datos_clasificacion(ruta_csv: str) -> Tuple[List[str], List[int], Dict[str, int], Dict[int, str]]:
    textos = []
    etiquetas = []
    
    with open(ruta_csv, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            textos.append(row["texto"])
            etiquetas.append(row["etiqueta_unificada"])
            
    clases_unicas = sorted(list(set(etiquetas)))
    label2id = {c: i for i, c in enumerate(clases_unicas)}
    id2label = {i: c for i, c in enumerate(clases_unicas)}
    
    etiquetas_id = [label2id[e] for e in etiquetas]
    return textos, etiquetas_id, label2id, id2label


def cargar_datos_crisis(ruta_csv: str) -> Tuple[List[str], List[int], Dict[str, int], Dict[int, str]]:
    textos = []
    etiquetas = []
    
    with open(ruta_csv, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            textos.append(row["texto"])
            etiquetas.append(int(row["es_crisis"]))
            
    label2id = {"no_crisis": 0, "crisis": 1}
    id2label = {0: "no_crisis", 1: "crisis"}
    return textos, etiquetas, label2id, id2label


# ------------------------------------------------------------------------------
# Función de métricas
# ------------------------------------------------------------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    acc = (preds == labels).mean()
    return {
        "accuracy": float(acc),
        "f1_macro": float(f1),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
    }


# ------------------------------------------------------------------------------
# Entrenamiento principal
# ------------------------------------------------------------------------------

def entrenar(task: str, epochs: int, batch_size: int, lr: float):
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print(f"[INFO] Iniciando entrenamiento de BETO: Tarea [{task.upper()}]")
    print(f"Dispositivo de cómputo detectado: {dispositivo.upper()}")
    if dispositivo == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / (1024**2):.0f} MB")
    print("=" * 60)

    # 1. Cargar datos según la tarea
    if task == "clasificacion":
        ruta_csv = os.path.join(directorio_actual, "dataset_clasificacion.csv")
        textos, etiquetas, label2id, id2label = cargar_datos_clasificacion(ruta_csv)
    elif task == "crisis":
        ruta_csv = os.path.join(directorio_actual, "dataset_crisis.csv")
        textos, etiquetas, label2id, id2label = cargar_datos_crisis(ruta_csv)
    else:
        raise ValueError(f"Tarea desconocida: {task}")

    # Split estratificado 80% train / 20% val
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        textos, etiquetas, test_size=0.2, random_state=42, stratify=etiquetas
    )

    print(f"Muestras de entrenamiento: {len(train_texts)}")
    print(f"Muestras de validación:    {len(val_texts)}")
    print(f"Número de clases:         {len(label2id)} -> {list(label2id.keys())}")

    # 2. Tokenización
    print(f"\nDescargando / cargando tokenizador [{MODELO_BASE}]...")
    tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE, do_lower_case=False)

    train_encodings = tokenizer(
        train_texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt"
    )
    val_encodings = tokenizer(
        val_texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt"
    )

    train_dataset = TextDataset(train_encodings, train_labels)
    val_dataset = TextDataset(val_encodings, val_labels)

    # 3. Cargar Modelo BETO
    print(f"Cargando arquitectura BETO con cabeza de clasificación ({len(label2id)} clases)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODELO_BASE,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    # 4. Configurar argumentos de entrenamiento (optimizados para 4GB VRAM)
    salida_checkpoints = os.path.join(directorio_actual, f"checkpoints_{task}")
    directorio_modelo_final = os.path.join(directorio_actual, "..", "ml-service", "modelos", task)
    os.makedirs(directorio_modelo_final, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=salida_checkpoints,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=2,
        weight_decay=0.01,
        learning_rate=lr,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 5. Ejecutar Entrenamiento
    print("\nIniciando optimización de pesos (Fine-Tuning)...")
    trainer.train()

    # 6. Evaluación final
    print("\nEvaluando mejor modelo en conjunto de validación...")
    eval_results = trainer.evaluate()
    print("\n" + "=" * 50)
    print("[EVALUACION] RESULTADOS:")
    for k, v in eval_results.items():
        if isinstance(v, float):
            print(f"  - {k}: {v:.4f}")
    print("=" * 50)

    # 7. Guardar modelo y tokenizador para el microservicio de inferencia
    print(f"\n[GUARDADO] Guardando artefactos entrenados en: {directorio_modelo_final}")
    model.save_pretrained(directorio_modelo_final)
    tokenizer.save_pretrained(directorio_modelo_final)

    # Guardar metadatos y mapeo de clases
    with open(os.path.join(directorio_modelo_final, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2, ensure_ascii=False)

    print("¡Entrenamiento y exportación finalizados con éxito!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar BETO para Agente BETO")
    parser.add_argument(
        "--task",
        type=str,
        default="clasificacion",
        choices=["clasificacion", "crisis"],
        help="Tarea a entrenar: clasificacion o crisis",
    )
    parser.add_argument("--epochs", type=int, default=4, help="Número de épocas")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size por dispositivo (8 recomendado para 4GB)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate para AdamW")

    args = parser.parse_args()
    entrenar(task=args.task, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

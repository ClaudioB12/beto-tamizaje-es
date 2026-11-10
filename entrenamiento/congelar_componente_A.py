"""Congela en un solo archivo la configuración del Componente A antes de tocar la prueba.

Escribe <salida>/componente_A_frozen.yaml y <salida>/componente_A_frozen.sha256 con:
arquitecturas e hiperparámetros, w por arquitectura (de la decisión), semillas, umbrales
(rejilla y regla, elegidos solo en validación), regla de decisión de tres clases,
hashes de la partición (total y por train/val/test), del CSV de datos, de la lista de
términos de la ablación y, si se indica, del parche y del commit base.

Si la decisión es provisional, el archivo lo dice: no sirve para tocar la prueba.

Uso:
    python congelar_componente_A.py --decision resultados/decision_w.json --fuente reddit --salida resultados/
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
from pathlib import Path

import pandas as pd
import yaml

import registro
from configuracion import RUTA_POR_DEFECTO, cargar_config_entrenamiento
from datos.reddit import RUTA_TERMINOS
from evaluacion.umbrales import DESCRIPCION_REJILLA


def hashes_particion(asignaciones: Path) -> dict:
    a = pd.read_csv(asignaciones, dtype=str)
    salida = {"asignaciones_csv": registro.sha256_archivo(asignaciones)}
    for nombre in ("train", "val", "test"):
        filas = a.loc[a["particion"] == nombre, ["fragmento_id", "sujeto_id"]].sort_values("fragmento_id")
        texto = "".join(f"{f},{s}\n" for f, s in filas.itertuples(index=False))
        salida[nombre] = {"fragmentos": int(len(filas)), "sujetos": int(filas["sujeto_id"].nunique()),
                          "sha256_ids": hashlib.sha256(texto.encode()).hexdigest()}
    return salida


def construir(config_ruta: Path, decision: dict, fuente: str, particiones: Path, extra: dict) -> dict:
    cfg = cargar_config_entrenamiento(config_ruta)
    b = cfg.beto
    w = decision["w"]
    inf = registro.leer_json(particiones / fuente / "informe.json")
    return {
        "estado": "provisional (no habilita la prueba)" if decision.get("provisional") else "definitivo",
        "decision": {k: decision.get(k) for k in ("fecha", "aprobado_por", "justificacion_sonda", "provisional")},
        "arquitecturas": {
            "beto_ajustado": {"modelo_base": b.modelo_base, "revision": b.revision, "ajuste": "completo",
                              "max_length": b.max_length, "optimizador": "AdamW", "learning_rate": b.learning_rate,
                              "batch_size": b.batch_size, "max_epochs": b.max_epochs,
                              "parada_temprana": {"metrica": "eval_loss", "paciencia": b.paciencia},
                              "dropout": b.dropout, "weight_decay": b.weight_decay, "w": w["beto_ajustado"]},
            "beto_sonda": {"modelo_base": b.modelo_base, "codificador": "congelado", "cabeza": "lineal sobre [CLS]",
                           "learning_rate_elegida_en_validacion_entre": list(cfg.sonda_learning_rates),
                           "max_epochs": cfg.sonda_max_epochs, "w": w["beto_sonda"]},
            "tfidf_lr": {"ngramas": [1, cfg.tfidf.ngram_max], "modelo": "regresión logística uno-contra-el-resto",
                         "C": cfg.tfidf.C, "max_iter": cfg.tfidf.max_iter, "w": w["tfidf_lr"]},
        },
        "semillas": list(cfg.semillas),
        "umbrales": {"seleccion": "solo en validación, por etiqueta, maximizando F1 en filas con la etiqueta conocida",
                     "rejilla": DESCRIPCION_REJILLA},
        "regla_tres_clases": "etiqueta de mayor probabilidad entre las que superan su umbral; control si ninguna",
        "metrica_primaria": "macro-F1 de tres clases (ansiedad, depresión, control); casos con doble etiqueta aparte",
        "bootstrap": {"replicas": 10000, "unidad": "sujeto", "principal": "semillas promediadas en cada réplica",
                      "sensibilidad": "jerárquico (semillas y luego sujetos)", "ic": "percentil 95 %"},
        "particion": {"fuente": fuente, "sha256_asignaciones": inf["sha256_asignaciones"],
                      "sha256_datos_entrada": inf["entrada"]["sha256"],
                      **hashes_particion(particiones / fuente / "asignaciones.csv")},
        "ablacion": {"terminos": RUTA_TERMINOS.name, "sha256_terminos": registro.sha256_archivo(RUTA_TERMINOS)},
        "config_base": {"archivo": config_ruta.name, "sha256": registro.sha256_archivo(config_ruta)},
        **({"codigo": extra} if extra else {}),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--decision", type=Path, required=True)
    ap.add_argument("--fuente", default="reddit")
    ap.add_argument("--config", type=Path, default=RUTA_POR_DEFECTO)
    ap.add_argument("--particiones", type=Path, default=registro.CARPETA_PARTICIONES)
    ap.add_argument("--parche", type=Path, default=None)
    ap.add_argument("--commit-base", default=None)
    ap.add_argument("--salida", type=Path, required=True)
    a = ap.parse_args(argv)
    extra = {}
    if a.parche:
        extra["sha256_parche"] = registro.sha256_archivo(a.parche)
    if a.commit_base:
        extra["commit_base"] = a.commit_base
    congelado = construir(a.config, registro.leer_json(a.decision), a.fuente, a.particiones, extra)
    texto = yaml.safe_dump(congelado, allow_unicode=True, sort_keys=False)
    a.salida.mkdir(parents=True, exist_ok=True)
    (a.salida / "componente_A_frozen.yaml").write_text(texto, encoding="utf-8")
    h = hashlib.sha256(texto.encode("utf-8")).hexdigest()
    (a.salida / "componente_A_frozen.sha256").write_text(f"{h}  componente_A_frozen.yaml\n", encoding="utf-8")
    print(texto)
    print("sha256:", h)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())

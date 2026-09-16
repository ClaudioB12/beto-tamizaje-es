"""Baseline clásico: TF–IDF + regresión logística (PPI §2.2, nivel c de la VI).

El PPI dice "regresión logística one-vs-rest". ``OneVsRestClassifier`` no sabe
ignorar etiquetas desconocidas (-1, decisión D2), así que se entrena una
regresión binaria por etiqueta, cada una solo sobre las filas donde su
etiqueta es conocida. Es la misma formulación con enmascaramiento explícito.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from configuracion import ConfigTfidf
from datos.esquema import DESCONOCIDA, ETIQUETAS, POSITIVA

ARQUITECTURA = "tfidf_lr"


class ClasificadorTfidf:
    arquitectura = ARQUITECTURA

    def __init__(self, semilla: int, config: ConfigTfidf):
        self.semilla = semilla
        self.config = config
        self.vectorizador: TfidfVectorizer | None = None
        self.modelos: dict[str, LogisticRegression] = {}

    def entrenar(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> dict:
        """``val`` no se usa: la regresión logística no tiene parada temprana.

        Se acepta para que las tres arquitecturas compartan interfaz. Con el
        solver lbfgs el resultado no depende de la semilla, así que la varianza
        entre las cinco semillas será prácticamente nula; se reporta igual por
        simetría del diseño 3 × 2 × 5.
        """
        self.vectorizador = TfidfVectorizer(ngram_range=(1, self.config.ngram_max))
        matriz = self.vectorizador.fit_transform(train["texto"])

        self.modelos = {}
        filas_por_etiqueta = {}
        for etiqueta in ETIQUETAS:
            conocidas = (train[etiqueta] != DESCONOCIDA).to_numpy()
            y = train.loc[conocidas, etiqueta].to_numpy()
            if len(np.unique(y)) < 2:
                raise ValueError(
                    f"{etiqueta}: las filas conocidas de entrenamiento tienen una sola "
                    "clase; no se puede ajustar un clasificador binario"
                )
            modelo = LogisticRegression(
                C=self.config.C, max_iter=self.config.max_iter, random_state=self.semilla
            )
            modelo.fit(matriz[conocidas], y)
            self.modelos[etiqueta] = modelo
            filas_por_etiqueta[etiqueta] = int(conocidas.sum())

        return {
            "vocabulario": len(self.vectorizador.vocabulary_),
            "filas_entrenamiento_por_etiqueta": filas_por_etiqueta,
        }

    def predecir_probabilidades(self, df: pd.DataFrame) -> np.ndarray:
        """Matriz (n, 2): probabilidad de positivo, en el orden de ``ETIQUETAS``."""
        if self.vectorizador is None:
            raise RuntimeError("El clasificador no está entrenado ni cargado")
        matriz = self.vectorizador.transform(df["texto"])
        columnas = []
        for etiqueta in ETIQUETAS:
            modelo = self.modelos[etiqueta]
            indice_positiva = list(modelo.classes_).index(POSITIVA)
            columnas.append(modelo.predict_proba(matriz)[:, indice_positiva])
        return np.column_stack(columnas)

    def guardar(self, carpeta: Path) -> None:
        carpeta.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizador": self.vectorizador, "modelos": self.modelos}, carpeta / "tfidf.joblib")
        (carpeta / "clasificador.json").write_text(
            json.dumps(
                {"arquitectura": ARQUITECTURA, "semilla": self.semilla, "config": self.config.como_dict()},
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def cargar(cls, carpeta: Path) -> "ClasificadorTfidf":
        meta = json.loads((carpeta / "clasificador.json").read_text(encoding="utf-8"))
        clasificador = cls(meta["semilla"], ConfigTfidf(**meta["config"]))
        estado = joblib.load(carpeta / "tfidf.joblib")
        clasificador.vectorizador = estado["vectorizador"]
        clasificador.modelos = estado["modelos"]
        return clasificador

"""Partición 70/10/20 por sujeto (PPI §2.2, preparación de datos).

Todos los fragmentos de un sujeto caen en la misma partición: el modelo nunca
se evalúa sobre la persona con la que aprendió. La partición se calcula una
vez, con una semilla propia, y se congela. Las cinco semillas de
entrenamiento no la tocan, para que las tres arquitecturas se comparen sobre
exactamente los mismos fragmentos.
"""
import math
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from datos.esquema import ETIQUETAS

PROPORCIONES_PPI = (0.70, 0.10, 0.20)


class FugaDeDatosError(RuntimeError):
    """Un sujeto o un texto aparece en más de una partición."""


@dataclass(frozen=True)
class Particion:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def por_nombre(self) -> dict[str, pd.DataFrame]:
        return {"train": self.train, "val": self.val, "test": self.test}

    def asignaciones(self) -> pd.DataFrame:
        """Una fila por fragmento: el artefacto que se congela y se publica."""
        partes = [
            df[["fragmento_id", "sujeto_id"]].assign(particion=nombre)
            for nombre, df in self.por_nombre().items()
        ]
        return pd.concat(partes).sort_values("fragmento_id").reset_index(drop=True)

    def resumen(self) -> dict[str, dict]:
        total = sum(len(df) for df in self.por_nombre().values())
        return {
            nombre: {
                "fragmentos": len(df),
                "sujetos": int(df["sujeto_id"].nunique()),
                "fraccion_fragmentos": round(len(df) / total, 4),
                "estratos": {
                    k: int(v) for k, v in estrato(df).value_counts().sort_index().items()
                },
            }
            for nombre, df in self.por_nombre().items()
        }


def estrato(df: pd.DataFrame) -> pd.Series:
    """Combinación de etiquetas como texto: "1|0" = ansiedad sí, depresión no."""
    return df[list(ETIQUETAS)].astype(str).agg("|".join, axis=1)


def particionar(
    df: pd.DataFrame,
    proporciones: tuple[float, float, float] = PROPORCIONES_PPI,
    semilla: int = 42,
) -> Particion:
    """Parte ``df`` por ``sujeto_id``, estratificando por combinación de etiquetas.

    Espera un conjunto ya deduplicado: si un mismo texto normalizado queda en
    dos particiones, lanza ``FugaDeDatosError`` en vez de devolver una
    partición contaminada.

    Con grupos, las proporciones son aproximadas: un sujeto es indivisible.
    """
    k_test, k_val = _pliegues(proporciones)

    n_sujetos = df["sujeto_id"].nunique()
    if n_sujetos < k_test:
        raise ValueError(
            f"Hay {n_sujetos} sujetos y separar el {proporciones[2]:.0%} de prueba "
            f"requiere al menos {k_test}."
        )

    etiquetas = estrato(df)
    primero = StratifiedGroupKFold(n_splits=k_test, shuffle=True, random_state=semilla)
    idx_resto, idx_test = next(primero.split(df, etiquetas, groups=df["sujeto_id"]))

    resto = df.iloc[idx_resto]
    sujetos_resto = resto["sujeto_id"].nunique()
    if sujetos_resto < k_val:
        raise ValueError(
            f"Tras separar prueba quedan {sujetos_resto} sujetos y separar validación "
            f"requiere al menos {k_val}."
        )
    segundo = StratifiedGroupKFold(n_splits=k_val, shuffle=True, random_state=semilla)
    idx_train, idx_val = next(
        segundo.split(resto, etiquetas.iloc[idx_resto], groups=resto["sujeto_id"])
    )

    particion = Particion(
        train=resto.iloc[idx_train].reset_index(drop=True),
        val=resto.iloc[idx_val].reset_index(drop=True),
        test=df.iloc[idx_test].reset_index(drop=True),
    )
    verificar_sin_fugas(particion)
    return particion


def verificar_sin_fugas(particion: Particion) -> None:
    """Comprueba que ningún sujeto ni texto normalizado cruce particiones."""
    partes = particion.por_nombre()
    nombres = list(partes)
    for i, a in enumerate(nombres):
        for b in nombres[i + 1 :]:
            for columna in ("sujeto_id", "texto_normalizado"):
                compartidos = set(partes[a][columna]) & set(partes[b][columna])
                if compartidos:
                    raise FugaDeDatosError(
                        f"{len(compartidos)} valores de {columna} están en {a} y en {b} "
                        f"(ejemplos: {sorted(compartidos)[:3]}). Si son textos, "
                        "deduplica antes de particionar."
                    )


def _pliegues(proporciones: tuple[float, float, float]) -> tuple[int, int]:
    """Traduce las proporciones a número de pliegues de StratifiedGroupKFold.

    Prueba = 1 pliegue de k_test sobre el total. Validación = 1 pliegue de
    k_val sobre lo que queda. Para 70/10/20: k_test = 5 y k_val = 8
    (0,10 / 0,80 = 1/8).
    """
    if len(proporciones) != 3 or min(proporciones) <= 0:
        raise ValueError(f"Se esperan tres proporciones positivas; llegó {proporciones}")
    p_train, p_val, p_test = proporciones
    suma = p_train + p_val + p_test
    if not math.isclose(suma, 1.0, abs_tol=1e-9):
        raise ValueError(f"Las proporciones deben sumar 1; suman {suma}")
    return _inverso_entero(p_test, "prueba"), _inverso_entero(p_val / (1 - p_test), "validación")


def _inverso_entero(fraccion: float, nombre: str) -> int:
    k = round(1 / fraccion)
    if k < 2 or not math.isclose(1 / k, fraccion, rel_tol=1e-6):
        raise ValueError(
            f"La fracción de {nombre} ({fraccion:.4f}) debe ser 1/k con k entero ≥ 2 "
            "para partir por pliegues; 70/10/20 cumple la condición."
        )
    return k

"""Las tres arquitecturas del Componente A detrás de una sola fábrica.

Las tres exponen la misma interfaz: ``entrenar``, ``predecir_probabilidades``
(matriz n × 2 en el orden de ``datos.esquema.ETIQUETAS``), ``guardar`` y
``cargar``. La evaluación y la estadística no necesitan saber cuál es cuál.
"""
import json
from pathlib import Path

from configuracion import ConfigEntrenamiento


def cargar_clasificador(carpeta: Path):
    """Carga cualquier arquitectura guardada, según su ``clasificador.json``."""
    meta = json.loads((Path(carpeta) / "clasificador.json").read_text(encoding="utf-8"))
    if meta["arquitectura"] == "tfidf_lr":
        from modelos.tfidf import ClasificadorTfidf

        return ClasificadorTfidf.cargar(Path(carpeta))
    from modelos.beto import ClasificadorBeto

    return ClasificadorBeto.cargar(Path(carpeta))


def construir_clasificador(
    arquitectura: str,
    semilla: int,
    config: ConfigEntrenamiento,
    learning_rate_sonda: float | None = None,
):
    if arquitectura == "tfidf_lr":
        from modelos.tfidf import ClasificadorTfidf

        return ClasificadorTfidf(semilla, config.tfidf)

    # Import diferido: TF–IDF no debe cargar torch.
    from modelos.beto import ClasificadorBeto

    if arquitectura == "beto_ajustado":
        return ClasificadorBeto(arquitectura, semilla, config.beto)
    if arquitectura == "beto_sonda":
        if learning_rate_sonda is None:
            raise ValueError(
                "beto_sonda necesita learning_rate_sonda, uno de los valores del barrido "
                f"declarado: {list(config.sonda_learning_rates)}"
            )
        return ClasificadorBeto(arquitectura, semilla, config.hiperparametros_sonda(learning_rate_sonda))
    raise ValueError(f"Arquitectura desconocida: {arquitectura!r}")

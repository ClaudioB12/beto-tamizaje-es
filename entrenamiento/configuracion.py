"""Lectura validada de config/experimento.yaml."""
import dataclasses
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

RUTA_POR_DEFECTO = Path(__file__).resolve().parent / "config" / "experimento.yaml"

ARQUITECTURAS = ("beto_ajustado", "beto_sonda", "tfidf_lr")


class ConfiguracionInvalidaError(ValueError):
    """El archivo de configuración no tiene la forma esperada."""


@dataclass(frozen=True)
class ConfigParticion:
    proporciones: tuple[float, float, float]
    semilla: int

    def como_dict(self) -> dict:
        return {"proporciones": list(self.proporciones), "semilla": self.semilla}


@dataclass(frozen=True)
class HiperparametrosBeto:
    modelo_base: str
    revision: str | None
    max_length: int
    batch_size: int
    batch_por_dispositivo: int
    learning_rate: float
    weight_decay: float
    dropout: float
    max_epochs: int
    paciencia: int

    @property
    def acumulacion(self) -> int:
        """Pasos de acumulación para llegar al lote efectivo del PPI."""
        return self.batch_size // self.batch_por_dispositivo

    def con(self, **cambios) -> "HiperparametrosBeto":
        return dataclasses.replace(self, **cambios)

    def como_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ConfigTfidf:
    ngram_max: int
    C: float
    max_iter: int

    def como_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ConfigEntrenamiento:
    beto: HiperparametrosBeto
    sonda_learning_rates: tuple[float, ...]
    sonda_max_epochs: int
    tfidf: ConfigTfidf
    semillas: tuple[int, ...]
    arquitecturas: tuple[str, ...]

    def hiperparametros_sonda(self, learning_rate: float) -> HiperparametrosBeto:
        """La sonda comparte todo con el ajuste fino salvo tasa y épocas (D5)."""
        if learning_rate not in self.sonda_learning_rates:
            raise ValueError(
                f"learning_rate {learning_rate} no está en el barrido declarado "
                f"{list(self.sonda_learning_rates)}"
            )
        return self.beto.con(learning_rate=learning_rate, max_epochs=self.sonda_max_epochs)


def _leer(ruta: Path | str) -> dict:
    return yaml.safe_load(Path(ruta).read_text(encoding="utf-8")) or {}


def cargar_config_particion(ruta: Path | str = RUTA_POR_DEFECTO) -> ConfigParticion:
    datos = _leer(ruta)
    seccion = datos.get("particion")
    if not isinstance(seccion, dict):
        raise ConfiguracionInvalidaError(f"{ruta}: falta la sección 'particion'")

    proporciones = seccion.get("proporciones")
    if (
        not isinstance(proporciones, list)
        or len(proporciones) != 3
        or not all(_es_numero(p) for p in proporciones)
    ):
        raise ConfiguracionInvalidaError(
            f"{ruta}: particion.proporciones debe ser una lista de tres números "
            f"(train, val, test); es {proporciones!r}"
        )
    return ConfigParticion(
        proporciones=tuple(float(p) for p in proporciones),
        semilla=_entero(seccion, "semilla", ruta, "particion.", minimo=0),
    )


def cargar_config_entrenamiento(ruta: Path | str = RUTA_POR_DEFECTO) -> ConfigEntrenamiento:
    datos = _leer(ruta)

    modelo_base = datos.get("modelo_base")
    if not isinstance(modelo_base, str) or not modelo_base:
        raise ConfiguracionInvalidaError(f"{ruta}: falta modelo_base")
    revision = datos.get("revision_modelo_base")
    # Un identificador del Hub sin revisión fijada deja de ser reproducible el día
    # que el autor suba otro commit. Una carpeta local (tests) no la necesita.
    if not Path(modelo_base).is_dir() and not (isinstance(revision, str) and revision):
        raise ConfiguracionInvalidaError(
            f"{ruta}: revision_modelo_base es obligatoria para {modelo_base}; "
            "usa el hash de commit del modelo en Hugging Face"
        )

    parada = datos.get("parada_temprana") or {}
    if parada.get("metrica") != "eval_loss":
        raise ConfiguracionInvalidaError(
            f"{ruta}: parada_temprana.metrica debe ser eval_loss (PPI §2.2); es "
            f"{parada.get('metrica')!r}"
        )

    beto = HiperparametrosBeto(
        modelo_base=modelo_base,
        revision=revision or None,
        max_length=_entero(datos, "max_length", ruta),
        batch_size=_entero(datos, "batch_size", ruta),
        batch_por_dispositivo=_entero(datos, "batch_por_dispositivo", ruta),
        learning_rate=_flotante(datos, "learning_rate", ruta),
        weight_decay=_flotante(datos, "weight_decay", ruta, positivo=False),
        dropout=_flotante(datos, "dropout", ruta, positivo=False),
        max_epochs=_entero(datos, "max_epochs", ruta),
        paciencia=_entero(parada, "paciencia", ruta, "parada_temprana."),
    )
    if beto.batch_size % beto.batch_por_dispositivo:
        raise ConfiguracionInvalidaError(
            f"{ruta}: batch_size ({beto.batch_size}) debe ser múltiplo de "
            f"batch_por_dispositivo ({beto.batch_por_dispositivo})"
        )

    sonda = datos.get("sonda") or {}
    tasas = sonda.get("learning_rate")
    if not isinstance(tasas, list) or not tasas or not all(_es_numero(t) for t in tasas):
        raise ConfiguracionInvalidaError(
            f"{ruta}: sonda.learning_rate debe ser una lista de números; es {tasas!r}. "
            "Ojo: YAML lee 1e-3 como texto; escribe 1.0e-3."
        )

    tfidf = datos.get("tfidf") or {}
    semillas = datos.get("semillas")
    if (
        not isinstance(semillas, list)
        or not semillas
        or not all(isinstance(s, int) and not isinstance(s, bool) for s in semillas)
        or len(set(semillas)) != len(semillas)
    ):
        raise ConfiguracionInvalidaError(f"{ruta}: semillas debe ser una lista de enteros distintos")

    arquitecturas = datos.get("arquitecturas")
    if not isinstance(arquitecturas, list) or not set(arquitecturas) <= set(ARQUITECTURAS):
        raise ConfiguracionInvalidaError(
            f"{ruta}: arquitecturas debe ser un subconjunto de {list(ARQUITECTURAS)}"
        )

    return ConfigEntrenamiento(
        beto=beto,
        sonda_learning_rates=tuple(float(t) for t in tasas),
        sonda_max_epochs=_entero(sonda, "max_epochs", ruta, "sonda."),
        tfidf=ConfigTfidf(
            ngram_max=_entero(tfidf, "ngram_max", ruta, "tfidf."),
            C=_flotante(tfidf, "C", ruta, "tfidf."),
            max_iter=_entero(tfidf, "max_iter", ruta, "tfidf."),
        ),
        semillas=tuple(semillas),
        arquitecturas=tuple(arquitecturas),
    )


def _es_numero(valor) -> bool:
    return isinstance(valor, (int, float)) and not isinstance(valor, bool)


def _entero(seccion: dict, clave: str, ruta, prefijo: str = "", minimo: int = 1) -> int:
    valor = seccion.get(clave)
    # bool es subclase de int: `semilla: true` no debe pasar como 1.
    if not isinstance(valor, int) or isinstance(valor, bool) or valor < minimo:
        raise ConfiguracionInvalidaError(
            f"{ruta}: {prefijo}{clave} debe ser un entero ≥ {minimo}; es {valor!r}"
        )
    return valor


def _flotante(seccion: dict, clave: str, ruta, prefijo: str = "", positivo: bool = True) -> float:
    valor = seccion.get(clave)
    if not _es_numero(valor) or valor < 0 or (positivo and valor == 0):
        pista = " Ojo: YAML lee 2e-5 como texto; escribe 2.0e-5." if isinstance(valor, str) else ""
        raise ConfiguracionInvalidaError(
            f"{ruta}: {prefijo}{clave} debe ser un número{' positivo' if positivo else ''}; "
            f"es {valor!r}.{pista}"
        )
    return float(valor)

"""Ejecuta las corridas del Componente A: arquitectura × semilla (PPI §2.2).

Uso, desde entrenamiento/:
    python entrenar.py --fuente sintetico
    python entrenar.py --fuente sintetico --arquitecturas tfidf_lr --semillas 13 42
    python entrenar.py --fuente sintetico --humo

--humo entrena 1 época con textos de 64 tokens y una sola tasa de sonda. Sirve
para comprobar el pipeline de punta a punta antes de gastar GPU; escribe en una
carpeta aparte y sus resultados no se reportan.

Cada corrida escribe en corridas/<fuente>[-humo]/<arquitectura>-s<semilla>/:
    modelo/            pesos (BETO) o joblib (TF–IDF)
    entorno.json       versiones de paquetes y dispositivo
    manifiesto.json    se escribe AL FINAL: su existencia marca la corrida completa

Una corrida con manifiesto se salta. Tras una desconexión de Colab basta con
volver a lanzar el mismo comando: lo interrumpido se borra y se repite.
"""
import argparse
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import registro
from configuracion import RUTA_POR_DEFECTO as CONFIG_POR_DEFECTO
from configuracion import ConfigEntrenamiento, cargar_config_entrenamiento
from datos.fuentes import FUENTES
from datos.particion import Particion
from modelos import construir_clasificador

HUMO_MAX_EPOCHS = 1
HUMO_MAX_LENGTH = 64


def main(argv: list[str] | None = None) -> int:
    args = _argumentos(argv)
    config = cargar_config_entrenamiento(args.config)

    arquitecturas = args.arquitecturas or list(config.arquitecturas)
    semillas = args.semillas or list(config.semillas)
    # El PPI fija las cinco semillas; una semilla no declarada produciría una
    # corrida que el análisis mezclaría con las oficiales.
    fuera = [s for s in semillas if s not in config.semillas]
    if fuera:
        print(f"Semillas no declaradas en la configuración: {fuera}. Declaradas: {list(config.semillas)}", file=sys.stderr)
        return 2
    fuera = [a for a in arquitecturas if a not in config.arquitecturas]
    if fuera:
        print(f"Arquitecturas no declaradas: {fuera}. Declaradas: {list(config.arquitecturas)}", file=sys.stderr)
        return 2

    particion, informe_particion = registro.cargar_particion_congelada(args.fuente, args.particiones)
    raiz = args.corridas / registro.nombre_experimento(args.fuente, args.humo)
    entorno = registro.describir_entorno()

    for arquitectura in arquitecturas:
        for semilla in semillas:
            carpeta = raiz / f"{arquitectura}-s{semilla}"
            if (carpeta / "manifiesto.json").exists():
                print(f"= {carpeta.name}: completa, se salta")
                continue
            if carpeta.exists():
                print(f"~ {carpeta.name}: interrumpida, se borra y se repite")
                shutil.rmtree(carpeta)

            print(f"> {carpeta.name}: entrenando", flush=True)
            manifiesto = ejecutar_corrida(
                arquitectura, semilla, config, particion, informe_particion, carpeta, args.fuente, args.humo
            )
            registro.escribir_json(carpeta / "entorno.json", entorno)
            registro.escribir_json(carpeta / "manifiesto.json", manifiesto)
            perdida = manifiesto["mejor_eval_loss"]
            detalle = f"eval_loss={perdida:.4f}" if perdida is not None else "sin pérdida de validación"
            print(f"  {carpeta.name}: lista en {manifiesto['duracion_s']} s, {detalle}", flush=True)
    return 0


def ejecutar_corrida(
    arquitectura: str,
    semilla: int,
    config: ConfigEntrenamiento,
    particion: Particion,
    informe_particion: dict,
    carpeta: Path,
    fuente: str,
    humo: bool,
) -> dict:
    inicio = registro.ahora_utc()
    reloj = time.perf_counter()
    trabajo = carpeta / "trabajo"
    barrido = None

    if arquitectura == "beto_sonda":
        tasas = config.sonda_learning_rates[:1] if humo else config.sonda_learning_rates
        clasificador, info, barrido = _barrido_sonda(semilla, config, tasas, particion, trabajo, humo)
    else:
        clasificador = construir_clasificador(arquitectura, semilla, config)
        _aplicar_humo(clasificador, humo)
        if arquitectura == "tfidf_lr":
            info = clasificador.entrenar(particion.train, particion.val)
        else:
            info = clasificador.entrenar(particion.train, particion.val, carpeta_trabajo=trabajo)

    clasificador.guardar(carpeta / "modelo")
    # Los puntos de control intermedios ocupan ~1,3 GB por corrida de BETO y el
    # mejor ya quedó en modelo/.
    shutil.rmtree(trabajo, ignore_errors=True)

    hiper = getattr(clasificador, "hiper", None)
    return {
        "estado": "completa",
        "arquitectura": arquitectura,
        "semilla": semilla,
        "fuente": fuente,
        "humo": humo,
        "inicio_utc": inicio,
        "fin_utc": registro.ahora_utc(),
        "duracion_s": round(time.perf_counter() - reloj, 1),
        "config_sha256": registro.sha256_json(asdict(config)),
        "hiperparametros": hiper.como_dict() if hiper else config.tfidf.como_dict(),
        "particion": {
            "sha256_asignaciones": informe_particion["sha256_asignaciones"],
            "sha256_entrada": informe_particion["entrada"]["sha256"],
            "fragmentos": {nombre: len(df) for nombre, df in particion.por_nombre().items()},
        },
        "mejor_eval_loss": info.get("mejor_eval_loss"),
        "barrido_sonda": barrido,
        "entrenamiento": {k: v for k, v in info.items() if k != "mejor_punto_de_control"},
    }


def _barrido_sonda(semilla, config, tasas, particion, trabajo, humo):
    """Entrena la sonda con cada tasa declarada y conserva la de menor eval_loss.

    La elección usa solo validación (decisión D5, pendiente de aprobación). Cada
    semilla elige su tasa, y todo el barrido queda en el manifiesto.
    """
    mejor_clasificador, mejor_info, resultados = None, None, []
    for tasa in tasas:
        clasificador = construir_clasificador("beto_sonda", semilla, config, learning_rate_sonda=tasa)
        _aplicar_humo(clasificador, humo)
        info = clasificador.entrenar(particion.train, particion.val, carpeta_trabajo=trabajo / f"lr-{tasa:g}")
        resultados.append(
            {
                "learning_rate": tasa,
                "mejor_eval_loss": info["mejor_eval_loss"],
                "epocas_completadas": info["epocas_completadas"],
            }
        )
        if mejor_info is None or info["mejor_eval_loss"] < mejor_info["mejor_eval_loss"]:
            mejor_clasificador, mejor_info = clasificador, info
        else:
            del clasificador
    barrido = {
        "criterio": "menor eval_loss en validación",
        "learning_rate_elegida": mejor_clasificador.hiper.learning_rate,
        "resultados": resultados,
    }
    return mejor_clasificador, mejor_info, barrido


def _aplicar_humo(clasificador, humo: bool) -> None:
    if humo and hasattr(clasificador, "hiper"):
        clasificador.hiper = clasificador.hiper.con(max_epochs=HUMO_MAX_EPOCHS, max_length=HUMO_MAX_LENGTH)


def _argumentos(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fuente", required=True, choices=sorted(FUENTES))
    parser.add_argument("--config", type=Path, default=CONFIG_POR_DEFECTO)
    parser.add_argument("--arquitecturas", nargs="+", default=None)
    parser.add_argument("--semillas", nargs="+", type=int, default=None)
    parser.add_argument("--humo", action="store_true")
    parser.add_argument("--corridas", type=Path, default=registro.CARPETA_CORRIDAS)
    parser.add_argument("--particiones", type=Path, default=registro.CARPETA_PARTICIONES)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())

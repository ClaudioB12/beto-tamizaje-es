"""Congela la partición del experimento.

Uso, desde entrenamiento/:
    python particionar.py --fuente sintetico
    python particionar.py --fuente sintetico --sobrescribir

Escribe en particiones/<fuente>/:
    asignaciones.csv   fragmento_id, sujeto_id, particion
    informe.json       hashes, configuración, versiones y conteos

Ambos archivos se versionan: son las particiones que el PPI (§2.2) promete
publicar. Una partición congelada no se regenera sin --sobrescribir, porque
regenerarla después de ver resultados invalida la evaluación.
"""
import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sklearn

from configuracion import RUTA_POR_DEFECTO as CONFIG_POR_DEFECTO
from configuracion import cargar_config_particion
from datos.fuentes import FUENTES
from datos.particion import particionar
from datos.preparacion import deduplicar

RAIZ = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    args = _argumentos(argv)
    cargar, entrada = FUENTES[args.fuente]
    salida = args.salida or RAIZ / "particiones" / args.fuente

    if (salida / "asignaciones.csv").exists() and not args.sobrescribir:
        print(
            f"Ya existe una partición congelada en {salida}. Regenerarla cambia qué "
            "fragmentos son de prueba; si es intencional, usa --sobrescribir.",
            file=sys.stderr,
        )
        return 1

    config = cargar_config_particion(args.config)
    datos, informe_dedup = deduplicar(cargar(entrada))
    particion = particionar(datos, config.proporciones, config.semilla)

    salida.mkdir(parents=True, exist_ok=True)
    asignaciones_csv = particion.asignaciones().to_csv(index=False, lineterminator="\n")
    (salida / "asignaciones.csv").write_text(asignaciones_csv, encoding="utf-8", newline="")

    informe = {
        "fuente": args.fuente,
        "generado_en_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "entrada": {"archivo": entrada.name, "sha256": _sha256(entrada.read_bytes())},
        "configuracion": config.como_dict(),
        "sha256_asignaciones": _sha256(asignaciones_csv.encode("utf-8")),
        "deduplicacion": informe_dedup.como_dict(),
        "particiones": particion.resumen(),
        "versiones": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }
    (salida / "informe.json").write_text(
        json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    _imprimir_resumen(salida, informe)
    return 0


def _argumentos(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fuente", required=True, choices=sorted(FUENTES))
    parser.add_argument("--config", type=Path, default=CONFIG_POR_DEFECTO)
    parser.add_argument("--salida", type=Path, default=None)
    parser.add_argument("--sobrescribir", action="store_true")
    return parser.parse_args(argv)


def _sha256(contenido: bytes) -> str:
    return hashlib.sha256(contenido).hexdigest()


def _imprimir_resumen(salida: Path, informe: dict) -> None:
    dedup = informe["deduplicacion"]
    print(f"Partición congelada en {salida}")
    print(
        f"  deduplicación: {dedup['filas_entrada']} -> {dedup['filas_salida']} filas "
        f"({dedup['duplicados_eliminados']} duplicados, "
        f"{dedup['filas_en_conflicto_eliminadas']} en conflicto)"
    )
    for nombre, datos in informe["particiones"].items():
        print(
            f"  {nombre:5}  {datos['fragmentos']:4} fragmentos  {datos['sujetos']:3} sujetos  "
            f"{datos['fraccion_fragmentos']:.1%}  {datos['estratos']}"
        )
    print(f"  sha256 asignaciones: {informe['sha256_asignaciones'][:16]}…")


if __name__ == "__main__":
    raise SystemExit(main())

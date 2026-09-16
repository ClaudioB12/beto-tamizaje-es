"""Empaqueta el modelo a desplegar, elegido SOLO por validación (fase 7, primera parte).

Uso, desde entrenamiento/:
    python exportar.py --fuente mentalriskes
    python exportar.py --fuente sintetico            # permitido, marcado como solo pruebas

Elige, entre las corridas completas de la arquitectura, la semilla con menor
eval_loss en validación. Nunca mira la prueba: elegir con prueba la convertiría
en validación. Escribe en exportados/<fuente>/<arquitectura>-s<semilla>/:
    modelo/            copia del modelo de la corrida
    umbrales.json      calibrados en validación por evaluar.py
    exportacion.json   criterio, candidatos, hashes de cada archivo

NO copia nada a ml-service/: el modelo multietiqueta cambia el contrato de
/ml/clasificar, y ese cambio requiere las bandas de D4 y plantillas de
comorbilidad aprobadas por los psicólogos (PD1). Ver la fase 7 de la guía.
"""
import argparse
import shutil
import sys
from pathlib import Path

import registro
from datos.fuentes import FUENTES

ARQUITECTURAS_EXPORTABLES = ("beto_ajustado",)


def main(argv: list[str] | None = None) -> int:
    args = _argumentos(argv)
    raiz = args.corridas / registro.nombre_experimento(args.fuente, humo=False)
    candidatos = []
    for ruta in sorted(raiz.glob(f"{args.arquitectura}-s*/manifiesto.json")):
        manifiesto = registro.leer_json(ruta)
        candidatos.append((ruta.parent, manifiesto))
    if not candidatos:
        print(f"No hay corridas completas de {args.arquitectura} en {raiz}.", file=sys.stderr)
        return 1

    sin_umbrales = [c.name for c, _ in candidatos if not (c / "umbrales.json").exists()]
    if sin_umbrales:
        print(f"Faltan umbrales en {sin_umbrales}: ejecuta evaluar.py antes de exportar.", file=sys.stderr)
        return 1

    carpeta, elegido = seleccionar_por_validacion(candidatos)
    destino = args.exportados / args.fuente / carpeta.name
    if destino.exists():
        if not args.sobrescribir:
            print(f"{destino} ya existe; usa --sobrescribir para reemplazarlo.", file=sys.stderr)
            return 1
        shutil.rmtree(destino)

    shutil.copytree(carpeta / "modelo", destino / "modelo")
    for nombre in ("umbrales.json", "manifiesto.json", "entorno.json"):
        shutil.copy2(carpeta / nombre, destino / nombre)

    registro.escribir_json(
        destino / "exportacion.json",
        {
            "exportado_en_utc": registro.ahora_utc(),
            "fuente": args.fuente,
            "solo_pruebas": args.fuente == "sintetico",
            "arquitectura": args.arquitectura,
            "semilla_elegida": elegido["semilla"],
            "criterio": "menor eval_loss en validación; la prueba no interviene",
            "candidatos": [
                {"semilla": m["semilla"], "mejor_eval_loss": m["mejor_eval_loss"]} for _, m in candidatos
            ],
            "archivos": {
                str(ruta.relative_to(destino)).replace("\\", "/"): registro.sha256_archivo(ruta)
                for ruta in sorted(destino.rglob("*"))
                if ruta.is_file()
            },
        },
    )
    aviso = "  (SOLO PRUEBAS: datos sintéticos)" if args.fuente == "sintetico" else ""
    print(f"Exportado {carpeta.name} (eval_loss {elegido['mejor_eval_loss']:.4f}) en {destino}{aviso}")
    return 0


def seleccionar_por_validacion(candidatos: list[tuple[Path, dict]]) -> tuple[Path, dict]:
    """Menor eval_loss; ante empate, la semilla menor para que la elección sea determinista."""
    sin_perdida = [m["semilla"] for _, m in candidatos if m.get("mejor_eval_loss") is None]
    if sin_perdida:
        raise ValueError(f"Corridas sin pérdida de validación (semillas {sin_perdida}): no se pueden comparar")
    return min(candidatos, key=lambda c: (c[1]["mejor_eval_loss"], c[1]["semilla"]))


def _argumentos(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fuente", required=True, choices=sorted(FUENTES))
    parser.add_argument("--arquitectura", default="beto_ajustado", choices=ARQUITECTURAS_EXPORTABLES)
    parser.add_argument("--sobrescribir", action="store_true")
    parser.add_argument("--corridas", type=Path, default=registro.CARPETA_CORRIDAS)
    parser.add_argument("--exportados", type=Path, default=registro.CARPETA_EXPORTADOS)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())

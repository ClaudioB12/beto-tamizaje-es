"""Evalúa las corridas completas sobre la partición de prueba congelada.

Uso, desde entrenamiento/:
    python evaluar.py --fuente sintetico
    python evaluar.py --fuente sintetico --humo
    python evaluar.py --fuente sintetico --bandas config/bandas_severidad.json

Para cada corrida, en este orden y sin ajustar nada del modelo:
    1. calibra un umbral por etiqueta en VALIDACIÓN -> corrida/umbrales.json
    2. mide la confusión cruzada en VALIDACIÓN      -> indice.json (diagnóstico)
    3. predice PRUEBA con esos umbrales            -> predicciones/<experimento>/<corrida>-<dominio>.csv
    4. si se pasan bandas congeladas (D4), asigna severidad

predicciones/<experimento>/indice.json registra el hash de cada archivo; analizar.py
se niega a usar predicciones modificadas después.

El conjunto clínico no pasa por aquí mientras no exista su cargador: se evalúa
una sola vez, con los 15 modelos, umbrales y bandas ya congelados.
"""
import argparse
import sys
from pathlib import Path

import registro
from datos.fuentes import FUENTES
from evaluacion.predicciones import construir_predicciones
from evaluacion.severidad import cargar_bandas
from datos.esquema import ETIQUETAS
from estadistica.metricas import confusion_cruzada
from evaluacion.umbrales import aplicar_umbrales, calibrar_umbrales
from modelos import cargar_clasificador


def main(argv: list[str] | None = None) -> int:
    args = _argumentos(argv)
    experimento = registro.nombre_experimento(args.fuente, args.humo)
    corridas = sorted(ruta.parent for ruta in (args.corridas / experimento).glob("*/manifiesto.json"))
    if not corridas:
        print(f"No hay corridas completas en {args.corridas / experimento}. Ejecuta entrenar.py antes.", file=sys.stderr)
        return 1

    particion, informe_particion = registro.cargar_particion_congelada(args.fuente, args.particiones)
    bandas = cargar_bandas(args.bandas) if args.bandas else None
    dominios = particion.test["dominio"].unique()
    if len(dominios) != 1:
        print(f"La prueba mezcla dominios {list(dominios)}; se esperaba uno por fuente.", file=sys.stderr)
        return 1
    dominio = dominios[0]

    salida = args.predicciones / experimento
    salida.mkdir(parents=True, exist_ok=True)
    indice = {
        "experimento": experimento,
        "fuente": args.fuente,
        "humo": args.humo,
        "generado_en_utc": registro.ahora_utc(),
        "sha256_asignaciones": informe_particion["sha256_asignaciones"],
        "bandas": {"archivo": args.bandas.name, "sha256": registro.sha256_archivo(args.bandas)} if args.bandas else None,
        "archivos": [],
    }

    for carpeta in corridas:
        manifiesto = registro.leer_json(carpeta / "manifiesto.json")
        if manifiesto["particion"]["sha256_asignaciones"] != informe_particion["sha256_asignaciones"]:
            print(
                f"{carpeta.name} se entrenó con otra partición; no se evalúa contra la actual.",
                file=sys.stderr,
            )
            return 1

        clasificador = cargar_clasificador(carpeta / "modelo")
        probabilidades_val = clasificador.predecir_probabilidades(particion.val)
        umbrales = calibrar_umbrales(particion.val, probabilidades_val)
        registro.escribir_json(carpeta / "umbrales.json", umbrales)
        # En validación, no en prueba: es la cifra que puede guiar decisiones (w).
        cruzada_val = confusion_cruzada(
            particion.val[list(ETIQUETAS)].to_numpy(), aplicar_umbrales(probabilidades_val, umbrales)
        )

        tabla = construir_predicciones(
            particion.test, clasificador.predecir_probabilidades(particion.test), umbrales, bandas
        )
        nombre = f"{manifiesto['arquitectura']}-s{manifiesto['semilla']}-{dominio}.csv"
        contenido = tabla.to_csv(index=False, lineterminator="\n").encode("utf-8")
        (salida / nombre).write_bytes(contenido)

        indice["archivos"].append(
            {
                "archivo": nombre,
                "arquitectura": manifiesto["arquitectura"],
                "semilla": manifiesto["semilla"],
                "dominio": dominio,
                "sha256": registro.sha256_bytes(contenido),
                "umbrales": {etiqueta: datos["umbral"] for etiqueta, datos in umbrales.items()},
                "peso_negativo_debil": float(manifiesto["hiperparametros"].get("peso_negativo_debil", 0.0)),
                "confusion_cruzada_validacion": cruzada_val,
            }
        )
        print(f"  {nombre}: {len(tabla)} fragmentos, umbrales {indice['archivos'][-1]['umbrales']}")
        print(f"    confusión cruzada en validación: {_texto_cruzada(cruzada_val)}")
        en_borde = [etiqueta for etiqueta, datos in umbrales.items() if datos["en_borde_de_rejilla"]]
        if en_borde:
            indice["archivos"][-1]["umbrales_en_borde"] = en_borde
            print(
                f"    AVISO: umbral en el borde de la rejilla para {en_borde}; el modelo apenas "
                "separa esas clases en validación."
            )

    registro.escribir_json(salida / "indice.json", indice)
    print(f"Predicciones en {salida}")
    return 0


def _texto_cruzada(cruzada: dict) -> str:
    return " | ".join(
        f"{par} {datos['proporcion']:.0%}" if datos["proporcion"] is not None else f"{par} —"
        for par, datos in cruzada.items()
    )


def _argumentos(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fuente", required=True, choices=sorted(FUENTES))
    parser.add_argument("--humo", action="store_true")
    parser.add_argument("--bandas", type=Path, default=None)
    parser.add_argument("--corridas", type=Path, default=registro.CARPETA_CORRIDAS)
    parser.add_argument("--particiones", type=Path, default=registro.CARPETA_PARTICIONES)
    parser.add_argument("--predicciones", type=Path, default=registro.CARPETA_PREDICCIONES)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())

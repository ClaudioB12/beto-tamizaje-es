"""Ablación de términos SOLO en prueba (M1): mismos modelos, mismos umbrales, sin reentrenar.

Enmascara en el texto de PRUEBA los términos de config/terminos_etiqueta.txt y vuelve a
predecir con cada corrida ya evaluada, usando los umbrales que evaluar.py calibró en
validación SIN enmascarar (corrida/umbrales.json). Nada se ajusta mirando la prueba.

Deja las predicciones con el mismo formato e indice.json que evaluar.py, de modo que
    python analizar_pi1.py --predicciones <originales> --ablacion <estas> ...
calcula la caída de macro-F1 por modelo con IC bootstrap pareado (mismos sujetos y réplicas).

Interpretación: una caída indica dependencia de los términos enmascarados; por sí sola no
distingue un atajo léxico del uso de información clínicamente pertinente.

Uso:
    python ablacion_prueba.py --fuente reddit --corridas <carpeta de corridas> --predicciones <salida>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import registro
from datos.fuentes import FUENTES
from datos.normalizacion import normalizar_texto
from datos.reddit import RUTA_TERMINOS, enmascarar
from evaluacion.predicciones import construir_predicciones
from modelos import cargar_clasificador


def enmascarar_prueba(test):
    t = test.copy()
    t["texto"] = t["texto"].map(enmascarar)
    t["texto_normalizado"] = t["texto"].map(normalizar_texto)
    return t


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fuente", required=True, choices=sorted(FUENTES))
    ap.add_argument("--humo", action="store_true")
    ap.add_argument("--corridas", type=Path, default=registro.CARPETA_CORRIDAS)
    ap.add_argument("--particiones", type=Path, default=registro.CARPETA_PARTICIONES)
    ap.add_argument("--predicciones", type=Path, required=True)
    a = ap.parse_args(argv)

    experimento = registro.nombre_experimento(a.fuente, a.humo)
    corridas = sorted(r.parent for r in (a.corridas / experimento).glob("*/manifiesto.json"))
    if not corridas:
        print(f"No hay corridas completas en {a.corridas / experimento}.", file=sys.stderr)
        return 1
    faltan = [c.name for c in corridas if not (c / "umbrales.json").exists()]
    if faltan:
        print(f"Faltan umbrales.json (corre evaluar.py antes): {faltan}", file=sys.stderr)
        return 1

    particion, informe = registro.cargar_particion_congelada(a.fuente, a.particiones)
    test = enmascarar_prueba(particion.test)
    cambiadas = int((test["texto"] != particion.test["texto"]).sum())
    dominio = test["dominio"].unique()[0]
    salida = a.predicciones / experimento
    salida.mkdir(parents=True, exist_ok=True)
    indice = {
        "experimento": experimento, "fuente": a.fuente, "humo": a.humo,
        "generado_en_utc": registro.ahora_utc(),
        "sha256_asignaciones": informe["sha256_asignaciones"],
        "ablacion": {"tipo": "solo prueba, sin reentrenar, umbrales de validación sin enmascarar",
                     "terminos": RUTA_TERMINOS.name, "sha256_terminos": registro.sha256_archivo(RUTA_TERMINOS),
                     "fragmentos_modificados": cambiadas, "fragmentos_prueba": int(len(test))},
        "archivos": [],
    }
    for carpeta in corridas:
        manifiesto = registro.leer_json(carpeta / "manifiesto.json")
        if manifiesto["particion"]["sha256_asignaciones"] != informe["sha256_asignaciones"]:
            print(f"{carpeta.name} se entrenó con otra partición.", file=sys.stderr)
            return 1
        umbrales = registro.leer_json(carpeta / "umbrales.json")
        clasificador = cargar_clasificador(carpeta / "modelo")
        tabla = construir_predicciones(test, clasificador.predecir_probabilidades(test), umbrales, None)
        nombre = f"{manifiesto['arquitectura']}-s{manifiesto['semilla']}-{dominio}.csv"
        contenido = tabla.to_csv(index=False, lineterminator="\n").encode("utf-8")
        (salida / nombre).write_bytes(contenido)
        indice["archivos"].append({"archivo": nombre, "arquitectura": manifiesto["arquitectura"],
                                   "semilla": manifiesto["semilla"], "dominio": dominio,
                                   "sha256": registro.sha256_bytes(contenido)})
        print(f"  {nombre}: {cambiadas} de {len(test)} fragmentos con términos enmascarados")
    registro.escribir_json(salida / "indice.json", indice)
    print(f"Predicciones con prueba enmascarada en {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

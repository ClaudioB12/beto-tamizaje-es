"""Compara valores de peso_negativo_debil (opción B de D2) usando SOLO validación.

Uso, desde entrenamiento/:
    python buscar_peso.py --fuente reddit --humo
    python buscar_peso.py --fuente reddit --humo --pesos 0 0.1 0.3 0.5 1
    python buscar_peso.py --fuente reddit --arquitecturas beto_ajustado --semilla 42

Para cada arquitectura y cada w, con una sola semilla:
    1. entrena igual que entrenar.py (misma configuración salvo w)
    2. calibra los umbrales en validación
    3. mide en validación el F1 por etiqueta, el F1 macro y la confusión cruzada

El conjunto de prueba no se usa: elegir w mirando la prueba invalidaría H1.

Escribe en resultados/busqueda_peso/<experimento>/:
    por_corrida/<arquitectura>-w<w>.json   una por combinación; permite retomar
    tabla.csv                              una fila por combinación
    resumen.md                             la tabla legible, para el asesor

No elige w: entrega la tabla para que la decisión la tome el equipo. Los
modelos no se guardan; solo sus métricas.
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

import entrenar
import registro
from configuracion import RUTA_POR_DEFECTO as CONFIG_POR_DEFECTO
from configuracion import cargar_config_entrenamiento
from datos.esquema import DESCONOCIDA, ETIQUETAS
from datos.fuentes import FUENTES
from estadistica.metricas import confusion_cruzada, f1_macro, metricas_por_etiqueta
from evaluacion.umbrales import aplicar_umbrales, calibrar_umbrales
from modelos import cargar_clasificador

PESOS_POR_DEFECTO = (0.0, 0.1, 0.3, 0.5, 1.0)
ARQUITECTURAS_POR_DEFECTO = ("beto_ajustado", "tfidf_lr")
CARPETA_BUSQUEDA = registro.CARPETA_RESULTADOS / "busqueda_peso"
CARPETA_TRABAJO = registro.CARPETA_CORRIDAS / "busqueda_peso"


def main(argv: list[str] | None = None) -> int:
    args = _argumentos(argv)
    config = cargar_config_entrenamiento(args.config)
    semilla = args.semilla if args.semilla is not None else config.semillas[0]
    if semilla not in config.semillas:
        print(f"Semilla {semilla} no declarada. Declaradas: {list(config.semillas)}", file=sys.stderr)
        return 2
    fuera = [a for a in args.arquitecturas if a not in config.arquitecturas]
    if fuera:
        print(f"Arquitecturas no declaradas: {fuera}. Declaradas: {list(config.arquitecturas)}", file=sys.stderr)
        return 2
    try:
        pesos = [config.con_peso_negativo_debil(w).peso_negativo_debil for w in args.pesos]
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2

    particion, informe_particion = registro.cargar_particion_congelada(args.fuente, args.particiones)
    experimento = registro.nombre_experimento(args.fuente, args.humo)
    salida = args.resultados / experimento
    trabajo = args.trabajo / experimento
    sha_particion = informe_particion["sha256_asignaciones"]
    sin_desconocidas = not (particion.train[list(ETIQUETAS)] == DESCONOCIDA).to_numpy().any()
    if sin_desconocidas:
        print(f"AVISO: {args.fuente} no tiene etiquetas desconocidas en entrenamiento; w no cambia nada.")

    filas = []
    for arquitectura in args.arquitecturas:
        for peso in pesos:
            nombre = f"{arquitectura}-w{peso:g}"
            archivo = salida / "por_corrida" / f"{nombre}.json"
            identidad = {
                "arquitectura": arquitectura,
                "peso_negativo_debil": peso,
                "semilla": semilla,
                "humo": args.humo,
                "sha256_asignaciones": sha_particion,
            }
            if archivo.exists():
                previa = registro.leer_json(archivo)
                if all(previa.get(clave) == valor for clave, valor in identidad.items()):
                    print(f"= {nombre}: ya medida, se reutiliza")
                    filas.append(previa)
                    continue
                print(f"~ {nombre}: medida con otra configuración, se repite")

            print(f"> {nombre}: entrenando (semilla {semilla})", flush=True)
            carpeta = trabajo / nombre
            shutil.rmtree(carpeta, ignore_errors=True)
            manifiesto = entrenar.ejecutar_corrida(
                arquitectura,
                semilla,
                config.con_peso_negativo_debil(peso),
                particion,
                informe_particion,
                carpeta,
                args.fuente,
                args.humo,
            )
            fila = {**identidad, **medir_en_validacion(carpeta / "modelo", particion.val)}
            fila["mejor_eval_loss"] = manifiesto["mejor_eval_loss"]
            fila["duracion_s"] = manifiesto["duracion_s"]
            shutil.rmtree(carpeta, ignore_errors=True)
            registro.escribir_json(archivo, fila)
            filas.append(fila)
            print(f"  {nombre}: {_texto_fila(fila)}", flush=True)

    tabla = pd.DataFrame([_aplanar(f) for f in filas])
    salida.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(salida / "tabla.csv", index=False, lineterminator="\n")
    resumen = resumen_markdown(tabla, args.fuente, args.humo, semilla, sin_desconocidas)
    (salida / "resumen.md").write_text(resumen, encoding="utf-8")
    print(resumen)
    print(f"Resultados en {salida}")
    return 0


def medir_en_validacion(carpeta_modelo: Path, val: pd.DataFrame) -> dict:
    """Umbrales y métricas, ambos sobre validación.

    Medir con los mismos datos con que se calibraron los umbrales da cifras algo
    optimistas, pero el sesgo es el mismo para todos los w: la comparación entre
    ellos sigue siendo válida. Las cifras citables salen de evaluar.py y analizar.py.
    """
    clasificador = cargar_clasificador(carpeta_modelo)
    probabilidades = clasificador.predecir_probabilidades(val)
    umbrales = calibrar_umbrales(val, probabilidades)
    y = val[list(ETIQUETAS)].to_numpy()
    pred = aplicar_umbrales(probabilidades, umbrales)
    por_etiqueta = metricas_por_etiqueta(y, pred)
    return {
        "f1_macro": f1_macro(y, pred),
        "f1": {etiqueta: por_etiqueta[etiqueta]["f1"] for etiqueta in ETIQUETAS},
        "confusion_cruzada": confusion_cruzada(y, pred),
        "umbrales": {etiqueta: datos["umbral"] for etiqueta, datos in umbrales.items()},
        "umbrales_en_borde": [e for e, datos in umbrales.items() if datos["en_borde_de_rejilla"]],
        "filas_validacion": len(val),
    }


def _aplanar(fila: dict) -> dict:
    plana = {
        "arquitectura": fila["arquitectura"],
        "w": fila["peso_negativo_debil"],
        "f1_macro": fila["f1_macro"],
    }
    plana.update({f"f1_{etiqueta}": valor for etiqueta, valor in fila["f1"].items()})
    plana.update({par: datos["proporcion"] for par, datos in fila["confusion_cruzada"].items()})
    plana.update({f"umbral_{etiqueta}": valor for etiqueta, valor in fila["umbrales"].items()})
    plana["umbrales_en_borde"] = ",".join(fila["umbrales_en_borde"])
    plana["mejor_eval_loss"] = fila["mejor_eval_loss"]
    plana["duracion_s"] = fila["duracion_s"]
    return plana


def _texto_fila(fila: dict) -> str:
    cruzada = " | ".join(
        f"{par} {datos['proporcion']:.0%}" if datos["proporcion"] is not None else f"{par} —"
        for par, datos in fila["confusion_cruzada"].items()
    )
    return f"F1 macro {fila['f1_macro']:.3f} | {cruzada}"


def _porcentaje(valor) -> str:
    return "—" if valor is None or pd.isna(valor) else f"{valor:.1%}"


def resumen_markdown(
    tabla: pd.DataFrame, fuente: str, humo: bool, semilla: int, sin_desconocidas: bool = False
) -> str:
    pares = [c for c in tabla.columns if "→" in c]
    lineas = [f"# Búsqueda de w (peso_negativo_debil) — {fuente}{' · humo' if humo else ''}", ""]
    if humo:
        lineas += ["> **PRUEBA DE HUMO (1 época, 64 tokens): orienta la decisión, NO REPORTAR como resultado.**", ""]
    if sin_desconocidas:
        lineas += [f"> **{fuente} no tiene etiquetas desconocidas: w no tiene efecto y todas las filas coinciden.**", ""]
    lineas += [
        f"Semilla {semilla}. Todas las cifras son de VALIDACIÓN; la prueba no se usó.",
        "",
        "| Arquitectura | w | F1 macro | F1 ansiedad | F1 depresión | " + " | ".join(pares) + " |",
        "|---|---|---|---|---|" + "---|" * len(pares),
    ]
    for _, f in tabla.sort_values(["arquitectura", "w"]).iterrows():
        lineas.append(
            f"| {f['arquitectura']} | {f['w']:g} | {f['f1_macro']:.3f} | {f['f1_ansiedad']:.3f} | "
            f"{f['f1_depresion']:.3f} | " + " | ".join(_porcentaje(f[p]) for p in pares) + " |"
        )
    lineas += [
        "",
        "Cómo leerla:",
        "- w = 0 es la pérdida enmascarada (D2); w = 1 trata la etiqueta no observada como negativa.",
        "- El F1 compara cada clase de riesgo con el control. La confusión cruzada mide si el modelo "
        "distingue ansiedad de depresión; en Reddit el F1 no la ve.",
        "- Un buen w baja la confusión cruzada sin hundir el F1 ni subir «control→alguna». "
        "La elección es del equipo y del asesor, y se registra en experimento.yaml.",
        "- No se compara eval_loss entre valores de w: cada w define una pérdida distinta.",
    ]
    borde = tabla[tabla["umbrales_en_borde"].astype(str).str.len() > 0]
    if len(borde):
        lineas += [
            "",
            "Umbral en el borde de la rejilla (el modelo apenas separa esa etiqueta en validación): "
            + ", ".join(f"{f['arquitectura']} w={f['w']:g} ({f['umbrales_en_borde']})" for _, f in borde.iterrows()),
        ]
    return "\n".join(lineas) + "\n"


def _argumentos(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fuente", required=True, choices=sorted(FUENTES))
    parser.add_argument("--config", type=Path, default=CONFIG_POR_DEFECTO)
    parser.add_argument("--pesos", nargs="+", type=float, default=list(PESOS_POR_DEFECTO))
    parser.add_argument("--arquitecturas", nargs="+", default=list(ARQUITECTURAS_POR_DEFECTO))
    parser.add_argument("--semilla", type=int, default=None, help="Por defecto, la primera declarada")
    parser.add_argument("--humo", action="store_true")
    parser.add_argument("--particiones", type=Path, default=registro.CARPETA_PARTICIONES)
    parser.add_argument("--resultados", type=Path, default=CARPETA_BUSQUEDA)
    parser.add_argument("--trabajo", type=Path, default=CARPETA_TRABAJO)
    return parser.parse_args(argv)


if __name__ == "__main__":
    # La tabla usa «→»; la consola de Windows (cp1252) no puede imprimirlo.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())

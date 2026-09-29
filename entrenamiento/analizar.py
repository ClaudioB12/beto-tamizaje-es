"""Convierte las predicciones de prueba en los resultados de PE1, PE2, PE3 y H1.

Uso, desde entrenamiento/:
    python analizar.py --fuente sintetico
    python analizar.py --fuente sintetico --humo --bootstrap 1000

Escribe en resultados/<experimento>/:
    resultados.json   todas las cifras, con IC y parámetros del bootstrap
    resumen.md        tablas legibles para el informe

Solo lee predicciones cuyo hash coincide con predicciones/<experimento>/indice.json.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import registro
from datos.esquema import ETIQUETAS
from datos.fuentes import FUENTES
from estadistica.bootstrap import N_BOOTSTRAP_PPI, ic_brecha_dominio, ic_diferencia, ic_media_semillas
from estadistica.concordancia import kappa_ponderado
from estadistica.metricas import DEFINICION_F1_MACRO, f1_control, f1_macro, metricas_por_etiqueta
from estadistica.pruebas import prueba_semillas
from evaluacion.instrumentos import INSTRUMENTO_POR_ETIQUETA, ORDEN_POR_INSTRUMENTO, categoria

# H1: el ajuste fino supera a la sonda y al baseline (PPI §1.6).
COMPARACIONES_H1 = (("beto_ajustado", "beto_sonda"), ("beto_ajustado", "tfidf_lr"))
# PE2: brecha = F1(redes sociales) − F1(conversación clínica).
BRECHA = ("mentalriskes", "clinico")
ORDEN_ARQUITECTURAS = ("beto_ajustado", "beto_sonda", "tfidf_lr")


class PrediccionesInvalidasError(RuntimeError):
    pass


def main(argv: list[str] | None = None) -> int:
    args = _argumentos(argv)
    experimento = registro.nombre_experimento(args.fuente, args.humo)
    carpeta = args.predicciones / experimento
    if not (carpeta / "indice.json").exists():
        print(f"No hay predicciones en {carpeta}. Ejecuta evaluar.py antes.", file=sys.stderr)
        return 1
    try:
        tablas = cargar_predicciones(carpeta)
    except PrediccionesInvalidasError as error:
        print(error, file=sys.stderr)
        return 1

    resultados = analizar(tablas, args.bootstrap, args.semilla_bootstrap)
    resultados.update(
        {
            "experimento": experimento,
            "fuente": args.fuente,
            "humo": args.humo,
            "generado_en_utc": registro.ahora_utc(),
            "advertencias": _advertencias(args.fuente, args.humo, args.bootstrap),
        }
    )
    salida = args.resultados / experimento
    registro.escribir_json(salida / "resultados.json", resultados)
    (salida / "resumen.md").write_text(resumen_markdown(resultados), encoding="utf-8")
    print(resumen_markdown(resultados))
    print(f"Resultados en {salida}")
    return 0


def cargar_predicciones(carpeta: Path) -> dict[tuple[str, str], dict[int, pd.DataFrame]]:
    """(arquitectura, dominio) -> {semilla: tabla}, verificando hashes y alineación."""
    indice = registro.leer_json(carpeta / "indice.json")
    tablas: dict[tuple[str, str], dict[int, pd.DataFrame]] = defaultdict(dict)
    for entrada in indice["archivos"]:
        contenido = (carpeta / entrada["archivo"]).read_bytes()
        if registro.sha256_bytes(contenido) != entrada["sha256"]:
            raise PrediccionesInvalidasError(
                f"{entrada['archivo']} no coincide con su hash en indice.json: se modificó después de evaluar."
            )
        tabla = pd.read_csv(carpeta / entrada["archivo"], dtype={"fragmento_id": str, "sujeto_id": str})
        tablas[(entrada["arquitectura"], entrada["dominio"])][entrada["semilla"]] = tabla

    # Todas las tablas de un dominio deben describir los mismos fragmentos con las
    # mismas etiquetas; si no, las comparaciones pareadas no tendrían sentido.
    referencia_por_dominio: dict[str, pd.DataFrame] = {}
    for (arquitectura, dominio), por_semilla in tablas.items():
        for semilla, tabla in por_semilla.items():
            columnas = ["fragmento_id", *(f"y_{e}" for e in ETIQUETAS)]
            referencia = referencia_por_dominio.setdefault(dominio, tabla[columnas])
            if not tabla[columnas].equals(referencia):
                raise PrediccionesInvalidasError(
                    f"{arquitectura} s{semilla} ({dominio}) no tiene los mismos fragmentos o etiquetas que el resto."
                )
    return tablas


def _matrices(tabla: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y = tabla[[f"y_{e}" for e in ETIQUETAS]].to_numpy()
    pred = tabla[[f"pred_{e}" for e in ETIQUETAS]].to_numpy()
    return y, pred


def analizar(tablas, n_bootstrap: int, semilla_bootstrap: int) -> dict:
    por_arquitectura, comparaciones, brechas, concordancia = [], [], [], []

    for (arquitectura, dominio), por_semilla in _ordenadas(tablas):
        semillas = sorted(por_semilla)
        y, _ = _matrices(por_semilla[semillas[0]])
        preds = [_matrices(por_semilla[s])[1] for s in semillas]
        f1s = [f1_macro(y, p) for p in preds]
        por_etiqueta = [metricas_por_etiqueta(y, p) for p in preds]
        controles = [f1_control(y, p) for p in preds]

        por_arquitectura.append(
            {
                "arquitectura": arquitectura,
                "dominio": dominio,
                "semillas": semillas,
                "f1_macro_por_semilla": dict(zip(semillas, f1s)),
                "f1_macro_media": float(np.mean(f1s)),
                "f1_macro_de": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else None,
                "f1_macro_min": float(np.min(f1s)),
                "f1_macro_max": float(np.max(f1s)),
                # Todas las semillas idénticas: las corridas son una sola observación.
                "determinista_entre_semillas": len(f1s) > 1 and float(np.ptp(f1s)) == 0.0,
                "ic_bootstrap": ic_media_semillas(y, preds, n_bootstrap, semilla=semilla_bootstrap),
                "por_etiqueta_media": {
                    etiqueta: {
                        metrica: float(np.mean([m[etiqueta][metrica] for m in por_etiqueta]))
                        for metrica in ("f1", "sensibilidad", "precision")
                    }
                    for etiqueta in ETIQUETAS
                },
                "f1_control_media": float(np.mean(controles)) if None not in controles else None,
            }
        )
        concordancia.extend(_concordancia(arquitectura, dominio, por_semilla, n_bootstrap, semilla_bootstrap))

    dominios = sorted({dominio for _, dominio in tablas})
    for dominio in dominios:
        for a, b in COMPARACIONES_H1:
            if (a, dominio) not in tablas or (b, dominio) not in tablas:
                continue
            semillas_a, semillas_b = sorted(tablas[(a, dominio)]), sorted(tablas[(b, dominio)])
            y, _ = _matrices(tablas[(a, dominio)][semillas_a[0]])
            preds_a = [_matrices(tablas[(a, dominio)][s])[1] for s in semillas_a]
            preds_b = [_matrices(tablas[(b, dominio)][s])[1] for s in semillas_b]
            prueba = None
            if len(preds_a) >= 2 and len(preds_b) >= 2:
                prueba = prueba_semillas([f1_macro(y, p) for p in preds_a], [f1_macro(y, p) for p in preds_b])
            comparaciones.append(
                {
                    "dominio": dominio,
                    "a": a,
                    "b": b,
                    "diferencia_f1_macro": ic_diferencia(y, preds_a, preds_b, n_bootstrap, semilla=semilla_bootstrap),
                    "prueba_semillas": prueba,
                }
            )

    origen, destino = BRECHA
    for arquitectura in ORDEN_ARQUITECTURAS:
        if (arquitectura, origen) in tablas and (arquitectura, destino) in tablas:
            y_o, preds_o = _todas(tablas[(arquitectura, origen)])
            y_d, preds_d = _todas(tablas[(arquitectura, destino)])
            brechas.append(
                {
                    "arquitectura": arquitectura,
                    "definicion": f"F1 macro({origen}) − F1 macro({destino})",
                    **ic_brecha_dominio(y_o, preds_o, y_d, preds_d, n_bootstrap, semilla=semilla_bootstrap),
                }
            )

    return {
        "definicion_f1_macro": DEFINICION_F1_MACRO,
        "bootstrap": {"n": n_bootstrap, "semilla": semilla_bootstrap, "metodo": "percentiles"},
        "por_arquitectura": por_arquitectura,
        "comparaciones_h1": comparaciones,
        "brechas_dominio": brechas,
        "concordancia_severidad": concordancia,
    }


def _todas(por_semilla):
    semillas = sorted(por_semilla)
    y, _ = _matrices(por_semilla[semillas[0]])
    return y, [_matrices(por_semilla[s])[1] for s in semillas]


def _ordenadas(tablas):
    def clave(item):
        (arquitectura, dominio), _ = item
        return (dominio, ORDEN_ARQUITECTURAS.index(arquitectura) if arquitectura in ORDEN_ARQUITECTURAS else 99)

    return sorted(tablas.items(), key=clave)


def _concordancia(arquitectura, dominio, por_semilla, n_bootstrap, semilla_bootstrap) -> list[dict]:
    """Kappa ponderado de la severidad inferida contra PHQ-9/GAD-7 (VD2).

    Solo aplica cuando hay bandas congeladas (D4) y puntajes del instrumento,
    es decir, en el conjunto clínico.
    """
    salida = []
    for etiqueta in ETIQUETAS:
        instrumento = INSTRUMENTO_POR_ETIQUETA[etiqueta]
        kappas = {}
        for semilla, tabla in sorted(por_semilla.items()):
            validas = tabla[f"sev_{etiqueta}"].notna() & tabla[instrumento].notna()
            if not validas.any():
                continue
            referencia = categoria(instrumento, tabla.loc[validas, instrumento].to_numpy())
            kappas[semilla] = kappa_ponderado(
                referencia,
                tabla.loc[validas, f"sev_{etiqueta}"].to_numpy(),
                ORDEN_POR_INSTRUMENTO[instrumento],
                n_bootstrap,
                semilla=semilla_bootstrap,
            )
        if kappas:
            salida.append(
                {
                    "arquitectura": arquitectura,
                    "dominio": dominio,
                    "etiqueta": etiqueta,
                    "instrumento": instrumento,
                    "kappa_media_semillas": float(np.mean([k["kappa"] for k in kappas.values()])),
                    "por_semilla": kappas,
                }
            )
    return salida


def _advertencias(fuente: str, humo: bool, n_bootstrap: int) -> list[str]:
    advertencias = []
    if humo:
        advertencias.append("PRUEBA DE HUMO (1 época, 64 tokens): estas cifras solo verifican el pipeline. NO REPORTAR.")
    if fuente == "sintetico":
        advertencias.append("Datos sintéticos: no miden el desempeño del tamizaje. NO REPORTAR.")
    if n_bootstrap != N_BOOTSTRAP_PPI:
        advertencias.append(f"Bootstrap con {n_bootstrap} remuestreos; el PPI declara {N_BOOTSTRAP_PPI}.")
    return advertencias


def _ic(d: dict) -> str:
    return f"{d['estimacion']:.3f} [{d['ic_inferior']:.3f}, {d['ic_superior']:.3f}]"


NOTA_VARIABILIDAD = (
    "Las dos medidas de variabilidad no son intercambiables. La **DE y el rango entre semillas** "
    "describen cuánto cambia el F1 al cambiar la inicialización y el orden de los lotes, con los "
    "fragmentos fijos. El **IC bootstrap sobre fragmentos** describe la incertidumbre por muestreo "
    "del conjunto de prueba, con las semillas fijas, y por tanto NO incorpora la variabilidad entre "
    "semillas. Un IC más estrecho que ±1 DE no es un error de cálculo."
)


def _marca_no_citable(r: dict) -> list[str]:
    """Rótulo pegado a cada tabla, no solo en la cabecera: las tablas se copian sueltas."""
    if not r.get("advertencias"):
        return []
    motivos = []
    if r.get("humo"):
        motivos.append("prueba de humo")
    if r.get("fuente") == "sintetico":
        motivos.append("datos sintéticos")
    return [f"**[NO CITABLE — {' · '.join(motivos) or 'ver advertencias'}]**", ""]


def _celda_prueba(prueba: dict | None) -> str:
    if not prueba or not prueba.get("prueba_usada"):
        return "— (sin variabilidad entre semillas)"
    if prueba["prueba_usada"] == "mann_whitney_unilateral":
        mw = prueba["mann_whitney_unilateral"]
        return f"Mann–Whitney p = {mw['p']:.4f} (r = {mw['r_rango_biserial']:.2f})"
    w = prueba["wilcoxon_una_muestra_unilateral"]
    lado = "b" if prueba["b_determinista"] else "a"
    return f"Wilcoxon 1 muestra p = {w['p']:.4f} ({lado} determinista)"


def resumen_markdown(r: dict) -> str:
    lineas = [f"# Resultados — {r.get('experimento', '')}", ""]
    for advertencia in r.get("advertencias", []):
        lineas.append(f"> **{advertencia}**")
    lineas += ["", f"F1 macro: {r['definicion_f1_macro']}.", ""]

    lineas += ["## PE1 — F1 macro por arquitectura", "", *_marca_no_citable(r)]
    lineas += [
        "| Dominio | Arquitectura | F1 media ± DE entre semillas | Rango entre semillas | "
        "IC 95 % bootstrap sobre fragmentos | F1 ansiedad | F1 depresión |",
        "|---|---|---|---|---|---|---|",
    ]
    for f in r["por_arquitectura"]:
        de = f"{f['f1_macro_de']:.3f}" if f["f1_macro_de"] is not None else "—"
        determinista = " (determinista)" if f.get("determinista_entre_semillas") else ""
        lineas.append(
            f"| {f['dominio']} | {f['arquitectura']}{determinista} | {f['f1_macro_media']:.3f} ± {de} | "
            f"{f['f1_macro_min']:.3f}–{f['f1_macro_max']:.3f} | {_ic(f['ic_bootstrap'])} | "
            f"{f['por_etiqueta_media']['ansiedad']['f1']:.3f} | {f['por_etiqueta_media']['depresion']['f1']:.3f} |"
        )
    lineas += ["", NOTA_VARIABILIDAD]
    if any(f.get("determinista_entre_semillas") for f in r["por_arquitectura"]):
        lineas += [
            "",
            "«Determinista»: todas sus semillas dan el mismo F1, así que sus corridas equivalen a una "
            "sola observación. En H1 se contrasta contra ese valor con Wilcoxon de una muestra.",
        ]

    lineas += ["", "## H1 — Diferencias entre arquitecturas", ""]
    if r["comparaciones_h1"]:
        lineas += [
            *_marca_no_citable(r),
            "| Dominio | Comparación | Diferencia [IC 95 % bootstrap sobre fragmentos] | IC excluye 0 | "
            "Prueba sobre semillas (unilateral) |",
            "|---|---|---|---|---|",
        ]
        for c in r["comparaciones_h1"]:
            lineas.append(
                f"| {c['dominio']} | {c['a']} − {c['b']} | {_ic(c['diferencia_f1_macro'])} | "
                f"{'sí' if c['diferencia_f1_macro']['excluye_cero'] else 'no'} | {_celda_prueba(c['prueba_semillas'])} |"
            )
        lineas += [
            "",
            "Con 5 contra 5 semillas, el p mínimo de Mann–Whitney es 1/252 ≈ 0,004 y con una "
            "arquitectura determinista el de Wilcoxon de una muestra es 1/32 ≈ 0,031: son pisos que la "
            "prueba no puede superar por grande que sea la diferencia real.",
        ]
    else:
        lineas.append("Sin pares de arquitecturas para comparar.")

    lineas += ["", "## PE2 — Brecha de transferencia de dominio", ""]
    if r["brechas_dominio"]:
        for b in r["brechas_dominio"]:
            lineas.append(f"- {b['arquitectura']}: {b['definicion']} = {_ic(b)}")
    else:
        lineas.append("No disponible: requiere predicciones de MentalRiskES y del conjunto clínico.")

    lineas += ["", "## PE3 — Concordancia de severidad (kappa ponderado)", ""]
    if r["concordancia_severidad"]:
        for k in r["concordancia_severidad"]:
            lineas.append(
                f"- {k['dominio']} · {k['arquitectura']} · {k['etiqueta']} vs {k['instrumento']}: "
                f"κ medio = {k['kappa_media_semillas']:.3f}"
            )
    else:
        lineas.append("No disponible: requiere bandas congeladas (D4) y puntajes PHQ-9/GAD-7 del conjunto clínico.")
    return "\n".join(lineas) + "\n"


def _argumentos(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fuente", required=True, choices=sorted(FUENTES))
    parser.add_argument("--humo", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP_PPI)
    parser.add_argument("--semilla-bootstrap", type=int, default=2026)
    parser.add_argument("--predicciones", type=Path, default=registro.CARPETA_PREDICCIONES)
    parser.add_argument("--resultados", type=Path, default=registro.CARPETA_RESULTADOS)
    return parser.parse_args(argv)


if __name__ == "__main__":
    # El resumen usa «−», «±» y «κ». La consola de Windows (cp1252) no puede
    # imprimir «−» ni «κ» y abortaría después de escribir los archivos.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())

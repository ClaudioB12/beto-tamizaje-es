"""PI1 en Reddit con la métrica principal del short paper (§IV-D).

Lee las predicciones de prueba que deja ``evaluar.py`` (sin reentrenar nada) y
calcula, por arquitectura:
- macro-F1 de tres clases con IC bootstrap al 95 % por sujeto (media de semillas)
  y su media ± DE entre semillas;
- las diferencias pareadas BETO-FT − BETO-LP y BETO-FT − TF–IDF+RL con IC; en
  Reddit, H1 se cumpliría si ambos IC excluyen el 0 (la parte clínica de la
  prueba de intersección-unión queda pendiente del conjunto clínico);
- secundarias por etiqueta (F1, sensibilidad, especificidad, AUROC) sobre filas
  con la etiqueta conocida, la doble asignación y el macro-F1 enmascarado (D3);
- opcional (``--ablacion``): la misma métrica con los términos de la etiqueta
  enmascarados y la diferencia original − enmascarado, pareada por sujeto.

Uso, desde entrenamiento/:
    python analizar_pi1.py --predicciones predicciones_w03/reddit
    python analizar_pi1.py --predicciones predicciones_w03/reddit \
        --ablacion predicciones_w03/reddit_enmascarado --salida resultados_w03/pi1_reddit

Escribe en --salida: ``pi1.json``, ``resumen.md`` y ``tabla_pi1.tex``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import registro
from analizar import ORDEN_ARQUITECTURAS, cargar_predicciones
from datos.esquema import ETIQUETAS
from estadistica.bootstrap import N_BOOTSTRAP_PPI
from estadistica.metricas import confusion_cruzada, f1_macro
from estadistica import tres_clases as tc

NOMBRES = {"beto_ajustado": "BETO-FT", "beto_sonda": "BETO-LP", "tfidf_lr": "TF--IDF+RL"}
COMPARACIONES = (("beto_ajustado", "beto_sonda"), ("beto_ajustado", "tfidf_lr"))


def _media_de(valores) -> dict:
    v = np.asarray([x for x in valores if x is not None], dtype=float)
    if v.size == 0:
        return {"media": None, "de": None, "n": 0}
    return {"media": float(v.mean()), "de": float(v.std(ddof=1)) if v.size > 1 else 0.0, "n": int(v.size)}


def _secundarias(tabla) -> dict:
    y = tabla[[f"y_{e}" for e in ETIQUETAS]].to_numpy()
    pred = tabla[[f"pred_{e}" for e in ETIQUETAS]].to_numpy()
    salida = {}
    for j, etiqueta in enumerate(ETIQUETAS):
        conocidas = y[:, j] >= 0
        yj, pj = y[conocidas, j], pred[conocidas, j]
        tp, fp = int(((yj == 1) & (pj == 1)).sum()), int(((yj == 0) & (pj == 1)).sum())
        fn, tn = int(((yj == 1) & (pj == 0)).sum()), int(((yj == 0) & (pj == 0)).sum())
        auroc = None
        if len(set(yj.tolist())) == 2:
            auroc = float(roc_auc_score(yj, tabla.loc[conocidas, f"p_{etiqueta}"].to_numpy()))
        salida[etiqueta] = {
            "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
            "sensibilidad": tp / (tp + fn) if tp + fn else None,
            "especificidad": tn / (tn + fp) if tn + fp else None,
            "auroc": auroc,
        }
    cruzada = confusion_cruzada(y, pred)
    salida["doble_asignacion"] = {k: v["proporcion"] for k, v in cruzada.items()}
    salida["f1_macro_enmascarado"] = f1_macro(y, pred)
    return salida


def _resumir_secundarias(por_semilla: list[dict]) -> dict:
    salida = {}
    for etiqueta in ETIQUETAS:
        salida[etiqueta] = {
            m: _media_de(s[etiqueta][m] for s in por_semilla)
            for m in ("f1", "sensibilidad", "especificidad", "auroc")
        }
    claves = por_semilla[0]["doble_asignacion"].keys()
    salida["doble_asignacion"] = {k: _media_de(s["doble_asignacion"][k] for s in por_semilla) for k in claves}
    salida["f1_macro_enmascarado"] = _media_de(s["f1_macro_enmascarado"] for s in por_semilla)
    return salida


def _un_dominio(tablas: dict, n_bootstrap: int, semilla: int) -> tuple[dict, tc.Remuestreo, dict]:
    if len({d for _, d in tablas}) != 1:
        raise ValueError("La carpeta de predicciones debe tener un solo dominio.")
    por_arq = {a: s for (a, _), s in tablas.items()}
    primera = por_arq[next(a for a in ORDEN_ARQUITECTURAS if a in por_arq)]
    referencia = primera[sorted(primera)[0]]
    remuestreo = tc.Remuestreo(referencia, n_bootstrap, semilla)
    distribuciones, resultado = {}, {}
    for arq in ORDEN_ARQUITECTURAS:
        if arq not in por_arq:
            continue
        dist, f1_semillas = remuestreo.distribucion(por_arq[arq])
        distribuciones[arq] = dist
        inf, sup = tc.intervalo(dist)
        resultado[arq] = {
            "semillas": sorted(por_arq[arq]),
            "macro_f1_tres_clases": {
                "media_semillas": float(np.mean(f1_semillas)),
                "ic95": [inf, sup],
                "por_semilla": dict(zip(map(str, sorted(por_arq[arq])), f1_semillas)),
                "de_semillas": _media_de(f1_semillas)["de"],
            },
            "secundarias": _resumir_secundarias([_secundarias(por_arq[arq][s]) for s in sorted(por_arq[arq])]),
        }
    return resultado, remuestreo, distribuciones


def cargar_varias(carpetas) -> dict:
    """Une las predicciones de varias carpetas (p. ej., una por arquitectura
    cuando cada una usa su propio w). Una arquitectura no puede repetirse."""
    if isinstance(carpetas, (str, Path)):
        carpetas = [carpetas]
    unidas: dict = {}
    for carpeta in carpetas:
        for clave, por_semilla in cargar_predicciones(Path(carpeta)).items():
            if clave in unidas:
                raise ValueError(f"{clave} aparece en más de una carpeta de predicciones")
            unidas[clave] = por_semilla
    return unidas


def analizar(carpeta, ablacion, n_bootstrap: int, semilla: int) -> dict:
    tablas = cargar_varias(carpeta)
    resultado, remuestreo, dist = _un_dominio(tablas, n_bootstrap, semilla)

    comparaciones = []
    for a, b in COMPARACIONES:
        if a in dist and b in dist:
            d = dist[a] - dist[b]
            inf, sup = tc.intervalo(d)
            estimacion = (
                resultado[a]["macro_f1_tres_clases"]["media_semillas"]
                - resultado[b]["macro_f1_tres_clases"]["media_semillas"]
            )
            comparaciones.append(
                {"a": a, "b": b, "diferencia": estimacion, "ic95": [inf, sup], "excluye_cero": inf > 0 or sup < 0}
            )

    salida = {
        "definicion": tc.DEFINICION,
        "bootstrap": {"n": n_bootstrap, "semilla": semilla, "unidad": "sujeto", "metodo": "percentiles"},
        "fragmentos": int(len(remuestreo.fragmento_id)),
        "sujetos": remuestreo.n_sujetos,
        "casos_doble_etiqueta_excluidos": remuestreo.n_dobles,
        "por_arquitectura": resultado,
        "comparaciones_h1": comparaciones,
        "h1_parte_reddit": bool(comparaciones) and all(c["excluye_cero"] and c["diferencia"] > 0 for c in comparaciones),
        "ablacion": None,
    }

    if ablacion is not None:
        tablas_abl = cargar_varias(ablacion)
        por_arq_abl = {a: s for (a, _), s in tablas_abl.items()}
        filas = []
        for arq, por_semilla in por_arq_abl.items():
            if arq not in dist:
                continue
            dist_abl, f1_abl = remuestreo.distribucion(por_semilla)  # mismos sujetos y réplicas
            d = dist[arq] - dist_abl
            inf, sup = tc.intervalo(d)
            filas.append(
                {
                    "arquitectura": arq,
                    "macro_f1_enmascarado_terminos": float(np.mean(f1_abl)),
                    "diferencia_original_menos_ablacion": resultado[arq]["macro_f1_tres_clases"]["media_semillas"]
                    - float(np.mean(f1_abl)),
                    "ic95": [inf, sup],
                }
            )
        salida["ablacion"] = filas
    return salida


def _f(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


def _pct(m):
    return "—" if m["media"] is None else f"{100 * m['media']:.1f} ± {100 * m['de']:.1f}"


def resumen_markdown(r: dict) -> str:
    l = [
        "# PI1 en Reddit (prueba): macro-F1 de tres clases",
        "",
        f"- Definición: {r['definicion']}.",
        f"- {r['fragmentos']} publicaciones de {r['sujetos']} sujetos; bootstrap por sujeto, "
        f"{r['bootstrap']['n']} réplicas (semilla {r['bootstrap']['semilla']}).",
        "",
        "| Modelo | Macro-F1 (media de semillas) [IC 95 %] | DE entre semillas | Macro-F1 enmascarado | Ans.→Dep. (%) | Dep.→Ans. (%) | Control→alguna (%) |",
        "|---|---|---|---|---|---|---|",
    ]
    for arq, x in r["por_arquitectura"].items():
        m, s = x["macro_f1_tres_clases"], x["secundarias"]
        da = s["doble_asignacion"]
        l.append(
            f"| {NOMBRES[arq]} | {_f(m['media_semillas'])} [{_f(m['ic95'][0])}, {_f(m['ic95'][1])}] | "
            f"{_f(m['de_semillas'])} | {_f(s['f1_macro_enmascarado']['media'])} | "
            f"{_pct(da['ansiedad→depresion'])} | {_pct(da['depresion→ansiedad'])} | {_pct(da['control→alguna'])} |"
        )
    l += ["", "## Diferencias pareadas (H1, parte Reddit)", ""]
    for c in r["comparaciones_h1"]:
        l.append(
            f"- {NOMBRES[c['a']]} − {NOMBRES[c['b']]}: {_f(c['diferencia'])} "
            f"[{_f(c['ic95'][0])}, {_f(c['ic95'][1])}]; excluye 0: {'sí' if c['excluye_cero'] else 'no'}"
        )
    l.append(f"- Parte Reddit de H1 {'se cumple' if r['h1_parte_reddit'] else 'no se cumple'} "
             "(falta la parte clínica de la prueba de intersección-unión).")
    l += ["", "## Secundarias por etiqueta (media ± DE entre semillas)", "",
          "| Modelo | Etiqueta | F1 | Sensibilidad | Especificidad | AUROC |", "|---|---|---|---|---|---|"]
    for arq, x in r["por_arquitectura"].items():
        for e in ETIQUETAS:
            s = x["secundarias"][e]
            l.append(f"| {NOMBRES[arq]} | {e} | " + " | ".join(
                "—" if s[k]["media"] is None else f"{s[k]['media']:.3f} ± {s[k]['de']:.3f}"
                for k in ("f1", "sensibilidad", "especificidad", "auroc")) + " |")
    if r["ablacion"]:
        l += ["", "## Ablación: términos de la etiqueta enmascarados", ""]
        for a in r["ablacion"]:
            l.append(
                f"- {NOMBRES[a['arquitectura']]}: macro-F1 {_f(a['macro_f1_enmascarado_terminos'])}; "
                f"original − ablación {_f(a['diferencia_original_menos_ablacion'])} "
                f"[{_f(a['ic95'][0])}, {_f(a['ic95'][1])}]"
            )
    return "\n".join(l) + "\n"


def tabla_latex(r: dict) -> str:
    filas = []
    for arq, x in r["por_arquitectura"].items():
        m, da = x["macro_f1_tres_clases"], x["secundarias"]["doble_asignacion"]
        filas.append(
            f"{NOMBRES[arq]} & {m['media_semillas']:.3f} [{m['ic95'][0]:.3f}, {m['ic95'][1]:.3f}] & "
            f"{m['de_semillas']:.3f} & {100 * da['ansiedad→depresion']['media']:.1f}\\% & "
            f"{100 * da['depresion→ansiedad']['media']:.1f}\\% \\\\"
        )
    return (
        "% Generado por analizar_pi1.py: no editar a mano.\n"
        "\\begin{tabular}{|l|c|c|c|c|}\n\\hline\n"
        "\\textbf{Modelo} & \\textbf{Macro-F1 [IC 95\\%]} & \\textbf{DE} & \\textbf{Ans.$\\to$Dep.} & \\textbf{Dep.$\\to$Ans.} \\\\\n\\hline\n"
        + "\n".join(filas)
        + "\n\\hline\n\\end{tabular}\n"
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--predicciones", type=Path, nargs="+", required=True,
                   help="una o varias carpetas con indice.json del mismo dominio (p. ej., una por arquitectura)")
    p.add_argument("--ablacion", type=Path, nargs="+", help="predicciones de reddit_enmascarado (opcional)")
    p.add_argument("--salida", type=Path, default=registro.CARPETA_RESULTADOS / "pi1_reddit")
    p.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP_PPI)
    p.add_argument("--semilla-bootstrap", type=int, default=2026)
    a = p.parse_args(argv)
    r = analizar(a.predicciones, a.ablacion, a.bootstrap, a.semilla_bootstrap)
    a.salida.mkdir(parents=True, exist_ok=True)
    registro.escribir_json(a.salida / "pi1.json", r)
    (a.salida / "resumen.md").write_text(resumen_markdown(r), encoding="utf-8")
    (a.salida / "tabla_pi1.tex").write_text(tabla_latex(r), encoding="utf-8")
    print(resumen_markdown(r))
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())

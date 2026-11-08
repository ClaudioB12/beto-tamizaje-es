"""Aplica la regla del short paper (§IV-B) para elegir w con varias semillas.

Lee los ``tabla.csv`` que deja ``buscar_peso.py`` (uno por semilla, todos con
la misma rejilla de umbrales) y, por arquitectura, promedia entre semillas:
- macro-F1 de validación (la columna ``f1_macro`` de buscar_peso);
- doble asignación: media de «ansiedad→depresion» y «depresion→ansiedad»;
- proporción de controles con alguna etiqueta («control→alguna»).

Regla, fijada antes de ver los resultados: entre los w que pierden como mucho
0,01 de macro-F1 respecto de w = 0 y marcan como mucho el 10 % de los controles,
se elige el de menor doble asignación media (empate: el w menor). Como w = 0
cumple siempre ambas condiciones, la regla siempre da un resultado.

No decide nada por sí solo: deja la tabla y la propuesta para el asesor.

Uso:
    python elegir_peso.py --tablas s13/reddit/tabla.csv s42/reddit/tabla.csv s77/reddit/tabla.csv \
        --salida resultados/eleccion_w
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PERDIDA_MAXIMA = 0.01
CONTROLES_MAXIMOS = 0.10
_TOL = 1e-9


def cargar(tablas: list[Path]) -> pd.DataFrame:
    partes = []
    for i, ruta in enumerate(tablas):
        t = pd.read_csv(ruta)
        faltan = {"arquitectura", "w", "f1_macro", "ansiedad→depresion", "depresion→ansiedad", "control→alguna"} - set(t.columns)
        if faltan:
            raise ValueError(f"{ruta}: faltan columnas {sorted(faltan)}")
        partes.append(t.assign(tabla=str(ruta), semilla_idx=i))
    return pd.concat(partes, ignore_index=True)


def resumir(datos: pd.DataFrame) -> pd.DataFrame:
    d = datos.assign(doble=(datos["ansiedad→depresion"] + datos["depresion→ansiedad"]) / 2)
    g = d.groupby(["arquitectura", "w"])
    r = g.agg(
        semillas=("semilla_idx", "nunique"),
        f1_macro=("f1_macro", "mean"),
        f1_macro_de=("f1_macro", "std"),
        doble=("doble", "mean"),
        doble_de=("doble", "std"),
        controles=("control→alguna", "mean"),
    ).reset_index()
    if "umbrales_en_borde" in d.columns:
        borde = g["umbrales_en_borde"].apply(lambda s: int(s.fillna("").astype(str).str.len().gt(0).sum()))
        r = r.merge(borde.rename("semillas_con_umbral_en_borde").reset_index(), on=["arquitectura", "w"])
    base = r[r["w"] == 0].set_index("arquitectura")["f1_macro"]
    r["perdida_vs_w0"] = r["f1_macro"].sub(r["arquitectura"].map(base))
    r["cumple"] = (r["perdida_vs_w0"].mul(-1) <= PERDIDA_MAXIMA + _TOL) & (r["controles"] <= CONTROLES_MAXIMOS + _TOL)
    return r.sort_values(["arquitectura", "w"]).reset_index(drop=True)


def elegir(resumen: pd.DataFrame) -> dict:
    eleccion = {}
    for arq, t in resumen.groupby("arquitectura"):
        if 0 not in set(t["w"]):
            raise ValueError(f"{arq}: falta w = 0, que es la referencia de la regla")
        candidatos = t[t["cumple"]].sort_values(["doble", "w"])
        fila = candidatos.iloc[0]
        eleccion[arq] = {
            "w": float(fila["w"]),
            "doble_asignacion_media": float(fila["doble"]),
            "perdida_vs_w0": float(fila["perdida_vs_w0"]),
            "controles_marcados": float(fila["controles"]),
            "semillas": int(fila["semillas"]),
            "solo_cumple_w0": bool(len(candidatos) == 1 and fila["w"] == 0),
        }
    return eleccion


def resumen_markdown(resumen: pd.DataFrame, eleccion: dict) -> str:
    L = ["# Elección de w (validación, media de semillas)", "",
         f"Regla: pérdida de macro-F1 ≤ {PERDIDA_MAXIMA} frente a w = 0 y controles marcados ≤ {CONTROLES_MAXIMOS:.0%}; "
         "entre los que cumplen, menor doble asignación (empate: w menor).", "",
         "| Arquitectura | w | Semillas | Macro-F1 (media ± DE) | Pérdida vs. w=0 | Doble asignación (media ± DE) | Controles marcados | Cumple |",
         "|---|---|---|---|---|---|---|---|"]
    for _, f in resumen.iterrows():
        de = lambda x: "—" if pd.isna(x) else f"{x:.3f}"
        L.append(f"| {f['arquitectura']} | {f['w']:g} | {f['semillas']} | {f['f1_macro']:.3f} ± {de(f['f1_macro_de'])} | "
                 f"{f['perdida_vs_w0']:+.3f} | {100*f['doble']:.1f} % ± {100*f['doble_de'] if not pd.isna(f['doble_de']) else 0:.1f} | "
                 f"{100*f['controles']:.1f} % | {'sí' if f['cumple'] else 'no'} |")
    L += ["", "## Propuesta para el asesor", ""]
    for arq, e in eleccion.items():
        extra = " (solo cumple w = 0: reportar la doble asignación como limitación)" if e["solo_cumple_w0"] else ""
        L.append(f"- **{arq}: w = {e['w']:g}**{extra}; doble asignación {100*e['doble_asignacion_media']:.1f} %, "
                 f"pérdida {e['perdida_vs_w0']:+.3f}, controles {100*e['controles_marcados']:.1f} %.")
    if "semillas_con_umbral_en_borde" in resumen.columns and resumen["semillas_con_umbral_en_borde"].gt(0).any():
        L += ["", "Atención: hay combinaciones con algún umbral en el borde de la rejilla "
              "(columna «semillas_con_umbral_en_borde» de resumen.csv)."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tablas", nargs="+", type=Path, required=True)
    ap.add_argument("--salida", type=Path, required=True)
    a = ap.parse_args(argv)
    resumen = resumir(cargar(a.tablas))
    eleccion = elegir(resumen)
    a.salida.mkdir(parents=True, exist_ok=True)
    resumen.to_csv(a.salida / "resumen.csv", index=False)
    (a.salida / "eleccion.json").write_text(json.dumps(eleccion, ensure_ascii=False, indent=2), encoding="utf-8")
    md = resumen_markdown(resumen, eleccion)
    (a.salida / "resumen.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())

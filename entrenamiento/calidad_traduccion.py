"""Control de calidad de la traducción: acuerdo entre dos evaluadores (M4).

Cada evaluador completa su copia de ``muestra_calidad.csv`` (la que deja
``traducir_reddit.py``), con las columnas ``adecuacion_1a5``, ``fluidez_1a5``,
``senal_salud_mental_1a5``, ``aceptable_si_no`` e ``inversion_polaridad_si_no``
(sí cuando la traducción invierte el sentido clínico, p. ej., «no estoy
deprimido» → «estoy deprimido»). La polaridad es OBLIGATORIA: el paper exige
ninguna inversión, y sin esa columna el criterio no puede comprobarse.

Calcula:
- κ de Cohen con ponderación cuadrática para cada criterio de 1 a 5;
- κ de Cohen para «aceptable»;
- porcentaje de pares aceptables según cada evaluador y según ambos;
- la lista de pares en desacuerdo, para resolverlos por consenso, y las
  inversiones de polaridad.

El criterio del paper es ≥ 90 % de pares aceptables y ninguna inversión de
polaridad, sobre la etiqueta de consenso (después de resolver los desacuerdos).

Uso:
    python calidad_traduccion.py --evaluador1 eval1.csv --evaluador2 eval2.csv --salida resultados/calidad
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

CRITERIOS = ("adecuacion_1a5", "fluidez_1a5", "senal_salud_mental_1a5")
SI = {"sí", "si", "s", "1", "true", "x"}
BINARIOS_VALIDOS = {"sí", "si", "no"}


def validar_evaluacion(e: pd.DataFrame, nombre: str) -> dict:
    """Comprueba que una planilla sea una evaluación HUMANA completa y en escala.

    - sin las columnas IA_ de la pre-revisión (se califica sobre la plantilla limpia);
    - un único nombre en ``evaluador``;
    - puntajes enteros de 1 a 5 y respuestas sí/no (``revisar`` u otros valores se rechazan,
      en lugar de contarse silenciosamente como «no»).
    Devuelve el nombre del evaluador y, si existe la columna, si vio sugerencias de IA.
    """
    ia = [c for c in e.columns if str(c).startswith("IA_")]
    if ia:
        raise ValueError(f"{nombre}: contiene columnas de la pre-revisión automática {ia}; "
                         "la evaluación humana se hace sobre la plantilla sin ellas.")
    if "evaluador" not in e.columns or e["evaluador"].fillna("").str.strip().eq("").any():
        raise ValueError(f"{nombre}: falta el nombre del evaluador en alguna fila (columna «evaluador»).")
    nombres = set(e["evaluador"].str.strip())
    if len(nombres) != 1:
        raise ValueError(f"{nombre}: la columna «evaluador» tiene varios nombres: {sorted(nombres)}")
    malos = []
    for c in CRITERIOS:
        v = pd.to_numeric(e[c], errors="coerce")
        malos += [f"{c}:{i}" for i in e.loc[~v.isin([1, 2, 3, 4, 5]), "fragmento_id"]]
    for c in ("aceptable_si_no", "inversion_polaridad_si_no"):
        v = e[c].fillna("").astype(str).str.strip().str.lower()
        malos += [f"{c}:{i}" for i in e.loc[~v.isin(BINARIOS_VALIDOS), "fragmento_id"]]
    if malos:
        raise ValueError(f"{nombre}: valores fuera de escala o vacíos (se aceptan 1-5 y sí/no): {malos[:10]}"
                         + (f" y {len(malos) - 10} más" if len(malos) > 10 else ""))
    vio = None
    if "vio_sugerencias_ia" in e.columns:
        vio = bool(e["vio_sugerencias_ia"].fillna("").astype(str).str.strip().str.lower().isin(SI).any())
    return {"evaluador": nombres.pop(), "vio_sugerencias_ia": vio}


def _si(serie: pd.Series) -> pd.Series:
    return serie.fillna("").astype(str).str.strip().str.lower().isin(SI)


def leer(ruta: Path) -> pd.DataFrame:
    for codificacion in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(ruta, dtype={"fragmento_id": str}, encoding=codificacion)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"No pude leer {ruta}")


def comparar(e1: pd.DataFrame, e2: pd.DataFrame) -> dict:
    if set(e1["fragmento_id"]) != set(e2["fragmento_id"]):
        raise ValueError("Los dos archivos no tienen los mismos pares (fragmento_id).")
    d = e1.merge(e2, on="fragmento_id", suffixes=("_1", "_2"))
    obligatorias = CRITERIOS + ("aceptable_si_no", "inversion_polaridad_si_no")
    ausentes = [f"{c} (evaluador {i})" for i, e in ((1, e1), (2, e2)) for c in obligatorias if c not in e.columns]
    if ausentes:
        raise ValueError(f"Faltan columnas obligatorias: {ausentes}")
    info = [validar_evaluacion(e, f"evaluador {i}") for i, e in ((1, e1), (2, e2))]
    if info[0]["evaluador"].lower() == info[1]["evaluador"].lower():
        raise ValueError("Las dos planillas tienen el mismo evaluador: se necesitan dos personas distintas.")
    faltan = [c for c in obligatorias for k in ("_1", "_2") if d[c + k].isna().any()]
    if faltan:
        raise ValueError(f"Hay celdas vacías en: {sorted(set(faltan))}")
    r = {"pares": len(d), "kappa": {}, "evaluadores": info,
         "kappa_tipo": {"criterios_1a5": "Cohen ponderado cuadrático (etiquetas 1-5)",
                        "aceptable": "Cohen sin ponderar (sí/no)"},
         "momento": "acuerdo previo al consenso"}
    for c in CRITERIOS:
        r["kappa"][c] = float(cohen_kappa_score(d[c + "_1"].astype(int), d[c + "_2"].astype(int),
                                                labels=[1, 2, 3, 4, 5], weights="quadratic"))
    a1, a2 = _si(d["aceptable_si_no_1"]), _si(d["aceptable_si_no_2"])
    r["kappa"]["aceptable"] = float(cohen_kappa_score(a1, a2))
    r["aceptables"] = {"evaluador1": float(a1.mean()), "evaluador2": float(a2.mean()), "ambos": float((a1 & a2).mean())}
    r["desacuerdos_aceptable"] = d.loc[a1 != a2, "fragmento_id"].tolist()
    inv = _si(d["inversion_polaridad_si_no_1"]) | _si(d["inversion_polaridad_si_no_2"])
    r["inversiones_polaridad"] = d.loc[inv, "fragmento_id"].tolist()
    return r


def resumen_markdown(r: dict) -> str:
    L = [f"# Control de calidad de la traducción ({r['pares']} pares)", "",
         "| Criterio | κ |", "|---|---|"]
    L += [f"| {c} | {k:.2f} |" for c, k in r["kappa"].items()]
    a = r["aceptables"]
    L += ["", f"- Aceptables según el evaluador 1: {a['evaluador1']:.1%}; según el evaluador 2: {a['evaluador2']:.1%}; "
          f"según ambos: {a['ambos']:.1%}.",
          f"- Pares en desacuerdo sobre «aceptable» ({len(r['desacuerdos_aceptable'])}), a resolver por consenso: "
          + (", ".join(r["desacuerdos_aceptable"]) or "ninguno") + "."]
    L.append(f"- Inversiones de polaridad: {len(r['inversiones_polaridad'])}"
             + (f" ({', '.join(r['inversiones_polaridad'])})" if r["inversiones_polaridad"] else "") + ".")
    L += ["", "Criterio del paper: ≥ 90 % de pares aceptables por consenso y ninguna inversión de polaridad. "
          "«Según ambos» es una cota inferior del consenso."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--evaluador1", type=Path, required=True)
    ap.add_argument("--evaluador2", type=Path, required=True)
    ap.add_argument("--salida", type=Path, required=True)
    a = ap.parse_args(argv)
    r = comparar(leer(a.evaluador1), leer(a.evaluador2))
    a.salida.mkdir(parents=True, exist_ok=True)
    (a.salida / "calidad.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    md = resumen_markdown(r)
    (a.salida / "resumen.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())

"""Desempeño en la prueba de Reddit según la calidad de la traducción (M4).

Es un análisis exploratorio. Separa las publicaciones de prueba en estratos
de calidad de la traducción y calcula en cada uno el macro-F1 de tres clases
(media de semillas, IC bootstrap por sujeto):
- «marcada»: la publicación figura en diagnostico_traduccion.csv (cobertura
  baja, bucle o inglés residual);
- terciles de la razón de palabras traducidas (palabras en español / palabras
  en inglés, sobre las primeras 400 palabras), entre las no marcadas.

Si el desempeño cae en los estratos de peor traducción, parte de la brecha
podría deberse a la traducción y no al dominio.

Uso:
    python analizar_traduccion.py --predicciones PRED/reddit [PRED2/reddit ...] \
        --datos datos_reddit/reddit_es.csv --diagnostico datos_reddit/diagnostico_traduccion.csv \
        --salida resultados/traduccion
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analizar_pi1 import NOMBRES, cargar_varias
from estadistica import tres_clases as tc

MAX_PALABRAS = 400


def _palabras(texto) -> int:
    return sum(any(c.isalpha() for c in w) for w in str(texto).split())


def estratos(datos: pd.DataFrame, diagnostico: pd.DataFrame | None) -> pd.DataFrame:
    """fragmento_id -> estrato de calidad."""
    d = datos[["fragmento_id", "texto_en", "texto_es"]].copy()
    d["fragmento_id"] = d["fragmento_id"].astype(str)
    n_en = d["texto_en"].map(lambda t: min(_palabras(t), MAX_PALABRAS))
    d["razon"] = d["texto_es"].map(_palabras) / n_en.clip(lower=1)
    marcadas = set(diagnostico["fragmento_id"].astype(str)) if diagnostico is not None else set()
    d["estrato"] = np.where(d["fragmento_id"].isin(marcadas), "marcada", "")
    libres = d["estrato"] == ""
    d.loc[libres, "estrato"] = pd.qcut(d.loc[libres, "razon"], 3, labels=["razón baja", "razón media", "razón alta"]).astype(str)
    return d[["fragmento_id", "razon", "estrato"]]


def analizar(carpetas, datos: pd.DataFrame, diagnostico, n_bootstrap: int, semilla: int) -> dict:
    tablas = cargar_varias(carpetas)
    est = estratos(datos, diagnostico).set_index("fragmento_id")
    salida = {"estratos": {}, "n_bootstrap": n_bootstrap}
    por_arq = {a: s for (a, _), s in tablas.items()}
    referencia = next(iter(next(iter(por_arq.values())).values()))
    ids = referencia["fragmento_id"].astype(str)
    if not ids.isin(est.index).all():
        raise ValueError("Hay publicaciones de prueba que no están en el archivo de datos traducidos.")
    etiqueta = ids.map(est["estrato"])
    for nombre in ["marcada", "razón baja", "razón media", "razón alta"]:
        mascara = (etiqueta == nombre).to_numpy()
        if mascara.sum() == 0:
            continue
        sub = {a: {s: t.loc[mascara].reset_index(drop=True) for s, t in ps.items()} for a, ps in por_arq.items()}
        ref = next(iter(next(iter(sub.values())).values()))
        rem = tc.Remuestreo(ref, n_bootstrap, semilla)
        fila = {"publicaciones": int(mascara.sum()), "sujetos": rem.n_sujetos,
                "razon_media": float(est.loc[ids[mascara], "razon"].mean()), "por_arquitectura": {}}
        for a, ps in sub.items():
            dist, f1s = rem.distribucion(ps)
            fila["por_arquitectura"][a] = {"macro_f1": float(np.mean(f1s)), "ic95": list(tc.intervalo(dist))}
        salida["estratos"][nombre] = fila
    return salida


def resumen_markdown(r: dict) -> str:
    arqs = [a for a in NOMBRES if any(a in e["por_arquitectura"] for e in r["estratos"].values())]
    L = ["# Desempeño según la calidad de la traducción (prueba de Reddit, exploratorio)", "",
         "| Estrato | Publicaciones | Razón media | " + " | ".join(NOMBRES[a] for a in arqs) + " |",
         "|---|---|---|" + "---|" * len(arqs)]
    for nombre, e in r["estratos"].items():
        celdas = []
        for a in arqs:
            x = e["por_arquitectura"].get(a)
            celdas.append("—" if x is None else f"{x['macro_f1']:.3f} [{x['ic95'][0]:.3f}, {x['ic95'][1]:.3f}]")
        L.append(f"| {nombre} | {e['publicaciones']} | {e['razon_media']:.2f} | " + " | ".join(celdas) + " |")
    L += ["", "Macro-F1 de tres clases (media de semillas) e IC 95 % por sujeto. Los estratos «razón» "
          "son terciles entre las traducciones no marcadas."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--predicciones", type=Path, nargs="+", required=True)
    ap.add_argument("--datos", type=Path, required=True, help="reddit_es.csv con texto_en y texto_es")
    ap.add_argument("--diagnostico", type=Path, help="diagnostico_traduccion.csv (opcional)")
    ap.add_argument("--salida", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--semilla-bootstrap", type=int, default=2026)
    a = ap.parse_args(argv)
    datos = pd.read_csv(a.datos, dtype={"fragmento_id": str})
    diag = pd.read_csv(a.diagnostico, dtype={"fragmento_id": str}) if a.diagnostico and a.diagnostico.exists() else None
    r = analizar(a.predicciones, datos, diag, a.bootstrap, a.semilla_bootstrap)
    a.salida.mkdir(parents=True, exist_ok=True)
    (a.salida / "traduccion.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    md = resumen_markdown(r)
    (a.salida / "resumen.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())

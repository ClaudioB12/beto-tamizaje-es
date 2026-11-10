"""Comprueba la integridad de la búsqueda de w antes de aplicar la regla.

Lee ``<raiz>/s<semilla>/<fuente>/por_corrida/*.json`` y ``tabla.csv`` de cada
semilla y exige:
- exactamente una medición por (arquitectura, w, semilla) esperada, sin
  duplicados ni combinaciones de más;
- la misma partición (sha256_asignaciones) y la misma rejilla de umbrales en
  todas;
- ninguna corrida de humo y la semilla del JSON igual a la de su carpeta;
- que cada fila de tabla.csv coincida con su JSON (f1_macro);
- si existe la huella de configuración (sha256_config), que sea la misma en
  todas las corridas de la misma semilla.

Escribe ``auditoria.json`` con el resultado y el sha256 de cada tabla.csv, que
el cuaderno guarda junto a la decisión de w. Termina con código 1 si algo falla.

Uso:
    python auditar_busqueda.py --raiz busqueda_w --semillas 13 42 77 \
        --pesos 0 0.1 0.3 0.5 1 --arquitecturas beto_ajustado tfidf_lr --salida resultados/auditoria_w
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

import registro


def auditar(raiz: Path, fuente: str, semillas, pesos, arquitecturas) -> dict:
    problemas: list[str] = []
    vistos: dict[tuple, Path] = {}
    particiones, rejillas, tablas = set(), set(), {}
    for semilla in semillas:
        carpeta = raiz / f"s{semilla}" / fuente
        tabla_ruta = carpeta / "tabla.csv"
        if not tabla_ruta.exists():
            problemas.append(f"falta {tabla_ruta}")
            continue
        tablas[str(tabla_ruta)] = registro.sha256_archivo(tabla_ruta)
        tabla = pd.read_csv(tabla_ruta)
        huellas = set()
        for archivo in sorted((carpeta / "por_corrida").glob("*.json")):
            j = registro.leer_json(archivo)
            clave = (j["arquitectura"], float(j["peso_negativo_debil"]), int(j["semilla"]))
            if clave[2] != semilla:
                problemas.append(f"{archivo}: semilla {clave[2]} en la carpeta s{semilla}")
            if clave in vistos:
                problemas.append(f"duplicado {clave}: {vistos[clave]} y {archivo}")
            vistos[clave] = archivo
            if j.get("humo"):
                problemas.append(f"{archivo}: corrida de humo")
            particiones.add(j.get("sha256_asignaciones"))
            rejillas.add(j.get("rejilla_umbrales"))
            if "sha256_config" in j:
                huellas.add(j["sha256_config"])
            fila = tabla[(tabla["arquitectura"] == clave[0]) & (tabla["w"].astype(float) == clave[1])]
            if len(fila) != 1:
                problemas.append(f"{tabla_ruta}: {len(fila)} filas para {clave[:2]}")
            elif abs(float(fila["f1_macro"].iloc[0]) - float(j["f1_macro"])) > 1e-12:
                problemas.append(f"{tabla_ruta}: f1_macro de {clave[:2]} no coincide con {archivo.name}")
        if len(huellas) > 1:
            problemas.append(f"s{semilla}: corridas con huellas de configuración distintas")
    esperadas = {(a, float(w), int(s)) for a, w, s in itertools.product(arquitecturas, pesos, semillas)}
    faltan = sorted(esperadas - set(vistos))
    sobran = sorted(set(vistos) - esperadas)
    problemas += [f"falta la combinación {c}" for c in faltan]
    problemas += [f"combinación no esperada {c}" for c in sobran]
    if len(particiones) != 1:
        problemas.append(f"particiones distintas: {sorted(map(str, particiones))}")
    if len(rejillas) != 1:
        problemas.append(f"rejillas distintas: {sorted(map(str, rejillas))}")
    return {
        "ok": not problemas,
        "problemas": problemas,
        "combinaciones": len(vistos),
        "esperadas": len(esperadas),
        "sha256_asignaciones": next(iter(particiones)) if len(particiones) == 1 else None,
        "rejilla_umbrales": next(iter(rejillas)) if len(rejillas) == 1 else None,
        "sha256_tablas": tablas,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raiz", type=Path, required=True)
    ap.add_argument("--fuente", default="reddit")
    ap.add_argument("--semillas", nargs="+", type=int, required=True)
    ap.add_argument("--pesos", nargs="+", type=float, required=True)
    ap.add_argument("--arquitecturas", nargs="+", required=True)
    ap.add_argument("--salida", type=Path, required=True)
    a = ap.parse_args(argv)
    r = auditar(a.raiz, a.fuente, a.semillas, a.pesos, a.arquitecturas)
    registro.escribir_json(a.salida / "auditoria.json", r)
    print(f"{r['combinaciones']}/{r['esperadas']} combinaciones; partición {r['sha256_asignaciones']}; rejilla {r['rejilla_umbrales']}")
    for p in r["problemas"]:
        print("PROBLEMA:", p)
    print("Integridad: OK" if r["ok"] else "Integridad: FALLA")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())

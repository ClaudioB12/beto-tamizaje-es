"""Comprueba, ANTES de entrenar, que dos particiones congeladas coinciden.

Para la ablación de términos (M1): ``reddit_enmascarado`` debe tener
exactamente las mismas publicaciones, con el mismo sujeto y en la misma
partición (train/val/test) que ``reddit``. El enmascaramiento podría alterar la
deduplicación o los casos retenidos; si eso ocurre, la comparación con la
partición original dejaría de ser pareada.

Uso:
    python comparar_particiones.py --a reddit --b reddit_enmascarado
Termina con código 1 si difieren.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

import registro


def comparar(ruta_a: Path, ruta_b: Path) -> dict:
    cols = ["fragmento_id", "sujeto_id", "particion"]
    a = pd.read_csv(ruta_a, dtype=str)[cols]
    b = pd.read_csv(ruta_b, dtype=str)[cols]
    solo_a = sorted(set(a["fragmento_id"]) - set(b["fragmento_id"]))
    solo_b = sorted(set(b["fragmento_id"]) - set(a["fragmento_id"]))
    m = a.merge(b, on="fragmento_id", suffixes=("_a", "_b"))
    distinta = m[(m["sujeto_id_a"] != m["sujeto_id_b"]) | (m["particion_a"] != m["particion_b"])]
    return {
        "iguales": not solo_a and not solo_b and distinta.empty,
        "fragmentos_a": len(a),
        "fragmentos_b": len(b),
        "solo_en_a": solo_a[:20],
        "solo_en_b": solo_b[:20],
        "n_solo_en_a": len(solo_a),
        "n_solo_en_b": len(solo_b),
        "n_asignacion_distinta": int(len(distinta)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a", required=True, help="Fuente de referencia, p. ej. reddit")
    ap.add_argument("--b", required=True, help="Fuente a comparar, p. ej. reddit_enmascarado")
    ap.add_argument("--particiones", type=Path, default=registro.CARPETA_PARTICIONES)
    a = ap.parse_args(argv)
    r = comparar(a.particiones / a.a / "asignaciones.csv", a.particiones / a.b / "asignaciones.csv")
    print(r)
    print("Particiones iguales: OK" if r["iguales"] else "Particiones DISTINTAS: no entrenar la ablación")
    return 0 if r["iguales"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

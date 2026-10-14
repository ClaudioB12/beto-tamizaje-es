"""Prepara el Reddit Mental Health Dataset para el PPI (paso previo a traducir).

Uso, desde entrenamiento/:
    python preparar_reddit.py --entrada "../../PPI/Data - Corpus"
    python preparar_reddit.py --entrada RUTA --por_clase 3000 --max_por_autor 5

Lee los 8 CSV descargados de Zenodo (registro 3941387), verifica su MD5,
limpia, deduplica, excluye a los autores que publicaron en más de una clase,
limita las publicaciones por autor, balancea por clase y anonimiza.

Escribe en datos_reddit/ (ignorado por git: contiene textos de Reddit):
    reddit_en.csv              fragmento_id, sujeto_id, subreddit, fecha, etiqueta, texto_en
Y en particiones/reddit_preparacion.json (versionable: solo conteos y hashes).

Fuente: Low et al. (2020), JMIR 22(10):e22635. Licencia PDDL 1.0; la
reutilización está sujeta a los términos de la API de Reddit, por eso los
textos nunca se versionan ni se publican: solo los identificadores y scripts.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent
SALIDA_POR_DEFECTO = RAIZ / "datos_reddit"
INFORME_POR_DEFECTO = RAIZ / "particiones" / "reddit_preparacion.json"

# Archivo de Zenodo -> (etiqueta, md5 publicado en el registro 3941387).
# Periodos "2018" (ene–abr 2018) y "pre" (dic 2018–dic 2019): sin el periodo
# COVID ("post") y sin "2019", que se solapa con "pre".
ARCHIVOS = {
    "depression_2018_features_tfidf_256.csv": ("depresion", "3b3bcc5d049fd0ccd782a53865e36911"),
    "depression_pre_features_tfidf_256.csv": ("depresion", "48406719aa1d6efc0d1f46f7b6ab127e"),
    "anxiety_2018_features_tfidf_256.csv": ("ansiedad", "054262ae8299ab50592d3bc88796907b"),
    "anxiety_pre_features_tfidf_256.csv": ("ansiedad", "a7cb3f928b9d4c34030ce014290d09e0"),
    "relationships_2018_features_tfidf_256.csv": ("control", "671da0c5c705732322a3a458b52e44d4"),
    "relationships_pre_features_tfidf_256.csv": ("control", "9efaa70f5e9e77baa8344ae70752d3b1"),
    "parenting_2018_features_tfidf_256.csv": ("control", "0b96864b1507b3fe45e9579be5f49d30"),
    "parenting_pre_features_tfidf_256.csv": ("control", "aa495d72197c4cd87545dd448461d030"),
}
COLUMNAS = ["subreddit", "author", "date", "post"]
TEXTOS_ELIMINADOS = {"[removed]", "[deleted]", ""}


class ArchivoInvalidoError(RuntimeError):
    """Falta un archivo de Zenodo o su contenido no coincide con el publicado."""


def limpiar(texto: str) -> str:
    """Quita enlaces y menciones de usuario; colapsa espacios."""
    texto = re.sub(r"https?://\S+|www\.\S+", " ", texto)
    texto = re.sub(r"(?<!\w)/?u/[A-Za-z0-9_-]+", " ", texto)
    texto = texto.replace("&amp;", "&").replace("&gt;", ">").replace("&lt;", "<")
    return " ".join(texto.split())


def anonimizar(autor: str, sal: str) -> str:
    return "rd-" + hashlib.sha256((sal + autor).encode("utf-8")).hexdigest()[:16]


def md5_archivo(ruta: Path) -> str:
    h = hashlib.md5()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def verificar_archivos(entrada: Path, verificar_md5: bool = True) -> dict[str, str]:
    """Comprueba que estén los 8 archivos y que su MD5 coincida con Zenodo."""
    faltantes = [n for n in ARCHIVOS if not (entrada / n).exists()]
    if faltantes:
        raise ArchivoInvalidoError(f"Faltan en {entrada}: {faltantes}")
    if not verificar_md5:
        return {}
    distintos, md5s = [], {}
    for nombre, (_, esperado) in ARCHIVOS.items():
        md5s[nombre] = md5_archivo(entrada / nombre)
        if md5s[nombre] != esperado:
            distintos.append(nombre)
    if distintos:
        raise ArchivoInvalidoError(
            f"MD5 distinto al publicado en Zenodo: {distintos}. Vuelve a descargarlos; "
            "si los abriste y guardaste con Excel, se modificaron."
        )
    return md5s


def preparar(
    crudo: pd.DataFrame,
    *,
    por_clase: int,
    max_por_autor: int,
    min_palabras: int,
    semilla: int,
    sal: str,
) -> tuple[pd.DataFrame, dict]:
    """De las filas crudas (con columna ``etiqueta``) al CSV en inglés."""
    traza = {"leidas": len(crudo)}

    datos = crudo.dropna(subset=["post", "author"]).copy()
    datos = datos[~datos["post"].str.strip().isin(TEXTOS_ELIMINADOS)]
    datos = datos[~datos["author"].isin({"[deleted]", "AutoModerator"})]
    datos["texto_en"] = datos["post"].map(limpiar)
    datos = datos[datos["texto_en"].str.split().str.len() >= min_palabras]
    traza["tras_limpieza"] = len(datos)

    clave = datos["texto_en"].str.lower()
    datos = datos[~clave.duplicated(keep="first")]
    traza["tras_duplicados"] = len(datos)

    clases_por_autor = datos.groupby("author")["etiqueta"].nunique()
    en_varias = clases_por_autor[clases_por_autor > 1].index
    datos = datos[~datos["author"].isin(en_varias)]
    traza["autores_en_varias_clases_excluidos"] = int(len(en_varias))
    traza["tras_excluir_autores"] = len(datos)

    # Ningún autor domina su clase: como máximo max_por_autor publicaciones.
    datos = datos.sample(frac=1, random_state=semilla)
    datos = datos.groupby("author", sort=False).head(max_por_autor)
    traza["tras_limite_por_autor"] = len(datos)

    disponibles = datos["etiqueta"].value_counts().to_dict()
    datos = pd.concat(
        [g.sample(n=min(por_clase, len(g)), random_state=semilla) for _, g in datos.groupby("etiqueta")],
        ignore_index=True,
    )
    traza["disponibles_por_clase"] = {k: int(v) for k, v in sorted(disponibles.items())}
    traza["seleccionadas_por_clase"] = {
        k: int(v) for k, v in datos["etiqueta"].value_counts().sort_index().items()
    }

    datos = datos.sort_values(["etiqueta", "author", "date", "texto_en"]).reset_index(drop=True)
    salida = pd.DataFrame(
        {
            "fragmento_id": [f"rd-{i:06d}" for i in range(len(datos))],
            "sujeto_id": datos["author"].map(lambda a: anonimizar(a, sal)),
            "subreddit": datos["subreddit"],
            "fecha": datos["date"],
            "etiqueta": datos["etiqueta"],
            "texto_en": datos["texto_en"],
        }
    )
    palabras = salida["texto_en"].str.split().str.len()
    traza["palabras_en"] = {
        "media": round(float(palabras.mean()), 1),
        "mediana": float(palabras.median()),
        "p90": float(palabras.quantile(0.9)),
        "pct_mas_de_190": round(100 * float((palabras > 190).mean()), 1),
    }
    traza["sujetos"] = int(salida["sujeto_id"].nunique())
    return salida, traza


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--entrada", type=Path, required=True, help="Carpeta con los 8 CSV de Zenodo")
    p.add_argument("--salida", type=Path, default=SALIDA_POR_DEFECTO)
    p.add_argument("--informe", type=Path, default=INFORME_POR_DEFECTO)
    p.add_argument("--por_clase", type=int, default=3000)
    p.add_argument("--max_por_autor", type=int, default=5)
    p.add_argument("--min_palabras", type=int, default=10)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--sal", default="ppi-upeu-2026")
    p.add_argument("--sin_md5", action="store_true", help="Omite la verificación MD5 (no recomendado)")
    p.add_argument("--sobrescribir", action="store_true")
    args = p.parse_args(argv)

    destino = args.salida / "reddit_en.csv"
    if destino.exists() and not args.sobrescribir:
        print(f"Ya existe {destino}. Usa --sobrescribir si es intencional.", file=sys.stderr)
        return 1

    try:
        md5s = verificar_archivos(args.entrada, verificar_md5=not args.sin_md5)
    except ArchivoInvalidoError as error:
        print(error, file=sys.stderr)
        return 1

    partes = []
    for nombre, (etiqueta, _) in ARCHIVOS.items():
        df = pd.read_csv(args.entrada / nombre, usecols=COLUMNAS, dtype=str)
        partes.append(df.assign(etiqueta=etiqueta, archivo=nombre))
    salida, traza = preparar(
        pd.concat(partes, ignore_index=True),
        por_clase=args.por_clase,
        max_por_autor=args.max_por_autor,
        min_palabras=args.min_palabras,
        semilla=args.semilla,
        sal=args.sal,
    )

    args.salida.mkdir(parents=True, exist_ok=True)
    salida.to_csv(destino, index=False, encoding="utf-8", lineterminator="\n")

    informe = {
        "fuente": "Reddit Mental Health Dataset (Low et al., 2020), Zenodo 10.17605/OSF.IO/7PEYQ, PDDL 1.0",
        "parametros": {k: v for k, v in vars(args).items() if k not in {"entrada", "salida", "informe", "sal"}},
        "md5_verificados": md5s or "omitido",
        "sha256_reddit_en": hashlib.sha256(destino.read_bytes()).hexdigest(),
        "traza": traza,
    }
    args.informe.parent.mkdir(parents=True, exist_ok=True)
    args.informe.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(traza, ensure_ascii=False, indent=2))
    print(f"\nEscrito {destino} ({len(salida)} filas) e informe en {args.informe}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Traduce datos_reddit/reddit_en.csv al español con MarianMT, en local.

Uso, desde entrenamiento/ (GPU recomendada: Colab T4):
    pip install -r requirements-traduccion.txt
    python traducir_reddit.py
    python traducir_reddit.py --lote 32 --max_palabras 400
    python traducir_reddit.py --limite 200          # piloto: mide la velocidad y estima el total

Sin APIs externas (proposición de diseño PD3): el modelo se descarga una vez
de Hugging Face y se ejecuta en la máquina. Si Colab se desconecta, vuelve a
lanzar el mismo comando: retoma desde el último bloque guardado.

Escribe en datos_reddit/ (ignorado por git):
    reddit_es.csv            reddit_en.csv + columna texto_es
    muestra_calidad.csv      pares EN/ES para el control de calidad manual
Y en particiones/reddit_traduccion.json los metadatos de la traducción.

Control de calidad (PPI, sección 2.2, nueva subfase): dos evaluadores
bilingües califican de 1 a 5 la adecuación, la fluidez y la preservación de
la señal de salud mental, siguiendo a Casamayor-Segarra et al. (2026).
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Callable

import pandas as pd

RAIZ = Path(__file__).resolve().parent
CARPETA = RAIZ / "datos_reddit"
INFORME = RAIZ / "particiones" / "reddit_traduccion.json"
MODELO_POR_DEFECTO = "Helsinki-NLP/opus-mt-en-es"

# Marian admite 512 tokens; se deja margen para no truncar en silencio.
MAX_TOKENS_TROZO = 400

_FIN_DE_ORACION = re.compile(r"(?<=[.!?])\s+")


def recortar_palabras(texto: str, max_palabras: int) -> str:
    """BETO lee como máximo 256–512 tokens: traducir más es gastar GPU en balde."""
    palabras = texto.split()
    return texto if len(palabras) <= max_palabras else " ".join(palabras[:max_palabras])


def trocear(texto: str, max_tokens: int, contar: Callable[[str], int]) -> list[str]:
    """Agrupa oraciones en trozos de como máximo ``max_tokens`` tokens.

    Una oración más larga que el límite se parte por palabras. Ningún trozo
    queda vacío y, unidos con espacios, conservan todas las palabras.
    """
    trozos: list[str] = []
    actual = ""
    for oracion in _FIN_DE_ORACION.split(texto.strip()):
        if not oracion:
            continue
        for pieza in _partir_oracion(oracion, max_tokens, contar):
            candidato = f"{actual} {pieza}".strip()
            if actual and contar(candidato) > max_tokens:
                trozos.append(actual)
                actual = pieza
            else:
                actual = candidato
    if actual:
        trozos.append(actual)
    return trozos


def _partir_oracion(oracion: str, max_tokens: int, contar: Callable[[str], int]) -> list[str]:
    if contar(oracion) <= max_tokens:
        return [oracion]
    piezas, actual = [], []
    for palabra in oracion.split():
        if actual and contar(" ".join([*actual, palabra])) > max_tokens:
            piezas.append(" ".join(actual))
            actual = [palabra]
        else:
            actual.append(palabra)
    if actual:
        piezas.append(" ".join(actual))
    return piezas


def muestra_estratificada(df: pd.DataFrame, n: int, semilla: int) -> pd.DataFrame:
    """Muestra para el control de calidad, estratificada por etiqueta y longitud."""
    largo = pd.qcut(df["texto_en"].str.split().str.len(), q=3, labels=["corto", "medio", "largo"], duplicates="drop")
    estratos = df.assign(_estrato=df["etiqueta"] + "|" + largo.astype(str))
    por_estrato = max(1, n // estratos["_estrato"].nunique())
    muestra = (
        estratos.groupby("_estrato", group_keys=False)
        .sample(n=por_estrato, random_state=semilla, replace=False)
        if estratos["_estrato"].value_counts().min() >= por_estrato
        else estratos.sample(n=min(n, len(estratos)), random_state=semilla)
    )
    columnas = ["fragmento_id", "etiqueta", "texto_en", "texto_es"]
    salida = muestra[columnas].reset_index(drop=True)
    for criterio in ("adecuacion_1a5", "fluidez_1a5", "senal_salud_mental_1a5", "aceptable_si_no", "evaluador", "comentario"):
        salida[criterio] = ""
    return salida


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--entrada", type=Path, default=CARPETA / "reddit_en.csv")
    p.add_argument("--salida", type=Path, default=CARPETA / "reddit_es.csv")
    p.add_argument("--modelo", default=MODELO_POR_DEFECTO)
    p.add_argument("--revision", default=None, help="Commit del modelo en Hugging Face (fíjalo y regístralo)")
    p.add_argument("--lote", type=int, default=16)
    p.add_argument("--max_palabras", type=int, default=400)
    p.add_argument("--guardar_cada", type=int, default=500)
    p.add_argument("--muestra", type=int, default=120)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--limite", type=int, default=None,
                   help="Traduce solo N pendientes y estima el tiempo total (piloto)")
    args = p.parse_args(argv)

    import torch
    from transformers import MarianMTModel, MarianTokenizer

    origen = pd.read_csv(args.entrada, dtype=str, keep_default_na=False, encoding="utf-8")
    hechos: dict[str, str] = {}
    if args.salida.exists():
        previo = pd.read_csv(args.salida, dtype=str, keep_default_na=False, encoding="utf-8")
        hechos = dict(zip(previo["fragmento_id"], previo["texto_es"]))
        hechos = {k: v for k, v in hechos.items() if v.strip()}
    pendientes = origen[~origen["fragmento_id"].isin(hechos)]
    print(f"{len(hechos)} ya traducidas, {len(pendientes)} pendientes")
    total_pendientes = len(pendientes)
    if args.limite:
        pendientes = pendientes.head(args.limite)
    inicio_reloj = time.perf_counter()

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = MarianTokenizer.from_pretrained(args.modelo, revision=args.revision)
    modelo = MarianMTModel.from_pretrained(args.modelo, revision=args.revision).to(dispositivo).eval()
    contar = lambda t: len(tokenizer(t, add_special_tokens=True)["input_ids"])

    def traducir_lote(textos: list[str]) -> list[str]:
        entrada = tokenizer(textos, return_tensors="pt", padding=True, truncation=True, max_length=512).to(dispositivo)
        with torch.no_grad():
            salida = modelo.generate(**entrada, num_beams=4, max_new_tokens=512)
        return tokenizer.batch_decode(salida, skip_special_tokens=True)

    def guardar() -> None:
        resultado = origen.assign(texto_es=origen["fragmento_id"].map(hechos).fillna(""))
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        temporal = args.salida.with_suffix(".tmp")
        resultado.to_csv(temporal, index=False, encoding="utf-8", lineterminator="\n")
        temporal.replace(args.salida)

    desde_guardado = 0
    for inicio in range(0, len(pendientes), args.lote):
        bloque = pendientes.iloc[inicio : inicio + args.lote]
        trozos, dueno = [], []
        for fid, texto in zip(bloque["fragmento_id"], bloque["texto_en"]):
            for trozo in trocear(recortar_palabras(texto, args.max_palabras), MAX_TOKENS_TROZO, contar):
                trozos.append(trozo)
                dueno.append(fid)
        traducidos = []
        for i in range(0, len(trozos), args.lote):
            traducidos.extend(traducir_lote(trozos[i : i + args.lote]))
        por_fragmento: dict[str, list[str]] = {}
        for fid, es in zip(dueno, traducidos):
            por_fragmento.setdefault(fid, []).append(es)
        hechos.update({fid: " ".join(partes) for fid, partes in por_fragmento.items()})
        desde_guardado += len(bloque)
        if desde_guardado >= args.guardar_cada:
            guardar()
            desde_guardado = 0
            print(f"  {len(hechos)}/{len(origen)} traducidas", flush=True)
    guardar()

    if args.limite:
        segundos = time.perf_counter() - inicio_reloj
        por_texto = segundos / max(1, len(pendientes))
        restante_h = por_texto * (total_pendientes - len(pendientes)) / 3600
        print(
            f"Piloto: {len(pendientes)} textos en {segundos / 60:.1f} min "
            f"({por_texto:.2f} s/texto en {dispositivo}). "
            f"Faltan {total_pendientes - len(pendientes)}: unas {restante_h:.1f} h. "
            "Revisa algunas traducciones en el CSV y lanza el comando sin --limite: retoma donde quedó."
        )
        return 0

    final = pd.read_csv(args.salida, dtype=str, keep_default_na=False, encoding="utf-8")
    vacias = int((final["texto_es"].str.strip() == "").sum())
    if vacias:
        print(f"Quedaron {vacias} filas sin traducir; vuelve a lanzar el comando.", file=sys.stderr)
        return 1

    muestra = muestra_estratificada(final, args.muestra, args.semilla)
    muestra.to_csv(args.salida.parent / "muestra_calidad.csv", index=False, encoding="utf-8-sig")

    import transformers

    informe = {
        "modelo": args.modelo,
        "revision": args.revision or "no fijada (registrar el commit usado)",
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "dispositivo": dispositivo,
        "decodificacion": {"num_beams": 4, "max_new_tokens": 512, "max_tokens_trozo": MAX_TOKENS_TROZO},
        "max_palabras_origen": args.max_palabras,
        "filas": len(final),
        "sha256_reddit_en": hashlib.sha256(args.entrada.read_bytes()).hexdigest(),
        "sha256_reddit_es": hashlib.sha256(args.salida.read_bytes()).hexdigest(),
        "muestra_calidad": len(muestra),
    }
    INFORME.parent.mkdir(parents=True, exist_ok=True)
    INFORME.write_text(json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Listo: {args.salida} y {args.salida.parent / 'muestra_calidad.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

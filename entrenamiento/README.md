# Entrenamiento — Componente A del PPI

Pipeline completo del experimento computacional (PPI §2.2 y §2.5): datos,
partición, tres arquitecturas × cinco semillas, evaluación, estadística y
exportación. Funciona con el dataset sintético (pruebas) y con la fuente `reddit`, que
sustituye a MentalRiskES como dominio de redes sociales (ver más abajo).

## Flujo

```bash
python particionar.py --fuente sintetico            # 1. congela la partición (una vez)
python entrenar.py    --fuente sintetico --humo     # 2a. prueba de humo: 1 época, rápida
python entrenar.py    --fuente sintetico            # 2b. las 15 corridas reales
python evaluar.py     --fuente sintetico            # 3. umbrales en val, predicciones en test
python analizar.py    --fuente sintetico            # 4. PE1, PE2, PE3, H1 con IC bootstrap
python exportar.py    --fuente sintetico            # 5. modelo elegido por validación
```

Cada paso verifica por hash que trabaja sobre lo que produjo el anterior:

| Paso | Lee | Escribe | Se niega si |
|---|---|---|---|
| particionar | fuente | `particiones/<fuente>/` | ya existe (sin `--sobrescribir`) |
| entrenar | partición congelada | `corridas/<exp>/<arq>-s<semilla>/` | la entrada o las asignaciones cambiaron |
| evaluar | corridas completas | `umbrales.json`, `predicciones/<exp>/` | una corrida usó otra partición |
| analizar | predicciones | `resultados/<exp>/resultados.json`, `resumen.md` | un CSV no coincide con su hash |
| exportar | corridas + umbrales | `exportados/<fuente>/<arq>-s<semilla>/` | faltan umbrales o el destino existe |

`<exp>` es la fuente, o `<fuente>-humo` con `--humo`. Los resultados de humo
nunca comparten carpeta con los reales, y `exportar.py` no puede leerlos.

## Fuente `reddit` (Reddit Mental Health Dataset, traducido)

MentalRiskES no está disponible (solicitud sin respuesta). Se sustituye por el
Reddit Mental Health Dataset (Low et al., 2020; Zenodo 10.17605/OSF.IO/7PEYQ;
licencia PDDL 1.0, reutilización sujeta a los términos de la API de Reddit).
Está en inglés y se traduce en local con MarianMT. Los textos van a
`datos_reddit/`, que git ignora: solo se versionan scripts, conteos y hashes.

```bash
pip install -r requirements.txt -r requirements-traduccion.txt
python preparar_reddit.py --entrada "RUTA/Data - Corpus"   # verifica MD5, limpia, balancea, anonimiza
python traducir_reddit.py                                  # EN -> ES; reanudable; deja muestra_calidad.csv
python traducir_reddit.py --diagnosticar                   # revisa una salida: cobertura, bucles, inglés residual
python traducir_reddit.py --rehacer_marcadas               # vuelve a traducir solo las filas marcadas
python traducir_reddit.py --solo_muestra                   # regenera muestra_calidad.csv (original recortado a lo traducido)
python particionar.py --fuente reddit                      # y desde aquí, el flujo de siempre
```

- **Archivos de entrada (8):** `depression`, `anxiety`, `relationships` y `parenting`, periodos `2018` y `pre`. Sin `post` (COVID) ni `2019` (se solapa con `pre`).
- **Etiquetas (D2):** r/depression → depresión 1, ansiedad desconocida (-1). r/anxiety → ansiedad 1, depresión -1. Control → 0 y 0.
- **Sujeto:** autor anonimizado con hash. Se excluyen los autores que publicaron en más de una clase y se limita a 5 publicaciones por autor.
- **Control de calidad de la traducción:** dos evaluadores bilingües califican `muestra_calidad.csv` antes de congelar la partición.
- **Diagnóstico automático:** al terminar (o con `--diagnosticar`) se escribe `diagnostico_traduccion.csv` con las filas cuya traducción omite contenido (cobertura < 0,8), entra en bucle o deja oraciones en inglés. La traducción se hace en trozos de hasta 100 tokens: con trozos de 400, opus-mt omitía en torno a un tercio de los textos de más de 200 palabras.
- **Publicaciones en otro idioma:** el cargador descarta las que no están escritas en alfabeto latino (traducirlas EN→ES no tiene sentido).
- **Ablación de fuga léxica:** `--fuente reddit_enmascarado` lee el mismo archivo con los términos de la etiqueta (depresión, ansiedad, pánico…) reemplazados por `[MASK]`.

## Colab

`requirements.txt` instala torch 2.13.0, pero Colab trae un `torchvision`
compilado para otra versión de torch. transformers lo importa si existe y falla
con `operator torchvision::nms does not exist` (o `Could not import module
'EarlyStoppingCallback'`). El pipeline no usa torchvision: desinstálalo después
de instalar los requisitos.

```bash
pip install -q -r requirements.txt
pip uninstall -y -q torchvision torchaudio
```

Los avisos de pip sobre `google-colab` y `pandas` no afectan a los scripts.

## Retomar tras un corte

`entrenar.py` escribe `manifiesto.json` al final de cada corrida. Si Colab se
desconecta, se vuelve a lanzar el mismo comando: lo completo se salta y lo
interrumpido se borra y se repite.

## Estructura

```
entrenamiento/
├── config/experimento.yaml   configuración completa del experimento
├── configuracion.py          lectura validada del YAML
├── registro.py               hashes, partición congelada, entorno
├── datos/                    esquema, cargadores, deduplicación, partición
├── modelos/                  beto.py (ajustado y sonda), tfidf.py, fábrica
├── evaluacion/               umbrales, instrumentos PHQ-9/GAD-7, severidad, predicciones
├── estadistica/              métricas, bootstrap, pruebas, kappas, escala
├── particionar.py · entrenar.py · evaluar.py · analizar.py · exportar.py
├── tests/                    132 tests, ~1 min en CPU
│
├── generar_dataset.py        LEGADO: genera el CSV sintético (no regenerar)
└── train.py                  LEGADO: no usar
```

## Decisiones pendientes que el código ya contempla

| Decisión | Dónde vive | Estado en el código |
|---|---|---|
| D2 etiqueta desconocida | `datos/esquema.py`, pérdida enmascarada | implementada con `-1` |
| D3 definición de F1 macro | `estadistica/metricas.py` | propuesta; cambiarla es tocar una función |
| D4 bandas de severidad | `evaluacion/severidad.py` | **sin cortes**: se leen de un archivo congelado |
| D5 barrido de la sonda | `config/experimento.yaml` → `sonda` | cada semilla elige su tasa por `eval_loss` |
| D6 prueba de semillas | `estadistica/pruebas.py` | Mann–Whitney unilateral + Wilcoxon del PPI con su límite |

### Bandas de severidad (D4)

`evaluar.py --bandas archivo.json` asigna severidad solo si el equipo congeló
los cortes. Formato:

```json
{
  "congelado_en": "AAAA-MM-DD",
  "referencia": "acta o decisión que aprueba los cortes",
  "bandas": {
    "ansiedad":  {"cortes": [c1, c2, c3]},
    "depresion": {"cortes": [c1, c2, c3, c4]}
  }
}
```

Ansiedad se ancla a GAD-7 (4 categorías) y depresión a PHQ-9 (5 categorías).
Los rangos de los instrumentos están en `evaluacion/instrumentos.py`. Sin este
archivo, las columnas `sev_*` quedan vacías y PE3 aparece como no disponible.

## Tres detalles técnicos con tests que los fijan

- **BETO no trae pesos del pooler** (0 de 205 tensores). La sonda lee el vector
  [CLS]. Un modelo de sonda se carga siempre con `ClasificadorBeto.cargar`.
- **Acumulación de gradiente.** Con pérdida propia, transformers 5.x no divide
  entre pasos de acumulación; `TrainerEnmascarado` lo corrige.
- **Wilcoxon con 5 semillas** tiene p mínimo 0,0625 bilateral; el análisis usa
  Mann–Whitney exacta unilateral (p mínimo 1/252) y reporta ambas.

## Lo que falta y por qué no está aquí

- **Cargador de MentalRiskES**: el corpus no llegó; lo sustituye la fuente `reddit`.
- **Conjunto clínico**: requiere su cargador; se evalúa una sola vez con todo congelado.
- **Integración con el ml-service y el backend**: el modelo multietiqueta cambia
  el contrato de `/ml/clasificar`, la tabla `mensajes` y el banco de plantillas.
  Necesita las bandas de D4 y plantillas de comorbilidad aprobadas por los
  psicólogos (PD1). `exportar.py` deja el paquete listo, pero no toca `ml-service/`.

## Por qué `train.py` es legado

Parte por fila (el 100 % de su validación tenía la frase base en entrenamiento),
sobrescribe `ml-service/modelos/` al terminar y es multiclase de 7 clases.

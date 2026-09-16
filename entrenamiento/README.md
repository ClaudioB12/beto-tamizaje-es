# Entrenamiento — Componente A del PPI

Pipeline completo del experimento computacional (PPI §2.2 y §2.5): datos,
partición, tres arquitecturas × cinco semillas, evaluación, estadística y
exportación. Todo funciona hoy con el dataset sintético; con MentalRiskES solo
falta escribir su cargador en `datos/` y registrarlo en `datos/fuentes.py`.

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

- **Cargador de MentalRiskES**: requiere el corpus y la decisión D1.
- **Conjunto clínico**: requiere su cargador; se evalúa una sola vez con todo congelado.
- **Integración con el ml-service y el backend**: el modelo multietiqueta cambia
  el contrato de `/ml/clasificar`, la tabla `mensajes` y el banco de plantillas.
  Necesita las bandas de D4 y plantillas de comorbilidad aprobadas por los
  psicólogos (PD1). `exportar.py` deja el paquete listo, pero no toca `ml-service/`.

## Por qué `train.py` es legado

Parte por fila (el 100 % de su validación tenía la frase base en entrenamiento),
sobrescribe `ml-service/modelos/` al terminar y es multiclase de 7 clases.

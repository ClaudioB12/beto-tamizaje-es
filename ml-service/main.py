"""Servicio ML — Agente BETO con Modelos de Deep Learning (BETO).

Carga los modelos entrenados con BERT en español (BETO):
1. Modelos/Crisis: Inferencia de riesgo vital / crisis clínica (binario con BETO).
2. Modelos/Clasificacion: Inferencia de emoción y nivel de severidad (multiclase con BETO).
3. Base de Recuperación: Conocimiento semántico de psicoeducación clínica.
"""
import os
import json
import logging
import time
from typing import Literal, Optional, Tuple

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml-service")

app = FastAPI(
    title="Servicio ML — Agente BETO (Deep Learning Inferencia)",
    version="1.0.0",
    description="Microservicio de inferencia NLP con modelos BETO entrenados y recuperación clínica",
)

Emocion = Literal["ansiedad", "depresion", "control"]
Severidad = Literal["leve", "moderado", "severo"]

DIRECTORIO_BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO_CLASIF = os.path.join(DIRECTORIO_BASE, "modelos", "clasificacion")
RUTA_MODELO_CRISIS = os.path.join(DIRECTORIO_BASE, "modelos", "crisis")

# ----------------------------------------------------------------------
# Esquemas de entrada y salida
# ----------------------------------------------------------------------

class CrisisRequest(BaseModel):
    texto: str
    paciente_id: str

class CrisisMLResponse(BaseModel):
    riesgo: bool
    motivo: Optional[str] = None

class ClasificarRequest(BaseModel):
    texto: str

class ClasificacionMLResponse(BaseModel):
    emocion: Emocion
    severidad: Severidad

class RecuperarRequest(BaseModel):
    emocion: str
    severidad: str

class FragmentoML(BaseModel):
    id: str
    texto: str
    categoria: str

class RecuperacionMLResponse(BaseModel):
    fragmentos: list[FragmentoML]


# ----------------------------------------------------------------------
# Carga de Modelos Neuronales BETO
# ----------------------------------------------------------------------

def _precalentar(modelo, tokenizer, nombre: str) -> None:
    """Ejecuta una inferencia descartable al arrancar.

    La primera pasada de PyTorch paga la inicialización de kernels y tarda
    10-14 s en CPU, muy por encima del timeout del backend
    (BETO_ML_TIMEOUT_SEGUNDOS, 10 s por defecto). Como el gate de crisis falla
    cerrado, sin este precalentamiento el primer mensaje de paciente tras cada
    arranque se rechazaría con 503. Tras la primera pasada la latencia baja a
    ~100 ms.
    """
    entradas = tokenizer(
        "texto de precalentamiento",
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    ).to(device)
    inicio = time.perf_counter()
    with torch.no_grad():
        modelo(**entradas)
    logger.info(
        "Modelo %s precalentado en %.1f s", nombre, time.perf_counter() - inicio
    )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Dispositivo de inferencia para BETO: {device}")

# 1. Cargar clasificador de emoción y severidad
modelo_clasif = None
tokenizer_clasif = None
id2label_clasif = {}

if os.path.exists(RUTA_MODELO_CLASIF):
    try:
        tokenizer_clasif = AutoTokenizer.from_pretrained(RUTA_MODELO_CLASIF)
        modelo_clasif = AutoModelForSequenceClassification.from_pretrained(RUTA_MODELO_CLASIF).to(device)
        modelo_clasif.eval()
        
        with open(os.path.join(RUTA_MODELO_CLASIF, "label_map.json"), "r", encoding="utf-8") as f:
            datos_map = json.load(f)
            id2label_clasif = {int(k): v for k, v in datos_map["id2label"].items()}
        logger.info(f"Modelo BETO Clasificación cargado exitosamente ({len(id2label_clasif)} clases).")
        _precalentar(modelo_clasif, tokenizer_clasif, "Clasificación")
    except Exception as e:
        logger.error(f"Error cargando modelo de clasificación: {e}")

# 2. Cargar detector de crisis
modelo_crisis = None
tokenizer_crisis = None

if os.path.exists(RUTA_MODELO_CRISIS):
    try:
        tokenizer_crisis = AutoTokenizer.from_pretrained(RUTA_MODELO_CRISIS)
        modelo_crisis = AutoModelForSequenceClassification.from_pretrained(RUTA_MODELO_CRISIS).to(device)
        modelo_crisis.eval()
        logger.info("Modelo BETO Detector de Crisis cargado exitosamente.")
        _precalentar(modelo_crisis, tokenizer_crisis, "Crisis")
    except Exception as e:
        logger.error(f"Error cargando modelo de crisis: {e}")


# ----------------------------------------------------------------------
# Base de conocimientos psicoeducativa (fragmentos)
# ----------------------------------------------------------------------

BASE_FRAGMENTOS: dict[str, dict[str, list[str]]] = {
    "ansiedad": {
        "leve": [
            "La respiración diafragmática pausada (inhalar en 4 segundos, exhalar en 4) ayuda a regular la activación del cuerpo.",
            "Tomar breves pausas de 2 minutos para estirarse y soltar la tensión en hombros y mandíbula suele ser reconfortante."
        ],
        "moderado": [
            "La técnica de anclaje 5-4-3-2-1 (nombrar cosas que ves, tocas, oyes) ayuda a reconectar con el momento presente cuando los pensamientos se aceleran.",
            "Escribir las preocupaciones en una lista permite externalizarlas y evitar que sigan dando vueltas constantes en la mente."
        ],
        "severo": [
            "Cuando la ansiedad es muy intensa, colocar las manos bajo agua fresca o lavarse el rostro ayuda a enviar una señal de calma al sistema nervioso.",
            "Recuerda que las sensaciones físicas intensas son temporales y disminuyen progresivamente al concentrarse en respiraciones lentas."
        ]
    },
    "depresion": {
        "leve": [
            "Realizar una actividad pequeña y alcanzable al día (como caminar 10 minutos) ayuda a activar gradualmente la motivación.",
            "Mantener horarios regulares para levantarse y acostarse favorece la estabilidad del estado de ánimo."
        ],
        "moderado": [
            "Dividir las tareas grandes en micro-pasos de 5 minutos reduce la sensación de sobrecarga cuando el ánimo está bajo.",
            "Estar en contacto con la luz del día y mantener breves interacciones con personas de confianza apoya el bienestar emocional."
        ],
        "severo": [
            "En momentos de profunda tristeza, es valioso ser muy compasivo con uno mismo y priorizar el descanso y las necesidades básicas sin autoexigencia.",
            "Recuerda que no tienes que resolver todo hoy; dar un paso a la vez y aceptar apoyo es lo primordial."
        ]
    },
    "control": {
        "leve": [
            "Continuar registrando cómo te sientes a lo largo de la semana te permitirá identificar patrones que favorecen tu bienestar.",
            "Dedicar momentos del día a actividades recreativas o descanso consciente fortalece tu equilibrio diario."
        ]
    }
}


# ----------------------------------------------------------------------
# Endpoints de Inferencia
# ----------------------------------------------------------------------

@app.get("/health", tags=["salud"])
def health():
    return {
        "estado": "ok",
        "modelo_clasificacion": "cargado" if modelo_clasif is not None else "no_cargado",
        "modelo_crisis": "cargado" if modelo_crisis is not None else "no_cargado",
        "dispositivo": str(device)
    }


@app.post("/ml/detectar-crisis", response_model=CrisisMLResponse, tags=["ml"])
def detectar_crisis(datos: CrisisRequest) -> CrisisMLResponse:
    """Evaluación neuronal de riesgo/crisis clínica mediante BETO."""
    if modelo_crisis is not None and tokenizer_crisis is not None:
        inputs = tokenizer_crisis(
            datos.texto,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = modelo_crisis(**inputs)
            probabilidades = torch.softmax(outputs.logits, dim=-1)[0]
            # Probabilidad de clase 1 (crisis)
            prob_crisis = probabilidades[1].item()
            
        # Umbral preventivo de seguridad clínica: >= 0.40 se considera riesgo
        if prob_crisis >= 0.40:
            logger.warning(f"BETO Crisis detectada con confianza {prob_crisis:.4f}")
            return CrisisMLResponse(
                riesgo=True,
                motivo=f"beto_riesgo_detectado (confianza: {prob_crisis:.2%})"
            )
        return CrisisMLResponse(riesgo=False, motivo=None)
        
    # Heurística de respaldo en caso de no cargar pesos
    texto_limpio = datos.texto.lower()
    if any(k in texto_limpio for k in ["matar", "suicid", "morir", "acabar con mi vida", "autolesion", "quitarme la vida"]):
        return CrisisMLResponse(riesgo=True, motivo="respaldo_patron_detectado")
    return CrisisMLResponse(riesgo=False, motivo=None)


@app.post("/ml/clasificar", response_model=ClasificacionMLResponse, tags=["ml"])
def clasificar(datos: ClasificarRequest) -> ClasificacionMLResponse:
    """Clasificación neuronal de emoción y severidad mediante BETO."""
    if modelo_clasif is not None and tokenizer_clasif is not None:
        inputs = tokenizer_clasif(
            datos.texto,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = modelo_clasif(**inputs)
            pred_id = int(torch.argmax(outputs.logits, dim=-1)[0].item())
            etiqueta = id2label_clasif.get(pred_id, "control")
            
        # Parsear etiqueta (e.g. 'ansiedad_moderado', 'depresion_severo', 'control')
        if "_" in etiqueta:
            emocion_str, severidad_str = etiqueta.split("_", 1)
        else:
            emocion_str, severidad_str = "control", "leve"
            
        emocion: Emocion = emocion_str if emocion_str in ["ansiedad", "depresion", "control"] else "control"
        severidad: Severidad = severidad_str if severidad_str in ["leve", "moderado", "severo"] else "leve"
        
        # NO se registra el texto del paciente. La política de privacidad del
        # proyecto (Ley 29733) limita los logs a metadatos de la inferencia;
        # el backend aplica la misma regla en app/core/logging.py.
        logger.info("BETO Clasificación -> %s (%s)", emocion, severidad)
        return ClasificacionMLResponse(emocion=emocion, severidad=severidad)
        
    return ClasificacionMLResponse(emocion="ansiedad", severidad="moderado")


@app.post("/ml/recuperar", response_model=RecuperacionMLResponse, tags=["ml"])
def recuperar(datos: RecuperarRequest) -> RecuperacionMLResponse:
    """Recuperación de fragmentos psicoeducativos acordes al estado emocional."""
    emocion = datos.emocion if datos.emocion in BASE_FRAGMENTOS else "control"
    severidad = datos.severidad if datos.severidad in BASE_FRAGMENTOS.get(emocion, {}) else "leve"
    
    textos = BASE_FRAGMENTOS.get(emocion, {}).get(severidad, BASE_FRAGMENTOS["control"]["leve"])
    
    fragmentos = [
        FragmentoML(
            id=f"frag_{emocion}_{severidad}_{i+1}",
            texto=texto,
            categoria="psicoeducacion"
        )
        for i, texto in enumerate(textos)
    ]
    
    return RecuperacionMLResponse(fragmentos=fragmentos)

"""Generador de dataset clínico sintético en español para entrenamiento de BETO.
Utiliza solo la biblioteca estándar de Python (módulo csv).

Genera dos conjuntos de datos:
1. dataset_clasificacion.csv: Emoción (ansiedad, depresion, control) y Severidad (leve, moderado, severo).
2. dataset_crisis.csv: Detección binaria de crisis / riesgo vital (0: no crisis, 1: crisis).
"""
import os
import csv
import random
from collections import Counter

# Fijar semilla para reproducibilidad
random.seed(42)

# ==============================================================================
# PLANTILLAS Y FRASES BASE PARA CLASIFICACIÓN
# ==============================================================================

FRASES_ANSIEDAD = {
    "leve": [
        "Siento un poco de nervios por la presentación del viernes.",
        "A veces me preocupo más de lo normal por las cosas del trabajo.",
        "Tengo una ligera inquietud en el pecho cuando pienso en mis exámenes.",
        "Me cuesta un poco relajarme después de un día largo de trabajo.",
        "Siento algo de tensión en los hombros cuando hay mucho que hacer.",
        "Últimamente ando algo intranquilo con los cambios en la oficina.",
        "Me pongo algo nervioso cuando tengo que hablar en público.",
        "A veces sobrepienso las cosas antes de dormir y tardo un rato en conciliar el sueño.",
        "Siento una leve preocupación por temas económicos pero puedo seguir con mi día.",
        "Hoy me sentí algo agitado con tanto tráfico y prisas.",
        "Tengo un poco de estrés acumulado por la universidad.",
        "Me siento algo ansioso cuando no tengo el control de mis horarios.",
        "A ratos me da una pequeña sensación de desasosiego sin motivo aparente.",
        "Siento cierta impaciencia y me muevo de un lado a otro.",
        "Me preocupa un poco que las cosas no salgan perfectas."
    ],
    "moderado": [
        "Llevo varios días con taquicardia constante y mucha dificultad para concentrarme.",
        "La angustia no me deja estar tranquilo en ningún momento del día.",
        "Siento una opresión en el pecho casi a diario y me falta el aire a ratos.",
        "Tengo miedo constante de que algo muy malo vaya a pasar con mi familia.",
        "El insomnio por preocupación ya lleva dos semanas y me siento muy agotado.",
        "Me tiemblan las manos cuando entro en reuniones y me cuesta respirar hondo.",
        "Siento que mi cabeza no para de pensar en catástrofes y me cuesta calmarme.",
        "Tengo ataques de ansiedad recurrentes donde siento que voy a perder el control.",
        "Me da miedo salir a lugares concurridos porque siento que me va a faltar el aire.",
        "No puedo desconectarme de la tensión laboral, me despierto sobresaltado en la noche.",
        "Siento náuseas y mareos leves debido al estado de alerta constante en el que estoy.",
        "La preocupación es tan constante que me cuesta comer y trabajar con normalidad.",
        "Siento mucha irritabilidad y nerviosismo que ya no puedo disimular.",
        "Todo el tiempo tengo la sensación de que se me acaba el tiempo y me desespero.",
        "Me cuesta mucho respirar con calma cuando tengo que tomar decisiones importantes."
    ],
    "severo": [
        "Siento que voy a volverme loco o que me va a dar un infarto del pánico tan horrible que tengo.",
        "Los ataques de pánico son diarios y no puedo salir de mi habitación del terror que siento.",
        "Es un terror constante, mi corazón late desbocado y siento que me asfixio por completo.",
        "La ansiedad es totalmente insoportable, estoy temblando sin poder parar desde hace horas.",
        "Siento una desesperación desgarradora que me paraliza por completo el cuerpo.",
        "El pánico me sobrepasa por completo, no puedo hablar ni pensar de la angustia.",
        "Siento que me estoy desconectando de la realidad y me aterra perder el juicio.",
        "La taquicardia es tan extrema y dolorosa que siento que colapsaré en cualquier instante.",
        "No aguanto esta sensación de asfixia y pánico continuo que no cede con nada.",
        "Siento un terror incontrolable a cada segundo, no puedo sostenerme en pie de la agitación."
    ]
}

FRASES_DEPRESION = {
    "leve": [
        "Estos días he estado con el ánimo un poco bajo y con menos energía.",
        "Me ha costado un poco disfrutar de las cosas que antes me entretenían.",
        "Hoy me sentí algo melancólico recordando momentos del pasado.",
        "Tengo menos ganas de salir con amigos este fin de semana, prefiero quedarme en casa.",
        "Me siento algo desmotivado con mi rutina actual.",
        "A veces siento una pequeña tristeza sin una razón clara.",
        "Me noto más cansado de lo habitual aunque duerma las horas completas.",
        "He perdido un poco el interés por mis pasatiempos favoritos.",
        "Siento cierta apatía pero igual cumplo con mis responsabilidades diarias.",
        "El día de hoy me pareció algo gris y sin muchas ganas de empezar actividades.",
        "Me da un poco de pereza socializar últimamente.",
        "Tengo el ánimo algo decaído desde que empezó la semana.",
        "Siento que me falta un poco de entusiasmo en mis proyectos.",
        "A ratos me da nostalgia y me cuesta sonreír.",
        "Me siento un poco desconectado de mi entorno pero voy avanzando."
    ],
    "moderado": [
        "Llevo semanas sintiendo un vacío muy grande y llorando casi todas las tardes.",
        "No tengo energía para levantarme de la cama y me cuesta mucho asearme o comer.",
        "Siento una tristeza profunda que no se va con nada y nada me provoca ilusión.",
        "He dejado de hablar con mis amigos y familiares porque me siento una carga para todos.",
        "Siento que nada de lo que hago tiene sentido ni propósito desde hace un mes.",
        "Todo me resulta un esfuerzo titánico, hasta ducharme o preparar comida me agota.",
        "He perdido totalmente el apetito y me paso el día mirando al techo sin ganas de vivir.",
        "Me siento culpable por todo lo que pasa a mi alrededor y siento que no sirvo para nada.",
        "La soledad que siento es inmensa aunque esté rodeado de gente.",
        "No recuerdo la última vez que sentí alegría genuina, todo se siente plano y gris.",
        "Me cuesta mucho concentrarme en el trabajo porque solo pienso en lo triste que me siento.",
        "Paso días enteros encerrado en mi cuarto sin querer ver la luz del día.",
        "Siento un dolor emocional muy pesado en el pecho que me quita las fuerzas.",
        "Siento que he fracasado en todo y me cuesta encontrar motivos para continuar.",
        "La desesperanza se ha vuelto mi estado natural durante el último mes."
    ],
    "severo": [
        "Siento un dolor y una agonía emocional tan profunda que ya no soporto existir.",
        "La oscuridad es total, ya no tengo fuerzas para seguir viviendo este sufrimiento diario.",
        "Siento que soy un estorbo absoluto y el dolor es tan desgarrador que no puedo más.",
        "No puedo parar de llorar del sufrimiento tan hondo que me consume por dentro.",
        "Estoy completamente roto por dentro, no tengo ningún futuro ni esperanza de nada.",
        "El vacío es insoportable, siento que estoy muerto en vida y ya no aguanto el dolor.",
        "No hay ninguna salida a este sufrimiento, cada segundo de existencia me tortura.",
        "Me siento totalmente hundido en un pozo sin fondo del que jamás podré salir.",
        "El dolor psicológico es tan insoportable que ya no tengo fuerzas para respirar.",
        "Siento una desesperación absoluta y una agonía constante que me quita la vida."
    ]
}

FRASES_CONTROL = [
    "Hoy fue un día normal, fui al trabajo y regresé a casa a descansar.",
    "El fin de semana salí a caminar por el parque con mi perro.",
    "Estuve cocinando una receta nueva que vi en internet.",
    "Me siento tranquilo hoy, he podido avanzar con mis pendientes sin problemas.",
    "Estuve viendo una serie que me recomendaron y me pareció entretenida.",
    "Hoy dormí bien y me desperté con buen ánimo para la jornada.",
    "Fui a almorzar con unos compañeros de la oficina y conversamos un buen rato.",
    "Hice un poco de ejercicio por la mañana y me sentí con buena energía.",
    "Estoy leyendo un libro nuevo sobre historia que me está gustando mucho.",
    "Todo ha estado marchando con tranquilidad en el trabajo esta semana.",
    "Estuve limpiando y organizando mi habitación el día de hoy.",
    "Ayer hablé por teléfono con mi mamá para saber cómo estaba la familia.",
    "Me siento en paz con mi rutina diaria, sin mayores novedades.",
    "Fui al supermercado a hacer las compras de la semana.",
    "Disfruté de una tarde de descanso escuchando música relajante."
]

# ==============================================================================
# FRASES PARA DETECCIÓN DE CRISIS (0: NO CRISIS, 1: CRISIS)
# ==============================================================================

FRASES_CRISIS_POSITIVAS = [
    "Ya no quiero vivir, quiero acabar con mi vida esta misma noche.",
    "Tengo pastillas guardadas y me las voy a tomar todas para no despertar.",
    "Quiero matarme, ya no soporto más este dolor.",
    "Me voy a cortar las venas, ya tengo el cuchillo listo.",
    "Estoy en el puente pensando en tirarme para terminar con todo.",
    "He escrito cartas de despedida para mi familia porque hoy me voy a suicidar.",
    "No vale la pena seguir vivo, voy a quitarme la vida hoy.",
    "Ya no puedo aguantar, he decidido que hoy es mi último día en este mundo.",
    "Me quiero colgar, ya preparé la soga en mi habitación.",
    "Adiós a todos, gracias por todo pero ya tomé la decisión de acabar con mi vida.",
    "Tengo un arma en la mano y voy a dispararme ahora mismo.",
    "Voy a estrellar mi auto a toda velocidad para morir de una vez.",
    "Me voy a envenenar para dejar de sufrir para siempre.",
    "Ya no hay vuelta atrás, me estoy despidiendo de todos porque no estaré mañana.",
    "Quiero morir ahora mismo, no me importa nada más."
]

PREFIJOS = ["", "Doctor, ", "Hola, ", "La verdad es que ", "Quería contarte que ", "Siento que ", "Últimamente "]
SUFIJOS = ["", " y no sé qué hacer.", " y me cuesta lidiar con eso.", " desde hace varios días.", " todo el tiempo.", " a cada momento."]

def aumentar_texto(texto: str) -> str:
    pref = random.choice(PREFIJOS)
    suf = random.choice(SUFIJOS)
    resultado = f"{pref}{texto.lower().strip()}{suf}".strip()
    return resultado.capitalize()

def generar_dataset():
    filas_clasificacion = []
    filas_crisis = []
    
    # 1. Generar Ansiedad
    for severidad, frases in FRASES_ANSIEDAD.items():
        for frase in frases:
            filas_clasificacion.append({
                "texto": frase,
                "emocion": "ansiedad",
                "severidad": severidad,
                "etiqueta_unificada": f"ansiedad_{severidad}"
            })
            filas_crisis.append({"texto": frase, "es_crisis": 0})
            
            for _ in range(4):
                aumentada = aumentar_texto(frase)
                filas_clasificacion.append({
                    "texto": aumentada,
                    "emocion": "ansiedad",
                    "severidad": severidad,
                    "etiqueta_unificada": f"ansiedad_{severidad}"
                })
                filas_crisis.append({"texto": aumentada, "es_crisis": 0})
                
    # 2. Generar Depresión
    for severidad, frases in FRASES_DEPRESION.items():
        for frase in frases:
            filas_clasificacion.append({
                "texto": frase,
                "emocion": "depresion",
                "severidad": severidad,
                "etiqueta_unificada": f"depresion_{severidad}"
            })
            filas_crisis.append({"texto": frase, "es_crisis": 0})
            
            for _ in range(4):
                aumentada = aumentar_texto(frase)
                filas_clasificacion.append({
                    "texto": aumentada,
                    "emocion": "depresion",
                    "severidad": severidad,
                    "etiqueta_unificada": f"depresion_{severidad}"
                })
                filas_crisis.append({"texto": aumentada, "es_crisis": 0})
                
    # 3. Generar Control
    for frase in FRASES_CONTROL:
        filas_clasificacion.append({
            "texto": frase,
            "emocion": "control",
            "severidad": "leve",
            "etiqueta_unificada": "control"
        })
        filas_crisis.append({"texto": frase, "es_crisis": 0})
        
        for _ in range(8):
            aumentada = aumentar_texto(frase)
            filas_clasificacion.append({
                "texto": aumentada,
                "emocion": "control",
                "severidad": "leve",
                "etiqueta_unificada": "control"
            })
            filas_crisis.append({"texto": aumentada, "es_crisis": 0})

    # 4. Generar Crisis (Positivas = 1)
    for frase in FRASES_CRISIS_POSITIVAS:
        filas_crisis.append({"texto": frase, "es_crisis": 1})
        for _ in range(15):
            aumentada = aumentar_texto(frase)
            filas_crisis.append({"texto": aumentada, "es_crisis": 1})
            
    # Barajar
    random.shuffle(filas_clasificacion)
    random.shuffle(filas_crisis)
    
    directorio_salida = os.path.dirname(os.path.abspath(__file__))
    ruta_clasificacion = os.path.join(directorio_salida, "dataset_clasificacion.csv")
    ruta_crisis = os.path.join(directorio_salida, "dataset_crisis.csv")
    
    # Escribir dataset_clasificacion.csv
    with open(ruta_clasificacion, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["texto", "emocion", "severidad", "etiqueta_unificada"])
        writer.writeheader()
        writer.writerows(filas_clasificacion)
        
    # Escribir dataset_crisis.csv
    with open(ruta_crisis, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["texto", "es_crisis"])
        writer.writeheader()
        writer.writerows(filas_crisis)
        
    print(f"Dataset de Clasificación generado: {len(filas_clasificacion)} muestras en {ruta_clasificacion}")
    dist_clasif = Counter(r["etiqueta_unificada"] for r in filas_clasificacion)
    for k, v in dist_clasif.items():
        print(f"  - {k}: {v}")
        
    print("-" * 50)
    print(f"Dataset de Crisis generado: {len(filas_crisis)} muestras en {ruta_crisis}")
    dist_crisis = Counter(r["es_crisis"] for r in filas_crisis)
    print(f"  - No Crisis (0): {dist_crisis[0]}")
    print(f"  - Crisis (1): {dist_crisis[1]}")

if __name__ == "__main__":
    generar_dataset()

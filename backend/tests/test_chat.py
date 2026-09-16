from sqlalchemy import select

from app.models import EstadoSesion, Mensaje, RemitenteMensaje, SesionConversacion


def test_mensaje_normal_flujo_completo(client, headers_paciente, ml_falso):
    respuesta = client.post(
        "/chat/mensaje",
        json={"texto": "Últimamente me siento muy nervioso en el trabajo"},
        headers=headers_paciente,
    )
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert datos["crisis_detectada"] is False
    assert datos["respuesta"]
    assert datos["sesion_id"]
    # El gate de crisis SIEMPRE corre antes de clasificar.
    assert ml_falso.llamadas.index("detectar_crisis") < ml_falso.llamadas.index("clasificar")
    assert "recuperar" in ml_falso.llamadas


def test_texto_persistido_desidentificado(client, headers_paciente, db):
    respuesta = client.post(
        "/chat/mensaje",
        json={"texto": "Mi correo es juan.perez@gmail.com y mi DNI es 45678123"},
        headers=headers_paciente,
    )
    assert respuesta.status_code == 200
    sesion_id = respuesta.json()["sesion_id"]
    mensaje_paciente = db.scalar(
        select(Mensaje).where(
            Mensaje.sesion_id == sesion_id,
            Mensaje.remitente == RemitenteMensaje.PACIENTE,
        )
    )
    assert "[EMAIL]" in mensaje_paciente.texto
    assert "[DNI]" in mensaje_paciente.texto
    assert "juan.perez@gmail.com" not in mensaje_paciente.texto
    assert "45678123" not in mensaje_paciente.texto


def test_crisis_lexica_no_consulta_al_modelo(client, headers_paciente, ml_falso, db):
    respuesta = client.post(
        "/chat/mensaje",
        json={"texto": "Siento que ya no quiero vivir"},
        headers=headers_paciente,
    )
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert datos["crisis_detectada"] is True
    assert datos["recursos_ayuda"]
    # El gate léxico resolvió de forma determinista: ni clasificación ni
    # detección ML llegaron a ejecutarse.
    assert "clasificar" not in ml_falso.llamadas
    assert "detectar_crisis" not in ml_falso.llamadas
    sesion = db.get(SesionConversacion, datos["sesion_id"])
    assert sesion.estado == EstadoSesion.ESCALADA_CRISIS


def test_crisis_detectada_por_ml(client, headers_paciente, ml_falso):
    ml_falso.riesgo = True
    ml_falso.motivo = "patron_riesgo_alto"
    respuesta = client.post(
        "/chat/mensaje",
        json={"texto": "Hoy fue un día complicado"},
        headers=headers_paciente,
    )
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert datos["crisis_detectada"] is True
    assert "clasificar" not in ml_falso.llamadas


def test_ml_caido_falla_explicito_sin_persistir(client, headers_paciente, ml_falso, db):
    ml_falso.disponible = False
    respuesta = client.post(
        "/chat/mensaje",
        json={"texto": "Hoy fue un día complicado"},
        headers=headers_paciente,
    )
    assert respuesta.status_code == 503
    assert respuesta.json()["codigo"] == "servicio_ml_no_disponible"
    # Sin evaluación de crisis completa no se persiste ningún mensaje.
    assert db.scalar(select(Mensaje)) is None


def test_continuar_sesion_existente(client, headers_paciente):
    primera = client.post(
        "/chat/mensaje", json={"texto": "Me siento tenso"}, headers=headers_paciente
    )
    sesion_id = primera.json()["sesion_id"]
    segunda = client.post(
        "/chat/mensaje",
        json={"texto": "Sigo dándole vueltas a lo mismo", "sesion_id": sesion_id},
        headers=headers_paciente,
    )
    assert segunda.status_code == 200
    assert segunda.json()["sesion_id"] == sesion_id


def test_sesion_ajena_devuelve_404(client, headers_paciente, headers_otro_paciente):
    primera = client.post(
        "/chat/mensaje", json={"texto": "Me siento tenso"}, headers=headers_paciente
    )
    sesion_id = primera.json()["sesion_id"]
    respuesta = client.post(
        "/chat/mensaje",
        json={"texto": "hola, soy otra persona", "sesion_id": sesion_id},
        headers=headers_otro_paciente,
    )
    assert respuesta.status_code == 404


def test_psicologo_no_puede_usar_chat(client, headers_psicologo):
    respuesta = client.post(
        "/chat/mensaje", json={"texto": "hola"}, headers=headers_psicologo
    )
    assert respuesta.status_code == 403
    assert respuesta.json()["codigo"] == "acceso_denegado"


def test_plantillas_rotan_entre_turnos(client, headers_paciente, db):
    """Regresión: el índice de rotación contaba paciente + agente, avanzaba de
    dos en dos y con dos plantillas por combinación se quedaba siempre en la
    misma. El paciente recibía la respuesta idéntica en cada turno."""
    sesion_id = None
    respuestas = []
    for _ in range(3):
        payload = {"texto": "me siento muy nervioso en el trabajo"}
        if sesion_id is not None:
            payload["sesion_id"] = sesion_id
        respuesta = client.post("/chat/mensaje", json=payload, headers=headers_paciente)
        assert respuesta.status_code == 200
        datos = respuesta.json()
        sesion_id = datos["sesion_id"]
        respuestas.append(datos["respuesta"])

    assert len(set(respuestas)) > 1, "el agente repitió la misma plantilla en cada turno"

    plantillas = [
        m.plantilla_id
        for m in db.scalars(
            select(Mensaje)
            .where(
                Mensaje.sesion_id == sesion_id,
                Mensaje.remitente == RemitenteMensaje.AGENTE,
            )
            .order_by(Mensaje.creado_en)
        ).all()
    ]
    assert plantillas[0] != plantillas[1]

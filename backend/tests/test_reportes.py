def _enviar_mensajes(client, headers_paciente, textos):
    for texto in textos:
        respuesta = client.post(
            "/chat/mensaje", json={"texto": texto}, headers=headers_paciente
        )
        assert respuesta.status_code == 200


def test_psicologo_vinculado_obtiene_reporte(
    client, headers_paciente, headers_psicologo, paciente, vinculo
):
    _enviar_mensajes(
        client,
        headers_paciente,
        ["Me siento muy tenso", "No puedo dejar de preocuparme"],
    )
    respuesta = client.get(f"/reportes/{paciente.id}", headers=headers_psicologo)
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert datos["version"] == 1
    contenido = datos["contenido"]
    assert contenido["total_mensajes_paciente"] == 2
    assert contenido["distribucion_emociones"].get("ansiedad") == 2
    assert contenido["resumen_automatico"]
    assert contenido["advertencias"]


def test_regenerar_crea_version_nueva(
    client, headers_paciente, headers_psicologo, paciente, vinculo
):
    _enviar_mensajes(client, headers_paciente, ["Me siento tenso"])
    primera = client.get(f"/reportes/{paciente.id}", headers=headers_psicologo)
    assert primera.json()["version"] == 1
    # Sin regenerar devuelve la misma versión (no duplica).
    repetida = client.get(f"/reportes/{paciente.id}", headers=headers_psicologo)
    assert repetida.json()["version"] == 1
    regenerada = client.get(
        f"/reportes/{paciente.id}?regenerar=true", headers=headers_psicologo
    )
    assert regenerada.json()["version"] == 2


def test_reporte_incluye_eventos_de_crisis(
    client, headers_paciente, headers_psicologo, paciente, vinculo
):
    _enviar_mensajes(client, headers_paciente, ["Siento que ya no quiero vivir"])
    respuesta = client.get(f"/reportes/{paciente.id}", headers=headers_psicologo)
    contenido = respuesta.json()["contenido"]
    assert len(contenido["eventos_crisis"]) == 1
    assert contenido["eventos_crisis"][0]["motivo"] == "lexico:ideacion_suicida"


def test_psicologo_sin_vinculo_denegado(client, headers_psicologo, paciente):
    # Sin fixture de vínculo: el psicólogo existe pero no está vinculado.
    respuesta = client.get(f"/reportes/{paciente.id}", headers=headers_psicologo)
    assert respuesta.status_code == 403
    assert respuesta.json()["codigo"] == "acceso_denegado"


def test_paciente_no_accede_a_reportes(client, headers_paciente, paciente):
    respuesta = client.get(f"/reportes/{paciente.id}", headers=headers_paciente)
    assert respuesta.status_code == 403


def test_reporte_paciente_inexistente(client, headers_psicologo):
    respuesta = client.get("/reportes/id-que-no-existe", headers=headers_psicologo)
    assert respuesta.status_code == 404

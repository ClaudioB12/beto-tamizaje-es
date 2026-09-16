def _abrir_sesion_con_mensaje(client, headers_paciente) -> str:
    respuesta = client.post(
        "/chat/mensaje", json={"texto": "Me siento tenso"}, headers=headers_paciente
    )
    assert respuesta.status_code == 200
    return respuesta.json()["sesion_id"]


def test_paciente_ve_su_historial(client, headers_paciente):
    sesion_id = _abrir_sesion_con_mensaje(client, headers_paciente)
    respuesta = client.get(f"/sesiones/{sesion_id}/historial", headers=headers_paciente)
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert datos["sesion_id"] == sesion_id
    assert len(datos["mensajes"]) == 2
    remitentes = {m["remitente"] for m in datos["mensajes"]}
    assert remitentes == {"paciente", "agente"}


def test_otro_paciente_no_ve_historial_ajeno(
    client, headers_paciente, headers_otro_paciente
):
    sesion_id = _abrir_sesion_con_mensaje(client, headers_paciente)
    respuesta = client.get(
        f"/sesiones/{sesion_id}/historial", headers=headers_otro_paciente
    )
    # 404 y no 403: no se revela la existencia de sesiones ajenas.
    assert respuesta.status_code == 404


def test_psicologo_vinculado_ve_historial(
    client, headers_paciente, headers_psicologo, vinculo
):
    sesion_id = _abrir_sesion_con_mensaje(client, headers_paciente)
    respuesta = client.get(f"/sesiones/{sesion_id}/historial", headers=headers_psicologo)
    assert respuesta.status_code == 200


def test_psicologo_sin_vinculo_no_ve_historial(
    client, headers_paciente, headers_psicologo
):
    sesion_id = _abrir_sesion_con_mensaje(client, headers_paciente)
    respuesta = client.get(f"/sesiones/{sesion_id}/historial", headers=headers_psicologo)
    assert respuesta.status_code == 403


def test_sesion_inexistente_404(client, headers_paciente):
    respuesta = client.get("/sesiones/no-existe/historial", headers=headers_paciente)
    assert respuesta.status_code == 404

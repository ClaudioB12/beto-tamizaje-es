from tests.conftest import PASSWORD_TEST


def test_login_correcto_devuelve_token(client, paciente):
    respuesta = client.post(
        "/auth/login",
        json={"email": paciente.email, "password": PASSWORD_TEST},
    )
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert datos["access_token"]
    assert datos["token_type"] == "bearer"


def test_login_password_incorrecta(client, paciente):
    respuesta = client.post(
        "/auth/login",
        json={"email": paciente.email, "password": "Incorrecta123!"},
    )
    assert respuesta.status_code == 401
    assert respuesta.json()["codigo"] == "credenciales_invalidas"


def test_login_email_inexistente_mismo_error(client):
    respuesta = client.post(
        "/auth/login",
        json={"email": "nadie@test.pe", "password": "Cualquiera123!"},
    )
    # Mismo código y mensaje que password incorrecta: no revela si el email existe.
    assert respuesta.status_code == 401
    assert respuesta.json()["codigo"] == "credenciales_invalidas"


def test_endpoint_protegido_sin_token(client):
    respuesta = client.post("/chat/mensaje", json={"texto": "hola"})
    assert respuesta.status_code == 401

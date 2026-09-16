# Backend — Agente conversacional BETO

Backend FastAPI del agente de dos capas (Capa 1: chat con paciente mediante
plantillas validadas; Capa 2: reporte clínico para el psicólogo). No contiene
lógica de ML: consume el servicio ML por HTTP (`/ml/detectar-crisis`,
`/ml/clasificar`, `/ml/recuperar`).

## Ejecución local (prototipo, SQLite)

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python -m app.db.seed          # usuarios demo + vínculo
uvicorn app.main:app --reload  # http://localhost:8000/docs
```

Variables de entorno: ver `.env.example` (prefijo `BETO_`).

## Tests

```bash
pytest
```

El cliente ML se sustituye por un doble de prueba (`tests/conftest.py`);
no se necesita el servicio ML levantado.

## Migraciones (PostgreSQL)

En el prototipo (SQLite) las tablas se crean al arrancar. Para PostgreSQL:

```bash
set BETO_DATABASE_URL=postgresql+psycopg://beto:beto@localhost:5432/beto
alembic upgrade head
```

Nueva migración tras cambiar modelos: `alembic revision --autogenerate -m "descripcion"`.

## Invariantes de seguridad

- **Gate de crisis**: léxico determinista local (`app/services/crisis_gate.py`)
  + detector del servicio ML, siempre síncrono y ANTES de generar respuesta.
  Si el servicio ML no responde, la petición falla con 503: nunca se responde
  sin evaluación de crisis completa.
- **Privacidad**: todo texto del paciente pasa por
  `app/services/desidentificacion.py` antes de persistirse; los logs solo
  registran identificadores y metadatos. Cero APIs de LLM externas.
- **Roles**: paciente solo accede a sus propios datos; psicólogo solo a
  pacientes con vínculo activo (`vinculos_paciente_psicologo`).
- **Plantillas**: el agente nunca genera texto libre; instancia el banco
  versionado `app/plantillas/banco_plantillas.json`, cuya aprobación por los
  psicólogos se gestiona vía pull request.

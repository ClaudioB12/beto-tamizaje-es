"""Orquestador: flujo de Capa 1 (chat) y Capa 2 (reporte clínico).

Los routers no contienen lógica de negocio: validan input, delegan aquí y
formatean la respuesta. El flujo de Capa 1 es estrictamente:

    gate de crisis (léxico local + ML, síncrono y bloqueante)
        -> clasificación -> recuperación semántica
        -> instanciación de plantilla -> persistencia -> respuesta
"""
from collections import Counter
from statistics import mean

from fastapi import Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.exceptions import (
    AccesoDenegadoError,
    ConflictoError,
    RecursoNoEncontradoError,
)
from app.core.logging import obtener_logger
from app.db.session import get_db
from app.models import (
    EstadoSesion,
    Mensaje,
    RemitenteMensaje,
    ReporteClinico,
    RolUsuario,
    SesionConversacion,
    Usuario,
    VinculoPacientePsicologo,
)
from app.schemas.chat import MensajeChatRequest, MensajeChatResponse
from app.schemas.reporte import EventoCrisis, ReporteContenido
from app.services.crisis_gate import evaluar_crisis_lexica
from app.services.desidentificacion import desidentificar
from app.services.ml_client import MLClient, get_ml_client
from app.services.plantillas import BancoPlantillas, get_banco_plantillas

logger = obtener_logger("orquestador")

RECURSOS_AYUDA_CRISIS = [
    "Línea 113 del Ministerio de Salud, opción 5 (salud mental): gratuita, 24 horas",
    "SAMU 106: emergencias médicas",
    "Acude al servicio de emergencia del establecimiento de salud más cercano",
]

_VALOR_SEVERIDAD = {"leve": 1, "moderado": 2, "severo": 3}


class Orquestador:
    def __init__(self, db: Session, ml_client: MLClient, banco: BancoPlantillas):
        self.db = db
        self.ml_client = ml_client
        self.banco = banco

    # ------------------------------------------------------------------
    # Capa 1: chat con el paciente
    # ------------------------------------------------------------------

    async def procesar_mensaje(
        self, paciente: Usuario, datos: MensajeChatRequest
    ) -> MensajeChatResponse:
        sesion = self._obtener_o_crear_sesion(paciente, datos.sesion_id)
        texto_seguro = desidentificar(datos.texto)

        # GATE DE CRISIS: síncrono, bloqueante y determinista. Primero el
        # léxico local (sin red ni modelo); si no dispara, el detector del
        # servicio ML. Si el servicio ML no responde, la petición completa
        # falla con 503: NUNCA se genera una respuesta sin haber completado
        # la evaluación de crisis.
        hay_crisis, motivo = evaluar_crisis_lexica(datos.texto)
        if not hay_crisis:
            deteccion_ml = await self.ml_client.detectar_crisis(datos.texto, paciente.id)
            if deteccion_ml.riesgo:
                hay_crisis = True
                motivo = f"ml:{deteccion_ml.motivo or 'riesgo_detectado'}"

        if hay_crisis:
            return self._responder_crisis(sesion, texto_seguro, motivo)

        clasificacion = await self.ml_client.clasificar(datos.texto)
        recuperacion = await self.ml_client.recuperar(
            clasificacion.emocion, clasificacion.severidad
        )
        fragmento = recuperacion.fragmentos[0].texto if recuperacion.fragmentos else None

        # Índice de rotación: SOLO los mensajes del paciente. Contar todos los
        # mensajes de la sesión avanza de dos en dos (paciente + agente), de
        # modo que con un número par de plantillas por combinación el índice
        # cae siempre en la misma y la rotación queda congelada.
        mensajes_previos = (
            self.db.scalar(
                select(func.count())
                .select_from(Mensaje)
                .where(
                    Mensaje.sesion_id == sesion.id,
                    Mensaje.remitente == RemitenteMensaje.PACIENTE,
                )
            )
            or 0
        )
        plantilla = self.banco.seleccionar(
            clasificacion.emocion, clasificacion.severidad, mensajes_previos
        )
        respuesta = plantilla.instanciar(fragmento)

        mensaje_paciente = Mensaje(
            sesion_id=sesion.id,
            remitente=RemitenteMensaje.PACIENTE,
            texto=texto_seguro,
            emocion=clasificacion.emocion,
            severidad=clasificacion.severidad,
        )
        mensaje_agente = Mensaje(
            sesion_id=sesion.id,
            remitente=RemitenteMensaje.AGENTE,
            texto=respuesta,
            plantilla_id=plantilla.id,
        )
        self.db.add_all([mensaje_paciente, mensaje_agente])
        self.db.commit()
        self.db.refresh(mensaje_agente)

        logger.info(
            "Mensaje procesado: sesion=%s emocion=%s severidad=%s plantilla=%s",
            sesion.id,
            clasificacion.emocion,
            clasificacion.severidad,
            plantilla.id,
        )
        return MensajeChatResponse(
            sesion_id=sesion.id,
            mensaje_id=mensaje_agente.id,
            respuesta=respuesta,
            crisis_detectada=False,
        )

    def _responder_crisis(
        self, sesion: SesionConversacion, texto_seguro: str, motivo: str | None
    ) -> MensajeChatResponse:
        plantilla = self.banco.respuesta_crisis
        respuesta = plantilla.instanciar(None)

        mensaje_paciente = Mensaje(
            sesion_id=sesion.id,
            remitente=RemitenteMensaje.PACIENTE,
            texto=texto_seguro,
            crisis_detectada=True,
            crisis_motivo=motivo,
        )
        mensaje_agente = Mensaje(
            sesion_id=sesion.id,
            remitente=RemitenteMensaje.AGENTE,
            texto=respuesta,
            plantilla_id=plantilla.id,
        )
        sesion.estado = EstadoSesion.ESCALADA_CRISIS
        self.db.add_all([mensaje_paciente, mensaje_agente])
        self.db.commit()
        self.db.refresh(mensaje_agente)

        logger.warning("Crisis detectada: sesion=%s motivo=%s", sesion.id, motivo)
        return MensajeChatResponse(
            sesion_id=sesion.id,
            mensaje_id=mensaje_agente.id,
            respuesta=respuesta,
            crisis_detectada=True,
            recursos_ayuda=RECURSOS_AYUDA_CRISIS,
        )

    def _obtener_o_crear_sesion(
        self, paciente: Usuario, sesion_id: str | None
    ) -> SesionConversacion:
        if sesion_id is None:
            sesion = SesionConversacion(paciente_id=paciente.id)
            self.db.add(sesion)
            self.db.commit()
            self.db.refresh(sesion)
            return sesion
        sesion = self.db.get(SesionConversacion, sesion_id)
        # 404 también cuando la sesión es de otro paciente: no se revela
        # la existencia de recursos ajenos.
        if sesion is None or sesion.paciente_id != paciente.id:
            raise RecursoNoEncontradoError("Sesión no encontrada")
        if sesion.estado == EstadoSesion.CERRADA:
            raise ConflictoError("La sesión está cerrada; inicia una nueva")
        return sesion

    # ------------------------------------------------------------------
    # Historial de sesión (paciente dueño o psicólogo vinculado)
    # ------------------------------------------------------------------

    def obtener_historial(self, sesion_id: str, usuario: Usuario) -> SesionConversacion:
        sesion = self.db.get(SesionConversacion, sesion_id)
        if sesion is None:
            raise RecursoNoEncontradoError("Sesión no encontrada")
        if usuario.rol == RolUsuario.PACIENTE:
            if sesion.paciente_id != usuario.id:
                raise RecursoNoEncontradoError("Sesión no encontrada")
        else:
            self._verificar_vinculo(sesion.paciente_id, usuario.id)
        return sesion

    # ------------------------------------------------------------------
    # Capa 2: reporte clínico para el psicólogo
    # ------------------------------------------------------------------

    def obtener_o_generar_reporte(
        self, paciente_id: str, psicologo: Usuario, regenerar: bool
    ) -> ReporteClinico:
        self._verificar_vinculo(paciente_id, psicologo.id)

        if not regenerar:
            ultimo = self.db.scalar(
                select(ReporteClinico)
                .where(ReporteClinico.paciente_id == paciente_id)
                .order_by(ReporteClinico.version.desc())
                .limit(1)
            )
            if ultimo is not None:
                return ultimo

        contenido = self._construir_contenido_reporte(paciente_id)
        version_maxima = (
            self.db.scalar(
                select(func.max(ReporteClinico.version)).where(
                    ReporteClinico.paciente_id == paciente_id
                )
            )
            or 0
        )
        reporte = ReporteClinico(
            paciente_id=paciente_id,
            generado_por_id=psicologo.id,
            version=version_maxima + 1,
            contenido=contenido.model_dump(mode="json"),
        )
        self.db.add(reporte)
        self.db.commit()
        self.db.refresh(reporte)

        logger.info(
            "Reporte generado: paciente=%s version=%s psicologo=%s",
            paciente_id,
            reporte.version,
            psicologo.id,
        )
        return reporte

    def _verificar_vinculo(self, paciente_id: str, psicologo_id: str) -> None:
        paciente = self.db.get(Usuario, paciente_id)
        if paciente is None or paciente.rol != RolUsuario.PACIENTE:
            raise RecursoNoEncontradoError("Paciente no encontrado")
        vinculo = self.db.scalar(
            select(VinculoPacientePsicologo).where(
                VinculoPacientePsicologo.paciente_id == paciente_id,
                VinculoPacientePsicologo.psicologo_id == psicologo_id,
                VinculoPacientePsicologo.activo.is_(True),
            )
        )
        if vinculo is None:
            raise AccesoDenegadoError("No existe un vínculo activo con este paciente")

    def _construir_contenido_reporte(self, paciente_id: str) -> ReporteContenido:
        mensajes = self.db.scalars(
            select(Mensaje)
            .join(SesionConversacion, Mensaje.sesion_id == SesionConversacion.id)
            .where(
                SesionConversacion.paciente_id == paciente_id,
                Mensaje.remitente == RemitenteMensaje.PACIENTE,
            )
            .order_by(Mensaje.creado_en)
        ).all()
        total_sesiones = (
            self.db.scalar(
                select(func.count())
                .select_from(SesionConversacion)
                .where(SesionConversacion.paciente_id == paciente_id)
            )
            or 0
        )

        distribucion_emociones = Counter(m.emocion for m in mensajes if m.emocion)
        distribucion_severidad = Counter(m.severidad for m in mensajes if m.severidad)
        eventos_crisis = [
            EventoCrisis(
                mensaje_id=m.id,
                sesion_id=m.sesion_id,
                fecha=m.creado_en,
                motivo=m.crisis_motivo,
            )
            for m in mensajes
            if m.crisis_detectada
        ]
        tendencia = self._calcular_tendencia(
            [m.severidad for m in mensajes if m.severidad]
        )

        advertencias = [
            "Reporte generado automáticamente por reglas deterministas; "
            "requiere validación clínica antes de cualquier decisión."
        ]
        if len(mensajes) < 5:
            advertencias.append(
                "Volumen de mensajes bajo: interpretar las distribuciones con cautela."
            )
        if eventos_crisis:
            advertencias.append(
                "Se registraron eventos de crisis: priorizar el contacto con el paciente."
            )

        if mensajes:
            # Los mensajes de crisis no pasan por clasificación: la
            # distribución puede estar vacía aunque haya mensajes.
            if distribucion_emociones:
                predominante = distribucion_emociones.most_common(1)[0]
                emocion_predominante = f"{predominante[0]} ({predominante[1]} mensajes)"
            else:
                emocion_predominante = "sin clasificar"
            severidad_predominante = (
                distribucion_severidad.most_common(1)[0][0]
                if distribucion_severidad
                else "sin clasificar"
            )
            resumen = (
                f"El paciente registró {len(mensajes)} mensajes en {total_sesiones} "
                f"sesiones. Emoción predominante: {emocion_predominante}. "
                f"Severidad predominante: {severidad_predominante}. "
                f"Eventos de crisis: {len(eventos_crisis)}. "
                f"Tendencia de severidad: {tendencia}."
            )
        else:
            resumen = "El paciente aún no registra mensajes en el sistema."

        return ReporteContenido(
            rango_inicio=mensajes[0].creado_en if mensajes else None,
            rango_fin=mensajes[-1].creado_en if mensajes else None,
            total_sesiones=total_sesiones,
            total_mensajes_paciente=len(mensajes),
            distribucion_emociones=dict(distribucion_emociones),
            distribucion_severidad=dict(distribucion_severidad),
            eventos_crisis=eventos_crisis,
            tendencia_severidad=tendencia,
            resumen_automatico=resumen,
            advertencias=advertencias,
        )

    @staticmethod
    def _calcular_tendencia(severidades: list[str]) -> str:
        valores = [_VALOR_SEVERIDAD[s] for s in severidades if s in _VALOR_SEVERIDAD]
        if len(valores) < 4:
            return "sin_datos"
        mitad = len(valores) // 2
        delta = mean(valores[mitad:]) - mean(valores[:mitad])
        if delta <= -0.3:
            return "mejora"
        if delta >= 0.3:
            return "empeoramiento"
        return "estable"


def get_orquestador(
    db: Session = Depends(get_db),
    ml_client: MLClient = Depends(get_ml_client),
    banco: BancoPlantillas = Depends(get_banco_plantillas),
) -> Orquestador:
    return Orquestador(db=db, ml_client=ml_client, banco=banco)

import asyncio
import json
from types import SimpleNamespace

import fakeredis
import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app import jobs, worker
from app.graph import construir_grafo


@pytest.fixture
def r():
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


class GrafoFalso:
    """Imita la interfaz async del grafo compilado que usa el worker."""

    def __init__(self, interrupts=(), respuesta="respuesta final", error=None):
        self.interrupts = list(interrupts)
        self.respuesta = respuesta
        self.error = error
        self.entradas = []

    async def ainvoke(self, entrada, config):
        self.entradas.append((entrada, config))
        if self.error:
            raise self.error

    async def aget_state(self, config):
        return SimpleNamespace(
            interrupts=self.interrupts,
            values={"messages": [AIMessage(content=self.respuesta)]},
        )


async def _job_en_cola(r, pregunta="¿Puedo tener mascota?") -> tuple[str, dict]:
    job_id = await jobs.crear_job(r, pregunta)
    _, crudo = await r.brpop([jobs.COLA], timeout=1)
    return job_id, json.loads(crudo)


async def test_exito_queda_done_con_respuesta(r):
    job_id, mensaje = await _job_en_cola(r)
    grafo = GrafoFalso()

    await worker.procesar_mensaje(r, grafo, mensaje)

    job = await jobs.obtener_job(r, job_id)
    assert job["status"] == "DONE"
    assert job["respuesta"] == "respuesta final"
    entrada, config = grafo.entradas[0]
    assert entrada["messages"][0].content == "¿Puedo tener mascota?"
    assert config == {"configurable": {"thread_id": job_id}}


async def test_excepcion_queda_failed_con_error(r):
    job_id, mensaje = await _job_en_cola(r)

    await worker.procesar_mensaje(r, GrafoFalso(error=RuntimeError("boom")), mensaje)

    job = await jobs.obtener_job(r, job_id)
    assert job["status"] == "FAILED"
    assert "RuntimeError" in job["error"] and "boom" in job["error"]


async def test_excepcion_al_reanudar_queda_failed(r):
    job_id, _ = await _job_en_cola(r)
    mensaje = {"job_id": job_id, "tipo": "resume", "resume": {"aprobado": True, "comentario": ""}}
    grafo = GrafoFalso(error=ConnectionError("Redis se cayó"))

    await worker.procesar_mensaje(r, grafo, mensaje)

    job = await jobs.obtener_job(r, job_id)
    assert job["status"] == "FAILED"
    assert "Redis se cayó" in job["error"]
    entrada, _ = grafo.entradas[0]
    assert isinstance(entrada, Command) and entrada.resume == {"aprobado": True, "comentario": ""}


async def test_interrupt_queda_waiting_approval(r):
    job_id, mensaje = await _job_en_cola(r)
    pendiente = SimpleNamespace(value={"accion": "Registrar mudanza", "razon": "r", "pregunta": "p"})

    await worker.procesar_mensaje(r, GrafoFalso(interrupts=[pendiente]), mensaje)

    job = await jobs.obtener_job(r, job_id)
    assert job["status"] == "WAITING_APPROVAL"
    assert job["aprobacion_pendiente"]["accion"] == "Registrar mudanza"
    assert job["respuesta"] is None


async def test_mensaje_de_job_inexistente_no_crea_job(r):
    await worker.procesar_mensaje(r, GrafoFalso(), {"job_id": "fantasma", "tipo": "start"})
    assert await jobs.obtener_job(r, "fantasma") is None


async def test_bucle_worker_consume_la_cola_y_se_cancela_limpio(r):
    job_id = await jobs.crear_job(r, "x")
    tarea = asyncio.create_task(worker.bucle_worker(r, GrafoFalso(), "w-test"))

    for _ in range(40):
        if (await jobs.obtener_job(r, job_id))["status"] == "DONE":
            break
        await asyncio.sleep(0.05)
    tarea.cancel()
    await asyncio.gather(tarea, return_exceptions=True)

    assert (await jobs.obtener_job(r, job_id))["status"] == "DONE"
    assert tarea.cancelled()


async def test_bucle_worker_ignora_mensaje_corrupto(r):
    await r.lpush(jobs.COLA, "esto no es json")
    job_id = await jobs.crear_job(r, "x")
    tarea = asyncio.create_task(worker.bucle_worker(r, GrafoFalso(), "w-test"))

    for _ in range(40):
        if (await jobs.obtener_job(r, job_id))["status"] == "DONE":
            break
        await asyncio.sleep(0.05)
    tarea.cancel()
    await asyncio.gather(tarea, return_exceptions=True)

    assert (await jobs.obtener_job(r, job_id))["status"] == "DONE"


# --- Ciclo completo con el grafo real (nodos falsos) y la cola de Redis ----------

def _supervisor(state):
    if any(c["agente"] == "gestor" for c in state.get("contribuciones", [])):
        return {"next_agent": "FINISH", "instruccion": "", "razon": "fake", "task_completed": True}
    return {"next_agent": "gestor", "instruccion": "Registrar mudanza", "razon": "pedida", "task_completed": False}


def _gestor(state):
    return {"contribuciones": [{"agente": "gestor", "instruccion": "x", "aporte": "registrada"}], "pasos": 1}


def _sintesis(state):
    return {"messages": [AIMessage(content=f"Final: {state['contribuciones'][-1]['aporte']}")]}


async def test_ciclo_hitl_completo_por_la_cola(r):
    grafo = construir_grafo(InMemorySaver(), nodos={"supervisor": _supervisor, "gestor": _gestor, "sintesis": _sintesis})
    job_id, mensaje = await _job_en_cola(r, "Registra mi mudanza")

    await worker.procesar_mensaje(r, grafo, mensaje)
    assert (await jobs.obtener_job(r, job_id))["status"] == "WAITING_APPROVAL"

    assert await jobs.solicitar_reanudacion(r, job_id, True, "")
    _, crudo = await r.brpop([jobs.COLA], timeout=1)
    await worker.procesar_mensaje(r, grafo, json.loads(crudo))

    job = await jobs.obtener_job(r, job_id)
    assert job["status"] == "DONE"
    assert job["respuesta"] == "Final: registrada"

import json

import fakeredis
import pytest

from app import jobs


@pytest.fixture
def r():
    return fakeredis.aioredis.FakeRedis(decode_responses=True)


async def _mensajes_en_cola(r) -> list[dict]:
    return [json.loads(m) for m in await r.lrange(jobs.COLA, 0, -1)]


async def test_crear_job_queda_pending_y_encolado(r):
    job_id = await jobs.crear_job(r, "¿Puedo tener mascota?")

    job = await jobs.obtener_job(r, job_id)
    assert job["job_id"] == job_id
    assert job["status"] == "PENDING"
    assert job["pregunta"] == "¿Puedo tener mascota?"
    assert job["respuesta"] is None and job["error"] is None and job["aprobacion_pendiente"] is None
    assert job["created_at"] and job["updated_at"]
    assert await _mensajes_en_cola(r) == [{"job_id": job_id, "tipo": "start"}]


async def test_cola_es_fifo(r):
    primero = await jobs.crear_job(r, "uno")
    await jobs.crear_job(r, "dos")

    _, crudo = await r.brpop([jobs.COLA], timeout=1)
    assert json.loads(crudo)["job_id"] == primero


async def test_obtener_job_inexistente_devuelve_none(r):
    assert await jobs.obtener_job(r, "no-existe") is None


async def test_actualizar_job_serializa_dict_y_limpia_con_none(r):
    job_id = await jobs.crear_job(r, "x")

    await jobs.actualizar_job(r, job_id, status="WAITING_APPROVAL", aprobacion_pendiente={"accion": "mudanza"})
    assert (await jobs.obtener_job(r, job_id))["aprobacion_pendiente"] == {"accion": "mudanza"}

    await jobs.actualizar_job(r, job_id, aprobacion_pendiente=None)
    assert (await jobs.obtener_job(r, job_id))["aprobacion_pendiente"] is None


async def test_reanudar_solo_desde_waiting_approval(r):
    job_id = await jobs.crear_job(r, "x")
    await r.delete(jobs.COLA)
    await jobs.actualizar_job(r, job_id, status="RUNNING")

    assert await jobs.solicitar_reanudacion(r, job_id, True, "") is False
    assert await _mensajes_en_cola(r) == []
    assert (await jobs.obtener_job(r, job_id))["status"] == "RUNNING"


async def test_doble_aprobacion_encola_un_solo_resume(r):
    job_id = await jobs.crear_job(r, "x")
    await r.delete(jobs.COLA)
    await jobs.actualizar_job(r, job_id, status="WAITING_APPROVAL", aprobacion_pendiente={"accion": "a"})

    assert await jobs.solicitar_reanudacion(r, job_id, True, "ok") is True
    assert await jobs.solicitar_reanudacion(r, job_id, False, "doble clic") is False

    assert await _mensajes_en_cola(r) == [
        {"job_id": job_id, "tipo": "resume", "resume": {"aprobado": True, "comentario": "ok"}}
    ]
    job = await jobs.obtener_job(r, job_id)
    assert job["status"] == "PENDING"
    assert job["aprobacion_pendiente"] is None


async def test_reanudar_job_inexistente_devuelve_false(r):
    assert await jobs.solicitar_reanudacion(r, "no-existe", True, "") is False
    assert await _mensajes_en_cola(r) == []

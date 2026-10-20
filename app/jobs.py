"""Estado de los trabajos y cola de tareas en Redis.

Cada job es un hash `job:{id}`; la cola es la lista `cola:tareas` (LPUSH para encolar,
BRPOP para consumir: FIFO). Todas las funciones son async y reciben un cliente
redis.asyncio creado con decode_responses=True.
"""
import json
import uuid
from datetime import datetime, timezone
from typing import Literal

from redis.asyncio import Redis
from redis.exceptions import WatchError

Estado = Literal["PENDING", "RUNNING", "WAITING_APPROVAL", "DONE", "FAILED"]

COLA = "cola:tareas"
# Campos que se guardan como JSON (el resto son strings).
_CAMPOS_JSON = {"aprobacion_pendiente"}
# Campos opcionales: vacíos en Redis, None hacia afuera.
_CAMPOS_OPCIONALES = ("respuesta", "error", "aprobacion_pendiente")


def _clave(job_id: str) -> str:
    return f"job:{job_id}"


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serializar(campo: str, valor) -> str:
    """Redis guarda strings: None se guarda vacío y los dicts como JSON."""
    if valor is None:
        return ""
    if campo in _CAMPOS_JSON:
        return json.dumps(valor, ensure_ascii=False)
    return str(valor)


async def crear_job(r: Redis, pregunta: str) -> str:
    """Registra el job en PENDING y lo encola. Devuelve el job_id."""
    job_id = uuid.uuid4().hex
    ahora = _ahora()
    async with r.pipeline(transaction=True) as pipe:
        pipe.hset(_clave(job_id), mapping={"status": "PENDING", "pregunta": pregunta,
                                           "created_at": ahora, "updated_at": ahora})
        pipe.lpush(COLA, json.dumps({"job_id": job_id, "tipo": "start"}))
        await pipe.execute()
    return job_id


async def obtener_job(r: Redis, job_id: str) -> dict | None:
    """Devuelve el job con sus campos opcionales en None si están vacíos, o None si no existe."""
    datos = await r.hgetall(_clave(job_id))
    if not datos:
        return None
    job = {"job_id": job_id, **datos}
    for campo in _CAMPOS_OPCIONALES:
        valor = job.get(campo) or None
        job[campo] = json.loads(valor) if valor and campo in _CAMPOS_JSON else valor
    return job


async def actualizar_job(r: Redis, job_id: str, **campos) -> None:
    """Actualiza campos del job (status, respuesta, error, aprobacion_pendiente) y su updated_at."""
    mapping = {campo: _serializar(campo, valor) for campo, valor in campos.items()}
    mapping["updated_at"] = _ahora()
    await r.hset(_clave(job_id), mapping=mapping)


async def solicitar_reanudacion(r: Redis, job_id: str, aprobado: bool, comentario: str) -> bool:
    """Pasa el job de WAITING_APPROVAL a PENDING y encola su reanudación, de forma atómica.

    Devuelve False si el job no existe o no está esperando aprobación (por ejemplo, un doble clic).
    WATCH hace que la transacción falle si otro cliente cambió el job entre la lectura y la escritura.
    """
    clave = _clave(job_id)
    mensaje = {"job_id": job_id, "tipo": "resume", "resume": {"aprobado": aprobado, "comentario": comentario}}
    async with r.pipeline(transaction=True) as pipe:
        while True:
            try:
                await pipe.watch(clave)
                if await pipe.hget(clave, "status") != "WAITING_APPROVAL":
                    await pipe.unwatch()
                    return False
                pipe.multi()
                pipe.hset(clave, mapping={"status": "PENDING", "aprobacion_pendiente": "", "updated_at": _ahora()})
                pipe.lpush(COLA, json.dumps(mensaje, ensure_ascii=False))
                await pipe.execute()
                return True
            except WatchError:
                continue  # otro cliente tocó el job: se vuelve a leer su estado

"""Configuración común de los tests.

Los agentes del M6 crean sus LLM al importarse y config.validar_credenciales() exige
claves no vacías. Se definen claves falsas antes de importar `app`: construir los
clientes no llama a la red, así que los tests no necesitan credenciales reales.
"""
import os
import uuid

import pytest
import redis

os.environ.setdefault("OPENAI_API_KEY", "sk-test-falsa")
os.environ.setdefault("PINECONE_API_KEY", "pc-test-falsa")
os.environ.setdefault("PINECONE_INDEX_NAME", "edificio-riesco")

REDIS_URL_TEST = os.environ.get("REDIS_URL", "redis://localhost:6379/0")


def _redis_disponible() -> bool:
    try:
        return bool(redis.Redis.from_url(REDIS_URL_TEST, socket_connect_timeout=0.5).ping())
    except redis.exceptions.RedisError:
        return False


def pytest_collection_modifyitems(config, items):
    """Salta los tests marcados con @pytest.mark.redis si no hay un Redis levantado."""
    if _redis_disponible():
        return
    saltar = pytest.mark.skip(reason=f"Redis no disponible en {REDIS_URL_TEST} (docker compose up -d redis)")
    for item in items:
        if "redis" in item.keywords:
            item.add_marker(saltar)


@pytest.fixture
def redis_url() -> str:
    return REDIS_URL_TEST


@pytest.fixture
def prefijo_unico() -> str:
    """Prefijo de thread_id por test, para no chocar con datos de corridas anteriores."""
    return f"test-{uuid.uuid4().hex[:8]}"

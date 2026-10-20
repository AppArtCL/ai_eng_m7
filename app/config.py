import os
import re
import dotenv
import logging
from pathlib import Path
from pydantic import SecretStr
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

dotenv.load_dotenv(BASE_DIR / ".env")
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
    
def validar_credenciales():
    if not OPENAI_API_KEY:
        mensaje = "OPENAI_API_KEY no puede ser None o vacío."
        logger.error(mensaje)
        raise ValueError(mensaje)
    if not PINECONE_API_KEY:
        mensaje = "PINECONE_API_KEY no puede ser None o vacío."
        logger.error(mensaje)
        raise ValueError(mensaje)
    if not PINECONE_INDEX_NAME:
        mensaje = "PINECONE_INDEX_NAME no puede ser None o vacío."
        logger.error(mensaje)
        raise ValueError(mensaje)
    if not re.fullmatch(r"[a-z0-9-]{1,45}", PINECONE_INDEX_NAME):
        mensaje = f"PINECONE_INDEX_NAME inválido: '{PINECONE_INDEX_NAME}'. Solo se permiten minúsculas, números y guiones (máx. 45 caracteres)."
        logger.error(mensaje)
        raise ValueError(mensaje)

DIM = 1536
METRIC = "cosine"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
ENCODING_NAME = "cl100k_base"
TOP_K = 5
PINECONE_NAMESPACE = "consultas-convivencia"
EMBEDDING_MODEL = "text-embedding-3-small"
MAX_PASOS = 6

def get_embeddings():
    validar_credenciales()
    assert OPENAI_API_KEY is not None
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=SecretStr(OPENAI_API_KEY),
        dimensions=DIM,
    )

def get_llm():
    validar_credenciales()
    assert OPENAI_API_KEY is not None
    return ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=SecretStr(OPENAI_API_KEY),
        temperature=0,
        max_retries=3,
    )
"""Configuración común de los tests.

Los agentes del M6 crean sus LLM al importarse y config.validar_credenciales() exige
claves no vacías. Se definen claves falsas antes de importar `app`: construir los
clientes no llama a la red, así que los tests no necesitan credenciales reales.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-falsa")
os.environ.setdefault("PINECONE_API_KEY", "pc-test-falsa")
os.environ.setdefault("PINECONE_INDEX_NAME", "edificio-riesco")

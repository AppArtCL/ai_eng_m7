from pinecone import Pinecone, ServerlessSpec
import config
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

config.validar_credenciales()

pc = Pinecone(api_key=config.PINECONE_API_KEY)

if not config.PINECONE_INDEX_NAME:
    logging.error("PINECONE_INDEX_NAME no puede ser None o vacío.")
    raise ValueError("PINECONE_INDEX_NAME no puede ser None o vacío.")

indices_existentes = [i["name"] for i in pc.list_indexes()]

if config.PINECONE_INDEX_NAME not in indices_existentes:
    logging.info(f"Creando índice: {config.PINECONE_INDEX_NAME}.")
    pc.create_index(
        name=config.PINECONE_INDEX_NAME,
        dimension=config.DIM,
        metric=config.METRIC,
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    logging.info(f"Índice {config.PINECONE_INDEX_NAME} creado exitosamente.")
else:
    dim_actual = pc.describe_index(config.PINECONE_INDEX_NAME).dimension
    if dim_actual != config.DIM:
        logging.error(f"Dimensión del índice existente ({dim_actual}) no coincide con la configuración ({config.DIM}).")
        raise ValueError(f"Dimensión del índice existente ({dim_actual}) no coincide con la configuración ({config.DIM}).")
    logging.info(f"Usando índice existente: {config.PINECONE_INDEX_NAME}.")

indice = pc.Index(config.PINECONE_INDEX_NAME)
logging.info(f"Estado del índice {config.PINECONE_INDEX_NAME}: {indice.describe_index_stats()}")

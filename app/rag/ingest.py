import os
import logging
import config
import time
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

logger = logging.getLogger(__name__)

# Determina si una línea es un título.
def es_titulo(linea: str) -> bool:    
    return (
        0 < len(linea) < 110
        and not linea[0].isspace()
        and not linea.startswith("⚬")
        and not linea.rstrip().endswith((".", ":"))
    )

# Lee y genera secciones de un documento.
def parsear_documento(ruta: str) -> list[Document]:
    with open(ruta, encoding="utf-8") as f:
        lineas = f.read().strip().splitlines()
    
    source = os.path.basename(ruta)
    doc_id = source.removesuffix(".txt")
    titulo = lineas[0].strip()

    secciones = []
    seccion_actual = "Introducción"
    buffer = []
    
    for linea in lineas[1:]:
        if es_titulo(linea):
            if buffer:
                secciones.append((seccion_actual, "\n".join(buffer)))
            seccion_actual = linea.strip()
            buffer = []
        else:
            buffer.append(linea)

    if buffer:
        secciones.append((seccion_actual, "\n".join(buffer)))
    
    return [
        Document(
            page_content=f"{seccion}\n{texto}",
            metadata={
                "doc_id": doc_id,
                "source": source,
                "titulo": titulo,
                "seccion": seccion,
            }
        )
        for seccion, texto in secciones
    ]

def cargar_chunks():
    secciones = []
    for ruta in sorted(config.DATA_DIR.glob("*.txt")):
        secciones.extend(parsear_documento(str(ruta)))

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=config.ENCODING_NAME,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(secciones)
    if not chunks:
        logger.error(f"No se encontraron documentos en {config.DATA_DIR}.")
        raise ValueError(f"No se encontraron documentos en {config.DATA_DIR}.")
    
    contadores = {}
    for chunk in chunks:
        doc_id = chunk.metadata.get("doc_id")
        i = contadores.get(doc_id, 0)
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_id"] = f"{doc_id}-{i:03d}"
        contadores[doc_id] = i + 1

    return chunks

# Espero hasta que el conteo sea exacto o se espere 30 segundos.
def esperar_conteo(index, esperado, timeout=30):
    inicio = time.time()
    n = 0
    while time.time() - inicio < timeout:
        stats = index.describe_index_stats()
        n = stats["namespaces"].get(config.PINECONE_NAMESPACE, {}).get("vector_count", 0)
        if n >= esperado:
            return n
        time.sleep(2)
    logger.error(f"Timeout alcanzado. Se esperaban {esperado} vectores, pero solo se contaron {n}.")
    return n

def subir_a_pinecone(chunks):
    config.validar_credenciales()
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=config.get_embeddings(),
        index_name=config.PINECONE_INDEX_NAME,
        namespace=config.PINECONE_NAMESPACE,
        ids=[c.metadata["chunk_id"] for c in chunks],
    )
    
    n = esperar_conteo(vectorstore.index, esperado=len(chunks))
    
    logger.info(f"Vectores en el namespace {config.PINECONE_NAMESPACE}: {n}.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    chunks = cargar_chunks()
    logger.info(f"Total de chunks procesados: {len(chunks)}.")
    logger.info("Metadata de ejemplo: %s", chunks[0].metadata if chunks else {})
    subir_a_pinecone(chunks)

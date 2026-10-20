import re
import config
import unicodedata
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_pinecone import PineconeVectorStore

STOPWORDS = {"el", "la", "los", "las", "un", "una", "de", "del", "al", "a", "en", "y", "o", "que", "se", "su", "sus", "por", "para", "con", "lo", "es", "son", "como", "mas"}

def quitar_tildes(texto: str) -> str:
    descompuesto = unicodedata.normalize('NFKD', texto)
    return "".join(c for c in descompuesto if not unicodedata.combining(c))

def preprocesar(texto: str) -> list[str]:
    texto = quitar_tildes(texto.lower())
    palabras = re.findall(r"\w+", texto)
    palabras_filtradas = [palabra for palabra in palabras if palabra not in STOPWORDS]
    return [p[:-1] if len(p) > 4 and p.endswith("s") else p for p in palabras_filtradas]

# Retriever léxico BM25.
def crear_bm25(chunks: list[Document], k: int = config.TOP_K) -> BM25Retriever:
    return BM25Retriever.from_documents(chunks, k=k, preprocess_func=preprocesar)

# Retriever vectorial.
def crear_vectorial(k: int = config.TOP_K) -> BaseRetriever:
    config.validar_credenciales()
    vectorstore = PineconeVectorStore(
        index_name=config.PINECONE_INDEX_NAME,
        embedding=config.get_embeddings(),
        namespace=config.PINECONE_NAMESPACE,
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})
 
 
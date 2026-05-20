"""
Thin wrapper around ChromaDB for persistent vector storage.
ChromaDB runs locally — no external service needed.
"""
import chromadb
from chromadb.config import Settings

_client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="./data/chroma",
        anonymized_telemetry=False,
    )
)

COLLECTION_NAME = "rag_documents"


def _get_collection():
    return _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict]):
    col = _get_collection()
    col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


def query_chunks(query_embedding: list[float], top_k: int = 3):
    col = _get_collection()
    results = col.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({"text": doc, "metadata": meta, "score": round(1 - dist, 4)})
    return chunks


def get_all_chunks():
    col = _get_collection()
    result = col.get(include=["documents", "metadatas"])
    return [
        {"text": d[:200] + "...", "metadata": m}
        for d, m in zip(result["documents"], result["metadatas"])
    ]

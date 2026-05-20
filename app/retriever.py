"""
Retriever — embeds the query and fetches top-k chunks from ChromaDB.
"""
from app.embedder import embed_query
from app.vectorstore import query_chunks


def retrieve_chunks(question: str, top_k: int = 3) -> list[dict]:
    """
    Embed the user's question and return the most relevant document chunks.

    Returns a list of dicts:
        [{"text": "...", "metadata": {...}, "score": 0.87}, ...]
    """
    query_vec = embed_query(question)
    chunks = query_chunks(query_vec, top_k=top_k)
    return chunks

"""
Embedder — wraps sentence-transformers for local, free embeddings.
Model: all-MiniLM-L6-v2 (fast, 384-dim, great for semantic search).

Swap this module for OpenAI embeddings by changing embed_texts() only.
"""
from functools import lru_cache
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load once, reuse across requests."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return a list of embedding vectors for the given texts."""
    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]

"""
Ingestor: reads a file, splits it into chunks, embeds them, stores in ChromaDB.
"""
import hashlib
import io
import uuid
from typing import Any

from app.embedder import embed_texts
from app.vectorstore import upsert_chunks


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks by character count."""
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 20]   # drop tiny tail chunks


# ── Extraction ────────────────────────────────────────────────────────────────

def extract_text(contents: bytes, filename: str) -> str:
    if filename.endswith(".pdf"):
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(contents)) as pdf:
                return "\n\n".join(
                    page.extract_text() or "" for page in pdf.pages
                )
        except ImportError:
            raise RuntimeError("pdfplumber not installed. Run: pip install pdfplumber")
    else:
        return contents.decode("utf-8", errors="replace")


# ── Public API ────────────────────────────────────────────────────────────────

def ingest_document(contents: bytes, filename: str) -> dict[str, Any]:
    text = extract_text(contents, filename)
    chunks = chunk_text(text)

    # Stable IDs so re-uploading the same file updates in place
    doc_hash = hashlib.md5(contents).hexdigest()[:8]
    ids = [f"{doc_hash}-{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

    embeddings = embed_texts(chunks)
    upsert_chunks(ids, embeddings, chunks, metadatas)

    return {
        "filename": filename,
        "chunks_indexed": len(chunks),
        "doc_id": doc_hash,
    }

"""
Basic tests — run with: pytest tests/
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app
from app.ingestor import chunk_text

client = TestClient(app)


# ── Unit: chunking ────────────────────────────────────────────────────────────

def test_chunk_text_basic():
    text = "word " * 300          # 1500 chars
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) > 20 for c in chunks)


def test_chunk_text_overlap():
    text = "abcdefghij" * 100
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    # Each chunk should share the last 20 chars of the previous
    assert chunks[0][-20:] == chunks[1][:20]


# ── Integration: API ──────────────────────────────────────────────────────────

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_upload_bad_extension():
    r = client.post("/upload", files={"file": ("test.docx", b"data", "application/octet-stream")})
    assert r.status_code == 400


@patch("app.ingestor.embed_texts", return_value=[[0.1] * 384])
@patch("app.vectorstore.upsert_chunks")
def test_upload_txt(mock_upsert, mock_embed):
    content = b"Hello world. " * 50
    r = client.post("/upload", files={"file": ("test.txt", content, "text/plain")})
    assert r.status_code == 200
    data = r.json()
    assert data["filename"] == "test.txt"
    assert data["chunks_indexed"] >= 1


@patch("app.retriever.retrieve_chunks", return_value=[
    {"text": "Paris is the capital of France.", "metadata": {"source": "test.txt", "chunk_index": 0}, "score": 0.95}
])
@patch("app.generator.generate_answer_stream", return_value=iter(["data: Paris\n\n", "data: [DONE]\n\n"]))
def test_query(mock_gen, mock_ret):
    r = client.post("/query", json={"question": "What is the capital of France?"})
    assert r.status_code == 200

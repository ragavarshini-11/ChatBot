from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from app.ingestor import ingest_document
from app.retriever import retrieve_chunks
from app.generator import generate_answer_stream

app = FastAPI(
    title="RAG Chatbot API",
    description="Retrieval-Augmented Generation chatbot — upload docs, ask questions.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Ingest a PDF or TXT file into the vector store."""
    if not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(400, "Only .pdf and .txt files are supported.")
    contents = await file.read()
    result = ingest_document(contents, file.filename)
    return result


@app.post("/query")
def query(req: QueryRequest):
    """Retrieve relevant chunks and stream an answer."""
    chunks = retrieve_chunks(req.question, top_k=req.top_k)
    if not chunks:
        raise HTTPException(404, "No documents ingested yet. Upload a document first.")

    return StreamingResponse(
        generate_answer_stream(req.question, chunks),
        media_type="text/event-stream",
    )


@app.get("/chunks")
def list_chunks():
    """Return all stored chunk metadata (for debugging)."""
    from app.vectorstore import get_all_chunks
    return {"chunks": get_all_chunks()}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

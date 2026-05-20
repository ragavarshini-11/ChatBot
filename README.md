# RAG Chatbot

A production-ready Retrieval-Augmented Generation (RAG) API that lets you upload documents and ask questions about them. Built with FastAPI, ChromaDB, sentence-transformers, and Claude.

## Architecture

```
User Question
     │
     ▼
┌─────────────┐     embed query     ┌──────────────────┐
│   FastAPI   │ ──────────────────► │  sentence-trans  │
│   /query    │                     │  (all-MiniLM-L6) │
└─────────────┘                     └──────────────────┘
     │                                       │
     │          cosine similarity            ▼
     │ ◄──────────────────────── ┌────────────────────┐
     │                           │     ChromaDB       │
     │   top-k chunks            │  (local vector DB) │
     ▼                           └────────────────────┘
┌─────────────┐
│   Claude    │  streams answer token-by-token (SSE)
│  Generator  │
└─────────────┘
```

## Tech Stack

| Component | Technology |
|---|---|
| API framework | FastAPI |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| Vector store | ChromaDB (local, persistent) |
| LLM | Anthropic Claude (streaming) |
| PDF parsing | pdfplumber |
| Tests | pytest |
| Container | Docker |

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
cd rag-chatbot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Add your API key
cp .env.example .env
# Edit .env and paste your Anthropic API key

# 3. Run
uvicorn app.main:app --reload
```

API docs available at `http://localhost:8000/docs`

## API Endpoints

### `POST /upload`
Upload a `.pdf` or `.txt` file. The server chunks it, embeds it, and stores it in ChromaDB.

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@my_document.pdf"
```

Response:
```json
{
  "filename": "my_document.pdf",
  "chunks_indexed": 42,
  "doc_id": "a3f1b2c4"
}
```

### `POST /query`
Ask a question. Returns a streaming SSE response.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main topics covered?", "top_k": 3}'
```

### `GET /chunks`
Inspect all indexed chunks (useful for debugging retrieval quality).

## Docker

```bash
docker build -t rag-chatbot .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=your_key rag-chatbot
```

## Tests

```bash
pytest tests/ -v
```

## Project Structure

```
rag-chatbot/
├── app/
│   ├── main.py          # FastAPI routes
│   ├── ingestor.py      # File reading + chunking
│   ├── embedder.py      # sentence-transformers wrapper
│   ├── vectorstore.py   # ChromaDB interface
│   ├── retriever.py     # Query embedding + similarity search
│   └── generator.py     # Claude streaming response
├── tests/
│   └── test_rag.py
├── data/                # ChromaDB persists here (gitignored)
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Key Design Decisions

- **Local embeddings** — `all-MiniLM-L6-v2` runs on CPU, no API cost. Swap `app/embedder.py` for OpenAI embeddings by changing one function.
- **Overlapping chunks** — 100-char overlap prevents context from being cut at chunk boundaries.
- **Streaming responses** — SSE lets the frontend render tokens as they arrive, improving perceived latency.
- **Stable chunk IDs** — re-uploading the same file updates existing vectors instead of duplicating them.

## License

MIT

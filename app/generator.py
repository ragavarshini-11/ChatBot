"""
Generator — builds the RAG prompt and streams the answer from Claude.
Uses Server-Sent Events (SSE) format so the frontend can render tokens live.
"""
import os
from typing import Generator

import anthropic

_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the document \
context provided. If the answer is not found in the context, say so clearly — do not make \
things up. Always mention which source chunk your answer came from."""


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk["metadata"].get("source", "unknown")
        score = chunk.get("score", 0)
        parts.append(f"[Chunk {i} | source: {source} | relevance: {score:.2f}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def generate_answer_stream(question: str, chunks: list[dict]) -> Generator[str, None, None]:
    """Yield SSE-formatted tokens from Claude."""
    context = _build_context(chunks)
    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    with _client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        for text in stream.text_stream:
            # SSE format: "data: <token>\n\n"
            yield f"data: {text}\n\n"
        yield "data: [DONE]\n\n"

"""RAG pipeline: retrieve context, build prompt, generate answer."""

from __future__ import annotations

import time

from app.rag.llm import make_generator
from app.rag.retriever import get_retriever
from app.schemas import Citation, QueryRequest, QueryResponse, RetrievedChunk


def _build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context_lines = "\n".join(
        f"[{c.doc_id}] {c.text}" for c in chunks
    )
    return (
        "You are a compliance expert. Answer the question using ONLY the provided context.\n"
        "If the answer is not in the context, say \"I don't have information on that.\"\n"
        "Cite the source documents by their doc_id.\n\n"
        f"Context:\n{context_lines}\n\n"
        f"Question: {question}"
    )


def _chunks_to_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    """Deduplicate by doc_id, keeping the highest-scoring chunk per doc."""
    best: dict[str, RetrievedChunk] = {}
    for chunk in chunks:
        if chunk.doc_id not in best or chunk.score > best[chunk.doc_id].score:
            best[chunk.doc_id] = chunk
    return [
        Citation(
            doc_id=c.doc_id,
            title=c.title,
            jurisdiction=c.jurisdiction,
            source_url=c.source_url,
            score=c.score,
            snippet=c.text[:200],
        )
        for c in best.values()
    ]


async def answer_query(request: QueryRequest) -> QueryResponse:
    t0 = time.perf_counter()

    chunks = await get_retriever().retrieve(
        request.question, request.k, request.jurisdiction
    )
    prompt = _build_prompt(request.question, chunks)

    generator = make_generator()
    answer = await generator.generate(prompt)

    latency_ms = (time.perf_counter() - t0) * 1000
    return QueryResponse(
        answer=answer,
        citations=_chunks_to_citations(chunks),
        latency_ms=latency_ms,
        provider=generator.provider,
        model=generator.model,
    )

"""RAG pipeline: retrieve context, build prompt, generate answer."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator

from app.rag.llm import AnthropicCachingGenerator, make_generator
from app.rag.retriever import get_retriever
from app.schemas import Citation, QueryRequest, QueryResponse, RetrievedChunk


def _system_prompt() -> str:
    return (
        "You are a compliance expert specialising in employment law across multiple jurisdictions.\n\n"
        "When answering a question:\n"
        "1. Identify which parts of the provided context directly address the question.\n"
        "2. Reason step by step, noting any jurisdiction-specific differences.\n"
        "3. Synthesise a clear, accurate answer and cite each source by its doc_id in brackets, "
        "e.g. [doc_id].\n\n"
        "If the answer cannot be found in the context, respond: "
        "'I don't have information on that.' "
        "Do not invent facts not supported by the context."
    )


def _context_block(chunks: list[RetrievedChunk]) -> str:
    return "Context:\n" + "\n".join(f"[{c.doc_id}] {c.text}" for c in chunks)


def _question_block(question: str) -> str:
    return f"Question: {question}"


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


async def stream_query(request: QueryRequest) -> AsyncGenerator[dict, None]:
    t0 = time.monotonic()
    chunks = await get_retriever().retrieve(request.question, request.k, request.jurisdiction)
    generator = make_generator()

    accumulated = ""
    if isinstance(generator, AnthropicCachingGenerator):
        token_stream = generator.astream(_system_prompt(), _context_block(chunks), _question_block(request.question))
    else:
        prompt = f"{_system_prompt()}\n\n{_context_block(chunks)}\n\n{_question_block(request.question)}"
        token_stream = generator.astream(prompt)

    async for token in token_stream:
        accumulated += token
        yield {"event": "token", "data": {"token": token}}

    citations = _chunks_to_citations(chunks)
    latency_ms = (time.monotonic() - t0) * 1000
    yield {
        "event": "done",
        "data": QueryResponse(
            answer=accumulated,
            citations=citations,
            latency_ms=latency_ms,
            provider=generator.provider,
            model=generator.model,
        ).model_dump(),
    }


async def answer_query(request: QueryRequest) -> QueryResponse:
    t0 = time.perf_counter()

    chunks = await get_retriever().retrieve(
        request.question, request.k, request.jurisdiction
    )

    generator = make_generator()
    if isinstance(generator, AnthropicCachingGenerator):
        answer = await generator.generate(
            system=_system_prompt(),
            context=_context_block(chunks),
            question=_question_block(request.question),
        )
    else:
        prompt = f"{_system_prompt()}\n\n{_context_block(chunks)}\n\n{_question_block(request.question)}"
        answer = await generator.generate(prompt)

    latency_ms = (time.perf_counter() - t0) * 1000
    return QueryResponse(
        answer=answer,
        citations=_chunks_to_citations(chunks),
        latency_ms=latency_ms,
        provider=generator.provider,
        model=generator.model,
    )

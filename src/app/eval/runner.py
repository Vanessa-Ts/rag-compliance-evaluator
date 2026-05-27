"""Evaluation runner: executes the golden set through the RAG pipeline."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from app.core.config import settings
from app.eval.dataset import load_golden
from app.eval.metrics import hit_at_k, judge_faithfulness, p95, precision_at_k
from app.rag.llm import make_generator
from app.rag.pipeline import answer_query
from app.schemas import (
    EvalItemResult,
    EvalRequest,
    EvalResponse,
    EvalSummary,
    QueryRequest,
)

_LAST_RESULT: EvalResponse | None = None
_CONCURRENCY = 4


async def _eval_one(
    item: object,
    k: int,
    generator: object,
    sem: asyncio.Semaphore,
) -> EvalItemResult:
    from app.eval.dataset import GoldenItem

    assert isinstance(item, GoldenItem)

    async with sem:
        resp = await answer_query(
            QueryRequest(question=item.question, k=k, jurisdiction=item.jurisdiction)
        )

    retrieved_doc_ids = [c.doc_id for c in resp.citations]
    prec = precision_at_k(retrieved_doc_ids, item.expected_doc_ids, k)
    hit = hit_at_k(retrieved_doc_ids, item.expected_doc_ids, k)

    context = "\n".join(f"[{c.doc_id}] {c.snippet}" for c in resp.citations)
    faithful, faith_score = await judge_faithfulness(
        item.question, context, resp.answer, generator
    )

    return EvalItemResult(
        id=item.id,
        question=item.question,
        precision_at_k=prec,
        hit=hit,
        faithful=faithful,
        faithfulness_score=faith_score,
        latency_ms=resp.latency_ms,
        retrieved_doc_ids=retrieved_doc_ids,
    )


async def run_eval(request: EvalRequest) -> EvalResponse:
    global _LAST_RESULT

    k = request.k if request.k is not None else settings.top_k
    golden = load_golden(subset=request.subset)

    generator = make_generator()
    sem = asyncio.Semaphore(_CONCURRENCY)

    results = await asyncio.gather(
        *[_eval_one(item, k, generator, sem) for item in golden]
    )

    latencies = [r.latency_ms for r in results]
    precisions = [r.precision_at_k for r in results]
    faithfulness_scores = [r.faithfulness_score for r in results]

    summary = EvalSummary(
        n=len(results),
        retrieval_precision_at_k=sum(precisions) / len(precisions) if precisions else 0.0,
        hit_rate_at_k=sum(1 for r in results if r.hit) / len(results) if results else 0.0,
        mean_faithfulness=sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0,
        mean_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
        p95_latency_ms=p95(latencies),
    )

    response = EvalResponse(
        summary=summary,
        per_question=list(results),
        config={
            "k": k,
            "provider": generator.provider,
            "model": generator.model,
            "embedding_model": settings.embedding_model,
        },
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    _LAST_RESULT = response
    return response


def get_last() -> EvalResponse | None:
    return _LAST_RESULT

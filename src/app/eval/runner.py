"""Evaluation runner: executes the golden set through the RAG pipeline."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
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
_RUNNING = False


@dataclass
class JobState:
    n_total: int
    results: list[EvalItemResult] = field(default_factory=list)
    event: asyncio.Event = field(default_factory=asyncio.Event)
    done: bool = False
    response: EvalResponse | None = None
    error: str | None = None


_JOBS: dict[str, JobState] = {}


async def _eval_one(item: object, k: int, generator: object) -> EvalItemResult:
    from app.eval.dataset import GoldenItem

    assert isinstance(item, GoldenItem)

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
        jurisdiction=item.jurisdiction,
        precision_at_k=prec,
        hit=hit,
        faithful=faithful,
        faithfulness_score=faith_score,
        latency_ms=resp.latency_ms,
        retrieved_doc_ids=retrieved_doc_ids,
    )


def is_running() -> bool:
    return _RUNNING


def register_job() -> str:
    """Mark eval as running, create a job entry, and return its ID."""
    global _RUNNING
    _RUNNING = True
    job_id = str(uuid.uuid4())
    _JOBS[job_id] = JobState(n_total=0)
    return job_id


def get_job(job_id: str) -> JobState | None:
    return _JOBS.get(job_id)


async def run_eval_bg(job_id: str, request: EvalRequest) -> None:
    global _LAST_RESULT, _RUNNING
    job = _JOBS[job_id]
    try:
        k = request.k if request.k is not None else settings.top_k
        golden = load_golden(subset=request.subset)
        job.n_total = len(golden)
        generator = make_generator()

        for item in golden:
            result = await _eval_one(item, k, generator)
            job.results.append(result)
            job.event.set()

        results = job.results
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
        job.response = response
        _LAST_RESULT = response
    except Exception as exc:
        job.error = str(exc)
    finally:
        job.done = True
        job.event.set()
        _RUNNING = False


def get_last() -> EvalResponse | None:
    return _LAST_RESULT

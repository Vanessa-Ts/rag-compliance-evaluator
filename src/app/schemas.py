"""Pydantic request/response models — the shared contract for the RAG service.

Everything (API routes, pipeline, eval) keys off these shapes, so they are kept
free of heavy imports and behaviour. See plan WP0.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A source attribution attached to a generated answer."""

    doc_id: str
    title: str
    jurisdiction: str
    source_url: str
    score: float
    snippet: str


class RetrievedChunk(BaseModel):
    """A chunk returned by the retriever, with its source metadata and score."""

    doc_id: str
    title: str
    jurisdiction: str
    source_url: str
    text: str
    score: float


# --- /query ---
class QueryRequest(BaseModel):
    question: str
    k: int = 4
    jurisdiction: str | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    latency_ms: float
    provider: str
    model: str


# --- /ingest ---
class IngestRequest(BaseModel):
    force: bool = False


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    duration_ms: float


# --- /evaluate ---
class EvalRequest(BaseModel):
    k: int | None = None
    subset: list[str] | None = None


class EvalItemResult(BaseModel):
    id: str
    question: str
    jurisdiction: str
    precision_at_k: float
    hit: bool
    faithful: bool
    faithfulness_score: float
    latency_ms: float
    retrieved_doc_ids: list[str]


class EvalSummary(BaseModel):
    n: int
    retrieval_precision_at_k: float
    hit_rate_at_k: float
    mean_faithfulness: float
    mean_latency_ms: float
    p95_latency_ms: float


class EvalResponse(BaseModel):
    summary: EvalSummary
    per_question: list[EvalItemResult]
    config: dict[str, Any] = Field(default_factory=dict)
    timestamp: str


class EvalStartResponse(BaseModel):
    job_id: str

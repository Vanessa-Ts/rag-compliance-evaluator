"""API contract tests via TestClient — pipeline and runner are monkeypatched."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.schemas import (
    Citation,
    EvalItemResult,
    EvalResponse,
    EvalSummary,
    IngestResponse,
    QueryResponse,
)


def _make_query_response() -> QueryResponse:
    return QueryResponse(
        answer="Workers are entitled to 48h max.",
        citations=[
            Citation(
                doc_id="eu-working-time-directive",
                title="EU Working Time Directive",
                jurisdiction="EU",
                source_url="https://example.com",
                score=0.9,
                snippet="max 48 hours per week",
            )
        ],
        latency_ms=123.4,
        provider="ollama",
        model="llama3.2:3b",
    )


def _make_eval_response() -> EvalResponse:
    return EvalResponse(
        summary=EvalSummary(
            n=2,
            retrieval_precision_at_k=0.75,
            hit_rate_at_k=1.0,
            mean_faithfulness=0.8,
            mean_latency_ms=150.0,
            p95_latency_ms=200.0,
        ),
        per_question=[
            EvalItemResult(
                id="q1",
                question="How many hours?",
                precision_at_k=0.75,
                hit=True,
                faithful=True,
                faithfulness_score=0.8,
                latency_ms=150.0,
                retrieved_doc_ids=["eu-working-time-directive"],
            )
        ],
        config={"k": 4, "provider": "ollama", "model": "llama3.2:3b", "embedding_model": "all-MiniLM-L6-v2"},
        timestamp="2026-01-01T00:00:00+00:00",
    )


@pytest.fixture()
def client() -> TestClient:
    from app.main import app
    return TestClient(app, raise_server_exceptions=False)


def test_ingest_endpoint(client: TestClient) -> None:
    with patch("app.api.routes_rag.ingest", new=AsyncMock(return_value=IngestResponse(documents=7, chunks=37, duration_ms=123.0))):
        resp = client.post("/ingest", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["documents"] == 7
    assert data["chunks"] == 37


def test_query_endpoint(client: TestClient) -> None:
    with patch("app.api.routes_rag.answer_query", new=AsyncMock(return_value=_make_query_response())):
        resp = client.post("/query", json={"question": "Max weekly hours?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert data["provider"] == "ollama"
    assert len(data["citations"]) == 1


def test_query_llm_unavailable_returns_503(client: TestClient) -> None:
    from app.rag.llm import LLMUnavailable

    with patch("app.api.routes_rag.answer_query", new=AsyncMock(side_effect=LLMUnavailable("no model"))):
        resp = client.post("/query", json={"question": "test"})
    assert resp.status_code == 503


def test_evaluate_endpoint(client: TestClient) -> None:
    with patch("app.api.routes_eval.run_eval", new=AsyncMock(return_value=_make_eval_response())):
        resp = client.post("/evaluate", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert data["summary"]["n"] == 2
    assert "per_question" in data


def test_evaluate_last_404_when_empty(client: TestClient) -> None:
    with patch("app.api.routes_eval.get_last", return_value=None):
        resp = client.get("/evaluate/last")
    assert resp.status_code == 404


def test_evaluate_last_returns_cached(client: TestClient) -> None:
    cached = _make_eval_response()
    with patch("app.api.routes_eval.get_last", return_value=cached):
        resp = client.get("/evaluate/last")
    assert resp.status_code == 200
    assert resp.json()["summary"]["n"] == 2


def test_health_live(client: TestClient) -> None:
    resp = client.get("/health/live")
    assert resp.status_code == 200


def test_health_ready_when_indexed(client: TestClient) -> None:
    with patch("app.rag.store.collection_count", return_value=37):
        resp = client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json()["vectors"] == 37

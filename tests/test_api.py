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
                jurisdiction="EU",
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


def test_evaluate_endpoint_returns_202(client: TestClient) -> None:
    with (
        patch("app.api.routes_eval.is_running", return_value=False),
        patch("app.api.routes_eval.register_job", return_value="test-job-id"),
        patch("app.api.routes_eval.run_eval_bg", new=AsyncMock(return_value=None)),
    ):
        resp = client.post("/evaluate", json={})
    assert resp.status_code == 202
    data = resp.json()
    assert data["job_id"] == "test-job-id"


def test_evaluate_endpoint_409_when_running(client: TestClient) -> None:
    with patch("app.api.routes_eval.is_running", return_value=True):
        resp = client.post("/evaluate", json={})
    assert resp.status_code == 409


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


def test_health_ready_503_when_empty(client: TestClient) -> None:
    with patch("app.rag.store.collection_count", return_value=0):
        resp = client.get("/health/ready")
    assert resp.status_code == 503


# ---------- SSE stream endpoint ----------

def test_evaluate_stream_404_when_job_not_found(client: TestClient) -> None:
    with patch("app.api.routes_eval.get_job", return_value=None):
        resp = client.get("/evaluate/stream/nonexistent-job-id")
    assert resp.status_code == 404


def test_query_stream_returns_token_and_done_events(client: TestClient) -> None:
    query_resp = _make_query_response()

    async def fake_stream_query(request):  # noqa: ARG001
        yield {"event": "token", "data": {"token": "Workers "}}
        yield {"event": "token", "data": {"token": "are entitled."}}
        yield {"event": "done", "data": query_resp.model_dump()}

    with patch("app.api.routes_rag.stream_query", side_effect=fake_stream_query):
        resp = client.post("/query/stream", json={"question": "Max weekly hours?"})

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    lines = resp.text.splitlines()
    token_events = [l for l in lines if l == "event: token"]
    done_events = [l for l in lines if l == "event: done"]
    assert len(token_events) >= 1
    assert len(done_events) == 1


def test_query_stream_llm_unavailable_returns_error_event(client: TestClient) -> None:
    from app.rag.llm import LLMUnavailable

    async def failing_stream(request):  # noqa: ARG001
        raise LLMUnavailable("no model")
        yield  # make it an async generator

    with patch("app.api.routes_rag.stream_query", side_effect=failing_stream):
        resp = client.post("/query/stream", json={"question": "test"})

    assert resp.status_code == 200
    assert "event: error" in resp.text


def test_evaluate_stream_returns_sse_events_for_completed_job(client: TestClient) -> None:
    from app.eval.runner import JobState

    item = EvalItemResult(
        id="q1",
        question="Max weekly hours?",
        jurisdiction="EU",
        precision_at_k=1.0,
        hit=True,
        faithful=True,
        faithfulness_score=0.95,
        latency_ms=110.0,
        retrieved_doc_ids=["eu-working-time-directive"],
    )
    job = JobState(n_total=1)
    job.results.append(item)
    job.done = True
    job.response = _make_eval_response()

    with patch("app.api.routes_eval.get_job", return_value=job):
        resp = client.get("/evaluate/stream/test-job-id")

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    body = resp.text
    assert '"event": "progress"' in body
    assert '"event": "done"' in body
    assert "eu-working-time-directive" in body

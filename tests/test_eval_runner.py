"""Unit tests for the eval runner: job registry, _eval_one, run_eval_bg."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schemas import Citation, EvalItemResult, QueryResponse


def _make_query_response() -> QueryResponse:
    return QueryResponse(
        answer="Max 48 hours per week.",
        citations=[
            Citation(
                doc_id="eu-working-time-directive",
                title="EU WTD",
                jurisdiction="EU",
                source_url="https://example.com",
                score=0.9,
                snippet="48 hour weekly limit",
            )
        ],
        latency_ms=120.0,
        provider="ollama",
        model="llama3.2:3b",
    )


# ---------- job registry ----------

def test_register_job_returns_unique_ids() -> None:
    import importlib

    import app.eval.runner as runner_mod

    importlib.reload(runner_mod)
    id1 = runner_mod.register_job()
    runner_mod._RUNNING = False  # reset so we can register a second
    id2 = runner_mod.register_job()
    runner_mod._RUNNING = False

    assert id1 != id2
    assert runner_mod.get_job(id1) is not None
    assert runner_mod.get_job(id2) is not None


def test_get_job_returns_none_for_unknown_id() -> None:
    from app.eval.runner import get_job

    assert get_job("does-not-exist-xyz") is None


def test_is_running_reflects_state() -> None:
    import importlib

    import app.eval.runner as runner_mod

    importlib.reload(runner_mod)
    assert runner_mod.is_running() is False
    runner_mod.register_job()
    assert runner_mod.is_running() is True
    runner_mod._RUNNING = False
    assert runner_mod.is_running() is False


# ---------- _eval_one ----------

@pytest.mark.asyncio
async def test_eval_one_returns_item_result() -> None:
    from app.eval.dataset import GoldenItem
    from app.eval.runner import _eval_one

    item = GoldenItem(
        id="q1",
        question="Max weekly hours?",
        jurisdiction="EU",
        expected_doc_ids=["eu-working-time-directive"],
        reference_answer="48 hours.",
    )

    class _MockGen:
        provider = "test"
        model = "test"

        async def generate(self, prompt: str) -> str:
            return '{"faithful": true, "score": 0.95, "reasoning": "ok"}'

    with patch("app.eval.runner.answer_query", new=AsyncMock(return_value=_make_query_response())):
        result = await _eval_one(item, k=4, generator=_MockGen())

    assert isinstance(result, EvalItemResult)
    assert result.id == "q1"
    assert result.jurisdiction == "EU"
    assert result.hit is True
    assert result.precision_at_k == pytest.approx(0.25)  # 1 hit / k=4
    assert result.faithful is True
    assert result.latency_ms == pytest.approx(120.0)


@pytest.mark.asyncio
async def test_eval_one_miss_when_wrong_doc_id() -> None:
    from app.eval.dataset import GoldenItem
    from app.eval.runner import _eval_one

    item = GoldenItem(
        id="q2",
        question="Something else?",
        jurisdiction="DE",
        expected_doc_ids=["de-some-other-doc"],
        reference_answer="answer",
    )

    class _MockGen:
        provider = "test"
        model = "test"

        async def generate(self, prompt: str) -> str:
            return '{"faithful": false, "score": 0.1, "reasoning": "miss"}'

    with patch("app.eval.runner.answer_query", new=AsyncMock(return_value=_make_query_response())):
        result = await _eval_one(item, k=4, generator=_MockGen())

    assert result.hit is False
    assert result.precision_at_k == 0.0


# ---------- run_eval_bg ----------

@pytest.mark.asyncio
async def test_run_eval_bg_populates_job_and_sets_last() -> None:
    import importlib

    import app.eval.runner as runner_mod

    importlib.reload(runner_mod)

    golden_item = MagicMock()
    golden_item.id = "q1"
    golden_item.question = "Test?"
    golden_item.jurisdiction = "EU"
    golden_item.expected_doc_ids = ["eu-working-time-directive"]

    fake_result = EvalItemResult(
        id="q1",
        question="Test?",
        jurisdiction="EU",
        precision_at_k=1.0,
        hit=True,
        faithful=True,
        faithfulness_score=0.9,
        latency_ms=100.0,
        retrieved_doc_ids=["eu-working-time-directive"],
    )

    from app.schemas import EvalRequest

    job_id = runner_mod.register_job()

    with (
        patch("app.eval.runner.load_golden", return_value=[golden_item]),
        patch("app.eval.runner.make_generator", return_value=MagicMock(provider="test", model="test")),
        patch("app.eval.runner._eval_one", new=AsyncMock(return_value=fake_result)),
    ):
        await runner_mod.run_eval_bg(job_id, EvalRequest())

    job = runner_mod.get_job(job_id)
    assert job is not None
    assert job.done is True
    assert len(job.results) == 1
    assert job.response is not None
    assert job.response.summary.n == 1
    assert runner_mod.get_last() is job.response
    assert runner_mod.is_running() is False


@pytest.mark.asyncio
async def test_run_eval_bg_sets_done_on_error() -> None:
    import importlib

    import app.eval.runner as runner_mod

    importlib.reload(runner_mod)

    from app.schemas import EvalRequest

    job_id = runner_mod.register_job()

    with (
        patch("app.eval.runner.load_golden", side_effect=ValueError("bad data")),
    ):
        await runner_mod.run_eval_bg(job_id, EvalRequest())

    job = runner_mod.get_job(job_id)
    assert job is not None
    assert job.done is True
    assert job.error == "bad data"
    assert runner_mod.is_running() is False

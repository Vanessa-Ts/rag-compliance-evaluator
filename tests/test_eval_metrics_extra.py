"""Tests for judge_faithfulness — the LLM-backed eval metric."""

from __future__ import annotations

import pytest


class _FaithfulGenerator:
    provider = "test"
    model = "test"

    async def generate(self, prompt: str) -> str:
        return '{"faithful": true, "score": 0.9, "reasoning": "ok"}'


class _UnfaithfulGenerator:
    provider = "test"
    model = "test"

    async def generate(self, prompt: str) -> str:
        return '{"faithful": false, "score": 0.2, "reasoning": "not supported"}'


class _MarkdownWrappedGenerator:
    """Simulates a model that wraps JSON in a markdown code fence."""

    provider = "test"
    model = "test"

    async def generate(self, prompt: str) -> str:
        return '```json\n{"faithful": true, "score": 0.85, "reasoning": "wrapped"}\n```'


class _BadJsonGenerator:
    provider = "test"
    model = "test"

    async def generate(self, prompt: str) -> str:
        return "Sorry, I cannot evaluate this."


class _RaisingGenerator:
    provider = "test"
    model = "test"

    async def generate(self, prompt: str) -> str:
        raise RuntimeError("network error")


@pytest.mark.asyncio
async def test_judge_faithfulness_faithful() -> None:
    from app.eval.metrics import judge_faithfulness

    faithful, score = await judge_faithfulness("Q?", "context", "answer", _FaithfulGenerator())
    assert faithful is True
    assert score == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_judge_faithfulness_unfaithful() -> None:
    from app.eval.metrics import judge_faithfulness

    faithful, score = await judge_faithfulness("Q?", "context", "answer", _UnfaithfulGenerator())
    assert faithful is False
    assert score == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_judge_faithfulness_markdown_fence_stripped() -> None:
    from app.eval.metrics import judge_faithfulness

    faithful, score = await judge_faithfulness("Q?", "ctx", "ans", _MarkdownWrappedGenerator())
    assert faithful is True
    assert score == pytest.approx(0.85)


@pytest.mark.asyncio
async def test_judge_faithfulness_bad_json_returns_false() -> None:
    from app.eval.metrics import judge_faithfulness

    faithful, score = await judge_faithfulness("Q?", "ctx", "ans", _BadJsonGenerator())
    assert faithful is False
    assert score == 0.0


@pytest.mark.asyncio
async def test_judge_faithfulness_generator_raises_returns_false() -> None:
    from app.eval.metrics import judge_faithfulness

    faithful, score = await judge_faithfulness("Q?", "ctx", "ans", _RaisingGenerator())
    assert faithful is False
    assert score == 0.0


@pytest.mark.asyncio
async def test_judge_faithfulness_non_generator_returns_false() -> None:
    from app.eval.metrics import judge_faithfulness

    faithful, score = await judge_faithfulness("Q?", "ctx", "ans", object())
    assert faithful is False
    assert score == 0.0


@pytest.mark.asyncio
async def test_judge_faithfulness_score_clamped_to_one() -> None:
    from app.eval.metrics import judge_faithfulness

    class OverflowGenerator:
        provider = "test"
        model = "test"

        async def generate(self, prompt: str) -> str:
            return '{"faithful": true, "score": 99.0, "reasoning": "overflow"}'

    faithful, score = await judge_faithfulness("Q?", "ctx", "ans", OverflowGenerator())
    assert faithful is True
    assert score == pytest.approx(1.0)

"""Tests for judge_faithfulness — the LLM-backed eval metric."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

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

    faithful, score, reasoning = await judge_faithfulness("Q?", "context", "answer", _FaithfulGenerator())
    assert faithful is True
    assert score == pytest.approx(0.9)
    assert reasoning is None


@pytest.mark.asyncio
async def test_judge_faithfulness_unfaithful() -> None:
    from app.eval.metrics import judge_faithfulness

    faithful, score, reasoning = await judge_faithfulness("Q?", "context", "answer", _UnfaithfulGenerator())
    assert faithful is False
    assert score == pytest.approx(0.2)
    assert reasoning is None


@pytest.mark.asyncio
async def test_judge_faithfulness_markdown_fence_stripped() -> None:
    from app.eval.metrics import judge_faithfulness

    faithful, score, reasoning = await judge_faithfulness("Q?", "ctx", "ans", _MarkdownWrappedGenerator())
    assert faithful is True
    assert score == pytest.approx(0.85)
    assert reasoning is None


@pytest.mark.asyncio
async def test_judge_faithfulness_bad_json_returns_false() -> None:
    from app.eval.metrics import judge_faithfulness

    faithful, score, reasoning = await judge_faithfulness("Q?", "ctx", "ans", _BadJsonGenerator())
    assert faithful is False
    assert score == 0.0
    assert reasoning is None


@pytest.mark.asyncio
async def test_judge_faithfulness_generator_raises_returns_false() -> None:
    from app.eval.metrics import judge_faithfulness

    faithful, score, reasoning = await judge_faithfulness("Q?", "ctx", "ans", _RaisingGenerator())
    assert faithful is False
    assert score == 0.0
    assert reasoning is None


@pytest.mark.asyncio
async def test_judge_faithfulness_non_generator_returns_false() -> None:
    from app.eval.metrics import judge_faithfulness

    faithful, score, reasoning = await judge_faithfulness("Q?", "ctx", "ans", object())
    assert faithful is False
    assert score == 0.0
    assert reasoning is None


@pytest.mark.asyncio
async def test_judge_faithfulness_score_clamped_to_one() -> None:
    from app.eval.metrics import judge_faithfulness

    class OverflowGenerator:
        provider = "test"
        model = "test"

        async def generate(self, prompt: str) -> str:
            return '{"faithful": true, "score": 99.0, "reasoning": "overflow"}'

    faithful, score, reasoning = await judge_faithfulness("Q?", "ctx", "ans", OverflowGenerator())
    assert faithful is True
    assert score == pytest.approx(1.0)
    assert reasoning is None


@pytest.mark.asyncio
async def test_judge_faithfulness_anthropic_tool_use() -> None:
    """AnthropicCachingGenerator uses tool_choice to get structured output without JSON parsing."""
    from app.eval.metrics import judge_faithfulness
    from app.rag.llm import AnthropicCachingGenerator

    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = {"faithful": True, "score": 0.95, "reasoning": "All claims verified."}

    mock_response = MagicMock()
    mock_response.content = [tool_block]

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    gen = AnthropicCachingGenerator(mock_client, "claude-sonnet-4-6")
    faithful, score, reasoning = await judge_faithfulness("Q?", "ctx", "ans", gen)

    assert faithful is True
    assert score == pytest.approx(0.95)
    assert reasoning == "All claims verified."

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["tool_choice"] == {"type": "any"}
    assert call_kwargs["tools"][0]["name"] == "faithfulness_verdict"


@pytest.mark.asyncio
async def test_judge_faithfulness_fallback_langchain_ollama() -> None:
    """LangChainGenerator with provider='ollama' uses JSON-parsing fallback, reasoning is None."""
    from app.eval.metrics import judge_faithfulness
    from app.rag.llm import LangChainGenerator

    fake_chat = MagicMock()
    fake_response = MagicMock()
    fake_response.content = '{"faithful": false, "score": 0.3, "reasoning": "unsupported claim"}'
    fake_chat.ainvoke = AsyncMock(return_value=fake_response)

    gen = LangChainGenerator(fake_chat, provider="ollama", model="llama3.2:3b")
    faithful, score, reasoning = await judge_faithfulness("Q?", "ctx", "ans", gen)

    assert faithful is False
    assert score == pytest.approx(0.3)
    assert reasoning is None


@pytest.mark.asyncio
async def test_judge_context_relevance_anthropic_tool_use() -> None:
    """AnthropicCachingGenerator path uses tool_choice and returns the score from tool input."""
    from app.eval.metrics import judge_context_relevance
    from app.rag.llm import AnthropicCachingGenerator

    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = {"score": 0.82, "reasoning": "Passages directly address the question."}

    mock_response = MagicMock()
    mock_response.content = [tool_block]

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    gen = AnthropicCachingGenerator(mock_client, "claude-sonnet-4-6")
    score = await judge_context_relevance("What is overtime pay?", "context about overtime", gen)

    assert score == pytest.approx(0.82)

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["tool_choice"] == {"type": "any"}
    assert call_kwargs["tools"][0]["name"] == "context_relevance_verdict"


@pytest.mark.asyncio
async def test_judge_context_relevance_fallback_json() -> None:
    """Non-Anthropic generator uses JSON-parsing fallback."""
    from app.eval.metrics import judge_context_relevance

    class _ScoreGenerator:
        provider = "test"
        model = "test"

        async def generate(self, prompt: str) -> str:
            return '{"score": 0.75}'

    score = await judge_context_relevance("Q?", "some context", _ScoreGenerator())
    assert score == pytest.approx(0.75)

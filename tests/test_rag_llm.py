"""Tests for the LLM provider factory and LangChainGenerator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.rag.llm import AnthropicCachingGenerator, LLMUnavailable, LangChainGenerator, make_generator


# ---------- make_generator ----------

def test_make_generator_anthropic_raises_without_api_key() -> None:
    with patch("app.rag.llm.settings") as s:
        s.llm_provider = "anthropic"
        s.anthropic_api_key = None
        with pytest.raises(LLMUnavailable, match="ANTHROPIC_API_KEY"):
            make_generator()


def test_make_generator_openai_raises_without_api_key() -> None:
    with patch("app.rag.llm.settings") as s:
        s.llm_provider = "openai"
        s.openai_api_key = None
        with pytest.raises(LLMUnavailable, match="OPENAI_API_KEY"):
            make_generator()


def test_make_generator_unknown_provider_raises() -> None:
    with patch("app.rag.llm.settings") as s:
        s.llm_provider = "nonexistent-provider"
        with pytest.raises(LLMUnavailable, match="Unsupported LLM provider"):
            make_generator()


def test_make_generator_ollama_returns_generator() -> None:
    fake_model = MagicMock()
    with (
        patch("app.rag.llm.settings") as s,
        patch("app.rag.llm.ChatOllama", return_value=fake_model, create=True),
    ):
        s.llm_provider = "ollama"
        s.generation_model = None
        s.ollama_model = "llama3.2:3b"
        s.ollama_base_url = "http://localhost:11434"

        # Import after patching so the import inside make_generator picks up the mock
        import importlib
        import sys

        langchain_ollama_mock = MagicMock()
        langchain_ollama_mock.ChatOllama = MagicMock(return_value=fake_model)
        sys.modules["langchain_ollama"] = langchain_ollama_mock

        gen = make_generator()

    assert isinstance(gen, LangChainGenerator)
    assert gen.provider == "ollama"
    assert gen.model == "llama3.2:3b"


# ---------- LangChainGenerator.generate ----------

@pytest.mark.asyncio
async def test_langchain_generator_generate_returns_content() -> None:
    fake_chat = MagicMock()
    fake_response = MagicMock()
    fake_response.content = "The answer is 48 hours."
    fake_chat.ainvoke = AsyncMock(return_value=fake_response)

    gen = LangChainGenerator(fake_chat, provider="test", model="test-model")
    result = await gen.generate("What is the max?")

    assert result == "The answer is 48 hours."
    fake_chat.ainvoke.assert_awaited_once_with("What is the max?")


@pytest.mark.asyncio
async def test_langchain_generator_generate_wraps_exception_as_llm_unavailable() -> None:
    fake_chat = MagicMock()
    fake_chat.ainvoke = AsyncMock(side_effect=ConnectionError("refused"))

    with patch("app.rag.llm.settings") as s:
        s.ollama_base_url = "http://localhost:11434"
        gen = LangChainGenerator(fake_chat, provider="ollama", model="llama3.2:3b")
        with pytest.raises(LLMUnavailable, match="ollama"):
            await gen.generate("test prompt")


# ---------- make_generator (anthropic caching path) ----------

def test_make_generator_anthropic_caching_returns_caching_generator() -> None:
    fake_client = MagicMock()
    with (
        patch("app.rag.llm.settings") as s,
        patch("anthropic.AsyncAnthropic", return_value=fake_client),
    ):
        s.llm_provider = "anthropic"
        s.anthropic_api_key = "test-key"
        s.enable_prompt_caching = True
        s.generation_model = None

        gen = make_generator()

    assert isinstance(gen, AnthropicCachingGenerator)
    assert gen.model == "claude-sonnet-4-6"
    assert gen._client is fake_client


# ---------- AnthropicCachingGenerator.generate ----------

@pytest.mark.asyncio
async def test_anthropic_caching_generator_generate_returns_text() -> None:
    mock_client = MagicMock()
    mock_content = MagicMock()
    mock_content.text = "Minimum wage in California is $16/hr."
    mock_response = MagicMock()
    mock_response.content = [mock_content]
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    gen = AnthropicCachingGenerator(mock_client, "claude-sonnet-4-6")
    result = await gen.generate(
        system="You are an expert.",
        context="California: $16/hr.",
        question="What is the minimum wage in California?",
    )

    assert result == "Minimum wage in California is $16/hr."

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    user_content = call_kwargs["messages"][0]["content"]
    assert user_content[0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in user_content[1]


# ---------- AnthropicCachingGenerator.astream ----------

@pytest.mark.asyncio
async def test_anthropic_caching_generator_astream_yields_chunks() -> None:
    mock_client = MagicMock()

    async def fake_text_stream() -> object:
        yield "Minimum wage "
        yield "is $16/hr."

    mock_stream_ctx = MagicMock()
    mock_stream_ctx.text_stream = fake_text_stream()

    mock_stream_cm = MagicMock()
    mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_stream_ctx)
    mock_stream_cm.__aexit__ = AsyncMock(return_value=None)
    mock_client.messages.stream.return_value = mock_stream_cm

    gen = AnthropicCachingGenerator(mock_client, "claude-sonnet-4-6")
    chunks: list[str] = []
    async for chunk in gen.astream(
        system="You are an expert.",
        context="California: $16/hr.",
        question="What is the minimum wage in California?",
    ):
        chunks.append(chunk)

    assert chunks == ["Minimum wage ", "is $16/hr."]

    call_kwargs = mock_client.messages.stream.call_args.kwargs
    assert call_kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    user_content = call_kwargs["messages"][0]["content"]
    assert user_content[0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in user_content[1]

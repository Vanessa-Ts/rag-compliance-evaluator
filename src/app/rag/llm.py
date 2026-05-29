"""LLM provider factory for the RAG generation step.

Builds a concrete :class:`Generator` (see ``app.rag.interfaces.Generator``)
wrapping a LangChain chat model, selected at runtime by
``settings.llm_provider``. Construction is lazy: no network calls or API-key
checks happen for Ollama until ``generate()`` is invoked.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from functools import lru_cache
from typing import TYPE_CHECKING

from app.core.config import settings

if TYPE_CHECKING:
    import anthropic as anthropic_sdk
    from langchain_core.language_models.chat_models import BaseChatModel


class LLMUnavailable(RuntimeError):
    """Raised when the configured LLM backend cannot be used.

    Callers should map this to an HTTP 503 (service unavailable).
    """


class LangChainGenerator:
    """Concrete :class:`Generator` wrapping a LangChain chat model."""

    def __init__(self, chat_model: BaseChatModel, provider: str, model: str) -> None:
        self._chat_model = chat_model
        self.provider = provider
        self.model = model

    async def generate(self, prompt: str) -> str:
        """Run the chat model on ``prompt`` and return its text content."""
        try:
            response = await self._chat_model.ainvoke(prompt)
        except Exception as exc:  # noqa: BLE001 - normalize backend failures
            raise LLMUnavailable(
                f"LLM provider {self.provider!r} (model {self.model!r}) is "
                "unreachable. If using Ollama, verify OLLAMA_BASE_URL "
                f"({settings.ollama_base_url}) is correct, the server is "
                f"running, and the model {self.model!r} has been pulled "
                f"(`ollama pull {self.model}`). Underlying error: {exc}"
            ) from exc
        return str(response.content)

    async def astream(self, prompt: str) -> AsyncGenerator[str, None]:
        """Stream tokens from the chat model for ``prompt``."""
        from langchain_core.messages import HumanMessage

        async for chunk in self._chat_model.astream([HumanMessage(content=prompt)]):
            if chunk.content:
                yield chunk.content


class AnthropicCachingGenerator:
    """Anthropic generator with cache_control on system and context blocks."""

    provider = "anthropic"

    def __init__(self, client: anthropic_sdk.AsyncAnthropic, model: str) -> None:
        self._client = client
        self.model = model

    async def generate(self, system: str, context: str, question: str) -> str:
        """Non-streaming call with cache_control on system + context blocks."""
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": context, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": question},
                ],
            }],
        )
        block = response.content[0]
        if not hasattr(block, "text"):
            raise LLMUnavailable(f"Unexpected response block type: {type(block).__name__}")
        return block.text

    async def tool_judge(
        self,
        prompt: str,
        tool: dict,
        max_tokens: int = 256,
    ) -> dict:
        """Tool-use call for structured judge output; returns the tool_use block's input dict."""
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            tools=[tool],
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": prompt}],
        )
        return next(b for b in response.content if b.type == "tool_use").input

    async def astream(
        self, system: str, context: str, question: str
    ) -> AsyncGenerator[str, None]:
        """Streaming call with cache_control on system + context blocks."""
        async with self._client.messages.stream(
            model=self.model,
            max_tokens=1024,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": context, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": question},
                ],
            }],
        ) as stream:
            async for text in stream.text_stream:
                yield text


def make_generator() -> LangChainGenerator | AnthropicCachingGenerator:
    """Build a generator for the configured ``settings.llm_provider``."""
    provider = settings.llm_provider

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        model = settings.generation_model or settings.ollama_model
        chat_model = ChatOllama(
            base_url=settings.ollama_base_url,
            model=model,
            temperature=0,
        )
        return LangChainGenerator(chat_model, provider="ollama", model=model)

    if provider == "anthropic":
        if settings.anthropic_api_key is None:
            raise LLMUnavailable(
                "ANTHROPIC_API_KEY is not set; cannot use the 'anthropic' "
                "LLM provider."
            )
        model = settings.generation_model or "claude-sonnet-4-6"
        if settings.enable_prompt_caching:
            import anthropic as anthropic_sdk

            client = anthropic_sdk.AsyncAnthropic(api_key=settings.anthropic_api_key)
            return AnthropicCachingGenerator(client, model)

        from langchain_anthropic import ChatAnthropic

        chat_model = ChatAnthropic(
            api_key=settings.anthropic_api_key,
            model=model,
            temperature=0,
        )
        return LangChainGenerator(chat_model, provider="anthropic", model=model)

    if provider == "openai":
        if settings.openai_api_key is None:
            raise LLMUnavailable(
                "OPENAI_API_KEY is not set; cannot use the 'openai' LLM "
                "provider."
            )
        from langchain_openai import ChatOpenAI

        model = settings.generation_model or "gpt-4o-mini"
        chat_model = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=model,
            temperature=0,
        )
        return LangChainGenerator(chat_model, provider="openai", model=model)

    raise LLMUnavailable(f"Unsupported LLM provider: {provider!r}")


@lru_cache(maxsize=1)
def get_generator() -> LangChainGenerator | AnthropicCachingGenerator:
    return make_generator()

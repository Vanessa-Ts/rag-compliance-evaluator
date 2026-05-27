"""LLM provider factory for the RAG generation step.

Builds a concrete :class:`Generator` (see ``app.rag.interfaces.Generator``)
wrapping a LangChain chat model, selected at runtime by
``settings.llm_provider``. Construction is lazy: no network calls or API-key
checks happen for Ollama until ``generate()`` is invoked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.config import settings

if TYPE_CHECKING:
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


def make_generator() -> LangChainGenerator:
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
        from langchain_anthropic import ChatAnthropic

        model = settings.generation_model or "claude-haiku-4-5-20251001"
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

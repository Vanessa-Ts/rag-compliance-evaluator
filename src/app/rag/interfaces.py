"""Structural interfaces (typing.Protocol) that decouple RAG components.

Implementations live in sibling modules (retriever.py, llm.py); consumers
(pipeline, eval) depend only on these Protocols so the pieces can be built and
tested in parallel with mocks. See plan WP0.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.schemas import RetrievedChunk


@runtime_checkable
class Retriever(Protocol):
    """Returns the top-k chunks most relevant to a query."""

    async def retrieve(
        self, query: str, k: int, jurisdiction: str | None = None
    ) -> list[RetrievedChunk]: ...


@runtime_checkable
class Generator(Protocol):
    """A text generator backed by some LLM provider."""

    provider: str
    model: str

    async def generate(self, prompt: str) -> str: ...

"""Tests for RAG components — all run offline with mocks (no embeddings/Ollama)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schemas import QueryRequest, RetrievedChunk


# ---------- prompt helpers ----------

def test_system_prompt_contains_cot_instructions() -> None:
    from app.rag.pipeline import _system_prompt

    sp = _system_prompt()
    assert "step by step" in sp
    assert "Identify" in sp


def test_context_block_format() -> None:
    from app.rag.pipeline import _context_block

    chunks = [_make_chunk(doc_id="doc-a", text="Text A"), _make_chunk(doc_id="doc-b", text="Text B")]
    result = _context_block(chunks)
    assert result.startswith("Context:\n")
    assert "[doc-a] Text A" in result
    assert "[doc-b] Text B" in result


def test_question_block_format() -> None:
    from app.rag.pipeline import _question_block

    assert _question_block("How much leave?") == "Question: How much leave?"


# ---------- pipeline ----------

def _make_chunk(**kwargs: object) -> RetrievedChunk:
    defaults = dict(
        doc_id="eu-working-time-directive",
        title="EU WTD",
        jurisdiction="EU",
        source_url="https://example.com",
        text="Workers are entitled to 4 weeks annual leave.",
        score=0.9,
    )
    defaults.update(kwargs)  # type: ignore[arg-type]
    return RetrievedChunk(**defaults)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_pipeline_returns_answer_and_citations() -> None:
    chunk = _make_chunk()
    with (
        patch("app.rag.pipeline.get_retriever") as mock_ret,
        patch("app.rag.pipeline.get_generator") as mock_gen,
    ):
        mock_ret.return_value.retrieve = AsyncMock(return_value=[chunk])
        mock_gen.return_value.generate = AsyncMock(return_value="4 weeks annual leave.")
        mock_gen.return_value.provider = "ollama"
        mock_gen.return_value.model = "llama3.2:3b"

        from app.rag.pipeline import answer_query

        resp = await answer_query(QueryRequest(question="How much annual leave?"))

    assert resp.answer == "4 weeks annual leave."
    assert len(resp.citations) == 1
    assert resp.citations[0].doc_id == "eu-working-time-directive"
    assert resp.latency_ms > 0


@pytest.mark.asyncio
async def test_pipeline_deduplicates_citations_by_doc_id() -> None:
    chunks = [_make_chunk(score=0.9), _make_chunk(score=0.7)]
    with (
        patch("app.rag.pipeline.get_retriever") as mock_ret,
        patch("app.rag.pipeline.get_generator") as mock_gen,
    ):
        mock_ret.return_value.retrieve = AsyncMock(return_value=chunks)
        mock_gen.return_value.generate = AsyncMock(return_value="4 weeks.")
        mock_gen.return_value.provider = "ollama"
        mock_gen.return_value.model = "llama3.2:3b"

        from app.rag.pipeline import answer_query

        resp = await answer_query(QueryRequest(question="Leave?"))

    assert len(resp.citations) == 1
    assert resp.citations[0].score == 0.9  # highest kept


# ---------- ingest ----------

@pytest.mark.asyncio
async def test_ingest_returns_response_shape() -> None:
    fake_doc = MagicMock()
    fake_store = MagicMock()
    fake_store.add_documents = MagicMock(return_value=None)
    fake_store._collection.get.return_value = {"ids": []}

    with patch("app.rag.ingest.get_vectorstore", return_value=fake_store):
        from app.rag.ingest import ingest

        result = await ingest()

    assert result.documents >= 0
    assert result.chunks >= result.documents
    assert result.duration_ms > 0

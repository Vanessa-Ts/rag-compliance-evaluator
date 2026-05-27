"""Chroma-backed retriever conforming to the Retriever Protocol."""

from __future__ import annotations

import asyncio
from functools import lru_cache

from app.rag.store import get_vectorstore
from app.schemas import RetrievedChunk


class ChromaRetriever:
    async def retrieve(
        self, query: str, k: int, jurisdiction: str | None = None
    ) -> list[RetrievedChunk]:
        filter_dict = {"jurisdiction": jurisdiction} if jurisdiction else None
        results = await asyncio.to_thread(
            self._search, query, k, filter_dict
        )
        return results

    def _search(
        self,
        query: str,
        k: int,
        filter_dict: dict[str, str] | None,
    ) -> list[RetrievedChunk]:
        vs = get_vectorstore()
        raw = vs.similarity_search_with_relevance_scores(
            query, k=k, filter=filter_dict
        )
        chunks = []
        for doc, score in raw:
            m = doc.metadata
            chunks.append(
                RetrievedChunk(
                    doc_id=m.get("doc_id", ""),
                    title=m.get("title", ""),
                    jurisdiction=m.get("jurisdiction", ""),
                    source_url=m.get("source_url", ""),
                    text=doc.page_content,
                    score=float(score),
                )
            )
        return chunks


@lru_cache(maxsize=1)
def get_retriever() -> ChromaRetriever:
    return ChromaRetriever()

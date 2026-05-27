"""Corpus ingestion: load markdown docs, chunk, and upsert into Chroma."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import yaml
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.rag.store import get_vectorstore
from app.schemas import IngestResponse

_CORPUS_DIR = Path(__file__).resolve().parents[3] / "data" / "corpus"


def _parse_markdown(path: Path) -> tuple[dict[str, str], str]:
    """Return (frontmatter_dict, body_text) from a markdown file with YAML front matter."""
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---"):
        return {}, raw
    parts = raw.split("---", 2)
    if len(parts) < 3:
        return {}, raw
    front: dict[str, str] = yaml.safe_load(parts[1]) or {}
    body = parts[2].strip()
    return front, body


def _ingest_sync(force: bool) -> IngestResponse:
    t0 = time.perf_counter()
    vs = get_vectorstore()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    if force:
        # Clear the collection by deleting all documents
        col = vs._collection
        existing = col.get()
        if existing["ids"]:
            col.delete(ids=existing["ids"])

    docs: list[Document] = []
    ids: list[str] = []

    corpus_files = sorted(_CORPUS_DIR.glob("*.md"))
    doc_count = 0

    for md_path in corpus_files:
        if md_path.name == "README.md":
            continue
        front, body = _parse_markdown(md_path)
        if not front.get("doc_id"):
            continue
        doc_count += 1
        chunks = splitter.split_text(body)
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{front['doc_id']}-{i}"
            docs.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "doc_id": front.get("doc_id", ""),
                        "title": front.get("title", ""),
                        "jurisdiction": front.get("jurisdiction", ""),
                        "source_url": front.get("source_url", ""),
                        "chunk_index": i,
                    },
                )
            )
            ids.append(chunk_id)

    if docs:
        vs.add_documents(docs, ids=ids)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return IngestResponse(documents=doc_count, chunks=len(docs), duration_ms=elapsed_ms)


async def ingest(force: bool = False) -> IngestResponse:
    """Load corpus, chunk, and upsert into the vector store. Idempotent."""
    return await asyncio.to_thread(_ingest_sync, force)

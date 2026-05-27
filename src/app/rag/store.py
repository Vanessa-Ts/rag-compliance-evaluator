"""Vector store factory — lazy, cached Chroma persisted on disk.

The persist directory is created if missing. The Chroma instance is a
process-wide singleton so repeated calls return the same object. See plan WP3.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma

from app.core.config import settings
from app.rag.embeddings import get_embeddings


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """Return a process-wide singleton Chroma vector store.

    Ensures the persist directory exists before constructing the store.
    """
    persist_dir = Path(settings.chroma_path)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(persist_dir),
    )


def collection_count() -> int:
    """Return the number of items currently in the collection."""
    return int(get_vectorstore()._collection.count())

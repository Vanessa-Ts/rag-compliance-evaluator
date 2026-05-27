"""Embedding model factory — lazy, cached HuggingFace embeddings.

Construction downloads the model (~90MB) on first call, so it is kept lazy:
importing this module does not trigger a download. See plan WP3.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a process-wide singleton embeddings model.

    The model is constructed once per process on first call.
    """
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


import logging
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    app_name: str = "rag-compliance-evaluator"
    version: str = "0.2.0"
    environment: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"

    # --- LLM provider (generation + faithfulness judge) ---
    llm_provider: Literal["ollama", "anthropic", "openai"] = "ollama"
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "llama3.2:3b"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    generation_model: str | None = None  # provider-specific override

    # --- Embeddings / vector store ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_path: str = "data/chroma"
    collection_name: str = "employment_law"

    # --- Retrieval / chunking ---
    top_k: int = 4
    chunk_size: int = 800
    chunk_overlap: int = 120


@lru_cache
def get_settings() -> Settings:
    """Cached settings factory — one parse per process."""
    return Settings()
 
 
def configure_logging(s: Settings) -> None:
    """Set up structured logging based on environment."""
    logging.basicConfig(
        level=s.log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
 
 
settings = get_settings()
configure_logging(settings)
 
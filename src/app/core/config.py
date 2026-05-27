
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
    app_name: str = "docker-dev-template"
    version: str = "0.2.0"
    environment: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"


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
 
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    app_env: Literal["dev", "staging", "prod"] = "dev"
    database_url: str = Field(..., description="Database connection string.")

    youtube_api_key: str | None = None
    youtube_transcript_io_api_key: str | None = None
    openai_api_key: str | None = None
    openai_vector_store_id: str | None = None
    metadata_extraction_default_model: str = "gpt-5.4-nano"
    metadata_extraction_enhanced_model: str = "gpt-5.4-mini"
    metadata_extraction_timeout_seconds: float = 90.0
    mcp_auth_bearer_token: str | None = None

    redis_url: str | None = None

    gmail_smtp_username: str | None = None
    gmail_smtp_app_password: str | None = None
    notify_email_to: str | None = None

    transcript_proxy_http_url: str | None = None
    transcript_proxy_https_url: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings so the app has a single config source per process."""
    return Settings()

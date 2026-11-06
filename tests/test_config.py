from __future__ import annotations

import pytest
from pydantic import ValidationError

from learn_from.config import Settings, get_settings


def test_settings_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+psycopg://user:pass@localhost:5432/learn_from_dev",
    )
    monkeypatch.setenv("APP_ENV", "staging")
    monkeypatch.setenv("YOUTUBE_API_KEY", "yt-key")
    monkeypatch.setenv("YOUTUBE_TRANSCRIPT_IO_API_KEY", "ytio-key")
    monkeypatch.setenv("TRANSCRIPT_PROXY_HTTP_URL", "http://proxy.example:8080")
    monkeypatch.setenv("METADATA_EXTRACTION_DEFAULT_MODEL", "gpt-5.4-nano")
    monkeypatch.setenv("METADATA_EXTRACTION_ENHANCED_MODEL", "gpt-5.4-mini")

    get_settings.cache_clear()
    settings = get_settings()
    assert settings.database_url.endswith("learn_from_dev")
    assert settings.app_env == "staging"
    assert settings.youtube_api_key == "yt-key"
    assert settings.youtube_transcript_io_api_key == "ytio-key"
    assert settings.transcript_proxy_http_url == "http://proxy.example:8080"
    assert settings.metadata_extraction_default_model == "gpt-5.4-nano"
    assert settings.metadata_extraction_enhanced_model == "gpt-5.4-mini"

    get_settings.cache_clear()


def test_settings_requires_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)

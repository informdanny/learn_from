from __future__ import annotations

import pytest
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    select,
    update,
)

from learn_from.ingestion import backfill_channel, find_reusable_transcript, ingest_ad_hoc_video
from learn_from.transcripts import TranscriptRetrievalError, TranscriptSegment, YouTubeTranscript
from learn_from.youtube import YouTubeChannelMetadata, YouTubeVideoMetadata


def _build_test_engine():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    metadata = MetaData()

    Table(
        "channels",
        metadata,
        Column("channel_id", String(255), primary_key=True),
        Column("source_id", String(50), nullable=False),
        Column("channel_name", String(255), nullable=False),
        Column("channel_url", Text, nullable=False),
        Column("description", Text),
        Column("thumbnail_url", Text),
        Column("subscriber_count", Integer),
        Column("last_synced_at", DateTime),
    )
    Table(
        "media",
        metadata,
        Column("media_id", String(255), primary_key=True),
        Column("source_id", String(50), nullable=False),
        Column("channel_id", String(255)),
        Column("title", Text, nullable=False),
        Column("description", Text),
        Column("url", Text, nullable=False),
        Column("thumbnail_url", Text),
        Column("duration", Integer),
        Column("publish_date", DateTime),
        Column("views", Integer),
        Column("likes", Integer),
        Column("comments", Integer),
        Column("source_type", String(20), nullable=False, default="ad_hoc"),
        Column("last_updated", DateTime),
    )
    Table(
        "doc_content",
        metadata,
        Column("content_id", Integer, primary_key=True, autoincrement=True),
        Column("media_id", String(255), nullable=False),
        Column("full_text", Text, nullable=False),
        Column("formatted_text", Text),
        Column("timestamped_segments", JSON),
        Column("chunk_type", String(20), nullable=False, default="complete"),
        Column("language", Text, nullable=False, default="en"),
        Column("word_count", Integer),
        Column("character_count", Integer),
        Column("updated_at", DateTime),
    )
    Table(
        "sync_jobs",
        metadata,
        Column("job_id", String(255), primary_key=True),
        Column("job_type", String(50), nullable=False),
        Column("channel_id", String(255)),
        Column("media_ids", JSON),
        Column("status", String(20), nullable=False),
        Column("progress", Integer, nullable=False),
        Column("items_total", Integer),
        Column("items_processed", Integer),
        Column("items_succeeded", Integer),
        Column("items_failed", Integer),
        Column("error_message", Text),
        Column("started_at", DateTime),
        Column("completed_at", DateTime),
        Column("last_checkpoint_at", DateTime),
    )

    metadata.create_all(engine)
    return engine


def _fake_video_metadata(video_id: str) -> YouTubeVideoMetadata:
    return YouTubeVideoMetadata(
        video_id=video_id,
        title=f"title-{video_id}",
        description=f"description-{video_id}",
        channel_id="UCwsrIijdkrYaQmjnKO2FKRQ",
        channel_name="Data Over Dogma",
        url=f"https://www.youtube.com/watch?v={video_id}",
        publish_date="2026-04-01T08:00:00Z",
        duration_seconds=600,
        views=1000,
        likes=100,
        comments=10,
        thumbnail_url="https://example.com/thumb.jpg",
    )


def _fake_transcript(video_id: str) -> YouTubeTranscript:
    return YouTubeTranscript(
        video_id=video_id,
        language_code="en",
        is_generated=False,
        segments=[
            TranscriptSegment(text=f"{video_id} line 1", start_seconds=0.0, duration_seconds=2.0),
            TranscriptSegment(text=f"{video_id} line 2", start_seconds=2.0, duration_seconds=2.0),
        ],
    )


def test_backfill_channel_idempotent_with_sync_job_tracking(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _build_test_engine()
    transcript_calls: list[str] = []
    metadata_calls: list[int] = []

    monkeypatch.setattr(
        "learn_from.ingestion.fetch_channel_metadata",
        lambda **kwargs: YouTubeChannelMetadata(
            channel_id="UCwsrIijdkrYaQmjnKO2FKRQ",
            channel_name="Data Over Dogma",
            channel_url="https://www.youtube.com/channel/UCwsrIijdkrYaQmjnKO2FKRQ",
            description="",
            thumbnail_url="https://example.com/channel.jpg",
            subscriber_count=34300,
        ),
    )
    monkeypatch.setattr(
        "learn_from.ingestion.list_channel_video_ids",
        lambda **kwargs: ["ZPdtcM6dPrQ", "0ei_IMYuvzA"],
    )
    monkeypatch.setattr(
        "learn_from.ingestion.fetch_video_metadata",
        lambda **kwargs: _fake_video_metadata(kwargs["video_id_or_url"]),
    )
    def _count_transcript(**kwargs):
        transcript_calls.append(kwargs["video_id_or_url"])
        return _fake_transcript(kwargs["video_id_or_url"])

    def _count_metadata(**kwargs):
        metadata_calls.append(kwargs["content_id"])
        return {"content_id": kwargs["content_id"], "model": kwargs["model"]}

    monkeypatch.setattr("learn_from.ingestion.fetch_transcript", _count_transcript)
    monkeypatch.setattr("learn_from.ingestion.extract_and_persist_metadata_for_content", _count_metadata)

    summary_one = backfill_channel(
        engine=engine,
        channel_id="UCwsrIijdkrYaQmjnKO2FKRQ",
        youtube_api_key="test-key",
        transcript_io_api_key="transcript-key",
        include_metadata=True,
        openai_api_key="openai-key",
        metadata_model="gpt-5.4-nano",
    )
    summary_two = backfill_channel(
        engine=engine,
        channel_id="UCwsrIijdkrYaQmjnKO2FKRQ",
        youtube_api_key="test-key",
        transcript_io_api_key="transcript-key",
        include_metadata=True,
        openai_api_key="openai-key",
        metadata_model="gpt-5.4-nano",
    )

    assert summary_one["succeeded_count"] == 2
    assert summary_one["failed_count"] == 0
    assert summary_two["succeeded_count"] == 2
    assert summary_two["failed_count"] == 0
    assert summary_one["acquired_count"] == 2
    assert summary_two["reused_count"] == 2
    assert transcript_calls == ["ZPdtcM6dPrQ", "0ei_IMYuvzA"]
    assert len(metadata_calls) == 4  # Explicit extraction still runs for reused transcripts.

    metadata = MetaData()
    channels = Table("channels", metadata, autoload_with=engine)
    media = Table("media", metadata, autoload_with=engine)
    doc_content = Table("doc_content", metadata, autoload_with=engine)
    sync_jobs = Table("sync_jobs", metadata, autoload_with=engine)

    with engine.connect() as connection:
        assert connection.execute(select(channels.c.channel_id)).scalars().all() == [
            "UCwsrIijdkrYaQmjnKO2FKRQ"
        ]
        assert set(connection.execute(select(media.c.media_id)).scalars().all()) == {
            "0ei_IMYuvzA",
            "ZPdtcM6dPrQ",
        }
        assert set(connection.execute(select(doc_content.c.media_id)).scalars().all()) == {
            "ZPdtcM6dPrQ",
            "0ei_IMYuvzA",
        }
        statuses = connection.execute(select(sync_jobs.c.status)).scalars().all()
        assert statuses == ["completed", "completed"]


def test_backfill_channel_continues_when_one_item_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _build_test_engine()
    failing_video = "0ei_IMYuvzA"

    monkeypatch.setattr(
        "learn_from.ingestion.fetch_channel_metadata",
        lambda **kwargs: YouTubeChannelMetadata(
            channel_id="UCwsrIijdkrYaQmjnKO2FKRQ",
            channel_name="Data Over Dogma",
            channel_url="https://www.youtube.com/channel/UCwsrIijdkrYaQmjnKO2FKRQ",
            description="",
            thumbnail_url=None,
            subscriber_count=34300,
        ),
    )
    monkeypatch.setattr(
        "learn_from.ingestion.list_channel_video_ids",
        lambda **kwargs: ["ZPdtcM6dPrQ", failing_video],
    )
    monkeypatch.setattr(
        "learn_from.ingestion.fetch_video_metadata",
        lambda **kwargs: _fake_video_metadata(kwargs["video_id_or_url"]),
    )

    def _fake_fetch_transcript(**kwargs):
        if kwargs["video_id_or_url"] == failing_video:
            raise TranscriptRetrievalError("transcript unavailable")
        return _fake_transcript(kwargs["video_id_or_url"])

    monkeypatch.setattr("learn_from.ingestion.fetch_transcript", _fake_fetch_transcript)

    summary = backfill_channel(
        engine=engine,
        channel_id="UCwsrIijdkrYaQmjnKO2FKRQ",
        youtube_api_key="test-key",
        transcript_io_api_key="transcript-key",
        include_metadata=False,
    )

    assert summary["succeeded_count"] == 1
    assert summary["failed_count"] == 1
    assert summary["failed_items"][0]["video_id"] == failing_video


def test_ad_hoc_ingest_is_idempotent_and_keeps_source_type(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _build_test_engine()

    monkeypatch.setattr(
        "learn_from.ingestion.fetch_video_metadata",
        lambda **kwargs: _fake_video_metadata("dQw4w9WgXcQ"),
    )
    monkeypatch.setattr(
        "learn_from.ingestion.fetch_transcript",
        lambda **kwargs: _fake_transcript("dQw4w9WgXcQ"),
    )

    summary_one = ingest_ad_hoc_video(
        engine=engine,
        video_id_or_url="dQw4w9WgXcQ",
        youtube_api_key="test-key",
        include_metadata=False,
    )

    def _unexpected_provider(**kwargs):
        pytest.fail("A reusable transcript must not contact a provider or run extraction.")

    monkeypatch.setattr("learn_from.ingestion.fetch_video_metadata", _unexpected_provider)
    monkeypatch.setattr("learn_from.ingestion.fetch_transcript", _unexpected_provider)
    monkeypatch.setattr("learn_from.ingestion.extract_and_persist_metadata_for_content", _unexpected_provider)
    summary_two = ingest_ad_hoc_video(
        engine=engine,
        video_id_or_url="dQw4w9WgXcQ",
        youtube_api_key="test-key",
        include_metadata=False,
    )

    assert summary_one["succeeded_count"] == 1
    assert summary_two["succeeded_count"] == 1
    assert summary_one["item"]["transcript_status"] == "acquired"
    assert summary_two["item"]["transcript_status"] == "reused"
    assert summary_two["item"]["metadata_summary"] is None

    metadata = MetaData()
    media = Table("media", metadata, autoload_with=engine)
    doc_content = Table("doc_content", metadata, autoload_with=engine)
    sync_jobs = Table("sync_jobs", metadata, autoload_with=engine)

    with engine.connect() as connection:
        media_rows = connection.execute(
            select(media.c.media_id, media.c.source_type, media.c.channel_id)
        ).mappings().all()
        content_rows = connection.execute(select(doc_content.c.content_id)).mappings().all()
        job_status_rows = connection.execute(select(sync_jobs.c.status)).scalars().all()

    assert len(media_rows) == 1
    assert media_rows[0]["media_id"] == "dQw4w9WgXcQ"
    assert media_rows[0]["source_type"] == "ad_hoc"
    assert media_rows[0]["channel_id"] is None
    assert len(content_rows) == 1
    assert job_status_rows == ["completed", "completed"]


@pytest.mark.parametrize(
    "damaged_fields",
    [
        {"full_text": "  "},
        {"timestamped_segments": []},
        {"timestamped_segments": None},
        {"timestamped_segments": [{"text": "partial", "start_seconds": 0.0}]},
        {"timestamped_segments": [{"text": "dQw4w9WgXcQ line 1\ndQw4w9WgXcQ line 2", "start_seconds": -1}]},
        {"timestamped_segments": [{"text": "dQw4w9WgXcQ line 1\ndQw4w9WgXcQ line 2", "start_seconds": None}]},
        {"language": "es"},
        {"language": "und"},
    ],
)
def test_incomplete_or_wrong_language_transcript_is_fetched(
    monkeypatch: pytest.MonkeyPatch, damaged_fields: dict,
) -> None:
    engine = _build_test_engine()
    calls: list[str] = []

    def _fetch(**kwargs):
        calls.append(kwargs["video_id_or_url"])
        return _fake_transcript("dQw4w9WgXcQ")

    monkeypatch.setattr("learn_from.ingestion.fetch_video_metadata", lambda **kwargs: _fake_video_metadata("dQw4w9WgXcQ"))
    monkeypatch.setattr("learn_from.ingestion.fetch_transcript", _fetch)
    ingest_ad_hoc_video(engine=engine, video_id_or_url="dQw4w9WgXcQ", youtube_api_key="test-key", include_metadata=False)
    with engine.begin() as connection:
        content = Table("doc_content", MetaData(), autoload_with=connection)
        connection.execute(update(content).values(**damaged_fields))

    result = ingest_ad_hoc_video(engine=engine, video_id_or_url="dQw4w9WgXcQ", youtube_api_key="test-key", include_metadata=False)

    assert calls == ["dQw4w9WgXcQ", "dQw4w9WgXcQ"]
    assert result["item"]["transcript_status"] == "acquired"
    assert find_reusable_transcript(engine=engine, video_id_or_url="dQw4w9WgXcQ") is not None


@pytest.mark.parametrize("language", ["en-US", "en-Latn-US-fonipa", "es", None])
def test_ingestion_preserves_provider_language(
    monkeypatch: pytest.MonkeyPatch, language: str | None,
) -> None:
    engine = _build_test_engine()
    transcript = _fake_transcript("dQw4w9WgXcQ")
    monkeypatch.setattr("learn_from.ingestion.fetch_video_metadata", lambda **kwargs: _fake_video_metadata("dQw4w9WgXcQ"))
    monkeypatch.setattr(
        "learn_from.ingestion.fetch_transcript",
        lambda **kwargs: YouTubeTranscript(
            video_id=transcript.video_id, language_code=language,
            is_generated=False, segments=transcript.segments,
        ),
    )
    result = ingest_ad_hoc_video(engine=engine, video_id_or_url="dQw4w9WgXcQ", youtube_api_key="test-key", include_metadata=False)
    expected_language = language or "und"
    with engine.connect() as connection:
        content = Table("doc_content", MetaData(), autoload_with=connection)
        assert connection.execute(select(content.c.language)).scalar_one() == expected_language
    assert result["item"]["language"] == expected_language
    reused = find_reusable_transcript(engine=engine, video_id_or_url="https://youtu.be/dQw4w9WgXcQ", languages=("en",))
    assert (reused is not None) is (language in {"en-US", "en-Latn-US-fonipa"})


def test_prefer_existing_false_refreshes_transcript(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _build_test_engine()
    calls: list[str] = []

    def _fetch(**kwargs):
        calls.append(kwargs["video_id_or_url"])
        return _fake_transcript("dQw4w9WgXcQ")

    monkeypatch.setattr("learn_from.ingestion.fetch_video_metadata", lambda **kwargs: _fake_video_metadata("dQw4w9WgXcQ"))
    monkeypatch.setattr("learn_from.ingestion.fetch_transcript", _fetch)
    for _ in range(2):
        result = ingest_ad_hoc_video(
            engine=engine, video_id_or_url="dQw4w9WgXcQ", youtube_api_key="test-key",
            include_metadata=False, prefer_existing=False,
        )
        assert result["item"]["transcript_status"] == "acquired"
    assert len(calls) == 2

from __future__ import annotations

import math
from datetime import UTC, datetime
from uuid import uuid4
from collections.abc import Mapping, Sequence
from typing import Any

from sqlalchemy import MetaData, Table, insert, select, update
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError

from learn_from.metadata import (
    DEFAULT_EXTRACTION_MODEL,
    MetadataExtractionError,
    MetadataPersistenceError,
    extract_and_persist_metadata_for_content,
)
from learn_from.transcripts import (
    TranscriptNotAvailableError,
    TranscriptRetrievalError,
    fetch_transcript,
    format_transcript_with_timestamps,
)
from learn_from.youtube import (
    YouTubeApiError,
    YouTubeChannelMetadata,
    YouTubeVideoMetadata,
    extract_video_id,
    fetch_channel_metadata,
    fetch_video_metadata,
    list_channel_video_ids,
)


class IngestionError(RuntimeError):
    """Raised when ingestion fails at the workflow level."""


class IngestionConfigurationError(IngestionError):
    """Raised when required ingestion configuration is missing or invalid."""


class IngestionItemError(IngestionError):
    """Raised when one media item fails ingest processing."""


def backfill_channel(
    *,
    engine: Engine,
    channel_id: str,
    youtube_api_key: str,
    transcript_io_api_key: str | None = None,
    languages: Sequence[str] = ("en",),
    max_videos: int = 30,
    include_metadata: bool = True,
    openai_api_key: str | None = None,
    metadata_model: str = DEFAULT_EXTRACTION_MODEL,
    metadata_timeout_seconds: float = 90.0,
    transcript_proxy_http_url: str | None = None,
    transcript_proxy_https_url: str | None = None,
    prefer_existing: bool = True,
) -> dict[str, Any]:
    """
    Ingest recent channel videos into `media` + `doc_content` with sync-job tracking.
    """
    if not youtube_api_key:
        raise IngestionConfigurationError("YOUTUBE_API_KEY is required for channel backfill.")
    if max_videos <= 0:
        raise IngestionConfigurationError("max_videos must be greater than 0.")
    if include_metadata and not openai_api_key:
        raise IngestionConfigurationError(
            "OPENAI_API_KEY is required for channel backfill when metadata extraction is enabled."
        )

    normalized_channel_id = channel_id.strip()
    if not normalized_channel_id:
        raise IngestionConfigurationError("Channel ID is required.")

    channel = fetch_channel_metadata(
        channel_id=normalized_channel_id,
        api_key=youtube_api_key,
    )
    video_ids = list_channel_video_ids(
        channel_id=normalized_channel_id,
        api_key=youtube_api_key,
        max_results=max_videos,
    )

    with engine.begin() as connection:
        tables = _reflect_ingestion_tables(connection)
        _upsert_channel(connection=connection, table=tables["channels"], channel=channel)
        job_id = _create_sync_job(
            connection=connection,
            table=tables["sync_jobs"],
            job_type="channel_backfill",
            channel_id=normalized_channel_id,
            media_ids=video_ids,
            items_total=len(video_ids),
        )

    succeeded_items: list[dict[str, Any]] = []
    failed_items: list[dict[str, Any]] = []
    processed = 0
    succeeded = 0
    failed = 0

    try:
        for video_id in video_ids:
            processed += 1
            try:
                result = _ingest_video(
                    engine=engine,
                    video_id_or_url=video_id,
                    youtube_api_key=youtube_api_key,
                    transcript_io_api_key=transcript_io_api_key,
                    languages=languages,
                    source_type="channel",
                    channel_id_for_media=normalized_channel_id,
                    include_metadata=include_metadata,
                    openai_api_key=openai_api_key,
                    metadata_model=metadata_model,
                    metadata_timeout_seconds=metadata_timeout_seconds,
                    transcript_proxy_http_url=transcript_proxy_http_url,
                    transcript_proxy_https_url=transcript_proxy_https_url,
                    prefer_existing=prefer_existing,
                )
                succeeded += 1
                succeeded_items.append(result)
            except Exception as exc:
                failed += 1
                failed_items.append({"video_id": video_id, "error": str(exc)})

            with engine.begin() as connection:
                tables = _reflect_ingestion_tables(connection)
                _update_sync_job_progress(
                    connection=connection,
                    table=tables["sync_jobs"],
                    job_id=job_id,
                    items_total=len(video_ids),
                    items_processed=processed,
                    items_succeeded=succeeded,
                    items_failed=failed,
                    error_message=(
                        f"{failed} item(s) failed during channel backfill."
                        if failed > 0 and processed == len(video_ids)
                        else None
                    ),
                )

        with engine.begin() as connection:
            tables = _reflect_ingestion_tables(connection)
            _finalize_sync_job(
                connection=connection,
                table=tables["sync_jobs"],
                job_id=job_id,
                status="completed",
                items_total=len(video_ids),
                items_processed=processed,
                items_succeeded=succeeded,
                items_failed=failed,
                error_message=(f"{failed} item(s) failed." if failed else None),
            )
    except Exception as exc:  # pragma: no cover - defensive job-state handling
        with engine.begin() as connection:
            tables = _reflect_ingestion_tables(connection)
            _finalize_sync_job(
                connection=connection,
                table=tables["sync_jobs"],
                job_id=job_id,
                status="failed",
                items_total=len(video_ids),
                items_processed=processed,
                items_succeeded=succeeded,
                items_failed=failed,
                error_message=str(exc),
            )
        raise IngestionError(f"Channel backfill failed: {exc}") from exc

    return {
        "job_id": job_id,
        "channel_id": normalized_channel_id,
        "requested_video_count": len(video_ids),
        "processed_count": processed,
        "succeeded_count": succeeded,
        "failed_count": failed,
        "reused_count": sum(item["transcript_status"] == "reused" for item in succeeded_items),
        "acquired_count": sum(item["transcript_status"] == "acquired" for item in succeeded_items),
        "succeeded_items": succeeded_items,
        "failed_items": failed_items,
        "metadata_enabled": include_metadata,
        "metadata_model": metadata_model if include_metadata else None,
    }


def ingest_ad_hoc_video(
    *,
    engine: Engine,
    video_id_or_url: str,
    youtube_api_key: str,
    transcript_io_api_key: str | None = None,
    languages: Sequence[str] = ("en",),
    include_metadata: bool = True,
    openai_api_key: str | None = None,
    metadata_model: str = DEFAULT_EXTRACTION_MODEL,
    metadata_timeout_seconds: float = 90.0,
    transcript_proxy_http_url: str | None = None,
    transcript_proxy_https_url: str | None = None,
    prefer_existing: bool = True,
) -> dict[str, Any]:
    """
    Ingest a single ad-hoc video URL/ID into `media` + `doc_content` with sync-job tracking.
    """
    if not youtube_api_key:
        raise IngestionConfigurationError("YOUTUBE_API_KEY is required for ad-hoc ingestion.")
    if include_metadata and not openai_api_key:
        raise IngestionConfigurationError(
            "OPENAI_API_KEY is required for ad-hoc ingestion when metadata extraction is enabled."
        )

    video_id = extract_video_id(video_id_or_url)
    with engine.begin() as connection:
        tables = _reflect_ingestion_tables(connection)
        job_id = _create_sync_job(
            connection=connection,
            table=tables["sync_jobs"],
            job_type="ad_hoc_media",
            media_ids=[video_id],
            items_total=1,
        )

    try:
        result = _ingest_video(
            engine=engine,
            video_id_or_url=video_id_or_url,
            youtube_api_key=youtube_api_key,
            transcript_io_api_key=transcript_io_api_key,
            languages=languages,
            source_type="ad_hoc",
            channel_id_for_media=None,
            include_metadata=include_metadata,
            openai_api_key=openai_api_key,
            metadata_model=metadata_model,
            metadata_timeout_seconds=metadata_timeout_seconds,
            transcript_proxy_http_url=transcript_proxy_http_url,
            transcript_proxy_https_url=transcript_proxy_https_url,
            prefer_existing=prefer_existing,
        )
    except Exception as exc:
        with engine.begin() as connection:
            tables = _reflect_ingestion_tables(connection)
            _finalize_sync_job(
                connection=connection,
                table=tables["sync_jobs"],
                job_id=job_id,
                status="failed",
                items_total=1,
                items_processed=1,
                items_succeeded=0,
                items_failed=1,
                error_message=str(exc),
            )
        raise IngestionError(f"Ad-hoc ingestion failed for '{video_id_or_url}': {exc}") from exc

    with engine.begin() as connection:
        tables = _reflect_ingestion_tables(connection)
        _finalize_sync_job(
            connection=connection,
            table=tables["sync_jobs"],
            job_id=job_id,
            status="completed",
            items_total=1,
            items_processed=1,
            items_succeeded=1,
            items_failed=0,
            error_message=None,
        )

    return {
        "job_id": job_id,
        "video_id": video_id,
        "processed_count": 1,
        "succeeded_count": 1,
        "failed_count": 0,
        "reused_count": int(result["transcript_status"] == "reused"),
        "acquired_count": int(result["transcript_status"] == "acquired"),
        "item": result,
        "metadata_enabled": include_metadata,
        "metadata_model": metadata_model if include_metadata else None,
    }


def find_reusable_transcript(
    *,
    engine: Engine,
    video_id_or_url: str,
    languages: Sequence[str] = ("en",),
) -> dict[str, Any] | None:
    """Read a complete, internally consistent transcript without contacting a provider."""
    video_id = extract_video_id(video_id_or_url)
    with engine.connect() as connection:
        metadata = MetaData()
        media = Table("media", metadata, autoload_with=connection)
        content = Table("doc_content", metadata, autoload_with=connection)
        rows = connection.execute(
            select(
                content.c.content_id,
                content.c.full_text,
                content.c.timestamped_segments,
                content.c.language,
                media.c.title,
                media.c.publish_date,
                media.c.source_type,
            )
            .select_from(content.join(media, content.c.media_id == media.c.media_id))
            .where(content.c.media_id == video_id, content.c.chunk_type == "complete")
            .order_by(content.c.content_id)
        ).mappings().all()

    for row in rows:
        language = str(row["language"] or "").strip()
        if not _matches_language(language, languages):
            continue
        full_text = row["full_text"]
        segments = row["timestamped_segments"]
        if not _has_usable_transcript(full_text, segments):
            continue
        published_at = row["publish_date"]
        return {
            "video_id": video_id,
            "media_id": video_id,
            "content_id": int(row["content_id"]),
            "title": row["title"],
            "published_at": published_at.isoformat() if published_at else None,
            "word_count": len(full_text.split()),
            "source_type": row["source_type"],
            "language": language,
            "transcript_status": "reused",
        }
    return None


def _matches_language(language: str, requested: Sequence[str]) -> bool:
    normalized = language.lower().replace("_", "-")
    if not normalized or normalized == "und":
        return False
    for preference in requested:
        preferred = preference.strip().lower().replace("_", "-")
        if preferred and (normalized == preferred or normalized.startswith(f"{preferred}-")):
            return True
    return not requested


def _has_usable_transcript(full_text: Any, segments: Any) -> bool:
    if not isinstance(full_text, str) or not full_text.strip():
        return False
    if not isinstance(segments, list) or not segments:
        return False
    segment_text: list[str] = []
    previous_start = -1.0
    for segment in segments:
        if not isinstance(segment, Mapping) or not isinstance(segment.get("text"), str):
            return False
        start = _finite_float_or_none(segment.get("start_seconds"))
        if start is None or start < 0 or start < previous_start:
            return False
        duration = segment.get("duration_seconds")
        if duration is not None:
            normalized_duration = _finite_float_or_none(duration)
            if normalized_duration is None or normalized_duration < 0:
                return False
        previous_start = start
        segment_text.append(segment["text"])
    return " ".join(full_text.split()) == " ".join(" ".join(segment_text).split())


def _ingest_video(
    *,
    engine: Engine,
    video_id_or_url: str,
    youtube_api_key: str,
    transcript_io_api_key: str | None,
    languages: Sequence[str],
    source_type: str,
    channel_id_for_media: str | None,
    include_metadata: bool,
    openai_api_key: str | None,
    metadata_model: str,
    metadata_timeout_seconds: float,
    transcript_proxy_http_url: str | None,
    transcript_proxy_https_url: str | None,
    prefer_existing: bool = True,
) -> dict[str, Any]:
    result = find_reusable_transcript(
        engine=engine,
        video_id_or_url=video_id_or_url,
        languages=languages,
    ) if prefer_existing else None
    if result is None:
        result = _acquire_video_transcript(
            engine=engine,
            video_id_or_url=video_id_or_url,
            youtube_api_key=youtube_api_key,
            transcript_io_api_key=transcript_io_api_key,
            languages=languages,
            source_type=source_type,
            channel_id_for_media=channel_id_for_media,
            transcript_proxy_http_url=transcript_proxy_http_url,
            transcript_proxy_https_url=transcript_proxy_https_url,
        )

    metadata_summary: dict[str, Any] | None = None
    if include_metadata:
        if not openai_api_key:
            raise IngestionConfigurationError("OPENAI_API_KEY is required when metadata extraction is enabled.")
        try:
            metadata_summary = extract_and_persist_metadata_for_content(
                engine=engine,
                content_id=result["content_id"],
                openai_api_key=openai_api_key,
                model=metadata_model,
                timeout_seconds=metadata_timeout_seconds,
            )
        except (MetadataExtractionError, MetadataPersistenceError, SQLAlchemyError, ValueError) as exc:
            raise IngestionItemError(f"Failed metadata extraction/persistence: {exc}") from exc

    result["metadata_summary"] = metadata_summary
    return result


def _acquire_video_transcript(
    *,
    engine: Engine,
    video_id_or_url: str,
    youtube_api_key: str,
    transcript_io_api_key: str | None,
    languages: Sequence[str],
    source_type: str,
    channel_id_for_media: str | None,
    transcript_proxy_http_url: str | None,
    transcript_proxy_https_url: str | None,
) -> dict[str, Any]:
    try:
        metadata = fetch_video_metadata(
            video_id_or_url=video_id_or_url,
            api_key=youtube_api_key,
        )
    except (YouTubeApiError, ValueError) as exc:
        raise IngestionItemError(f"Failed to fetch video metadata: {exc}") from exc

    try:
        transcript = fetch_transcript(
            video_id_or_url=video_id_or_url,
            languages=languages,
            transcript_io_api_key=transcript_io_api_key,
            proxy_http_url=transcript_proxy_http_url,
            proxy_https_url=transcript_proxy_https_url,
        )
    except (TranscriptNotAvailableError, TranscriptRetrievalError, ValueError) as exc:
        raise IngestionItemError(f"Failed to fetch transcript: {exc}") from exc

    full_text = transcript.full_text.strip()
    if not full_text:
        raise IngestionItemError("Transcript was empty after normalization.")

    sanitized_segments = _sanitize_timestamped_segments(transcript.timestamped_segments)
    formatted_text = format_transcript_with_timestamps(transcript)
    word_count = len(full_text.split())
    character_count = len(full_text)
    language = (transcript.language_code or "").strip() or "und"

    try:
        with engine.begin() as connection:
            tables = _reflect_ingestion_tables(connection)
            _upsert_media(
                connection=connection,
                table=tables["media"],
                metadata=metadata,
                source_type=source_type,
                channel_id=channel_id_for_media,
            )
            content_id = _upsert_complete_doc_content(
                connection=connection,
                table=tables["doc_content"],
                media_id=metadata.video_id,
                full_text=full_text,
                formatted_text=formatted_text,
                timestamped_segments=sanitized_segments,
                word_count=word_count,
                character_count=character_count,
                language=language,
            )
    except SQLAlchemyError as exc:
        raise IngestionItemError(f"Failed writing media/doc_content rows: {exc}") from exc

    return {
        "video_id": metadata.video_id,
        "media_id": metadata.video_id,
        "content_id": content_id,
        "title": metadata.title,
        "published_at": metadata.publish_date,
        "word_count": word_count,
        "source_type": source_type,
        "language": language,
        "transcript_status": "acquired",
    }


def _reflect_ingestion_tables(connection: Connection) -> dict[str, Table]:
    metadata = MetaData()
    tables: dict[str, Table] = {}
    for table_name in ("channels", "media", "doc_content", "sync_jobs"):
        tables[table_name] = Table(table_name, metadata, autoload_with=connection)
    return tables


def _upsert_channel(*, connection: Connection, table: Table, channel: YouTubeChannelMetadata) -> None:
    row = (
        connection.execute(
            select(table.c.channel_id).where(table.c.channel_id == channel.channel_id)
        )
        .mappings()
        .first()
    )
    now = _utc_now()
    values = _filtered_values(
        table,
        {
            "source_id": channel.source_id,
            "channel_name": channel.channel_name,
            "channel_url": channel.channel_url,
            "description": channel.description,
            "thumbnail_url": channel.thumbnail_url,
            "subscriber_count": channel.subscriber_count,
            "last_synced_at": now,
        },
    )
    if row is None:
        insert_values = {"channel_id": channel.channel_id, **values}
        connection.execute(insert(table).values(**insert_values))
        return

    connection.execute(
        update(table)
        .where(table.c.channel_id == channel.channel_id)
        .values(**values)
    )


def _upsert_media(
    *,
    connection: Connection,
    table: Table,
    metadata: YouTubeVideoMetadata,
    source_type: str,
    channel_id: str | None,
) -> None:
    existing = (
        connection.execute(select(table.c.media_id).where(table.c.media_id == metadata.video_id))
        .mappings()
        .first()
    )
    now = _utc_now()
    values = _filtered_values(
        table,
        {
            "source_id": metadata.source_id,
            "channel_id": channel_id,
            "title": metadata.title,
            "description": metadata.description,
            "url": metadata.url,
            "thumbnail_url": metadata.thumbnail_url,
            "duration": metadata.duration_seconds,
            "publish_date": _parse_publish_date(metadata.publish_date),
            "views": metadata.views,
            "likes": metadata.likes,
            "comments": metadata.comments,
            "source_type": source_type,
            "last_updated": now,
        },
    )

    if existing is None:
        insert_values = {"media_id": metadata.video_id, **values}
        connection.execute(insert(table).values(**insert_values))
        return

    connection.execute(
        update(table)
        .where(table.c.media_id == metadata.video_id)
        .values(**values)
    )


def _upsert_complete_doc_content(
    *,
    connection: Connection,
    table: Table,
    media_id: str,
    full_text: str,
    formatted_text: str,
    timestamped_segments: Sequence[Mapping[str, Any]],
    word_count: int,
    character_count: int,
    language: str = "und",
) -> int:
    existing = (
        connection.execute(
            select(table.c.content_id)
            .where(
                table.c.media_id == media_id,
                table.c.chunk_type == "complete",
            )
            .order_by(table.c.content_id)
        )
        .mappings()
        .first()
    )
    now = _utc_now()
    values = _filtered_values(
        table,
        {
            "media_id": media_id,
            "full_text": full_text,
            "formatted_text": formatted_text,
            "timestamped_segments": list(timestamped_segments),
            "chunk_type": "complete",
            "language": language,
            "word_count": word_count,
            "character_count": character_count,
            "updated_at": now,
        },
    )

    if existing is None:
        result = connection.execute(insert(table).values(**values))
        inserted = result.inserted_primary_key[0] if result.inserted_primary_key else None
        if inserted is not None:
            return int(inserted)
        lookup = (
            connection.execute(
                select(table.c.content_id)
                .where(
                    table.c.media_id == media_id,
                    table.c.chunk_type == "complete",
                )
                .order_by(table.c.content_id)
            )
            .mappings()
            .first()
        )
        if lookup is None:
            raise IngestionItemError(f"Unable to resolve inserted doc_content row for media_id={media_id}.")
        return int(lookup["content_id"])

    content_id = int(existing["content_id"])
    update_values = dict(values)
    update_values.pop("media_id", None)
    update_values.pop("chunk_type", None)
    connection.execute(
        update(table)
        .where(table.c.content_id == content_id)
        .values(**update_values)
    )
    return content_id


def _create_sync_job(
    *,
    connection: Connection,
    table: Table,
    job_type: str,
    channel_id: str | None = None,
    media_ids: Sequence[str] | None = None,
    items_total: int,
) -> str:
    job_id = f"job_{uuid4().hex}"
    now = _utc_now()
    values = _filtered_values(
        table,
        {
            "job_id": job_id,
            "job_type": job_type,
            "channel_id": channel_id,
            "media_ids": list(media_ids or []),
            "status": "processing",
            "progress": 0,
            "is_resumable": False,
            "resume_priority": 0,
            "items_total": items_total,
            "items_processed": 0,
            "items_succeeded": 0,
            "items_failed": 0,
            "created_at": now,
            "started_at": now,
            "last_checkpoint_at": now,
        },
    )
    connection.execute(insert(table).values(**values))
    return job_id


def _update_sync_job_progress(
    *,
    connection: Connection,
    table: Table,
    job_id: str,
    items_total: int,
    items_processed: int,
    items_succeeded: int,
    items_failed: int,
    error_message: str | None = None,
) -> None:
    progress = 100 if items_total == 0 else int((items_processed / items_total) * 100)
    values = _filtered_values(
        table,
        {
            "status": "processing",
            "progress": min(max(progress, 0), 100),
            "items_total": items_total,
            "items_processed": items_processed,
            "items_succeeded": items_succeeded,
            "items_failed": items_failed,
            "last_checkpoint_at": _utc_now(),
            "error_message": error_message,
        },
    )
    connection.execute(update(table).where(table.c.job_id == job_id).values(**values))


def _finalize_sync_job(
    *,
    connection: Connection,
    table: Table,
    job_id: str,
    status: str,
    items_total: int,
    items_processed: int,
    items_succeeded: int,
    items_failed: int,
    error_message: str | None,
) -> None:
    values = _filtered_values(
        table,
        {
            "status": status,
            "progress": 100,
            "items_total": items_total,
            "items_processed": items_processed,
            "items_succeeded": items_succeeded,
            "items_failed": items_failed,
            "error_message": error_message,
            "completed_at": _utc_now(),
            "last_checkpoint_at": _utc_now(),
        },
    )
    connection.execute(update(table).where(table.c.job_id == job_id).values(**values))


def _sanitize_timestamped_segments(
    segments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for segment in segments:
        text_value = str(segment.get("text", "")).strip()
        start_value = _finite_float_or_none(segment.get("start_seconds"))
        duration_value = _finite_float_or_none(segment.get("duration_seconds"))
        sanitized.append(
            {
                "text": text_value,
                "start_seconds": 0.0 if start_value is None else start_value,
                "duration_seconds": duration_value,
            }
        )
    return sanitized


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _parse_publish_date(raw_value: str | None) -> datetime | None:
    if not raw_value:
        return None
    try:
        return datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _filtered_values(table: Table, values: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if key in table.c}


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)

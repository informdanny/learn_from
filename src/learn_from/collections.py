"""Durable, bounded transcript collections using the existing cloud job table."""
from __future__ import annotations

import copy
import hashlib
import json
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any
from urllib.error import HTTPError

from sqlalchemy import MetaData, Table, cast, select, text, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine

from learn_from.collection_discovery import discover_collection_plan
from learn_from.ingestion import (
    IngestionError,
    _create_sync_job,
    _ingest_video,
    _upsert_channel,
    find_reusable_transcript,
)
from learn_from.metadata import DEFAULT_EXTRACTION_MODEL
from learn_from.transcripts import TranscriptNotAvailableError
from learn_from.youtube import VideoNotFoundError, YouTubeChannelMetadata, extract_video_id


MAX_PROVIDER_REQUESTS = 6000
PROVIDER_REQUEST_INTERVAL_SECONDS = 3.0
_ITEM_STATUSES = ("pending", "inflight", "acquired", "reused", "unavailable", "failed", "uncertain")


class CollectionBusyError(IngestionError):
    """Another worker already holds the collection's database lock."""


def _now() -> datetime:
    return datetime.now(UTC)


class _CollectionRepository:
    def __init__(self, engine: Engine):
        self.engine = engine
        with engine.connect() as connection:
            self.jobs = Table("sync_jobs", MetaData(), autoload_with=connection)

    def load(self, job_id: str) -> tuple[dict[str, Any], str]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(self.jobs.c.checkpoint_data, self.jobs.c.status)
                .where(self.jobs.c.job_id == job_id)
            ).mappings().first()
        if row is None:
            raise ValueError(f"Collection job not found: {job_id}")
        checkpoint = row["checkpoint_data"]
        if not isinstance(checkpoint, dict) or checkpoint.get("collection_version") != 1:
            raise ValueError("This job does not contain a supported collection checkpoint.")
        return copy.deepcopy(checkpoint), row["status"]

    def save(self, job_id: str, checkpoint: dict[str, Any], status: str) -> None:
        counts = Counter(item["status"] for item in checkpoint["items"])
        total = len(checkpoint["items"])
        processed = total - counts["pending"] - counts["inflight"]
        now = _now()
        values = {
            "checkpoint_data": checkpoint,
            "status": status,
            "items_total": total,
            "items_processed": processed,
            "items_succeeded": counts["acquired"] + counts["reused"],
            "items_failed": counts["unavailable"] + counts["failed"] + counts["uncertain"],
            "progress": int(processed * 100 / total) if total else 100,
            "last_checkpoint_at": now,
            "completed_at": now if status == "completed" else None,
            "is_resumable": bool(counts["pending"] or counts["inflight"]),
            "budget_halt_reason": checkpoint.get("halt_reason") if status == "budget_halted" else None,
            "error_message": checkpoint.get("halt_reason"),
        }
        with self.engine.begin() as connection:
            # The index worker owns this independent subkey. Do not overwrite a newer
            # index manifest with the older copy loaded at the start of collection.
            collection_checkpoint = {key: value for key, value in checkpoint.items() if key != "hosted_index"}
            if self.engine.dialect.name == "postgresql":
                values["checkpoint_data"] = self.jobs.c.checkpoint_data.op("||")(
                    cast(collection_checkpoint, JSONB)
                )
            else:
                current = connection.execute(
                    select(self.jobs.c.checkpoint_data).where(self.jobs.c.job_id == job_id)
                ).scalar_one_or_none() or {}
                values["checkpoint_data"] = {**current, **collection_checkpoint}
            result = connection.execute(
                update(self.jobs).where(self.jobs.c.job_id == job_id)
                .values(**{key: value for key, value in values.items() if key in self.jobs.c})
            )
            if result.rowcount != 1:
                raise ValueError(f"Collection job disappeared: {job_id}")


@contextmanager
def _collection_lock(engine: Engine, job_id: str) -> Iterator[Callable[[], None]]:
    if engine.dialect.name != "postgresql":
        # SQLite is only used by local isolated tests; production collections use Neon.
        yield lambda: None
        return
    lock_key = int.from_bytes(hashlib.sha256(job_id.encode()).digest()[:8], "big", signed=True)
    with engine.connect() as connection:
        acquired = connection.execute(
            text("SELECT pg_try_advisory_xact_lock(:key)"), {"key": lock_key}
        ).scalar_one()
        if not acquired:
            connection.rollback()
            raise CollectionBusyError(f"Collection {job_id} is already running.")
        try:
            # Keep the transaction open to pin the connection through PgBouncer's
            # transaction pool. Commit would release this transaction-scoped lock.
            def check_lock() -> None:
                # Losing this connection also loses its transaction lock. A long
                # worker must stop rather than continue spending on other connections.
                connection.execute(text("SELECT 1")).scalar_one()

            yield check_lock
        finally:
            connection.rollback()


def create_collection(engine: Engine, plan: Mapping[str, Any], max_provider_requests: int) -> str:
    """Persist a collection and its request cap; this does not acquire transcripts."""
    if isinstance(max_provider_requests, bool) or not isinstance(max_provider_requests, int):
        raise ValueError("max_provider_requests must be an integer.")
    if not 0 <= max_provider_requests <= MAX_PROVIDER_REQUESTS:
        raise ValueError(f"max_provider_requests must be between 0 and {MAX_PROVIDER_REQUESTS}.")
    channels = plan.get("channels")
    selected = plan.get("items")
    if not isinstance(channels, list) or not channels or not isinstance(selected, list):
        raise ValueError("A collection plan requires channels and an items list.")
    channel_ids = {channel["channel_id"] for channel in channels}
    if len(channel_ids) != len(channels):
        raise ValueError("Collection channels must be unique.")
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in selected:
        video_id = extract_video_id(raw["video_id"])
        if video_id in seen:
            raise ValueError(f"Duplicate collection video: {video_id}")
        if raw.get("channel_id") not in channel_ids:
            raise ValueError(f"Video {video_id} does not belong to a collection channel.")
        seen.add(video_id)
        items.append({
            "video_id": video_id,
            "channel_id": raw["channel_id"],
            "title": str(raw.get("title") or "")[:500],
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "published_at": raw.get("published_at"),
            "duration_seconds": raw.get("duration_seconds"),
            "status": "pending",
            "attempts_reserved": 0,
        })
    checkpoint = {
        "collection_version": 1,
        "created_at": _now().isoformat(),
        "plan": {key: value for key, value in plan.items() if key != "items"},
        "items": items,
        "max_provider_requests": max_provider_requests,
        "provider_requests_reserved": 0,
        "languages": ["en"],
        "halt_reason": None,
    }
    # Descriptions stay at channel level. Reject unexpectedly huge plans before any write.
    if len(json.dumps(checkpoint, allow_nan=False).encode()) > 1_048_576:
        raise ValueError("Collection checkpoint exceeds 1 MB; split the collection into smaller date windows.")
    with engine.begin() as connection:
        metadata = MetaData()
        channel_table = Table("channels", metadata, autoload_with=connection)
        jobs = Table("sync_jobs", metadata, autoload_with=connection)
        for channel in channels:
            _upsert_channel(connection=connection, table=channel_table, channel=YouTubeChannelMetadata(**channel))
        job_id = _create_sync_job(
            connection=connection, table=jobs, job_type="channel_backfill",
            media_ids=[item["video_id"] for item in items], items_total=len(items),
        )
        connection.execute(update(jobs).where(jobs.c.job_id == job_id).values(
            checkpoint_data=checkpoint, status="pending",
        ))
    return job_id


def _summary(job_id: str, checkpoint: Mapping[str, Any], status: str) -> dict[str, Any]:
    counts = Counter(item["status"] for item in checkpoint["items"])
    per_channel = []
    for channel in checkpoint["plan"]["channels"]:
        channel_counts = Counter(
            item["status"] for item in checkpoint["items"]
            if item["channel_id"] == channel["channel_id"]
        )
        per_channel.append({
            "channel_id": channel["channel_id"],
            "channel_name": channel["channel_name"],
            "total": sum(channel_counts.values()),
            **{name: channel_counts[name] for name in _ITEM_STATUSES},
        })
    return {
        "job_id": job_id,
        "status": status,
        "total": len(checkpoint["items"]),
        **{f"{name}_count": counts[name] for name in _ITEM_STATUSES},
        "provider_requests_reserved": checkpoint["provider_requests_reserved"],
        "max_provider_requests": checkpoint["max_provider_requests"],
        "provider_requests_remaining": max(0, checkpoint["max_provider_requests"] - checkpoint["provider_requests_reserved"]),
        "budget_basis": "Per-job request cap, not verified remaining provider account allowance.",
        "halt_reason": checkpoint.get("halt_reason"),
        "discovery_complete": checkpoint["plan"].get("discovery_complete", False),
        "discovery_warning": checkpoint["plan"].get("discovery_warning"),
        "per_channel": per_channel,
    }


def collection_status(engine: Engine, job_id: str) -> dict[str, Any]:
    checkpoint, status = _CollectionRepository(engine).load(job_id)
    return _summary(job_id, checkpoint, status)


def _classify_error(error: Exception) -> tuple[str, str | None]:
    causes: list[BaseException] = []
    current: BaseException | None = error
    while current is not None and current not in causes:
        causes.append(current)
        current = current.__cause__ or current.__context__
    if any(isinstance(cause, (TranscriptNotAvailableError, VideoNotFoundError)) for cause in causes):
        return "unavailable", None
    codes = {cause.code for cause in causes if isinstance(cause, HTTPError)}
    if codes & {401, 403}:
        return "failed", "provider_authentication"
    if 402 in codes:
        return "failed", "provider_credit_limit"
    if 429 in codes:
        return "failed", "provider_rate_limit"
    message = str(error).lower()
    if any(marker in message for marker in ("insufficient credit", "no credits", "credit limit", "quota exceeded", "token limit", "out of token")):
        return "failed", "provider_credit_limit"
    # An unclassified provider/network/storage failure may follow a chargeable request.
    return "uncertain", "request_outcome_uncertain"


def _safe_error(error: Exception, settings: Any) -> str:
    message = str(error)
    for field in ("youtube_api_key", "youtube_transcript_io_api_key", "openai_api_key", "database_url"):
        secret = getattr(settings, field, None)
        if isinstance(secret, str) and secret:
            message = message.replace(secret, "[redacted]")
    return message[:1000]


def run_collection(
    engine: Engine,
    job_id: str,
    settings: Any,
    max_items: int | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Continue pending items, reserving every request durably before external work."""
    if max_items is not None and (isinstance(max_items, bool) or not isinstance(max_items, int) or max_items < 0):
        raise ValueError("max_items must be a nonnegative integer or None.")
    repository = _CollectionRepository(engine)
    with _collection_lock(engine, job_id) as check_lock:
        checkpoint, _ = repository.load(job_id)
        cap = checkpoint["max_provider_requests"]
        reserved = checkpoint["provider_requests_reserved"]
        if not isinstance(cap, int) or not 0 <= cap <= MAX_PROVIDER_REQUESTS or not isinstance(reserved, int) or reserved < 0 or reserved > cap:
            raise ValueError("Invalid saved collection request budget.")
        languages = tuple(checkpoint.get("languages") or ["en"])

        def save(status: str) -> dict[str, Any]:
            repository.save(job_id, checkpoint, status)
            result = _summary(job_id, checkpoint, status)
            if on_progress is not None:
                on_progress(result)
            return result

        # A crashed request may already have persisted its transcript. Never spend twice
        # merely because the process died before updating the collection checkpoint.
        recovered_uncertain = False
        for item in checkpoint["items"]:
            if item["status"] != "inflight":
                continue
            cached = find_reusable_transcript(engine=engine, video_id_or_url=item["video_id"], languages=languages)
            if cached is not None:
                item.update(status="acquired", content_id=cached["content_id"], language=cached["language"], recovered_after_interrupt=True)
            else:
                item.update(status="uncertain", error="Interrupted request has no verified persisted transcript; no automatic retry.")
                recovered_uncertain = True
            item["finished_at"] = _now().isoformat()
        if recovered_uncertain:
            checkpoint["halt_reason"] = "interrupted_request_uncertain"
            return save("paused")

        if not any(item["status"] == "pending" for item in checkpoint["items"]):
            status = "paused" if any(item["status"] in {"failed", "uncertain"} for item in checkpoint["items"]) else "completed"
            return save(status)
        checkpoint["halt_reason"] = None
        save("processing")
        processed = 0
        last_attempt = checkpoint.get("last_request_reserved_at")
        last_attempt_monotonic: float | None = None
        if last_attempt:
            elapsed = max(0.0, (_now() - datetime.fromisoformat(last_attempt)).total_seconds())
            last_attempt_monotonic = time.monotonic() - elapsed

        for item in checkpoint["items"]:
            if item["status"] != "pending":
                continue
            check_lock()
            if max_items is not None and processed >= max_items:
                checkpoint["halt_reason"] = "invocation_item_limit"
                return save("paused")
            cached = find_reusable_transcript(engine=engine, video_id_or_url=item["video_id"], languages=languages)
            if cached is not None:
                item.update(status="reused", content_id=cached["content_id"], language=cached["language"], finished_at=_now().isoformat())
                processed += 1
                save("processing")
                continue
            if checkpoint["provider_requests_reserved"] >= cap:
                checkpoint["halt_reason"] = "provider_request_cap"
                # Continue looking for cached items; no acquisition is allowed.
                continue
            if not getattr(settings, "youtube_api_key", None) or not getattr(settings, "youtube_transcript_io_api_key", None):
                checkpoint["halt_reason"] = "provider_credentials_missing"
                return save("paused")
            if last_attempt_monotonic is not None:
                delay = PROVIDER_REQUEST_INTERVAL_SECONDS - (time.monotonic() - last_attempt_monotonic)
                if delay > 0:
                    time.sleep(delay)
            check_lock()
            item.update(status="inflight", attempts_reserved=item["attempts_reserved"] + 1, reserved_at=_now().isoformat())
            checkpoint["provider_requests_reserved"] += 1
            checkpoint["last_request_reserved_at"] = item["reserved_at"]
            # This transaction must succeed before any provider request begins.
            repository.save(job_id, checkpoint, "processing")
            last_attempt_monotonic = time.monotonic()
            try:
                result = _ingest_video(
                    engine=engine, video_id_or_url=item["video_id"],
                    youtube_api_key=settings.youtube_api_key,
                    transcript_io_api_key=settings.youtube_transcript_io_api_key,
                    languages=languages, source_type="channel", channel_id_for_media=item["channel_id"],
                    include_metadata=False, openai_api_key=None, metadata_model=DEFAULT_EXTRACTION_MODEL,
                    metadata_timeout_seconds=90.0,
                    transcript_proxy_http_url=getattr(settings, "transcript_proxy_http_url", None),
                    transcript_proxy_https_url=getattr(settings, "transcript_proxy_https_url", None),
                    prefer_existing=True,
                )
                item.update(status=result["transcript_status"], content_id=result["content_id"], language=result["language"])
                if result["transcript_status"] == "reused":
                    # Another writer supplied the transcript between the two cache reads.
                    checkpoint["provider_requests_reserved"] -= 1
                    item["attempts_reserved"] -= 1
            except Exception as exc:
                item["status"], halt_reason = _classify_error(exc)
                item["error"] = _safe_error(exc, settings)
                item["finished_at"] = _now().isoformat()
                processed += 1
                if halt_reason is not None:
                    checkpoint["halt_reason"] = halt_reason
                    return save("paused")
                save("processing")
                continue
            item["finished_at"] = _now().isoformat()
            processed += 1
            save("processing")

        if any(item["status"] == "pending" for item in checkpoint["items"]):
            checkpoint["halt_reason"] = "provider_request_cap"
            return save("budget_halted")
        if any(item["status"] in {"failed", "uncertain"} for item in checkpoint["items"]):
            checkpoint["halt_reason"] = "unresolved_item_outcomes"
            return save("paused")
        checkpoint["halt_reason"] = None
        return save("completed")

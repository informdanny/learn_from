from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest
from sqlalchemy import JSON, Column, DateTime, Integer, MetaData, String, Table, Text, create_engine, select, update
from sqlalchemy.dialects.postgresql import JSONB, psycopg

from learn_from import collections
from learn_from.ingestion import IngestionItemError
from learn_from.transcripts import TranscriptNotAvailableError


def _plan(count: int = 3) -> dict:
    channel_id = "UCwsrIijdkrYaQmjnKO2FKRQ"
    return {
        "channels": [{
            "channel_id": channel_id, "channel_name": "Example", "channel_url": "https://www.youtube.com/@example",
            "description": "", "thumbnail_url": None, "subscriber_count": None, "source_id": "youtube",
        }],
        "items": [{
            "video_id": f"video{i:06d}", "channel_id": channel_id, "title": f"Episode {i}",
            "published_at": "2026-11-01T00:00:00+00:00", "duration_seconds": 3600,
        } for i in range(count)],
        "discovery_complete": True,
    }


@pytest.fixture
def harness(monkeypatch):
    engine = create_engine("sqlite+pysqlite:///:memory:")
    metadata = MetaData()
    Table("channels", metadata,
          Column("channel_id", String, primary_key=True), Column("source_id", String),
          Column("channel_name", String), Column("channel_url", Text))
    jobs = Table("sync_jobs", metadata,
                 Column("job_id", String, primary_key=True), Column("job_type", String),
                 Column("media_ids", JSON), Column("checkpoint_data", JSON), Column("status", String),
                 Column("progress", Integer), Column("items_total", Integer), Column("items_processed", Integer),
                 Column("items_succeeded", Integer), Column("items_failed", Integer),
                 Column("last_checkpoint_at", DateTime), Column("completed_at", DateTime),
                 Column("error_message", Text), Column("budget_halt_reason", Text))
    metadata.create_all(engine)
    stored = {}
    calls = []
    settings = SimpleNamespace(youtube_api_key="youtube-secret", youtube_transcript_io_api_key="transcript-secret")

    def cached(**kwargs):
        return stored.get(kwargs["video_id_or_url"])

    def acquire(**kwargs):
        calls.append(kwargs["video_id_or_url"])
        assert kwargs["include_metadata"] is False
        with engine.connect() as connection:
            checkpoint = connection.execute(select(jobs.c.checkpoint_data)).scalar_one()
        item = next(item for item in checkpoint["items"] if item["video_id"] == kwargs["video_id_or_url"])
        assert item["status"] == "inflight"
        assert item["attempts_reserved"] == 1
        assert checkpoint["provider_requests_reserved"] >= len(calls)
        result = {"content_id": len(stored) + 1, "language": "en", "transcript_status": "acquired"}
        stored[kwargs["video_id_or_url"]] = result
        return result

    monkeypatch.setattr(collections, "find_reusable_transcript", cached)
    monkeypatch.setattr(collections, "_ingest_video", acquire)
    monkeypatch.setattr(collections, "PROVIDER_REQUEST_INTERVAL_SECONDS", 0)
    return SimpleNamespace(engine=engine, jobs=jobs, stored=stored, calls=calls, settings=settings, acquire=acquire)


def _checkpoint(harness, job_id):
    return collections._CollectionRepository(harness.engine).load(job_id)[0]


def test_cap_is_durable_and_cached_items_remain_available_after_cap(harness):
    harness.stored["video000002"] = {"content_id": 40, "language": "en"}
    job = collections.create_collection(harness.engine, _plan(), 1)
    result = collections.run_collection(harness.engine, job, harness.settings)
    assert harness.calls == ["video000000"]
    assert result["status"] == "budget_halted"
    assert result["provider_requests_reserved"] == 1
    assert result["acquired_count"] == result["reused_count"] == result["pending_count"] == 1
    assert result["per_channel"][0]["reused"] == 1
    assert collections.run_collection(harness.engine, job, harness.settings)["provider_requests_reserved"] == 1
    assert harness.calls == ["video000000"]


@pytest.mark.parametrize("cap", [-1, 6001, 1.5, True])
def test_invalid_request_cap_is_rejected_before_creating_job(harness, cap):
    with pytest.raises(ValueError):
        collections.create_collection(harness.engine, _plan(), cap)
    with harness.engine.connect() as connection:
        assert connection.execute(select(harness.jobs.c.job_id)).all() == []


def test_completed_run_repeats_no_provider_calls(harness):
    job = collections.create_collection(harness.engine, _plan(), 3)
    result = collections.run_collection(harness.engine, job, harness.settings)
    assert result["status"] == "completed"
    assert result["acquired_count"] == 3
    assert collections.run_collection(harness.engine, job, harness.settings)["status"] == "completed"
    assert harness.calls == ["video000000", "video000001", "video000002"]


def test_invocation_limit_resumes_only_remaining_items(harness):
    job = collections.create_collection(harness.engine, _plan(), 3)
    first = collections.run_collection(harness.engine, job, harness.settings, max_items=1)
    assert first["status"] == "paused"
    assert first["pending_count"] == 2
    second = collections.run_collection(harness.engine, job, harness.settings)
    assert second["status"] == "completed"
    assert second["provider_requests_reserved"] == 3
    assert len(harness.calls) == 3


def test_unavailable_transcript_consumes_reservation_and_continues(harness, monkeypatch):
    def acquire(**kwargs):
        if kwargs["video_id_or_url"] == "video000000":
            harness.calls.append(kwargs["video_id_or_url"])
            try:
                raise TranscriptNotAvailableError("No matching transcript")
            except TranscriptNotAvailableError as exc:
                raise IngestionItemError("Failed to fetch transcript") from exc
        return harness.acquire(**kwargs)

    monkeypatch.setattr(collections, "_ingest_video", acquire)
    job = collections.create_collection(harness.engine, _plan(), 3)
    result = collections.run_collection(harness.engine, job, harness.settings)
    assert result["status"] == "completed"
    assert result["unavailable_count"] == 1
    assert result["acquired_count"] == 2
    assert result["provider_requests_reserved"] == 3


@pytest.mark.parametrize("code,reason", [(401, "provider_authentication"), (402, "provider_credit_limit"), (403, "provider_authentication"), (429, "provider_rate_limit")])
def test_provider_rejection_stops_without_retry(harness, monkeypatch, code, reason):
    def acquire(**kwargs):
        harness.calls.append(kwargs["video_id_or_url"])
        try:
            raise HTTPError("https://provider.example", code, "rejected", {}, None)
        except HTTPError as exc:
            raise IngestionItemError("provider rejected transcript-secret") from exc

    monkeypatch.setattr(collections, "_ingest_video", acquire)
    job = collections.create_collection(harness.engine, _plan(), 3)
    result = collections.run_collection(harness.engine, job, harness.settings)
    assert len(harness.calls) == 1
    assert result["status"] == "paused"
    assert result["halt_reason"] == reason
    assert result["failed_count"] == 1
    assert result["pending_count"] == 2
    assert result["provider_requests_reserved"] == 1
    assert "transcript-secret" not in _checkpoint(harness, job)["items"][0]["error"]


def test_unknown_network_failure_is_uncertain_and_not_automatically_retried(harness, monkeypatch):
    def acquire(**kwargs):
        harness.calls.append(kwargs["video_id_or_url"])
        raise TimeoutError("response timed out")

    monkeypatch.setattr(collections, "_ingest_video", acquire)
    job = collections.create_collection(harness.engine, _plan(1), 3)
    first = collections.run_collection(harness.engine, job, harness.settings)
    assert first["uncertain_count"] == 1
    assert first["provider_requests_reserved"] == 1
    second = collections.run_collection(harness.engine, job, harness.settings)
    assert second["uncertain_count"] == 1
    assert len(harness.calls) == 1


@pytest.mark.parametrize("persisted", [False, True])
def test_interrupted_request_reconciles_without_duplicate_request(harness, monkeypatch, persisted):
    def crash(**kwargs):
        harness.calls.append(kwargs["video_id_or_url"])
        if persisted:
            harness.stored[kwargs["video_id_or_url"]] = {"content_id": 17, "language": "en"}
        raise KeyboardInterrupt()

    monkeypatch.setattr(collections, "_ingest_video", crash)
    job = collections.create_collection(harness.engine, _plan(1), 3)
    with pytest.raises(KeyboardInterrupt):
        collections.run_collection(harness.engine, job, harness.settings)
    assert _checkpoint(harness, job)["items"][0]["status"] == "inflight"
    result = collections.run_collection(harness.engine, job, harness.settings)
    assert len(harness.calls) == 1
    assert result["provider_requests_reserved"] == 1
    assert result["status"] == ("completed" if persisted else "paused")
    assert result["acquired_count"] == int(persisted)
    assert result["uncertain_count"] == int(not persisted)


def test_reservation_write_failure_prevents_external_call(harness, monkeypatch):
    original_save = collections._CollectionRepository.save

    def fail_reservation(self, job_id, checkpoint, status):
        if any(item["status"] == "inflight" for item in checkpoint["items"]):
            raise RuntimeError("database unavailable")
        return original_save(self, job_id, checkpoint, status)

    monkeypatch.setattr(collections._CollectionRepository, "save", fail_reservation)
    job = collections.create_collection(harness.engine, _plan(), 3)
    with pytest.raises(RuntimeError, match="database unavailable"):
        collections.run_collection(harness.engine, job, harness.settings)
    assert harness.calls == []
    assert collections.collection_status(harness.engine, job)["provider_requests_reserved"] == 0


def test_other_worker_hosted_index_manifest_is_preserved(harness):
    job = collections.create_collection(harness.engine, _plan(), 3)
    repository = collections._CollectionRepository(harness.engine)
    old, _ = repository.load(job)
    old["hosted_index"] = {"version": "stale"}
    newer, _ = repository.load(job)
    newer["hosted_index"] = {"version": "current"}
    with harness.engine.begin() as connection:
        connection.execute(update(harness.jobs).where(harness.jobs.c.job_id == job).values(checkpoint_data=newer))
    repository.save(job, old, "paused")
    assert _checkpoint(harness, job)["hosted_index"] == {"version": "current"}


def test_postgres_checkpoint_merge_binds_json_object_not_encoded_string():
    jobs = Table("sync_jobs", MetaData(), Column("job_id", String), Column("checkpoint_data", JSONB), Column("status", String))
    seen = []

    class Connection:
        def execute(self, statement):
            compiled = statement.compile(dialect=psycopg.dialect())
            assert "|| CAST(" in str(compiled)
            payloads = [value for value in compiled.params.values() if isinstance(value, dict)]
            assert len(payloads) == 1
            assert payloads[0]["collection_version"] == 1
            assert "hosted_index" not in payloads[0]
            for key, value in compiled.params.items():
                if value is payloads[0]:
                    # Psycopg receives a Jsonb wrapper around the dict, not around
                    # an already JSON-encoded string (which would corrupt || merge).
                    assert isinstance(compiled._bind_processors[key](value).obj, dict)
            seen.append(True)
            return SimpleNamespace(rowcount=1)

    @contextmanager
    def begin():
        yield Connection()

    repository = object.__new__(collections._CollectionRepository)
    repository.engine = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"), begin=begin)
    repository.jobs = jobs
    repository.save("job", {"collection_version": 1, "items": [], "hosted_index": {"stale": True}}, "paused")
    assert seen == [True]


def test_race_with_another_cache_writer_releases_unused_reservation(harness, monkeypatch):
    monkeypatch.setattr(collections, "_ingest_video", lambda **kwargs: {"content_id": 20, "language": "en", "transcript_status": "reused"})
    job = collections.create_collection(harness.engine, _plan(1), 1)
    result = collections.run_collection(harness.engine, job, harness.settings)
    assert result["reused_count"] == 1
    assert result["provider_requests_reserved"] == 0
    assert _checkpoint(harness, job)["items"][0]["attempts_reserved"] == 0


def test_requests_are_paced_three_seconds_apart(harness, monkeypatch):
    clock = [0.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(collections, "PROVIDER_REQUEST_INTERVAL_SECONDS", 3.0)
    monkeypatch.setattr(collections.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(collections.time, "sleep", sleep)
    job = collections.create_collection(harness.engine, _plan(3), 3)
    collections.run_collection(harness.engine, job, harness.settings)
    assert sleeps == [3.0, 3.0]


@pytest.mark.parametrize("acquired", [False, True])
def test_postgres_advisory_lock_prevents_overlap_and_releases(acquired):
    statements = []
    rollbacks = []

    class Connection:
        def execute(self, statement, params=None):
            statements.append(str(statement))
            return SimpleNamespace(scalar_one=lambda: acquired)

        def rollback(self):
            rollbacks.append(True)

    @contextmanager
    def connect():
        yield Connection()

    engine = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"), connect=connect)
    if acquired:
        with pytest.raises(RuntimeError):
            with collections._collection_lock(engine, "job_test") as check_lock:
                check_lock()
                raise RuntimeError("worker stopped")
        assert statements == ["SELECT pg_try_advisory_xact_lock(:key)", "SELECT 1"]
        assert rollbacks == [True]
    else:
        with pytest.raises(collections.CollectionBusyError):
            with collections._collection_lock(engine, "job_test"):
                pytest.fail("A busy lock cannot run work.")
        assert len(statements) == 1
        assert rollbacks == [True]


def test_lost_advisory_lock_stops_before_provider_reservation(harness, monkeypatch):
    @contextmanager
    def lost_lock(engine, job_id):
        def check():
            raise RuntimeError("lock connection lost")
        yield check

    monkeypatch.setattr(collections, "_collection_lock", lost_lock)
    job = collections.create_collection(harness.engine, _plan(), 3)
    with pytest.raises(RuntimeError, match="lock connection lost"):
        collections.run_collection(harness.engine, job, harness.settings)
    assert harness.calls == []
    assert collections.collection_status(harness.engine, job)["provider_requests_reserved"] == 0

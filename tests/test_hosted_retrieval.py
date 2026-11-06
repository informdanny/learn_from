from __future__ import annotations

import copy
import io
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest
from sqlalchemy import JSON, Column, Integer, MetaData, String, Table, Text, create_engine, select, update

from learn_from import hosted_retrieval as hosted


CHANNEL = "UCwsrIijdkrYaQmjnKO2FKRQ"
OTHER_CHANNEL = "UC" + "a" * 22
JOB = "job_test"


@pytest.fixture
def harness(monkeypatch):
    engine = create_engine("sqlite+pysqlite:///:memory:")
    metadata = MetaData()
    channels = Table("channels", metadata, Column("channel_id", String, primary_key=True), Column("channel_name", String))
    media = Table("media", metadata, Column("media_id", String, primary_key=True), Column("channel_id", String),
                  Column("title", Text), Column("publish_date", String))
    content = Table("doc_content", metadata, Column("content_id", Integer, primary_key=True), Column("media_id", String),
                    Column("full_text", Text), Column("timestamped_segments", JSON), Column("chunk_type", String), Column("language", String))
    jobs = Table("sync_jobs", metadata, Column("job_id", String, primary_key=True), Column("checkpoint_data", JSON))
    metadata.create_all(engine)
    items = [{"video_id": f"video{i:06d}", "channel_id": CHANNEL, "content_id": i + 1,
              "status": "acquired" if i < 2 else "pending"} for i in range(3)]
    documents = [{"media_id": item["video_id"], "channel_id": CHANNEL, "channel_name": "Example channel",
                  "content_id": item["content_id"], "title": f"Episode {i}", "publish_date": "2026-11-01",
                  "full_text": "First statement.\nImportant closing qualification.", "language": "en",
                  "timestamped_segments": [{"text": "First statement.", "start_seconds": 0},
                                           {"text": "Important closing qualification.", "start_seconds": 3600.5}]}
                 for i, item in enumerate(items)]
    with engine.begin() as connection:
        connection.execute(channels.insert(), {"channel_id": CHANNEL, "channel_name": "Example channel"})
        for doc in documents:
            connection.execute(media.insert(), {key: doc[key] for key in ("media_id", "channel_id", "title", "publish_date")})
            connection.execute(content.insert(), {**{key: doc[key] for key in ("content_id", "media_id", "full_text", "timestamped_segments", "language")}, "chunk_type": "complete"})
        connection.execute(jobs.insert(), {"job_id": JOB, "checkpoint_data": {"collection_version": 1, "items": items, "provider_requests_reserved": 2}})
    calls, uploads, attachments = [], [], {}
    file_states = {}
    usage = [0]

    def checkpoint():
        with engine.connect() as connection:
            return connection.execute(select(jobs.c.checkpoint_data)).scalar_one()

    def request(method, path, key, **kwargs):
        calls.append((method, path, copy.deepcopy(kwargs)))
        if method == "POST" and path == "/vector_stores":
            assert "expires_after" not in kwargs["payload"]
            assert checkpoint()["hosted_index"]["store_state"] == "creating"
            return {"id": "vs_test", "status": "completed", "usage_bytes": usage[0]}
        if method == "GET" and path == "/vector_stores/vs_test":
            return {"id": "vs_test", "status": "completed", "usage_bytes": usage[0]}
        if method == "POST" and path == "/vector_stores/vs_test/files":
            payload = kwargs["payload"]
            entry = checkpoint()["hosted_index"]["files"][payload["attributes"]["video_id"]]
            assert entry["file_id"] == payload["file_id"]  # Upload ID durably saved before attachment.
            assert entry["status"] == "attaching"
            attachments[payload["file_id"]] = payload
            usage[0] += 100
            return {"id": payload["file_id"], "status": "in_progress"}
        if method == "GET" and path.startswith("/vector_stores/vs_test/files/"):
            file_id = path.rsplit("/", 1)[1]
            return {"id": file_id, "vector_store_id": "vs_test", "status": file_states.get(file_id, "completed"), "usage_bytes": 100}
        raise AssertionError((method, path, kwargs))

    def upload(key, filename, document):
        assert checkpoint()["hosted_index"]["files"][documents[len(uploads)]["media_id"]]["status"] == "uploading"
        uploads.append((filename, document))
        return {"id": f"file_{len(uploads)}", "bytes": len(document)}

    monkeypatch.setattr(hosted, "_request", request)
    monkeypatch.setattr(hosted, "_upload", upload)
    return SimpleNamespace(engine=engine, jobs=jobs, content=content, documents=documents,
                           checkpoint=checkpoint, request=request, upload=upload, calls=calls,
                           uploads=uploads, attachments=attachments, usage=usage, file_states=file_states)


def sync(harness, **kwargs):
    return hosted.sync_collection_index(engine=harness.engine, job_id=JOB, api_key="test-key", poll_interval_seconds=0, **kwargs)


def test_saved_items_only_and_repeated_run_reuses_index(harness):
    first = sync(harness)
    assert first["status"] == "completed"
    assert first["completed_count"] == first["eligible_count"] == 2
    assert first["usage_bytes"] == 200
    assert len(harness.uploads) == len(harness.attachments) == 2
    assert "video000002" not in first["files"]  # Saved row alone does not enroll a pending item.
    before = len(harness.calls)
    second = sync(harness)
    assert second["files"] == first["files"]
    assert len(harness.uploads) == 2
    assert [(method, path) for method, path, _ in harness.calls[before:]] == [("GET", "/vector_stores/vs_test")]
    assert harness.checkpoint()["provider_requests_reserved"] == 2
    assert harness.checkpoint()["items"][2]["status"] == "pending"
    with harness.engine.connect() as connection:
        assert connection.execute(select(harness.content.c.full_text)).scalars().all() == [doc["full_text"] for doc in harness.documents]


def test_timestamped_document_preserves_end_and_source_attribution(harness):
    rendered, attrs = hosted.source_document(harness.documents[0])
    assert b"[t=3600.500s] Important closing qualification." in rendered
    assert b"guests may express their own views" in rendered
    assert b"https://www.youtube.com/watch?v=video000000" in rendered
    assert attrs["channel_id"] == CHANNEL
    assert len(attrs["source_hash"]) == 64
    assert hosted.source_document(harness.documents[0]) == (rendered, attrs)


def test_unaligned_segments_never_drop_text_or_invent_timing(harness):
    doc = copy.deepcopy(harness.documents[0])
    doc["full_text"] += "\nAn untimed additional sentence."
    rendered, _ = hosted.source_document(doc)
    assert doc["full_text"].encode() in rendered
    assert b"not aligned; do not infer timestamps" in rendered
    assert b"[t=" not in rendered


def test_file_limit_resumes_without_uploading_completed_files(harness):
    first = sync(harness, max_files=1)
    assert first["status"] == "file_limit_reached"
    assert len(harness.uploads) == 1
    second = sync(harness, max_files=1)
    assert second["status"] == "completed"
    assert len(harness.uploads) == 2


def test_storage_threshold_stops_after_measured_file_not_claiming_hard_cap(harness):
    result = sync(harness, max_index_bytes=90)
    assert result["status"] == "storage_limit_reached"
    assert result["usage_bytes"] == 100
    assert len(harness.uploads) == 1
    sync(harness, max_index_bytes=90)
    assert len(harness.uploads) == 1


def test_pending_indexing_stops_next_file_and_can_resume(harness):
    harness.file_states["file_1"] = "in_progress"
    first = sync(harness, poll_attempts=2)
    assert first["status"] == "indexing_pending"
    assert len(harness.uploads) == 1
    assert first["completed_count"] == 0
    harness.file_states["file_1"] = "completed"
    assert sync(harness)["completed_count"] == 2
    assert len(harness.uploads) == 2


def test_source_changes_pause_instead_of_indexing_stale_and_new_versions(harness):
    sync(harness)
    with harness.engine.begin() as connection:
        connection.execute(update(harness.content).where(harness.content.c.content_id == 1).values(full_text="Changed source"))
    result = sync(harness)
    assert result["status"] == "source_changed"
    assert len(harness.uploads) == 2


def test_uncertain_upload_recovered_by_filename_and_bytes_without_reupload(harness, monkeypatch):
    orphan = {}
    def interrupted(key, filename, document):
        orphan.update(filename=filename, document=document)
        raise hosted.HostedAPIError("timeout")
    monkeypatch.setattr(hosted, "_upload", interrupted)
    with pytest.raises(hosted.HostedAPIError):
        sync(harness, max_files=1)
    assert harness.checkpoint()["hosted_index"]["files"]["video000000"]["status"] == "upload_unknown"
    def recovered(method, path, key, **kwargs):
        if path.startswith("/files?"):
            return {"data": [{"id": "file_orphan", "filename": orphan["filename"]}], "has_more": False}
        if path == "/files/file_orphan/content":
            return orphan["document"]
        return harness.request(method, path, key, **kwargs)
    monkeypatch.setattr(hosted, "_request", recovered)
    result = sync(harness, max_files=1)
    assert result["files"]["video000000"]["file_id"] == "file_orphan"
    assert result["files"]["video000000"]["status"] == "completed"
    assert len(harness.uploads) == 0


def test_unresolved_upload_never_blindly_retries(harness, monkeypatch):
    attempts = []
    def interrupted(*args):
        attempts.append(1)
        raise hosted.HostedAPIError("timeout")
    monkeypatch.setattr(hosted, "_upload", interrupted)
    with pytest.raises(hosted.HostedAPIError):
        sync(harness)
    def empty(method, path, key, **kwargs):
        if path.startswith("/files?"):
            return {"data": [], "has_more": False}
        return harness.request(method, path, key, **kwargs)
    monkeypatch.setattr(hosted, "_request", empty)
    assert sync(harness)["status"] == "reconciliation_required"
    assert len(attempts) == 1


def test_attachment_timeout_recovers_by_get_without_second_post(harness, monkeypatch):
    def interrupted(method, path, key, **kwargs):
        result = harness.request(method, path, key, **kwargs)
        if method == "POST" and path.endswith("/files"):
            raise hosted.HostedAPIError("timeout")
        return result
    monkeypatch.setattr(hosted, "_request", interrupted)
    with pytest.raises(hosted.HostedAPIError):
        sync(harness)
    monkeypatch.setattr(hosted, "_request", harness.request)
    result = sync(harness, max_files=1)
    assert result["files"]["video000000"]["status"] == "completed"
    assert len(harness.uploads) == 1
    assert sum(method == "POST" and path.endswith("/files") for method, path, _ in harness.calls) == 1


def test_uncertain_store_creation_does_not_create_another(harness, monkeypatch):
    attempts = []
    def fail(*args, **kwargs):
        attempts.append(1)
        raise hosted.HostedAPIError("timeout")
    monkeypatch.setattr(hosted, "_request", fail)
    with pytest.raises(hosted.HostedAPIError):
        sync(harness)
    with pytest.raises(hosted.HostedRetrievalError, match="no second store"):
        sync(harness)
    assert len(attempts) == 1


def test_recovered_upload_must_match_original_bytes(monkeypatch):
    def request(method, path, key, **kwargs):
        return b"wrong" if path.endswith("/content") else {"data": [{"id": "file_found", "filename": "expected.txt"}], "has_more": False}
    monkeypatch.setattr(hosted, "_request", request)
    with pytest.raises(hosted.HostedRetrievalError, match="does not match"):
        hosted._recover_upload("key", "expected.txt", b"original")


def test_hosted_search_filters_channels_and_uses_provider_ranking(monkeypatch):
    calls = []
    def request(method, path, key, **kwargs):
        calls.append((method, path, kwargs))
        return {"data": [{"file_id": "file_1", "attributes": {"channel_id": CHANNEL}, "content": [{"type": "text", "text": "source"}]}], "has_more": False}
    monkeypatch.setattr(hosted, "_request", request)
    result = hosted.hosted_search(api_key="key", vector_store_id="vs_test", query="customer acquisition", channel_ids=[CHANNEL], max_results=5)
    assert len(result["data"]) == 1
    assert calls[0][2]["payload"] == {"query": "customer acquisition", "filters": {"type": "in", "key": "channel_id", "value": [CHANNEL]}, "max_num_results": 5}


def test_search_fails_closed_on_wrong_channel(monkeypatch):
    monkeypatch.setattr(hosted, "_request", lambda *args, **kwargs: {"data": [{"file_id": "file_1", "attributes": {"channel_id": OTHER_CHANNEL}}]})
    with pytest.raises(hosted.HostedRetrievalError, match="out-of-scope"):
        hosted.hosted_search(api_key="key", vector_store_id="vs_test", query="offers", channel_ids=[CHANNEL])


def test_http_error_does_not_expose_credentials_or_provider_body(monkeypatch):
    def fail(*args, **kwargs):
        raise HTTPError("https://api.openai.com/v1/files", 401, "unauthorized", {}, io.BytesIO(b"secret body"))
    monkeypatch.setattr(hosted, "urlopen", fail)
    with pytest.raises(hosted.HostedAPIError) as error:
        hosted._request("POST", "/files", "private-key", payload={})
    assert str(error.value) == "OpenAI returned HTTP 401; no automatic retry was made."
    assert error.value.ambiguous is False


def test_upload_builds_complete_multipart_without_sdk(monkeypatch):
    captured = []
    monkeypatch.setattr(hosted, "_request", lambda *args, **kwargs: captured.append((args, kwargs)) or {"id": "file_1"})
    assert hosted._upload("key", "source.txt", b"start\nend")["id"] == "file_1"
    body = captured[0][1]["body"]
    assert b'name="purpose"\r\n\r\nassistants' in body
    assert b'filename="source.txt"' in body
    assert b"start\nend" in body
    assert captured[0][1]["content_type"].startswith("multipart/form-data; boundary=")


@pytest.mark.parametrize("kwargs", [{"max_files": 0}, {"max_files": True}, {"max_index_bytes": 0}, {"poll_attempts": 0}, {"poll_attempts": 1.5}, {"poll_attempts": True}])
def test_invalid_limits_do_not_call_provider(harness, kwargs):
    with pytest.raises(ValueError):
        sync(harness, **kwargs)
    assert harness.calls == []


def test_lost_collection_lock_stops_before_provider_mutation(harness, monkeypatch):
    from contextlib import contextmanager

    @contextmanager
    def lost_lock(*args):
        def check():
            raise RuntimeError("pinned connection lost")
        yield check

    monkeypatch.setattr(hosted, "_collection_lock", lost_lock)
    with pytest.raises(RuntimeError, match="pinned connection lost"):
        sync(harness)
    assert harness.calls == []

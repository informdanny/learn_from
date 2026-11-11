from __future__ import annotations

import copy
import hashlib
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from learn_from import hosted_retrieval as hosted


CHANNEL = "UC" + "a" * 22
OTHER_CHANNEL = "UC" + "b" * 22
VIDEO = "video000001"


@pytest.fixture
def synced_source(monkeypatch):
    document = {"media_id": VIDEO, "content_id": 1, "channel_id": CHANNEL, "channel_name": "Example",
                "title": "A business", "publish_date": "2026-11-01", "language": "en",
                "full_text": "A local service business.", "timestamped_segments": [{"text": "A local service business.", "start_seconds": 0}]}
    _, base = hosted.source_document(document)
    checkpoint = {"items": [{"video_id": VIDEO, "channel_id": CHANNEL, "content_id": 1, "status": "acquired"}],
                  "provider_requests_reserved": 1, "halt_reason": "paused", "hosted_index": {"vector_store_id": "vs_test", "status": "reconciliation_required",
                    "files": {VIDEO: {"file_id": "file_test", "status": "completed", "content_id": 1, "source_hash": base["source_hash"]}}}}
    extracted = {"version": "index-1.0", "source_hash": hashlib.sha256(document["full_text"].encode()).hexdigest(),
                 "fields": [{"key": "business_model", "status": "trusted", "value": "local_service", "openai_key": "business_model", "openai_eligible": True}],
                 "openai_attributes": {"business_model": "local_service"}}
    observed = {"id": "file_test", "vector_store_id": "vs_test", "status": "completed", "attributes": dict(base)}
    calls = []
    controls = {"fail_after_update": False, "bad_readback": False}

    class Repo:
        def load(self, job_id):
            return copy.deepcopy(checkpoint)

        def document(self, item):
            return copy.deepcopy(document)

        def save(self, job_id, manifest):
            checkpoint["hosted_index"] = copy.deepcopy(manifest)

    @contextmanager
    def lock(engine, job_id):
        yield lambda: None

    def request(method, path, api_key, **kwargs):
        calls.append((method, path, copy.deepcopy(kwargs)))
        assert path == "/vector_stores/vs_test/files/file_test", "Must use existing attachment, never upload or create a store."
        if method == "POST":
            assert checkpoint["hosted_index"]["files"][VIDEO]["metadata"]["status"] == "updating"
            assert set(kwargs["payload"]) == {"attributes"}
            observed["attributes"] = copy.deepcopy(kwargs["payload"]["attributes"])
            if controls["fail_after_update"]:
                raise hosted.HostedRetrievalError("Unknown update response.")
        result = copy.deepcopy(observed)
        if method == "GET" and controls["bad_readback"] and any(call[0] == "POST" for call in calls):
            result["status"] = "in_progress"
        return result

    monkeypatch.setattr(hosted, "_IndexRepository", lambda engine: Repo())
    monkeypatch.setattr(hosted, "_collection_lock", lock)
    monkeypatch.setattr(hosted, "_latest_index_metadata", lambda engine, content_id: (42, extracted))
    monkeypatch.setattr(hosted, "_request", request)
    monkeypatch.setattr(hosted, "_upload", lambda *a, **k: pytest.fail("Metadata sync must never upload."))
    return SimpleNamespace(document=document, checkpoint=checkpoint, extracted=extracted, observed=observed, calls=calls, controls=controls)


def _sync(**kwargs):
    return hosted.sync_collection_metadata(engine=object(), job_id="job_test", api_key="test-key", video_ids=[VIDEO], **kwargs)


def test_attribute_update_uses_existing_file_and_preserves_stopped_collection(synced_source):
    h = synced_source
    outcome = _sync()
    assert [call[0] for call in h.calls] == ["GET", "POST", "GET"]
    assert outcome["results"][0]["status"] == "synced"
    assert h.checkpoint["provider_requests_reserved"] == 1
    assert h.checkpoint["halt_reason"] == "paused"
    assert h.checkpoint["hosted_index"]["status"] == "reconciliation_required"
    assert h.observed["attributes"]["channel_id"] == CHANNEL
    assert h.observed["attributes"]["metadata_run_id"] == "42"
    assert len(h.observed["attributes"]) == 9
    h.calls.clear()
    assert _sync()["results"][0]["status"] == "unchanged"
    assert [call[0] for call in h.calls] == ["GET"]


def test_uncertain_update_reconciles_with_get_without_repeating_success(synced_source):
    h = synced_source
    h.controls["fail_after_update"] = True
    with pytest.raises(hosted.HostedRetrievalError):
        _sync()
    assert h.checkpoint["hosted_index"]["files"][VIDEO]["metadata"]["status"] == "reconciliation_required"
    h.controls["fail_after_update"] = False
    h.calls.clear()
    assert _sync()["results"][0]["status"] == "unchanged"
    assert [call[0] for call in h.calls] == ["GET"]


@pytest.mark.parametrize("change", ["extraction_hash", "upload_hash", "content_id"])
def test_stale_source_or_evidence_stops_before_provider_calls(synced_source, change):
    h = synced_source
    if change == "extraction_hash":
        h.extracted["source_hash"] = "0" * 64
    elif change == "upload_hash":
        h.checkpoint["hosted_index"]["files"][VIDEO]["source_hash"] = "0" * 64
    else:
        h.checkpoint["hosted_index"]["files"][VIDEO]["content_id"] = 999
    with pytest.raises(hosted.HostedRetrievalError):
        _sync()
    assert h.calls == []


@pytest.mark.parametrize("change", ["provisional", "conflicting", "unsupported"])
def test_projection_must_match_trusted_unambiguous_fields(synced_source, change):
    h = synced_source
    if change == "provisional":
        h.extracted["fields"][0]["status"] = "provisional"
    elif change == "conflicting":
        h.extracted["fields"].append(h.extracted["fields"][0] | {"value": "subscription"})
    else:
        h.extracted["openai_attributes"]["channel_id"] = OTHER_CHANNEL
    with pytest.raises(hosted.HostedRetrievalError):
        _sync()
    assert h.calls == []


def test_readback_requires_completed_status(synced_source):
    h = synced_source
    h.controls["bad_readback"] = True
    with pytest.raises(hosted.HostedRetrievalError, match="readback differs"):
        _sync()
    assert h.checkpoint["hosted_index"]["files"][VIDEO]["metadata"]["status"] == "reconciliation_required"


def test_eventual_attribute_visibility_retries_reads_not_update(synced_source, monkeypatch):
    h = synced_source
    original = hosted._request
    stale = copy.deepcopy(h.observed)
    remaining = [2]
    monkeypatch.setattr(hosted.time, "sleep", lambda seconds: None)

    def delayed(method, path, key, **kwargs):
        result = original(method, path, key, **kwargs)
        if method == "GET" and any(call[0] == "POST" for call in h.calls) and remaining[0]:
            remaining[0] -= 1
            return stale
        return result

    monkeypatch.setattr(hosted, "_request", delayed)
    assert _sync()["results"][0]["status"] == "synced"
    assert [call[0] for call in h.calls] == ["GET", "POST", "GET", "GET", "GET"]


def test_pending_attachment_is_skipped_without_calls(synced_source):
    h = synced_source
    h.checkpoint["hosted_index"]["files"][VIDEO]["status"] = "in_progress"
    assert _sync()["results"][0]["status"] == "skipped"
    assert h.calls == []


def test_search_combines_channel_and_metadata_filters_and_checks_returned_scope(monkeypatch):
    calls = []
    row = {"file_id": "file_test", "attributes": {"channel_id": CHANNEL, "industry": "home_services"}}
    def request(method, path, key, **kwargs):
        calls.append(kwargs["payload"])
        return {"data": [row]}
    monkeypatch.setattr(hosted, "_request", request)
    hosted.hosted_search(api_key="test", vector_store_id="vs_test", query="Finding customers", channel_ids=[CHANNEL], metadata_filters={"industry": "home_services"}, rewrite_query=True)
    assert calls[0]["filters"] == {"type": "and", "filters": [{"type": "in", "key": "channel_id", "value": [CHANNEL]}, {"type": "eq", "key": "industry", "value": "home_services"}]}
    assert calls[0]["rewrite_query"] is True
    row["attributes"]["industry"] = "software"
    with pytest.raises(hosted.HostedRetrievalError, match="outside the requested metadata"):
        hosted.hosted_search(api_key="test", vector_store_id="vs_test", query="Customers", channel_ids=[CHANNEL], metadata_filters={"industry": "home_services"})


@pytest.mark.parametrize("filters", [{"channel_id": CHANNEL}, {"industry": "Home services"}, {"business_model": "service,subscription"}, {"business_stage": "invented"}, {"industry": ["software"]}, {"industry": ""}])
def test_invalid_filters_make_no_provider_request(monkeypatch, filters):
    monkeypatch.setattr(hosted, "_request", lambda *a, **k: pytest.fail("Invalid filters must fail before provider call."))
    with pytest.raises(ValueError):
        hosted.hosted_search(api_key="test", vector_store_id="vs_test", query="Customers", channel_ids=[CHANNEL], metadata_filters=filters)


def test_video_cohort_uses_managed_search_with_channel_scope_and_no_scalar_facets(monkeypatch):
    calls = []
    row = {"file_id": "file_test", "attributes": {"channel_id": CHANNEL, "video_id": VIDEO}}
    def request(method, path, key, **kwargs):
        calls.append(kwargs["payload"])
        return {"data": [row]}
    monkeypatch.setattr(hosted, "_request", request)
    hosted.hosted_search(api_key="test", vector_store_id="vs_test", query="Finding customers", channel_ids=[CHANNEL], video_ids=[VIDEO])
    assert calls[0]["filters"] == {"type": "and", "filters": [
        {"type": "in", "key": "channel_id", "value": [CHANNEL]},
        {"type": "in", "key": "video_id", "value": [VIDEO]},
    ]}
    row["attributes"]["video_id"] = "video000002"
    with pytest.raises(hosted.HostedRetrievalError, match="outside the selected"):
        hosted.hosted_search(api_key="test", vector_store_id="vs_test", query="Customers", channel_ids=[CHANNEL], video_ids=[VIDEO])


def test_empty_video_selection_does_not_spend_on_search(monkeypatch):
    monkeypatch.setattr(hosted, "_request", lambda *a, **k: pytest.fail("Empty cohort must not call OpenAI."))
    result = hosted.hosted_search(api_key="test", vector_store_id="vs_test", query="Customers", channel_ids=[CHANNEL], video_ids=[])
    assert result["data"] == []
    assert result["search_performed"] is False


@pytest.mark.parametrize("videos", ["video000001", ["invalid"], [f"video{i:06d}" for i in range(1001)]])
def test_video_selection_validation_precedes_request(monkeypatch, videos):
    monkeypatch.setattr(hosted, "_request", lambda *a, **k: pytest.fail("Invalid cohort must not call OpenAI."))
    with pytest.raises(ValueError):
        hosted.hosted_search(api_key="test", vector_store_id="vs_test", query="Customers", channel_ids=[CHANNEL], video_ids=videos)

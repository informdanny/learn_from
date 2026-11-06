from __future__ import annotations

from types import SimpleNamespace

import pytest

from learn_from.worker import run_worker


@pytest.fixture
def worker_calls(monkeypatch):
    calls = []
    index_results = []
    monkeypatch.setenv("LEARN_FROM_COLLECTION_JOB_ID", "job_test")
    monkeypatch.delenv("LEARN_FROM_INDEX_BEFORE_COLLECTION", raising=False)
    monkeypatch.delenv("LEARN_FROM_INDEX_AFTER_COLLECTION", raising=False)
    monkeypatch.delenv("LEARN_FROM_MAX_INDEX_FILES", raising=False)

    def collect(**kwargs):
        calls.append(("collect", kwargs))
        return {"status": "completed", "acquired_count": 2, "reused_count": 0}

    def index(**kwargs):
        calls.append(("index", kwargs))
        return {"status": index_results.pop(0) if index_results else "completed"}

    monkeypatch.setattr("learn_from.collections.run_collection", collect)
    monkeypatch.setattr("learn_from.hosted_retrieval.sync_collection_index", index)
    return calls, index_results, SimpleNamespace(openai_api_key="test-key"), object()


def test_default_worker_still_collects_then_indexes(worker_calls):
    calls, _, settings, engine = worker_calls
    assert run_worker(settings=settings, engine=engine) == 0
    assert [call[0] for call in calls] == ["collect", "index"]


def test_opt_in_indexes_saved_sources_before_collection_and_new_sources_after(monkeypatch, worker_calls):
    calls, _, settings, engine = worker_calls
    monkeypatch.setenv("LEARN_FROM_INDEX_BEFORE_COLLECTION", "true")
    monkeypatch.setenv("LEARN_FROM_MAX_INDEX_FILES", "5")

    assert run_worker(settings=settings, engine=engine) == 0

    assert [call[0] for call in calls] == ["index", "collect", "index"]
    for _, kwargs in (calls[0], calls[2]):
        assert kwargs["max_files"] == 5
        assert kwargs["api_key"] == "test-key"
        assert kwargs["job_id"] == "job_test"


@pytest.mark.parametrize("status", ["indexing_pending", "reconciliation_required", "file_failed", "unexpected"])
def test_incomplete_first_index_stops_before_acquisition(monkeypatch, worker_calls, status):
    calls, index_results, settings, engine = worker_calls
    monkeypatch.setenv("LEARN_FROM_INDEX_BEFORE_COLLECTION", "true")
    index_results.append(status)

    assert run_worker(settings=settings, engine=engine) == 1
    assert [call[0] for call in calls] == ["index"]


@pytest.mark.parametrize("status", ["completed", "file_limit_reached", "storage_limit_reached", "no_saved_transcripts"])
def test_first_index_can_continue_without_enabling_after_index(monkeypatch, worker_calls, status):
    calls, index_results, settings, engine = worker_calls
    monkeypatch.setenv("LEARN_FROM_INDEX_BEFORE_COLLECTION", "true")
    monkeypatch.setenv("LEARN_FROM_INDEX_AFTER_COLLECTION", "false")
    index_results.append(status)

    assert run_worker(settings=settings, engine=engine) == 0
    assert [call[0] for call in calls] == ["index", "collect"]


@pytest.mark.parametrize("variable,value", [
    ("LEARN_FROM_INDEX_BEFORE_COLLECTION", "yes"),
    ("LEARN_FROM_INDEX_AFTER_COLLECTION", "no"),
    ("LEARN_FROM_MAX_INDEX_FILES", "0"),
    ("LEARN_FROM_MAX_INDEX_FILES", "10001"),
    ("LEARN_FROM_MAX_INDEX_FILES", "not-a-number"),
])
def test_settings_are_validated_before_any_api_or_collection(monkeypatch, worker_calls, variable, value):
    calls, _, settings, engine = worker_calls
    monkeypatch.setenv("LEARN_FROM_INDEX_BEFORE_COLLECTION", "true")
    monkeypatch.setenv("LEARN_FROM_INDEX_AFTER_COLLECTION", "false")
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValueError, match=variable):
        run_worker(settings=settings, engine=engine)
    assert calls == []

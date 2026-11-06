from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from learn_from import collections, consultation, hosted_retrieval, operations, worker


CHANNEL = "UC" + "a" * 22
VIDEO_A = "a" * 11
VIDEO_B = "b" * 11


def parser():
    value = argparse.ArgumentParser()
    operations.add_operation_parsers(value.add_subparsers(dest="command", required=True))
    return value


def consult_args(*extra):
    return parser().parse_args(["consult", "Challenge my service idea", "--channel", CHANNEL, "--job-id", "job-test", *extra])


@pytest.fixture
def runtime(monkeypatch):
    settings = SimpleNamespace(openai_api_key="secret-openai", youtube_api_key="secret-youtube",
                               youtube_transcript_io_api_key="secret-transcript", database_url="secret-database")
    engine = SimpleNamespace(dispose=Mock())
    monkeypatch.setattr(operations, "get_settings", lambda: settings)
    monkeypatch.setattr(operations, "create_db_engine", lambda _: engine)
    monkeypatch.setattr(operations, "_index_for_job", lambda *args: {"vector_store_id": "vs_test"})
    return settings, engine


@pytest.mark.parametrize("value", ["0", "-1", "11", "1.5"])
def test_document_limit_rejected_by_parser(value):
    with pytest.raises(SystemExit):
        consult_args("--max-documents", value)


@pytest.mark.parametrize("value", ["0", "-0.01", "0.11", "nan", "inf"])
def test_budget_rejected_by_parser(value):
    with pytest.raises(SystemExit):
        consult_args("--max-cost-usd", value)


def test_hosted_search_selects_unique_videos_then_reads_complete_documents(monkeypatch, runtime, capsys):
    search = Mock(return_value={"data": [{"attributes": {"video_id": video}} for video in (VIDEO_A, VIDEO_A, VIDEO_B, "c" * 11)]})
    consult = Mock(return_value={"status": "completed", "usage": {"input_tokens": 20}})
    monkeypatch.setattr(hosted_retrieval, "hosted_search", search)
    monkeypatch.setattr(consultation, "consult_published_views", consult)
    assert operations.run_operation(consult_args("--max-documents", "2")) == 0
    assert search.call_args.kwargs["max_results"] == 6
    assert search.call_args.kwargs["channel_ids"] == [CHANNEL]
    assert consult.call_args.kwargs["video_ids"] == [VIDEO_A, VIDEO_B]
    assert consult.call_args.kwargs["max_context_chars"] == 300000
    assert consult.call_args.kwargs["max_cost_usd"] == 0.10
    output = json.loads(capsys.readouterr().out)
    assert output["source_selection"]["matched_passages"] == 4
    runtime[1].dispose.assert_called_once()


def test_direct_video_mode_does_not_search(monkeypatch, runtime):
    monkeypatch.setattr(hosted_retrieval, "hosted_search", lambda **kwargs: pytest.fail("Direct selection must not search"))
    consult = Mock(return_value={"status": "completed"})
    monkeypatch.setattr(consultation, "consult_published_views", consult)
    args = parser().parse_args(["consult", "Challenge this", "--channel", CHANNEL, "--video", VIDEO_A])
    assert operations.run_operation(args) == 0
    assert consult.call_args.kwargs["video_ids"] == [VIDEO_A]


def test_no_evidence_writes_requested_output_and_skips_model(monkeypatch, runtime, tmp_path, capsys):
    monkeypatch.setattr(hosted_retrieval, "hosted_search", lambda **kwargs: {"data": []})
    monkeypatch.setattr(consultation, "consult_published_views", lambda **kwargs: pytest.fail("No evidence must not call model"))
    target = tmp_path / "new-directory" / "consultation.json"
    assert operations.run_operation(consult_args("--output", str(target))) == 0
    stored = json.loads(target.read_text())
    assert stored["status"] == "no_evidence"
    assert stored["estimated_cost_usd"] == 0
    assert stored["source_selection"]["selected_video_ids"] == []
    assert json.loads(capsys.readouterr().out)["output"] == str(target)


@pytest.mark.parametrize("field,value", [("max_documents", 0), ("max_documents", 11), ("max_cost_usd", float("nan"))])
def test_programmatic_invalid_limits_fail_before_hosted_search(monkeypatch, runtime, field, value):
    monkeypatch.setattr(hosted_retrieval, "hosted_search", lambda **kwargs: pytest.fail("Invalid limits must not search"))
    args = consult_args()
    setattr(args, field, value)
    assert operations.run_operation(args) == 1
    runtime[1].dispose.assert_called_once()


def test_output_directory_rejected_before_provider_call(monkeypatch, runtime, tmp_path):
    monkeypatch.setattr(hosted_retrieval, "hosted_search", lambda **kwargs: pytest.fail("Invalid output must not search"))
    assert operations.run_operation(consult_args("--output", str(tmp_path))) == 1


def test_errors_redact_configured_secrets(monkeypatch, runtime, capsys):
    monkeypatch.setattr(hosted_retrieval, "hosted_search", Mock(side_effect=RuntimeError("Bad secret-openai at secret-database")))
    assert operations.run_operation(consult_args()) == 1
    output = capsys.readouterr().out
    assert "secret-openai" not in output and "secret-database" not in output
    assert "[redacted]" in output


@pytest.mark.parametrize("status,exit_code", [("completed", 0), ("paused", 1), ("budget_halted", 0)])
def test_worker_skips_empty_index_and_reports_collection_outcome(monkeypatch, runtime, capsys, status, exit_code):
    monkeypatch.setenv("LEARN_FROM_COLLECTION_JOB_ID", "job-test")
    monkeypatch.setenv("LEARN_FROM_INDEX_AFTER_COLLECTION", "true")
    monkeypatch.setattr(collections, "run_collection", lambda **kwargs: {"status": status, "acquired_count": 0, "reused_count": 0})
    monkeypatch.setattr(hosted_retrieval, "sync_collection_index", lambda **kwargs: pytest.fail("An empty collection must not create an index"))
    assert worker.run_worker(settings=runtime[0], engine=runtime[1]) == exit_code
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[-1]["result"]["status"] == "skipped"


def test_worker_indexes_usable_partial_results_but_reports_failure(monkeypatch, runtime):
    monkeypatch.setenv("LEARN_FROM_COLLECTION_JOB_ID", "job-test")
    monkeypatch.setenv("LEARN_FROM_INDEX_AFTER_COLLECTION", "true")
    monkeypatch.setenv("LEARN_FROM_MAX_INDEX_FILES", "7")
    monkeypatch.setattr(collections, "run_collection", lambda **kwargs: {"status": "paused", "acquired_count": 1, "reused_count": 0})
    index = Mock(return_value={"status": "completed"})
    monkeypatch.setattr(hosted_retrieval, "sync_collection_index", index)
    assert worker.run_worker(settings=runtime[0], engine=runtime[1]) == 1
    assert index.call_args.kwargs["max_files"] == 7


@pytest.mark.parametrize("name,value", [("LEARN_FROM_INDEX_AFTER_COLLECTION", "maybe"), ("LEARN_FROM_MAX_INDEX_FILES", "0")])
def test_worker_invalid_configuration_fails_before_collection(monkeypatch, runtime, name, value):
    monkeypatch.setenv("LEARN_FROM_COLLECTION_JOB_ID", "job-test")
    monkeypatch.setenv("LEARN_FROM_INDEX_AFTER_COLLECTION", "true")
    monkeypatch.setenv(name, value)
    monkeypatch.setattr(collections, "run_collection", lambda **kwargs: pytest.fail("Invalid configuration must not collect"))
    with pytest.raises(ValueError):
        worker.run_worker(settings=runtime[0], engine=runtime[1])


def test_worker_can_skip_indexing(monkeypatch, runtime):
    monkeypatch.setenv("LEARN_FROM_COLLECTION_JOB_ID", "job-test")
    monkeypatch.setenv("LEARN_FROM_INDEX_AFTER_COLLECTION", "false")
    monkeypatch.setattr(collections, "run_collection", lambda **kwargs: {"status": "completed", "acquired_count": 1})
    monkeypatch.setattr(hosted_retrieval, "sync_collection_index", lambda **kwargs: pytest.fail("Indexing is disabled"))
    assert worker.run_worker(settings=runtime[0], engine=runtime[1]) == 0


@pytest.mark.parametrize("status,exit_code", [("completed", 0), ("file_limit_reached", 0), ("indexing_pending", 1), ("file_failed", 1)])
def test_worker_exit_reflects_hosted_index_outcome(monkeypatch, runtime, status, exit_code):
    monkeypatch.setenv("LEARN_FROM_COLLECTION_JOB_ID", "job-test")
    monkeypatch.setenv("LEARN_FROM_INDEX_AFTER_COLLECTION", "true")
    monkeypatch.setattr(collections, "run_collection", lambda **kwargs: {"status": "completed", "reused_count": 1})
    monkeypatch.setattr(hosted_retrieval, "sync_collection_index", lambda **kwargs: {"status": status})
    assert worker.run_worker(settings=runtime[0], engine=runtime[1]) == exit_code


@pytest.mark.parametrize("command", ["collection-index", "collection-worker"])
def test_index_progress_and_final_output_exclude_full_manifest(monkeypatch, runtime, capsys, command):
    manifest = {
        "status": "completed", "eligible_count": 1000, "completed_count": 1000,
        "usage_bytes": 2500000, "vector_store_id": "vs_test", "provider_status": "completed",
        "files": {f"video-{i}": {"status": "completed", "filename": "PRIVATE-FILE-DATA", "source_hash": "x" * 64} for i in range(1000)},
        "future_verbose_field": "PRIVATE-EXTRA-DATA",
    }
    def index(**kwargs):
        kwargs["on_progress"](manifest)
        return manifest
    monkeypatch.setattr(hosted_retrieval, "sync_collection_index", index)
    if command == "collection-worker":
        monkeypatch.setenv("LEARN_FROM_COLLECTION_JOB_ID", "job-test")
        monkeypatch.setenv("LEARN_FROM_INDEX_AFTER_COLLECTION", "true")
        monkeypatch.setattr(collections, "run_collection", lambda **kwargs: {"status": "completed", "acquired_count": 1000})
        assert worker.run_worker(settings=runtime[0], engine=runtime[1]) == 0
    else:
        assert operations.run_operation(parser().parse_args([command, "job-test"])) == 0
    output = capsys.readouterr().out
    assert len(output) < 1500
    assert '"files"' not in output and "PRIVATE-" not in output
    events = [json.loads(line) for line in output.splitlines()]
    summaries = [event.get("result", event) for event in events if event.get("stage") != "collection"]
    assert len(summaries) == 2
    for summary in summaries:
        assert summary["status"] == "completed"
        assert summary["file_count"] == summary["completed_count"] == summary["eligible_count"] == 1000
        assert summary["remaining_count"] == 0
        assert summary["usage_bytes"] == 2500000
        assert summary["vector_store_id"] == "vs_test"
    assert len(manifest["files"]) == 1000
    assert manifest["files"]["video-0"]["filename"] == "PRIVATE-FILE-DATA"


def test_index_summary_preserves_bounded_error_and_halt_information():
    manifest = {"status": "file_failed", "error": "x" * 4000, "halt_reason": "Provider rejected file.", "files": {}}
    result = operations._index_summary(manifest)
    assert result["status"] == "file_failed"
    assert len(result["error"]) == 1000
    assert result["halt_reason"] == "Provider rejected file."
    assert len(manifest["error"]) == 4000

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from learn_from import collections, consultation, hosted_retrieval, operations, worker


CHANNEL = "UC" + "a" * 22
CHANNEL_B = "UC" + "b" * 22
VIDEO_A = "a" * 11
VIDEO_B = "b" * 11
VIDEO_C = "c" * 11


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


@pytest.mark.parametrize("selection", ["direct", "hosted"])
def test_prepare_only_exports_complete_evidence_without_responses(monkeypatch, runtime, tmp_path, selection):
    transcript = "Beginning of source. " + "Substantive discussion. " * 700 + "End of source."
    monkeypatch.setattr(consultation, "_load_full_documents", lambda *args: [{
        "media_id": VIDEO_A, "channel_id": CHANNEL, "channel_name": "Example channel",
        "content_id": 1, "title": "A useful interview", "publish_date": "2026-11-01",
        "full_text": transcript, "timestamped_segments": [],
    }])
    monkeypatch.setattr(consultation, "_request_response", lambda *args: pytest.fail("Prepare mode must never request a model response"))
    if selection == "hosted":
        monkeypatch.setattr(hosted_retrieval, "hosted_search", lambda **kwargs: {"data": [{"attributes": {"video_id": VIDEO_A}}]})
        scope = ["--job-id", "job-test"]
    else:
        runtime[0].openai_api_key = None
        monkeypatch.setattr(hosted_retrieval, "hosted_search", lambda **kwargs: pytest.fail("Direct preparation must not search"))
        scope = ["--video", VIDEO_A]
    target = tmp_path / "prepared" / "evidence.json"
    args = parser().parse_args(["consult", "Challenge my service idea", "--channel", CHANNEL,
                               *scope, "--prepare-only", "--output", str(target)])
    assert operations.run_operation(args) == 0
    result = json.loads(target.read_text())
    assert result["status"] == "prepared"
    assert result["model_request_made"] is False
    assert result["usage"] is None and result["estimated_cost_usd"] == 0
    assert result["coverage"]["complete_selected_documents"] is True
    assert result["coverage"]["read_video_ids"] == [VIDEO_A]
    assert result["evidence"][0]["text"] == transcript
    assert result["evidence"][0]["url"] == f"https://www.youtube.com/watch?v={VIDEO_A}"


def balanced_args(*extra):
    return consult_args("--channel", CHANNEL_B, *extra)


def balanced_search(monkeypatch, *, missing_second=False):
    def search(**kwargs):
        channel = kwargs["channel_ids"][0]
        videos = [VIDEO_A, VIDEO_A, VIDEO_C] if channel == CHANNEL else ([] if missing_second else [VIDEO_B, VIDEO_B])
        return {"data": [{"attributes": {"video_id": video, "channel_id": channel}} for video in videos]}
    mock = Mock(side_effect=search)
    monkeypatch.setattr(hosted_retrieval, "hosted_search", mock)
    return mock


@pytest.mark.parametrize("prepare_only", [False, True])
def test_balanced_selection_reads_both_channels_in_provider_order(monkeypatch, runtime, capsys, prepare_only):
    search = balanced_search(monkeypatch)
    sizes = Mock(return_value={VIDEO_A: {"channel_id": CHANNEL, "text_characters": 10000},
                               VIDEO_B: {"channel_id": CHANNEL_B, "text_characters": 10000},
                               VIDEO_C: {"channel_id": CHANNEL, "text_characters": 10000}})
    monkeypatch.setattr(operations, "_complete_document_sizes", sizes)
    consult = Mock(return_value={"status": "completed"})
    read = Mock(return_value={"evidence": [], "coverage": {"complete_selected_documents": True}})
    monkeypatch.setattr(consultation, "consult_published_views", consult)
    monkeypatch.setattr(consultation, "read_consultation_documents", read)
    # A repeated channel must not cause an additional search or consume a document slot.
    args = balanced_args("--channel", CHANNEL, *(["--prepare-only"] if prepare_only else []))
    assert operations.run_operation(args) == 0
    assert [call.kwargs["channel_ids"] for call in search.call_args_list] == [[CHANNEL], [CHANNEL_B]]
    sizes.assert_called_once_with(runtime[1], [CHANNEL, CHANNEL_B], [VIDEO_A, VIDEO_B, VIDEO_C])
    active, inactive = (read, consult) if prepare_only else (consult, read)
    active.assert_called_once()
    inactive.assert_not_called()
    assert active.call_args.kwargs["video_ids"] == [VIDEO_A, VIDEO_B, VIDEO_C]
    assert active.call_args.kwargs["max_context_chars"] == 300000
    output = json.loads(capsys.readouterr().out)
    selection = output["source_selection"]
    assert selection["required_video_ids"] == [VIDEO_A, VIDEO_B]
    assert selection["by_channel"][CHANNEL_B]["selected_video_ids"] == [VIDEO_B]
    assert selection["matched_passages"] == 5


def test_balanced_missing_channel_writes_explicit_result_without_model(monkeypatch, runtime, tmp_path):
    balanced_search(monkeypatch, missing_second=True)
    monkeypatch.setattr(operations, "_complete_document_sizes", lambda *args: pytest.fail("Missing coverage must not read documents"))
    monkeypatch.setattr(consultation, "consult_published_views", lambda **kwargs: pytest.fail("Missing coverage must not generate a comparison"))
    target = tmp_path / "coverage" / "result.json"
    assert operations.run_operation(balanced_args("--output", str(target))) == 0
    result = json.loads(target.read_text())
    assert result["status"] == "insufficient_channel_evidence"
    assert result["coverage"]["missing_channel_ids"] == [CHANNEL_B]
    assert result["coverage"]["read_video_ids"] == []
    assert result["answer"] is None and result["model_request_made"] is False


def test_balanced_document_cap_fails_before_search(monkeypatch, runtime, capsys):
    monkeypatch.setattr(hosted_retrieval, "hosted_search", lambda **kwargs: pytest.fail("Invalid cap must not search"))
    monkeypatch.setattr(operations, "_index_for_job", lambda *args: pytest.fail("Invalid cap must fail before index lookup"))
    assert operations.run_operation(balanced_args("--max-documents", "1")) == 1
    assert "one document per requested channel" in json.loads(capsys.readouterr().out)["message"]


def test_balanced_context_limit_drops_only_optional_whole_document(monkeypatch, runtime, capsys):
    balanced_search(monkeypatch)
    sizes = Mock(return_value={VIDEO_A: {"channel_id": CHANNEL, "text_characters": 100000},
                               VIDEO_B: {"channel_id": CHANNEL_B, "text_characters": 100000},
                               VIDEO_C: {"channel_id": CHANNEL, "text_characters": 150000}})
    monkeypatch.setattr(operations, "_complete_document_sizes", sizes)
    consult = Mock(return_value={"status": "completed"})
    monkeypatch.setattr(consultation, "consult_published_views", consult)
    assert operations.run_operation(balanced_args()) == 0
    sizes.assert_called_once()
    assert consult.call_args.kwargs["video_ids"] == [VIDEO_A, VIDEO_B]
    result = json.loads(capsys.readouterr().out)
    assert result["source_selection"]["selected_text_characters"] == 200000
    assert result["source_selection"]["omitted_documents"] == [{
        "video_id": VIDEO_C, "channel_id": CHANNEL, "reason": "complete_document_context_limit", "text_characters": 150000}]


def test_balanced_required_documents_over_limit_never_drop_a_channel(monkeypatch, runtime, capsys):
    balanced_search(monkeypatch)
    monkeypatch.setattr(operations, "_complete_document_sizes", lambda *args: {
        VIDEO_A: {"channel_id": CHANNEL, "text_characters": 200000},
        VIDEO_B: {"channel_id": CHANNEL_B, "text_characters": 150000},
        VIDEO_C: {"channel_id": CHANNEL, "text_characters": 1000},
    })
    monkeypatch.setattr(consultation, "consult_published_views", lambda **kwargs: pytest.fail("Oversized required set must not call model"))
    assert operations.run_operation(balanced_args()) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "context_limit_exceeded"
    assert result["coverage"]["required_video_ids"] == [VIDEO_A, VIDEO_B]
    assert result["coverage"]["required_text_characters"] == 350000
    assert result["coverage"]["read_video_ids"] == []
    assert result["answer"] is None and result["model_request_made"] is False


def test_balanced_selection_rejects_mismatched_source_channel(monkeypatch, runtime):
    balanced_search(monkeypatch)
    monkeypatch.setattr(operations, "_complete_document_sizes", lambda *args: {
        VIDEO_A: {"channel_id": CHANNEL, "text_characters": 1000},
        VIDEO_B: {"channel_id": CHANNEL, "text_characters": 1000},
        VIDEO_C: {"channel_id": CHANNEL, "text_characters": 1000},
    })
    monkeypatch.setattr(consultation, "consult_published_views", lambda **kwargs: pytest.fail("Mismatched attribution must not call model"))
    assert operations.run_operation(balanced_args()) == 1


def test_complete_document_sizes_use_latest_complete_nonempty_scoped_source():
    from sqlalchemy import create_engine, text
    engine = create_engine("sqlite:///:memory:")
    try:
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE channels (channel_id TEXT PRIMARY KEY)"))
            connection.execute(text("CREATE TABLE media (media_id TEXT PRIMARY KEY, channel_id TEXT)"))
            connection.execute(text("CREATE TABLE doc_content (content_id INTEGER PRIMARY KEY, media_id TEXT, chunk_type TEXT, full_text TEXT)"))
            connection.execute(text("INSERT INTO channels VALUES (:id)"), [{"id": CHANNEL}, {"id": CHANNEL_B}])
            connection.execute(text("INSERT INTO media VALUES (:video, :channel)"), [{"video": VIDEO_A, "channel": CHANNEL}, {"video": VIDEO_B, "channel": CHANNEL_B}])
            connection.execute(text("INSERT INTO doc_content VALUES (:id, :video, :kind, :body)"), [
                {"id": 1, "video": VIDEO_A, "kind": "complete", "body": "old text"},
                {"id": 2, "video": VIDEO_A, "kind": "complete", "body": "é字"},
                {"id": 3, "video": VIDEO_A, "kind": "segment", "body": "irrelevant partial"},
                {"id": 4, "video": VIDEO_A, "kind": "complete", "body": "   "},
                {"id": 5, "video": VIDEO_B, "kind": "complete", "body": "outside channel scope"},
            ])
        assert operations._complete_document_sizes(engine, [CHANNEL], [VIDEO_A, VIDEO_B]) == {
            VIDEO_A: {"channel_id": CHANNEL, "text_characters": 2}}
    finally:
        engine.dispose()

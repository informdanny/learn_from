from __future__ import annotations

import argparse
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
from sqlalchemy import JSON, Column, Integer, MetaData, String, Table, Text, create_engine, select, update

from learn_from import metadata_catalog, metadata_operations as operations

CHANNEL = "UC" + "a" * 22


def _parser():
    parser = argparse.ArgumentParser()
    operations.add_metadata_parsers(parser.add_subparsers(dest="command", required=True))
    return parser


def _field(*, known=("home_services",), more=False):
    return {"key": "industry", "openai": {"eligible": True, "key": "industry"}, "filter_operators": ["eq"],
            "known_values": [{"value": value} for value in known], "known_has_more": more}


def test_cli_filter_validation_requires_accepted_value_for_each_channel(monkeypatch):
    calls = []
    def schema(**kwargs):
        calls.append(kwargs)
        return {"fields": [_field(known=("home_services",) if kwargs["channel_id"] == "one" else ())]}
    monkeypatch.setattr(metadata_catalog, "get_metadata_schema", schema)
    with pytest.raises(ValueError, match="not accepted for every"):
        operations._validate_catalog_filters(engine=object(), channel_ids=["one", "two"], filters={"industry": "home_services"})
    assert [call["channel_id"] for call in calls] == ["one", "two"]


def test_approved_values_can_be_found_on_later_catalog_pages(monkeypatch):
    calls = []
    def schema(**kwargs):
        calls.append(kwargs["value_offset"])
        return {"fields": [_field(known=() if kwargs["value_offset"] == 0 else ("home_services",), more=kwargs["value_offset"] == 0)]}
    monkeypatch.setattr(metadata_catalog, "get_metadata_schema", schema)
    operations._validate_catalog_filters(engine=object(), channel_ids=["one"], filters={"industry": "home_services"})
    assert calls == [0, 1000]


@pytest.mark.parametrize("fields", [[], [_field() | {"filter_operators": []}], [_field() | {"openai": {"eligible": False, "key": "industry"}}], [_field() | {"openai": {"eligible": True, "key": "other"}}]])
def test_nonfilterable_or_remapped_fields_are_rejected(monkeypatch, fields):
    monkeypatch.setattr(metadata_catalog, "get_metadata_schema", lambda **kwargs: {"fields": fields})
    with pytest.raises(ValueError, match="not filterable"):
        operations._validate_catalog_filters(engine=object(), channel_ids=["one"], filters={"industry": "home_services"})


def test_no_metadata_filters_requires_no_catalog_reads(monkeypatch):
    monkeypatch.setattr(metadata_catalog, "get_metadata_schema", lambda **kwargs: pytest.fail("No metadata scope should not need a business catalog."))
    operations._validate_catalog_filters(engine=object(), channel_ids=["legacy"], filters={})


@pytest.mark.parametrize("expressions", [["industry"], ["=x"], ["industry="], ["industry=software", "industry=retail"]])
def test_malformed_or_duplicate_cli_filters_are_rejected(expressions):
    with pytest.raises(ValueError):
        operations._where(expressions)


@pytest.fixture
def cli(monkeypatch):
    settings = SimpleNamespace(database_url="safe-test-url", openai_api_key="test-key", youtube_api_key=None, youtube_transcript_io_api_key=None)
    disposed = []
    engine = SimpleNamespace(dispose=lambda: disposed.append(True))
    monkeypatch.setattr(operations, "get_settings", lambda: settings)
    monkeypatch.setattr(operations, "create_db_engine", lambda chosen: engine)
    return engine, disposed


def test_catalog_cli_forwards_parameter_names_and_disposes_engine(monkeypatch, cli, capsys):
    captured = []
    monkeypatch.setattr(metadata_catalog, "get_metadata_schema", lambda **kwargs: captured.append(kwargs) or {"fields": []})
    args = _parser().parse_args(["get-metadata-schema", "--channel-id", "one", "--category-id", "10", "--field", "industry", "--values-limit", "15", "--value-offset", "2"])
    assert operations.run_metadata_operation(args) == 0
    assert captured[0] == {"engine": cli[0], "channel_id": "one", "category_id": 10, "source_id": "youtube", "field": "industry", "values_limit": 15, "value_offset": 2}
    assert json.loads(capsys.readouterr().out) == {"fields": []}
    assert cli[1] == [True]


def test_extraction_cli_passes_explicit_budget_and_force(monkeypatch, cli, capsys):
    from learn_from import metadata_v2
    captured = []
    monkeypatch.setattr(metadata_v2, "run_index_metadata_for_content", lambda **kwargs: captured.append(kwargs) or {"run_id": 1})
    args = _parser().parse_args(["metadata-extract", "7", "--max-cost-usd", "0.05", "--force"])
    assert operations.run_metadata_operation(args) == 0
    assert captured[0] == {"engine": cli[0], "content_id": 7, "openai_api_key": "test-key", "max_cost_usd": 0.05, "force": True}


def test_search_rejects_provisional_vocabulary_before_any_provider_request(monkeypatch, cli, capsys):
    from learn_from import hosted_retrieval
    monkeypatch.setattr(metadata_catalog, "get_metadata_schema", lambda **kwargs: {"fields": [_field(known=())]})
    monkeypatch.setattr(hosted_retrieval, "hosted_search", lambda **kwargs: pytest.fail("No provider request for provisional filters."))
    args = _parser().parse_args(["metadata-search", "job_test", "Customers", "--channel", CHANNEL, "--where", "industry=home_services"])
    assert operations.run_metadata_operation(args) == 1
    assert "not accepted" in json.loads(capsys.readouterr().out)["message"]
    assert cli[1] == [True]


def test_metadata_sync_cli_selects_exact_files_without_starting_collection(monkeypatch, cli, capsys):
    from learn_from import hosted_retrieval
    captured = []
    monkeypatch.setattr(hosted_retrieval, "sync_collection_metadata", lambda **kwargs: captured.append(kwargs) or {"results": []})
    args = _parser().parse_args(["collection-metadata-sync", "job_test", "--video", "video000001", "--max-files", "1"])
    assert operations.run_metadata_operation(args) == 0
    assert captured[0] == {"engine": cli[0], "job_id": "job_test", "api_key": "test-key", "video_ids": ["video000001"], "max_files": 1}


@pytest.fixture
def cohort_db(monkeypatch):
    from learn_from.hosted_retrieval import source_document
    engine = create_engine("sqlite+pysqlite:///:memory:")
    meta = MetaData()
    jobs = Table("sync_jobs", meta, Column("job_id", String, primary_key=True), Column("checkpoint_data", JSON))
    channels = Table("channels", meta, Column("channel_id", String, primary_key=True), Column("channel_name", String))
    media = Table("media", meta, Column("media_id", String, primary_key=True), Column("channel_id", String), Column("title", String), Column("publish_date", String))
    content = Table("doc_content", meta, Column("content_id", Integer, primary_key=True), Column("media_id", String), Column("full_text", Text), Column("timestamped_segments", JSON), Column("chunk_type", String), Column("language", String))
    runs = Table("extraction_runs", meta, Column("run_id", Integer, primary_key=True), Column("content_id", Integer), Column("media_id", String), Column("status", String), Column("resolved_output", JSON), Column("completed_at", String))
    meta.create_all(engine)
    docs = [{"media_id": f"video{i:06d}", "content_id": i, "channel_id": CHANNEL, "channel_name": "Example", "title": f"Episode {i}", "publish_date": "2026-11-01", "language": "en", "full_text": f"Source {i} about business.", "timestamped_segments": [{"text": f"Source {i} about business.", "start_seconds": 0}]} for i in range(1,4)]
    checkpoint = {"items": [{"video_id": doc["media_id"], "channel_id": CHANNEL, "content_id": doc["content_id"], "status": "acquired"} for doc in docs],
                  "hosted_index": {"vector_store_id": "vs_test", "files": {doc["media_id"]: {"file_id": f"file_{doc['content_id']}", "status": "completed", "content_id": doc["content_id"], "source_hash": source_document(doc)[1]["source_hash"]} for doc in docs}}}
    run_rows = []
    with engine.begin() as c:
        c.execute(channels.insert().values(channel_id=CHANNEL, channel_name="Example"))
        c.execute(jobs.insert().values(job_id="job_test", checkpoint_data=checkpoint))
        for i, doc in enumerate(docs,1):
            c.execute(media.insert().values(**{k: doc[k] for k in ("media_id","channel_id","title","publish_date")}))
            c.execute(content.insert().values(**{k: doc[k] for k in ("content_id","media_id","full_text","timestamped_segments","language")},chunk_type="complete"))
            fields = [{"key": "industry", "value": "home_services" if i != 3 else "retail", "status": "provisional" if i == 2 else "trusted"},
                      {"key": "decision_area", "value": "pricing", "status": "trusted"},
                      {"key": "decision_area", "value": "customer_acquisition", "status": "trusted"}]
            result = {"version": "index-1", "source_hash": hashlib.sha256(doc["full_text"].encode()).hexdigest(), "fields": fields, "openai_attributes": {}}
            row = {"run_id": i, "content_id": i, "media_id": doc["media_id"], "status": "completed", "completed_at": "2026-11-11", "resolved_output": {"index_metadata": result}}
            c.execute(runs.insert().values(**row))
            run_rows.append(row)
    monkeypatch.setattr(operations, "_validate_catalog_filters", lambda **kwargs: None)
    return SimpleNamespace(engine=engine, jobs=jobs, media=media, content=content, runs=runs, docs=docs, checkpoint=checkpoint, run_rows=run_rows)


def _select(h, filters=None):
    return operations.select_metadata_video_ids(engine=h.engine, job_id="job_test", channel_ids=[CHANNEL], filters=filters or {"industry":"home_services","decision_area":"pricing"})


def test_multivalue_metadata_matches_any_trusted_value_and_all_requested_fields(cohort_db):
    h = cohort_db
    selected = _select(h)
    assert selected["video_ids"] == ["video000001"]
    assert selected["coverage"] == {"indexed_in_scope":3,"current_indexed_sources":3,"with_current_metadata":3,"matching_sources":1}
    assert selected["vector_store_id"] == "vs_test"
    assert h.run_rows[0]["resolved_output"]["index_metadata"]["openai_attributes"] == {}
    assert _select(h, {"industry":"home_services","decision_area":"customer_acquisition"})["video_ids"] == ["video000001"]
    with h.engine.connect() as c:
        assert c.execute(select(h.jobs.c.checkpoint_data)).scalar_one() == h.checkpoint


@pytest.mark.parametrize("change", ["rendered_hash", "content_id", "pending_index", "pending_item", "channel", "metadata_hash", "metadata_status", "provisional", "rejected", "newer_content"])
def test_stale_pending_untrusted_and_wrong_scope_sources_never_enter_selection(cohort_db, change):
    h = cohort_db
    checkpoint = copy.deepcopy(h.checkpoint)
    run = copy.deepcopy(h.run_rows[0])
    with h.engine.begin() as c:
        if change == "rendered_hash": checkpoint["hosted_index"]["files"]["video000001"]["source_hash"] = "stale"
        elif change == "content_id": checkpoint["hosted_index"]["files"]["video000001"]["content_id"] = 99
        elif change == "pending_index": checkpoint["hosted_index"]["files"]["video000001"]["status"] = "in_progress"
        elif change == "pending_item": checkpoint["items"][0]["status"] = "pending"
        elif change == "channel": c.execute(update(h.media).where(h.media.c.media_id=="video000001").values(channel_id="UC"+"b"*22))
        elif change == "metadata_hash": run["resolved_output"]["index_metadata"]["source_hash"] = "stale"
        elif change == "metadata_status": run["status"] = "failed"
        elif change in {"provisional","rejected"}: run["resolved_output"]["index_metadata"]["fields"][0]["status"] = change
        elif change == "newer_content": c.execute(h.content.insert().values(content_id=99,media_id="video000001",full_text="Replacement source.",timestamped_segments=[],chunk_type="complete",language="en"))
        c.execute(update(h.jobs).values(checkpoint_data=checkpoint))
        c.execute(update(h.runs).where(h.runs.c.run_id==1).values(**{k:v for k,v in run.items() if k!='run_id'}))
    assert _select(h)["video_ids"] == []


def test_latest_completed_metadata_supersedes_older_matching_values(cohort_db):
    h = cohort_db
    newer = copy.deepcopy(h.run_rows[0])
    newer.update(run_id=10,completed_at="2026-11-12")
    newer["resolved_output"]["index_metadata"]["fields"][0]["value"] = "software"
    with h.engine.begin() as c:
        c.execute(h.runs.insert().values(**newer))
    assert _select(h)["video_ids"] == []


def test_cohort_bound_fails_instead_of_silently_truncating(cohort_db):
    h = cohort_db
    checkpoint = {"items": [], "hosted_index":{"vector_store_id":"vs_test","files":{}}}
    for i in range(1001):
        video=f"video{i:06d}"
        checkpoint["items"].append({"video_id":video,"channel_id":CHANNEL,"status":"acquired"})
        checkpoint["hosted_index"]["files"][video]={"status":"completed","file_id":"file_test"}
    with h.engine.begin() as c:
        c.execute(update(h.jobs).values(checkpoint_data=checkpoint))
    with pytest.raises(ValueError,match="exceeds 1,000"):
        _select(h)


def test_cli_sends_cohort_ids_without_scalar_metadata_filter(monkeypatch, cli, capsys):
    from learn_from import hosted_retrieval
    captured=[]
    cohort={"video_ids":["video000001"],"vector_store_id":"vs_test","coverage":{"matching_sources":1},"filters":{"industry":"home_services"}}
    monkeypatch.setattr(operations,"select_metadata_video_ids",lambda **kwargs:cohort)
    monkeypatch.setattr(hosted_retrieval,"hosted_search",lambda **kwargs:captured.append(kwargs) or {"data":[]})
    args=_parser().parse_args(["metadata-search","job_test","Customers","--channel",CHANNEL,"--where","industry=home_services"])
    assert operations.run_metadata_operation(args)==0
    assert captured[0]["video_ids"]==["video000001"]
    assert captured[0]["channel_ids"]==[CHANNEL]
    assert "metadata_filters" not in captured[0]
    assert json.loads(capsys.readouterr().out)["metadata_selection"]==cohort

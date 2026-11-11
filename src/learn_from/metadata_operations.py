"""Inspectable metadata catalog, bounded extraction, and existing-index updates."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from learn_from.config import get_settings
from learn_from.db import create_db_engine

METADATA_COMMANDS = {
    "get-metadata-schema", "metadata-catalog-init", "metadata-extract",
    "metadata-value-accept", "collection-metadata-sync", "metadata-search",
}


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Expected a positive integer.") from None
    if result <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return result


def add_metadata_parsers(subparsers: Any) -> None:
    p = subparsers.add_parser("get-metadata-schema", help="Read metadata definitions and current accepted/proposed options without model calls.")
    p.add_argument("--channel-id")
    p.add_argument("--category-id", type=_positive_int)
    p.add_argument("--source-id", default="youtube")
    p.add_argument("--field")
    p.add_argument("--values-limit", type=_positive_int, default=50)
    p.add_argument("--value-offset", type=int, default=0)
    p = subparsers.add_parser("metadata-catalog-init", help="Initialize versioned business metadata definitions for selected existing channels.")
    p.add_argument("--channel", action="append", required=True, dest="channels")
    p = subparsers.add_parser("metadata-extract", help="Extract full-source index metadata with Luna, deterministic normalization, and cached reuse.")
    p.add_argument("content_id", type=_positive_int)
    p.add_argument("--max-cost-usd", type=float, default=0.10)
    p.add_argument("--force", action="store_true", help="Explicitly allow a fresh model run instead of reusing or stopping at a previous attempt.")
    p.add_argument("--output")
    p = subparsers.add_parser("metadata-value-accept", help="Accept one proposed catalog value for future extraction and filtering.")
    p.add_argument("value_id", type=_positive_int)
    p = subparsers.add_parser("collection-metadata-sync", help="Update metadata on selected existing indexed files; does not acquire or upload transcripts.")
    p.add_argument("job_id")
    p.add_argument("--video", action="append", required=True, dest="videos")
    p.add_argument("--max-files", type=_positive_int, default=10)
    p = subparsers.add_parser("metadata-search", help="Search the existing OpenAI index with explicit channel and optional metadata filters.")
    p.add_argument("job_id")
    p.add_argument("query")
    p.add_argument("--channel", action="append", required=True, dest="channels")
    p.add_argument("--where", action="append", default=[], help="Canonical field=value filter; repeat to combine with AND.")
    p.add_argument("--max-results", type=_positive_int, default=10)
    p.add_argument("--no-query-rewrite", action="store_true")
    p.add_argument("--output")


def _where(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for expression in values:
        key, separator, value = expression.partition("=")
        if not separator or not key.strip() or not value.strip() or key.strip() in result:
            raise ValueError("Use distinct field=value filters.")
        result[key.strip()] = value.strip()
    return result


def _validate_catalog_filters(*, engine: Any, channel_ids: list[str], filters: dict[str, str]) -> None:
    """Every selected channel must expose the requested accepted filter value."""
    from learn_from.metadata_catalog import get_metadata_schema
    from learn_from.hosted_retrieval import _metadata_filters

    _metadata_filters(filters)
    for channel_id in dict.fromkeys(channel_ids):
        if not filters:
            continue
        first_page = get_metadata_schema(engine=engine, channel_id=channel_id,
                                         values_limit=1000, value_offset=0)
        for key, value in filters.items():
            offset = 0
            while True:
                catalog = first_page if offset == 0 else get_metadata_schema(
                    engine=engine, channel_id=channel_id, field=key, values_limit=1000, value_offset=offset)
                fields = [field for field in catalog["fields"] if field["key"] == key]
                if len(fields) != 1 or not fields[0]["openai"]["eligible"] or fields[0]["openai"]["key"] != key or "eq" not in fields[0]["filter_operators"]:
                    raise ValueError(f"Metadata field {key} is not filterable for every selected channel.")
                field = fields[0]
                if any(option["value"] == value for option in field["known_values"]):
                    break
                if not field["known_has_more"]:
                    raise ValueError(f"Metadata value {key}={value} is not accepted for every selected channel; inspect get-metadata-schema.")
                offset += 1000


def select_metadata_video_ids(*, engine: Any, job_id: str, channel_ids: list[str], filters: dict[str, str]) -> dict[str, Any]:
    """Select current indexed sources by multi-valued metadata, without retrieval.

    A value must be trusted in the latest completed result for the current source.
    Different fields are ANDed; any trusted value within a field may match. The
    managed provider still handles text search/ranking after this bounded scope.
    """
    from sqlalchemy import MetaData, func, select
    from learn_from.hosted_retrieval import HostedRetrievalError, _provider_id, source_document
    from learn_from.metadata_v2 import INDEX_METADATA_VERSION

    scope = list(dict.fromkeys(channel_ids))
    if not 1 <= len(scope) <= 10 or any(not isinstance(v, str) or not re.fullmatch(r"UC[A-Za-z0-9_-]{22}", v) for v in scope):
        raise ValueError("Select one to ten canonical YouTube channel IDs.")
    _validate_catalog_filters(engine=engine, channel_ids=scope, filters=filters)
    with engine.connect() as connection:
        if connection.dialect.name == "postgresql":
            connection = connection.execution_options(isolation_level="REPEATABLE READ")
            connection.exec_driver_sql("SET TRANSACTION READ ONLY")
        metadata = MetaData()
        names = ("sync_jobs", "media", "doc_content", "channels", "extraction_runs")
        metadata.reflect(bind=connection, only=names, resolve_fks=False)
        tables = metadata.tables
        jobs, media, content, channels, runs = (tables[name] for name in
                  ("sync_jobs", "media", "doc_content", "channels", "extraction_runs"))
        checkpoint = connection.execute(select(jobs.c.checkpoint_data).where(jobs.c.job_id == job_id)).scalar_one_or_none()
        manifest = checkpoint.get("hosted_index") if isinstance(checkpoint, dict) else None
        if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
            raise ValueError("No current hosted index exists for this collection.")
        store_id = _provider_id(manifest.get("vector_store_id"), "vs_")
        items = {item.get("video_id"): item for item in checkpoint.get("items", []) if isinstance(item, dict)}
        candidates = {}
        for video_id, entry in manifest["files"].items():
            item = items.get(video_id)
            if (not isinstance(entry, dict) or entry.get("status") != "completed" or not item
                    or item.get("status") not in {"acquired", "reused"} or item.get("channel_id") not in scope):
                continue
            if not isinstance(video_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
                raise ValueError("Current index contains an invalid video identifier.")
            _provider_id(entry.get("file_id"), "file")
            candidates[video_id] = entry
        if len(candidates) > 1000:
            raise ValueError("Metadata selection exceeds 1,000 indexed sources; choose a narrower channel scope.")
        coverage = {"indexed_in_scope": len(candidates), "current_indexed_sources": 0, "with_current_metadata": 0, "matching_sources": 0}
        selected = []
        if candidates:
            latest = select(content.c.media_id, func.max(content.c.content_id).label("latest_id")).where(
                content.c.media_id.in_(candidates), content.c.chunk_type == "complete").group_by(content.c.media_id).subquery()
            documents = connection.execute(select(
                media.c.media_id, media.c.channel_id, media.c.title, media.c.publish_date, channels.c.channel_name,
                content.c.content_id, content.c.full_text, content.c.timestamped_segments, content.c.language,
            ).select_from(content.join(latest, content.c.content_id == latest.c.latest_id)
                .join(media, media.c.media_id == content.c.media_id)
                .outerjoin(channels, channels.c.channel_id == media.c.channel_id))
                .where(media.c.channel_id.in_(scope))).mappings().all()
            current = {}
            for row in documents:
                video_id, content_id = row["media_id"], row["content_id"]
                if (str(candidates[video_id].get("content_id")) != str(content_id)
                        or str(items[video_id].get("content_id")) != str(content_id)
                        or items[video_id].get("channel_id") != row["channel_id"]):
                    continue
                try:
                    _, identity = source_document(row)
                except (HostedRetrievalError, ValueError, TypeError):
                    continue
                if candidates[video_id].get("source_hash") != identity["source_hash"]:
                    continue
                current[content_id] = {"video_id": video_id, "hash": hashlib.sha256(row["full_text"].strip().encode()).hexdigest()}
            coverage["current_indexed_sources"] = len(current)
            latest_results = {}
            if current:
                rows = connection.execute(select(runs.c.content_id, runs.c.media_id, runs.c.resolved_output).where(
                    runs.c.content_id.in_(current), runs.c.status == "completed")
                    .order_by(runs.c.completed_at.desc().nulls_last(), runs.c.run_id.desc())).mappings()
                for row in rows:
                    resolved = row["resolved_output"]
                    result = resolved.get("index_metadata") if isinstance(resolved, dict) else None
                    if isinstance(result, dict) and row["content_id"] not in latest_results:
                        latest_results[row["content_id"]] = (row["media_id"], result)
            for content_id, source in current.items():
                found = latest_results.get(content_id)
                if not found:
                    if not filters:
                        selected.append(source["video_id"])
                    continue
                media_id, result = found
                if media_id != source["video_id"] or result.get("version") != INDEX_METADATA_VERSION or result.get("source_hash") != source["hash"]:
                    continue
                coverage["with_current_metadata"] += 1
                field_values: dict[str, set[str]] = {}
                for field in result.get("fields", []):
                    if isinstance(field, dict) and field.get("status") == "trusted" and isinstance(field.get("key"), str) and isinstance(field.get("value"), str):
                        field_values.setdefault(field["key"], set()).add(field["value"])
                if all(value in field_values.get(key, set()) for key, value in filters.items()):
                    selected.append(source["video_id"])
        selected.sort()
        coverage["matching_sources"] = len(selected)
        return {"video_ids": selected, "vector_store_id": store_id, "coverage": coverage, "filters": dict(filters)}


def run_metadata_operation(args: argparse.Namespace) -> int:
    from learn_from import metadata_catalog as catalog
    settings = get_settings()
    engine = create_db_engine(settings)
    try:
        if args.command == "get-metadata-schema":
            result = catalog.get_metadata_schema(engine=engine, channel_id=args.channel_id,
                category_id=args.category_id, source_id=args.source_id, field=args.field,
                values_limit=args.values_limit, value_offset=args.value_offset)
        elif args.command == "metadata-catalog-init":
            result = catalog.ensure_business_metadata_catalog(engine=engine, channel_ids=args.channels)
        elif args.command == "metadata-value-accept":
            result = catalog.accept_metadata_value(engine=engine, value_id=args.value_id)
        elif args.command == "metadata-extract":
            from learn_from.metadata_v2 import run_index_metadata_for_content
            result = run_index_metadata_for_content(engine=engine, content_id=args.content_id,
                openai_api_key=settings.openai_api_key, max_cost_usd=args.max_cost_usd, force=args.force)
        elif args.command == "collection-metadata-sync":
            from learn_from.hosted_retrieval import sync_collection_metadata
            result = sync_collection_metadata(engine=engine, job_id=args.job_id,
                api_key=settings.openai_api_key, video_ids=args.videos, max_files=args.max_files)
        elif args.command == "metadata-search":
            from learn_from.hosted_retrieval import hosted_search
            from learn_from.operations import _index_for_job
            filters = _where(args.where)
            cohort = select_metadata_video_ids(engine=engine, job_id=args.job_id, channel_ids=args.channels, filters=filters) if filters else None
            manifest = cohort if cohort is not None else _index_for_job(engine, args.job_id)
            result = hosted_search(api_key=settings.openai_api_key,
                vector_store_id=manifest.get("vector_store_id"), query=args.query,
                channel_ids=args.channels, video_ids=cohort["video_ids"] if cohort is not None else None,
                max_results=args.max_results, rewrite_query=not args.no_query_rewrite)
            if cohort is not None:
                result["metadata_selection"] = cohort
            result["coverage_note"] = "Metadata filters select current indexed transcripts using all trusted extracted values. OpenAI searches and ranks only those videos. Missing or provisional metadata is not evidence of absence."
        else:
            raise ValueError("Unknown metadata command.")
        if getattr(args, "output", None):
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
            result = {"output": str(destination.resolve()), "run_id": result.get("run_id"),
                      "reused": result.get("reused"), "usage": result.get("usage")}
        print(json.dumps(result, ensure_ascii=False, default=str))
        return 0
    except Exception as exc:
        from sqlalchemy.exc import SQLAlchemyError
        if isinstance(exc, SQLAlchemyError):
            message = "Metadata database operation failed; no database details were exposed."
        else:
            message = str(exc)
            for secret in (settings.database_url, settings.openai_api_key,
                           settings.youtube_api_key, settings.youtube_transcript_io_api_key):
                if secret:
                    message = message.replace(secret, "[redacted]")
        print(json.dumps({"error": type(exc).__name__, "message": message[:1500]}))
        return 1
    finally:
        engine.dispose()

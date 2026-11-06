"""Owner CLI for durable collections and OpenAI-backed consultation."""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
from typing import Any

from sqlalchemy import bindparam, text

from learn_from.config import get_settings
from learn_from.db import create_db_engine

OPERATION_COMMANDS = {
    "collection-plan", "collection-create", "collection-run", "collection-status",
    "collection-index", "consult", "collection-worker",
}


def _document_limit(value: Any) -> int:
    try:
        number = int(value)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("Document limit must be an integer from 1 to 10.") from None
    if isinstance(value, bool) or str(number) != str(value) or not 1 <= number <= 10:
        raise argparse.ArgumentTypeError("Document limit must be an integer from 1 to 10.")
    return number


def _consultation_budget(value: Any) -> float:
    try:
        budget = Decimal(str(value))
    except InvalidOperation:
        raise argparse.ArgumentTypeError("Consultation budget must be positive and at most $0.10.") from None
    if not budget.is_finite() or not Decimal("0") < budget <= Decimal("0.10"):
        raise argparse.ArgumentTypeError("Consultation budget must be positive and at most $0.10.")
    return float(budget)


def add_operation_parsers(subparsers: Any) -> None:
    p = subparsers.add_parser("collection-plan", help="Discover channel uploads; no transcript acquisition.")
    p.add_argument("channels", nargs="+")
    p.add_argument("--after", default="2005-01-01T00:00:00+00:00")
    p.add_argument("--before", default=None)
    p.add_argument("--min-seconds", type=int, default=180)
    p.add_argument("--max-pages", type=int, default=150)
    p.add_argument("--output", required=True)
    p = subparsers.add_parser("collection-create", help="Save a selected collection and request cap in Neon.")
    p.add_argument("plan")
    p.add_argument("--max-provider-requests", type=int, required=True)
    for name, help_text in (("collection-run", "Run/resume a saved collection without rich metadata."),
                            ("collection-status", "Read cloud collection progress."),
                            ("collection-index", "Index saved collection transcripts with OpenAI.")):
        p = subparsers.add_parser(name, help=help_text)
        p.add_argument("job_id")
        if name == "collection-run":
            p.add_argument("--max-items", type=int, default=None)
        if name == "collection-index":
            p.add_argument("--max-files", type=int, default=1000)
    p = subparsers.add_parser("consult", help="Consult published views using hosted search and complete transcripts.")
    p.add_argument("question")
    p.add_argument("--channel", action="append", required=True, dest="channels")
    scope = p.add_mutually_exclusive_group(required=True)
    scope.add_argument("--video", action="append", dest="videos")
    scope.add_argument("--job-id")
    p.add_argument("--max-documents", type=_document_limit, default=3)
    p.add_argument("--max-cost-usd", type=_consultation_budget, default=0.10)
    p.add_argument("--prepare-only", action="store_true",
                   help="Export complete selected transcript evidence without a model request.")
    p.add_argument("--output")
    subparsers.add_parser("collection-worker", help="One-shot cloud worker configured with environment variables.")


def _emit(value: Any) -> None:
    print(json.dumps(value, default=str, ensure_ascii=False), flush=True)


def _index_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    """Keep per-file source data in the durable manifest, never in progress logs."""
    fields = (
        "status", "completed_count", "eligible_count", "usage_bytes", "max_index_bytes",
        "max_files_per_run", "vector_store_id", "provider_status", "store_state",
        "halt_reason", "error", "updated_at",
    )
    summary = {key: manifest[key] for key in fields if key in manifest}
    for key in ("error", "halt_reason"):
        if key in summary and summary[key] is not None:
            summary[key] = str(summary[key])[:1000]
    entries = manifest.get("files")
    if isinstance(entries, dict):
        summary["file_count"] = len(entries)
    if isinstance(manifest.get("eligible_count"), int) and isinstance(manifest.get("completed_count"), int):
        summary["remaining_count"] = max(0, manifest["eligible_count"] - manifest["completed_count"])
    return summary


def _emit_index(manifest: dict[str, Any]) -> None:
    _emit({"stage": "hosted_index", "result": _index_summary(manifest)})


def _index_for_job(engine: Any, job_id: str) -> dict[str, Any]:
    with engine.connect() as conn:
        state = conn.execute(text("SELECT checkpoint_data FROM sync_jobs WHERE job_id=:job"), {"job": job_id}).scalar_one()
    if not isinstance(state, dict) or not isinstance(state.get("hosted_index", {}), dict):
        raise ValueError("Collection checkpoint has no valid hosted index manifest.")
    return state.get("hosted_index", {})


def _complete_document_sizes(engine: Any, channel_ids: list[str], video_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Inspect latest complete sources in one query without loading transcript bodies."""
    query = text("""
        SELECT m.media_id, m.channel_id, length(dc.full_text) AS text_characters
        FROM media m
        JOIN channels ch ON ch.channel_id = m.channel_id
        JOIN doc_content dc ON dc.media_id = m.media_id
        WHERE m.channel_id IN :channel_ids AND m.media_id IN :video_ids
          AND dc.chunk_type = 'complete' AND length(trim(dc.full_text)) > 0
        ORDER BY dc.content_id DESC
    """).bindparams(bindparam("channel_ids", expanding=True), bindparam("video_ids", expanding=True))
    with engine.connect() as connection:
        rows = connection.execute(query, {"channel_ids": channel_ids, "video_ids": video_ids}).mappings()
        sizes: dict[str, dict[str, Any]] = {}
        for row in rows:
            sizes.setdefault(row["media_id"], {"channel_id": row["channel_id"], "text_characters": int(row["text_characters"])})
    return sizes


def _balanced_hosted_selection(*, engine: Any, api_key: str, vector_store_id: str,
                              question: str, channel_ids: list[str], max_documents: int,
                              max_context_chars: int = 300000) -> tuple[list[str], dict[str, Any], dict[str, Any] | None]:
    """Reserve a complete source per channel using the provider's ranking only."""
    from learn_from.hosted_retrieval import hosted_search
    scope = list(dict.fromkeys(channel_ids))
    if not 1 <= len(scope) <= 10 or any(not isinstance(value, str) or not re.fullmatch(r"UC[A-Za-z0-9_-]{22}", value) for value in scope):
        raise ValueError("Select one to ten canonical YouTube channel IDs.")
    if max_documents < len(scope):
        raise ValueError("--max-documents must cover at least one document per requested channel.")
    candidates: dict[str, list[str]] = {}
    owners: dict[str, str] = {}
    selection: dict[str, Any] = {
        "provider": "openai_vector_store", "vector_store_id": vector_store_id,
        "selection_policy": "one_per_channel_then_round_robin_in_provider_order",
        "requested_channel_ids": scope, "by_channel": {}, "matched_passages": 0,
        "selected_video_ids": [], "omitted_documents": [], "max_context_characters": max_context_chars,
    }
    for channel in scope:
        result = hosted_search(api_key=api_key, vector_store_id=vector_store_id,
                               query=question, channel_ids=[channel], max_results=max_documents * 3)
        hits = result.get("data", [])
        videos: list[str] = []
        for hit in hits:
            attrs = hit.get("attributes", {})
            if attrs.get("channel_id") != channel:
                raise ValueError("Hosted search returned a source outside the requested channel.")
            video = attrs.get("video_id")
            if not isinstance(video, str) or not re.fullmatch(r"[A-Za-z0-9_-]{11}", video):
                continue
            if video in owners and owners[video] != channel:
                raise ValueError("Hosted search attributed one video to conflicting channels.")
            owners[video] = channel
            if video not in videos:
                videos.append(video)
        candidates[channel] = videos
        selection["matched_passages"] += len(hits)
        selection["by_channel"][channel] = {"matched_passages": len(hits), "candidate_videos": len(videos), "selected_video_ids": []}
    missing = [channel for channel in scope if not candidates[channel]]
    if missing:
        return [], selection, {
            "status": "insufficient_channel_evidence", "answer": None,
            "message": "Managed search found no usable matches for every requested channel; no comparison was generated.",
            "coverage": {"requested_channel_ids": scope, "missing_channel_ids": missing,
                         "read_video_ids": [], "complete_selected_documents": False},
            "model_request_made": False, "usage": None, "estimated_cost_usd": 0.0,
        }
    required = [candidates[channel][0] for channel in scope]
    proposed = list(required)
    for rank in range(1, max(len(videos) for videos in candidates.values())):
        for channel in scope:
            if len(proposed) < max_documents and rank < len(candidates[channel]):
                proposed.append(candidates[channel][rank])
    sizes = _complete_document_sizes(engine, scope, proposed)
    for video in proposed:
        if video not in sizes or sizes[video]["channel_id"] != owners[video]:
            raise ValueError("A selected indexed transcript is missing or has mismatched channel attribution; no partial read was substituted.")
    selection["required_video_ids"] = required
    selection["document_text_characters"] = {video: sizes[video]["text_characters"] for video in proposed}
    total = sum(sizes[video]["text_characters"] for video in required)
    if total > max_context_chars:
        return [], selection, {
            "status": "context_limit_exceeded", "answer": None,
            "message": f"The top complete transcript from each channel requires {total:,} characters, exceeding the {max_context_chars:,} limit. Select shorter videos explicitly; no source was truncated or channel omitted.",
            "coverage": {"requested_channel_ids": scope, "required_video_ids": required,
                         "required_text_characters": total, "max_context_characters": max_context_chars,
                         "read_video_ids": [], "complete_selected_documents": False},
            "model_request_made": False, "usage": None, "estimated_cost_usd": 0.0,
        }
    selected = list(required)
    for video in proposed[len(required):]:
        count = sizes[video]["text_characters"]
        if total + count > max_context_chars:
            selection["omitted_documents"].append({"video_id": video, "channel_id": owners[video],
                                                    "reason": "complete_document_context_limit", "text_characters": count})
        else:
            selected.append(video)
            total += count
    selection["selected_video_ids"] = selected
    selection["selected_text_characters"] = total
    for channel in scope:
        selection["by_channel"][channel]["selected_video_ids"] = [video for video in selected if owners[video] == channel]
    return selected, selection, None


def run_operation(args: argparse.Namespace) -> int:
    settings = get_settings()
    engine = create_db_engine(settings)
    try:
        from learn_from.collections import (
            collection_status, create_collection, discover_collection_plan, run_collection,
        )
        if args.command == "collection-plan":
            plan = discover_collection_plan(
                channel_inputs=args.channels, youtube_api_key=settings.youtube_api_key,
                published_after=datetime.fromisoformat(args.after.replace("Z", "+00:00")),
                published_before=datetime.fromisoformat(args.before.replace("Z", "+00:00")) if args.before else datetime.now(UTC),
                min_duration_seconds=args.min_seconds, max_pages_per_channel=args.max_pages,
            )
            target = Path(args.output)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
            result = {"path": str(target.resolve()), "selected_count": plan["selected_count"],
                      "discovery_complete": plan["discovery_complete"], "channels": plan["channel_discovery"]}
        elif args.command == "collection-create":
            plan = json.loads(Path(args.plan).read_text())
            job_id = create_collection(engine=engine, plan=plan, max_provider_requests=args.max_provider_requests)
            result = collection_status(engine=engine, job_id=job_id)
        elif args.command == "collection-run":
            result = run_collection(engine=engine, job_id=args.job_id, settings=settings,
                                    max_items=args.max_items, on_progress=_emit)
        elif args.command == "collection-status":
            result = collection_status(engine=engine, job_id=args.job_id)
        elif args.command == "collection-index":
            from learn_from.hosted_retrieval import sync_collection_index
            indexed = sync_collection_index(engine=engine, job_id=args.job_id,
                api_key=settings.openai_api_key, max_files=args.max_files, on_progress=_emit_index)
            result = _index_summary(indexed)
        elif args.command == "collection-worker":
            from learn_from.worker import run_worker
            return run_worker(settings=settings, engine=engine)
        elif args.command == "consult":
            from learn_from.consultation import consult_published_views, read_consultation_documents
            max_documents = _document_limit(args.max_documents)
            max_cost_usd = _consultation_budget(args.max_cost_usd)
            target = Path(args.output) if args.output else None
            if target is not None:
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.is_dir():
                    raise ValueError("Consultation output must be a file path.")
            video_ids = args.videos
            search_metadata = None
            if not video_ids:
                from learn_from.hosted_retrieval import hosted_search
                channels = list(dict.fromkeys(args.channels))
                if len(channels) > 1 and max_documents < len(channels):
                    raise ValueError("--max-documents must cover at least one document per requested channel.")
                manifest = _index_for_job(engine, args.job_id)
                store_id = manifest.get("vector_store_id")
                if not store_id:
                    raise ValueError("This collection has no hosted index. Run collection-index or select --video.")
                if len(channels) > 1:
                    video_ids, search_metadata, unavailable = _balanced_hosted_selection(
                        engine=engine, api_key=settings.openai_api_key, vector_store_id=store_id,
                        question=args.question, channel_ids=channels, max_documents=max_documents)
                    if unavailable is not None:
                        result = unavailable
                else:
                    search_result = hosted_search(api_key=settings.openai_api_key, vector_store_id=store_id,
                        query=args.question, channel_ids=channels, max_results=max_documents * 3)
                    video_ids = []
                    for hit in search_result.get("data", []):
                        video_id = hit.get("attributes", {}).get("video_id")
                        if video_id and video_id not in video_ids:
                            video_ids.append(video_id)
                        if len(video_ids) >= max_documents:
                            break
                    if not video_ids:
                        result = {"status": "no_evidence", "answer": None,
                                  "message": "Hosted search found no matching indexed transcripts; no model request was made.",
                                  "vector_store_id": store_id, "usage": None, "estimated_cost_usd": 0.0}
                    search_metadata = {"provider": "openai_vector_store", "vector_store_id": store_id,
                                       "matched_passages": len(search_result.get("data", [])), "selected_video_ids": video_ids}
            if video_ids:
                if getattr(args, "prepare_only", False):
                    packet = read_consultation_documents(engine=engine, question=args.question,
                        channel_ids=args.channels, video_ids=video_ids, max_context_chars=300000)
                    result = {"status": "prepared", **packet, "model_request_made": False,
                              "usage": None, "estimated_cost_usd": 0.0}
                else:
                    result = consult_published_views(engine=engine, question=args.question,
                        channel_ids=args.channels, video_ids=video_ids, openai_api_key=settings.openai_api_key,
                        max_cost_usd=max_cost_usd, max_context_chars=300000)
            if search_metadata:
                result["source_selection"] = search_metadata
            if target is not None:
                target.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
                result = {"output": str(target.resolve()), "status": result.get("status"), "usage": result.get("usage")}
        else:
            raise ValueError(f"Unknown operation: {args.command}")
        _emit(result)
        return 0
    except Exception as exc:
        # Provider exceptions may contain request detail: never let configured secrets escape.
        message = str(exc)
        for secret in (settings.openai_api_key, settings.youtube_api_key, settings.youtube_transcript_io_api_key,
                       settings.database_url):
            if secret:
                message = message.replace(secret, "[redacted]")
        _emit({"error": type(exc).__name__, "message": message})
        return 1
    finally:
        engine.dispose()

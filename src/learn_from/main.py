from __future__ import annotations

import argparse
import json
from typing import Sequence

from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from learn_from.config import get_settings
from learn_from.db import check_database_connection, create_db_engine
from learn_from.explorer import serve_explorer
from learn_from.ingestion import (
    IngestionConfigurationError,
    IngestionError,
    backfill_channel,
    ingest_ad_hoc_video,
)
from learn_from.metadata import (
    MetadataExtractionError,
    MetadataPersistenceError,
    extract_and_persist_metadata_for_content,
)
from learn_from.metadata_v2 import (
    MetadataV2Error,
    create_profile_draft,
    list_extraction_profiles,
    list_extraction_runs,
    publish_profile_version,
    run_metadata_v2_for_content,
)
from learn_from.retrieval_profiles import RetrievalProfileError, get_retrieval_profiles
from learn_from.transcripts import (
    TranscriptNotAvailableError,
    TranscriptRetrievalError,
    fetch_transcript,
    format_transcript_with_timestamps,
)
from learn_from.youtube import YouTubeApiError, fetch_video_metadata


def _redact(secret: str | None) -> str:
    if not secret:
        return "missing"
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}...{secret[-4:]}"


def run_config_check() -> int:
    try:
        settings = get_settings()
    except ValidationError as exc:
        print("Configuration validation failed.")
        print(exc)
        return 1

    summary = {
        "app_env": settings.app_env,
        "database_url": "set",
        "youtube_api_key": _redact(settings.youtube_api_key),
        "youtube_transcript_io_api_key": _redact(settings.youtube_transcript_io_api_key),
        "openai_api_key": _redact(settings.openai_api_key),
        "openai_vector_store_id": _redact(settings.openai_vector_store_id),
        "metadata_extraction_default_model": settings.metadata_extraction_default_model,
        "metadata_extraction_enhanced_model": settings.metadata_extraction_enhanced_model,
        "metadata_extraction_timeout_seconds": settings.metadata_extraction_timeout_seconds,
        "mcp_auth_bearer_token": _redact(settings.mcp_auth_bearer_token),
        "redis_url": _redact(settings.redis_url),
        "gmail_smtp_username": _redact(settings.gmail_smtp_username),
        "gmail_smtp_app_password": _redact(settings.gmail_smtp_app_password),
        "notify_email_to": _redact(settings.notify_email_to),
        "transcript_proxy_http_url": _redact(settings.transcript_proxy_http_url),
        "transcript_proxy_https_url": _redact(settings.transcript_proxy_https_url),
    }
    print(json.dumps(summary, indent=2))
    return 0


def run_db_check() -> int:
    try:
        settings = get_settings()
    except ValidationError as exc:
        print("Configuration validation failed.")
        print(exc)
        return 1

    engine = create_db_engine(settings)
    try:
        check_database_connection(engine)
    except SQLAlchemyError as exc:
        print(f"Database connection failed: {exc}")
        return 1
    finally:
        engine.dispose()

    print("Database connection check passed.")
    return 0


def run_youtube_metadata(video_id_or_url: str) -> int:
    try:
        settings = get_settings()
    except ValidationError as exc:
        print("Configuration validation failed.")
        print(exc)
        return 1

    if not settings.youtube_api_key:
        print("YOUTUBE_API_KEY is required for youtube-metadata.")
        return 1

    try:
        metadata = fetch_video_metadata(
            video_id_or_url=video_id_or_url,
            api_key=settings.youtube_api_key,
        )
    except (ValueError, YouTubeApiError) as exc:
        print(f"YouTube metadata retrieval failed: {exc}")
        return 1

    print(json.dumps(metadata.to_dict(), indent=2))
    return 0


def run_youtube_transcript(video_id_or_url: str, languages: Sequence[str]) -> int:
    try:
        settings = get_settings()
    except ValidationError as exc:
        print("Configuration validation failed.")
        print(exc)
        return 1

    try:
        transcript = fetch_transcript(
            video_id_or_url=video_id_or_url,
            languages=languages,
            transcript_io_api_key=settings.youtube_transcript_io_api_key,
            proxy_http_url=settings.transcript_proxy_http_url,
            proxy_https_url=settings.transcript_proxy_https_url,
        )
    except ValueError as exc:
        print(f"Transcript retrieval failed: {exc}")
        return 1
    except TranscriptNotAvailableError as exc:
        print(f"Transcript unavailable: {exc}")
        return 1
    except TranscriptRetrievalError as exc:
        print(f"Transcript retrieval failed: {exc}")
        return 1

    payload = transcript.to_dict()
    payload["formatted_text"] = format_transcript_with_timestamps(transcript)
    print(json.dumps(payload, indent=2))
    return 0


def run_metadata_extract_persist(content_id: int, *, model: str | None, enhanced: bool) -> int:
    try:
        settings = get_settings()
    except ValidationError as exc:
        print("Configuration validation failed.")
        print(exc)
        return 1

    if not settings.openai_api_key:
        print("OPENAI_API_KEY is required for metadata-extract-persist.")
        return 1

    selected_model = model
    if not selected_model:
        selected_model = (
            settings.metadata_extraction_enhanced_model
            if enhanced
            else settings.metadata_extraction_default_model
        )

    engine = create_db_engine(settings)
    try:
        summary = extract_and_persist_metadata_for_content(
            engine=engine,
            content_id=content_id,
            openai_api_key=settings.openai_api_key,
            model=selected_model,
            timeout_seconds=settings.metadata_extraction_timeout_seconds,
        )
    except (MetadataExtractionError, MetadataPersistenceError, ValueError, SQLAlchemyError) as exc:
        print(f"Metadata extraction/persistence failed: {exc}")
        return 1
    finally:
        engine.dispose()

    print(json.dumps(summary, indent=2))
    return 0


def run_metadata_v2_extract(
    content_id: int,
    *,
    dry_run: bool,
    skip_resolver_model: bool,
    skip_learner_model: bool,
    extractor_model: str | None,
    resolver_model: str | None,
    learner_model: str | None,
    service_tier: str,
) -> int:
    try:
        settings = get_settings()
    except ValidationError as exc:
        print("Configuration validation failed.")
        print(exc)
        return 1

    if not settings.openai_api_key:
        print("OPENAI_API_KEY is required for metadata-v2-extract.")
        return 1

    engine = create_db_engine(settings)
    try:
        summary = run_metadata_v2_for_content(
            engine=engine,
            content_id=content_id,
            openai_api_key=settings.openai_api_key,
            persist_run=not dry_run,
            use_resolver_model=not skip_resolver_model,
            use_learner_model=not skip_learner_model,
            extractor_model=extractor_model,
            resolver_model=resolver_model,
            learner_model=learner_model,
            service_tier=service_tier,
            timeout_seconds=max(900.0, settings.metadata_extraction_timeout_seconds),
        )
    except (MetadataV2Error, MetadataExtractionError, MetadataPersistenceError, ValueError, SQLAlchemyError) as exc:
        print(f"Metadata V2 extraction failed: {exc}")
        return 1
    finally:
        engine.dispose()

    print(json.dumps(summary, indent=2, default=str))
    return 0


def run_metadata_v2_profiles() -> int:
    try:
        settings = get_settings()
    except ValidationError as exc:
        print("Configuration validation failed.")
        print(exc)
        return 1
    engine = create_db_engine(settings)
    try:
        profiles = list_extraction_profiles(engine=engine)
    except (MetadataV2Error, SQLAlchemyError) as exc:
        print(f"Metadata V2 profile lookup failed: {exc}")
        return 1
    finally:
        engine.dispose()
    print(json.dumps({"count": len(profiles), "profiles": profiles}, indent=2, default=str))
    return 0


def run_metadata_v2_profile_create(
    *,
    profile_name: str,
    scope: str,
    source_id: str | None,
    channel_id: str | None,
    category_id: int | None,
    learning_mode: str,
    enable_scripture: bool,
    prompt_instructions: str,
    category_schema_json: str | None,
    known_values_json: str | None,
    blocked_values_json: str | None,
) -> int:
    try:
        settings = get_settings()
    except ValidationError as exc:
        print("Configuration validation failed.")
        print(exc)
        return 1
    try:
        category_schema = _parse_json_option(
            category_schema_json,
            option_name="--category-schema-json",
            expected_type=list,
        )
        known_values = _parse_json_option(
            known_values_json,
            option_name="--known-values-json",
            expected_type=dict,
        )
        blocked_values = _parse_json_option(
            blocked_values_json,
            option_name="--blocked-values-json",
            expected_type=dict,
        )
    except ValueError as exc:
        print(f"Metadata V2 profile draft creation failed: {exc}")
        return 1

    engine = create_db_engine(settings)
    try:
        result = create_profile_draft(
            engine=engine,
            profile_name=profile_name,
            scope=scope,
            source_id=source_id,
            channel_id=channel_id,
            category_id=category_id,
            learning_mode=learning_mode,
            enable_scripture=enable_scripture,
            prompt_instructions=prompt_instructions,
            category_schema=category_schema,
            known_values=known_values,
            blocked_values=blocked_values,
        )
    except (MetadataV2Error, ValueError, SQLAlchemyError) as exc:
        print(f"Metadata V2 profile draft creation failed: {exc}")
        return 1
    finally:
        engine.dispose()
    print(json.dumps(result, indent=2, default=str))
    return 0


def _parse_json_option(
    value: str | None,
    *,
    option_name: str,
    expected_type: type[list] | type[dict],
) -> list | dict | None:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{option_name} must contain valid JSON: {exc.msg}") from exc
    if not isinstance(parsed, expected_type):
        expected_name = "array" if expected_type is list else "object"
        raise ValueError(f"{option_name} must be a JSON {expected_name}.")
    return parsed


def run_metadata_v2_profile_publish(profile_version_id: int) -> int:
    try:
        settings = get_settings()
    except ValidationError as exc:
        print("Configuration validation failed.")
        print(exc)
        return 1
    engine = create_db_engine(settings)
    try:
        result = publish_profile_version(engine=engine, profile_version_id=profile_version_id)
    except (MetadataV2Error, SQLAlchemyError) as exc:
        print(f"Metadata V2 profile publication failed: {exc}")
        return 1
    finally:
        engine.dispose()
    print(json.dumps(result, indent=2, default=str))
    return 0


def run_metadata_v2_runs(*, content_id: int | None, limit: int) -> int:
    try:
        settings = get_settings()
    except ValidationError as exc:
        print("Configuration validation failed.")
        print(exc)
        return 1
    engine = create_db_engine(settings)
    try:
        runs = list_extraction_runs(engine=engine, content_id=content_id, limit=limit)
    except (MetadataV2Error, SQLAlchemyError) as exc:
        print(f"Metadata V2 run lookup failed: {exc}")
        return 1
    finally:
        engine.dispose()
    print(json.dumps({"count": len(runs), "runs": runs}, indent=2, default=str))
    return 0


def run_get_retrieval_profiles(*, active_only: bool) -> int:
    try:
        settings = get_settings()
    except ValidationError as exc:
        print("Configuration validation failed.")
        print(exc)
        return 1

    engine = create_db_engine(settings)
    try:
        profiles = get_retrieval_profiles(engine=engine, active_only=active_only)
    except (RetrievalProfileError, SQLAlchemyError) as exc:
        print(f"Retrieval profile lookup failed: {exc}")
        return 1
    finally:
        engine.dispose()

    payload = {
        "active_only": active_only,
        "count": len(profiles),
        "profiles": profiles,
    }
    print(json.dumps(payload, indent=2))
    return 0


def run_channel_backfill(
    channel_id: str,
    *,
    max_videos: int,
    languages: Sequence[str],
    model: str | None,
    enhanced: bool,
    skip_metadata: bool,
) -> int:
    try:
        settings = get_settings()
    except ValidationError as exc:
        print("Configuration validation failed.")
        print(exc)
        return 1

    if not settings.youtube_api_key:
        print("YOUTUBE_API_KEY is required for channel-backfill.")
        return 1

    selected_model = model
    if not selected_model:
        selected_model = (
            settings.metadata_extraction_enhanced_model
            if enhanced
            else settings.metadata_extraction_default_model
        )

    if not skip_metadata and not settings.openai_api_key:
        print("OPENAI_API_KEY is required for channel-backfill unless --skip-metadata is set.")
        return 1

    engine = create_db_engine(settings)
    try:
        summary = backfill_channel(
            engine=engine,
            channel_id=channel_id,
            youtube_api_key=settings.youtube_api_key,
            transcript_io_api_key=settings.youtube_transcript_io_api_key,
            languages=languages,
            max_videos=max_videos,
            include_metadata=not skip_metadata,
            openai_api_key=settings.openai_api_key,
            metadata_model=selected_model,
            metadata_timeout_seconds=settings.metadata_extraction_timeout_seconds,
            transcript_proxy_http_url=settings.transcript_proxy_http_url,
            transcript_proxy_https_url=settings.transcript_proxy_https_url,
        )
    except (IngestionConfigurationError, IngestionError, SQLAlchemyError, ValueError) as exc:
        print(f"Channel backfill failed: {exc}")
        return 1
    finally:
        engine.dispose()

    print(json.dumps(summary, indent=2))
    return 0


def run_ad_hoc_ingest(
    video_id_or_url: str,
    *,
    languages: Sequence[str],
    model: str | None,
    enhanced: bool,
    skip_metadata: bool,
) -> int:
    try:
        settings = get_settings()
    except ValidationError as exc:
        print("Configuration validation failed.")
        print(exc)
        return 1

    if not settings.youtube_api_key:
        print("YOUTUBE_API_KEY is required for ad-hoc-ingest.")
        return 1

    selected_model = model
    if not selected_model:
        selected_model = (
            settings.metadata_extraction_enhanced_model
            if enhanced
            else settings.metadata_extraction_default_model
        )

    if not skip_metadata and not settings.openai_api_key:
        print("OPENAI_API_KEY is required for ad-hoc-ingest unless --skip-metadata is set.")
        return 1

    engine = create_db_engine(settings)
    try:
        summary = ingest_ad_hoc_video(
            engine=engine,
            video_id_or_url=video_id_or_url,
            youtube_api_key=settings.youtube_api_key,
            transcript_io_api_key=settings.youtube_transcript_io_api_key,
            languages=languages,
            include_metadata=not skip_metadata,
            openai_api_key=settings.openai_api_key,
            metadata_model=selected_model,
            metadata_timeout_seconds=settings.metadata_extraction_timeout_seconds,
            transcript_proxy_http_url=settings.transcript_proxy_http_url,
            transcript_proxy_https_url=settings.transcript_proxy_https_url,
        )
    except (IngestionConfigurationError, IngestionError, SQLAlchemyError, ValueError) as exc:
        print(f"Ad-hoc ingest failed: {exc}")
        return 1
    finally:
        engine.dispose()

    print(json.dumps(summary, indent=2))
    return 0


def run_explorer(*, host: str, port: int) -> int:
    try:
        settings = get_settings()
    except ValidationError as exc:
        print("Configuration validation failed.")
        print(exc)
        return 1

    engine = create_db_engine(settings)
    try:
        check_database_connection(engine)
        serve_explorer(engine=engine, host=host, port=port)
    except (OSError, SQLAlchemyError) as exc:
        engine.dispose()
        print(f"Explorer failed: {exc}")
        return 1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="learn-from",
        description="Learn From development CLI utilities.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    from learn_from.operations import add_operation_parsers
    add_operation_parsers(subparsers)
    subparsers.add_parser("config-check", help="Validate and print config summary.")
    subparsers.add_parser("db-check", help="Validate database connectivity.")
    explorer_parser = subparsers.add_parser(
        "explorer",
        help="Run the local read-only transcript and metadata explorer.",
    )
    explorer_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind (default: 127.0.0.1).",
    )
    explorer_parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Local HTTP port (default: 8765).",
    )
    youtube_parser = subparsers.add_parser(
        "youtube-metadata",
        help="Fetch YouTube metadata for a video id or URL.",
    )
    youtube_parser.add_argument("video", help="YouTube video ID or URL")
    transcript_parser = subparsers.add_parser(
        "youtube-transcript",
        help="Fetch transcript segments for a YouTube video id or URL.",
    )
    transcript_parser.add_argument("video", help="YouTube video ID or URL")
    transcript_parser.add_argument(
        "--languages",
        default="en",
        help="Comma-separated language codes in priority order (default: en).",
    )
    metadata_parser = subparsers.add_parser(
        "metadata-extract-persist",
        help="Extract metadata for an existing doc_content row and persist to doc_* tables.",
    )
    metadata_parser.add_argument("content_id", type=int, help="doc_content.content_id to process")
    metadata_parser.add_argument(
        "--model",
        default=None,
        help="OpenAI model override (defaults to configured metadata extraction model).",
    )
    metadata_parser.add_argument(
        "--enhanced",
        action="store_true",
        help="Use configured enhanced extraction model when --model is not supplied.",
    )
    metadata_v2_parser = subparsers.add_parser(
        "metadata-v2-extract",
        help="Run the side-by-side Metadata V2 three-worker pipeline for one content row.",
    )
    metadata_v2_parser.add_argument("content_id", type=int, help="doc_content.content_id to process")
    metadata_v2_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all selected workers without writing V2 run, observation, or learning records.",
    )
    metadata_v2_parser.add_argument(
        "--skip-resolver-model",
        action="store_true",
        help="Use deterministic resolution only; do not call Worker 2's model.",
    )
    metadata_v2_parser.add_argument(
        "--skip-learner-model",
        action="store_true",
        help="Do not call Worker 3 for channel learning proposals.",
    )
    metadata_v2_parser.add_argument("--extractor-model", default=None, help="Worker 1 model override.")
    metadata_v2_parser.add_argument("--resolver-model", default=None, help="Worker 2 model override.")
    metadata_v2_parser.add_argument("--learner-model", default=None, help="Worker 3 model override.")
    metadata_v2_parser.add_argument(
        "--service-tier",
        choices=("auto", "flex"),
        default="flex",
        help="Responses service tier for Workers 2 and 3 (default: flex).",
    )
    subparsers.add_parser(
        "metadata-v2-profiles",
        help="List Metadata V2 extraction profiles and all draft/active versions.",
    )
    profile_create_parser = subparsers.add_parser(
        "metadata-v2-profile-create",
        help="Create a versioned Metadata V2 profile draft.",
    )
    profile_create_parser.add_argument("profile_name", help="Operator-facing profile name.")
    profile_create_parser.add_argument(
        "--scope",
        required=True,
        choices=("global", "source", "category", "channel", "channel_category"),
    )
    profile_create_parser.add_argument("--source-id", default=None)
    profile_create_parser.add_argument("--channel-id", default=None)
    profile_create_parser.add_argument("--category-id", type=int, default=None)
    profile_create_parser.add_argument(
        "--learning-mode",
        choices=("off", "observe", "assisted", "autopilot"),
        default="autopilot",
    )
    profile_create_parser.add_argument(
        "--enable-scripture",
        action="store_true",
        help="Enable exhaustive scripture references and structured scripture resolution.",
    )
    profile_create_parser.add_argument(
        "--prompt-instructions",
        default="",
        help="Scoped prompt instructions stored in the draft profile.",
    )
    profile_create_parser.add_argument(
        "--category-schema-json",
        default=None,
        help="JSON array defining category-specific extraction attributes.",
    )
    profile_create_parser.add_argument(
        "--known-values-json",
        default=None,
        help="JSON object of scoped participant, topic, entity, or reference value hints.",
    )
    profile_create_parser.add_argument(
        "--blocked-values-json",
        default=None,
        help="JSON object of scoped values that extraction or resolution must reject.",
    )
    profile_publish_parser = subparsers.add_parser(
        "metadata-v2-profile-publish",
        help="Publish one draft/testing profile version and retire its prior active version.",
    )
    profile_publish_parser.add_argument("profile_version_id", type=int)
    runs_parser = subparsers.add_parser(
        "metadata-v2-runs",
        help="List immutable Metadata V2 run manifests and outputs.",
    )
    runs_parser.add_argument("--content-id", type=int, default=None)
    runs_parser.add_argument("--limit", type=int, default=50)
    retrieval_profiles_parser = subparsers.add_parser(
        "get-retrieval-profiles",
        help="Read retrieval profile presets from the retrieval_profiles source table.",
    )
    retrieval_profiles_parser.add_argument(
        "--active-only",
        action="store_true",
        help="Return only active retrieval profiles.",
    )
    backfill_parser = subparsers.add_parser(
        "channel-backfill",
        help="Backfill recent channel videos into media/doc_content with sync_jobs tracking.",
    )
    backfill_parser.add_argument("channel_id", help="YouTube channel ID to ingest.")
    backfill_parser.add_argument(
        "--max-videos",
        type=int,
        default=30,
        help="Maximum number of recent channel videos to process (default: 30).",
    )
    backfill_parser.add_argument(
        "--languages",
        default="en",
        help="Comma-separated language codes in priority order (default: en).",
    )
    backfill_parser.add_argument(
        "--model",
        default=None,
        help="OpenAI model override for metadata extraction (default from config).",
    )
    backfill_parser.add_argument(
        "--enhanced",
        action="store_true",
        help="Use configured enhanced metadata model when --model is not supplied.",
    )
    backfill_parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Ingest transcripts without running metadata extraction persistence.",
    )
    adhoc_parser = subparsers.add_parser(
        "ad-hoc-ingest",
        help="Ingest one ad-hoc video URL/ID into media/doc_content with sync_jobs tracking.",
    )
    adhoc_parser.add_argument("video", help="YouTube video ID or URL")
    adhoc_parser.add_argument(
        "--languages",
        default="en",
        help="Comma-separated language codes in priority order (default: en).",
    )
    adhoc_parser.add_argument(
        "--model",
        default=None,
        help="OpenAI model override for metadata extraction (default from config).",
    )
    adhoc_parser.add_argument(
        "--enhanced",
        action="store_true",
        help="Use configured enhanced metadata model when --model is not supplied.",
    )
    adhoc_parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Ingest transcripts without running metadata extraction persistence.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    from learn_from.operations import OPERATION_COMMANDS, run_operation
    if args.command in OPERATION_COMMANDS:
        return run_operation(args)

    if args.command == "config-check":
        return run_config_check()
    if args.command == "db-check":
        return run_db_check()
    if args.command == "explorer":
        return run_explorer(host=args.host, port=args.port)
    if args.command == "youtube-metadata":
        return run_youtube_metadata(args.video)
    if args.command == "youtube-transcript":
        language_codes = [code.strip() for code in args.languages.split(",") if code.strip()]
        return run_youtube_transcript(args.video, language_codes or ["en"])
    if args.command == "metadata-extract-persist":
        return run_metadata_extract_persist(args.content_id, model=args.model, enhanced=args.enhanced)
    if args.command == "metadata-v2-extract":
        return run_metadata_v2_extract(
            args.content_id,
            dry_run=args.dry_run,
            skip_resolver_model=args.skip_resolver_model,
            skip_learner_model=args.skip_learner_model,
            extractor_model=args.extractor_model,
            resolver_model=args.resolver_model,
            learner_model=args.learner_model,
            service_tier=args.service_tier,
        )
    if args.command == "metadata-v2-profiles":
        return run_metadata_v2_profiles()
    if args.command == "metadata-v2-profile-create":
        return run_metadata_v2_profile_create(
            profile_name=args.profile_name,
            scope=args.scope,
            source_id=args.source_id,
            channel_id=args.channel_id,
            category_id=args.category_id,
            learning_mode=args.learning_mode,
            enable_scripture=args.enable_scripture,
            prompt_instructions=args.prompt_instructions,
            category_schema_json=args.category_schema_json,
            known_values_json=args.known_values_json,
            blocked_values_json=args.blocked_values_json,
        )
    if args.command == "metadata-v2-profile-publish":
        return run_metadata_v2_profile_publish(args.profile_version_id)
    if args.command == "metadata-v2-runs":
        return run_metadata_v2_runs(content_id=args.content_id, limit=args.limit)
    if args.command == "get-retrieval-profiles":
        return run_get_retrieval_profiles(active_only=args.active_only)
    if args.command == "channel-backfill":
        language_codes = [code.strip() for code in args.languages.split(",") if code.strip()]
        return run_channel_backfill(
            args.channel_id,
            max_videos=args.max_videos,
            languages=language_codes or ["en"],
            model=args.model,
            enhanced=args.enhanced,
            skip_metadata=args.skip_metadata,
        )
    if args.command == "ad-hoc-ingest":
        language_codes = [code.strip() for code in args.languages.split(",") if code.strip()]
        return run_ad_hoc_ingest(
            args.video,
            languages=language_codes or ["en"],
            model=args.model,
            enhanced=args.enhanced,
            skip_metadata=args.skip_metadata,
        )

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

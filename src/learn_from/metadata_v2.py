from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import MetaData, Table, and_, func, insert, or_, select, update
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import NoSuchTableError, SQLAlchemyError

from learn_from.metadata import (
    MetadataExtractionError,
    MetadataPersistenceError,
    _extract_json_from_response_payload,
    _normalize_key,
    _post_json,
)
from learn_from.scripture import find_explicit_scripture_references, parse_scripture_reference


V2_RELEASE_VERSION = "2.0"
V2_NORMALIZER_VERSION = "light-1"
V2_STABLE_MODEL = "gpt-5.6-luna"
V2_ENHANCED_MODEL = "gpt-5.6-terra"
V2_RESOLVER_MODEL = "gpt-5.6-terra"
V2_LEARNER_MODEL = "gpt-5.6-terra"
V2_PROVIDER = "openai"

_MODEL_TOKEN_PRICING = {
    "gpt-5.6-luna": {"input": 1.00, "cached_input": 0.10, "output": 6.00},
    "gpt-5.6-terra": {"input": 2.50, "cached_input": 0.25, "output": 15.00},
}
_USAGE_COLLECTOR: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "metadata_v2_usage_collector",
    default=None,
)
_USAGE_STEP: ContextVar[str] = ContextVar("metadata_v2_usage_step", default="unknown")

DEFAULT_CHUNK_MAX_CHARS = 18_000
DEFAULT_CHUNK_OVERLAP_SEGMENTS = 3
TRUSTED_CONFIDENCE = 0.95
PROVISIONAL_CONFIDENCE = 0.80
ALIAS_DOCUMENT_THRESHOLD = 3

_GENERIC_PARTICIPANT_NAMES = {
    "speaker",
    "host",
    "guest",
    "narrator",
    "interviewer",
    "interviewee",
    "unknown speaker",
    "unknown person",
    "random guy",
    "random person",
}
_REFERENCE_TYPES = {
    "book",
    "paper",
    "study",
    "scripture",
    "talk",
    "podcast",
    "youtube_video",
    "author_creator",
    "magazine_article",
    "website",
    "media_fragment",
    "other",
}
_ENTITY_TYPES = {
    "person",
    "organization",
    "brand",
    "product",
    "service",
    "software",
    "framework",
    "metric",
    "location",
    "timeframe",
    "event",
    "topic",
    "concept",
    "other",
}
_USAGE_TYPES = {
    "primary_topic",
    "recommended",
    "criticized",
    "quoted",
    "evidence",
    "example",
    "comparison",
    "background",
    "neutral",
}


class MetadataV2Error(RuntimeError):
    """Raised when the V2 metadata pipeline cannot complete."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AttributeCandidate(StrictModel):
    key: str
    value: str
    evidence: str
    confidence: float = Field(ge=0.0, le=1.0)


class ParticipantCandidate(StrictModel):
    raw_name: str
    role: str | None
    evidence: str
    confidence: float = Field(ge=0.0, le=1.0)


class EntityCandidate(StrictModel):
    type: Literal[
        "person",
        "organization",
        "brand",
        "product",
        "service",
        "software",
        "framework",
        "metric",
        "location",
        "timeframe",
        "event",
        "topic",
        "concept",
        "other",
    ]
    raw_label: str
    context: str | None
    usage: Literal[
        "primary_topic",
        "recommended",
        "criticized",
        "quoted",
        "evidence",
        "example",
        "comparison",
        "background",
        "neutral",
    ]
    evidence: str
    confidence: float = Field(ge=0.0, le=1.0)


class ReferenceCandidate(StrictModel):
    type: Literal[
        "book",
        "paper",
        "study",
        "scripture",
        "talk",
        "podcast",
        "youtube_video",
        "author_creator",
        "magazine_article",
        "website",
        "media_fragment",
        "other",
    ]
    raw_mention: str
    suggested_label: str | None
    context: str | None
    usage: Literal[
        "primary_topic",
        "recommended",
        "criticized",
        "quoted",
        "evidence",
        "example",
        "comparison",
        "background",
        "neutral",
    ]
    evidence: str
    confidence: float = Field(ge=0.0, le=1.0)
    resolution_basis: Literal["explicit", "normalized", "inferred", "unresolved"]


class ChunkExtraction(StrictModel):
    category: str | None
    topics: list[str]
    keywords: list[str]
    participants: list[ParticipantCandidate]
    entities: list[EntityCandidate]
    references: list[ReferenceCandidate]
    category_attributes: list[AttributeCandidate]


class ResolvedParticipant(StrictModel):
    raw_name: str
    canonical_name: str | None
    role: str | None
    evidence: str
    confidence: float = Field(ge=0.0, le=1.0)
    resolution_status: Literal["provisional", "trusted", "rejected", "unresolved"]
    resolution_basis: Literal["explicit", "normalized", "alias", "inferred", "externally_verified", "unresolved"]


class ResolvedEntity(StrictModel):
    type: str
    raw_label: str
    canonical_label: str | None
    context: str | None
    usage: str
    evidence: str
    confidence: float = Field(ge=0.0, le=1.0)
    resolution_status: Literal["provisional", "trusted", "rejected", "unresolved"]
    resolution_basis: Literal["explicit", "normalized", "alias", "inferred", "externally_verified", "unresolved"]


class ScriptureResolution(StrictModel):
    canon: str | None
    collection: str | None
    work: str | None
    book: str | None
    chapter_start: int | None
    chapter_end: int | None
    verse_start: int | None
    verse_end: int | None
    section_start: int | None
    section_end: int | None
    canonical_locator: str | None


class ResolvedReference(StrictModel):
    type: str
    raw_mention: str
    canonical_label: str | None
    context: str | None
    usage: str
    evidence: str
    confidence: float = Field(ge=0.0, le=1.0)
    resolution_status: Literal["provisional", "trusted", "rejected", "unresolved"]
    resolution_basis: Literal["explicit", "normalized", "alias", "inferred", "externally_verified", "unresolved"]
    scripture: ScriptureResolution | None


class ResolverOutput(StrictModel):
    participants: list[ResolvedParticipant]
    entities: list[ResolvedEntity]
    references: list[ResolvedReference]


class LearningSuggestion(StrictModel):
    dimension: str
    action: str
    raw_value: str | None
    canonical_value: str | None
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class LearnerOutput(StrictModel):
    suggestions: list[LearningSuggestion]
    prompt_additions: list[str]
    blocked_values: list[str]


@dataclass(frozen=True)
class TranscriptChunk:
    index: int
    text: str
    start_seconds: float | None
    end_seconds: float | None


@dataclass(frozen=True)
class ContentContext:
    content_id: int
    media_id: str
    source_id: str
    source_type: str
    channel_id: str | None
    channel_name: str | None
    category_id: int | None
    category_name: str | None
    title: str
    description: str | None
    transcript_text: str
    timestamped_segments: list[dict[str, Any]]
    channel_document_count: int


@dataclass(frozen=True)
class EffectiveProfile:
    profile_version_id: int | None
    profile_ids: tuple[int, ...]
    version_ids: tuple[int, ...]
    learning_mode: str
    config: dict[str, Any]


@dataclass(frozen=True)
class ModelRoute:
    run_mode: Literal["bootstrap", "stable", "audit", "ad_hoc"]
    extractor_model: str
    extractor_reasoning: str
    resolver_model: str
    learner_model: str


StructuredRequester = Callable[..., Mapping[str, Any]]


def default_profile_config() -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "reference_modules": [],
        "coverage": {
            "topics": "top_n",
            "keywords": "top_n",
            "participants": "exhaustive_named",
            "entities": "high_signal",
            "references": "exhaustive",
        },
        "model_routing": {
            "bootstrap_documents": 3,
            "audit_interval": 10,
            "stable_model": V2_STABLE_MODEL,
            "enhanced_model": V2_ENHANCED_MODEL,
            "resolver_model": V2_RESOLVER_MODEL,
            "learner_model": V2_LEARNER_MODEL,
        },
        "confidence": {
            "trusted": TRUSTED_CONFIDENCE,
            "provisional": PROVISIONAL_CONFIDENCE,
            "alias_documents": ALIAS_DOCUMENT_THRESHOLD,
        },
        "prompt_instructions": "",
        "category_schema": [],
        "known_values": {},
        "blocked_values": {"participants": sorted(_GENERIC_PARTICIPANT_NAMES)},
    }


def determine_model_route(
    *,
    context: ContentContext,
    profile: EffectiveProfile,
    extractor_override: str | None = None,
    resolver_override: str | None = None,
    learner_override: str | None = None,
) -> ModelRoute:
    routing = profile.config.get("model_routing", {})
    bootstrap_documents = int(routing.get("bootstrap_documents", 3))
    audit_interval = max(1, int(routing.get("audit_interval", 10)))

    if context.source_type == "ad_hoc" or not context.channel_id:
        run_mode: Literal["bootstrap", "stable", "audit", "ad_hoc"] = "ad_hoc"
    elif context.channel_document_count < bootstrap_documents:
        run_mode = "bootstrap"
    elif (context.channel_document_count + 1) % audit_interval == 0:
        run_mode = "audit"
    else:
        run_mode = "stable"

    if run_mode in {"bootstrap", "audit"}:
        extractor_model = str(routing.get("enhanced_model") or V2_ENHANCED_MODEL)
        extractor_reasoning = "medium"
    else:
        extractor_model = str(routing.get("stable_model") or V2_STABLE_MODEL)
        extractor_reasoning = "low"

    return ModelRoute(
        run_mode=run_mode,
        extractor_model=extractor_override or extractor_model,
        extractor_reasoning=extractor_reasoning,
        resolver_model=resolver_override or str(routing.get("resolver_model") or V2_RESOLVER_MODEL),
        learner_model=learner_override or str(routing.get("learner_model") or V2_LEARNER_MODEL),
    )


def light_normalize_text(value: str) -> str:
    text = value.replace("\u00a0", " ")
    text = re.sub(r"\b(um|uh)(?:\s*[,.-]?\s*\1){1,}\b", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def build_transcript_chunks(
    *,
    transcript_text: str,
    timestamped_segments: Sequence[Mapping[str, Any]] | None = None,
    max_chars: int = DEFAULT_CHUNK_MAX_CHARS,
    overlap_segments: int = DEFAULT_CHUNK_OVERLAP_SEGMENTS,
) -> tuple[str, list[dict[str, Any]], list[TranscriptChunk]]:
    if max_chars < 1000:
        raise ValueError("max_chars must be at least 1000.")

    clean_segments: list[dict[str, Any]] = []
    for segment in timestamped_segments or ():
        text = light_normalize_text(str(segment.get("text") or ""))
        if not text:
            continue
        start = _optional_float(segment.get("start_seconds"))
        duration = _optional_float(segment.get("duration_seconds"))
        clean_segments.append(
            {
                "text": text,
                "start_seconds": start,
                "end_seconds": start + duration if start is not None and duration is not None else start,
            }
        )

    if not clean_segments:
        normalized = light_normalize_text(transcript_text)
        chunks: list[TranscriptChunk] = []
        start = 0
        index = 0
        overlap_chars = min(1500, max_chars // 5)
        while start < len(normalized):
            end = min(len(normalized), start + max_chars)
            if end < len(normalized):
                boundary = normalized.rfind(" ", start + max_chars // 2, end)
                if boundary > start:
                    end = boundary
            chunks.append(TranscriptChunk(index=index, text=normalized[start:end], start_seconds=None, end_seconds=None))
            if end >= len(normalized):
                break
            start = max(start + 1, end - overlap_chars)
            index += 1
        return normalized, [], chunks

    chunks = []
    chunk_segments: list[dict[str, Any]] = []
    chunk_chars = 0
    index = 0
    cursor = 0
    while cursor < len(clean_segments):
        segment = clean_segments[cursor]
        addition = len(segment["text"]) + 24
        if chunk_segments and chunk_chars + addition > max_chars:
            chunks.append(_chunk_from_segments(index, chunk_segments))
            index += 1
            overlap = chunk_segments[-overlap_segments:] if overlap_segments else []
            chunk_segments = list(overlap)
            chunk_chars = sum(len(item["text"]) + 24 for item in chunk_segments)
            continue
        chunk_segments.append(segment)
        chunk_chars += addition
        cursor += 1
    if chunk_segments:
        chunks.append(_chunk_from_segments(index, chunk_segments))

    analysis_text = " ".join(segment["text"] for segment in clean_segments)
    return analysis_text, clean_segments, chunks


def _chunk_from_segments(index: int, segments: Sequence[Mapping[str, Any]]) -> TranscriptChunk:
    lines = []
    for segment in segments:
        start = _optional_float(segment.get("start_seconds"))
        prefix = f"[{_format_timestamp(start)}] " if start is not None else ""
        lines.append(f"{prefix}{segment['text']}")
    return TranscriptChunk(
        index=index,
        text="\n".join(lines),
        start_seconds=_optional_float(segments[0].get("start_seconds")),
        end_seconds=_optional_float(segments[-1].get("end_seconds")),
    )


def _format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def build_extractor_instructions(*, profile: EffectiveProfile) -> str:
    coverage = profile.config.get("coverage", {})
    modules = profile.config.get("reference_modules", [])
    custom_instructions = str(profile.config.get("prompt_instructions") or "").strip()
    category_schema = profile.config.get("category_schema", [])
    return "\n".join(
        value
        for value in (
            "# Identity\nYou are Worker 1, the transcript metadata extractor for a retrieval library.",
            "# Hard boundaries\n"
            "Extract candidates from supplied evidence only. Preserve uncertainty and changes of position. "
            "Do not perform external research, silently identify unnamed people, or use generic roles such as "
            "speaker, host, guest, or narrator as participant names. Participants must speak or be present in "
            "this document. A person mentioned as a guest in a previous episode is an entity mention, not a "
            "participant in the current document. Scripture citations belong in references, not keywords.",
            "# Coverage\n"
            f"Topics={coverage.get('topics', 'top_n')}; keywords={coverage.get('keywords', 'top_n')}; "
            f"participants={coverage.get('participants', 'exhaustive_named')}; "
            f"entities={coverage.get('entities', 'high_signal')}; "
            f"references={coverage.get('references', 'exhaustive')}.",
            "# Reference modules\n"
            f"Enabled modules: {', '.join(str(item) for item in modules) or 'none'}. "
            "Emit one reference candidate per distinct work or passage; never combine multiple citations into one candidate. "
            "When scripture is enabled, capture every explicit scripture mention and probable allusion, but mark "
            "allusions as inferred and retain the exact transcript wording.",
            "# Category attributes\n" + json.dumps(category_schema, sort_keys=True),
            "# Profile instructions\n" + custom_instructions if custom_instructions else "",
            "# Output\nReturn only the strict structured output requested by the response schema.",
        )
        if value
    )


def build_extractor_input(
    *,
    context: ContentContext,
    profile: EffectiveProfile,
    chunk: TranscriptChunk,
    chunk_count: int,
) -> str:
    known_values = profile.config.get("known_values", {})
    blocked_values = profile.config.get("blocked_values", {})
    return (
        "<document_context>\n"
        f"title: {context.title}\n"
        f"source: {context.source_id}\n"
        f"channel: {context.channel_name or 'ad hoc'}\n"
        f"category: {context.category_name or 'unassigned'}\n"
        f"description: {context.description or ''}\n"
        f"known_values: {json.dumps(known_values, sort_keys=True)}\n"
        f"blocked_values: {json.dumps(blocked_values, sort_keys=True)}\n"
        "</document_context>\n"
        f"<transcript_chunk index=\"{chunk.index}\" count=\"{chunk_count}\" "
        f"start_seconds=\"{chunk.start_seconds}\" end_seconds=\"{chunk.end_seconds}\">\n"
        f"{chunk.text}\n"
        "</transcript_chunk>"
    )


def run_extractor_worker(
    *,
    context: ContentContext,
    profile: EffectiveProfile,
    route: ModelRoute,
    openai_api_key: str,
    timeout_seconds: float = 180.0,
    requester: StructuredRequester | None = None,
) -> tuple[dict[str, Any], str, list[dict[str, Any]], list[TranscriptChunk]]:
    analysis_text, segment_map, chunks = build_transcript_chunks(
        transcript_text=context.transcript_text,
        timestamped_segments=context.timestamped_segments,
    )
    if not chunks:
        raise MetadataV2Error("Transcript normalization produced no extraction chunks.")

    request = requester or _request_openai_structured
    instructions = build_extractor_instructions(profile=profile)
    chunk_results: list[ChunkExtraction] = []
    for chunk in chunks:
        raw = request(
            api_key=openai_api_key,
            model=route.extractor_model,
            reasoning_effort=route.extractor_reasoning,
            instructions=instructions,
            input_text=build_extractor_input(
                context=context,
                profile=profile,
                chunk=chunk,
                chunk_count=len(chunks),
            ),
            schema_name="metadata_v2_chunk",
            schema=_strict_schema(ChunkExtraction),
            timeout_seconds=timeout_seconds,
            service_tier="auto",
        )
        try:
            chunk_results.append(ChunkExtraction.model_validate(raw))
        except ValidationError as exc:
            raise MetadataV2Error(f"Worker 1 returned an invalid chunk extraction: {exc}") from exc

    aggregate = _aggregate_chunk_extractions(chunk_results)
    if "scripture" in {str(item) for item in profile.config.get("reference_modules", [])}:
        aggregate["references"] = _merge_deterministic_scripture_references(
            references=aggregate["references"],
            transcript_text=context.transcript_text,
        )
    return aggregate, analysis_text, segment_map, chunks


def _aggregate_chunk_extractions(results: Sequence[ChunkExtraction]) -> dict[str, Any]:
    categories = [item.category.strip() for item in results if item.category and item.category.strip()]
    category = Counter(categories).most_common(1)[0][0] if categories else None

    topics = _dedupe_strings(value for item in results for value in item.topics)[:12]
    keywords = [
        value
        for value in _dedupe_strings(value.lower() for item in results for value in item.keywords)
        if not is_scripture_keyword(value)
    ][:20]
    attributes = _dedupe_records(
        (value.model_dump() for item in results for value in item.category_attributes),
        key_fields=("key", "value"),
    )
    participants = _dedupe_records(
        (value.model_dump() for item in results for value in item.participants),
        key_fields=("raw_name",),
        prefer_confidence=True,
    )
    entities = _dedupe_records(
        (value.model_dump() for item in results for value in item.entities),
        key_fields=("type", "raw_label"),
        prefer_confidence=True,
    )
    references = _dedupe_records(
        (value.model_dump() for item in results for value in item.references),
        key_fields=("type", "raw_mention"),
        prefer_confidence=True,
    )
    return {
        "schema_version": "2.0",
        "category": category,
        "topics": topics,
        "keywords": keywords,
        "participants": participants,
        "entities": entities,
        "references": references,
        "category_attributes": attributes,
    }


def _merge_deterministic_scripture_references(
    *,
    references: Sequence[Mapping[str, Any]],
    transcript_text: str,
) -> list[dict[str, Any]]:
    merged = [dict(item) for item in references]
    seen = {
        (
            str(item.get("type") or "").lower(),
            _normalize_key(str(item.get("raw_mention") or item.get("suggested_label") or "")),
        )
        for item in merged
    }
    for detected in find_explicit_scripture_references(transcript_text):
        locator = detected["locator"]
        key = ("scripture", _normalize_key(str(detected["raw_mention"])))
        if key in seen:
            continue
        seen.add(key)
        merged.append(
            {
                "type": "scripture",
                "raw_mention": detected["raw_mention"],
                "suggested_label": locator["canonical_label"],
                "context": None,
                "usage": "neutral",
                "evidence": detected["evidence"],
                "confidence": 0.99,
                "resolution_basis": "explicit",
            }
        )
    return merged


def _dedupe_strings(values: Sequence[str] | Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = re.sub(r"\s+", " ", str(value)).strip()
        key = _normalize_key(text)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _dedupe_records(
    values: Sequence[Mapping[str, Any]] | Any,
    *,
    key_fields: Sequence[str],
    prefer_confidence: bool = False,
) -> list[dict[str, Any]]:
    selected: dict[tuple[str, ...], dict[str, Any]] = {}
    order: list[tuple[str, ...]] = []
    for item in values:
        record = dict(item)
        key = tuple(_normalize_key(str(record.get(field) or "")) for field in key_fields)
        if not any(key):
            continue
        if key not in selected:
            selected[key] = record
            order.append(key)
            continue
        if prefer_confidence and float(record.get("confidence") or 0) > float(
            selected[key].get("confidence") or 0
        ):
            selected[key] = record
    return [selected[key] for key in order]


def _strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    _make_schema_strict(schema)
    return schema


def _make_schema_strict(node: Any) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            properties = node.get("properties", {})
            node["additionalProperties"] = False
            node["required"] = list(properties)
        for value in node.values():
            _make_schema_strict(value)
    elif isinstance(node, list):
        for item in node:
            _make_schema_strict(item)


def _request_openai_structured(
    *,
    api_key: str,
    model: str,
    reasoning_effort: str,
    instructions: str,
    input_text: str,
    schema_name: str,
    schema: Mapping[str, Any],
    timeout_seconds: float,
    service_tier: str,
) -> Mapping[str, Any]:
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for Metadata V2.")
    payload: dict[str, Any] = {
        "model": model,
        "reasoning": {"effort": reasoning_effort},
        "instructions": instructions,
        "input": input_text,
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": dict(schema),
                "strict": True,
            }
        },
    }
    if service_tier != "auto":
        payload["service_tier"] = service_tier

    response = _post_json(
        url="https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "learn-from/0.1.0",
        },
        payload=payload,
        timeout_seconds=timeout_seconds,
    )
    _capture_response_usage(
        response=response,
        model=model,
        requested_service_tier=service_tier,
    )
    parsed = _extract_json_from_response_payload(response)
    if not isinstance(parsed, Mapping):
        raise MetadataExtractionError("OpenAI returned a non-object V2 extraction payload.")
    return dict(parsed)


def _capture_response_usage(
    *,
    response: Mapping[str, Any],
    model: str,
    requested_service_tier: str,
) -> None:
    collector = _USAGE_COLLECTOR.get()
    if collector is None:
        return
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        return
    input_details = usage.get("input_tokens_details")
    output_details = usage.get("output_tokens_details")
    input_details = input_details if isinstance(input_details, Mapping) else {}
    output_details = output_details if isinstance(output_details, Mapping) else {}
    input_tokens = int(usage.get("input_tokens") or 0)
    cached_input_tokens = int(input_details.get("cached_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    event = {
        "step": _USAGE_STEP.get(),
        "model": model,
        "service_tier": response.get("service_tier") or requested_service_tier,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": int(output_details.get("reasoning_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or input_tokens + output_tokens),
    }
    event["estimated_cost"] = _estimate_token_cost(event)
    collector.append(event)


def _estimate_token_cost(event: Mapping[str, Any]) -> float | None:
    rates = _MODEL_TOKEN_PRICING.get(str(event.get("model") or ""))
    if not rates:
        return None
    input_tokens = int(event.get("input_tokens") or 0)
    cached_tokens = min(input_tokens, int(event.get("cached_input_tokens") or 0))
    output_tokens = int(event.get("output_tokens") or 0)
    cost = (
        (input_tokens - cached_tokens) * rates["input"]
        + cached_tokens * rates["cached_input"]
        + output_tokens * rates["output"]
    ) / 1_000_000
    return round(cost, 8)


def summarize_run_usage(
    *,
    events: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    chunk_count: int,
    durations: Mapping[str, float],
) -> dict[str, Any]:
    workers = (
        ("extractor", manifest.get("extractor", {}), chunk_count),
        ("resolver", manifest.get("resolver", {}), None),
        ("learner", manifest.get("learner", {}), None),
    )
    steps: list[dict[str, Any]] = [
        {
            "step": "normalization",
            "label": "Transcript normalization",
            "kind": "deterministic",
            "invoked": True,
            "request_count": 0,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "estimated_cost": 0.0,
            "duration_seconds": round(float(durations.get("normalization", 0)), 3),
        }
    ]
    for step_name, worker_manifest, expected_requests in workers:
        step_events = [event for event in events if event.get("step") == step_name]
        invoked = bool(worker_manifest.get("invoked"))
        estimated_costs = [event.get("estimated_cost") for event in step_events]
        steps.append(
            {
                "step": step_name,
                "label": {
                    "extractor": "Worker 1 · Extractor",
                    "resolver": "Worker 2 · Resolver",
                    "learner": "Worker 3 · Learner",
                }[step_name],
                "kind": "model",
                "invoked": invoked,
                "model": worker_manifest.get("model"),
                "reasoning_effort": worker_manifest.get("reasoning_effort"),
                "request_count": len(step_events) if step_events else (expected_requests if not invoked else None),
                "expected_request_count": expected_requests,
                "input_tokens": sum(int(event.get("input_tokens") or 0) for event in step_events) if step_events else None,
                "cached_input_tokens": sum(int(event.get("cached_input_tokens") or 0) for event in step_events) if step_events else None,
                "output_tokens": sum(int(event.get("output_tokens") or 0) for event in step_events) if step_events else None,
                "reasoning_tokens": sum(int(event.get("reasoning_tokens") or 0) for event in step_events) if step_events else None,
                "total_tokens": sum(int(event.get("total_tokens") or 0) for event in step_events) if step_events else None,
                "estimated_cost": round(sum(float(value) for value in estimated_costs if value is not None), 8) if step_events and all(value is not None for value in estimated_costs) else None,
                "service_tiers": sorted({str(event.get("service_tier")) for event in step_events if event.get("service_tier")}),
                "duration_seconds": round(float(durations.get(step_name, 0)), 3),
            }
        )
    steps.append(
        {
            "step": "persistence",
            "label": "Run persistence",
            "kind": "deterministic",
            "invoked": True,
            "request_count": 0,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "estimated_cost": 0.0,
            "duration_seconds": None,
        }
    )
    available_events = [event for event in events if event.get("total_tokens") is not None]
    estimated_costs = [event.get("estimated_cost") for event in available_events]
    return {
        "tracking_status": "recorded" if available_events else "not_recorded",
        "pricing_basis": "Published standard API token rates captured 2026-07-12; billed cost may differ by service tier.",
        "total_tokens": sum(int(event.get("total_tokens") or 0) for event in available_events) if available_events else None,
        "input_tokens": sum(int(event.get("input_tokens") or 0) for event in available_events) if available_events else None,
        "cached_input_tokens": sum(int(event.get("cached_input_tokens") or 0) for event in available_events) if available_events else None,
        "output_tokens": sum(int(event.get("output_tokens") or 0) for event in available_events) if available_events else None,
        "reasoning_tokens": sum(int(event.get("reasoning_tokens") or 0) for event in available_events) if available_events else None,
        "estimated_cost": round(sum(float(value) for value in estimated_costs if value is not None), 8) if available_events and all(value is not None for value in estimated_costs) else None,
        "steps": steps,
    }


def run_resolver_worker(
    *,
    context: ContentContext,
    profile: EffectiveProfile,
    route: ModelRoute,
    extraction: Mapping[str, Any],
    aliases: Mapping[str, Mapping[str, str]] | None,
    openai_api_key: str,
    timeout_seconds: float = 900.0,
    service_tier: str = "flex",
    requester: StructuredRequester | None = None,
    use_model: bool = True,
) -> dict[str, Any]:
    alias_maps = {key: dict(value) for key, value in (aliases or {}).items()}
    participant_candidates = [dict(item) for item in extraction.get("participants", [])]
    entity_candidates = [dict(item) for item in extraction.get("entities", [])]
    mentioned_people = [
        _participant_mention_entity(item)
        for item in participant_candidates
        if is_mentioned_only_participant(item)
    ]
    existing_people = {
        _normalize_key(str(item.get("raw_label") or ""))
        for item in entity_candidates
        if _controlled_value(item.get("type"), _ENTITY_TYPES, "other") == "person"
    }
    entity_candidates.extend(
        item for item in mentioned_people if _normalize_key(str(item.get("raw_label") or "")) not in existing_people
    )
    deterministic = ResolverOutput(
        participants=_dedupe_resolved_participants([
            _resolve_participant_candidate(
                item,
                context=context,
                aliases=alias_maps.get("participant", {}),
                known_values=_known_participant_values(profile.config.get("known_values", {})),
            )
            for item in participant_candidates
        ]),
        entities=[_resolve_entity_candidate(item) for item in entity_candidates],
        references=[_resolve_reference_candidate(item) for item in extraction.get("references", [])],
    )

    unresolved = ResolverOutput(
        participants=[item for item in deterministic.participants if item.resolution_status != "trusted"],
        entities=[item for item in deterministic.entities if item.resolution_status != "trusted"],
        references=[item for item in deterministic.references if item.resolution_status != "trusted"],
    )
    if not use_model or not (
        unresolved.participants or unresolved.entities or unresolved.references
    ):
        return deterministic.model_dump()

    request = requester or _request_openai_structured
    instructions = build_resolver_instructions(profile=profile)
    raw = request(
        api_key=openai_api_key,
        model=route.resolver_model,
        reasoning_effort="high",
        instructions=instructions,
        input_text=build_resolver_input(
            context=context,
            profile=profile,
            unresolved=unresolved,
            aliases=alias_maps,
        ),
        schema_name="metadata_v2_resolution",
        schema=_strict_schema(ResolverOutput),
        timeout_seconds=timeout_seconds,
        service_tier=service_tier,
    )
    try:
        model_resolution = ResolverOutput.model_validate(raw)
    except ValidationError as exc:
        raise MetadataV2Error(f"Worker 2 returned invalid resolution output: {exc}") from exc
    return _merge_resolver_outputs(deterministic, model_resolution).model_dump()


def build_resolver_instructions(*, profile: EffectiveProfile) -> str:
    modules = profile.config.get("reference_modules", [])
    return (
        "# Identity\n"
        "You are Worker 2, the canonical resolver and enrichment librarian.\n"
        "# Goal\n"
        "Resolve raw metadata candidates into canonical values while preserving the original evidence.\n"
        "# Rules\n"
        "Do not invent an identity for unnamed people. Generic roles are not names. Keep uncertain results "
        "provisional or unresolved. Distinguish explicit, normalized, alias, inferred, and externally_verified "
        "resolution bases. A model-knowledge inference is not externally verified.\n"
        "# References\n"
        f"Enabled specialized modules: {', '.join(str(item) for item in modules) or 'none'}. "
        "For scripture, return structured passage coordinates when supported by the evidence. If a transcript "
        "only alludes to a passage, retain inferred as the basis.\n"
        "# Output\nReturn every supplied unresolved candidate exactly once in the strict response schema."
    )


def build_resolver_input(
    *,
    context: ContentContext,
    profile: EffectiveProfile,
    unresolved: ResolverOutput,
    aliases: Mapping[str, Mapping[str, str]],
) -> str:
    return (
        "<document_context>\n"
        f"title: {context.title}\n"
        f"channel: {context.channel_name or 'ad hoc'}\n"
        f"category: {context.category_name or 'unassigned'}\n"
        f"approved_aliases: {json.dumps(aliases, sort_keys=True)}\n"
        f"known_values: {json.dumps(profile.config.get('known_values', {}), sort_keys=True)}\n"
        "</document_context>\n"
        "<unresolved_candidates>\n"
        f"{unresolved.model_dump_json()}\n"
        "</unresolved_candidates>"
    )


def _resolve_participant_candidate(
    item: Mapping[str, Any],
    *,
    context: ContentContext,
    aliases: Mapping[str, str],
    known_values: Sequence[str] = (),
) -> ResolvedParticipant:
    raw_name = _clean_value(item.get("raw_name"))
    role = _clean_value(item.get("role")) or None
    evidence = _clean_value(item.get("evidence")) or ""
    confidence = _bounded_confidence(item.get("confidence"))
    normalized = _normalize_key(raw_name)
    if is_mentioned_only_participant(item):
        return ResolvedParticipant(
            raw_name=raw_name,
            canonical_name=None,
            role=role or "mentioned person",
            evidence=evidence,
            confidence=confidence,
            resolution_status="rejected",
            resolution_basis="explicit",
        )
    if normalized in {_normalize_key(value) for value in _GENERIC_PARTICIPANT_NAMES}:
        return ResolvedParticipant(
            raw_name=raw_name,
            canonical_name=None,
            role=role or raw_name.lower(),
            evidence=evidence,
            confidence=1.0,
            resolution_status="rejected",
            resolution_basis="normalized",
        )

    canonical = aliases.get(normalized)
    if canonical:
        return ResolvedParticipant(
            raw_name=raw_name,
            canonical_name=canonical,
            role=role,
            evidence=evidence,
            confidence=max(confidence, TRUSTED_CONFIDENCE),
            resolution_status="trusted",
            resolution_basis="alias",
        )

    if _ambiguous_known_first_name(raw_name, known_values):
        return ResolvedParticipant(
            raw_name=raw_name,
            canonical_name=None,
            role=role,
            evidence=evidence,
            confidence=min(confidence, 0.94),
            resolution_status="provisional",
            resolution_basis="unresolved",
        )

    for known_value in known_values:
        if _looks_like_same_person(raw_name, known_value):
            return ResolvedParticipant(
                raw_name=raw_name,
                canonical_name=known_value,
                role=role,
                evidence=evidence,
                confidence=max(confidence, TRUSTED_CONFIDENCE),
                resolution_status="trusted",
                resolution_basis="normalized",
            )

    if context.channel_name and _looks_like_same_person(raw_name, context.channel_name):
        return ResolvedParticipant(
            raw_name=raw_name,
            canonical_name=context.channel_name,
            role=role or "channel host",
            evidence=evidence,
            confidence=max(confidence, TRUSTED_CONFIDENCE),
            resolution_status="trusted",
            resolution_basis="alias" if _normalize_key(raw_name) != _normalize_key(context.channel_name) else "explicit",
        )

    status = _confidence_status(confidence)
    return ResolvedParticipant(
        raw_name=raw_name,
        canonical_name=raw_name if status == "trusted" else None,
        role=role,
        evidence=evidence,
        confidence=confidence,
        resolution_status=status,
        resolution_basis="explicit" if status == "trusted" else "unresolved",
    )


def is_scripture_keyword(value: Any) -> bool:
    text = _clean_value(value)
    return bool(text and parse_scripture_reference(text))


def is_mentioned_only_participant(item: Mapping[str, Any]) -> bool:
    role = _clean_value(item.get("role")).lower()
    evidence = _clean_value(item.get("evidence")).lower()
    combined = f"{role} {evidence}"
    if re.search(r"\b(previous|prior|former|past|earlier)\b.{0,40}\bguest\b", combined):
        return True
    if "mentioned in discussion" in role or "mentioned only" in role:
        return True
    if "absent" in role or "not present" in role:
        return True
    if re.search(r"\b(do not have|don't have|is absent|was absent)\b", evidence):
        return True
    return bool(
        re.search(r"\b(recall|remember|mentioned)\b.{0,100}\b(on the show|on the podcast|as a guest)\b", evidence)
    )


def _ambiguous_known_first_name(raw_name: str, known_values: Sequence[str]) -> bool:
    raw_parts = re.findall(r"[a-z0-9]+", raw_name.lower())
    if len(raw_parts) != 1:
        return False
    matching = [
        value
        for value in known_values
        if re.findall(r"[a-z0-9]+", str(value).lower())[:1] == raw_parts
    ]
    return len(matching) > 1


def _participant_mention_entity(item: Mapping[str, Any]) -> dict[str, Any]:
    raw_name = _clean_value(item.get("raw_name"))
    role = _clean_value(item.get("role"))
    return {
        "type": "person",
        "raw_label": raw_name,
        "context": role or "Person mentioned in background; not present in this document.",
        "usage": "background",
        "evidence": _clean_value(item.get("evidence")),
        "confidence": _bounded_confidence(item.get("confidence")),
    }


def _dedupe_resolved_participants(items: Sequence[ResolvedParticipant]) -> list[ResolvedParticipant]:
    selected: dict[str, ResolvedParticipant] = {}
    order: list[str] = []
    for item in items:
        key = _normalize_key(item.canonical_name or item.raw_name)
        if not key:
            continue
        existing = selected.get(key)
        if existing is None:
            selected[key] = item
            order.append(key)
            continue
        existing_score = (existing.resolution_status == "trusted", existing.confidence)
        item_score = (item.resolution_status == "trusted", item.confidence)
        if item_score > existing_score:
            selected[key] = item
    return [selected[key] for key in order]


def _known_participant_values(known_values: Any) -> list[str]:
    if not isinstance(known_values, Mapping):
        return []
    participants = known_values.get("participants", [])
    if isinstance(participants, Mapping):
        participants = participants.values()
    if isinstance(participants, str) or not isinstance(participants, Sequence):
        return []
    return _dedupe_strings(str(value) for value in participants)


def _resolve_entity_candidate(item: Mapping[str, Any]) -> ResolvedEntity:
    raw_label = _clean_value(item.get("raw_label"))
    entity_type = _controlled_value(item.get("type"), _ENTITY_TYPES, "other")
    confidence = _bounded_confidence(item.get("confidence"))
    status = _confidence_status(confidence)
    return ResolvedEntity(
        type=entity_type,
        raw_label=raw_label,
        canonical_label=raw_label if status == "trusted" else None,
        context=_clean_value(item.get("context")),
        usage=_controlled_value(item.get("usage"), _USAGE_TYPES, "neutral"),
        evidence=_clean_value(item.get("evidence")) or "",
        confidence=confidence,
        resolution_status=status,
        resolution_basis="explicit" if status == "trusted" else "unresolved",
    )


def _resolve_reference_candidate(item: Mapping[str, Any]) -> ResolvedReference:
    raw_mention = _clean_value(item.get("raw_mention"))
    suggested_label = _clean_value(item.get("suggested_label"))
    reference_type = _controlled_value(item.get("type"), _REFERENCE_TYPES, "other")
    confidence = _bounded_confidence(item.get("confidence"))
    candidate_basis = _controlled_value(
        item.get("resolution_basis"),
        {"explicit", "normalized", "alias", "inferred", "externally_verified", "unresolved"},
        "unresolved",
    )
    locator = None
    if reference_type == "scripture" or suggested_label:
        locator = parse_scripture_reference(suggested_label or raw_mention)
    if locator:
        status = "trusted" if confidence >= TRUSTED_CONFIDENCE and candidate_basis != "inferred" else "provisional"
        return ResolvedReference(
            type="scripture",
            raw_mention=raw_mention,
            canonical_label=locator.canonical_label,
            context=_clean_value(item.get("context")),
            usage=_controlled_value(item.get("usage"), _USAGE_TYPES, "neutral"),
            evidence=_clean_value(item.get("evidence")) or "",
            confidence=confidence,
            resolution_status=status,
            resolution_basis=candidate_basis if candidate_basis != "unresolved" else "normalized",
            scripture=ScriptureResolution(**{
                key: value
                for key, value in locator.to_dict().items()
                if key not in {"canonical_label"}
            }),
        )

    status = _confidence_status(confidence)
    canonical = suggested_label or raw_mention
    return ResolvedReference(
        type=reference_type,
        raw_mention=raw_mention,
        canonical_label=canonical if status == "trusted" else None,
        context=_clean_value(item.get("context")),
        usage=_controlled_value(item.get("usage"), _USAGE_TYPES, "neutral"),
        evidence=_clean_value(item.get("evidence")) or "",
        confidence=confidence,
        resolution_status=status,
        resolution_basis="normalized" if status == "trusted" else candidate_basis,
        scripture=None,
    )


def _merge_resolver_outputs(primary: ResolverOutput, secondary: ResolverOutput) -> ResolverOutput:
    return ResolverOutput(
        participants=_merge_resolution_list(
            primary.participants,
            secondary.participants,
            key=lambda item: _normalize_key(item.raw_name),
        ),
        entities=_merge_resolution_list(
            primary.entities,
            secondary.entities,
            key=lambda item: (_normalize_key(item.type), _normalize_key(item.raw_label)),
        ),
        references=_merge_resolution_list(
            primary.references,
            secondary.references,
            key=lambda item: (_normalize_key(item.type), _normalize_key(item.raw_mention)),
        ),
    )


def _merge_resolution_list(primary: Sequence[Any], secondary: Sequence[Any], *, key: Callable[[Any], Any]) -> list[Any]:
    replacements = {key(item): item for item in secondary}
    return [replacements.get(key(item), item) for item in primary]


def _looks_like_same_person(raw_name: str, canonical_name: str) -> bool:
    raw_parts = re.findall(r"[a-z0-9]+", raw_name.lower())
    canonical_parts = re.findall(r"[a-z0-9]+", canonical_name.lower())
    if not raw_parts or not canonical_parts or raw_parts[0] != canonical_parts[0]:
        return False
    if len(canonical_parts) < 2:
        return False
    return SequenceMatcher(None, "".join(raw_parts), "".join(canonical_parts)).ratio() >= 0.72


def _confidence_status(confidence: float) -> str:
    if confidence >= TRUSTED_CONFIDENCE:
        return "trusted"
    if confidence >= PROVISIONAL_CONFIDENCE:
        return "provisional"
    return "unresolved"


def _bounded_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _clean_value(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _controlled_value(value: Any, allowed: set[str], fallback: str) -> str:
    normalized = _clean_value(value).lower().replace("-", "_").replace(" ", "_")
    return normalized if normalized in allowed else fallback


_PROFILE_SCOPE_ORDER = {
    "global": 0,
    "source": 1,
    "category": 2,
    "channel": 3,
    "channel_category": 4,
}


def load_content_context(*, engine: Engine, content_id: int) -> ContentContext:
    with engine.connect() as connection:
        metadata = MetaData()
        try:
            media = Table("media", metadata, autoload_with=connection)
            doc_content = Table("doc_content", metadata, autoload_with=connection)
            channels = Table("channels", metadata, autoload_with=connection)
            categories = Table("doc_categories", metadata, autoload_with=connection)
        except NoSuchTableError as exc:
            raise MetadataV2Error("Core ingestion tables are missing; run migrations first.") from exc

        row = (
            connection.execute(
                select(
                    doc_content.c.content_id,
                    doc_content.c.media_id,
                    doc_content.c.full_text,
                    doc_content.c.timestamped_segments,
                    media.c.source_id,
                    media.c.source_type,
                    media.c.channel_id,
                    media.c.category_id,
                    media.c.title,
                    media.c.description,
                    channels.c.channel_name,
                    categories.c.category_name,
                )
                .select_from(
                    doc_content.join(media, media.c.media_id == doc_content.c.media_id)
                    .outerjoin(channels, channels.c.channel_id == media.c.channel_id)
                    .outerjoin(categories, categories.c.category_id == media.c.category_id)
                )
                .where(doc_content.c.content_id == content_id)
            )
            .mappings()
            .first()
        )
        if not row:
            raise MetadataV2Error(f"No doc_content row found for content_id={content_id}.")
        transcript_text = str(row.get("full_text") or "").strip()
        if not transcript_text:
            raise MetadataV2Error(f"doc_content.full_text is empty for content_id={content_id}.")

        channel_id = str(row["channel_id"]) if row.get("channel_id") else None
        channel_document_count = 0
        if channel_id:
            try:
                extraction_runs = Table("extraction_runs", metadata, autoload_with=connection)
            except NoSuchTableError:
                extraction_runs = None
            if extraction_runs is not None:
                channel_document_count = int(
                    connection.execute(
                        select(func.count(func.distinct(extraction_runs.c.content_id)))
                        .select_from(
                            extraction_runs.join(media, media.c.media_id == extraction_runs.c.media_id)
                        )
                        .where(
                            media.c.channel_id == channel_id,
                            extraction_runs.c.status == "completed",
                        )
                    ).scalar_one()
                )

    segments = row.get("timestamped_segments")
    return ContentContext(
        content_id=int(row["content_id"]),
        media_id=str(row["media_id"]),
        source_id=str(row["source_id"]),
        source_type=str(row.get("source_type") or "ad_hoc"),
        channel_id=channel_id,
        channel_name=str(row["channel_name"]) if row.get("channel_name") else None,
        category_id=int(row["category_id"]) if row.get("category_id") is not None else None,
        category_name=str(row["category_name"]) if row.get("category_name") else None,
        title=str(row["title"]),
        description=str(row["description"]) if row.get("description") else None,
        transcript_text=transcript_text,
        timestamped_segments=[dict(item) for item in segments] if isinstance(segments, Sequence) else [],
        channel_document_count=channel_document_count,
    )


def load_effective_profile(*, engine: Engine, context: ContentContext) -> EffectiveProfile:
    with engine.connect() as connection:
        tables = _reflect_v2_tables(connection, names=("extraction_profiles", "extraction_profile_versions"))
        profiles = tables["extraction_profiles"]
        versions = tables["extraction_profile_versions"]
        selectors = [profiles.c.scope == "global"]
        selectors.append(
            and_(profiles.c.scope == "source", profiles.c.source_id == context.source_id)
        )
        if context.category_id is not None:
            selectors.append(
                and_(profiles.c.scope == "category", profiles.c.category_id == context.category_id)
            )
        if context.channel_id:
            selectors.append(
                and_(profiles.c.scope == "channel", profiles.c.channel_id == context.channel_id)
            )
            if context.category_id is not None:
                selectors.append(
                    and_(
                        profiles.c.scope == "channel_category",
                        profiles.c.channel_id == context.channel_id,
                        profiles.c.category_id == context.category_id,
                    )
                )

        rows = (
            connection.execute(
                select(
                    profiles.c.profile_id,
                    profiles.c.scope,
                    profiles.c.learning_mode,
                    versions.c.profile_version_id,
                    versions.c.config,
                )
                .select_from(profiles.join(versions, versions.c.profile_id == profiles.c.profile_id))
                .where(profiles.c.is_active.is_(True), versions.c.status == "active", or_(*selectors))
            )
            .mappings()
            .all()
        )

    config = default_profile_config()
    ordered = sorted(rows, key=lambda row: _PROFILE_SCOPE_ORDER.get(str(row["scope"]), 99))
    learning_mode = "autopilot"
    for row in ordered:
        _deep_merge(config, dict(row.get("config") or {}))
        learning_mode = str(row.get("learning_mode") or learning_mode)
    return EffectiveProfile(
        profile_version_id=int(ordered[-1]["profile_version_id"]) if ordered else None,
        profile_ids=tuple(int(row["profile_id"]) for row in ordered),
        version_ids=tuple(int(row["profile_version_id"]) for row in ordered),
        learning_mode=learning_mode,
        config=config,
    )


def load_value_aliases(*, engine: Engine, context: ContentContext) -> dict[str, dict[str, str]]:
    with engine.connect() as connection:
        table = _reflect_v2_tables(connection, names=("metadata_value_aliases",))["metadata_value_aliases"]
        scope_match = and_(
            or_(table.c.source_id.is_(None), table.c.source_id == context.source_id),
            or_(table.c.channel_id.is_(None), table.c.channel_id == context.channel_id),
            or_(table.c.category_id.is_(None), table.c.category_id == context.category_id),
        )
        rows = (
            connection.execute(
                select(
                    table.c.dimension,
                    table.c.alias_normalized,
                    table.c.canonical_value,
                    table.c.is_blocked,
                ).where(table.c.trust_status == "trusted", scope_match)
            )
            .mappings()
            .all()
        )
    aliases: dict[str, dict[str, str]] = defaultdict(dict)
    for row in rows:
        if row.get("is_blocked"):
            continue
        aliases[str(row["dimension"])][str(row["alias_normalized"])] = str(row["canonical_value"])
    return dict(aliases)


def _deep_merge(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value


def _reflect_v2_tables(connection: Connection, *, names: Sequence[str]) -> dict[str, Table]:
    metadata = MetaData()
    tables: dict[str, Table] = {}
    try:
        for name in names:
            tables[name] = Table(name, metadata, autoload_with=connection)
    except NoSuchTableError as exc:
        raise MetadataV2Error(
            "Metadata V2 tables are missing. Run `poetry run alembic upgrade head` first."
        ) from exc
    return tables


def is_learner_due(*, context: ContentContext, profile: EffectiveProfile) -> bool:
    if profile.learning_mode in {"off", "observe"} or not context.channel_id:
        return False
    routing = profile.config.get("model_routing", {})
    bootstrap_documents = int(routing.get("bootstrap_documents", 3))
    audit_interval = max(1, int(routing.get("audit_interval", 10)))
    return (
        context.channel_document_count < bootstrap_documents
        or (context.channel_document_count + 1) % audit_interval == 0
    )


def load_recent_resolved_runs(
    *,
    engine: Engine,
    context: ContentContext,
    limit: int = 10,
) -> list[dict[str, Any]]:
    if not context.channel_id:
        return []
    with engine.connect() as connection:
        tables = _reflect_v2_tables(connection, names=("extraction_runs", "media"))
        runs = tables["extraction_runs"]
        media = tables["media"]
        rows = (
            connection.execute(
                select(
                    runs.c.run_id,
                    runs.c.content_id,
                    runs.c.release_version,
                    runs.c.resolved_output,
                    runs.c.completed_at,
                )
                .select_from(runs.join(media, media.c.media_id == runs.c.media_id))
                .where(
                    media.c.channel_id == context.channel_id,
                    runs.c.status == "completed",
                    runs.c.resolved_output.is_not(None),
                )
                .order_by(runs.c.completed_at.desc())
                .limit(limit)
            )
            .mappings()
            .all()
        )
    return [dict(row) for row in rows]


def run_learner_worker(
    *,
    context: ContentContext,
    profile: EffectiveProfile,
    route: ModelRoute,
    recent_runs: Sequence[Mapping[str, Any]],
    current_resolution: Mapping[str, Any],
    openai_api_key: str,
    timeout_seconds: float = 900.0,
    service_tier: str = "flex",
    requester: StructuredRequester | None = None,
) -> dict[str, Any]:
    if not is_learner_due(context=context, profile=profile):
        return LearnerOutput(suggestions=[], prompt_additions=[], blocked_values=[]).model_dump()

    request = requester or _request_openai_structured
    instructions = build_learner_instructions()
    payload = {
        "channel_id": context.channel_id,
        "channel_name": context.channel_name,
        "category": context.category_name,
        "learning_mode": profile.learning_mode,
        "current_resolution": current_resolution,
        "recent_runs": [
            {
                "content_id": row.get("content_id"),
                "release_version": row.get("release_version"),
                "resolved_output": row.get("resolved_output"),
            }
            for row in recent_runs
        ],
    }
    raw = request(
        api_key=openai_api_key,
        model=route.learner_model,
        reasoning_effort="high",
        instructions=instructions,
        input_text=json.dumps(payload, default=str, sort_keys=True),
        schema_name="metadata_v2_learning",
        schema=_strict_schema(LearnerOutput),
        timeout_seconds=timeout_seconds,
        service_tier=service_tier,
    )
    try:
        return LearnerOutput.model_validate(raw).model_dump()
    except ValidationError as exc:
        raise MetadataV2Error(f"Worker 3 returned invalid learning output: {exc}") from exc


def build_learner_instructions() -> str:
    """Return the exact Worker 3 instruction block used at runtime."""
    return (
        "# Identity\nYou are Worker 3, the channel learner and evaluator.\n"
        "# Goal\nFind repeatable channel knowledge that can reduce future extraction cost without converting "
        "repeated errors into truth.\n"
        "# Memory\nUse only the supplied versioned run history. Treat one extraction as an observation, not a fact.\n"
        "# Rules\nPropose alias or value learning only when supported by authoritative channel identity or at "
        "least three distinct documents without contradiction. Generic participant roles should be blocked. "
        "Prompt and schema improvements are proposals only and must never be activated by this worker.\n"
        "# Safety\nDo not alter document metadata, publish profiles, or infer unnamed identities.\n"
        "# Output\nReturn only the strict learning-suggestion schema."
    )


def get_metadata_v2_prompt_inspection() -> dict[str, Any]:
    """Return V2 prompt contracts with visible placeholders instead of document data."""
    profile = EffectiveProfile(
        profile_version_id=None,
        profile_ids=(),
        version_ids=(),
        learning_mode="{{ effective_learning_mode }}",
        config=default_profile_config(),
    )
    extractor_input = """<document_context>
title: {{ media.title }}
source: {{ media.source_id }}
channel: {{ channel.name | default('ad hoc') }}
category: {{ category.name | default('unassigned') }}
description: {{ media.description }}
known_values: {{ effective_profile.known_values | json }}
blocked_values: {{ effective_profile.blocked_values | json }}
</document_context>
<transcript_chunk index="{{ chunk.index }}" count="{{ chunk_count }}"
start_seconds="{{ chunk.start_seconds }}" end_seconds="{{ chunk.end_seconds }}">
{{ chunk.text }}
</transcript_chunk>"""
    resolver_input = """<document_context>
title: {{ media.title }}
channel: {{ channel.name | default('ad hoc') }}
category: {{ category.name | default('unassigned') }}
approved_aliases: {{ scoped_aliases | json }}
known_values: {{ effective_profile.known_values | json }}
</document_context>
<unresolved_candidates>
{{ deterministic_resolution.unresolved | json }}
</unresolved_candidates>"""
    learner_input = json.dumps(
        {
            "channel_id": "{{ channel.id }}",
            "channel_name": "{{ channel.name }}",
            "category": "{{ category.name }}",
            "learning_mode": "{{ effective_learning_mode }}",
            "current_resolution": "{{ current_resolution | json }}",
            "recent_runs": "{{ recent_channel_runs | json }}",
        },
        indent=2,
    )
    return {
        "name": "Metadata Extraction V2",
        "schema_version": V2_RELEASE_VERSION,
        "default_model": V2_STABLE_MODEL,
        "enhanced_model": V2_ENHANCED_MODEL,
        "transcript_max_chars": DEFAULT_CHUNK_MAX_CHARS,
        "prompt_contracts": {
            "extractor": {
                "instructions": build_extractor_instructions(profile=profile),
                "input_template": extractor_input,
                "json_schema": _strict_schema(ChunkExtraction),
            },
            "resolver": {
                "instructions": build_resolver_instructions(profile=profile),
                "input_template": resolver_input,
                "json_schema": _strict_schema(ResolverOutput),
            },
            "learner": {
                "instructions": build_learner_instructions(),
                "input_template": learner_input,
                "json_schema": _strict_schema(LearnerOutput),
            },
        },
    }


def preview_profile_instruction_contracts(config: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Render base and effective worker contracts through the runtime prompt builders."""
    if not isinstance(config, Mapping):
        raise ValueError("Effective profile config must be a JSON object.")
    base_config = default_profile_config()
    effective_config = default_profile_config()
    _deep_merge(effective_config, dict(config))
    base_profile = EffectiveProfile(
        profile_version_id=None,
        profile_ids=(),
        version_ids=(),
        learning_mode="autopilot",
        config=base_config,
    )
    effective_profile = EffectiveProfile(
        profile_version_id=None,
        profile_ids=(),
        version_ids=(),
        learning_mode="autopilot",
        config=effective_config,
    )
    workers = [
        {
            "key": "extractor",
            "label": "Worker 1 · Extractor",
            "role": "Evidence-bearing candidate extraction",
            "base_instructions": build_extractor_instructions(profile=base_profile),
            "effective_instructions": build_extractor_instructions(profile=effective_profile),
            "profile_inputs": [
                "coverage",
                "reference_modules",
                "category_schema",
                "prompt_instructions",
                "known_values",
                "blocked_values",
            ],
        },
        {
            "key": "resolver",
            "label": "Worker 2 · Resolver",
            "role": "Rules-first canonical resolution with optional model help",
            "base_instructions": build_resolver_instructions(profile=base_profile),
            "effective_instructions": build_resolver_instructions(profile=effective_profile),
            "profile_inputs": ["reference_modules", "known_values", "metadata_value_aliases"],
        },
        {
            "key": "learner",
            "label": "Worker 3 · Learner",
            "role": "Conditional channel learning proposals",
            "base_instructions": build_learner_instructions(),
            "effective_instructions": build_learner_instructions(),
            "profile_inputs": ["learning_mode", "model_routing.audit_interval", "recent_channel_runs"],
        },
    ]
    return {
        "workers": workers,
        "prompt_instruction_behavior": {
            "applies_to": "extractor",
            "merge": "most_specific_replaces",
            "detail": "prompt_instructions is a scalar profile field; the most specific configured scope replaces broader values for Worker 1.",
        },
    }


def run_metadata_v2_for_content(
    *,
    engine: Engine,
    content_id: int,
    openai_api_key: str,
    persist_run: bool = True,
    use_resolver_model: bool = True,
    use_learner_model: bool = True,
    extractor_model: str | None = None,
    resolver_model: str | None = None,
    learner_model: str | None = None,
    service_tier: str = "flex",
    timeout_seconds: float = 900.0,
    requester: StructuredRequester | None = None,
) -> dict[str, Any]:
    context = load_content_context(engine=engine, content_id=content_id)
    profile = load_effective_profile(engine=engine, context=context)
    aliases = load_value_aliases(engine=engine, context=context)
    route = determine_model_route(
        context=context,
        profile=profile,
        extractor_override=extractor_model,
        resolver_override=resolver_model,
        learner_override=learner_model,
    )
    manifest = build_release_manifest(
        profile=profile,
        route=route,
        resolver_model_enabled=use_resolver_model,
        learner_model_enabled=use_learner_model,
    )
    run_id = _create_run(
        engine=engine,
        context=context,
        profile=profile,
        route=route,
        manifest=manifest,
    ) if persist_run else None

    usage_events: list[dict[str, Any]] = []
    durations: dict[str, float] = {}
    collector_token = _USAGE_COLLECTOR.set(usage_events)
    try:
        step_token = _USAGE_STEP.set("extractor")
        started = time.perf_counter()
        extraction, analysis_text, segment_map, chunks = run_extractor_worker(
            context=context,
            profile=profile,
            route=route,
            openai_api_key=openai_api_key,
            timeout_seconds=timeout_seconds,
            requester=requester,
        )
        durations["extractor"] = time.perf_counter() - started
        _USAGE_STEP.reset(step_token)

        step_token = _USAGE_STEP.set("resolver")
        started = time.perf_counter()
        resolution = run_resolver_worker(
            context=context,
            profile=profile,
            route=route,
            extraction=extraction,
            aliases=aliases,
            openai_api_key=openai_api_key,
            timeout_seconds=timeout_seconds,
            service_tier=service_tier,
            requester=requester,
            use_model=use_resolver_model,
        )
        durations["resolver"] = time.perf_counter() - started
        _USAGE_STEP.reset(step_token)

        recent_runs = load_recent_resolved_runs(engine=engine, context=context) if use_learner_model else []
        step_token = _USAGE_STEP.set("learner")
        started = time.perf_counter()
        learning = run_learner_worker(
            context=context,
            profile=profile,
            route=route,
            recent_runs=recent_runs,
            current_resolution=resolution,
            openai_api_key=openai_api_key,
            timeout_seconds=timeout_seconds,
            service_tier=service_tier,
            requester=requester,
        ) if use_learner_model else LearnerOutput(
            suggestions=[], prompt_additions=[], blocked_values=[]
        ).model_dump()
        durations["learner"] = time.perf_counter() - started
        _USAGE_STEP.reset(step_token)

        usage_summary = summarize_run_usage(
            events=usage_events,
            manifest=manifest,
            chunk_count=len(chunks),
            durations=durations,
        )
        manifest_with_usage = {**manifest, "usage": usage_summary}

        if persist_run and run_id is not None:
            _persist_completed_run(
                engine=engine,
                run_id=run_id,
                context=context,
                profile=profile,
                extraction=extraction,
                resolution=resolution,
                learning=learning,
                manifest=manifest_with_usage,
                analysis_text=analysis_text,
                segment_map=segment_map,
                chunk_count=len(chunks),
            )
    except Exception as exc:
        if persist_run and run_id is not None:
            _mark_run_failed(engine=engine, run_id=run_id, error=str(exc))
        raise
    finally:
        _USAGE_COLLECTOR.reset(collector_token)

    return {
        "run_id": run_id,
        "persisted": persist_run,
        "release_version": V2_RELEASE_VERSION,
        "content_id": content_id,
        "media_id": context.media_id,
        "run_mode": route.run_mode,
        "models": {
            "extractor": route.extractor_model,
            "resolver": route.resolver_model,
            "learner": route.learner_model,
        },
        "profile_version_ids": list(profile.version_ids),
        "manifest_hash": manifest["manifest_hash"],
        "chunk_count": len(chunks),
        "input_characters": len(context.transcript_text),
        "usage": usage_summary,
        "extraction": extraction,
        "resolution": resolution,
        "learning": learning,
    }


def build_release_manifest(
    *,
    profile: EffectiveProfile,
    route: ModelRoute,
    resolver_model_enabled: bool = True,
    learner_model_enabled: bool = True,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "pipeline_version": V2_RELEASE_VERSION,
        "category_schema_version": str(profile.config.get("schema_version") or "2.0"),
        "profile_version_ids": list(profile.version_ids),
        "normalizer_version": V2_NORMALIZER_VERSION,
        "extractor": {
            "provider": V2_PROVIDER,
            "model": route.extractor_model,
            "reasoning_effort": route.extractor_reasoning,
            "invoked": True,
        },
        "resolver": {
            "provider": V2_PROVIDER,
            "model": route.resolver_model,
            "reasoning_effort": "high",
            "invoked": resolver_model_enabled,
        },
        "learner": {
            "provider": V2_PROVIDER,
            "model": route.learner_model,
            "reasoning_effort": "high",
            "invoked": learner_model_enabled,
        },
        "learning_mode": profile.learning_mode,
        "reference_modules": list(profile.config.get("reference_modules", [])),
        "coverage": dict(profile.config.get("coverage", {})),
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest["manifest_hash"] = hashlib.sha256(encoded).hexdigest()
    return manifest


def _create_run(
    *,
    engine: Engine,
    context: ContentContext,
    profile: EffectiveProfile,
    route: ModelRoute,
    manifest: Mapping[str, Any],
) -> int:
    with engine.begin() as connection:
        table = _reflect_v2_tables(connection, names=("extraction_runs",))["extraction_runs"]
        result = connection.execute(
            insert(table).values(
                content_id=context.content_id,
                media_id=context.media_id,
                profile_version_id=profile.profile_version_id,
                release_version=V2_RELEASE_VERSION,
                run_mode=route.run_mode,
                status="running",
                extractor_model=route.extractor_model,
                resolver_model=route.resolver_model,
                learner_model=route.learner_model,
                manifest=dict(manifest),
                input_characters=len(context.transcript_text),
            )
        )
        if not result.inserted_primary_key:
            raise MetadataPersistenceError("Failed to create Metadata V2 extraction run.")
        return int(result.inserted_primary_key[0])


def _persist_completed_run(
    *,
    engine: Engine,
    run_id: int,
    context: ContentContext,
    profile: EffectiveProfile,
    extraction: Mapping[str, Any],
    resolution: Mapping[str, Any],
    learning: Mapping[str, Any],
    manifest: Mapping[str, Any],
    analysis_text: str,
    segment_map: Sequence[Mapping[str, Any]],
    chunk_count: int,
) -> None:
    names = (
        "extraction_runs",
        "transcript_analysis_versions",
        "extraction_observations",
        "doc_reference_mentions",
        "metadata_learning_events",
        "metadata_value_aliases",
        "extraction_profiles",
        "extraction_profile_versions",
    )
    with engine.begin() as connection:
        tables = _reflect_v2_tables(connection, names=names)
        connection.execute(
            update(tables["extraction_runs"])
            .where(tables["extraction_runs"].c.run_id == run_id)
            .values(
                status="completed",
                raw_output=dict(extraction),
                resolved_output=dict(resolution),
                manifest=dict(manifest),
                chunk_count=chunk_count,
                completed_at=func.current_timestamp(),
            )
        )
        connection.execute(
            insert(tables["transcript_analysis_versions"]).values(
                content_id=context.content_id,
                run_id=run_id,
                normalizer_version=V2_NORMALIZER_VERSION,
                analysis_text=analysis_text,
                segment_map=[dict(item) for item in segment_map],
            )
        )
        _persist_observations(
            connection=connection,
            tables=tables,
            run_id=run_id,
            context=context,
            extraction=extraction,
            resolution=resolution,
        )
        _persist_learning_output(
            connection=connection,
            tables=tables,
            run_id=run_id,
            context=context,
            profile=profile,
            learning=learning,
        )
        _apply_corroborated_aliases(
            connection=connection,
            tables=tables,
            run_id=run_id,
            context=context,
            profile=profile,
            resolution=resolution,
        )


def _mark_run_failed(*, engine: Engine, run_id: int, error: str) -> None:
    try:
        with engine.begin() as connection:
            table = _reflect_v2_tables(connection, names=("extraction_runs",))["extraction_runs"]
            connection.execute(
                update(table)
                .where(table.c.run_id == run_id)
                .values(status="failed", error=error[:4000], completed_at=func.current_timestamp())
            )
    except (SQLAlchemyError, MetadataV2Error):
        return


def _persist_observations(
    *,
    connection: Connection,
    tables: Mapping[str, Table],
    run_id: int,
    context: ContentContext,
    extraction: Mapping[str, Any],
    resolution: Mapping[str, Any],
) -> None:
    observations = tables["extraction_observations"]
    mentions = tables["doc_reference_mentions"]

    def add_observation(
        *,
        dimension: str,
        raw_value: str,
        normalized_value: str | None,
        canonical_value: str | None,
        evidence: str | None,
        confidence: float,
        status: str,
        basis: str,
        details: Mapping[str, Any],
    ) -> None:
        start_seconds, end_seconds = _find_evidence_time(evidence, context.timestamped_segments)
        connection.execute(
            insert(observations).values(
                run_id=run_id,
                content_id=context.content_id,
                dimension=dimension,
                raw_value=raw_value[:512],
                normalized_value=normalized_value[:512] if normalized_value else None,
                canonical_value=canonical_value[:512] if canonical_value else None,
                evidence_text=evidence,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                confidence=confidence,
                resolution_status=_controlled_value(
                    status,
                    {"observed", "provisional", "trusted", "rejected", "unresolved"},
                    "unresolved",
                ),
                resolution_basis=_controlled_value(
                    basis,
                    {"explicit", "normalized", "alias", "inferred", "externally_verified", "unresolved"},
                    "unresolved",
                ),
                details=dict(details),
            )
        )

    category = _clean_value(extraction.get("category"))
    if category:
        add_observation(
            dimension="category",
            raw_value=category,
            normalized_value=_normalize_key(category),
            canonical_value=category,
            evidence=None,
            confidence=0.95,
            status="observed",
            basis="explicit",
            details={},
        )
    for dimension, values in (
        ("topic", extraction.get("topics", [])),
        ("keyword", extraction.get("keywords", [])),
    ):
        for value in values:
            text = _clean_value(value)
            if not text:
                continue
            add_observation(
                dimension=dimension,
                raw_value=text,
                normalized_value=_normalize_key(text),
                canonical_value=text,
                evidence=None,
                confidence=0.95,
                status="observed",
                basis="explicit",
                details={},
            )

    for item in resolution.get("participants", []):
        raw_value = _clean_value(item.get("raw_name"))
        add_observation(
            dimension="participant",
            raw_value=raw_value,
            normalized_value=_normalize_key(raw_value),
            canonical_value=_clean_value(item.get("canonical_name")) or None,
            evidence=_clean_value(item.get("evidence")) or None,
            confidence=_bounded_confidence(item.get("confidence")),
            status=_clean_value(item.get("resolution_status")),
            basis=_clean_value(item.get("resolution_basis")),
            details=dict(item),
        )
    for item in resolution.get("entities", []):
        raw_value = _clean_value(item.get("raw_label"))
        add_observation(
            dimension="entity",
            raw_value=raw_value,
            normalized_value=_normalize_key(raw_value),
            canonical_value=_clean_value(item.get("canonical_label")) or None,
            evidence=_clean_value(item.get("evidence")) or None,
            confidence=_bounded_confidence(item.get("confidence")),
            status=_clean_value(item.get("resolution_status")),
            basis=_clean_value(item.get("resolution_basis")),
            details=dict(item),
        )
    for item in resolution.get("references", []):
        raw_value = _clean_value(item.get("raw_mention"))
        evidence = _clean_value(item.get("evidence")) or None
        confidence = _bounded_confidence(item.get("confidence"))
        basis = _clean_value(item.get("resolution_basis"))
        add_observation(
            dimension="reference",
            raw_value=raw_value,
            normalized_value=_normalize_key(raw_value),
            canonical_value=_clean_value(item.get("canonical_label")) or None,
            evidence=evidence,
            confidence=confidence,
            status=_clean_value(item.get("resolution_status")),
            basis=basis,
            details=dict(item),
        )
        start_seconds, end_seconds = _find_evidence_time(evidence, context.timestamped_segments)
        connection.execute(
            insert(mentions).values(
                content_id=context.content_id,
                reference_id=None,
                run_id=run_id,
                raw_mention=raw_value[:512],
                evidence_text=evidence,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                resolution_basis=_controlled_value(
                    basis,
                    {"explicit", "normalized", "alias", "inferred", "externally_verified", "unresolved"},
                    "unresolved",
                ),
                confidence=confidence,
            )
        )


def _persist_learning_output(
    *,
    connection: Connection,
    tables: Mapping[str, Table],
    run_id: int,
    context: ContentContext,
    profile: EffectiveProfile,
    learning: Mapping[str, Any],
) -> None:
    events = tables["metadata_learning_events"]
    profile_id = profile.profile_ids[-1] if profile.profile_ids else None
    for suggestion in learning.get("suggestions", []):
        connection.execute(
            insert(events).values(
                profile_id=profile_id,
                run_id=run_id,
                dimension=_clean_value(suggestion.get("dimension")) or "profile",
                action=_clean_value(suggestion.get("action")) or "propose",
                raw_value=_clean_value(suggestion.get("raw_value")) or None,
                canonical_value=_clean_value(suggestion.get("canonical_value")) or None,
                confidence=_bounded_confidence(suggestion.get("confidence")),
                reason=_clean_value(suggestion.get("reason")) or "Worker 3 proposal",
                status="proposed",
                details=dict(suggestion),
            )
        )

    prompt_additions = [
        _clean_value(value) for value in learning.get("prompt_additions", []) if _clean_value(value)
    ]
    blocked_values = [
        _clean_value(value) for value in learning.get("blocked_values", []) if _clean_value(value)
    ]
    if context.channel_id and (prompt_additions or blocked_values):
        _ensure_channel_profile_draft(
            connection=connection,
            tables=tables,
            context=context,
            profile=profile,
            prompt_additions=prompt_additions,
            blocked_values=blocked_values,
        )


def _apply_corroborated_aliases(
    *,
    connection: Connection,
    tables: Mapping[str, Table],
    run_id: int,
    context: ContentContext,
    profile: EffectiveProfile,
    resolution: Mapping[str, Any],
) -> None:
    if profile.learning_mode not in {"assisted", "autopilot"} or not context.channel_id:
        return
    aliases = tables["metadata_value_aliases"]
    observations = tables["extraction_observations"]
    events = tables["metadata_learning_events"]
    threshold = int(profile.config.get("confidence", {}).get("alias_documents", ALIAS_DOCUMENT_THRESHOLD))
    profile_id = profile.profile_ids[-1] if profile.profile_ids else None

    for item in resolution.get("participants", []):
        raw_name = _clean_value(item.get("raw_name"))
        canonical = _clean_value(item.get("canonical_name"))
        if (
            not raw_name
            or not canonical
            or _normalize_key(raw_name) == _normalize_key(canonical)
            or _clean_value(item.get("resolution_status")) != "trusted"
        ):
            continue
        raw_key = _normalize_key(raw_name)
        evidence_count = int(
            connection.execute(
                select(func.count(func.distinct(observations.c.content_id))).where(
                    observations.c.dimension == "participant",
                    observations.c.normalized_value == raw_key,
                    observations.c.canonical_value == canonical,
                    observations.c.resolution_status == "trusted",
                )
            ).scalar_one()
        )
        authoritative = bool(
            context.channel_name and _normalize_key(canonical) == _normalize_key(context.channel_name)
        )
        trusted = authoritative or evidence_count >= threshold
        status = "trusted" if trusted and profile.learning_mode == "autopilot" else "provisional"
        existing = (
            connection.execute(
                select(aliases.c.alias_id).where(
                    aliases.c.dimension == "participant",
                    aliases.c.alias_normalized == raw_key,
                    aliases.c.channel_id == context.channel_id,
                )
            )
            .mappings()
            .first()
        )
        values = {
            "canonical_value": canonical,
            "alias_text": raw_name,
            "confidence": _bounded_confidence(item.get("confidence")),
            "trust_status": status,
            "evidence_count": max(1, evidence_count),
            "updated_at": func.current_timestamp(),
        }
        if existing:
            connection.execute(
                update(aliases).where(aliases.c.alias_id == existing["alias_id"]).values(**values)
            )
        else:
            connection.execute(
                insert(aliases).values(
                    dimension="participant",
                    alias_normalized=raw_key,
                    source_id=context.source_id,
                    channel_id=context.channel_id,
                    category_id=context.category_id,
                    is_blocked=False,
                    created_by="learner",
                    **values,
                )
            )
        connection.execute(
            insert(events).values(
                profile_id=profile_id,
                run_id=run_id,
                dimension="participant",
                action="alias_promoted" if status == "trusted" else "alias_observed",
                raw_value=raw_name,
                canonical_value=canonical,
                confidence=_bounded_confidence(item.get("confidence")),
                reason=(
                    "Canonical value matches authoritative channel identity."
                    if authoritative
                    else f"Observed in {evidence_count} distinct documents."
                ),
                status="applied" if status == "trusted" else "proposed",
                details={"evidence_count": evidence_count, "authoritative": authoritative},
                applied_at=func.current_timestamp() if status == "trusted" else None,
            )
        )


def _ensure_channel_profile_draft(
    *,
    connection: Connection,
    tables: Mapping[str, Table],
    context: ContentContext,
    profile: EffectiveProfile,
    prompt_additions: Sequence[str],
    blocked_values: Sequence[str],
) -> None:
    profiles = tables["extraction_profiles"]
    versions = tables["extraction_profile_versions"]
    profile_row = (
        connection.execute(
            select(profiles.c.profile_id).where(
                profiles.c.scope == "channel",
                profiles.c.channel_id == context.channel_id,
                profiles.c.is_active.is_(True),
            )
        )
        .mappings()
        .first()
    )
    if profile_row:
        channel_profile_id = int(profile_row["profile_id"])
    else:
        result = connection.execute(
            insert(profiles).values(
                profile_name=f"{context.channel_name or context.channel_id} Metadata V2",
                scope="channel",
                source_id=context.source_id,
                channel_id=context.channel_id,
                learning_mode=profile.learning_mode,
                is_active=True,
            )
        )
        channel_profile_id = int(result.inserted_primary_key[0])

    existing_draft = connection.execute(
        select(versions.c.profile_version_id).where(
            versions.c.profile_id == channel_profile_id,
            versions.c.status == "draft",
        )
    ).first()
    if existing_draft:
        return

    max_version = connection.execute(
        select(func.max(versions.c.version)).where(versions.c.profile_id == channel_profile_id)
    ).scalar_one()
    config = json.loads(json.dumps(profile.config))
    additions = "\n".join(prompt_additions)
    current_instructions = str(config.get("prompt_instructions") or "").strip()
    config["prompt_instructions"] = "\n".join(
        value for value in (current_instructions, additions) if value
    )
    blocked = config.setdefault("blocked_values", {}).setdefault("participants", [])
    config["blocked_values"]["participants"] = _dedupe_strings([*blocked, *blocked_values])
    connection.execute(
        insert(versions).values(
            profile_id=channel_profile_id,
            version=int(max_version or 0) + 1,
            status="draft",
            config=config,
        )
    )


def _find_evidence_time(
    evidence: str | None,
    segments: Sequence[Mapping[str, Any]],
) -> tuple[float | None, float | None]:
    if not evidence:
        return None, None
    evidence_key = _normalize_key(evidence)
    if not evidence_key:
        return None, None
    for segment in segments:
        segment_text = _clean_value(segment.get("text"))
        segment_key = _normalize_key(segment_text)
        if segment_key and (segment_key in evidence_key or evidence_key in segment_key):
            start = _optional_float(segment.get("start_seconds"))
            duration = _optional_float(segment.get("duration_seconds"))
            end = start + duration if start is not None and duration is not None else start
            return start, end
    return None, None


def create_profile_draft(
    *,
    engine: Engine,
    profile_name: str,
    scope: str,
    source_id: str | None = None,
    channel_id: str | None = None,
    category_id: int | None = None,
    learning_mode: str = "autopilot",
    enable_scripture: bool = False,
    prompt_instructions: str = "",
    category_schema: Sequence[Mapping[str, Any]] | None = None,
    known_values: Mapping[str, Any] | None = None,
    blocked_values: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if scope not in _PROFILE_SCOPE_ORDER:
        raise ValueError(f"Unsupported profile scope: {scope}")
    if learning_mode not in {"off", "observe", "assisted", "autopilot"}:
        raise ValueError(f"Unsupported learning mode: {learning_mode}")
    _validate_profile_selector(
        scope=scope,
        source_id=source_id,
        channel_id=channel_id,
        category_id=category_id,
    )

    with engine.begin() as connection:
        tables = _reflect_v2_tables(
            connection,
            names=("extraction_profiles", "extraction_profile_versions"),
        )
        profiles = tables["extraction_profiles"]
        versions = tables["extraction_profile_versions"]
        selector = [profiles.c.scope == scope]
        selector.append(
            profiles.c.source_id == source_id if source_id is not None else profiles.c.source_id.is_(None)
        )
        selector.append(
            profiles.c.channel_id == channel_id if channel_id is not None else profiles.c.channel_id.is_(None)
        )
        selector.append(
            profiles.c.category_id == category_id if category_id is not None else profiles.c.category_id.is_(None)
        )
        row = connection.execute(select(profiles.c.profile_id).where(*selector)).mappings().first()
        if row:
            profile_id = int(row["profile_id"])
            connection.execute(
                update(profiles)
                .where(profiles.c.profile_id == profile_id)
                .values(
                    profile_name=profile_name,
                    learning_mode=learning_mode,
                    is_active=True,
                    updated_at=func.current_timestamp(),
                )
            )
        else:
            result = connection.execute(
                insert(profiles).values(
                    profile_name=profile_name,
                    scope=scope,
                    source_id=source_id,
                    channel_id=channel_id,
                    category_id=category_id,
                    learning_mode=learning_mode,
                    is_active=True,
                )
            )
            profile_id = int(result.inserted_primary_key[0])

        active_config = connection.execute(
            select(versions.c.config)
            .where(versions.c.profile_id == profile_id, versions.c.status == "active")
            .order_by(versions.c.version.desc())
            .limit(1)
        ).scalar_one_or_none()
        if active_config is not None:
            config = dict(active_config)
        elif scope == "global":
            config = default_profile_config()
        else:
            config = {}
        if enable_scripture:
            config["reference_modules"] = _dedupe_strings(
                [*config.get("reference_modules", []), "scripture"]
            )
            config.setdefault("coverage", {})["references"] = "exhaustive"
        if prompt_instructions.strip():
            config["prompt_instructions"] = prompt_instructions.strip()
        if category_schema is not None:
            config["category_schema"] = [dict(item) for item in category_schema]
        if known_values is not None:
            config["known_values"] = dict(known_values)
        if blocked_values is not None:
            config["blocked_values"] = dict(blocked_values)

        max_version = connection.execute(
            select(func.max(versions.c.version)).where(versions.c.profile_id == profile_id)
        ).scalar_one()
        version = int(max_version or 0) + 1
        result = connection.execute(
            insert(versions).values(
                profile_id=profile_id,
                version=version,
                status="draft",
                config=config,
            )
        )
        profile_version_id = int(result.inserted_primary_key[0])
    return {
        "profile_id": profile_id,
        "profile_version_id": profile_version_id,
        "version": version,
        "status": "draft",
        "config": config,
    }


def publish_profile_version(*, engine: Engine, profile_version_id: int) -> dict[str, Any]:
    with engine.begin() as connection:
        versions = _reflect_v2_tables(connection, names=("extraction_profile_versions",))[
            "extraction_profile_versions"
        ]
        row = (
            connection.execute(
                select(
                    versions.c.profile_version_id,
                    versions.c.profile_id,
                    versions.c.version,
                    versions.c.status,
                ).where(versions.c.profile_version_id == profile_version_id)
            )
            .mappings()
            .first()
        )
        if not row:
            raise MetadataV2Error(f"Profile version {profile_version_id} was not found.")
        if row["status"] not in {"draft", "testing"}:
            raise MetadataV2Error(
                f"Profile version {profile_version_id} cannot be published from status={row['status']}."
            )
        profile_id = int(row["profile_id"])
        connection.execute(
            update(versions)
            .where(versions.c.profile_id == profile_id, versions.c.status == "active")
            .values(status="retired", retired_at=func.current_timestamp())
        )
        connection.execute(
            update(versions)
            .where(versions.c.profile_version_id == profile_version_id)
            .values(status="active", published_at=func.current_timestamp())
        )
    return {
        "profile_version_id": profile_version_id,
        "profile_id": profile_id,
        "version": int(row["version"]),
        "status": "active",
    }


def save_scoped_instruction_draft(
    *,
    engine: Engine,
    scope: str,
    prompt_instructions: str,
    category_id: int | None = None,
    channel_id: str | None = None,
) -> dict[str, Any]:
    """Create or update a category/channel draft without mutating the active version."""
    if scope not in {"category", "channel"}:
        raise ValueError("Instruction editing is limited to category and channel profiles.")
    if scope == "category" and category_id is None:
        raise ValueError("Category instruction editing requires category_id.")
    if scope == "channel" and not channel_id:
        raise ValueError("Channel instruction editing requires channel_id.")
    instructions = str(prompt_instructions).strip()
    if len(instructions) > 20_000:
        raise ValueError("Prompt instructions cannot exceed 20,000 characters.")

    with engine.begin() as connection:
        tables = _reflect_v2_tables(
            connection,
            names=(
                "extraction_profiles",
                "extraction_profile_versions",
                "doc_categories",
                "channels",
            ),
        )
        profiles = tables["extraction_profiles"]
        versions = tables["extraction_profile_versions"]
        source_id: str | None = None
        if scope == "category":
            category_name = connection.execute(
                select(tables["doc_categories"].c.category_name).where(
                    tables["doc_categories"].c.category_id == category_id
                )
            ).scalar_one_or_none()
            if category_name is None:
                raise MetadataV2Error(f"Category {category_id} was not found.")
            profile_name = f"{category_name} Metadata V2"
            selector = and_(
                profiles.c.scope == "category",
                profiles.c.category_id == category_id,
                profiles.c.channel_id.is_(None),
            )
        else:
            channel_row = connection.execute(
                select(
                    tables["channels"].c.channel_name,
                    tables["channels"].c.source_id,
                ).where(tables["channels"].c.channel_id == channel_id)
            ).mappings().first()
            if channel_row is None:
                raise MetadataV2Error(f"Channel {channel_id} was not found.")
            source_id = str(channel_row["source_id"])
            profile_name = f"{channel_row['channel_name']} Metadata V2"
            selector = and_(
                profiles.c.scope == "channel",
                profiles.c.channel_id == channel_id,
                profiles.c.category_id.is_(None),
            )

        profile_row = connection.execute(
            select(profiles.c.profile_id, profiles.c.learning_mode).where(selector)
        ).mappings().first()
        if profile_row is None:
            result = connection.execute(
                insert(profiles).values(
                    profile_name=profile_name,
                    scope=scope,
                    source_id=source_id,
                    channel_id=channel_id if scope == "channel" else None,
                    category_id=category_id if scope == "category" else None,
                    learning_mode="autopilot",
                    is_active=True,
                )
            )
            profile_id = int(result.inserted_primary_key[0])
        else:
            profile_id = int(profile_row["profile_id"])

        draft = connection.execute(
            select(
                versions.c.profile_version_id,
                versions.c.version,
                versions.c.config,
            )
            .where(versions.c.profile_id == profile_id, versions.c.status == "draft")
            .order_by(versions.c.version.desc())
            .limit(1)
        ).mappings().first()
        if draft is not None:
            config = dict(draft.get("config") or {})
            config["prompt_instructions"] = instructions
            profile_version_id = int(draft["profile_version_id"])
            version = int(draft["version"])
            connection.execute(
                update(versions)
                .where(versions.c.profile_version_id == profile_version_id)
                .values(config=config)
            )
        else:
            active_config = connection.execute(
                select(versions.c.config)
                .where(versions.c.profile_id == profile_id, versions.c.status == "active")
                .order_by(versions.c.version.desc())
                .limit(1)
            ).scalar_one_or_none()
            config = dict(active_config or {})
            config["prompt_instructions"] = instructions
            max_version = connection.execute(
                select(func.max(versions.c.version)).where(versions.c.profile_id == profile_id)
            ).scalar_one()
            version = int(max_version or 0) + 1
            result = connection.execute(
                insert(versions).values(
                    profile_id=profile_id,
                    version=version,
                    status="draft",
                    config=config,
                )
            )
            profile_version_id = int(result.inserted_primary_key[0])

    return {
        "profile_id": profile_id,
        "profile_version_id": profile_version_id,
        "version": version,
        "status": "draft",
        "scope": scope,
        "category_id": category_id,
        "channel_id": channel_id,
        "prompt_instructions": instructions,
    }


def list_extraction_profiles(*, engine: Engine) -> list[dict[str, Any]]:
    with engine.connect() as connection:
        tables = _reflect_v2_tables(
            connection,
            names=("extraction_profiles", "extraction_profile_versions"),
        )
        profiles = tables["extraction_profiles"]
        versions = tables["extraction_profile_versions"]
        rows = (
            connection.execute(
                select(
                    profiles.c.profile_id,
                    profiles.c.profile_name,
                    profiles.c.scope,
                    profiles.c.source_id,
                    profiles.c.channel_id,
                    profiles.c.category_id,
                    profiles.c.learning_mode,
                    profiles.c.is_active,
                    versions.c.profile_version_id,
                    versions.c.version,
                    versions.c.status,
                    versions.c.config,
                    versions.c.created_at,
                    versions.c.published_at,
                )
                .select_from(profiles.join(versions, versions.c.profile_id == profiles.c.profile_id))
                .order_by(profiles.c.profile_id, versions.c.version.desc())
            )
            .mappings()
            .all()
        )
    return [dict(row) for row in rows]


def list_extraction_runs(
    *,
    engine: Engine,
    content_id: int | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    with engine.connect() as connection:
        runs = _reflect_v2_tables(connection, names=("extraction_runs",))["extraction_runs"]
        statement = select(runs).order_by(runs.c.started_at.desc()).limit(max(1, min(limit, 500)))
        if content_id is not None:
            statement = statement.where(runs.c.content_id == content_id)
        rows = connection.execute(statement).mappings().all()
    return [dict(row) for row in rows]


def _validate_profile_selector(
    *,
    scope: str,
    source_id: str | None,
    channel_id: str | None,
    category_id: int | None,
) -> None:
    requirements = {
        "global": (False, False, False),
        "source": (True, False, False),
        "category": (False, False, True),
        "channel": (False, True, False),
        "channel_category": (False, True, True),
    }
    needs_source, needs_channel, needs_category = requirements[scope]
    if needs_source and not source_id:
        raise ValueError(f"scope={scope} requires source_id.")
    if needs_channel and not channel_id:
        raise ValueError(f"scope={scope} requires channel_id.")
    if needs_category and category_id is None:
        raise ValueError(f"scope={scope} requires category_id.")

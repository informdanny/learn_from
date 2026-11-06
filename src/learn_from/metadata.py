from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from sqlalchemy import MetaData, Table, case, delete, func, insert, select, update
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError


DEFAULT_EXTRACTION_MODEL = "gpt-5.4-nano"
ENHANCED_EXTRACTION_MODEL = "gpt-5.4-mini"

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
_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "schema_version": {"type": "string"},
        "category": {"type": "string"},
        "topics": {"type": "array", "items": {"type": "string"}},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "participants": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": ["string", "null"]},
                },
                "required": ["name", "role"],
                "additionalProperties": False,
            },
        },
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "label": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "context": {"type": ["string", "null"]},
                    "usage": {"type": ["string", "null"]},
                    "mention_count": {"type": ["integer", "null"]},
                    "ref": {
                        "type": ["object", "null"],
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
                "required": ["type", "label", "aliases", "context", "usage", "mention_count", "ref"],
                "additionalProperties": False,
            },
        },
        "references": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "label": {"type": "string"},
                    "context": {"type": ["string", "null"]},
                    "usage": {"type": ["string", "null"]},
                    "mention_count": {"type": ["integer", "null"]},
                    "ref": {
                        "type": ["object", "null"],
                        "properties": {},
                        "additionalProperties": False,
                    },
                    "identifiers": {
                        "type": ["object", "null"],
                        "properties": {},
                        "additionalProperties": False,
                    },
                    "attributes": {
                        "type": ["object", "null"],
                        "properties": {},
                        "additionalProperties": False,
                    },
                    "fragment_kind": {"type": ["string", "null"]},
                },
                "required": [
                    "type",
                    "label",
                    "context",
                    "usage",
                    "mention_count",
                    "ref",
                    "identifiers",
                    "attributes",
                    "fragment_kind",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["schema_version", "category", "topics", "keywords", "participants", "entities", "references"],
    "additionalProperties": False,
}
_CATEGORY_KEY_RE = re.compile(r"[^a-z0-9]+")
_SCRIPTURE_REFERENCE_RE = re.compile(
    r"^(?:[1-3]\s+)?[A-Za-z][A-Za-z\s]+?\s+\d+:\d+(?:\s*-\s*\d+)?$",
    flags=re.IGNORECASE,
)
_FRAGMENT_KINDS = {
    "slide",
    "chart",
    "whiteboard",
    "screenshot",
    "demo",
    "code_snippet",
    "diagram",
}
_TRANSCRIPT_MAX_CHARS = 12000


class MetadataExtractionError(RuntimeError):
    """Raised when metadata extraction fails."""


class MetadataPersistenceError(RuntimeError):
    """Raised when metadata persistence fails."""


def extract_metadata(
    *,
    transcript_text: str,
    openai_api_key: str,
    model: str = DEFAULT_EXTRACTION_MODEL,
    timeout_seconds: float = 90.0,
    category_hints: Sequence[str] | None = None,
    max_transcript_chars: int = _TRANSCRIPT_MAX_CHARS,
) -> dict[str, Any]:
    """
    Extract structured metadata from transcript text using the OpenAI Responses API.
    """
    if not transcript_text.strip():
        raise ValueError("Transcript text is required for metadata extraction.")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for metadata extraction.")

    prompt = build_metadata_extraction_prompt(
        transcript_text=transcript_text,
        category_hints=category_hints or (),
        max_transcript_chars=max_transcript_chars,
    )
    raw_output = _request_openai_metadata(
        prompt=prompt,
        api_key=openai_api_key,
        model=model,
        timeout_seconds=timeout_seconds,
    )
    return normalize_extracted_metadata(raw_output)


def build_metadata_extraction_prompt(
    *,
    transcript_text: str,
    category_hints: Sequence[str],
    max_transcript_chars: int,
) -> str:
    """
    Build a compact prompt aligned to the repository's metadata spec.
    """
    categories = [value.strip() for value in category_hints if value and value.strip()]
    category_lines = "\n".join(f"- {name}" for name in categories) if categories else "- Any known category"
    transcript_chunk = transcript_text[:max_transcript_chars]

    return f"""
You are extracting structured metadata from a transcript for a retrieval system.
Return ONLY valid JSON with this shape:
{{
  "schema_version": "2.6",
  "category": "string",
  "topics": ["string"],
  "keywords": ["string"],
  "participants": [{{"name": "string", "role": "string"}}],
  "entities": [
    {{
      "type": "person|organization|brand|product|service|software|framework|metric|location|timeframe|event|topic|concept|other",
      "label": "string",
      "aliases": ["string"],
      "context": "string",
      "usage": "primary_topic|recommended|criticized|quoted|evidence|example|comparison|background|neutral",
      "mention_count": 1,
      "ref": {{}}
    }}
  ],
  "references": [
    {{
      "type": "book|paper|study|scripture|talk|podcast|youtube_video|author_creator|magazine_article|website|media_fragment|other",
      "label": "string",
      "context": "string",
      "usage": "primary_topic|recommended|criticized|quoted|evidence|example|comparison|background|neutral",
      "mention_count": 1,
      "ref": {{}},
      "identifiers": {{}},
      "attributes": {{}}
    }}
  ]
}}

Category preference list:
{category_lines}

Rules:
- Keep topics short (1-3 words).
- Keep keywords lowercase.
- Only include high-signal entities/references, not exhaustive mentions.
- For references, keep `label` canonical and retrieval-friendly; put nuance/interpretation in `context`.
- Scripture labels should be canonical (examples: `Acts 12:2`, `1 Enoch 6-11`, `Book of Revelation`).
- For scripture, remove interpretive parentheticals from `label` and move those details to `context`.
- If a citation is verse-like (`Book 12:2` style), classify it as `scripture` rather than `paper` or `study`.
- `talk` is the correct type for conference talks, sermons, and lectures.
- Use `magazine_article` for magazine citations; label best-effort as publication + known title/date/issue when available.
- Keep Bible translations (NIV/KJV/NRSV/etc.) in `context` and/or `identifiers.translation`, not as standalone reference labels.
- `media_fragment` labels may include descriptive parenthetical detail when needed to disambiguate the fragment.

Transcript:
{transcript_chunk}
""".strip()


def get_metadata_extraction_inspection(*, category_hints: Sequence[str]) -> dict[str, Any]:
    """Return the live extraction contract used by the read-only pipeline inspector."""
    transcript_placeholder = "{{ transcript_text | truncated to 12,000 characters }}"
    prompt = build_metadata_extraction_prompt(
        transcript_text=transcript_placeholder,
        category_hints=category_hints,
        max_transcript_chars=_TRANSCRIPT_MAX_CHARS,
    )
    return {
        "name": "Metadata extraction",
        "schema_version": "2.6",
        "default_model": DEFAULT_EXTRACTION_MODEL,
        "enhanced_model": ENHANCED_EXTRACTION_MODEL,
        "transcript_max_chars": _TRANSCRIPT_MAX_CHARS,
        "endpoint": "https://api.openai.com/v1/responses",
        "assembled_prompt": prompt,
        "json_schema": _JSON_SCHEMA,
        "request_envelope": {
            "model": "{{ selected_model }}",
            "input": "{{ assembled_prompt }}",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "metadata_extraction",
                    "schema": "{{ json_schema shown in this inspector }}",
                    "strict": True,
                }
            },
        },
        "normalization": [
            "Clean and deduplicate category, topics, keywords, and participants",
            "Canonicalize entity and reference labels and controlled types",
            "Limit topics to 8 and keywords to 10",
            "Persist normalized output to catalog and doc relationship tables",
        ],
        "persistence_targets": [
            "media.category_id",
            "media.extraction_version",
            "doc_topics",
            "doc_keywords",
            "doc_participants",
            "entities_catalog + doc_entities",
            "references_catalog + doc_references",
        ],
    }


def normalize_extracted_metadata(payload: Mapping[str, Any] | Any) -> dict[str, Any]:
    """
    Normalize extractor output into a stable, DB-ready shape.
    """
    if not isinstance(payload, Mapping):
        raise MetadataExtractionError("Metadata extractor output must be a JSON object.")

    return {
        "schema_version": _clean_text(payload.get("schema_version"), max_length=20) or "2.6",
        "category": _clean_text(payload.get("category"), max_length=120),
        "topics": _normalize_string_list(payload.get("topics"), max_items=8, max_length=100),
        "keywords": _normalize_string_list(
            payload.get("keywords"),
            max_items=10,
            max_length=100,
            lowercase=True,
        ),
        "participants": _normalize_participants(payload.get("participants")),
        "entities": _normalize_entities(payload.get("entities")),
        "references": _normalize_references(payload.get("references")),
    }


def extract_and_persist_metadata_for_content(
    *,
    engine: Engine,
    content_id: int,
    openai_api_key: str,
    model: str = DEFAULT_EXTRACTION_MODEL,
    timeout_seconds: float = 90.0,
) -> dict[str, Any]:
    """
    Run extraction against one existing `doc_content` row and persist output to `doc_*` tables.
    """
    with engine.begin() as connection:
        tables = _reflect_required_tables(connection)
        row = (
            connection.execute(
                select(
                    tables["doc_content"].c.content_id,
                    tables["doc_content"].c.media_id,
                    tables["doc_content"].c.full_text,
                    tables["media"].c.extraction_version,
                )
                .select_from(
                    tables["doc_content"].join(
                        tables["media"],
                        tables["media"].c.media_id == tables["doc_content"].c.media_id,
                    )
                )
                .where(tables["doc_content"].c.content_id == content_id)
            )
            .mappings()
            .first()
        )
        if row is None:
            raise MetadataPersistenceError(f"No doc_content row found for content_id={content_id}.")
        transcript_text = str(row.get("full_text") or "").strip()
        if not transcript_text:
            raise MetadataPersistenceError(
                f"doc_content.full_text is empty for content_id={content_id}; cannot extract metadata."
            )
        category_hints = [
            str(category_name)
            for category_name in connection.execute(
                select(tables["doc_categories"].c.category_name)
            ).scalars()
        ]
        extraction_version = str(row.get("extraction_version") or "1.0")

    extracted = extract_metadata(
        transcript_text=transcript_text,
        openai_api_key=openai_api_key,
        model=model,
        timeout_seconds=timeout_seconds,
        category_hints=category_hints,
    )
    persist_extracted_metadata(
        engine=engine,
        content_id=content_id,
        extraction=extracted,
        extraction_version=extraction_version,
    )

    return {
        "content_id": content_id,
        "media_id": row["media_id"],
        "model": model,
        "category": extracted.get("category"),
        "topic_count": len(extracted.get("topics", [])),
        "keyword_count": len(extracted.get("keywords", [])),
        "participant_count": len(extracted.get("participants", [])),
        "entity_count": len(extracted.get("entities", [])),
        "reference_count": len(extracted.get("references", [])),
    }


def persist_extracted_metadata(
    *,
    engine: Engine,
    content_id: int,
    extraction: Mapping[str, Any] | Any,
    extraction_version: str = "1.0",
    extraction_quality_score: float | None = None,
) -> None:
    """
    Persist extracted metadata for one `doc_content` row in normalized `doc_*` tables.
    """
    normalized = normalize_extracted_metadata(extraction)

    try:
        with engine.begin() as connection:
            tables = _reflect_required_tables(connection)
            media_row = (
                connection.execute(
                    select(tables["doc_content"].c.media_id).where(
                        tables["doc_content"].c.content_id == content_id
                    )
                )
                .mappings()
                .first()
            )
            if media_row is None:
                raise MetadataPersistenceError(f"No doc_content row found for content_id={content_id}.")
            media_id = str(media_row["media_id"])

            existing_entity_ids = set(
                connection.execute(
                    select(tables["doc_entities"].c.entity_id).where(
                        tables["doc_entities"].c.content_id == content_id
                    )
                ).scalars()
            )
            existing_reference_ids = set(
                connection.execute(
                    select(tables["doc_references"].c.reference_id).where(
                        tables["doc_references"].c.content_id == content_id
                    )
                ).scalars()
            )

            _delete_existing_doc_metadata_rows(connection, tables=tables, content_id=content_id)
            _persist_topics(connection, table=tables["doc_topics"], content_id=content_id, topics=normalized["topics"])
            _persist_keywords(
                connection, table=tables["doc_keywords"], content_id=content_id, keywords=normalized["keywords"]
            )
            _persist_participants(
                connection,
                table=tables["doc_participants"],
                content_id=content_id,
                participants=normalized["participants"],
            )

            new_entity_ids = _persist_entities(
                connection=connection,
                tables=tables,
                content_id=content_id,
                entities=normalized["entities"],
                existing_entity_ids=existing_entity_ids,
            )
            new_reference_ids = _persist_references(
                connection=connection,
                tables=tables,
                content_id=content_id,
                references=normalized["references"],
                existing_reference_ids=existing_reference_ids,
            )

            _decrement_removed_catalog_usage(
                connection=connection,
                table=tables["entities_catalog"],
                id_column_name="entity_id",
                removed_ids=existing_entity_ids - new_entity_ids,
            )
            _decrement_removed_catalog_usage(
                connection=connection,
                table=tables["references_catalog"],
                id_column_name="reference_id",
                removed_ids=existing_reference_ids - new_reference_ids,
            )

            category_id = _resolve_category_id(
                connection=connection,
                table=tables["doc_categories"],
                extracted_category=normalized.get("category"),
            )
            media_update_values: dict[str, Any] = {
                "last_updated": func.current_timestamp(),
                "extraction_version": extraction_version,
            }
            if category_id is not None:
                media_update_values["category_id"] = category_id
            if extraction_quality_score is not None:
                media_update_values["extraction_quality_score"] = extraction_quality_score
            connection.execute(
                update(tables["media"])
                .where(tables["media"].c.media_id == media_id)
                .values(**media_update_values)
            )
            connection.execute(
                update(tables["doc_content"])
                .where(tables["doc_content"].c.content_id == content_id)
                .values(
                    metadata_updated_at=func.current_timestamp(),
                    updated_at=func.current_timestamp(),
                )
            )
    except SQLAlchemyError as exc:
        raise MetadataPersistenceError(f"Failed to persist metadata for content_id={content_id}: {exc}") from exc


def _persist_entities(
    *,
    connection: Connection,
    tables: dict[str, Table],
    content_id: int,
    entities: Sequence[Mapping[str, Any]],
    existing_entity_ids: set[int],
) -> set[int]:
    new_entity_ids: set[int] = set()
    for entity in entities:
        entity_id, created = _upsert_catalog_row(
            connection=connection,
            table=tables["entities_catalog"],
            id_column_name="entity_id",
            type_value=str(entity["type"]),
            label_value=str(entity["label"]),
            aliases=entity.get("aliases"),
            identifiers=None,
        )
        if entity_id is None:
            continue
        new_entity_ids.add(entity_id)
        if not created and entity_id not in existing_entity_ids:
            _increment_catalog_usage(
                connection=connection,
                table=tables["entities_catalog"],
                id_column_name="entity_id",
                row_id=entity_id,
            )
        connection.execute(
            insert(tables["doc_entities"]).values(
                content_id=content_id,
                entity_id=entity_id,
                context=entity.get("context"),
                usage=entity.get("usage", "neutral"),
                mention_count=entity.get("mention_count", 1),
                ref=entity.get("ref"),
            )
        )
    return new_entity_ids


def _persist_references(
    *,
    connection: Connection,
    tables: dict[str, Table],
    content_id: int,
    references: Sequence[Mapping[str, Any]],
    existing_reference_ids: set[int],
) -> set[int]:
    new_reference_ids: set[int] = set()
    for reference in references:
        reference_id, created = _upsert_catalog_row(
            connection=connection,
            table=tables["references_catalog"],
            id_column_name="reference_id",
            type_value=str(reference["type"]),
            label_value=str(reference["label"]),
            aliases=None,
            identifiers=reference.get("identifiers"),
        )
        if reference_id is None:
            continue
        new_reference_ids.add(reference_id)
        if not created and reference_id not in existing_reference_ids:
            _increment_catalog_usage(
                connection=connection,
                table=tables["references_catalog"],
                id_column_name="reference_id",
                row_id=reference_id,
            )
        connection.execute(
            insert(tables["doc_references"]).values(
                content_id=content_id,
                reference_id=reference_id,
                context=reference.get("context"),
                usage=reference.get("usage", "neutral"),
                mention_count=reference.get("mention_count", 1),
                ref=reference.get("ref"),
                attributes=reference.get("attributes", {}),
            )
        )
    return new_reference_ids


def _upsert_catalog_row(
    *,
    connection: Connection,
    table: Table,
    id_column_name: str,
    type_value: str,
    label_value: str,
    aliases: Sequence[str] | None,
    identifiers: Mapping[str, Any] | None,
) -> tuple[int | None, bool]:
    label_normalized = _normalize_key(label_value)
    if not label_normalized:
        return None, False

    row = (
        connection.execute(
            select(table.c[id_column_name]).where(
                table.c.type == type_value,
                table.c.label_normalized == label_normalized,
            )
        )
        .mappings()
        .first()
    )
    if row:
        row_id = int(row[id_column_name])
        update_values: dict[str, Any] = {
            "label": label_value,
            "last_seen_at": func.current_timestamp(),
        }
        if aliases is not None and "aliases" in table.c:
            update_values["aliases"] = list(aliases)
        if identifiers is not None and "identifiers" in table.c:
            update_values["identifiers"] = dict(identifiers)
        connection.execute(update(table).where(table.c[id_column_name] == row_id).values(**update_values))
        return row_id, False

    insert_values: dict[str, Any] = {
        "type": type_value,
        "label": label_value,
        "label_normalized": label_normalized,
    }
    if "usage_count" in table.c:
        insert_values["usage_count"] = 1
    if "first_seen_at" in table.c:
        insert_values["first_seen_at"] = func.current_timestamp()
    if "last_seen_at" in table.c:
        insert_values["last_seen_at"] = func.current_timestamp()
    if aliases is not None and "aliases" in table.c:
        insert_values["aliases"] = list(aliases)
    if identifiers is not None and "identifiers" in table.c:
        insert_values["identifiers"] = dict(identifiers)

    result = connection.execute(insert(table).values(**insert_values))
    inserted_id = result.inserted_primary_key[0] if result.inserted_primary_key else None
    if inserted_id is not None:
        return int(inserted_id), True

    lookup = (
        connection.execute(
            select(table.c[id_column_name]).where(
                table.c.type == type_value,
                table.c.label_normalized == label_normalized,
            )
        )
        .mappings()
        .first()
    )
    return (int(lookup[id_column_name]), True) if lookup else (None, True)


def _increment_catalog_usage(
    *,
    connection: Connection,
    table: Table,
    id_column_name: str,
    row_id: int,
) -> None:
    connection.execute(
        update(table)
        .where(table.c[id_column_name] == row_id)
        .values(
            usage_count=table.c.usage_count + 1,
            last_seen_at=func.current_timestamp(),
        )
    )


def _decrement_removed_catalog_usage(
    *,
    connection: Connection,
    table: Table,
    id_column_name: str,
    removed_ids: set[int],
) -> None:
    for row_id in removed_ids:
        connection.execute(
            update(table)
            .where(table.c[id_column_name] == row_id)
            .values(
                usage_count=case((table.c.usage_count > 1, table.c.usage_count - 1), else_=1),
                last_seen_at=func.current_timestamp(),
            )
        )


def _delete_existing_doc_metadata_rows(
    connection: Connection,
    *,
    tables: dict[str, Table],
    content_id: int,
) -> None:
    for table_name in ("doc_topics", "doc_keywords", "doc_participants", "doc_entities", "doc_references"):
        connection.execute(delete(tables[table_name]).where(tables[table_name].c.content_id == content_id))


def _persist_topics(
    connection: Connection,
    *,
    table: Table,
    content_id: int,
    topics: Sequence[str],
) -> None:
    for topic in topics:
        connection.execute(insert(table).values(content_id=content_id, topic=topic))


def _persist_keywords(
    connection: Connection,
    *,
    table: Table,
    content_id: int,
    keywords: Sequence[str],
) -> None:
    for keyword in keywords:
        connection.execute(insert(table).values(content_id=content_id, keyword=keyword))


def _persist_participants(
    connection: Connection,
    *,
    table: Table,
    content_id: int,
    participants: Sequence[Mapping[str, Any]],
) -> None:
    for participant in participants:
        connection.execute(
            insert(table).values(
                content_id=content_id,
                name=participant["name"],
                role=participant.get("role"),
            )
        )


def _reflect_required_tables(connection: Connection) -> dict[str, Table]:
    required = (
        "media",
        "doc_content",
        "doc_categories",
        "doc_topics",
        "doc_keywords",
        "doc_participants",
        "entities_catalog",
        "references_catalog",
        "doc_entities",
        "doc_references",
    )
    metadata = MetaData()
    tables: dict[str, Table] = {}
    for table_name in required:
        tables[table_name] = Table(table_name, metadata, autoload_with=connection)
    return tables


def _resolve_category_id(
    *,
    connection: Connection,
    table: Table,
    extracted_category: str | None,
) -> int | None:
    if not extracted_category:
        return None

    extracted_variants = _category_variants(extracted_category)
    rows = connection.execute(select(table.c.category_id, table.c.category_name)).mappings().all()
    for row in rows:
        row_name = str(row["category_name"])
        if _normalize_key(row_name) in extracted_variants:
            return int(row["category_id"])
    return None


def _category_variants(category: str) -> set[str]:
    raw = category.strip()
    variants = {
        raw,
        raw.lower(),
        raw.replace("_", " "),
        raw.replace("_", "/"),
        raw.replace("-", " "),
        raw.replace("/", " "),
    }
    return {_normalize_key(value) for value in variants if value.strip()}


def _normalize_participants(raw_participants: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_participants, Sequence) or isinstance(raw_participants, (str, bytes)):
        return []

    normalized: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for item in raw_participants:
        if not isinstance(item, Mapping):
            continue
        name = _clean_text(item.get("name"), max_length=255)
        if not name:
            continue
        name_key = _normalize_key(name)
        if name_key in seen_names:
            continue
        seen_names.add(name_key)
        normalized.append(
            {
                "name": name,
                "role": _clean_text(item.get("role"), max_length=50),
            }
        )
    return normalized


def _normalize_entities(raw_entities: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_entities, Sequence) or isinstance(raw_entities, (str, bytes)):
        return []

    normalized: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for item in raw_entities:
        if not isinstance(item, Mapping):
            continue
        label = _clean_text(item.get("label"), max_length=255)
        if not label:
            continue
        entity_type = _normalize_entity_type(item.get("type"))
        dedupe_key = (entity_type, _normalize_key(label))
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        normalized.append(
            {
                "type": entity_type,
                "label": label,
                "aliases": _normalize_string_list(item.get("aliases"), max_items=10, max_length=255),
                "context": _clean_text(item.get("context"), max_length=150),
                "usage": _normalize_usage(item.get("usage")),
                "mention_count": _safe_positive_int(item.get("mention_count")),
                "ref": _normalize_ref(item.get("ref")),
            }
        )
    return normalized


def _normalize_references(raw_references: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_references, Sequence) or isinstance(raw_references, (str, bytes)):
        return []

    normalized: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for item in raw_references:
        if not isinstance(item, Mapping):
            continue
        label = _clean_text(item.get("label"), max_length=255)
        if not label:
            continue
        reference_type = _normalize_reference_type(item.get("type"))
        if reference_type != "scripture" and _looks_like_scripture_reference(label):
            reference_type = "scripture"
        label = _canonicalize_reference_label(reference_type, label)
        dedupe_key = (reference_type, _normalize_key(label))
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)

        attributes: dict[str, Any] = {}
        if isinstance(item.get("attributes"), Mapping):
            attributes.update(dict(item.get("attributes")))
        fragment_kind = _clean_text(item.get("fragment_kind"), max_length=50)
        if fragment_kind:
            fragment_kind = fragment_kind.lower().replace("-", "_").replace(" ", "_")
        if (
            reference_type == "media_fragment"
            and fragment_kind
            and fragment_kind in _FRAGMENT_KINDS
            and "fragment_kind" not in attributes
        ):
            attributes["fragment_kind"] = fragment_kind

        identifiers = item.get("identifiers")
        normalized.append(
            {
                "type": reference_type,
                "label": label,
                "context": _clean_text(item.get("context"), max_length=150),
                "usage": _normalize_usage(item.get("usage")),
                "mention_count": _safe_positive_int(item.get("mention_count")),
                "ref": _normalize_ref(item.get("ref")),
                "identifiers": dict(identifiers) if isinstance(identifiers, Mapping) else {},
                "attributes": attributes,
            }
        )
    return normalized


def _normalize_usage(value: Any) -> str:
    candidate = _clean_text(value, max_length=20)
    if not candidate:
        return "neutral"
    normalized = candidate.lower().replace("-", "_").replace(" ", "_")
    return normalized if normalized in _USAGE_TYPES else "neutral"


def _normalize_entity_type(value: Any) -> str:
    candidate = _clean_text(value, max_length=30)
    if not candidate:
        return "other"
    normalized = candidate.lower().replace("-", "_").replace(" ", "_")
    return normalized if normalized in _ENTITY_TYPES else "other"


def _normalize_reference_type(value: Any) -> str:
    candidate = _clean_text(value, max_length=30)
    if not candidate:
        return "other"
    normalized = candidate.lower().replace("-", "_").replace(" ", "_")
    return normalized if normalized in _REFERENCE_TYPES else "other"


def _looks_like_scripture_reference(label: str) -> bool:
    compact = re.sub(r"\s+", " ", label).strip()
    return bool(_SCRIPTURE_REFERENCE_RE.match(compact))


def _canonicalize_reference_label(reference_type: str, label: str) -> str:
    if reference_type != "scripture":
        return label

    # Keep labels broad/canonical for retrieval; interpretation belongs in context.
    cleaned = re.sub(r"\s*\([^)]*\)\s*$", "", label).strip()
    cleaned = re.sub(r"\bchapters?\s+(\d+\s*-\s*\d+)\b", r"\1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bchapter\s+(\d+)\s+verse\s+(\d+)\b", r"\1:\2", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:.")

    if _normalize_key(cleaned) == "revelation":
        return "Book of Revelation"
    return cleaned or label


def _normalize_ref(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None


def _normalize_string_list(
    value: Any,
    *,
    max_items: int,
    max_length: int,
    lowercase: bool = False,
) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _clean_text(item, max_length=max_length)
        if not text:
            continue
        if lowercase:
            text = text.lower()
        dedupe_key = _normalize_key(text)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        cleaned.append(text)
        if len(cleaned) >= max_items:
            break
    return cleaned


def _safe_positive_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 1
    return parsed if parsed > 0 else 1


def _clean_text(value: Any, *, max_length: int) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"\s+", " ", str(value)).strip()
    if not cleaned:
        return None
    return cleaned[:max_length]


def _normalize_key(value: str) -> str:
    return _CATEGORY_KEY_RE.sub("", value.lower().strip())


def _request_openai_metadata(
    *,
    prompt: str,
    api_key: str,
    model: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    request_payload_v1 = {
        "model": model,
        "input": prompt,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "metadata_extraction",
                "schema": _JSON_SCHEMA,
                "strict": True,
            }
        },
    }

    try:
        response_payload = _post_json(
            url="https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "learn-from/0.1.0",
            },
            payload=request_payload_v1,
            timeout_seconds=timeout_seconds,
        )
    except MetadataExtractionError as exc:
        message = str(exc).lower()
        if "unknown parameter" not in message and "invalid request" not in message:
            raise
        request_payload_fallback = {
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "metadata_extraction", "schema": _JSON_SCHEMA},
            },
        }
        response_payload = _post_json(
            url="https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "learn-from/0.1.0",
            },
            payload=request_payload_fallback,
            timeout_seconds=timeout_seconds,
        )

    parsed = _extract_json_from_response_payload(response_payload)
    if not isinstance(parsed, Mapping):
        raise MetadataExtractionError("OpenAI returned a non-object extraction payload.")
    return dict(parsed)


def _post_json(
    *,
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    request = Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
        raise MetadataExtractionError(f"OpenAI extraction request failed (HTTP {exc.code}): {details}") from exc
    except URLError as exc:
        raise MetadataExtractionError(f"OpenAI extraction connection error: {exc}") from exc

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MetadataExtractionError("OpenAI extraction returned invalid JSON.") from exc

    if isinstance(parsed, Mapping) and parsed.get("error"):
        raise MetadataExtractionError(f"OpenAI extraction error payload: {parsed['error']}")
    return dict(parsed) if isinstance(parsed, Mapping) else {"output_text": raw}


def _extract_json_from_response_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("output_parsed", "parsed_output", "parsed"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            return dict(candidate)

    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return _parse_json_object(output_text)

    output_items = payload.get("output")
    if isinstance(output_items, Sequence):
        for item in output_items:
            if not isinstance(item, Mapping):
                continue
            content = item.get("content")
            if isinstance(content, Sequence):
                for block in content:
                    if not isinstance(block, Mapping):
                        continue
                    for field in ("parsed", "json"):
                        candidate = block.get(field)
                        if isinstance(candidate, Mapping):
                            return dict(candidate)
                    text_field = block.get("text")
                    if isinstance(text_field, str) and text_field.strip():
                        return _parse_json_object(text_field)
            # Some responses place plain text directly on the item.
            for field in ("text", "arguments"):
                candidate = item.get(field)
                if isinstance(candidate, str) and candidate.strip():
                    return _parse_json_object(candidate)

    raise MetadataExtractionError("Unable to parse structured metadata JSON from OpenAI response payload.")


def _parse_json_object(raw_json: str) -> Mapping[str, Any]:
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise MetadataExtractionError("OpenAI output_text was not valid JSON.") from exc
    if not isinstance(parsed, Mapping):
        raise MetadataExtractionError("OpenAI JSON output was not an object.")
    return dict(parsed)

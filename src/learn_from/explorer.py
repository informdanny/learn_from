from __future__ import annotations

import json
import mimetypes
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from learn_from.metadata import get_metadata_extraction_inspection
from learn_from.metadata_v2 import (
    MetadataV2Error,
    get_metadata_v2_prompt_inspection,
    is_mentioned_only_participant,
    is_scripture_keyword,
    list_extraction_profiles,
    list_extraction_runs,
    preview_profile_instruction_contracts,
    save_scoped_instruction_draft,
)


STATIC_DIR = Path(__file__).with_name("explorer_static")
ATLAS_KINDS = {"category", "channel", "topic", "keyword", "participant", "entity", "reference"}
SORT_SQL = {
    "newest": "m.publish_date DESC NULLS LAST, m.title",
    "oldest": "m.publish_date ASC NULLS LAST, m.title",
    "views": "m.views DESC NULLS LAST, m.title",
    "title": "m.title ASC",
}


def _json_value(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def _rows(result: Any) -> list[dict[str, Any]]:
    return [{key: _json_value(value) for key, value in row.items()} for row in result.mappings()]


def _one(result: Any) -> dict[str, Any] | None:
    row = result.mappings().first()
    return None if row is None else {key: _json_value(value) for key, value in row.items()}


def _normalize_atlas_key(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def _validate_extraction_version(value: str | None) -> str:
    version = str(value or "2.0")
    if version not in {"1.0", "2.0"}:
        raise ValueError(f"Unsupported extraction version: {version}")
    return version


def _pipeline_stage(
    key: str,
    label: str,
    detail: str,
    *,
    owner: str,
    mechanism: str,
    invocation: str,
    summary: str,
    inputs: Sequence[str],
    actions: Sequence[str],
    outputs: Sequence[str],
    code_views: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "detail": detail,
        "owner": owner,
        "mechanism": mechanism,
        "invocation": invocation,
        "summary": summary,
        "inputs": list(inputs),
        "actions": list(actions),
        "outputs": list(outputs),
        "code_views": [dict(item) for item in code_views],
    }


def _shared_ingestion_stages() -> list[dict[str, Any]]:
    return [
        _pipeline_stage(
            "source", "YouTube metadata", "Title, channel, dates, metrics",
            owner="YouTube Data API client", mechanism="Provider API + deterministic mapping", invocation="Every ingest",
            summary="Retrieves the source record used to identify and describe the video. No LLM is involved.",
            inputs=["{{ video_id_or_url }}", "{{ youtube_api_key }}"],
            actions=["Resolve the video ID", "Fetch snippet, statistics, and content details", "Map provider fields into the media contract"],
            outputs=["Video identity", "Title and description", "Channel and publication metadata", "Duration and engagement metrics"],
        ),
        _pipeline_stage(
            "transcript", "Transcript retrieval", "Timestamped segments + full text",
            owner="Transcript provider adapter", mechanism="Provider API with local normalization", invocation="Every ingest",
            summary="Fetches caption segments and converts provider-specific output into one transcript shape. No LLM is involved.",
            inputs=["{{ video_id }}", "{{ preferred_languages }}", "{{ transcript_provider_credentials }}"],
            actions=["Try youtube-transcript.io when configured", "Otherwise use youtube-transcript-api", "Sanitize segment timing and render timestamped text"],
            outputs=["full_text", "formatted_text", "timestamped_segments", "language and generated-caption flags"],
        ),
        _pipeline_stage(
            "content", "Content persistence", "media + complete doc_content",
            owner="Ingestion service", mechanism="Transactional PostgreSQL upsert", invocation="Every successful ingest",
            summary="Writes the source record and complete transcript before metadata extraction begins. No agent or model is involved.",
            inputs=["{{ youtube_metadata }}", "{{ normalized_transcript }}", "{{ source_type }}"],
            actions=["Upsert media", "Upsert the complete transcript", "Calculate word and character counts"],
            outputs=["media row", "doc_content row", "content_id used by extraction"],
        ),
    ]


def _v1_pipeline_stages(inspection: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _shared_ingestion_stages() + [
        _pipeline_stage(
            "extractor", "Metadata extraction", "Single Responses API structured-output pass",
            owner="V1 metadata extractor", mechanism="One LLM call", invocation="When metadata extraction is enabled",
            summary="Assembles category hints and a transcript placeholder into the production V1 prompt, then requests strict JSON.",
            inputs=["{{ transcript_text | truncated }}", "{{ category_hints }}", "{{ selected_model }}"],
            actions=["Assemble the live prompt", "Call the OpenAI Responses API", "Validate the strict metadata schema"],
            outputs=["Category", "Topics and keywords", "Participants", "Entities", "References"],
            code_views=[
                {"key": "prompt", "label": "Assembled prompt", "content": inspection["assembled_prompt"]},
                {"key": "schema", "label": "JSON schema", "content": json.dumps(inspection["json_schema"], indent=2)},
                {"key": "request", "label": "Request envelope", "content": json.dumps(inspection["request_envelope"], indent=2)},
            ],
        ),
        _pipeline_stage(
            "normalize", "Normalization", "Canonical labels + controlled vocabularies",
            owner="V1 normalization functions", mechanism="Deterministic Python rules", invocation="After every V1 model response",
            summary="Cleans and constrains model output. This is not an agent and does not make a second model call.",
            inputs=["{{ validated_extractor_json }}"],
            actions=inspection["normalization"],
            outputs=["Deduplicated metadata", "Controlled entity/reference types", "Canonical reference labels", "Database-ready records"],
        ),
        _pipeline_stage(
            "persist", "Extraction persistence", "doc_* links + global catalogs",
            owner="V1 persistence service", mechanism="Transactional PostgreSQL writes", invocation="After normalization succeeds",
            summary="Replaces document relationship rows, upserts shared catalogs, and updates extraction timestamps/version. No LLM is involved.",
            inputs=["{{ content_id }}", "{{ normalized_metadata }}", "{{ extraction_version }}"],
            actions=["Replace document metadata links", "Upsert entity and reference catalogs", "Resolve category ID", "Update media and content timestamps"],
            outputs=inspection["persistence_targets"],
        ),
    ]


def _v2_pipeline_stages(inspection: Mapping[str, Any]) -> list[dict[str, Any]]:
    contracts = inspection["prompt_contracts"]
    prompt_views = lambda key: [
        {"key": "instructions", "label": "Instructions", "content": contracts[key]["instructions"]},
        {"key": "input", "label": "Runtime input", "content": contracts[key]["input_template"]},
        {"key": "schema", "label": "JSON schema", "content": json.dumps(contracts[key]["json_schema"], indent=2)},
    ]
    return _shared_ingestion_stages() + [
        _pipeline_stage(
            "normalize", "Transcript preparation", "Light cleanup + timestamp-aware chunks",
            owner="V2 normalizer light-1", mechanism="Deterministic Python rules", invocation="Every V2 run",
            summary="Lightly cleans transcript text and creates overlapping timestamp-aware chunks. This is not one of the three workers and makes no model call.",
            inputs=["{{ doc_content.full_text }}", "{{ doc_content.timestamped_segments }}"],
            actions=["Collapse repeated filler and whitespace", "Preserve timestamps", "Create chunks up to 18,000 characters", "Overlap three boundary segments"],
            outputs=["analysis_text", "segment_map", "transcript_chunks"],
        ),
        _pipeline_stage(
            "extractor", "Worker 1 · Extractor", "Evidence-bearing candidates from every chunk",
            owner="Worker 1", mechanism="OpenAI Responses API · LLM", invocation="Every V2 run, once per transcript chunk",
            summary="Extracts candidates from transcript evidence. Runtime profile values are interpolated into the instructions and document context shown below.",
            inputs=["{{ effective_profile }}", "{{ document_context }}", "{{ transcript_chunk }}"],
            actions=["Extract strict candidates per chunk", "Aggregate and deduplicate candidates", "Supplement explicit scripture references when enabled"],
            outputs=["Raw category, topics, and keywords", "Participant candidates", "Entity candidates", "Reference candidates with evidence"],
            code_views=prompt_views("extractor"),
        ),
        _pipeline_stage(
            "resolver", "Worker 2 · Resolver", "Rules first; model only for unresolved values",
            owner="Worker 2 + resolver rules", mechanism="Hybrid deterministic + optional LLM", invocation="Every V2 run; model call only when unresolved candidates remain",
            summary="Canonicalization starts with aliases, known values, confidence rules, generic-role rejection, and scripture parsing. Worker 2 receives only the remaining unresolved candidates.",
            inputs=["{{ worker_1_candidates }}", "{{ scoped_aliases }}", "{{ effective_profile.known_values }}"],
            actions=["Apply deterministic resolution", "Separate trusted and unresolved candidates", "Optionally resolve the unresolved subset with Worker 2", "Merge model resolutions back into the full result"],
            outputs=["Canonical participants", "Canonical entities", "Structured references", "Confidence, status, basis, and preserved evidence"],
            code_views=prompt_views("resolver"),
        ),
        _pipeline_stage(
            "learner", "Worker 3 · Learner", "Conditional channel-level improvement proposals",
            owner="Worker 3", mechanism="Conditional OpenAI Responses API · LLM", invocation="Bootstrap/audit schedule only; may be disabled",
            summary="Reviews current and recent channel runs for repeatable knowledge. It proposes aliases, blocked values, or prompt additions but cannot publish profile changes itself.",
            inputs=["{{ current_resolution }}", "{{ recent_channel_runs }}", "{{ effective_learning_mode }}"],
            actions=["Check whether learning is due", "Compare repeated evidence across documents", "Return suggestions under strict corroboration rules"],
            outputs=["Alias/value suggestions", "Draft prompt additions", "Draft blocked values", "No direct metadata or profile mutation"],
            code_views=prompt_views("learner"),
        ),
        _pipeline_stage(
            "persist", "Run persistence", "Immutable run + evidence observations",
            owner="V2 persistence service", mechanism="Transactional PostgreSQL writes", invocation="After successful V2 resolution/learning",
            summary="Stores the raw, resolved, and learning outputs with provenance. V2 remains side-by-side and does not overwrite V1 doc metadata.",
            inputs=["{{ extraction }}", "{{ resolution }}", "{{ learning }}", "{{ release_manifest }}", "{{ usage_summary }}"],
            actions=["Complete the immutable extraction run", "Store transcript analysis and observations", "Store reference mentions and learning events", "Retain profile/model/token provenance"],
            outputs=["extraction_runs", "transcript_analysis_versions", "metadata_observations", "doc_reference_mentions", "metadata_learning_events"],
        ),
    ]


def _v2_records(data: Mapping[str, Any], *, kind: str) -> list[dict[str, Any]]:
    if kind == "topic":
        return [{"label": value, "type": "topic", "mention_count": 1} for value in data.get("topics", [])]
    if kind == "keyword":
        return [{"label": value, "type": "keyword", "mention_count": 1} for value in data.get("keywords", [])]
    if kind == "participant":
        return [
            {
                "label": item["name"],
                "type": item.get("role") or "participant",
                "context": item.get("role"),
                "evidence": item.get("evidence"),
                "usage": "participant",
                "mention_count": 1,
            }
            for item in data.get("participants", [])
        ]
    if kind in {"entity", "reference"}:
        field = "entities" if kind == "entity" else "references"
        return [dict(item) for item in data.get(field, [])]
    return []


class ExplorerRepository:
    """Local corpus explorer with narrowly scoped profile-draft writes."""

    def __init__(self, engine: Engine):
        self.engine = engine

    def overview(self) -> dict[str, Any]:
        with self.engine.connect() as connection:
            totals = _one(
                connection.execute(
                    text(
                        """
                        SELECT
                          (SELECT count(*) FROM media) AS documents,
                          (SELECT count(*) FROM doc_content WHERE chunk_type = 'complete') AS transcripts,
                          (SELECT coalesce(sum(word_count), 0) FROM doc_content WHERE chunk_type = 'complete') AS words,
                          (SELECT count(*) FROM entities_catalog) AS entities,
                          (SELECT count(*) FROM references_catalog) AS references,
                          (SELECT count(*) FROM doc_topics) AS topic_links
                        """
                    )
                )
            )
            categories = _rows(
                connection.execute(
                    text(
                        """
                        SELECT c.category_name AS label, count(m.media_id) AS count
                        FROM doc_categories c
                        JOIN media m ON m.category_id = c.category_id
                        GROUP BY c.category_id, c.category_name
                        ORDER BY count DESC, label
                        LIMIT 8
                        """
                    )
                )
            )
            recent = _rows(connection.execute(text(_DOCUMENT_SELECT + " ORDER BY m.publish_date DESC NULLS LAST LIMIT 5")))
            coverage = self._coverage(connection)
        return {"totals": totals, "categories": categories, "recent": recent, "coverage": coverage}

    def facets(self) -> dict[str, Any]:
        with self.engine.connect() as connection:
            categories = _rows(
                connection.execute(
                    text(
                        """
                        SELECT c.category_id AS value, c.category_name AS label, count(m.media_id) AS count
                        FROM doc_categories c
                        LEFT JOIN media m ON m.category_id = c.category_id
                        GROUP BY c.category_id, c.category_name
                        HAVING count(m.media_id) > 0
                        ORDER BY label
                        """
                    )
                )
            )
        return {"categories": categories}

    def documents(
        self,
        *,
        search: str = "",
        category_id: int | None = None,
        sort: str = "newest",
        limit: int = 100,
        version: str = "2.0",
    ) -> dict[str, Any]:
        version = _validate_extraction_version(version)
        clauses = ["1 = 1"]
        params: dict[str, Any] = {"limit": min(max(limit, 1), 200)}
        if search and version == "1.0":
            clauses.append(
                """(
                  lower(m.title) LIKE :search
                  OR lower(coalesce(m.description, '')) LIKE :search
                  OR EXISTS (SELECT 1 FROM doc_topics x WHERE x.content_id = dc.content_id AND lower(x.topic) LIKE :search)
                  OR EXISTS (SELECT 1 FROM doc_keywords x WHERE x.content_id = dc.content_id AND lower(x.keyword) LIKE :search)
                  OR EXISTS (SELECT 1 FROM doc_participants x WHERE x.content_id = dc.content_id AND lower(x.name) LIKE :search)
                  OR EXISTS (
                    SELECT 1 FROM doc_entities x JOIN entities_catalog e ON e.entity_id = x.entity_id
                    WHERE x.content_id = dc.content_id AND lower(e.label) LIKE :search
                  )
                  OR EXISTS (
                    SELECT 1 FROM doc_references x JOIN references_catalog r ON r.reference_id = x.reference_id
                    WHERE x.content_id = dc.content_id AND lower(r.label) LIKE :search
                  )
                )"""
            )
            params["search"] = f"%{search.lower()}%"
        if category_id is not None:
            clauses.append("m.category_id = :category_id")
            params["category_id"] = category_id
        order_by = SORT_SQL.get(sort, SORT_SQL["newest"])
        query = text(
            _DOCUMENT_SELECT
            + f" WHERE {' AND '.join(clauses)} ORDER BY {order_by} LIMIT :limit"
        )
        with self.engine.connect() as connection:
            documents = _rows(connection.execute(query, params))
            if version == "2.0":
                self._overlay_latest_v2_summaries(connection, documents)
        if search and version == "2.0":
            search_key = search.lower()
            documents = [
                document
                for document in documents
                if search_key in str(document.get("_extraction_search_blob") or "").lower()
                or search_key in str(document.get("title") or "").lower()
                or search_key in str(document.get("description") or "").lower()
            ]
        for document in documents:
            document.pop("_extraction_search_blob", None)
        return {"documents": documents, "count": len(documents), "version": version}

    @staticmethod
    def _overlay_latest_v2_summaries(connection: Any, documents: list[dict[str, Any]]) -> None:
        content_ids = [int(document["content_id"]) for document in documents if document.get("content_id") is not None]
        if not content_ids:
            return
        statement = text(
            """
            SELECT DISTINCT ON (content_id)
                   run_id, content_id, release_version, run_mode, status, extractor_model,
                   resolver_model, learner_model, manifest, raw_output, resolved_output,
                   input_characters, chunk_count, started_at, completed_at,
                   EXTRACT(EPOCH FROM (completed_at - started_at)) AS duration_seconds
            FROM extraction_runs
            WHERE status = 'completed' AND content_id IN :content_ids
            ORDER BY content_id, completed_at DESC NULLS LAST, run_id DESC
            """
        ).bindparams(bindparam("content_ids", expanding=True))
        try:
            runs = _rows(connection.execute(statement, {"content_ids": content_ids}))
        except SQLAlchemyError:
            return
        by_content = {int(run["content_id"]): run for run in runs}
        for document in documents:
            run = by_content.get(int(document["content_id"])) if document.get("content_id") is not None else None
            if not run:
                document["latest_extraction"] = "legacy"
                continue
            snapshot = _v2_extraction_snapshot(run)
            data = snapshot["data"]
            document["extraction_version"] = snapshot["version"]
            document["extraction_label"] = snapshot["label"]
            document["latest_extraction"] = "v2"
            document["latest_extraction_run_id"] = snapshot["run_id"]
            document["extraction_completed_at"] = snapshot["completed_at"]
            document["topic_count"] = len(data["topics"])
            document["keyword_count"] = len(data["keywords"])
            document["entity_count"] = len(data["entities"])
            document["reference_count"] = len(data["references"])
            document["participant_count"] = len(data["participants"])
            document["_extraction_search_blob"] = " ".join(
                [
                    *data["topics"],
                    *data["keywords"],
                    *(item["name"] for item in data["participants"]),
                    *(item["label"] for item in data["entities"]),
                    *(item["label"] for item in data["references"]),
                ]
            )
            if data.get("category"):
                document["category_name"] = data["category"]

    def document(self, media_id: str) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            document = _one(
                connection.execute(
                    text(_DOCUMENT_SELECT + " WHERE m.media_id = :media_id LIMIT 1"),
                    {"media_id": media_id},
                )
            )
            if document is None:
                return None
            content_id = document["content_id"]
            content = _one(
                connection.execute(
                    text(
                        """
                        SELECT full_text, formatted_text, timestamped_segments, language,
                               word_count, character_count, metadata_updated_at
                        FROM doc_content WHERE content_id = :content_id
                        """
                    ),
                    {"content_id": content_id},
                )
            )
            topics = connection.execute(
                text("SELECT topic FROM doc_topics WHERE content_id = :content_id ORDER BY topic"),
                {"content_id": content_id},
            ).scalars().all()
            keywords = connection.execute(
                text("SELECT keyword FROM doc_keywords WHERE content_id = :content_id ORDER BY keyword"),
                {"content_id": content_id},
            ).scalars().all()
            participants = _rows(
                connection.execute(
                    text("SELECT name, role FROM doc_participants WHERE content_id = :content_id ORDER BY name"),
                    {"content_id": content_id},
                )
            )
            entities = _rows(
                connection.execute(
                    text(
                        """
                        SELECT e.entity_id AS id, e.label, e.type, de.context, de.usage,
                               de.mention_count, de.ref
                        FROM doc_entities de
                        JOIN entities_catalog e ON e.entity_id = de.entity_id
                        WHERE de.content_id = :content_id
                        ORDER BY de.mention_count DESC, e.label
                        """
                    ),
                    {"content_id": content_id},
                )
            )
            references = _rows(
                connection.execute(
                    text(
                        """
                        SELECT r.reference_id AS id, r.label, r.type, r.identifiers,
                               dr.context, dr.usage, dr.mention_count, dr.ref, dr.attributes
                        FROM doc_references dr
                        JOIN references_catalog r ON r.reference_id = dr.reference_id
                        WHERE dr.content_id = :content_id
                        ORDER BY dr.mention_count DESC, r.label
                        """
                    ),
                    {"content_id": content_id},
                )
            )
            try:
                runs = _rows(
                    connection.execute(
                        text(
                            """
                            SELECT run_id, content_id, media_id, release_version, run_mode, status,
                                   extractor_model, resolver_model, learner_model, manifest,
                                   raw_output, resolved_output, input_characters, chunk_count,
                                   started_at, completed_at,
                                   EXTRACT(EPOCH FROM (completed_at - started_at)) AS duration_seconds
                            FROM extraction_runs
                            WHERE content_id = :content_id
                            ORDER BY started_at DESC, run_id DESC
                            LIMIT 20
                            """
                        ),
                        {"content_id": content_id},
                    )
                )
            except SQLAlchemyError:
                runs = []
        document["content"] = content
        document["extractions"] = {
            "topics": topics,
            "keywords": keywords,
            "participants": participants,
            "entities": entities,
            "references": references,
        }
        document["extraction_history"] = self._document_extraction_history(document, runs=runs)
        document["active_extraction_key"] = document["extraction_history"][0]["key"]
        return document

    def _document_extraction_history(
        self,
        document: dict[str, Any],
        *,
        runs: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        for run in runs:
            if run.get("status") != "completed" or not run.get("resolved_output"):
                continue
            history.append(_v2_extraction_snapshot(run))

        history.append(
            {
                "key": f"v1-{document.get('extraction_version') or '1.0'}",
                "version": str(document.get("extraction_version") or "1.0"),
                "label": f"V{document.get('extraction_version') or '1.0'}",
                "status": "production",
                "run_id": None,
                "run_mode": "production",
                "model": None,
                "profile_version_ids": [],
                "completed_at": document.get("content", {}).get("metadata_updated_at"),
                "data": document["extractions"],
            }
        )
        return history

    def atlas(
        self,
        *,
        kind: str,
        search: str = "",
        limit: int = 100,
        version: str = "2.0",
    ) -> dict[str, Any]:
        if kind not in ATLAS_KINDS:
            raise ValueError(f"Unsupported atlas kind: {kind}")
        version = _validate_extraction_version(version)
        params: dict[str, Any] = {"limit": min(max(limit, 1), 250)}
        search_clause = ""
        if search:
            search_clause = " WHERE lower(label) LIKE :search"
            params["search"] = f"%{search.lower()}%"
        with self.engine.connect() as connection:
            if version == "2.0" and kind not in {"category", "channel"}:
                items = self._v2_atlas_items(connection, kind=kind)
                if search:
                    items = [item for item in items if search.lower() in item["label"].lower()]
                items = items[: params["limit"]]
            else:
                source_sql = _ATLAS_SOURCE_SQL[kind]
                items = _rows(
                    connection.execute(
                        text(
                            f"SELECT * FROM ({source_sql}) atlas{search_clause} "
                            "ORDER BY document_count DESC, mention_count DESC, label LIMIT :limit"
                        ),
                        params,
                    )
                )
        return {"kind": kind, "items": items, "count": len(items), "version": version}

    def atlas_documents(
        self,
        *,
        kind: str,
        key: str,
        version: str = "2.0",
    ) -> dict[str, Any]:
        if kind not in ATLAS_KINDS:
            raise ValueError(f"Unsupported atlas kind: {kind}")
        version = _validate_extraction_version(version)
        id_value: int | str = int(key) if kind == "category" or (version == "1.0" and kind in {"entity", "reference"}) else key
        with self.engine.connect() as connection:
            if version == "2.0" and kind not in {"category", "channel"}:
                item, documents = self._v2_atlas_documents(connection, kind=kind, key=key)
            else:
                item = _one(connection.execute(text(_ATLAS_ITEM_SQL[kind]), {"key": id_value}))
                documents = _rows(connection.execute(text(_ATLAS_DOCUMENT_SQL[kind]), {"key": id_value}))
            profile = self._collection_profile(
                connection,
                kind=kind,
                key=id_value,
                version=version,
            ) if kind in {"category", "channel"} else None
        return {"kind": kind, "item": item, "documents": documents, "profile": profile, "version": version}

    def _collection_profile(
        self,
        connection: Any,
        *,
        kind: str,
        key: int | str,
        version: str,
    ) -> dict[str, Any]:
        if version == "2.0":
            corpus = self._latest_v2_corpus(connection)
            selected = [
                item
                for item in corpus
                if (
                    kind == "category" and item["document"].get("category_id") == key
                ) or (
                    kind == "channel" and item["document"].get("channel_id") == key
                )
            ]
            stats = self._collection_stats(connection, kind=kind, key=key)
            return {
                "stats": stats,
                "topics": self._aggregate_v2_items(selected, kind="topic")[:8],
                "entities": self._aggregate_v2_items(selected, kind="entity")[:8],
                "references": self._aggregate_v2_items(selected, kind="reference")[:8],
            }
        return self._collection_profile_v1(connection, kind=kind, key=key)

    @staticmethod
    def _collection_profile_v1(connection: Any, *, kind: str, key: int | str) -> dict[str, Any]:
        media_condition = "m.category_id = :key" if kind == "category" else "m.channel_id = :key"
        stats = _one(
            connection.execute(
                text(
                    f"""
                    SELECT count(DISTINCT m.media_id) AS documents,
                           coalesce(sum(dc.word_count), 0) AS words,
                           coalesce(sum(m.views), 0) AS views,
                           coalesce(sum(m.duration), 0) AS duration_seconds
                    FROM media m
                    LEFT JOIN doc_content dc ON dc.media_id = m.media_id AND dc.chunk_type = 'complete'
                    WHERE {media_condition}
                    """
                ),
                {"key": key},
            )
        )
        topics = _rows(
            connection.execute(
                text(
                    f"""
                    SELECT dt.topic AS key, dt.topic AS label, count(DISTINCT dc.content_id) AS count
                    FROM media m
                    JOIN doc_content dc ON dc.media_id = m.media_id AND dc.chunk_type = 'complete'
                    JOIN doc_topics dt ON dt.content_id = dc.content_id
                    WHERE {media_condition}
                    GROUP BY dt.topic ORDER BY count DESC, label LIMIT 8
                    """
                ),
                {"key": key},
            )
        )
        entities = _rows(
            connection.execute(
                text(
                    f"""
                    SELECT e.entity_id AS key, e.label, count(DISTINCT dc.content_id) AS count
                    FROM media m
                    JOIN doc_content dc ON dc.media_id = m.media_id AND dc.chunk_type = 'complete'
                    JOIN doc_entities de ON de.content_id = dc.content_id
                    JOIN entities_catalog e ON e.entity_id = de.entity_id
                    WHERE {media_condition}
                    GROUP BY e.entity_id, e.label ORDER BY count DESC, label LIMIT 8
                    """
                ),
                {"key": key},
            )
        )
        references = _rows(
            connection.execute(
                text(
                    f"""
                    SELECT r.reference_id AS key, r.label, count(DISTINCT dc.content_id) AS count
                    FROM media m
                    JOIN doc_content dc ON dc.media_id = m.media_id AND dc.chunk_type = 'complete'
                    JOIN doc_references dr ON dr.content_id = dc.content_id
                    JOIN references_catalog r ON r.reference_id = dr.reference_id
                    WHERE {media_condition}
                    GROUP BY r.reference_id, r.label ORDER BY count DESC, label LIMIT 8
                    """
                ),
                {"key": key},
            )
        )
        return {"stats": stats, "topics": topics, "entities": entities, "references": references}

    @staticmethod
    def _collection_stats(connection: Any, *, kind: str, key: int | str) -> dict[str, Any] | None:
        media_condition = "m.category_id = :key" if kind == "category" else "m.channel_id = :key"
        return _one(
            connection.execute(
                text(
                    f"""
                    SELECT count(DISTINCT m.media_id) AS documents,
                           coalesce(sum(dc.word_count), 0) AS words,
                           coalesce(sum(m.views), 0) AS views,
                           coalesce(sum(m.duration), 0) AS duration_seconds
                    FROM media m
                    LEFT JOIN doc_content dc ON dc.media_id = m.media_id AND dc.chunk_type = 'complete'
                    WHERE {media_condition}
                    """
                ),
                {"key": key},
            )
        )

    def _latest_v2_corpus(self, connection: Any) -> list[dict[str, Any]]:
        rows = _rows(
            connection.execute(
                text(
                    """
                    WITH latest AS (
                        SELECT DISTINCT ON (content_id)
                               run_id, content_id, media_id, release_version, run_mode, status,
                               extractor_model, resolver_model, learner_model, manifest,
                               raw_output, resolved_output, input_characters, chunk_count,
                               started_at, completed_at,
                               EXTRACT(EPOCH FROM (completed_at - started_at)) AS duration_seconds
                        FROM extraction_runs
                        WHERE status = 'completed' AND release_version = '2.0'
                        ORDER BY content_id, completed_at DESC NULLS LAST, run_id DESC
                    )
                    SELECT latest.*, m.title, m.thumbnail_url, m.publish_date, m.views, m.duration,
                           m.category_id, m.channel_id, c.category_name, ch.channel_name, dc.word_count
                    FROM latest
                    JOIN media m ON m.media_id = latest.media_id
                    JOIN doc_content dc ON dc.content_id = latest.content_id
                    LEFT JOIN doc_categories c ON c.category_id = m.category_id
                    LEFT JOIN channels ch ON ch.channel_id = m.channel_id
                    """
                )
            )
        )
        corpus: list[dict[str, Any]] = []
        for row in rows:
            snapshot = _v2_extraction_snapshot(row)
            if snapshot["data"].get("category"):
                row["category_name"] = snapshot["data"]["category"]
            corpus.append({"document": row, "snapshot": snapshot})
        return corpus

    def _v2_atlas_items(self, connection: Any, *, kind: str) -> list[dict[str, Any]]:
        return self._aggregate_v2_items(self._latest_v2_corpus(connection), kind=kind)

    @staticmethod
    def _aggregate_v2_items(corpus: Sequence[dict[str, Any]], *, kind: str) -> list[dict[str, Any]]:
        aggregated: dict[str, dict[str, Any]] = {}
        for entry in corpus:
            content_id = int(entry["document"]["content_id"])
            seen_in_document: set[str] = set()
            for record in _v2_records(entry["snapshot"]["data"], kind=kind):
                normalized = _normalize_atlas_key(record["label"])
                if not normalized:
                    continue
                item = aggregated.setdefault(
                    normalized,
                    {
                        "key": record["label"],
                        "label": record["label"],
                        "type": record["type"],
                        "document_ids": set(),
                        "mention_count": 0,
                    },
                )
                item["mention_count"] += int(record.get("mention_count") or 1)
                if normalized not in seen_in_document:
                    item["document_ids"].add(content_id)
                    seen_in_document.add(normalized)
        items = [
            {
                "key": item["key"],
                "label": item["label"],
                "type": item["type"],
                "document_count": len(item["document_ids"]),
                "mention_count": item["mention_count"],
                "count": len(item["document_ids"]),
            }
            for item in aggregated.values()
        ]
        if kind == "participant":
            multiword_first_names = {
                item["label"].split()[0].lower()
                for item in items
                if len(item["label"].split()) > 1
            }
            items = [
                item
                for item in items
                if not (
                    len(item["label"].split()) == 1
                    and item["label"].lower() in multiword_first_names
                )
            ]
        return sorted(items, key=lambda item: (-item["document_count"], -item["mention_count"], item["label"].lower()))

    def _v2_atlas_documents(
        self,
        connection: Any,
        *,
        kind: str,
        key: str,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        normalized_key = _normalize_atlas_key(key)
        item: dict[str, Any] | None = None
        documents: list[dict[str, Any]] = []
        for entry in self._latest_v2_corpus(connection):
            matches = [
                record
                for record in _v2_records(entry["snapshot"]["data"], kind=kind)
                if _normalize_atlas_key(record["label"]) == normalized_key
            ]
            if not matches:
                continue
            record = matches[0]
            if item is None:
                item = {"key": record["label"], "label": record["label"], "type": record["type"]}
            document = entry["document"]
            documents.append(
                {
                    "media_id": document["media_id"],
                    "title": document["title"],
                    "thumbnail_url": document.get("thumbnail_url"),
                    "publish_date": document.get("publish_date"),
                    "category_name": document.get("category_name"),
                    "content_id": document["content_id"],
                    "context": record.get("context") or record.get("evidence"),
                    "usage": record.get("usage") or kind,
                    "mention_count": sum(int(value.get("mention_count") or 1) for value in matches),
                }
            )
        documents.sort(key=lambda value: (value.get("publish_date") is None, value.get("publish_date")), reverse=True)
        return item, documents

    def quality(self, *, version: str = "2.0") -> dict[str, Any]:
        version = _validate_extraction_version(version)
        with self.engine.connect() as connection:
            if version == "2.0":
                corpus = self._latest_v2_corpus(connection)
                documents = []
                for entry in corpus:
                    source = entry["document"]
                    data = entry["snapshot"]["data"]
                    documents.append(
                        {
                            "media_id": source["media_id"],
                            "title": source["title"],
                            "thumbnail_url": source.get("thumbnail_url"),
                            "publish_date": source.get("publish_date"),
                            "category_name": data.get("category") or source.get("category_name"),
                            "content_id": source["content_id"],
                            "word_count": source.get("word_count"),
                            "topics": len(data["topics"]),
                            "keywords": len(data["keywords"]),
                            "participants": len(data["participants"]),
                            "entities": len(data["entities"]),
                            "references": len(data["references"]),
                        }
                    )
                labels = (
                    ("Category", "category_name"),
                    ("Topics", "topics"),
                    ("Keywords", "keywords"),
                    ("Participants", "participants"),
                    ("Entities", "entities"),
                    ("References", "references"),
                )
                coverage = [
                    {
                        "label": label,
                        "covered": sum(bool(document.get(field)) for document in documents),
                        "total": len(documents),
                        "percent": round(sum(bool(document.get(field)) for document in documents) / len(documents) * 100) if documents else 0,
                    }
                    for label, field in labels
                ]
            else:
                coverage = self._coverage(connection)
                documents = _rows(
                    connection.execute(
                        text(
                            """
                            SELECT m.media_id, m.title, m.thumbnail_url, m.publish_date,
                                   c.category_name, dc.content_id, dc.word_count,
                                   (SELECT count(*) FROM doc_topics x WHERE x.content_id = dc.content_id) AS topics,
                                   (SELECT count(*) FROM doc_keywords x WHERE x.content_id = dc.content_id) AS keywords,
                                   (SELECT count(*) FROM doc_participants x WHERE x.content_id = dc.content_id) AS participants,
                                   (SELECT count(*) FROM doc_entities x WHERE x.content_id = dc.content_id) AS entities,
                                   (SELECT count(*) FROM doc_references x WHERE x.content_id = dc.content_id) AS references
                            FROM media m
                            JOIN doc_content dc ON dc.media_id = m.media_id AND dc.chunk_type = 'complete'
                            LEFT JOIN doc_categories c ON c.category_id = m.category_id
                            ORDER BY m.publish_date DESC NULLS LAST, m.title
                            """
                        )
                    )
                )
        for document in documents:
            checks = [document.get("category_name"), document["topics"], document["keywords"], document["participants"], document["entities"], document["references"]]
            document["coverage_score"] = round(sum(bool(value) for value in checks) / len(checks) * 100)
        documents.sort(key=lambda item: (item["coverage_score"], item["title"]))
        return {"coverage": coverage, "documents": documents, "version": version}

    def pipeline_prompt(self, *, version: str = "2.0") -> dict[str, Any]:
        version = _validate_extraction_version(version)
        with self.engine.connect() as connection:
            categories = connection.execute(
                text("SELECT category_name FROM doc_categories ORDER BY lower(category_name)")
            ).scalars().all()
            versions = _rows(
                connection.execute(
                    text(
                        """
                        SELECT version, release_date, is_current, changes_summary, prompt_changes,
                               schema_changes, rollout_status, media_extracted, success_rate
                        FROM extraction_versions ORDER BY release_date DESC
                        """
                    )
                )
            )
            media_versions = _rows(
                connection.execute(
                    text(
                        """
                        SELECT extraction_version AS version, count(*) AS document_count
                        FROM media GROUP BY extraction_version ORDER BY extraction_version
                        """
                    )
                )
            )
            run_versions = _rows(
                connection.execute(
                    text(
                        """
                        SELECT release_version AS version,
                               count(DISTINCT content_id) AS document_count,
                               count(*) AS run_count
                        FROM extraction_runs
                        WHERE status = 'completed'
                        GROUP BY release_version
                        ORDER BY release_version
                        """
                    )
                )
            )
        inspection = (
            get_metadata_v2_prompt_inspection()
            if version == "2.0"
            else get_metadata_extraction_inspection(category_hints=categories)
        )
        inspection["version"] = version
        inspection["category_hints"] = categories
        inspection["version_registry"] = versions
        inspection["media_versions"] = media_versions
        inspection["run_versions"] = run_versions
        inspection["stages"] = (
            _v2_pipeline_stages(inspection)
            if version == "2.0"
            else _v1_pipeline_stages(inspection)
        )
        return inspection

    def extraction_v2_profiles(self) -> dict[str, Any]:
        try:
            profiles = list_extraction_profiles(engine=self.engine)
        except MetadataV2Error as exc:
            return {"available": False, "profiles": [], "detail": str(exc)}
        return {"available": True, "count": len(profiles), "profiles": profiles}

    def profile_contexts(self) -> dict[str, Any]:
        """Return the V2 profile hierarchy and its adjacent memory stores."""
        with self.engine.connect() as connection:
            profile_rows = _rows(
                connection.execute(
                    text(
                        """
                        SELECT p.profile_id, p.profile_name, p.scope, p.source_id, p.channel_id,
                               p.category_id, p.learning_mode, p.is_active, p.created_at, p.updated_at,
                               s.source_name, ch.channel_name, c.category_name,
                               v.profile_version_id, v.version, v.status, v.config,
                               v.created_at AS version_created_at, v.published_at, v.retired_at
                        FROM extraction_profiles p
                        LEFT JOIN media_sources s ON s.source_id = p.source_id
                        LEFT JOIN channels ch ON ch.channel_id = p.channel_id
                        LEFT JOIN doc_categories c ON c.category_id = p.category_id
                        JOIN extraction_profile_versions v ON v.profile_id = p.profile_id
                        ORDER BY p.profile_id, v.version DESC
                        """
                    )
                )
            )
            sources = _rows(
                connection.execute(
                    text(
                        """
                        SELECT s.source_id, s.source_name, s.description, s.is_active,
                               count(DISTINCT m.media_id) AS document_count
                        FROM media_sources s
                        LEFT JOIN media m ON m.source_id = s.source_id
                        GROUP BY s.source_id, s.source_name, s.description, s.is_active
                        ORDER BY lower(s.source_name)
                        """
                    )
                )
            )
            categories = _rows(
                connection.execute(
                    text(
                        """
                        SELECT c.category_id, c.category_name, c.description, c.is_default,
                               count(DISTINCT m.media_id) AS document_count
                        FROM doc_categories c
                        LEFT JOIN media m ON m.category_id = c.category_id
                        GROUP BY c.category_id, c.category_name, c.description, c.is_default
                        ORDER BY lower(c.category_name)
                        """
                    )
                )
            )
            channels = _rows(
                connection.execute(
                    text(
                        """
                        SELECT ch.channel_id, ch.channel_name, ch.source_id, ch.description,
                               ch.thumbnail_url, count(DISTINCT m.media_id) AS document_count
                        FROM channels ch
                        LEFT JOIN media m ON m.channel_id = ch.channel_id
                        GROUP BY ch.channel_id, ch.channel_name, ch.source_id,
                                 ch.description, ch.thumbnail_url
                        ORDER BY lower(ch.channel_name)
                        """
                    )
                )
            )
            channel_categories = _rows(
                connection.execute(
                    text(
                        """
                        SELECT m.channel_id, c.category_id, c.category_name,
                               count(DISTINCT m.media_id) AS document_count
                        FROM media m
                        JOIN doc_categories c ON c.category_id = m.category_id
                        WHERE m.channel_id IS NOT NULL
                        GROUP BY m.channel_id, c.category_id, c.category_name
                        ORDER BY m.channel_id, count(DISTINCT m.media_id) DESC, lower(c.category_name)
                        """
                    )
                )
            )
            aliases = _rows(
                connection.execute(
                    text(
                        """
                        SELECT a.alias_id, a.dimension, a.canonical_value, a.alias_text,
                               a.source_id, a.channel_id, a.category_id, a.confidence,
                               a.trust_status, a.evidence_count, a.is_blocked, a.created_by,
                               a.created_at, a.updated_at, s.source_name, ch.channel_name,
                               c.category_name
                        FROM metadata_value_aliases a
                        LEFT JOIN media_sources s ON s.source_id = a.source_id
                        LEFT JOIN channels ch ON ch.channel_id = a.channel_id
                        LEFT JOIN doc_categories c ON c.category_id = a.category_id
                        ORDER BY a.updated_at DESC, lower(a.alias_text)
                        """
                    )
                )
            )
            learning_events = _rows(
                connection.execute(
                    text(
                        """
                        SELECT e.event_id, e.profile_id, e.run_id, e.dimension, e.action,
                               e.raw_value, e.canonical_value, e.confidence, e.reason,
                               e.status, e.details, e.created_at, e.applied_at, e.reversed_at,
                               p.profile_name, r.media_id, m.title
                        FROM metadata_learning_events e
                        LEFT JOIN extraction_profiles p ON p.profile_id = e.profile_id
                        LEFT JOIN extraction_runs r ON r.run_id = e.run_id
                        LEFT JOIN media m ON m.media_id = r.media_id
                        ORDER BY e.created_at DESC
                        """
                    )
                )
            )

        grouped: dict[int, dict[str, Any]] = {}
        for row in profile_rows:
            profile_id = int(row["profile_id"])
            profile = grouped.setdefault(
                profile_id,
                {
                    key: row.get(key)
                    for key in (
                        "profile_id", "profile_name", "scope", "source_id", "source_name",
                        "channel_id", "channel_name", "category_id", "category_name",
                        "learning_mode", "is_active", "created_at", "updated_at",
                    )
                } | {"versions": []},
            )
            profile["versions"].append(
                {
                    "profile_version_id": row.get("profile_version_id"),
                    "version": row.get("version"),
                    "status": row.get("status"),
                    "config": row.get("config") or {},
                    "created_at": row.get("version_created_at"),
                    "published_at": row.get("published_at"),
                    "retired_at": row.get("retired_at"),
                }
            )

        categories_by_channel: dict[str, list[dict[str, Any]]] = {}
        for item in channel_categories:
            categories_by_channel.setdefault(str(item["channel_id"]), []).append(
                {key: item[key] for key in ("category_id", "category_name", "document_count")}
            )
        for channel in channels:
            channel["categories"] = categories_by_channel.get(str(channel["channel_id"]), [])

        return {
            "scope_order": ["global", "source", "category", "channel", "channel_category"],
            "profiles": list(grouped.values()),
            "sources": sources,
            "categories": categories,
            "channels": channels,
            "aliases": aliases,
            "learning_events": learning_events,
            "counts": {
                "profiles": len(grouped),
                "profile_versions": len(profile_rows),
                "aliases": len(aliases),
                "learning_events": len(learning_events),
            },
        }

    def save_profile_instructions(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        scope = str(payload.get("scope") or "")
        category_id = payload.get("category_id")
        return save_scoped_instruction_draft(
            engine=self.engine,
            scope=scope,
            category_id=int(category_id) if category_id is not None else None,
            channel_id=str(payload["channel_id"]) if payload.get("channel_id") else None,
            prompt_instructions=str(payload.get("prompt_instructions") or ""),
        )

    @staticmethod
    def preview_profile_instructions(payload: Mapping[str, Any]) -> dict[str, Any]:
        return preview_profile_instruction_contracts(payload.get("config"))

    def extraction_v2_runs(self, *, content_id: int | None = None, limit: int = 50) -> dict[str, Any]:
        try:
            runs = list_extraction_runs(
                engine=self.engine,
                content_id=content_id,
                limit=limit,
            )
        except MetadataV2Error as exc:
            return {"available": False, "runs": [], "detail": str(exc)}
        return {"available": True, "count": len(runs), "runs": runs}

    @staticmethod
    def _coverage(connection: Any) -> list[dict[str, Any]]:
        rows = connection.execute(
            text(
                """
                SELECT 'Category' AS label, count(*) AS covered FROM media WHERE category_id IS NOT NULL
                UNION ALL SELECT 'Topics', count(DISTINCT content_id) FROM doc_topics
                UNION ALL SELECT 'Keywords', count(DISTINCT content_id) FROM doc_keywords
                UNION ALL SELECT 'Participants', count(DISTINCT content_id) FROM doc_participants
                UNION ALL SELECT 'Entities', count(DISTINCT content_id) FROM doc_entities
                UNION ALL SELECT 'References', count(DISTINCT content_id) FROM doc_references
                """
            )
        )
        total = connection.execute(text("SELECT count(*) FROM doc_content WHERE chunk_type = 'complete'")).scalar_one()
        return [
            {"label": row.label, "covered": row.covered, "total": total, "percent": round(row.covered / total * 100) if total else 0}
            for row in rows
        ]


def _v2_extraction_snapshot(run: dict[str, Any]) -> dict[str, Any]:
    raw = dict(run.get("raw_output") or {})
    resolved = dict(run.get("resolved_output") or {})
    mentioned_participants = [
        item for item in resolved.get("participants", []) if is_mentioned_only_participant(item)
    ]
    participants = [
        {
            "name": item.get("canonical_name") or item.get("raw_name") or "Unnamed participant",
            "raw_name": item.get("raw_name"),
            "role": item.get("role"),
            "evidence": item.get("evidence"),
            "confidence": item.get("confidence"),
            "resolution_status": item.get("resolution_status"),
            "resolution_basis": item.get("resolution_basis"),
        }
        for item in resolved.get("participants", [])
        if item.get("resolution_status") != "rejected" and not is_mentioned_only_participant(item)
    ]
    entities = [
        {
            "label": item.get("canonical_label") or item.get("raw_label") or "Unresolved entity",
            "raw_label": item.get("raw_label"),
            "type": item.get("type") or "other",
            "context": item.get("context"),
            "usage": item.get("usage") or "neutral",
            "mention_count": 1,
            "evidence": item.get("evidence"),
            "confidence": item.get("confidence"),
            "resolution_status": item.get("resolution_status"),
            "resolution_basis": item.get("resolution_basis"),
        }
        for item in resolved.get("entities", [])
        if item.get("resolution_status") != "rejected"
    ]
    existing_entity_names = {_normalize_atlas_key(item["label"]) for item in entities}
    for item in mentioned_participants:
        name = item.get("canonical_name") or item.get("raw_name") or "Mentioned person"
        if _normalize_atlas_key(name) in existing_entity_names:
            continue
        entities.append(
            {
                "label": name,
                "raw_label": item.get("raw_name"),
                "type": "person",
                "context": item.get("role") or "Mentioned person; not present in this document.",
                "usage": "background",
                "mention_count": 1,
                "evidence": item.get("evidence"),
                "confidence": item.get("confidence"),
                "resolution_status": "trusted",
                "resolution_basis": "explicit",
            }
        )
    references = [
        {
            "label": item.get("canonical_label") or item.get("raw_mention") or "Unresolved reference",
            "raw_label": item.get("raw_mention"),
            "type": item.get("type") or "other",
            "context": item.get("context"),
            "usage": item.get("usage") or "neutral",
            "mention_count": 1,
            "evidence": item.get("evidence"),
            "confidence": item.get("confidence"),
            "resolution_status": item.get("resolution_status"),
            "resolution_basis": item.get("resolution_basis"),
            "scripture": item.get("scripture"),
        }
        for item in resolved.get("references", [])
        if item.get("resolution_status") != "rejected"
    ]
    release_version = str(run.get("release_version") or "2.0")
    started_at = run.get("started_at")
    completed_at = run.get("completed_at")
    duration_seconds = _json_value(run.get("duration_seconds"))
    if isinstance(started_at, datetime) and isinstance(completed_at, datetime):
        duration_seconds = round((completed_at - started_at).total_seconds(), 3)
    elif duration_seconds is not None:
        duration_seconds = round(float(duration_seconds), 3)
    return {
        "key": f"v{release_version}-run-{run.get('run_id')}",
        "version": release_version,
        "label": f"V{release_version} · Run {run.get('run_id')}",
        "status": "testing",
        "run_id": run.get("run_id"),
        "run_mode": run.get("run_mode"),
        "model": run.get("extractor_model"),
        "profile_version_ids": list((run.get("manifest") or {}).get("profile_version_ids", [])),
        "completed_at": run.get("completed_at"),
        "input_characters": run.get("input_characters"),
        "chunk_count": run.get("chunk_count"),
        "duration_seconds": duration_seconds,
        "usage": _run_usage_snapshot(run),
        "data": {
            "category": raw.get("category"),
            "topics": list(raw.get("topics") or []),
            "keywords": [value for value in (raw.get("keywords") or []) if not is_scripture_keyword(value)],
            "participants": participants,
            "entities": entities,
            "references": references,
        },
    }


def _run_usage_snapshot(run: Mapping[str, Any]) -> dict[str, Any]:
    manifest = dict(run.get("manifest") or {})
    recorded = manifest.get("usage")
    if isinstance(recorded, Mapping):
        return dict(recorded)
    steps = [
        {
            "step": "normalization",
            "label": "Transcript normalization",
            "kind": "deterministic",
            "invoked": True,
            "request_count": 0,
            "total_tokens": 0,
            "estimated_cost": 0.0,
        }
    ]
    labels = {
        "extractor": "Worker 1 · Extractor",
        "resolver": "Worker 2 · Resolver",
        "learner": "Worker 3 · Learner",
    }
    for key in ("extractor", "resolver", "learner"):
        worker = dict(manifest.get(key) or {})
        steps.append(
            {
                "step": key,
                "label": labels[key],
                "kind": "model",
                "invoked": bool(worker.get("invoked")),
                "model": worker.get("model") or run.get(f"{key}_model"),
                "reasoning_effort": worker.get("reasoning_effort"),
                "request_count": None,
                "input_tokens": None,
                "cached_input_tokens": None,
                "output_tokens": None,
                "reasoning_tokens": None,
                "total_tokens": None,
                "estimated_cost": None,
            }
        )
    steps.append(
        {
            "step": "persistence",
            "label": "Run persistence",
            "kind": "deterministic",
            "invoked": True,
            "request_count": 0,
            "total_tokens": 0,
            "estimated_cost": 0.0,
        }
    )
    return {
        "tracking_status": "not_recorded",
        "pricing_basis": "Token usage was not retained for this historical run.",
        "total_tokens": None,
        "input_tokens": None,
        "cached_input_tokens": None,
        "output_tokens": None,
        "reasoning_tokens": None,
        "estimated_cost": None,
        "steps": steps,
    }


_DOCUMENT_SELECT = """
SELECT m.media_id, m.title, m.description, m.url, m.thumbnail_url, m.duration,
       m.publish_date, m.views, m.likes, m.comments, m.source_type,
       m.openai_sync_status, m.extraction_version, m.extraction_quality_score,
       m.ingestion_date, c.category_id, c.category_name,
       ch.channel_name, dc.content_id, dc.word_count, dc.character_count, dc.language,
       (SELECT count(*) FROM doc_topics x WHERE x.content_id = dc.content_id) AS topic_count,
       (SELECT count(*) FROM doc_keywords x WHERE x.content_id = dc.content_id) AS keyword_count,
       (SELECT count(*) FROM doc_entities x WHERE x.content_id = dc.content_id) AS entity_count,
       (SELECT count(*) FROM doc_references x WHERE x.content_id = dc.content_id) AS reference_count
FROM media m
LEFT JOIN doc_categories c ON c.category_id = m.category_id
LEFT JOIN channels ch ON ch.channel_id = m.channel_id
LEFT JOIN doc_content dc ON dc.media_id = m.media_id AND dc.chunk_type = 'complete'
"""

_ATLAS_SOURCE_SQL = {
    "category": "SELECT c.category_id AS key, c.category_name AS label, 'category' AS type, count(DISTINCT m.media_id) AS document_count, count(DISTINCT m.media_id) AS mention_count FROM doc_categories c JOIN media m ON m.category_id = c.category_id GROUP BY c.category_id, c.category_name",
    "channel": "SELECT ch.channel_id AS key, ch.channel_name AS label, 'channel' AS type, count(DISTINCT m.media_id) AS document_count, count(DISTINCT m.media_id) AS mention_count FROM channels ch JOIN media m ON m.channel_id = ch.channel_id GROUP BY ch.channel_id, ch.channel_name",
    "topic": "SELECT topic AS key, topic AS label, 'topic' AS type, count(DISTINCT content_id) AS document_count, count(*) AS mention_count FROM doc_topics GROUP BY topic",
    "keyword": "SELECT keyword AS key, keyword AS label, 'keyword' AS type, count(DISTINCT content_id) AS document_count, count(*) AS mention_count FROM doc_keywords GROUP BY keyword",
    "participant": "SELECT name AS key, name AS label, coalesce(max(role), 'participant') AS type, count(DISTINCT content_id) AS document_count, count(*) AS mention_count FROM doc_participants GROUP BY name",
    "entity": "SELECT e.entity_id AS key, e.label, e.type, count(DISTINCT de.content_id) AS document_count, coalesce(sum(de.mention_count), 0) AS mention_count FROM entities_catalog e JOIN doc_entities de ON de.entity_id = e.entity_id GROUP BY e.entity_id, e.label, e.type",
    "reference": "SELECT r.reference_id AS key, r.label, r.type, count(DISTINCT dr.content_id) AS document_count, coalesce(sum(dr.mention_count), 0) AS mention_count FROM references_catalog r JOIN doc_references dr ON dr.reference_id = r.reference_id GROUP BY r.reference_id, r.label, r.type",
}

_ATLAS_ITEM_SQL = {
    "category": "SELECT category_id AS key, category_name AS label, 'category' AS type, description AS details FROM doc_categories WHERE category_id = :key",
    "channel": "SELECT channel_id AS key, channel_name AS label, 'channel' AS type, channel_url AS details FROM channels WHERE channel_id = :key",
    "topic": "SELECT :key AS key, :key AS label, 'topic' AS type",
    "keyword": "SELECT :key AS key, :key AS label, 'keyword' AS type",
    "participant": "SELECT :key AS key, :key AS label, coalesce(max(role), 'participant') AS type FROM doc_participants WHERE name = :key",
    "entity": "SELECT entity_id AS key, label, type, aliases AS details FROM entities_catalog WHERE entity_id = :key",
    "reference": "SELECT reference_id AS key, label, type, identifiers AS details FROM references_catalog WHERE reference_id = :key",
}

_ATLAS_DOCUMENT_BASE = """
SELECT m.media_id, m.title, m.thumbnail_url, m.publish_date, c.category_name,
       dc.content_id, {context} AS context, {usage} AS usage, {mentions} AS mention_count
FROM {link_table} link
JOIN doc_content dc ON dc.content_id = link.content_id
JOIN media m ON m.media_id = dc.media_id
LEFT JOIN doc_categories c ON c.category_id = m.category_id
WHERE {condition}
ORDER BY m.publish_date DESC NULLS LAST, m.title
"""

_ATLAS_DOCUMENT_SQL = {
    "category": """
        SELECT m.media_id, m.title, m.thumbnail_url, m.publish_date, c.category_name,
               dc.content_id, NULL AS context, 'category' AS usage, 1 AS mention_count
        FROM media m
        JOIN doc_content dc ON dc.media_id = m.media_id AND dc.chunk_type = 'complete'
        LEFT JOIN doc_categories c ON c.category_id = m.category_id
        WHERE m.category_id = :key
        ORDER BY m.publish_date DESC NULLS LAST, m.title
    """,
    "channel": """
        SELECT m.media_id, m.title, m.thumbnail_url, m.publish_date, c.category_name,
               dc.content_id, NULL AS context, 'channel' AS usage, 1 AS mention_count
        FROM media m
        JOIN doc_content dc ON dc.media_id = m.media_id AND dc.chunk_type = 'complete'
        LEFT JOIN doc_categories c ON c.category_id = m.category_id
        WHERE m.channel_id = :key
        ORDER BY m.publish_date DESC NULLS LAST, m.title
    """,
    "topic": _ATLAS_DOCUMENT_BASE.format(context="NULL", usage="'topic'", mentions="1", link_table="doc_topics", condition="link.topic = :key"),
    "keyword": _ATLAS_DOCUMENT_BASE.format(context="NULL", usage="'keyword'", mentions="1", link_table="doc_keywords", condition="link.keyword = :key"),
    "participant": _ATLAS_DOCUMENT_BASE.format(context="link.role", usage="'participant'", mentions="1", link_table="doc_participants", condition="link.name = :key"),
    "entity": _ATLAS_DOCUMENT_BASE.format(context="link.context", usage="link.usage", mentions="link.mention_count", link_table="doc_entities", condition="link.entity_id = :key"),
    "reference": _ATLAS_DOCUMENT_BASE.format(context="link.context", usage="link.usage", mentions="link.mention_count", link_table="doc_references", condition="link.reference_id = :key"),
}


def make_handler(repository: ExplorerRepository) -> type[BaseHTTPRequestHandler]:
    class ExplorerHandler(BaseHTTPRequestHandler):
        repo = repository

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            parsed = urlparse(self.path)
            try:
                if parsed.path.startswith("/api/"):
                    self._handle_api(parsed.path, parse_qs(parsed.query))
                else:
                    self._serve_static(parsed.path)
            except (ValueError, TypeError) as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:  # pragma: no cover - defensive HTTP boundary
                self._send_json({"error": "Explorer request failed", "detail": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_HEAD(self) -> None:  # noqa: N802 - stdlib handler API
            parsed = urlparse(self.path)
            relative = "index.html" if parsed.path in {"", "/"} else unquote(parsed.path.lstrip("/"))
            candidate = (STATIC_DIR / relative).resolve()
            if not candidate.is_file():
                candidate = STATIC_DIR / "index.html"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mimetypes.guess_type(candidate.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(candidate.stat().st_size))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            parsed = urlparse(self.path)
            try:
                if parsed.path not in {
                    "/api/extraction/v2/profile-instructions",
                    "/api/extraction/v2/instruction-preview",
                }:
                    return self._send_json({"error": "API route not found"}, HTTPStatus.NOT_FOUND)
                payload = self._read_json()
                if parsed.path == "/api/extraction/v2/profile-instructions":
                    return self._send_json(self.repo.save_profile_instructions(payload), HTTPStatus.CREATED)
                if parsed.path == "/api/extraction/v2/instruction-preview":
                    return self._send_json(self.repo.preview_profile_instructions(payload))
            except (ValueError, TypeError, MetadataV2Error) as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:  # pragma: no cover - defensive HTTP boundary
                self._send_json({"error": "Explorer request failed", "detail": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def _handle_api(self, path: str, query: dict[str, list[str]]) -> None:
            get = lambda key, default="": query.get(key, [default])[0]
            if path == "/api/overview":
                return self._send_json(self.repo.overview())
            if path == "/api/facets":
                return self._send_json(self.repo.facets())
            if path == "/api/documents":
                category = get("category")
                return self._send_json(self.repo.documents(search=get("search"), category_id=int(category) if category else None, sort=get("sort", "newest"), version=get("version", "2.0")))
            if path.startswith("/api/documents/"):
                document = self.repo.document(unquote(path.removeprefix("/api/documents/")))
                return self._send_json(document or {"error": "Document not found"}, HTTPStatus.OK if document else HTTPStatus.NOT_FOUND)
            if path == "/api/atlas":
                return self._send_json(self.repo.atlas(kind=get("kind", "entity"), search=get("search"), version=get("version", "2.0")))
            if path == "/api/atlas/documents":
                return self._send_json(self.repo.atlas_documents(kind=get("kind", "entity"), key=get("key"), version=get("version", "2.0")))
            if path == "/api/quality":
                return self._send_json(self.repo.quality(version=get("version", "2.0")))
            if path == "/api/pipeline/prompt":
                return self._send_json(self.repo.pipeline_prompt(version=get("version", "2.0")))
            if path == "/api/extraction/v2/profiles":
                return self._send_json(self.repo.extraction_v2_profiles())
            if path == "/api/extraction/v2/contexts":
                return self._send_json(self.repo.profile_contexts())
            if path == "/api/extraction/v2/runs":
                content_id = int(get("content_id")) if get("content_id") else None
                limit = int(get("limit", "50"))
                return self._send_json(self.repo.extraction_v2_runs(content_id=content_id, limit=limit))
            self._send_json({"error": "API route not found"}, HTTPStatus.NOT_FOUND)

        def _serve_static(self, path: str) -> None:
            relative = "index.html" if path in {"", "/"} else unquote(path.lstrip("/"))
            candidate = (STATIC_DIR / relative).resolve()
            if STATIC_DIR.resolve() not in candidate.parents and candidate != STATIC_DIR.resolve():
                return self.send_error(HTTPStatus.NOT_FOUND)
            if not candidate.is_file():
                candidate = STATIC_DIR / "index.html"
            content = candidate.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mimetypes.guess_type(candidate.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                return

        def _read_json(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ValueError("Invalid Content-Length header.") from exc
            if length <= 0 or length > 100_000:
                raise ValueError("JSON request body must be between 1 and 100,000 bytes.")
            try:
                payload = json.loads(self.rfile.read(length))
            except json.JSONDecodeError as exc:
                raise ValueError("Request body must be valid JSON.") from exc
            if not isinstance(payload, dict):
                raise ValueError("Request body must be a JSON object.")
            return payload

        def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            content = json.dumps(payload, default=_json_value).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                return

        def log_message(self, format: str, *args: Any) -> None:
            return

    return ExplorerHandler


def serve_explorer(*, engine: Engine, host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(ExplorerRepository(engine)))
    print(f"Learn From explorer running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        engine.dispose()

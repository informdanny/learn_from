"""Read the existing metadata registry and normalize evidence-backed business facets.

Nothing is seeded on import or read. Vocabulary expansion remains a proposal until
explicit acceptance; extraction snapshots are never rewritten by catalog changes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from importlib.resources import files
from typing import Any, get_args

from sqlalchemy import MetaData, Table, func, insert, select, update
from sqlalchemy.engine import Engine


CATALOG_VERSION = "business-facets-v1"
NORMALIZATION_VERSION = "evidence-quotes-v2"
_RESERVED = {"video_id", "channel_id", "content_id", "language", "source_hash", "published_at", "metadata_run_id", "metadata_version"}
BUSINESS_FIELDS = {
    "industry": {
        "label": "Industry", "policy": "expandable",
        "description": "Industry explicitly discussed. Use the most specific supported industry; omit if unclear.",
        "values": ["liquidation_auctions", "christmas_light_installation", "ai_freelancing", "productized_ai_services", "home_services", "construction_trades", "real_estate", "short_term_rentals", "ecommerce", "retail", "software", "marketing_services", "professional_services", "education", "health_fitness", "hospitality", "manufacturing", "logistics"],
    },
    "business_model": {
        "label": "Business model", "policy": "expandable",
        "description": "How the discussed business earns revenue, only when supported by a transcript quote.",
        "values": ["local_service", "freelancing", "productized_service", "agency", "consulting", "subscription", "software_as_a_service", "marketplace", "resale", "retail", "ecommerce", "rental", "licensing", "franchise", "education_training", "advertising_sponsorship"],
    },
    "business_stage": {
        "label": "Business stage", "policy": "fixed",
        "description": "Stage of the discussed business or advice. Do not infer the stage from the speaker's fame.",
        "values": ["idea", "validation", "launch", "growth", "scaling", "mature", "turnaround", "exit"],
    },
    "decision_area": {
        "label": "Decision area", "policy": "expandable",
        "description": "Business decision addressed by the passage; do not infer a recommendation from a passing mention.",
        "values": ["idea_selection", "market_research", "customer_acquisition", "sales", "pricing", "offer_design", "operations", "hiring", "finance", "funding", "automation", "retention", "expansion", "acquisition", "exit", "risk"],
    },
}


class MetadataCatalogError(ValueError):
    """Catalog configuration cannot be interpreted safely."""


def _tables(connection: Any, *names: str) -> dict[str, Table]:
    metadata = MetaData()
    return {name: Table(name, metadata, autoload_with=connection) for name in names}


def _normalize(value: str) -> str:
    return re.sub(r"[\s-]+", "_", value.strip().casefold())


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _active_channel_config(connection: Any, tables: Mapping[str, Table], channel_id: str | None, source_id: str) -> dict[str, Any]:
    if channel_id is None:
        return {}
    profiles, versions = tables["extraction_profiles"], tables["extraction_profile_versions"]
    rows = connection.execute(select(versions.c.config).select_from(profiles.join(versions))
        .where(profiles.c.channel_id == channel_id, profiles.c.source_id == source_id,
               profiles.c.scope == "channel", profiles.c.is_active.is_(True), versions.c.status == "active")
        .order_by(versions.c.profile_version_id)).scalars().all()
    if len(rows) > 1:
        raise MetadataCatalogError("Multiple active channel profiles require review.")
    return dict(rows[0]) if rows else {}


def _core_dimensions(categories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from .metadata_v2 import EntityCandidate, ReferenceCandidate

    return [
        {"key": "category", "type": "string", "policy": "fixed", "options": categories},
        {"key": "topics", "type": "string_array", "policy": "open"},
        {"key": "keywords", "type": "string_array", "policy": "open"},
        {"key": "participants", "type": "object_array", "policy": "open"},
        {"key": "entities", "type": "object_array", "policy": "open", "type_options": list(get_args(EntityCandidate.model_fields["type"].annotation))},
        {"key": "references", "type": "object_array", "policy": "open", "type_options": list(get_args(ReferenceCandidate.model_fields["type"].annotation))},
    ]


def get_metadata_schema(*, engine: Engine, channel_id: str | None = None,
                        category_id: int | None = None, source_id: str = "youtube",
                        values_limit: int = 50, value_offset: int = 0,
                        field: str | None = None) -> dict[str, Any]:
    """Return applicable active fields and bounded vocabulary pages, without writes.

    Scope precedence is category > source > universal. Same-precedence duplicate
    keys are rejected rather than picking an arbitrary row. Channel profile config
    can supply the category without changing legacy media.category_id values.
    """
    if isinstance(values_limit, bool) or not isinstance(values_limit, int) or not 1 <= values_limit <= 1000:
        raise MetadataCatalogError("values_limit must be between 1 and 1000.")
    if isinstance(value_offset, bool) or not isinstance(value_offset, int) or value_offset < 0:
        raise MetadataCatalogError("value_offset must be a nonnegative integer.")
    with engine.connect() as connection:
        t = _tables(connection, "metadata_schemas", "metadata_values", "doc_categories", "extraction_profiles", "extraction_profile_versions")
        config = _active_channel_config(connection, t, channel_id, source_id)
        profile_catalog = config.get("metadata_catalog", {})
        if category_id is None:
            category_id = profile_catalog.get("category_id")
        elif profile_catalog.get("category_id") not in (None, category_id):
            profile_catalog = {}
        categories = [dict(r) for r in connection.execute(select(t["doc_categories"].c.category_id, t["doc_categories"].c.category_name).order_by(t["doc_categories"].c.category_id)).mappings()]
        if category_id is not None and category_id not in {r["category_id"] for r in categories}:
            raise MetadataCatalogError("Unknown category_id.")
        selected: dict[str, tuple[int, dict[str, Any]]] = {}
        for row in connection.execute(select(t["metadata_schemas"]).where(t["metadata_schemas"].c.is_active.is_(True))).mappings():
            row = dict(row)
            if field is not None and row["attribute_key"] != field:
                continue
            scope = row["scope"]
            rank = None
            if scope == "universal" and row["source_id"] is None and row["category_id"] is None:
                rank = 0
            elif scope == "source" and row["source_id"] == source_id and row["category_id"] is None:
                rank = 1
            elif scope == "category" and category_id is not None and row["category_id"] == category_id and row["source_id"] in (None, source_id):
                rank = 2
            if rank is None:
                continue
            previous = selected.get(row["attribute_key"])
            if previous and previous[0] == rank:
                raise MetadataCatalogError("Duplicate applicable metadata field definitions require review.")
            if previous is None or previous[0] < rank:
                selected[row["attribute_key"]] = (rank, row)
        fields, identities = [], []
        for _, schema in sorted(selected.values(), key=lambda pair: (pair[1]["openai_priority"], pair[1]["attribute_key"])):
            key = schema["attribute_key"]
            values = [dict(r) for r in connection.execute(select(t["metadata_values"]).where(
                t["metadata_values"].c.schema_id == schema["schema_id"], t["metadata_values"].c.is_deprecated.is_(False))
                .order_by(t["metadata_values"].c.value_text, t["metadata_values"].c.value_id)).mappings()]
            known = [{"value_id": r["value_id"], "value": r["value_text"], "label": r["value_text"].replace("_", " ")} for r in values if r["is_verified"]]
            provisional = [{"value_id": r["value_id"], "value": r["value_text"], "label": r["value_text"].replace("_", " ")} for r in values if not r["is_verified"]]
            known_set = {r["value"] for r in known}
            aliases = []
            for alias in profile_catalog.get("aliases", {}).get(key, []):
                if isinstance(alias, Mapping) and isinstance(alias.get("alias"), str) and alias.get("value") in known_set:
                    aliases.append({"alias": alias["alias"], "value": alias["value"]})
            aliases = sorted(aliases, key=lambda item: (item["alias"], item["value"]))
            policy = profile_catalog.get("policies", {}).get(key, BUSINESS_FIELDS.get(key, {}).get("policy", "expandable"))
            if policy not in {"fixed", "expandable"}:
                raise MetadataCatalogError("Unsupported metadata vocabulary policy.")
            export_key = schema["openai_key"] or key
            eligible = bool(schema["sync_to_openai"] and schema["attribute_type"] in {"string", "number", "boolean"} and export_key not in _RESERVED and 0 < len(export_key) <= 64)
            item = {"schema_id": schema["schema_id"], "key": key, "type": schema["attribute_type"], "label": schema["display_name"],
                    "description": schema["description"], "scope": schema["scope"], "category_id": schema["category_id"], "policy": policy,
                    "known_values": known[value_offset:value_offset + values_limit], "provisional_values": provisional[value_offset:value_offset + values_limit],
                    "counts": {"known": len(known), "provisional": len(provisional)}, "aliases": aliases,
                    "has_more": len(known) > value_offset + values_limit or len(provisional) > value_offset + values_limit,
                    "known_has_more": len(known) > value_offset + values_limit, "value_offset": value_offset,
                    "filter_operators": ["eq"] if eligible and schema["is_searchable"] else [],
                    "openai": {"eligible": eligible, "key": export_key, "reason": "eligible_scalar" if eligible else "disabled_or_unsupported", "priority": schema["openai_priority"]}}
            fields.append(item)
            identities.append({k: v for k, v in item.items() if k not in {"known_values", "provisional_values", "counts", "has_more", "known_has_more", "value_offset"}} | {"known_values": known})
        export_keys = [f["openai"]["key"] for f in fields if f["openai"]["eligible"]]
        if len(set(export_keys)) != len(export_keys):
            raise MetadataCatalogError("Duplicate OpenAI metadata keys require review.")
    version = _fingerprint({"implementation": CATALOG_VERSION, "normalization": NORMALIZATION_VERSION, "category_id": category_id, "fields": identities})
    return {"version": version, "catalog_version": version, "category_id": category_id, "channel_id": channel_id,
            "source_id": source_id, "categories": categories, "core_dimensions": _core_dimensions(categories), "fields": fields,
            "values_limit": values_limit, "value_offset": value_offset}


def build_category_schema(catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{"key": f["key"], "type": f["type"], "description": f["description"], "policy": f["policy"],
             "known_values": [v["value"] for v in f["known_values"]]} for f in catalog["fields"]]


def _normalize_evidence(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    excerpt = value.strip()
    pairs = {"\"": "\"", "'": "'", "“": "”", "‘": "’"}
    if len(excerpt) >= 2 and pairs.get(excerpt[0]) == excerpt[-1]:
        excerpt = excerpt[1:-1]
    return " ".join(excerpt.split())


def normalize_category_attributes(*, attributes: Sequence[Mapping[str, Any]], catalog: Mapping[str, Any], transcript_text: str) -> dict[str, Any]:
    """Pure normalization. Exact quoted evidence is necessary, not a truth guarantee."""
    if any(f.get("known_has_more") or f.get("value_offset", 0) for f in catalog["fields"]):
        raise MetadataCatalogError("Normalization requires a complete verified vocabulary page.")
    fields = {f["key"]: f for f in catalog["fields"]}
    source = " ".join(transcript_text.split())
    output, rejected = [], []
    for candidate in attributes:
        item = dict(candidate)
        key, raw, raw_evidence = item.get("key"), item.get("value"), item.get("evidence")
        evidence = _normalize_evidence(raw_evidence)
        field = fields.get(key) if isinstance(key, str) else None
        base = {"key": key, "schema_id": field["schema_id"] if field else None, "raw_value": raw,
                "value": None, "evidence": evidence, "raw_evidence": raw_evidence, "confidence": item.get("confidence"), "status": "rejected",
                "openai_key": field["openai"]["key"] if field else None,
                "openai_eligible": bool(field and field["openai"]["eligible"])}
        reason = None
        confidence = item.get("confidence")
        if field is None:
            reason = "unsupported_key"
        elif field["type"] != "string":
            reason = "unsupported_attribute_type"
        elif not isinstance(raw, str) or not raw.strip() or len(raw) > 512 or not re.fullmatch(r"[\w\s-]+", raw, re.UNICODE):
            reason = "invalid_scalar_value"
        elif not isinstance(evidence, str) or not evidence or len(raw_evidence) > 2000 or evidence not in source:
            reason = "evidence_not_in_transcript"
        elif isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            reason = "invalid_confidence"
        elif confidence < 0.8:
            reason = "low_confidence"
        else:
            normalized = _normalize(raw)
            known = {_normalize(v["value"]): v["value"] for v in field["known_values"]}
            alias_targets = {a["value"] for a in field["aliases"] if _normalize(a["alias"]) == normalized}
            if len(alias_targets) > 1 or (normalized in known and alias_targets and alias_targets != {known[normalized]}):
                reason = "ambiguous_alias"
            else:
                canonical = known.get(normalized) or (next(iter(alias_targets)) if alias_targets else None)
                if canonical:
                    base.update(value=canonical, status="trusted", reason="known_value" if normalized in known else "trusted_alias")
                elif field["policy"] == "fixed":
                    reason = "not_in_fixed_options"
                else:
                    base.update(value=normalized, status="provisional", reason="new_value_proposal")
        if reason:
            base["reason"] = reason
            if field is None:
                rejected.append(base)
                continue
        output.append(base)
    result = {"catalog_version": catalog["version"], "fields": output, "rejected": rejected}
    result["openai_attributes"] = project_openai_metadata(result)
    return result


def project_openai_metadata(results: Mapping[str, Any]) -> dict[str, str]:
    """Export at most four business scalars; omit provisional/conflicting fields."""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for item in results.get("fields", []):
        if item.get("key") in BUSINESS_FIELDS:
            grouped.setdefault(item["key"], []).append(item)
    output = {}
    for key in BUSINESS_FIELDS:
        rows = grouped.get(key, [])
        if not rows or any(r.get("status") != "trusted" or not r.get("openai_eligible") for r in rows):
            continue
        values = {r.get("value") for r in rows if isinstance(r.get("value"), str)}
        export_keys = {r.get("openai_key") for r in rows}
        if len(values) != 1 or len(export_keys) != 1:
            continue
        value, export_key = next(iter(values)), next(iter(export_keys))
        if not isinstance(export_key, str) or export_key in _RESERVED or not 0 < len(export_key) <= 64 or export_key in output:
            continue
        if value and len(value) <= 512 and re.fullmatch(r"[\w-]+", value, re.UNICODE):
            output[export_key] = value
    return output


def persist_metadata_candidates(*, engine: Engine, run_id: int, results: Mapping[str, Any], profile_id: int | None = None) -> None:
    """Record each new value once per run; never promote it or mutate the run."""
    with engine.begin() as connection:
        t = _tables(connection, "metadata_values", "metadata_learning_events", "extraction_runs", "metadata_schemas")
        # Serialize concurrent recording for the same run in PostgreSQL.
        if connection.execute(select(t["extraction_runs"].c.run_id).where(t["extraction_runs"].c.run_id == run_id).with_for_update()).scalar_one_or_none() is None:
            raise MetadataCatalogError("Extraction run was not found.")
        values, events = t["metadata_values"], t["metadata_learning_events"]
        schema_ids = sorted({item["schema_id"] for item in results.get("fields", []) if item.get("status") == "provisional" and item.get("key") in BUSINESS_FIELDS})
        # Different extraction runs can propose the same value concurrently.
        connection.execute(select(t["metadata_schemas"].c.schema_id).where(t["metadata_schemas"].c.schema_id.in_(schema_ids))
                           .order_by(t["metadata_schemas"].c.schema_id).with_for_update()).all()
        for item in results.get("fields", []):
            if item.get("status") != "provisional" or item.get("key") not in BUSINESS_FIELDS:
                continue
            existing = connection.execute(select(events.c.event_id).where(events.c.run_id == run_id,
                events.c.dimension == item["key"], events.c.action == "propose_value", events.c.canonical_value == item["value"])).first()
            if existing:
                continue
            value = connection.execute(select(values).where(values.c.schema_id == item["schema_id"], values.c.value_normalized == item["value"])).mappings().first()
            if value is None:
                value_id = connection.execute(insert(values).values(schema_id=item["schema_id"], value_text=item["value"], value_normalized=item["value"], usage_count=1, is_verified=False, is_deprecated=False)).inserted_primary_key[0]
            else:
                value_id = value["value_id"]
                connection.execute(update(values).where(values.c.value_id == value_id).values(usage_count=values.c.usage_count + 1, last_seen=func.current_timestamp()))
            connection.execute(insert(events).values(profile_id=profile_id, run_id=run_id, dimension=item["key"], action="propose_value",
                raw_value=item["raw_value"], canonical_value=item["value"], confidence=item["confidence"], reason="Evidence-backed candidate requires explicit vocabulary review.",
                status="proposed", details={"value_id": value_id, "schema_id": item["schema_id"], "catalog_version": results["catalog_version"], "evidence": item["evidence"]}))


def accept_metadata_value(*, engine: Engine, value_id: int) -> dict[str, Any]:
    """Explicitly accept an expandable proposal. Prior extraction results stay fixed."""
    with engine.begin() as connection:
        t = _tables(connection, "metadata_values", "metadata_schemas", "metadata_learning_events")
        values, schemas = t["metadata_values"], t["metadata_schemas"]
        row = connection.execute(select(values, schemas.c.attribute_key).join(schemas).where(values.c.value_id == value_id).with_for_update()).mappings().first()
        if row is None or row["is_deprecated"]:
            raise MetadataCatalogError("Active metadata value was not found.")
        if BUSINESS_FIELDS.get(row["attribute_key"], {}).get("policy") != "expandable":
            raise MetadataCatalogError("Only expandable business vocabulary can be accepted here.")
        connection.execute(update(values).where(values.c.value_id == value_id).values(is_verified=True, last_seen=func.current_timestamp()))
        events = t["metadata_learning_events"]
        for event in connection.execute(select(events).where(events.c.action == "propose_value", events.c.status == "proposed", events.c.canonical_value == row["value_text"])).mappings():
            if event["details"].get("value_id") == value_id:
                connection.execute(update(events).where(events.c.event_id == event["event_id"]).values(status="applied", applied_at=func.current_timestamp()))
    return {"value_id": value_id, "value": row["value_text"], "status": "verified", "affects_existing_extractions": False}


def ensure_business_metadata_catalog(*, engine: Engine, channel_ids: Sequence[str]) -> dict[str, Any]:
    """Explicit, idempotent seed using existing tables; no DDL or model requests.

    Existing channel profiles are extended with a new active version; global and
    legacy category/channel profiles remain untouched. No media row is recategorized.
    """
    ids = list(dict.fromkeys(channel_ids))
    if not ids or any(not isinstance(v, str) or not v for v in ids):
        raise MetadataCatalogError("Provide at least one existing channel_id.")
    prompt = json.loads(files("learn_from").joinpath("prompts.json").read_text())["metadata_v2"]["business_profile_instructions"]
    with engine.begin() as connection:
        t = _tables(connection, "metadata_schemas", "metadata_values", "doc_categories", "channels", "extraction_profiles", "extraction_profile_versions")
        # This table lock protects nullable-key uniqueness and profile version allocation.
        if connection.dialect.name == "postgresql":
            from sqlalchemy import text
            connection.execute(text("LOCK TABLE metadata_schemas, metadata_values, extraction_profiles, extraction_profile_versions IN SHARE ROW EXCLUSIVE MODE"))
        categories, schemas, values = t["doc_categories"], t["metadata_schemas"], t["metadata_values"]
        category_id = connection.execute(select(categories.c.category_id).where(func.lower(categories.c.category_name) == "business")).scalar_one_or_none()
        if category_id is None:
            raise MetadataCatalogError("The business category must exist before seeding.")
        channels = {r["channel_id"]: r["channel_name"] for r in connection.execute(select(t["channels"].c.channel_id, t["channels"].c.channel_name).where(t["channels"].c.channel_id.in_(ids), t["channels"].c.source_id == "youtube")).mappings()}
        if set(channels) != set(ids):
            raise MetadataCatalogError("Every channel must already exist as a YouTube channel.")
        schema_ids = {}
        for priority, (key, spec) in enumerate(BUSINESS_FIELDS.items(), 1):
            rows = connection.execute(select(schemas).where(schemas.c.attribute_key == key, schemas.c.scope == "category", schemas.c.category_id == category_id)).mappings().all()
            if len(rows) > 1:
                raise MetadataCatalogError("Duplicate business schemas require review.")
            if rows:
                row = rows[0]
                if row["attribute_type"] != "string" or not row["is_active"] or row["source_id"] not in (None, "youtube"):
                    raise MetadataCatalogError("An incompatible business schema already exists.")
                schema_id = row["schema_id"]
            else:
                schema_id = connection.execute(insert(schemas).values(attribute_key=key, attribute_type="string", scope="category", source_id="youtube", category_id=category_id,
                    display_name=spec["label"], description=spec["description"], is_searchable=True, is_required=False, sync_to_openai=True, openai_key=key,
                    openai_priority=priority, extraction_version="business-v1", is_active=True)).inserted_primary_key[0]
            schema_ids[key] = schema_id
            for value in spec["values"]:
                existing = connection.execute(select(values.c.value_id).where(values.c.schema_id == schema_id, values.c.value_normalized == value)).scalar_one_or_none()
                if existing is None:
                    connection.execute(insert(values).values(schema_id=schema_id, value_text=value, value_normalized=value, usage_count=0, is_verified=True, is_deprecated=False))
                # Existing proposals/deprecations are not silently promoted by reseeding.
        profiles, versions = t["extraction_profiles"], t["extraction_profile_versions"]
        profile_results = []
        for channel_id in ids:
            rows = connection.execute(select(profiles).where(profiles.c.scope == "channel", profiles.c.source_id == "youtube", profiles.c.channel_id == channel_id, profiles.c.category_id.is_(None))).mappings().all()
            if len(rows) > 1:
                raise MetadataCatalogError("Duplicate channel profiles require review.")
            if rows:
                profile_id = rows[0]["profile_id"]
            else:
                profile_id = connection.execute(insert(profiles).values(profile_name=f"{channels[channel_id]} Business Metadata", scope="channel", source_id="youtube", channel_id=channel_id,
                    learning_mode="observe", is_active=True)).inserted_primary_key[0]
            active = connection.execute(select(versions).where(versions.c.profile_id == profile_id, versions.c.status == "active")).mappings().all()
            if len(active) > 1:
                raise MetadataCatalogError("Multiple active profile versions require review.")
            config = dict(active[0]["config"]) if active else {}
            previous_known = config.get("known_values", {})
            known_values = {**previous_known, **{
                key: list(dict.fromkeys([*previous_known.get(key, []), *spec["values"]]))
                for key, spec in BUSINESS_FIELDS.items()
            }}
            category_schema = [item for item in config.get("category_schema", []) if item.get("key") not in BUSINESS_FIELDS]
            category_schema.extend({"key": key, "type": "string", "description": spec["description"], "policy": spec["policy"]} for key, spec in BUSINESS_FIELDS.items())
            config.update(metadata_catalog={**config.get("metadata_catalog", {}), "version": CATALOG_VERSION, "category_id": category_id,
                "policies": {**config.get("metadata_catalog", {}).get("policies", {}), **{k: v["policy"] for k, v in BUSINESS_FIELDS.items()}}},
                category_schema=category_schema,
                known_values=known_values,
                model_routing={"stable_model": "gpt-6-luna", "enhanced_model": "gpt-6-luna", "resolver_model": "gpt-6-luna", "learner_model": "gpt-6-luna", "bootstrap_documents": 0, "audit_interval": 10})
            if prompt not in config.get("prompt_instructions", ""):
                config["prompt_instructions"] = "\n\n".join(v for v in [config.get("prompt_instructions", ""), prompt] if v)
            if active and config == active[0]["config"]:
                profile_results.append({"channel_id": channel_id, "profile_id": profile_id, "profile_version_id": active[0]["profile_version_id"], "reused": True})
                continue
            version = (connection.execute(select(func.max(versions.c.version)).where(versions.c.profile_id == profile_id)).scalar_one() or 0) + 1
            connection.execute(update(versions).where(versions.c.profile_id == profile_id, versions.c.status == "active").values(status="retired", retired_at=func.current_timestamp()))
            version_id = connection.execute(insert(versions).values(profile_id=profile_id, version=version, status="active", config=config, published_at=func.current_timestamp())).inserted_primary_key[0]
            connection.execute(update(profiles).where(profiles.c.profile_id == profile_id).values(learning_mode="observe", is_active=True, updated_at=func.current_timestamp()))
            profile_results.append({"channel_id": channel_id, "profile_id": profile_id, "profile_version_id": version_id, "reused": False})
    return {"category_id": category_id, "schema_ids": schema_ids, "profiles": profile_results}

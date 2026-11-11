from __future__ import annotations

import copy

import pytest
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON, MetaData, String, Table, UniqueConstraint, create_engine, insert, select, update

from learn_from.metadata_catalog import (
    MetadataCatalogError, accept_metadata_value, build_category_schema,
    ensure_business_metadata_catalog, get_metadata_schema,
    normalize_category_attributes, persist_metadata_candidates, project_openai_metadata,
)


def _engine():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    meta = MetaData()
    Table("doc_categories", meta, Column("category_id", Integer, primary_key=True), Column("category_name", String))
    Table("channels", meta, Column("channel_id", String, primary_key=True), Column("channel_name", String), Column("source_id", String))
    Table("extraction_profiles", meta, Column("profile_id", Integer, primary_key=True), Column("profile_name", String), Column("scope", String), Column("source_id", String), Column("channel_id", String), Column("category_id", Integer), Column("learning_mode", String), Column("is_active", Boolean), Column("updated_at", DateTime))
    Table("extraction_profile_versions", meta, Column("profile_version_id", Integer, primary_key=True), Column("profile_id", Integer, ForeignKey("extraction_profiles.profile_id")), Column("version", Integer), Column("status", String), Column("config", JSON), Column("published_at", DateTime), Column("retired_at", DateTime))
    Table("metadata_schemas", meta, Column("schema_id", Integer, primary_key=True), Column("attribute_key", String), Column("attribute_type", String), Column("scope", String), Column("source_id", String), Column("category_id", Integer), Column("display_name", String), Column("description", String), Column("is_searchable", Boolean), Column("is_required", Boolean), Column("sync_to_openai", Boolean), Column("openai_key", String), Column("openai_priority", Integer), Column("extraction_version", String), Column("is_active", Boolean))
    Table("metadata_values", meta, Column("value_id", Integer, primary_key=True), Column("schema_id", Integer, ForeignKey("metadata_schemas.schema_id")), Column("value_text", String), Column("value_normalized", String), Column("usage_count", Integer), Column("is_verified", Boolean), Column("is_deprecated", Boolean), Column("last_seen", DateTime), UniqueConstraint("schema_id", "value_normalized"))
    Table("extraction_runs", meta, Column("run_id", Integer, primary_key=True))
    Table("metadata_learning_events", meta, Column("event_id", Integer, primary_key=True), Column("profile_id", Integer), Column("run_id", Integer), Column("dimension", String), Column("action", String), Column("raw_value", String), Column("canonical_value", String), Column("confidence", Float), Column("reason", String), Column("status", String), Column("details", JSON), Column("applied_at", DateTime))
    meta.create_all(engine)
    with engine.begin() as c:
        c.execute(insert(meta.tables["doc_categories"]), [{"category_id": 1, "category_name": "Faith"}, {"category_id": 10, "category_name": "business"}])
        c.execute(insert(meta.tables["channels"]), [{"channel_id": "one", "channel_name": "One", "source_id": "youtube"}, {"channel_id": "two", "channel_name": "Two", "source_id": "youtube"}])
        c.execute(insert(meta.tables["extraction_runs"]), [{"run_id": 1}, {"run_id": 2}])
    return engine, meta.tables


@pytest.fixture
def seeded():
    engine, tables = _engine()
    ensure_business_metadata_catalog(engine=engine, channel_ids=["one", "two"])
    return engine, tables


def _catalog(engine, **kwargs):
    return get_metadata_schema(engine=engine, channel_id="one", **kwargs)


def _candidate(value="local_service", *, key="business_model", evidence="A local service business.", confidence=0.95):
    return {"key": key, "value": value, "evidence": evidence, "confidence": confidence}


def test_empty_registry_is_not_fabricated_and_core_types_match_models():
    engine, _ = _engine()
    catalog = get_metadata_schema(engine=engine)
    assert catalog["fields"] == []
    assert catalog["categories"][1] == {"category_id": 10, "category_name": "business"}
    core = {x["key"]: x for x in catalog["core_dimensions"]}
    assert core["topics"]["policy"] == "open"
    assert "software" in core["entities"]["type_options"]
    assert "scripture" in core["references"]["type_options"]


def test_seed_is_idempotent_and_leaves_global_profile_unchanged():
    engine, tables = _engine()
    with engine.begin() as c:
        c.execute(insert(tables["extraction_profiles"]).values(profile_id=20, profile_name="Global", scope="global", learning_mode="autopilot", is_active=True))
        c.execute(insert(tables["extraction_profile_versions"]).values(profile_id=20, version=1, status="active", config={"legacy": True}))
    first = ensure_business_metadata_catalog(engine=engine, channel_ids=["one"])
    second = ensure_business_metadata_catalog(engine=engine, channel_ids=["one"])
    assert first["schema_ids"] == second["schema_ids"]
    assert first["profiles"][0]["profile_version_id"] == second["profiles"][0]["profile_version_id"]
    assert second["profiles"][0]["reused"] is True
    with engine.connect() as c:
        assert c.execute(select(tables["extraction_profile_versions"].c.config).where(tables["extraction_profile_versions"].c.profile_id == 20)).scalar_one() == {"legacy": True}
    catalog = _catalog(engine)
    assert catalog["category_id"] == 10
    assert len(catalog["fields"]) == 4
    assert {f["key"]: f["policy"] for f in catalog["fields"]}["business_stage"] == "fixed"
    assert len(build_category_schema(catalog)) == 4
    assert get_metadata_schema(engine=engine, channel_id="two")["fields"] == []


def test_seed_fails_without_partial_changes_for_unknown_channel():
    engine, tables = _engine()
    with pytest.raises(MetadataCatalogError, match="already exist"):
        ensure_business_metadata_catalog(engine=engine, channel_ids=["missing"])
    with engine.connect() as c:
        assert c.execute(select(tables["metadata_schemas"])).all() == []


def test_seed_extends_existing_channel_config_and_preserves_prior_version():
    engine, tables = _engine()
    original = {"known_values": {"industry": ["custom_industry"], "participants": ["A Host"]},
                "category_schema": [{"key": "legacy_field", "type": "string"}], "prompt_instructions": "A custom instruction."}
    with engine.begin() as c:
        c.execute(insert(tables["extraction_profiles"]).values(profile_id=30, profile_name="Existing", scope="channel", source_id="youtube", channel_id="one", learning_mode="assisted", is_active=True))
        c.execute(insert(tables["extraction_profile_versions"]).values(profile_id=30, version=1, status="active", config=original))
    ensure_business_metadata_catalog(engine=engine, channel_ids=["one"])
    with engine.connect() as c:
        versions = c.execute(select(tables["extraction_profile_versions"]).where(tables["extraction_profile_versions"].c.profile_id == 30).order_by(tables["extraction_profile_versions"].c.version)).mappings().all()
    assert versions[0]["status"] == "retired"
    assert versions[0]["config"] == original
    active = versions[1]["config"]
    assert "custom_industry" in active["known_values"]["industry"]
    assert active["known_values"]["participants"] == ["A Host"]
    assert active["category_schema"][0]["key"] == "legacy_field"
    assert active["prompt_instructions"].startswith("A custom instruction.")


def test_scope_precedence_and_duplicates_fail_closed(seeded):
    engine, tables = seeded
    schemas = tables["metadata_schemas"]
    with engine.begin() as c:
        original = dict(c.execute(select(schemas).where(schemas.c.attribute_key == "industry")).mappings().one())
        original.pop("schema_id")
        c.execute(insert(schemas).values(**(original | {"scope": "universal", "source_id": None, "category_id": None, "display_name": "Generic industry"})))
    assert next(f for f in _catalog(engine)["fields"] if f["key"] == "industry")["label"] == "Industry"
    assert next(f for f in get_metadata_schema(engine=engine)["fields"] if f["key"] == "industry")["label"] == "Generic industry"
    with engine.begin() as c:
        c.execute(insert(schemas).values(**original))
    with pytest.raises(MetadataCatalogError, match="Duplicate applicable"):
        _catalog(engine)


def test_pagination_keeps_version_stable_and_incomplete_vocab_cannot_normalize(seeded):
    engine, _ = seeded
    full = _catalog(engine)
    paged = _catalog(engine, values_limit=2, value_offset=2)
    assert paged["version"] == full["version"]
    assert all(f["known_has_more"] for f in paged["fields"])
    with pytest.raises(MetadataCatalogError, match="complete verified"):
        normalize_category_attributes(attributes=[], catalog=paged, transcript_text="")


@pytest.mark.parametrize("params", [{"values_limit": 0}, {"values_limit": True}, {"values_limit": 1001}, {"value_offset": -1}, {"category_id": 999}])
def test_catalog_parameter_bounds(seeded, params):
    with pytest.raises(MetadataCatalogError):
        _catalog(seeded[0], **params)


def test_verified_values_normalize_with_exact_whitespace_evidence(seeded):
    catalog = _catalog(seeded[0])
    result = normalize_category_attributes(attributes=[_candidate("Local Service")], catalog=catalog, transcript_text="Introduction. A local\n service business. More.")
    assert result["fields"][0]["status"] == "trusted"
    assert result["openai_attributes"] == {"business_model": "local_service"}


@pytest.mark.parametrize("opening,closing", [("\"", "\""), ("'", "'"), ("“", "”"), ("‘", "’")])
def test_matching_outer_quote_pair_is_removed_and_raw_evidence_preserved(seeded, opening, closing):
    raw = f"  {opening}A local\n service business.{closing}  "
    result = normalize_category_attributes(attributes=[_candidate(evidence=raw)], catalog=_catalog(seeded[0]), transcript_text="Intro. A local service business. End.")
    field = result["fields"][0]
    assert field["status"] == "trusted"
    assert field["raw_evidence"] == raw
    assert field["evidence"] == "A local service business."


@pytest.mark.parametrize("evidence", [
    "“An invented service business.”", "“A local ... business.”", "“A local — business.”",
    "“A local service business.'", '“\"A local service business.\"”', "“”",
])
def test_quote_normalization_does_not_rewrite_inner_text_or_strip_multiple_pairs(seeded, evidence):
    result = normalize_category_attributes(attributes=[_candidate(evidence=evidence)], catalog=_catalog(seeded[0]), transcript_text="A local service business.")
    assert result["fields"][0]["reason"] == "evidence_not_in_transcript"
    assert result["fields"][0]["raw_evidence"] == evidence
    assert result["openai_attributes"] == {}


@pytest.mark.parametrize("candidate,reason", [
    (_candidate(evidence="Not actually said."), "evidence_not_in_transcript"),
    (_candidate(confidence=float("nan")), "invalid_confidence"),
    (_candidate(confidence=0.4), "low_confidence"),
    (_candidate("local_service,subscription"), "invalid_scalar_value"),
    (_candidate("made_up_stage", key="business_stage"), "not_in_fixed_options"),
])
def test_invalid_or_unsupported_evidence_never_reaches_export(seeded, candidate, reason):
    result = normalize_category_attributes(attributes=[candidate], catalog=_catalog(seeded[0]), transcript_text="A local service business.")
    assert result["fields"][0]["reason"] == reason
    assert result["openai_attributes"] == {}


def test_unsupported_key_is_explicitly_rejected(seeded):
    result = normalize_category_attributes(attributes=[_candidate(key="secret_custom_field")], catalog=_catalog(seeded[0]), transcript_text="A local service business.")
    assert result["fields"] == []
    assert result["rejected"][0]["reason"] == "unsupported_key"


def test_trusted_alias_requires_verified_target_and_conflicting_alias_is_rejected(seeded):
    engine, tables = seeded
    with engine.begin() as c:
        version = c.execute(select(tables["extraction_profile_versions"]).limit(1)).mappings().one()
        config = copy.deepcopy(version["config"])
        config["metadata_catalog"]["aliases"] = {"business_model": [{"alias": "local services", "value": "local_service"}, {"alias": "unsupported", "value": "nonexistent"}]}
        c.execute(update(tables["extraction_profile_versions"]).where(tables["extraction_profile_versions"].c.profile_version_id == version["profile_version_id"]).values(config=config))
    catalog = _catalog(engine)
    field = next(f for f in catalog["fields"] if f["key"] == "business_model")
    assert field["aliases"] == [{"alias": "local services", "value": "local_service"}]
    result = normalize_category_attributes(attributes=[_candidate("local services")], catalog=catalog, transcript_text="A local service business.")
    assert result["fields"][0]["reason"] == "trusted_alias"
    field["aliases"].append({"alias": "local services", "value": "subscription"})
    result = normalize_category_attributes(attributes=[_candidate("local services")], catalog=catalog, transcript_text="A local service business.")
    assert result["fields"][0]["reason"] == "ambiguous_alias"
    assert result["openai_attributes"] == {}


def test_multiple_values_or_provisional_value_omit_the_scalar_field(seeded):
    catalog = _catalog(seeded[0])
    for values in [("local_service", "subscription"), ("local_service", "new_model")]:
        result = normalize_category_attributes(attributes=[_candidate(v) for v in values], catalog=catalog, transcript_text="A local service business.")
        assert result["openai_attributes"] == {}
    result = normalize_category_attributes(attributes=[_candidate(), _candidate()], catalog=catalog, transcript_text="A local service business.")
    assert project_openai_metadata(result) == {"business_model": "local_service"}


def test_proposals_are_idempotent_and_acceptance_does_not_rewrite_old_results(seeded):
    engine, tables = seeded
    catalog = _catalog(engine)
    result = normalize_category_attributes(attributes=[_candidate("mobile_pet_grooming", key="industry")], catalog=catalog, transcript_text="A local service business.")
    snapshot = copy.deepcopy(result)
    persist_metadata_candidates(engine=engine, run_id=1, results=result)
    persist_metadata_candidates(engine=engine, run_id=1, results=result)
    after_proposal = _catalog(engine)
    assert after_proposal["version"] == catalog["version"]
    with engine.connect() as c:
        events = c.execute(select(tables["metadata_learning_events"])).mappings().all()
        assert len(events) == 1
        value_id = events[0]["details"]["value_id"]
        assert c.execute(select(tables["metadata_values"].c.usage_count).where(tables["metadata_values"].c.value_id == value_id)).scalar_one() == 1
    accepted = accept_metadata_value(engine=engine, value_id=value_id)
    assert accepted["affects_existing_extractions"] is False
    assert result == snapshot
    assert result["fields"][0]["status"] == "provisional"
    assert _catalog(engine)["version"] != catalog["version"]
    with engine.connect() as c:
        assert c.execute(select(tables["metadata_learning_events"].c.status)).scalar_one() == "applied"


def test_accept_cannot_expand_fixed_vocabulary(seeded):
    engine, tables = seeded
    with engine.connect() as c:
        value_id = c.execute(select(tables["metadata_values"].c.value_id).where(tables["metadata_values"].c.value_text == "idea")).scalar_one()
    with pytest.raises(MetadataCatalogError, match="Only expandable"):
        accept_metadata_value(engine=engine, value_id=value_id)


def test_reserved_and_duplicate_export_keys_are_not_accepted(seeded):
    engine, tables = seeded
    schemas = tables["metadata_schemas"]
    with engine.begin() as c:
        c.execute(update(schemas).where(schemas.c.attribute_key == "industry").values(openai_key="channel_id"))
    field = next(f for f in _catalog(engine)["fields"] if f["key"] == "industry")
    assert field["openai"]["eligible"] is False
    with engine.begin() as c:
        c.execute(update(schemas).where(schemas.c.attribute_key == "industry").values(openai_key="business_model"))
    with pytest.raises(MetadataCatalogError, match="Duplicate OpenAI"):
        _catalog(engine)

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    insert,
    select,
)

from learn_from.metadata import (
    build_metadata_extraction_prompt,
    extract_and_persist_metadata_for_content,
    normalize_extracted_metadata,
    persist_extracted_metadata,
)


def _build_test_engine():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    metadata = MetaData()

    Table(
        "media",
        metadata,
        Column("media_id", String(255), primary_key=True),
        Column("category_id", Integer),
        Column("extraction_version", String(20), nullable=False, default="1.0"),
        Column("extraction_quality_score", Float),
        Column("last_updated", DateTime),
    )
    Table(
        "doc_content",
        metadata,
        Column("content_id", Integer, primary_key=True, autoincrement=True),
        Column("media_id", String(255), nullable=False),
        Column("full_text", Text, nullable=False),
        Column("metadata_updated_at", DateTime),
        Column("updated_at", DateTime),
    )
    Table(
        "doc_categories",
        metadata,
        Column("category_id", Integer, primary_key=True, autoincrement=True),
        Column("category_name", String(100), nullable=False, unique=True),
    )
    Table(
        "doc_topics",
        metadata,
        Column("content_id", Integer, nullable=False, primary_key=True),
        Column("topic", String(100), nullable=False, primary_key=True),
    )
    Table(
        "doc_keywords",
        metadata,
        Column("content_id", Integer, nullable=False, primary_key=True),
        Column("keyword", String(100), nullable=False, primary_key=True),
    )
    Table(
        "doc_participants",
        metadata,
        Column("content_id", Integer, nullable=False, primary_key=True),
        Column("name", String(255), nullable=False, primary_key=True),
        Column("role", String(50)),
    )
    Table(
        "entities_catalog",
        metadata,
        Column("entity_id", Integer, primary_key=True, autoincrement=True),
        Column("type", String(30), nullable=False),
        Column("label", String(255), nullable=False),
        Column("label_normalized", String(255), nullable=False),
        Column("aliases", JSON, nullable=False, default=list),
        Column("usage_count", Integer, nullable=False, default=1),
        Column("first_seen_at", DateTime, nullable=False, default=datetime.utcnow),
        Column("last_seen_at", DateTime, nullable=False, default=datetime.utcnow),
    )
    Table(
        "references_catalog",
        metadata,
        Column("reference_id", Integer, primary_key=True, autoincrement=True),
        Column("type", String(30), nullable=False),
        Column("label", String(255), nullable=False),
        Column("label_normalized", String(255), nullable=False),
        Column("identifiers", JSON, nullable=False, default=dict),
        Column("usage_count", Integer, nullable=False, default=1),
        Column("first_seen_at", DateTime, nullable=False, default=datetime.utcnow),
        Column("last_seen_at", DateTime, nullable=False, default=datetime.utcnow),
    )
    Table(
        "doc_entities",
        metadata,
        Column("content_id", Integer, nullable=False, primary_key=True),
        Column("entity_id", Integer, nullable=False, primary_key=True),
        Column("context", String(150)),
        Column("usage", String(20), nullable=False, default="neutral"),
        Column("mention_count", Integer, nullable=False, default=1),
        Column("ref", JSON),
    )
    Table(
        "doc_references",
        metadata,
        Column("content_id", Integer, nullable=False, primary_key=True),
        Column("reference_id", Integer, nullable=False, primary_key=True),
        Column("context", String(150)),
        Column("usage", String(20), nullable=False, default="neutral"),
        Column("mention_count", Integer, nullable=False, default=1),
        Column("ref", JSON),
        Column("attributes", JSON, nullable=False, default=dict),
    )

    metadata.create_all(engine)
    return engine


def _seed_content(engine) -> None:
    metadata = MetaData()
    media = Table("media", metadata, autoload_with=engine)
    doc_content = Table("doc_content", metadata, autoload_with=engine)
    doc_categories = Table("doc_categories", metadata, autoload_with=engine)

    with engine.begin() as connection:
        connection.execute(insert(doc_categories), [{"category_name": "short term rental"}])
        connection.execute(insert(doc_categories), [{"category_name": "technology"}])
        connection.execute(insert(media).values(media_id="vid-1", extraction_version="1.0"))
        connection.execute(
            insert(doc_content).values(
                content_id=1,
                media_id="vid-1",
                full_text="We discuss Airbnb dynamic pricing with PriceLabs.",
            )
        )


def test_normalize_extracted_metadata_handles_dedupes_and_fallbacks() -> None:
    payload = {
        "category": "short_term_rental",
        "topics": ["Dynamic Pricing", "Dynamic Pricing", "  "],
        "keywords": ["Airbnb", "AIRBNB", "PriceLabs"],
        "participants": [{"name": "Host A", "role": "host"}, {"name": "host a"}],
        "entities": [
            {"type": "software", "label": "PriceLabs", "mention_count": 2, "usage": "recommended"},
            {"type": "unknown_type", "label": "Dynamic Pricing"},
        ],
        "references": [
            {
                "type": "media_fragment",
                "label": "AirDNA",
                "identifiers": {"url": "https://airdna.co"},
                "fragment_kind": "chart",
            },
            {"type": "not_real", "label": "AirDNA"},
        ],
    }

    normalized = normalize_extracted_metadata(payload)
    assert normalized["category"] == "short_term_rental"
    assert normalized["topics"] == ["Dynamic Pricing"]
    assert normalized["keywords"] == ["airbnb", "pricelabs"]
    assert normalized["participants"] == [{"name": "Host A", "role": "host"}]
    assert normalized["entities"][1]["type"] == "other"
    assert normalized["references"][0]["attributes"]["fragment_kind"] == "chart"
    assert normalized["references"][1]["type"] == "other"


def test_build_metadata_extraction_prompt_includes_reference_labeling_rules() -> None:
    prompt = build_metadata_extraction_prompt(
        transcript_text="Acts chapter 12 verse 2 and NIV translation notes.",
        category_hints=["christian_faith"],
        max_transcript_chars=500,
    )
    assert "magazine_article" in prompt
    assert "canonical and retrieval-friendly" in prompt
    assert "Bible translations" in prompt


def test_normalize_extracted_metadata_reference_canonicalization() -> None:
    payload = {
        "references": [
            {"type": "paper", "label": "Acts 12:2"},
            {
                "type": "scripture",
                "label": "Revelation (Euphrates River / angels bound tradition)",
            },
            {"type": "scripture", "label": "1 Enoch chapters 6-11"},
            {"type": "magazine_article", "label": "Christianity Today (1980s issue)"},
            {
                "type": "media_fragment",
                "label": "Data Over Dogma transcript panel",
                "fragment_kind": "transcript_excerpt",
            },
        ]
    }

    normalized = normalize_extracted_metadata(payload)
    assert normalized["references"][0]["type"] == "scripture"
    assert normalized["references"][0]["label"] == "Acts 12:2"
    assert normalized["references"][1]["label"] == "Book of Revelation"
    assert normalized["references"][2]["label"] == "1 Enoch 6-11"
    assert normalized["references"][3]["type"] == "magazine_article"
    assert "fragment_kind" not in normalized["references"][4]["attributes"]


def test_persist_extracted_metadata_writes_doc_tables_and_updates_media() -> None:
    engine = _build_test_engine()
    _seed_content(engine)
    metadata = MetaData()
    media = Table("media", metadata, autoload_with=engine)
    doc_topics = Table("doc_topics", metadata, autoload_with=engine)
    doc_keywords = Table("doc_keywords", metadata, autoload_with=engine)
    doc_participants = Table("doc_participants", metadata, autoload_with=engine)
    doc_entities = Table("doc_entities", metadata, autoload_with=engine)
    doc_references = Table("doc_references", metadata, autoload_with=engine)
    entities_catalog = Table("entities_catalog", metadata, autoload_with=engine)

    extraction = {
        "category": "short_term_rental",
        "topics": ["dynamic pricing"],
        "keywords": ["airbnb", "pricing"],
        "participants": [{"name": "Mike Chen", "role": "host"}],
        "entities": [{"type": "software", "label": "PriceLabs", "usage": "recommended", "mention_count": 4}],
        "references": [{"type": "website", "label": "AirDNA", "usage": "evidence"}],
    }

    persist_extracted_metadata(engine=engine, content_id=1, extraction=extraction, extraction_version="1.0")
    persist_extracted_metadata(engine=engine, content_id=1, extraction=extraction, extraction_version="1.0")

    with engine.connect() as connection:
        assert connection.execute(select(doc_topics.c.topic)).scalars().all() == ["dynamic pricing"]
        assert set(connection.execute(select(doc_keywords.c.keyword)).scalars().all()) == {"airbnb", "pricing"}
        assert connection.execute(select(doc_participants.c.name)).scalars().all() == ["Mike Chen"]
        assert len(connection.execute(select(doc_entities)).all()) == 1
        assert len(connection.execute(select(doc_references)).all()) == 1
        assert connection.execute(select(media.c.category_id)).scalar_one() == 1
        assert connection.execute(select(entities_catalog.c.usage_count)).scalar_one() == 1


def test_extract_and_persist_metadata_for_content_uses_extractor(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _build_test_engine()
    _seed_content(engine)
    metadata = MetaData()
    doc_keywords = Table("doc_keywords", metadata, autoload_with=engine)

    def _fake_extract_metadata(**kwargs):
        return {
            "category": "technology",
            "topics": ["rag systems"],
            "keywords": ["openai", "retrieval"],
            "participants": [],
            "entities": [],
            "references": [],
        }

    monkeypatch.setattr("learn_from.metadata.extract_metadata", _fake_extract_metadata)
    summary = extract_and_persist_metadata_for_content(
        engine=engine,
        content_id=1,
        openai_api_key="test-key",
        model="gpt-5.4-nano",
    )

    assert summary["content_id"] == 1
    assert summary["category"] == "technology"
    with engine.connect() as connection:
        assert set(connection.execute(select(doc_keywords.c.keyword)).scalars().all()) == {"openai", "retrieval"}

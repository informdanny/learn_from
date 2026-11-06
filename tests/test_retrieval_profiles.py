from __future__ import annotations

import pytest
from sqlalchemy import Boolean, Column, DateTime, JSON, MetaData, String, Table, create_engine, insert

from learn_from.retrieval_profiles import RetrievalProfileError, get_retrieval_profiles


def _build_test_engine():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    metadata = MetaData()

    retrieval_profiles = Table(
        "retrieval_profiles",
        metadata,
        Column("profile_name", String(20), primary_key=True),
        Column("is_active", Boolean, nullable=False, default=True),
        Column("is_default", Boolean, nullable=False, default=False),
        Column("profile_config", JSON, nullable=False),
        Column("created_at", DateTime),
        Column("updated_at", DateTime),
    )

    metadata.create_all(engine)
    return engine, retrieval_profiles


def _seed_profiles(connection, table) -> None:
    connection.execute(
        insert(table),
        [
            {
                "profile_name": "max",
                "is_active": True,
                "is_default": False,
                "profile_config": {
                    "k_per_query": 30,
                    "max_query_variants": 6,
                    "max_docs": 24,
                    "max_hits_total": 72,
                    "max_hits_per_doc": 8,
                    "max_windows": 12,
                },
            },
            {
                "profile_name": "standard",
                "is_active": True,
                "is_default": True,
                "profile_config": {
                    "k_per_query": 12,
                    "max_query_variants": 3,
                    "max_docs": 10,
                    "max_hits_total": 30,
                    "max_hits_per_doc": 5,
                    "max_windows": 8,
                },
            },
            {
                "profile_name": "light",
                "is_active": True,
                "is_default": False,
                "profile_config": {
                    "k_per_query": 6,
                    "max_query_variants": 2,
                    "max_docs": 6,
                    "max_hits_total": 12,
                    "max_hits_per_doc": 3,
                    "max_windows": 4,
                },
            },
            {
                "profile_name": "deep",
                "is_active": True,
                "is_default": False,
                "profile_config": {
                    "k_per_query": 20,
                    "max_query_variants": 4,
                    "max_docs": 16,
                    "max_hits_total": 48,
                    "max_hits_per_doc": 6,
                    "max_windows": 10,
                },
            },
        ],
    )


def test_get_retrieval_profiles_returns_contract_fields_in_canonical_order() -> None:
    engine, retrieval_profiles = _build_test_engine()

    with engine.begin() as connection:
        _seed_profiles(connection, retrieval_profiles)

    profiles = get_retrieval_profiles(engine=engine)

    assert [profile["profile_name"] for profile in profiles] == ["light", "standard", "deep", "max"]
    assert profiles[1] == {
        "profile_name": "standard",
        "k_per_query": 12,
        "max_query_variants": 3,
        "max_docs": 10,
        "max_hits_total": 30,
        "max_hits_per_doc": 5,
        "max_windows": 8,
    }


def test_get_retrieval_profiles_active_only_filters_inactive_rows() -> None:
    engine, retrieval_profiles = _build_test_engine()

    with engine.begin() as connection:
        _seed_profiles(connection, retrieval_profiles)
        connection.execute(
            retrieval_profiles.update()
            .where(retrieval_profiles.c.profile_name == "deep")
            .values(is_active=False)
        )

    profiles = get_retrieval_profiles(engine=engine, active_only=True)
    assert [profile["profile_name"] for profile in profiles] == ["light", "standard", "max"]


def test_get_retrieval_profiles_raises_for_missing_required_config_key() -> None:
    engine, retrieval_profiles = _build_test_engine()

    with engine.begin() as connection:
        _seed_profiles(connection, retrieval_profiles)
        connection.execute(
            retrieval_profiles.update()
            .where(retrieval_profiles.c.profile_name == "light")
            .values(profile_config={"k_per_query": 6})
        )

    with pytest.raises(RetrievalProfileError, match="missing required config keys"):
        get_retrieval_profiles(engine=engine)


def test_get_retrieval_profiles_raises_for_non_positive_values() -> None:
    engine, retrieval_profiles = _build_test_engine()

    with engine.begin() as connection:
        _seed_profiles(connection, retrieval_profiles)
        connection.execute(
            retrieval_profiles.update()
            .where(retrieval_profiles.c.profile_name == "max")
            .values(
                profile_config={
                    "k_per_query": 0,
                    "max_query_variants": 6,
                    "max_docs": 24,
                    "max_hits_total": 72,
                    "max_hits_per_doc": 8,
                    "max_windows": 12,
                }
            )
        )

    with pytest.raises(RetrievalProfileError, match="non-positive 'k_per_query' value"):
        get_retrieval_profiles(engine=engine)

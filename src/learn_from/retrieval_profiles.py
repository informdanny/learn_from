from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sqlalchemy import MetaData, Table, case, select
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError


CANONICAL_PROFILE_ORDER: tuple[str, ...] = ("light", "standard", "deep", "max")
REQUIRED_PROFILE_CONFIG_KEYS: tuple[str, ...] = (
    "k_per_query",
    "max_query_variants",
    "max_docs",
    "max_hits_total",
    "max_hits_per_doc",
    "max_windows",
)


class RetrievalProfileError(RuntimeError):
    """Raised when retrieval profiles cannot be loaded or validated."""


def get_retrieval_profiles(*, engine: Engine, active_only: bool = False) -> list[dict[str, Any]]:
    """
    Return retrieval-profile presets from the canonical `retrieval_profiles` source table.

    Returned payload aligns with the v3 Layer 1 contract fields:
    - profile_name
    - k_per_query
    - max_query_variants
    - max_docs
    - max_hits_total
    - max_hits_per_doc
    - max_windows
    """
    with engine.connect() as connection:
        table = _reflect_retrieval_profiles_table(connection)
        rows = _fetch_profile_rows(connection=connection, table=table, active_only=active_only)

    return [_normalize_profile_row(row) for row in rows]


def _reflect_retrieval_profiles_table(connection: Connection) -> Table:
    metadata = MetaData()
    try:
        return Table("retrieval_profiles", metadata, autoload_with=connection)
    except SQLAlchemyError as exc:
        raise RetrievalProfileError(
            "Could not load retrieval_profiles table. Run migrations before querying profiles."
        ) from exc


def _fetch_profile_rows(
    *,
    connection: Connection,
    table: Table,
    active_only: bool,
) -> list[Mapping[str, Any]]:
    sort_order = case(
        *[(table.c.profile_name == name, idx) for idx, name in enumerate(CANONICAL_PROFILE_ORDER)],
        else_=len(CANONICAL_PROFILE_ORDER),
    )

    query = select(
        table.c.profile_name,
        table.c.profile_config,
    ).order_by(sort_order, table.c.profile_name)

    if active_only:
        query = query.where(table.c.is_active.is_(True))

    try:
        return connection.execute(query).mappings().all()
    except SQLAlchemyError as exc:
        raise RetrievalProfileError(f"Failed to query retrieval profiles: {exc}") from exc


def _normalize_profile_row(row: Mapping[str, Any]) -> dict[str, Any]:
    profile_name = str(row["profile_name"])
    config = row["profile_config"]
    if not isinstance(config, Mapping):
        raise RetrievalProfileError(
            f"Profile '{profile_name}' has invalid profile_config shape; expected an object."
        )

    missing_keys = [key for key in REQUIRED_PROFILE_CONFIG_KEYS if key not in config]
    if missing_keys:
        raise RetrievalProfileError(
            f"Profile '{profile_name}' is missing required config keys: {', '.join(missing_keys)}"
        )

    normalized: dict[str, Any] = {"profile_name": profile_name}
    for key in REQUIRED_PROFILE_CONFIG_KEYS:
        normalized[key] = _coerce_positive_int(config[key], profile_name=profile_name, key=key)

    return normalized


def _coerce_positive_int(value: Any, *, profile_name: str, key: str) -> int:
    if isinstance(value, bool):
        raise RetrievalProfileError(
            f"Profile '{profile_name}' has non-integer '{key}' value: {value!r}"
        )
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float) and value.is_integer():
        parsed = int(value)
    elif isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
    else:
        raise RetrievalProfileError(
            f"Profile '{profile_name}' has non-integer '{key}' value: {value!r}"
        )

    if parsed <= 0:
        raise RetrievalProfileError(
            f"Profile '{profile_name}' has non-positive '{key}' value: {parsed}"
        )

    return parsed

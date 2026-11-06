"""Add magazine_article reference type and targeted reference backfill.

Revision ID: 20260411_0002
Revises: 20260411_0001
Create Date: 2026-04-11 20:40:00
"""

from __future__ import annotations

import re

from alembic import context, op
import sqlalchemy as sa
from sqlalchemy.engine import Connection


# revision identifiers, used by Alembic.
revision = "20260411_0002"
down_revision = "20260411_0001"
branch_labels = None
depends_on = None


def _normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower().strip())


def _replace_reference_type_constraint(*, include_magazine_article: bool) -> None:
    op.drop_constraint("ck_references_catalog_type", "references_catalog", type_="check")
    if include_magazine_article:
        constraint_sql = (
            "type IN ('book', 'paper', 'study', 'scripture', 'talk', 'podcast', "
            "'youtube_video', 'author_creator', 'magazine_article', 'website', 'media_fragment', 'other')"
        )
    else:
        constraint_sql = (
            "type IN ('book', 'paper', 'study', 'scripture', 'talk', 'podcast', "
            "'youtube_video', 'author_creator', 'website', 'media_fragment', 'other')"
        )
    op.create_check_constraint("ck_references_catalog_type", "references_catalog", constraint_sql)


def _upsert_targeted_reference(
    *,
    connection: Connection,
    source_type: str,
    source_label: str,
    target_type: str,
    target_label: str,
) -> None:
    source_label_normalized = _normalize_label(source_label)
    target_label_normalized = _normalize_label(target_label)

    source_row = (
        connection.execute(
            sa.text(
                """
                SELECT reference_id, usage_count
                FROM references_catalog
                WHERE type = :source_type
                  AND label_normalized = :source_label_normalized
                ORDER BY reference_id
                LIMIT 1
                """
            ),
            {
                "source_type": source_type,
                "source_label_normalized": source_label_normalized,
            },
        )
        .mappings()
        .first()
    )
    if source_row is None:
        return

    source_reference_id = int(source_row["reference_id"])
    source_usage_count = int(source_row["usage_count"] or 1)

    target_row = (
        connection.execute(
            sa.text(
                """
                SELECT reference_id
                FROM references_catalog
                WHERE type = :target_type
                  AND label_normalized = :target_label_normalized
                ORDER BY reference_id
                LIMIT 1
                """
            ),
            {
                "target_type": target_type,
                "target_label_normalized": target_label_normalized,
            },
        )
        .mappings()
        .first()
    )

    if target_row is not None and int(target_row["reference_id"]) != source_reference_id:
        target_reference_id = int(target_row["reference_id"])
        connection.execute(
            sa.text(
                """
                INSERT INTO doc_references (
                    content_id,
                    reference_id,
                    context,
                    usage,
                    mention_count,
                    ref,
                    attributes,
                    created_at
                )
                SELECT
                    content_id,
                    :target_reference_id,
                    context,
                    usage,
                    mention_count,
                    ref,
                    attributes,
                    created_at
                FROM doc_references
                WHERE reference_id = :source_reference_id
                ON CONFLICT (content_id, reference_id) DO NOTHING
                """
            ),
            {
                "source_reference_id": source_reference_id,
                "target_reference_id": target_reference_id,
            },
        )
        connection.execute(
            sa.text("DELETE FROM doc_references WHERE reference_id = :source_reference_id"),
            {"source_reference_id": source_reference_id},
        )
        connection.execute(
            sa.text(
                """
                UPDATE references_catalog
                SET usage_count = CASE
                        WHEN COALESCE(usage_count, 1) + :source_usage_count > 1
                            THEN COALESCE(usage_count, 1) + :source_usage_count
                        ELSE 1
                    END,
                    last_seen_at = CURRENT_TIMESTAMP
                WHERE reference_id = :target_reference_id
                """
            ),
            {
                "source_usage_count": source_usage_count,
                "target_reference_id": target_reference_id,
            },
        )
        connection.execute(
            sa.text("DELETE FROM references_catalog WHERE reference_id = :source_reference_id"),
            {"source_reference_id": source_reference_id},
        )
        return

    connection.execute(
        sa.text(
            """
            UPDATE references_catalog
            SET type = :target_type,
                label = :target_label,
                label_normalized = :target_label_normalized,
                last_seen_at = CURRENT_TIMESTAMP
            WHERE reference_id = :source_reference_id
            """
        ),
        {
            "source_reference_id": source_reference_id,
            "target_type": target_type,
            "target_label": target_label,
            "target_label_normalized": target_label_normalized,
        },
    )


def upgrade() -> None:
    _replace_reference_type_constraint(include_magazine_article=True)
    if context.is_offline_mode():
        return

    connection = op.get_bind()

    _upsert_targeted_reference(
        connection=connection,
        source_type="paper",
        source_label="Acts 12:2",
        target_type="scripture",
        target_label="Acts 12:2",
    )
    _upsert_targeted_reference(
        connection=connection,
        source_type="scripture",
        source_label="Revelation (Euphrates River / angels bound tradition)",
        target_type="scripture",
        target_label="Book of Revelation",
    )
    _upsert_targeted_reference(
        connection=connection,
        source_type="scripture",
        source_label="1 Enoch chapters 6-11",
        target_type="scripture",
        target_label="1 Enoch 6-11",
    )


def downgrade() -> None:
    op.execute("UPDATE references_catalog SET type = 'other' WHERE type = 'magazine_article'")
    _replace_reference_type_constraint(include_magazine_article=False)

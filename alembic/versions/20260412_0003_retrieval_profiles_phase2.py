"""Create retrieval profiles for Phase 2 Layer 1 capability tools.

Revision ID: 20260412_0003
Revises: 20260411_0002
Create Date: 2026-04-12 09:10:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "20260412_0003"
down_revision = "20260411_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "retrieval_profiles",
        sa.Column("profile_name", sa.String(length=20), primary_key=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("is_default", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("profile_config", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.CheckConstraint(
            "profile_name IN ('light', 'standard', 'deep', 'max')",
            name="ck_retrieval_profiles_name",
        ),
    )
    op.create_index("idx_retrieval_profiles_active", "retrieval_profiles", ["is_active"])

    op.execute(
        """
        INSERT INTO retrieval_profiles (profile_name, is_active, is_default, profile_config)
        VALUES
            ('light', true, false, '{
                "k_per_query": 6,
                "max_query_variants": 2,
                "max_docs": 6,
                "max_hits_total": 12,
                "max_hits_per_doc": 3,
                "max_windows": 4
            }'::jsonb),
            ('standard', true, true, '{
                "k_per_query": 12,
                "max_query_variants": 3,
                "max_docs": 10,
                "max_hits_total": 30,
                "max_hits_per_doc": 5,
                "max_windows": 8
            }'::jsonb),
            ('deep', true, false, '{
                "k_per_query": 20,
                "max_query_variants": 4,
                "max_docs": 16,
                "max_hits_total": 48,
                "max_hits_per_doc": 6,
                "max_windows": 10
            }'::jsonb),
            ('max', true, false, '{
                "k_per_query": 30,
                "max_query_variants": 6,
                "max_docs": 24,
                "max_hits_total": 72,
                "max_hits_per_doc": 8,
                "max_windows": 12
            }'::jsonb)
        ON CONFLICT (profile_name)
        DO UPDATE SET
            is_active = EXCLUDED.is_active,
            is_default = EXCLUDED.is_default,
            profile_config = EXCLUDED.profile_config,
            updated_at = CURRENT_TIMESTAMP
        """
    )


def downgrade() -> None:
    op.drop_index("idx_retrieval_profiles_active", table_name="retrieval_profiles")
    op.drop_table("retrieval_profiles")

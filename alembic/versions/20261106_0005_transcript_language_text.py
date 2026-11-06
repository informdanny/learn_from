"""Preserve complete provider transcript language identifiers.

Revision ID: 20261106_0005
Revises: 20260711_0004
Create Date: 2026-11-06
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20261106_0005"
down_revision = "20260711_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "doc_content", "language", existing_type=sa.String(length=10), type_=sa.Text()
    )


def downgrade() -> None:
    # Hold the same lock as ALTER TYPE while checking, including trailing spaces
    # that PostgreSQL would otherwise silently trim to the varchar limit.
    op.execute(sa.text("""
        DO $$
        BEGIN
            LOCK TABLE doc_content IN ACCESS EXCLUSIVE MODE;
            IF EXISTS (SELECT 1 FROM doc_content WHERE char_length(language) > 10) THEN
                RAISE EXCEPTION 'Cannot downgrade doc_content.language: values exceed 10 characters'
                    USING ERRCODE = '22001';
            END IF;
        END $$;
    """))
    op.alter_column(
        "doc_content", "language", existing_type=sa.Text(), type_=sa.String(length=10)
    )

"""Add the versioned metadata extraction V2 foundation.

Revision ID: 20260711_0004
Revises: 20260412_0003
Create Date: 2026-07-11 12:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260711_0004"
down_revision = "20260412_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "extraction_profiles",
        sa.Column("profile_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("profile_name", sa.String(length=200), nullable=False),
        sa.Column("scope", sa.String(length=30), nullable=False),
        sa.Column("source_id", sa.String(length=50), nullable=True),
        sa.Column("channel_id", sa.String(length=255), nullable=True),
        sa.Column("category_id", sa.Integer(), nullable=True),
        sa.Column("learning_mode", sa.String(length=20), nullable=False, server_default="autopilot"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.CheckConstraint(
            "scope IN ('global', 'source', 'category', 'channel', 'channel_category')",
            name="ck_extraction_profiles_scope",
        ),
        sa.CheckConstraint(
            "learning_mode IN ('off', 'observe', 'assisted', 'autopilot')",
            name="ck_extraction_profiles_learning_mode",
        ),
        sa.ForeignKeyConstraint(["source_id"], ["media_sources.source_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["channel_id"], ["channels.channel_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["category_id"], ["doc_categories.category_id"], ondelete="CASCADE"),
    )
    op.create_index("idx_extraction_profiles_scope", "extraction_profiles", ["scope", "is_active"])
    op.create_index("idx_extraction_profiles_source", "extraction_profiles", ["source_id"])
    op.create_index("idx_extraction_profiles_channel", "extraction_profiles", ["channel_id"])
    op.create_index("idx_extraction_profiles_category", "extraction_profiles", ["category_id"])

    op.create_table(
        "extraction_profile_versions",
        sa.Column("profile_version_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("profile_id", sa.Integer(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="draft"),
        sa.Column("config", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("published_at", sa.DateTime(), nullable=True),
        sa.Column("retired_at", sa.DateTime(), nullable=True),
        sa.CheckConstraint(
            "status IN ('draft', 'testing', 'active', 'retired')",
            name="ck_extraction_profile_versions_status",
        ),
        sa.ForeignKeyConstraint(["profile_id"], ["extraction_profiles.profile_id"], ondelete="CASCADE"),
        sa.UniqueConstraint("profile_id", "version", name="uq_extraction_profile_versions_version"),
    )
    op.create_index(
        "idx_extraction_profile_versions_active",
        "extraction_profile_versions",
        ["profile_id", "status"],
    )

    op.create_table(
        "extraction_runs",
        sa.Column("run_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("content_id", sa.Integer(), nullable=False),
        sa.Column("media_id", sa.String(length=255), nullable=False),
        sa.Column("profile_version_id", sa.Integer(), nullable=True),
        sa.Column("release_version", sa.String(length=20), nullable=False),
        sa.Column("run_mode", sa.String(length=20), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="running"),
        sa.Column("extractor_model", sa.String(length=100), nullable=False),
        sa.Column("resolver_model", sa.String(length=100), nullable=False),
        sa.Column("learner_model", sa.String(length=100), nullable=False),
        sa.Column("manifest", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("raw_output", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("resolved_output", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("input_characters", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("started_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.CheckConstraint(
            "run_mode IN ('bootstrap', 'stable', 'audit', 'ad_hoc')",
            name="ck_extraction_runs_mode",
        ),
        sa.CheckConstraint(
            "status IN ('running', 'extracted', 'resolved', 'completed', 'failed')",
            name="ck_extraction_runs_status",
        ),
        sa.ForeignKeyConstraint(["content_id"], ["doc_content.content_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["media_id"], ["media.media_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["profile_version_id"],
            ["extraction_profile_versions.profile_version_id"],
            ondelete="SET NULL",
        ),
    )
    op.create_index("idx_extraction_runs_content", "extraction_runs", ["content_id", "started_at"])
    op.create_index("idx_extraction_runs_media", "extraction_runs", ["media_id", "started_at"])
    op.create_index("idx_extraction_runs_status", "extraction_runs", ["status"])
    op.create_index("idx_extraction_runs_release", "extraction_runs", ["release_version"])

    op.create_table(
        "transcript_analysis_versions",
        sa.Column("analysis_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("content_id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=True),
        sa.Column("normalizer_version", sa.String(length=30), nullable=False),
        sa.Column("analysis_text", sa.Text(), nullable=False),
        sa.Column("segment_map", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["content_id"], ["doc_content.content_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["run_id"], ["extraction_runs.run_id"], ondelete="SET NULL"),
    )
    op.create_index(
        "idx_transcript_analysis_content",
        "transcript_analysis_versions",
        ["content_id", "created_at"],
    )

    op.create_table(
        "extraction_observations",
        sa.Column("observation_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("content_id", sa.Integer(), nullable=False),
        sa.Column("dimension", sa.String(length=30), nullable=False),
        sa.Column("raw_value", sa.String(length=512), nullable=False),
        sa.Column("normalized_value", sa.String(length=512), nullable=True),
        sa.Column("canonical_value", sa.String(length=512), nullable=True),
        sa.Column("evidence_text", sa.Text(), nullable=True),
        sa.Column("start_seconds", sa.Numeric(12, 3), nullable=True),
        sa.Column("end_seconds", sa.Numeric(12, 3), nullable=True),
        sa.Column("confidence", sa.Numeric(5, 4), nullable=False),
        sa.Column("resolution_status", sa.String(length=20), nullable=False),
        sa.Column("resolution_basis", sa.String(length=30), nullable=False),
        sa.Column("details", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.CheckConstraint(
            "dimension IN ('category', 'topic', 'keyword', 'participant', 'entity', 'reference')",
            name="ck_extraction_observations_dimension",
        ),
        sa.CheckConstraint(
            "resolution_status IN ('observed', 'provisional', 'trusted', 'rejected', 'unresolved')",
            name="ck_extraction_observations_status",
        ),
        sa.CheckConstraint(
            "resolution_basis IN ('explicit', 'normalized', 'alias', 'inferred', 'externally_verified', 'unresolved')",
            name="ck_extraction_observations_basis",
        ),
        sa.ForeignKeyConstraint(["run_id"], ["extraction_runs.run_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["content_id"], ["doc_content.content_id"], ondelete="CASCADE"),
    )
    op.create_index("idx_extraction_observations_run", "extraction_observations", ["run_id"])
    op.create_index(
        "idx_extraction_observations_value",
        "extraction_observations",
        ["dimension", "normalized_value"],
    )
    op.create_index(
        "idx_extraction_observations_status",
        "extraction_observations",
        ["resolution_status", "confidence"],
    )

    op.create_table(
        "metadata_value_aliases",
        sa.Column("alias_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("dimension", sa.String(length=30), nullable=False),
        sa.Column("canonical_value", sa.String(length=512), nullable=False),
        sa.Column("alias_text", sa.String(length=512), nullable=False),
        sa.Column("alias_normalized", sa.String(length=512), nullable=False),
        sa.Column("source_id", sa.String(length=50), nullable=True),
        sa.Column("channel_id", sa.String(length=255), nullable=True),
        sa.Column("category_id", sa.Integer(), nullable=True),
        sa.Column("confidence", sa.Numeric(5, 4), nullable=False),
        sa.Column("trust_status", sa.String(length=20), nullable=False, server_default="observed"),
        sa.Column("evidence_count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("is_blocked", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_by", sa.String(length=30), nullable=False, server_default="learner"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.CheckConstraint(
            "dimension IN ('topic', 'keyword', 'participant', 'entity', 'reference')",
            name="ck_metadata_value_aliases_dimension",
        ),
        sa.CheckConstraint(
            "trust_status IN ('observed', 'provisional', 'trusted', 'rejected')",
            name="ck_metadata_value_aliases_trust",
        ),
        sa.CheckConstraint(
            "created_by IN ('learner', 'resolver', 'user', 'migration')",
            name="ck_metadata_value_aliases_creator",
        ),
        sa.ForeignKeyConstraint(["source_id"], ["media_sources.source_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["channel_id"], ["channels.channel_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["category_id"], ["doc_categories.category_id"], ondelete="CASCADE"),
        sa.UniqueConstraint(
            "dimension",
            "alias_normalized",
            "source_id",
            "channel_id",
            "category_id",
            name="uq_metadata_value_aliases_scope",
        ),
    )
    op.create_index(
        "idx_metadata_value_aliases_lookup",
        "metadata_value_aliases",
        ["dimension", "alias_normalized", "trust_status"],
    )
    op.create_index("idx_metadata_value_aliases_channel", "metadata_value_aliases", ["channel_id"])

    op.create_table(
        "metadata_learning_events",
        sa.Column("event_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("profile_id", sa.Integer(), nullable=True),
        sa.Column("run_id", sa.Integer(), nullable=True),
        sa.Column("dimension", sa.String(length=30), nullable=False),
        sa.Column("action", sa.String(length=30), nullable=False),
        sa.Column("raw_value", sa.String(length=512), nullable=True),
        sa.Column("canonical_value", sa.String(length=512), nullable=True),
        sa.Column("confidence", sa.Numeric(5, 4), nullable=True),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="proposed"),
        sa.Column("details", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("applied_at", sa.DateTime(), nullable=True),
        sa.Column("reversed_at", sa.DateTime(), nullable=True),
        sa.CheckConstraint(
            "status IN ('proposed', 'applied', 'rejected', 'reversed')",
            name="ck_metadata_learning_events_status",
        ),
        sa.ForeignKeyConstraint(["profile_id"], ["extraction_profiles.profile_id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["run_id"], ["extraction_runs.run_id"], ondelete="SET NULL"),
    )
    op.create_index("idx_metadata_learning_events_profile", "metadata_learning_events", ["profile_id", "status"])
    op.create_index("idx_metadata_learning_events_run", "metadata_learning_events", ["run_id"])

    op.create_table(
        "enrichment_jobs",
        sa.Column("job_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.Integer(), nullable=True),
        sa.Column("content_id", sa.Integer(), nullable=False),
        sa.Column("job_type", sa.String(length=30), nullable=False),
        sa.Column("priority", sa.String(length=20), nullable=False, server_default="background"),
        sa.Column("provider", sa.String(length=30), nullable=False, server_default="openai"),
        sa.Column("model", sa.String(length=100), nullable=False),
        sa.Column("service_tier", sa.String(length=20), nullable=False, server_default="flex"),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.CheckConstraint(
            "job_type IN ('resolve', 'learn', 'backfill')",
            name="ck_enrichment_jobs_type",
        ),
        sa.CheckConstraint(
            "priority IN ('immediate', 'background', 'batch')",
            name="ck_enrichment_jobs_priority",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed', 'cancelled')",
            name="ck_enrichment_jobs_status",
        ),
        sa.ForeignKeyConstraint(["run_id"], ["extraction_runs.run_id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["content_id"], ["doc_content.content_id"], ondelete="CASCADE"),
    )
    op.create_index("idx_enrichment_jobs_queue", "enrichment_jobs", ["status", "priority", "created_at"])

    op.create_table(
        "reference_scripture_details",
        sa.Column("reference_id", sa.Integer(), primary_key=True),
        sa.Column("canon", sa.String(length=100), nullable=False),
        sa.Column("collection", sa.String(length=100), nullable=False),
        sa.Column("work", sa.String(length=150), nullable=False),
        sa.Column("book", sa.String(length=150), nullable=True),
        sa.Column("chapter_start", sa.Integer(), nullable=True),
        sa.Column("chapter_end", sa.Integer(), nullable=True),
        sa.Column("verse_start", sa.Integer(), nullable=True),
        sa.Column("verse_end", sa.Integer(), nullable=True),
        sa.Column("section_start", sa.Integer(), nullable=True),
        sa.Column("section_end", sa.Integer(), nullable=True),
        sa.Column("canonical_locator", sa.String(length=512), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["reference_id"], ["references_catalog.reference_id"], ondelete="CASCADE"),
    )
    op.create_index("idx_reference_scripture_book", "reference_scripture_details", ["canon", "book"])
    op.create_index(
        "idx_reference_scripture_chapter",
        "reference_scripture_details",
        ["canon", "book", "chapter_start"],
    )

    op.create_table(
        "doc_reference_mentions",
        sa.Column("mention_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("content_id", sa.Integer(), nullable=False),
        sa.Column("reference_id", sa.Integer(), nullable=True),
        sa.Column("run_id", sa.Integer(), nullable=True),
        sa.Column("raw_mention", sa.String(length=512), nullable=False),
        sa.Column("evidence_text", sa.Text(), nullable=True),
        sa.Column("start_seconds", sa.Numeric(12, 3), nullable=True),
        sa.Column("end_seconds", sa.Numeric(12, 3), nullable=True),
        sa.Column("resolution_basis", sa.String(length=30), nullable=False),
        sa.Column("confidence", sa.Numeric(5, 4), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.CheckConstraint(
            "resolution_basis IN ('explicit', 'normalized', 'alias', 'inferred', 'externally_verified', 'unresolved')",
            name="ck_doc_reference_mentions_basis",
        ),
        sa.ForeignKeyConstraint(["content_id"], ["doc_content.content_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["reference_id"], ["references_catalog.reference_id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["run_id"], ["extraction_runs.run_id"], ondelete="SET NULL"),
    )
    op.create_index("idx_doc_reference_mentions_content", "doc_reference_mentions", ["content_id"])
    op.create_index("idx_doc_reference_mentions_reference", "doc_reference_mentions", ["reference_id"])

    op.execute(
        """
        INSERT INTO extraction_profiles (profile_name, scope, learning_mode, is_active)
        VALUES ('Global Metadata V2', 'global', 'autopilot', true)
        """
    )
    op.execute(
        """
        INSERT INTO extraction_profile_versions (profile_id, version, status, config, published_at)
        SELECT profile_id, 1, 'active', '{
            "schema_version": "2.0",
            "reference_modules": [],
            "coverage": {
                "topics": "top_n",
                "keywords": "top_n",
                "participants": "exhaustive_named",
                "entities": "high_signal",
                "references": "exhaustive"
            },
            "model_routing": {
                "bootstrap_documents": 3,
                "audit_interval": 10,
                "stable_model": "gpt-5.6-luna",
                "enhanced_model": "gpt-5.6-terra",
                "resolver_model": "gpt-5.6-terra",
                "learner_model": "gpt-5.6-terra"
            },
            "confidence": {
                "trusted": 0.95,
                "provisional": 0.80,
                "alias_documents": 3
            },
            "prompt_instructions": ""
        }'::jsonb, CURRENT_TIMESTAMP
        FROM extraction_profiles
        WHERE profile_name = 'Global Metadata V2' AND scope = 'global'
        """
    )
    op.execute(
        """
        INSERT INTO extraction_versions (
            version, is_current, changes_summary, prompt_changes, schema_changes, rollout_status
        )
        VALUES (
            '2.0', false,
            'Three-worker metadata extraction with versioned profiles, full-transcript evidence, and learning.',
            'Category, source, channel, value-hint, and coverage-policy prompt composition.',
            'Adds V2 run, observation, learning, enrichment, and specialized scripture-reference records.',
            'testing'
        )
        ON CONFLICT (version) DO NOTHING
        """
    )


def downgrade() -> None:
    op.execute("DELETE FROM extraction_versions WHERE version = '2.0'")
    op.drop_index("idx_doc_reference_mentions_reference", table_name="doc_reference_mentions")
    op.drop_index("idx_doc_reference_mentions_content", table_name="doc_reference_mentions")
    op.drop_table("doc_reference_mentions")
    op.drop_index("idx_reference_scripture_chapter", table_name="reference_scripture_details")
    op.drop_index("idx_reference_scripture_book", table_name="reference_scripture_details")
    op.drop_table("reference_scripture_details")
    op.drop_index("idx_enrichment_jobs_queue", table_name="enrichment_jobs")
    op.drop_table("enrichment_jobs")
    op.drop_index("idx_metadata_learning_events_run", table_name="metadata_learning_events")
    op.drop_index("idx_metadata_learning_events_profile", table_name="metadata_learning_events")
    op.drop_table("metadata_learning_events")
    op.drop_index("idx_metadata_value_aliases_channel", table_name="metadata_value_aliases")
    op.drop_index("idx_metadata_value_aliases_lookup", table_name="metadata_value_aliases")
    op.drop_table("metadata_value_aliases")
    op.drop_index("idx_extraction_observations_status", table_name="extraction_observations")
    op.drop_index("idx_extraction_observations_value", table_name="extraction_observations")
    op.drop_index("idx_extraction_observations_run", table_name="extraction_observations")
    op.drop_table("extraction_observations")
    op.drop_index("idx_transcript_analysis_content", table_name="transcript_analysis_versions")
    op.drop_table("transcript_analysis_versions")
    op.drop_index("idx_extraction_runs_release", table_name="extraction_runs")
    op.drop_index("idx_extraction_runs_status", table_name="extraction_runs")
    op.drop_index("idx_extraction_runs_media", table_name="extraction_runs")
    op.drop_index("idx_extraction_runs_content", table_name="extraction_runs")
    op.drop_table("extraction_runs")
    op.drop_index("idx_extraction_profile_versions_active", table_name="extraction_profile_versions")
    op.drop_table("extraction_profile_versions")
    op.drop_index("idx_extraction_profiles_category", table_name="extraction_profiles")
    op.drop_index("idx_extraction_profiles_channel", table_name="extraction_profiles")
    op.drop_index("idx_extraction_profiles_source", table_name="extraction_profiles")
    op.drop_index("idx_extraction_profiles_scope", table_name="extraction_profiles")
    op.drop_table("extraction_profiles")

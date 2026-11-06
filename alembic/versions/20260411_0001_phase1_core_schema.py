"""Create Phase 1 core schema.

Revision ID: 20260411_0001
Revises:
Create Date: 2026-04-11 10:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "20260411_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "media_sources",
        sa.Column("source_id", sa.String(length=50), primary_key=True),
        sa.Column("source_name", sa.String(length=100), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("base_url", sa.String(length=255), nullable=True),
        sa.Column("supports_channels", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("supports_transcripts", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("requires_api_key", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )

    op.create_table(
        "channels",
        sa.Column("channel_id", sa.String(length=255), primary_key=True),
        sa.Column("source_id", sa.String(length=50), nullable=False),
        sa.Column("channel_name", sa.String(length=255), nullable=False),
        sa.Column("channel_url", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("thumbnail_url", sa.Text(), nullable=True),
        sa.Column("subscriber_count", sa.BigInteger(), nullable=True),
        sa.Column(
            "intelligent_metadata_enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column(
            "always_sync_to_openai",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column("backfill_months", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("last_synced_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["source_id"], ["media_sources.source_id"]),
    )
    op.create_index("idx_channels_source", "channels", ["source_id"])
    op.create_index("idx_channels_last_synced", "channels", ["last_synced_at"])

    op.create_table(
        "doc_categories",
        sa.Column("category_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("category_name", sa.String(length=100), nullable=False, unique=True),
        sa.Column("is_default", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index("idx_categories_default", "doc_categories", ["is_default"])

    op.create_table(
        "media",
        sa.Column("media_id", sa.String(length=255), primary_key=True),
        sa.Column("source_id", sa.String(length=50), nullable=False),
        sa.Column("channel_id", sa.String(length=255), nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("url", sa.Text(), nullable=False, unique=True),
        sa.Column("thumbnail_url", sa.Text(), nullable=True),
        sa.Column("duration", sa.Integer(), nullable=True),
        sa.Column("publish_date", sa.DateTime(), nullable=True),
        sa.Column("views", sa.BigInteger(), nullable=True),
        sa.Column("likes", sa.BigInteger(), nullable=True),
        sa.Column("comments", sa.BigInteger(), nullable=True),
        sa.Column("source_type", sa.String(length=20), nullable=False, server_default="ad_hoc"),
        sa.Column("category_id", sa.Integer(), nullable=True),
        sa.Column("is_user_saved", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("openai_sync_status", sa.String(length=20), nullable=False, server_default="not_synced"),
        sa.Column("openai_file_id", sa.String(length=255), nullable=True),
        sa.Column("openai_synced_at", sa.DateTime(), nullable=True),
        sa.Column("openai_sync_error", sa.Text(), nullable=True),
        sa.Column("extraction_version", sa.String(length=20), nullable=False, server_default="1.0"),
        sa.Column("extraction_quality_score", sa.Numeric(3, 2), nullable=True),
        sa.Column("ingestion_date", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("last_updated", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.CheckConstraint("source_type IN ('channel', 'ad_hoc')", name="ck_media_source_type"),
        sa.CheckConstraint(
            "openai_sync_status IN ('not_synced', 'pending', 'syncing', 'synced', 'failed')",
            name="ck_media_openai_sync_status",
        ),
        sa.ForeignKeyConstraint(["source_id"], ["media_sources.source_id"]),
        sa.ForeignKeyConstraint(["channel_id"], ["channels.channel_id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["category_id"], ["doc_categories.category_id"], ondelete="SET NULL"),
    )
    op.create_index("idx_media_source", "media", ["source_id"])
    op.create_index("idx_media_channel", "media", ["channel_id"])
    op.create_index("idx_media_source_type", "media", ["source_type"])
    op.create_index("idx_media_category", "media", ["category_id"])
    op.create_index("idx_media_publish_date", "media", ["publish_date"])
    op.create_index("idx_media_openai_sync_status", "media", ["openai_sync_status"])
    op.create_index("idx_media_extraction_version", "media", ["extraction_version"])
    op.create_index(
        "idx_media_title_search",
        "media",
        [sa.text("to_tsvector('english', title)")],
        postgresql_using="gin",
    )
    op.create_index(
        "idx_media_description_search",
        "media",
        [sa.text("to_tsvector('english', description)")],
        postgresql_using="gin",
    )

    op.create_table(
        "doc_content",
        sa.Column("content_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("media_id", sa.String(length=255), nullable=False),
        sa.Column("full_text", sa.Text(), nullable=False),
        sa.Column("formatted_text", sa.Text(), nullable=True),
        sa.Column("timestamped_segments", postgresql.JSONB(), nullable=True),
        sa.Column("chunk_type", sa.String(length=20), nullable=False, server_default="complete"),
        sa.Column("chunk_index", sa.Integer(), nullable=True),
        sa.Column("parent_content_id", sa.Integer(), nullable=True),
        sa.Column("word_count", sa.Integer(), nullable=True),
        sa.Column("character_count", sa.Integer(), nullable=True),
        sa.Column("language", sa.String(length=10), nullable=False, server_default="en"),
        sa.Column("metadata_updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.CheckConstraint("chunk_type IN ('complete', 'chunk')", name="ck_doc_content_chunk_type"),
        sa.ForeignKeyConstraint(["media_id"], ["media.media_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["parent_content_id"], ["doc_content.content_id"], ondelete="CASCADE"),
    )
    op.create_index("idx_doc_content_media", "doc_content", ["media_id"])
    op.create_index("idx_doc_content_chunk_type", "doc_content", ["chunk_type"])
    op.create_index("idx_doc_content_parent", "doc_content", ["parent_content_id"])
    op.create_index(
        "idx_doc_content_search",
        "doc_content",
        [sa.text("to_tsvector('english', full_text)")],
        postgresql_using="gin",
    )

    op.create_table(
        "sync_jobs",
        sa.Column("job_id", sa.String(length=255), primary_key=True),
        sa.Column("job_type", sa.String(length=50), nullable=False),
        sa.Column("channel_id", sa.String(length=255), nullable=True),
        sa.Column("media_ids", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("progress", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_processed_media_id", sa.String(length=255), nullable=True),
        sa.Column("checkpoint_data", postgresql.JSONB(), nullable=True),
        sa.Column("is_resumable", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("resume_priority", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("items_total", sa.Integer(), nullable=True),
        sa.Column("items_processed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("items_succeeded", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("items_failed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("budget_spent", sa.Numeric(10, 6), nullable=False, server_default="0.00"),
        sa.Column("budget_halt_reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("last_checkpoint_at", sa.DateTime(), nullable=True),
        sa.Column("resume_after", sa.DateTime(), nullable=True),
        sa.CheckConstraint(
            "job_type IN ('channel_backfill', 'daily_sync', 'ad_hoc_media', 'openai_sync', 'category_classification')",
            name="ck_sync_jobs_job_type",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'processing', 'completed', 'failed', 'budget_halted', 'paused', 'cancelled')",
            name="ck_sync_jobs_status",
        ),
        sa.ForeignKeyConstraint(["channel_id"], ["channels.channel_id"], ondelete="SET NULL"),
    )
    op.create_index("idx_sync_jobs_type", "sync_jobs", ["job_type"])
    op.create_index("idx_sync_jobs_status", "sync_jobs", ["status"])
    op.create_index("idx_sync_jobs_channel", "sync_jobs", ["channel_id"])
    op.create_index("idx_sync_jobs_created", "sync_jobs", ["created_at"])
    op.create_index("idx_sync_jobs_resumable", "sync_jobs", ["is_resumable", "status"])
    op.create_index(
        "idx_sync_jobs_resume_priority",
        "sync_jobs",
        [sa.text("resume_priority DESC"), sa.text("created_at ASC")],
    )
    op.create_index("idx_sync_jobs_resume_after", "sync_jobs", ["resume_after"])

    op.create_table(
        "openai_usage_log",
        sa.Column("log_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("request_type", sa.String(length=50), nullable=False),
        sa.Column("model", sa.String(length=100), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("cost", sa.Numeric(10, 6), nullable=True),
        sa.Column("media_id", sa.String(length=255), nullable=True),
        sa.Column("content_id", sa.Integer(), nullable=True),
        sa.Column("job_id", sa.String(length=255), nullable=True),
        sa.Column("user_query", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["media_id"], ["media.media_id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["content_id"], ["doc_content.content_id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["job_id"], ["sync_jobs.job_id"], ondelete="SET NULL"),
    )
    op.create_index("idx_usage_log_type", "openai_usage_log", ["request_type"])
    op.create_index("idx_usage_log_created", "openai_usage_log", ["created_at"])
    op.create_index("idx_usage_log_media", "openai_usage_log", ["media_id"])
    op.create_index("idx_usage_log_content", "openai_usage_log", ["content_id"])

    op.create_table(
        "metadata_schemas",
        sa.Column("schema_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("attribute_key", sa.String(length=100), nullable=False),
        sa.Column("attribute_type", sa.String(length=20), nullable=False),
        sa.Column("scope", sa.String(length=20), nullable=False),
        sa.Column("source_id", sa.String(length=50), nullable=True),
        sa.Column("category_id", sa.Integer(), nullable=True),
        sa.Column("display_name", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("is_searchable", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("is_required", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("sync_to_openai", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("openai_key", sa.String(length=64), nullable=True),
        sa.Column("openai_priority", sa.Integer(), nullable=False, server_default="5"),
        sa.Column("extraction_version", sa.String(length=20), nullable=False, server_default="1.0"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.CheckConstraint(
            "attribute_type IN ('string', 'string_array', 'number', 'boolean', 'date')",
            name="ck_metadata_schemas_attribute_type",
        ),
        sa.CheckConstraint("scope IN ('universal', 'source', 'category')", name="ck_metadata_schemas_scope"),
        sa.ForeignKeyConstraint(["source_id"], ["media_sources.source_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["category_id"], ["doc_categories.category_id"], ondelete="CASCADE"),
        sa.UniqueConstraint(
            "attribute_key",
            "scope",
            "source_id",
            "category_id",
            name="uq_metadata_schemas_scope_key",
        ),
    )
    op.create_index("idx_metadata_schemas_scope", "metadata_schemas", ["scope"])
    op.create_index("idx_metadata_schemas_source", "metadata_schemas", ["source_id"])
    op.create_index("idx_metadata_schemas_category", "metadata_schemas", ["category_id"])
    op.create_index("idx_metadata_schemas_searchable", "metadata_schemas", ["is_searchable"])
    op.create_index(
        "idx_metadata_schemas_openai",
        "metadata_schemas",
        ["sync_to_openai", "openai_priority"],
    )

    op.create_table(
        "metadata_values",
        sa.Column("value_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("schema_id", sa.Integer(), nullable=False),
        sa.Column("value_text", sa.String(length=512), nullable=False),
        sa.Column("value_normalized", sa.String(length=512), nullable=True),
        sa.Column("usage_count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("first_seen", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("last_seen", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("is_verified", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("is_deprecated", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("replaced_by", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["schema_id"], ["metadata_schemas.schema_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["replaced_by"], ["metadata_values.value_id"], ondelete="SET NULL"),
        sa.UniqueConstraint("schema_id", "value_normalized", name="uq_metadata_values_schema_value"),
    )
    op.create_index("idx_metadata_values_schema", "metadata_values", ["schema_id"])
    op.create_index("idx_metadata_values_usage", "metadata_values", [sa.text("usage_count DESC")])
    op.create_index("idx_metadata_values_verified", "metadata_values", ["is_verified"])
    op.create_index("idx_metadata_values_normalized", "metadata_values", ["value_normalized"])

    op.create_table(
        "extraction_versions",
        sa.Column("version_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("version", sa.String(length=20), nullable=False, unique=True),
        sa.Column("release_date", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("is_current", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("changes_summary", sa.Text(), nullable=True),
        sa.Column("prompt_changes", sa.Text(), nullable=True),
        sa.Column("schema_changes", sa.Text(), nullable=True),
        sa.Column("avg_extraction_cost", sa.Numeric(10, 6), nullable=True),
        sa.Column("avg_extraction_time", sa.Integer(), nullable=True),
        sa.Column("rollout_status", sa.String(length=20), nullable=False, server_default="testing"),
        sa.Column("rollout_started", sa.DateTime(), nullable=True),
        sa.Column("rollout_completed", sa.DateTime(), nullable=True),
        sa.Column("media_extracted", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("success_rate", sa.Numeric(5, 2), nullable=True),
        sa.CheckConstraint(
            "rollout_status IN ('testing', 'staged', 'production', 'deprecated')",
            name="ck_extraction_versions_rollout_status",
        ),
    )
    op.create_index("idx_extraction_versions_current", "extraction_versions", ["is_current"])
    op.create_index("idx_extraction_versions_rollout", "extraction_versions", ["rollout_status"])

    op.create_table(
        "entities_catalog",
        sa.Column("entity_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("type", sa.String(length=30), nullable=False),
        sa.Column("label", sa.String(length=255), nullable=False),
        sa.Column("label_normalized", sa.String(length=255), nullable=False),
        sa.Column("aliases", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("usage_count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("first_seen_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("last_seen_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.CheckConstraint(
            "type IN ('person', 'organization', 'brand', 'product', 'service', "
            "'software', 'framework', 'metric', 'location', 'timeframe', "
            "'event', 'topic', 'concept', 'other')",
            name="ck_entities_catalog_type",
        ),
        sa.UniqueConstraint("type", "label_normalized", name="uq_entities_catalog_type_label"),
    )
    op.create_index("idx_entities_type", "entities_catalog", ["type"])
    op.create_index("idx_entities_label_normalized", "entities_catalog", ["label_normalized"])
    op.create_index("idx_entities_usage_count", "entities_catalog", [sa.text("usage_count DESC")])

    op.create_table(
        "references_catalog",
        sa.Column("reference_id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("type", sa.String(length=30), nullable=False),
        sa.Column("label", sa.String(length=255), nullable=False),
        sa.Column("label_normalized", sa.String(length=255), nullable=False),
        sa.Column("identifiers", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("usage_count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("first_seen_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("last_seen_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.CheckConstraint(
            "type IN ('book', 'paper', 'study', 'scripture', 'talk', 'podcast', "
            "'youtube_video', 'author_creator', 'website', 'media_fragment', 'other')",
            name="ck_references_catalog_type",
        ),
        sa.UniqueConstraint("type", "label_normalized", name="uq_references_catalog_type_label"),
    )
    op.create_index("idx_references_type", "references_catalog", ["type"])
    op.create_index("idx_references_label_normalized", "references_catalog", ["label_normalized"])
    op.create_index(
        "idx_references_identifiers",
        "references_catalog",
        ["identifiers"],
        postgresql_using="gin",
    )

    op.create_table(
        "doc_entities",
        sa.Column("content_id", sa.Integer(), nullable=False),
        sa.Column("entity_id", sa.Integer(), nullable=False),
        sa.Column("context", sa.String(length=150), nullable=True),
        sa.Column("usage", sa.String(length=20), nullable=False, server_default="neutral"),
        sa.Column("mention_count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("ref", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.CheckConstraint(
            "usage IN ('primary_topic', 'recommended', 'criticized', 'quoted', "
            "'evidence', 'example', 'comparison', 'background', 'neutral')",
            name="ck_doc_entities_usage",
        ),
        sa.ForeignKeyConstraint(["content_id"], ["doc_content.content_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["entity_id"], ["entities_catalog.entity_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("content_id", "entity_id"),
    )
    op.create_index("idx_doc_entities_content", "doc_entities", ["content_id"])
    op.create_index("idx_doc_entities_entity", "doc_entities", ["entity_id"])
    op.create_index("idx_doc_entities_usage", "doc_entities", ["usage"])

    op.create_table(
        "doc_references",
        sa.Column("content_id", sa.Integer(), nullable=False),
        sa.Column("reference_id", sa.Integer(), nullable=False),
        sa.Column("context", sa.String(length=150), nullable=True),
        sa.Column("usage", sa.String(length=20), nullable=False, server_default="neutral"),
        sa.Column("mention_count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("ref", postgresql.JSONB(), nullable=True),
        sa.Column("attributes", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.CheckConstraint(
            "usage IN ('primary_topic', 'recommended', 'criticized', 'quoted', "
            "'evidence', 'example', 'comparison', 'background', 'neutral')",
            name="ck_doc_references_usage",
        ),
        sa.ForeignKeyConstraint(["content_id"], ["doc_content.content_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["reference_id"], ["references_catalog.reference_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("content_id", "reference_id"),
    )
    op.create_index("idx_doc_references_content", "doc_references", ["content_id"])
    op.create_index("idx_doc_references_reference", "doc_references", ["reference_id"])
    op.create_index("idx_doc_references_usage", "doc_references", ["usage"])

    op.create_table(
        "doc_participants",
        sa.Column("content_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=50), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["content_id"], ["doc_content.content_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("content_id", "name"),
    )
    op.create_index("idx_doc_participants_content", "doc_participants", ["content_id"])
    op.create_index("idx_doc_participants_name", "doc_participants", ["name"])

    op.create_table(
        "doc_topics",
        sa.Column("content_id", sa.Integer(), nullable=False),
        sa.Column("topic", sa.String(length=100), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["content_id"], ["doc_content.content_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("content_id", "topic"),
    )
    op.create_index("idx_doc_topics_content", "doc_topics", ["content_id"])
    op.create_index("idx_doc_topics_topic", "doc_topics", ["topic"])

    op.create_table(
        "doc_keywords",
        sa.Column("content_id", sa.Integer(), nullable=False),
        sa.Column("keyword", sa.String(length=100), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.ForeignKeyConstraint(["content_id"], ["doc_content.content_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("content_id", "keyword"),
    )
    op.create_index("idx_doc_keywords_content", "doc_keywords", ["content_id"])
    op.create_index("idx_doc_keywords_keyword", "doc_keywords", ["keyword"])

    op.execute(
        """
        CREATE VIEW monthly_openai_spending AS
        SELECT
            DATE_TRUNC('month', created_at) AS month,
            request_type,
            COUNT(*) AS request_count,
            SUM(total_tokens) AS total_tokens,
            SUM(cost) AS total_cost
        FROM openai_usage_log
        GROUP BY DATE_TRUNC('month', created_at), request_type
        ORDER BY month DESC, total_cost DESC
        """
    )

    media_sources = sa.table(
        "media_sources",
        sa.column("source_id", sa.String()),
        sa.column("source_name", sa.String()),
        sa.column("description", sa.Text()),
        sa.column("base_url", sa.String()),
        sa.column("supports_channels", sa.Boolean()),
        sa.column("supports_transcripts", sa.Boolean()),
    )
    op.bulk_insert(
        media_sources,
        [
            {
                "source_id": "youtube",
                "source_name": "YouTube",
                "description": "YouTube videos and channels",
                "base_url": "https://www.youtube.com",
                "supports_channels": True,
                "supports_transcripts": True,
            },
            {
                "source_id": "rss",
                "source_name": "RSS Feed",
                "description": "Generic RSS feeds",
                "base_url": None,
                "supports_channels": True,
                "supports_transcripts": False,
            },
            {
                "source_id": "podcast",
                "source_name": "Podcast",
                "description": "Podcast episodes",
                "base_url": None,
                "supports_channels": True,
                "supports_transcripts": True,
            },
        ],
    )

    doc_categories = sa.table(
        "doc_categories",
        sa.column("category_name", sa.String()),
        sa.column("is_default", sa.Boolean()),
    )
    op.bulk_insert(
        doc_categories,
        [
            {"category_name": "LDS Faith", "is_default": True},
            {"category_name": "Christian Faith", "is_default": True},
            {"category_name": "Self Help", "is_default": True},
            {"category_name": "construction/trade", "is_default": True},
            {"category_name": "product reviews", "is_default": True},
            {"category_name": "speculation", "is_default": True},
            {"category_name": "DIY", "is_default": True},
            {"category_name": "Educational", "is_default": True},
            {"category_name": "cooking", "is_default": True},
            {"category_name": "business", "is_default": True},
            {"category_name": "space", "is_default": True},
            {"category_name": "short term rental", "is_default": True},
            {"category_name": "science", "is_default": True},
            {"category_name": "news_politics", "is_default": True},
            {"category_name": "news_other", "is_default": True},
            {"category_name": "AI engineering", "is_default": True},
            {"category_name": "health", "is_default": True},
            {"category_name": "financial", "is_default": True},
            {"category_name": "technology", "is_default": True},
        ],
    )

    extraction_versions = sa.table(
        "extraction_versions",
        sa.column("version", sa.String()),
        sa.column("is_current", sa.Boolean()),
        sa.column("changes_summary", sa.Text()),
        sa.column("rollout_status", sa.String()),
    )
    op.bulk_insert(
        extraction_versions,
        [
            {
                "version": "1.0",
                "is_current": True,
                "changes_summary": "Initial Phase 1 extraction baseline.",
                "rollout_status": "production",
            }
        ],
    )


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS monthly_openai_spending")

    op.drop_index("idx_doc_keywords_keyword", table_name="doc_keywords")
    op.drop_index("idx_doc_keywords_content", table_name="doc_keywords")
    op.drop_table("doc_keywords")

    op.drop_index("idx_doc_topics_topic", table_name="doc_topics")
    op.drop_index("idx_doc_topics_content", table_name="doc_topics")
    op.drop_table("doc_topics")

    op.drop_index("idx_doc_participants_name", table_name="doc_participants")
    op.drop_index("idx_doc_participants_content", table_name="doc_participants")
    op.drop_table("doc_participants")

    op.drop_index("idx_doc_references_usage", table_name="doc_references")
    op.drop_index("idx_doc_references_reference", table_name="doc_references")
    op.drop_index("idx_doc_references_content", table_name="doc_references")
    op.drop_table("doc_references")

    op.drop_index("idx_doc_entities_usage", table_name="doc_entities")
    op.drop_index("idx_doc_entities_entity", table_name="doc_entities")
    op.drop_index("idx_doc_entities_content", table_name="doc_entities")
    op.drop_table("doc_entities")

    op.drop_index("idx_references_identifiers", table_name="references_catalog")
    op.drop_index("idx_references_label_normalized", table_name="references_catalog")
    op.drop_index("idx_references_type", table_name="references_catalog")
    op.drop_table("references_catalog")

    op.drop_index("idx_entities_usage_count", table_name="entities_catalog")
    op.drop_index("idx_entities_label_normalized", table_name="entities_catalog")
    op.drop_index("idx_entities_type", table_name="entities_catalog")
    op.drop_table("entities_catalog")

    op.drop_index("idx_extraction_versions_rollout", table_name="extraction_versions")
    op.drop_index("idx_extraction_versions_current", table_name="extraction_versions")
    op.drop_table("extraction_versions")

    op.drop_index("idx_metadata_values_normalized", table_name="metadata_values")
    op.drop_index("idx_metadata_values_verified", table_name="metadata_values")
    op.drop_index("idx_metadata_values_usage", table_name="metadata_values")
    op.drop_index("idx_metadata_values_schema", table_name="metadata_values")
    op.drop_table("metadata_values")

    op.drop_index("idx_metadata_schemas_openai", table_name="metadata_schemas")
    op.drop_index("idx_metadata_schemas_searchable", table_name="metadata_schemas")
    op.drop_index("idx_metadata_schemas_category", table_name="metadata_schemas")
    op.drop_index("idx_metadata_schemas_source", table_name="metadata_schemas")
    op.drop_index("idx_metadata_schemas_scope", table_name="metadata_schemas")
    op.drop_table("metadata_schemas")

    op.drop_index("idx_usage_log_content", table_name="openai_usage_log")
    op.drop_index("idx_usage_log_media", table_name="openai_usage_log")
    op.drop_index("idx_usage_log_created", table_name="openai_usage_log")
    op.drop_index("idx_usage_log_type", table_name="openai_usage_log")
    op.drop_table("openai_usage_log")

    op.drop_index("idx_sync_jobs_resume_after", table_name="sync_jobs")
    op.drop_index("idx_sync_jobs_resume_priority", table_name="sync_jobs")
    op.drop_index("idx_sync_jobs_resumable", table_name="sync_jobs")
    op.drop_index("idx_sync_jobs_created", table_name="sync_jobs")
    op.drop_index("idx_sync_jobs_channel", table_name="sync_jobs")
    op.drop_index("idx_sync_jobs_status", table_name="sync_jobs")
    op.drop_index("idx_sync_jobs_type", table_name="sync_jobs")
    op.drop_table("sync_jobs")

    op.drop_index("idx_doc_content_search", table_name="doc_content")
    op.drop_index("idx_doc_content_parent", table_name="doc_content")
    op.drop_index("idx_doc_content_chunk_type", table_name="doc_content")
    op.drop_index("idx_doc_content_media", table_name="doc_content")
    op.drop_table("doc_content")

    op.drop_index("idx_media_description_search", table_name="media")
    op.drop_index("idx_media_title_search", table_name="media")
    op.drop_index("idx_media_extraction_version", table_name="media")
    op.drop_index("idx_media_openai_sync_status", table_name="media")
    op.drop_index("idx_media_publish_date", table_name="media")
    op.drop_index("idx_media_category", table_name="media")
    op.drop_index("idx_media_source_type", table_name="media")
    op.drop_index("idx_media_channel", table_name="media")
    op.drop_index("idx_media_source", table_name="media")
    op.drop_table("media")

    op.drop_index("idx_categories_default", table_name="doc_categories")
    op.drop_table("doc_categories")

    op.drop_index("idx_channels_last_synced", table_name="channels")
    op.drop_index("idx_channels_source", table_name="channels")
    op.drop_table("channels")

    op.drop_table("media_sources")

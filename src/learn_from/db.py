from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from learn_from.config import Settings


def create_db_engine(settings: Settings) -> Engine:
    """Create the SQLAlchemy engine used by ingestion and operational flows."""
    return create_engine(settings.database_url, pool_pre_ping=True)


def check_database_connection(engine: Engine) -> None:
    """Run a minimal query to validate that the DB is reachable and usable."""
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))

from __future__ import annotations

from learn_from.config import Settings
from learn_from.db import check_database_connection, create_db_engine


def test_create_db_engine_and_check_with_sqlite() -> None:
    settings = Settings(_env_file=None, database_url="sqlite+pysqlite:///:memory:")
    engine = create_db_engine(settings)

    try:
        check_database_connection(engine)
    finally:
        engine.dispose()

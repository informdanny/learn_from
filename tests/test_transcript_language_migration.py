from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

from alembic.migration import MigrationContext
from alembic.operations import Operations
import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import DataError


_MIGRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "alembic/versions/20261106_0005_transcript_language_text.py"
)
_LONG_LANGUAGE = "en-Latn-US-fonipa"


def _migration():
    spec = importlib.util.spec_from_file_location("transcript_language_migration", _MIGRATION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migration_renders_only_language_type_change() -> None:
    migration = _migration()
    output = io.StringIO()
    context = MigrationContext.configure(
        dialect_name="postgresql", opts={"as_sql": True, "output_buffer": output}
    )
    with Operations.context(context):
        migration.upgrade()
    assert migration.down_revision == "20260711_0004"
    assert output.getvalue().strip() == "ALTER TABLE doc_content ALTER COLUMN language TYPE TEXT;"


@pytest.fixture(scope="module")
def local_postgres():
    """Use an isolated socket-only local cluster; never read the configured database."""
    initdb, pg_ctl = shutil.which("initdb"), shutil.which("pg_ctl")
    if not initdb or not pg_ctl:
        pytest.skip("Local PostgreSQL binaries are required for migration integration tests")
    with TemporaryDirectory(prefix="learn-from-language-", dir="/tmp") as directory:
        root = Path(directory)
        data, sockets = root / "data", root / "socket"
        sockets.mkdir()
        subprocess.run(
            [initdb, "-D", str(data), "-A", "trust", "--no-locale", "-E", "UTF8", "-U", "learn_from_test"],
            check=True, capture_output=True, timeout=30,
        )
        subprocess.run(
            [pg_ctl, "-D", str(data), "-l", str(root / "postgres.log"), "-o",
             f"-c listen_addresses='' -k {sockets}", "-w", "start"],
            check=True, capture_output=True, timeout=30,
        )
        engine = create_engine(URL.create(
            "postgresql+psycopg", username="learn_from_test", database="postgres",
            query={"host": str(sockets)},
        ))
        try:
            yield engine
        finally:
            engine.dispose()
            subprocess.run(
                [pg_ctl, "-D", str(data), "-m", "fast", "-w", "stop"],
                check=True, capture_output=True, timeout=30,
            )


@pytest.fixture
def language_table(local_postgres):
    with local_postgres.begin() as connection:
        connection.execute(text("CREATE TABLE doc_content (language varchar(10) NOT NULL DEFAULT 'en')"))
        connection.execute(text("INSERT INTO doc_content DEFAULT VALUES"))
    yield local_postgres
    with local_postgres.begin() as connection:
        connection.execute(text("DROP TABLE doc_content"))


def _apply(connection, direction: str) -> None:
    with Operations.context(MigrationContext.configure(connection)):
        getattr(_migration(), direction)()


def test_upgrade_preserves_existing_values_and_full_language_tag(language_table) -> None:
    with language_table.begin() as connection:
        _apply(connection, "upgrade")
        connection.execute(text("INSERT INTO doc_content VALUES (:language)"), {"language": _LONG_LANGUAGE})
        assert connection.execute(text("SELECT language FROM doc_content ORDER BY language")).scalars().all() == [
            "en", _LONG_LANGUAGE,
        ]
        column = inspect(connection).get_columns("doc_content")[0]
        assert str(column["type"]) == "TEXT"
        assert column["nullable"] is False
        assert column["default"] is not None


@pytest.mark.parametrize("language", [_LONG_LANGUAGE, "en" + " " * 9])
def test_downgrade_rejects_overlength_value_without_truncation(language_table, language) -> None:
    with language_table.begin() as connection:
        _apply(connection, "upgrade")
        connection.execute(text("INSERT INTO doc_content VALUES (:language)"), {"language": language})
    with pytest.raises(DataError, match="Cannot downgrade doc_content.language"):
        with language_table.begin() as connection:
            _apply(connection, "downgrade")
    with language_table.connect() as connection:
        assert str(inspect(connection).get_columns("doc_content")[0]["type"]) == "TEXT"
        assert language in connection.execute(text("SELECT language FROM doc_content")).scalars().all()


def test_downgrade_preserves_values_that_fit(language_table) -> None:
    with language_table.begin() as connection:
        _apply(connection, "upgrade")
        connection.execute(text("INSERT INTO doc_content VALUES ('zh-Hant-TW')"))
        _apply(connection, "downgrade")
        column = inspect(connection).get_columns("doc_content")[0]
        assert str(column["type"]) == "VARCHAR(10)"
        assert column["nullable"] is False
        assert column["default"] is not None
        assert connection.execute(text("SELECT language FROM doc_content ORDER BY language")).scalars().all() == [
            "en", "zh-Hant-TW",
        ]

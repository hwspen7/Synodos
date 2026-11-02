"""PostgreSQL connections and versioned, transactional schema migrations."""

from __future__ import annotations

from pathlib import Path


class PostgresDatabase:
    def __init__(self, dsn: str, pool_max: int = 5) -> None:
        try:
            from psycopg_pool import ConnectionPool
        except ImportError as exc:
            raise RuntimeError("PostgreSQL support requires: uv sync --extra postgres") from exc
        self.pool = ConnectionPool(
            conninfo=dsn, min_size=1, max_size=pool_max,
            timeout=10.0, open=False,
            kwargs={"connect_timeout": 5, "application_name": "meeting-foundation"},
        )
        self.pool.open(wait=True, timeout=15.0)
        try:
            self.migrate()
        except Exception:
            self.pool.close()
            raise

    def migrate(self) -> None:
        scripts = sorted((Path(__file__).parent / "migrations").glob("[0-9]*.sql"))
        with self.pool.connection() as conn:
            # Serialize DDL migration among concurrent startup processes.
            conn.execute("SELECT pg_advisory_xact_lock(938471223)")
            conn.execute("CREATE TABLE IF NOT EXISTS meeting_schema_migrations (version integer PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())")
            for script in scripts:
                version = int(script.stem.split("_", 1)[0])
                found = conn.execute("SELECT 1 FROM meeting_schema_migrations WHERE version = %s", (version,)).fetchone()
                if found:
                    continue
                for statement in script.read_text(encoding="utf-8").split(";"):
                    if statement.strip():
                        conn.execute(statement)
                conn.execute("INSERT INTO meeting_schema_migrations (version) VALUES (%s)", (version,))

    def close(self) -> None:
        self.pool.close()

    def check(self) -> bool:
        with self.pool.connection() as conn:
            return conn.execute("SELECT 1").fetchone()[0] == 1

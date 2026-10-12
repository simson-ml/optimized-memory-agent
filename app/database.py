import threading
from contextlib import contextmanager
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    memory_type TEXT NOT NULL DEFAULT 'conversation',
    importance REAL NOT NULL DEFAULT 0.3
        CHECK (importance >= 0 AND importance <= 1),
    source TEXT NOT NULL DEFAULT 'user_conversation',
    indexed BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_conversations_created_at
ON conversations(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_conversations_memory_type
ON conversations(memory_type);

CREATE INDEX IF NOT EXISTS idx_conversations_unindexed
ON conversations(id) WHERE indexed = FALSE;
"""

_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                conninfo = (
                    f"dbname={settings.dbname} "
                    f"user={settings.user} "
                    f"password={settings.password} "
                    f"host={settings.host} "
                    f"port={settings.port}"
                )
                _pool = ConnectionPool(
                    conninfo=conninfo,
                    min_size=settings.pg_pool_min,
                    max_size=settings.pg_pool_max,
                    timeout=settings.pg_pool_timeout,
                    kwargs={"row_factory": dict_row},
                    open=True,
                )
    return _pool


import atexit


def close_pool():
    global _pool
    with _pool_lock:
        if _pool is not None:
            try:
                _pool.close()
            except Exception:
                pass
            _pool = None


atexit.register(close_pool)


@contextmanager
def connect():
    """Yields a connection from the connection pool, auto-committing on normal exit."""
    pool = get_pool()
    with pool.connection() as conn:
        yield conn


def initialize():
    with connect() as c:
        with c.cursor() as cur:
            cur.execute(SCHEMA)


def insert_record(
    prompt,
    response,
    memory_type="conversation",
    importance=0.3,
    source="user_conversation",
):
    with connect() as c:
        with c.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversations
                (prompt, response, memory_type, importance, source)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING *
                """,
                (prompt, response, memory_type, importance, source),
            )
            return cur.fetchone()


def mark_indexed(i, v=True):
    with connect() as c:
        with c.cursor() as cur:
            cur.execute(
                "UPDATE conversations SET indexed=%s WHERE id=%s",
                (v, i),
            )


def mark_batch_indexed(ids: list[int], v=True):
    if not ids:
        return
    with connect() as c:
        with c.cursor() as cur:
            cur.execute(
                "UPDATE conversations SET indexed=%s WHERE id = ANY(%s)",
                (v, ids),
            )


def get_unindexed(limit=1000):
    with connect() as c:
        with c.cursor() as cur:
            cur.execute(
                """
                SELECT *
                FROM conversations
                WHERE indexed=FALSE
                ORDER BY id
                LIMIT %s
                """,
                (limit,),
            )
            return cur.fetchall()


def get_records_by_ids(ids: list[int]):
    if not ids:
        return []
    with connect() as c:
        with c.cursor() as cur:
            cur.execute(
                """
                SELECT *
                FROM conversations
                WHERE id = ANY(%s)
                ORDER BY id
                """,
                (ids,),
            )
            return cur.fetchall()


def get_all_record_ids() -> list[int]:
    with connect() as c:
        with c.cursor() as cur:
            cur.execute("SELECT id FROM conversations ORDER BY id")
            return [row["id"] for row in cur.fetchall()]


def delete_record(i):
    with connect() as c:
        with c.cursor() as cur:
            cur.execute(
                "DELETE FROM conversations WHERE id=%s RETURNING id",
                (i,),
            )
            return cur.fetchone()


def stats():
    with connect() as c:
        with c.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*) AS total,
                    COUNT(*) FILTER (WHERE indexed) AS indexed
                FROM conversations
                """)
            return cur.fetchone()

import psycopg
from psycopg.rows import dict_row
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

-- Migrations for older database versions
ALTER TABLE conversations
ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ
DEFAULT CURRENT_TIMESTAMP;

ALTER TABLE conversations
ADD COLUMN IF NOT EXISTS prompt TEXT;

ALTER TABLE conversations
ADD COLUMN IF NOT EXISTS response TEXT;

ALTER TABLE conversations
ADD COLUMN IF NOT EXISTS memory_type TEXT
DEFAULT 'conversation';

ALTER TABLE conversations
ADD COLUMN IF NOT EXISTS importance REAL
DEFAULT 0.3;

ALTER TABLE conversations
ADD COLUMN IF NOT EXISTS source TEXT
DEFAULT 'user_conversation';

ALTER TABLE conversations
ADD COLUMN IF NOT EXISTS indexed BOOLEAN
DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_conversations_indexed
ON conversations(indexed);

CREATE INDEX IF NOT EXISTS idx_conversations_created_at
ON conversations(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_conversations_memory_type
ON conversations(memory_type);
"""


def connect():
    return psycopg.connect(
        dbname=settings.dbname,
        user=settings.user,
        password=settings.password,
        host=settings.host,
        port=settings.port,
        row_factory=dict_row,
    )


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

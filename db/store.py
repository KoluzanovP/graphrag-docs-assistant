"""psycopg2 helpers for persisting conversations, documents, and feedback."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import psycopg2
import psycopg2.extras

from config import settings

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


@contextmanager
def get_connection() -> Iterator["psycopg2.extensions.connection"]:
    """Yield a psycopg2 connection, committing on success and closing always."""
    conn = psycopg2.connect(settings.postgres_dsn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_schema() -> None:
    """Create tables from ``schema.sql`` if they do not yet exist."""
    sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql)
    logger.info("PostgreSQL schema initialised")


def create_session(title: str | None = None) -> str:
    """Insert a new chat session and return its UUID."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO chat_sessions (title) VALUES (%s) RETURNING id",
            (title,),
        )
        return str(cur.fetchone()[0])


def add_message(
    session_id: str,
    role: str,
    content: str,
    citations: list[str] | None = None,
) -> int:
    """Insert a message into a session and return its primary key."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO messages (session_id, role, content, citations)
            VALUES (%s, %s, %s, %s) RETURNING id
            """,
            (session_id, role, content, json.dumps(citations or [])),
        )
        return int(cur.fetchone()[0])


def get_messages(session_id: str) -> list[dict[str, Any]]:
    """Return all messages for a session ordered chronologically."""
    with get_connection() as conn, conn.cursor(
        cursor_factory=psycopg2.extras.RealDictCursor
    ) as cur:
        cur.execute(
            """
            SELECT id, role, content, citations, created_at
            FROM messages
            WHERE session_id = %s
            ORDER BY id ASC
            """,
            (session_id,),
        )
        return [dict(row) for row in cur.fetchall()]


def record_document(
    source: str,
    file_path: str | None,
    num_chunks: int,
    num_nodes: int,
    num_edges: int,
) -> None:
    """Upsert ingestion metadata for a source document."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (source, file_path, num_chunks, num_nodes, num_edges)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (source) DO UPDATE SET
                file_path = EXCLUDED.file_path,
                num_chunks = EXCLUDED.num_chunks,
                num_nodes = EXCLUDED.num_nodes,
                num_edges = EXCLUDED.num_edges,
                ingested_at = now()
            """,
            (source, file_path, num_chunks, num_nodes, num_edges),
        )


def add_feedback(message_id: int, rating: int, comment: str | None = None) -> int:
    """Record thumbs-up (1) / thumbs-down (-1) feedback for a message."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO feedback (message_id, rating, comment) VALUES (%s, %s, %s) RETURNING id",
            (message_id, rating, comment),
        )
        return int(cur.fetchone()[0])

"""PostgreSQL data access layer for enrolled face embeddings.

Why this file exists:
- Isolate SQL operations from business logic.
- Use connection pooling and transactions for production readiness.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg2
from psycopg2.pool import SimpleConnectionPool

from app.logger import get_logger
from config import settings

LOGGER = get_logger(__name__)

_pool: SimpleConnectionPool | None = None


def _get_pool() -> SimpleConnectionPool:
    """Lazily create and return a shared DB connection pool."""
    global _pool
    if _pool is None:
        _pool = SimpleConnectionPool(
            minconn=settings.db_min_connections,
            maxconn=settings.db_max_connections,
            dbname=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
            host=settings.db_host,
            port=settings.db_port,
        )
        LOGGER.info(
            "Initialized DB pool host=%s db=%s min=%d max=%d",
            settings.db_host,
            settings.db_name,
            settings.db_min_connections,
            settings.db_max_connections,
        )
    return _pool


@contextmanager
def _cursor() -> Iterator[psycopg2.extensions.cursor]:
    """Yield a transactional cursor from the shared connection pool."""
    conn = _get_pool().getconn()
    try:
        with conn.cursor() as cur:
            yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _get_pool().putconn(conn)


def ensure_schema() -> None:
    """Create required tables if they do not already exist."""
    with _cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS employees (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                embedding DOUBLE PRECISION[] NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            );
            """
        )
    LOGGER.info("Database schema ensured")


def insert_user(name: str, embedding: list[float]) -> None:
    """Insert one user record with embedding vector."""
    with _cursor() as cur:
        cur.execute(
            "INSERT INTO employees (name, embedding) VALUES (%s, %s)",
            (name, embedding),
        )


def get_all_users() -> list[tuple[str, list[float]]]:
    """Fetch all registered users and embeddings."""
    with _cursor() as cur:
        cur.execute("SELECT name, embedding FROM employees")
        rows = cur.fetchall()
    return rows


def close_pool() -> None:
    """Close all pooled DB connections."""
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None
        LOGGER.info("Closed DB pool")

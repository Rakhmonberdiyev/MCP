"""PostgreSQL-backed multi-user memory event log.

Replaces the single-file mem0_history.db (SQLite) with a proper server database
so multiple users and processes can write/query history concurrently.

Schema
------
memory_events
  id          — auto-increment PK
  user_id     — who owns this memory
  memory_id   — Mem0's internal UUID for the fact (nullable for legacy)
  event       — ADD | UPDATE | DELETE
  old_memory  — previous fact text (populated for UPDATE)
  new_memory  — current fact text (ADD / UPDATE); deleted fact for DELETE
  created_at  — UTC timestamp
"""

import asyncpg
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Read connection URL from the central config so all store settings live in one place
from config import MEM0_CONFIG
DATABASE_URL: str = MEM0_CONFIG["history_store"]["config"]["url"]

_pool: Optional[asyncpg.Pool] = None
_pool_lock = asyncio.Lock()


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is not None:
        return _pool
    async with _pool_lock:
        if _pool is None:
            try:
                _pool = await asyncpg.create_pool(
                    DATABASE_URL,
                    min_size=1,
                    max_size=10,
                    command_timeout=10,
                )
                await _init_schema(_pool)
            except Exception as exc:
                logger.warning(f"[db] PostgreSQL unavailable ({exc}) — history disabled")
                _pool = None
                raise
    return _pool


async def _init_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_events (
                id          BIGSERIAL PRIMARY KEY,
                user_id     TEXT        NOT NULL,
                memory_id   TEXT,
                event       TEXT        NOT NULL,
                old_memory  TEXT,
                new_memory  TEXT,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_mem_events_user_time
                ON memory_events (user_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_mem_events_event
                ON memory_events (user_id, event);
        """)


# ── Write ──────────────────────────────────────────────────────────────────────

async def write_events(user_id: str, results: list[dict]) -> None:
    """
    Persist ADD / UPDATE / DELETE events returned by ltm_memory.add().

    Each result dict is expected to contain at minimum:
      {"event": "ADD"|"UPDATE"|"DELETE", "memory": "<fact text>", "id": "<uuid>"}
    For UPDATE, the caller should also include {"old_memory": "<old text>"}.
    """
    rows = [
        (
            user_id,
            r.get("id"),
            r["event"],
            r.get("old_memory"),   # None for ADD; populated for UPDATE
            r.get("memory"),       # new / current fact text
        )
        for r in results
        if r.get("event") in ("ADD", "UPDATE", "DELETE")
    ]
    if not rows:
        return

    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO memory_events
                    (user_id, memory_id, event, old_memory, new_memory)
                VALUES ($1, $2, $3, $4, $5)
                """,
                rows,
            )
    except Exception as exc:
        logger.warning(f"[db] write_events failed (non-fatal): {exc}")


# ── Read ───────────────────────────────────────────────────────────────────────

def _parse_iso(since_iso: str) -> datetime:
    """Convert an ISO-8601 string to a timezone-aware datetime for asyncpg."""
    dt = datetime.fromisoformat(since_iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


async def query_events(
    user_id: str,
    since_iso: str,
    event_types: tuple[str, ...] = ("ADD", "UPDATE", "DELETE"),
) -> list[dict]:
    """
    Return all events for user_id that occurred on or after since_iso (ISO-8601).
    Results are ordered oldest-first, matching the previous SQLite query behaviour.
    """
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT event, old_memory, new_memory, created_at, memory_id
                FROM   memory_events
                WHERE  user_id    = $1
                  AND  created_at >= $2
                  AND  event      = ANY($3)
                ORDER  BY created_at ASC
                """,
                user_id,
                _parse_iso(since_iso),
                list(event_types),
            )
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning(f"[db] query_events failed: {exc}")
        return []


async def count_events(user_id: str, since_iso: str) -> int:
    """Quick count — used by the eval polling loop."""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT COUNT(*) FROM memory_events "
                "WHERE user_id = $1 AND created_at >= $2",
                user_id,
                _parse_iso(since_iso),
            )
    except Exception:
        return 0

"""PostgreSQL access: structured SNB / Curia Vista data, ETag cube state, ingestion job state.

Per architecture spec §3.3, numeric time series and structured records are queried
exactly via SQL — never embedded/semantically searched — so this module, not the vector
store, is the source of truth for that data.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import psycopg
from psycopg.rows import dict_row

from src.config import get_settings

logger = logging.getLogger("swiss_legal_ai.storage.postgres")

if sys.platform == "win32":
    # psycopg's async mode requires a selector-based event loop; Windows defaults to
    # ProactorEventLoop, under which every async connection attempt raises
    # psycopg.InterfaceError. Set at import time so any entrypoint using this module
    # (scripts, the API, notebooks) gets a working event loop policy for free.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS snb_cube_state (
    cube_id TEXT PRIMARY KEY,
    etag TEXT,
    last_fetched_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS snb_observations (
    id BIGSERIAL PRIMARY KEY,
    cube_id TEXT NOT NULL,
    dimensions JSONB NOT NULL,
    observation_date TEXT,
    value DOUBLE PRECISION,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (cube_id, dimensions, observation_date)
);
CREATE INDEX IF NOT EXISTS idx_snb_observations_cube ON snb_observations (cube_id);

CREATE TABLE IF NOT EXISTS curia_vista_business (
    id TEXT PRIMARY KEY,
    short_number TEXT,
    title TEXT,
    summary TEXT,
    status TEXT,
    submission_date TEXT,
    business_type TEXT,
    related_law_references JSONB,
    source_url TEXT,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id BIGSERIAL PRIMARY KEY,
    job_name TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    details JSONB
);

-- Cross-references between law articles (e.g. "art. 21 LIFD rimanda all'art. 210 CO"),
-- extracted by processing.reference_extractor and used for bounded 1-hop retrieval
-- expansion in orchestration.query_engine. `to_systematic_number`/`to_article` are
-- nullable: a reference whose law abbreviation doesn't resolve to a known/ingested
-- prefix is still stored (raw text only), so it can be re-resolved later without
-- re-parsing the source text.
CREATE TABLE IF NOT EXISTS article_references (
    id BIGSERIAL PRIMARY KEY,
    from_systematic_number TEXT NOT NULL,
    from_article TEXT NOT NULL,
    to_systematic_number TEXT,
    to_article TEXT,
    raw_reference_text TEXT NOT NULL,
    language TEXT NOT NULL,
    source_url TEXT NOT NULL,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (from_systematic_number, from_article, to_systematic_number, to_article, raw_reference_text)
);
CREATE INDEX IF NOT EXISTS idx_article_references_from
    ON article_references (from_systematic_number, from_article);
CREATE INDEX IF NOT EXISTS idx_article_references_to
    ON article_references (to_systematic_number, to_article);
"""


@asynccontextmanager
async def get_connection() -> AsyncIterator[psycopg.AsyncConnection]:
    settings = get_settings()
    conn = await psycopg.AsyncConnection.connect(settings.postgres_dsn, row_factory=dict_row)
    try:
        yield conn
    finally:
        await conn.close()


class PostgresStore:
    def __init__(self, dsn: str | None = None) -> None:
        self._dsn = dsn or get_settings().postgres_dsn

    async def _connect(self) -> psycopg.AsyncConnection:
        return await psycopg.AsyncConnection.connect(self._dsn, row_factory=dict_row)

    async def init_schema(self) -> None:
        """Create tables if missing. Guarded by a session-level advisory lock: plain
        'CREATE TABLE IF NOT EXISTS' is not safe against two connections racing to create
        the same table concurrently (e.g. SNB and Curia Vista ingestion steps both calling
        this at startup) and can raise a UniqueViolation on pg_type."""
        # Arbitrary fixed lock key, unique to this schema-init operation.
        lock_key = 727386412
        async with await self._connect() as conn:
            await conn.execute("SELECT pg_advisory_lock(%s)", (lock_key,))
            try:
                await conn.execute(SCHEMA_SQL)
                await conn.commit()
            finally:
                await conn.execute("SELECT pg_advisory_unlock(%s)", (lock_key,))
                await conn.commit()
        logger.info("PostgreSQL schema initialized")

    # --- SNB cube ETag / lastUpdate state -----------------------------------

    async def get_cube_etag(self, cube_id: str) -> str | None:
        async with await self._connect() as conn:
            row = await (
                await conn.execute("SELECT etag FROM snb_cube_state WHERE cube_id = %s", (cube_id,))
            ).fetchone()
            return row["etag"] if row else None

    async def set_cube_etag(self, cube_id: str, etag: str | None) -> None:
        async with await self._connect() as conn:
            await conn.execute(
                """
                INSERT INTO snb_cube_state (cube_id, etag, last_fetched_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (cube_id) DO UPDATE
                SET etag = EXCLUDED.etag, last_fetched_at = EXCLUDED.last_fetched_at
                """,
                (cube_id, etag, datetime.utcnow()),
            )
            await conn.commit()

    # --- SNB observations -----------------------------------------------------

    async def upsert_snb_observations(self, cube_id: str, records: list[dict[str, Any]]) -> int:
        """Each record: {"dimensions": {...}, "observation_date": str, "value": float}."""
        if not records:
            return 0
        async with await self._connect() as conn:
            async with conn.cursor() as cur:
                for rec in records:
                    await cur.execute(
                        """
                        INSERT INTO snb_observations (cube_id, dimensions, observation_date, value)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (cube_id, dimensions, observation_date) DO UPDATE
                        SET value = EXCLUDED.value
                        """,
                        (
                            cube_id,
                            json.dumps(rec["dimensions"], sort_keys=True),
                            rec.get("observation_date"),
                            rec.get("value"),
                        ),
                    )
            await conn.commit()
        logger.info("Upserted %d observations for cube %s", len(records), cube_id)
        return len(records)

    # --- Curia Vista business --------------------------------------------------

    async def upsert_curia_vista_business(self, items: list[dict[str, Any]]) -> int:
        if not items:
            return 0
        async with await self._connect() as conn:
            async with conn.cursor() as cur:
                for item in items:
                    await cur.execute(
                        """
                        INSERT INTO curia_vista_business
                            (id, short_number, title, summary, status, submission_date,
                             business_type, related_law_references, source_url)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            short_number = EXCLUDED.short_number,
                            title = EXCLUDED.title,
                            summary = EXCLUDED.summary,
                            status = EXCLUDED.status,
                            submission_date = EXCLUDED.submission_date,
                            business_type = EXCLUDED.business_type,
                            related_law_references = EXCLUDED.related_law_references,
                            source_url = EXCLUDED.source_url
                        """,
                        (
                            item["id"],
                            item.get("short_number"),
                            item.get("title"),
                            item.get("summary"),
                            item.get("status"),
                            item.get("submission_date"),
                            item.get("business_type"),
                            json.dumps(item.get("related_law_references", [])),
                            item.get("source_url"),
                        ),
                    )
            await conn.commit()
        logger.info("Upserted %d Curia Vista business items", len(items))
        return len(items)

    # --- Ingestion job tracking --------------------------------------------------

    async def start_job(self, job_name: str) -> int:
        async with await self._connect() as conn:
            row = await (
                await conn.execute(
                    """
                    INSERT INTO ingestion_jobs (job_name, status)
                    VALUES (%s, 'running')
                    RETURNING id
                    """,
                    (job_name,),
                )
            ).fetchone()
            await conn.commit()
            return row["id"]

    async def finish_job(
        self, job_id: int, status: str, details: dict[str, Any] | None = None
    ) -> None:
        async with await self._connect() as conn:
            await conn.execute(
                """
                UPDATE ingestion_jobs
                SET status = %s, finished_at = now(), details = %s
                WHERE id = %s
                """,
                (status, json.dumps(details or {}), job_id),
            )
            await conn.commit()

    async def update_job_details(self, job_id: int, details: dict[str, Any]) -> None:
        """Update `details` on a still-`running` job without touching `status`/
        `finished_at` — used to report incremental per-prefix progress on long bulk
        ingestion jobs (see `/ingest/fedlex/bulk`), as opposed to `finish_job`, which is
        only called once at the very end."""
        async with await self._connect() as conn:
            await conn.execute(
                "UPDATE ingestion_jobs SET details = %s WHERE id = %s",
                (json.dumps(details), job_id),
            )
            await conn.commit()

    async def get_job(self, job_id: int) -> dict[str, Any] | None:
        async with await self._connect() as conn:
            row = await (
                await conn.execute(
                    "SELECT id, job_name, status, started_at, finished_at, details "
                    "FROM ingestion_jobs WHERE id = %s",
                    (job_id,),
                )
            ).fetchone()
            return dict(row) if row else None

    # --- Article cross-references -----------------------------------------------

    async def upsert_article_references(self, refs: list[dict[str, Any]]) -> int:
        """Replace all stored references *from* each (from_systematic_number,
        from_article) pair present in `refs` with the freshly extracted set.

        Delete-then-insert per from-article, rather than an `ON CONFLICT` upsert, because
        `to_systematic_number`/`to_article` are nullable (unresolved references) and
        PostgreSQL treats NULL as distinct from NULL for uniqueness purposes — an
        `ON CONFLICT` on a unique constraint containing those columns would not dedupe
        re-extracted unresolved references, silently accumulating duplicates across
        ingestion runs. Re-extraction is idempotent this way: re-running ingestion for an
        unchanged article yields the same reference rows.

        `refs` is deduped in Python before inserting: a long article split into several
        paragraph chunks (`processing.chunker`'s `paragraph_index` splitting) can have the
        *same* reference re-extracted once per split chunk, since
        `processing.reference_extractor.extract_references` only dedupes within a single
        chunk's text. Two identical, fully-resolved rows in the same batch would otherwise
        hit the real (non-NULL) unique constraint on the second INSERT — this dedup avoids
        that regardless of where the duplicate came from.
        """
        if not refs:
            return 0
        deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
        for ref in refs:
            key = (
                ref["from_systematic_number"],
                ref["from_article"],
                ref.get("to_systematic_number"),
                ref.get("to_article"),
                ref["raw_reference_text"],
            )
            deduped[key] = ref
        refs = list(deduped.values())

        from_pairs = {(r["from_systematic_number"], r["from_article"]) for r in refs}
        async with await self._connect() as conn:
            async with conn.cursor() as cur:
                for from_systematic_number, from_article in from_pairs:
                    await cur.execute(
                        "DELETE FROM article_references "
                        "WHERE from_systematic_number = %s AND from_article = %s",
                        (from_systematic_number, from_article),
                    )
                for ref in refs:
                    await cur.execute(
                        """
                        INSERT INTO article_references
                            (from_systematic_number, from_article, to_systematic_number,
                             to_article, raw_reference_text, language, source_url)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            ref["from_systematic_number"],
                            ref["from_article"],
                            ref.get("to_systematic_number"),
                            ref.get("to_article"),
                            ref["raw_reference_text"],
                            ref["language"],
                            ref["source_url"],
                        ),
                    )
            await conn.commit()
        logger.info("Upserted %d article reference(s)", len(refs))
        return len(refs)

    async def get_outgoing_references(
        self, systematic_number: str, article: str
    ) -> list[dict[str, Any]]:
        """Resolved outgoing references (to_systematic_number IS NOT NULL) from one
        article — used for 1-hop retrieval expansion in `orchestration.query_engine`."""
        async with await self._connect() as conn:
            rows = await (
                await conn.execute(
                    """
                    SELECT to_systematic_number, to_article
                    FROM article_references
                    WHERE from_systematic_number = %s AND from_article = %s
                      AND to_systematic_number IS NOT NULL
                    """,
                    (systematic_number, article),
                )
            ).fetchall()
            return [dict(row) for row in rows]

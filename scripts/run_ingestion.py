#!/usr/bin/env python
"""Bounded, end-to-end manual ingestion for local verification (architecture spec §6.13 / §9).

Runs three small, independent ingestion steps and reports what happened for each:
1. Fedlex: a handful of consolidated acts under FEDLEX_SYSTEMATIC_PREFIX -> chunked,
   embedded, and upserted into Qdrant.
2. SNB: one or two cubes from SNB_TEST_CUBES -> parsed and upserted into PostgreSQL
   (ETag-aware, so a second run should mostly no-op).
3. Curia Vista: one month of Business items starting at CURIA_VISTA_DATE_FROM -> upserted
   into PostgreSQL.

Each step is independently try/excepted so a failure in one (e.g. no network access to a
given source) doesn't prevent verifying the others. This script does not raise on partial
failure; check the printed summary.
"""
from __future__ import annotations

import asyncio
import sys
import traceback
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path

import structlog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings  # noqa: E402
from src.embedding.embed_service import get_embed_service  # noqa: E402
from src.ingestion.curia_vista_client import CuriaVistaClient  # noqa: E402
from src.ingestion.fedlex_client import FedlexClient  # noqa: E402
from src.ingestion.snb_client import SNBClient  # noqa: E402
from src.logging_config import configure_logging  # noqa: E402
from src.processing.chunker import chunk_act_text, chunk_business_item  # noqa: E402
from src.processing.metadata import build_metadata  # noqa: E402
from src.processing.reference_extractor import extract_references  # noqa: E402
from src.storage.postgres_store import PostgresStore  # noqa: E402
from src.storage.qdrant_store import ChunkRecord, QdrantStore  # noqa: E402

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"

MAX_FEDLEX_ACTS_PER_PREFIX = 3


async def ingest_fedlex() -> str:
    """Ingest every SR/RS prefix configured in `settings.fedlex_systematic_prefix_list`
    (e.g. tax law, civil code, criminal code, ... — see config.py), accumulating chunks
    across all prefixes before a single batched embed + upsert call, which is cheaper
    than embedding once per prefix."""
    settings = get_settings()
    client = FedlexClient()
    embed_service = get_embed_service()
    store = QdrantStore()
    pg_store = PostgresStore()
    await pg_store.init_schema()

    chunk_payloads: list[tuple[str, dict]] = []
    all_refs: list[dict] = []
    per_prefix_summary: list[str] = []
    for prefix in settings.fedlex_systematic_prefix_list:
        # Language preference: Italian first, then French, then German (an act missing
        # an IT title/text falls back down the list — see
        # FedlexClient.DEFAULT_LANGUAGE_PRIORITY).
        acts = await asyncio.to_thread(
            client.list_consolidated_acts_by_prefix_preferred, prefix
        )
        if not acts:
            per_prefix_summary.append(f"{prefix}: no acts found")
            continue

        acts_processed = 0
        for act_summary in acts[:MAX_FEDLEX_ACTS_PER_PREFIX]:
            try:
                doc = await asyncio.to_thread(
                    client.fetch_act_preferred, act_summary.act_uri, include_text=True
                )
            except Exception:
                # One bad act (network hiccup, no HTML manifestation, parsing edge
                # case) must not sink the whole multi-prefix run and discard every
                # chunk already collected for other prefixes — log and move on.
                structlog.get_logger("swiss_legal_ai.scripts.run_ingestion").warning(
                    "act_fetch_failed", prefix=prefix, act_uri=act_summary.act_uri, exc_info=True
                )
                continue
            acts_processed += 1
            for chunk in chunk_act_text(doc.raw_text):
                meta = build_metadata(
                    chunk,
                    source="fedlex",
                    language=doc.language,
                    source_url=doc.source_url,
                    law_short_name=doc.title,
                    systematic_number=doc.systematic_number,
                    valid_from=doc.valid_from,
                    valid_to=doc.valid_to,
                )
                chunk_payloads.append((chunk.text, meta.model_dump()))
                if chunk.article:
                    all_refs.extend(
                        asdict(ref)
                        for ref in extract_references(
                            chunk.text,
                            from_systematic_number=doc.systematic_number,
                            from_article=chunk.article,
                            language=doc.language,
                            source_url=doc.source_url,
                        )
                    )
        per_prefix_summary.append(f"{prefix}: {acts_processed} act(s)")

    if all_refs:
        await pg_store.upsert_article_references(all_refs)

    if not chunk_payloads:
        return "Fedlex: " + "; ".join(per_prefix_summary) + " — 0 chunks produced"

    embedded = embed_service.embed_batch([text for text, _ in chunk_payloads])
    store.create_collection(dense_dim=len(embedded[0].dense))
    records = [
        ChunkRecord(text=text, metadata=meta, embedding=embedding)
        for (text, meta), embedding in zip(chunk_payloads, embedded)
    ]
    upserted = store.upsert_chunks(records)
    return (
        "Fedlex: " + "; ".join(per_prefix_summary)
        + f"; {upserted} chunk(s) upserted into Qdrant, {len(all_refs)} reference(s) extracted"
    )


async def ingest_snb() -> str:
    settings = get_settings()
    pg_store = PostgresStore()
    await pg_store.init_schema()

    summaries = []
    async with SNBClient() as client:
        for cube_id in settings.snb_test_cubes.split(","):
            cube_id = cube_id.strip()
            if not cube_id:
                continue
            result = await client.sync_cube(cube_id, pg_store)
            summaries.append(
                f"{cube_id} (modified={result.modified}, {result.records_upserted} records)"
            )
    return "SNB: " + ", ".join(summaries) if summaries else "SNB: no cubes configured"


async def ingest_curia_vista() -> str:
    """Curia Vista business items are dual-written (architecture spec §3.1/§3.3):
    the full raw fields go to PostgreSQL (structured, exact lookup), and one synthetic
    summary chunk per item is embedded and upserted into Qdrant, using the same metadata
    schema as Fedlex, so hybrid search can filter/retrieve across both sources.
    """
    settings = get_settings()
    pg_store = PostgresStore()
    await pg_store.init_schema()
    embed_service = get_embed_service()
    qdrant_store = QdrantStore()

    date_from = date.fromisoformat(settings.curia_vista_date_from)
    date_to = date_from + timedelta(days=30)

    async with CuriaVistaClient() as client:
        items = await client.fetch_business(date_from, date_to)

    pg_upserted = await pg_store.upsert_curia_vista_business(
        [
            {
                "id": item.id,
                "short_number": item.short_number,
                "title": item.title,
                "summary": item.summary,
                "status": item.status,
                "submission_date": item.submission_date,
                "business_type": item.business_type,
                "related_law_references": item.related_law_references,
                "source_url": item.source_url,
            }
            for item in items
        ]
    )

    chunk_payloads: list[tuple[str, dict]] = []
    for item in items:
        chunk = chunk_business_item(item.short_number, item.title, item.status, item.summary)
        meta = build_metadata(
            chunk,
            source="curia_vista",
            language=item.language,
            source_url=item.source_url,
            law_short_name=item.business_type,
            valid_from=item.submission_date,
        )
        chunk_payloads.append((chunk.text, meta.model_dump()))

    qdrant_upserted = 0
    if chunk_payloads:
        embedded = embed_service.embed_batch([text for text, _ in chunk_payloads])
        qdrant_store.create_collection(dense_dim=len(embedded[0].dense))
        records = [
            ChunkRecord(text=text, metadata=meta, embedding=embedding)
            for (text, meta), embedding in zip(chunk_payloads, embedded)
        ]
        qdrant_upserted = qdrant_store.upsert_chunks(records)

    return (
        f"Curia Vista: {pg_upserted} business item(s) upserted into Postgres, "
        f"{qdrant_upserted} chunk(s) upserted into Qdrant, for {date_from}..{date_to}"
    )


async def run_step(name: str, coro) -> str:
    try:
        return await coro
    except Exception:  # noqa: BLE001 - report and keep going, don't crash the whole run
        # Log the full traceback (not just print it) so it lands in the log file too —
        # otherwise a failure is only ever visible in the console window that ran this
        # script, not in logs/ingestion_*.log, which makes after-the-fact diagnosis
        # (e.g. from a double-clicked start_ingestion.bat window that's already closed)
        # impossible without re-running.
        structlog.get_logger("swiss_legal_ai.scripts.run_ingestion").error(
            "step_failed", step=name, traceback=traceback.format_exc()
        )
        return f"{name}: FAILED\n{traceback.format_exc(limit=3)}"


async def main() -> None:
    log_file = LOGS_DIR / f"ingestion_{datetime.now():%Y%m%d_%H%M%S}.log"
    configure_logging(json_output=False, log_file=log_file)
    print(f"Logging to: {log_file}")

    results = await asyncio.gather(
        run_step("Fedlex", ingest_fedlex()),
        run_step("SNB", ingest_snb()),
        run_step("Curia Vista", ingest_curia_vista()),
    )
    print("\n=== Ingestion summary ===")
    for line in results:
        print(f"- {line}")
    structlog.get_logger("swiss_legal_ai.scripts.run_ingestion").info(
        "ingestion_summary", results=results
    )
    print(f"\nFull log: {log_file}")


if __name__ == "__main__":
    asyncio.run(main())

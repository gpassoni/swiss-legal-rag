"""Minimal FastAPI app: POST /query plus manual ingestion trigger endpoints for testing."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from datetime import date, timedelta
from functools import lru_cache
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from src.config import get_settings
from src.embedding.embed_service import EmbedService, get_embed_service
from src.ingestion.curia_vista_client import CuriaVistaClient
from src.ingestion.fedlex_client import FedlexClient
from src.ingestion.snb_client import SNBClient
from src.logging_config import configure_logging
from src.orchestration.query_engine import QueryEngine, build_llm_client_from_settings
from src.processing.chunker import chunk_act_text, chunk_business_item
from src.processing.metadata import build_metadata
from src.processing.reference_extractor import extract_references
from src.retrieval.reranker import Reranker, get_reranker
from src.storage.postgres_store import PostgresStore
from src.storage.qdrant_store import ChunkRecord, QdrantStore

configure_logging(json_output=True)
logger = logging.getLogger("swiss_legal_ai.api")

app = FastAPI(
    title="Swiss Legal & Financial AI Assistant — Phase 1",
    description="Ingestion pipeline + RAG core over Swiss federal law, parliamentary "
    "activity, and monetary/banking data. Informational only, not legal or tax advice.",
    version="0.1.0",
)


@lru_cache
def _store() -> QdrantStore:
    return QdrantStore()


class QueryRequest(BaseModel):
    question: str
    filters: dict[str, Any] | None = None
    top_k: int = 8


class QueryResponse(BaseModel):
    answer: str
    verified_sources: list[dict[str, str]]
    unverified_citation_count: int
    retrieved_chunk_count: int


class IngestFedlexRequest(BaseModel):
    systematic_prefix: str | None = None
    language: str | None = None  # None = use the IT -> FR -> DE preference order
    max_acts: int = 5


class IngestFedlexResponse(BaseModel):
    acts_processed: int
    chunks_upserted: int


class IngestFedlexBulkRequest(BaseModel):
    prefixes: list[str] | None = None  # None = use settings.fedlex_systematic_prefix_list
    max_acts_per_prefix: int = 5


class IngestFedlexBulkResponse(BaseModel):
    job_id: int


class IngestJobStatusResponse(BaseModel):
    id: int
    job_name: str
    status: str
    started_at: str
    finished_at: str | None
    details: dict[str, Any]


class IngestSnbRequest(BaseModel):
    cube_ids: list[str] | None = None
    lang: str = "en"


class IngestSnbResponse(BaseModel):
    results: list[dict[str, Any]]


class IngestCuriaVistaRequest(BaseModel):
    date_from: str | None = None  # ISO date; defaults to settings.curia_vista_date_from
    date_to: str | None = None  # ISO date; defaults to date_from + 30 days


class IngestCuriaVistaResponse(BaseModel):
    items_upserted_postgres: int
    chunks_upserted_qdrant: int


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    embed_service: EmbedService = get_embed_service()
    reranker: Reranker = get_reranker()
    try:
        llm_client = build_llm_client_from_settings()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    engine = QueryEngine(embed_service, _store(), reranker, llm_client, PostgresStore())
    result = await engine.answer(request.question, filters=request.filters, top_k=request.top_k)
    return QueryResponse(
        answer=result.answer,
        verified_sources=[
            {"article": c.article, "source_url": c.source_url} for c in result.verified_sources
        ],
        unverified_citation_count=result.unverified_citation_count,
        retrieved_chunk_count=result.retrieved_chunk_count,
    )


@app.post("/ingest/fedlex", response_model=IngestFedlexResponse)
async def ingest_fedlex(request: IngestFedlexRequest) -> IngestFedlexResponse:
    """Bounded, synchronous-triggered ingestion of a Fedlex systematic-number prefix, for
    local testing. Not a production scheduler — see scripts/run_ingestion.py."""
    settings = get_settings()
    prefix = request.systematic_prefix or settings.fedlex_systematic_prefix
    embed_service = get_embed_service()
    store = _store()

    def _fetch_and_chunk() -> tuple[int, list[tuple[str, dict]], list[dict]]:
        client = FedlexClient()
        if request.language:
            acts = client.list_consolidated_acts_by_prefix(prefix, language=request.language)
        else:
            acts = client.list_consolidated_acts_by_prefix_preferred(prefix)
        chunk_payloads: list[tuple[str, dict]] = []
        refs: list[dict] = []
        acts_processed = 0
        for act_summary in acts[: request.max_acts]:
            try:
                if request.language:
                    doc = client.fetch_act(
                        act_summary.act_uri, language=request.language, include_text=True
                    )
                else:
                    doc = client.fetch_act_preferred(act_summary.act_uri, include_text=True)
            except Exception:
                logger.exception("Failed to fetch act %s", act_summary.act_uri)
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
                    refs.extend(
                        asdict(ref)
                        for ref in extract_references(
                            chunk.text,
                            from_systematic_number=doc.systematic_number,
                            from_article=chunk.article,
                            language=doc.language,
                            source_url=doc.source_url,
                        )
                    )
        return acts_processed, chunk_payloads, refs

    acts_processed, chunk_payloads, refs = await asyncio.to_thread(_fetch_and_chunk)
    if refs:
        pg_store = PostgresStore()
        await pg_store.init_schema()
        await pg_store.upsert_article_references(refs)

    if not chunk_payloads:
        return IngestFedlexResponse(acts_processed=acts_processed, chunks_upserted=0)

    texts = [text for text, _ in chunk_payloads]
    embedded = embed_service.embed_batch(texts)
    dense_dim = len(embedded[0].dense)
    store.create_collection(dense_dim=dense_dim)

    records = [
        ChunkRecord(text=text, metadata=meta, embedding=embedding)
        for (text, meta), embedding in zip(chunk_payloads, embedded)
    ]
    upserted = store.upsert_chunks(records)

    return IngestFedlexResponse(acts_processed=acts_processed, chunks_upserted=upserted)


async def _run_fedlex_bulk_job(
    job_id: int, prefixes: list[str], max_acts_per_prefix: int
) -> None:
    """Background body of `/ingest/fedlex/bulk`: ingest every prefix, updating
    `ingestion_jobs.details` incrementally so `/ingest/fedlex/bulk/{job_id}` can report
    per-prefix progress while the job is still running. Runs after the triggering
    request has already returned a response (see FastAPI `BackgroundTasks`)."""
    pg_store = PostgresStore()
    await pg_store.init_schema()
    embed_service = get_embed_service()
    store = _store()
    client = FedlexClient()

    details: dict[str, Any] = {"prefixes": {p: {"status": "pending"} for p in prefixes}}
    await pg_store.update_job_details(job_id, details)

    chunk_payloads: list[tuple[str, dict]] = []
    all_refs: list[dict] = []
    try:
        for prefix in prefixes:
            details["prefixes"][prefix] = {"status": "running"}
            await pg_store.update_job_details(job_id, details)
            try:
                acts = await asyncio.to_thread(
                    client.list_consolidated_acts_by_prefix_preferred, prefix
                )
                acts_processed = 0
                prefix_chunks = 0
                for act_summary in acts[:max_acts_per_prefix]:
                    doc = await asyncio.to_thread(
                        client.fetch_act_preferred, act_summary.act_uri, include_text=True
                    )
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
                        prefix_chunks += 1
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
                details["prefixes"][prefix] = {
                    "status": "done",
                    "acts": acts_processed,
                    "chunks": prefix_chunks,
                }
            except Exception:
                logger.exception("Bulk ingestion failed for prefix %s", prefix)
                details["prefixes"][prefix] = {"status": "failed"}
            await pg_store.update_job_details(job_id, details)

        if all_refs:
            await pg_store.upsert_article_references(all_refs)
            details["references_extracted"] = len(all_refs)

        if chunk_payloads:
            texts = [text for text, _ in chunk_payloads]
            embedded = embed_service.embed_batch(texts)
            store.create_collection(dense_dim=len(embedded[0].dense))
            records = [
                ChunkRecord(text=text, metadata=meta, embedding=embedding)
                for (text, meta), embedding in zip(chunk_payloads, embedded)
            ]
            upserted = store.upsert_chunks(records)
            details["chunks_upserted"] = upserted
        else:
            details["chunks_upserted"] = 0

        await pg_store.finish_job(job_id, "completed", details)
    except Exception:
        logger.exception("Bulk ingestion job %d failed", job_id)
        await pg_store.finish_job(job_id, "failed", details)


@app.post("/ingest/fedlex/bulk", response_model=IngestFedlexBulkResponse, status_code=202)
async def ingest_fedlex_bulk(
    request: IngestFedlexBulkRequest, background_tasks: BackgroundTasks
) -> IngestFedlexBulkResponse:
    """Trigger ingestion across every configured (or explicitly requested) Fedlex
    systematic-number prefix as a background job, returning immediately with a job id.

    Unlike `/ingest/fedlex` (single prefix, synchronous, fine for one ad-hoc law), this
    endpoint is for ingesting several codes (e.g. tax + civil + criminal + ...) in one
    call — synchronously, that would comfortably exceed typical HTTP timeouts, since each
    act requires two SPARQL round-trips plus an HTML fetch plus embedding.
    """
    settings = get_settings()
    prefixes = request.prefixes or settings.fedlex_systematic_prefix_list
    pg_store = PostgresStore()
    await pg_store.init_schema()
    job_id = await pg_store.start_job("fedlex_bulk")
    background_tasks.add_task(
        _run_fedlex_bulk_job, job_id, prefixes, request.max_acts_per_prefix
    )
    return IngestFedlexBulkResponse(job_id=job_id)


@app.get("/ingest/fedlex/bulk/{job_id}", response_model=IngestJobStatusResponse)
async def ingest_fedlex_bulk_status(job_id: int) -> IngestJobStatusResponse:
    pg_store = PostgresStore()
    job = await pg_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No ingestion job with id {job_id}")
    return IngestJobStatusResponse(
        id=job["id"],
        job_name=job["job_name"],
        status=job["status"],
        started_at=job["started_at"].isoformat(),
        finished_at=job["finished_at"].isoformat() if job["finished_at"] else None,
        details=job["details"] or {},
    )


@app.post("/ingest/snb", response_model=IngestSnbResponse)
async def ingest_snb(request: IngestSnbRequest) -> IngestSnbResponse:
    settings = get_settings()
    cube_ids = request.cube_ids or settings.snb_test_cubes.split(",")
    pg_store = PostgresStore()
    await pg_store.init_schema()

    results = []
    async with SNBClient() as snb_client:
        for cube_id in cube_ids:
            cube_id = cube_id.strip()
            if not cube_id:
                continue
            result = await snb_client.sync_cube(cube_id, pg_store, lang=request.lang)
            results.append(
                {
                    "cube_id": result.cube_id,
                    "modified": result.modified,
                    "records_upserted": result.records_upserted,
                }
            )
    return IngestSnbResponse(results=results)


@app.post("/ingest/curia_vista", response_model=IngestCuriaVistaResponse)
async def ingest_curia_vista(request: IngestCuriaVistaRequest) -> IngestCuriaVistaResponse:
    """Dual-writes Curia Vista business items: raw fields to PostgreSQL, plus one
    synthetic chunk per item embedded into Qdrant using the same metadata schema as
    Fedlex (architecture spec §3.1/§3.3), so hybrid search can filter across both."""
    settings = get_settings()
    date_from = date.fromisoformat(request.date_from or settings.curia_vista_date_from)
    date_to = date.fromisoformat(request.date_to) if request.date_to else date_from + timedelta(days=30)

    pg_store = PostgresStore()
    await pg_store.init_schema()
    embed_service = get_embed_service()
    qdrant_store = _store()

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

    return IngestCuriaVistaResponse(
        items_upserted_postgres=pg_upserted, chunks_upserted_qdrant=qdrant_upserted
    )

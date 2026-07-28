# Swiss Legal & Financial AI Assistant — Phase 1 Architecture Spec

**Status:** Phase 1 only — ingestion pipeline + RAG core.
**Explicitly out of scope for this phase:** automatic tax return compilation/filing, cantonal law scrapers (require per-canton legal/technical verification), any calculation engine.
**Goal of this phase:** stand up a working, testable RAG pipeline over Swiss federal law, federal parliamentary activity, and Swiss financial/monetary public data, using only sources with simple, documented, no-auth (or key-based) HTTP/API access.

---

## 1. Project Summary

We are building an AI assistant that will eventually help users understand Swiss law (federal + cantonal), understand how Swiss taxation and personal finance work, and get non-binding guidance on legal/tax questions. This phase builds the **foundation**: a data ingestion pipeline that pulls structured legal and financial data from official Swiss sources, transforms it into well-chunked, metadata-rich documents, embeds them, stores them in a vector database, and exposes a retrieval-augmented generation (RAG) query layer with citation verification.

No user-facing tax calculation, no cantonal scraping, no filing automation in this phase. Just: **ingest → chunk → embed → store → retrieve → answer with citations**.

---

## 2. Data Sources (Phase 1 — all reachable via simple HTTP/API calls)

### 2.1 Fedlex — Swiss Federal Legislation
- **What:** Classified Compilation of Federal Legislation (Systematische Rechtssammlung / Recueil systématique), Official Compilation, Federal Gazette, international treaties.
- **Access:** SPARQL endpoint at `https://fedlex.data.admin.ch/sparqlendpoint`. Data modeled with the JOLux ontology (RDF/Linked Data).
- **Reuse rights:** text and metadata are published for reuse, including commercial use, per the conditions at `https://www.fedlex.admin.ch/de/broadcasters`. Always re-check that page for current terms before large-scale ingestion.
- **Scope (expanded beyond Phase 1's tax-law-only start):** the Classified Compilation (`/eli/cc/...` namespace), consolidated (in-force) versions only, preferring Italian, then French, then German (see the project-wide language preference note in §3.1). Ingestion now covers the main federal codes, not just tax law — configured as a list of SR/RS systematic-number prefixes (`Settings.fedlex_systematic_prefix_list`, `config.PREFIX_LABELS`): tax law (640/642), civil code — ZGB/CC (210), code of obligations — OR/CO (220), criminal code — StGB/CP (311.0), criminal procedure — StPO/CPP (312.0), civil procedure — ZPO/CPC (272), federal constitution (101). Still not the full ~17,000-act Fedlex corpus — add more prefixes to widen further.
- **Practical note:** SPARQL has a learning curve. `FedlexClient.list_consolidated_acts_by_prefix` resolves a listed act's title via the `ConsolidationAbstract`'s own direct `jolux:isRealizedBy` expression — **not** the dated per-`Consolidation` member's `jolux:isRealizedBy` expression, which for foundational codes (ZGB, OR, StGB, ZPO, Constitution) carries no `jolux:title` at all (confirmed empirically against the live endpoint while widening scope beyond tax law — the original query silently returned zero acts for exactly these codes, since most 640/642 tax ordinances happen to expose title via the per-consolidation path too, masking the gap). The per-consolidation node is still used for `dateApplicability`/`dateEndApplicability` (an act's title doesn't change across dated versions, but its in-force date range does).

### 2.2 Curia Vista — Federal Parliamentary Business
- **What:** motions, parliamentary initiatives, votes, session data, council members — everything needed for "summarize this new law proposal / this vote."
- **Access:** OData API at `https://ws.parlament.ch/OData.svc` (a legacy endpoint also exists at `ws-old.parlament.ch`; check `parlament.ch` "Open Data / Webservices" page for current status, since the underlying system was replaced in July 2023).
- **Auth:** none required.
- **What to fetch first:** `Business` (Geschäfte) and `Voting` (Abstimmungen) entities, filtered by date, for the summarization use case.

### 2.3 SNB Data Portal — Swiss Monetary & Banking Statistics
- **What:** exchange rates, interest rates, SARON, monetary aggregates, banking statistics, balance of payments — the factual backbone for "how Swiss finance works" content.
- **Access:** public REST API, no authentication.
  - Cube API: `https://data.snb.ch/api/cube/{cubeId}/data/csv/{lang}` and `/api/cube/{cubeId}/dimensions/{lang}`
  - Warehouse API (granular banking stats): `https://data.snb.ch/api/warehouse/cube/{cubeId}/data/csv/{lang}`
- **Rate limiting:** the SNB explicitly asks integrators to only re-fetch when new data is actually available (use the `lastUpdate` method or ETags) and reserves the right to block IPs on excessive use — respect this in the scraper (see §4.1).
- **Documentation:** `https://data.snb.ch/en/help_api`.

### 2.4 Deferred to a later phase (do not implement now)
- Cantonal law repositories (26 different systems, licensing/ToS to be verified individually — lexfind.ch is a reasonable starting aggregator to investigate later).
- Cantonal/court jurisprudence databases.
- AFC (Federal Tax Administration) and FINMA circulars — worth adding once the pipeline pattern is validated, since they are simpler HTML pages rather than APIs and need a lightweight HTML-to-text scraper, not full API integration.
- opendata.swiss (CKAN catalogue): removed from scope. It was used in an earlier draft
  purely for dataset *discovery* (not as a content source) and was never wired into any
  ingestion job; the client and its tests have been deleted.
- Any tax-calculation logic.

---

## 3. System Architecture

```
                     ┌─────────────────────────────┐
                     │        Source APIs          │
                     │  Fedlex SPARQL | Curia Vista │
                     │        | SNB API             │
                     └──────────────┬───────────────┘
                                    │  (scheduled ingestion jobs)
                                    ▼
                     ┌─────────────────────────────┐
                     │   Ingestion / ETL Layer     │
                     │  fetch → normalize → chunk  │
                     │  → attach metadata          │
                     └──────────────┬───────────────┘
                                    ▼
                     ┌─────────────────────────────┐
                     │      Embedding Service       │
                     │  (batch embed new/changed    │
                     │   chunks only)                │
                     └──────────────┬───────────────┘
                                    ▼
                     ┌─────────────────────────────┐
                     │      Vector Database         │
                     │      (Qdrant, self-hosted)   │
                     │  dense + sparse vectors,      │
                     │  rich metadata filters        │
                     └──────────────┬───────────────┘
                                    ▼
                     ┌─────────────────────────────┐
                     │      Retrieval Layer         │
                     │  hybrid search → rerank      │
                     └──────────────┬───────────────┘
                                    ▼
                     ┌─────────────────────────────┐
                     │   LLM Orchestrator (API)     │
                     │  answer + forced citations   │
                     │  + citation verifier          │
                     └──────────────┬───────────────┘
                                    ▼
                              User-facing answer
                          (with source links, non-binding disclaimer)
```

### 3.1 Chunking Strategy

Chunk by **legal/logical unit**, never by fixed character count:
- Fedlex: one chunk per article (or per paragraph if the article is very long, e.g. >1500 tokens), keeping article numbering intact.
- Curia Vista: one chunk per parliamentary business item (Geschäft), with a short synthetic summary chunk plus the raw fields.
- SNB data: not chunked as prose — stored as structured time series in a separate lightweight store (see §3.3), only referenced by the LLM through a data-lookup tool, not through the vector DB.

**Language preference:** where a source publishes the same content in multiple official
languages (Fedlex acts, Curia Vista business items), fetch Italian first, then French,
then German — falling further down the list only for the rare item missing a translation
in a higher-priority language. Implemented as `DEFAULT_LANGUAGE_PRIORITY = ("it"/"IT",
"fr"/"FR", "de"/"DE")` in `fedlex_client.py` and `curia_vista_client.py`.

### 3.2 Metadata Schema (attached to every text chunk before embedding)

```json
{
  "source": "fedlex | curia_vista",
  "level": "federal",
  "law_short_name": "string, e.g. LIFD",
  "article": "string or null",
  "systematic_number": "string, e.g. SR 642.11",
  "area_of_law": "string or null, e.g. civile | penale | tributario | procedura_civile | procedura_penale | costituzionale",
  "language": "de | fr | it | rm",
  "valid_from": "ISO date",
  "valid_to": "ISO date or null (null = still in force)",
  "source_url": "canonical URL to the original document",
  "ingested_at": "ISO datetime",
  "content_hash": "sha256 of raw text, used to detect changes"
}
```

This schema is what makes filtered retrieval possible later (e.g., "only federal tax law, only versions in force today"). `area_of_law` (`config.label_for_systematic_number`) is a coarse, non-authoritative convenience label derived from `systematic_number` via longest-prefix match against `config.PREFIX_LABELS` — a UI/API filtering aid ("solo diritto penale"), not a legal taxonomy. `null` for chunks whose systematic number doesn't map to a configured prefix (e.g. Curia Vista items, or a Fedlex act outside the configured scope).

### 3.3 Storage

- **Vector DB:** Qdrant (self-hosted via Docker). Supports hybrid dense+sparse vectors and rich payload (metadata) filtering natively — needed for the schema above. Payload indexes: `source`, `language`, `systematic_number`, `law_short_name`, `valid_to`, `area_of_law`, `article`.
- **Structured/tabular data (SNB time series, Curia Vista raw fields, ingestion job state, article cross-references):** a simple PostgreSQL database, not the vector store. The LLM accesses SNB/Curia Vista data via a defined query tool, not via embeddings — numeric time series should never be "semantically searched," they should be queried exactly. See §3.7 for the cross-reference table.

### 3.4 Embedding Model

- **Model:** configurable (`Settings.embedding_model`); `jina-embeddings-v3` was tried first per the original spec but its pinned `trust_remote_code` revision was incompatible with the installed `transformers` version, so the running default is a standard multilingual model with no custom remote code (see `.env` / `.env.example` for the current pin) — any strong multilingual DE/FR/IT model works.
- Runs as a small local inference service. **Measured on this project's hardware (CPU):** ~465ms/chunk for a `sentence-transformers` batch encode, i.e. ~10 minutes to embed a single foundational code's full article set (e.g. ZGB, ~1,280 chunks) — this is the dominant cost of expanding legal-area coverage, not the Fedlex SPARQL fetch itself (see §3.7.1's ingestion-job design, which runs multi-code ingestion as a background job for exactly this reason). GPU recommended once corpus grows further.
- Store both dense vectors and a sparse (custom hashed BM25-style) representation per chunk in Qdrant for hybrid search.

### 3.5 Retrieval

- Hybrid query: dense vector search + sparse keyword search combined (Qdrant native hybrid query).
- Metadata pre-filter applied before vector search whenever the query implies a filter (e.g. a canton name, "in force today").
- Rerank top ~50 candidates down to top ~5–8 using an open-source cross-encoder reranker (e.g. BGE-reranker) run locally.

### 3.6 LLM Orchestration & Citation Verification

- LLM called via API (model-agnostic in this phase — the code should accept any OpenAI-compatible or Anthropic-compatible client).
- System prompt constraints:
  1. Answer only using the retrieved chunks provided in context.
  2. Every factual/legal claim must reference a chunk's `article` + `source_url`.
  3. If the retrieved chunks do not contain the answer, say so explicitly rather than guessing.
  4. Always include the non-binding disclaimer for legal/tax content.
- **Citation verifier (deterministic code, not LLM):** after generation, parse the citations the model produced and confirm each one matches a chunk actually present in the retrieved context. Reject/flag any citation that doesn't match — this is a hard filter, not a suggestion. Also checked against articles pulled in via cross-reference expansion (§3.7) — both pools are legitimate citation targets.

### 3.7 Cross-Reference Graph (article-to-article citations)

Swiss statutes constantly cross-reference each other and internally (e.g. "art. 21 LIFD rimanda all'art. 210 CO"). Capturing this improves multi-hop retrieval (a tax-deduction question may need the referenced CO/ZGB article too) and answer grounding.

- **Extraction (`processing.reference_extractor.extract_references`):** runs per-article, after chunking, on already-normalized text. A multilingual regex family (IT/FR/DE — see `_ARTICLE_HEADER`) finds "art./articolo/Art./Artikel N" mentions, then looks a short distance ahead for a trailing law abbreviation:
  - No abbreviation found → treated as an **internal** reference (same law) — by far the most common case in real Fedlex text.
  - A known abbreviation (`processing.law_abbreviations.LAW_ABBREVIATIONS`, hand-curated, seeded from the same codes as `config.PREFIX_LABELS`) → resolved to that law's systematic number.
  - An abbreviation-*shaped* token not in the lookup table → stored **unresolved** (`to_systematic_number=None`), re-resolvable later without re-parsing the source text.
  - Design principle (matches the citation verifier's philosophy): better to under-extract than to fabricate a link. Coverage across DE/FR/IT stylistic variation is intentionally incomplete — this is not a document-structure parser.
- **Storage (`storage.postgres_store`, table `article_references`):** `(from_systematic_number, from_article) → (to_systematic_number, to_article)` edges, with the nullable `to_*` columns for unresolved references and `raw_reference_text` preserved regardless. Re-extraction on re-ingestion is idempotent via delete-then-insert per `(from_systematic_number, from_article)` (an `ON CONFLICT` upsert doesn't work here since PostgreSQL treats NULL as distinct from NULL for uniqueness, which would silently accumulate duplicate unresolved rows across runs).
- **Retrieval-time use (`orchestration.query_engine.QueryEngine._expand_references`):** a bounded 1-hop expansion between rerank and prompt-building — for each reranked chunk with a known `(systematic_number, article)`, look up resolved outgoing references and fetch each target article from Qdrant (`storage.qdrant_store.QdrantStore.get_by_article`, an exact-match `scroll()`, no vector search) if not already among the retrieved chunks. Capped at `MAX_EXPANDED_CHUNKS = 5` per query. Expanded chunks are shown to the model in a visually separate "supporting context" section of the prompt (`orchestration.prompt_templates.build_user_prompt`), but are equally citable — `citation_verifier.verify_citations` is called against the combined retrieved+expanded pool.
- **Why not a graph database or LlamaIndex's `PropertyGraphIndex`:** the actual need is a narrow, well-typed one (article→article edges, 1-hop bounded expansion, no path-finding/ranking) — a Postgres edge table with two indexed lookups covers it, consistent with the project's existing "Postgres for structured/exact data, Qdrant for semantic" split, without the operational overhead of a dedicated graph store or the LLM-based entity/relation extraction those tools assume.

### 3.8 Structured Logging & Observability

`structlog` (already a declared dependency, previously unused) is wired up once per entry point (`logging_config.configure_logging`) — JSON output for the API, console output for local/interactive use (Streamlit, scripts). `orchestration.query_engine.QueryEngine.answer()` emits one structured `query_completed` log line per query, correlated by a per-request `request_id`, with: retrieval/rerank/expansion/LLM latencies, top retrieval/rerank scores, an approximate prompt token count, LLM input/output token usage, a rough `$` cost estimate (`config.estimate_cost_usd`, illustrative only — not for billing), and the citation-verification outcome (verified/unverified counts). This is the pragmatic first step toward observability as query volume grows; a dedicated tracing backend (e.g. Langfuse) is a reasonable later addition if/when team-facing trace search or dashboards are needed, but isn't justified yet (it would mean another self-hosted service for a benefit the project doesn't currently need).

### 3.9 Evaluation

A small, hand-picked golden set (`eval/golden_qa.yaml`) — Q&A pairs across each legal area with expected `(systematic_number, article)` citations, including cases that specifically exercise the §3.7 cross-reference expansion — is scored by `scripts/run_eval.py`: retrieval recall@k against the expected citations, and the citation-verifier pass rate, both computed deterministically against this pipeline's own `QueryResult` output. Deliberately **not** Ragas or an LLM-as-judge framework: those metrics are themselves LLM-judged, which reintroduces the same cost/latency/non-determinism in the *evaluation* layer that the citation verifier was built to avoid in the *answer* layer. Keep the golden set small as legal-area coverage grows — it's a high-signal spot-check, not a second corpus to maintain. `tests/eval/test_golden_set.py` only validates the YAML schema (fast, no network); the actual scored run needs a live Qdrant + Postgres + configured LLM API key and is a manual/CI-optional tool, not a unit test.

### 3.10 Why not LlamaIndex or LangChain

Evaluated and deliberately not adopted, in whole or in part. The custom stack already implements hybrid dense+sparse retrieval with native Qdrant RRF fusion, a local cross-encoder reranker, and — the load-bearing choice — deterministic regex-based citation verification, all with more transparency than the equivalent framework wrapper would offer. Where LlamaIndex looks most relevant on paper (`CitationQueryEngine`) it is actually less rigorous than what's already here: it trusts the LLM to tie citations to node IDs, with no independent post-hoc verification step, which would be a regression on exactly the property that matters most for legal accuracy. `PropertyGraphIndex` is overkill for §3.7's actual need (a narrow, well-typed edge table, not open-ended entity/relation graph traversal), and LlamaIndex's evaluation modules reintroduce the LLM-judge non-determinism §3.9 deliberately avoids. Revisit only if a genuinely new capability gap emerges that isn't already closed more simply by the custom stack.

---

## 4. Technology Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | best ecosystem for RAG/ETL tooling |
| HTTP client | `httpx` | async support, needed for polite rate-limited scraping |
| SPARQL client | `SPARQLWrapper` | standard Python SPARQL client for Fedlex |
| OData client | plain `httpx` + manual query building (Curia Vista OData is simple enough not to need a heavy OData library) |
| Scheduling | `APScheduler` (declared dependency; bulk multi-code ingestion currently runs as an on-demand `BackgroundTasks` job, not a recurring schedule — see §5's `/ingest/fedlex/bulk`) | simple, no infra overhead for a first version |
| Embedding model | configurable multilingual model via `sentence-transformers` (see §3.4) | multilingual, self-hostable |
| Vector DB | Qdrant (Docker) | hybrid search, metadata filtering, self-hostable |
| Relational DB | PostgreSQL (Docker) | structured SNB/Curia Vista data, job state, article cross-references (§3.7) |
| Reranker | `BAAI/bge-reranker-v2-m3` (open-weight, local) | closes the top-50→top-5 gap cheaply |
| Orchestration/API layer | Plain Python (FastAPI) with manual function-calling logic — no LlamaIndex/LangChain, see §3.10 | full control, easier to debug than a heavy agent framework for this scope; preserves the deterministic citation verifier's anti-hallucination guarantee |
| Structured logging | `structlog` (§3.8) | one machine-parseable log line per query with latency/cost/citation-outcome fields |
| Containerization | Docker Compose | one command to bring up Qdrant + Postgres + API locally |

---

## 5. Repository Structure

```
swiss-legal-ai/
├── docker-compose.yml
├── .env.example
├── README.md
├── pyproject.toml
├── src/
│   ├── config.py                     # Settings, PREFIX_LABELS/label_for_systematic_number, LLM cost table
│   ├── logging_config.py             # structlog setup (§3.8), called once per entry point
│   ├── app_streamlit.py              # Phase 3 manual test UI
│   ├── ingestion/
│   │   ├── fedlex_client.py        # SPARQL queries against fedlex.data.admin.ch
│   │   ├── curia_vista_client.py    # OData client for ws.parlament.ch
│   │   ├── snb_client.py            # REST client for data.snb.ch, respects ETag/lastUpdate
│   │   └── base.py                  # shared rate-limiting, retry, and logging helpers
│   ├── processing/
│   │   ├── chunker.py                # article/paragraph-level chunking
│   │   ├── metadata.py               # builds the metadata schema from §3.2
│   │   ├── normalizer.py             # language detection, text cleanup
│   │   ├── reference_extractor.py    # cross-reference extraction (§3.7)
│   │   └── law_abbreviations.py      # abbreviation → SR/RS lookup (§3.7)
│   ├── embedding/
│   │   └── embed_service.py          # batch embedding, dense + sparse
│   ├── storage/
│   │   ├── qdrant_store.py           # collection setup, upsert, hybrid query, get_by_article
│   │   └── postgres_store.py         # structured data, ingestion job state, article_references (§3.7)
│   ├── retrieval/
│   │   ├── hybrid_search.py
│   │   └── reranker.py
│   ├── orchestration/
│   │   ├── prompt_templates.py
│   │   ├── citation_verifier.py
│   │   └── query_engine.py           # ties retrieval + LLM call together; 1-hop expansion (§3.7)
│   └── api/
│       └── main.py                   # FastAPI app: /query, /ingest/*, /ingest/fedlex/bulk (§5)
├── scripts/
│   ├── run_ingestion.py              # one-shot manual ingestion trigger (all configured prefixes)
│   ├── init_qdrant_collection.py
│   └── run_eval.py                   # golden-set scorer (§3.9)
├── eval/
│   └── golden_qa.yaml                # golden Q&A set (§3.9)
└── tests/
    ├── test_fedlex_client.py
    ├── test_chunker.py
    ├── test_hybrid_search.py
    ├── test_citation_verifier.py
    ├── test_config.py
    ├── test_metadata.py
    ├── test_reference_extractor.py
    └── eval/
        └── test_golden_set.py         # schema-only check, no network (§3.9)
```

---

## 6. Implementation Tasks (ordered — hand this list to Codex as the build plan)

1. **Scaffold the repo** with the structure above, `pyproject.toml`, and a `docker-compose.yml` bringing up Qdrant and PostgreSQL with health checks.
2. **Build `fedlex_client.py`**: a SPARQL client that can (a) list consolidated acts under a given systematic number prefix (e.g. `640` for tax law), (b) fetch the full text + metadata (title, articles, dates, language) for a given act. Include a small, hardcoded test query set for validation.
3. **Build `curia_vista_client.py`**: fetch `Business` entities filtered by a date range via the OData endpoint; parse into a normalized dict (title, summary, status, date, related law references if present).
4. **Build `snb_client.py`**: fetch one or two example cubes (e.g. exchange rates, SARON) as CSV, parse into structured records, and store in PostgreSQL. Implement the "only fetch if updated" pattern using ETags as documented by SNB.
5. **Build `chunker.py` + `metadata.py`**: given raw Fedlex act text, split into per-article chunks and attach the metadata schema from §3.2.
6. **Build `embed_service.py`**: load `jina-embeddings-v3`, batch-embed a list of chunks, return dense vectors (and a sparse representation if using Qdrant's built-in sparse support).
7. **Build `qdrant_store.py`**: create the collection with the right vector config (dense + sparse), implement `upsert_chunks()` and `hybrid_search(query, filters)`.
8. **Build `reranker.py`**: load the local cross-encoder reranker, rerank a candidate list against a query.
9. **Build `query_engine.py`**: tie it together — take a user question, run hybrid search + rerank, build the LLM prompt with retrieved chunks, call the LLM, run `citation_verifier.py` on the output, return the final answer with verified sources.
10. **Build `api/main.py`**: a minimal FastAPI app with a single `POST /query` endpoint wrapping `query_engine.py`, plus a `POST /ingest/fedlex` and `POST /ingest/snb` endpoint to manually trigger ingestion for testing.
11. **Write tests** for each client (mocking HTTP calls), the chunker, and the citation verifier.
12. **Write `scripts/run_ingestion.py`**: a manual script to run a small, bounded ingestion (e.g. one systematic number prefix from Fedlex, one SNB cube, one month of Curia Vista business) end-to-end, for local verification before scaling up.

---

## 7. Configuration (.env.example)

```
# Vector DB
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Relational DB
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=swiss_legal_ai
POSTGRES_USER=postgres
POSTGRES_PASSWORD=changeme

# LLM provider (choose one)
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=
# or
OPENAI_API_KEY=

# Embedding model (jina-embeddings-v3 was tried first per the original spec above, but
# its pinned trust_remote_code revision is incompatible with the installed transformers
# version — see .env.example for the current pin)
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=cpu   # or cuda

# Ingestion scope — comma-separated SR/RS systematic-number prefixes (§2.1); see
# config.PREFIX_LABELS for the area-of-law label attached to each.
FEDLEX_SYSTEMATIC_PREFIXES=640,642,210,220,311.0,312.0,272,101
SNB_TEST_CUBES=rendoblim,snbmonagg
CURIA_VISTA_DATE_FROM=2025-01-01
```

---

## 8. Legal & Compliance Notes (recap for this phase)

- Fedlex text/metadata reuse, including commercial, is authorized per the conditions published at `fedlex.admin.ch/de/broadcasters` — re-verify before scaling ingestion volume.
- Curia Vista and the SNB data portal are both public, no-auth APIs intended for reuse — still respect each one's stated rate-limiting etiquette (especially SNB's ETag/`lastUpdate` guidance).
- No personal data is being ingested in this phase (all sources are institutional/legislative/statistical, not case law with personal data) — this simplifies nLPD exposure considerably for Phase 1.
- Every response generated by the system must carry a clear "informational, not legally binding" disclaimer — this is a product requirement, not just a legal one, and should be enforced in `prompt_templates.py`, not left to model discretion.

---

## 9. Definition of Done for Phase 1

- `docker-compose up` brings up Qdrant + Postgres cleanly.
- `scripts/run_ingestion.py` successfully ingests: a bounded set of Fedlex articles (one systematic number prefix), one to two SNB cubes, and one month of Curia Vista business items.
- `POST /query` returns an answer to a basic test question (e.g. "what does the federal law say about X") with at least one verified citation pointing to a real ingested chunk.
- All unit tests pass.
- No cantonal scraping, no tax calculation logic present — confirmed out of scope for this phase.

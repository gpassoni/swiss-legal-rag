# Swiss Legal & Financial AI Assistant — Phase 1

Ingestion pipeline + RAG core over Swiss federal law (Fedlex), federal parliamentary
activity (Curia Vista), and SNB monetary/banking data.

**Scope of this phase:** ingest → chunk → embed → store → retrieve → answer with citations.
No cantonal scraping, no tax calculation logic, no filing automation. See
`swiss-legal-ai-architecture.md` for the full spec.

## Quickstart

Dependency management uses [uv](https://docs.astral.sh/uv/) (not a manually-managed venv).
Install uv itself first (`pip install uv`, or the standalone installer at
https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
cp .env.example .env   # fill in ANTHROPIC_API_KEY or OPENAI_API_KEY
docker compose up -d   # brings up Qdrant + PostgreSQL
uv sync --extra dev --extra notebook   # creates .venv + uv.lock, installs everything

uv run python scripts/init_qdrant_collection.py
uv run python scripts/run_ingestion.py

uv run uvicorn src.api.main:app --reload
```

Then:

```bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question": "What does federal tax law say about deductions?"}'
```

See `VERIFY_SETUP.md` for a step-by-step checklist to confirm the environment works, and
`notebooks/data_showcase.ipynb` for live examples of each ingestion source's raw API
response next to its cleaned/normalized form.

## Running ingestion (no manual API calls needed)

**Double-click `start_ingestion.bat`** (repo root) — it starts Qdrant/PostgreSQL via
Docker Compose if they aren't already running, then runs `scripts/run_ingestion.py`,
which ingests **every configured Fedlex code** (`FEDLEX_SYSTEMATIC_PREFIXES` in `.env` —
tax law, civil code, code of obligations, criminal code + procedure, civil procedure,
constitution) plus SNB and Curia Vista, in one go. The window stays open with a summary
when it's done; close it or press any key.

Equivalent from a terminal: `uv run python scripts/run_ingestion.py`.

The `/ingest/fedlex/bulk` API endpoint does the same thing for the Fedlex side only, as a
background job you trigger over HTTP (`POST`, then poll `GET /ingest/fedlex/bulk/{job_id}`)
— use that only if you're driving ingestion from the running API rather than the
command line.

**Where the logs are:** every run of `scripts/run_ingestion.py` writes a structured,
timestamped log file to `logs/ingestion_<YYYYMMDD_HHMMSS>.log` (created automatically),
in addition to printing the same lines to the console/window. Open the newest file in
`logs/` to see what happened in the last run — retrieval/embedding/upsert steps, per-code
act/chunk counts, and any errors. The API (`src/api/main.py`) and the eval script log
structured JSON to the console only (no file) — see `swiss-legal-ai-architecture.md` §3.8.

**GPU:** if `nvidia-smi` shows a GPU and `EMBEDDING_DEVICE=cuda` is set in `.env` (already
the case on this machine — see `pyproject.toml`'s `[tool.uv.sources]` for how the
CUDA-enabled `torch` build is pinned), embedding runs on the GPU automatically; tune
`EMBEDDING_BATCH_SIZE` in `.env` to your VRAM. This only speeds up the embedding step —
the Fedlex SPARQL fetch stays throttled to one request per 0.5s regardless
(`FedlexClient._throttle`), which is intentional and should not be lowered or
parallelized (see §3.7/§3.4 risk notes in the architecture doc).

## Repository layout

```
src/
├── config.py          # settings, area-of-law labels, LLM cost table
├── logging_config.py   # structlog setup (console + optional file)
├── ingestion/       # Fedlex (SPARQL), Curia Vista (OData), SNB (REST)
├── processing/       # chunking, metadata schema, normalization, cross-reference extraction
├── embedding/         # batch embedding service (dense + sparse)
├── storage/           # Qdrant (vectors) + PostgreSQL (structured data, job state, cross-refs)
├── retrieval/          # hybrid search + reranking
├── orchestration/      # prompt templates, citation verifier, query engine
└── api/                 # FastAPI app
scripts/
├── run_ingestion.py      # full ingestion — all configured codes + SNB + Curia Vista
├── run_eval.py           # golden-set scorer (see eval/golden_qa.yaml)
└── init_qdrant_collection.py
start_ingestion.bat       # double-click to run the full ingestion, no CLI needed
eval/
└── golden_qa.yaml        # golden Q&A set for scripts/run_eval.py
notebooks/
└── data_showcase.ipynb   # raw API responses vs. cleaned data, per source
tests/
├── ...                   # unit tests (clients, chunker, citation verifier, config,
│                           metadata, cross-reference extractor — all mocked/pure, no
│                           live services needed)
└── eval/test_golden_set.py   # golden-set schema check only (fast, no network)
```

## Testing

```bash
uv run pytest
```

Ingestion-relevant unit tests specifically: `tests/test_fedlex_client.py` (mocked SPARQL
responses), `tests/test_chunker.py`, `tests/test_metadata.py`, `tests/test_config.py`
(prefix list / area-of-law labels), and `tests/test_reference_extractor.py`
(cross-reference extraction, IT/FR/DE). None of these need Docker/Qdrant/Postgres running.

## Legal & compliance

Every answer returned by `/query` carries a non-binding "informational only" disclaimer.
See §8 of `swiss-legal-ai-architecture.md` for source reuse terms and rate-limiting etiquette
(SNB in particular expects ETag/`lastUpdate`-based polling, not blind re-fetching).

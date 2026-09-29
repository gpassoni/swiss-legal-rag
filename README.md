# Swiss Legal RAG

Question answering over Swiss federal law, grounded in official sources and checked
citation by citation.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

## What it is

A retrieval-augmented generation (RAG) pipeline that ingests Swiss federal legislation from
Fedlex, parliamentary business from Curia Vista, and monetary statistics from the Swiss
National Bank. It turns laws into article-level chunks, indexes them for hybrid search, and
answers legal questions with an LLM. Every citation in an answer is checked against the
articles the model was actually shown. It is exposed as a FastAPI service, with a
Streamlit UI for manual testing.

## Why it matters

Legal answers are only useful if they point to the exact article they rely on, and LLMs
readily invent plausible-looking citations. Swiss law adds its own difficulties: three
official languages, laws that constantly cross-reference each other ("Art. 18 CC applies
by analogy…"), and official data behind a SPARQL linked-data endpoint rather than a simple
API. This project treats grounding as a hard constraint. Citations that don't match
retrieved context are rejected by a deterministic check, not by a second LLM call.

## Results

A full ingestion run over the default scope
([`results/ingestion_2026-07-28.txt`](results/ingestion_2026-07-28.txt)):

| | |
|---|---|
| Federal acts ingested | 14 consolidated acts across 7 areas (tax, civil, obligations, criminal, criminal procedure, civil procedure, constitution) |
| Indexed chunks | 4,973 article-level chunks in Qdrant |
| Cross-references extracted | 1,625 article-to-article links |
| SNB observations | 13,632 records from 2 data cubes, stored in PostgreSQL |
| Parliamentary items | 32 Curia Vista business items (January 2025) |
| GPU embedding speedup | ~34x vs. CPU (8 GB consumer GPU) |
| Test suite | 118 unit tests, all external calls mocked |

The pipeline runs end to end and comes with a golden-set evaluation harness
(`scripts/run_eval.py`: recall@k, precision@k, MRR, citation pass rate). Retrieval quality
still needs improvement and is the main focus of future work.

## How it works

```
Fedlex (SPARQL)      ─┐
Curia Vista (OData)  ─┼─> normalize -> chunk per article -> metadata -> embed ──> Qdrant (dense + sparse)
SNB (REST + ETag)    ─┘                            └─> cross-reference graph ──> PostgreSQL

question -> hybrid search (RRF) -> cross-encoder rerank -> 1-hop cross-reference expansion
         -> token-budgeted prompt -> LLM -> citation verifier -> answer + verified sources
```

- **Ingestion**: hand-written SPARQL queries against Fedlex's JOLux ontology, an OData
  client for Curia Vista, and ETag-conditional fetches for SNB data. All clients are
  throttled and retried to stay polite to public endpoints. Italian is preferred, with
  French and German as fallbacks.
- **Chunking**: one chunk per `Art. N`, split by paragraph above ~1,500 tokens, each with a
  shared metadata schema (law, article, language, validity dates, source URL).
- **Retrieval**: multilingual dense embeddings (`BAAI/bge-m3` by default) plus a hashed
  BM25-style sparse vector, fused in Qdrant with Reciprocal Rank Fusion. Results are
  filtered to law in force today and reranked with `BAAI/bge-reranker-v2-m3`.
- **Cross-reference expansion**: references like "Art. 31 CO" are extracted at ingestion
  time (IT/FR/DE abbreviations), and the articles they point to are added to the context.
- **Grounded answers**: the LLM (Anthropic or OpenAI, behind a small adapter interface)
  must cite `(Art. N, <url>)`. A regex verifier accepts only citations that match chunks
  in the prompt. A non-binding disclaimer is always appended.
- **Observability**: structured logs (structlog) per query with per-stage latency, token
  counts, and cost estimate.

**Stack:** Python 3.11, FastAPI, Qdrant, PostgreSQL, sentence-transformers, PyTorch,
Anthropic/OpenAI SDKs, Streamlit, Docker Compose, uv, pytest, ruff.

## How to run it

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), Docker, and an Anthropic or
OpenAI API key. All data comes from public APIs, so no dataset download is needed.

```bash
git clone https://github.com/gpassoni/swiss-legal-rag.git
cd swiss-legal-rag
uv sync --extra dev
cp .env.example .env              # set ANTHROPIC_API_KEY (or LLM_PROVIDER=openai + OPENAI_API_KEY)

docker compose up -d              # Qdrant + PostgreSQL
uv run python scripts/init_qdrant_collection.py
uv run python scripts/run_ingestion.py    # fetch, chunk, embed, store (logs in logs/)

uv run uvicorn src.api.main:app
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question": "Cosa dice il Codice civile svizzero sul principio della buona fede?"}'
```

Other entry points:

```bash
uv run streamlit run scripts/streamlit_app.py   # interactive test UI
uv run python scripts/run_eval.py               # golden-set evaluation
uv run pytest                                   # unit tests, no services needed
```

On CPU, embedding a large code like the Civil Code takes about 10 minutes. For GPU setup,
a live connectivity check, and troubleshooting, see [`docs/setup.md`](docs/setup.md).

## Project structure

```
src/
├── ingestion/       # Fedlex (SPARQL), Curia Vista (OData), SNB (REST) clients
├── processing/      # normalization, article chunking, metadata, cross-reference extraction
├── embedding/       # batched dense + sparse embedding
├── storage/         # Qdrant (vectors) and PostgreSQL (structured data, references, jobs)
├── retrieval/       # hybrid search and cross-encoder reranking
├── orchestration/   # prompts, LLM adapters, query engine, citation verifier
├── api/             # FastAPI app
└── config.py        # settings from .env
scripts/             # ingestion, evaluation, Qdrant tools, Streamlit UI
eval/                # golden Q&A set and evaluation outputs
notebooks/           # raw vs. cleaned data from each live source
docs/                # architecture spec and setup guide
tests/               # unit tests
```

## Next steps

- Improve retrieval quality, which is currently the weakest stage (better chunking of
  long articles, query translation across languages, tuning the fusion and rerank depth).
- Grow the golden evaluation set beyond a handful of questions per legal area.
- Add query-result caching and broaden coverage toward cantonal law.

This project provides legal information, not legal advice.

## License

[MIT](LICENSE)

# Swiss Legal RAG

**Ask a question about Swiss law, get an answer you can check: every claim traced back to
the exact article of the official federal text.**

[![CI](https://github.com/gpassoni/swiss-legal-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/gpassoni/swiss-legal-rag/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

## The problem

Switzerland is a direct democracy. Several times a year, citizens are asked to vote on
laws, constitutional amendments, and popular initiatives. Most of them never read the text
they are voting on, and it is hard to blame them. Federal law is spread across thousands
of acts, published in German, French, and Italian, and full of cross-references
("Art. 18 CC applies by analogy…"). The official source, Fedlex, is a linked-data
endpoint built for machines, not a search box built for people.

The same gap hits anyone with an everyday legal question: a tenant, a small-business owner,
someone reading a tax notice. Lawyers are expensive, and search engines return opinions,
not the law. General-purpose chatbots are worse still: they answer fluently and cite
articles that don't exist.

## The goal

The aim is an assistant that makes Swiss law **accessible without making it less
accurate**:

- **Precise**: every statement is tied to a specific article of the official text, with a
  link to read it at the source.
- **Relevant**: find the few articles that actually answer the question among thousands,
  including the ones they cross-reference.
- **Trustworthy by construction**: a citation that doesn't match what the system actually
  retrieved is rejected. This is enforced in code, not left to the model.
- **Connected to democratic life**: alongside the laws in force, the system ingests the
  federal parliament's current business (motions, bills, initiatives), the pipeline that
  new laws, and many popular votes, come from.

This repository is the foundation for that goal: the data pipeline and the question
answering engine.

## What it looks like

A real response from the running system (`POST /query`), asked in Italian:

> **Q:** *Cosa dice il Codice civile svizzero sul principio della buona fede?*
> (What does the Swiss Civil Code say about good faith?)
>
> **A:** L'articolo 2 stabilisce che: "Ognuno è tenuto ad agire secondo la buona fede così
> nell'esercizio dei propri diritti come nell'adempimento dei propri obblighi." (Art. 2,
> https://fedlex.data.admin.ch/eli/cc/24/233_245_233) Inoltre, "Il manifesto abuso del
> proprio diritto non è protetto dalla legge." …
>
> `verified_sources`: Art. 2, Art. 3 (Civil Code) · `unverified_citation_count`: 0

The answer quotes the law instead of paraphrasing it from memory. Every citation points to
the official text and has been checked against the retrieved articles.

## What's been built

Ingesting the default scope of federal law, parliamentary business, and National Bank data
([`results/ingestion_2026-07-28.txt`](results/ingestion_2026-07-28.txt)):

| | |
|---|---|
| Federal law | 14 consolidated acts across 7 areas: tax, civil, obligations, criminal, criminal procedure, civil procedure, constitution |
| Searchable index | 4,973 article-level chunks |
| Legal cross-reference graph | 1,625 article-to-article links |
| Parliamentary business | 32 Curia Vista items (January 2025) |
| Financial data | 13,632 Swiss National Bank observations |
| Embedding on GPU | ~34x faster than CPU |
| Test suite | 118 unit tests, all external services mocked |

The pipeline works end to end. The next step is retrieval quality: finding the right
article at the top of the list for every kind of question still needs improvement, and a
golden-set evaluation harness (`scripts/run_eval.py`) is in place to measure it.

## How it works

```
Fedlex (SPARQL)      ─┐
Curia Vista (OData)  ─┼─> normalize -> chunk per article -> metadata -> embed ──> Qdrant (dense + sparse)
SNB (REST + ETag)    ─┘                            └─> cross-reference graph ──> PostgreSQL

question -> hybrid search (RRF) -> cross-encoder rerank -> 1-hop cross-reference expansion
         -> token-budgeted prompt -> LLM -> citation verifier -> answer + verified sources
```

- **Reading the law at the source**: hand-written SPARQL queries against Fedlex's JOLux
  ontology, an OData client for parliament, and ETag-conditional fetches for SNB data.
  Every client is rate-limited and retried to stay polite to public infrastructure.
  Italian is preferred, with French and German as fallbacks.
- **One article, one unit of meaning**: laws are split on `Art. N` headers (by paragraph
  above ~1,500 tokens), and each chunk carries the law, article, language, validity dates,
  and source URL.
- **Finding what matters**: multilingual dense embeddings (`BAAI/bge-m3`) and a hashed
  BM25-style sparse vector, fused with Reciprocal Rank Fusion in Qdrant. Results are
  restricted to law in force today and reranked by a cross-encoder
  (`BAAI/bge-reranker-v2-m3`).
- **Following the references**: citations such as "Art. 31 CO" are extracted at ingestion
  (IT/FR/DE abbreviations) into a graph, and the articles they point to are pulled into the
  context, the way a lawyer would flip to them.
- **Refusing to make things up**: the LLM (Anthropic or OpenAI, behind a small adapter
  interface) must cite in a fixed `(Art. N, <url>)` format. A deterministic verifier keeps
  only citations that match what the model was shown, and every answer carries a
  "not legal advice" disclaimer.
- **Observable**: structured logs per query with per-stage latency, token counts, and cost
  estimate.

**Stack:** Python 3.11 · FastAPI · Qdrant · PostgreSQL · sentence-transformers · PyTorch ·
Anthropic / OpenAI · Streamlit · Docker Compose · uv · pytest · ruff

## Quickstart

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), Docker, and an Anthropic or OpenAI
API key. All data comes from public APIs; there is nothing to download by hand.

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
a live connectivity check, and troubleshooting (including a Windows note), see
[`docs/setup.md`](docs/setup.md). The full design is in
[`docs/architecture.md`](docs/architecture.md).

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

## Roadmap

- **Sharper retrieval**: better handling of long articles, cross-language queries, and
  tuning of fusion and rerank depth, measured against a larger golden set.
- **Closer to the ballot box**: link parliamentary business to the articles it would
  change, so a voter can see what a proposal actually amends.
- **Wider coverage**: more federal law, then cantonal law, where much of everyday life
  (taxes, housing, schools) is actually regulated.

This project provides legal information, not legal advice.

## License

[MIT](LICENSE)

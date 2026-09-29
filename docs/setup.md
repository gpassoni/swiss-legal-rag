# Setup and verification

A step-by-step checklist for a fresh clone. Each step says what a pass looks like.

## 1. Environment

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/) (`pip install uv` works too).

```bash
uv sync --extra dev --extra notebook
uv run pytest            # all tests are mocked: no network, Docker, or API key needed
uv run ruff check .
```

**GPU (optional).** PyPI's `torch` wheel is CPU-only on Windows. For CUDA, install the
matching build from PyTorch's index after syncing, then set `EMBEDDING_DEVICE=cuda` in `.env`:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu126
```

## 2. Live connectivity to the three sources

No API key or Docker needed; Fedlex, Curia Vista, and SNB are public endpoints.

```bash
uv run python -c "
import asyncio
from datetime import date
from src.ingestion.fedlex_client import FedlexClient
from src.ingestion.curia_vista_client import CuriaVistaClient
from src.ingestion.snb_client import SNBClient

acts = FedlexClient().list_consolidated_acts_by_prefix('642', limit=3)
print('Fedlex:', len(acts), 'acts')

async def main():
    async with CuriaVistaClient() as c:
        items = await c.fetch_business(date(2025, 1, 1), date(2025, 1, 31), top=3)
        print('Curia Vista:', len(items), 'items')
    async with SNBClient() as c:
        csv_text, etag = await c.fetch_cube_csv('rendoblim', lang='en')
        print('SNB:', len(c.parse_cube_csv(csv_text)), 'records')

asyncio.run(main())
"
```

**Pass:** all three counts are non-zero. A zero where data is expected usually means an
upstream API or schema change, not "no data" (see the quirks listed at the top of
`notebooks/data_showcase.ipynb`).

## 3. Notebook

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
  notebooks/data_showcase.ipynb --ExecutePreprocessor.timeout=180
```

**Pass:** no `CellExecutionError`, and the summary section at the bottom renders.

## 4. Full pipeline

```bash
cp .env.example .env                              # add ANTHROPIC_API_KEY or OPENAI_API_KEY
docker compose up -d                              # Qdrant + PostgreSQL
uv run python scripts/init_qdrant_collection.py
uv run python scripts/run_ingestion.py            # logs to logs/ingestion_<timestamp>.log
uv run uvicorn src.api.main:app --reload
```

**Pass:** both containers report healthy, both scripts exit 0, and a `POST /query` returns
JSON with an `answer` and `verified_sources`.

## Troubleshooting

- **`uv: command not found`**: uv isn't on `PATH`; use `python -m uv ...` or the full path
  to the `uv` executable.
- **Unexpected Python version**: `.python-version` pins 3.11; `uv python list` shows what
  uv can see.
- **Embedding stalls on GPU**: lower `EMBEDDING_BATCH_SIZE`. Long outlier chunks make every
  item in a batch pad to their length (see the notes in `.env.example`).
- **Windows: `/query` fails with `psycopg.InterfaceError ... ProactorEventLoop`**: recent
  uvicorn versions create their own event loop and ignore the selector policy set in
  `src/storage/postgres_store.py`. Known issue; until it is fixed, start the API with the
  policy applied:

  ```bash
  uv run python -c "import asyncio, uvicorn; from src.api.main import app; asyncio.run(uvicorn.Server(uvicorn.Config(app)).serve())"
  ```

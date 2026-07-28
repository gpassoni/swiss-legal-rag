# Verifying the uv-based environment

This project moved from a manually-managed `venv/` to [uv](https://docs.astral.sh/uv/).
`uv` reads `pyproject.toml`, resolves versions into `uv.lock`, and manages a project-local
`.venv/` for you — there is nothing to activate by hand; every command below is run through
`uv run ...`, which transparently uses `.venv/`.

This document is a step-by-step checklist to confirm the migration actually works: that
the environment builds, the package imports, the test suite passes, the four live
ingestion sources are reachable, and the notebook runs. Work through it top to bottom;
each step tells you what a pass looks like.

## 0. Prerequisites

- Python 3.11+ available somewhere on the machine (uv will also download its own managed
  CPython if none is found — see step 1's output for which one it picked).
- `uv` itself installed. Check with:

  ```bash
  uv --version
  ```

  If missing: `pip install uv` (into any Python/conda env you have handy — it only needs
  to exist once, globally, to bootstrap everything else), or the standalone installer at
  https://docs.astral.sh/uv/getting-started/installation/.

## 1. Create/refresh the environment

From the project root (the folder with `pyproject.toml`):

```bash
uv sync --extra dev --extra notebook
```

**What this does**: creates `.venv/` (if it doesn't exist), resolves every dependency in
`pyproject.toml` — including the `dev` extra (pytest, ruff, respx, ...) and the `notebook`
extra (jupyter, ipykernel, pandas) — into `uv.lock`, and installs them all.

**Pass criteria**: exits 0, ends with a list of `Installed N packages`/`+ package==version`
lines, no `error:` lines. Re-running it should be near-instant and print little to nothing
new (nothing to do).

```bash
uv run python -c "import sys; print(sys.version)"
```

**Pass criteria**: prints a `3.11.x` version string, confirming `.venv/` has the right
interpreter (pinned via `.python-version`).

## 2. Confirm the package and its dependencies import cleanly

```bash
uv run python -c "import fastapi, qdrant_client, sentence_transformers, anthropic, openai, psycopg, SPARQLWrapper, structlog; print('OK')"
```

**Pass criteria**: prints `OK` with no `ModuleNotFoundError` / `ImportError`.

## 3. Run the test suite

```bash
uv run pytest -q
```

**Pass criteria**: all tests pass (`N passed` in green, `0 failed`). As of this migration
that's **31 passed**, including 3 new regression tests added for bugs found while
validating against the live APIs (see §5 below and the "Notes" section in
`notebooks/data_showcase.ipynb`):

- `tests/test_curia_vista_client.py::test_fetch_business_handles_plain_list_payload`
- `tests/test_snb_client.py::test_parse_cube_csv_skips_metadata_preamble_and_bom`
- (the `follow_redirects` fix in `src/ingestion/base.py` is covered indirectly by every
  existing `respx`-mocked test still passing, plus the live check in §5)

If a test fails, `pytest -q` output name the failing test — re-run just that one with
`uv run pytest tests/test_x.py::test_name -v` for full detail.

## 4. Lint check (optional but quick)

```bash
uv run ruff check .
```

**Pass criteria for the migration specifically**: this command *runs* (proves `ruff` is
installed and usable through `uv run`). As of this writing it reports ~24 pre-existing
findings scattered across `src/` and `tests/` (unrelated to the uv migration — e.g. `Self`
typing modernizations, an unused `# noqa: E402`, `datetime.utcnow()` deprecation) that were
never cleaned up before. That's a separate pre-existing lint-debt item, not a sign the
environment switch broke anything — don't be alarmed by non-zero output here.

## 5. Confirm live connectivity to all three ingestion sources

This is the step that actually exercises the network + the real external APIs, not just
mocks. It requires internet access; no API keys or Docker services are needed (Fedlex,
Curia Vista, and SNB are all public/unauthenticated).

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
        records = c.parse_cube_csv(csv_text)
        print('SNB:', len(records), 'records, etag =', etag[:12] + '...')

asyncio.run(main())
"
```

**Pass criteria**: all three lines print non-zero counts, e.g.:

```
Fedlex: 3 acts
Curia Vista: 3 items
SNB: 5744 records, etag = "092c612ae59...
```

If any count is `0` where you'd expect data, that's a real signal something changed
upstream (see the "known bugs fixed" note in the notebook for prior examples of this
exact failure mode — an empty result is often silent breakage, not "no data available").

## 6. Run the notebook

The `notebook` extra installs Jupyter + a kernel is registered against this project's
`.venv/`. Register it once (already done as part of this migration, but here's the command
for reference / a fresh clone):

```bash
uv run python -m ipykernel install --user --name swiss-legal-ai --display-name "Python (swiss-legal-ai uv)"
```

Then either open `notebooks/data_showcase.ipynb` in VS Code / JupyterLab and select the
**"Python (swiss-legal-ai uv)"** kernel and run all cells, or execute it headlessly:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/data_showcase.ipynb --ExecutePreprocessor.timeout=180
```

**Pass criteria**: no `CellExecutionError`; every cell has an output; the "Summary" section
at the bottom renders. The notebook shows, per source, the raw API response next to the
cleaned Python object — if you can read through it and see real Swiss legal/parliamentary/
financial data (not stack traces), the ingestion layer works end-to-end on the new
environment.

## 7. (Optional) Full local pipeline

Steps 1–6 don't need Docker. If you also want to verify the storage/embedding layer:

```bash
docker compose up -d                          # Qdrant + PostgreSQL
uv run python scripts/init_qdrant_collection.py
uv run python scripts/run_ingestion.py
uv run uvicorn src.api.main:app --reload
```

Then, in another terminal:

```bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question": "What does federal tax law say about deductions?"}'
```

**Pass criteria**: `docker compose up -d` reports both services healthy/running; the
`init_qdrant_collection.py` and `run_ingestion.py` scripts exit 0; the `curl` call returns
a JSON response (not a connection error), which confirms FastAPI, Qdrant, and PostgreSQL
are all reachable from the uv-managed environment.

## Troubleshooting

- **`uv: command not found`** — uv isn't on `PATH` for this shell. If you installed it into
  a specific Python env (e.g. `pip install uv` inside a conda env), either activate that
  env first or call it via `python -m uv ...` / the env's `Scripts/uv.exe` directly.
- **`uv sync` picks an unexpected Python version** — check `.python-version` at the repo
  root (currently pinned to `3.11`); `uv python list` shows every interpreter uv can see
  and where it came from.
- **A source in step 5 returns 0 results where it shouldn't** — don't assume it's your
  environment; it may be an upstream API/schema change. Compare against the specific
  counts and endpoints documented in `notebooks/data_showcase.ipynb`.
- **Old `venv/` artifacts confuse things** — this project no longer uses a manually created
  `venv/`; if one still exists from before, delete it (`.venv/` — the uv-managed one — is
  the only environment this project needs, and it's already gitignored).

@echo off
REM Double-click this file to run the full ingestion (all configured Fedlex codes +
REM SNB + Curia Vista) — no manual POST request needed, see scripts/run_ingestion.py.
REM Logs are written to logs\ingestion_<timestamp>.log (see swiss-legal-ai-architecture.md
REM section 3.8) as well as printed in this window.

cd /d "%~dp0"

echo Starting Qdrant + PostgreSQL (no-op if already running)...
docker compose up -d

echo.
echo Running ingestion — this can take a while (CPU embedding is the bottleneck; a
echo single large code like the Civil Code takes roughly 10 minutes on CPU, seconds
echo to a couple of minutes on GPU). Do not close this window while it says "Running".
echo.

uv run python scripts\run_ingestion.py

echo.
echo Ingestion finished. Press any key to close this window.
pause >nul

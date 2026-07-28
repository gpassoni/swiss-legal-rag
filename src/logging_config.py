"""Structured logging setup (structlog) — one place, called once per entry point.

`structlog` was already a declared dependency but unused anywhere in `src/` (every
module called plain `logging.getLogger(...)`); this wires it up so
`orchestration.query_engine` can emit one structured, machine-parseable log line per
query (retrieval/rerank/LLM latency, token usage, cost estimate, citation-verification
outcome) instead of scattered plain-text log calls.
"""
from __future__ import annotations

import logging
from pathlib import Path

import structlog


def configure_logging(json_output: bool, log_file: Path | str | None = None) -> None:
    """Call once at process start. `json_output=True` for the API (machine-parseable
    logs in production); `json_output=False` for local/interactive use (Streamlit,
    scripts), which renders to a readable console format instead.

    `log_file`, when given, additionally writes every log line to that file (appended,
    UTF-8) alongside the console — used by `scripts/run_ingestion.py` so a run started
    by double-clicking a launcher still leaves a persistent, inspectable record (see
    `start_ingestion.bat`). Without it, logs only ever go to stdout/the console.
    """
    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]
    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=log_file is None)
    )

    if log_file is None:
        structlog.configure(
            processors=[*shared_processors, renderer],
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
        return

    # Routing through stdlib logging (rather than PrintLoggerFactory) is what lets one
    # structlog call fan out to two destinations (console + file) at once.
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()  # avoid duplicate lines if configure_logging is re-called

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

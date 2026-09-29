"""REST client for the SNB Data Portal (data.snb.ch): exchange rates, SARON, monetary
aggregates, banking statistics.

The SNB explicitly asks integrators to only re-fetch when new data is actually available
(architecture spec §2.4) — this client always sends the previously stored ETag as
``If-None-Match`` and treats HTTP 304 as "nothing to do", never blindly re-downloading a
cube on every run.

Docs: https://data.snb.ch/en/help_api
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass

from src.storage.postgres_store import PostgresStore

from .base import PoliteAsyncClient

logger = logging.getLogger("swiss_legal_ai.ingestion.snb")

SNB_BASE_URL = "https://data.snb.ch"


@dataclass
class CubeSyncResult:
    cube_id: str
    modified: bool
    records_upserted: int


class SNBClient:
    def __init__(self, base_url: str = SNB_BASE_URL, rate_limit_per_sec: int = 2) -> None:
        self._client = PoliteAsyncClient(base_url=base_url, rate_limit_per_sec=rate_limit_per_sec)

    async def __aenter__(self) -> SNBClient:
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    async def fetch_cube_csv(
        self, cube_id: str, lang: str = "en", etag: str | None = None
    ) -> tuple[str | None, str | None]:
        """GET the cube's CSV export. Returns (csv_text_or_None_if_304, new_etag).

        A conditional request is made whenever a prior ``etag`` is supplied; the SNB API
        is expected to answer with 304 Not Modified when nothing has changed.
        """
        headers = {"If-None-Match": etag} if etag else {}
        response = await self._client.get(f"/api/cube/{cube_id}/data/csv/{lang}", headers=headers)
        if response.status_code == 304:
            logger.info("Cube %s not modified since last fetch (etag=%s)", cube_id, etag)
            return None, etag
        new_etag = response.headers.get("ETag", etag)
        return response.text, new_etag

    @staticmethod
    def parse_cube_csv(csv_text: str) -> list[dict]:
        """Parse an SNB cube CSV export into records of {dimensions, observation_date, value}.

        SNB cube CSVs use ``Date`` and ``Value`` columns plus one or more dimension
        columns; delimiter is sniffed since it varies between comma and semicolon across
        cubes. This is a best-effort generic parser — validate column names against the
        specific cube's ``/dimensions`` endpoint before relying on it for a new cube.

        Real exports are prefixed with a UTF-8 BOM and a few metadata lines (``CubeId``,
        ``PublishingDate``, a blank line) before the actual ``Date;...;Value`` header row,
        so this skips ahead to whichever line actually starts with a ``Date`` field instead
        of assuming the header is line 1.
        """
        text = csv_text.lstrip("﻿")
        sample = text[:2048]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,")
        except csv.Error:
            dialect = csv.excel
        lines = text.splitlines()
        header_idx = next(
            (
                i
                for i, line in enumerate(lines)
                if line.split(dialect.delimiter, 1)[0].strip().strip('"').lower() == "date"
            ),
            0,
        )
        reader = csv.DictReader(io.StringIO("\n".join(lines[header_idx:])), dialect=dialect)
        records: list[dict] = []
        for row in reader:
            if not row:
                continue
            date_value = row.get("Date") or row.get("date")
            value_raw = row.get("Value") or row.get("value")
            if value_raw in (None, ""):
                continue
            try:
                value = float(value_raw)
            except ValueError:
                continue
            dimensions = {
                k: v for k, v in row.items() if k not in ("Date", "date", "Value", "value")
            }
            records.append(
                {"dimensions": dimensions, "observation_date": date_value, "value": value}
            )
        return records

    async def sync_cube(
        self, cube_id: str, store: PostgresStore, lang: str = "en"
    ) -> CubeSyncResult:
        """Fetch a cube only if changed (ETag-aware) and upsert new observations into Postgres."""
        stored_etag = await store.get_cube_etag(cube_id)
        csv_text, new_etag = await self.fetch_cube_csv(cube_id, lang=lang, etag=stored_etag)
        if csv_text is None:
            return CubeSyncResult(cube_id=cube_id, modified=False, records_upserted=0)

        records = self.parse_cube_csv(csv_text)
        count = await store.upsert_snb_observations(cube_id, records)
        await store.set_cube_etag(cube_id, new_etag)
        logger.info("Synced cube %s: %d records", cube_id, count)
        return CubeSyncResult(cube_id=cube_id, modified=True, records_upserted=count)

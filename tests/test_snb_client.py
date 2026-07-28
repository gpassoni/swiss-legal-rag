from unittest.mock import AsyncMock

import httpx
import pytest
import respx

from src.ingestion.snb_client import SNB_BASE_URL, SNBClient


def test_parse_cube_csv_extracts_records():
    csv_text = "Date;D0;Value\n2025-01-01;USD;0.91\n2025-01-02;USD;0.92\n"

    records = SNBClient.parse_cube_csv(csv_text)

    assert len(records) == 2
    assert records[0]["observation_date"] == "2025-01-01"
    assert records[0]["value"] == 0.91
    assert records[0]["dimensions"] == {"D0": "USD"}


def test_parse_cube_csv_skips_metadata_preamble_and_bom():
    # Real data.snb.ch exports look like this: BOM + CubeId/PublishingDate lines +
    # a blank line, before the actual header row.
    csv_text = (
        '﻿"CubeId";"rendoblim"\n'
        '"PublishingDate";"2025-09-01 14:29"\n'
        "\n"
        '"Date";"D0";"Value"\n'
        '"1988-01";"1J";"2.887"\n'
        '"1988-01";"2J";"3.218"\n'
    )

    records = SNBClient.parse_cube_csv(csv_text)

    assert len(records) == 2
    assert records[0]["observation_date"] == "1988-01"
    assert records[0]["value"] == 2.887
    assert records[0]["dimensions"] == {"D0": "1J"}


def test_parse_cube_csv_skips_rows_with_missing_value():
    csv_text = "Date;Value\n2025-01-01;\n2025-01-02;1.23\n"

    records = SNBClient.parse_cube_csv(csv_text)

    assert len(records) == 1
    assert records[0]["observation_date"] == "2025-01-02"


@pytest.mark.asyncio
async def test_fetch_cube_csv_returns_text_and_etag():
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{SNB_BASE_URL}/api/cube/rendoblim/data/csv/en").mock(
            return_value=httpx.Response(
                200, text="Date;Value\n2025-01-01;1.0\n", headers={"ETag": "abc123"}
            )
        )
        async with SNBClient() as client:
            text, etag = await client.fetch_cube_csv("rendoblim", lang="en")

    assert "2025-01-01" in text
    assert etag == "abc123"


@pytest.mark.asyncio
async def test_fetch_cube_csv_returns_none_on_304():
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{SNB_BASE_URL}/api/cube/rendoblim/data/csv/en").mock(
            return_value=httpx.Response(304)
        )
        async with SNBClient() as client:
            text, etag = await client.fetch_cube_csv("rendoblim", lang="en", etag="abc123")

    assert text is None
    assert etag == "abc123"


@pytest.mark.asyncio
async def test_sync_cube_skips_upsert_when_not_modified():
    store = AsyncMock()
    store.get_cube_etag.return_value = "abc123"

    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{SNB_BASE_URL}/api/cube/rendoblim/data/csv/en").mock(
            return_value=httpx.Response(304)
        )
        async with SNBClient() as client:
            result = await client.sync_cube("rendoblim", store, lang="en")

    assert result.modified is False
    assert result.records_upserted == 0
    store.upsert_snb_observations.assert_not_called()


@pytest.mark.asyncio
async def test_sync_cube_upserts_when_modified():
    store = AsyncMock()
    store.get_cube_etag.return_value = None
    store.upsert_snb_observations.return_value = 1

    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{SNB_BASE_URL}/api/cube/rendoblim/data/csv/en").mock(
            return_value=httpx.Response(
                200, text="Date;Value\n2025-01-01;1.0\n", headers={"ETag": "new-etag"}
            )
        )
        async with SNBClient() as client:
            result = await client.sync_cube("rendoblim", store, lang="en")

    assert result.modified is True
    assert result.records_upserted == 1
    store.set_cube_etag.assert_awaited_once_with("rendoblim", "new-etag")

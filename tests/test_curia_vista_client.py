from datetime import date

import httpx
import pytest
import respx

from src.ingestion.curia_vista_client import (
    CURIA_VISTA_BASE_URL,
    CuriaVistaClient,
    _parse_odata_date,
)


def _language_of(request: httpx.Request) -> str:
    filter_value = request.url.params.get("$filter", "")
    for lang in ("IT", "FR", "DE"):
        if f"Language eq '{lang}'" in filter_value:
            return lang
    raise AssertionError(f"unexpected $filter (missing Language clause): {filter_value!r}")


def test_parse_odata_date_converts_ms_epoch_to_iso_date():
    assert _parse_odata_date("/Date(1735689600000)/") == "2025-01-01"


def test_parse_odata_date_passes_through_unparseable_values():
    assert _parse_odata_date("not-a-date") == "not-a-date"
    assert _parse_odata_date(None) is None


@pytest.mark.asyncio
async def test_fetch_business_prefers_italian_and_strips_html():
    # The live ws.parlament.ch endpoint stores one row per (ID, Language); only the
    # Italian-language query returns anything here, the other two are empty.
    it_payload = {
        "d": {
            "results": [
                {
                    "ID": "20260046",
                    "Language": "IT",
                    "BusinessShortNumber": "26.046",
                    "Title": "Legge sulle epizoozie. Modifica",
                    "InitialSituation": (
                        "<h2>Comunicato stampa</h2><p>Il Consiglio federale "
                        "<strong>adotta</strong> il messaggio.</p>"
                    ),
                    "BusinessStatusText": "Nella Commissione del Consiglio degli Stati",
                    "SubmissionDate": "/Date(1735689600000)/",
                    "BusinessTypeName": "Oggetto del Consiglio federale",
                }
            ]
        }
    }
    empty_payload = {"d": {"results": []}}

    def _responder(request: httpx.Request) -> httpx.Response:
        lang = _language_of(request)
        return httpx.Response(200, json=it_payload if lang == "IT" else empty_payload)

    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{CURIA_VISTA_BASE_URL}/Business").mock(side_effect=_responder)
        async with CuriaVistaClient() as client:
            items = await client.fetch_business(date(2025, 1, 1), date(2025, 1, 31))

    assert len(items) == 1
    item = items[0]
    assert item.id == "20260046"
    assert item.language == "it"
    assert "<" not in item.summary and ">" not in item.summary
    assert "Consiglio federale" in item.summary
    assert "adotta" in item.summary
    assert item.submission_date == "2025-01-01"
    assert "/it/ratsbetrieb/suche-curia-vista/geschaeft?AffairId=20260046" in item.source_url


@pytest.mark.asyncio
async def test_fetch_business_falls_back_to_french_when_italian_missing():
    fr_payload = {
        "d": {
            "results": [
                {
                    "ID": "20260050",
                    "Language": "FR",
                    "BusinessShortNumber": "26.050",
                    "Title": "Motion on SR 642.11 deductions",
                    "SubmittedText": "Concerns SR 642.11 and SR 641.20.",
                    "BusinessStatusText": "Pending",
                    "SubmissionDate": "/Date(1735689600000)/",
                    "BusinessTypeName": "Motion",
                }
            ]
        }
    }
    empty_payload = {"d": {"results": []}}

    def _responder(request: httpx.Request) -> httpx.Response:
        lang = _language_of(request)
        return httpx.Response(200, json=fr_payload if lang == "FR" else empty_payload)

    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{CURIA_VISTA_BASE_URL}/Business").mock(side_effect=_responder)
        async with CuriaVistaClient() as client:
            items = await client.fetch_business(date(2025, 1, 1), date(2025, 1, 31))

    assert len(items) == 1
    assert items[0].id == "20260050"
    assert items[0].language == "fr"
    assert set(items[0].related_law_references) == {"SR 642.11", "SR 641.20"}
    assert "/fr/ratsbetrieb/" in items[0].source_url


@pytest.mark.asyncio
async def test_fetch_business_handles_plain_list_payload():
    # ws.parlament.ch's live $format=json response nests entries directly under "d"
    # as a list, not under "d.results" (the older OData verbose-JSON shape).
    de_payload = {
        "d": [
            {
                "ID": "20250002",
                "Language": "DE",
                "BusinessShortNumber": "25.002",
                "Title": "Postulat zu SR 641.20",
                "SubmittedText": "Betrifft SR 641.20.",
                "BusinessStatusText": "Pending",
                "SubmissionDate": "/Date(1735689600000)/",
                "BusinessTypeName": "Postulat",
            }
        ]
    }
    empty_payload = {"d": []}

    def _responder(request: httpx.Request) -> httpx.Response:
        lang = _language_of(request)
        return httpx.Response(200, json=de_payload if lang == "DE" else empty_payload)

    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{CURIA_VISTA_BASE_URL}/Business").mock(side_effect=_responder)
        async with CuriaVistaClient() as client:
            items = await client.fetch_business(date(2025, 1, 1), date(2025, 1, 31))

    assert len(items) == 1
    assert items[0].id == "20250002"
    assert items[0].language == "de"
    assert items[0].related_law_references == ["SR 641.20"]


@pytest.mark.asyncio
async def test_fetch_business_handles_empty_results():
    empty_payload = {"d": {"results": []}}
    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{CURIA_VISTA_BASE_URL}/Business").mock(
            return_value=httpx.Response(200, json=empty_payload)
        )
        async with CuriaVistaClient() as client:
            items = await client.fetch_business(date(2025, 1, 1), date(2025, 1, 31))

    assert items == []


@pytest.mark.asyncio
async def test_fetch_business_uses_dollar_filter_and_dollar_top():
    """Confirms the query is bounded via OData $filter/$top rather than pulling everything."""
    empty_payload = {"d": {"results": []}}
    seen_params: list[httpx.QueryParams] = []

    def _responder(request: httpx.Request) -> httpx.Response:
        seen_params.append(request.url.params)
        return httpx.Response(200, json=empty_payload)

    with respx.mock(assert_all_called=True) as mock:
        mock.get(f"{CURIA_VISTA_BASE_URL}/Business").mock(side_effect=_responder)
        async with CuriaVistaClient() as client:
            await client.fetch_business(date(2025, 1, 1), date(2025, 1, 31), top=10)

    assert len(seen_params) == 3  # one call per language in the IT -> FR -> DE priority
    for params in seen_params:
        assert params["$top"] == "10"
        assert "SubmissionDate ge" in params["$filter"]
        assert "SubmissionDate le" in params["$filter"]

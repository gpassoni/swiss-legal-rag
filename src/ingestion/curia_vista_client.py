"""OData client for Curia Vista (Swiss federal parliamentary business) via ws.parlament.ch.

The endpoint is a plain OData v2 service, simple enough that a full OData library is
unnecessary (see architecture spec §4) — this module builds filter query strings by hand
and parses the JSON response into a normalized dict.

Reference: https://www.parlament.ch/en/services/open-data-webservices
(the underlying system was replaced in July 2023; re-check that page if the endpoint
below starts returning errors, since a legacy mirror also exists at ws-old.parlament.ch).

Each ``Business`` row is keyed by ``(ID, Language)`` — the live service stores one fully
localized row per business item per language (confirmed against the live endpoint: the
same ``ID`` comes back with ``Language`` "DE"/"FR"/"IT"/"EN", each with its own
``Title``/``InitialSituation`` text). ``fetch_business`` therefore queries per language and
merges by ID, honoring the project-wide IT -> FR -> DE preference (architecture spec
"Misc") — a business item without an Italian localization yet falls back to French, then
German, rather than being dropped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
import logging

from src.processing.normalizer import clean_text, strip_html

from .base import PoliteAsyncClient

logger = logging.getLogger("swiss_legal_ai.ingestion.curia_vista")

CURIA_VISTA_BASE_URL = "https://ws.parlament.ch/odata.svc"

_ODATE_FMT = "%Y-%m-%dT00:00:00"
_ODATA_MS_RE = re.compile(r"/Date\((-?\d+)\)/")

# Project-wide language preference: try Italian first, then French, then German (see
# architecture spec "Misc"). "EN" exists on some business types but is not one of our
# four supported content languages (de/fr/it/rm), so it is never requested here.
DEFAULT_LANGUAGE_PRIORITY: tuple[str, ...] = ("IT", "FR", "DE")

_LOCALE_PATH = {"DE": "de", "FR": "fr", "IT": "it", "EN": "en"}


@dataclass
class BusinessItem:
    id: str
    short_number: str | None
    title: str
    summary: str | None
    status: str | None
    submission_date: str | None
    business_type: str | None
    language: str
    related_law_references: list[str]
    source_url: str


def _odata_datetime(d: date) -> str:
    return f"datetime'{d.strftime(_ODATE_FMT)}'"


def _parse_odata_date(raw: str | None) -> str | None:
    """Convert OData's ``/Date(1780617600000)/`` wire format into a plain ISO date
    string (UTC), matching the ISO dates Fedlex emits so both sources share one shape in
    the metadata schema."""
    if not raw:
        return None
    match = _ODATA_MS_RE.match(raw)
    if not match:
        return raw
    ms = int(match.group(1))
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date().isoformat()


def _extract_law_references(text: str | None) -> list[str]:
    """Best-effort extraction of SR-number-like references (e.g. "SR 642.11") from free text.

    This is a light heuristic, not a legal citation parser — good enough in Phase 1 to flag
    business items worth cross-linking to Fedlex acts during retrieval.
    """
    if not text:
        return []
    return sorted(set(re.findall(r"SR\s?\d{3}(?:\.\d+)*", text)))


class CuriaVistaClient:
    def __init__(self, base_url: str = CURIA_VISTA_BASE_URL, rate_limit_per_sec: int = 5) -> None:
        self._base_url = base_url
        self._client = PoliteAsyncClient(base_url=base_url, rate_limit_per_sec=rate_limit_per_sec)

    async def __aenter__(self) -> "CuriaVistaClient":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    async def fetch_business(
        self,
        date_from: date,
        date_to: date,
        top: int = 100,
        skip: int = 0,
        languages: tuple[str, ...] = DEFAULT_LANGUAGE_PRIORITY,
    ) -> list[BusinessItem]:
        """Fetch Business (Geschäfte) entities with a SubmissionDate in [date_from, date_to].

        Queries `languages` in order (default IT -> FR -> DE) and merges by business ID,
        keeping the first (highest-priority) language a given item was found in.
        """
        by_id: dict[str, BusinessItem] = {}
        for language in languages:
            items = await self._fetch_business_single_language(
                date_from, date_to, language, top=top, skip=skip
            )
            for item in items:
                by_id.setdefault(item.id, item)
        logger.info(
            "Fetched %d distinct Curia Vista business items for %s..%s (languages=%s)",
            len(by_id),
            date_from,
            date_to,
            languages,
        )
        return list(by_id.values())

    async def _fetch_business_single_language(
        self,
        date_from: date,
        date_to: date,
        language: str,
        top: int,
        skip: int,
    ) -> list[BusinessItem]:
        filter_clause = (
            f"SubmissionDate ge {_odata_datetime(date_from)} "
            f"and SubmissionDate le {_odata_datetime(date_to)} "
            f"and Language eq '{language}'"
        )
        params = {
            "$filter": filter_clause,
            "$top": str(top),
            "$skip": str(skip),
            "$format": "json",
        }
        response = await self._client.get("/Business", params=params)
        payload = response.json()
        d = payload.get("d", [])
        entries = d.get("results", []) if isinstance(d, dict) else d
        return [self._normalize(entry) for entry in entries]

    @staticmethod
    def _normalize(entry: dict) -> BusinessItem:
        raw_title = entry.get("Title") or entry.get("ShortTitle") or ""
        raw_summary = entry.get("SubmittedText") or entry.get("InitialSituation")
        title = clean_text(strip_html(raw_title))
        summary = clean_text(strip_html(raw_summary)) if raw_summary else None
        business_id = str(entry.get("ID") or entry.get("Id") or "")
        language_code = (entry.get("Language") or "DE").upper()
        locale = _LOCALE_PATH.get(language_code, "en")
        return BusinessItem(
            id=business_id,
            short_number=entry.get("BusinessShortNumber"),
            title=title,
            summary=summary,
            status=entry.get("BusinessStatusText") or entry.get("StateText"),
            submission_date=_parse_odata_date(entry.get("SubmissionDate")),
            business_type=entry.get("BusinessTypeName"),
            language=language_code.lower(),
            related_law_references=_extract_law_references(
                " ".join(filter(None, [title, summary]))
            ),
            source_url=f"https://www.parlament.ch/{locale}/ratsbetrieb/suche-curia-vista/geschaeft?AffairId={business_id}",
        )

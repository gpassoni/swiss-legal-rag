"""SPARQL client for Fedlex (fedlex.data.admin.ch), modeled with the JOLux ontology.

Fedlex's SPARQL endpoint has a real learning curve (see architecture spec §2.1), so this
client intentionally exposes a *small* set of hand-written, hardcoded queries rather than
a generic query builder:

1. ``list_consolidated_acts_by_prefix`` — enumerate consolidated (in-force) acts whose
   systematic number (SR/RS notation) starts with a given prefix (e.g. "640" for tax law).
2. ``fetch_act`` — fetch title + article-level text + validity dates for one act.

Fedlex's public ELI pages (``https://fedlex.data.admin.ch/eli/...``, no file extension) are
a JavaScript single-page app: a plain GET on one of those URLs returns an near-empty HTML
shell with a "please enable JavaScript" notice, not the legal text. The actual article text
is a *separate*, static HTML (or PDF) file hosted in Fedlex's public filestore
(``https://fedlex.data.admin.ch/filestore/...``), whose URI is only discoverable via SPARQL
(``jolux:isExemplifiedBy`` on a ``jolux:Manifestation``). So fetching real text is always a
two-step process: resolve the filestore file URI via SPARQL first (``fetch_act_file_url``),
then GET *that* URL (``fetch_act_html_text``) — never GET ``act_uri`` itself for text.

The exact JOLux predicate names below follow Fedlex's published SPARQL examples as of
this writing. Because the ontology and endpoint are maintained externally, re-validate
these queries against ``https://fedlex.data.admin.ch/sparqlendpoint`` (there is a browser
query form at that URL) before relying on them for anything beyond Phase 1 testing.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import httpx
from bs4 import BeautifulSoup
from SPARQLWrapper import JSON, SPARQLWrapper
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger("swiss_legal_ai.ingestion.fedlex")

FEDLEX_SPARQL_ENDPOINT = "https://fedlex.data.admin.ch/sparqlendpoint"

# JOLux uses EU Publications Office language authority URIs.
_LANGUAGE_URIS = {
    "de": "http://publications.europa.eu/resource/authority/language/DEU",
    "fr": "http://publications.europa.eu/resource/authority/language/FRA",
    "it": "http://publications.europa.eu/resource/authority/language/ITA",
    "rm": "http://publications.europa.eu/resource/authority/language/ROH",
}

# Project-wide language preference: try Italian first, then French, then German
# (see architecture spec "Misc" — most acts exist in all three official languages, but
# fall back down the list for the rare one that doesn't).
DEFAULT_LANGUAGE_PRIORITY: tuple[str, ...] = ("it", "fr", "de")

_SPARQL_PREFIXES = """
PREFIX jolux: <http://data.legilux.public.lu/resource/ontology/jolux#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
""".strip()

# Hardcoded, hand-validated query templates (task spec: "small, hardcoded test query set").
TEST_QUERIES: dict[str, str] = {
    # Title comes from the ConsolidationAbstract's OWN direct `jolux:isRealizedBy`
    # expression (matching the `fetch_act_metadata` query below), NOT from the dated
    # `jolux:Consolidation` member's `jolux:isRealizedBy` expression. Both exist in
    # parallel in Fedlex's JOLux data, but only the former reliably carries a real
    # `jolux:title` — the discovery, while widening ingestion beyond tax law (SR 640/642)
    # to the foundational codes (ZGB 210, OR 220, StGB 311.0, StPO 312.0, ZPO 272,
    # Constitution 101), was that those foundational codes' *dated per-consolidation*
    # expressions carry no `jolux:title` at all (confirmed empirically against the live
    # endpoint), so the original query silently returned zero acts for exactly the codes
    # this expansion needs. The per-consolidation node is still used, but only to get
    # `dateApplicability`/`dateEndApplicability` (an act's title doesn't change across its
    # dated versions, but the in-force date range does).
    "list_consolidated_acts_by_prefix": (
        _SPARQL_PREFIXES
        + """
SELECT DISTINCT ?consolidationAbstract ?srNotation ?title ?dateApplicability ?dateEndApplicability
WHERE {{
  ?consolidationAbstract a jolux:ConsolidationAbstract ;
                          jolux:classifiedByTaxonomyEntry ?taxEntry ;
                          jolux:isRealizedBy ?expression .
  ?taxEntry skos:notation ?srNotation .
  FILTER(STRSTARTS(STR(?srNotation), "{prefix}"))
  ?expression jolux:title ?title ;
              jolux:language <{language_uri}> .

  ?consolidation jolux:isMemberOf ?consolidationAbstract ;
                  jolux:dateApplicability ?dateApplicability .
  FILTER NOT EXISTS {{ ?consolidation jolux:dateEndApplicability ?end . }}
}}
ORDER BY ?srNotation
LIMIT {limit}
"""
    ),
    # dateApplicability/dateEndApplicability are properties of a jolux:Consolidation
    # (a dated "member" of the act), never of the act/ConsolidationAbstract itself — see
    # the same UNION shape used in fetch_act_file_url. Querying them directly on
    # <act_uri>, as an earlier version of this query did, silently returns nothing
    # (OPTIONAL just leaves the variable unbound) for every consolidated act, which is
    # the common case; it happened to work for the rare "simple act" shape where a Work
    # carries the dates directly.
    "fetch_act_metadata": (
        _SPARQL_PREFIXES
        + """
SELECT DISTINCT ?title ?srNotation ?dateApplicability ?dateEndApplicability ?expressionUri
WHERE {{
  <{act_uri}> jolux:classifiedByTaxonomyEntry ?taxEntry .
  ?taxEntry skos:notation ?srNotation .
  <{act_uri}> jolux:isRealizedBy ?expressionUri .
  ?expressionUri jolux:title ?title ;
                 jolux:language <{language_uri}> .
  OPTIONAL {{
    {{
      <{act_uri}> jolux:dateApplicability ?dateApplicability .
      OPTIONAL {{ <{act_uri}> jolux:dateEndApplicability ?dateEndApplicability . }}
    }}
    UNION
    {{
      # Pin dateEndApplicability's absence to the SAME in-force consolidation that
      # dateApplicability comes from — a separate OPTIONAL with its own variable would
      # let SPARQL bind dateEndApplicability from a *different*, superseded
      # consolidation of the same act, producing a valid_to earlier than valid_from.
      ?consolidation jolux:isMemberOf <{act_uri}> ;
                     jolux:dateApplicability ?dateApplicability .
      FILTER NOT EXISTS {{ ?consolidation jolux:dateEndApplicability ?end . }}
    }}
  }}
}}
"""
    ),
    # Resolves the public filestore URI of the HTML manifestation of the current in-force
    # expression of an act, in the requested language. Handles both Fedlex act shapes seen
    # live on the endpoint:
    #  - "simple" acts, where the act itself is a Work with jolux:isRealizedBy expressions
    #    directly (no separate dated Consolidation nodes), and
    #  - "consolidated" acts, where dated jolux:Consolidation versions are jolux:isMemberOf
    #    the ConsolidationAbstract, each with its own jolux:isRealizedBy expressions; the
    #    in-force one is the Consolidation with no jolux:dateEndApplicability.
    "fetch_act_file_url": (
        _SPARQL_PREFIXES
        + """
SELECT ?fileUrl
WHERE {{
  {{
    <{act_uri}> jolux:isRealizedBy ?expression .
  }}
  UNION
  {{
    ?consolidation jolux:isMemberOf <{act_uri}> ;
                   jolux:isRealizedBy ?expression .
    FILTER NOT EXISTS {{ ?consolidation jolux:dateEndApplicability ?end . }}
  }}
  ?expression jolux:language <{language_uri}> ;
              jolux:isEmbodiedBy ?manifestation .
  ?manifestation jolux:isExemplifiedBy ?fileUrl ;
                 jolux:format ?format .
  FILTER(CONTAINS(STR(?format), "HTML"))
}}
LIMIT 1
"""
    ),
}


@dataclass
class ActSummary:
    act_uri: str
    systematic_number: str
    title: str
    language: str
    valid_from: str | None
    valid_to: str | None


@dataclass
class Article:
    number: str
    text: str


@dataclass
class ActDocument:
    act_uri: str
    systematic_number: str
    title: str
    language: str
    valid_from: str | None
    valid_to: str | None
    source_url: str
    articles: list[Article] = field(default_factory=list)
    raw_text: str = ""


class FedlexClient:
    """Thin, synchronous wrapper around SPARQLWrapper for the Fedlex endpoint.

    SPARQLWrapper has no native async support, so this client stays synchronous; callers
    running inside an async pipeline should invoke it via ``asyncio.to_thread``.
    """

    def __init__(
        self,
        endpoint: str = FEDLEX_SPARQL_ENDPOINT,
        min_interval_sec: float = 0.5,
    ) -> None:
        self._wrapper = SPARQLWrapper(endpoint)
        self._wrapper.setReturnFormat(JSON)
        self._min_interval_sec = min_interval_sec
        self._last_call_ts = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call_ts
        if elapsed < self._min_interval_sec:
            time.sleep(self._min_interval_sec - elapsed)
        self._last_call_ts = time.monotonic()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def _run_query(self, query: str) -> list[dict]:
        self._throttle()
        logger.info("Running Fedlex SPARQL query (%d chars)", len(query))
        self._wrapper.setQuery(query)
        results = self._wrapper.query().convert()
        return results["results"]["bindings"]  # type: ignore[index]

    def list_consolidated_acts_by_prefix(
        self, systematic_prefix: str, language: str = "de", limit: int = 100
    ) -> list[ActSummary]:
        """List in-force (dateEndApplicability absent) consolidated acts under a systematic prefix."""
        if language not in _LANGUAGE_URIS:
            raise ValueError(f"Unsupported language: {language}")
        query = TEST_QUERIES["list_consolidated_acts_by_prefix"].format(
            prefix=systematic_prefix,
            language_uri=_LANGUAGE_URIS[language],
            limit=limit,
        )
        bindings = self._run_query(query)
        acts = []
        for row in bindings:
            acts.append(
                ActSummary(
                    act_uri=row["consolidationAbstract"]["value"],
                    systematic_number=row["srNotation"]["value"],
                    title=row["title"]["value"],
                    language=language,
                    valid_from=row.get("dateApplicability", {}).get("value"),
                    valid_to=row.get("dateEndApplicability", {}).get("value"),
                )
            )
        return acts

    def list_consolidated_acts_by_prefix_preferred(
        self,
        systematic_prefix: str,
        languages: tuple[str, ...] = DEFAULT_LANGUAGE_PRIORITY,
        limit: int = 100,
    ) -> list[ActSummary]:
        """Like ``list_consolidated_acts_by_prefix``, but merges results across
        `languages` (default IT -> FR -> DE): an act missing a title in the
        higher-priority language (rare, but possible for very recent acts) is still
        returned, using the first language in the list it *does* have a title in.
        """
        by_uri: dict[str, ActSummary] = {}
        for language in languages:
            for act in self.list_consolidated_acts_by_prefix(
                systematic_prefix, language=language, limit=limit
            ):
                by_uri.setdefault(act.act_uri, act)
        return list(by_uri.values())

    def fetch_act(self, act_uri: str, language: str = "de", include_text: bool = False) -> ActDocument:
        """Fetch title + validity metadata for one act (SPARQL), optionally followed by an
        HTML fetch of the consolidated text (`include_text=True`) for downstream chunking.
        """
        if language not in _LANGUAGE_URIS:
            raise ValueError(f"Unsupported language: {language}")
        query = TEST_QUERIES["fetch_act_metadata"].format(
            act_uri=act_uri, language_uri=_LANGUAGE_URIS[language]
        )
        bindings = self._run_query(query)
        if not bindings:
            raise LookupError(f"No metadata found for act {act_uri} in language {language}")
        row = bindings[0]
        doc = ActDocument(
            act_uri=act_uri,
            systematic_number=row["srNotation"]["value"],
            title=row["title"]["value"],
            language=language,
            valid_from=row.get("dateApplicability", {}).get("value"),
            valid_to=row.get("dateEndApplicability", {}).get("value"),
            source_url=act_uri,
        )
        if include_text:
            doc.raw_text = self.fetch_act_html_text(act_uri, language=language)
        return doc

    def fetch_act_preferred(
        self,
        act_uri: str,
        languages: tuple[str, ...] = DEFAULT_LANGUAGE_PRIORITY,
        include_text: bool = False,
    ) -> ActDocument:
        """Fetch `act_uri` trying `languages` in order (default IT -> FR -> DE), returning
        the first language that actually has metadata (and, if `include_text`, HTML text)
        for this act. Raises the last ``LookupError`` if none of `languages` match."""
        last_exc: LookupError | None = None
        for language in languages:
            try:
                return self.fetch_act(act_uri, language=language, include_text=include_text)
            except LookupError as exc:
                last_exc = exc
                continue
        assert last_exc is not None
        raise last_exc

    def fetch_act_file_url(self, act_uri: str, language: str = "de") -> str:
        """Resolve, via SPARQL, the public Fedlex filestore URI of the static HTML file
        that holds the current in-force text of `act_uri` in `language`.

        This is the actual document file (`https://fedlex.data.admin.ch/filestore/...`),
        distinct from `act_uri` itself (an ELI page served by Fedlex's JavaScript app,
        which returns no usable text on a plain GET).
        """
        if language not in _LANGUAGE_URIS:
            raise ValueError(f"Unsupported language: {language}")
        query = TEST_QUERIES["fetch_act_file_url"].format(
            act_uri=act_uri, language_uri=_LANGUAGE_URIS[language]
        )
        bindings = self._run_query(query)
        if not bindings:
            raise LookupError(
                f"No HTML filestore manifestation found for act {act_uri} in language {language}"
            )
        return bindings[0]["fileUrl"]["value"]

    def fetch_act_html_text(self, act_uri: str, language: str = "de") -> str:
        """Fetch the act's actual text, stripped to plain text.

        Two-step flow: resolve the filestore file URI via SPARQL (`fetch_act_file_url`),
        then GET *that* URL — it is a static HTML file, not a JavaScript app, so a plain GET
        + BeautifulSoup works. The exact HTML structure is not guaranteed stable across
        acts, so this is a coarse extraction meant to feed `processing.chunker` (which
        looks for "Art. N" headers), not a precise document-structure parser.
        """
        file_url = self.fetch_act_file_url(act_uri, language=language)
        self._throttle()
        headers = {"Accept": "text/html", "User-Agent": "swiss-legal-ai-phase1/0.1"}
        response = httpx.get(file_url, headers=headers, follow_redirects=True, timeout=30.0)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        return soup.get_text(separator="\n")

    def close(self) -> None:
        # SPARQLWrapper does not hold a persistent connection that needs explicit closing.
        pass

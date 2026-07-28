"""Best-effort extraction of article cross-references (e.g. "art. 97 CO", "art. 21a",
"articolo 83 capoverso 2 CC") from already-chunked Fedlex article text.

Runs *after* chunking (`processing.chunker.chunk_act_text`), per article, not on raw
pre-chunk HTML — chunk boundaries already tell us the "from" article, and the text has
already been through `processing.normalizer.clean_text`.

Design principle (deliberate, matches `orchestration.citation_verifier`'s philosophy):
better to under-extract than to fabricate a link. A bare "art. N" with nothing
recognizable trailing it is treated as an *internal* reference (same law) — by far the
most common case in Swiss statutory text. A trailing token that looks abbreviation-shaped
but isn't in `law_abbreviations.LAW_ABBREVIATIONS` is stored as an *unresolved* external
reference (`to_systematic_number=None`) rather than guessed at or discarded — it can be
re-resolved later (a new code added to `LAW_ABBREVIATIONS`) without re-parsing the source
text. Coverage across DE/FR/IT stylistic variation is intentionally incomplete; this is
not a document-structure parser.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .law_abbreviations import LAW_ABBREVIATIONS

# "Art."/"Artikel" (DE), "Art."/"Article" (FR), "Art."/"Articolo" (IT) — all three also
# accept the bare number directly attached with no space (observed in real Fedlex text
# after HTML-to-text stripping, e.g. "articoli280").
_ARTICLE_HEADER: dict[str, str] = {
    "it": r"art(?:\.|icolo|icoli)?",
    "fr": r"art(?:\.|icle)?",
    "de": r"art(?:\.|ikel)?",
}

# How far past the end of the article-number match to look for a trailing law
# abbreviation. Sized from real examples (e.g. "articolo83\nb\n capoverso2 CC" — HTML
# sup-tag stripping inserts newlines/whitespace between the number and a paragraph
# marker before the abbreviation).
_LOOKAHEAD_WINDOW = 30

# Longest-first so "CPC"/"CPP"/"StGB"/"StPO" aren't shadowed by a shorter key ("CP").
_KNOWN_ABBREV_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(k) for k in sorted(LAW_ABBREVIATIONS, key=len, reverse=True)) + r")\b"
)
# Generic "looks like an abbreviation" shape for laws not (yet) in LAW_ABBREVIATIONS:
# a short run of uppercase letters (CO, ZGB, ...), optionally with a trailing period.
# Does not match mixed-case shapes like "StGB" that aren't already known — accepted gap.
_GENERIC_ABBREV_SHAPE = re.compile(r"\b[A-Z]{2,6}\.?\b")


@dataclass
class ArticleReference:
    from_systematic_number: str
    from_article: str
    to_systematic_number: str | None
    to_article: str
    raw_reference_text: str
    language: str
    source_url: str


def extract_references(
    chunk_text: str,
    from_systematic_number: str,
    from_article: str,
    language: str,
    source_url: str,
) -> list[ArticleReference]:
    header = _ARTICLE_HEADER.get(language)
    if header is None:
        return []

    pattern = re.compile(rf"\b{header}\s*(\d+[a-z]?)", re.IGNORECASE)
    refs: list[ArticleReference] = []
    seen: set[tuple[str | None, str]] = set()

    for m in pattern.finditer(chunk_text):
        to_article = m.group(1)
        window_start = m.end()
        window = chunk_text[window_start : window_start + _LOOKAHEAD_WINDOW]

        known_match = _KNOWN_ABBREV_PATTERN.search(window)
        if known_match:
            to_systematic_number: str | None = LAW_ABBREVIATIONS[known_match.group(0)]
            raw_end = window_start + known_match.end()
        else:
            generic_match = _GENERIC_ABBREV_SHAPE.search(window)
            if generic_match:
                # Abbreviation-shaped but not in our lookup table — store unresolved
                # rather than guessing, or discarding, or assuming "internal".
                to_systematic_number = None
                raw_end = window_start + generic_match.end()
            else:
                # Nothing abbreviation-shaped nearby: treat as internal (same law).
                to_systematic_number = from_systematic_number
                raw_end = m.end()

        if to_systematic_number == from_systematic_number and to_article == from_article:
            continue  # the article's own header re-matched, or a self-mention

        key = (to_systematic_number, to_article)
        if key in seen:
            continue
        seen.add(key)

        refs.append(
            ArticleReference(
                from_systematic_number=from_systematic_number,
                from_article=from_article,
                to_systematic_number=to_systematic_number,
                to_article=to_article,
                raw_reference_text=chunk_text[m.start() : raw_end].strip(),
                language=language,
                source_url=source_url,
            )
        )

    return refs

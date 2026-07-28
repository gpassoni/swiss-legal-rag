"""Deterministic citation verification (architecture spec §3.6).

After the LLM generates an answer, this module parses the citations it produced and
confirms each one matches a chunk actually present in the retrieved context. This is a
hard filter, not a suggestion: citations that don't match a retrieved chunk are flagged
as unverified and must not be presented to the user as sources.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.storage.qdrant_store import SearchResult

_CITATION_RE = re.compile(r"\(Art\.\s*([^,()]+),\s*(https?://[^\s)]+)\)")


@dataclass
class Citation:
    article: str
    source_url: str
    verified: bool


@dataclass
class VerificationResult:
    citations: list[Citation]
    all_verified: bool

    @property
    def verified_citations(self) -> list[Citation]:
        return [c for c in self.citations if c.verified]

    @property
    def unverified_citations(self) -> list[Citation]:
        return [c for c in self.citations if not c.verified]


def extract_citations(answer: str) -> list[tuple[str, str]]:
    """Extract (article, source_url) pairs from the model's answer text."""
    return [(article.strip(), url.strip()) for article, url in _CITATION_RE.findall(answer)]


def verify_citations(answer: str, retrieved_chunks: list[SearchResult]) -> VerificationResult:
    """Check every citation in `answer` against the chunks actually retrieved for this query."""
    known = {
        (str(chunk.metadata.get("article") or "n/a"), chunk.metadata.get("source_url", ""))
        for chunk in retrieved_chunks
    }
    citations = []
    for article, source_url in extract_citations(answer):
        verified = (article, source_url) in known
        citations.append(Citation(article=article, source_url=source_url, verified=verified))
    all_verified = all(c.verified for c in citations) if citations else False
    return VerificationResult(citations=citations, all_verified=all_verified)

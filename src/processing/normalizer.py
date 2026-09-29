"""Text cleanup and language detection shared by the chunking pipeline."""

from __future__ import annotations

import re
import unicodedata

from bs4 import BeautifulSoup
from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0  # deterministic language detection

_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_SUPPORTED_LANGUAGES = {"de", "fr", "it", "rm"}


def strip_html(text: str | None) -> str:
    """Strip HTML tags/entities, returning plain text.

    Some sources (e.g. Curia Vista's ``InitialSituation``/``SubmittedText`` fields) come
    back as HTML fragments (``<p>``, ``<h2>``, ``&nbsp;``, embedded ``<a>`` links) rather
    than plain text — unlike Fedlex's already-clean article text, these must be stripped
    before chunking/embedding, or the model would embed markup instead of legal content.
    """
    if not text:
        return ""
    return BeautifulSoup(text, "html.parser").get_text(separator=" ")


def clean_text(text: str) -> str:
    """Normalize unicode, collapse repeated whitespace, strip control characters."""
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if ch == "\n" or ch.isprintable())
    text = _WHITESPACE_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def detect_language(text: str, default: str = "de") -> str:
    """Detect DE/FR/IT/RM; falls back to `default` if detection fails or is out of scope.

    langdetect doesn't distinguish Romansh (rm) reliably, so anything it can't map onto
    our four supported languages falls back to `default` rather than guessing.
    """
    sample = text.strip()
    if not sample:
        return default
    try:
        detected = detect(sample[:2000])
    except LangDetectException:
        return default
    return detected if detected in _SUPPORTED_LANGUAGES else default

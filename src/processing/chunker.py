"""Article/paragraph-level chunking for Fedlex act text.

Chunks are cut by legal/logical unit (article, or paragraph within an over-long article),
never by fixed character count — see architecture spec §3.1.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .normalizer import clean_text

# "Art." is the abbreviation used for "Artikel" (de), "Article" (fr/it share "Art."),
# at the start of a line, followed by a number and optional letter suffix (e.g. "21a").
_ARTICLE_HEADER_RE = re.compile(r"^Art\.\s*(\d+[a-z]*)\s*", re.MULTILINE)

MAX_CHUNK_TOKENS = 1500


@dataclass
class Chunk:
    article: str | None
    text: str
    paragraph_index: int | None = None  # set when an article was itself split further


def _approx_token_count(text: str) -> int:
    """Cheap whitespace-based proxy for token count — good enough to decide when an
    article needs to be split further, without pulling in a tokenizer dependency."""
    return len(text.split())


def _split_long_article(article: str, text: str, max_tokens: int) -> list[Chunk]:
    """Split an over-long article into paragraph-level chunks, keeping article numbering."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) <= 1:
        # No paragraph breaks to exploit — return as a single (oversized) chunk rather
        # than cutting mid-sentence, which would break the "legal unit" chunking rule.
        return [Chunk(article=article, text=text)]

    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_tokens = 0
    part = 1
    for paragraph in paragraphs:
        paragraph_tokens = _approx_token_count(paragraph)
        if buffer and buffer_tokens + paragraph_tokens > max_tokens:
            chunks.append(
                Chunk(article=article, text="\n\n".join(buffer), paragraph_index=part)
            )
            part += 1
            buffer = []
            buffer_tokens = 0
        buffer.append(paragraph)
        buffer_tokens += paragraph_tokens
    if buffer:
        chunks.append(Chunk(article=article, text="\n\n".join(buffer), paragraph_index=part))
    return chunks


def chunk_act_text(raw_text: str, max_chunk_tokens: int = MAX_CHUNK_TOKENS) -> list[Chunk]:
    """Split a full act's text into one chunk per article (or per paragraph group for
    articles longer than `max_chunk_tokens`).

    Any text preceding the first "Art." header (title, ingress, table of contents) is kept
    as a single preamble chunk with `article=None`.
    """
    text = clean_text(raw_text)
    matches = list(_ARTICLE_HEADER_RE.finditer(text))

    chunks: list[Chunk] = []
    if not matches:
        if text:
            chunks.append(Chunk(article=None, text=text))
        return chunks

    preamble = text[: matches[0].start()].strip()
    if preamble:
        chunks.append(Chunk(article=None, text=preamble))

    for i, match in enumerate(matches):
        article_number = match.group(1)
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        article_text = text[start:end].strip()

        if _approx_token_count(article_text) > max_chunk_tokens:
            chunks.extend(_split_long_article(article_number, article_text, max_chunk_tokens))
        else:
            chunks.append(Chunk(article=article_number, text=article_text))

    return chunks


def chunk_business_item(
    short_number: str | None,
    title: str,
    status: str | None,
    summary: str | None,
) -> Chunk:
    """Build the single synthetic chunk for one Curia Vista business item (architecture
    spec §3.1): a Geschäft isn't naturally split into legal sub-units the way a Fedlex act
    is, so it gets one chunk combining its short number, title, status, and free text —
    the item's full raw fields are stored separately in Postgres, not in this chunk.
    """
    parts = [
        f"{short_number} — {title}" if short_number else title,
        f"Status: {status}" if status else None,
        summary,
    ]
    text = clean_text("\n\n".join(p for p in parts if p))
    return Chunk(article=short_number, text=text)

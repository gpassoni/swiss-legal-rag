"""Builds the metadata schema (architecture spec §3.2) attached to every chunk before embedding."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel

from src.config import label_for_systematic_number

from .chunker import Chunk

Source = Literal["fedlex", "curia_vista"]


class ChunkMetadata(BaseModel):
    source: Source
    level: Literal["federal"] = "federal"
    law_short_name: str | None = None
    article: str | None = None
    systematic_number: str | None = None
    # Coarse, non-authoritative area-of-law label (e.g. "civile", "penale") derived from
    # `systematic_number` via `label_for_systematic_number` — a UI/API filtering
    # convenience, not a legal taxonomy. None when the systematic number doesn't map to
    # a known prefix (e.g. non-Fedlex sources, or a law outside the configured scope).
    area_of_law: str | None = None
    language: Literal["de", "fr", "it", "rm"]
    valid_from: str | None = None
    valid_to: str | None = None
    source_url: str
    ingested_at: str
    content_hash: str


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_metadata(
    chunk: Chunk,
    *,
    source: Source,
    language: str,
    source_url: str,
    law_short_name: str | None = None,
    systematic_number: str | None = None,
    valid_from: str | None = None,
    valid_to: str | None = None,
    ingested_at: datetime | None = None,
) -> ChunkMetadata:
    """Build the metadata payload for one chunk, computing its content hash and timestamp."""
    ts = ingested_at or datetime.now(UTC)
    return ChunkMetadata(
        source=source,
        law_short_name=law_short_name,
        article=chunk.article,
        systematic_number=systematic_number,
        area_of_law=label_for_systematic_number(systematic_number),
        language=language,  # type: ignore[arg-type]
        valid_from=valid_from,
        valid_to=valid_to,
        source_url=source_url,
        ingested_at=ts.isoformat(),
        content_hash=content_hash(chunk.text),
    )

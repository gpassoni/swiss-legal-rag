"""Local cross-encoder reranker: closes the top-50 -> top-5/8 gap (architecture spec §3.5)."""
from __future__ import annotations

import logging
from functools import lru_cache

from src.config import get_settings
from src.storage.qdrant_store import SearchResult

logger = logging.getLogger("swiss_legal_ai.retrieval.reranker")

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


class Reranker:
    """Wraps a sentence-transformers CrossEncoder, loaded lazily on first use."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: str | None = None,
        max_length: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        settings = get_settings()
        self._model_name = model_name
        self._device = device
        self._max_length = max_length or settings.reranker_max_length
        self._batch_size = batch_size or settings.reranker_batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            logger.info(
                "Loading reranker model %s (max_length=%d, batch_size=%d)",
                self._model_name,
                self._max_length,
                self._batch_size,
            )
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self._model_name, device=self._device, max_length=self._max_length
            )
        return self._model

    def rerank(
        self, query: str, candidates: list[SearchResult], top_k: int = 8
    ) -> list[SearchResult]:
        """Score each candidate against `query` and return the top_k, sorted by rerank score."""
        if not candidates:
            return []
        pairs = [(query, candidate.text) for candidate in candidates]
        scores = self.model.predict(
            pairs, batch_size=self._batch_size, show_progress_bar=False
        )
        reranked = sorted(
            zip(candidates, scores), key=lambda pair: pair[1], reverse=True
        )[:top_k]
        results = []
        for candidate, score in reranked:
            results.append(
                SearchResult(
                    id=candidate.id,
                    score=float(score),
                    text=candidate.text,
                    metadata=candidate.metadata,
                )
            )
        return results


@lru_cache
def get_reranker() -> Reranker:
    return Reranker()

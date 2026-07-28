"""Batch embedding service: dense vectors via jina-embeddings-v3, plus a lightweight
hashed-BM25-style sparse representation for Qdrant's native hybrid (dense+sparse) search.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from src.config import get_settings

logger = logging.getLogger("swiss_legal_ai.embedding")

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
DEFAULT_SPARSE_DIM = 2**18  # hashing-trick vocabulary size


@dataclass
class SparseVector:
    indices: list[int]
    values: list[float]


@dataclass
class EmbeddedChunk:
    dense: list[float]
    sparse: SparseVector


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _hash_token(token: str, dim: int) -> int:
    return hash(token) % dim


class EmbedService:
    """Wraps a sentence-transformers model. The model is loaded lazily on first use so
    importing this module (e.g. from tests) doesn't require downloading weights."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        sparse_dim: int = DEFAULT_SPARSE_DIM,
    ) -> None:
        settings = get_settings()
        self._model_name = model_name or settings.embedding_model
        self._device = device or settings.embedding_device
        self._sparse_dim = sparse_dim
        self._model = None

    @property
    def model(self):
        if self._model is None:
            logger.info("Loading embedding model %s on %s", self._model_name, self._device)
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self._model_name, device=self._device, trust_remote_code=True
            )
        return self._model

    def embed_dense(
        self,
        texts: list[str],
        task: Literal["retrieval.query", "retrieval.passage"] = "retrieval.passage",
        batch_size: int | None = None,
    ) -> list[list[float]]:
        if batch_size is None:
            batch_size = get_settings().embedding_batch_size
        if not texts:
            return []
        model_name_lower = self._model_name.lower()
        if model_name_lower.startswith("jinaai/"):
            # jina-embeddings-v3's custom remote code takes an explicit task adapter.
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                task=task,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        elif "e5" in model_name_lower:
            # Standard sentence-transformers models have no task parameter; e5-style
            # "query: "/"passage: " prefixing is the documented way to get the same
            # asymmetric retrieval benefit for models like intfloat/multilingual-e5-*.
            prefix = "query: " if task == "retrieval.query" else "passage: "
            embeddings = self.model.encode(
                [prefix + t for t in texts],
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        else:
            # e.g. BAAI/bge-m3: prefix-free multilingual retrieval, symmetric
            # query/passage encoding — no special task handling needed.
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        return [vec.tolist() for vec in embeddings]

    def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        """Hashed term-frequency sparse vectors (BM25-style saturation, no IDF corpus
        state needed) — good enough for Qdrant's sparse index without a separate fitted
        vocabulary or an extra heavyweight dependency."""
        results = []
        for text in texts:
            tokens = _tokenize(text)
            counts = Counter(_hash_token(tok, self._sparse_dim) for tok in tokens)
            indices = sorted(counts)
            # log-saturated term frequency, similar in spirit to BM25's TF component
            values = [1.0 + math.log(counts[idx]) for idx in indices]
            results.append(SparseVector(indices=indices, values=values))
        return results

    def embed_batch(
        self,
        texts: list[str],
        task: Literal["retrieval.query", "retrieval.passage"] = "retrieval.passage",
    ) -> list[EmbeddedChunk]:
        dense_vectors = self.embed_dense(texts, task=task)
        sparse_vectors = self.embed_sparse(texts)
        return [
            EmbeddedChunk(dense=d, sparse=s) for d, s in zip(dense_vectors, sparse_vectors)
        ]


@lru_cache
def get_embed_service() -> EmbedService:
    return EmbedService()

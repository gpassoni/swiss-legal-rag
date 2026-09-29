"""Batch embedding service: dense vectors via jina-embeddings-v3, plus a lightweight
hashed-BM25-style sparse representation for Qdrant's native hybrid (dense+sparse) search.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

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
    """Deterministic token -> index hash for the sparse vector's hashing trick.

    Must be stable across separate Python processes (the ingestion process that embeds
    passages and the API process that embeds queries at request time are frequently not
    the same process, and a long-running server gets restarted). Python's builtin
    `hash()` on `str` is randomized per-process by default (`PYTHONHASHSEED`), which would
    silently map the same token to a different index in each process — breaking the
    sparse half of hybrid search without raising any error. blake2b is deterministic
    across processes/machines regardless of hash-seed randomization.
    """
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


def _approx_token_len(text: str) -> int:
    return len(text.split()) or 1


def _length_bucketed_batches(
    lengths: list[int], batch_size: int, max_tokens_per_batch: int | None
) -> list[list[int]]:
    """Group text indices into sub-batches for `SentenceTransformer.encode()`, bounded by
    both an item-count cap (`batch_size`) and, if given, an approximate total-token cap
    (`max_tokens_per_batch`).

    `encode()` already sorts its input by length internally before forming batches, but
    only within a single call — with a large enough `batch_size` (e.g. once raised for
    GPU throughput), that still allows one sub-batch to end up entirely made of long
    outlier chunks, which is what previously forced `EMBEDDING_BATCH_SIZE` to stay low
    across the board. Sorting here first and applying the token budget lets short/medium
    chunks batch at the full configured size while long outliers automatically split into
    smaller sub-batches instead of a fixed small batch size everywhere.
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    for idx in order:
        length = lengths[idx]
        would_exceed_tokens = (
            max_tokens_per_batch is not None
            and current
            and current_tokens + length > max_tokens_per_batch
        )
        if current and (len(current) >= batch_size or would_exceed_tokens):
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(idx)
        current_tokens += length
    if current:
        batches.append(current)
    return batches


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
        max_tokens_per_batch: int | None = None,
    ) -> list[list[float]]:
        settings = get_settings()
        if batch_size is None:
            batch_size = settings.embedding_batch_size
        if max_tokens_per_batch is None:
            max_tokens_per_batch = settings.embedding_max_tokens_per_batch
        if not texts:
            return []

        model_name_lower = self._model_name.lower()
        encode_kwargs: dict[str, Any] = {}
        if model_name_lower.startswith("jinaai/"):
            # jina-embeddings-v3's custom remote code takes an explicit task adapter.
            prepared = texts
            encode_kwargs["task"] = task
        elif "e5" in model_name_lower:
            # Standard sentence-transformers models have no task parameter; e5-style
            # "query: "/"passage: " prefixing is the documented way to get the same
            # asymmetric retrieval benefit for models like intfloat/multilingual-e5-*.
            prefix = "query: " if task == "retrieval.query" else "passage: "
            prepared = [prefix + t for t in texts]
        else:
            # e.g. BAAI/bge-m3: prefix-free multilingual retrieval, symmetric
            # query/passage encoding — no special task handling needed.
            prepared = texts

        lengths = [_approx_token_len(t) for t in prepared]
        batches = _length_bucketed_batches(lengths, batch_size, max_tokens_per_batch)

        embeddings: list[Any] = [None] * len(prepared)
        for batch_indices in batches:
            batch_texts = [prepared[i] for i in batch_indices]
            batch_embeddings = self.model.encode(
                batch_texts,
                batch_size=len(batch_texts),
                show_progress_bar=False,
                convert_to_numpy=True,
                **encode_kwargs,
            )
            for i, vec in zip(batch_indices, batch_embeddings):
                embeddings[i] = vec
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
        return [EmbeddedChunk(dense=d, sparse=s) for d, s in zip(dense_vectors, sparse_vectors)]


@lru_cache
def get_embed_service() -> EmbedService:
    return EmbedService()

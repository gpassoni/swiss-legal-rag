"""Ties the embedding service and the Qdrant store together: embed a text query (dense +
sparse) and run the hybrid search against the collection.
"""
from __future__ import annotations

from typing import Any

from src.embedding.embed_service import EmbedService
from src.storage.qdrant_store import QdrantStore, SearchResult


def search(
    query: str,
    embed_service: EmbedService,
    store: QdrantStore,
    filters: dict[str, Any] | None = None,
    limit: int = 50,
) -> list[SearchResult]:
    """Embed `query` and run hybrid dense+sparse retrieval, pre-filtered by `filters`."""
    dense = embed_service.embed_dense([query], task="retrieval.query")[0]
    sparse = embed_service.embed_sparse([query])[0]
    return store.hybrid_search(
        query_dense=dense,
        query_sparse_indices=sparse.indices,
        query_sparse_values=sparse.values,
        filters=filters,
        limit=limit,
    )

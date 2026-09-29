"""Ties the embedding service and the Qdrant store together: embed a text query (dense +
sparse) and run the hybrid search against the collection.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from src.embedding.embed_service import EmbedService
from src.storage.qdrant_store import QdrantStore, SearchResult


def search(
    query: str,
    embed_service: EmbedService,
    store: QdrantStore,
    filters: dict[str, Any] | None = None,
    limit: int = 50,
    in_force_on: date | bool | None = True,
) -> list[SearchResult]:
    """Embed `query` and run hybrid dense+sparse retrieval, pre-filtered by `filters`.

    `in_force_on` (default `True`, meaning "today") excludes chunks not in force on that
    date — see `QdrantStore.hybrid_search`. Pass `False`/`None` to search across
    historical/expired versions too.
    """
    dense = embed_service.embed_dense([query], task="retrieval.query")[0]
    sparse = embed_service.embed_sparse([query])[0]
    return store.hybrid_search(
        query_dense=dense,
        query_sparse_indices=sparse.indices,
        query_sparse_values=sparse.values,
        filters=filters,
        limit=limit,
        in_force_on=in_force_on,
    )

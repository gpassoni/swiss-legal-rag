"""Qdrant vector store: collection setup, chunk upsert, hybrid (dense+sparse) search
with metadata filtering — see architecture spec §3.3/§3.5.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models

from src.config import get_settings
from src.embedding.embed_service import EmbeddedChunk

logger = logging.getLogger("swiss_legal_ai.storage.qdrant")

DEFAULT_COLLECTION = "swiss_legal_chunks"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


@dataclass
class ChunkRecord:
    text: str
    metadata: dict[str, Any]
    embedding: EmbeddedChunk


@dataclass
class SearchResult:
    id: str
    score: float
    text: str
    metadata: dict[str, Any]


def point_id_for(content_hash: str) -> str:
    """Deterministic UUID from a chunk's content hash, so re-ingesting unchanged content
    is a no-op upsert rather than a duplicate point."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, content_hash))


class QdrantStore:
    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        settings = get_settings()
        self._collection = collection_name
        self._client = QdrantClient(
            host=host or settings.qdrant_host, port=port or settings.qdrant_port
        )

    def create_collection(self, dense_dim: int, recreate: bool = False) -> None:
        exists = self._client.collection_exists(self._collection)
        if exists and not recreate:
            logger.info("Qdrant collection %s already exists", self._collection)
            return
        if exists and recreate:
            self._client.delete_collection(self._collection)

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=dense_dim, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )
        for field_name, schema in (
            ("source", models.PayloadSchemaType.KEYWORD),
            ("language", models.PayloadSchemaType.KEYWORD),
            ("systematic_number", models.PayloadSchemaType.KEYWORD),
            ("law_short_name", models.PayloadSchemaType.KEYWORD),
            ("valid_to", models.PayloadSchemaType.KEYWORD),
            ("area_of_law", models.PayloadSchemaType.KEYWORD),
            ("article", models.PayloadSchemaType.KEYWORD),
        ):
            self._client.create_payload_index(
                collection_name=self._collection, field_name=field_name, field_schema=schema
            )
        logger.info("Created Qdrant collection %s (dense_dim=%d)", self._collection, dense_dim)

    # Points per `upsert()` HTTP call. Each point carries a dense vector (~4 bytes/dim
    # as JSON floats, larger than that in text form), a sparse vector, chunk text, and
    # metadata — a single request covering many codes at once can otherwise exceed
    # Qdrant's default 32MB request-body limit (observed: 8 Fedlex codes in one batch
    # produced a ~119MB payload and a 400 Bad Request). 256 points comfortably stays
    # under that limit even for longer chunks; tune down further only if upserting
    # unusually large payloads (e.g. a much bigger embedding model/dimension).
    UPSERT_BATCH_SIZE = 256

    def upsert_chunks(self, records: list[ChunkRecord]) -> int:
        if not records:
            return 0
        points = []
        for record in records:
            content_hash = record.metadata.get("content_hash")
            if not content_hash:
                raise ValueError("Each chunk's metadata must include content_hash")
            points.append(
                models.PointStruct(
                    id=point_id_for(content_hash),
                    vector={
                        DENSE_VECTOR_NAME: record.embedding.dense,
                        SPARSE_VECTOR_NAME: models.SparseVector(
                            indices=record.embedding.sparse.indices,
                            values=record.embedding.sparse.values,
                        ),
                    },
                    payload={"text": record.text, **record.metadata},
                )
            )
        for i in range(0, len(points), self.UPSERT_BATCH_SIZE):
            batch = points[i : i + self.UPSERT_BATCH_SIZE]
            self._client.upsert(collection_name=self._collection, points=batch)
        logger.info("Upserted %d chunks into %s", len(points), self._collection)
        return len(points)

    def get_by_article(self, systematic_number: str, article: str) -> SearchResult | None:
        """Exact-match lookup of a single chunk by (systematic_number, article), with no
        vector search — used for 1-hop cross-reference expansion (see
        `processing.reference_extractor` / `orchestration.query_engine`), where the target
        of a reference is already known and only needs to be fetched, not searched for."""
        points, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="systematic_number", match=models.MatchValue(value=systematic_number)
                    ),
                    models.FieldCondition(
                        key="article", match=models.MatchValue(value=article)
                    ),
                ]
            ),
            limit=1,
            with_payload=True,
        )
        if not points:
            return None
        point = points[0]
        payload = dict(point.payload or {})
        text = payload.pop("text", "")
        return SearchResult(id=str(point.id), score=1.0, text=text, metadata=payload)

    @staticmethod
    def _build_filter(filters: dict[str, Any] | None) -> models.Filter | None:
        if not filters:
            return None
        must = [
            models.FieldCondition(key=key, match=models.MatchValue(value=value))
            for key, value in filters.items()
        ]
        return models.Filter(must=must)

    def hybrid_search(
        self,
        query_dense: list[float],
        query_sparse_indices: list[int],
        query_sparse_values: list[float],
        filters: dict[str, Any] | None = None,
        limit: int = 8,
        prefetch_limit: int = 50,
    ) -> list[SearchResult]:
        """Dense + sparse candidates fused with Reciprocal Rank Fusion, metadata pre-filtered."""
        qdrant_filter = self._build_filter(filters)
        response = self._client.query_points(
            collection_name=self._collection,
            prefetch=[
                models.Prefetch(
                    query=query_dense,
                    using=DENSE_VECTOR_NAME,
                    filter=qdrant_filter,
                    limit=prefetch_limit,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=query_sparse_indices, values=query_sparse_values
                    ),
                    using=SPARSE_VECTOR_NAME,
                    filter=qdrant_filter,
                    limit=prefetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=qdrant_filter,
            limit=limit,
            with_payload=True,
        )
        results = []
        for point in response.points:
            payload = dict(point.payload or {})
            text = payload.pop("text", "")
            results.append(
                SearchResult(id=str(point.id), score=point.score, text=text, metadata=payload)
            )
        return results

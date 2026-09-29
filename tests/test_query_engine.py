import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.embedding.embed_service import SparseVector
from src.orchestration.query_engine import QueryEngine
from src.storage.qdrant_store import SearchResult


def _chunk(article="21", source_url="https://fedlex.example/act/1", words=0):
    text = " ".join(f"word{i}" for i in range(words)) if words else "Art. 21 text"
    return SearchResult(
        id=article,
        score=0.9,
        text=text,
        metadata={"article": article, "source_url": source_url, "systematic_number": "642.11"},
    )


def _make_engine(llm_answer: str, chunks: list[SearchResult]):
    embed_service = MagicMock()
    embed_service.embed_dense.return_value = [[0.1]]
    embed_service.embed_sparse.return_value = [SparseVector(indices=[], values=[])]

    store = MagicMock()
    store.hybrid_search.return_value = chunks

    reranker = MagicMock()
    reranker.rerank.return_value = chunks

    llm_client = MagicMock()
    llm_client.complete = AsyncMock(return_value=llm_answer)
    llm_client.last_usage = None
    llm_client.model = "test-model"

    return QueryEngine(embed_service, store, reranker, llm_client, postgres_store=None)


@pytest.mark.asyncio
async def test_answer_has_citations_true_when_answer_cites_retrieved_chunk():
    chunks = [_chunk()]
    engine = _make_engine("Deductions apply (Art. 21, https://fedlex.example/act/1).", chunks)

    result = await engine.answer("question?")

    assert result.has_citations is True
    assert result.unverified_citation_count == 0
    assert len(result.verified_sources) == 1


@pytest.mark.asyncio
async def test_answer_has_citations_false_when_answer_cites_nothing():
    # Regression for the P0 bug: chunks were retrieved and shown to the model, but the
    # generated answer contains no citation at all. unverified_citation_count is still 0
    # here (nothing to be "unverified") — has_citations is the only field that flags it.
    chunks = [_chunk()]
    engine = _make_engine("This is a general statement with no citation.", chunks)

    result = await engine.answer("question?")

    assert result.unverified_citation_count == 0
    assert result.has_citations is False


@pytest.mark.asyncio
async def test_answer_has_citations_true_even_when_citation_is_hallucinated():
    chunks = [_chunk()]
    engine = _make_engine("Deductions apply (Art. 99, https://fedlex.example/act/999).", chunks)

    result = await engine.answer("question?")

    assert result.has_citations is True
    assert result.unverified_citation_count == 1


@pytest.mark.asyncio
async def test_answer_drops_low_priority_chunks_over_a_tight_token_budget():
    kept = _chunk(article="1", words=5)
    dropped = _chunk(article="99", words=5000)
    engine = _make_engine("A general answer with no citation.", [kept, dropped])

    with patch("src.orchestration.query_engine.get_settings") as mock_settings:
        mock_settings.return_value.max_prompt_context_tokens = 50
        result = await engine.answer("question?")

    assert result.dropped_chunk_count == 1
    assert len(result.chunks) == 1
    assert result.chunks[0].metadata["article"] == "1"


@pytest.mark.asyncio
async def test_answer_excludes_dropped_chunk_from_citation_verification():
    # Regression: a citation matching a chunk that got cut by the token budget must not
    # verify as legitimate — that chunk was never actually shown to the model.
    kept = _chunk(article="1", words=5)
    dropped = _chunk(article="99", words=5000)
    engine = _make_engine("See (Art. 99, https://fedlex.example/act/1).", [kept, dropped])

    with patch("src.orchestration.query_engine.get_settings") as mock_settings:
        mock_settings.return_value.max_prompt_context_tokens = 50
        result = await engine.answer("question?")

    assert result.dropped_chunk_count == 1
    assert result.unverified_citation_count == 1
    assert result.verified_sources == []


@pytest.mark.asyncio
async def test_answer_does_not_serialize_concurrent_requests_during_rerank():
    # Regression: hybrid_search.search / reranker.rerank used to run synchronously inside
    # `async def answer()`, blocking the whole event loop for their duration. Wrapped in
    # asyncio.to_thread, two concurrent answer() calls should overlap instead of the
    # second waiting for the first's blocking work to finish on the event loop.
    chunks = [_chunk()]

    def slow_rerank(question, candidates, top_k=8):
        time.sleep(0.2)
        return candidates

    embed_service = MagicMock()
    embed_service.embed_dense.return_value = [[0.1]]
    embed_service.embed_sparse.return_value = [SparseVector(indices=[], values=[])]
    store = MagicMock()
    store.hybrid_search.return_value = chunks
    reranker = MagicMock()
    reranker.rerank.side_effect = slow_rerank
    llm_client = MagicMock()
    llm_client.complete = AsyncMock(return_value="A general answer with no citation.")
    llm_client.last_usage = None
    llm_client.model = "test-model"

    engine = QueryEngine(embed_service, store, reranker, llm_client, postgres_store=None)

    start = time.perf_counter()
    await asyncio.gather(engine.answer("q1"), engine.answer("q2"))
    elapsed = time.perf_counter() - start

    # Two serialized 0.2s reranks would take >=0.4s; overlapping ones stay well under
    # that even with thread-pool scheduling overhead.
    assert elapsed < 0.35

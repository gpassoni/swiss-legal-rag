from unittest.mock import MagicMock

from src.embedding.embed_service import SparseVector
from src.retrieval import hybrid_search
from src.storage.qdrant_store import SearchResult


def test_search_embeds_query_and_delegates_to_store():
    embed_service = MagicMock()
    embed_service.embed_dense.return_value = [[0.1, 0.2, 0.3]]
    embed_service.embed_sparse.return_value = [SparseVector(indices=[1, 5], values=[0.9, 0.4])]

    expected_results = [
        SearchResult(id="1", score=0.87, text="Art. 21 text", metadata={"article": "21"})
    ]
    store = MagicMock()
    store.hybrid_search.return_value = expected_results

    results = hybrid_search.search(
        "What does Art. 21 say?", embed_service, store, filters={"language": "de"}, limit=10
    )

    embed_service.embed_dense.assert_called_once_with(
        ["What does Art. 21 say?"], task="retrieval.query"
    )
    embed_service.embed_sparse.assert_called_once_with(["What does Art. 21 say?"])
    store.hybrid_search.assert_called_once_with(
        query_dense=[0.1, 0.2, 0.3],
        query_sparse_indices=[1, 5],
        query_sparse_values=[0.9, 0.4],
        filters={"language": "de"},
        limit=10,
        in_force_on=True,
    )
    assert results == expected_results


def test_search_defaults_to_no_filters():
    embed_service = MagicMock()
    embed_service.embed_dense.return_value = [[0.0]]
    embed_service.embed_sparse.return_value = [SparseVector(indices=[], values=[])]
    store = MagicMock()
    store.hybrid_search.return_value = []

    hybrid_search.search("query", embed_service, store)

    _, kwargs = store.hybrid_search.call_args
    assert kwargs["filters"] is None


def test_search_defaults_to_in_force_today():
    embed_service = MagicMock()
    embed_service.embed_dense.return_value = [[0.0]]
    embed_service.embed_sparse.return_value = [SparseVector(indices=[], values=[])]
    store = MagicMock()
    store.hybrid_search.return_value = []

    hybrid_search.search("query", embed_service, store)

    _, kwargs = store.hybrid_search.call_args
    assert kwargs["in_force_on"] is True


def test_search_forwards_explicit_in_force_on():
    embed_service = MagicMock()
    embed_service.embed_dense.return_value = [[0.0]]
    embed_service.embed_sparse.return_value = [SparseVector(indices=[], values=[])]
    store = MagicMock()
    store.hybrid_search.return_value = []

    hybrid_search.search("query", embed_service, store, in_force_on=False)

    _, kwargs = store.hybrid_search.call_args
    assert kwargs["in_force_on"] is False

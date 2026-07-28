from datetime import date
from unittest.mock import MagicMock

import pytest
from qdrant_client import models

from src.storage.qdrant_store import QdrantStore


def _make_store() -> QdrantStore:
    store = QdrantStore()
    store._client = MagicMock()
    return store


def test_build_filter_returns_none_when_nothing_requested():
    assert QdrantStore._build_filter(None, in_force_on=None) is None
    assert QdrantStore._build_filter({}, in_force_on=None) is None


def test_build_filter_equality_only():
    result = QdrantStore._build_filter({"language": "de"}, in_force_on=None)
    assert isinstance(result, models.Filter)
    assert result.must == [
        models.FieldCondition(key="language", match=models.MatchValue(value="de"))
    ]


def test_build_filter_combines_equality_and_in_force():
    as_of = date(2026, 7, 28)
    result = QdrantStore._build_filter({"language": "it"}, in_force_on=as_of)
    assert len(result.must) == 2
    assert result.must[0] == models.FieldCondition(
        key="language", match=models.MatchValue(value="it")
    )
    # Second condition is the nested in-force sub-filter.
    assert isinstance(result.must[1], models.Filter)


def test_in_force_condition_checks_both_bounds_with_null_escape_hatch():
    as_of = date(2026, 7, 28)
    condition = QdrantStore._in_force_condition(as_of)
    assert len(condition.must) == 2

    valid_from_subfilter, valid_to_subfilter = condition.must

    assert models.IsNullCondition(is_null=models.PayloadField(key="valid_from")) in (
        valid_from_subfilter.should
    )
    assert (
        models.FieldCondition(key="valid_from", range=models.DatetimeRange(lte=as_of))
        in valid_from_subfilter.should
    )

    assert models.IsNullCondition(is_null=models.PayloadField(key="valid_to")) in (
        valid_to_subfilter.should
    )
    assert (
        models.FieldCondition(key="valid_to", range=models.DatetimeRange(gte=as_of))
        in valid_to_subfilter.should
    )


def _run_hybrid_search(store: QdrantStore, **kwargs):
    store._client.query_points.return_value = MagicMock(points=[])
    store.hybrid_search(
        query_dense=[0.1], query_sparse_indices=[], query_sparse_values=[], **kwargs
    )
    _, call_kwargs = store._client.query_points.call_args
    return call_kwargs["query_filter"]


def test_hybrid_search_defaults_to_in_force_today():
    store = _make_store()
    qdrant_filter = _run_hybrid_search(store)
    assert qdrant_filter is not None
    # The in-force sub-filter is the only `must` entry when no equality filters are given.
    in_force_subfilter = qdrant_filter.must[0]
    valid_to_subfilter = in_force_subfilter.must[1]
    range_condition = next(
        c for c in valid_to_subfilter.should if isinstance(c, models.FieldCondition)
    )
    assert range_condition.range.gte == date.today()


def test_hybrid_search_in_force_on_false_disables_date_filter():
    store = _make_store()
    qdrant_filter = _run_hybrid_search(store, in_force_on=False)
    assert qdrant_filter is None


def test_hybrid_search_in_force_on_explicit_date():
    store = _make_store()
    as_of = date(2020, 1, 1)
    qdrant_filter = _run_hybrid_search(store, in_force_on=as_of)
    in_force_subfilter = qdrant_filter.must[0]
    valid_to_subfilter = in_force_subfilter.must[1]
    range_condition = next(
        c for c in valid_to_subfilter.should if isinstance(c, models.FieldCondition)
    )
    assert range_condition.range.gte == as_of


def _fake_collection_info(dense_dim: int | None):
    info = MagicMock()
    if dense_dim is None:
        info.config.params.vectors = None
    else:
        vector_params = MagicMock()
        vector_params.size = dense_dim
        info.config.params.vectors = {"dense": vector_params}
    return info


def test_create_collection_noop_when_dimension_matches():
    store = _make_store()
    store._client.collection_exists.return_value = True
    store._client.get_collection.return_value = _fake_collection_info(1024)

    store.create_collection(dense_dim=1024)

    store._client.create_collection.assert_not_called()


def test_create_collection_raises_clear_error_on_dimension_mismatch():
    # Regression: previously this silently returned, so the first upsert with a
    # different-dimension embedding model would fail with a much less legible error
    # straight from Qdrant's HTTP layer instead of a clear, actionable message here.
    store = _make_store()
    store._client.collection_exists.return_value = True
    store._client.get_collection.return_value = _fake_collection_info(1024)

    with pytest.raises(ValueError, match="768"):
        store.create_collection(dense_dim=768)

    store._client.create_collection.assert_not_called()


def test_create_collection_recreate_bypasses_dimension_check():
    store = _make_store()
    store._client.collection_exists.return_value = True
    store._client.get_collection.return_value = _fake_collection_info(1024)

    store.create_collection(dense_dim=768, recreate=True)

    store._client.delete_collection.assert_called_once()
    store._client.create_collection.assert_called_once()


def test_create_collection_creates_fresh_when_absent():
    store = _make_store()
    store._client.collection_exists.return_value = False

    store.create_collection(dense_dim=768)

    store._client.get_collection.assert_not_called()
    store._client.create_collection.assert_called_once()


def test_hybrid_search_combines_filters_with_in_force():
    store = _make_store()
    qdrant_filter = _run_hybrid_search(store, filters={"language": "fr"}, in_force_on=True)
    assert len(qdrant_filter.must) == 2
    assert qdrant_filter.must[0] == models.FieldCondition(
        key="language", match=models.MatchValue(value="fr")
    )

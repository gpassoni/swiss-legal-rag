from scripts.run_eval import _avg, _ranked_chunk_keys, mean_reciprocal_rank, precision_at_k
from src.storage.qdrant_store import SearchResult


def _chunk(sr: str, article: str) -> SearchResult:
    return SearchResult(
        id=f"{sr}-{article}",
        score=1.0,
        text="irrelevant",
        metadata={"systematic_number": sr, "article": article},
    )


def _chunk_missing_metadata() -> SearchResult:
    return SearchResult(id="x", score=1.0, text="irrelevant", metadata={})


def test_ranked_chunk_keys_preserves_order_and_skips_missing_metadata():
    chunks = [_chunk("642.11", "21"), _chunk_missing_metadata(), _chunk("210", "2")]
    assert _ranked_chunk_keys(chunks) == [("642.11", "21"), ("210", "2")]


def test_precision_at_k_none_when_nothing_retrieved():
    assert precision_at_k([], expected={("642.11", "21")}) is None


def test_precision_at_k_all_relevant():
    ranked = [("642.11", "21"), ("210", "2")]
    expected = {("642.11", "21"), ("210", "2")}
    assert precision_at_k(ranked, expected) == 1.0


def test_precision_at_k_partial_relevance():
    ranked = [("642.11", "21"), ("999", "1"), ("999", "2"), ("999", "3")]
    expected = {("642.11", "21")}
    assert precision_at_k(ranked, expected) == 0.25


def test_precision_at_k_zero_when_nothing_relevant():
    ranked = [("999", "1"), ("999", "2")]
    expected = {("642.11", "21")}
    assert precision_at_k(ranked, expected) == 0.0


def test_mrr_none_when_nothing_retrieved():
    assert mean_reciprocal_rank([], expected={("642.11", "21")}) is None


def test_mrr_full_score_when_first_result_is_relevant():
    ranked = [("642.11", "21"), ("999", "1")]
    expected = {("642.11", "21")}
    assert mean_reciprocal_rank(ranked, expected) == 1.0


def test_mrr_reciprocal_of_position_when_relevant_result_is_later():
    ranked = [("999", "1"), ("999", "2"), ("642.11", "21")]
    expected = {("642.11", "21")}
    assert mean_reciprocal_rank(ranked, expected) == 1.0 / 3


def test_mrr_zero_when_nothing_relevant_in_ranked_list():
    ranked = [("999", "1"), ("999", "2")]
    expected = {("642.11", "21")}
    assert mean_reciprocal_rank(ranked, expected) == 0.0


def test_avg_ignores_none_values():
    assert _avg([1.0, None, 0.5, None]) == 0.75


def test_avg_none_when_all_none_or_empty():
    assert _avg([None, None]) is None
    assert _avg([]) is None

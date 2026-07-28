from src.orchestration.citation_verifier import extract_citations, verify_citations
from src.storage.qdrant_store import SearchResult


def _chunk(article, source_url):
    return SearchResult(
        id="1", score=1.0, text="irrelevant", metadata={"article": article, "source_url": source_url}
    )


def test_extract_citations_parses_expected_format():
    answer = "Tax deductions are allowed (Art. 21, https://fedlex.example/act/1)."
    citations = extract_citations(answer)
    assert citations == [("21", "https://fedlex.example/act/1")]


def test_extract_citations_handles_no_article():
    answer = "See the summary (Art. n/a, https://parlament.example/business/1)."
    citations = extract_citations(answer)
    assert citations == [("n/a", "https://parlament.example/business/1")]


def test_verify_citations_marks_matching_citation_as_verified():
    chunks = [_chunk("21", "https://fedlex.example/act/1")]
    answer = "Deductions apply (Art. 21, https://fedlex.example/act/1)."

    result = verify_citations(answer, chunks)

    assert result.all_verified is True
    assert len(result.verified_citations) == 1
    assert result.unverified_citations == []


def test_verify_citations_flags_hallucinated_citation():
    chunks = [_chunk("21", "https://fedlex.example/act/1")]
    answer = "Deductions apply (Art. 99, https://fedlex.example/act/999)."

    result = verify_citations(answer, chunks)

    assert result.all_verified is False
    assert len(result.unverified_citations) == 1
    assert result.verified_citations == []


def test_verify_citations_with_no_citations_is_not_all_verified():
    chunks = [_chunk("21", "https://fedlex.example/act/1")]
    answer = "This answer cites nothing."

    result = verify_citations(answer, chunks)

    assert result.citations == []
    assert result.all_verified is False


def test_has_citations_false_when_answer_cites_nothing():
    # Regression: `unverified_citations` is also empty in this case (there's nothing to
    # be unverified), so a caller checking only `unverified_citation_count == 0` would
    # wrongly treat an ungrounded, zero-citation answer as "fully verified". `has_citations`
    # is the field that actually distinguishes the two.
    chunks = [_chunk("21", "https://fedlex.example/act/1")]
    answer = "This answer cites nothing."

    result = verify_citations(answer, chunks)

    assert result.unverified_citations == []
    assert result.has_citations is False


def test_has_citations_true_when_all_verified():
    chunks = [_chunk("21", "https://fedlex.example/act/1")]
    answer = "Deductions apply (Art. 21, https://fedlex.example/act/1)."

    result = verify_citations(answer, chunks)

    assert result.has_citations is True
    assert result.all_verified is True


def test_has_citations_true_even_when_hallucinated():
    # has_citations only signals "the answer attempted at least one citation" — it says
    # nothing about whether that citation checked out; `all_verified`/`unverified_citations`
    # remain the source of truth for correctness of the citations that are present.
    chunks = [_chunk("21", "https://fedlex.example/act/1")]
    answer = "Deductions apply (Art. 99, https://fedlex.example/act/999)."

    result = verify_citations(answer, chunks)

    assert result.has_citations is True
    assert result.all_verified is False


def test_verify_citations_mixed_verified_and_unverified():
    chunks = [_chunk("21", "https://fedlex.example/act/1")]
    answer = (
        "First claim (Art. 21, https://fedlex.example/act/1). "
        "Second, fabricated claim (Art. 30, https://fedlex.example/act/30)."
    )

    result = verify_citations(answer, chunks)

    assert result.all_verified is False
    assert len(result.verified_citations) == 1
    assert len(result.unverified_citations) == 1

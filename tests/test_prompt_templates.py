from src.orchestration.prompt_templates import (
    build_user_prompt,
    estimate_tokens,
    fit_chunks_to_token_budget,
    format_chunk_for_prompt,
)
from src.storage.qdrant_store import SearchResult


def _chunk(article: str, words: int) -> SearchResult:
    text = " ".join(f"word{i}" for i in range(words))
    return SearchResult(
        id=article,
        score=1.0,
        text=text,
        metadata={"article": article, "source_url": f"https://example/{article}"},
    )


def test_estimate_tokens_empty_string():
    assert estimate_tokens("") == 0


def test_estimate_tokens_scales_with_word_count():
    short = estimate_tokens("one two three")
    long = estimate_tokens(" ".join(["word"] * 300))
    assert long > short


def test_fit_chunks_keeps_everything_when_under_budget():
    chunks = [_chunk("1", 10), _chunk("2", 10)]
    kept_chunks, kept_expanded, dropped = fit_chunks_to_token_budget(chunks, [], max_tokens=10_000)
    assert kept_chunks == chunks
    assert kept_expanded == []
    assert dropped == 0


def test_fit_chunks_drops_lowest_priority_tail_first():
    # chunks is priority-ordered best-first (rerank score order); a tight budget should
    # keep the earliest (highest-priority) ones and drop later ones.
    chunks = [_chunk(str(i), 200) for i in range(5)]
    kept_chunks, _kept_expanded, dropped = fit_chunks_to_token_budget(chunks, [], max_tokens=400)
    assert kept_chunks == chunks[: len(kept_chunks)]
    assert len(kept_chunks) < len(chunks)
    assert dropped == len(chunks) - len(kept_chunks)


def test_fit_chunks_always_keeps_first_chunk_even_if_it_exceeds_budget_alone():
    chunks = [_chunk("huge", 5000)]
    kept_chunks, _kept_expanded, dropped = fit_chunks_to_token_budget(chunks, [], max_tokens=10)
    assert kept_chunks == chunks
    assert dropped == 0


def test_fit_chunks_expanded_chunks_are_lower_priority_than_main_chunks():
    main_chunks = [_chunk("main", 200)]
    expanded = [_chunk("expanded", 200)]
    # Budget fits the main chunk but not the expanded one on top of it.
    budget = estimate_tokens(format_chunk_for_prompt(1, main_chunks[0])) + 5
    kept_chunks, kept_expanded, dropped = fit_chunks_to_token_budget(
        main_chunks, expanded, max_tokens=budget
    )
    assert kept_chunks == main_chunks
    assert kept_expanded == []
    assert dropped == 1


def test_fit_chunks_expanded_chunk_can_be_dropped_without_touching_main_chunks():
    main_chunks = [_chunk("a", 10), _chunk("b", 10)]
    expanded = [_chunk("c", 5000)]
    kept_chunks, kept_expanded, dropped = fit_chunks_to_token_budget(
        main_chunks, expanded, max_tokens=100
    )
    assert kept_chunks == main_chunks
    assert kept_expanded == []
    assert dropped == 1


def test_build_user_prompt_still_works_with_pretrimmed_lists():
    chunks = [_chunk("21", 5)]
    prompt = build_user_prompt("What does Art. 21 say?", chunks, [])
    assert "article=21" in prompt
    assert "word0" in prompt

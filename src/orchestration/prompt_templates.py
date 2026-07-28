"""System/user prompt construction and the non-binding disclaimer.

Per architecture spec §3.6 and §8, the disclaimer is a product requirement enforced here
in code — ``ensure_disclaimer`` guarantees it is present on every answer, rather than
relying on the model to remember to include it.
"""
from __future__ import annotations

import math

from src.storage.qdrant_store import SearchResult

DISCLAIMER = (
    "This information is provided for general informational purposes only and does not "
    "constitute legal or tax advice. It is not binding and may not reflect the current "
    "state of the law. Consult a qualified professional before acting on it."
)

# Citation format the model is instructed to use inline, and that citation_verifier.py
# parses deterministically: (Art. <article>, <source_url>)
CITATION_FORMAT_INSTRUCTIONS = (
    "When you state a fact drawn from a retrieved chunk, cite it inline immediately after "
    "the claim using exactly this format: (Art. <article>, <source_url>) — using the "
    "`article` and `source_url` values given for that chunk below. If a chunk has no "
    "article (e.g. a parliamentary business summary), write (Art. n/a, <source_url>). "
    "Do not invent citations that are not backed by a retrieved chunk."
)

SYSTEM_PROMPT = f"""You are a Swiss legal and financial information assistant (Phase 1: \
federal law, federal parliamentary activity, and Swiss monetary/banking statistics only).

Rules:
1. Answer ONLY using the retrieved chunks provided in the user message. Do not use outside \
knowledge, even if you believe it is correct.
2. {CITATION_FORMAT_INSTRUCTIONS}
3. If the retrieved chunks do not contain enough information to answer, say so explicitly \
instead of guessing.
4. Always end your answer with this exact disclaimer on its own line:
"{DISCLAIMER}"
"""


def format_chunk_for_prompt(index: int, result: SearchResult) -> str:
    meta = result.metadata
    article = meta.get("article") or "n/a"
    source_url = meta.get("source_url", "")
    law = meta.get("law_short_name") or meta.get("systematic_number") or meta.get("source", "")
    return (
        f"[Chunk {index}] law={law} article={article} language={meta.get('language')} "
        f"source_url={source_url}\n{result.text}"
    )


def estimate_tokens(text: str) -> int:
    """Approximate token count for prompt-budget purposes.

    Not tied to a specific provider's tokenizer: `LLM_PROVIDER` can be Anthropic or
    OpenAI (see query_engine.py), and Anthropic doesn't ship a fast local tokenizer the
    way OpenAI's tiktoken does, so a single provider-specific dependency wouldn't even
    apply to both. A ~1.3x multiplier over a whitespace word count is a deliberately
    cautious over-estimate of subword/punctuation splitting for truncation purposes —
    good enough to budget against, not an exact count.
    """
    if not text:
        return 0
    return math.ceil(len(text.split()) * 1.3)


def fit_chunks_to_token_budget(
    chunks: list[SearchResult],
    expanded_chunks: list[SearchResult],
    max_tokens: int,
) -> tuple[list[SearchResult], list[SearchResult], int]:
    """Keep as many chunks as fit in `max_tokens`, dropping the lowest-priority ones
    first: `chunks` is already rerank-score-ordered (best first) and `expanded_chunks`
    is lower priority still (cross-reference "supporting context", see
    `orchestration.query_engine._expand_references`), so truncating from each list's
    tail preserves relevance ordering. Always keeps at least the single highest-priority
    chunk even if it alone exceeds the budget, so one oversized chunk can't empty the
    prompt entirely. Returns `(kept_chunks, kept_expanded_chunks, dropped_count)`.
    """
    kept_chunks: list[SearchResult] = []
    kept_expanded: list[SearchResult] = []
    used_tokens = 0
    dropped = 0

    for i, chunk in enumerate(chunks):
        cost = estimate_tokens(format_chunk_for_prompt(i + 1, chunk))
        if kept_chunks and used_tokens + cost > max_tokens:
            dropped += 1
            continue
        kept_chunks.append(chunk)
        used_tokens += cost

    for i, chunk in enumerate(expanded_chunks):
        cost = estimate_tokens(format_chunk_for_prompt(i + 1, chunk))
        if used_tokens + cost > max_tokens:
            dropped += 1
            continue
        kept_expanded.append(chunk)
        used_tokens += cost

    return kept_chunks, kept_expanded, dropped


def build_user_prompt(
    question: str,
    chunks: list[SearchResult],
    expanded_chunks: list[SearchResult] | None = None,
) -> str:
    """`expanded_chunks` are articles pulled in via 1-hop cross-reference expansion
    (see `orchestration.query_engine`) — kept in a visually separate "supporting
    context" section so the model (and a reader of the prompt) can tell a directly
    retrieved chunk from one pulled in because a retrieved chunk referenced it. Both
    pools are equally citable: `citation_verifier.verify_citations` is called against
    the combined set, so a citation naming an expanded chunk's article is verified,
    not rejected.
    """
    chunk_blocks = "\n\n".join(
        format_chunk_for_prompt(i + 1, chunk) for i, chunk in enumerate(chunks)
    )
    if not chunk_blocks:
        chunk_blocks = "(no chunks retrieved)"

    expanded_section = ""
    if expanded_chunks:
        expanded_blocks = "\n\n".join(
            format_chunk_for_prompt(i + 1, chunk) for i, chunk in enumerate(expanded_chunks)
        )
        expanded_section = (
            "\n\nSupporting context (articles referenced by the chunks above, pulled in "
            "automatically — cite them the same way if you use them):\n" + expanded_blocks
        )

    return (
        f"Question: {question}\n\n"
        f"Retrieved chunks:\n{chunk_blocks}"
        f"{expanded_section}\n\n"
        "Answer the question using only the chunks above, following the citation format "
        "and disclaimer rules from the system prompt."
    )


def ensure_disclaimer(answer: str) -> str:
    """Guarantee the disclaimer is present, appending it if the model omitted it."""
    if DISCLAIMER in answer:
        return answer
    return f"{answer.rstrip()}\n\n{DISCLAIMER}"

"""Ties retrieval (hybrid search + rerank) and the LLM call together, with deterministic
citation verification on the output (architecture spec §3.6).

The LLM layer is provider-agnostic: `QueryEngine` depends on a small `LLMClient`
protocol, with adapters for Anthropic and OpenAI-compatible APIs. Callers can also pass
any other object implementing `complete(system, user) -> str`.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

import anthropic
import openai
import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import Settings, estimate_cost_usd, get_settings
from src.embedding.embed_service import EmbedService
from src.orchestration.citation_verifier import Citation, verify_citations
from src.orchestration.prompt_templates import (
    SYSTEM_PROMPT,
    build_user_prompt,
    ensure_disclaimer,
    fit_chunks_to_token_budget,
)
from src.retrieval import hybrid_search
from src.retrieval.reranker import Reranker
from src.storage.postgres_store import PostgresStore
from src.storage.qdrant_store import QdrantStore, SearchResult

logger = structlog.get_logger("swiss_legal_ai.orchestration.query_engine")


class LLMClient(Protocol):
    async def complete(self, system: str, user: str) -> str: ...


# Retry only on genuinely transient provider errors (rate limit, connection issues,
# request timeout, 5xx/overload) — never on 4xx client errors (bad request, auth), which
# a retry can't fix and would just add latency to a request that's going to fail anyway.
# Mirrors the ingestion HTTP retry policy (ingestion/base.py::default_retry).
def _anthropic_retry():
    return retry(
        retry=retry_if_exception_type(
            (
                anthropic.APIConnectionError,
                anthropic.APITimeoutError,
                anthropic.RateLimitError,
                anthropic.InternalServerError,
                anthropic.OverloadedError,
            )
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )


def _openai_retry():
    return retry(
        retry=retry_if_exception_type(
            (
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.RateLimitError,
                openai.InternalServerError,
            )
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )


class AnthropicAdapter:
    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-5",
        temperature: float = 0.1,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._timeout = timeout
        # Token usage of the most recent `complete()` call, for cost-tracking UIs
        # (e.g. the Streamlit test app) — not part of the LLMClient protocol.
        self.last_usage: dict[str, int] | None = None

    @property
    def model(self) -> str:
        return self._model

    @_anthropic_retry()
    async def complete(self, system: str, user: str) -> str:
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=self._api_key, timeout=self._timeout)
        response = await client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            temperature=self._temperature,
            messages=[{"role": "user", "content": user}],
        )
        self.last_usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        return "".join(block.text for block in response.content if block.type == "text")


class OpenAIAdapter:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._timeout = timeout
        self.last_usage: dict[str, int] | None = None

    @property
    def model(self) -> str:
        return self._model

    @_openai_retry()
    async def complete(self, system: str, user: str) -> str:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self._api_key, timeout=self._timeout)
        response = await client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        if response.usage:
            self.last_usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
        return response.choices[0].message.content or ""


def build_llm_client_from_settings(settings: Settings | None = None) -> LLMClient:
    settings = settings or get_settings()
    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set")
        return AnthropicAdapter(
            api_key=settings.anthropic_api_key,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
        )
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set")
        return OpenAIAdapter(
            api_key=settings.openai_api_key,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
        )
    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")


@dataclass
class QueryResult:
    answer: str
    verified_sources: list[Citation]
    unverified_citation_count: int
    retrieved_chunk_count: int
    # Whether the answer contained any parseable citation at all — distinct from
    # `unverified_citation_count == 0`, which is also true for an answer with zero
    # citations (see `citation_verifier.VerificationResult.has_citations`). A caller must
    # check both to tell "properly grounded" apart from "cited nothing".
    has_citations: bool = True
    # Chunks actually included in the prompt shown to the LLM, i.e. after
    # `prompt_templates.fit_chunks_to_token_budget` truncation — not necessarily every
    # chunk retrieval+rerank returned, see `dropped_chunk_count`.
    chunks: list[SearchResult] = field(default_factory=list)
    # Articles pulled in via 1-hop cross-reference expansion (see `_expand_references`),
    # kept separate from `chunks` (the directly retrieved+reranked set) for transparency
    # in the Streamlit test UI — both pools are equally citable/verified, though.
    expanded_chunks: list[SearchResult] = field(default_factory=list)
    # How many retrieved/expanded chunks were cut by the token budget before prompting
    # (see prompt_templates.fit_chunks_to_token_budget) — 0 when nothing was dropped.
    dropped_chunk_count: int = 0


# Bound on 1-hop cross-reference expansion per query — caps prompt growth from a heavily
# cross-referenced article; see `QueryEngine._expand_references`.
MAX_EXPANDED_CHUNKS = 5


class QueryEngine:
    def __init__(
        self,
        embed_service: EmbedService,
        store: QdrantStore,
        reranker: Reranker,
        llm_client: LLMClient,
        postgres_store: PostgresStore | None = None,
    ) -> None:
        self._embed_service = embed_service
        self._store = store
        self._reranker = reranker
        self._llm_client = llm_client
        # Optional: without it, cross-reference expansion is simply skipped (e.g. for
        # callers/tests that only care about hybrid search + rerank + LLM).
        self._postgres_store = postgres_store

    async def _expand_references(self, top_chunks: list[SearchResult]) -> list[SearchResult]:
        """Bounded 1-hop expansion: for each retrieved chunk with a known
        (systematic_number, article), look up its resolved outgoing cross-references
        (processing.reference_extractor / storage.postgres_store) and fetch each
        referenced article from Qdrant if it isn't already among `top_chunks`."""
        if self._postgres_store is None:
            return []

        already_present = {
            (c.metadata.get("systematic_number"), c.metadata.get("article")) for c in top_chunks
        }
        expanded: list[SearchResult] = []
        seen: set[tuple[str | None, str | None]] = set()
        for chunk in top_chunks:
            systematic_number = chunk.metadata.get("systematic_number")
            article = chunk.metadata.get("article")
            if not systematic_number or not article:
                continue
            outgoing = await self._postgres_store.get_outgoing_references(
                systematic_number, article
            )
            for ref in outgoing:
                target = (ref["to_systematic_number"], ref["to_article"])
                if target in already_present or target in seen:
                    continue
                seen.add(target)
                result = self._store.get_by_article(ref["to_systematic_number"], ref["to_article"])
                if result is not None:
                    expanded.append(result)
                if len(expanded) >= MAX_EXPANDED_CHUNKS:
                    return expanded
        return expanded

    async def answer(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        retrieval_limit: int = 50,
        top_k: int = 8,
    ) -> QueryResult:
        request_id = uuid.uuid4().hex[:12]
        log = logger.bind(request_id=request_id)
        t_start = time.perf_counter()

        # Both hybrid_search.search (a sync QdrantClient call) and reranker.rerank (a
        # real CPU/GPU cross-encoder inference batch, not I/O) are blocking calls — run
        # them off the event loop so one request's retrieval/rerank doesn't stall every
        # other concurrent request under FastAPI's single event loop (same approach as
        # api/main.py::_fetch_and_chunk).
        t0 = time.perf_counter()
        candidates = await asyncio.to_thread(
            hybrid_search.search,
            question,
            self._embed_service,
            self._store,
            filters=filters,
            limit=retrieval_limit,
        )
        retrieval_latency_ms = (time.perf_counter() - t0) * 1000
        top_candidate_score = candidates[0].score if candidates else None

        t0 = time.perf_counter()
        top_chunks: list[SearchResult] = await asyncio.to_thread(
            self._reranker.rerank, question, candidates, top_k=top_k
        )
        rerank_latency_ms = (time.perf_counter() - t0) * 1000
        top_rerank_score = top_chunks[0].score if top_chunks else None

        t0 = time.perf_counter()
        expanded_chunks = await self._expand_references(top_chunks)
        expansion_latency_ms = (time.perf_counter() - t0) * 1000

        # Truncate to a token budget before prompting — top_chunks is already rerank-score
        # ordered and expanded_chunks is lower priority still, so this drops the least
        # relevant material first (see prompt_templates.fit_chunks_to_token_budget). Both
        # the prompt and the citation verification below use the *trimmed* sets, since a
        # citation matching a chunk that got dropped was never actually shown to the model.
        prompt_chunks, prompt_expanded_chunks, dropped_chunk_count = fit_chunks_to_token_budget(
            top_chunks, expanded_chunks, get_settings().max_prompt_context_tokens
        )

        user_prompt = build_user_prompt(question, prompt_chunks, prompt_expanded_chunks)
        prompt_tokens_approx = len(user_prompt.split())

        t0 = time.perf_counter()
        raw_answer = await self._llm_client.complete(SYSTEM_PROMPT, user_prompt)
        llm_latency_ms = (time.perf_counter() - t0) * 1000
        answer = ensure_disclaimer(raw_answer)

        usage = getattr(self._llm_client, "last_usage", None)
        cost_usd = None
        model = getattr(self._llm_client, "model", None)
        if usage and model:
            cost_usd = estimate_cost_usd(model, usage["input_tokens"], usage["output_tokens"])

        # `known` is built from both pools: a citation naming an expanded (cross-
        # referenced) chunk is legitimate — it was shown to the model — while a citation
        # naming anything else is still rejected. See prompt_templates.build_user_prompt.
        verification = verify_citations(answer, prompt_chunks + prompt_expanded_chunks)
        if verification.unverified_citations:
            log.warning(
                "unverified_citations_rejected",
                question=question,
                unverified_count=len(verification.unverified_citations),
                unverified_articles=[c.article for c in verification.unverified_citations],
            )
        if not verification.has_citations and prompt_chunks:
            # Chunks were retrieved and shown to the model, yet the answer cites none of
            # them — `unverified_citation_count` alone is 0 here too, so it would otherwise
            # pass silently as "no unverified citations" (see citation_verifier
            # docstring). Distinct from the legitimate "insufficient information" case,
            # this is still worth a warning since it's the one signal that lets an
            # ungrounded answer be told apart from a well-cited one downstream.
            log.warning(
                "answer_has_no_citations", question=question, chunk_count=len(prompt_chunks)
            )

        log.info(
            "query_completed",
            question=question,
            retrieval_latency_ms=round(retrieval_latency_ms, 1),
            retrieval_candidate_count=len(candidates),
            top_candidate_score=top_candidate_score,
            rerank_latency_ms=round(rerank_latency_ms, 1),
            top_rerank_score=top_rerank_score,
            expansion_latency_ms=round(expansion_latency_ms, 1),
            expanded_chunk_count=len(expanded_chunks),
            dropped_chunk_count=dropped_chunk_count,
            prompt_tokens_approx=prompt_tokens_approx,
            llm_latency_ms=round(llm_latency_ms, 1),
            llm_model=model,
            llm_input_tokens=usage["input_tokens"] if usage else None,
            llm_output_tokens=usage["output_tokens"] if usage else None,
            llm_cost_usd_estimate=round(cost_usd, 6) if cost_usd is not None else None,
            verified_citation_count=len(verification.verified_citations),
            unverified_citation_count=len(verification.unverified_citations),
            has_citations=verification.has_citations,
            total_latency_ms=round((time.perf_counter() - t_start) * 1000, 1),
        )

        return QueryResult(
            answer=answer,
            verified_sources=verification.verified_citations,
            unverified_citation_count=len(verification.unverified_citations),
            retrieved_chunk_count=len(top_chunks),
            has_citations=verification.has_citations,
            chunks=prompt_chunks,
            expanded_chunks=prompt_expanded_chunks,
            dropped_chunk_count=dropped_chunk_count,
        )

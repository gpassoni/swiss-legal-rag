"""Centralized settings loaded from environment / .env."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Vector DB
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # Relational DB
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "swiss_legal_ai"
    postgres_user: str = "postgres"
    postgres_password: str = "changeme"

    # LLM provider
    llm_provider: str = "anthropic"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    # Cheapest current Anthropic model — good default while iterating (Phase 3 test UI).
    # Override via LLM_MODEL in .env.
    llm_model: str = "claude-haiku-4-5-20251001"
    # Low, not 0.0: some provider APIs reject/warn on a hard-zero temperature for certain
    # models, and a touch of randomness is harmless for this task. Kept low rather than
    # the provider default (Anthropic's default is 1.0) because the answer must follow a
    # rigid inline citation grammar that `citation_verifier` parses with a strict regex —
    # lower temperature measurably improves format adherence and factual consistency for
    # a task that should behave close to deterministically in the first place.
    llm_temperature: float = 0.1
    llm_timeout_seconds: float = 30.0

    # Embedding model
    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_device: str = "cpu"
    # sentence-transformers encode() batch size. 16 is a safe CPU default; on GPU this
    # under-utilizes the device — bump it via .env (EMBEDDING_BATCH_SIZE) once
    # EMBEDDING_DEVICE=cuda, tuned to available VRAM (128 is a reasonable start for an
    # 8GB card with a ~1GB-class multilingual embedding model).
    embedding_batch_size: int = 16

    # Caps the approximate total token count of any single encode() sub-batch, on top of
    # the item-count cap (EMBEDDING_BATCH_SIZE) — see embed_service._length_bucketed_batches.
    # sentence-transformers' own encode() already sorts inputs by length internally before
    # batching, but only within one encode() call; that still allows a batch entirely made
    # of long outlier chunks (chunker.py allows up to ~1500 tokens/article) once
    # EMBEDDING_BATCH_SIZE is raised for GPU throughput. 24000 mirrors the previously
    # tested-safe worst case (batch_size=16 x ~1500-token chunk) as a default: below that
    # per-batch token total, chunks are bucketed at the full configured batch size; above
    # it, a bucket of long outliers gets split into several smaller sub-batches
    # automatically. Re-tune against actual VRAM headroom before relying on it.
    embedding_max_tokens_per_batch: int = 24000

    # Reranker (cross-encoder). No dedicated device setting: sentence-transformers'
    # CrossEncoder auto-detects CUDA when available, same as the embedding model.
    # max_length caps tokens per (query, chunk) pair — without it, a single outlier
    # chunk (chunker.py allows up to ~1000 words per paragraph split) forces the whole
    # batch to pad to its length; combined with the embedding model already resident on
    # the same GPU, this pushed VRAM to ~7.9/8GB and made inference appear to hang
    # (100% GPU util, no progress) rather than erroring — same outlier-padding failure
    # mode already documented above for EMBEDDING_BATCH_SIZE, just unaddressed here.
    # 512 tokens covers the large majority of article chunks; longer ones get truncated
    # rather than blowing up latency/memory.
    reranker_max_length: int = 512
    # Kept well below the sentence-transformers default (32) for the same VRAM-headroom
    # reason as EMBEDDING_BATCH_SIZE — bump if profiling shows headroom on your GPU.
    reranker_batch_size: int = 8

    # Cap on the *retrieved + cross-reference-expanded chunks* portion of the user prompt
    # (approximate tokens, see orchestration.prompt_templates.estimate_tokens) — without
    # it, top_k=8 chunks up to ~1500 tokens each plus MAX_EXPANDED_CHUNKS=5 more had no
    # ceiling at all (worst case >20k tokens of context for one query, uncontrolled cost/
    # latency and no guarantee relevant information isn't diluted by marginal chunks).
    # 6000 is a starting point comfortably above what a typical top_k=8 query needs;
    # re-tune against observed prompt_tokens_approx / cost in the query_completed log.
    max_prompt_context_tokens: int = 6000

    # Ingestion scope — comma-separated SR/RS systematic-number prefixes to ingest from
    # Fedlex (same comma-separated-string convention as `snb_test_cubes` below; pydantic-
    # settings only decodes list-typed env vars as JSON, not comma lists, so this stays a
    # plain string and is parsed via `fedlex_systematic_prefix_list`).
    # Covers the main federal codes, not the full ~17,000-act Fedlex corpus:
    # 640/642 = tax law, 210 = civil code (ZGB/CC), 220 = code of obligations (OR/CO),
    # 311.0 = criminal code (StGB/CP), 312.0 = criminal procedure (StPO/CPP),
    # 272 = civil procedure (ZPO/CPC), 101 = federal constitution.
    fedlex_systematic_prefixes: str = "640,642,210,220,311.0,312.0,272,101"
    snb_test_cubes: str = "rendoblim,snbmonagg"
    curia_vista_date_from: str = "2025-01-01"

    @property
    def fedlex_systematic_prefix_list(self) -> list[str]:
        return [p.strip() for p in self.fedlex_systematic_prefixes.split(",") if p.strip()]

    @property
    def fedlex_systematic_prefix(self) -> str:
        """Deprecated: first entry of `fedlex_systematic_prefix_list`, kept for callers
        that only ever handled a single prefix (`IngestFedlexRequest.systematic_prefix`
        default, `scripts/run_ingestion.py`'s original single-prefix path)."""
        return self.fedlex_systematic_prefix_list[0]

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


# Coarse, non-authoritative "area of law" label per SR/RS systematic-number prefix —
# a convenience for UI/API filtering (e.g. "solo diritto penale"), not a legal taxonomy.
# Longer prefixes must be checked before shorter ones that share a leading substring
# (none currently collide, but `label_for_systematic_number` sorts defensively anyway).
PREFIX_LABELS: dict[str, str] = {
    "640": "tributario",
    "642": "tributario",
    "210": "civile",
    "220": "civile",
    "311.0": "penale",
    "312.0": "procedura_penale",
    "272": "procedura_civile",
    "101": "costituzionale",
}


def label_for_systematic_number(systematic_number: str | None) -> str | None:
    """Map an SR/RS systematic number (e.g. "642.11") to its coarse area-of-law label
    via longest-prefix match against `PREFIX_LABELS`, mirroring the SPARQL
    `FILTER(STRSTARTS(...))` prefix logic already used in `fedlex_client.py`."""
    if not systematic_number:
        return None
    for prefix in sorted(PREFIX_LABELS, key=len, reverse=True):
        if systematic_number.startswith(prefix):
            return PREFIX_LABELS[prefix]
    return None


# Illustrative $/million-token rates for the LLM-call cost estimate logged per query
# (structured logging, orchestration.query_engine) — NOT used for billing, just a rough
# order-of-magnitude signal alongside latency. Anthropic rates below are current first-
# party API list prices as of this writing; re-check platform.claude.com/docs/en/pricing
# before relying on them for anything beyond a log line. The OpenAI rate is an approximate
# public list price, not independently re-verified here, and should be treated as rougher
# still. Unlisted models fall back to `None` (no cost estimate logged, latency still is).
LLM_PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    # model_id: (input $/1M tokens, output $/1M tokens)
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-5": (5.00, 25.00),
    "gpt-4o-mini": (0.15, 0.60),  # approximate — not re-verified against current OpenAI pricing
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Rough $ estimate for one LLM call, for structured-log visibility only. Returns
    None for a model not in `LLM_PRICING_USD_PER_MTOK` rather than guessing."""
    rates = LLM_PRICING_USD_PER_MTOK.get(model)
    if rates is None:
        return None
    input_rate, output_rate = rates
    return (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate

#!/usr/bin/env python
"""Evaluation harness over `eval/golden_qa.yaml` (architecture doc §9-eval).

Custom scorer, not Ragas: Ragas's core metrics (faithfulness, context precision, ...)
are themselves LLM-judged, which reintroduces the same cost/latency/non-determinism in
the *evaluation* layer that `orchestration.citation_verifier` was deliberately built to
avoid on the *answer* layer via regex, not an LLM call. The metrics computed here —
retrieval recall@k against expected citations and the citation-verifier pass rate — are
cheap to compute deterministically against this pipeline's own output shapes
(`QueryResult`), without adopting a general-purpose framework whose main value (working
across many frameworks' output shapes) isn't needed for a single, specific pipeline.

Usage:
    uv run python scripts/run_eval.py                  # full golden set
    uv run python scripts/run_eval.py --area civile     # one area_of_law only
    uv run python scripts/run_eval.py --ids civile-001  # specific case(s)
    uv run python scripts/run_eval.py --output results.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embedding.embed_service import get_embed_service  # noqa: E402
from src.logging_config import configure_logging  # noqa: E402
from src.orchestration.query_engine import QueryEngine, build_llm_client_from_settings  # noqa: E402
from src.retrieval.reranker import get_reranker  # noqa: E402
from src.storage.postgres_store import PostgresStore  # noqa: E402
from src.storage.qdrant_store import QdrantStore  # noqa: E402

GOLDEN_SET_PATH = Path(__file__).resolve().parent.parent / "eval" / "golden_qa.yaml"

# Default `top_k` for `QueryEngine.answer()` — matches the production `/query` endpoint's
# default (src/api/main.py::QueryRequest.top_k), so recall@k here reflects what a real
# user would actually see, not an eval-only inflated value.
DEFAULT_TOP_K = 8


@dataclass
class EvalResult:
    id: str
    area_of_law: str
    question: str
    expected: set[tuple[str, str]]
    retrieved: set[tuple[str, str]]
    recall_at_k: float | None
    citation_pass: bool
    unverified_citation_count: int
    answer_preview: str


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        cases = yaml.safe_load(f)
    return cases or []


def _chunk_keys(chunks) -> set[tuple[str, str]]:  # type: ignore[no-untyped-def]
    keys = set()
    for chunk in chunks:
        sr = chunk.metadata.get("systematic_number")
        article = chunk.metadata.get("article")
        if sr and article:
            keys.add((sr, article))
    return keys


async def evaluate_case(engine: QueryEngine, case: dict, top_k: int) -> EvalResult:
    result = await engine.answer(case["question"], top_k=top_k)
    expected = {(c["systematic_number"], c["article"]) for c in case["expected_citations"]}
    retrieved = _chunk_keys(result.chunks) | _chunk_keys(result.expanded_chunks)
    recall_at_k = len(expected & retrieved) / len(expected) if expected else None
    return EvalResult(
        id=case["id"],
        area_of_law=case["area_of_law"],
        question=case["question"],
        expected=expected,
        retrieved=retrieved,
        recall_at_k=recall_at_k,
        citation_pass=result.unverified_citation_count == 0,
        unverified_citation_count=result.unverified_citation_count,
        answer_preview=result.answer[:150].replace("\n", " "),
    )


def print_report(results: list[EvalResult]) -> None:
    print("\n=== Per-case results ===")
    for r in results:
        recall_str = f"{r.recall_at_k:.0%}" if r.recall_at_k is not None else "n/a"
        cite_str = "OK" if r.citation_pass else f"FAIL ({r.unverified_citation_count} unverified)"
        print(f"- [{r.area_of_law}] {r.id}: recall@k={recall_str}  citations={cite_str}")
        if r.recall_at_k is not None and r.recall_at_k < 1.0:
            missing = r.expected - r.retrieved
            print(f"    missing expected citation(s): {sorted(missing)}")

    print("\n=== Breakdown by area_of_law ===")
    by_area: dict[str, list[EvalResult]] = {}
    for r in results:
        by_area.setdefault(r.area_of_law, []).append(r)
    for area, area_results in sorted(by_area.items()):
        recalls = [r.recall_at_k for r in area_results if r.recall_at_k is not None]
        avg_recall = sum(recalls) / len(recalls) if recalls else None
        citation_pass_rate = sum(r.citation_pass for r in area_results) / len(area_results)
        avg_recall_str = f"{avg_recall:.0%}" if avg_recall is not None else "n/a"
        print(
            f"- {area}: n={len(area_results)}  avg_recall@k={avg_recall_str}  "
            f"citation_pass_rate={citation_pass_rate:.0%}"
        )

    overall_recalls = [r.recall_at_k for r in results if r.recall_at_k is not None]
    overall_avg = sum(overall_recalls) / len(overall_recalls) if overall_recalls else None
    overall_citation_rate = sum(r.citation_pass for r in results) / len(results) if results else 0
    print("\n=== Overall ===")
    print(f"cases: {len(results)}")
    print(f"avg_recall@k: {overall_avg:.0%}" if overall_avg is not None else "avg_recall@k: n/a")
    print(f"citation_pass_rate: {overall_citation_rate:.0%}")


def write_json(results: list[EvalResult], path: Path) -> None:
    payload = [
        {**asdict(r), "expected": sorted(r.expected), "retrieved": sorted(r.retrieved)}
        for r in results
    ]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(results)} result(s) to {path}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area", help="Only run cases with this area_of_law")
    parser.add_argument("--ids", nargs="+", help="Only run these case ids")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--output", type=Path, help="Optional path to write JSON results")
    args = parser.parse_args()

    configure_logging(json_output=False)

    cases = load_golden_set()
    if args.area:
        cases = [c for c in cases if c["area_of_law"] == args.area]
    if args.ids:
        cases = [c for c in cases if c["id"] in args.ids]
    if not cases:
        print("No matching cases in the golden set.")
        return

    embed_service = get_embed_service()
    store = QdrantStore()
    reranker = get_reranker()
    postgres_store = PostgresStore()
    await postgres_store.init_schema()
    llm_client = build_llm_client_from_settings()
    engine = QueryEngine(embed_service, store, reranker, llm_client, postgres_store)

    results = [await evaluate_case(engine, case, args.top_k) for case in cases]

    print_report(results)
    if args.output:
        write_json(results, args.output)


if __name__ == "__main__":
    asyncio.run(main())

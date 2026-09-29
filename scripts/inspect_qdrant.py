#!/usr/bin/env python
"""CLI to explore the Qdrant collection(s) without writing ad-hoc queries by hand.

Three subcommands:
  list    - all collections + point counts
  search  - filter by metadata (source, systematic_number, language, law_short_name)
            and/or a keyword substring in the chunk text; prints readable results
  export  - dump a sample of N points (optionally filtered) to a JSON file

This is a metadata/keyword browser for manual review, not the RAG retrieval path — it
does not load the embedding model, so it starts instantly. Keyword search is a plain
case-insensitive substring match against the stored chunk text (client-side, since the
collection has no full-text index on "text" — only the metadata fields listed in
QdrantStore.create_collection are indexed). For real semantic (hybrid) search, use the
Streamlit app instead.

Examples:
  uv run python scripts/inspect_qdrant.py list
  uv run python scripts/inspect_qdrant.py search --source fedlex --keyword "imposta"
  uv run python scripts/inspect_qdrant.py search --systematic-number 642.114
  uv run python scripts/inspect_qdrant.py export --n 20 --source fedlex --out sample.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient, models  # noqa: E402

from src.config import get_settings  # noqa: E402
from src.storage.qdrant_store import DEFAULT_COLLECTION  # noqa: E402

FILTERABLE_FIELDS = ("source", "systematic_number", "language", "law_short_name")


def get_client() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(
        host=settings.qdrant_host, port=settings.qdrant_port, check_compatibility=False
    )


def build_filter(args: argparse.Namespace) -> models.Filter | None:
    must = []
    for field in FILTERABLE_FIELDS:
        value = getattr(args, field, None)
        if value:
            must.append(models.FieldCondition(key=field, match=models.MatchValue(value=value)))
    return models.Filter(must=must) if must else None


def cmd_list(client: QdrantClient, _args: argparse.Namespace) -> None:
    collections = client.get_collections().collections
    if not collections:
        print("No collections found.")
        return
    print(f"{'collection':<30} {'points':>10}")
    print("-" * 42)
    for coll in collections:
        info = client.get_collection(coll.name)
        print(f"{coll.name:<30} {info.points_count:>10}")


def _scroll_matching(
    client: QdrantClient,
    collection: str,
    qfilter: models.Filter | None,
    keyword: str | None,
    limit: int,
    scan_batch: int = 256,
) -> list[Any]:
    """Scroll through points applying the metadata filter server-side and, if given,
    a case-insensitive keyword substring match on `text` client-side, stopping once
    `limit` matches are collected."""
    keyword_lower = keyword.lower() if keyword else None
    matches: list[Any] = []
    offset = None
    while len(matches) < limit:
        points, offset = client.scroll(
            collection,
            scroll_filter=qfilter,
            limit=scan_batch,
            offset=offset,
            with_payload=True,
        )
        if not points:
            break
        for p in points:
            if keyword_lower and keyword_lower not in (p.payload.get("text") or "").lower():
                continue
            matches.append(p)
            if len(matches) >= limit:
                break
        if offset is None:
            break
    return matches


def cmd_search(client: QdrantClient, args: argparse.Namespace) -> None:
    qfilter = build_filter(args)
    matches = _scroll_matching(client, args.collection, qfilter, args.keyword, args.limit)
    if not matches:
        print("No matching points found.")
        return
    for p in matches:
        payload = dict(p.payload)
        text = payload.pop("text", "")
        print("=" * 80)
        print(f"id: {p.id}")
        print(
            f"source={payload.get('source')}  language={payload.get('language')}  "
            f"article={payload.get('article')}  systematic_number={payload.get('systematic_number')}"
        )
        print(f"law_short_name: {payload.get('law_short_name')}")
        print(f"valid_from={payload.get('valid_from')}  valid_to={payload.get('valid_to')}")
        print(f"source_url: {payload.get('source_url')}")
        print("-" * 80)
        snippet = text if len(text) <= 800 else text[:800] + "... [truncated]"
        print(snippet)
    print("=" * 80)
    print(f"{len(matches)} result(s) shown (limit={args.limit})")


def cmd_export(client: QdrantClient, args: argparse.Namespace) -> None:
    qfilter = build_filter(args)
    matches = _scroll_matching(client, args.collection, qfilter, args.keyword, args.n)
    records = [{"id": str(p.id), **p.payload} for p in matches]
    out_path = Path(args.out)
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported {len(records)} point(s) to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Qdrant collection name")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List collections and point counts")

    search_parser = subparsers.add_parser("search", help="Search by metadata filter and/or keyword")
    search_parser.add_argument("--source", help="e.g. fedlex, curia_vista")
    search_parser.add_argument("--systematic-number", dest="systematic_number", help="e.g. 642.114")
    search_parser.add_argument("--language", help="e.g. it, fr, de, rm")
    search_parser.add_argument("--law-short-name", dest="law_short_name", help="exact match")
    search_parser.add_argument("--keyword", help="case-insensitive substring match on chunk text")
    search_parser.add_argument("--limit", type=int, default=10, help="max results (default 10)")

    export_parser = subparsers.add_parser("export", help="Export a sample of points to JSON")
    export_parser.add_argument("--source", help="e.g. fedlex, curia_vista")
    export_parser.add_argument("--systematic-number", dest="systematic_number", help="e.g. 642.114")
    export_parser.add_argument("--language", help="e.g. it, fr, de, rm")
    export_parser.add_argument("--law-short-name", dest="law_short_name", help="exact match")
    export_parser.add_argument("--keyword", help="case-insensitive substring match on chunk text")
    export_parser.add_argument(
        "--n", type=int, default=20, help="number of points to export (default 20)"
    )
    export_parser.add_argument("--out", default="qdrant_sample.json", help="output JSON file path")

    args = parser.parse_args()
    client = get_client()

    if args.command == "list":
        cmd_list(client, args)
    elif args.command == "search":
        cmd_search(client, args)
    elif args.command == "export":
        cmd_export(client, args)


if __name__ == "__main__":
    main()

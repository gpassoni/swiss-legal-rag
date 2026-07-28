#!/usr/bin/env python
"""One-shot script: create (or recreate) the Qdrant collection with the right dense +
sparse vector configuration, so `/ingest/*` endpoints and `run_ingestion.py` have
somewhere to write to.

Usage:
    python scripts/init_qdrant_collection.py [--dim 1024] [--recreate]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.storage.qdrant_store import QdrantStore  # noqa: E402

# jina-embeddings-v3's default (non-truncated) output dimension.
JINA_V3_DEFAULT_DIM = 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=JINA_V3_DEFAULT_DIM)
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate if it exists")
    args = parser.parse_args()

    store = QdrantStore()
    store.create_collection(dense_dim=args.dim, recreate=args.recreate)
    print(f"Qdrant collection ready (dense_dim={args.dim}, recreate={args.recreate}).")


if __name__ == "__main__":
    main()

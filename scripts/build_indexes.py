"""Build both ChromaDB collections from the knowledge base.


    python -m scripts.build_indexes


Run this first, before calibration, comparison, evaluation or the API. It
chunks every knowledge-base document with BOTH strategies and upserts each
strategy into its own collection.
"""


from __future__ import annotations


import argparse
import sys


from app.config import SETTINGS, ensure_runtime_directories
from app.logging_config import configure_logging
from rag.chunking import load_knowledge_base
from rag.embeddings import EmbeddingModelUnavailableError
from rag.indexer import VectorIndexError, build_indexes




def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help=(
            "upsert without dropping the collections first. Faster, but a document "
            "you deleted or renamed would leave orphan vectors behind."
        ),
    )
    args = parser.parse_args(argv)


    configure_logging(SETTINGS)
    ensure_runtime_directories(SETTINGS)


    documents = load_knowledge_base()
    print(f"[build] loaded {len(documents)} knowledge-base documents")
    for document in documents:
        marker = "required" if document.is_required_topic else "additional"
        print(f"  - {document.document_id:<28} {marker:<10} {document.title}")


    try:
        result = build_indexes(SETTINGS, reset=not args.keep_existing)
    except EmbeddingModelUnavailableError as exc:
        print(f"\n[build] FAILED: {exc}", file=sys.stderr)
        return 2
    except VectorIndexError as exc:
        print(f"\n[build] FAILED: {exc}", file=sys.stderr)
        return 3


    print()
    for line in result.summary_lines():
        print(f"[build] {line}")
    print(f"[build] store: {SETTINGS.chroma_persist_directory}")
    print("[build] next: python -m scripts.calibrate_threshold")
    return 0




if __name__ == "__main__":
    sys.exit(main())




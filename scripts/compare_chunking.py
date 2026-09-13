"""Compare both chunking strategies with document-level precision and recall.


    python -m scripts.compare_chunking


Scores both collections over the same five in-scope queries, shows the
per-query arithmetic, checks the out-of-scope fallback, writes
``reports/chunking_comparison.md`` and prints the recommendation.
"""


from __future__ import annotations


import argparse
import sys


from app.config import SETTINGS, ensure_runtime_directories
from app.logging_config import configure_logging
from rag.embeddings import EmbeddingModelUnavailableError
from rag.evaluation import (
    IN_SCOPE_DEMO_QUERIES,
    OUT_OF_SCOPE_DEMO_QUERIES,
    format_readme_section,
    recommend_strategy,
    score_collection,
    write_report,
)
from rag.grounded_generation import CalibrationRequiredError, GroundedGenerator
from rag.retriever import RetrievalError, Retriever
from scripts.readme import MARKER_CHUNKING, ReadmeSectionError, replace_section


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=None, help="override TOP_K")
    parser.add_argument(
        "--no-readme",
        action="store_true",
        help="skip writing the measured numbers into README.md",
    )
    args = parser.parse_args(argv)


    configure_logging(SETTINGS)
    ensure_runtime_directories(SETTINGS)


    retriever = Retriever(SETTINGS)


    try:
        scores = {
            name: score_collection(
                retriever, name, queries=IN_SCOPE_DEMO_QUERIES, top_k=args.top_k
            )
            for name in SETTINGS.all_collection_names
        }
    except (EmbeddingModelUnavailableError, RetrievalError) as exc:
        print(f"[compare] FAILED: {exc}", file=sys.stderr)
        return 2


    for score in scores.values():
        print(f"\n[compare] strategy {score.strategy} (collection {score.collection_name})")
        for query_score in score.scores:
            print(f"[compare]   {query_score.query_id}")
            print(f"[compare]     retrieved docs : {list(query_score.retrieved_documents)}")
            print(f"[compare]     relevant docs  : {sorted(query_score.relevant)}")
            print(f"[compare]     precision@k    : {query_score.precision_arithmetic}")
            print(f"[compare]     recall@k       : {query_score.recall_arithmetic}")
        print(
            f"[compare]   MEAN precision={score.mean_precision:.4f} "
            f"recall={score.mean_recall:.4f} f1={score.mean_f1:.4f} "
            f"docs/query={score.mean_documents_per_query}"
        )


    recommendation = recommend_strategy(scores)


    # The same out-of-scope query used in Task 4, re-checked here so the
    # comparison report also carries the fallback evidence.
    fallback_check = None
    try:
        generator = GroundedGenerator(retriever=retriever, settings=SETTINGS)
        out_of_scope = OUT_OF_SCOPE_DEMO_QUERIES[0]
        answer = generator.generate(
            out_of_scope.query, collection_name=recommendation[0], top_k=args.top_k
        )
        fallback_check = {
            "query": out_of_scope.query,
            "top_similarity": answer.top_similarity,
            "threshold": answer.threshold,
            "grounded": answer.grounded,
            "answer": answer.answer,
        }
    except CalibrationRequiredError as exc:
        print(
            f"\n[compare] fallback check skipped: {exc}",
            file=sys.stderr,
        )


    path = write_report(scores, recommendation, fallback_check=fallback_check)


    print()
    print(f"[compare] {recommendation[1]}")
    print(f"[compare] report -> {path}")


    if not args.no_readme:
        try:
            readme_path = replace_section(
                MARKER_CHUNKING, format_readme_section(scores, recommendation)
            )
            print(f"[compare] README section -> {readme_path} (AUTO:CHUNKING)")
        except ReadmeSectionError as exc:
            print(f"[compare] WARNING: could not update README: {exc}", file=sys.stderr)


    print(f"[compare] set RECOMMENDED_COLLECTION_NAME={recommendation[0]} in .env")
    return 0


if __name__ == "__main__":
    sys.exit(main())



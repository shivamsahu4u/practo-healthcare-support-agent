"""Measure the "I don't know" similarity threshold empirically.


    python -m scripts.calibrate_threshold


Measures top-1 cosine similarity for 5 in-scope and 3 out-of-scope probes
against BOTH collections, then places the threshold at the midpoint of the
observed gap for the recommended collection. Writes
``data/generated/calibration.json`` (read by grounded generation),
``reports/calibration_report.md`` (for a human), and - because Task 4 requires
the measured values and the chosen threshold to be stated in the README itself -
the ``AUTO:CALIBRATION`` section of ``README.md``.


If the clusters overlap, this exits non-zero and tells you what to fix. It never
picks a number anyway - that is the failure a preset threshold hides.
"""


from __future__ import annotations


import argparse
import sys


from app.config import SETTINGS, ensure_runtime_directories
from app.logging_config import configure_logging
from rag.calibration import (
    CalibrationOverlapError,
    format_readme_section,
    measure,
    write_calibration,
    write_report,
)
from rag.embeddings import EmbeddingModelUnavailableError
from rag.retriever import RetrievalError, Retriever
from scripts.readme import MARKER_CALIBRATION, ReadmeSectionError, replace_section




def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection",
        default=None,
        help=(
            "collection whose measurement sets the deployed threshold. Defaults to "
            "RECOMMENDED_COLLECTION_NAME."
        ),
    )
    parser.add_argument(
        "--no-readme",
        action="store_true",
        help="skip writing the measured values into README.md",
    )
    args = parser.parse_args(argv)


    configure_logging(SETTINGS)
    ensure_runtime_directories(SETTINGS)


    retriever = Retriever(SETTINGS)
    chosen_name = args.collection or SETTINGS.recommended_collection_name


    try:
        results = [
            measure(retriever, collection_name=name, settings=SETTINGS)
            for name in SETTINGS.all_collection_names
        ]
    except (EmbeddingModelUnavailableError, RetrievalError) as exc:
        print(f"[calibrate] FAILED: {exc}", file=sys.stderr)
        return 2


    chosen = next(
        (result for result in results if result.collection_name == chosen_name), None
    )
    if chosen is None:
        print(
            f"[calibrate] FAILED: no measurement for collection {chosen_name!r}.",
            file=sys.stderr,
        )
        return 4


    for result in results:
        print(f"\n[calibrate] collection {result.collection_name}")
        print("[calibrate]   in-scope probes:")
        for measurement in result.in_scope:
            print(
                f"[calibrate]     {measurement.top_similarity:.4f}  "
                f"{measurement.top_document_id:<28} {measurement.query}"
            )
        print("[calibrate]   out-of-scope probes:")
        for measurement in result.out_of_scope:
            print(
                f"[calibrate]     {measurement.top_similarity:.4f}  "
                f"{measurement.top_document_id:<28} {measurement.query}"
            )
        print(
            f"[calibrate]   min(in-scope)={result.min_in_scope:.4f}  "
            f"max(out-of-scope)={result.max_out_of_scope:.4f}  "
            f"margin={result.margin:.4f}  separable={result.separable}"
        )


    try:
        threshold = chosen.recommended_threshold
    except CalibrationOverlapError as exc:
        print(f"\n[calibrate] FAILED: {exc}", file=sys.stderr)
        return 5


    calibration_path = write_calibration(chosen)
    report_path = write_report(results, chosen)


    print()
    print(f"[calibrate] deployed threshold = {threshold:.4f} (collection {chosen_name})")
    print(f"[calibrate] machine-readable  -> {calibration_path}")
    print(f"[calibrate] human-readable    -> {report_path}")


    if not args.no_readme:
        # Task 4 requires the measured values and the chosen threshold to be
        # stated in README.md itself, so they are written there directly.
        try:
            readme_path = replace_section(
                MARKER_CALIBRATION, format_readme_section(chosen)
            )
            print(f"[calibrate] README section  -> {readme_path} (AUTO:CALIBRATION)")
        except ReadmeSectionError as exc:
            print(f"[calibrate] WARNING: could not update README: {exc}", file=sys.stderr)


    print(
        f"[calibrate] pin it for reproducibility with SIMILARITY_THRESHOLD="
        f"{threshold:.4f} in .env"
    )
    return 0




if __name__ == "__main__":
    sys.exit(main())




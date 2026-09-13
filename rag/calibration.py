"""Task 4 (part 3) - empirical calibration of the "I don't know" threshold.


Method: measure the top-1 cosine similarity for a set of in-scope queries and a
set of deliberately out-of-scope queries against the real index, then place the
threshold in the gap between the two observed clusters:


    threshold = (min(in_scope) + max(out_of_scope)) / 2


If the clusters overlap - ``min(in_scope) <= max(out_of_scope)`` - no threshold
can separate them, and this module says so loudly instead of picking a number
anyway. That is the whole point: the brief forbids an untested preset, and a
silently-overlapping calibration is exactly the failure a preset hides.
"""


from __future__ import annotations


import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final


from app.config import CALIBRATION_FILE, REPORTS_DIR, SETTINGS, Settings
from rag.retriever import Retriever


LOGGER: Final = logging.getLogger(__name__)


#: Six in-scope probes (the brief requires >= 3), including the safety-critical
#: emergency scenario so its required policy response is calibrated explicitly.
IN_SCOPE_CALIBRATION_QUERIES: Final[tuple[str, ...]] = (
    "How long before my appointment can I cancel without paying a fee?",
    "What is the consultation fee for a cardiology visit?",
    "How many days does a culture and sensitivity lab test take?",
    "Is a video consultation allowed for a young child?",
    "What discount applies to a follow-up visit within two weeks?",
    "What should someone do about sudden chest pain and breathlessness right now?",
)


#: Three out-of-scope probes (the brief requires >= 2), chosen to be plainly
#: unrelated to clinic policy while still being fluent English sentences - a
#: string of nonsense would make the separation look better than it is.
OUT_OF_SCOPE_CALIBRATION_QUERIES: Final[tuple[str, ...]] = (
    "What is the current share price of a large technology company?",
    "Who won the football league final last season?",
    "Give me a recipe for chocolate cake with buttercream icing.",
)


class CalibrationOverlapError(RuntimeError):
    """Raised when in-scope and out-of-scope similarities do not separate."""


@dataclass(frozen=True, slots=True)
class QueryMeasurement:
    """One measured probe."""


    query: str
    top_similarity: float
    top_document_id: str
    in_scope: bool


    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "top_similarity": round(self.top_similarity, 4),
            "top_document_id": self.top_document_id,
            "in_scope": self.in_scope,
        }


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """The measured separation and the threshold derived from it."""


    collection_name: str
    measured_at: str
    embedder: str
    in_scope: tuple[QueryMeasurement, ...]
    out_of_scope: tuple[QueryMeasurement, ...]


    @property
    def min_in_scope(self) -> float:
        return min(measurement.top_similarity for measurement in self.in_scope)


    @property
    def max_out_of_scope(self) -> float:
        return max(measurement.top_similarity for measurement in self.out_of_scope)


    @property
    def separable(self) -> bool:
        return self.min_in_scope > self.max_out_of_scope


    @property
    def margin(self) -> float:
        """Width of the gap between the clusters. Negative means they overlap."""
        return self.min_in_scope - self.max_out_of_scope


    @property
    def recommended_threshold(self) -> float:
        """Midpoint of the observed gap.


        Raises:
            CalibrationOverlapError: when the clusters overlap.
        """
        if not self.separable:
            raise CalibrationOverlapError(
                "in-scope and out-of-scope similarities overlap: "
                f"min(in-scope)={self.min_in_scope:.4f} <= "
                f"max(out-of-scope)={self.max_out_of_scope:.4f}. No single threshold "
                "separates them. Fix the cause rather than picking a number: check "
                "that the index was built with the embedder currently configured, "
                "that the out-of-scope probes really are unrelated, and that "
                "EMBEDDING_BACKEND is not deterministic_hash."
            )
        return round((self.min_in_scope + self.max_out_of_scope) / 2.0, 4)


    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "collection_name": self.collection_name,
            "measured_at": self.measured_at,
            "embedder": self.embedder,
            "in_scope": [measurement.as_dict() for measurement in self.in_scope],
            "out_of_scope": [measurement.as_dict() for measurement in self.out_of_scope],
            "min_in_scope": round(self.min_in_scope, 4),
            "max_out_of_scope": round(self.max_out_of_scope, 4),
            "margin": round(self.margin, 4),
            "separable": self.separable,
            "method": "midpoint of the observed gap: (min(in_scope) + max(out_of_scope)) / 2",
        }
        if self.separable:
            payload["recommended_threshold"] = self.recommended_threshold
        return payload


def measure(
    retriever: Retriever,
    *,
    collection_name: str | None = None,
    settings: Settings = SETTINGS,
    in_scope: tuple[str, ...] = IN_SCOPE_CALIBRATION_QUERIES,
    out_of_scope: tuple[str, ...] = OUT_OF_SCOPE_CALIBRATION_QUERIES,
) -> CalibrationResult:
    """Measure top-1 similarity for both probe sets against one collection."""
    if len(in_scope) < 3:
        raise ValueError("the brief requires at least 3 in-scope calibration queries.")
    if len(out_of_scope) < 2:
        raise ValueError("the brief requires at least 2 out-of-scope calibration queries.")


    name = collection_name or settings.recommended_collection_name


    def probe(query: str, is_in_scope: bool) -> QueryMeasurement:
        # top_k=1: the threshold is compared against the top-1 similarity, so
        # that is the only number worth measuring here.
        result = retriever.search(query, collection_name=name, top_k=1)
        top = result.chunks[0] if result.chunks else None
        return QueryMeasurement(
            query=query,
            top_similarity=top.similarity if top else 0.0,
            top_document_id=top.document_id if top else "",
            in_scope=is_in_scope,
        )


    return CalibrationResult(
        collection_name=name,
        measured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        embedder=retriever.embedder.name,
        in_scope=tuple(probe(query, True) for query in in_scope),
        out_of_scope=tuple(probe(query, False) for query in out_of_scope),
    )


def write_calibration(
    result: CalibrationResult, *, path: Path = CALIBRATION_FILE
) -> Path:
    """Persist the machine-readable calibration consumed by grounded generation.


    Raises:
        CalibrationOverlapError: refuses to write an unusable calibration.
    """
    _ = result.recommended_threshold  # raises when the clusters overlap
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.as_dict(), indent=2) + "\n", encoding="utf-8")
    LOGGER.info("wrote calibration to %s", path)
    return path


def format_report(results: list[CalibrationResult], chosen: CalibrationResult) -> str:
    """Render the human-readable calibration report."""
    lines = [
        "# Task 4 - Similarity threshold calibration report",
        "",
        "Generated by `python -m scripts.calibrate_threshold`. Every number below is",
        "a measured top-1 cosine similarity against the live index - none of it is a",
        "preset, and none of it was written by hand.",
        "",
        f"- **Measured at:** {chosen.measured_at}",
        f"- **Embedder:** `{chosen.embedder}`",
        f"- **Collection used for the deployed threshold:** `{chosen.collection_name}`",
        "- **Method:** threshold = (min in-scope similarity + max out-of-scope similarity) / 2",
        "",
    ]


    for result in results:
        lines += [
            f"## Collection `{result.collection_name}`",
            "",
            "### In-scope probes",
            "",
            "| Query | Top-1 similarity | Best-matching document |",
            "| --- | --- | --- |",
        ]
        lines.extend(
            f"| {m.query} | {m.top_similarity:.4f} | `{m.top_document_id}` |"
            for m in result.in_scope
        )
        lines += [
            "",
            "### Deliberately out-of-scope probes",
            "",
            "| Query | Top-1 similarity | Best-matching document |",
            "| --- | --- | --- |",
        ]
        lines.extend(
            f"| {m.query} | {m.top_similarity:.4f} | `{m.top_document_id}` |"
            for m in result.out_of_scope
        )
        lines += [
            "",
            f"- min(in-scope) = **{result.min_in_scope:.4f}**",
            f"- max(out-of-scope) = **{result.max_out_of_scope:.4f}**",
            f"- gap (margin) = **{result.margin:.4f}**",
            f"- clusters separable = **{result.separable}**",
        ]
        if result.separable:
            lines.append(
                f"- midpoint threshold = ({result.min_in_scope:.4f} + "
                f"{result.max_out_of_scope:.4f}) / 2 = **{result.recommended_threshold:.4f}**"
            )
        else:
            lines.append(
                "- **the clusters overlap, so no threshold is recommended for this "
                "collection**"
            )
        lines.append("")


    lines += [
        "## Deployed threshold",
        "",
        f"`SIMILARITY_THRESHOLD = {chosen.recommended_threshold:.4f}` "
        f"(from collection `{chosen.collection_name}`)",
        "",
        "This value is written to `data/generated/calibration.json` and read by",
        "`rag/grounded_generation.py`. Pin it in `.env` as `SIMILARITY_THRESHOLD` to",
        "reproduce this exact run on a fresh checkout.",
        "",
        "### Why not 0.5 / 0.6 / 0.7",
        "",
        "Those are tutorial defaults. Short policy sentences embedded with a",
        "MiniLM-class model do not place in-scope and unrelated queries on either side",
        "of a round number reliably - the measured clusters above are where the real",
        "boundary is for *this* knowledge base and *this* embedder, and the midpoint",
        "of the measured gap is the only defensible choice.",
        "",
    ]
    return "\n".join(lines)


def write_report(
    results: list[CalibrationResult],
    chosen: CalibrationResult,
    *,
    path: Path | None = None,
) -> Path:
    """Write the Markdown calibration report and return its path."""
    target = path or (REPORTS_DIR / "calibration_report.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(format_report(results, chosen), encoding="utf-8")
    return target


def format_readme_section(chosen: CalibrationResult) -> str:
    """The compact block written into README.md's AUTO:CALIBRATION markers.


    Task 4 requires the measured values and the chosen threshold to appear in
    the README itself, so this is the canonical short form of them.
    """
    lines = [
        f"Measured on {chosen.measured_at} against collection "
        f"`{chosen.collection_name}` with embedder `{chosen.embedder}`.",
        "",
        "| Probe | In scope? | Top-1 cosine similarity | Best-matching document |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {m.query} | yes | **{m.top_similarity:.4f}** | `{m.top_document_id}` |"
        for m in chosen.in_scope
    )
    lines.extend(
        f"| {m.query} | no | **{m.top_similarity:.4f}** | `{m.top_document_id}` |"
        for m in chosen.out_of_scope
    )
    lines += [
        "",
        f"- `min(in-scope)` = **{chosen.min_in_scope:.4f}**",
        f"- `max(out-of-scope)` = **{chosen.max_out_of_scope:.4f}**",
        f"- observed gap between the two clusters = **{chosen.margin:.4f}**",
        f"- **chosen threshold = ({chosen.min_in_scope:.4f} + "
        f"{chosen.max_out_of_scope:.4f}) / 2 = "
        f"`{chosen.recommended_threshold:.4f}`**",
        "",
        f"The two clusters separate cleanly, so the midpoint "
        f"`{chosen.recommended_threshold:.4f}` sits in the observed gap with no "
        f"in-scope query below it and no out-of-scope query above it. Pin it with "
        f"`SIMILARITY_THRESHOLD={chosen.recommended_threshold:.4f}` in `.env`.",
    ]
    return "\n".join(lines)



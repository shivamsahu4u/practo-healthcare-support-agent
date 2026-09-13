"""Task 5 - document-level precision and recall for BOTH chunking strategies.


Scoring is at the **document** level, not the chunk level: retrieved chunks are
mapped back to their parent ``document_id`` and deduplicated before anything is
counted. That matters because fixed-size chunking with overlap frequently
returns three chunks of the *same* document, which would otherwise be scored as
three separate hits and flatter that strategy.


    precision@k = |relevant ∩ retrieved_docs| / |retrieved_docs|
    recall@k    = |relevant ∩ retrieved_docs| / |relevant|


The per-query arithmetic is rendered as a string so the report shows the actual
sets and division, not just the final number.
"""


from __future__ import annotations


from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final


from app.config import REPORTS_DIR, SETTINGS, STRATEGY_FIXED, STRATEGY_SENTENCE, Settings
from rag.retriever import Retriever




@dataclass(frozen=True, slots=True)
class DemoQuery:
    """One evaluation query with its document-level ground truth."""


    query_id: str
    query: str
    relevant_document_ids: frozenset[str]
    in_scope: bool = True
    note: str = ""




#: The canonical query set. Tasks 4 and 5 share it, exactly as the brief
#: requires ("the same >= 5 queries"): five in-scope queries with ground truth,
#: plus one deliberately out-of-scope query that must trigger the fallback.
#:
#: Ground truth is the set of documents that genuinely answer the question.
#: ``fees_and_home_visit`` intentionally spans two documents so recall has
#: something to measure beyond 0 or 1.
DEMO_QUERIES: Final[tuple[DemoQuery, ...]] = (
    DemoQuery(
        query_id="cancel_window",
        query="How long before my appointment can I cancel without paying a fee?",
        relevant_document_ids=frozenset({"cancellation_rescheduling"}),
    ),
    DemoQuery(
        query_id="fees_and_home_visit",
        query=(
            "What is the consultation fee for cardiology and how much extra does a "
            "home visit cost?"
        ),
        relevant_document_ids=frozenset({"consultation_fees", "home_visits"}),
        note="Two-document ground truth, so recall@k can land between 0 and 1.",
    ),
    DemoQuery(
        query_id="lab_culture",
        query="How many days does a culture and sensitivity lab test take to report?",
        relevant_document_ids=frozenset({"lab_turnaround"}),
    ),
    DemoQuery(
        query_id="telemedicine_child",
        query="Is a video consultation allowed for a three year old child?",
        relevant_document_ids=frozenset({"telemedicine"}),
    ),
    DemoQuery(
        query_id="followup_discount",
        query="What discount applies to a follow-up visit within two weeks?",
        relevant_document_ids=frozenset({"follow_up_discount"}),
    ),
    DemoQuery(
        query_id="out_of_scope_stock",
        query="What is the current share price of a large technology company?",
        relevant_document_ids=frozenset(),
        in_scope=False,
        note="Deliberately out of scope - must trigger the 'I don't know' fallback.",
    ),
)


IN_SCOPE_DEMO_QUERIES: Final[tuple[DemoQuery, ...]] = tuple(
    query for query in DEMO_QUERIES if query.in_scope
)
OUT_OF_SCOPE_DEMO_QUERIES: Final[tuple[DemoQuery, ...]] = tuple(
    query for query in DEMO_QUERIES if not query.in_scope
)




def _render_set(values: list[str] | frozenset[str]) -> str:
    ordered = sorted(values)
    return "{" + ", ".join(ordered) + "}" if ordered else "{}"




@dataclass(frozen=True, slots=True)
class QueryScore:
    """Precision / recall for one query against one collection."""


    query_id: str
    query: str
    collection_name: str
    strategy: str
    relevant: frozenset[str]
    retrieved_documents: tuple[str, ...]
    retrieved_chunk_ids: tuple[str, ...]
    top_similarity: float


    @property
    def true_positives(self) -> frozenset[str]:
        return self.relevant & frozenset(self.retrieved_documents)


    @property
    def precision(self) -> float:
        if not self.retrieved_documents:
            return 0.0
        return len(self.true_positives) / len(self.retrieved_documents)


    @property
    def recall(self) -> float:
        if not self.relevant:
            return 0.0
        return len(self.true_positives) / len(self.relevant)


    @property
    def precision_arithmetic(self) -> str:
        return (
            f"|TP| / |retrieved docs| = |{_render_set(self.true_positives)}| / "
            f"|{_render_set(list(self.retrieved_documents))}| = "
            f"{len(self.true_positives)}/{len(self.retrieved_documents)} = "
            f"{self.precision:.4f}"
        )


    @property
    def recall_arithmetic(self) -> str:
        return (
            f"|TP| / |relevant docs| = |{_render_set(self.true_positives)}| / "
            f"|{_render_set(self.relevant)}| = "
            f"{len(self.true_positives)}/{len(self.relevant)} = {self.recall:.4f}"
        )


    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "collection_name": self.collection_name,
            "strategy": self.strategy,
            "relevant_document_ids": sorted(self.relevant),
            "retrieved_document_ids": list(self.retrieved_documents),
            "retrieved_chunk_ids": list(self.retrieved_chunk_ids),
            "true_positives": sorted(self.true_positives),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "top_similarity": round(self.top_similarity, 4),
        }




@dataclass(frozen=True, slots=True)
class StrategyScore:
    """Aggregated precision / recall for one collection across all queries."""


    collection_name: str
    strategy: str
    scores: tuple[QueryScore, ...]


    @property
    def mean_precision(self) -> float:
        return round(sum(score.precision for score in self.scores) / len(self.scores), 4)


    @property
    def mean_recall(self) -> float:
        return round(sum(score.recall for score in self.scores) / len(self.scores), 4)


    @property
    def mean_f1(self) -> float:
        precision, recall = self.mean_precision, self.mean_recall
        if precision + recall == 0.0:
            return 0.0
        return round(2 * precision * recall / (precision + recall), 4)


    @property
    def mean_chunks_per_query(self) -> float:
        return round(
            sum(len(score.retrieved_chunk_ids) for score in self.scores) / len(self.scores), 2
        )


    @property
    def mean_documents_per_query(self) -> float:
        return round(
            sum(len(score.retrieved_documents) for score in self.scores) / len(self.scores), 2
        )


    def as_dict(self) -> dict[str, Any]:
        return {
            "collection_name": self.collection_name,
            "strategy": self.strategy,
            "mean_precision": self.mean_precision,
            "mean_recall": self.mean_recall,
            "mean_f1": self.mean_f1,
            "mean_chunks_per_query": self.mean_chunks_per_query,
            "mean_documents_per_query": self.mean_documents_per_query,
            "per_query": [score.as_dict() for score in self.scores],
        }




def score_collection(
    retriever: Retriever,
    collection_name: str,
    *,
    queries: tuple[DemoQuery, ...] = IN_SCOPE_DEMO_QUERIES,
    top_k: int | None = None,
    settings: Settings = SETTINGS,
) -> StrategyScore:
    """Score one collection over the in-scope query set."""
    if not queries:
        raise ValueError("at least one query is required.")
    strategy = settings.strategy_for_collection.get(collection_name, "unknown")


    scores: list[QueryScore] = []
    for demo in queries:
        result = retriever.search(demo.query, collection_name=collection_name, top_k=top_k)
        scores.append(
            QueryScore(
                query_id=demo.query_id,
                query=demo.query,
                collection_name=collection_name,
                strategy=strategy,
                relevant=demo.relevant_document_ids,
                # Already deduplicated and best-rank-first by RetrievalResult.
                retrieved_documents=tuple(result.document_ids),
                retrieved_chunk_ids=tuple(chunk.chunk_id for chunk in result.chunks),
                top_similarity=result.top_similarity,
            )
        )
    return StrategyScore(
        collection_name=collection_name, strategy=strategy, scores=tuple(scores)
    )




def recommend_strategy(scores: dict[str, StrategyScore]) -> tuple[str, str]:
    """Pick the collection to deploy and explain why, from the measured numbers.


    Decision rule, applied in order and stated up front so the recommendation is
    reproducible rather than a judgement call:


    1. higher mean F1 (the balance of precision and recall) wins;
    2. on a tie, higher mean recall wins - a support agent that misses the
       relevant policy is worse than one that shows an extra document;
    3. on a further tie, the strategy that retrieves fewer parent documents per
       query wins, because it feeds less irrelevant context to generation.
    """
    if not scores:
        raise ValueError("no strategy scores to compare.")


    ranked = sorted(
        scores.values(),
        key=lambda score: (-score.mean_f1, -score.mean_recall, score.mean_documents_per_query),
    )
    winner = ranked[0]
    others = ranked[1:]


    sentences = [
        f"Deploy the `{winner.strategy}` strategy (collection `{winner.collection_name}`): "
        f"it scored mean precision {winner.mean_precision:.4f}, mean recall "
        f"{winner.mean_recall:.4f} and mean F1 {winner.mean_f1:.4f} across the five "
        f"in-scope queries, retrieving {winner.mean_documents_per_query} parent "
        f"document(s) per query on average."
    ]
    for other in others:
        sentences.append(
            f"The `{other.strategy}` strategy (collection `{other.collection_name}`) scored "
            f"mean precision {other.mean_precision:.4f}, mean recall {other.mean_recall:.4f} "
            f"and mean F1 {other.mean_f1:.4f} over the same queries, at "
            f"{other.mean_documents_per_query} parent document(s) per query."
        )
    sentences.append(
        "The decision rule was fixed before the numbers were read - highest mean F1, "
        "then highest mean recall, then fewest parent documents per query - so the "
        "recommendation follows from the measurements rather than from preference."
    )
    return winner.collection_name, " ".join(sentences)




def format_report(
    scores: dict[str, StrategyScore],
    recommendation: tuple[str, str],
    *,
    fallback_check: dict[str, Any] | None = None,
) -> str:
    """Render the Task 5 comparison report."""
    winner_collection, rationale = recommendation
    lines = [
        "# Task 5 - Chunking strategy comparison (document-level precision / recall)",
        "",
        "Generated by `python -m scripts.compare_chunking`. Every number is measured",
        "against the live ChromaDB collections; nothing here is hand-written.",
        "",
        f"- **Measured at:** {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "- **Scoring unit:** parent document (chunks mapped back to `document_id` and",
        "  deduplicated before counting)",
        "- **precision@k** = |relevant ∩ retrieved_docs| / |retrieved_docs|",
        "- **recall@k** = |relevant ∩ retrieved_docs| / |relevant|",
        "",
    ]


    for strategy in (STRATEGY_FIXED, STRATEGY_SENTENCE):
        score = next(
            (item for item in scores.values() if item.strategy == strategy), None
        )
        if score is None:
            continue
        lines += [
            f"## Strategy `{score.strategy}` - collection `{score.collection_name}`",
            "",
        ]
        for query_score in score.scores:
            lines += [
                f"### {query_score.query_id}",
                "",
                f"> {query_score.query}",
                "",
                f"- relevant documents: `{_render_set(query_score.relevant)}`",
                f"- retrieved chunks ({len(query_score.retrieved_chunk_ids)}): "
                + ", ".join(f"`{cid}`" for cid in query_score.retrieved_chunk_ids),
                f"- retrieved documents after dedup: "
                f"`{_render_set(list(query_score.retrieved_documents))}`",
                f"- true positives: `{_render_set(query_score.true_positives)}`",
                f"- top-1 similarity: {query_score.top_similarity:.4f}",
                f"- **precision@k** = {query_score.precision_arithmetic}",
                f"- **recall@k** = {query_score.recall_arithmetic}",
                "",
            ]
        lines += [
            f"**Aggregate for `{score.strategy}`:** mean precision "
            f"{score.mean_precision:.4f}, mean recall {score.mean_recall:.4f}, "
            f"mean F1 {score.mean_f1:.4f}, "
            f"{score.mean_chunks_per_query} chunks and "
            f"{score.mean_documents_per_query} parent documents per query.",
            "",
        ]


    lines += [
        "## Side-by-side",
        "",
        "| Strategy | Collection | Mean precision | Mean recall | Mean F1 | Docs/query |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| `{score.strategy}` | `{score.collection_name}` | {score.mean_precision:.4f} "
        f"| {score.mean_recall:.4f} | {score.mean_f1:.4f} "
        f"| {score.mean_documents_per_query} |"
        for score in scores.values()
    )


    lines += [
        "",
        "## Recommendation",
        "",
        rationale,
        "",
        f"Set `RECOMMENDED_COLLECTION_NAME={winner_collection}` in `.env`; the CrewAI",
        "Retrieval Agent reads that setting.",
        "",
    ]


    if fallback_check:
        lines += [
            "## Out-of-scope fallback check",
            "",
            f"- query: _{fallback_check.get('query', '')}_",
            f"- top-1 similarity: {fallback_check.get('top_similarity', 0.0):.4f}",
            f"- calibrated threshold: {fallback_check.get('threshold', 0.0):.4f}",
            f"- grounded: **{fallback_check.get('grounded')}**",
            f"- answer: _{fallback_check.get('answer', '')}_",
            "",
        ]
    return "\n".join(lines)




def format_readme_section(
    scores: dict[str, StrategyScore], recommendation: tuple[str, str]
) -> str:
    """The compact block written into README.md's AUTO:CHUNKING markers.


    Task 5 asks for a 2-3 sentence recommendation citing your own two sets of
    numbers, so both sets and the recommendation live here together.
    """
    winner_collection, rationale = recommendation
    lines = [
        "| Strategy | Collection | Mean precision@k | Mean recall@k | Mean F1 | Docs/query |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| `{score.strategy}` | `{score.collection_name}` | "
        f"**{score.mean_precision:.4f}** | **{score.mean_recall:.4f}** | "
        f"{score.mean_f1:.4f} | {score.mean_documents_per_query} |"
        for score in scores.values()
    )
    lines += [
        "",
        f"**Recommendation:** {rationale}",
        "",
        f"Deployed collection: `{winner_collection}`. Per-query arithmetic for every "
        "query and both collections is in `reports/chunking_comparison.md`.",
    ]
    return "\n".join(lines)




def write_report(
    scores: dict[str, StrategyScore],
    recommendation: tuple[str, str],
    *,
    fallback_check: dict[str, Any] | None = None,
    path: Path | None = None,
) -> Path:
    """Write the Markdown comparison report and return its path."""
    target = path or (REPORTS_DIR / "chunking_comparison.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        format_report(scores, recommendation, fallback_check=fallback_check), encoding="utf-8"
    )
    return target




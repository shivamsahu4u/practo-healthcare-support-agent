"""Task 4 (part 2) - grounded generation with a *calibrated* threshold.


Two rules govern this module:


1. **Context-only answers.** Under ``MOCK_LLM`` the "generation" step selects
   the retrieved sentences that best cover the query and emits them verbatim
   behind a fixed frame. Nothing is paraphrased and nothing is invented, so the
   answer is grounded by construction rather than by hope.


2. **No preset threshold.** ``0.5`` / ``0.6`` / ``0.7`` are tutorial defaults
   that do not reliably separate short policy-sentence embeddings from
   unrelated queries. This module refuses to guess: the threshold comes from
   ``data/generated/calibration.json`` (written by
   ``scripts/calibrate_threshold.py``) or from an explicit
   ``SIMILARITY_THRESHOLD`` pin, and raises ``CalibrationRequiredError``
   otherwise.
"""


from __future__ import annotations


import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final


from app.config import (
    CALIBRATION_FILE,
    FALLBACK_ANSWER,
    SETTINGS,
    Settings,
)
from rag.cache import ResponseCache, make_cache_key, normalise_query
from rag.indexer import current_kb_version
from rag.retriever import RetrievalResult, Retriever
from rag.textutils import select_relevant_sentences


LOGGER: Final = logging.getLogger(__name__)


#: How many retrieved sentences an answer may quote. Three keeps the answer
#: readable while still covering multi-part policy questions.
MAX_ANSWER_SENTENCES: Final[int] = 3


class CalibrationRequiredError(RuntimeError):
    """Raised when no measured similarity threshold is available."""


# --------------------------------------------------------------------------- #
# Threshold resolution
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ThresholdSource:
    """Where the active threshold came from - reported in transcripts and logs."""


    value: float
    origin: str
    detail: str


def resolve_threshold(
    settings: Settings = SETTINGS, *, calibration_file: Path = CALIBRATION_FILE
) -> ThresholdSource:
    """Resolve the similarity threshold, refusing to invent one.


    Precedence:
        1. ``SIMILARITY_THRESHOLD`` - an explicit pin of a previously measured
           value, so a graded run reproduces on a fresh checkout.
        2. ``data/generated/calibration.json`` - written by the calibration script.


    Raises:
        CalibrationRequiredError: when neither is available, or the calibration
            file exists but is unusable.
    """
    override = settings.similarity_threshold_override
    if override is not None:
        if not -1.0 <= override <= 1.0:
            raise CalibrationRequiredError(
                f"SIMILARITY_THRESHOLD={override} is outside the cosine range [-1, 1]."
            )
        return ThresholdSource(
            value=override,
            origin="environment",
            detail="SIMILARITY_THRESHOLD pinned in the environment",
        )


    if not calibration_file.is_file():
        raise CalibrationRequiredError(
            "no calibrated similarity threshold is available. The brief forbids an "
            "untested preset, so this step refuses to guess one. Run:\n"
            "    python -m scripts.build_indexes\n"
            "    python -m scripts.calibrate_threshold\n"
            f"which measures in-scope vs out-of-scope similarity and writes {calibration_file}."
        )


    try:
        payload = json.loads(calibration_file.read_text(encoding="utf-8"))
        value = float(payload["recommended_threshold"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise CalibrationRequiredError(
            f"calibration file {calibration_file} is unusable ({exc}). Delete it and "
            "re-run `python -m scripts.calibrate_threshold`."
        ) from exc


    return ThresholdSource(
        value=value,
        origin="calibration_file",
        detail=(
            f"measured by scripts/calibrate_threshold.py on "
            f"{payload.get('measured_at', 'an unrecorded date')} "
            f"(collection {payload.get('collection_name', 'unknown')})"
        ),
    )


# --------------------------------------------------------------------------- #
# Answers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    """The result of one grounded-generation call."""


    query: str
    answer: str
    grounded: bool
    sources: tuple[str, ...]
    top_similarity: float
    threshold: float
    collection_name: str
    context_text: str
    chunk_ids: tuple[str, ...]
    cache_hit: bool = False
    elapsed_ms: float = 0.0
    retrieved: tuple[dict[str, Any], ...] = field(default_factory=tuple)


    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "grounded": self.grounded,
            "sources": list(self.sources),
            "top_similarity": round(self.top_similarity, 4),
            "threshold": round(self.threshold, 4),
            "collection_name": self.collection_name,
            "chunk_ids": list(self.chunk_ids),
            "cache_hit": self.cache_hit,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


    def with_cache_hit(self, *, elapsed_ms: float) -> "GroundedAnswer":
        """Copy marked as a cache hit, with this call's own timing."""
        return GroundedAnswer(
            query=self.query,
            answer=self.answer,
            grounded=self.grounded,
            sources=self.sources,
            top_similarity=self.top_similarity,
            threshold=self.threshold,
            collection_name=self.collection_name,
            context_text=self.context_text,
            chunk_ids=self.chunk_ids,
            cache_hit=True,
            elapsed_ms=elapsed_ms,
            retrieved=self.retrieved,
        )


def _attributed_topic(sentence: str, result: RetrievalResult) -> str:
    """Topic title of the retrieved chunk that actually contains ``sentence``."""
    needle = " ".join(sentence.split())
    for chunk in result.chunks:
        if needle in " ".join(chunk.text.split()):
            return chunk.topic_title or "policy"
    # A sentence straddling a chunk boundary belongs to no single chunk: fixed-size
    # chunking snaps to word boundaries, not sentence boundaries. Fall back to the
    # best-ranked chunk, which is what the whole answer used to be attributed to.
    return result.chunks[0].topic_title or "policy"


def compose_grounded_answer(query: str, result: RetrievalResult) -> str:
    """Build the answer text from retrieved context only.


    Each frame names the policy topic the sentences beneath it were actually
    retrieved from; the body is verbatim retrieved sentences. No paraphrase, no
    synthesis, no outside knowledge.


    Attribution is per source document, not per answer. ``context_text`` is the
    concatenation of *every* retrieved chunk, so a single frame naming
    ``chunks[0]`` misattributed any sentence drawn from a lower-ranked chunk of a
    different document - visible on a genuinely multi-document question such as
    "the cardiology fee and the home-visit surcharge", where the prose credited
    one document while ``sources`` correctly listed two.
    """
    sentences = select_relevant_sentences(query, result.context_text, MAX_ANSWER_SENTENCES)
    if not sentences:
        return FALLBACK_ANSWER


    # Group consecutive sentences under their own source document, preserving
    # context order so the answer still reads in the order the policy is written.
    groups: list[tuple[str, list[str]]] = []
    for sentence in sentences:
        topic = _attributed_topic(sentence, result)
        if groups and groups[-1][0] == topic:
            groups[-1][1].append(sentence)
        else:
            groups.append((topic, [sentence]))


    return " ".join(
        f"According to Practo's {topic}: " + " ".join(quoted) for topic, quoted in groups
    )


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class GenerationStats:
    """Call counters - the before/after evidence for Task 16."""


    generations: int = 0
    cache_hits: int = 0
    fallbacks: int = 0


    def as_dict(self) -> dict[str, Any]:
        return {
            "generations": self.generations,
            "cache_hits": self.cache_hits,
            "fallbacks": self.fallbacks,
        }


class GroundedGenerator:
    """Retrieval + context-only generation + response cache.


    ``stats.generations`` counts only the calls that actually did retrieval work.
    A cache hit leaves it untouched, which is precisely the Task 16 evidence.
    """


    def __init__(
        self,
        retriever: Retriever | None = None,
        cache: ResponseCache[GroundedAnswer] | None = None,
        settings: Settings = SETTINGS,
        *,
        threshold_source: ThresholdSource | None = None,
    ) -> None:
        self._settings = settings
        self.retriever = retriever or Retriever(settings)
        self.cache: ResponseCache[GroundedAnswer] = cache or ResponseCache(
            max_entries=settings.cache_max_entries, enabled=settings.cache_enabled
        )
        self._threshold_source = threshold_source
        self.stats = GenerationStats()


    # ------------------------------------------------------------------ #


    @property
    def threshold_source(self) -> ThresholdSource:
        """Resolved lazily so importing this module never demands calibration."""
        if self._threshold_source is None:
            self._threshold_source = resolve_threshold(self._settings)
        return self._threshold_source


    @property
    def threshold(self) -> float:
        return self.threshold_source.value


    @property
    def embedder_identity(self) -> str:
        """Configuration identity of the active embedder, for the cache key.


        Derived from settings rather than from ``retriever.embedder`` on purpose:
        building a cache key must never construct an embedder or touch a model
        artefact, because ``GET /health`` reads cache state and is documented as
        never loading a model.
        """
        model = (
            self._settings.embedding_model_path or self._settings.embedding_model_name
        )
        return f"{self._settings.embedding_backend}:{model}"


    def reset_counters(self) -> None:
        self.stats = GenerationStats()
        self.retriever.reset_counters()
        self.cache.reset_stats()


    def invalidate_cache(self) -> int:
        """Drop every cached answer. Called by ``POST /add-document``."""
        return self.cache.invalidate()


    def refresh_threshold(self) -> bool:
        """Forget the resolved threshold so the next call re-reads calibration.


        ``threshold_source`` memoises on first resolution, and this object lives
        as long as the process, so re-running ``scripts.calibrate_threshold``
        against a running server had no effect until restart.


        Note: this also discards a ``threshold_source`` that was injected through
        the constructor, so the next resolution goes to the environment pin or
        the calibration file. Callers that injected one should re-inject it.


        Returns:
            Whether a memoised value was actually dropped.
        """
        had_value = self._threshold_source is not None
        self._threshold_source = None
        return had_value


    # ------------------------------------------------------------------ #


    def generate(
        self,
        query: str,
        *,
        collection_name: str | None = None,
        top_k: int | None = None,
    ) -> GroundedAnswer:
        """Answer ``query`` from retrieved context, or return the fallback.


        A cache hit short-circuits before any retrieval or generation work, so
        neither ``Retriever.query_count`` nor ``stats.generations`` advances.
        """
        started = time.perf_counter()
        text = (query or "").strip()
        if not text:
            raise ValueError("query must not be empty.")


        name = collection_name or self._settings.recommended_collection_name
        k = top_k if top_k is not None else self._settings.top_k
        threshold = self.threshold


        key = make_cache_key(
            text,
            collection_name=name,
            top_k=k,
            threshold=threshold,
            kb_version=current_kb_version(),
            embedder=self.embedder_identity,
        )
        cached = self.cache.get(key)
        if cached is not None:
            self.stats.cache_hits += 1
            elapsed = (time.perf_counter() - started) * 1000.0
            LOGGER.debug("grounded-generation cache hit for %r", normalise_query(text))
            return cached.with_cache_hit(elapsed_ms=elapsed)


        result = self.retriever.search(text, collection_name=name, top_k=k)
        self.stats.generations += 1


        grounded = bool(result.chunks) and result.top_similarity >= threshold
        if grounded:
            answer = compose_grounded_answer(text, result)
            sources = tuple(result.document_ids)
        else:
            # Below the calibrated threshold, retrieval is the only groundedness
            # signal available under MOCK_LLM - so refuse rather than answer.
            answer = FALLBACK_ANSWER
            sources = ()
            self.stats.fallbacks += 1


        generated = GroundedAnswer(
            query=text,
            answer=answer,
            grounded=grounded,
            sources=sources,
            top_similarity=result.top_similarity,
            threshold=threshold,
            collection_name=name,
            context_text=result.context_text,
            chunk_ids=tuple(chunk.chunk_id for chunk in result.chunks),
            cache_hit=False,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            retrieved=tuple(chunk.as_dict() for chunk in result.chunks),
        )
        self.cache.set(key, generated)
        return generated



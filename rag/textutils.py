"""Shared deterministic text utilities.


One implementation, three consumers: query-relevant sentence selection during
grounded generation, the output-side groundedness guardrail, and the Autogen
reviewer's unsupported-claim detector. Keeping it here is what stops the same
overlap arithmetic from being reimplemented three slightly different ways.
"""


from __future__ import annotations


import re
from typing import Final


_TOKEN: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+")


#: Function words carry no topical signal, so they are excluded before any
#: overlap is measured. Kept small and explicit rather than pulled from a
#: heavyweight NLP dependency.
STOPWORDS: Final[frozenset[str]] = frozenset(
    """
    a about above after again against all am an and any are aren as at be because been
    before being below between both but by can cannot could couldn did didn do does
    doesn doing don down during each few for from further had hadn has hasn have haven
    having he her here hers herself him himself his how i if in into is isn it its
    itself just me more most must my myself no nor not now of off on once only or other
    ought our ours ourselves out over own same shan she should shouldn so some such than
    that the their theirs them themselves then there these they this those through to
    too under until up very was wasn we were weren what when where which while who whom
    why will with won would wouldn you your yours yourself yourselves get got tell
    please need want know like also may might shall
    """.split()
)


_SENTENCE_BOUNDARY: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into non-empty, stripped sentences."""
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(text or "") if part.strip()]


def content_words(text: str) -> list[str]:
    """Tokenise to lowercase content words, dropping stopwords and 1-2 letter noise.


    Digits are always kept: ``"4"`` in "4 hours" and ``"30"`` in "30 days" are
    exactly the tokens that distinguish one Practo policy answer from another.
    """
    words: list[str] = []
    for token in _TOKEN.findall((text or "").lower()):
        if token in STOPWORDS:
            continue
        if len(token) < 3 and not token.isdigit():
            continue
        words.append(token)
    return words


def content_word_set(text: str) -> frozenset[str]:
    """``content_words`` as a set, for intersection arithmetic."""
    return frozenset(content_words(text))


def overlap_ratio(candidate: str, reference: str) -> float:
    """Fraction of ``candidate``'s content words that also occur in ``reference``.


    Returns ``1.0`` for a candidate with no content words at all: a sentence
    made entirely of function words asserts nothing, so there is nothing in it
    that could be ungrounded.
    """
    candidate_words = content_word_set(candidate)
    if not candidate_words:
        return 1.0
    reference_words = content_word_set(reference)
    shared = candidate_words & reference_words
    return len(shared) / len(candidate_words)


def keyword_score(query: str, sentence: str) -> float:
    """How much of ``query``'s vocabulary the sentence covers, in ``[0, 1]``.


    Used to pick which retrieved sentences answer the question. Symmetric to
    ``overlap_ratio`` but oriented the other way round: here the query is the
    candidate whose coverage is being measured.
    """
    query_words = content_word_set(query)
    if not query_words:
        return 0.0
    return len(query_words & content_word_set(sentence)) / len(query_words)


def select_relevant_sentences(query: str, context: str, limit: int) -> list[str]:
    """Pick the ``limit`` sentences from ``context`` that best cover ``query``.


    Sentences are returned **verbatim** and in their original context order, so
    the generated answer contains only text that is literally present in the
    retrieved chunks. Selection is stable: ties break on original position.
    """
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}.")


    sentences = split_sentences(context)
    if not sentences:
        return []


    # Deduplicate: fixed-size chunking overlaps, so the same sentence can be
    # retrieved twice in two adjacent chunks.
    unique: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        fingerprint = " ".join(content_words(sentence))
        if fingerprint and fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(sentence)


    ranked = sorted(
        enumerate(unique),
        key=lambda pair: (-keyword_score(query, pair[1]), pair[0]),
    )
    chosen = sorted(ranked[:limit], key=lambda pair: pair[0])
    return [sentence for _, sentence in chosen]


def unsupported_sentences(
    draft: str,
    support: str,
    minimum_overlap: float,
    exempt: frozenset[str] = frozenset(),
) -> list[str]:
    """Sentences in ``draft`` whose content-word overlap with ``support`` is too low.


    This is the deterministic stand-in for a semantic entailment check under
    ``MOCK_LLM``. It is lexical, and that limitation is stated in the README: a
    paraphrase of a supported fact could be flagged, and a fabricated claim
    built entirely from context vocabulary could slip through.


    Args:
        exempt: sentences to accept **by identity** rather than by vocabulary.


            This exists so fixed control text - the "I don't know" refusal, the
            request for an appointment id - does not have to be poured into
            ``support`` to pass. Dumping it there granted every request a free
            vocabulary ("know", "available", "knowledge", "specific", "share"),
            so any drafted sentence reusing those words scored as supported
            regardless of provenance. Exempting the exact strings removes that
            loophole while keeping the control answers passing.
    """
    if not 0.0 < minimum_overlap <= 1.0:
        raise ValueError(f"minimum_overlap must be in (0, 1], got {minimum_overlap}.")
    return [
        sentence
        for sentence in split_sentences(draft)
        if sentence.strip() not in exempt
        and overlap_ratio(sentence, support) < minimum_overlap
    ]



"""Task 3 (part 1) - knowledge-base loading and the two chunking strategies.


Two strategies are implemented side by side so Task 5 can compare them:


* ``fixed_size_overlap`` - a sliding character window of ``FIXED_CHUNK_SIZE``
  with ``FIXED_CHUNK_OVERLAP`` characters of carry-over, snapped back to a word
  boundary so a chunk never ends mid-word.
* ``sentence_based`` - sentences grouped ``SENTENCES_PER_CHUNK`` at a time with
  no overlap, so every chunk is a whole number of policy sentences.


Both strategies emit ``Chunk`` objects carrying the metadata Task 5 needs to map
a retrieved chunk back to its parent document.
"""


from __future__ import annotations


import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final


from app.config import (
    KB_TOPIC_BY_SLUG,
    KNOWLEDGE_BASE_DIR,
    STRATEGIES,
    STRATEGY_FIXED,
    STRATEGY_SENTENCE,
    SETTINGS,
    Settings,
)


#: Sentence boundary: a terminator followed by whitespace. Deliberately simple
#: and dependency-free - the knowledge base is authored without abbreviations
#: like "e.g." precisely so this stays correct.
_SENTENCE_BOUNDARY: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?])\s+")


_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")
_HEADING: Final[re.Pattern[str]] = re.compile(r"^#\s+(?P<title>.+?)\s*$", re.MULTILINE)


class KnowledgeBaseError(RuntimeError):
    """Raised when the knowledge base on disk is missing or malformed."""


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class KbDocument:
    """One knowledge-base document.


    ``document_id`` is the filename stem and is the unit of relevance for the
    document-level precision / recall in Task 5.
    """


    document_id: str
    title: str
    body: str
    source_filename: str


    @property
    def is_required_topic(self) -> bool:
        """True when this document covers one of the twelve required topics."""
        return self.document_id in KB_TOPIC_BY_SLUG


def normalise_whitespace(text: str) -> str:
    """Collapse all whitespace runs to single spaces and strip the ends."""
    return _WHITESPACE.sub(" ", text).strip()


def parse_document(path: Path) -> KbDocument:
    """Parse one Markdown knowledge-base file into a ``KbDocument``.


    The first ``# Heading`` becomes the title and is removed from the body, so
    the title lives in chunk metadata rather than being duplicated into every
    chunk's text.


    Raises:
        KnowledgeBaseError: when the file has no heading or no body text.
    """
    raw = path.read_text(encoding="utf-8")
    match = _HEADING.search(raw)
    if match is None:
        raise KnowledgeBaseError(
            f"{path.name} has no '# Title' heading; every knowledge-base document needs one."
        )
    title = match.group("title").strip()
    body = normalise_whitespace(raw[: match.start()] + raw[match.end() :])
    if not body:
        raise KnowledgeBaseError(f"{path.name} has a heading but no body text.")
    return KbDocument(
        document_id=path.stem,
        title=title,
        body=body,
        source_filename=path.name,
    )


def load_knowledge_base(directory: Path | None = None) -> list[KbDocument]:
    """Load every ``*.md`` document, sorted by ``document_id`` for determinism.


    Raises:
        KnowledgeBaseError: when the directory is absent, empty, or any of the
            twelve required topics has no document.
    """
    target = directory or KNOWLEDGE_BASE_DIR
    if not target.is_dir():
        raise KnowledgeBaseError(f"knowledge-base directory not found: {target}")


    documents = [parse_document(path) for path in sorted(target.glob("*.md"))]
    if not documents:
        raise KnowledgeBaseError(f"no *.md documents found in {target}")


    present = {document.document_id for document in documents}
    missing = [slug for slug in KB_TOPIC_BY_SLUG if slug not in present]
    if missing:
        raise KnowledgeBaseError(
            "knowledge base is missing required topic document(s): " + ", ".join(missing)
        )
    return documents


# --------------------------------------------------------------------------- #
# Chunks
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Chunk:
    """One embeddable unit of text plus the metadata needed to score it."""


    chunk_id: str
    document_id: str
    source_filename: str
    topic_title: str
    strategy: str
    chunk_index: int
    text: str


    def metadata(self) -> dict[str, Any]:
        """ChromaDB metadata payload. Scalars only - Chroma rejects nesting."""
        return {
            "document_id": self.document_id,
            "source_filename": self.source_filename,
            "topic_title": self.topic_title,
            "strategy": self.strategy,
            "chunk_index": self.chunk_index,
        }


def split_sentences(text: str) -> list[str]:
    """Split normalised text into non-empty sentences."""
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(normalise_whitespace(text)) if part.strip()]


def chunk_fixed_size(text: str, size: int, overlap: int) -> list[str]:
    """Fixed-size character windows with overlap, snapped to word boundaries.


    Args:
        text: source text (whitespace is normalised first).
        size: maximum characters per window.
        overlap: characters of the previous window carried into the next one.


    Raises:
        ValueError: if ``overlap >= size``, which could not advance.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}.")
    if overlap < 0:
        raise ValueError(f"overlap must not be negative, got {overlap}.")
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size}).")


    normalised = normalise_whitespace(text)
    if not normalised:
        return []
    if len(normalised) <= size:
        return [normalised]


    windows: list[str] = []
    start = 0
    length = len(normalised)
    while start < length:
        # The overlap offset can land inside the last word of the previous
        # window. Move to the next word boundary before opening the new one.
        if start > 0 and normalised[start - 1] != " ":
            boundary = normalised.find(" ", start)
            if boundary < 0:
                break
            start = boundary + 1
        end = min(start + size, length)
        if end < length:
            # Snap back to the last space so words stay intact. `start + 1`
            # keeps rfind from returning `start` itself and stalling.
            boundary = normalised.rfind(" ", start + 1, end)
            if boundary > start:
                end = boundary
        piece = normalised[start:end].strip()
        if piece:
            windows.append(piece)
        if end >= length:
            break
        # max(..., start + 1) is the belt-and-braces guard against a window so
        # short that `end - overlap` would not move forward.
        start = max(end - overlap, start + 1)
    return windows


def chunk_sentences(text: str, sentences_per_chunk: int) -> list[str]:
    """Group whole sentences into chunks of ``sentences_per_chunk``, no overlap."""
    if sentences_per_chunk <= 0:
        raise ValueError(f"sentences_per_chunk must be positive, got {sentences_per_chunk}.")
    sentences = split_sentences(text)
    return [
        " ".join(sentences[index : index + sentences_per_chunk])
        for index in range(0, len(sentences), sentences_per_chunk)
    ]


def chunk_document(
    document: KbDocument, strategy: str, settings: Settings = SETTINGS
) -> list[Chunk]:
    """Chunk one document with the named strategy.


    Raises:
        ValueError: for an unknown strategy name.
    """
    if strategy == STRATEGY_FIXED:
        pieces = chunk_fixed_size(
            document.body, settings.fixed_chunk_size, settings.fixed_chunk_overlap
        )
    elif strategy == STRATEGY_SENTENCE:
        pieces = chunk_sentences(document.body, settings.sentences_per_chunk)
    else:
        raise ValueError(f"unknown chunking strategy {strategy!r}; expected one of {list(STRATEGIES)}.")


    return [
        Chunk(
            chunk_id=f"{strategy}::{document.document_id}::{index:03d}",
            document_id=document.document_id,
            source_filename=document.source_filename,
            topic_title=document.title,
            strategy=strategy,
            chunk_index=index,
            text=piece,
        )
        for index, piece in enumerate(pieces)
    ]


def chunk_corpus(
    documents: list[KbDocument], strategy: str, settings: Settings = SETTINGS
) -> list[Chunk]:
    """Chunk every document with one strategy, preserving document order."""
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, strategy, settings))
    if not chunks:
        raise KnowledgeBaseError(f"strategy {strategy!r} produced no chunks.")
    return chunks


def chunk_corpus_all_strategies(
    documents: list[KbDocument], settings: Settings = SETTINGS
) -> dict[str, list[Chunk]]:
    """Chunk the corpus with both strategies. Keys are strategy identifiers."""
    return {strategy: chunk_corpus(documents, strategy, settings) for strategy in STRATEGIES}

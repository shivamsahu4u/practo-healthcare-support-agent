"""Task 3 (part 3) - two separate ChromaDB collections, one per chunking strategy.


Each strategy gets its OWN persistent collection and the two are never mixed.
Embeddings are always computed here and passed to ``collection.upsert()``
explicitly, with ``embedding_function=None`` on the collection - that keeps
ChromaDB from quietly constructing its default embedding function (which would
try to fetch its own model artefact and break the offline guarantee).


The module also owns the knowledge-base version counter. Every rebuild or
document addition bumps it, and the response cache keys on it, so adding a
document can never serve a stale cached answer.
"""


from __future__ import annotations


import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final


from app.config import (
    INDEX_STATE_FILE,
    KNOWLEDGE_BASE_DIR,
    SETTINGS,
    STRATEGIES,
    Settings,
    ensure_runtime_directories,
)
from rag.chunking import (
    Chunk,
    KbDocument,
    KnowledgeBaseError,
    chunk_corpus,
    load_knowledge_base,
    normalise_whitespace,
    parse_document,
)
from rag.embeddings import Embedder, build_embedder


LOGGER: Final = logging.getLogger(__name__)


#: A document slug must be a plain lowercase identifier. This also makes path
#: traversal impossible: no dots, no slashes, no backslashes can match.
SLUG_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]{2,47}$")


MIN_DOCUMENT_SENTENCES: Final[int] = 2
MAX_DOCUMENT_CHARACTERS: Final[int] = 4000


class VectorIndexError(RuntimeError):
    """Raised when the vector index cannot be built or read."""


class DocumentRejectedError(ValueError):
    """Raised when a submitted knowledge-base document fails validation."""


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


def get_chroma_client(settings: Settings = SETTINGS) -> Any:
    """Open (or create) the persistent ChromaDB store.


    Raises:
        VectorIndexError: when chromadb is not installed or the store cannot open.
    """
    ensure_runtime_directories(settings)
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
    except ImportError as exc:
        raise VectorIndexError(
            "chromadb is not installed. Install the declared baseline with "
            "`pip install -r requirements.txt`."
        ) from exc


    try:
        return chromadb.PersistentClient(
            path=str(settings.chroma_persist_directory),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with the resolved path
        raise VectorIndexError(
            f"could not open the ChromaDB store at {settings.chroma_persist_directory}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def get_collection(client: Any, name: str) -> Any:
    """Get or create a cosine-space collection with no embedding function.


    ``embedding_function=None`` is deliberate: every write passes explicit
    vectors, so ChromaDB must never try to embed anything itself.
    """
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )


# --------------------------------------------------------------------------- #
# Index state / knowledge-base version
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class IndexState:
    """Persisted description of what is currently indexed."""


    kb_version: int = 0
    built_at: str = ""
    embedder: str = ""
    document_ids: list[str] = field(default_factory=list)
    chunk_counts: dict[str, int] = field(default_factory=dict)
    collection_names: dict[str, str] = field(default_factory=dict)


    def as_dict(self) -> dict[str, Any]:
        return {
            "kb_version": self.kb_version,
            "built_at": self.built_at,
            "embedder": self.embedder,
            "document_ids": list(self.document_ids),
            "chunk_counts": dict(self.chunk_counts),
            "collection_names": dict(self.collection_names),
        }


def read_index_state(path: Path = INDEX_STATE_FILE) -> IndexState:
    """Read the index state file, returning a zeroed state when absent."""
    if not path.is_file():
        return IndexState()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VectorIndexError(
            f"index state file {path} is unreadable ({exc}). Delete it and re-run "
            "`python -m scripts.build_indexes`."
        ) from exc
    return IndexState(
        kb_version=int(payload.get("kb_version", 0)),
        built_at=str(payload.get("built_at", "")),
        embedder=str(payload.get("embedder", "")),
        document_ids=list(payload.get("document_ids", [])),
        chunk_counts=dict(payload.get("chunk_counts", {})),
        collection_names=dict(payload.get("collection_names", {})),
    )


def write_index_state(state: IndexState, path: Path = INDEX_STATE_FILE) -> None:
    """Persist the index state file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.as_dict(), indent=2) + "\n", encoding="utf-8")


def current_kb_version(path: Path = INDEX_STATE_FILE) -> int:
    """The knowledge-base version the cache keys on."""
    return read_index_state(path).kb_version


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #


def upsert_chunks(collection: Any, chunks: list[Chunk], embedder: Embedder) -> int:
    """Embed and upsert ``chunks`` into ``collection``. Returns the count written."""
    if not chunks:
        return 0
    embeddings = embedder.encode([chunk.text for chunk in chunks])
    if len(embeddings) != len(chunks):
        raise VectorIndexError(
            f"embedder returned {len(embeddings)} vectors for {len(chunks)} chunks."
        )
    collection.upsert(
        ids=[chunk.chunk_id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        metadatas=[chunk.metadata() for chunk in chunks],
        embeddings=embeddings,
    )
    return len(chunks)


@dataclass(slots=True)
class BuildResult:
    """Outcome of a full index build."""


    documents: int
    chunk_counts: dict[str, int]
    collection_counts: dict[str, int]
    embedder: str
    kb_version: int


    def summary_lines(self) -> list[str]:
        lines = [
            f"documents indexed : {self.documents}",
            f"embedder          : {self.embedder}",
            f"kb_version        : {self.kb_version}",
        ]
        lines.extend(
            f"strategy {strategy:<20} -> {count} chunks"
            for strategy, count in self.chunk_counts.items()
        )
        lines.extend(
            f"collection {name:<18} -> {count} stored vectors"
            for name, count in self.collection_counts.items()
        )
        return lines


def build_indexes(
    settings: Settings = SETTINGS,
    *,
    client: Any | None = None,
    embedder: Embedder | None = None,
    documents: list[KbDocument] | None = None,
    reset: bool = True,
) -> BuildResult:
    """Chunk the knowledge base both ways and index each strategy separately.


    Args:
        reset: when True (default) each collection is dropped first, so a
            removed or renamed document cannot leave orphan vectors behind.
    """
    ensure_runtime_directories(settings)
    active_client = client or get_chroma_client(settings)
    active_embedder = embedder or build_embedder(settings)
    corpus = documents if documents is not None else load_knowledge_base()


    collection_for_strategy = settings.collection_for_strategy
    chunk_counts: dict[str, int] = {}
    collection_counts: dict[str, int] = {}


    for strategy in STRATEGIES:
        name = collection_for_strategy[strategy]
        if reset:
            try:
                active_client.delete_collection(name=name)
            except Exception:  # noqa: BLE001
                # Absent on a first build; nothing to drop. Any real failure
                # surfaces immediately below when the collection is recreated.
                LOGGER.debug("collection %s did not exist prior to build", name)


        collection = get_collection(active_client, name)
        chunks = chunk_corpus(corpus, strategy, settings)
        chunk_counts[strategy] = upsert_chunks(collection, chunks, active_embedder)
        collection_counts[name] = collection.count()
        LOGGER.info(
            "indexed %s chunks for strategy %s into collection %s",
            chunk_counts[strategy],
            strategy,
            name,
        )


    previous = read_index_state()
    state = IndexState(
        kb_version=previous.kb_version + 1,
        built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        embedder=active_embedder.name,
        document_ids=[document.document_id for document in corpus],
        chunk_counts=chunk_counts,
        collection_names=dict(collection_for_strategy),
    )
    write_index_state(state)


    return BuildResult(
        documents=len(corpus),
        chunk_counts=chunk_counts,
        collection_counts=collection_counts,
        embedder=active_embedder.name,
        kb_version=state.kb_version,
    )


# --------------------------------------------------------------------------- #
# Runtime document addition (POST /add-document)
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class AddDocumentResult:
    """Outcome of indexing one new document at runtime."""


    document_id: str
    title: str
    source_filename: str
    chunk_ids: dict[str, list[str]]
    kb_version: int


    @property
    def total_chunks(self) -> int:
        return sum(len(ids) for ids in self.chunk_ids.values())


def validate_document_submission(topic_slug: str, title: str, content: str) -> tuple[str, str, str]:
    """Validate a submitted document. Returns the cleaned ``(slug, title, body)``.


    Raises:
        DocumentRejectedError: on a bad slug, an empty title, content that is
            too short or too long, or a slug that already exists.
    """
    slug = (topic_slug or "").strip()
    if not SLUG_PATTERN.fullmatch(slug):
        raise DocumentRejectedError(
            f"topic_slug {topic_slug!r} is invalid. Use 3-48 characters, lowercase "
            "letters, digits and underscores, starting with a letter. This pattern "
            "also makes filesystem path traversal impossible."
        )


    clean_title = normalise_whitespace(title or "")
    if not clean_title:
        raise DocumentRejectedError("title must not be empty.")
    if len(clean_title) > 120:
        raise DocumentRejectedError(f"title is {len(clean_title)} characters; keep it under 120.")


    body = normalise_whitespace(content or "")
    if not body:
        raise DocumentRejectedError("content must not be empty.")
    if len(body) > MAX_DOCUMENT_CHARACTERS:
        raise DocumentRejectedError(
            f"content is {len(body)} characters; the limit is {MAX_DOCUMENT_CHARACTERS}."
        )
    sentence_count = sum(1 for char in body if char in ".!?")
    if sentence_count < MIN_DOCUMENT_SENTENCES:
        raise DocumentRejectedError(
            f"content needs at least {MIN_DOCUMENT_SENTENCES} sentences; found "
            f"{sentence_count} sentence terminator(s)."
        )


    destination = KNOWLEDGE_BASE_DIR / f"{slug}.md"
    if destination.exists():
        raise DocumentRejectedError(
            f"a knowledge-base document already exists for topic_slug {slug!r}. "
            "Pick a different slug; existing policy documents are not overwritten "
            "through the API."
        )
    return slug, clean_title, body


def add_document(
    topic_slug: str,
    title: str,
    content: str,
    settings: Settings = SETTINGS,
    *,
    client: Any | None = None,
    embedder: Embedder | None = None,
) -> AddDocumentResult:
    """Validate, persist and index one new knowledge-base document.


    The document is written under ``data/knowledge_base/`` using only the
    validated slug - the caller can never influence the path. It is then chunked
    with BOTH strategies and upserted into BOTH collections, and the
    knowledge-base version is bumped so cached answers are invalidated.
    """
    slug, clean_title, body = validate_document_submission(topic_slug, title, content)


    ensure_runtime_directories(settings)
    destination = KNOWLEDGE_BASE_DIR / f"{slug}.md"
    destination.write_text(f"# {clean_title}\n\n{body}\n", encoding="utf-8")


    try:
        document = parse_document(destination)
    except KnowledgeBaseError:
        destination.unlink(missing_ok=True)
        raise


    active_client = client or get_chroma_client(settings)
    active_embedder = embedder or build_embedder(settings)
    collection_for_strategy = settings.collection_for_strategy


    chunk_ids: dict[str, list[str]] = {}
    for strategy in STRATEGIES:
        collection = get_collection(active_client, collection_for_strategy[strategy])
        chunks = chunk_corpus([document], strategy, settings)
        upsert_chunks(collection, chunks, active_embedder)
        chunk_ids[strategy] = [chunk.chunk_id for chunk in chunks]


    state = read_index_state()
    state.kb_version += 1
    state.built_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    state.embedder = active_embedder.name
    if document.document_id not in state.document_ids:
        state.document_ids.append(document.document_id)
    for strategy, ids in chunk_ids.items():
        state.chunk_counts[strategy] = state.chunk_counts.get(strategy, 0) + len(ids)
    state.collection_names = dict(collection_for_strategy)
    write_index_state(state)


    LOGGER.info(
        "added knowledge-base document %s (%s chunks) -> kb_version %s",
        document.document_id,
        sum(len(ids) for ids in chunk_ids.values()),
        state.kb_version,
    )
    return AddDocumentResult(
        document_id=document.document_id,
        title=document.title,
        source_filename=document.source_filename,
        chunk_ids=chunk_ids,
        kb_version=state.kb_version,
    )



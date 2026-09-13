"""Task 4 (part 1) - top-k retrieval from either ChromaDB collection.


Collections are created with ``hnsw:space = cosine``, so ChromaDB returns a
cosine *distance*. Similarity is ``1 - distance``, which for unit-length
embeddings is exactly cosine similarity in ``[-1, 1]``. Every score reported by
this module - and therefore every calibration measurement - is that similarity.
"""


from __future__ import annotations


import logging
from dataclasses import dataclass
from typing import Any, Final


from app.config import SETTINGS, Settings
from rag.embeddings import Embedder, build_embedder
from rag.indexer import VectorIndexError, get_chroma_client, get_collection


LOGGER: Final = logging.getLogger(__name__)




class RetrievalError(RuntimeError):
    """Raised when a retrieval query cannot be served."""




@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """One retrieved chunk plus its similarity to the query."""


    chunk_id: str
    document_id: str
    topic_title: str
    source_filename: str
    strategy: str
    chunk_index: int
    text: str
    similarity: float
    distance: float


    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "topic_title": self.topic_title,
            "strategy": self.strategy,
            "chunk_index": self.chunk_index,
            "similarity": round(self.similarity, 4),
            "text": self.text,
        }




@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """The full outcome of one retrieval query."""


    query: str
    collection_name: str
    top_k: int
    chunks: tuple[RetrievedChunk, ...]


    @property
    def top_similarity(self) -> float:
        """Similarity of the best chunk, or ``0.0`` when nothing was retrieved."""
        return self.chunks[0].similarity if self.chunks else 0.0


    @property
    def document_ids(self) -> list[str]:
        """Parent document ids, deduplicated, best-rank first.


        This is the unit of relevance for the Task 5 precision / recall scoring.
        """
        seen: list[str] = []
        for chunk in self.chunks:
            if chunk.document_id not in seen:
                seen.append(chunk.document_id)
        return seen


    @property
    def context_text(self) -> str:
        """The retrieved chunks joined into the only context generation may use."""
        return "\n".join(chunk.text for chunk in self.chunks)




class Retriever:
    """Queries one of the two collections and normalises ChromaDB's response.


    ``query_count`` is a plain call counter used as before/after evidence in the
    Task 16 cache demonstration.
    """


    def __init__(
        self,
        settings: Settings = SETTINGS,
        *,
        client: Any | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self._settings = settings
        self._client = client
        self._embedder = embedder
        self._collections: dict[str, Any] = {}
        self.query_count = 0


    # ------------------------------------------------------------------ #


    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = get_chroma_client(self._settings)
        return self._client


    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = build_embedder(self._settings)
        return self._embedder


    def collection(self, name: str) -> Any:
        """Cache collection handles so repeated queries do not re-open them."""
        if name not in self._collections:
            self._collections[name] = get_collection(self.client, name)
        return self._collections[name]


    def reset_counters(self) -> None:
        self.query_count = 0


    def refresh_collections(self) -> int:
        """Drop cached collection handles. Returns how many were dropped.


        ``rag/indexer.py:build_indexes`` with ``reset=True`` *deletes* each
        collection before recreating it. A long-lived ``Retriever`` would keep
        handles to the deleted objects and keep querying them, so a rebuild has
        to be followed by this call (or a restart).
        """
        count = len(self._collections)
        self._collections.clear()
        return count


    # ------------------------------------------------------------------ #


    def search(
        self,
        query: str,
        *,
        collection_name: str | None = None,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """Retrieve the top-k most similar chunks for ``query``.


        Raises:
            RetrievalError: for an empty query, an unknown collection name, or
                an empty collection (which means the index was never built).
        """
        text = (query or "").strip()
        if not text:
            raise RetrievalError("query must not be empty.")


        name = collection_name or self._settings.recommended_collection_name
        if name not in self._settings.all_collection_names:
            raise RetrievalError(
                f"unknown collection {name!r}; expected one of "
                f"{list(self._settings.all_collection_names)}."
            )
        k = top_k if top_k is not None else self._settings.top_k
        if k <= 0:
            raise RetrievalError(f"top_k must be positive, got {k}.")


        collection = self.collection(name)
        try:
            stored = collection.count()
        except Exception as exc:  # noqa: BLE001 - surfaced with a fix instruction
            raise RetrievalError(
                f"could not read collection {name!r}: {type(exc).__name__}: {exc}"
            ) from exc
        if stored == 0:
            raise RetrievalError(
                f"collection {name!r} is empty. Build the indexes first: "
                "`python -m scripts.build_indexes`."
            )


        embedding = self.embedder.encode([text])[0]
        self.query_count += 1
        try:
            raw = collection.query(
                query_embeddings=[embedding],
                n_results=min(k, stored),
                include=["documents", "metadatas", "distances"],
            )
        except VectorIndexError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced with context
            raise RetrievalError(
                f"ChromaDB query against {name!r} failed: {type(exc).__name__}: {exc}"
            ) from exc


        return RetrievalResult(
            query=text,
            collection_name=name,
            top_k=k,
            chunks=tuple(self._parse(raw)),
        )


    # ------------------------------------------------------------------ #


    @staticmethod
    def _parse(raw: dict[str, Any]) -> list[RetrievedChunk]:
        """Flatten ChromaDB's list-of-lists response for a single query."""
        ids = (raw.get("ids") or [[]])[0]
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]


        chunks: list[RetrievedChunk] = []
        for position, chunk_id in enumerate(ids):
            metadata = metadatas[position] if position < len(metadatas) else {}
            metadata = metadata or {}
            distance = float(distances[position]) if position < len(distances) else 1.0
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    document_id=str(metadata.get("document_id", "")),
                    topic_title=str(metadata.get("topic_title", "")),
                    source_filename=str(metadata.get("source_filename", "")),
                    strategy=str(metadata.get("strategy", "")),
                    chunk_index=int(metadata.get("chunk_index", position)),
                    text=str(documents[position]) if position < len(documents) else "",
                    # Cosine space: similarity = 1 - distance.
                    similarity=1.0 - distance,
                    distance=distance,
                )
            )
        # ChromaDB already sorts ascending by distance; sorting again makes the
        # ordering contract explicit and independent of that implementation detail.
        chunks.sort(key=lambda chunk: chunk.distance)
        return chunks




"""Offline doubles for the embedding model and the vector store.


The real SentenceTransformers model needs downloaded artefacts and the real
ChromaDB client needs a persistent directory. Neither is appropriate in a test
run that must work with no network and no model files, so both are replaced
here with deterministic in-memory equivalents that honour the same contracts:


* ``FakeEmbedder`` produces stable unit-length vectors from lexical features, so
  a query that shares words with a chunk scores higher than one that does not.
* ``FakeChromaClient`` implements the slice of the ChromaDB API this project
  uses - ``get_or_create_collection``, ``delete_collection``, ``upsert``,
  ``query``, ``count`` - with the same cosine-distance response shape.
"""


from __future__ import annotations


from typing import Any, Sequence


from rag.embeddings import DeterministicHashEmbedder, cosine_similarity


class FakeEmbedder(DeterministicHashEmbedder):
    """Deterministic hashing embedder with a call counter for assertions."""


    def __init__(self, dimension: int = 256) -> None:
        super().__init__(dimension=dimension)
        self.name = f"fake_hash:{dimension}"
        self.encode_calls = 0


    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        self.encode_calls += 1
        return super().encode(texts)


class FakeCollection:
    """In-memory stand-in for a ChromaDB collection in cosine space."""


    def __init__(self, name: str, metadata: dict[str, Any] | None = None) -> None:
        self.name = name
        self.metadata = dict(metadata or {})
        self._ids: list[str] = []
        self._documents: dict[str, str] = {}
        self._metadatas: dict[str, dict[str, Any]] = {}
        self._embeddings: dict[str, list[float]] = {}
        self.upsert_calls = 0
        self.query_calls = 0


    def upsert(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[dict[str, Any]],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if not (len(ids) == len(documents) == len(metadatas) == len(embeddings)):
            raise ValueError("upsert received ragged inputs")
        self.upsert_calls += 1
        for index, chunk_id in enumerate(ids):
            if chunk_id not in self._documents:
                self._ids.append(chunk_id)
            self._documents[chunk_id] = documents[index]
            self._metadatas[chunk_id] = dict(metadatas[index])
            self._embeddings[chunk_id] = [float(value) for value in embeddings[index]]


    def count(self) -> int:
        return len(self._ids)


    def query(
        self,
        *,
        query_embeddings: Sequence[Sequence[float]],
        n_results: int,
        include: Sequence[str] | None = None,
    ) -> dict[str, list[list[Any]]]:
        self.query_calls += 1
        embedding = list(query_embeddings[0])
        scored = sorted(
            (
                (1.0 - cosine_similarity(embedding, self._embeddings[chunk_id]), chunk_id)
                for chunk_id in self._ids
            ),
            key=lambda pair: (pair[0], pair[1]),
        )[: max(0, n_results)]


        return {
            "ids": [[chunk_id for _, chunk_id in scored]],
            "documents": [[self._documents[chunk_id] for _, chunk_id in scored]],
            "metadatas": [[self._metadatas[chunk_id] for _, chunk_id in scored]],
            "distances": [[distance for distance, _ in scored]],
        }


class FakeChromaClient:
    """In-memory stand-in for ``chromadb.PersistentClient``."""


    def __init__(self) -> None:
        self.collections: dict[str, FakeCollection] = {}


    def get_or_create_collection(
        self,
        *,
        name: str,
        metadata: dict[str, Any] | None = None,
        embedding_function: Any = None,
        **_ignored: Any,
    ) -> FakeCollection:
        if embedding_function is not None:
            raise AssertionError(
                "collections must be created with embedding_function=None so ChromaDB "
                "never embeds anything itself"
            )
        if name not in self.collections:
            self.collections[name] = FakeCollection(name, metadata)
        return self.collections[name]


    def delete_collection(self, *, name: str) -> None:
        if name not in self.collections:
            raise KeyError(name)
        del self.collections[name]



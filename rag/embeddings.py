"""Task 3 (part 2) - the embedding layer.


The graded path is ``sentence_transformers``: a free, local SentenceTransformers
model, no API key, no paid account. Two alternate backends exist, both
**explicitly opt-in** via ``EMBEDDING_BACKEND`` and neither ever selected
automatically:


* ``chroma_onnx`` - ChromaDB's own bundled ONNX build of ``all-MiniLM-L6-v2``.
  Identical model weights, ``onnxruntime`` instead of ``torch``. This exists for
  machines where torch cannot be installed at all.
* ``deterministic_hash`` - a hashing pseudo-embedder for the offline test suite.
  Retrieval quality is *not* meaningful; it only needs to be stable and
  self-consistent so the plumbing can be tested without model artefacts.


There is no silent fallback. When the configured backend cannot load, an
``EmbeddingModelUnavailableError`` is raised with the exact command to fix it.
"""


from __future__ import annotations


import hashlib
import logging
import math
from typing import Final, Protocol, Sequence, runtime_checkable


from app.config import (
    BACKEND_CHROMA_ONNX,
    BACKEND_DETERMINISTIC_HASH,
    BACKEND_SENTENCE_TRANSFORMERS,
    REPO_ROOT,
    SETTINGS,
    Settings,
)


LOGGER: Final = logging.getLogger(__name__)




class EmbeddingModelUnavailableError(RuntimeError):
    """Raised when the configured embedding backend cannot be loaded."""




@runtime_checkable
class Embedder(Protocol):
    """Minimal embedding interface. Tests substitute their own implementation."""


    #: Stable identifier written into the index state file.
    name: str


    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one unit-length embedding per input text."""
        ...




# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #




def l2_normalise(vector: Sequence[float]) -> list[float]:
    """Scale ``vector`` to unit length so a dot product equals cosine similarity.


    A zero vector is returned unchanged rather than raising: it can only come
    from empty input, and ChromaDB handles it without error.
    """
    magnitude = math.sqrt(sum(component * component for component in vector))
    if magnitude == 0.0:
        return list(vector)
    return [component / magnitude for component in vector]




def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity, clamped to [-1, 1] to absorb float drift."""
    if len(left) != len(right):
        raise ValueError(f"dimension mismatch: {len(left)} vs {len(right)}.")
    dot = sum(a * b for a, b in zip(left, right))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    if norm == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / norm))




def _validate_texts(texts: Sequence[str]) -> list[str]:
    if isinstance(texts, str):
        raise TypeError("encode() takes a sequence of strings, not a single string.")
    listed = list(texts)
    if any(not isinstance(item, str) for item in listed):
        raise TypeError("every item passed to encode() must be a string.")
    return listed




# --------------------------------------------------------------------------- #
# Backend: SentenceTransformers (default, graded path)
# --------------------------------------------------------------------------- #




class SentenceTransformerEmbedder:
    """Free local SentenceTransformers embedder.


    The model is loaded lazily on first ``encode()`` so that importing this
    module - or hitting ``GET /health`` - never touches the model files.
    """


    def __init__(self, settings: Settings = SETTINGS) -> None:
        self._settings = settings
        self._model: object | None = None
        self._source = self._resolve_source(settings)
        self.name = f"sentence_transformers:{self._source}"


    @staticmethod
    def _resolve_source(settings: Settings) -> str:
        """Decide what to hand SentenceTransformers: a local path or a model id."""
        if not settings.embedding_model_path:
            return settings.embedding_model_name
        candidate = (REPO_ROOT / settings.embedding_model_path).expanduser().resolve()
        if not candidate.is_dir():
            raise EmbeddingModelUnavailableError(
                f"EMBEDDING_MODEL_PATH={settings.embedding_model_path!r} resolved to "
                f"{candidate}, which is not a directory. Point it at a pre-downloaded "
                "SentenceTransformers model folder, or clear it to use EMBEDDING_MODEL_NAME."
            )
        return str(candidate)


    def _load(self) -> object:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingModelUnavailableError(
                "sentence-transformers is not installed. Install the declared "
                "baseline with `pip install -r requirements.txt`, or switch to the "
                "documented torch-free backend by setting "
                "EMBEDDING_BACKEND=chroma_onnx in your .env."
            ) from exc


        try:
            self._model = SentenceTransformer(self._source)
        except Exception as exc:  # noqa: BLE001 - re-raised with guidance below
            offline = self._settings.offline_mode and not self._settings.allow_model_download
            hint = (
                "OFFLINE_MODE=true and ALLOW_MODEL_DOWNLOAD=false, so no download was "
                "attempted. Either pre-download the model once on a machine that "
                "permits it and set EMBEDDING_MODEL_PATH to that folder, or set "
                "ALLOW_MODEL_DOWNLOAD=true for a single online run."
                if offline
                else "The model could not be loaded or downloaded."
            )
            raise EmbeddingModelUnavailableError(
                f"could not load SentenceTransformers model {self._source!r}. {hint} "
                f"Underlying error: {type(exc).__name__}: {exc}"
            ) from exc
        LOGGER.info("loaded embedding model %s", self._source)
        return self._model


    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        listed = _validate_texts(texts)
        if not listed:
            return []
        model = self._load()
        vectors = model.encode(  # type: ignore[attr-defined]
            listed,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [l2_normalise([float(value) for value in row]) for row in vectors]




# --------------------------------------------------------------------------- #
# Backend: ChromaDB bundled ONNX MiniLM (opt-in, torch-free)
# --------------------------------------------------------------------------- #




class ChromaOnnxEmbedder:
    """``all-MiniLM-L6-v2`` via ChromaDB's own bundled ONNX runtime build.


    Same model as the default backend, without the torch dependency. ChromaDB
    fetches the ONNX artefact once and caches it under its own home directory.
    """


    def __init__(self, settings: Settings = SETTINGS) -> None:
        self._settings = settings
        self._function: object | None = None
        self.name = "chroma_onnx:all-MiniLM-L6-v2"


    def _load(self) -> object:
        if self._function is not None:
            return self._function
        try:
            from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        except ImportError as exc:
            raise EmbeddingModelUnavailableError(
                "chromadb's ONNX embedding function is unavailable. Install the "
                "torch-free baseline with `pip install -r requirements-no-torch.txt`."
            ) from exc
        try:
            self._function = ONNXMiniLM_L6_V2()
        except Exception as exc:  # noqa: BLE001 - re-raised with guidance below
            raise EmbeddingModelUnavailableError(
                "could not initialise ChromaDB's bundled ONNX MiniLM model. Its "
                "artefact is downloaded once and then cached; a strictly offline "
                "machine needs that cache to be present already. Underlying error: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        LOGGER.info("loaded ChromaDB bundled ONNX MiniLM embedder")
        return self._function


    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        listed = _validate_texts(texts)
        if not listed:
            return []
        function = self._load()
        vectors = function(listed)  # type: ignore[operator]
        return [l2_normalise([float(value) for value in row]) for row in vectors]




# --------------------------------------------------------------------------- #
# Backend: deterministic hashing (test-suite only, opt-in)
# --------------------------------------------------------------------------- #




class DeterministicHashEmbedder:
    """Stable hashing pseudo-embedder with no model artefacts.


    Word-level bag of hashed features, so lexical overlap between a query and a
    chunk still produces a higher cosine similarity than no overlap at all -
    enough to exercise indexing, retrieval, thresholds and caching offline.
    ``hashlib`` is used rather than ``hash()`` because the built-in is salted
    per process and would not be reproducible.


    Not suitable for any graded similarity number.
    """


    def __init__(self, dimension: int = 256) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}.")
        self.dimension = dimension
        self.name = f"deterministic_hash:{dimension}"


    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % self.dimension
        # Sign from a separate byte keeps unrelated tokens from all adding up.
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        return index, sign


    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        listed = _validate_texts(texts)
        vectors: list[list[float]] = []
        for text in listed:
            accumulator = [0.0] * self.dimension
            tokens = [token for token in text.lower().split() if token]
            for token in tokens:
                stripped = token.strip(".,;:!?()[]\"'")
                if not stripped:
                    continue
                index, sign = self._bucket(stripped)
                accumulator[index] += sign
            vectors.append(l2_normalise(accumulator))
        return vectors




# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #




def build_embedder(settings: Settings = SETTINGS) -> Embedder:
    """Construct the embedder named by ``EMBEDDING_BACKEND``.


    Raises:
        EmbeddingModelUnavailableError: for an unrecognised backend name.
    """
    backend = settings.embedding_backend
    if backend == BACKEND_SENTENCE_TRANSFORMERS:
        return SentenceTransformerEmbedder(settings)
    if backend == BACKEND_CHROMA_ONNX:
        LOGGER.warning(
            "EMBEDDING_BACKEND=%s selected: ChromaDB's bundled ONNX MiniLM build. "
            "Same model as the default, torch-free. This is a documented opt-in "
            "alternative, not the graded default.",
            backend,
        )
        return ChromaOnnxEmbedder(settings)
    if backend == BACKEND_DETERMINISTIC_HASH:
        LOGGER.warning(
            "EMBEDDING_BACKEND=%s selected: hashing pseudo-embeddings. Retrieval "
            "quality is NOT meaningful and no similarity number produced in this "
            "mode may be reported as a graded result.",
            backend,
        )
        return DeterministicHashEmbedder()
    raise EmbeddingModelUnavailableError(f"unknown EMBEDDING_BACKEND {backend!r}.")




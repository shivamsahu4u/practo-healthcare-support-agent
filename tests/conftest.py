"""Shared fixtures. Everything here is offline and deterministic.


The vector store and the embedder are replaced with the doubles in
``tests/fakes.py``, and the knowledge-base version counter is redirected to a
mutable in-test value so cache invalidation can be exercised without touching
``data/generated/``.


The similarity threshold is **measured** with the same calibration code the
production path uses, rather than hard-coded. With the hashing double the
absolute similarities differ from the real model's, so a fixed number would be
meaningless - and measuring it here keeps the tests honest about the fact that
the threshold is derived, not chosen.
"""


from __future__ import annotations


import dataclasses
import json
from pathlib import Path
from typing import Iterator


import pytest


from app.config import CREW_MODE_DIRECT, SETTINGS, Settings
from app.services.session_store import SessionStore
from app.services.support_service import SupportService
from rag.calibration import CalibrationOverlapError, measure
from rag.chunking import KbDocument, load_knowledge_base
from rag.grounded_generation import GroundedGenerator, ThresholdSource
from rag.indexer import IndexState, build_indexes
from rag.retriever import Retriever
from tests.fakes import FakeChromaClient, FakeEmbedder




@pytest.fixture(scope="session")
def documents() -> list[KbDocument]:
    """The real knowledge base, parsed from disk."""
    return load_knowledge_base()




@pytest.fixture()
def test_settings() -> Settings:
    """Settings for the offline suite.


    ``CREW_MODE=direct`` and ``REVIEW_ENABLED=False`` keep the base fixtures
    importable without crewai or autogen installed. Tests that exercise those
    integrations build their own settings and use ``pytest.importorskip``.
    """
    return dataclasses.replace(
        SETTINGS,
        crew_mode=CREW_MODE_DIRECT,
        review_enabled=False,
        cache_enabled=True,
        similarity_threshold_override=None,
    )




@pytest.fixture()
def kb_version(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, int]]:
    """Redirect the knowledge-base version counter to an in-test value."""
    holder = {"value": 1}
    monkeypatch.setattr(
        "rag.grounded_generation.current_kb_version", lambda *_a, **_k: holder["value"]
    )
    yield holder




@pytest.fixture()
def index_state_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect index-state reads and writes to a temporary file.


    ``read_index_state`` / ``write_index_state`` bind their default path at
    definition time, so the functions themselves are replaced rather than the
    module constant.
    """
    target = tmp_path / "index_state.json"


    def fake_read(path: Path | None = None) -> IndexState:
        if not target.is_file():
            return IndexState()
        payload = json.loads(target.read_text(encoding="utf-8"))
        return IndexState(
            kb_version=int(payload.get("kb_version", 0)),
            built_at=str(payload.get("built_at", "")),
            embedder=str(payload.get("embedder", "")),
            document_ids=list(payload.get("document_ids", [])),
            chunk_counts=dict(payload.get("chunk_counts", {})),
            collection_names=dict(payload.get("collection_names", {})),
        )


    def fake_write(state: IndexState, path: Path | None = None) -> None:
        target.write_text(json.dumps(state.as_dict(), indent=2), encoding="utf-8")


    monkeypatch.setattr("rag.indexer.read_index_state", fake_read)
    monkeypatch.setattr("rag.indexer.write_index_state", fake_write)
    return target




@pytest.fixture()
def embedder() -> FakeEmbedder:
    return FakeEmbedder()




@pytest.fixture()
def chroma_client() -> FakeChromaClient:
    return FakeChromaClient()




@pytest.fixture()
def built_index(
    test_settings: Settings,
    chroma_client: FakeChromaClient,
    embedder: FakeEmbedder,
    documents: list[KbDocument],
    index_state_file: Path,
) -> FakeChromaClient:
    """Both collections, populated from the real knowledge base."""
    build_indexes(
        test_settings,
        client=chroma_client,
        embedder=embedder,
        documents=documents,
        reset=True,
    )
    return chroma_client




@pytest.fixture()
def retriever(
    test_settings: Settings, built_index: FakeChromaClient, embedder: FakeEmbedder
) -> Retriever:
    return Retriever(test_settings, client=built_index, embedder=embedder)




@pytest.fixture()
def measured_threshold(test_settings: Settings, retriever: Retriever) -> ThresholdSource:
    """Measure the threshold with the production calibration algorithm.


    Skips - rather than guessing - when the hashing double cannot separate the
    two clusters, because a fabricated threshold would make every downstream
    assertion meaningless.
    """
    result = measure(retriever, settings=test_settings)
    try:
        value = result.recommended_threshold
    except CalibrationOverlapError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"fake embedder did not separate the calibration clusters: {exc}")
    return ThresholdSource(
        value=value,
        origin="test_measured",
        detail=(
            f"measured in-test on {result.collection_name}: "
            f"min(in)={result.min_in_scope:.4f}, out(max)={result.max_out_of_scope:.4f}"
        ),
    )




@pytest.fixture()
def generator(
    test_settings: Settings,
    retriever: Retriever,
    measured_threshold: ThresholdSource,
    kb_version: dict[str, int],
) -> GroundedGenerator:
    return GroundedGenerator(
        retriever=retriever,
        settings=test_settings,
        threshold_source=measured_threshold,
    )




@pytest.fixture()
def service(test_settings: Settings, generator: GroundedGenerator) -> SupportService:
    return SupportService(
        test_settings, generator=generator, sessions=SessionStore()
    )




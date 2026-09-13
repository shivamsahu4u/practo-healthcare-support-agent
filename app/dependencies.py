"""Dependency wiring for the FastAPI app.


The ``SupportService`` is built lazily and cached. Lazily matters: constructing
it must not load the embedding model, because ``GET /health`` has to answer
without touching model artefacts and without any network activity. The embedder
is loaded on the first retrieval instead.
"""


from __future__ import annotations


import logging
from functools import lru_cache
from typing import Final


from app.config import SETTINGS, Settings
from app.services.support_service import SupportService


LOGGER: Final = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_support_service() -> SupportService:
    """The process-wide ``SupportService`` singleton, built on first use."""
    LOGGER.info(
        "constructing SupportService (mock_llm=%s, crew_mode=%s, backend=%s)",
        SETTINGS.mock_llm,
        SETTINGS.crew_mode,
        SETTINGS.embedding_backend,
    )
    return SupportService(SETTINGS)


def get_settings() -> Settings:
    """FastAPI dependency for the resolved settings."""
    return SETTINGS


def reset_support_service() -> None:
    """Drop the cached singleton. Used by the test suite between cases."""
    get_support_service.cache_clear()



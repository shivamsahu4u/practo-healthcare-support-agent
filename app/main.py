"""Task 11 - the FastAPI deployment.


Endpoints:


* ``GET  /health``       - liveness plus the operating mode. Loads no model and
  makes no network call, so it is safe as a container probe.
* ``POST /ask``          - the full support pipeline behind Pydantic models.
* ``POST /add-document`` - add one knowledge-base document, chunk it with both
  strategies, upsert it into both collections, bump the knowledge-base version
  and invalidate the response cache.
* ``GET  /governance``   - the four-layer governance model, machine-readable.
* ``WS   /ws/chat/{session_id}`` - real-time multi-turn chat over the same
  pipeline and the same guardrails, which survives a client disconnecting
  mid-conversation.


The WebSocket handler catches ``WebSocketDisconnect`` and returns cleanly, so one
client vanishing never takes the server - or any other client's socket - with
it. A malformed frame gets a structured error frame back and the socket stays
open.
"""


from __future__ import annotations


import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Final


from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.responses import RedirectResponse
from pydantic import ValidationError


from app import __version__
from app.config import SETTINGS, Settings, ensure_runtime_directories
from app.dependencies import get_settings, get_support_service
from app.logging_config import configure_logging, log_request
from app.models import (
    AddDocumentRequest,
    AddDocumentResponse,
    AskRequest,
    BudgetRejection,
    ErrorResponse,
    GovernanceResponse,
    HealthResponse,
    SupportResponse,
    WsClientMessage,
    WsServerAck,
    WsServerError,
)
from app.services.support_service import SupportService
from agents.crew import CrewExecutionError
from agents.governance import (
    BudgetExceededError,
    ToolPermissionError,
    governance_snapshot,
)
from agents.review_team import ReviewUnavailableError
from agents.tools import ToolDispatchError
from rag.chunking import KnowledgeBaseError
from rag.embeddings import EmbeddingModelUnavailableError
from rag.grounded_generation import CalibrationRequiredError
from rag.indexer import (
    DocumentRejectedError,
    IndexState,
    VectorIndexError,
    add_document,
    read_index_state,
)
from rag.retriever import RetrievalError


LOGGER: Final = logging.getLogger(__name__)


TRACK: Final[str] = "Practo (Healthcare)"


#: Typed failures that mean "a dependency of the pipeline is unavailable", as
#: distinct from "the request was bad".
#:
#: Only ``CalibrationRequiredError`` / ``RetrievalError`` / ``VectorIndexError``
#: used to be handled. Everything else here - a missing embedding-model artefact,
#: an absent Autogen or CrewAI install, a crew failure, an unclassifiable tool -
#: subclasses plain ``RuntimeError`` and propagated as an **unhandled HTTP 500**
#: with no structured body. That contradicted the Runtime governance layer's
#: claim that "typed exceptions [are] surfaced as structured errors; internals
#: never returned to the client", and it was the single most likely failure on a
#: machine without the model artefacts.
PIPELINE_UNAVAILABLE_ERRORS: Final[tuple[type[Exception], ...]] = (
    EmbeddingModelUnavailableError,
    CrewExecutionError,
    ReviewUnavailableError,
    ToolDispatchError,
    ToolPermissionError,
    KnowledgeBaseError,
)


#: Everything a transport should report as "service unavailable". Defined once so
#: the HTTP and WebSocket paths cannot drift apart in what they recognise - the
#: WebSocket handler previously caught a narrower set, so an embedding failure
#: fell through to the catch-all and *closed the socket*, contradicting the
#: documented promise that a failed turn leaves the connection open.
SERVICE_UNAVAILABLE_ERRORS: Final[tuple[type[Exception], ...]] = (
    CalibrationRequiredError,
    RetrievalError,
    VectorIndexError,
) + PIPELINE_UNAVAILABLE_ERRORS


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Start-up and shut-down.


    Deliberately cheap: configure logging, make sure the writable directories
    exist, and stop. No index build, no model load, no outbound call.
    """
    configure_logging(SETTINGS)
    ensure_runtime_directories(SETTINGS)
    LOGGER.info(
        "Practo support agent starting (track=%s, mock_llm=%s, offline=%s, backend=%s)",
        TRACK,
        SETTINGS.mock_llm,
        SETTINGS.offline_mode,
        SETTINGS.embedding_backend,
    )
    yield
    LOGGER.info("Practo support agent shutting down")


app = FastAPI(
    title="Practo Domain Support Agent",
    description=(
        "Final Capstone - Practo (Healthcare) track. RAG core, CrewAI crew, Autogen "
        "review stage and governance, all running under MOCK_LLM with zero API keys."
    ),
    version=__version__,
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# GET /
# --------------------------------------------------------------------------- #


@app.get("/", include_in_schema=False, tags=["ops"])
async def root() -> RedirectResponse:
    """Send browser visitors to the interactive Swagger UI."""
    return RedirectResponse(url="/docs")


# GET /health
# --------------------------------------------------------------------------- #


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health(
    settings: Settings = Depends(get_settings),
    service: SupportService = Depends(get_support_service),
) -> HealthResponse:
    """Liveness and operating mode. Never loads the embedding model."""
    try:
        state = read_index_state()
        index_error: str | None = None
    except VectorIndexError as exc:
        # A liveness probe must not fail because an on-disk state file is
        # corrupt. `read_index_state` raises in that case, and an unhandled 500
        # here would make an orchestrator restart-loop the container forever,
        # because the corrupt file survives every restart. Report an empty index
        # plus the reason instead, and let the probe stay green.
        LOGGER.warning("index state unreadable; reporting an empty index: %s", exc)
        state = IndexState()
        index_error = str(exc)


    return HealthResponse(
        status="ok",
        version=__version__,
        track=TRACK,
        mode=settings.describe(),
        index={
            "kb_version": state.kb_version,
            "built_at": state.built_at,
            "embedder": state.embedder,
            "documents": len(state.document_ids),
            "chunk_counts": state.chunk_counts,
            "collections": state.collection_names,
            "error": index_error,
        },
        cache=service.generator.cache.snapshot(),
    )


# --------------------------------------------------------------------------- #
# GET /governance
# --------------------------------------------------------------------------- #


@app.get("/governance", response_model=GovernanceResponse, tags=["governance"])
async def governance(settings: Settings = Depends(get_settings)) -> GovernanceResponse:
    """The four-layer governance model and the risk classification."""
    return GovernanceResponse.model_validate(governance_snapshot(settings))


# --------------------------------------------------------------------------- #
# POST /ask
# --------------------------------------------------------------------------- #


@app.post(
    "/ask",
    response_model=SupportResponse,
    tags=["support"],
    responses={
        413: {"model": BudgetRejection, "description": "Per-request budget cap exceeded"},
    },
)
async def ask(
    request: AskRequest,
    service: SupportService = Depends(get_support_service),
) -> SupportResponse:
    """Answer one patient question through the full pipeline."""
    try:
        return await service.answer(
            request.query,
            session_id=request.session_id,
            collection_name=request.collection_name,
            top_k=request.top_k,
            endpoint="POST /ask",
            transport="http",
        )
    except BudgetExceededError as exc:
        decision = exc.decision
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=BudgetRejection(
                trace_id="rejected-before-trace",
                message=decision.message,
                limit_name=decision.limit_name or "unknown",
                limit_value=decision.limit_value or 0,
                observed_value=decision.observed_value or 0,
            ).model_dump(),
        ) from exc
    except CalibrationRequiredError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ErrorResponse(
                error="calibration_required",
                trace_id=uuid.uuid4().hex,
                message=str(exc),
            ).model_dump(),
        ) from exc
    except (RetrievalError, VectorIndexError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ErrorResponse(
                error="index_unavailable",
                trace_id=uuid.uuid4().hex,
                message=str(exc),
            ).model_dump(),
        ) from exc
    except PIPELINE_UNAVAILABLE_ERRORS as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ErrorResponse(
                error="pipeline_unavailable",
                trace_id=uuid.uuid4().hex,
                message=str(exc),
            ).model_dump(),
        ) from exc


# --------------------------------------------------------------------------- #
# POST /add-document
# --------------------------------------------------------------------------- #


@app.post("/add-document", response_model=AddDocumentResponse, tags=["knowledge-base"])
async def add_knowledge_document(
    request: AddDocumentRequest,
    service: SupportService = Depends(get_support_service),
    settings: Settings = Depends(get_settings),
) -> AddDocumentResponse:
    """Index one new knowledge-base document into BOTH collections.


    ``topic_slug`` is validated against a plain-identifier pattern, so the
    destination path is derived entirely from validated input - the caller
    cannot reach an arbitrary filesystem location.
    """
    trace_id = uuid.uuid4().hex
    started = time.perf_counter()


    try:
        result = add_document(
            request.topic_slug, request.title, request.content, settings
        )
    except DocumentRejectedError as exc:
        log_request(
            trace_id=trace_id,
            endpoint="POST /add-document",
            transport="http",
            session_id="n/a",
            query=request.topic_slug,
            response_type="blocked",
            grounded=False,
            cache_status="n/a",
            duration_ms=(time.perf_counter() - started) * 1000.0,
            status_code=400,
            error_category="document_rejected",
            settings=settings,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorResponse(
                error="document_rejected", trace_id=trace_id, message=str(exc)
            ).model_dump(),
        ) from exc
    except VectorIndexError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ErrorResponse(
                error="index_unavailable", trace_id=trace_id, message=str(exc)
            ).model_dump(),
        ) from exc


    # A new document can change any answer, so every cached answer goes. The
    # kb_version bump in the cache key would already do this; the explicit
    # invalidation makes the intent obvious and gives the transcript a number.
    invalidated = service.generator.invalidate_cache()


    log_request(
        trace_id=trace_id,
        endpoint="POST /add-document",
        transport="http",
        session_id="n/a",
        query=f"added document {result.document_id}",
        response_type="policy",
        grounded=True,
        cache_status="invalidated",
        duration_ms=(time.perf_counter() - started) * 1000.0,
        status_code=200,
        guardrails_fired=[],
        extra={
            "document_id": result.document_id,
            "total_chunks": result.total_chunks,
            "kb_version": result.kb_version,
            "cache_entries_invalidated": invalidated,
        },
        settings=settings,
    )


    return AddDocumentResponse(
        trace_id=trace_id,
        document_id=result.document_id,
        title=result.title,
        source_filename=result.source_filename,
        indexed=True,
        chunk_ids=result.chunk_ids,
        total_chunks=result.total_chunks,
        kb_version=result.kb_version,
        cache_entries_invalidated=invalidated,
    )


# --------------------------------------------------------------------------- #
# WS /ws/chat/{session_id}
# --------------------------------------------------------------------------- #


@app.websocket("/ws/chat/{session_id}")
async def chat_socket(
    websocket: WebSocket,
    session_id: str,
    service: SupportService = Depends(get_support_service),
) -> None:
    """Real-time multi-turn chat over the same pipeline and guardrails.


    Accepted frames::


        {"query": "How long before my appointment can I cancel?"}
        {"reset": true}


    Every reply is a serialised ``SupportResponse``, a ``WsServerAck`` or a
    ``WsServerError``. A malformed frame is answered with an error frame and the
    socket stays open; a disconnect is caught and the handler returns without
    disturbing the server or any other client.
    """
    await websocket.accept()
    LOGGER.info("websocket connected for session %s", session_id)


    try:
        while True:
            raw = await websocket.receive_text()


            try:
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError("frame must be a JSON object")
                frame = WsClientMessage.model_validate(payload)
            except (json.JSONDecodeError, ValueError, ValidationError) as exc:
                await websocket.send_text(
                    WsServerError(
                        error="malformed_frame",
                        message=(
                            'Send a JSON object such as {"query": "..."} or '
                            f'{{"reset": true}}. Rejected: {exc}'
                        ),
                    ).model_dump_json()
                )
                continue


            if frame.reset:
                cleared = service.sessions.reset(session_id)
                await websocket.send_text(
                    WsServerAck(
                        action="reset",
                        session_id=session_id,
                        detail=(
                            "conversation history cleared"
                            if cleared
                            else "no history to clear"
                        ),
                    ).model_dump_json()
                )
                continue


            if not frame.query or not frame.query.strip():
                await websocket.send_text(
                    WsServerError(
                        error="empty_query",
                        message='Provide a non-empty "query" field.',
                    ).model_dump_json()
                )
                continue


            try:
                response = await service.answer(
                    frame.query,
                    session_id=session_id,
                    endpoint="WS /ws/chat",
                    transport="websocket",
                )
                await websocket.send_text(response.model_dump_json())
            except BudgetExceededError as exc:
                await websocket.send_text(
                    WsServerError(
                        error="budget_exceeded",
                        message=exc.decision.message,
                    ).model_dump_json()
                )
            except SERVICE_UNAVAILABLE_ERRORS as exc:
                # Widened deliberately: this used to catch only three of the
                # typed failures, so a missing embedding model or an absent
                # Autogen install reached the outer catch-all and closed the
                # socket. Every recognised unavailability now returns an error
                # frame and leaves the conversation open.
                await websocket.send_text(
                    WsServerError(
                        error="service_unavailable",
                        message=str(exc),
                    ).model_dump_json()
                )


    except WebSocketDisconnect:
        # The expected way a conversation ends. Nothing to clean up beyond the
        # log line: session history is keyed by session_id and deliberately
        # outlives the socket so a reconnect resumes the conversation.
        LOGGER.info("websocket disconnected for session %s", session_id)
    except Exception:  # noqa: BLE001 - logged, then the socket is closed cleanly
        LOGGER.exception("unexpected websocket failure for session %s", session_id)
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except RuntimeError:
            # Already closed by the peer; nothing further to do.
            pass


def describe_routes() -> list[dict[str, Any]]:
    """Every registered route. Used by the acceptance-check script."""
    described: list[dict[str, Any]] = []
    for route in app.routes:
        described.append(
            {
                "path": getattr(route, "path", ""),
                "name": getattr(route, "name", ""),
                "methods": sorted(getattr(route, "methods", []) or []),
                "type": type(route).__name__,
            }
        )
    return described



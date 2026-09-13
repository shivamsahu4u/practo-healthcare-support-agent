"""The support pipeline - where every layer is composed into one turn.


Order of operations for ``POST /ask`` and for one WebSocket frame:


1. **Input guardrails** - mask contact numbers, then run injection detection on
   the masked text. An injection blocks the turn here: no crew, no cache, no
   memory write.
2. **Runtime budget cap** - checked on the masked text *before* the crew is
   constructed, so an oversized request costs one length check.
3. **Session memory** - the masked query is handed to
   ``RunnableWithMessageHistory``, which supplies prior turns and stores this
   one. Only masked text ever reaches the history.
4. **Deterministic routing** - policy, appointment, or both.
5. **CrewAI crew** - ``crew.kickoff()`` produces the draft.
6. **Retrieval gate** - if retrieval never cleared the calibrated similarity
   threshold there is no admissible context, so the turn refuses and the review
   stage is skipped.
7. **Autogen review** - the independent two-agent team approves or revises.
8. **Final groundedness check** - the reviewed answer is re-checked
   sentence-by-sentence against the support text. Defence in depth: if anything
   unsupported survived the reviewer, the turn refuses rather than ships it.
9. **Structured validation** - the result is validated as a ``SupportResponse``
   before it can leave the process.
10. **Structured logging** - one JSON-Lines entry with the trace id.
"""


from __future__ import annotations


import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Final


from langchain_core.messages import AIMessage


from app.config import (
    FALLBACK_ANSWER,
    SETTINGS,
    Settings,
)
from app.logging_config import log_request
from app.models import (
    RESPONSE_FORMAT,
    AppointmentResult,
    ReviewVerdict,
    SupportResponse,
)
from app.services.session_store import SessionStore
from agents.composition import (
    EXEMPT_ANSWER_SENTENCES,
    build_support_text,
    compose_appointment_section,
    compose_sources,
    determine_response_type,
)
from agents.context import CrewRunContext
from agents.crew import CrewDraft, run_crew
from agents.governance import BudgetExceededError, enforce_budget
from agents.guardrails import (
    GROUNDEDNESS_REFUSAL_MESSAGE,
    INJECTION_SAFETY_MESSAGE,
    GuardrailReport,
    apply_input_guardrails,
    check_groundedness,
)
from agents.memory import (
    RESPONSE_PAYLOAD_KEY,
    build_memory_runnable,
    resolve_record_id_from_history,
)
from agents.review_team import ReviewSession, ReviewUnavailableError, review_draft
from agents.routing import classify_route, extract_record_id
from rag.grounded_generation import GroundedGenerator


LOGGER: Final = logging.getLogger(__name__)


GUARDRAIL_GROUNDEDNESS: Final[str] = "output_groundedness_refusal"


@dataclass(frozen=True, slots=True)
class TurnParams:
    """Per-turn inputs that are not part of the memory payload."""


    trace_id: str
    session_id: str
    guard: GuardrailReport
    collection_name: str | None = None
    top_k: int | None = None
    inject_unsupported_claim: bool = False


class SupportService:
    """Stateful, process-wide support pipeline.


    ``crew_calls`` is a plain counter used as the Task 15 evidence that a
    budget-rejected request never reaches the crew.
    """


    def __init__(
        self,
        settings: Settings = SETTINGS,
        *,
        generator: GroundedGenerator | None = None,
        sessions: SessionStore | None = None,
    ) -> None:
        self.settings = settings
        self.generator = generator or GroundedGenerator(settings=settings)
        self.sessions = sessions or SessionStore()
        self.crew_calls = 0
        self.review_calls = 0
        self.blocked_calls = 0
        self.budget_rejections = 0
        self.pipeline_failures = 0


    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #


    def reset_counters(self) -> None:
        """Zero every counter, including the generator's. Used by the demos."""
        self.crew_calls = 0
        self.review_calls = 0
        self.blocked_calls = 0
        self.budget_rejections = 0
        self.pipeline_failures = 0
        self.generator.reset_counters()


    def refresh_runtime_state(self) -> dict[str, int]:
        """Drop every cached view of on-disk state. Returns what was cleared.


        The service is a process-wide singleton
        (``app/dependencies.py:get_support_service``), so three things were
        memoised for the life of the process with no way to reload them:


        * the calibrated threshold, resolved once and kept forever - re-running
          ``scripts.calibrate_threshold`` had no effect on a live server;
        * ChromaDB collection handles - ``build_indexes(reset=True)`` *deletes*
          those collections, leaving the server holding stale handles;
        * cached answers, which are keyed on the threshold and index version.


        Call this after rebuilding an index or re-calibrating.
        """
        return {
            "cache_entries_invalidated": self.generator.invalidate_cache(),
            "collection_handles_dropped": self.generator.retriever.refresh_collections(),
            "threshold_reloaded": int(self.generator.refresh_threshold()),
        }


    def stats(self) -> dict[str, Any]:
        return {
            "crew_calls": self.crew_calls,
            "review_calls": self.review_calls,
            "blocked_calls": self.blocked_calls,
            "budget_rejections": self.budget_rejections,
            "pipeline_failures": self.pipeline_failures,
            "generation": self.generator.stats.as_dict(),
            "retrieval_queries": self.generator.retriever.query_count,
            "cache": self.generator.cache.snapshot(),
            "sessions": self.sessions.snapshot(),
        }


    async def answer(
        self,
        query: str,
        *,
        session_id: str = "default",
        collection_name: str | None = None,
        top_k: int | None = None,
        endpoint: str = "POST /ask",
        transport: str = "http",
        inject_unsupported_claim: bool = False,
    ) -> SupportResponse:
        """Handle one support turn end to end.


        Raises:
            BudgetExceededError: when the request exceeds a runtime cap. The
                rejection is logged before it propagates, and the crew is never
                constructed.
        """
        trace_id = uuid.uuid4().hex
        started = time.perf_counter()


        guard = apply_input_guardrails(query)


        # --- 1. injection blocks the turn outright ---------------------- #
        if guard.blocked:
            self.blocked_calls += 1
            response = SupportResponse(
                trace_id=trace_id,
                session_id=session_id,
                query=guard.masked_text,
                answer=INJECTION_SAFETY_MESSAGE,
                response_type="blocked",
                sources=[],
                grounded=False,
                safety_message=(
                    "blocked by the prompt-injection guardrail: "
                    + ", ".join(
                        finding.pattern_name for finding in guard.injection_findings
                    )
                ),
                pii_masked=guard.pii_masked,
                guardrails_fired=list(guard.fired),
                mock_llm=self.settings.mock_llm,
                crew_mode=self.settings.crew_mode,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
            self._log(
                response,
                endpoint=endpoint,
                transport=transport,
                status_code=200,
                cache_status="bypass",
                error_category="prompt_injection_blocked",
            )
            return response


        # --- 2. runtime budget cap, before any crew work ---------------- #
        try:
            enforce_budget(guard.masked_text, self.settings)
        except BudgetExceededError as exc:
            self.budget_rejections += 1
            log_request(
                trace_id=trace_id,
                endpoint=endpoint,
                transport=transport,
                session_id=session_id,
                query=guard.masked_text[:200],
                response_type="blocked",
                grounded=False,
                cache_status="bypass",
                duration_ms=(time.perf_counter() - started) * 1000.0,
                status_code=413,
                error_category="budget_exceeded",
                guardrails_fired=list(guard.fired),
                extra=exc.decision.as_dict() | {"crew_invoked": False},
                settings=self.settings,
            )
            raise


        # --- 3-9. memory-wrapped pipeline ------------------------------- #
        turn = TurnParams(
            trace_id=trace_id,
            session_id=session_id,
            guard=guard,
            collection_name=collection_name,
            top_k=top_k,
            inject_unsupported_claim=inject_unsupported_claim,
        )


        async def resolve(payload: dict[str, Any]) -> AIMessage:
            return await self._resolve_turn(payload, turn)


        try:
            runnable = build_memory_runnable(resolve, self.sessions.memory)
            message = await runnable.ainvoke(
                {"input": guard.masked_text},
                config={"configurable": {"session_id": session_id}},
            )


            # Task 9: validate against the declared response_format in code, on
            # the payload that round-tripped through session memory.
            response = RESPONSE_FORMAT.model_validate(
                message.additional_kwargs[RESPONSE_PAYLOAD_KEY]
            )
            response = response.model_copy(
                update={"latency_ms": (time.perf_counter() - started) * 1000.0}
            )
        except Exception as exc:  # noqa: BLE001 - logged, then re-raised unchanged
            # Task 12 promises exactly one JSON-Lines entry per request. Every
            # log call used to sit *after* the pipeline, so any failure below
            # this point - a missing embedding model, an absent Autogen install,
            # a crew error - escaped before anything was written and the request
            # vanished from the audit trail. The most likely production failure
            # was also the one guaranteed to leave no trace.
            #
            # The exception is re-raised untouched: the API layer owns the status
            # code. This branch only guarantees the audit line.
            self.pipeline_failures += 1
            log_request(
                trace_id=trace_id,
                endpoint=endpoint,
                transport=transport,
                session_id=session_id,
                query=guard.masked_text[:200],
                response_type="error",
                grounded=False,
                cache_status="bypass",
                duration_ms=(time.perf_counter() - started) * 1000.0,
                status_code=503,
                error_category=type(exc).__name__,
                guardrails_fired=list(guard.fired),
                extra={"error": str(exc)[:500], "crew_invoked": self.crew_calls > 0},
                settings=self.settings,
            )
            raise


        # --- 10. one structured log line -------------------------------- #
        self._log(
            response,
            endpoint=endpoint,
            transport=transport,
            status_code=200,
            cache_status="hit" if response.cache_hit else "miss",
            error_category=None,
        )
        return response


    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #


    async def _resolve_turn(self, payload: dict[str, Any], turn: TurnParams) -> AIMessage:
        """The memory-wrapped body of one turn.


        Receives ``{"input": <masked query>, "history": [BaseMessage, ...]}``
        from ``RunnableWithMessageHistory`` and returns an ``AIMessage`` carrying
        the validated ``SupportResponse`` payload alongside the answer text.
        """
        masked_query = str(payload.get("input", ""))
        history = list(payload.get("history") or [])


        record_id_in_text = extract_record_id(masked_query)
        record_id_in_memory = (
            None if record_id_in_text else resolve_record_id_from_history(history)
        )
        record_id = record_id_in_text or record_id_in_memory


        context = CrewRunContext(
            trace_id=turn.trace_id,
            session_id=turn.session_id,
            query=masked_query,
            route=classify_route(masked_query, record_id_from_memory=record_id_in_memory),
            record_id=record_id,
            collection_name=turn.collection_name,
            top_k=turn.top_k,
            record_id_from_memory=record_id_in_memory is not None,
            inject_unsupported_claim=turn.inject_unsupported_claim,
        )


        draft = run_crew(context, self.generator, self.settings)
        self.crew_calls += 1


        response = await self._finalise(context, draft, turn)
        return AIMessage(
            content=response.answer,
            additional_kwargs={RESPONSE_PAYLOAD_KEY: response.model_dump()},
        )


    async def _finalise(
        self, context: CrewRunContext, draft: CrewDraft, turn: TurnParams
    ) -> SupportResponse:
        """Apply the output guardrails and the review stage, then validate."""
        guardrails_fired = list(turn.guard.fired)
        support_text = build_support_text(context)
        retrieval_grounded = bool(context.grounded and context.grounded.grounded)


        answer = draft.draft
        response_type = determine_response_type(context)
        sources = compose_sources(context)
        verdict: ReviewVerdict | None = None
        review_revised = False


        # --- 6. retrieval gate ----------------------------------------- #
        if context.needs_policy and not retrieval_grounded:
            gate = check_groundedness(
                answer, support_text, retrieval_grounded=False, settings=self.settings
            )
            LOGGER.info("output groundedness refusal: %s", gate.reason)
            guardrails_fired.append(GUARDRAIL_GROUNDEDNESS)
            # The policy half is discarded. On a combined turn the appointment
            # half is still admissible - it comes from structured record facts,
            # not from retrieval - so it is kept and prefixed with the refusal.
            appointment_section = (
                compose_appointment_section(context) if context.lookup else ""
            )
            if appointment_section:
                answer = f"{FALLBACK_ANSWER} {appointment_section}"
                response_type = "appointment"
                sources = compose_sources(context)
            else:
                answer = FALLBACK_ANSWER
                response_type = "fallback"
                sources = []
        else:
            # --- 7. independent Autogen review ------------------------- #
            if self.settings.review_enabled:
                session = ReviewSession(
                    query=context.query,
                    draft=answer,
                    context_text=context.context_text,
                    support_text=support_text,
                    source_ids=list(context.source_ids),
                    lookup=context.lookup,
                    retrieval_grounded=retrieval_grounded or not context.needs_policy,
                    minimum_overlap=self.settings.groundedness_overlap_min,
                    exempt=EXEMPT_ANSWER_SENTENCES,
                )
                try:
                    outcome = await review_draft(session, self.settings)
                except ReviewUnavailableError as exc:
                    # Never ship an unreviewed answer as though it were reviewed.
                    LOGGER.error("review stage unavailable: %s", exc)
                    raise
                self.review_calls += 1
                verdict = outcome.verdict
                review_revised = outcome.revised
                answer = verdict.final_answer


            # --- 8. final groundedness check --------------------------- #
            final_check = check_groundedness(
                answer,
                support_text,
                retrieval_grounded=retrieval_grounded or not context.needs_policy,
                # Required, not cosmetic: the reviewer may have replaced the
                # draft with the fixed refusal, and that refusal is no longer in
                # the support corpus. Without the exemption the final check
                # would flag the system's own refusal as ungrounded.
                exempt=EXEMPT_ANSWER_SENTENCES,
                settings=self.settings,
            )
            if not final_check.grounded:
                LOGGER.warning(
                    "answer refused after review: %s (unsupported: %s)",
                    final_check.reason,
                    final_check.unsupported,
                )
                guardrails_fired.append(GUARDRAIL_GROUNDEDNESS)
                answer = GROUNDEDNESS_REFUSAL_MESSAGE
                response_type = "fallback"
                sources = []


        appointment = (
            AppointmentResult.model_validate(context.lookup) if context.lookup else None
        )


        # --- 9. structured validation ---------------------------------- #
        return SupportResponse(
            trace_id=turn.trace_id,
            session_id=turn.session_id,
            query=context.query,
            answer=answer,
            response_type=response_type,
            sources=sources,
            appointment=appointment,
            escalation_recommended=bool(
                context.lookup and context.lookup.get("escalation_recommended")
            ),
            grounded=retrieval_grounded,
            safety_message=None,
            collection_name=(
                context.grounded.collection_name if context.grounded else None
            ),
            top_similarity=context.grounded.top_similarity if context.grounded else 0.0,
            threshold=context.grounded.threshold if context.grounded else 0.0,
            pii_masked=turn.guard.pii_masked,
            guardrails_fired=guardrails_fired,
            cache_hit=bool(context.grounded and context.grounded.cache_hit),
            mock_llm=self.settings.mock_llm,
            crew_mode=draft.crew_mode,
            tools_invoked=context.tools_invoked,
            review_approved=verdict.approved if verdict else None,
            review_reason=verdict.reason if verdict else None,
            review_revised=review_revised,
        )


    def _log(
        self,
        response: SupportResponse,
        *,
        endpoint: str,
        transport: str,
        status_code: int,
        cache_status: str,
        error_category: str | None,
    ) -> None:
        log_request(
            trace_id=response.trace_id,
            endpoint=endpoint,
            transport=transport,
            session_id=response.session_id,
            query=response.query,
            response_type=response.response_type,
            grounded=response.grounded,
            cache_status=cache_status,
            duration_ms=response.latency_ms,
            status_code=status_code,
            error_category=error_category,
            guardrails_fired=response.guardrails_fired,
            tools_invoked=response.tools_invoked,
            sources=response.sources,
            review_approved=response.review_approved,
            extra={
                "top_similarity": round(response.top_similarity, 4),
                "threshold": round(response.threshold, 4),
                "escalation_recommended": response.escalation_recommended,
                "review_revised": response.review_revised,
            },
            settings=self.settings,
        )



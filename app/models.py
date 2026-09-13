"""Task 9 - the Pydantic contracts every layer validates against.


``SupportResponse`` is the structured-output schema the brief requires: the crew
never returns a bare dictionary to the API or to the Autogen reviewer, and
``app/services/support_service.py`` validates against this model in code before
anything leaves the process.


``ReviewVerdict`` lives here too, because it is both the Autogen Final-Editor's
``output_content_type`` and part of the HTTP response contract - one definition,
two consumers.
"""


from __future__ import annotations


from typing import Any, Final, Literal


from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


#: The closed set of outcomes a support turn can have.
ResponseType = Literal["policy", "appointment", "combined", "fallback", "blocked"]


#: Hard ceiling on an inbound free-text field, enforced by Pydantic *before* any
#: application code runs.
#:
#: This is a **denial-of-service** bound, deliberately separate from the
#: **cost** bound in ``agents/governance.py`` (``MAX_REQUEST_CHARACTERS``, default
#: 2000). Without it, ``query`` was unbounded: a multi-megabyte body was parsed,
#: materialised, then scanned by the masking regex and nine injection patterns
#: *before* the budget cap was ever consulted, so the cap could not protect the
#: work it was supposed to gate.
#:
#: Kept comfortably above ``MAX_REQUEST_CHARACTERS`` on purpose: requests between
#: the two limits must still reach the service and be rejected with a structured
#: ``413 BudgetRejection`` (the Task 15 demonstration), not a bare 422. Only
#: absurd payloads are refused this early.
MAX_INBOUND_TEXT_CHARACTERS: Final[int] = 8000




# --------------------------------------------------------------------------- #
# Domain payloads
# --------------------------------------------------------------------------- #




class EscalationComponents(BaseModel):
    """Transparent breakdown of the designed escalation score."""


    model_config = ConfigDict(extra="forbid")


    follow_up_required: bool
    follow_up_component: float = Field(ge=0.0, le=1.0)
    days_since_created: int = Field(ge=0)
    aging_normalised: float = Field(ge=0.0, le=1.0)
    aging_component: float = Field(ge=0.0, le=1.0)
    weight_follow_up: float = Field(gt=0.0, le=1.0)
    weight_aging: float = Field(gt=0.0, le=1.0)
    aging_cap_days: int = Field(gt=0)




class AppointmentResult(BaseModel):
    """What ``check_appointment_status`` returns, as a validated model."""


    model_config = ConfigDict(extra="forbid")


    record_id: str
    found: bool
    status: str | None = None
    consultation_fee_inr: int | None = None
    category: str | None = None
    days_since_created: int | None = None
    follow_up_required: bool | None = None
    escalation_score: float | None = Field(default=None, ge=0.0, le=1.0)
    escalation_recommended: bool = False
    escalation_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    components: EscalationComponents | None = None
    message: str | None = None




class ReviewVerdict(BaseModel):
    """Task 14 / Part 4 - the propose-review pair's structured verdict.


    The brief names these three fields exactly: ``approved``, ``final_answer``
    and ``reason``. This is also the ``output_content_type`` declared on the
    Autogen Final-Editor agent, so the same three names are what the model is
    obliged to produce and what ``model_dump()`` emits.
    """


    model_config = ConfigDict(extra="forbid")


    approved: bool
    final_answer: str
    reason: str


    @property
    def reasoning(self) -> str:
        """Backward-compatible read access for older internal callers."""
        return self.reason




# --------------------------------------------------------------------------- #
# The crew's structured output contract
# --------------------------------------------------------------------------- #




class SupportResponse(BaseModel):
    """The single response shape every crew turn must conform to."""


    model_config = ConfigDict(extra="forbid")


    trace_id: str
    session_id: str
    query: str = Field(description="The PII-masked query. Never the raw user text.")
    answer: str
    response_type: ResponseType
    sources: list[str] = Field(default_factory=list)
    appointment: AppointmentResult | None = None
    escalation_recommended: bool = False
    grounded: bool = False
    safety_message: str | None = None


    # Retrieval provenance
    collection_name: str | None = None
    top_similarity: float = 0.0
    threshold: float = 0.0


    # Guardrails / observability
    pii_masked: bool = False
    guardrails_fired: list[str] = Field(default_factory=list)
    cache_hit: bool = False
    latency_ms: float = 0.0
    mock_llm: bool = True
    crew_mode: str = "crewai"
    tools_invoked: list[str] = Field(default_factory=list)


    # Autogen review stage
    review_approved: bool | None = None
    review_reason: str | None = None
    review_revised: bool = False


    @field_validator("sources", "guardrails_fired", "tools_invoked")
    @classmethod
    def _no_blank_entries(cls, value: list[str]) -> list[str]:
        return [item for item in value if item and item.strip()]




#: The declared ``response_format`` for the crew, in the sense Task 9 uses the
#: term: the single Pydantic model every crew response must conform to.
#:
#: It is referenced through this alias wherever a response is validated, so the
#: contract has exactly one name in the codebase. Two places enforce it:
#: ``SupportService._finalise()`` constructs it (so an invalid response cannot be
#: built at all), and ``SupportService.answer()`` re-validates the round-tripped
#: payload that came back through session memory with
#: ``RESPONSE_FORMAT.model_validate(...)``. ``POST /ask`` then declares it as its
#: FastAPI ``response_model``, which puts the same schema in the OpenAPI document.
RESPONSE_FORMAT: Final[type[SupportResponse]] = SupportResponse




# --------------------------------------------------------------------------- #
# HTTP request / response models
# --------------------------------------------------------------------------- #




class AskRequest(BaseModel):
    """``POST /ask`` request body."""


    model_config = ConfigDict(extra="forbid")


    query: str = Field(
        min_length=1,
        max_length=MAX_INBOUND_TEXT_CHARACTERS,
        description="The patient's question.",
    )
    session_id: str = Field(
        default="default",
        min_length=1,
        max_length=128,
        description="Conversation key for session memory. Reuse it to keep context.",
    )
    collection_name: str | None = Field(
        default=None,
        description="Override the retrieval collection. Defaults to the recommended one.",
    )
    top_k: int | None = Field(default=None, ge=1, le=20)


    @field_validator("query", "session_id")
    @classmethod
    def _not_blank(cls, value: str, info: ValidationInfo) -> str:
        """Reject whitespace-only text.


        ``min_length`` alone is not enough: ``" "`` has length 1, so it used to
        reach ``SessionMemory.get_history``, which strips before checking and
        raises ``ValueError`` - surfacing as an unhandled HTTP 500. Rejecting it
        here turns that into the 422 it always should have been.
        """
        if not value.strip():
            raise ValueError(f"{info.field_name} must not be blank")
        return value




class AddDocumentRequest(BaseModel):
    """``POST /add-document`` request body.


    ``topic_slug`` is a plain identifier, never a path: the indexer's
    ``SLUG_PATTERN`` rejects dots and separators outright.
    """


    model_config = ConfigDict(extra="forbid")


    topic_slug: str = Field(
        min_length=3,
        max_length=48,
        pattern=r"^[a-z][a-z0-9_]{2,47}$",
        description="Lowercase identifier used as the document id and filename stem.",
    )
    title: str = Field(min_length=1, max_length=120)
    content: str = Field(min_length=20, max_length=4000)




class AddDocumentResponse(BaseModel):
    """``POST /add-document`` response body."""


    model_config = ConfigDict(extra="forbid")


    trace_id: str
    document_id: str
    title: str
    source_filename: str
    indexed: bool
    chunk_ids: dict[str, list[str]]
    total_chunks: int
    kb_version: int
    cache_entries_invalidated: int




class HealthResponse(BaseModel):
    """``GET /health`` response body. Never triggers a model load."""


    model_config = ConfigDict(extra="forbid")


    status: Literal["ok"]
    version: str
    track: str
    mode: dict[str, Any]
    index: dict[str, Any]
    cache: dict[str, Any]


class GovernanceResponse(BaseModel):
    """``GET /governance`` response body - the four-layer model, machine-readable."""


    model_config = ConfigDict(extra="forbid")


    risk_level: Literal["Low", "Medium", "High"]
    risk_justification: str
    layers: dict[str, Any]
    tool_permissions: dict[str, list[str]]
    budget: dict[str, int]


class BudgetRejection(BaseModel):
    """Structured 413 body when the runtime budget cap rejects a request."""


    model_config = ConfigDict(extra="forbid")


    error: Literal["budget_exceeded"] = "budget_exceeded"
    trace_id: str
    message: str
    limit_name: str
    limit_value: int
    observed_value: int
    crew_invoked: Literal[False] = False




class ErrorResponse(BaseModel):
    """Generic structured error body."""


    model_config = ConfigDict(extra="forbid")


    error: str
    trace_id: str
    message: str




# --------------------------------------------------------------------------- #
# WebSocket frames
# --------------------------------------------------------------------------- #




class WsClientMessage(BaseModel):
    """One inbound WebSocket frame.


    ``query`` carries the same ``MAX_INBOUND_TEXT_CHARACTERS`` ceiling as
    ``AskRequest``: a WebSocket frame is read fully into memory by
    ``receive_text()`` before any handler sees it, so the bound has to exist on
    this model too. A frame between this ceiling and the budget cap still gets
    the structured ``budget_exceeded`` error frame.
    """


    model_config = ConfigDict(extra="forbid")


    query: str | None = Field(default=None, max_length=MAX_INBOUND_TEXT_CHARACTERS)
    reset: bool = False




class WsServerError(BaseModel):
    """One outbound WebSocket error frame. The socket stays open afterwards."""


    model_config = ConfigDict(extra="forbid")


    type: Literal["error"] = "error"
    error: str
    message: str
    trace_id: str | None = None




class WsServerAck(BaseModel):
    """Outbound acknowledgement for a control frame such as ``reset``."""


    model_config = ConfigDict(extra="forbid")


    type: Literal["ack"] = "ack"
    action: str
    session_id: str
    detail: str




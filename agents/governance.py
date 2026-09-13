"""Task 15 - the four-layer AI-governance model, implemented rather than described.


Application layer
    Least autonomy via an explicit tool registry. ``check_appointment_status``
    is authorised for the Lookup Agent and for nobody else. ``build_agents()``
    in ``agents/crew.py`` runs every proposed assignment through
    ``assert_tool_assignment()`` before an ``Agent`` is constructed, so wiring
    the tool to the Retrieval Agent or the Composer raises
    ``ToolPermissionError`` instead of quietly working.


Model / orchestration layer
    ``MOCK_LLM`` by default, deterministic rule-based routing, Pydantic
    structured outputs at every boundary, and bounded turns everywhere - CrewAI
    ``max_iter``, the mock LLM's own ``MAX_TOOL_ATTEMPTS``, and Autogen's
    ``max_turns=2``. There is no unbounded autonomous loop anywhere.


Runtime layer
    A per-request budget cap enforced *before* the crew runs, plus structured
    rejection, plus safe error handling that never leaks internals to the client.


Infrastructure / data layer
    Local ChromaDB, local synthetic data, masked logs, no committed secrets,
    telemetry disabled, offline-first embedding configuration.


**Risk classification: High.** Justified in ``RISK_JUSTIFICATION`` below.
"""


from __future__ import annotations


import math
from dataclasses import dataclass
from typing import Any, Final


from app.config import (
    AGENT_COMPOSER,
    AGENT_LOOKUP,
    AGENT_RETRIEVAL,
    SETTINGS,
    TOOL_APPOINTMENT_LOOKUP,
    TOOL_POLICY_LOOKUP,
    Settings,
)


# --------------------------------------------------------------------------- #
# Application layer - least autonomy
# --------------------------------------------------------------------------- #


#: The complete tool-permission registry. This is the single authority on which
#: agent may hold which tool; nothing else in the codebase decides it.
TOOL_PERMISSIONS: Final[dict[str, frozenset[str]]] = {
    AGENT_RETRIEVAL: frozenset({TOOL_POLICY_LOOKUP}),
    AGENT_LOOKUP: frozenset({TOOL_APPOINTMENT_LOOKUP}),
    # The Composer holds no tools. It only combines what the other two produced.
    AGENT_COMPOSER: frozenset(),
}


#: Tools that must never be reachable by more than one agent.
EXCLUSIVE_TOOLS: Final[frozenset[str]] = frozenset({TOOL_APPOINTMENT_LOOKUP})


class ToolPermissionError(PermissionError):
    """Raised when an agent is offered a tool it is not authorised to hold."""


def authorised_tools(agent_key: str) -> frozenset[str]:
    """Tool names ``agent_key`` may hold.


    Raises:
        ToolPermissionError: for an agent that is not in the registry at all.
    """
    if agent_key not in TOOL_PERMISSIONS:
        raise ToolPermissionError(
            f"agent {agent_key!r} is not in the tool-permission registry. Add it to "
            "TOOL_PERMISSIONS with an explicit tool set before wiring it into the crew."
        )
    return TOOL_PERMISSIONS[agent_key]


def assert_tool_assignment(agent_key: str, tool_names: list[str]) -> None:
    """Gate one agent's proposed tool set.


    Raises:
        ToolPermissionError: when any proposed tool is unauthorised for this agent.
    """
    allowed = authorised_tools(agent_key)
    unauthorised = sorted(set(tool_names) - allowed)
    if unauthorised:
        raise ToolPermissionError(
            f"agent {agent_key!r} may hold {sorted(allowed) or 'no tools'} but was "
            f"offered {unauthorised}. Least autonomy is enforced at construction "
            "time: only the Lookup Agent may call "
            f"{TOOL_APPOINTMENT_LOOKUP!r}, because that tool reads appointment "
            "records and every additional caller widens the blast radius for no "
            "functional gain."
        )


def agent_holding(tool_name: str) -> str | None:
    """Which single agent is authorised for ``tool_name``, if exactly one is."""
    holders = [
        agent for agent, tools in TOOL_PERMISSIONS.items() if tool_name in tools
    ]
    return holders[0] if len(holders) == 1 else None


def verify_registry_invariants() -> None:
    """Check the registry itself is sane. Called by the acceptance script.


    Raises:
        ToolPermissionError: when an exclusive tool has more than one holder.
    """
    for tool_name in EXCLUSIVE_TOOLS:
        holders = [
            agent for agent, tools in TOOL_PERMISSIONS.items() if tool_name in tools
        ]
        if len(holders) != 1:
            raise ToolPermissionError(
                f"{tool_name!r} must be authorised for exactly one agent, found "
                f"{holders}."
            )


# --------------------------------------------------------------------------- #
# Runtime layer - per-request budget cap
# --------------------------------------------------------------------------- #


#: Characters per token. A deterministic heuristic, not a real tokeniser: under
#: MOCK_LLM there is no model tokeniser to consult, and pulling one in just for
#: an estimate would add a dependency for no benefit. Four characters per token
#: is the usual rule of thumb for English and is stated as an estimate wherever
#: it is reported.
CHARACTERS_PER_TOKEN: Final[int] = 4


@dataclass(frozen=True, slots=True)
class BudgetDecision:
    """The outcome of a budget check."""


    allowed: bool
    characters: int
    estimated_tokens: int
    limit_name: str | None = None
    limit_value: int | None = None
    observed_value: int | None = None
    message: str = ""


    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "characters": self.characters,
            "estimated_tokens": self.estimated_tokens,
            "limit_name": self.limit_name,
            "limit_value": self.limit_value,
            "observed_value": self.observed_value,
            "message": self.message,
        }


class BudgetExceededError(RuntimeError):
    """Raised when a request exceeds the per-request token/cost budget."""


    def __init__(self, decision: BudgetDecision) -> None:
        self.decision = decision
        super().__init__(decision.message)


def estimate_tokens(text: str) -> int:
    """Deterministic token estimate: ``ceil(len(text) / CHARACTERS_PER_TOKEN)``."""
    return math.ceil(len(text or "") / CHARACTERS_PER_TOKEN)


def evaluate_budget(text: str, settings: Settings = SETTINGS) -> BudgetDecision:
    """Check ``text`` against both caps without raising."""
    characters = len(text or "")
    tokens = estimate_tokens(text)


    if characters > settings.max_request_characters:
        return BudgetDecision(
            allowed=False,
            characters=characters,
            estimated_tokens=tokens,
            limit_name="MAX_REQUEST_CHARACTERS",
            limit_value=settings.max_request_characters,
            observed_value=characters,
            message=(
                f"Request rejected before any agent ran: {characters} characters "
                f"exceeds the per-request cap of {settings.max_request_characters}. "
                "Please shorten the question."
            ),
        )


    if tokens > settings.max_estimated_tokens:
        return BudgetDecision(
            allowed=False,
            characters=characters,
            estimated_tokens=tokens,
            limit_name="MAX_ESTIMATED_TOKENS",
            limit_value=settings.max_estimated_tokens,
            observed_value=tokens,
            message=(
                f"Request rejected before any agent ran: an estimated {tokens} tokens "
                f"exceeds the per-request cap of {settings.max_estimated_tokens}. "
                "Please shorten the question."
            ),
        )


    return BudgetDecision(
        allowed=True,
        characters=characters,
        estimated_tokens=tokens,
        message="within the per-request budget",
    )


def enforce_budget(text: str, settings: Settings = SETTINGS) -> BudgetDecision:
    """Check ``text`` and raise when it exceeds a cap.


    Called *before* the crew is constructed, so an oversized request costs one
    length check rather than a crew run.


    Raises:
        BudgetExceededError: when either cap is exceeded.
    """
    decision = evaluate_budget(text, settings)
    if not decision.allowed:
        raise BudgetExceededError(decision)
    return decision


def budget_snapshot(settings: Settings = SETTINGS) -> dict[str, int]:
    """The active caps, for ``GET /governance`` and reports."""
    return {
        "max_request_characters": settings.max_request_characters,
        "max_estimated_tokens": settings.max_estimated_tokens,
        "characters_per_token_estimate": CHARACTERS_PER_TOKEN,
    }


# --------------------------------------------------------------------------- #
# Risk classification
# --------------------------------------------------------------------------- #


RISK_LEVEL: Final[str] = "High"


RISK_JUSTIFICATION: Final[str] = (
    "This system is classified High risk under the Low/Medium/High scheme. The "
    "scheme places medical data in the High band, and that is the band this system "
    "belongs in: it answers questions about clinical appointments, consultation "
    "fees, prescription refills, lab reports, insurance claims and patient-data "
    "privacy, and it is addressed to patients rather than to trained staff. It is "
    "not merely a support-ticket assistant, because a confidently wrong answer "
    "about an emergency route, a telemedicine eligibility rule or a refill policy "
    "can change what a patient does about their health. Two facts reduce the "
    "concrete exposure without changing the classification: every appointment "
    "record in this repository is synthetic and deterministically generated, and "
    "the agent has no write path to any clinical system - it reads policy text and "
    "one read-only appointment record. Mitigations are the guardrails, the "
    "retrieval-only answer construction, the independent Autogen review stage, the "
    "tool-permission registry and the budget cap. Residual risk remains: the "
    "groundedness check is lexical rather than semantic, the injection denylist is "
    "not exhaustive, and free-text PII such as a patient name, a condition or an "
    "insurance ID is not masked because it has no fixed format to match. The system "
    "does not diagnose, does not triage and does not replace clinical judgement, "
    "and the emergency policy document routes any suspected emergency to the "
    "national emergency number rather than answering it."
)


GOVERNANCE_LAYERS: Final[dict[str, dict[str, Any]]] = {
    "application": {
        "principle": "least autonomy",
        "controls": [
            "explicit tool-permission registry (TOOL_PERMISSIONS)",
            "assert_tool_assignment() gates every Agent construction",
            f"{TOOL_APPOINTMENT_LOOKUP} is authorised for {AGENT_LOOKUP} only",
            f"{AGENT_COMPOSER} holds no tools at all",
            "input guardrails: contact-number masking, prompt-injection denylist",
            "output guardrail: groundedness refusal",
        ],
        "enforced_in": ["agents/governance.py", "agents/crew.py", "agents/guardrails.py"],
    },
    "scope": {
        "principle": f"risk classified {RISK_LEVEL}; bounded, deterministic scope",
        "controls": [
            f"risk classification: {RISK_LEVEL} on the Low/Medium/High scheme, "
            "justified in RISK_JUSTIFICATION",
            "answer scope is retrieval-only: policy claims may come from the "
            "knowledge base and nothing else, with an explicit refusal below the "
            "calibrated similarity threshold",
            "rule-based routing (agents/routing.py), never model-decided, so the "
            "crew cannot widen its own scope",
            "Pydantic structured output on every boundary (app/models.py)",
            "bounded turns: CrewAI max_iter, MockCrewLLM MAX_TOOL_ATTEMPTS, "
            "Autogen max_turns=2; no unbounded autonomous loop and no "
            "agent-to-agent delegation (allow_delegation=False)",
            "data scope: synthetic deterministic dataset, local persistent "
            "ChromaDB, no hosted service, no real patient data anywhere",
            "no diagnosis, no triage: the emergency document routes to the "
            "national emergency number rather than advising",
            "telemetry disabled and HF_HUB_OFFLINE exported before any "
            "third-party import; .env git-ignored and .env.example secret-free",
        ],
        "enforced_in": [
            "agents/governance.py",
            "agents/routing.py",
            "app/config.py",
            "app/models.py",
            "rag/grounded_generation.py",
        ],
    },
    "runtime": {
        "principle": "performance monitoring and cost tracking",
        "controls": [
            "per-request cost cap on characters and estimated tokens, enforced "
            "before the crew is constructed so a rejection never costs a crew run",
            "structured HTTP 413 BudgetRejection body with crew_invoked=false",
            "inbound payload ceiling (MAX_INBOUND_TEXT_CHARACTERS) enforced by "
            "Pydantic before any application code runs - a DoS bound, separate "
            "from the cost cap",
            "performance monitoring: duration_ms on every JSON-Lines entry, "
            "latency_ms on every response, and per-stage counters "
            "(crew_calls, review_calls, generations, retrieval_queries, "
            "cache hits/misses, pipeline_failures) via SupportService.stats()",
            "exactly one structured log entry per request, written on the "
            "failure path as well as the success path",
            "typed exceptions surfaced as structured 503s; internals never "
            "returned to the client",
            "bounded session memory (DEFAULT_MAX_SESSIONS) with LRU eviction",
        ],
        "enforced_in": [
            "agents/governance.py",
            "app/main.py",
            "app/logging_config.py",
            "app/services/support_service.py",
            "agents/memory.py",
        ],
    },
    "cache": {
        "principle": "in-memory cache with normalised keys and explicit invalidation",
        "controls": [
            "in-memory bounded LRU (CACHE_MAX_ENTRIES) with eviction counted",
            "normalised cache keys: query trimmed, lowercased and "
            "whitespace-collapsed, so wording variants share one entry",
            "SHA-256 digest over length-prefixed components (query, collection, "
            "top_k, threshold, kb_version, embedder identity) so two components "
            "cannot collide by concatenation",
            "correctness bound to the retrieval configuration: a re-calibration, "
            "a rebuild or an embedding-backend change all change the key",
            "explicit invalidation on POST /add-document, in addition to the "
            "implicit kb_version bump",
            "deliberately not cached: blocked or prompt-injection requests, "
            "errors, and appointment lookups (mutable state could go stale)",
            "raw PII never reaches the cache, because the query is masked before "
            "generation",
            "thread-safe under an RLock: /ask is served from a worker thread "
            "pool while the WebSocket handler runs on the event loop",
        ],
        "enforced_in": ["rag/cache.py", "rag/grounded_generation.py", "app/main.py"],
    },
}


def governance_snapshot(settings: Settings = SETTINGS) -> dict[str, Any]:
    """Everything ``GET /governance`` reports."""
    return {
        "risk_level": RISK_LEVEL,
        "risk_justification": RISK_JUSTIFICATION,
        "layers": GOVERNANCE_LAYERS,
        "tool_permissions": {
            agent: sorted(tools) for agent, tools in TOOL_PERMISSIONS.items()
        },
        "budget": budget_snapshot(settings),
    }



"""Per-request orchestration context and the tool-invocation ledger.


One ``CrewRunContext`` is created per support turn and handed to the tools, the
mock LLM and the composer. It serves two purposes:


* it is the **tool-invocation ledger** - proof of which tool actually ran, with
  what arguments, used both as the Task 7 evidence and as the mock LLM's
  authoritative "has the tool run yet?" signal;
* it carries the retrieval and lookup results forward to the Composer agent and
  then to the Autogen review stage, so nothing has to be re-derived by parsing
  prompt text.


Using the ledger rather than scraping ``"Observation:"`` out of the prompt is
the direct mitigation for CrewAI pitfall (1) - see ``agents/mock_llm.py``.
"""


from __future__ import annotations


from dataclasses import dataclass, field
from typing import Any


from rag.grounded_generation import GroundedAnswer


#: Deterministic routing outcomes.
ROUTE_POLICY = "policy"
ROUTE_APPOINTMENT = "appointment"
ROUTE_COMBINED = "combined"


@dataclass(frozen=True, slots=True)
class ToolInvocation:
    """One recorded tool call."""


    tool_name: str
    agent_key: str
    arguments: dict[str, Any]
    ok: bool
    summary: str


    def as_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "agent_key": self.agent_key,
            "arguments": dict(self.arguments),
            "ok": self.ok,
            "summary": self.summary,
        }


@dataclass(slots=True)
class CrewRunContext:
    """Everything one support turn needs, and everything it produced."""


    trace_id: str
    session_id: str
    query: str
    """The **masked** query. Raw user text never reaches this object."""


    route: str
    record_id: str | None = None
    collection_name: str | None = None
    top_k: int | None = None


    grounded: GroundedAnswer | None = None
    lookup: dict[str, Any] | None = None
    invocations: list[ToolInvocation] = field(default_factory=list)


    record_id_from_memory: bool = False
    """True when the record id was recovered from session history, not this turn."""


    inject_unsupported_claim: bool = False
    """Test-only fixture flag for the Task 14 revision demonstration.


    When set, the Composer appends one deliberately ungrounded sentence so the
    Autogen reviewer has something real to catch. Defaults to ``False`` and is
    never set by ``POST /ask``, the WebSocket handler, or any normal runtime
    path - only by ``scripts/generate_demonstrations.py`` and the test suite.
    """


    # ------------------------------------------------------------------ #


    def record(
        self,
        *,
        tool_name: str,
        agent_key: str,
        arguments: dict[str, Any],
        ok: bool,
        summary: str,
    ) -> None:
        """Append an entry to the ledger."""
        self.invocations.append(
            ToolInvocation(
                tool_name=tool_name,
                agent_key=agent_key,
                arguments=dict(arguments),
                ok=ok,
                summary=summary,
            )
        )


    def was_invoked(self, tool_name: str) -> bool:
        """Has ``tool_name`` already run in this turn?"""
        return any(entry.tool_name == tool_name for entry in self.invocations)


    @property
    def tools_invoked(self) -> list[str]:
        """Distinct tool names, in first-invocation order."""
        ordered: list[str] = []
        for entry in self.invocations:
            if entry.tool_name not in ordered:
                ordered.append(entry.tool_name)
        return ordered


    @property
    def needs_policy(self) -> bool:
        return self.route in (ROUTE_POLICY, ROUTE_COMBINED)


    @property
    def needs_appointment(self) -> bool:
        return self.route in (ROUTE_APPOINTMENT, ROUTE_COMBINED)


    @property
    def context_text(self) -> str:
        """The retrieved context, i.e. the only policy text an answer may use."""
        return self.grounded.context_text if self.grounded else ""


    @property
    def source_ids(self) -> list[str]:
        return list(self.grounded.sources) if self.grounded else []


    def as_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "query": self.query,
            "route": self.route,
            "record_id": self.record_id,
            "record_id_from_memory": self.record_id_from_memory,
            "collection_name": self.collection_name,
            "grounded": self.grounded.as_dict() if self.grounded else None,
            "lookup": self.lookup,
            "invocations": [entry.as_dict() for entry in self.invocations],
        }



"""Task 7 - the CrewAI crew: Retrieval Agent, Lookup Agent, Response Composer.


Run with ``crew.kickoff()``. Which task legs are included is decided by
``agents/routing.py`` before the crew is built, so a policy question never
constructs an appointment leg and vice versa.


Tool permissions are checked against ``agents/governance.py`` at construction
time. Wiring ``appointment_status_lookup`` to the Retrieval Agent or the
Composer raises ``ToolPermissionError`` before ``Agent(...)`` is called - that
is the Task 15 least-autonomy demonstration, and it is a real guard rather than
a comment.


``CREW_MODE=direct`` runs the same tools and the same composition functions in a
straight line, without CrewAI. It exists as a diagnostic for environments where
CrewAI cannot be installed, and because both modes call
``agents.tools.invoke_*`` and ``agents.composition.compose_draft`` they produce
identical drafts. It is **not** the graded path.
"""


from __future__ import annotations


import logging
from dataclasses import dataclass, field
from typing import Any, Final


from app.config import (
    AGENT_COMPOSER,
    AGENT_LOOKUP,
    AGENT_RETRIEVAL,
    CREW_MODE_CREWAI,
    CREW_MODE_DIRECT,
    SETTINGS,
    TOOL_APPOINTMENT_LOOKUP,
    TOOL_POLICY_LOOKUP,
    Settings,
)
from agents.composition import compose_draft
from agents.context import CrewRunContext
from agents.governance import assert_tool_assignment
from agents.tools import (
    CrewTools,
    build_crew_tools,
    invoke_appointment_lookup,
    invoke_policy_lookup,
)
from rag.grounded_generation import GroundedGenerator


LOGGER: Final = logging.getLogger(__name__)




class CrewExecutionError(RuntimeError):
    """Raised when the crew cannot be constructed or run."""




# --------------------------------------------------------------------------- #
# Agent definitions
# --------------------------------------------------------------------------- #




@dataclass(frozen=True, slots=True)
class AgentSpec:
    """Static description of one crew agent."""


    key: str
    role: str
    goal: str
    backstory: str




AGENT_SPECS: Final[dict[str, AgentSpec]] = {
    AGENT_RETRIEVAL: AgentSpec(
        key=AGENT_RETRIEVAL,
        role="Practo Policy Retrieval Specialist",
        goal=(
            "Find the passages of Practo's clinic-policy knowledge base that answer "
            "the patient's question, and report them without embellishment."
        ),
        backstory=(
            "You are a patient-experience specialist who has read every Practo clinic "
            "policy document. You never answer a policy question from memory, because "
            "policies change and a wrong answer about fees or eligibility costs a "
            "patient real money or real time. You always search the knowledge base "
            "first, and if the knowledge base does not cover a question you say so."
        ),
    ),
    AGENT_LOOKUP: AgentSpec(
        key=AGENT_LOOKUP,
        role="Practo Appointment Records Officer",
        goal=(
            "Retrieve the authoritative status, consultation fee and escalation score "
            "for one named appointment record."
        ),
        backstory=(
            "You are the only member of this crew authorised to read appointment "
            "records, and you read exactly the one record the patient named - never a "
            "list, never a neighbouring record. If no appointment id was given you ask "
            "for one instead of guessing."
        ),
    ),
    AGENT_COMPOSER: AgentSpec(
        key=AGENT_COMPOSER,
        role="Practo Patient Response Composer",
        goal=(
            "Combine the retrieved policy text and the appointment record into one "
            "clear answer for the patient, adding nothing that neither of them says."
        ),
        backstory=(
            "You are a patient-communication editor. You hold no tools of your own: "
            "you cannot look anything up, which is exactly why you cannot invent "
            "anything. You work only from what the Retrieval Agent and the Records "
            "Officer handed you, and you never soften or embroider a policy."
        ),
    ),
}




# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #




@dataclass(slots=True)
class CrewDraft:
    """The crew's draft answer plus the evidence of how it was produced."""


    draft: str
    context: CrewRunContext
    crew_mode: str
    raw_output: str = ""
    agents_used: list[str] = field(default_factory=list)
    llm_calls: dict[str, int] = field(default_factory=dict)


    @property
    def tools_invoked(self) -> list[str]:
        return self.context.tools_invoked


    def as_dict(self) -> dict[str, Any]:
        return {
            "draft": self.draft,
            "crew_mode": self.crew_mode,
            "agents_used": list(self.agents_used),
            "llm_calls": dict(self.llm_calls),
            "tools_invoked": self.tools_invoked,
            "invocations": [entry.as_dict() for entry in self.context.invocations],
        }




# --------------------------------------------------------------------------- #
# CrewAI path
# --------------------------------------------------------------------------- #




def _authorised_tool_map(tools: CrewTools, agent_keys: list[str]) -> dict[str, list[Any]]:
    """Assign tools per agent and gate every assignment through governance.


    Raises:
        ToolPermissionError: when an assignment is unauthorised.
    """
    assignments: dict[str, list[Any]] = {}
    for agent_key in agent_keys:
        proposed = tools.for_agent(agent_key)
        assert_tool_assignment(agent_key, [tool.name for tool in proposed])
        assignments[agent_key] = proposed
    return assignments




def _agent_keys_for_route(context: CrewRunContext) -> list[str]:
    """Which agents this turn needs. The Composer always runs."""
    keys: list[str] = []
    if context.needs_policy:
        keys.append(AGENT_RETRIEVAL)
    if context.needs_appointment:
        keys.append(AGENT_LOOKUP)
    keys.append(AGENT_COMPOSER)
    return keys




def _run_with_crewai(
    context: CrewRunContext, generator: GroundedGenerator, settings: Settings
) -> CrewDraft:
    """Build and kick off the real CrewAI crew."""
    try:
        from crewai import Agent, Crew, Process, Task
    except ImportError as exc:
        raise CrewExecutionError(
            "crewai is not installed, so CREW_MODE=crewai cannot run. Install the "
            "declared baseline with `pip install -r requirements.txt`, or set "
            "CREW_MODE=direct for the diagnostic pipeline."
        ) from exc


    from agents.mock_llm import build_mock_llms


    tools = build_crew_tools(context, generator, settings)
    agent_keys = _agent_keys_for_route(context)
    assignments = _authorised_tool_map(tools, agent_keys)
    mock_llms = build_mock_llms(context, assignments)


    agents: dict[str, Any] = {}
    for agent_key in agent_keys:
        spec = AGENT_SPECS[agent_key]
        agents[agent_key] = Agent(
            role=spec.role,
            goal=spec.goal,
            backstory=spec.backstory,
            tools=assignments[agent_key],
            llm=mock_llms[agent_key],
            allow_delegation=False,
            verbose=False,
            max_iter=settings.crew_max_iter,
            # CrewAI's own tool-result cache would suppress a genuine second
            # tool call and make the invocation ledger under-report.
            cache=False,
        )


    # Values are baked into the descriptions rather than passed as kickoff
    # inputs, so no template interpolation ever runs over patient text (a query
    # containing a brace would otherwise break interpolation).
    tasks: list[Any] = []
    upstream: list[Any] = []


    if context.needs_policy:
        retrieval_task = Task(
            description=(
                "A Practo patient asked:\n"
                f"{context.query}\n\n"
                "Search the Practo clinic-policy knowledge base with your tool and "
                "report what the policy actually says. Do not answer from memory. If "
                "the knowledge base does not cover the question, say that plainly."
            ),
            expected_output=(
                "The policy answer, quoted from the retrieved knowledge-base text, or "
                "an explicit statement that the knowledge base does not cover it."
            ),
            agent=agents[AGENT_RETRIEVAL],
        )
        tasks.append(retrieval_task)
        upstream.append(retrieval_task)


    if context.needs_appointment:
        lookup_task = Task(
            description=(
                "A Practo patient asked:\n"
                f"{context.query}\n\n"
                f"Appointment id in scope: {context.record_id or '(none provided)'}\n\n"
                "If an appointment id is present, look up that one record with your "
                "tool and report its status, consultation fee and escalation score. "
                "If no id is present, ask the patient for one."
            ),
            expected_output=(
                "The appointment's status, consultation fee and escalation score, or a "
                "request for the appointment id."
            ),
            agent=agents[AGENT_LOOKUP],
        )
        tasks.append(lookup_task)
        upstream.append(lookup_task)


    compose_task = Task(
        description=(
            "A Practo patient asked:\n"
            f"{context.query}\n\n"
            "Combine the outputs of the preceding tasks into one answer for the "
            "patient. Add nothing that those outputs do not contain. You have no "
            "tools of your own."
        ),
        expected_output="One clear answer for the patient, with no invented detail.",
        agent=agents[AGENT_COMPOSER],
        context=upstream,
    )
    tasks.append(compose_task)


    crew = Crew(
        agents=[agents[key] for key in agent_keys],
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
        # memory=True would spin up CrewAI's own embedder and vector store,
        # which needs a model artefact and would break the offline guarantee.
        # Conversation memory is handled by agents/memory.py instead.
        memory=False,
        cache=False,
    )


    try:
        result = crew.kickoff()
    except Exception as exc:  # noqa: BLE001 - wrapped with actionable context
        raise CrewExecutionError(
            f"crew.kickoff() failed: {type(exc).__name__}: {exc}"
        ) from exc


    raw_output = str(result).strip()
    deterministic = compose_draft(context)
    if raw_output and raw_output != deterministic:
        # Not an error: CrewAI can trim or reformat a final answer. The
        # deterministic composition is logged so any divergence is visible.
        LOGGER.info(
            "crew output differs from the deterministic composition "
            "(crew=%d chars, deterministic=%d chars)",
            len(raw_output),
            len(deterministic),
        )


    return CrewDraft(
        draft=raw_output or deterministic,
        context=context,
        crew_mode=CREW_MODE_CREWAI,
        raw_output=raw_output,
        agents_used=[AGENT_SPECS[key].role for key in agent_keys],
        llm_calls={key: mock_llms[key].call_count for key in agent_keys},
    )




# --------------------------------------------------------------------------- #
# Direct path (diagnostic)
# --------------------------------------------------------------------------- #




def _run_direct(
    context: CrewRunContext, generator: GroundedGenerator, settings: Settings
) -> CrewDraft:
    """Straight-line pipeline over the same tools and the same composition."""
    agent_keys = _agent_keys_for_route(context)


    # The same governance gate the CrewAI path uses, so least autonomy holds in
    # both modes rather than only in the graded one.
    if context.needs_policy:
        assert_tool_assignment(AGENT_RETRIEVAL, [TOOL_POLICY_LOOKUP])
        invoke_policy_lookup(context, generator)


    if context.needs_appointment and context.record_id:
        assert_tool_assignment(AGENT_LOOKUP, [TOOL_APPOINTMENT_LOOKUP])
        invoke_appointment_lookup(context)


    return CrewDraft(
        draft=compose_draft(context),
        context=context,
        crew_mode=CREW_MODE_DIRECT,
        raw_output="",
        agents_used=[AGENT_SPECS[key].role for key in agent_keys],
        llm_calls={},
    )




# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #




def run_crew(
    context: CrewRunContext,
    generator: GroundedGenerator,
    settings: Settings = SETTINGS,
) -> CrewDraft:
    """Produce a draft answer for one turn, using the configured orchestration mode."""
    if settings.crew_mode == CREW_MODE_CREWAI:
        return _run_with_crewai(context, generator, settings)
    if settings.crew_mode == CREW_MODE_DIRECT:
        LOGGER.warning(
            "CREW_MODE=direct: running the diagnostic straight-line pipeline. This "
            "is not the graded path; set CREW_MODE=crewai for real crew.kickoff() "
            "orchestration."
        )
        return _run_direct(context, generator, settings)
    raise CrewExecutionError(f"unsupported CREW_MODE {settings.crew_mode!r}.")




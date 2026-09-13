"""``MOCK_LLM`` for CrewAI - a deterministic ``BaseLLM`` implementation.


Implemented by extending ``crewai.llms.base_llm.BaseLLM``, CrewAI's own
documented extension point for a non-``litellm`` model, rather than trying to
intercept calls from outside. One instance is created per agent, so each agent's
"model" knows which role it is playing and returns exactly the ReAct text CrewAI
expects for that role.


Both known pitfalls are handled explicitly.


**Pitfall 1 - the ``"Observation:"`` template trap.**
    CrewAI's built-in ReAct system prompt literally contains the example line
    ``"Observation: the result of the action"``. A mock that searches the whole
    conversation for ``"Observation:"`` to find a tool result matches that
    template text on the very first call - before any tool has run - and
    silently returns placeholder text as the final answer. No crash, just wrong
    output.


    Mitigation, in two layers:


    1. The authoritative "has the tool run?" signal is the
       ``CrewRunContext`` ledger, which the tool wrappers write to when they
       actually execute. Prompt text is never the source of truth.
    2. ``extract_generated_observation()`` exists as a cross-check and is
       deliberately hardened: it skips ``system`` messages entirely and rejects
       the template's own placeholder string. It is used for diagnostics, never
       as the sole trigger.


**Pitfall 2 - dispatching on the tool's name.**
    Deciding what to send a tool by testing whether ``"lookup"`` appears in its
    name misclassifies a tool called ``rag_lookup``. Dispatch here goes through
    ``agents.tools.classify_tool()``, which keys off the tool's own declared
    ``args_schema`` field names (``{"query"}`` vs ``{"record_id"}``). The tool's
    *name* is only ever used verbatim, as the string CrewAI needs to match the
    ``Action:`` line.
"""


from __future__ import annotations


import json
import logging
import re
from typing import Any, Final, Sequence


try:
    from crewai.llms.base_llm import BaseLLM
except ImportError:  # CrewAI 0.100.x exposes the same hook as `crewai.llm.LLM`.
    from crewai.llm import LLM as BaseLLM


from app.config import (
    AGENT_COMPOSER,
    AGENT_LOOKUP,
    AGENT_RETRIEVAL,
    FALLBACK_ANSWER,
    TOOL_APPOINTMENT_LOOKUP,
    TOOL_POLICY_LOOKUP,
)
from agents.composition import (
    MISSING_RECORD_ID_ANSWER,
    compose_appointment_section,
    compose_draft,
    compose_policy_section,
)
from agents.context import CrewRunContext
from agents.tools import ToolDispatchError, classify_tool


LOGGER: Final = logging.getLogger(__name__)


#: The exact placeholder text CrewAI's own ReAct template uses. Any
#: "observation" equal to this is template scaffolding, not a tool result.
CREWAI_TEMPLATE_OBSERVATION_PLACEHOLDER: Final[str] = "the result of the action"


#: Hard cap on how many times one agent's mock model will ask for a tool before
#: it gives up and answers. Belt and braces alongside CrewAI's own ``max_iter``.
MAX_TOOL_ATTEMPTS: Final[int] = 2


_OBSERVATION_MARKER: Final[re.Pattern[str]] = re.compile(r"Observation\s*:", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Message helpers
# --------------------------------------------------------------------------- #


def normalise_messages(messages: str | Sequence[Any]) -> list[dict[str, str]]:
    """Coerce CrewAI's ``messages`` argument into a list of role/content dicts."""
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]


    normalised: list[dict[str, str]] = []
    for message in messages or []:
        if isinstance(message, dict):
            role = str(message.get("role", "user"))
            content = message.get("content", "")
        else:
            role = str(getattr(message, "role", "user"))
            content = getattr(message, "content", "")
        normalised.append({"role": role, "content": "" if content is None else str(content)})
    return normalised


def extract_generated_observation(messages: str | Sequence[Any]) -> str | None:
    """Read the most recent *genuine* tool observation out of the conversation.


    Hardened against pitfall 1:


    * ``system`` messages are skipped outright, because that is where CrewAI's
      ReAct template - and its ``"Observation: the result of the action"``
      example - lives;
    * an extracted observation equal to the template placeholder is rejected;
    * an empty extraction is rejected.


    Returns ``None`` when there is no real observation yet. Diagnostic only: the
    ledger on ``CrewRunContext`` is what actually gates tool re-invocation.
    """
    for message in reversed(normalise_messages(messages)):
        if message["role"] == "system":
            continue
        content = message["content"]
        match = None
        for candidate in _OBSERVATION_MARKER.finditer(content):
            match = candidate
        if match is None:
            continue
        observed = content[match.end() :].strip()
        if not observed:
            continue
        if observed.lower().startswith(CREWAI_TEMPLATE_OBSERVATION_PLACEHOLDER):
            LOGGER.debug("ignored CrewAI template placeholder observation")
            continue
        return observed
    return None


# --------------------------------------------------------------------------- #
# ReAct formatting
# --------------------------------------------------------------------------- #


def format_action(tool_name: str, arguments: dict[str, Any], thought: str) -> str:
    """Render a CrewAI ReAct tool-call turn.


    ``Action Input`` is a single-line JSON object, and no ``Observation:`` line is
    emitted - CrewAI appends the real one after it runs the tool. Emitting both
    an ``Action`` and a ``Final Answer`` in one response makes CrewAI's parser
    raise, so exactly one of the two is produced per call.
    """
    payload = json.dumps(arguments, ensure_ascii=False)
    return f"Thought: {thought}\nAction: {tool_name}\nAction Input: {payload}"


def format_final_answer(answer: str) -> str:
    """Render a CrewAI ReAct final-answer turn."""
    return f"Thought: I now can give a great answer\nFinal Answer: {answer}"


# --------------------------------------------------------------------------- #
# The mock LLM
# --------------------------------------------------------------------------- #


class MockCrewLLM(BaseLLM):
    """Deterministic per-agent stand-in for a real model.


    Args:
        agent_key: which crew role this instance serves.
        context: the shared per-turn context and tool ledger.
        tools: the tool objects actually assigned to this agent. Passed in
            rather than read from CrewAI's ``tools`` argument, because with
            ``supports_function_calling() == False`` CrewAI advertises tools
            through the prompt and may pass nothing here at all.
    """


    def __init__(
        self,
        agent_key: str,
        context: CrewRunContext,
        tools: Sequence[Any] | None = None,
        *,
        model: str = "mock-practo-llm",
        temperature: float = 0.0,
    ) -> None:
        super().__init__(model=model, temperature=temperature)
        # Some CrewAI versions read `stop` off the LLM before any call.
        if not hasattr(self, "stop") or self.stop is None:
            self.stop = []
        self.agent_key = agent_key
        self.context = context
        self.tools = list(tools or [])
        self.call_count = 0
        self.tool_attempts = 0


    # -- CrewAI capability declarations --------------------------------- #


    def supports_function_calling(self) -> bool:
        """False on purpose: this drives CrewAI's prompt-based ReAct path.


        That path is what the two documented pitfalls live in, so exercising it
        is the point rather than something to route around.
        """
        return False


    def supports_stop_words(self) -> bool:
        return False


    def get_context_window_size(self) -> int:
        return 8192


    # -- Tool dispatch --------------------------------------------------- #


    def _tool_of_kind(self, kind: str) -> Any | None:
        """Find this agent's tool of the given kind, by argument schema."""
        for tool in self.tools:
            try:
                if classify_tool(tool) == kind:
                    return tool
            except ToolDispatchError:
                LOGGER.warning(
                    "tool %r could not be classified by its argument schema and was skipped",
                    getattr(tool, "name", tool),
                )
        return None


    def _arguments_for(self, kind: str) -> dict[str, Any]:
        """Build the ``Action Input`` payload for a tool kind.


        Keyed on the *kind* resolved from the argument schema, so the argument
        name is always the one the tool actually declares.
        """
        if kind == TOOL_POLICY_LOOKUP:
            return {"query": self.context.query}
        if kind == TOOL_APPOINTMENT_LOOKUP:
            return {"record_id": self.context.record_id or ""}
        raise ToolDispatchError(f"no argument builder for tool kind {kind!r}.")


    # -- The call itself -------------------------------------------------- #


    def call(
        self,
        messages: str | Sequence[Any],
        tools: Sequence[Any] | None = None,
        callbacks: Sequence[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
    ) -> str:
        """Return the next ReAct turn for this agent, deterministically."""
        self.call_count += 1


        observation = extract_generated_observation(messages)
        LOGGER.debug(
            "mock llm call #%s agent=%s ledger=%s genuine_observation=%s",
            self.call_count,
            self.agent_key,
            self.context.tools_invoked,
            observation is not None,
        )


        if self.agent_key == AGENT_RETRIEVAL:
            return self._retrieval_turn()
        if self.agent_key == AGENT_LOOKUP:
            return self._lookup_turn()
        if self.agent_key == AGENT_COMPOSER:
            return format_final_answer(compose_draft(self.context))


        raise ValueError(f"unknown agent_key {self.agent_key!r}.")


    # -- Per-role behaviour ---------------------------------------------- #


    def _retrieval_turn(self) -> str:
        tool = self._tool_of_kind(TOOL_POLICY_LOOKUP)
        already_ran = self.context.was_invoked(TOOL_POLICY_LOOKUP)


        if tool is not None and not already_ran and self.tool_attempts < MAX_TOOL_ATTEMPTS:
            self.tool_attempts += 1
            return format_action(
                tool_name=tool.name,
                arguments=self._arguments_for(TOOL_POLICY_LOOKUP),
                thought=(
                    "I must not answer a policy question from memory. I will search "
                    "the Practo policy knowledge base first."
                ),
            )


        answer = compose_policy_section(self.context) or FALLBACK_ANSWER
        return format_final_answer(answer)


    def _lookup_turn(self) -> str:
        tool = self._tool_of_kind(TOOL_APPOINTMENT_LOOKUP)
        already_ran = self.context.was_invoked(TOOL_APPOINTMENT_LOOKUP)


        if not self.context.record_id:
            # No id to look up, so no tool call is warranted - asking is the
            # correct behaviour, and it is what the fresh-session memory
            # transcript demonstrates.
            return format_final_answer(MISSING_RECORD_ID_ANSWER)


        if tool is not None and not already_ran and self.tool_attempts < MAX_TOOL_ATTEMPTS:
            self.tool_attempts += 1
            return format_action(
                tool_name=tool.name,
                arguments=self._arguments_for(TOOL_APPOINTMENT_LOOKUP),
                thought=(
                    f"The patient named appointment {self.context.record_id}. I will "
                    "look up its authoritative status rather than guess."
                ),
            )


        return format_final_answer(
            compose_appointment_section(self.context) or MISSING_RECORD_ID_ANSWER
        )


def build_mock_llms(
    context: CrewRunContext, tools_for_agent: dict[str, list[Any]]
) -> dict[str, MockCrewLLM]:
    """One ``MockCrewLLM`` per crew agent, each bound to that agent's own tools."""
    return {
        agent_key: MockCrewLLM(agent_key=agent_key, context=context, tools=assigned)
        for agent_key, assigned in tools_for_agent.items()
    }



"""Task 7 - the CrewAI crew and the MOCK_LLM BaseLLM implementation.


Skipped in full when crewai is not installed, so the rest of the suite stays
runnable on a machine without the heavier orchestration dependency.
"""


from __future__ import annotations


import dataclasses
import json


import pytest


pytest.importorskip("crewai")


from app.config import (  # noqa: E402 - must follow importorskip
    AGENT_COMPOSER,
    AGENT_LOOKUP,
    AGENT_RETRIEVAL,
    CREW_MODE_CREWAI,
    TOOL_APPOINTMENT_LOOKUP,
    TOOL_POLICY_LOOKUP,
    Settings,
)
from agents.composition import MISSING_RECORD_ID_ANSWER  # noqa: E402
from agents.context import (  # noqa: E402
    ROUTE_APPOINTMENT,
    ROUTE_COMBINED,
    ROUTE_POLICY,
    CrewRunContext,
)
from agents.crew import AGENT_SPECS, run_crew  # noqa: E402
from agents.governance import ToolPermissionError  # noqa: E402
from agents.mock_llm import (  # noqa: E402
    CREWAI_TEMPLATE_OBSERVATION_PLACEHOLDER,
    MAX_TOOL_ATTEMPTS,
    MockCrewLLM,
    extract_generated_observation,
    format_action,
    format_final_answer,
    normalise_messages,
)
from agents.routing import classify_route  # noqa: E402
from agents.tools import build_crew_tools, classify_tool  # noqa: E402
from dataset import APPOINTMENTS  # noqa: E402
from rag.grounded_generation import GroundedGenerator  # noqa: E402


pytestmark = pytest.mark.requires_crewai


RECORD_ID = APPOINTMENTS[6]["record_id"]
POLICY_QUERY = "How long before my appointment can I cancel without paying a fee?"




@pytest.fixture()
def crewai_settings(test_settings: Settings) -> Settings:
    return dataclasses.replace(test_settings, crew_mode=CREW_MODE_CREWAI)




def make_context(query: str) -> CrewRunContext:
    from agents.routing import extract_record_id


    return CrewRunContext(
        trace_id="test-trace",
        session_id="test-session",
        query=query,
        route=classify_route(query),
        record_id=extract_record_id(query),
    )




class TestReActFormatting:
    def test_an_action_turn_has_no_final_answer(self) -> None:
        text = format_action("some_tool", {"query": "q"}, "thinking")
        assert "Action: some_tool" in text
        assert 'Action Input: {"query": "q"}' in text
        assert "Final Answer:" not in text
        # CrewAI's parser raises if a response carries both.
        assert "Observation:" not in text


    def test_action_input_is_a_single_json_line(self) -> None:
        text = format_action("t", {"query": "multi\nline"}, "thought")
        payload_line = [
            line for line in text.splitlines() if line.startswith("Action Input:")
        ][0]
        json.loads(payload_line[len("Action Input:") :].strip())


    def test_a_final_answer_turn_has_no_action(self) -> None:
        text = format_final_answer("the answer")
        assert "Final Answer: the answer" in text
        assert "\nAction:" not in text




class TestObservationTemplateGuard:
    """Pitfall 1: CrewAI's own system prompt contains an Observation example."""


    def test_the_template_placeholder_is_ignored(self) -> None:
        messages = [
            {
                "role": "system",
                "content": (
                    "Thought: you should always think about what to do\n"
                    "Action: the action to take\n"
                    "Action Input: the input to the action\n"
                    f"Observation: {CREWAI_TEMPLATE_OBSERVATION_PLACEHOLDER}"
                ),
            },
            {"role": "user", "content": "Answer this policy question."},
        ]
        assert extract_generated_observation(messages) is None


    def test_a_genuine_observation_is_returned(self) -> None:
        messages = [
            {"role": "system", "content": f"Observation: {CREWAI_TEMPLATE_OBSERVATION_PLACEHOLDER}"},
            {"role": "user", "content": 'Observation: {"grounded": true}'},
        ]
        assert extract_generated_observation(messages) == '{"grounded": true}'


    def test_the_placeholder_is_ignored_even_outside_a_system_message(self) -> None:
        messages = [
            {"role": "user", "content": f"Observation: {CREWAI_TEMPLATE_OBSERVATION_PLACEHOLDER}"}
        ]
        assert extract_generated_observation(messages) is None


    def test_the_latest_observation_wins(self) -> None:
        messages = [
            {"role": "user", "content": "Observation: first"},
            {"role": "user", "content": "Observation: second"},
        ]
        assert extract_generated_observation(messages) == "second"


    def test_no_observation_returns_none(self) -> None:
        assert extract_generated_observation([{"role": "user", "content": "hello"}]) is None


    def test_a_bare_string_is_accepted(self) -> None:
        assert normalise_messages("just a prompt")[0]["role"] == "user"




class TestMockCrewLLM:
    def test_the_retrieval_agent_calls_its_tool_then_answers(
        self, crewai_settings: Settings, generator: GroundedGenerator
    ) -> None:
        context = make_context(POLICY_QUERY)
        tools = build_crew_tools(context, generator, crewai_settings)
        llm = MockCrewLLM(AGENT_RETRIEVAL, context, [tools.policy_tool])


        first = llm.call([{"role": "user", "content": POLICY_QUERY}])
        assert f"Action: {TOOL_POLICY_LOOKUP}" in first
        assert "Final Answer:" not in first


        # Simulate CrewAI executing the tool it just asked for.
        tools.policy_tool._run(query=context.query)
        assert context.was_invoked(TOOL_POLICY_LOOKUP)


        second = llm.call([{"role": "user", "content": POLICY_QUERY}])
        assert "Final Answer:" in second
        assert "Action:" not in second


    def test_the_lookup_agent_asks_for_a_missing_id_without_calling_a_tool(
        self, crewai_settings: Settings, generator: GroundedGenerator
    ) -> None:
        context = make_context("And what is its current status?")
        assert context.record_id is None
        tools = build_crew_tools(context, generator, crewai_settings)
        llm = MockCrewLLM(AGENT_LOOKUP, context, [tools.appointment_tool])


        text = llm.call([{"role": "user", "content": context.query}])
        assert MISSING_RECORD_ID_ANSWER in text
        assert "Action:" not in text
        assert context.invocations == []


    def test_the_lookup_agent_calls_its_tool_with_the_record_id(
        self, crewai_settings: Settings, generator: GroundedGenerator
    ) -> None:
        context = make_context(f"What is the status of {RECORD_ID}?")
        tools = build_crew_tools(context, generator, crewai_settings)
        llm = MockCrewLLM(AGENT_LOOKUP, context, [tools.appointment_tool])


        text = llm.call([{"role": "user", "content": context.query}])
        assert f"Action: {TOOL_APPOINTMENT_LOOKUP}" in text
        assert RECORD_ID in text


    def test_the_composer_answers_immediately(
        self, crewai_settings: Settings, generator: GroundedGenerator
    ) -> None:
        context = make_context(POLICY_QUERY)
        tools = build_crew_tools(context, generator, crewai_settings)
        tools.policy_tool._run(query=context.query)


        llm = MockCrewLLM(AGENT_COMPOSER, context, [])
        text = llm.call([{"role": "user", "content": "compose"}])
        assert "Final Answer:" in text
        assert "Action:" not in text


    def test_tool_attempts_are_bounded(
        self, crewai_settings: Settings, generator: GroundedGenerator
    ) -> None:
        context = make_context(POLICY_QUERY)
        tools = build_crew_tools(context, generator, crewai_settings)
        llm = MockCrewLLM(AGENT_RETRIEVAL, context, [tools.policy_tool])


        # The tool never "runs", so without the cap the mock would loop forever.
        for _ in range(MAX_TOOL_ATTEMPTS):
            assert "Action:" in llm.call([{"role": "user", "content": "q"}])
        assert "Final Answer:" in llm.call([{"role": "user", "content": "q"}])


    def test_capabilities_drive_the_react_path(
        self, crewai_settings: Settings, generator: GroundedGenerator
    ) -> None:
        context = make_context(POLICY_QUERY)
        llm = MockCrewLLM(AGENT_COMPOSER, context, [])
        assert llm.supports_function_calling() is False
        assert llm.supports_stop_words() is False
        assert llm.get_context_window_size() > 0


    def test_an_unknown_agent_key_is_rejected(
        self, crewai_settings: Settings, generator: GroundedGenerator
    ) -> None:
        context = make_context(POLICY_QUERY)
        llm = MockCrewLLM("mystery_agent", context, [])
        with pytest.raises(ValueError):
            llm.call([{"role": "user", "content": "q"}])




class TestCrewTools:
    def test_tools_are_classified_by_their_argument_schema(
        self, crewai_settings: Settings, generator: GroundedGenerator
    ) -> None:
        context = make_context(POLICY_QUERY)
        tools = build_crew_tools(context, generator, crewai_settings)
        assert classify_tool(tools.policy_tool) == TOOL_POLICY_LOOKUP
        assert classify_tool(tools.appointment_tool) == TOOL_APPOINTMENT_LOOKUP


    def test_each_agent_receives_only_its_own_tool(
        self, crewai_settings: Settings, generator: GroundedGenerator
    ) -> None:
        context = make_context(POLICY_QUERY)
        tools = build_crew_tools(context, generator, crewai_settings)
        assert [t.name for t in tools.for_agent(AGENT_RETRIEVAL)] == [TOOL_POLICY_LOOKUP]
        assert [t.name for t in tools.for_agent(AGENT_LOOKUP)] == [
            TOOL_APPOINTMENT_LOOKUP
        ]
        assert tools.for_agent(AGENT_COMPOSER) == []


    def test_the_policy_tool_records_its_result_on_the_context(
        self, crewai_settings: Settings, generator: GroundedGenerator
    ) -> None:
        context = make_context(POLICY_QUERY)
        tools = build_crew_tools(context, generator, crewai_settings)
        payload = json.loads(tools.policy_tool._run(query=context.query))
        assert set(payload) == {"grounded", "answer", "sources", "top_similarity"}
        assert context.grounded is not None
        assert context.invocations[0].tool_name == TOOL_POLICY_LOOKUP
        assert context.invocations[0].agent_key == AGENT_RETRIEVAL


    def test_the_appointment_tool_records_its_result_on_the_context(
        self, crewai_settings: Settings, generator: GroundedGenerator
    ) -> None:
        context = make_context(f"Status of {RECORD_ID}?")
        tools = build_crew_tools(context, generator, crewai_settings)
        payload = json.loads(tools.appointment_tool._run(record_id=RECORD_ID))
        assert payload["found"] is True
        assert context.lookup is not None
        assert context.invocations[0].tool_name == TOOL_APPOINTMENT_LOOKUP




class TestCrewKickoff:
    def test_a_policy_query_invokes_only_the_rag_tool(
        self, crewai_settings: Settings, generator: GroundedGenerator
    ) -> None:
        context = make_context(POLICY_QUERY)
        assert context.route == ROUTE_POLICY
        draft = run_crew(context, generator, crewai_settings)
        assert draft.tools_invoked == [TOOL_POLICY_LOOKUP]
        assert draft.crew_mode == CREW_MODE_CREWAI
        assert draft.draft
        assert AGENT_SPECS[AGENT_RETRIEVAL].role in draft.agents_used
        assert AGENT_SPECS[AGENT_LOOKUP].role not in draft.agents_used


    def test_an_appointment_query_invokes_only_the_lookup_tool(
        self, crewai_settings: Settings, generator: GroundedGenerator
    ) -> None:
        context = make_context(f"What is the status of {RECORD_ID}?")
        assert context.route == ROUTE_APPOINTMENT
        draft = run_crew(context, generator, crewai_settings)
        assert draft.tools_invoked == [TOOL_APPOINTMENT_LOOKUP]
        assert RECORD_ID in draft.draft


    def test_a_combined_query_invokes_both_tools(
        self, crewai_settings: Settings, generator: GroundedGenerator
    ) -> None:
        context = make_context(
            f"What is the cancellation window and the status of {RECORD_ID}?"
        )
        assert context.route == ROUTE_COMBINED
        draft = run_crew(context, generator, crewai_settings)
        assert set(draft.tools_invoked) == {
            TOOL_POLICY_LOOKUP,
            TOOL_APPOINTMENT_LOOKUP,
        }
        assert len(draft.agents_used) == 3


    def test_every_agent_reports_at_least_one_llm_call(
        self, crewai_settings: Settings, generator: GroundedGenerator
    ) -> None:
        context = make_context(POLICY_QUERY)
        draft = run_crew(context, generator, crewai_settings)
        assert draft.llm_calls
        assert all(count >= 1 for count in draft.llm_calls.values())


    def test_an_unsupported_crew_mode_is_rejected(
        self, generator: GroundedGenerator
    ) -> None:
        from agents.crew import CrewExecutionError


        class UnknownMode:
            """`run_crew` reads only `crew_mode`, so this is enough to drive it."""


            crew_mode = "teleportation"


        context = make_context(POLICY_QUERY)
        with pytest.raises(CrewExecutionError):
            run_crew(context, generator, UnknownMode())  # type: ignore[arg-type]




class TestLeastAutonomyInsideTheCrew:
    def test_wiring_the_appointment_tool_to_another_agent_is_blocked(
        self, crewai_settings: Settings, generator: GroundedGenerator
    ) -> None:
        from agents.crew import _authorised_tool_map


        context = make_context(POLICY_QUERY)
        tools = build_crew_tools(context, generator, crewai_settings)


        class Overreaching:
            """A tool map that hands the appointment tool to the Retrieval Agent."""


            def __init__(self, appointment_tool: object) -> None:
                self._appointment_tool = appointment_tool


            def for_agent(self, agent_key: str) -> list[object]:
                return [self._appointment_tool]


        with pytest.raises(ToolPermissionError):
            _authorised_tool_map(
                Overreaching(tools.appointment_tool), [AGENT_RETRIEVAL]  # type: ignore[arg-type]
            )




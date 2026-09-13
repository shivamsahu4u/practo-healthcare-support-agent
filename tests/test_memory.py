"""Task 8 - session memory: state carried within a session, absent across sessions."""


from __future__ import annotations


import pytest
from langchain_core.messages import AIMessage, HumanMessage


from app.services.support_service import SupportService
from agents.memory import (
    SessionMemory,
    resolve_record_id_from_history,
)
from agents.routing import (
    classify_route,
    extract_record_id,
    has_appointment_intent,
    has_policy_intent,
)
from agents.context import ROUTE_APPOINTMENT, ROUTE_COMBINED, ROUTE_POLICY
from dataset import APPOINTMENTS


FOLLOW_UP = "And what is its current status?"


class TestSessionMemory:
    def test_histories_are_isolated_by_session_id(self) -> None:
        memory = SessionMemory()
        memory.get_history("a").add_user_message("first")
        memory.get_history("b").add_user_message("second")
        assert len(memory.messages("a")) == 1
        assert len(memory.messages("b")) == 1
        assert memory.messages("a")[0].content == "first"


    def test_the_same_session_id_returns_the_same_history(self) -> None:
        memory = SessionMemory()
        assert memory.get_history("x") is memory.get_history("x")


    def test_reset_clears_one_session_only(self) -> None:
        memory = SessionMemory()
        memory.get_history("a").add_user_message("hello")
        memory.get_history("b").add_user_message("hello")
        assert memory.reset("a") is True
        assert memory.messages("a") == []
        assert len(memory.messages("b")) == 1


    def test_resetting_an_unknown_session_is_not_an_error(self) -> None:
        assert SessionMemory().reset("never-seen") is False


    def test_reset_all_reports_the_count(self) -> None:
        memory = SessionMemory()
        memory.get_history("a")
        memory.get_history("b")
        assert memory.reset_all() == 2
        assert memory.sessions() == []


    def test_empty_session_id_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            SessionMemory().get_history("   ")


    def test_transcript_is_readable(self) -> None:
        memory = SessionMemory()
        history = memory.get_history("a")
        history.add_user_message("question")
        history.add_ai_message("answer")
        transcript = memory.transcript("a")
        assert [entry["content"] for entry in transcript] == ["question", "answer"]


class TestRecordIdRecovery:
    def test_finds_an_id_in_a_human_message(self) -> None:
        history = [HumanMessage(content="Status of APT-1007 please?")]
        assert resolve_record_id_from_history(history) == "APT-1007"


    def test_finds_an_id_in_an_ai_message(self) -> None:
        history = [AIMessage(content="Appointment APT-1023 is currently Completed.")]
        assert resolve_record_id_from_history(history) == "APT-1023"


    def test_prefers_the_most_recent_mention(self) -> None:
        history = [
            HumanMessage(content="About APT-1001"),
            AIMessage(content="Noted."),
            HumanMessage(content="Now about APT-1044"),
        ]
        assert resolve_record_id_from_history(history) == "APT-1044"


    def test_returns_none_for_an_empty_history(self) -> None:
        assert resolve_record_id_from_history([]) is None


    @pytest.mark.parametrize(
        "text,expected",
        [
            ("APT-1007", "APT-1007"),
            ("apt-1007", "APT-1007"),
            ("apt 1007", "APT-1007"),
            ("APT_1007", "APT-1007"),
            ("apt1007", "APT-1007"),
            ("no id here", None),
            ("APT-107", None),
        ],
    )
    def test_extraction_normalises(self, text: str, expected: str | None) -> None:
        assert extract_record_id(text) == expected


class TestRouting:
    @pytest.mark.parametrize(
        "query,expected",
        [
            ("How long before my appointment can I cancel?", ROUTE_POLICY),
            ("What is the consultation fee for cardiology?", ROUTE_POLICY),
            ("What is the status of APT-1007?", ROUTE_APPOINTMENT),
            ("And what is its current status?", ROUTE_APPOINTMENT),
            (
                "What is the cancellation window and the status of APT-1007?",
                ROUTE_COMBINED,
            ),
        ],
    )
    def test_routes(self, query: str, expected: str) -> None:
        assert classify_route(query) == expected


    def test_a_policy_question_mentioning_appointment_stays_policy(self) -> None:
        query = "How long before my appointment can I cancel without paying a fee?"
        assert has_appointment_intent(query)
        assert has_policy_intent(query)
        assert classify_route(query) == ROUTE_POLICY


    def test_memory_promotes_a_contextless_follow_up(self) -> None:
        assert (
            classify_route("And what about that one?", record_id_from_memory="APT-1007")
            == ROUTE_APPOINTMENT
        )


    def test_memory_does_not_hijack_a_policy_follow_up(self) -> None:
        assert (
            classify_route(
                "And what is the refund policy?", record_id_from_memory="APT-1007"
            )
            == ROUTE_POLICY
        )


class TestEndToEndMemory:
    async def test_state_is_carried_within_one_session(
        self, service: SupportService
    ) -> None:
        record_id = APPOINTMENTS[6]["record_id"]
        session = "same-session"


        first = await service.answer(
            f"What is the status of appointment {record_id}?", session_id=session
        )
        assert first.appointment is not None
        assert first.appointment.record_id == record_id


        second = await service.answer(FOLLOW_UP, session_id=session)
        assert second.appointment is not None, "the record id was not carried forward"
        assert second.appointment.record_id == record_id
        assert second.response_type == "appointment"


    async def test_state_is_absent_in_a_fresh_session(
        self, service: SupportService
    ) -> None:
        fresh = await service.answer(FOLLOW_UP, session_id="fresh-session")
        assert fresh.appointment is None
        assert "appointment id" in fresh.answer.lower()


    async def test_two_sessions_do_not_leak_into_each_other(
        self, service: SupportService
    ) -> None:
        first_id = APPOINTMENTS[2]["record_id"]
        second_id = APPOINTMENTS[9]["record_id"]


        await service.answer(f"Status of {first_id}?", session_id="alpha")
        await service.answer(f"Status of {second_id}?", session_id="beta")


        alpha = await service.answer(FOLLOW_UP, session_id="alpha")
        beta = await service.answer(FOLLOW_UP, session_id="beta")


        assert alpha.appointment is not None and beta.appointment is not None
        assert alpha.appointment.record_id == first_id
        assert beta.appointment.record_id == second_id


    async def test_only_masked_text_is_stored(self, service: SupportService) -> None:
        session = "masked-session"
        await service.answer(
            "My number is 9876543210, what is the cancellation window?",
            session_id=session,
        )
        stored = " ".join(
            entry["content"] for entry in service.sessions.transcript(session)
        )
        assert "9876543210" not in stored
        assert "[CONTACT_MASKED]" in stored


    async def test_reset_removes_carried_state(self, service: SupportService) -> None:
        record_id = APPOINTMENTS[4]["record_id"]
        session = "reset-session"
        await service.answer(f"Status of {record_id}?", session_id=session)
        assert service.sessions.reset(session) is True
        after = await service.answer(FOLLOW_UP, session_id=session)
        assert after.appointment is None



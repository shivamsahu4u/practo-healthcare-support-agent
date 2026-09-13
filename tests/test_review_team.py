"""Task 14 - the Autogen review stage: verdict logic and the real round-robin team."""


from __future__ import annotations


import pytest
from pydantic import ValidationError


from app.config import FALLBACK_ANSWER, SETTINGS
from app.models import ReviewVerdict
from agents.composition import INJECTED_UNSUPPORTED_CLAIM
from agents.review_team import (
    EDITOR_SYSTEM_MESSAGE,
    MARKER_EDITOR,
    MARKER_REVIEWER,
    REVIEWER_SYSTEM_MESSAGE,
    ReviewSession,
    deterministic_verdict,
    find_unsupported,
    reviewer_critique,
    strip_sentences,
)


GROUNDED_CONTEXT = (
    "An appointment can be cancelled free of charge up to 4 hours before the "
    "scheduled start time, and the consultation fee is refunded in full to the "
    "original payment method. Cancelling inside the 4-hour window retains a 25 "
    "percent late-cancellation charge."
)




def make_session(draft: str, *, grounded: bool = True) -> ReviewSession:
    return ReviewSession(
        query="How long before my appointment can I cancel without paying a fee?",
        draft=draft,
        context_text=GROUNDED_CONTEXT,
        support_text=GROUNDED_CONTEXT,
        source_ids=["cancellation_rescheduling"],
        lookup=None,
        retrieval_grounded=grounded,
        minimum_overlap=0.6,
    )




class TestVerdictModel:
    def test_requires_all_three_fields(self) -> None:
        with pytest.raises(ValidationError):
            ReviewVerdict(approved=True, final_answer="x")  # type: ignore[call-arg]


    def test_rejects_unexpected_fields(self) -> None:
        with pytest.raises(ValidationError):
            ReviewVerdict(
                approved=True, final_answer="x", reason="y", extra="nope"
            )  # type: ignore[call-arg]


    def test_round_trips_through_json(self) -> None:
        verdict = ReviewVerdict(
            approved=True, final_answer="answer", reason="because"
        )
        assert ReviewVerdict.model_validate_json(verdict.model_dump_json()) == verdict




class TestSupportDetection:
    def test_a_verbatim_draft_has_nothing_unsupported(self) -> None:
        assert find_unsupported(make_session(GROUNDED_CONTEXT)) == []


    def test_the_injected_claim_is_detected(self) -> None:
        session = make_session(f"{GROUNDED_CONTEXT} {INJECTED_UNSUPPORTED_CLAIM}")
        unsupported = find_unsupported(session)
        assert len(unsupported) == 1
        assert "platinum" in unsupported[0]


    def test_strip_sentences_keeps_the_remainder_verbatim(self) -> None:
        draft = f"{GROUNDED_CONTEXT} {INJECTED_UNSUPPORTED_CLAIM}"
        stripped = strip_sentences(draft, [INJECTED_UNSUPPORTED_CLAIM])
        assert INJECTED_UNSUPPORTED_CLAIM not in stripped
        assert "cancelled free of charge" in stripped


    def test_stripping_an_absent_sentence_changes_nothing(self) -> None:
        assert strip_sentences(GROUNDED_CONTEXT, ["Not present."]).startswith(
            "An appointment can be cancelled"
        )




class TestDeterministicVerdict:
    def test_a_grounded_draft_is_approved_unchanged(self) -> None:
        verdict = deterministic_verdict(make_session(GROUNDED_CONTEXT))
        assert verdict.approved is True
        assert verdict.final_answer == GROUNDED_CONTEXT
        assert "supported" in verdict.reason


    def test_an_injected_claim_is_removed(self) -> None:
        draft = f"{GROUNDED_CONTEXT} {INJECTED_UNSUPPORTED_CLAIM}"
        verdict = deterministic_verdict(make_session(draft))
        assert verdict.approved is False
        assert INJECTED_UNSUPPORTED_CLAIM not in verdict.final_answer
        assert "cancelled free of charge" in verdict.final_answer
        assert "unsupported" in verdict.reason


    def test_an_empty_draft_is_rejected(self) -> None:
        verdict = deterministic_verdict(make_session("   "))
        assert verdict.approved is False
        assert verdict.final_answer == FALLBACK_ANSWER


    def test_failed_retrieval_forces_a_refusal(self) -> None:
        verdict = deterministic_verdict(make_session(GROUNDED_CONTEXT, grounded=False))
        assert verdict.approved is False
        assert verdict.final_answer == FALLBACK_ANSWER
        assert "calibrated similarity threshold" in verdict.reason


    def test_a_wholly_unsupported_draft_becomes_the_fallback(self) -> None:
        verdict = deterministic_verdict(make_session(INJECTED_UNSUPPORTED_CLAIM))
        assert verdict.approved is False
        assert verdict.final_answer == FALLBACK_ANSWER


    def test_the_verdict_is_stable(self) -> None:
        session = make_session(f"{GROUNDED_CONTEXT} {INJECTED_UNSUPPORTED_CLAIM}")
        assert deterministic_verdict(session) == deterministic_verdict(session)




class TestReviewerCritique:
    def test_a_clean_draft_reports_no_finding(self) -> None:
        critique = reviewer_critique(make_session(GROUNDED_CONTEXT))
        assert "FINDING: none" in critique
        assert "cancellation_rescheduling" in critique


    def test_an_injected_claim_is_named(self) -> None:
        critique = reviewer_critique(
            make_session(f"{GROUNDED_CONTEXT} {INJECTED_UNSUPPORTED_CLAIM}")
        )
        assert "1 unsupported sentence" in critique
        assert "platinum" in critique


    def test_failed_retrieval_is_reported(self) -> None:
        critique = reviewer_critique(make_session(GROUNDED_CONTEXT, grounded=False))
        assert "no admissible policy support" in critique




class TestTaskMessage:
    def test_carries_everything_the_reviewer_needs(self) -> None:
        message = make_session(GROUNDED_CONTEXT).task_message()
        assert "Patient question" in message
        assert "Draft answer" in message
        assert "Retrieved policy context" in message
        assert "cancellation_rescheduling" in message
        assert GROUNDED_CONTEXT in message


    def test_reports_when_no_lookup_happened(self) -> None:
        assert "no appointment lookup" in make_session(GROUNDED_CONTEXT).task_message()




class TestRoleMarkers:
    def test_each_system_message_carries_exactly_one_marker(self) -> None:
        assert MARKER_REVIEWER in REVIEWER_SYSTEM_MESSAGE
        assert MARKER_EDITOR not in REVIEWER_SYSTEM_MESSAGE
        assert MARKER_EDITOR in EDITOR_SYSTEM_MESSAGE
        assert MARKER_REVIEWER not in EDITOR_SYSTEM_MESSAGE




@pytest.mark.requires_autogen
class TestRealAutogenTeam:
    """Exercises the actual RoundRobinGroupChat, not just the verdict logic."""


    async def test_approves_a_grounded_draft_unchanged(self) -> None:
        pytest.importorskip("autogen_agentchat")
        from agents.review_team import review_draft


        session = make_session(GROUNDED_CONTEXT)
        outcome = await review_draft(session, SETTINGS)


        assert isinstance(outcome.verdict, ReviewVerdict)
        assert outcome.verdict.approved is True
        assert outcome.verdict.final_answer == GROUNDED_CONTEXT
        assert outcome.revised is False
        assert outcome.reviewer_calls == 1
        assert outcome.editor_calls == 1
        # Task message plus one turn each.
        assert len(outcome.messages) >= 3


    async def test_revises_a_draft_with_an_injected_claim(self) -> None:
        pytest.importorskip("autogen_agentchat")
        from agents.review_team import review_draft


        draft = f"{GROUNDED_CONTEXT} {INJECTED_UNSUPPORTED_CLAIM}"
        outcome = await review_draft(make_session(draft), SETTINGS)


        assert outcome.verdict.approved is False
        assert outcome.revised is True
        assert INJECTED_UNSUPPORTED_CLAIM not in outcome.verdict.final_answer
        assert "cancelled free of charge" in outcome.verdict.final_answer


    async def test_the_transcript_names_both_agents(self) -> None:
        pytest.importorskip("autogen_agentchat")
        from agents.review_team import EDITOR_NAME, REVIEWER_NAME, review_draft


        outcome = await review_draft(make_session(GROUNDED_CONTEXT), SETTINGS)
        sources = {message["source"] for message in outcome.messages}
        assert REVIEWER_NAME in sources
        assert EDITOR_NAME in sources


    async def test_the_teams_verdict_matches_the_derived_expectation(self) -> None:
        pytest.importorskip("autogen_agentchat")
        from agents.review_team import review_draft


        session = make_session(f"{GROUNDED_CONTEXT} {INJECTED_UNSUPPORTED_CLAIM}")
        outcome = await review_draft(session, SETTINGS)
        assert outcome.verdict == deterministic_verdict(session)


    async def test_a_missing_role_marker_is_refused(self) -> None:
        pytest.importorskip("autogen_core")
        from agents.review_team import ReviewUnavailableError, build_mock_client


        client = build_mock_client(make_session(GROUNDED_CONTEXT))


        class Bare:
            content = "no marker anywhere in here"


        with pytest.raises(ReviewUnavailableError):
            await client.create([Bare()])




"""Regression tests for defects found in an adversarial review of this codebase.


Each class below pins one specific bug that existed and is now fixed. They are
grouped here rather than scattered through the suite so the fixes are auditable
as a set: every test states what used to happen, so a future change that
reintroduces the defect fails with an explanation rather than a bare assertion.


Nothing here needs the network, a model artefact, crewai or autogen.
"""


from __future__ import annotations


import dataclasses


import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError


from app.config import EXEMPLAR_RECORD_ID, FALLBACK_ANSWER, Settings
from app.models import (
    MAX_INBOUND_TEXT_CHARACTERS,
    AskRequest,
    ReviewVerdict,
    WsClientMessage,
)
from agents.governance import GOVERNANCE_LAYERS, RISK_LEVEL
from agents.guardrails import validate_contact_number
from agents.composition import (
    COMPOSER_FACT_TEMPLATES,
    EXEMPT_ANSWER_SENTENCES,
    MISSING_RECORD_ID_ANSWER,
)
from agents.guardrails import check_groundedness
from agents.memory import SessionMemory, resolve_record_id_from_history
from agents.review_team import (
    MARKER_FINDINGS_CLOSE,
    MARKER_FINDINGS_OPEN,
    ReviewSession,
    deterministic_verdict,
    parse_reviewer_findings,
    reviewer_critique,
    verdict_from_findings,
)
from agents.routing import extract_record_id
from agents.tools import check_appointment_status
from rag.cache import make_cache_key
from rag.grounded_generation import compose_grounded_answer
from rag.retriever import RetrievalResult, RetrievedChunk


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


GROUNDED_SUPPORT = (
    "An appointment can be cancelled free of charge up to 4 hours before the "
    "scheduled start time. Cancelling inside the 4-hour window retains a 25 "
    "percent late-cancellation charge."
)




def make_chunk(document_id: str, topic_title: str, text: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"sentence_based::{document_id}::00{rank}",
        document_id=document_id,
        topic_title=topic_title,
        source_filename=f"{document_id}.md",
        strategy="sentence_based",
        chunk_index=rank,
        text=text,
        similarity=0.9 - (rank / 100),
        distance=0.1 + (rank / 100),
    )




def make_review_session(draft: str) -> ReviewSession:
    return ReviewSession(
        query="How long before my appointment can I cancel?",
        draft=draft,
        context_text=GROUNDED_SUPPORT,
        support_text=GROUNDED_SUPPORT,
        source_ids=["cancellation_rescheduling"],
        retrieval_grounded=True,
        minimum_overlap=0.6,
        exempt=EXEMPT_ANSWER_SENTENCES,
    )




# --------------------------------------------------------------------------- #
# C1 - the assistant's own guidance used to poison session memory
# --------------------------------------------------------------------------- #




class TestExemplarIdCannotPoisonMemory:
    """The fallback answer used to name ``APT-1007``, which is a real record.


    That string is returned as the answer and stored in session history, and
    ``resolve_record_id_from_history`` scans history newest-first. So a patient
    who asked a bare follow-up got the "which appointment?" reply, and the *next*
    turn resolved ``APT-1007`` out of the assistant's own sentence and disclosed
    a record they never named.
    """


    def test_the_exemplar_id_is_not_resolvable(self) -> None:
        assert extract_record_id(EXEMPLAR_RECORD_ID) is None


    def test_the_missing_id_answer_names_no_real_record(self) -> None:
        assert extract_record_id(MISSING_RECORD_ID_ANSWER) is None


    def test_the_not_found_message_names_no_real_record(self) -> None:
        payload = check_appointment_status("APT-9999")
        assert payload["found"] is False
        assert extract_record_id(str(payload["message"])) is None


    def test_the_missing_id_answer_in_history_resolves_nothing(self) -> None:
        history = [
            HumanMessage(content="And what is its current status?"),
            AIMessage(content=MISSING_RECORD_ID_ANSWER),
        ]
        assert resolve_record_id_from_history(history) is None


    def test_a_genuine_ai_echo_still_resolves(self) -> None:
        """Guards against over-fixing.


        Recovering an id from a real answer is wanted behaviour: it is what makes
        "and is it escalated?" work. Only the *exemplar* must be unresolvable.
        """
        history = [AIMessage(content="Appointment APT-1007 is currently Scheduled.")]
        assert resolve_record_id_from_history(history) == "APT-1007"




# --------------------------------------------------------------------------- #
# M1 / M3 - inbound bounds belong in the schema
# --------------------------------------------------------------------------- #




class TestInboundRequestBounds:
    """``query`` was unbounded and ``session_id`` accepted whitespace.


    Unbounded ``query`` meant a multi-megabyte body was parsed, masked and
    scanned by nine injection regexes *before* the budget cap was consulted, so
    the cap could not protect the work it gated. Whitespace ``session_id`` passed
    ``min_length=1`` and then raised ``ValueError`` inside ``SessionMemory``,
    surfacing as an unhandled HTTP 500.
    """


    def test_a_blank_session_id_is_rejected_by_the_schema(self) -> None:
        with pytest.raises(ValidationError):
            AskRequest(query="What is the cancellation window?", session_id="   ")


    def test_a_blank_query_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AskRequest(query="   ")


    def test_an_absurd_query_is_refused_before_any_application_code(self) -> None:
        with pytest.raises(ValidationError):
            AskRequest(query="x" * (MAX_INBOUND_TEXT_CHARACTERS + 1))


    def test_an_oversized_but_plausible_query_still_reaches_the_budget_cap(self) -> None:
        """The 413 demonstration must stay reachable.


        A request over ``MAX_REQUEST_CHARACTERS`` but under the schema ceiling
        has to arrive at the service and be rejected with a structured
        ``BudgetRejection``, not swallowed as a 422.
        """
        request = AskRequest(query="x" * 3240)
        assert len(request.query) == 3240


    def test_the_websocket_frame_carries_the_same_ceiling(self) -> None:
        assert WsClientMessage(query="x " * 3000).query is not None
        with pytest.raises(ValidationError):
            WsClientMessage(query="x" * (MAX_INBOUND_TEXT_CHARACTERS + 1))




# --------------------------------------------------------------------------- #
# M7 - the cache key ignored the embedding backend
# --------------------------------------------------------------------------- #




class TestCacheKeyIncludesTheEmbedder:
    """A cached answer is only valid for the embedding space it came from.


    With ``SIMILARITY_THRESHOLD`` pinned, switching ``EMBEDDING_BACKEND``
    produced identical keys, so answers retrieved under one embedder were served
    for another.
    """


    BASE = {
        "collection_name": "practo_kb_sentence",
        "top_k": 3,
        "threshold": 0.42,
        "kb_version": 1,
    }


    def test_a_different_embedder_is_a_different_key(self) -> None:
        left = make_cache_key("q", **self.BASE, embedder="sentence_transformers:minilm")
        right = make_cache_key("q", **self.BASE, embedder="deterministic_hash:256")
        assert left != right


    def test_the_embedder_component_is_optional(self) -> None:
        """Kept defaulted so existing callers and tests are unaffected."""
        assert make_cache_key("q", **self.BASE) == make_cache_key(
            "q", **self.BASE, embedder=""
        )




# --------------------------------------------------------------------------- #
# M2 - session memory was unbounded
# --------------------------------------------------------------------------- #




class TestSessionMemoryIsBounded:
    """``session_id`` is client-supplied and history outlives its socket.


    An unbounded registry let any caller grow process memory without limit by
    sending a fresh id each turn, while the Runtime governance layer claimed
    "bounded cost".
    """


    def test_the_oldest_conversation_is_evicted(self) -> None:
        memory = SessionMemory(max_sessions=2)
        memory.get_history("a")
        memory.get_history("b")
        memory.get_history("c")


        assert memory.sessions() == ["b", "c"]
        assert memory.evictions == 1
        assert not memory.has_session("a")


    def test_touching_a_conversation_protects_it(self) -> None:
        memory = SessionMemory(max_sessions=2)
        memory.get_history("a")
        memory.get_history("b")
        memory.get_history("a")  # 'a' is now most-recently-used
        memory.get_history("c")


        assert memory.has_session("a")
        assert not memory.has_session("b")


    def test_a_non_positive_cap_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            SessionMemory(max_sessions=0)




# --------------------------------------------------------------------------- #
# M5 - multi-document answers were misattributed
# --------------------------------------------------------------------------- #




class TestAnswerAttributionPerDocument:
    """The frame named ``chunks[0]`` while quoting from every retrieved chunk.


    On a genuinely two-document question the prose credited one document while
    ``sources`` correctly listed two.
    """


    def _two_document_result(self) -> RetrievalResult:
        return RetrievalResult(
            query="cardiology consultation cost and home visit surcharge",
            collection_name="practo_kb_sentence",
            top_k=3,
            chunks=(
                make_chunk(
                    "consultation_fees",
                    "Consultation Fee Structure by Specialty",
                    "A cardiology consultation costs 1500 INR.",
                    0,
                ),
                make_chunk(
                    "home_visits",
                    "Home Visit Eligibility",
                    "A home visit adds a 400 INR surcharge.",
                    1,
                ),
            ),
        )


    def test_each_document_gets_its_own_frame(self) -> None:
        answer = compose_grounded_answer(
            "cardiology consultation cost and home visit surcharge",
            self._two_document_result(),
        )
        assert "Consultation Fee Structure by Specialty" in answer
        assert "Home Visit Eligibility" in answer
        assert answer.count("According to Practo's") == 2


    def test_a_sentence_is_credited_to_its_own_document(self) -> None:
        answer = compose_grounded_answer(
            "cardiology consultation cost and home visit surcharge",
            self._two_document_result(),
        )
        # The surcharge sentence must sit under the home-visit heading, not the
        # fee heading, which is the misattribution that used to happen.
        home_frame = answer.index("According to Practo's Home Visit Eligibility")
        assert answer.index("400 INR surcharge") > home_frame


    def test_a_single_document_answer_keeps_one_frame(self) -> None:
        result = RetrievalResult(
            query="cancel window",
            collection_name="practo_kb_sentence",
            top_k=3,
            chunks=(
                make_chunk(
                    "cancellation_rescheduling",
                    "Cancellation and Rescheduling Window",
                    GROUNDED_SUPPORT,
                    0,
                ),
            ),
        )
        answer = compose_grounded_answer("cancel free of charge window", result)
        assert answer.count("According to Practo's") == 1




# --------------------------------------------------------------------------- #
# C4 - the groundedness support corpus was too wide
# --------------------------------------------------------------------------- #




class TestFixedAnswersAreExemptByIdentity:
    """The fixed control answers used to be poured into the support corpus.


    That handed *every* request a free vocabulary ("know", "available",
    "knowledge", "base", "specific", "share"), so any drafted sentence reusing
    those words scored as supported regardless of provenance. They are now
    exempted by identity instead, which keeps them passing without widening the
    check for everything else.
    """


    def test_the_fallback_is_exempt(self) -> None:
        assert FALLBACK_ANSWER in EXEMPT_ANSWER_SENTENCES


    def test_the_fixed_answers_are_no_longer_in_the_support_corpus(self) -> None:
        joined = " ".join(COMPOSER_FACT_TEMPLATES)
        assert FALLBACK_ANSWER not in joined
        assert "knowledge base" not in joined
        assert MISSING_RECORD_ID_ANSWER not in joined


    def test_the_refusal_passes_the_check_via_the_exemption(
        self, test_settings: Settings
    ) -> None:
        report = check_groundedness(
            FALLBACK_ANSWER,
            "entirely unrelated reference text about orthopedic surgery",
            retrieval_grounded=True,
            exempt=EXEMPT_ANSWER_SENTENCES,
            settings=test_settings,
        )
        assert report.grounded is True


    def test_without_the_exemption_the_same_refusal_would_be_flagged(
        self, test_settings: Settings
    ) -> None:
        """Proves the exemption is doing the work, not a wide support corpus."""
        report = check_groundedness(
            FALLBACK_ANSWER,
            "entirely unrelated reference text about orthopedic surgery",
            retrieval_grounded=True,
            settings=test_settings,
        )
        assert report.grounded is False


    def test_a_fabricated_claim_is_still_caught(self, test_settings: Settings) -> None:
        report = check_groundedness(
            "Practo upgrades every patient to a lifetime platinum membership.",
            GROUNDED_SUPPORT,
            retrieval_grounded=True,
            exempt=EXEMPT_ANSWER_SENTENCES,
            settings=test_settings,
        )
        assert report.grounded is False




# --------------------------------------------------------------------------- #
# M4 - the reviewer's critique had no causal effect
# --------------------------------------------------------------------------- #




class TestTheEditorActsOnTheReviewersFindings:
    """The editor used to re-derive the verdict from the session object.


    The reviewer's turn was generated, appended to the transcript, and read by
    nothing - renaming it "reviewer" was the only thing that made it one. The
    editor now removes exactly the sentences the reviewer named.
    """


    def test_the_critique_carries_a_machine_readable_block(self) -> None:
        critique = reviewer_critique(make_review_session(GROUNDED_SUPPORT))
        assert MARKER_FINDINGS_OPEN in critique
        assert MARKER_FINDINGS_CLOSE in critique
        # The human-readable prose is unchanged.
        assert "FINDING: none" in critique


    def test_the_block_round_trips(self) -> None:
        claim = "Practo upgrades every patient to a lifetime platinum membership."
        session = make_review_session(f"{GROUNDED_SUPPORT} {claim}")
        parsed = parse_reviewer_findings(reviewer_critique(session))


        assert parsed is not None
        retrieval_failed, named = parsed
        assert retrieval_failed is False
        assert named == [claim]


    def test_a_missing_block_is_unparseable(self) -> None:
        """The editor uses this to fail closed rather than invent a verdict."""
        assert parse_reviewer_findings("some prose with no findings block") is None


    def test_retrieval_failure_is_signalled_in_the_block(self) -> None:
        session = dataclasses.replace(
            make_review_session(GROUNDED_SUPPORT), retrieval_grounded=False
        )
        parsed = parse_reviewer_findings(reviewer_critique(session))
        assert parsed == (True, [])


    def test_the_editor_follows_a_doctored_critique(self) -> None:
        """The decisive test that the critique is causal.


        The named sentence is perfectly well supported. If the editor were still
        recomputing from the session it would approve the draft unchanged; it
        strips the sentence instead, because that is what the reviewer said.
        """
        first = (
            "An appointment can be cancelled free of charge up to 4 hours before "
            "the scheduled start time."
        )
        second = (
            "Cancelling inside the 4-hour window retains a 25 percent "
            "late-cancellation charge."
        )
        session = make_review_session(f"{first} {second}")


        verdict = verdict_from_findings(session, False, [second])


        assert verdict.approved is False
        assert second not in verdict.final_answer
        assert first in verdict.final_answer


    def test_the_derived_expectation_still_matches(self) -> None:
        """``deterministic_verdict`` remains the independent cross-check."""
        claim = "Practo upgrades every patient to a lifetime platinum membership."
        session = make_review_session(f"{GROUNDED_SUPPORT} {claim}")
        parsed = parse_reviewer_findings(reviewer_critique(session))
        assert parsed is not None


        assert verdict_from_findings(session, *parsed) == deterministic_verdict(session)




# --------------------------------------------------------------------------- #
# Contact-number validation (distinct from masking)
# --------------------------------------------------------------------------- #




class TestContactNumberValidation:
    """The brief asks for phone **validation** as well as masking.


    Only masking existed. Masking answers "is there a number hidden in this free
    text"; validation answers "is this field a well-formed contact number",
    which is what a booking or call-back flow needs before accepting one.
    """


    @pytest.mark.parametrize(
        "value",
        [
            "9876543210",
            "+91 98765 43210",
            "+919876543210",
            "+91-9876543210",
            "0091 9876543210",
            "09876543210",
            "98765-43210",
            "6123456789",
        ],
    )
    def test_well_formed_numbers_are_accepted(self, value: str) -> None:
        check = validate_contact_number(value)
        assert check.valid is True
        assert check.normalised is not None
        assert len(check.normalised) == 10
        assert check.normalised[0] in "6789"


    @pytest.mark.parametrize(
        "value,because",
        [
            ("", "empty"),
            ("   ", "whitespace only"),
            ("98765", "too short"),
            ("98765432100", "too long"),
            ("1234567890", "does not start 6-9"),
            ("5876543210", "does not start 6-9"),
            ("not a number", "no digits"),
        ],
    )
    def test_malformed_numbers_are_rejected(self, value: str, because: str) -> None:
        check = validate_contact_number(value)
        assert check.valid is False, because
        assert check.normalised is None
        assert check.reason


    def test_the_loggable_form_carries_no_digits_of_the_number(self) -> None:
        """``as_dict`` is the shape that reaches a log line."""
        payload = validate_contact_number("+91 98765 43210").as_dict()
        assert "normalised" not in payload
        assert "9876543210" not in str(payload)




# --------------------------------------------------------------------------- #
# Contract names the brief specifies literally
# --------------------------------------------------------------------------- #




class TestBriefContractNames:
    """Two names did not match the brief.


    The review verdict field must be ``reason`` as the brief specifies,
    and two of the four governance layers were named ``model_orchestration`` and
    ``infrastructure_data`` rather than ``scope`` and ``cache``.
    """


    def test_the_verdict_uses_the_field_name_the_brief_specifies(self) -> None:
        verdict = ReviewVerdict(
            approved=True, final_answer="answer", reason="because"
        )
        payload = verdict.model_dump()
        assert set(payload) == {"approved", "final_answer", "reason"}


    def test_reason_remains_readable_as_an_alias(self) -> None:
        verdict = ReviewVerdict(approved=False, final_answer="answer", reason="because")
        assert verdict.reason == "because"
        assert verdict.reasoning == "because"
        assert "reason" in ReviewVerdict.model_fields


    def test_the_four_governance_layers_are_the_ones_the_brief_names(self) -> None:
        assert set(GOVERNANCE_LAYERS) == {"application", "scope", "runtime", "cache"}


    def test_each_layer_declares_a_principle_controls_and_enforcement(self) -> None:
        for name, layer in GOVERNANCE_LAYERS.items():
            assert layer["principle"], name
            assert layer["controls"], name
            assert layer["enforced_in"], name


    def test_the_scope_layer_carries_the_risk_classification(self) -> None:
        assert RISK_LEVEL in ("Low", "Medium", "High")
        assert RISK_LEVEL in GOVERNANCE_LAYERS["scope"]["principle"]


    def test_the_runtime_layer_covers_monitoring_and_cost(self) -> None:
        controls = " ".join(GOVERNANCE_LAYERS["runtime"]["controls"]).lower()
        assert "duration_ms" in controls
        assert "cost cap" in controls


    def test_the_cache_layer_documents_normalised_keys(self) -> None:
        controls = " ".join(GOVERNANCE_LAYERS["cache"]["controls"]).lower()
        assert "normalised" in controls
        assert "invalidation" in controls or "invalidate" in controls




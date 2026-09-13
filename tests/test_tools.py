"""Task 6 - the lookup tool, the escalation score, and schema-based dispatch."""


from __future__ import annotations


import pytest
from pydantic import BaseModel


from app.config import TOOL_APPOINTMENT_LOOKUP, TOOL_POLICY_LOOKUP
from agents.tools import (
    AGING_CAP_DAYS,
    ARG_SCHEMA_ROUTING,
    ESCALATION_PERCENTILE,
    WEIGHT_AGING,
    WEIGHT_FOLLOW_UP,
    AppointmentLookupArgs,
    PolicyLookupArgs,
    ToolDispatchError,
    check_appointment_status,
    classify_tool,
    escalation_components,
    escalation_distribution,
    escalation_score,
    escalation_threshold,
)
from app.models import AppointmentResult
from dataset import APPOINTMENTS


class TestEscalationScore:
    def test_weights_sum_to_one(self) -> None:
        assert WEIGHT_FOLLOW_UP + WEIGHT_AGING == pytest.approx(1.0)


    def test_score_stays_inside_the_unit_interval(self) -> None:
        for record in APPOINTMENTS:
            assert 0.0 <= escalation_score(record) <= 1.0


    def test_a_brand_new_record_without_follow_up_scores_zero(self) -> None:
        record = {"days_since_created": 0, "follow_up_required": False}
        assert escalation_score(record) == pytest.approx(0.0)


    def test_the_oldest_follow_up_record_scores_one(self) -> None:
        record = {"days_since_created": AGING_CAP_DAYS, "follow_up_required": True}
        assert escalation_score(record) == pytest.approx(1.0)


    def test_aging_is_capped(self) -> None:
        capped = {"days_since_created": AGING_CAP_DAYS, "follow_up_required": False}
        beyond = {"days_since_created": AGING_CAP_DAYS * 5, "follow_up_required": False}
        assert escalation_score(capped) == escalation_score(beyond)


    def test_components_reproduce_the_score(self) -> None:
        for record in APPOINTMENTS[:10]:
            components = escalation_components(record)
            total = components["follow_up_component"] + components["aging_component"]
            assert escalation_score(record) == pytest.approx(round(total, 4))


    def test_the_two_signals_genuinely_compete(self) -> None:
        # Same flag, different age -> different score. A boolean OR could not
        # distinguish these two records at all.
        fresh = {"days_since_created": 1, "follow_up_required": True}
        stale = {"days_since_created": 29, "follow_up_required": True}
        assert escalation_score(fresh) < escalation_score(stale)


        # And age alone can outrank a fresh follow-up.
        very_old = {"days_since_created": 30, "follow_up_required": False}
        assert escalation_score(very_old) > escalation_score(fresh)


    def test_score_is_monotonic_in_age(self) -> None:
        scores = [
            escalation_score({"days_since_created": day, "follow_up_required": False})
            for day in range(0, AGING_CAP_DAYS + 1)
        ]
        assert scores == sorted(scores)


class TestEscalationThreshold:
    def test_threshold_is_a_valid_probability(self) -> None:
        assert 0.0 <= escalation_threshold() <= 1.0


    def test_threshold_is_derived_from_the_dataset(self) -> None:
        scores = sorted(escalation_score(record) for record in APPOINTMENTS)
        assert escalation_threshold() in [round(score, 4) for score in scores]


    def test_threshold_is_the_nearest_rank_percentile(self) -> None:
        import math


        scores = sorted(escalation_score(record) for record in APPOINTMENTS)
        rank = max(1, math.ceil(ESCALATION_PERCENTILE * len(scores)))
        assert escalation_threshold() == pytest.approx(round(scores[rank - 1], 4))


    def test_distribution_report_is_self_consistent(self) -> None:
        distribution = escalation_distribution()
        assert distribution["records"] == len(APPOINTMENTS)
        assert distribution["min"] <= distribution["median"] <= distribution["max"]
        assert distribution["threshold"] == escalation_threshold()
        assert 0 < distribution["records_at_or_above_threshold"] <= len(APPOINTMENTS)


    def test_the_score_has_not_collapsed_into_a_boolean(self) -> None:
        distribution = escalation_distribution()
        # If either list were empty the score would be behaving like
        # `follow_up_required OR is_old`, which the brief rules out.
        assert (
            distribution["escalated_without_follow_up_flag"]
            or distribution["follow_up_flag_but_not_escalated"]
        ), "the escalation score is not blending its two signals"


class TestCheckAppointmentStatus:
    def test_returns_the_record_for_a_known_id(self) -> None:
        record = APPOINTMENTS[6]
        payload = check_appointment_status(record["record_id"])
        assert payload["found"] is True
        assert payload["status"] == record["status"]
        assert payload["consultation_fee_inr"] == record["consultation_fee_inr"]
        assert payload["escalation_score"] == escalation_score(record)
        assert payload["escalation_threshold"] == escalation_threshold()
        assert payload["components"] is not None


    def test_escalation_recommendation_follows_the_threshold(self) -> None:
        threshold = escalation_threshold()
        for record in APPOINTMENTS:
            payload = check_appointment_status(record["record_id"])
            assert payload["escalation_recommended"] == (
                payload["escalation_score"] >= threshold
            )


    def test_matching_is_case_insensitive(self) -> None:
        record = APPOINTMENTS[2]
        assert (
            check_appointment_status(record["record_id"].lower())["record_id"]
            == record["record_id"]
        )


    @pytest.mark.parametrize("bad_id", ["APT-9999", "", "   ", "not-an-id"])
    def test_unknown_id_returns_a_safe_structured_result(self, bad_id: str) -> None:
        payload = check_appointment_status(bad_id)
        assert payload["found"] is False
        assert payload["status"] is None
        assert payload["escalation_score"] is None
        assert payload["escalation_recommended"] is False
        assert payload["message"]


    def test_payload_validates_against_the_pydantic_model(self) -> None:
        for record in APPOINTMENTS[:5]:
            AppointmentResult.model_validate(check_appointment_status(record["record_id"]))
        AppointmentResult.model_validate(check_appointment_status("APT-9999"))


class TestSchemaBasedDispatch:
    """Pitfall 2: dispatch must key off the argument schema, not the tool name."""


    def test_routing_table_covers_both_tools(self) -> None:
        assert set(ARG_SCHEMA_ROUTING.values()) == {
            TOOL_POLICY_LOOKUP,
            TOOL_APPOINTMENT_LOOKUP,
        }


    def test_policy_schema_routes_to_the_policy_tool(self) -> None:
        assert (
            ARG_SCHEMA_ROUTING[frozenset(PolicyLookupArgs.model_fields)]
            == TOOL_POLICY_LOOKUP
        )


    def test_appointment_schema_routes_to_the_appointment_tool(self) -> None:
        assert (
            ARG_SCHEMA_ROUTING[frozenset(AppointmentLookupArgs.model_fields)]
            == TOOL_APPOINTMENT_LOOKUP
        )


    def test_a_tool_named_rag_lookup_is_still_classified_correctly(self) -> None:
        """The exact trap the brief warns about.


        A name-substring test for ``"lookup"`` would misfile this tool as the
        appointment lookup. Classifying by ``args_schema`` cannot.
        """


        class RagLookupTool:
            name = "rag_lookup"
            args_schema = PolicyLookupArgs


        assert classify_tool(RagLookupTool()) == TOOL_POLICY_LOOKUP
        assert "lookup" in RagLookupTool.name  # the trap is genuinely present


    def test_a_misleadingly_named_appointment_tool_is_also_correct(self) -> None:
        class WeirdlyNamedTool:
            name = "policy_query_helper"
            args_schema = AppointmentLookupArgs


        assert classify_tool(WeirdlyNamedTool()) == TOOL_APPOINTMENT_LOOKUP


    def test_a_tool_without_a_schema_is_rejected(self) -> None:
        class NoSchemaTool:
            name = "mystery"
            args_schema = None


        with pytest.raises(ToolDispatchError):
            classify_tool(NoSchemaTool())


    def test_an_unrecognised_schema_is_rejected(self) -> None:
        class OtherArgs(BaseModel):
            something_else: str


        class OtherTool:
            name = "other"
            args_schema = OtherArgs


        with pytest.raises(ToolDispatchError):
            classify_tool(OtherTool())



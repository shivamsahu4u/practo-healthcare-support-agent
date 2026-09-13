"""Task 1 - dataset generation, structural thresholds and validation."""


from __future__ import annotations


from collections import Counter


import pytest


from app.config import CATEGORIES, STATUSES
from dataset import (
    DESIGN,
    FEE_BANDS_INR,
    FOLLOW_UP_BAND,
    GLOBAL_FEE_RANGE_INR,
    MAX_DAYS_SINCE_CREATED,
    MIN_RECORDS_PER_CATEGORY,
    APPOINTMENTS,
    DatasetValidationError,
    allocate_quota,
    generate_appointments,
    get_appointment,
    summarise,
    validate_appointments,
)




def test_generation_is_deterministic() -> None:
    assert generate_appointments() == generate_appointments()
    assert generate_appointments() == APPOINTMENTS




def test_at_least_forty_records() -> None:
    assert len(APPOINTMENTS) >= 40




def test_record_ids_are_unique_and_sequential() -> None:
    ids = [record["record_id"] for record in APPOINTMENTS]
    assert len(set(ids)) == len(ids)
    assert ids[0] == "APT-1001"
    assert ids[-1] == f"APT-{1000 + len(APPOINTMENTS)}"




def test_every_category_has_at_least_three_records() -> None:
    counts = Counter(record["category"] for record in APPOINTMENTS)
    for category in CATEGORIES:
        assert counts[category] >= MIN_RECORDS_PER_CATEGORY, category




def test_every_status_appears_at_least_once() -> None:
    counts = Counter(record["status"] for record in APPOINTMENTS)
    for status in STATUSES:
        assert counts[status] >= 1, status




def test_follow_up_share_lands_in_the_required_band() -> None:
    share = summarise()["follow_up_required_percentage"]
    low, high = FOLLOW_UP_BAND
    assert low <= share <= high




def test_field_types_and_ranges() -> None:
    global_low, global_high = GLOBAL_FEE_RANGE_INR
    for record in APPOINTMENTS:
        assert isinstance(record["days_since_created"], int)
        assert 0 <= record["days_since_created"] <= MAX_DAYS_SINCE_CREATED
        assert isinstance(record["follow_up_required"], bool)
        fee = record["consultation_fee_inr"]
        assert isinstance(fee, int)
        assert global_low <= fee <= global_high
        band_low, band_high = FEE_BANDS_INR[record["category"]]
        assert band_low <= fee <= band_high
        assert fee % DESIGN.fee_step_inr == 0




def test_lookup_is_case_insensitive() -> None:
    record = APPOINTMENTS[3]
    assert get_appointment(record["record_id"]) == record
    assert get_appointment(record["record_id"].lower()) == record
    assert get_appointment(f"  {record['record_id']}  ") == record
    assert get_appointment("APT-9999") is None
    assert get_appointment(None) is None  # type: ignore[arg-type]




class TestAllocateQuota:
    def test_sums_to_total(self) -> None:
        quota = allocate_quota(45, {"a": 4, "b": 3, "c": 3, "d": 3, "e": 2})
        assert sum(quota.values()) == 45
        assert quota == {"a": 12, "b": 9, "c": 9, "d": 9, "e": 6}


    def test_respects_a_minimum(self) -> None:
        quota = allocate_quota(10, {"a": 100, "b": 1}, minimum=2)
        assert sum(quota.values()) == 10
        assert quota["b"] >= 2


    def test_is_deterministic_under_ties(self) -> None:
        weights = {"a": 1, "b": 1, "c": 1}
        assert allocate_quota(10, weights) == allocate_quota(10, weights)


    @pytest.mark.parametrize(
        "total,weights,minimum",
        [
            (0, {"a": 1}, 0),
            (5, {}, 0),
            (5, {"a": 0}, 0),
            (5, {"a": 1, "b": 1, "c": 1}, 2),
        ],
    )
    def test_rejects_impossible_inputs(
        self, total: int, weights: dict[str, int], minimum: int
    ) -> None:
        with pytest.raises(ValueError):
            allocate_quota(total, weights, minimum=minimum)




class TestValidation:
    def test_accepts_the_generated_dataset(self) -> None:
        validate_appointments(APPOINTMENTS)


    def test_rejects_too_few_records(self) -> None:
        with pytest.raises(DatasetValidationError) as excinfo:
            validate_appointments(APPOINTMENTS[:10])
        assert any(">= 40" in failure for failure in excinfo.value.failures)


    def test_rejects_a_missing_field(self) -> None:
        broken = [dict(record) for record in APPOINTMENTS]
        broken[0].pop("consultation_fee_inr")
        with pytest.raises(DatasetValidationError) as excinfo:
            validate_appointments(broken)
        assert any("missing field" in failure for failure in excinfo.value.failures)


    def test_rejects_an_out_of_range_day_count(self) -> None:
        broken = [dict(record) for record in APPOINTMENTS]
        broken[0]["days_since_created"] = 99
        with pytest.raises(DatasetValidationError) as excinfo:
            validate_appointments(broken)
        assert any("days_since_created" in failure for failure in excinfo.value.failures)


    def test_rejects_a_duplicate_record_id(self) -> None:
        broken = [dict(record) for record in APPOINTMENTS]
        broken[1]["record_id"] = broken[0]["record_id"]
        with pytest.raises(DatasetValidationError) as excinfo:
            validate_appointments(broken)
        assert any("duplicate" in failure for failure in excinfo.value.failures)


    def test_rejects_a_follow_up_share_outside_the_band(self) -> None:
        broken = [dict(record) for record in APPOINTMENTS]
        for record in broken:
            record["follow_up_required"] = True
        with pytest.raises(DatasetValidationError) as excinfo:
            validate_appointments(broken)
        assert any("follow_up_required share" in f for f in excinfo.value.failures)


    def test_rejects_a_fee_outside_its_band(self) -> None:
        broken = [dict(record) for record in APPOINTMENTS]
        broken[0]["category"] = "General Medicine"
        broken[0]["consultation_fee_inr"] = 2500
        with pytest.raises(DatasetValidationError) as excinfo:
            validate_appointments(broken)
        assert any("band" in failure for failure in excinfo.value.failures)


    def test_rejects_a_boolean_masquerading_as_a_day_count(self) -> None:
        broken = [dict(record) for record in APPOINTMENTS]
        broken[0]["days_since_created"] = True
        with pytest.raises(DatasetValidationError):
            validate_appointments(broken)


    def test_collects_every_failure_not_just_the_first(self) -> None:
        broken = [dict(record) for record in APPOINTMENTS]
        broken[0]["days_since_created"] = 99
        broken[1]["record_id"] = broken[0]["record_id"]
        with pytest.raises(DatasetValidationError) as excinfo:
            validate_appointments(broken)
        assert len(excinfo.value.failures) >= 2




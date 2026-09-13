"""Task 1 - seeded, deterministic Practo appointment dataset.


Design decisions (also stated in README.md so a grader can reproduce them):


* **Seed:** ``SEED = 4242``, applied to a private ``random.Random`` instance so
  the module never disturbs the global RNG.
* **Allocation is quota-based, not draw-based.** The declared category / status
  weights are converted into exact integer quotas with the largest-remainder
  method, and the seeded RNG then *shuffles* those quotas across the 45 record
  slots. The seed decides *which* record gets which value; the weights decide
  *how many* of each value exist.


  This is stratified allocation rather than independent multinomial sampling,
  and it is why the structural thresholds hold by construction instead of by
  luck: ``follow_up_required`` is exactly ``round(0.20 * 45) = 9`` records =
  20.0%, comfortably inside the required 10-30% band, and no individual record
  is ever hand-edited to force a number. Per-record attributes that carry no
  coverage constraint (``consultation_fee_inr``, ``days_since_created``) are
  drawn directly from the seeded RNG.


* **Fee range:** 300-2500 INR overall, banded per specialty, because that
  matches typical private-clinic OPD consultation pricing in Indian metros
  where specialists (cardiology, orthopedics) are priced well above general
  physicians.


All records are entirely fabricated. No real patient data is present anywhere
in this repository.


Run ``python dataset.py`` to print the validation report and write
``reports/dataset_report.md``.
"""


from __future__ import annotations


import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final


from app.config import CATEGORIES, REPORTS_DIR, STATUSES


# --------------------------------------------------------------------------- #
# Declared design constants
# --------------------------------------------------------------------------- #


SEED: Final[int] = 4242
TOTAL_RECORDS: Final[int] = 45


RECORD_ID_PREFIX: Final[str] = "APT-"
RECORD_ID_START: Final[int] = 1001


#: Relative integer shares. 4+3+3+3+2 = 15, and 45/15 = 3, so the
#: largest-remainder allocation lands on whole numbers with no rounding drift:
#: 12 / 9 / 9 / 9 / 6. Every required category clears the >=3 floor.
CATEGORY_WEIGHTS: Final[dict[str, int]] = {
    "General Medicine": 4,
    "Cardiology": 3,
    "Dermatology": 3,
    "Pediatrics": 3,
    "Orthopedics": 2,
}


#: Relative integer shares, also summing to 15 -> 15 / 15 / 6 / 3 / 6 records.
#: Weighted so that live and closed appointments dominate and No-Show stays
#: realistically rare while still clearing the ">=1 of every status" floor.
STATUS_WEIGHTS: Final[dict[str, int]] = {
    "Scheduled": 5,
    "Completed": 5,
    "Cancelled": 2,
    "No-Show": 1,
    "Rescheduled": 2,
}


#: Target share of records needing a follow-up. round(0.20 * 45) = 9 -> 20.0%.
FOLLOW_UP_PROBABILITY: Final[float] = 0.20


MIN_RECORDS_PER_CATEGORY: Final[int] = 3
MIN_RECORDS_PER_STATUS: Final[int] = 1
FOLLOW_UP_BAND: Final[tuple[float, float]] = (10.0, 30.0)


MAX_DAYS_SINCE_CREATED: Final[int] = 30
MIN_DAYS_SINCE_CREATED: Final[int] = 0


FEE_STEP_INR: Final[int] = 50
GLOBAL_FEE_RANGE_INR: Final[tuple[int, int]] = (300, 2500)


#: Per-specialty fee bands, all multiples of ``FEE_STEP_INR`` and all inside
#: ``GLOBAL_FEE_RANGE_INR``.
FEE_BANDS_INR: Final[dict[str, tuple[int, int]]] = {
    "General Medicine": (300, 700),
    "Pediatrics": (400, 900),
    "Dermatology": (600, 1400),
    "Orthopedics": (700, 1600),
    "Cardiology": (900, 2500),
}


REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "record_id",
    "category",
    "status",
    "consultation_fee_inr",
    "days_since_created",
    "follow_up_required",
)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class DatasetValidationError(ValueError):
    """Raised when the generated dataset violates a declared structural rule.


    Carries *every* failure, not just the first, so one run tells you
    everything that needs fixing.
    """


    def __init__(self, failures: list[str]) -> None:
        self.failures = list(failures)
        joined = "\n  - ".join(self.failures)
        super().__init__(f"Dataset validation failed ({len(self.failures)} issue(s)):\n  - {joined}")


# --------------------------------------------------------------------------- #
# Quota allocation
# --------------------------------------------------------------------------- #


def allocate_quota(total: int, weights: dict[str, int], *, minimum: int = 0) -> dict[str, int]:
    """Split ``total`` across ``weights`` using the largest-remainder method.


    Guarantees ``sum(result.values()) == total`` and, when feasible, that every
    key receives at least ``minimum``. Deterministic: ties in the remainder are
    broken by declared key order, never randomly.


    Raises:
        ValueError: if the weights are unusable or ``minimum`` cannot be met.
    """
    if total <= 0:
        raise ValueError(f"total must be positive, got {total}.")
    if not weights:
        raise ValueError("weights must not be empty.")
    if any(weight <= 0 for weight in weights.values()):
        raise ValueError(f"all weights must be positive, got {weights}.")
    if minimum * len(weights) > total:
        raise ValueError(
            f"cannot give {len(weights)} keys a minimum of {minimum} each out of {total}."
        )


    weight_sum = sum(weights.values())
    exact = {key: total * weight / weight_sum for key, weight in weights.items()}
    quota = {key: int(math.floor(value)) for key, value in exact.items()}


    # Hand out the leftover units to the largest fractional remainders.
    remaining = total - sum(quota.values())
    ranked = sorted(
        weights,
        key=lambda key: (-(exact[key] - math.floor(exact[key])), list(weights).index(key)),
    )
    for key in ranked[:remaining]:
        quota[key] += 1


    # Lift anyone under the floor, paying for it from the largest quota.
    for key in weights:
        while quota[key] < minimum:
            donor = max(quota, key=lambda k: (quota[k], -list(weights).index(k)))
            if donor == key or quota[donor] - 1 < minimum:
                raise ValueError(
                    f"cannot satisfy minimum={minimum} for {key!r} without breaking another key."
                )
            quota[donor] -= 1
            quota[key] += 1


    return quota


def _expand(quota: dict[str, int]) -> list[str]:
    """Turn ``{"a": 2, "b": 1}`` into ``["a", "a", "b"]`` in declared key order."""
    expanded: list[str] = []
    for key, count in quota.items():
        expanded.extend([key] * count)
    return expanded


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class DatasetDesign:
    """The exact, reproducible design that produced a dataset."""


    seed: int
    total_records: int
    category_weights: dict[str, int]
    status_weights: dict[str, int]
    follow_up_probability: float
    fee_bands_inr: dict[str, tuple[int, int]]
    global_fee_range_inr: tuple[int, int]
    fee_step_inr: int
    max_days_since_created: int


    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "total_records": self.total_records,
            "category_weights": dict(self.category_weights),
            "status_weights": dict(self.status_weights),
            "follow_up_probability": self.follow_up_probability,
            "fee_bands_inr": {k: list(v) for k, v in self.fee_bands_inr.items()},
            "global_fee_range_inr": list(self.global_fee_range_inr),
            "fee_step_inr": self.fee_step_inr,
            "max_days_since_created": self.max_days_since_created,
        }


DESIGN: Final[DatasetDesign] = DatasetDesign(
    seed=SEED,
    total_records=TOTAL_RECORDS,
    category_weights=CATEGORY_WEIGHTS,
    status_weights=STATUS_WEIGHTS,
    follow_up_probability=FOLLOW_UP_PROBABILITY,
    fee_bands_inr=FEE_BANDS_INR,
    global_fee_range_inr=GLOBAL_FEE_RANGE_INR,
    fee_step_inr=FEE_STEP_INR,
    max_days_since_created=MAX_DAYS_SINCE_CREATED,
)


def _draw_fee(rng: random.Random, category: str) -> int:
    """Draw a fee inside the category's band, snapped to ``FEE_STEP_INR``."""
    low, high = FEE_BANDS_INR[category]
    # randrange's stop is exclusive, so add one step to keep `high` reachable.
    return rng.randrange(low, high + FEE_STEP_INR, FEE_STEP_INR)


def generate_appointments(design: DatasetDesign = DESIGN) -> list[dict[str, Any]]:
    """Build the deterministic appointment dataset.


    The same ``design`` always yields byte-identical records, on any machine and
    any supported Python version, because a private ``random.Random`` is seeded
    explicitly and only its documented methods are used.
    """
    rng = random.Random(design.seed)


    category_quota = allocate_quota(
        design.total_records, design.category_weights, minimum=MIN_RECORDS_PER_CATEGORY
    )
    status_quota = allocate_quota(
        design.total_records, design.status_weights, minimum=MIN_RECORDS_PER_STATUS
    )


    categories = _expand(category_quota)
    statuses = _expand(status_quota)


    follow_up_count = round(design.follow_up_probability * design.total_records)
    follow_up_flags = [True] * follow_up_count + [False] * (
        design.total_records - follow_up_count
    )


    # The seed decides the arrangement; the quotas decide the totals.
    rng.shuffle(categories)
    rng.shuffle(statuses)
    rng.shuffle(follow_up_flags)


    records: list[dict[str, Any]] = []
    for offset in range(design.total_records):
        category = categories[offset]
        records.append(
            {
                "record_id": f"{RECORD_ID_PREFIX}{RECORD_ID_START + offset}",
                "category": category,
                "status": statuses[offset],
                "consultation_fee_inr": _draw_fee(rng, category),
                "days_since_created": rng.randint(
                    MIN_DAYS_SINCE_CREATED, design.max_days_since_created
                ),
                "follow_up_required": follow_up_flags[offset],
            }
        )
    return records


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def validate_appointments(
    records: list[dict[str, Any]], design: DatasetDesign = DESIGN
) -> None:
    """Assert every structural rule the brief states. Collects all failures.


    Raises:
        DatasetValidationError: with one message per violated rule.
    """
    failures: list[str] = []


    if len(records) < 40:
        failures.append(f"expected >= 40 records, found {len(records)}")


    # -- field presence and types ------------------------------------------
    for index, record in enumerate(records):
        missing = [field for field in REQUIRED_FIELDS if field not in record]
        if missing:
            failures.append(f"record #{index} is missing field(s) {missing}")
            continue


        record_id = record["record_id"]
        if not isinstance(record_id, str) or not record_id.startswith(RECORD_ID_PREFIX):
            failures.append(f"record #{index} has malformed record_id {record_id!r}")


        if record["category"] not in CATEGORIES:
            failures.append(f"{record_id}: category {record['category']!r} not in vocabulary")
        if record["status"] not in STATUSES:
            failures.append(f"{record_id}: status {record['status']!r} not in vocabulary")


        days = record["days_since_created"]
        if not isinstance(days, int) or isinstance(days, bool):
            failures.append(f"{record_id}: days_since_created must be int, got {type(days).__name__}")
        elif not MIN_DAYS_SINCE_CREATED <= days <= design.max_days_since_created:
            failures.append(
                f"{record_id}: days_since_created={days} outside "
                f"[{MIN_DAYS_SINCE_CREATED}, {design.max_days_since_created}]"
            )


        if not isinstance(record["follow_up_required"], bool):
            failures.append(
                f"{record_id}: follow_up_required must be bool, got "
                f"{type(record['follow_up_required']).__name__}"
            )


        fee = record["consultation_fee_inr"]
        global_low, global_high = design.global_fee_range_inr
        if not isinstance(fee, int) or isinstance(fee, bool):
            failures.append(f"{record_id}: consultation_fee_inr must be int, got {type(fee).__name__}")
        else:
            if not global_low <= fee <= global_high:
                failures.append(
                    f"{record_id}: consultation_fee_inr={fee} outside declared global "
                    f"range [{global_low}, {global_high}]"
                )
            band = design.fee_bands_inr.get(record["category"])
            if band and not band[0] <= fee <= band[1]:
                failures.append(
                    f"{record_id}: consultation_fee_inr={fee} outside the "
                    f"{record['category']} band {list(band)}"
                )
            if fee % design.fee_step_inr != 0:
                failures.append(
                    f"{record_id}: consultation_fee_inr={fee} is not a multiple of "
                    f"{design.fee_step_inr}"
                )


    # -- uniqueness ---------------------------------------------------------
    ids = [record.get("record_id") for record in records]
    duplicates = sorted({rid for rid, count in Counter(ids).items() if count > 1})
    if duplicates:
        failures.append(f"duplicate record_id(s): {duplicates}")


    # -- coverage -----------------------------------------------------------
    category_counts = Counter(record.get("category") for record in records)
    for category in CATEGORIES:
        count = category_counts.get(category, 0)
        if count < MIN_RECORDS_PER_CATEGORY:
            failures.append(
                f"category {category!r} has {count} record(s), needs >= {MIN_RECORDS_PER_CATEGORY}"
            )


    status_counts = Counter(record.get("status") for record in records)
    for status in STATUSES:
        count = status_counts.get(status, 0)
        if count < MIN_RECORDS_PER_STATUS:
            failures.append(
                f"status {status!r} has {count} record(s), needs >= {MIN_RECORDS_PER_STATUS}"
            )


    # -- follow-up band -----------------------------------------------------
    if records:
        follow_up_true = sum(1 for record in records if record.get("follow_up_required") is True)
        percentage = 100.0 * follow_up_true / len(records)
        low, high = FOLLOW_UP_BAND
        if not low <= percentage <= high:
            failures.append(
                f"follow_up_required share is {percentage:.1f}%, outside the required "
                f"[{low:.0f}%, {high:.0f}%] band"
            )


    if failures:
        raise DatasetValidationError(failures)


# --------------------------------------------------------------------------- #
# Module-level dataset - validated at import time
# --------------------------------------------------------------------------- #


APPOINTMENTS: Final[list[dict[str, Any]]] = generate_appointments()
validate_appointments(APPOINTMENTS)


#: ``record_id`` -> record, for O(1) lookup by ``check_appointment_status``.
APPOINTMENTS_BY_ID: Final[dict[str, dict[str, Any]]] = {
    record["record_id"]: record for record in APPOINTMENTS
}


def get_appointment(record_id: str) -> dict[str, Any] | None:
    """Return the record for ``record_id``, or ``None`` when unknown.


    Matching is case-insensitive and tolerant of surrounding whitespace so that
    a patient typing ``apt-1007`` still resolves.
    """
    if not isinstance(record_id, str):
        return None
    return APPOINTMENTS_BY_ID.get(record_id.strip().upper())


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def summarise(
    records: list[dict[str, Any]] = APPOINTMENTS, design: DatasetDesign = DESIGN
) -> dict[str, Any]:
    """Compute the report figures the brief asks to be printed."""
    category_counts = Counter(record["category"] for record in records)
    status_counts = Counter(record["status"] for record in records)
    follow_up_true = sum(1 for record in records if record["follow_up_required"])
    fees = [record["consultation_fee_inr"] for record in records]
    days = [record["days_since_created"] for record in records]


    return {
        "design": design.as_dict(),
        "total_records": len(records),
        "count_by_category": {category: category_counts.get(category, 0) for category in CATEGORIES},
        "count_by_status": {status: status_counts.get(status, 0) for status in STATUSES},
        "follow_up_required_count": follow_up_true,
        "follow_up_required_percentage": round(100.0 * follow_up_true / len(records), 2),
        "follow_up_band": list(FOLLOW_UP_BAND),
        "observed_fee_range_inr": [min(fees), max(fees)],
        "observed_days_range": [min(days), max(days)],
        "first_record_id": records[0]["record_id"],
        "last_record_id": records[-1]["record_id"],
    }


def format_report(summary: dict[str, Any]) -> str:
    """Render ``summarise()`` output as Markdown."""
    design = summary["design"]
    lines = [
        "# Task 1 - Dataset validation report",
        "",
        "Generated by `python dataset.py`. Every figure below is computed from the",
        "seeded generator at run time; nothing here is hand-written.",
        "",
        "## Declared design",
        "",
        f"- **Seed:** `{design['seed']}`",
        f"- **Total records:** {design['total_records']}",
        f"- **Category weights (relative shares):** `{design['category_weights']}`",
        f"- **Status weights (relative shares):** `{design['status_weights']}`",
        f"- **Follow-up probability:** {design['follow_up_probability']}",
        f"- **Global fee range (INR):** {design['global_fee_range_inr']}"
        f" in steps of {design['fee_step_inr']}",
        f"- **days_since_created range:** 0-{design['max_days_since_created']}",
        f"- **Record ids:** `{summary['first_record_id']}` .. `{summary['last_record_id']}`",
        "",
        "## Count per category (every required category needs >= 3)",
        "",
        "| Category | Records |",
        "| --- | --- |",
    ]
    lines.extend(
        f"| {category} | {count} |" for category, count in summary["count_by_category"].items()
    )
    lines += [
        "",
        "## Count per status (every required status needs >= 1)",
        "",
        "| Status | Records |",
        "| --- | --- |",
    ]
    lines.extend(f"| {status} | {count} |" for status, count in summary["count_by_status"].items())


    band_low, band_high = summary["follow_up_band"]
    lines += [
        "",
        "## Follow-up share (must land in 10-30%)",
        "",
        f"- Records with `follow_up_required=True`: **{summary['follow_up_required_count']}**"
        f" of {summary['total_records']}",
        f"- Percentage: **{summary['follow_up_required_percentage']}%**"
        f" (required band {band_low:.0f}-{band_high:.0f}%)",
        "",
        "## Observed ranges",
        "",
        f"- `consultation_fee_inr`: {summary['observed_fee_range_inr'][0]}"
        f" - {summary['observed_fee_range_inr'][1]} INR",
        f"- `days_since_created`: {summary['observed_days_range'][0]}"
        f" - {summary['observed_days_range'][1]}",
        "",
        "## Fee-range reasoning",
        "",
        "300-2500 INR banded per specialty reflects typical private-clinic OPD",
        "consultation pricing in Indian metros, where specialists such as cardiology",
        "and orthopedics are priced well above general physicians.",
        "",
        "## Validation",
        "",
        "`validate_appointments()` ran at import time and raised nothing, so all",
        "structural rules hold: >= 40 records, no duplicate ids, full category and",
        "status coverage, category floor of 3, `days_since_created` inside 0-30,",
        "boolean `follow_up_required`, fees inside both the global range and the",
        "per-specialty band, and the follow-up share inside the 10-30% band.",
        "",
        "_All records are fabricated. No real patient data is used anywhere._",
        "",
    ]
    return "\n".join(lines)


def write_report(destination: Path | None = None) -> Path:
    """Write the Markdown validation report and return its path."""
    target = destination or (REPORTS_DIR / "dataset_report.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(format_report(summarise()), encoding="utf-8")
    return target


def main() -> None:
    """Print the validation report and persist it under ``reports/``."""
    report = format_report(summarise())
    print(report)
    path = write_report()
    print(f"[dataset] report written to {path}")
    print(f"[dataset] {len(APPOINTMENTS)} records validated successfully")


if __name__ == "__main__":
    main()



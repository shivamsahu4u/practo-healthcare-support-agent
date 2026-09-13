"""Task 6 - the appointment-lookup tool with a designed escalation score,
plus the RAG tool, plus the CrewAI wrappers for both.


**Escalation score design.** A bare ``follow_up_required OR old_record`` boolean
would throw away the difference between a follow-up flagged yesterday and one
flagged a month ago, so the score is a weighted blend of two normalised signals:


    aging_normalised = min(days_since_created, AGING_CAP_DAYS) / AGING_CAP_DAYS


    escalation_score = clamp(
          WEIGHT_FOLLOW_UP * (1 if follow_up_required else 0)
        + WEIGHT_AGING     * aging_normalised,
        0.0, 1.0)


with ``WEIGHT_FOLLOW_UP = 0.45``, ``WEIGHT_AGING = 0.55`` and
``AGING_CAP_DAYS = 30``.


Direction: **older** unresolved records escalate harder. A record created today
contributes nothing from the aging term; one at the 30-day cap contributes the
full 0.55.


The weights are chosen so the two signals genuinely interact rather than one
dominating. A record without a follow-up flag can still reach 0.55, and a
follow-up record starts at 0.45 - so the bands overlap in ``[0.45, 0.55]``, and
the threshold sits inside that overlap. Consequences: a *fresh* follow-up record
is **not** escalated, and a *very old* record with no follow-up flag **is**.
That is the behaviour a boolean OR cannot express.


**Threshold.** ``escalation_threshold()`` computes the 80th percentile
(nearest-rank) of the score distribution over the generated dataset, at run
time. It is not a hard-coded constant, so it stays honest if the dataset design
changes; ``escalation_distribution()`` prints the supporting distribution.


**Error asymmetry: in this domain a false negative is worse than a false
positive.** A missed follow-up is a patient who needed a call back and did not
get one; an unnecessary escalation costs a support lead a few minutes of review.
The two are not symmetric, so the design is deliberately biased toward
over-escalating:


* The **aging term carries the larger weight** (0.55 against 0.45). Age is the
  signal that a case has been sitting unresolved, so a record with no follow-up
  flag still escalates on age alone once it approaches the 30-day cap. A
  boolean ``follow_up_required`` test would miss every one of those - they are
  pure false negatives, and ``escalated_without_follow_up_flag`` lists them.
* The score is **continuous and the threshold is a percentile**, not a fixed
  cut-off, so if the dataset shifts toward older records the escalated set grows
  with it rather than silently under-reporting.
* Escalation is **advisory**: ``escalation_recommended`` routes a record to a
  human support lead. Nothing is auto-closed, auto-refunded or auto-declined on
  the strength of this score, which is what makes erring toward escalation the
  cheap direction.


**The one false negative this design accepts, stated plainly.** A *fresh*
follow-up record - flagged, but created today - scores about 0.47 and does not
clear the threshold. That is a follow-up the system does not escalate, so it is
exactly the error class this domain says to avoid. It is accepted because a
same-day follow-up is not yet actionable: nothing has been missed while the
appointment is still current, and the record escalates on its own within days as
the aging term grows. ``escalation_distribution()`` reports these as
``follow_up_flag_but_not_escalated`` so the set is auditable rather than
hidden. A deployment that disagreed with that trade-off would lower the
percentile - the threshold is one argument, and no other logic depends on its
value.
"""


from __future__ import annotations


import json
import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Final


from pydantic import BaseModel, Field


from app.config import (
    AGENT_LOOKUP,
    AGENT_RETRIEVAL,
    EXEMPLAR_RECORD_ID,
    SETTINGS,
    TOOL_APPOINTMENT_LOOKUP,
    TOOL_POLICY_LOOKUP,
    Settings,
)
from agents.context import CrewRunContext
from dataset import APPOINTMENTS, get_appointment
from rag.grounded_generation import GroundedAnswer, GroundedGenerator


LOGGER: Final = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Escalation design constants
# --------------------------------------------------------------------------- #


WEIGHT_FOLLOW_UP: Final[float] = 0.45
WEIGHT_AGING: Final[float] = 0.55
AGING_CAP_DAYS: Final[int] = 30
ESCALATION_PERCENTILE: Final[float] = 0.80


assert math.isclose(WEIGHT_FOLLOW_UP + WEIGHT_AGING, 1.0), (
    "the two escalation weights must sum to 1.0 so the score spans the full [0, 1]"
)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def escalation_components(record: dict[str, Any]) -> dict[str, Any]:
    """Compute the transparent per-signal breakdown for one record."""
    days = int(record["days_since_created"])
    follow_up = bool(record["follow_up_required"])
    aging_normalised = _clamp(min(days, AGING_CAP_DAYS) / AGING_CAP_DAYS)
    return {
        "follow_up_required": follow_up,
        "follow_up_component": round(WEIGHT_FOLLOW_UP * (1.0 if follow_up else 0.0), 4),
        "days_since_created": days,
        "aging_normalised": round(aging_normalised, 4),
        "aging_component": round(WEIGHT_AGING * aging_normalised, 4),
        "weight_follow_up": WEIGHT_FOLLOW_UP,
        "weight_aging": WEIGHT_AGING,
        "aging_cap_days": AGING_CAP_DAYS,
    }


def escalation_score(record: dict[str, Any]) -> float:
    """The designed escalation score in ``[0, 1]``, rounded to 4 decimals."""
    components = escalation_components(record)
    raw = components["follow_up_component"] + components["aging_component"]
    return round(_clamp(raw), 4)


def _nearest_rank_percentile(values: list[float], percentile: float) -> float:
    """Nearest-rank percentile: the smallest value at or above the given rank.


    Chosen over interpolation because it always returns a value that actually
    occurs in the data, which makes the threshold easy to justify from the
    dataset itself.
    """
    if not values:
        raise ValueError("cannot take a percentile of an empty sequence.")
    if not 0.0 < percentile <= 1.0:
        raise ValueError(f"percentile must be in (0, 1], got {percentile}.")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


@lru_cache(maxsize=4)
def escalation_threshold(percentile: float = ESCALATION_PERCENTILE) -> float:
    """The escalation threshold, derived from the dataset's own distribution.


    Computed as the nearest-rank ``percentile`` of the escalation scores across
    every generated appointment. Cached because the dataset is immutable.
    """
    scores = [escalation_score(record) for record in APPOINTMENTS]
    return round(_nearest_rank_percentile(scores, percentile), 4)


def escalation_distribution(percentile: float = ESCALATION_PERCENTILE) -> dict[str, Any]:
    """Distribution summary that justifies the threshold. Used by reports."""
    scores = sorted(escalation_score(record) for record in APPOINTMENTS)
    threshold = escalation_threshold(percentile)
    above = [score for score in scores if score >= threshold]
    follow_up_records = [r for r in APPOINTMENTS if r["follow_up_required"]]
    escalated_without_follow_up = [
        record["record_id"]
        for record in APPOINTMENTS
        if not record["follow_up_required"] and escalation_score(record) >= threshold
    ]
    suppressed_with_follow_up = [
        record["record_id"]
        for record in follow_up_records
        if escalation_score(record) < threshold
    ]
    return {
        "records": len(scores),
        "percentile": percentile,
        "threshold": threshold,
        "min": scores[0],
        "max": scores[-1],
        "mean": round(sum(scores) / len(scores), 4),
        "median": round(_nearest_rank_percentile(scores, 0.5), 4),
        "p90": round(_nearest_rank_percentile(scores, 0.9), 4),
        "records_at_or_above_threshold": len(above),
        "share_at_or_above_threshold": round(100.0 * len(above) / len(scores), 2),
        "follow_up_records": len(follow_up_records),
        # These two lists are the proof that the score is a genuine blend: some
        # non-follow-up records escalate on age alone, and some fresh follow-up
        # records do not escalate. A boolean OR could produce neither list.
        "escalated_without_follow_up_flag": escalated_without_follow_up,
        "follow_up_flag_but_not_escalated": suppressed_with_follow_up,
        "weights": {"follow_up": WEIGHT_FOLLOW_UP, "aging": WEIGHT_AGING},
        "aging_cap_days": AGING_CAP_DAYS,
    }


# --------------------------------------------------------------------------- #
# Task 6 - the lookup tool itself
# --------------------------------------------------------------------------- #


def format_readme_section(percentile: float = ESCALATION_PERCENTILE) -> str:
    """The compact block written into README.md's AUTO:ESCALATION markers.


    Task 6 requires the exact formula **and the threshold** to be stated, with
    the threshold justified from the dataset's own distribution. The formula is
    static so it is written in the README by hand; the numbers are not, so they
    come from here.
    """
    distribution = escalation_distribution(percentile)
    aged = distribution["escalated_without_follow_up_flag"]
    fresh = distribution["follow_up_flag_but_not_escalated"]
    lines = [
        f"**Escalation threshold = `{distribution['threshold']}`** - the nearest-rank "
        f"{int(percentile * 100)}th percentile of the escalation-score distribution "
        f"across all {distribution['records']} generated appointments, computed at run "
        "time by `escalation_threshold()`.",
        "",
        "| Statistic | Value |",
        "| --- | --- |",
        f"| records | {distribution['records']} |",
        f"| min score | {distribution['min']} |",
        f"| median score | {distribution['median']} |",
        f"| 90th percentile score | {distribution['p90']} |",
        f"| max score | {distribution['max']} |",
        f"| **threshold ({int(percentile * 100)}th percentile)** | "
        f"**{distribution['threshold']}** |",
        f"| records at or above the threshold | "
        f"{distribution['records_at_or_above_threshold']} "
        f"({distribution['share_at_or_above_threshold']}%) |",
        f"| records with `follow_up_required=True` | "
        f"{distribution['follow_up_records']} |",
        "",
        "**Proof it is not a boolean OR:**",
        "",
        f"- escalated on age alone, with **no** follow-up flag: "
        f"`{aged or 'none'}`",
        f"- carrying a follow-up flag but **not** escalated because still fresh: "
        f"`{fresh or 'none'}`",
        "",
        "Both outcomes are impossible under `follow_up_required OR is_old`. The first "
        "list is non-empty by construction, because the threshold is the 36th-smallest "
        "of 45 scores and at most one score separates it from the largest "
        "no-follow-up score.",
    ]
    return "\n".join(lines)


def check_appointment_status(record_id: str) -> dict[str, Any]:
    """Look up one appointment and score it for escalation.


    Args:
        record_id: an appointment id such as ``"APT-1007"``. Matching is
            case-insensitive and tolerant of surrounding whitespace.


    Returns:
        A dict that always contains ``record_id``, ``found`` and
        ``escalation_recommended``. When the record exists it also carries
        ``status``, ``consultation_fee_inr``, ``escalation_score``,
        ``escalation_threshold`` and the ``components`` breakdown. An unknown id
        returns a safe structured not-found payload rather than raising, so the
        agent can answer the patient instead of crashing the turn.
    """
    threshold = escalation_threshold()
    requested = (record_id or "").strip()
    record = get_appointment(requested)


    if record is None:
        return {
            "record_id": requested.upper(),
            "found": False,
            "status": None,
            "consultation_fee_inr": None,
            "category": None,
            "days_since_created": None,
            "follow_up_required": None,
            "escalation_score": None,
            "escalation_recommended": False,
            "escalation_threshold": threshold,
            "components": None,
            "message": (
                "No appointment record matches the requested id. "
                f"Appointment ids look like {EXEMPLAR_RECORD_ID}. "
                "Please re-check the id."
            ),
        }


    score = escalation_score(record)
    return {
        "record_id": record["record_id"],
        "found": True,
        "status": record["status"],
        "consultation_fee_inr": record["consultation_fee_inr"],
        "category": record["category"],
        "days_since_created": record["days_since_created"],
        "follow_up_required": record["follow_up_required"],
        "escalation_score": score,
        "escalation_recommended": score >= threshold,
        "escalation_threshold": threshold,
        "components": escalation_components(record),
        "message": None,
    }


# --------------------------------------------------------------------------- #
# The RAG tool
# --------------------------------------------------------------------------- #


def policy_knowledge_lookup(
    query: str,
    *,
    generator: GroundedGenerator,
    collection_name: str | None = None,
    top_k: int | None = None,
) -> GroundedAnswer:
    """Retrieve Practo policy context and generate a grounded answer.


    Thin wrapper over ``GroundedGenerator.generate`` so the crew tool, the
    direct pipeline and the evaluation harness all share one code path. Returns
    the domain object rather than a dict, so the retrieved context travels with
    the answer to the Composer and the Autogen reviewer.
    """
    return generator.generate(query, collection_name=collection_name, top_k=top_k)


# --------------------------------------------------------------------------- #
# Argument schemas - the dispatch key for the mock LLM
# --------------------------------------------------------------------------- #


class PolicyLookupArgs(BaseModel):
    """Arguments for the policy knowledge tool."""


    query: str = Field(description="The patient's policy question, in full.")


class AppointmentLookupArgs(BaseModel):
    """Arguments for the appointment status tool."""


    record_id: str = Field(description="An appointment record id such as APT-1007.")


#: Argument-name -> tool-name routing table.
#:
#: This is the mitigation for CrewAI pitfall (2). The mock LLM decides what to
#: pass a tool by inspecting the tool's own declared ``args_schema`` field names
#: and looking them up here - never by substring-matching the tool's *name*. A
#: tool literally called ``rag_lookup`` would be misclassified as an appointment
#: lookup by any ``"lookup" in name`` test; keying off ``{"query"}`` versus
#: ``{"record_id"}`` cannot make that mistake.
ARG_SCHEMA_ROUTING: Final[dict[frozenset[str], str]] = {
    frozenset(PolicyLookupArgs.model_fields): TOOL_POLICY_LOOKUP,
    frozenset(AppointmentLookupArgs.model_fields): TOOL_APPOINTMENT_LOOKUP,
}


class ToolDispatchError(RuntimeError):
    """Raised when a tool's argument schema matches no known tool kind."""


def classify_tool(tool: Any) -> str:
    """Identify a tool by its declared argument schema.


    Raises:
        ToolDispatchError: when the schema matches no entry in
            ``ARG_SCHEMA_ROUTING``.
    """
    schema = getattr(tool, "args_schema", None)
    fields = getattr(schema, "model_fields", None)
    if not fields:
        raise ToolDispatchError(
            f"tool {getattr(tool, 'name', tool)!r} declares no args_schema, so it "
            "cannot be dispatched by argument schema."
        )
    key = frozenset(fields)
    if key not in ARG_SCHEMA_ROUTING:
        raise ToolDispatchError(
            f"tool {getattr(tool, 'name', tool)!r} declares arguments {sorted(key)}, "
            f"which match no known tool kind. Known: "
            f"{[sorted(known) for known in ARG_SCHEMA_ROUTING]}."
        )
    return ARG_SCHEMA_ROUTING[key]


# --------------------------------------------------------------------------- #
# CrewAI tool wrappers
# --------------------------------------------------------------------------- #


POLICY_TOOL_DESCRIPTION: Final[str] = (
    "Search Practo's clinic-policy knowledge base and return a grounded answer "
    "built only from retrieved policy text. Use this for any question about "
    "booking, cancellation, rescheduling, fees, insurance, prescriptions, lab "
    "turnaround, telemedicine, emergencies, privacy, follow-up discounts, second "
    "opinions or home visits. Input: the patient's question as a single string."
)


APPOINTMENT_TOOL_DESCRIPTION: Final[str] = (
    "Look up one specific appointment record by its id (for example APT-1007) and "
    "return its status, consultation fee and a designed escalation score. Use this "
    "only when the patient names a concrete appointment id."
)


def _summarise_lookup(payload: dict[str, Any]) -> str:
    if not payload.get("found"):
        return f"{payload.get('record_id')}: not found"
    return (
        f"{payload['record_id']}: status={payload['status']}, "
        f"fee={payload['consultation_fee_inr']} INR, "
        f"escalation_score={payload['escalation_score']} "
        f"(threshold {payload['escalation_threshold']})"
    )


def _summarise_policy(answer: GroundedAnswer) -> str:
    return (
        f"grounded={answer.grounded}, "
        f"top_similarity={answer.top_similarity:.4f}, "
        f"threshold={answer.threshold:.4f}, "
        f"sources={list(answer.sources)}, "
        f"cache_hit={answer.cache_hit}"
    )


def invoke_policy_lookup(
    context: CrewRunContext, generator: GroundedGenerator, query: str | None = None
) -> GroundedAnswer:
    """Run the policy tool, write the result onto ``context``, ledger the call.


    Shared by the CrewAI tool wrapper and by ``run_direct()``, so the two
    orchestration modes cannot drift apart in what they record.
    """
    text = (query or context.query or "").strip()
    try:
        answer = policy_knowledge_lookup(
            text,
            generator=generator,
            collection_name=context.collection_name,
            top_k=context.top_k,
        )
    except Exception as exc:  # noqa: BLE001 - ledgered, then re-raised
        context.record(
            tool_name=TOOL_POLICY_LOOKUP,
            agent_key=AGENT_RETRIEVAL,
            arguments={"query": text},
            ok=False,
            summary=f"{type(exc).__name__}: {exc}",
        )
        raise


    context.grounded = answer
    context.record(
        tool_name=TOOL_POLICY_LOOKUP,
        agent_key=AGENT_RETRIEVAL,
        arguments={"query": text},
        ok=True,
        summary=_summarise_policy(answer),
    )
    return answer


def invoke_appointment_lookup(
    context: CrewRunContext, record_id: str | None = None
) -> dict[str, Any]:
    """Run the appointment tool, write the result onto ``context``, ledger the call."""
    requested = (record_id or context.record_id or "").strip()
    payload = check_appointment_status(requested)
    context.lookup = payload
    context.record(
        tool_name=TOOL_APPOINTMENT_LOOKUP,
        agent_key=AGENT_LOOKUP,
        arguments={"record_id": requested},
        ok=bool(payload["found"]),
        summary=_summarise_lookup(payload),
    )
    return payload


@dataclass(slots=True)
class CrewTools:
    """The tool objects for one crew run, already bound to this turn's context."""


    policy_tool: Any
    appointment_tool: Any


    def for_agent(self, agent_key: str) -> list[Any]:
        """Tools an agent is allowed to hold. Enforced by ``agents.governance``."""
        if agent_key == AGENT_RETRIEVAL:
            return [self.policy_tool]
        if agent_key == AGENT_LOOKUP:
            return [self.appointment_tool]
        # The Composer holds no tools at all - least autonomy.
        return []


def build_crew_tools(
    context: CrewRunContext,
    generator: GroundedGenerator,
    settings: Settings = SETTINGS,
) -> CrewTools:
    """Construct CrewAI tool objects bound to this turn's context and generator.


    Both tools write their full result into ``context`` and append to its
    ledger, then hand CrewAI a compact JSON string. The compact string keeps the
    ReAct prompt small; the Composer and the Autogen reviewer read the full
    objects from the context instead of re-parsing prompt text.


    Args:
        settings: accepted for signature parity with the rest of the
            orchestration layer. The tools read their per-turn parameters
            (``collection_name``, ``top_k``) from ``context`` instead, so that a
            single request cannot be served with two different configurations.


    Raises:
        RuntimeError: when crewai is not installed.
    """
    del settings  # see the note above; kept in the signature deliberately
    try:
        from crewai.tools import BaseTool
    except ImportError as exc:
        raise RuntimeError(
            "crewai is not installed, so the CrewAI tools cannot be built. Install "
            "the declared baseline with `pip install -r requirements.txt`, or set "
            "CREW_MODE=direct to use the diagnostic straight-line pipeline."
        ) from exc


    class PolicyKnowledgeLookupTool(BaseTool):  # type: ignore[misc,valid-type]
        """CrewAI wrapper around ``policy_knowledge_lookup``."""


        name: str = TOOL_POLICY_LOOKUP
        description: str = POLICY_TOOL_DESCRIPTION
        args_schema: type[BaseModel] = PolicyLookupArgs
        run_context: Any = None
        generator_ref: Any = None


        def _run(self, query: str = "", **_ignored: Any) -> str:
            answer = invoke_policy_lookup(self.run_context, self.generator_ref, query)
            # Compact payload only: the full GroundedAnswer (including the
            # retrieved context) is already on the run context.
            return json.dumps(
                {
                    "grounded": answer.grounded,
                    "answer": answer.answer,
                    "sources": list(answer.sources),
                    "top_similarity": round(answer.top_similarity, 4),
                },
                ensure_ascii=False,
            )


    class AppointmentStatusLookupTool(BaseTool):  # type: ignore[misc,valid-type]
        """CrewAI wrapper around ``check_appointment_status``."""


        name: str = TOOL_APPOINTMENT_LOOKUP
        description: str = APPOINTMENT_TOOL_DESCRIPTION
        args_schema: type[BaseModel] = AppointmentLookupArgs
        run_context: Any = None


        def _run(self, record_id: str = "", **_ignored: Any) -> str:
            payload = invoke_appointment_lookup(self.run_context, record_id)
            return json.dumps(payload, ensure_ascii=False)


    policy_tool = PolicyKnowledgeLookupTool(run_context=context, generator_ref=generator)
    appointment_tool = AppointmentStatusLookupTool(run_context=context)
    return CrewTools(policy_tool=policy_tool, appointment_tool=appointment_tool)



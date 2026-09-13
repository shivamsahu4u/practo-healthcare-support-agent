"""Deterministic request routing.


Routing is rule-based rather than model-decided. That is a governance choice as
much as an engineering one (see the Model/Orchestration layer in
``agents/governance.py``): the same question always takes the same path, so a
transcript is reproducible and the crew cannot talk itself into calling a tool
it should not.


Three routes:


* ``policy``      - knowledge-base retrieval only
* ``appointment`` - appointment lookup only
* ``combined``    - both
"""


from __future__ import annotations


import re
from typing import Final


from agents.context import ROUTE_APPOINTMENT, ROUTE_COMBINED, ROUTE_POLICY


#: ``APT-1007``, ``apt1007`` and ``APT 1007`` all resolve to ``APT-1007``.
RECORD_ID_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\bAPT[\s\-_]?(\d{4})\b", re.IGNORECASE
)


#: Substrings that mark a clinic-policy question. Stems are used deliberately
#: (``cancel`` covers cancel / cancelled / cancellation) so the table stays
#: small enough to read and audit.
POLICY_KEYWORD_STEMS: Final[tuple[str, ...]] = (
    "book", "cancel", "reschedul", "refund", "window", "charge", "fee", "cost",
    "price", "insur", "claim", "cashless", "reimburse", "prescription", "refill",
    "medicine", "dosage", "lab ", "lab test", "report", "turnaround", "pathology",
    "blood", "telemedicin", "video consult", "online consult", "emergency",
    "urgent", "privacy", "data ", "consent", "record export", "delete my",
    "follow-up", "follow up", "discount", "second opinion", "home visit",
    "polic", "eligib", "allowed", "walk-in", "walk in", "invoice",
)


#: Substrings that mark a question about a specific appointment's state.
APPOINTMENT_INTENT_STEMS: Final[tuple[str, ...]] = (
    "appointment", "status", "escalat", "my booking", "booking id", "record id",
    "apt-", "apt ",
)


def extract_record_id(text: str) -> str | None:
    """Return the first appointment id in ``text``, normalised, or ``None``."""
    match = RECORD_ID_PATTERN.search(text or "")
    return f"APT-{match.group(1)}" if match else None


def _contains_any(text: str, stems: tuple[str, ...]) -> bool:
    lowered = (text or "").lower()
    return any(stem in lowered for stem in stems)


def has_policy_intent(text: str) -> bool:
    """True when the text asks about clinic policy."""
    return _contains_any(text, POLICY_KEYWORD_STEMS)


def has_appointment_intent(text: str) -> bool:
    """True when the text asks about a specific appointment's state."""
    return _contains_any(text, APPOINTMENT_INTENT_STEMS)


def classify_route(text: str, *, record_id_from_memory: str | None = None) -> str:
    """Choose the route for one turn.


    Args:
        text: the masked query.
        record_id_from_memory: an id recovered from session history. It can
            promote a bare follow-up ("and what is its status?") to an
            appointment lookup, but it never *adds* a policy leg - a follow-up
            about policy stays a policy turn.


    The rules, in order:


    1. an appointment id in the text itself -> ``combined`` if the text also
       asks about policy, otherwise ``appointment``;
    2. no id in the text, but appointment intent and no policy intent ->
       ``appointment`` (the lookup then asks for the id, or memory supplies it);
    3. no intent signal at all, but session memory is holding an appointment id
       -> ``appointment``, which is what makes a bare follow-up such as "and
       what about that one?" resolve instead of falling through to retrieval;
    4. everything else -> ``policy``.
    """
    policy = has_policy_intent(text)
    appointment = has_appointment_intent(text)


    if extract_record_id(text):
        return ROUTE_COMBINED if policy else ROUTE_APPOINTMENT


    if appointment and not policy:
        return ROUTE_APPOINTMENT


    if record_id_from_memory and not policy and not appointment:
        return ROUTE_APPOINTMENT


    return ROUTE_POLICY



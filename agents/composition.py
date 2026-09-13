"""Deterministic answer composition - the Response Composer's actual logic.


This module is imported by **both** orchestration paths:


* ``agents/mock_llm.py``, when the CrewAI Composer agent's turn comes up, and
* ``agents/crew.py``'s ``run_direct()`` diagnostic pipeline.


Keeping it here is what stops the same composition rules from being written
twice and drifting apart: ``CREW_MODE=crewai`` and ``CREW_MODE=direct`` produce
byte-identical drafts because they call the same functions.


Composition is *frame plus fact*. Policy text is quoted verbatim from retrieved
chunks; appointment text is rendered from the structured lookup result. Nothing
is paraphrased, so the output-side groundedness guardrail and the Autogen
reviewer are checking real provenance rather than a paraphrase artefact.
"""


from __future__ import annotations


from typing import Any, Final


from app.config import EXEMPLAR_RECORD_ID, FALLBACK_ANSWER
from agents.context import CrewRunContext
from agents.routing import extract_record_id
from app.models import ResponseType
from rag.textutils import split_sentences


# Regression guard for the memory-poisoning defect described on
# ``app/config.py:EXEMPLAR_RECORD_ID``. If someone replaces the exemplar with a
# real-looking id, the assistant's own guidance becomes resolvable from session
# history and a follow-up turn discloses a record the patient never named. This
# assertion makes that mistake fail at import time rather than in production.
assert extract_record_id(EXEMPLAR_RECORD_ID) is None, (
    f"EXEMPLAR_RECORD_ID={EXEMPLAR_RECORD_ID!r} is a resolvable appointment id. "
    "It must stay digit-free, or the assistant's own boilerplate will poison "
    "session memory and disclose an unrelated appointment record."
)


#: Asked for an appointment status without naming an appointment. This is what
#: the fresh-session memory transcript must show (Task 8).
MISSING_RECORD_ID_ANSWER: Final[str] = (
    "I need the appointment id before I can check a specific appointment. "
    f"Appointment ids look like {EXEMPLAR_RECORD_ID} - please share yours and I "
    "will look it up."
)


#: A deliberately ungrounded sentence, appended ONLY when
#: ``CrewRunContext.inject_unsupported_claim`` is set. That flag is set by the
#: Task 14 revision demonstration and by the test suite, never by a runtime
#: request path. It exists so the Autogen reviewer has a genuine ungrounded
#: claim to catch rather than a staged one.
INJECTED_UNSUPPORTED_CLAIM: Final[str] = (
    "Practo also automatically upgrades every patient to a lifetime platinum "
    "membership with unlimited complimentary surgeries at any partner hospital."
)


#: Generic guidance emitted when a looked-up id does not exist. The id-specific
#: first sentence is supported separately, by ``render_lookup_facts``.
NOT_FOUND_GUIDANCE: Final[str] = (
    f"Appointment ids look like {EXEMPLAR_RECORD_ID}. Please re-check the id."
)


#: Sentence *templates* the composer renders from **structured facts**. Their
#: fixed wording ("is currently", "which is below", "so I am flagging it") is
#: legitimate support for the sentences built from it, because the facts they
#: carry - the id, status, category, fee, score and threshold - are supported
#: independently by ``render_lookup_facts``. These stay in the support corpus.
COMPOSER_FACT_TEMPLATES: Final[tuple[str, ...]] = (
    "According to Practo's policy:",
    "Appointment is currently in the category with a consultation fee of INR.",
    "Its escalation score is above the escalation threshold of so I am flagging "
    "it for a Practo support lead to review.",
    "Its escalation score is which is below the escalation threshold of so no "
    "escalation is needed right now.",
    "This appointment is flagged as requiring a follow-up visit.",
)


#: Fixed control answers this module emits verbatim. They assert nothing about
#: policy, so the output guardrail exempts them **by identity** rather than by
#: vocabulary - see ``EXEMPT_ANSWER_SENTENCES`` below.
FIXED_ANSWER_SENTENCES: Final[tuple[str, ...]] = (
    FALLBACK_ANSWER,
    MISSING_RECORD_ID_ANSWER,
    NOT_FOUND_GUIDANCE,
)


#: The control answers above, exploded into individual sentences, for identity
#: exemption in the groundedness check.
#:
#: Why identity and not support text: these strings used to be poured into the
#: groundedness *support corpus*, which handed **every** request a free
#: vocabulary of words like "know", "available", "knowledge", "base", "specific",
#: "share", "matches" and "re-check". Any drafted sentence reusing that
#: vocabulary scored as supported no matter where it came from, which widened the
#: check for every answer in order to let three fixed answers through. Exempting
#: the exact sentences keeps those three passing and narrows the corpus back to
#: genuine evidence: retrieved policy text, topic titles and looked-up facts.
EXEMPT_ANSWER_SENTENCES: Final[frozenset[str]] = frozenset(
    sentence
    for phrase in FIXED_ANSWER_SENTENCES
    for sentence in split_sentences(phrase)
)


#: Union of both sets, retained under the original name for any caller that
#: wants "every literal string this composer can emit".
COMPOSER_FRAMING_PHRASES: Final[tuple[str, ...]] = (
    COMPOSER_FACT_TEMPLATES + FIXED_ANSWER_SENTENCES
)




# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #




def compose_policy_section(context: CrewRunContext) -> str:
    """The policy half of the answer, or the fallback when retrieval was weak."""
    grounded = context.grounded
    if grounded is None:
        return ""
    return grounded.answer




def render_lookup_facts(lookup: dict[str, Any]) -> str:
    """Flatten the structured lookup result into support text.


    Used as groundedness support for the appointment sentences, so that a
    rendered fact counts as evidence for the sentence rendered from it.
    """
    if not lookup:
        return ""
    if not lookup.get("found"):
        return str(lookup.get("message") or "")
    parts = [
        f"record {lookup['record_id']}",
        f"status {lookup['status']}",
        f"category {lookup['category']}",
        f"consultation fee {lookup['consultation_fee_inr']} INR",
        f"days since created {lookup['days_since_created']}",
        f"follow up required {lookup['follow_up_required']}",
        f"escalation score {lookup['escalation_score']}",
        f"escalation threshold {lookup['escalation_threshold']}",
        f"escalation recommended {lookup['escalation_recommended']}",
    ]
    return ". ".join(parts) + "."




def compose_appointment_section(context: CrewRunContext) -> str:
    """The appointment half of the answer, rendered from structured facts only."""
    lookup = context.lookup
    if lookup is None:
        return MISSING_RECORD_ID_ANSWER if context.needs_appointment else ""


    if not lookup.get("found"):
        return str(
            lookup.get("message")
            or f"No appointment record matches {lookup.get('record_id')}."
        )


    sentences = [
        f"Appointment {lookup['record_id']} is currently {lookup['status']} in the "
        f"{lookup['category']} category, with a consultation fee of "
        f"{lookup['consultation_fee_inr']} INR."
    ]
    if lookup.get("follow_up_required"):
        sentences.append("This appointment is flagged as requiring a follow-up visit.")


    score = lookup.get("escalation_score")
    threshold = lookup.get("escalation_threshold")
    if lookup.get("escalation_recommended"):
        sentences.append(
            f"Its escalation score is {score}, above the escalation threshold of "
            f"{threshold}, so I am flagging it for a Practo support lead to review."
        )
    else:
        sentences.append(
            f"Its escalation score is {score}, which is below the escalation threshold "
            f"of {threshold}, so no escalation is needed right now."
        )
    return " ".join(sentences)




# --------------------------------------------------------------------------- #
# Draft
# --------------------------------------------------------------------------- #




def compose_draft(context: CrewRunContext) -> str:
    """Build the Composer's full draft answer for this turn."""
    sections: list[str] = []


    if context.needs_policy:
        policy = compose_policy_section(context)
        if policy:
            sections.append(policy)


    if context.needs_appointment:
        appointment = compose_appointment_section(context)
        if appointment:
            sections.append(appointment)


    if not sections:
        sections.append(FALLBACK_ANSWER)


    draft = " ".join(sections)


    if context.inject_unsupported_claim:
        draft = f"{draft} {INJECTED_UNSUPPORTED_CLAIM}"
    return draft




def determine_response_type(context: CrewRunContext) -> ResponseType:
    """Classify the turn against the closed ``ResponseType`` vocabulary."""
    has_policy = context.grounded is not None and context.grounded.grounded
    has_appointment = bool(context.lookup)


    if has_policy and has_appointment:
        return "combined"
    if has_appointment:
        return "appointment"
    if has_policy:
        return "policy"
    return "fallback"




def build_support_text(context: CrewRunContext) -> str:
    """The reference text a draft's sentences are checked against.


    Three components:


    1. the retrieved policy chunks - the only admissible source of policy claims;
    2. a rendering of the structured appointment facts, so a sentence derived
       from a looked-up fact is supported by that fact;
    3. this module's own **fact templates** - the fixed wording it wraps
       structured facts in, which is deterministic scaffolding rather than a
       model claim.


    Note what is deliberately *absent*: the fixed control answers
    (``FIXED_ANSWER_SENTENCES``). Those are handled by identity exemption via
    ``EXEMPT_ANSWER_SENTENCES`` instead of by being added here, because adding
    them widened the accepted vocabulary for every request in order to let three
    fixed answers through.


    The remaining limitation is stated plainly in the README: this is lexical
    overlap, not semantic entailment. It reliably catches novel vocabulary
    appearing out of nowhere - which is what a fabricated policy claim looks
    like - and it would not catch a fabrication assembled entirely from context
    vocabulary.
    """
    parts = [context.context_text]
    if context.grounded is not None:
        # The answer frame names the source topic, so the topic titles count as
        # support for their own mention. Without this, a title word that never
        # appears in the document body (for example "Cancellation" when the body
        # only says "cancelled") would read as an unsupported claim.
        parts.extend(
            str(chunk.get("topic_title", "")) for chunk in context.grounded.retrieved
        )
    if context.lookup:
        parts.append(render_lookup_facts(context.lookup))
    parts.extend(COMPOSER_FACT_TEMPLATES)
    return "\n".join(part for part in parts if part)




def compose_sources(context: CrewRunContext) -> list[str]:
    """Source document ids backing this answer, plus the appointment record."""
    sources = list(context.source_ids)
    lookup = context.lookup
    if lookup and lookup.get("found"):
        marker = f"appointment:{lookup['record_id']}"
        if marker not in sources:
            sources.append(marker)
    return sources




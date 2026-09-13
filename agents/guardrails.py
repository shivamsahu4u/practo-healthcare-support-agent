"""Task 10 - input-side and output-side guardrails.


Input side
    * **Contact-number masking.** Indian contact numbers are the one PII field
      in this scenario with a fixed, matchable format, so they are masked
      deterministically by regex *before* the agent sees the text and *before*
      anything is logged. The same masked string is used for both, which is why
      a raw number can never reach disk.
    * **Prompt-injection detection.** A transparent list of named patterns. When
      one fires the request is blocked and never reaches the crew or the cache.


Output side
    * **Groundedness.** A drafted answer's sentences must be supported by the
      retrieved context (plus the structured appointment facts and the
      composer's own framing). Unsupported drafts are refused, not shipped.


Scope, stated honestly:


* ``contact_number`` is in scope for masking - fixed format, matchable.
* ``patient name``, ``diagnosis / condition`` and ``insurance ID`` are free text
  with no universal format, so they are **out of scope** for a keyless,
  ``MOCK_LLM``-only masker. Every example of all three in this repository is
  fabricated. This is a documented limitation, not an oversight.
* The injection detector is a deterministic denylist. It is not claimed to be
  complete: a novel phrasing that avoids every listed pattern would pass.
"""


from __future__ import annotations


import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Final


from app.config import SETTINGS, Settings
from rag.textutils import unsupported_sentences


#: Replacement token. Contains no digits at all, so no part of the original
#: number survives masking - which is what the tests assert.
CONTACT_MASK_TOKEN: Final[str] = "[CONTACT_MASKED]"


#: Indian contact numbers: an optional ``+91`` / ``0091`` / ``0`` prefix, then a
#: ten-digit number starting 6-9, optionally broken by one space, dot or hyphen.
#: The leading ``(?<!\w)`` and trailing ``(?!\d)`` stop it from biting into a
#: longer token, so ``APT-1007`` and ``2500 INR`` are never touched.
CONTACT_NUMBER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"""
    (?<!\w)
    (?:(?:\+|00)?91[\s.\-]?|0)?     # optional country code or trunk prefix
    [6-9]\d{4}                      # Indian mobile numbers start 6, 7, 8 or 9
    [\s.\-]?
    \d{5}
    (?!\d)
    """,
    re.VERBOSE,
)


#: Deterministic prompt-injection denylist: ``(name, pattern)``.
INJECTION_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    (
        "ignore_instructions",
        re.compile(r"\b(ignore|disregard|forget)\b[^.?!]{0,40}\b(previous|prior|above|earlier|all)\b[^.?!]{0,20}\b(instruction|prompt|rule|polic)", re.IGNORECASE),
    ),
    (
        # "instructions" on its own is too generic - a patient may legitimately
        # ask to be shown the instructions for a lab test - so the second group
        # requires it to be qualified as the assistant's own instructions.
        "reveal_system_prompt",
        re.compile(r"\b(reveal|show|print|repeat|output|dump|leak)\b[^.?!]{0,30}\b(system\s*prompt|your\s*prompt|your\s+instructions|system\s+instructions|initial\s+instructions|hidden\s*rules)\b", re.IGNORECASE),
    ),
    (
        "exfiltrate_secrets",
        re.compile(r"\b(show|print|give|send|reveal|list|what|tell)\b[^.?!]{0,30}\b(api[\s_-]?key|secret|token|credential|password|env(?:ironment)?\s*variable)s?\b|\b(api[\s_-]?key|secret|token|credential|password)s?\b[^.?!]{0,30}\b(show|print|give|send|reveal|list)\b", re.IGNORECASE),
    ),
    (
        "role_override",
        re.compile(r"\b(you\s+are\s+now|from\s+now\s+on\s+you|act\s+as|pretend\s+to\s+be|roleplay\s+as)\b[^.?!]{0,40}\b(admin|administrator|root|developer|dba|unrestricted|no\s+rules|different\s+ai)\b", re.IGNORECASE),
    ),
    (
        "jailbreak_mode",
        re.compile(r"\b(developer\s+mode|dan\s+mode|jailbreak|god\s*mode|unfiltered\s+mode|bypass\s+(?:the\s+)?(?:safety|guardrail|filter|polic))\b", re.IGNORECASE),
    ),
    (
        "policy_override",
        re.compile(r"\b(override|disable|turn\s+off|switch\s+off|remove)\b[^.?!]{0,30}\b(guardrail|safety|polic|restriction|validation|groundedness|budget)\w*\b", re.IGNORECASE),
    ),
    (
        "tool_permission_escalation",
        re.compile(r"\b(grant|give|assign|attach|wire)\b[^.?!]{0,40}\b(tool|check_appointment_status|lookup\s+tool)\b|\bcall\s+check_appointment_status\b[^.?!]{0,30}\b(directly|yourself|without)\b", re.IGNORECASE),
    ),
    (
        "encoded_payload",
        re.compile(r"\b(base64|rot13|hex)\b[^.?!]{0,30}\b(decode|decrypt|then\s+(?:run|execute|follow))\b|\bdecode\s+(?:this|the\s+following)\b[^.?!]{0,20}\b(and|then)\b[^.?!]{0,20}\b(execute|run|obey|follow)\b", re.IGNORECASE),
    ),
    (
        "data_exfiltration",
        re.compile(r"\b(list|dump|export|show)\b[^.?!]{0,30}\b(all\s+(?:patients?|records?|appointments?|users?)|every\s+(?:patient|record|appointment))\b", re.IGNORECASE),
    ),
)


INJECTION_SAFETY_MESSAGE: Final[str] = (
    "This request was blocked by the input guardrail because it tries to change the "
    "assistant's instructions, permissions or safety rules. I can only answer "
    "questions about Practo clinic policy and the status of an appointment you own."
)


GROUNDEDNESS_REFUSAL_MESSAGE: Final[str] = (
    "The retrieved Practo policy context does not support an answer to this "
    "question, so I will not answer it rather than guess."
)


# --------------------------------------------------------------------------- #
# Findings
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PiiFinding:
    """One masked PII match.


    Deliberately carries **no** raw value - only a length and a one-way
    fingerprint. Findings end up in logs, so a finding that quoted the number
    would defeat the masking it is reporting.
    """


    field_name: str
    start: int
    end: int
    digit_count: int
    fingerprint: str


    def as_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "start": self.start,
            "end": self.end,
            "digit_count": self.digit_count,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class InjectionFinding:
    """One prompt-injection pattern match."""


    pattern_name: str
    matched_text: str


    def as_dict(self) -> dict[str, Any]:
        return {"pattern_name": self.pattern_name, "matched_text": self.matched_text}


@dataclass(slots=True)
class GuardrailReport:
    """Result of the input-side guardrails."""


    original_length: int
    masked_text: str
    pii_findings: tuple[PiiFinding, ...] = ()
    injection_findings: tuple[InjectionFinding, ...] = ()
    fired: list[str] = field(default_factory=list)


    @property
    def pii_masked(self) -> bool:
        return bool(self.pii_findings)


    @property
    def injection_detected(self) -> bool:
        return bool(self.injection_findings)


    @property
    def blocked(self) -> bool:
        """Injection blocks the turn; masking does not - masking is a repair."""
        return self.injection_detected


    def as_dict(self) -> dict[str, Any]:
        return {
            "original_length": self.original_length,
            "pii_masked": self.pii_masked,
            "pii_findings": [finding.as_dict() for finding in self.pii_findings],
            "injection_detected": self.injection_detected,
            "injection_findings": [finding.as_dict() for finding in self.injection_findings],
            "fired": list(self.fired),
            "blocked": self.blocked,
        }


@dataclass(frozen=True, slots=True)
class GroundednessReport:
    """Result of the output-side groundedness check."""


    grounded: bool
    minimum_overlap: float
    unsupported: tuple[str, ...]
    reason: str


    def as_dict(self) -> dict[str, Any]:
        return {
            "grounded": self.grounded,
            "minimum_overlap": self.minimum_overlap,
            "unsupported": list(self.unsupported),
            "reason": self.reason,
        }


# --------------------------------------------------------------------------- #
# Input side
# --------------------------------------------------------------------------- #


def _fingerprint(value: str) -> str:
    """Short, non-reversible fingerprint so two masks can be correlated safely."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def mask_contact_numbers(text: str) -> tuple[str, tuple[PiiFinding, ...]]:
    """Replace every Indian contact number with ``CONTACT_MASK_TOKEN``.


    Returns:
        The masked text and one finding per replacement. The findings never
        contain the matched digits.
    """
    source = text or ""
    findings: list[PiiFinding] = []
    pieces: list[str] = []
    cursor = 0


    for match in CONTACT_NUMBER_PATTERN.finditer(source):
        raw = match.group(0)
        digits = [char for char in raw if char.isdigit()]
        pieces.append(source[cursor : match.start()])
        pieces.append(CONTACT_MASK_TOKEN)
        findings.append(
            PiiFinding(
                field_name="contact_number",
                start=match.start(),
                end=match.end(),
                digit_count=len(digits),
                fingerprint=_fingerprint("".join(digits)),
            )
        )
        cursor = match.end()


    pieces.append(source[cursor:])
    return "".join(pieces), tuple(findings)


_NON_DIGITS: Final[re.Pattern[str]] = re.compile(r"[^\d]")


@dataclass(frozen=True, slots=True)
class ContactNumberCheck:
    """Result of validating one contact-number *field*."""


    valid: bool
    normalised: str | None
    digit_count: int
    reason: str


    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            # Only the reason and the digit count are safe to log; `normalised`
            # is returned to the caller but deliberately excluded here, because
            # this dict is the shape that ends up in a log line.
            "digit_count": self.digit_count,
            "reason": self.reason,
        }


def validate_contact_number(value: str) -> ContactNumberCheck:
    """Validate an Indian contact number's format.


    This is a different job from masking, and both are required. Masking answers
    "is there a number hidden in this free text, and can I redact it before the
    model or the log sees it". Validation answers "is this *field* a well-formed
    contact number", which is what a booking or call-back flow must ask before it
    accepts one - rejecting ``98765`` or ``1234567890`` up front rather than
    storing it and failing later.


    Accepted: an optional ``+91`` / ``0091`` / ``91`` / ``0`` prefix, then ten
    national digits beginning 6, 7, 8 or 9, with spaces, dots or hyphens
    anywhere.


    Returns:
        A ``ContactNumberCheck``. ``normalised`` is the ten national digits when
        valid and ``None`` otherwise; ``as_dict()`` omits it, because that form
        is what gets logged.
    """
    raw = (value or "").strip()
    if not raw:
        return ContactNumberCheck(False, None, 0, "no contact number was provided")


    digits = _NON_DIGITS.sub("", raw)
    national = digits
    for prefix in ("0091", "91", "0"):
        if len(national) > 10 and national.startswith(prefix):
            national = national[len(prefix) :]
            break


    if len(national) != 10:
        return ContactNumberCheck(
            False,
            None,
            len(digits),
            f"expected 10 national digits after any +91 or 0 prefix, found {len(national)}",
        )
    if national[0] not in "6789":
        return ContactNumberCheck(
            False,
            None,
            len(digits),
            "an Indian mobile number starts with 6, 7, 8 or 9",
        )
    return ContactNumberCheck(
        True, national, len(digits), "well-formed Indian mobile number"
    )


def detect_prompt_injection(text: str) -> tuple[InjectionFinding, ...]:
    """Match the deterministic injection denylist against ``text``.


    The matched substring is truncated to 120 characters so a very long payload
    cannot bloat a log line.
    """
    source = text or ""
    findings: list[InjectionFinding] = []
    for name, pattern in INJECTION_PATTERNS:
        match = pattern.search(source)
        if match:
            findings.append(
                InjectionFinding(pattern_name=name, matched_text=match.group(0)[:120])
            )
    return tuple(findings)


def apply_input_guardrails(text: str) -> GuardrailReport:
    """Run masking then injection detection, in that order.


    Masking runs first on purpose: the injection detector then sees the same
    masked string the agent will see, so a payload hidden inside a phone number
    cannot slip past by being redacted afterwards.
    """
    masked, pii_findings = mask_contact_numbers(text)
    injection_findings = detect_prompt_injection(masked)


    fired: list[str] = []
    if pii_findings:
        fired.append("pii_contact_number_masking")
    if injection_findings:
        fired.append("prompt_injection_detection")


    return GuardrailReport(
        original_length=len(text or ""),
        masked_text=masked,
        pii_findings=pii_findings,
        injection_findings=injection_findings,
        fired=fired,
    )


# --------------------------------------------------------------------------- #
# Output side
# --------------------------------------------------------------------------- #


def check_groundedness(
    draft: str,
    support_text: str,
    *,
    retrieval_grounded: bool,
    exempt: frozenset[str] = frozenset(),
    settings: Settings = SETTINGS,
) -> GroundednessReport:
    """Verify a draft answer against its support text.


    Two conditions must both hold:


    1. retrieval cleared the calibrated similarity threshold
       (``retrieval_grounded``), and
    2. every sentence in the draft clears ``GROUNDEDNESS_OVERLAP_MIN`` content-word
       overlap against the support text.


    Args:
        retrieval_grounded: pass ``True`` for an appointment-only turn, where
            there is no policy claim for retrieval to support.
        exempt: fixed control sentences to accept by identity rather than by
            vocabulary - see ``rag.textutils.unsupported_sentences``. The caller
            supplies these (``agents.composition.EXEMPT_ANSWER_SENTENCES``) so
            this module stays a pure mechanism and does not depend on the
            composer.
    """
    minimum = settings.groundedness_overlap_min


    if not retrieval_grounded:
        return GroundednessReport(
            grounded=False,
            minimum_overlap=minimum,
            unsupported=(),
            reason=(
                "retrieval did not clear the calibrated similarity threshold, so no "
                "policy claim in this draft has admissible support"
            ),
        )


    unsupported = tuple(unsupported_sentences(draft, support_text, minimum, exempt))
    if unsupported:
        return GroundednessReport(
            grounded=False,
            minimum_overlap=minimum,
            unsupported=unsupported,
            reason=(
                f"{len(unsupported)} sentence(s) fall below {minimum:.2f} content-word "
                "overlap with the retrieved context"
            ),
        )


    return GroundednessReport(
        grounded=True,
        minimum_overlap=minimum,
        unsupported=(),
        reason="every sentence is supported by the retrieved context",
    )



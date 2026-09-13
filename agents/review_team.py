"""Task 14 - an independent Autogen review stage after the CrewAI draft.


Two agents in a ``RoundRobinGroupChat`` bounded with ``max_turns=2``:


1. ``policy_compliance_reviewer`` - checks each drafted sentence against the
   retrieved context and the structured appointment facts, and names any
   sentence that is not supported.
2. ``final_editor`` - approves the draft unchanged or revises it, emitting a
   Pydantic ``ReviewVerdict`` via ``output_content_type=ReviewVerdict``.


Two constructor details the brief calls out, both of which are load-bearing:


* ``max_turns=2`` is the real parameter name (not ``max_iterations``). With two
  participants that is exactly one turn each.
* Giving the Final-Editor ``output_content_type`` requires the **team** to be
  built with ``custom_message_types=[StructuredMessage[ReviewVerdict]]``, or the
  run dies with ``ValueError: Message type ... is not registered``.


Under ``MOCK_LLM`` the model client is ``MockChatCompletionClient``, a real
``autogen_core.models.ChatCompletionClient`` implementation that returns
deterministic text for the reviewer and deterministic ``ReviewVerdict`` JSON for
the editor. The review logic itself is the lexical support check in
``rag/textutils.py`` - independent of the crew, but lexical rather than
semantic, which is stated as a limitation in the README.
"""


from __future__ import annotations


import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Final, Mapping, Sequence


from app.config import FALLBACK_ANSWER, SETTINGS, Settings
from app.models import ReviewVerdict
from rag.textutils import split_sentences, unsupported_sentences


LOGGER: Final = logging.getLogger(__name__)


#: Role markers embedded in each agent's system message. The mock client keys
#: its behaviour off these rather than off agent names, so renaming an agent
#: cannot silently change which branch runs.
MARKER_REVIEWER: Final[str] = "[[ROLE:POLICY_COMPLIANCE_REVIEWER]]"
MARKER_EDITOR: Final[str] = "[[ROLE:FINAL_EDITOR]]"


REVIEWER_NAME: Final[str] = "policy_compliance_reviewer"
EDITOR_NAME: Final[str] = "final_editor"


#: Machine-readable block the reviewer appends to its critique, and the only
#: thing the Final Editor acts on.
#:
#: This exists to make the round-robin **causal**. The editor used to re-derive
#: the verdict from the session object, so the reviewer's turn was generated,
#: appended to the transcript and then read by nothing: renaming it "reviewer"
#: was the only thing that made it one. Now the editor removes exactly the
#: sentences the reviewer named, and refuses to act at all if the reviewer has
#: not spoken - so the two-agent conversation carries the decision rather than
#: decorating it.
#:
#: The prose findings above the block stay human-readable; this is the contract.
MARKER_FINDINGS_OPEN: Final[str] = "[[UNSUPPORTED_SENTENCES]]"
MARKER_FINDINGS_CLOSE: Final[str] = "[[/UNSUPPORTED_SENTENCES]]"
MARKER_RETRIEVAL_FAILED: Final[str] = "[[RETRIEVAL_BELOW_THRESHOLD]]"




class ReviewUnavailableError(RuntimeError):
    """Raised when the Autogen review stage cannot run."""




# --------------------------------------------------------------------------- #
# Review session + deterministic verdict logic
# --------------------------------------------------------------------------- #




@dataclass(slots=True)
class ReviewSession:
    """Everything the review team is given about one draft."""


    query: str
    """The **masked** patient query."""


    draft: str
    context_text: str
    support_text: str
    source_ids: list[str] = field(default_factory=list)
    lookup: dict[str, Any] | None = None
    retrieval_grounded: bool = True
    minimum_overlap: float = 0.6


    exempt: frozenset[str] = field(default_factory=frozenset)
    """Fixed control sentences accepted by identity rather than by vocabulary.


    Must be kept in step with what the output guardrail is given
    (``agents.composition.EXEMPT_ANSWER_SENTENCES``). If the reviewer does not
    know that the "I don't know" refusal and the request-for-an-id are fixed
    control text, it strips them as unsupported and the patient loses the
    guidance the answer exists to give.
    """


    def task_message(self) -> str:
        """The initiating task message handed to the team."""
        lookup_block = (
            json.dumps(self.lookup, ensure_ascii=False, indent=2)
            if self.lookup
            else "(no appointment lookup was performed for this turn)"
        )
        context_block = self.context_text or "(retrieval returned no context)"
        sources = ", ".join(self.source_ids) if self.source_ids else "(none)"
        return (
            "Review the following draft answer for a Practo patient.\n\n"
            f"### Patient question (PII-masked)\n{self.query}\n\n"
            f"### Draft answer from the CrewAI Response Composer\n{self.draft}\n\n"
            f"### Retrieved policy context (the only admissible source of policy claims)\n"
            f"{context_block}\n\n"
            f"### Source document ids\n{sources}\n\n"
            f"### Appointment lookup result\n{lookup_block}\n\n"
            f"### Retrieval cleared the calibrated similarity threshold\n"
            f"{self.retrieval_grounded}\n"
        )




def find_unsupported(session: ReviewSession) -> list[str]:
    """Drafted sentences not supported by the session's support text."""
    return unsupported_sentences(
        session.draft,
        session.support_text,
        session.minimum_overlap,
        session.exempt,
    )




def strip_sentences(draft: str, remove: Sequence[str]) -> str:
    """Drop the named sentences from ``draft``, keeping the rest verbatim."""
    removal = {sentence.strip() for sentence in remove}
    kept = [
        sentence for sentence in split_sentences(draft) if sentence.strip() not in removal
    ]
    return " ".join(kept).strip()




def format_findings_block(retrieval_failed: bool, unsupported: Sequence[str]) -> str:
    """Render the reviewer's machine-readable findings block.


    Internal whitespace in each named sentence is collapsed: a sentence quoted
    across a chunk boundary can contain a newline, which would otherwise be read
    back as two separate findings.
    """
    if retrieval_failed:
        return MARKER_RETRIEVAL_FAILED
    lines = [MARKER_FINDINGS_OPEN]
    lines.extend(" ".join(sentence.split()) for sentence in unsupported)
    lines.append(MARKER_FINDINGS_CLOSE)
    return "\n".join(lines)




def parse_reviewer_findings(text: str) -> tuple[bool, list[str]] | None:
    """Read the reviewer's findings out of the conversation so far.


    Returns:
        ``(retrieval_failed, named_sentences)``, or ``None`` when no findings
        block is present at all - which means the Policy Compliance Reviewer has
        not spoken, and the Final Editor must refuse rather than invent a
        verdict of its own.
    """
    if MARKER_RETRIEVAL_FAILED in text:
        return True, []
    start = text.find(MARKER_FINDINGS_OPEN)
    if start < 0:
        return None
    end = text.find(MARKER_FINDINGS_CLOSE, start)
    if end < 0:
        return None
    body = text[start + len(MARKER_FINDINGS_OPEN) : end]
    return False, [line.strip() for line in body.splitlines() if line.strip()]




def verdict_from_findings(
    session: ReviewSession, retrieval_failed: bool, named: Sequence[str]
) -> ReviewVerdict:
    """Build the verdict from the sentences the **reviewer named**.


    This is the Final Editor's actual decision procedure: it removes exactly what
    it was told to remove and nothing else, which is what makes the editor unable
    to introduce a claim of its own.
    """
    draft = (session.draft or "").strip()
    if not draft:
        return ReviewVerdict(
            approved=False,
            final_answer=FALLBACK_ANSWER,
            reason="the draft was empty, so there was nothing to approve",
        )


    if retrieval_failed:
        return ReviewVerdict(
            approved=False,
            final_answer=FALLBACK_ANSWER,
            reason=(
                "retrieval did not clear the calibrated similarity threshold, so no "
                "policy claim in the draft has admissible support; replaced with the "
                "explicit refusal"
            ),
        )


    # Map the reviewer's names back onto the draft's own sentences, so the editor
    # strips exactly the draft text even if the transport altered whitespace.
    wanted = {" ".join(name.split()) for name in named}
    unsupported = [
        sentence
        for sentence in split_sentences(draft)
        if " ".join(sentence.split()) in wanted
    ]


    if not unsupported:
        return ReviewVerdict(
            approved=True,
            final_answer=draft,
            reason=(
                f"every sentence is supported by at least {session.minimum_overlap:.2f} "
                f"content-word overlap with the retrieved context and the appointment facts; "
                f"sources: {', '.join(session.source_ids) or 'none'}"
            ),
        )


    revised = strip_sentences(draft, unsupported)
    if not revised:
        return ReviewVerdict(
            approved=False,
            final_answer=FALLBACK_ANSWER,
            reason=(
                "every sentence in the draft was unsupported by the retrieved "
                "context, so the whole draft was replaced with the explicit refusal"
            ),
        )


    quoted = "; ".join(f'"{sentence}"' for sentence in unsupported)
    return ReviewVerdict(
        approved=False,
        final_answer=revised,
        reason=(
            f"removed {len(unsupported)} unsupported sentence(s) that the retrieved "
            f"context does not back: {quoted}"
        ),
    )




def deterministic_verdict(session: ReviewSession) -> ReviewVerdict:
    """The verdict the review team is expected to reach.


    Kept as the independently-derived expectation: tests assert the *team's*
    structured output against this, so a bug in the reviewer/editor message
    contract shows up as a mismatch rather than passing silently. The editor
    itself no longer calls this - it calls ``verdict_from_findings`` with what the
    reviewer actually said.
    """
    return verdict_from_findings(
        session, not session.retrieval_grounded, find_unsupported(session)
    )




def reviewer_critique(session: ReviewSession) -> str:
    """The reviewer's plain-text turn."""
    unsupported = find_unsupported(session)
    header = (
        f"Compliance review of the draft for the question: {session.query}\n"
        f"Source document ids: {', '.join(session.source_ids) or '(none)'}\n"
        f"Retrieval cleared the calibrated threshold: {session.retrieval_grounded}\n"
        f"Support test: content-word overlap >= {session.minimum_overlap:.2f} against "
        "the retrieved context plus the structured appointment facts.\n"
    )
    retrieval_failed = not session.retrieval_grounded


    if retrieval_failed:
        prose = (
            header
            + "FINDING: retrieval did not clear the calibrated similarity threshold, so "
            "this draft has no admissible policy support at all. The answer must be "
            "replaced with an explicit refusal."
        )
    elif not unsupported:
        prose = (
            header
            + "FINDING: none. Every sentence in the draft is supported. No safety "
            "concern: the draft makes no diagnostic claim and gives no clinical "
            "advice. Recommend approving unchanged."
        )
    else:
        listed = "\n".join(
            f"  {index}. {text}" for index, text in enumerate(unsupported, 1)
        )
        prose = (
            header
            + f"FINDING: {len(unsupported)} unsupported sentence(s). The retrieved "
            f"context does not back the following, so they must not reach the "
            f"patient:\n{listed}\n"
            "Recommend removing them and keeping the remainder unchanged."
        )


    # The prose above is for a human reading the transcript; the block below is
    # the contract the Final Editor actually acts on.
    return prose + "\n" + format_findings_block(retrieval_failed, unsupported)




# --------------------------------------------------------------------------- #
# The mock Autogen model client
# --------------------------------------------------------------------------- #




def build_mock_client(session: ReviewSession) -> Any:
    """Construct a deterministic ``ChatCompletionClient`` bound to ``session``.


    Defined inside a function so importing this module does not require Autogen
    until a review is actually requested.


    Raises:
        ReviewUnavailableError: when the Autogen packages are missing.
    """
    try:
        from autogen_core import CancellationToken
        from autogen_core.models import (
            ChatCompletionClient,
            CreateResult,
            LLMMessage,
            ModelInfo,
            RequestUsage,
        )
    except ImportError as exc:
        raise ReviewUnavailableError(
            "autogen-core / autogen-agentchat are not installed, so the review stage "
            "cannot run. Install the declared baseline with "
            "`pip install -r requirements.txt`, or set REVIEW_ENABLED=false to skip "
            "the review stage."
        ) from exc


    class MockChatCompletionClient(ChatCompletionClient):  # type: ignore[misc,valid-type]
        """Deterministic, keyless model client for the review team.


        Which branch runs is decided by the role marker in the incoming system
        message, so the two agents get different behaviour from one client
        without any network call.
        """


        #: Declared for Autogen's component machinery. This client is never
        #: serialised with ``dump_component()``, but the attribute costs nothing
        #: and keeps the subclass well-formed across Autogen releases.
        component_type = "model"


        def __init__(self, review_session: ReviewSession) -> None:
            self._session = review_session
            self._last_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
            self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
            self.reviewer_calls = 0
            self.editor_calls = 0
            #: What the reviewer actually reported on its own turn.
            #:
            #: The editor prefers to read the findings block out of the
            #: conversation, but this makes causality independent of how the
            #: framework propagates message text: either way the editor cannot
            #: produce a verdict until the reviewer has genuinely run.
            self._reviewer_findings: tuple[bool, list[str]] | None = None


        # -- helpers ---------------------------------------------------- #


        @staticmethod
        def _joined_system_text(messages: Sequence[Any]) -> str:
            parts: list[str] = []
            for message in messages:
                content = getattr(message, "content", "")
                if isinstance(content, str):
                    parts.append(content)
            return "\n".join(parts)


        def _respond(self, messages: Sequence[Any], json_output: Any) -> str:
            text = self._joined_system_text(messages)


            # An explicit structured-output request is unambiguous: only the
            # Final-Editor is configured with output_content_type.
            wants_structured = isinstance(json_output, type) or MARKER_EDITOR in text


            if wants_structured:
                self.editor_calls += 1
                # The editor acts on what the reviewer said, not on the session.
                # Without this the reviewer's turn was inert - generated, added
                # to the transcript, and read by nothing.
                findings = parse_reviewer_findings(text) or self._reviewer_findings
                if findings is None:
                    raise ReviewUnavailableError(
                        "the Final Editor was invoked with no findings block from "
                        f"{REVIEWER_NAME}. The editor removes only the sentences the "
                        "reviewer named, so it fails closed rather than re-deriving "
                        "a verdict of its own - that would make the review stage "
                        "decorative. Check that the reviewer runs first and that "
                        "its critique reaches the editor."
                    )
                retrieval_failed, named = findings
                verdict = verdict_from_findings(
                    self._session, retrieval_failed, named
                )
                return verdict.model_dump_json()


            if MARKER_REVIEWER in text:
                self.reviewer_calls += 1
                critique = reviewer_critique(self._session)
                # Remember our own findings so the editor is driven by this turn.
                self._reviewer_findings = parse_reviewer_findings(critique)
                return critique


            # Neither marker present: refuse to guess rather than emit text that
            # would be silently misattributed to one of the two agents.
            raise ReviewUnavailableError(
                "the review model client received a request with no role marker. Each "
                "review agent's system_message must contain either "
                f"{MARKER_REVIEWER} or {MARKER_EDITOR}."
            )


        def _account(self, prompt_text: str, completion_text: str) -> Any:
            prompt_tokens = max(1, len(prompt_text) // 4)
            completion_tokens = max(1, len(completion_text) // 4)
            self._last_usage = RequestUsage(
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
            )
            self._total_usage = RequestUsage(
                prompt_tokens=self._total_usage.prompt_tokens + prompt_tokens,
                completion_tokens=self._total_usage.completion_tokens + completion_tokens,
            )
            return self._last_usage


        # -- ChatCompletionClient interface ----------------------------- #


        async def create(  # type: ignore[override]
            self,
            messages: Sequence[LLMMessage],
            *,
            tools: Sequence[Any] = (),
            json_output: Any | None = None,
            extra_create_args: Mapping[str, Any] | None = None,
            cancellation_token: "CancellationToken | None" = None,
            **_forward_compatible: Any,
        ) -> Any:
            content = self._respond(messages, json_output)
            usage = self._account(self._joined_system_text(messages), content)
            return CreateResult(
                finish_reason="stop",
                content=content,
                usage=usage,
                cached=False,
            )


        async def create_stream(  # type: ignore[override]
            self,
            messages: Sequence[LLMMessage],
            *,
            tools: Sequence[Any] = (),
            json_output: Any | None = None,
            extra_create_args: Mapping[str, Any] | None = None,
            cancellation_token: "CancellationToken | None" = None,
            **_forward_compatible: Any,
        ) -> AsyncGenerator[Any, None]:
            result = await self.create(
                messages,
                tools=tools,
                json_output=json_output,
                extra_create_args=extra_create_args,
                cancellation_token=cancellation_token,
            )
            yield result


        async def close(self) -> None:
            return None


        def actual_usage(self) -> Any:
            return self._last_usage


        def total_usage(self) -> Any:
            return self._total_usage


        def count_tokens(self, messages: Sequence[Any], *, tools: Sequence[Any] = ()) -> int:
            return max(1, len(self._joined_system_text(messages)) // 4)


        def remaining_tokens(
            self, messages: Sequence[Any], *, tools: Sequence[Any] = ()
        ) -> int:
            return max(0, 8192 - self.count_tokens(messages, tools=tools))


        @property
        def model_info(self) -> "ModelInfo":  # type: ignore[override]
            # structured_output=True is required for output_content_type to be
            # accepted on the Final-Editor agent.
            return {
                "vision": False,
                "function_calling": False,
                "json_output": True,
                "family": "unknown",
                "structured_output": True,
            }


        @property
        def capabilities(self) -> Any:  # retained for older Autogen releases
            return self.model_info


    return MockChatCompletionClient(session)




# --------------------------------------------------------------------------- #
# The team
# --------------------------------------------------------------------------- #


REVIEWER_SYSTEM_MESSAGE: Final[str] = (
    f"{MARKER_REVIEWER}\n"
    "You are the Policy Compliance Reviewer for Practo's patient-support agent. "
    "You receive a draft answer, the retrieved policy context it was built from, the "
    "source document ids, and any appointment lookup result. Your job is to check "
    "three things and report them: (1) grounding - is every sentence in the draft "
    "supported by the retrieved context or by the structured appointment facts; "
    "(2) source support - are source document ids actually present; (3) safety - does "
    "the draft avoid diagnosis, clinical advice and emergency triage. Name every "
    "unsupported sentence explicitly. You do not rewrite the draft yourself."
)


EDITOR_SYSTEM_MESSAGE: Final[str] = (
    f"{MARKER_EDITOR}\n"
    "You are the Final Editor for Practo's patient-support agent. You receive the "
    "draft and the Policy Compliance Reviewer's findings. Either approve the draft "
    "unchanged, or revise it by removing exactly the unsupported sentences the "
    "reviewer named. You may never add a claim that the retrieved context does not "
    "contain, and you may never soften a policy. Respond with a ReviewVerdict "
    "containing approved, final_answer and reason."
)




@dataclass(slots=True)
class ReviewOutcome:
    """The review stage's result plus the transcript that produced it."""


    verdict: ReviewVerdict
    revised: bool
    messages: list[dict[str, str]] = field(default_factory=list)
    reviewer_calls: int = 0
    editor_calls: int = 0


    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.model_dump(),
            "revised": self.revised,
            "reviewer_calls": self.reviewer_calls,
            "editor_calls": self.editor_calls,
            "messages": list(self.messages),
        }




def _extract_verdict(messages: Sequence[Any]) -> ReviewVerdict | None:
    """Pull the structured verdict out of the team's message list."""
    for message in reversed(messages):
        content = getattr(message, "content", None)
        if isinstance(content, ReviewVerdict):
            return content
        # Some releases hand back the model as a dict on a StructuredMessage.
        if isinstance(content, dict) and {
            "approved",
            "final_answer",
            "reason",
        } <= set(content):
            return ReviewVerdict.model_validate(content)
    return None




async def review_draft(
    session: ReviewSession, settings: Settings = SETTINGS
) -> ReviewOutcome:
    """Run the two-agent Autogen review team over one draft.


    Raises:
        ReviewUnavailableError: when Autogen is missing or the team returns no
            structured verdict.
    """
    try:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.messages import StructuredMessage
        from autogen_agentchat.teams import RoundRobinGroupChat
    except ImportError as exc:
        raise ReviewUnavailableError(
            "autogen-agentchat is not installed, so the review stage cannot run. "
            "Install the declared baseline with `pip install -r requirements.txt`, or "
            "set REVIEW_ENABLED=false to skip the review stage."
        ) from exc


    client = build_mock_client(session)


    reviewer = AssistantAgent(
        name=REVIEWER_NAME,
        model_client=client,
        system_message=REVIEWER_SYSTEM_MESSAGE,
        description="Checks a draft answer for grounding, source support and safety.",
    )
    editor = AssistantAgent(
        name=EDITOR_NAME,
        model_client=client,
        system_message=EDITOR_SYSTEM_MESSAGE,
        description="Approves or revises the draft and emits a structured verdict.",
        # Structured Pydantic output on the agent...
        output_content_type=ReviewVerdict,
    )


    team = RoundRobinGroupChat(
        [reviewer, editor],
        # ...which obliges the team to register the matching message type, or
        # the run raises "Message type ... is not registered".
        custom_message_types=[StructuredMessage[ReviewVerdict]],
        max_turns=2,
    )


    try:
        result = await team.run(task=session.task_message())
    except Exception as exc:  # noqa: BLE001 - wrapped with actionable context
        raise ReviewUnavailableError(
            f"the Autogen review team failed to run: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        await client.close()


    transcript = [
        {
            "source": str(getattr(message, "source", "unknown")),
            "content": (
                message.content.model_dump_json()
                if isinstance(getattr(message, "content", None), ReviewVerdict)
                else str(getattr(message, "content", ""))
            ),
        }
        for message in result.messages
    ]


    verdict = _extract_verdict(result.messages)
    if verdict is None:
        raise ReviewUnavailableError(
            "the review team produced no structured ReviewVerdict. Check that the "
            "Final-Editor agent has output_content_type=ReviewVerdict and that the "
            "team was constructed with "
            "custom_message_types=[StructuredMessage[ReviewVerdict]]."
        )


    return ReviewOutcome(
        verdict=verdict,
        revised=verdict.final_answer.strip() != (session.draft or "").strip(),
        messages=transcript,
        reviewer_calls=getattr(client, "reviewer_calls", 0),
        editor_calls=getattr(client, "editor_calls", 0),
    )




def review_draft_sync(session: ReviewSession, settings: Settings = SETTINGS) -> ReviewOutcome:
    """Blocking wrapper for scripts and tests. Not for use inside a running loop."""
    return asyncio.run(review_draft(session, settings))




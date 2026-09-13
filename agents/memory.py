"""Task 8 - session memory with LangChain's session-based memory primitives.


``InMemoryChatMessageHistory`` holds one transcript per ``session_id``, and
``RunnableWithMessageHistory`` wires that history into the support pipeline so
each turn sees the ones before it.


``RunnableWithMessageHistory`` emits a ``LangChainDeprecationWarning`` pointing
at LangGraph's persistence layer. That is expected, the class still works
correctly here, and the brief explicitly says it does not need silencing - so it
is left visible.


Memory is in-process only: it does not survive a restart, which the brief states
is sufficient. What it *does* do is carry an appointment id forward, so a
follow-up such as "and what is its current status?" resolves inside one
conversation and correctly fails to resolve in a fresh one.


Only **masked** text is ever stored. The service masks the query before invoking
the runnable, so a contact number cannot reach the history object.
"""


from __future__ import annotations


import logging
from collections import OrderedDict
from typing import Any, Awaitable, Callable, Final


from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory


from agents.routing import extract_record_id


LOGGER: Final = logging.getLogger(__name__)


#: Key under which the validated ``SupportResponse`` payload rides along on the
#: stored ``AIMessage``. The message text is what a human reads back in the
#: transcript; this is what the API layer re-validates and returns.
RESPONSE_PAYLOAD_KEY: Final[str] = "support_response"




#: Ceiling on how many conversations are retained at once.
#:
#: ``session_id`` is client-supplied and conversation history deliberately
#: outlives its socket, so an unbounded registry let any caller grow process
#: memory without limit simply by sending a fresh id each turn. The Runtime
#: governance layer claims "bounded cost"; the response cache was bounded and
#: this - the larger allocation - was not.
DEFAULT_MAX_SESSIONS: Final[int] = 512




class SessionMemory:
    """A bounded, LRU registry of per-session chat histories."""


    def __init__(self, max_sessions: int = DEFAULT_MAX_SESSIONS) -> None:
        if max_sessions <= 0:
            raise ValueError(f"max_sessions must be positive, got {max_sessions}.")
        self._histories: OrderedDict[str, InMemoryChatMessageHistory] = OrderedDict()
        self._max_sessions = max_sessions
        self.evictions = 0


    @property
    def max_sessions(self) -> int:
        return self._max_sessions


    def get_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """Return (creating if needed) the history for ``session_id``.


        Touching a session marks it most-recently-used, so an active
        conversation is never evicted ahead of an idle one.
        """
        if not session_id or not session_id.strip():
            raise ValueError("session_id must not be empty.")
        key = session_id.strip()
        if key not in self._histories:
            self._histories[key] = InMemoryChatMessageHistory()
            LOGGER.debug("opened new session history %r", key)
        self._histories.move_to_end(key)
        self._evict_overflow()
        return self._histories[key]


    def _evict_overflow(self) -> None:
        """Drop least-recently-used conversations until the registry fits.


        An evicted conversation is gone, not archived: the next turn on that
        ``session_id`` starts fresh, which degrades a follow-up into a request
        for the appointment id rather than leaking or crashing.
        """
        while len(self._histories) > self._max_sessions:
            # The key is deliberately not logged: session ids are treated as
            # sensitive everywhere else (app/logging_config.py hashes them).
            _evicted_key, evicted = self._histories.popitem(last=False)
            evicted.clear()
            self.evictions += 1
            LOGGER.info(
                "evicted least-recently-used session history (cap=%s)",
                self._max_sessions,
            )


    def has_session(self, session_id: str) -> bool:
        return session_id.strip() in self._histories


    def reset(self, session_id: str) -> bool:
        """Clear one session. Returns whether anything was there to clear."""
        key = (session_id or "").strip()
        history = self._histories.pop(key, None)
        if history is None:
            return False
        history.clear()
        return True


    def reset_all(self) -> int:
        """Clear every session. Returns how many were removed."""
        count = len(self._histories)
        self._histories.clear()
        return count


    def sessions(self) -> list[str]:
        return sorted(self._histories)


    def messages(self, session_id: str) -> list[BaseMessage]:
        key = (session_id or "").strip()
        history = self._histories.get(key)
        return list(history.messages) if history else []


    def transcript(self, session_id: str) -> list[dict[str, str]]:
        """Readable transcript for one session, for the Task 8 evidence files."""
        return [
            {"role": message.type, "content": str(message.content)}
            for message in self.messages(session_id)
        ]


    def snapshot(self) -> dict[str, int]:
        return {session: len(self.messages(session)) for session in self.sessions()}




def resolve_record_id_from_history(history: list[BaseMessage]) -> str | None:
    """Find the most recently mentioned appointment id in a session history.


    Scans newest first so a conversation that discusses two appointments carries
    the one the patient is actually talking about now.
    """
    for message in reversed(history or []):
        found = extract_record_id(str(message.content))
        if found:
            return found
    return None




def build_memory_runnable(
    resolver: Callable[[dict[str, Any]], Awaitable[AIMessage]],
    memory: SessionMemory,
) -> RunnableWithMessageHistory:
    """Wrap ``resolver`` in LangChain session memory.


    Args:
        resolver: async callable receiving ``{"input": <masked query>,
            "history": [BaseMessage, ...]}`` and returning an ``AIMessage`` whose
            ``additional_kwargs[RESPONSE_PAYLOAD_KEY]`` carries the validated
            ``SupportResponse`` payload.
        memory: the session registry backing ``get_session_history``.


    Invoke with::


        await runnable.ainvoke(
            {"input": masked_query},
            config={"configurable": {"session_id": session_id}},
        )
    """
    chain = RunnableLambda(resolver)
    return RunnableWithMessageHistory(
        chain,
        memory.get_history,
        input_messages_key="input",
        history_messages_key="history",
    )




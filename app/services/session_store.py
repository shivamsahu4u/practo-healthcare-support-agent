"""Application-layer accessor for conversation memory.


Thin wrapper over ``agents.memory.SessionMemory``. It exists so the HTTP and
WebSocket layers depend on an application service rather than reaching into the
agents package, and so the WebSocket ``reset`` control frame has one obvious
place to call.
"""


from __future__ import annotations


from typing import Any


from agents.memory import SessionMemory


class SessionStore:
    """Owns the process-wide session memory registry."""


    def __init__(self, memory: SessionMemory | None = None) -> None:
        self.memory = memory or SessionMemory()


    def reset(self, session_id: str) -> bool:
        """Clear one conversation. Returns whether anything was cleared."""
        return self.memory.reset(session_id)


    def reset_all(self) -> int:
        return self.memory.reset_all()


    def transcript(self, session_id: str) -> list[dict[str, str]]:
        return self.memory.transcript(session_id)


    def active_sessions(self) -> list[str]:
        return self.memory.sessions()


    def snapshot(self) -> dict[str, Any]:
        return {
            "active_sessions": self.memory.sessions(),
            "message_counts": self.memory.snapshot(),
        }



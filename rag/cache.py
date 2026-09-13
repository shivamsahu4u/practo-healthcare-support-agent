"""Task 16 - in-memory response cache for the grounded-generation step.


The cache key is a SHA-256 digest of every input that can change the answer:


* the **normalised** query text (trimmed, lowercased, whitespace collapsed),
* the collection name, so the two chunking strategies never share an entry,
* ``top_k`` and the calibrated similarity threshold,
* the knowledge-base version, so adding a document invalidates every entry
  implicitly - on top of the explicit ``invalidate()`` call that
  ``POST /add-document`` also makes.


Deliberately *not* cached (the callers enforce this): blocked or
prompt-injection requests, errors, and appointment-status lookups. Appointment
state is mutable, so caching it could serve a stale status; only the immutable
policy layer is cached. Raw PII never reaches the cache either, because the
caller masks the query before generation.
"""


from __future__ import annotations


import hashlib
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Final, Generic, TypeVar


_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


T = TypeVar("T")




def normalise_query(query: str) -> str:
    """Deterministic query normalisation: strip, lowercase, collapse whitespace.


    ``"  What is the   CANCELLATION window? "`` and
    ``"what is the cancellation window?"`` therefore share one cache entry.
    """
    return _WHITESPACE.sub(" ", (query or "").strip().lower())




def make_cache_key(
    query: str,
    *,
    collection_name: str,
    top_k: int,
    threshold: float,
    kb_version: int,
    embedder: str = "",
) -> str:
    """Build the cache key. Components are length-prefixed so they cannot collide.


    Args:
        embedder: configuration identity of the active embedding backend, for
            example ``"sentence_transformers:sentence-transformers/all-MiniLM-L6-v2"``.


            This component matters because a cached answer is only valid for the
            embedding space it was retrieved in. Without it, pinning
            ``SIMILARITY_THRESHOLD`` and then switching ``EMBEDDING_BACKEND``
            produced *identical* keys, so answers retrieved under one embedder
            were served for another. Defaults to empty for backward
            compatibility; ``GroundedGenerator`` always supplies it.
    """
    parts = [
        normalise_query(query),
        collection_name,
        str(top_k),
        f"{threshold:.6f}",
        str(kb_version),
        embedder,
    ]
    payload = "|".join(f"{len(part)}:{part}" for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()




@dataclass(slots=True)
class CacheStats:
    """Counters used as the before/after evidence for the Task 16 demonstration."""


    hits: int = 0
    misses: int = 0
    evictions: int = 0
    invalidations: int = 0


    @property
    def lookups(self) -> int:
        return self.hits + self.misses


    @property
    def hit_rate(self) -> float:
        return 0.0 if self.lookups == 0 else round(self.hits / self.lookups, 4)


    def as_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "lookups": self.lookups,
            "hit_rate": self.hit_rate,
            "evictions": self.evictions,
            "invalidations": self.invalidations,
        }




class ResponseCache(Generic[T]):
    """Thread-safe bounded LRU cache.


    Thread safety matters because Uvicorn serves ``POST /ask`` from a worker
    thread pool while the WebSocket handler runs on the event loop.
    """


    def __init__(self, max_entries: int = 512, *, enabled: bool = True) -> None:
        if max_entries <= 0:
            raise ValueError(f"max_entries must be positive, got {max_entries}.")
        self._entries: OrderedDict[str, T] = OrderedDict()
        self._max_entries = max_entries
        self._enabled = enabled
        self._lock = threading.RLock()
        self.stats = CacheStats()


    @property
    def enabled(self) -> bool:
        return self._enabled


    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


    def get(self, key: str) -> T | None:
        """Look up ``key``, recording a hit or a miss. Disabled cache always misses."""
        if not self._enabled:
            return None
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                self.stats.hits += 1
                return self._entries[key]
            self.stats.misses += 1
            return None


    def set(self, key: str, value: T) -> None:
        """Store ``value``, evicting the least recently used entry when full."""
        if not self._enabled:
            return
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
                self.stats.evictions += 1


    def invalidate(self) -> int:
        """Drop every entry. Returns how many were removed."""
        with self._lock:
            removed = len(self._entries)
            self._entries.clear()
            self.stats.invalidations += 1
            return removed


    def reset_stats(self) -> None:
        """Zero the counters without touching the stored entries."""
        with self._lock:
            self.stats = CacheStats()


    def snapshot(self) -> dict[str, Any]:
        """JSON-safe view for ``GET /health`` and the cache transcript."""
        with self._lock:
            return {
                "enabled": self._enabled,
                "entries": len(self._entries),
                "max_entries": self._max_entries,
                **self.stats.as_dict(),
            }




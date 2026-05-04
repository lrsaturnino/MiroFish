"""
In-memory ``QueryCache`` for deduplicating research search results.

The cache is process-local: it stores the results of a search keyed on
the *normalised* query string and reuses them for any subsequent lookup
of the same normalised key while the TTL is still live. This is the
deduplication seam that lets two agents asking the same question within
``RESEARCH_CACHE_TTL_HOURS`` share a single upstream call.

Multi-process / multi-worker sharing is an explicit non-goal at this
layer — see the module-level TODO below for the planned Redis seam.
"""

from __future__ import annotations

import os
import re
import time  # module form intentional — tests monkeypatch cache.time.time

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import SearchResult


# TODO: externalize to Redis when multi-process scaling is needed (planning §5.6, §9.2).


class QueryCache:
    """Process-local TTL cache keyed on a normalised query string.

    The TTL is read once from ``RESEARCH_CACHE_TTL_HOURS`` (default 24)
    at construction time, so a cache instance carries a stable expiry
    window even if the environment changes mid-run. Cache keys are the
    output of :meth:`normalize`, which collapses internal whitespace,
    strips edges, and lowercases — different surface forms of the same
    query share a single cache entry.

    Multi-process safety is not provided. The cache lives in the local
    Python process only; cross-process sharing is the planned Redis
    migration documented in the module-level TODO.
    """

    def __init__(self) -> None:
        """Initialise an empty store and lock the TTL for this instance."""
        self._store: dict[str, tuple[list[SearchResult], float]] = {}
        self._ttl_seconds: int = (
            int(os.environ.get("RESEARCH_CACHE_TTL_HOURS", "24")) * 3600
        )

    @staticmethod
    def normalize(query: str) -> str:
        """Return the canonical cache key for ``query``.

        Order: collapse internal whitespace runs (``\\s+`` -> single
        space), then strip leading/trailing whitespace, then lowercase.
        The function is idempotent: ``normalize(normalize(q)) ==
        normalize(q)``.
        """
        return re.sub(r"\s+", " ", query).strip().lower()

    def set(self, query: str, results: list[SearchResult]) -> None:
        """Store ``results`` under the normalised form of ``query``.

        Overwrites any previous entry for the same normalised key. The
        entry expires at ``time.time() + self._ttl_seconds``.
        """
        key = self.normalize(query)
        self._store[key] = (results, time.time() + self._ttl_seconds)

    def get(self, query: str) -> list[SearchResult] | None:
        """Return cached results for ``query`` or ``None`` on miss/expiry.

        On expiry the stale entry is deleted from the internal store
        before ``None`` is returned, so long-running processes do not
        accumulate dead keys.
        """
        key = self.normalize(query)
        entry = self._store.get(key)
        if entry is None:
            return None
        results, expires_at = entry
        # Strict less-than: at exactly TTL the entry is still fresh.
        if expires_at < time.time():
            del self._store[key]
            return None
        return results

"""
Tests for the in-memory ``QueryCache`` at
``app.services.search.cache.QueryCache``.

The cache is a process-local memoiser keyed on a normalised query string,
with a TTL read once from ``RESEARCH_CACHE_TTL_HOURS`` at construction
time. The 13 scenarios below saturate the contract grid: hit / miss /
expiry / normalisation × constructor-time TTL × delete-on-expiry side
effect × boundary symmetry.

**Time-control convention** (load-bearing):

The cache module must use ``import time`` (module form), NOT
``from time import time``. Every TTL-bearing scenario monkeypatches
``app.services.search.cache.time.time`` to a controllable callable
returning a ``float`` from a list-style closure. Attribute lookup
``cache.time.time`` resolves only when ``cache.time`` is the ``time``
module object — a ``from time import time`` form binds ``cache.time``
to a function and breaks the patch. If a future refactor "tidies"
this, the boundary tests silently revert to wall-clock and the suite
either takes 24 real hours or flakes on clock skew.

**Forward-reference convention**:

The cache stores ``list[SearchResult]`` but ``SearchResult`` is owned
by sibling task T-006 (``app.services.search.base``) which may not
have landed yet. The cache module must use
``from __future__ import annotations`` plus a ``TYPE_CHECKING``-guarded
import so the module is runtime-importable without ``base``. These
tests construct ``SearchResult``-shaped dict literals inline and never
import from ``app.services.search.base``.

**Env-var convention**:

The autouse ``_clean_llm_env`` fixture in ``conftest.py`` clears 12
LLM env vars but does NOT touch ``RESEARCH_CACHE_TTL_HOURS``. Tests
that pin TTL behaviour use ``monkeypatch.setenv`` /
``monkeypatch.delenv`` explicitly — pytest's monkeypatch fixture
auto-restores between tests.
"""

import pytest

from app.services.search.cache import QueryCache


# --------------------------------------------------------------------------
# Per-test data helpers — inline SearchResult-shaped dicts. Distinct
# field values so an accidental swap surfaces immediately in assertions.
# Do NOT import SearchResult from app.services.search.base — that
# module is owned by sibling task T-006 and may not exist yet.
# --------------------------------------------------------------------------
R1 = {"title": "t1", "url": "u1", "snippet": "s1"}
R2 = {"title": "t2", "url": "u2", "snippet": "s2"}


# --------------------------------------------------------------------------
# Scenario 1 — Empty cache miss
# --------------------------------------------------------------------------
def test_get_returns_none_on_empty_cache_miss(monkeypatch):
    monkeypatch.delenv("RESEARCH_CACHE_TTL_HOURS", raising=False)

    cache = QueryCache()

    assert cache.get("q") is None


# --------------------------------------------------------------------------
# Scenario 2 — Set then get within TTL returns cached results
# --------------------------------------------------------------------------
def test_get_returns_results_after_set_within_ttl(monkeypatch):
    monkeypatch.delenv("RESEARCH_CACHE_TTL_HOURS", raising=False)

    cache = QueryCache()
    cache.set("q", [R1, R2])

    assert cache.get("q") == [R1, R2]


# --------------------------------------------------------------------------
# Scenario 3 — Set then get after TTL elapsed: returns None AND deletes
# the entry from the internal dict (memory-leak guard).
# --------------------------------------------------------------------------
def test_get_after_ttl_expiry_returns_none_and_removes_entry(monkeypatch):
    monkeypatch.delenv("RESEARCH_CACHE_TTL_HOURS", raising=False)

    ticks = [1000.0]
    monkeypatch.setattr(
        "app.services.search.cache.time.time", lambda: ticks[0]
    )

    cache = QueryCache()
    cache.set("q", [R1])

    # Advance virtual time past the default 24h TTL.
    ticks[0] += 24 * 3600 + 1

    assert cache.get("q") is None
    # The delete-on-expiry side effect: the normalised key must be
    # gone from the internal store. A naive impl that returns None
    # without ``del`` passes the first assertion but leaks memory in
    # long-running processes.
    assert "q" not in cache._store


# --------------------------------------------------------------------------
# Scenario 4 — Normalize trims edges
# --------------------------------------------------------------------------
def test_normalize_trims_edges():
    assert QueryCache.normalize("  hello  ") == "hello"


# --------------------------------------------------------------------------
# Scenario 5 — Normalize lowercases
# --------------------------------------------------------------------------
def test_normalize_lowercases():
    assert QueryCache.normalize("HELLO") == "hello"


# --------------------------------------------------------------------------
# Scenario 6 — Normalize collapses internal whitespace (tabs + newlines
# + multi-space runs all flatten to a single space).
# --------------------------------------------------------------------------
def test_normalize_collapses_internal_whitespace():
    assert QueryCache.normalize("hello   world\t\nfoo") == "hello world foo"


# --------------------------------------------------------------------------
# Scenario 7 — Normalize is idempotent: applying it twice equals
# applying it once, and both equal the canonical form.
# --------------------------------------------------------------------------
def test_normalize_is_idempotent():
    q = "  HELLO  WoRlD  "
    once = QueryCache.normalize(q)
    twice = QueryCache.normalize(once)

    assert once == twice
    assert once == "hello world"


# --------------------------------------------------------------------------
# Scenario 8 — Cache key uses the normalised form. Variants whose
# normalised text equals "hello" hit; a variant whose normalised text
# differs (e.g. "HE LLO" -> "he llo") misses.
# --------------------------------------------------------------------------
def test_cache_key_uses_normalized_form(monkeypatch):
    monkeypatch.delenv("RESEARCH_CACHE_TTL_HOURS", raising=False)

    cache = QueryCache()
    cache.set("HELLO", [R1])

    assert cache.get("hello") == [R1]
    assert cache.get("  hello  ") == [R1]
    # different normalised text — a miss is correct, not a bug.
    # ``normalize("HE LLO")`` is ``"he llo"``; that is not the same
    # cache key as ``"hello"``.
    assert cache.get("HE LLO") is None


# --------------------------------------------------------------------------
# Scenario 9 — TTL boundary: at exactly TTL - 1s the entry is still
# fresh. Locks the strict-less-than expiry comparison.
# --------------------------------------------------------------------------
def test_ttl_boundary_just_before_expiry_is_a_hit(monkeypatch):
    monkeypatch.setenv("RESEARCH_CACHE_TTL_HOURS", "24")

    ticks = [1000.0]
    monkeypatch.setattr(
        "app.services.search.cache.time.time", lambda: ticks[0]
    )

    cache = QueryCache()
    cache.set("q", [R1])

    # Advance to one second before the 24h TTL elapses.
    ticks[0] += 24 * 3600 - 1

    assert cache.get("q") == [R1]


# --------------------------------------------------------------------------
# Scenario 10 — TTL boundary plus one second: entry has expired.
# Symmetric lock with Scenario 9.
# --------------------------------------------------------------------------
def test_ttl_boundary_just_after_expiry_is_a_miss(monkeypatch):
    monkeypatch.setenv("RESEARCH_CACHE_TTL_HOURS", "24")

    ticks = [1000.0]
    monkeypatch.setattr(
        "app.services.search.cache.time.time", lambda: ticks[0]
    )

    cache = QueryCache()
    cache.set("q", [R1])

    # Advance one second past the 24h TTL.
    ticks[0] += 24 * 3600 + 1

    assert cache.get("q") is None


# --------------------------------------------------------------------------
# Scenario 11 — set overwrites a previous entry under the same key.
# --------------------------------------------------------------------------
def test_set_overwrites_previous_entry(monkeypatch):
    monkeypatch.delenv("RESEARCH_CACHE_TTL_HOURS", raising=False)

    cache = QueryCache()
    cache.set("q", [R1])
    cache.set("q", [R2])

    assert cache.get("q") == [R2]
    assert cache.get("q") != [R1]


# --------------------------------------------------------------------------
# Scenario 12 — Custom TTL via env is honoured at construction time.
# A 1-hour TTL with a 2-hour advance produces a miss. AC-6 lock
# (TTL is read at __init__, not at call time).
# --------------------------------------------------------------------------
def test_custom_ttl_via_env_is_honored(monkeypatch):
    # The setenv MUST happen before QueryCache() is constructed —
    # the cache reads RESEARCH_CACHE_TTL_HOURS once at __init__.
    monkeypatch.setenv("RESEARCH_CACHE_TTL_HOURS", "1")

    ticks = [1000.0]
    monkeypatch.setattr(
        "app.services.search.cache.time.time", lambda: ticks[0]
    )

    cache = QueryCache()
    cache.set("q", [R1])

    # Advance two hours plus one second — comfortably past the 1h TTL.
    ticks[0] += 2 * 3600 + 1

    assert cache.get("q") is None


# --------------------------------------------------------------------------
# Scenario 13 — Default TTL is 24 hours expressed in seconds. Direct
# introspection of the private ``_ttl_seconds`` field locks the
# constructor-time read site (a refactor that moves the read to
# ``get`` / ``set`` would have to keep this field exactly as-is).
# --------------------------------------------------------------------------
def test_default_ttl_is_24_hours_in_seconds(monkeypatch):
    monkeypatch.delenv("RESEARCH_CACHE_TTL_HOURS", raising=False)

    cache = QueryCache()

    assert cache._ttl_seconds == 24 * 3600

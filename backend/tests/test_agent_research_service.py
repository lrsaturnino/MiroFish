"""
Tests for ``AgentResearchService`` — per-agent search-budget formula.

The service exposes a pure-arithmetic ``@staticmethod budget(influence_weight,
activity_level) -> int`` that returns 0 for silent observers
(``activity_level < 0.1``) and otherwise returns
``clamp(round(RESEARCH_BASE_K * influence_weight), 1, MAX_RESEARCH_QUERIES_PER_AGENT)``.

Both env vars are read **inside** the staticmethod body, so
``monkeypatch.setenv`` works deterministically for every scenario. The
autouse ``_clean_llm_env`` fixture in ``conftest.py`` clears only LLM
env vars; it does NOT clear ``RESEARCH_BASE_K`` or
``MAX_RESEARCH_QUERIES_PER_AGENT``. Tests that need the env defaults
(scenario #12) therefore ``delenv`` both vars explicitly to avoid
ambient-state coupling on developer machines.

Rounding semantics: Python 3's built-in ``round()`` is banker's rounding
(round-half-to-even). ``round(4.5) == 4`` — scenario #3 locks this
behavior so the production code may use ``round`` directly without a
manual ``math.floor(x + 0.5)`` workaround.

The test class is ``class TestBudget:`` — sibling classes ``TestRun``
and ``TestIsEnabled`` will be added to this file by future tasks. The
class-based shape keeps the multi-task test file consistent.
"""

import pytest

from app.services.agent_research_service import AgentResearchService


# Env var names — kept as module-level constants so any rename
# surfaces every reference at once.
ENV_BASE_K = "RESEARCH_BASE_K"
ENV_MAX_Q = "MAX_RESEARCH_QUERIES_PER_AGENT"


class TestBudget:
    """Boundary tests for ``AgentResearchService.budget`` — pure arithmetic helper."""

    # ----------------------------------------------------------------------
    # Scenario 1 — Active agent, mid-range weight
    # ``round(3 * 2.0) == 6`` — neither floor nor cap applies.
    # ----------------------------------------------------------------------
    def test_active_agent_mid_range_weight(self, monkeypatch):
        monkeypatch.setenv(ENV_BASE_K, "3")
        monkeypatch.setenv(ENV_MAX_Q, "20")

        assert AgentResearchService.budget(2.0, 0.5) == 6

    # ----------------------------------------------------------------------
    # Scenario 2 — Active agent, weight 1.0 (identity case)
    # ``round(3 * 1.0) == 3`` — sanity-check that BASE_K is honored.
    # ----------------------------------------------------------------------
    def test_active_agent_weight_one(self, monkeypatch):
        monkeypatch.setenv(ENV_BASE_K, "3")

        assert AgentResearchService.budget(1.0, 0.5) == 3

    # ----------------------------------------------------------------------
    # Scenario 3 — Banker's rounding (Decision D1)
    # Python's built-in ``round(4.5) == 4`` (round-half-to-even).
    # The production code uses ``round`` directly; this test locks that
    # behavior so a future ``math.floor(x + 0.5)`` swap surfaces here.
    # ----------------------------------------------------------------------
    def test_active_agent_bankers_rounding_half_to_even(self, monkeypatch):
        monkeypatch.setenv(ENV_BASE_K, "3")

        assert AgentResearchService.budget(1.5, 0.5) == 4

    # ----------------------------------------------------------------------
    # Scenario 4 — Active agent, MAX cap engaged
    # ``round(3 * 10.0) == 30`` — clamped to MAX=20.
    # ----------------------------------------------------------------------
    def test_active_agent_max_cap(self, monkeypatch):
        monkeypatch.setenv(ENV_BASE_K, "3")
        monkeypatch.setenv(ENV_MAX_Q, "20")

        assert AgentResearchService.budget(10.0, 0.5) == 20

    # ----------------------------------------------------------------------
    # Scenario 5 — Active agent, floor enforced (zero weight)
    # An active agent with weight 0.0 still gets 1 query — the formula
    # floors at 1 to guarantee at least one search per active agent.
    # ----------------------------------------------------------------------
    def test_active_agent_floor_enforced_zero_weight(self, monkeypatch):
        monkeypatch.setenv(ENV_BASE_K, "3")

        assert AgentResearchService.budget(0.0, 0.5) == 1

    # ----------------------------------------------------------------------
    # Scenario 6 — Active agent, near-zero weight rounds to zero, then floor
    # ``round(3 * 0.1) == round(0.3) == 0`` — floor of 1 wins.
    # ----------------------------------------------------------------------
    def test_active_agent_floor_enforced_near_zero_weight(self, monkeypatch):
        monkeypatch.setenv(ENV_BASE_K, "3")

        assert AgentResearchService.budget(0.1, 0.5) == 1

    # ----------------------------------------------------------------------
    # Scenario 7 — Silent observer skipped
    # ``activity_level < 0.1`` returns 0 BEFORE any weight arithmetic.
    # The huge ``influence_weight`` proves the skip happens first
    # (otherwise the formula would still produce 20 via the cap).
    # ----------------------------------------------------------------------
    def test_silent_observer_skipped(self, monkeypatch):
        assert AgentResearchService.budget(99.9, 0.05) == 0

    # ----------------------------------------------------------------------
    # Scenario 8 — Boundary ``activity_level == 0.1`` is ACTIVE
    # The skip predicate is strict less-than. Equal-to-0.1 is active.
    # ----------------------------------------------------------------------
    def test_boundary_activity_level_exactly_one_tenth_is_active(self, monkeypatch):
        monkeypatch.setenv(ENV_BASE_K, "3")

        assert AgentResearchService.budget(1.0, 0.1) == 3

    # ----------------------------------------------------------------------
    # Scenario 9 — Boundary ``activity_level == 0.099`` is SKIPPED
    # Just below the threshold — skip predicate fires.
    # ----------------------------------------------------------------------
    def test_boundary_activity_level_just_below_one_tenth_is_skipped(self):
        assert AgentResearchService.budget(1.0, 0.099) == 0

    # ----------------------------------------------------------------------
    # Scenario 10 — Custom ``RESEARCH_BASE_K`` env value
    # ``round(5 * 2.0) == 10`` — proves env is read at call time.
    # ----------------------------------------------------------------------
    def test_custom_research_base_k_env(self, monkeypatch):
        monkeypatch.setenv(ENV_BASE_K, "5")

        assert AgentResearchService.budget(2.0, 0.5) == 10

    # ----------------------------------------------------------------------
    # Scenario 11 — Custom ``MAX_RESEARCH_QUERIES_PER_AGENT`` env value
    # ``round(3 * 10.0) == 30`` — clamped to a low MAX=4.
    # ----------------------------------------------------------------------
    def test_custom_max_queries_per_agent_env(self, monkeypatch):
        monkeypatch.setenv(ENV_BASE_K, "3")
        monkeypatch.setenv(ENV_MAX_Q, "4")

        assert AgentResearchService.budget(10.0, 0.5) == 4

    # ----------------------------------------------------------------------
    # Scenario 12 — Default env values (both unset)
    # ``RESEARCH_BASE_K`` defaults to 3, ``MAX`` defaults to 20.
    # ``delenv`` both vars explicitly: the autouse ``_clean_llm_env``
    # fixture in conftest.py does NOT clear research env vars (Risk R3).
    # On a developer machine where these are set in the shell, omitting
    # the explicit delenv would silently flip this assertion.
    # ----------------------------------------------------------------------
    def test_default_env_values_when_both_unset(self, monkeypatch):
        monkeypatch.delenv(ENV_BASE_K, raising=False)
        monkeypatch.delenv(ENV_MAX_Q, raising=False)

        assert AgentResearchService.budget(2.0, 0.5) == 6

    # ----------------------------------------------------------------------
    # Scenario 13 — Negative ``influence_weight`` (defensive against bad data)
    # ``round(3 * -1.0) == -3`` — clamp's floor of 1 sanitizes to 1.
    # ----------------------------------------------------------------------
    def test_negative_influence_weight_clamped_to_floor(self, monkeypatch):
        monkeypatch.setenv(ENV_BASE_K, "3")

        assert AgentResearchService.budget(-1.0, 0.5) == 1

    # ----------------------------------------------------------------------
    # Scenario 14 — ``activity_level == 0.0`` is SKIPPED
    # Hard-zero activity returns 0 (skip predicate ``< 0.1`` includes 0.0).
    # ----------------------------------------------------------------------
    def test_zero_activity_level_skipped(self):
        assert AgentResearchService.budget(1.0, 0.0) == 0

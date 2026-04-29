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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.services.agent_research_service import AgentResearchService
from app.services.search.cache import QueryCache


# Env var names — kept as module-level constants so any rename
# surfaces every reference at once.
ENV_BASE_K = "RESEARCH_BASE_K"
ENV_MAX_Q = "MAX_RESEARCH_QUERIES_PER_AGENT"


# Sample SearchResult-shaped dicts used across TestRun scenarios.
# SearchResult is a TypedDict so a plain dict satisfies the type. Using
# a module-level constant keeps the per-test setup compact and makes any
# field-shape change surface in one place.
SAMPLE_RESULT = {"title": "t", "url": "https://u", "snippet": "s"}


def _make_profile(user_id, persona, interested_topics, bio):
    """Lightweight ``OasisAgentProfile`` stub for orchestration tests.

    Returns a ``SimpleNamespace`` carrying only the four attributes
    ``AgentResearchService.run`` reads from a profile (``user_id``,
    ``persona``, ``interested_topics``, ``bio``). Avoids importing the
    real dataclass which transitively pulls ``openai`` and ``zep_cloud``
    at module load.
    """
    return SimpleNamespace(
        user_id=user_id,
        persona=persona,
        interested_topics=interested_topics,
        bio=bio,
    )


def _make_activity(activity_level, influence_weight, stance):
    """Lightweight ``AgentActivityConfig`` stub for orchestration tests.

    Returns a ``SimpleNamespace`` carrying only the three attributes
    ``run`` reads from an activity record. Same import-isolation
    rationale as ``_make_profile``.
    """
    return SimpleNamespace(
        activity_level=activity_level,
        influence_weight=influence_weight,
        stance=stance,
    )


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


class TestRun:
    """Orchestration tests for ``AgentResearchService.run`` — mocked LLM + provider.

    Every test here exercises the per-agent loop body in
    ``run(project_id, profiles, activity_by_user_id, topic_seed)``. The
    contract (mutation-in-place persona append, per-agent error
    isolation, cache reuse across agents, configurable persona-cap, same
    list-reference return) is locked by ``T-009.md`` and the
    pre-implementation strategy doc.

    Mock convention:

        * ``mock_llm = MagicMock()`` — ``chat_json`` returns a dict
          shaped ``{"queries": [...]}`` (the SDK contract — JSON-mode
          forces a JSON object, never a bare list). ``chat`` returns the
          synthesis string verbatim.
        * ``mock_provider = MagicMock()`` — ``search`` returns
          ``[SAMPLE_RESULT]`` per call (length-1 list satisfies ``k=1``).
        * ``cache = QueryCache()`` — REAL fresh instance per test. The
          cache integration is exercised end-to-end; ``T-007`` already
          locks the cache's normalisation and TTL behaviour separately.
        * Profile / activity stubs come from the module-level
          ``_make_profile`` / ``_make_activity`` helpers (``SimpleNamespace``).

    Tests do NOT use ``pytest.raises`` for the error-isolation scenarios
    (3, 4, 5). The contract is that ``run`` returns normally and the
    captured error lives on ``svc._last_run_errors``; wrapping in
    ``pytest.raises`` would bake the wrong contract (re-raise) into the
    test.
    """

    # ----------------------------------------------------------------------
    # Scenario 1 — Happy path single agent
    # ``BASE_K=3``, weight=1.0 → K=3. LLM mocked to return three queries
    # then a one-line synthesis; provider returns one result per query.
    # Asserts: persona is mutated (original prefix preserved + synthesis
    # appended), LLM called twice (1 ``chat_json`` + 1 ``chat``),
    # provider called K=3 times.
    # ----------------------------------------------------------------------
    def test_happy_path_single_agent(self, monkeypatch):
        monkeypatch.setenv(ENV_BASE_K, "3")

        mock_llm = MagicMock()
        mock_llm.chat_json.return_value = {"queries": ["q1", "q2", "q3"]}
        mock_llm.chat.return_value = "Synthesized opinion."
        mock_provider = MagicMock()
        mock_provider.search.return_value = [SAMPLE_RESULT]
        cache = QueryCache()

        svc = AgentResearchService(
            search_provider=mock_provider,
            cache=cache,
            llm_client=mock_llm,
        )
        profile = _make_profile(
            user_id=1,
            persona="ORIGINAL ",
            interested_topics=["climate"],
            bio="bio",
        )
        activity = _make_activity(
            activity_level=0.5,
            influence_weight=1.0,
            stance="supportive",
        )

        svc.run("p1", [profile], {1: activity}, topic_seed="topic")

        assert profile.persona.startswith("ORIGINAL ")
        assert "Synthesized opinion." in profile.persona
        assert mock_llm.chat_json.call_count == 1
        assert mock_llm.chat.call_count == 1
        assert mock_provider.search.call_count == 3

    # ----------------------------------------------------------------------
    # Scenario 2 — Skip silent observer
    # ``activity_level=0.05`` < 0.1 → ``budget`` returns 0 → agent block
    # short-circuits BEFORE any LLM or provider call. Persona is
    # untouched.
    # ----------------------------------------------------------------------
    def test_skip_silent_observer(self):
        mock_llm = MagicMock()
        mock_provider = MagicMock()
        cache = QueryCache()

        svc = AgentResearchService(
            search_provider=mock_provider,
            cache=cache,
            llm_client=mock_llm,
        )
        profile = _make_profile(
            user_id=1,
            persona="ORIGINAL",
            interested_topics=["climate"],
            bio="bio",
        )
        activity = _make_activity(
            activity_level=0.05,
            influence_weight=1.0,
            stance="supportive",
        )

        svc.run("p1", [profile], {1: activity}, topic_seed="topic")

        assert profile.persona == "ORIGINAL"
        assert mock_llm.chat_json.call_count == 0
        assert mock_llm.chat.call_count == 0
        assert mock_provider.search.call_count == 0

    # ----------------------------------------------------------------------
    # Scenario 3 — Per-agent error isolation: search fails for agent A
    # Two active profiles. Provider raises on every call for A's queries;
    # B's queries succeed normally. A's persona stays unchanged; B's
    # persona is mutated; ``_last_run_errors`` contains an entry for A.
    # ----------------------------------------------------------------------
    def test_error_isolation_search_fails(self, monkeypatch):
        monkeypatch.setenv(ENV_BASE_K, "3")

        mock_llm = MagicMock()
        # Both agents get identical query-gen result shape but disjoint
        # query strings so cache cannot hide the per-agent failure mode.
        mock_llm.chat_json.side_effect = [
            {"queries": ["a1", "a2", "a3"]},
            {"queries": ["b1", "b2", "b3"]},
        ]
        mock_llm.chat.return_value = "B synthesis."
        mock_provider = MagicMock()

        def _provider_side_effect(query, k=1):
            if query.startswith("a"):
                raise RuntimeError("network down for A")
            return [SAMPLE_RESULT]

        mock_provider.search.side_effect = _provider_side_effect
        cache = QueryCache()

        svc = AgentResearchService(
            search_provider=mock_provider,
            cache=cache,
            llm_client=mock_llm,
        )
        profile_a = _make_profile(
            user_id=1, persona="A_ORIG", interested_topics=["x"], bio="bio_a"
        )
        profile_b = _make_profile(
            user_id=2, persona="B_ORIG", interested_topics=["y"], bio="bio_b"
        )
        activity_a = _make_activity(0.5, 1.0, "supportive")
        activity_b = _make_activity(0.5, 1.0, "supportive")

        svc.run(
            "p1",
            [profile_a, profile_b],
            {1: activity_a, 2: activity_b},
            topic_seed="topic",
        )

        assert profile_a.persona == "A_ORIG"
        assert "B synthesis." in profile_b.persona
        assert len(svc._last_run_errors) == 1
        assert svc._last_run_errors[0]["agent_id"] == 1
        assert "error" in svc._last_run_errors[0]

    # ----------------------------------------------------------------------
    # Scenario 4 — Per-agent error isolation: query-gen LLM fails for A
    # First ``chat_json`` call raises; second returns normally. A's
    # persona stays unchanged; B's persona is mutated; the error for A
    # is captured.
    # ----------------------------------------------------------------------
    def test_error_isolation_query_gen_fails(self, monkeypatch):
        monkeypatch.setenv(ENV_BASE_K, "3")

        mock_llm = MagicMock()
        mock_llm.chat_json.side_effect = [
            RuntimeError("llm down for A"),
            {"queries": ["b1", "b2", "b3"]},
        ]
        mock_llm.chat.return_value = "B synthesis."
        mock_provider = MagicMock()
        mock_provider.search.return_value = [SAMPLE_RESULT]
        cache = QueryCache()

        svc = AgentResearchService(
            search_provider=mock_provider,
            cache=cache,
            llm_client=mock_llm,
        )
        profile_a = _make_profile(
            user_id=1, persona="A_ORIG", interested_topics=["x"], bio="bio_a"
        )
        profile_b = _make_profile(
            user_id=2, persona="B_ORIG", interested_topics=["y"], bio="bio_b"
        )
        activity_a = _make_activity(0.5, 1.0, "supportive")
        activity_b = _make_activity(0.5, 1.0, "supportive")

        svc.run(
            "p1",
            [profile_a, profile_b],
            {1: activity_a, 2: activity_b},
            topic_seed="topic",
        )

        assert profile_a.persona == "A_ORIG"
        assert "B synthesis." in profile_b.persona
        assert len(svc._last_run_errors) == 1
        assert svc._last_run_errors[0]["agent_id"] == 1

    # ----------------------------------------------------------------------
    # Scenario 5 — Per-agent error isolation: synthesis LLM fails for A
    # Both agents pass query-gen; ``chat`` raises for A then succeeds for
    # B. A's persona stays unchanged; B's persona is mutated; A's error
    # is captured.
    # ----------------------------------------------------------------------
    def test_error_isolation_synthesis_fails(self, monkeypatch):
        monkeypatch.setenv(ENV_BASE_K, "3")

        mock_llm = MagicMock()
        mock_llm.chat_json.side_effect = [
            {"queries": ["a1", "a2", "a3"]},
            {"queries": ["b1", "b2", "b3"]},
        ]
        mock_llm.chat.side_effect = [
            RuntimeError("synth down for A"),
            "B synthesis.",
        ]
        mock_provider = MagicMock()
        mock_provider.search.return_value = [SAMPLE_RESULT]
        cache = QueryCache()

        svc = AgentResearchService(
            search_provider=mock_provider,
            cache=cache,
            llm_client=mock_llm,
        )
        profile_a = _make_profile(
            user_id=1, persona="A_ORIG", interested_topics=["x"], bio="bio_a"
        )
        profile_b = _make_profile(
            user_id=2, persona="B_ORIG", interested_topics=["y"], bio="bio_b"
        )
        activity_a = _make_activity(0.5, 1.0, "supportive")
        activity_b = _make_activity(0.5, 1.0, "supportive")

        svc.run(
            "p1",
            [profile_a, profile_b],
            {1: activity_a, 2: activity_b},
            topic_seed="topic",
        )

        assert profile_a.persona == "A_ORIG"
        assert "B synthesis." in profile_b.persona
        assert len(svc._last_run_errors) == 1
        assert svc._last_run_errors[0]["agent_id"] == 1

    # ----------------------------------------------------------------------
    # Scenario 6 — Cache reuse across agents
    # Two active profiles; query-gen returns the same single query for
    # both. Provider is called ONCE total (the second agent's lookup
    # hits the cache populated by the first agent's call).
    # ----------------------------------------------------------------------
    def test_cache_reuse_across_agents(self, monkeypatch):
        monkeypatch.setenv(ENV_BASE_K, "1")  # one query per agent

        mock_llm = MagicMock()
        # ``return_value`` (not ``side_effect``): both agents get the
        # exact same singleton query → identical normalised cache key.
        mock_llm.chat_json.return_value = {"queries": ["climate change"]}
        mock_llm.chat.return_value = "Synthesis."
        mock_provider = MagicMock()
        mock_provider.search.return_value = [SAMPLE_RESULT]
        cache = QueryCache()

        svc = AgentResearchService(
            search_provider=mock_provider,
            cache=cache,
            llm_client=mock_llm,
        )
        profile_a = _make_profile(
            user_id=1, persona="A", interested_topics=["x"], bio="bio_a"
        )
        profile_b = _make_profile(
            user_id=2, persona="B", interested_topics=["y"], bio="bio_b"
        )
        activity_a = _make_activity(0.5, 1.0, "supportive")
        activity_b = _make_activity(0.5, 1.0, "supportive")

        svc.run(
            "p1",
            [profile_a, profile_b],
            {1: activity_a, 2: activity_b},
            topic_seed="topic",
        )

        assert mock_provider.search.call_count == 1

    # ----------------------------------------------------------------------
    # Scenario 7 — Persona cap enforcement (default 600 chars)
    # Synthesis returns a 5000-char string; the appended block (after
    # truncation) MUST be ≤ 600 chars. The total persona length grows by
    # at most ``600 + 1`` (the newline separator + truncated text).
    # ----------------------------------------------------------------------
    def test_persona_cap_enforcement_default(self, monkeypatch):
        monkeypatch.setenv(ENV_BASE_K, "1")

        long_synthesis = "x" * 5000
        mock_llm = MagicMock()
        mock_llm.chat_json.return_value = {"queries": ["q"]}
        mock_llm.chat.return_value = long_synthesis
        mock_provider = MagicMock()
        mock_provider.search.return_value = [SAMPLE_RESULT]
        cache = QueryCache()

        svc = AgentResearchService(
            search_provider=mock_provider,
            cache=cache,
            llm_client=mock_llm,
        )
        profile = _make_profile(
            user_id=1, persona="ORIG", interested_topics=["x"], bio="bio"
        )
        activity = _make_activity(0.5, 1.0, "supportive")

        svc.run("p1", [profile], {1: activity}, topic_seed="topic")

        # Appended block is everything after the original "ORIG" prefix.
        appended = profile.persona[len("ORIG"):]
        # Strip the leading "\n" separator from the appended block before
        # length check — the cap covers the synthesis body only.
        if appended.startswith("\n"):
            appended = appended[1:]
        assert len(appended) <= 600

    # ----------------------------------------------------------------------
    # Scenario 8 — Configurable cap
    # Constructor kwarg ``max_persona_append_chars=200`` overrides the
    # default 600. Same long synthesis input; appended block ≤ 200.
    # ----------------------------------------------------------------------
    def test_persona_cap_configurable(self, monkeypatch):
        monkeypatch.setenv(ENV_BASE_K, "1")

        long_synthesis = "y" * 5000
        mock_llm = MagicMock()
        mock_llm.chat_json.return_value = {"queries": ["q"]}
        mock_llm.chat.return_value = long_synthesis
        mock_provider = MagicMock()
        mock_provider.search.return_value = [SAMPLE_RESULT]
        cache = QueryCache()

        svc = AgentResearchService(
            search_provider=mock_provider,
            cache=cache,
            llm_client=mock_llm,
            max_persona_append_chars=200,
        )
        profile = _make_profile(
            user_id=1, persona="ORIG", interested_topics=["x"], bio="bio"
        )
        activity = _make_activity(0.5, 1.0, "supportive")

        svc.run("p1", [profile], {1: activity}, topic_seed="topic")

        appended = profile.persona[len("ORIG"):]
        if appended.startswith("\n"):
            appended = appended[1:]
        assert len(appended) <= 200

    # ----------------------------------------------------------------------
    # Scenario 9 — Returns same ``profiles`` list reference
    # AC-4: ``run`` returns the SAME list object passed in (not a copy).
    # Python identity check ``result is profiles`` is the contract.
    # ----------------------------------------------------------------------
    def test_returns_same_list_reference(self, monkeypatch):
        monkeypatch.setenv(ENV_BASE_K, "1")

        mock_llm = MagicMock()
        mock_llm.chat_json.return_value = {"queries": ["q"]}
        mock_llm.chat.return_value = "Synthesis."
        mock_provider = MagicMock()
        mock_provider.search.return_value = [SAMPLE_RESULT]
        cache = QueryCache()

        svc = AgentResearchService(
            search_provider=mock_provider,
            cache=cache,
            llm_client=mock_llm,
        )
        profile = _make_profile(
            user_id=1, persona="ORIG", interested_topics=["x"], bio="bio"
        )
        activity = _make_activity(0.5, 1.0, "supportive")
        profiles = [profile]

        result = svc.run("p1", profiles, {1: activity}, topic_seed="topic")

        assert result is profiles

    # ----------------------------------------------------------------------
    # Scenario 10 — All-skipped run
    # Every profile has ``activity_level=0.05`` → all skipped. LLM and
    # provider are never called; profiles list is returned unchanged
    # (identity); every persona is unchanged.
    # ----------------------------------------------------------------------
    def test_all_skipped_run(self):
        mock_llm = MagicMock()
        mock_provider = MagicMock()
        cache = QueryCache()

        svc = AgentResearchService(
            search_provider=mock_provider,
            cache=cache,
            llm_client=mock_llm,
        )
        profile_a = _make_profile(
            user_id=1, persona="A", interested_topics=["x"], bio="bio_a"
        )
        profile_b = _make_profile(
            user_id=2, persona="B", interested_topics=["y"], bio="bio_b"
        )
        activity_a = _make_activity(0.05, 1.0, "supportive")
        activity_b = _make_activity(0.05, 1.0, "supportive")
        profiles = [profile_a, profile_b]

        result = svc.run(
            "p1", profiles, {1: activity_a, 2: activity_b}, topic_seed="topic"
        )

        assert mock_llm.chat_json.call_count == 0
        assert mock_llm.chat.call_count == 0
        assert mock_provider.search.call_count == 0
        assert profile_a.persona == "A"
        assert profile_b.persona == "B"
        assert result is profiles

    # ----------------------------------------------------------------------
    # Scenario 11 — K computed per agent
    # Profile A weight=1.0 → K=3; profile B weight=3.0 → K=9. With
    # DISJOINT query strings (so cache cannot hide misses) total provider
    # calls == 3 + 9 == 12. ``side_effect`` returns agent-specific lists.
    # ----------------------------------------------------------------------
    def test_k_computed_per_agent(self, monkeypatch):
        monkeypatch.setenv(ENV_BASE_K, "3")
        monkeypatch.setenv(ENV_MAX_Q, "20")

        mock_llm = MagicMock()
        mock_llm.chat_json.side_effect = [
            {"queries": ["a1", "a2", "a3"]},
            {
                "queries": [
                    "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9",
                ]
            },
        ]
        mock_llm.chat.side_effect = ["A_synth.", "B_synth."]
        mock_provider = MagicMock()
        mock_provider.search.return_value = [SAMPLE_RESULT]
        cache = QueryCache()

        svc = AgentResearchService(
            search_provider=mock_provider,
            cache=cache,
            llm_client=mock_llm,
        )
        profile_a = _make_profile(
            user_id=1, persona="A", interested_topics=["x"], bio="bio_a"
        )
        profile_b = _make_profile(
            user_id=2, persona="B", interested_topics=["y"], bio="bio_b"
        )
        activity_a = _make_activity(0.5, 1.0, "supportive")
        activity_b = _make_activity(0.5, 3.0, "supportive")

        svc.run(
            "p1",
            [profile_a, profile_b],
            {1: activity_a, 2: activity_b},
            topic_seed="topic",
        )

        assert mock_provider.search.call_count == 12

    # ----------------------------------------------------------------------
    # Scenario 12 — Mutation is "append", not "replace"
    # Original persona ``"ORIGINAL "`` is preserved as a prefix; the
    # synthesis ``"OPINION"`` ends the persona. The ``"\n"`` separator
    # sits between them; ``endswith("OPINION")`` proves the synthesis is
    # the trailing segment (Decision D14).
    # ----------------------------------------------------------------------
    def test_mutation_is_append_not_replace(self, monkeypatch):
        monkeypatch.setenv(ENV_BASE_K, "1")

        mock_llm = MagicMock()
        mock_llm.chat_json.return_value = {"queries": ["q"]}
        mock_llm.chat.return_value = "OPINION"
        mock_provider = MagicMock()
        mock_provider.search.return_value = [SAMPLE_RESULT]
        cache = QueryCache()

        svc = AgentResearchService(
            search_provider=mock_provider,
            cache=cache,
            llm_client=mock_llm,
        )
        profile = _make_profile(
            user_id=1,
            persona="ORIGINAL ",
            interested_topics=["x"],
            bio="bio",
        )
        activity = _make_activity(0.5, 1.0, "supportive")

        svc.run("p1", [profile], {1: activity}, topic_seed="topic")

        assert profile.persona.startswith("ORIGINAL ")
        assert profile.persona.endswith("OPINION")

    # ----------------------------------------------------------------------
    # Scenario 13 — Confirmation-bias query shape (smoke)
    # Profile carries ``stance="opposing"`` and ``interested_topics=["X"]``.
    # The query-gen prompt MUST literally contain both substrings (no
    # case-folding, no transformation — Decision D4). Inspects
    # ``mock_llm.chat_json.call_args.kwargs["messages"]`` (Decision D13:
    # the call site uses the ``messages=`` keyword form).
    # ----------------------------------------------------------------------
    def test_confirmation_bias_query_shape_smoke(self, monkeypatch):
        monkeypatch.setenv(ENV_BASE_K, "1")

        mock_llm = MagicMock()
        mock_llm.chat_json.return_value = {"queries": ["q"]}
        mock_llm.chat.return_value = "Synthesis."
        mock_provider = MagicMock()
        mock_provider.search.return_value = [SAMPLE_RESULT]
        cache = QueryCache()

        svc = AgentResearchService(
            search_provider=mock_provider,
            cache=cache,
            llm_client=mock_llm,
        )
        profile = _make_profile(
            user_id=1,
            persona="ORIG",
            interested_topics=["X"],
            bio="some bio",
        )
        activity = _make_activity(0.5, 1.0, "opposing")

        svc.run("p1", [profile], {1: activity}, topic_seed="topic")

        messages = mock_llm.chat_json.call_args.kwargs["messages"]
        combined = " ".join(m["content"] for m in messages)
        assert "opposing" in combined
        assert "X" in combined

"""
Host-site integration tests for pipeline Step 2.5 (agent web research).

These tests pin the contract that ``SimulationManager.prepare_simulation``:

    1. Generates Agent profiles via ``OasisProfileGenerator``.
    2. Generates ``AgentActivityConfig`` records via ``SimulationConfigGenerator``.
    3. Calls ``AgentResearchService.run`` (when enabled), mutating profiles.
    4. Persists profiles via ``OasisProfileGenerator.save_profiles``.

The host-side wiring is what we test here — the per-agent loop body of
``AgentResearchService`` is covered exhaustively in
``test_agent_research_service.py``. We mock the service at the host-side
seam in unit tests and use the real service (with mocked LLM + mocked
search) in the single integration test (#2).

Important env hygiene:

    The autouse ``_clean_llm_env`` fixture in ``conftest.py`` defaults
    ``RESEARCH_ENABLED=true`` and ``TAVILY_API_KEY=test-key`` so the
    research gate is open by default. Tests that exercise the disabled
    path (#1, #6) MUST ``delenv`` both explicitly. The fixture also
    purges every LLM env var — tests that rely on ``LLMClient(role="builder")``
    construction at the host site MUST either ``setenv("LLM_API_KEY", ...)``
    or patch ``LLMClient`` at the host-site seam.

Patch-path discipline:

    Every collaborator (``AgentResearchService``, ``TavilyProvider``,
    ``QueryCache``, ``LLMClient``, ``ResearchJsonlLogger``) is patched at
    ``app.services.simulation_manager.<name>`` — the host module is
    where the imports get bound. Patching at the definition module
    leaves the host's local binding pointing at the original symbol.

Performance target: full file completes in <5 seconds. All filesystem
I/O is scoped to ``tmp_path`` via the ``tmp_sim_dir`` fixture.
"""

from __future__ import annotations

import json
import logging
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Import the host module once so per-test patches can target attributes
# that do NOT yet exist on it (the 5 research collaborators are imported
# by the GREEN-phase host edit). ``monkeypatch.setattr(module, name,
# value, raising=False)`` is the only path that creates a missing
# attribute — the dotted-string form requires the attribute to exist.
from app.services import simulation_manager as sm_module


def _patch_host_attr(monkeypatch, name: str, value):
    """Patch ``app.services.simulation_manager.<name>`` even when the
    attribute does not yet exist on the module (RED-phase tests run
    before the GREEN-phase imports are added)."""
    monkeypatch.setattr(sm_module, name, value, raising=False)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

ENV_RESEARCH = "RESEARCH_ENABLED"
ENV_TAVILY = "TAVILY_API_KEY"
ENV_LLM_API_KEY = "LLM_API_KEY"

# Sentinel topic seed — confirms exact-string forwarding from
# ``simulation_requirement`` to ``AgentResearchService.run``'s ``topic_seed``.
PROBE_TOPIC_SEED = "PROBE-TOPIC-SEED-7c4f"

# Sentinel synthesized opinion — confirms persona mutation in #2.
PROBE_OPINION = "PROBE-OPINION-7c4f"


# ---------------------------------------------------------------------------
# Stub builders — keep tests independent of dataclass import chains
# (``OasisAgentProfile`` transitively imports ``openai`` and ``zep_cloud``).
# ---------------------------------------------------------------------------


def _make_profile(user_id: int, persona: str, interested_topics, bio: str):
    """Lightweight ``OasisAgentProfile`` stub.

    Carries only the four attributes ``AgentResearchService.run`` and
    ``OasisProfileGenerator.save_profiles`` read from a profile.
    """
    return SimpleNamespace(
        user_id=user_id,
        persona=persona,
        interested_topics=interested_topics,
        bio=bio,
    )


def _make_activity(
    activity_level: float,
    influence_weight: float,
    stance: str,
    agent_id: int,
):
    """Lightweight ``AgentActivityConfig`` stub.

    The host builds ``activity_by_user_id = {ac.agent_id: ac for ac in
    sim_params.agent_configs}`` — the dict-comp reads ``agent_id``.
    Service-side reads include ``activity_level``, ``influence_weight``,
    and ``stance``.
    """
    return SimpleNamespace(
        agent_id=agent_id,
        activity_level=activity_level,
        influence_weight=influence_weight,
        stance=stance,
    )


def _make_filtered_entities(count: int):
    """Build a ``FilteredEntities``-shaped stub for the Phase-1 mock.

    The host reads ``.filtered_count`` (truthiness gate at line 402),
    ``.entity_types`` (assigned to ``state.entity_types``), and
    ``.entities`` (passed downstream).
    """
    entities = [SimpleNamespace(name=f"E{i}", uuid=f"u{i}") for i in range(count)]
    return SimpleNamespace(
        entities=entities,
        entity_types={"Student"},
        filtered_count=count,
        total_count=count,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_sim_dir(tmp_path, monkeypatch):
    """Redirect ``SimulationManager.SIMULATION_DATA_DIR`` to ``tmp_path``.

    The class attribute is bound at module import (no constructor
    injection), so ``monkeypatch.setattr`` is the only clean way to
    redirect it. Yields the ``tmp_path`` for tests that inspect files.
    """
    from app.services.simulation_manager import SimulationManager

    monkeypatch.setattr(SimulationManager, "SIMULATION_DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def seeded_manager(tmp_sim_dir):
    """Create a ``SimulationManager`` with one seeded simulation.

    ``prepare_simulation`` raises if ``state`` is missing
    (line 367-368), so every test must seed via ``create_simulation``
    before driving the host flow. Returns ``(manager, simulation_id)``.
    """
    from app.services.simulation_manager import SimulationManager

    manager = SimulationManager()
    state = manager.create_simulation(
        project_id="proj-test",
        graph_id="graph-test",
        enable_twitter=False,
        enable_reddit=True,
    )
    return manager, state.simulation_id


@pytest.fixture
def llm_key_set(monkeypatch):
    """Set ``LLM_API_KEY`` so the host-site ``LLMClient`` constructor
    does not raise ``ValueError("LLM_API_KEY 未配置")``.

    The autouse ``_clean_llm_env`` purges all 12 LLM env vars; tests
    that keep a real ``LLMClient`` at the host need this restored.
    """
    monkeypatch.setenv(ENV_LLM_API_KEY, "test-key")


@pytest.fixture
def patch_phase1_phase2_phase3(monkeypatch):
    """Patch the three boundaries the host calls before Step 2.5.

    Returns the three mocks so individual tests can override their
    ``return_value`` and inspect calls. Defaults provide a 2-profile
    happy path with matching activity configs (one per profile).
    """
    profile_gen_mock = MagicMock()
    config_gen_mock = MagicMock()
    zep_reader_mock = MagicMock()

    # Defaults: 2 profiles, 2 activity configs, ``user_id == agent_id``.
    profiles = [
        _make_profile(
            user_id=0,
            persona="ORIGINAL-0",
            interested_topics=["t0"],
            bio="bio-0",
        ),
        _make_profile(
            user_id=1,
            persona="ORIGINAL-1",
            interested_topics=["t1"],
            bio="bio-1",
        ),
    ]
    profile_gen_mock.return_value.generate_profiles_from_entities.return_value = profiles

    sim_params = SimpleNamespace(
        agent_configs=[
            _make_activity(0.5, 1.0, "supportive", agent_id=0),
            _make_activity(0.5, 1.0, "opposing", agent_id=1),
        ],
        generation_reasoning="test-reasoning",
        to_json=lambda: "{}",
    )
    config_gen_mock.return_value.generate_config.return_value = sim_params

    zep_reader_mock.return_value.filter_defined_entities.return_value = (
        _make_filtered_entities(count=2)
    )

    monkeypatch.setattr(
        "app.services.simulation_manager.OasisProfileGenerator",
        profile_gen_mock,
    )
    monkeypatch.setattr(
        "app.services.simulation_manager.SimulationConfigGenerator",
        config_gen_mock,
    )
    monkeypatch.setattr(
        "app.services.simulation_manager.ZepEntityReader",
        zep_reader_mock,
    )

    return SimpleNamespace(
        profile_gen=profile_gen_mock,
        config_gen=config_gen_mock,
        zep_reader=zep_reader_mock,
        profiles=profiles,
        sim_params=sim_params,
    )


# ---------------------------------------------------------------------------
# Scenario 1 — Disabled gate: profiles handed to save_profiles unchanged
# ---------------------------------------------------------------------------
def test_disabled_gate_profiles_unchanged_at_save(
    seeded_manager,
    patch_phase1_phase2_phase3,
    monkeypatch,
    caplog,
):
    """When ``RESEARCH_ENABLED`` and ``TAVILY_API_KEY`` are unset, the
    research step must short-circuit (one ``WARNING`` log) and the
    ``profiles`` list reaching ``save_profiles`` must carry the original
    ``persona`` strings byte-for-byte.

    Per-impl strategy: do NOT mock ``is_enabled``; let the real
    env-driven gate run. We patch ``LLMClient`` and ``TavilyProvider``
    /``QueryCache`` /``ResearchJsonlLogger`` at the host-site seam so the
    constructor chain does not raise (gate is checked AFTER the
    constructors run).
    """
    monkeypatch.delenv(ENV_RESEARCH, raising=False)
    monkeypatch.delenv(ENV_TAVILY, raising=False)

    # Patch every collaborator the host instantiates so constructors
    # never raise. The real ``AgentResearchService`` is kept so the
    # env-driven gate runs unmocked.
    _patch_host_attr(monkeypatch, "LLMClient", MagicMock())
    _patch_host_attr(monkeypatch, "TavilyProvider", MagicMock())
    _patch_host_attr(monkeypatch, "QueryCache", MagicMock())
    _patch_host_attr(monkeypatch, "ResearchJsonlLogger", MagicMock())

    save_profiles_mock = MagicMock()
    monkeypatch.setattr(
        patch_phase1_phase2_phase3.profile_gen.return_value,
        "save_profiles",
        save_profiles_mock,
    )

    manager, simulation_id = seeded_manager

    # Capture the same-named module logger ``AgentResearchService`` uses
    # for the disabled-path WARNING line.
    caplog.set_level(
        logging.WARNING, logger="app.services.agent_research_service"
    )

    manager.prepare_simulation(
        simulation_id=simulation_id,
        simulation_requirement="topic-seed-disabled",
        document_text="doc",
    )

    save_profiles_mock.assert_called()
    # The kwarg name is ``profiles`` per ``save_profiles``'s signature.
    saved_profiles = save_profiles_mock.call_args.kwargs["profiles"]
    assert [p.persona for p in saved_profiles] == ["ORIGINAL-0", "ORIGINAL-1"]
    # The disabled-path WARNING line must surface from the service.
    assert any(
        "research disabled" in r.message
        for r in caplog.records
        if r.name == "app.services.agent_research_service"
    )


# ---------------------------------------------------------------------------
# Scenario 2 — Enabled gate: persona mutated before save (integration)
# ---------------------------------------------------------------------------
def test_enabled_gate_profiles_mutated_before_save(
    seeded_manager,
    patch_phase1_phase2_phase3,
    monkeypatch,
    llm_key_set,
):
    """With the gate ON and a REAL ``AgentResearchService`` constructed
    at the host site (only its three external collaborators —
    ``TavilyProvider.search``, ``LLMClient.chat``, ``LLMClient.chat_json``
    — are stubbed), the ``profiles`` list reaching ``save_profiles`` must
    carry the synthesized opinion appended after a ``"\\n"`` separator.

    This is the only integration-style test in the file. The other
    seven mock the service at the host-site seam.
    """
    # Default conftest provides the gate ENV — no monkeypatch needed.

    # Stub ``LLMClient`` at the host so its construction does not call
    # the real OpenAI SDK. ``chat_json`` returns the query-gen contract;
    # ``chat`` returns the canned synthesized opinion.
    fake_llm = MagicMock()
    fake_llm.chat_json.return_value = {"queries": ["q1", "q2", "q3"]}
    fake_llm.chat.return_value = PROBE_OPINION
    _patch_host_attr(
        monkeypatch, "LLMClient", MagicMock(return_value=fake_llm)
    )

    # Stub ``TavilyProvider`` to return one canned snippet per query.
    fake_provider = MagicMock()
    fake_provider.search.return_value = [
        {"title": "t", "url": "https://u", "snippet": "s"}
    ]
    _patch_host_attr(
        monkeypatch, "TavilyProvider", MagicMock(return_value=fake_provider)
    )
    # Test #2 uses a REAL ``AgentResearchService`` — it must be in the
    # host's namespace BEFORE ``prepare_simulation`` runs (the host
    # imports it at module top once GREEN is done; here we shim it in
    # via the same helper so RED-phase patching is symmetric).
    from app.services.agent_research_service import (
        AgentResearchService,
        ResearchJsonlLogger,
    )
    from app.services.search.cache import QueryCache as _RealQueryCache

    _patch_host_attr(monkeypatch, "AgentResearchService", AgentResearchService)
    _patch_host_attr(monkeypatch, "ResearchJsonlLogger", ResearchJsonlLogger)
    _patch_host_attr(monkeypatch, "QueryCache", _RealQueryCache)

    save_profiles_mock = MagicMock()
    monkeypatch.setattr(
        patch_phase1_phase2_phase3.profile_gen.return_value,
        "save_profiles",
        save_profiles_mock,
    )

    manager, simulation_id = seeded_manager
    manager.prepare_simulation(
        simulation_id=simulation_id,
        simulation_requirement="topic-seed-enabled",
        document_text="doc",
    )

    save_profiles_mock.assert_called()
    saved_profiles = save_profiles_mock.call_args.kwargs["profiles"]
    # Each profile's persona must have the canned opinion appended after
    # a ``"\n"`` separator (the service's contract per
    # ``agent_research_service.py:247``).
    for prof, original in zip(saved_profiles, ["ORIGINAL-0", "ORIGINAL-1"]):
        assert prof.persona.startswith(original)
        assert prof.persona.endswith("\n" + PROBE_OPINION)


# ---------------------------------------------------------------------------
# Scenario 3 — Call order: profile_gen → research → save
# ---------------------------------------------------------------------------
def test_call_order_profile_gen_then_research_then_save(
    seeded_manager,
    patch_phase1_phase2_phase3,
    monkeypatch,
    llm_key_set,
):
    """Attach mocks for the three sequential boundaries. The recorded
    call list must show ``profile_gen`` BEFORE ``research`` BEFORE
    ``save``. Other intermediate calls (config-gen, progress callbacks)
    are acceptable between them.
    """
    # Mock the entire AgentResearchService class at the host seam so
    # ``run`` is observable and persona mutation is bypassed.
    research_instance_mock = MagicMock()
    # Mimic real-service behaviour: return profiles list reference.
    research_instance_mock.run.side_effect = (
        lambda pid, profiles, abi, ts: profiles
    )
    research_instance_mock.is_enabled.return_value = True
    research_class_mock = MagicMock(return_value=research_instance_mock)
    _patch_host_attr(monkeypatch, "AgentResearchService", research_class_mock)
    # Patch the three remaining collaborators so construction succeeds.
    _patch_host_attr(monkeypatch, "TavilyProvider", MagicMock())
    _patch_host_attr(monkeypatch, "QueryCache", MagicMock())
    _patch_host_attr(monkeypatch, "ResearchJsonlLogger", MagicMock())
    _patch_host_attr(monkeypatch, "LLMClient", MagicMock())

    save_profiles_mock = MagicMock()
    monkeypatch.setattr(
        patch_phase1_phase2_phase3.profile_gen.return_value,
        "save_profiles",
        save_profiles_mock,
    )

    manager_recorder = MagicMock()
    manager_recorder.attach_mock(
        patch_phase1_phase2_phase3.profile_gen.return_value
        .generate_profiles_from_entities,
        "profile_gen",
    )
    manager_recorder.attach_mock(research_instance_mock.run, "research")
    manager_recorder.attach_mock(save_profiles_mock, "save")

    manager, simulation_id = seeded_manager
    manager.prepare_simulation(
        simulation_id=simulation_id,
        simulation_requirement="topic-seed-call-order",
        document_text="doc",
    )

    # Filter to the three named seams; assert ordering.
    names = [c[0] for c in manager_recorder.mock_calls]
    relevant = [n for n in names if n in {"profile_gen", "research", "save"}]
    assert relevant.index("profile_gen") < relevant.index("research")
    assert relevant.index("research") < relevant.index("save")


# ---------------------------------------------------------------------------
# Scenario 4 — Topic seed forwarded from simulation_requirement
# ---------------------------------------------------------------------------
def test_topic_seed_forwarded_from_simulation_requirement(
    seeded_manager,
    patch_phase1_phase2_phase3,
    monkeypatch,
    llm_key_set,
):
    """The 4th arg to ``AgentResearchService.run`` (positional or
    ``topic_seed`` kwarg) must equal ``simulation_requirement`` exactly
    — no trim, no case-fold.
    """
    research_instance_mock = MagicMock()
    research_instance_mock.run.side_effect = (
        lambda pid, profiles, abi, ts: profiles
    )
    research_instance_mock.is_enabled.return_value = True
    _patch_host_attr(
        monkeypatch,
        "AgentResearchService",
        MagicMock(return_value=research_instance_mock),
    )
    _patch_host_attr(monkeypatch, "TavilyProvider", MagicMock())
    _patch_host_attr(monkeypatch, "QueryCache", MagicMock())
    _patch_host_attr(monkeypatch, "ResearchJsonlLogger", MagicMock())
    _patch_host_attr(monkeypatch, "LLMClient", MagicMock())

    monkeypatch.setattr(
        patch_phase1_phase2_phase3.profile_gen.return_value,
        "save_profiles",
        MagicMock(),
    )

    manager, simulation_id = seeded_manager
    manager.prepare_simulation(
        simulation_id=simulation_id,
        simulation_requirement=PROBE_TOPIC_SEED,
        document_text="doc",
    )

    research_instance_mock.run.assert_called_once()
    call = research_instance_mock.run.call_args
    # Accept either positional (4th arg) or kwarg form.
    if "topic_seed" in call.kwargs:
        assert call.kwargs["topic_seed"] == PROBE_TOPIC_SEED
    else:
        assert call.args[3] == PROBE_TOPIC_SEED


# ---------------------------------------------------------------------------
# Scenario 5 — activity_by_user_id construction matches profiles
# ---------------------------------------------------------------------------
def test_activity_by_user_id_construction_matches_profiles(
    seeded_manager,
    patch_phase1_phase2_phase3,
    monkeypatch,
    llm_key_set,
):
    """Override the default 2-profile fixture with three profiles whose
    ``user_id`` ∈ {0, 1, 2} and three matching ``AgentActivityConfig``
    stubs with distinguishable ``stance`` values. The dict passed to
    ``run`` must map every ``user_id`` to its matching activity record.
    """
    profiles = [
        _make_profile(
            user_id=0,
            persona="ORIG-0",
            interested_topics=["t0"],
            bio="b0",
        ),
        _make_profile(
            user_id=1,
            persona="ORIG-1",
            interested_topics=["t1"],
            bio="b1",
        ),
        _make_profile(
            user_id=2,
            persona="ORIG-2",
            interested_topics=["t2"],
            bio="b2",
        ),
    ]
    activities = [
        _make_activity(0.5, 1.0, "supportive", agent_id=0),
        _make_activity(0.5, 1.0, "opposing", agent_id=1),
        _make_activity(0.5, 1.0, "neutral", agent_id=2),
    ]
    patch_phase1_phase2_phase3.profile_gen.return_value \
        .generate_profiles_from_entities.return_value = profiles
    patch_phase1_phase2_phase3.config_gen.return_value \
        .generate_config.return_value = SimpleNamespace(
            agent_configs=activities,
            generation_reasoning="test",
            to_json=lambda: "{}",
        )
    patch_phase1_phase2_phase3.zep_reader.return_value \
        .filter_defined_entities.return_value = _make_filtered_entities(3)

    research_instance_mock = MagicMock()
    research_instance_mock.run.side_effect = (
        lambda pid, profiles, abi, ts: profiles
    )
    research_instance_mock.is_enabled.return_value = True
    _patch_host_attr(
        monkeypatch,
        "AgentResearchService",
        MagicMock(return_value=research_instance_mock),
    )
    _patch_host_attr(monkeypatch, "TavilyProvider", MagicMock())
    _patch_host_attr(monkeypatch, "QueryCache", MagicMock())
    _patch_host_attr(monkeypatch, "ResearchJsonlLogger", MagicMock())
    _patch_host_attr(monkeypatch, "LLMClient", MagicMock())

    monkeypatch.setattr(
        patch_phase1_phase2_phase3.profile_gen.return_value,
        "save_profiles",
        MagicMock(),
    )

    manager, simulation_id = seeded_manager
    manager.prepare_simulation(
        simulation_id=simulation_id,
        simulation_requirement="topic-seed-activity-map",
        document_text="doc",
    )

    research_instance_mock.run.assert_called_once()
    call = research_instance_mock.run.call_args
    # Accept positional (3rd arg) or kwarg form.
    if "activity_by_user_id" in call.kwargs:
        abi = call.kwargs["activity_by_user_id"]
    else:
        abi = call.args[2]
    assert set(abi.keys()) == {0, 1, 2}
    assert abi[1].stance == "opposing"
    assert abi[0].stance == "supportive"
    assert abi[2].stance == "neutral"


# ---------------------------------------------------------------------------
# Scenario 6 — No regression on save path when disabled
# ---------------------------------------------------------------------------
def test_no_regression_on_save_profiles_path_when_disabled(
    seeded_manager,
    patch_phase1_phase2_phase3,
    monkeypatch,
    tmp_sim_dir,
):
    """With the gate disabled, run the full host flow with a REAL
    ``save_profiles`` (writing to ``tmp_path``). The
    ``reddit_profiles.json`` file must contain the canonical record
    keys and the persona text must match the input byte-for-byte —
    proving Step 2.5 inserted nothing on the disabled path.
    """
    monkeypatch.delenv(ENV_RESEARCH, raising=False)
    monkeypatch.delenv(ENV_TAVILY, raising=False)

    # Patch only the ARS-construction collaborators so no real LLM /
    # network call is made even on the disabled path.
    _patch_host_attr(monkeypatch, "LLMClient", MagicMock())
    _patch_host_attr(monkeypatch, "TavilyProvider", MagicMock())
    _patch_host_attr(monkeypatch, "QueryCache", MagicMock())
    _patch_host_attr(monkeypatch, "ResearchJsonlLogger", MagicMock())

    # Replace the default 2 SimpleNamespace profiles with REAL
    # OasisAgentProfile instances so ``save_profiles`` (which calls
    # ``profile.user_name``, ``.bio``, ``.karma``, etc.) doesn't blow
    # up on missing attributes.
    from app.services.oasis_profile_generator import OasisAgentProfile

    real_profiles = [
        OasisAgentProfile(
            user_id=0,
            user_name="user_0",
            name="Name0",
            bio="Bio0",
            persona="ORIGINAL-PERSONA-0",
        ),
        OasisAgentProfile(
            user_id=1,
            user_name="user_1",
            name="Name1",
            bio="Bio1",
            persona="ORIGINAL-PERSONA-1",
        ),
    ]
    patch_phase1_phase2_phase3.profile_gen.return_value \
        .generate_profiles_from_entities.return_value = real_profiles

    # Real ``save_profiles`` writes to disk; restore the real bound
    # method on the mocked generator instance so it actually runs.
    # ``save_profiles`` dispatches to ``self._save_reddit_json`` /
    # ``self._save_twitter_csv`` — both also need to bridge through
    # the real implementations, otherwise the MagicMock instance
    # silently returns another mock when those inner methods are
    # accessed and no file is ever written. ``_normalize_gender`` is
    # used inside ``_save_reddit_json`` and needs the same bridge.
    from app.services.oasis_profile_generator import OasisProfileGenerator

    mock_inst = patch_phase1_phase2_phase3.profile_gen.return_value
    real_save = OasisProfileGenerator.save_profiles
    real_save_reddit = OasisProfileGenerator._save_reddit_json
    real_save_twitter = OasisProfileGenerator._save_twitter_csv
    real_normalize_gender = OasisProfileGenerator._normalize_gender

    mock_inst.save_profiles = (
        lambda profiles, file_path, platform: real_save(
            mock_inst, profiles, file_path, platform
        )
    )
    mock_inst._save_reddit_json = (
        lambda profiles, file_path: real_save_reddit(
            mock_inst, profiles, file_path
        )
    )
    mock_inst._save_twitter_csv = (
        lambda profiles, file_path: real_save_twitter(
            mock_inst, profiles, file_path
        )
    )
    mock_inst._normalize_gender = (
        lambda gender: real_normalize_gender(mock_inst, gender)
    )

    manager, simulation_id = seeded_manager
    manager.prepare_simulation(
        simulation_id=simulation_id,
        simulation_requirement="topic-seed-regression",
        document_text="doc",
    )

    # ``tmp_sim_dir`` redirects ``SIMULATION_DATA_DIR`` to ``tmp_path``;
    # the per-sim sub-dir is named after ``simulation_id``.
    reddit_path = tmp_sim_dir / simulation_id / "reddit_profiles.json"
    assert reddit_path.exists()
    payload = json.loads(reddit_path.read_text(encoding="utf-8"))
    assert len(payload) == 2
    canonical_keys = {
        "user_id", "username", "name", "bio", "persona", "karma", "created_at"
    }
    assert canonical_keys.issubset(set(payload[0].keys()))
    # Persona byte-for-byte equal to input — the disabled gate must NOT
    # mutate persona.
    assert payload[0]["persona"] == "ORIGINAL-PERSONA-0"
    assert payload[1]["persona"] == "ORIGINAL-PERSONA-1"


# ---------------------------------------------------------------------------
# Scenario 7 — AgentResearchService constructed exactly once per run
# ---------------------------------------------------------------------------
def test_research_service_constructed_once_per_run(
    seeded_manager,
    patch_phase1_phase2_phase3,
    monkeypatch,
    llm_key_set,
):
    """Patch the class itself at the host seam and inspect call_count.
    Class-level patching is cleaner than ``__init__``-spy patching
    because it never strands instance attributes.
    """
    research_instance_mock = MagicMock()
    research_instance_mock.run.side_effect = (
        lambda pid, profiles, abi, ts: profiles
    )
    research_instance_mock.is_enabled.return_value = True
    research_class_mock = MagicMock(return_value=research_instance_mock)
    _patch_host_attr(monkeypatch, "AgentResearchService", research_class_mock)
    _patch_host_attr(monkeypatch, "TavilyProvider", MagicMock())
    _patch_host_attr(monkeypatch, "QueryCache", MagicMock())
    _patch_host_attr(monkeypatch, "ResearchJsonlLogger", MagicMock())
    _patch_host_attr(monkeypatch, "LLMClient", MagicMock())

    # Bump to 3 profiles to confirm construction is loop-invariant.
    profiles = [
        _make_profile(i, f"ORIG-{i}", [f"t{i}"], f"b{i}") for i in range(3)
    ]
    activities = [
        _make_activity(0.5, 1.0, "neutral", agent_id=i) for i in range(3)
    ]
    patch_phase1_phase2_phase3.profile_gen.return_value \
        .generate_profiles_from_entities.return_value = profiles
    patch_phase1_phase2_phase3.config_gen.return_value \
        .generate_config.return_value = SimpleNamespace(
            agent_configs=activities,
            generation_reasoning="test",
            to_json=lambda: "{}",
        )
    patch_phase1_phase2_phase3.zep_reader.return_value \
        .filter_defined_entities.return_value = _make_filtered_entities(3)

    monkeypatch.setattr(
        patch_phase1_phase2_phase3.profile_gen.return_value,
        "save_profiles",
        MagicMock(),
    )

    manager, simulation_id = seeded_manager
    manager.prepare_simulation(
        simulation_id=simulation_id,
        simulation_requirement="topic-seed-once",
        document_text="doc",
    )

    assert research_class_mock.call_count == 1


# ---------------------------------------------------------------------------
# Scenario 8 — Catastrophic failure does not abort the pipeline
# ---------------------------------------------------------------------------
def test_research_failure_does_not_abort_pipeline(
    seeded_manager,
    patch_phase1_phase2_phase3,
    monkeypatch,
    llm_key_set,
    caplog,
):
    """The host site MUST defensively wrap ``service.run`` so a raised
    exception does not propagate. The caught exception must be logged
    at ERROR level (via ``logger.exception``) and ``save_profiles`` MUST
    still receive the original (un-mutated) profile list.
    """
    research_instance_mock = MagicMock()
    research_instance_mock.run.side_effect = RuntimeError(
        "simulated catastrophic failure"
    )
    research_instance_mock.is_enabled.return_value = True
    _patch_host_attr(
        monkeypatch,
        "AgentResearchService",
        MagicMock(return_value=research_instance_mock),
    )
    _patch_host_attr(monkeypatch, "TavilyProvider", MagicMock())
    _patch_host_attr(monkeypatch, "QueryCache", MagicMock())
    _patch_host_attr(monkeypatch, "ResearchJsonlLogger", MagicMock())
    _patch_host_attr(monkeypatch, "LLMClient", MagicMock())

    save_profiles_mock = MagicMock()
    monkeypatch.setattr(
        patch_phase1_phase2_phase3.profile_gen.return_value,
        "save_profiles",
        save_profiles_mock,
    )

    caplog.set_level(logging.ERROR)

    from app.services.simulation_manager import SimulationStatus

    manager, simulation_id = seeded_manager
    state = manager.prepare_simulation(
        simulation_id=simulation_id,
        simulation_requirement="topic-seed-failure",
        document_text="doc",
    )

    # 1. Pipeline finished — status flipped to READY, NOT FAILED.
    assert state.status == SimulationStatus.READY
    # 2. ERROR-level log line mentioning "research" exists.
    assert any(
        "research" in r.message.lower() and r.levelno >= logging.ERROR
        for r in caplog.records
    )
    # 3. ``save_profiles`` was invoked with the ORIGINAL profile list
    #    (un-mutated by the failed research call).
    save_profiles_mock.assert_called()
    saved = save_profiles_mock.call_args.kwargs["profiles"]
    assert [p.persona for p in saved] == ["ORIGINAL-0", "ORIGINAL-1"]

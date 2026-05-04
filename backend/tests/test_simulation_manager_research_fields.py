"""
Tests for the two operator-visible research fields on ``SimulationState``:
``research_enabled`` (bool) and ``research_base_k`` (int).

These fields are persisted for UI round-trip only. They do NOT, on their own,
gate the runtime web-research pipeline — that contract belongs to the
``RESEARCH_ENABLED`` env var consumed by ``AgentResearchService.is_enabled()``
at process start (planning §3.3, T-011 wiring). ``False`` / ``3`` are the
canonical defaults.

Coverage:
    * dataclass defaults                       (test #1)
    * round-trip persistence (set + default)   (tests #2, #3)
    * backward-compat with legacy state.json   (tests #4, #9)
    * serializer surface                       (test #5)
    * REST create/read accept + return         (tests #6, #7)
    * REST partial update                      (test #8)
    * regression guard for all existing fields (test #10)
    * REST input validation                    (tests #11, #12)

Performance target: full file completes in <3 seconds. All filesystem I/O is
scoped to ``tmp_path``; the Flask test client runs in-process.

The autouse ``_clean_llm_env`` fixture from ``conftest.py`` purges all twelve
LLM env vars per test. It is harmless here (no test exercises ``Config.llm_for``)
but stays active for cross-test isolation.
"""

import json
import os
from types import SimpleNamespace

import pytest

from app.services.simulation_manager import (
    SimulationManager,
    SimulationState,
    SimulationStatus,
)


# --------------------------------------------------------------------------
# Test-local fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def tmp_sim_dir(tmp_path, monkeypatch):
    """Redirect ``SimulationManager.SIMULATION_DATA_DIR`` to a per-test
    ``tmp_path`` so file I/O never escapes the test sandbox.

    The class attribute is bound at module import (no constructor injection),
    so ``monkeypatch.setattr`` is the only clean way to redirect it.
    Yields the ``tmp_path`` for tests that need to seed legacy JSON files.
    """
    monkeypatch.setattr(SimulationManager, "SIMULATION_DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def flask_client(tmp_sim_dir, monkeypatch):
    """Build a Flask test client with ``ProjectManager.get_project`` stubbed
    to return a minimal namespace carrying the ``graph_id`` the create
    handler reads. Avoids the need for a real project on disk.

    Function-scoped so monkeypatch resets cleanly between tests.
    """
    from app import create_app
    import app.api.simulation as sim_api

    # Stub: only ``project.graph_id`` and truthiness are read by the handler.
    def _stub_get_project(project_id):
        return SimpleNamespace(project_id=project_id, graph_id="g1")

    monkeypatch.setattr(sim_api.ProjectManager, "get_project", staticmethod(_stub_get_project))

    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


# --------------------------------------------------------------------------
# Scenario 1 — dataclass defaults
# --------------------------------------------------------------------------
def test_simulation_state_default_research_fields():
    """A fresh ``SimulationState`` with only the three required positional
    args must default ``research_enabled`` to ``False`` and
    ``research_base_k`` to ``3``."""
    state = SimulationState(simulation_id="s1", project_id="p1", graph_id="g1")

    assert state.research_enabled is False
    assert state.research_base_k == 3


# --------------------------------------------------------------------------
# Scenario 2 — round-trip preserves user-set research fields
# --------------------------------------------------------------------------
def test_save_load_round_trip_preserves_set_research_fields(tmp_sim_dir):
    """Save a state with non-default research fields, reload via a fresh
    manager (bypassing the in-memory cache), assert both values survive."""
    manager = SimulationManager()
    state = SimulationState(
        simulation_id="s2",
        project_id="p1",
        graph_id="g1",
        research_enabled=True,
        research_base_k=7,
    )
    manager._save_simulation_state(state)

    # Fresh manager instance to bypass per-instance ``_simulations`` cache.
    reloaded_manager = SimulationManager()
    reloaded = reloaded_manager._load_simulation_state("s2")

    assert reloaded is not None
    assert reloaded.research_enabled is True
    assert reloaded.research_base_k == 7


# --------------------------------------------------------------------------
# Scenario 3 — round-trip preserves explicit default research fields
# --------------------------------------------------------------------------
def test_save_load_round_trip_preserves_default_research_fields(tmp_sim_dir):
    """``False`` and ``3`` are the canonical defaults and must survive
    round-trip — they don't collapse to ``None`` or a missing key. Critical
    boundary against the unsafe ``or default`` null-coercion pattern that
    would silently rewrite a legitimate ``False`` / ``0`` to the default."""
    manager = SimulationManager()
    state = SimulationState(
        simulation_id="s3",
        project_id="p1",
        graph_id="g1",
        research_enabled=False,
        research_base_k=3,
    )
    manager._save_simulation_state(state)

    reloaded_manager = SimulationManager()
    reloaded = reloaded_manager._load_simulation_state("s3")

    assert reloaded is not None
    assert reloaded.research_enabled is False
    assert reloaded.research_base_k == 3


# --------------------------------------------------------------------------
# Scenario 4 — backward compatibility with legacy state.json
# --------------------------------------------------------------------------
def test_legacy_state_json_without_research_fields_loads_with_defaults(tmp_sim_dir):
    """Legacy project JSONs predate the two new fields. ``_load_simulation_state``
    must not raise ``KeyError`` and must default ``research_enabled`` to
    ``False`` and ``research_base_k`` to ``3``."""
    sim_id = "s4_legacy"
    sim_dir = os.path.join(str(tmp_sim_dir), sim_id)
    os.makedirs(sim_dir, exist_ok=True)

    # Legacy shape — explicitly omits both new keys. The three model-name
    # fields from T-004 are present so this fixture is a faithful pre-T-012
    # snapshot, not a pre-T-004 one.
    legacy_payload = {
        "simulation_id": sim_id,
        "project_id": "p_legacy",
        "graph_id": "g_legacy",
        "enable_twitter": True,
        "enable_reddit": True,
        "status": "created",
        "entities_count": 0,
        "profiles_count": 0,
        "entity_types": [],
        "config_generated": False,
        "config_reasoning": "",
        "current_round": 0,
        "twitter_status": "not_started",
        "reddit_status": "not_started",
        "created_at": "2025-12-01T10:00:00",
        "updated_at": "2025-12-01T10:00:00",
        "error": None,
        "builder_model_name": "",
        "swarm_model_name": "",
        "judge_model_name": "",
    }
    with open(os.path.join(sim_dir, "state.json"), "w", encoding="utf-8") as f:
        json.dump(legacy_payload, f)

    manager = SimulationManager()
    state = manager._load_simulation_state(sim_id)

    assert state is not None
    assert state.research_enabled is False
    assert state.research_base_k == 3


# --------------------------------------------------------------------------
# Scenario 5 — to_dict surfaces the two new keys
# --------------------------------------------------------------------------
def test_to_dict_includes_research_fields():
    """``to_dict`` is hand-written and explicit (no ``asdict``). Adding a
    dataclass field is not enough; the serializer must list the new keys."""
    state = SimulationState(simulation_id="s5", project_id="p1", graph_id="g1")
    payload = state.to_dict()

    assert "research_enabled" in payload
    assert "research_base_k" in payload
    assert payload["research_enabled"] is False
    assert payload["research_base_k"] == 3


# --------------------------------------------------------------------------
# Scenario 6 — REST create accepts the two fields and read returns them
# --------------------------------------------------------------------------
def test_rest_create_accepts_and_returns_research_fields(flask_client):
    """POST to ``/api/simulation/create`` with the two research fields, then
    GET the simulation; the response payload must surface both values at
    ``data.<field>`` (handler wraps ``state.to_dict()`` under "data")."""
    create_response = flask_client.post(
        "/api/simulation/create",
        json={
            "project_id": "p1",
            "graph_id": "g1",
            "research_enabled": True,
            "research_base_k": 5,
        },
    )
    assert create_response.status_code == 200
    create_body = create_response.get_json()
    assert create_body["success"] is True
    sim_id = create_body["data"]["simulation_id"]

    get_response = flask_client.get(f"/api/simulation/{sim_id}")
    assert get_response.status_code == 200
    payload = get_response.get_json()["data"]
    assert payload["research_enabled"] is True
    assert payload["research_base_k"] == 5


# --------------------------------------------------------------------------
# Scenario 7 — REST create without research fields stays backward-compatible
# --------------------------------------------------------------------------
def test_rest_create_without_research_fields_is_backward_compat(flask_client):
    """A POST that omits the two new keys must still succeed and the
    subsequent GET must return ``research_enabled=False`` and
    ``research_base_k=3``."""
    create_response = flask_client.post(
        "/api/simulation/create",
        json={"project_id": "p1", "graph_id": "g1"},
    )
    assert create_response.status_code == 200
    sim_id = create_response.get_json()["data"]["simulation_id"]

    get_response = flask_client.get(f"/api/simulation/{sim_id}")
    assert get_response.status_code == 200
    payload = get_response.get_json()["data"]
    assert payload["research_enabled"] is False
    assert payload["research_base_k"] == 3


# --------------------------------------------------------------------------
# Scenario 8 — REST partial update mutates only the supplied field
# --------------------------------------------------------------------------
def test_rest_update_mutates_only_supplied_research_fields(flask_client):
    """Create a simulation with both research fields set; PATCH with only
    ``research_base_k``; GET must show ``research_enabled`` unchanged and
    ``research_base_k`` updated. Tests the PATCH endpoint expansion to
    accept the two new keys."""
    create_response = flask_client.post(
        "/api/simulation/create",
        json={
            "project_id": "p1",
            "graph_id": "g1",
            "research_enabled": True,
            "research_base_k": 4,
        },
    )
    assert create_response.status_code == 200
    sim_id = create_response.get_json()["data"]["simulation_id"]

    patch_response = flask_client.patch(
        f"/api/simulation/{sim_id}",
        json={"research_base_k": 7},
    )
    assert patch_response.status_code == 200

    get_response = flask_client.get(f"/api/simulation/{sim_id}")
    assert get_response.status_code == 200
    payload = get_response.get_json()["data"]
    # research_enabled unchanged because it was not in the PATCH body.
    assert payload["research_enabled"] is True
    # research_base_k mutated by the PATCH.
    assert payload["research_base_k"] == 7


# --------------------------------------------------------------------------
# Scenario 9 — defensive coercion of explicit JSON null on load
# --------------------------------------------------------------------------
def test_load_coerces_null_in_json_to_default_for_research_fields(tmp_sim_dir):
    """A buggy client may persist ``"research_enabled": null`` or
    ``"research_base_k": null``. The load path must coerce ``None`` to the
    declared default so the bool / int contract holds end-to-end."""
    sim_id = "s9_null"
    sim_dir = os.path.join(str(tmp_sim_dir), sim_id)
    os.makedirs(sim_dir, exist_ok=True)

    payload_with_null = {
        "simulation_id": sim_id,
        "project_id": "p9",
        "graph_id": "g9",
        "enable_twitter": True,
        "enable_reddit": True,
        "status": "created",
        "entities_count": 0,
        "profiles_count": 0,
        "entity_types": [],
        "config_generated": False,
        "config_reasoning": "",
        "current_round": 0,
        "twitter_status": "not_started",
        "reddit_status": "not_started",
        "created_at": "2025-12-01T10:00:00",
        "updated_at": "2025-12-01T10:00:00",
        "error": None,
        "builder_model_name": "",
        "swarm_model_name": "",
        "judge_model_name": "",
        "research_enabled": None,
        "research_base_k": None,
    }
    with open(os.path.join(sim_dir, "state.json"), "w", encoding="utf-8") as f:
        json.dump(payload_with_null, f)

    manager = SimulationManager()
    state = manager._load_simulation_state(sim_id)

    assert state is not None
    assert state.research_enabled is False
    assert state.research_enabled is not None
    assert state.research_base_k == 3
    assert state.research_base_k is not None


# --------------------------------------------------------------------------
# Scenario 10 — regression guard: every existing field survives round-trip
# --------------------------------------------------------------------------
def test_round_trip_preserves_all_existing_simulation_state_fields(tmp_sim_dir):
    """Construct a ``SimulationState`` with every field set to a non-default
    value, save, reload, and assert serializer-level equality. Comparing
    ``to_dict()`` (rather than dataclass instances) sidesteps the
    ``updated_at`` mutation in ``_save_simulation_state``. Non-default
    research field values guarantee we exercise the new persistence path."""
    manager = SimulationManager()
    state = SimulationState(
        simulation_id="s10",
        project_id="p10",
        graph_id="g10",
        enable_twitter=False,
        enable_reddit=False,
        status=SimulationStatus.READY,
        entities_count=42,
        profiles_count=7,
        entity_types=["Student", "Teacher"],
        config_generated=True,
        config_reasoning="because",
        current_round=3,
        twitter_status="running",
        reddit_status="paused",
        error="something",
        builder_model_name="b_full",
        swarm_model_name="s_full",
        judge_model_name="j_full",
        research_enabled=True,
        research_base_k=8,
    )
    manager._save_simulation_state(state)
    expected_payload = state.to_dict()

    reloaded_manager = SimulationManager()
    reloaded = reloaded_manager._load_simulation_state("s10")

    assert reloaded is not None
    assert reloaded.to_dict() == expected_payload


# --------------------------------------------------------------------------
# Scenario 11 — REST create rejects non-bool research_enabled (strict-bool)
# --------------------------------------------------------------------------
def test_rest_create_rejects_non_bool_research_enabled(flask_client):
    """POST with ``research_enabled="true"`` (string, not bool) must return
    HTTP 400. Per Critical decisions §1, the policy is strict-bool: never
    silently coerce a string to a bool at the REST entry. JSON has a native
    bool literal; a string is a client bug."""
    response = flask_client.post(
        "/api/simulation/create",
        json={
            "project_id": "p1",
            "graph_id": "g1",
            "research_enabled": "true",
        },
    )
    assert response.status_code == 400
    body = response.get_json()
    assert body["success"] is False


# --------------------------------------------------------------------------
# Scenario 12 — REST create clamps research_base_k below 1 to 1
# --------------------------------------------------------------------------
def test_rest_create_clamps_research_base_k_below_one(flask_client):
    """POST with ``research_base_k=0`` must clamp to 1 (minimum). Per
    Critical decisions §2, the policy is clamp-to-bounds [1, MAX_RESEARCH_QUERIES_PER_AGENT]
    rather than reject — clamp is more forgiving for an integer input and
    matches the runtime budget calculator."""
    create_response = flask_client.post(
        "/api/simulation/create",
        json={
            "project_id": "p1",
            "graph_id": "g1",
            "research_base_k": 0,
        },
    )
    assert create_response.status_code == 200
    sim_id = create_response.get_json()["data"]["simulation_id"]

    get_response = flask_client.get(f"/api/simulation/{sim_id}")
    assert get_response.status_code == 200
    payload = get_response.get_json()["data"]
    assert payload["research_base_k"] == 1

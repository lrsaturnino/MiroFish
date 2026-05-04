"""
Tests for the three operator-visible model-name fields on ``SimulationState``:
``builder_model_name``, ``swarm_model_name``, ``judge_model_name``.

These fields are persisted for UI round-trip only. They do not, on their own,
change the runtime LLM resolution path — that contract belongs to
``Config.llm_for(role)`` which reads env vars at call time. Empty string ``""``
is the canonical "inherit from ``LLM_*``" sentinel.

Coverage:
    * dataclass defaults                       (test #1)
    * round-trip persistence (set + empty)     (tests #2, #3)
    * backward-compat with legacy state.json   (tests #4, #9)
    * serializer surface                       (test #5)
    * REST create/read accept + return         (tests #6, #7)
    * REST partial update                      (test #8)
    * regression guard for all existing fields (test #10)

Performance target: full file completes in <5 seconds. All filesystem I/O is
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
def test_simulation_state_default_model_fields_are_empty():
    """A fresh ``SimulationState`` with only the three required positional
    args must default all three model fields to ``""``."""
    state = SimulationState(simulation_id="s1", project_id="p1", graph_id="g1")

    assert state.builder_model_name == ""
    assert state.swarm_model_name == ""
    assert state.judge_model_name == ""


# --------------------------------------------------------------------------
# Scenario 2 — round-trip preserves user-set model fields
# --------------------------------------------------------------------------
def test_save_load_round_trip_preserves_set_model_fields(tmp_sim_dir):
    """Save a state with non-empty model fields, reload via a fresh manager
    (bypassing the in-memory cache), assert all three values survive."""
    manager = SimulationManager()
    state = SimulationState(
        simulation_id="s2",
        project_id="p1",
        graph_id="g1",
        builder_model_name="gpt-4o",
        swarm_model_name="qwen-plus",
        judge_model_name="claude-sonnet",
    )
    manager._save_simulation_state(state)

    # Fresh manager instance to bypass per-instance ``_simulations`` cache.
    reloaded_manager = SimulationManager()
    reloaded = reloaded_manager._load_simulation_state("s2")

    assert reloaded is not None
    assert reloaded.builder_model_name == "gpt-4o"
    assert reloaded.swarm_model_name == "qwen-plus"
    assert reloaded.judge_model_name == "claude-sonnet"


# --------------------------------------------------------------------------
# Scenario 3 — round-trip preserves explicit empty model fields
# --------------------------------------------------------------------------
def test_save_load_round_trip_preserves_empty_model_fields(tmp_sim_dir):
    """Empty string is the canonical 'inherit' sentinel and must survive
    round-trip — not collapse to ``None`` or a missing key."""
    manager = SimulationManager()
    state = SimulationState(
        simulation_id="s3",
        project_id="p1",
        graph_id="g1",
        builder_model_name="",
        swarm_model_name="",
        judge_model_name="",
    )
    manager._save_simulation_state(state)

    reloaded_manager = SimulationManager()
    reloaded = reloaded_manager._load_simulation_state("s3")

    assert reloaded is not None
    assert reloaded.builder_model_name == ""
    assert reloaded.swarm_model_name == ""
    assert reloaded.judge_model_name == ""


# --------------------------------------------------------------------------
# Scenario 4 — backward compatibility with legacy state.json
# --------------------------------------------------------------------------
def test_legacy_state_json_without_model_fields_loads_with_empty_defaults(tmp_sim_dir):
    """Legacy project JSONs predate the three new fields. ``_load_simulation_state``
    must not raise ``KeyError`` and must default each missing field to ``""``."""
    sim_id = "s4_legacy"
    sim_dir = os.path.join(str(tmp_sim_dir), sim_id)
    os.makedirs(sim_dir, exist_ok=True)

    # Legacy 17-field shape — explicitly omits the three new keys.
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
    }
    with open(os.path.join(sim_dir, "state.json"), "w", encoding="utf-8") as f:
        json.dump(legacy_payload, f)

    manager = SimulationManager()
    state = manager._load_simulation_state(sim_id)

    assert state is not None
    assert state.builder_model_name == ""
    assert state.swarm_model_name == ""
    assert state.judge_model_name == ""


# --------------------------------------------------------------------------
# Scenario 5 — to_dict surfaces the three new keys
# --------------------------------------------------------------------------
def test_to_dict_includes_three_model_fields():
    """``to_dict`` is hand-written and explicit (no ``asdict``). Adding a
    dataclass field is not enough; the serializer must list the new keys."""
    state = SimulationState(simulation_id="s5", project_id="p1", graph_id="g1")
    payload = state.to_dict()

    assert "builder_model_name" in payload
    assert "swarm_model_name" in payload
    assert "judge_model_name" in payload
    assert payload["builder_model_name"] == ""
    assert payload["swarm_model_name"] == ""
    assert payload["judge_model_name"] == ""


# --------------------------------------------------------------------------
# Scenario 6 — REST create accepts the three fields and read returns them
# --------------------------------------------------------------------------
def test_rest_create_accepts_and_returns_model_fields(flask_client):
    """POST to ``/api/simulation/create`` with the three model fields, then
    GET the simulation; the response payload must surface all three values
    at ``data.<field>`` (handler wraps state.to_dict() under "data")."""
    create_response = flask_client.post(
        "/api/simulation/create",
        json={
            "project_id": "p1",
            "graph_id": "g1",
            "builder_model_name": "gpt-4o",
            "swarm_model_name": "qwen-plus",
            "judge_model_name": "",
        },
    )
    assert create_response.status_code == 200
    create_body = create_response.get_json()
    assert create_body["success"] is True
    sim_id = create_body["data"]["simulation_id"]

    get_response = flask_client.get(f"/api/simulation/{sim_id}")
    assert get_response.status_code == 200
    payload = get_response.get_json()["data"]
    assert payload["builder_model_name"] == "gpt-4o"
    assert payload["swarm_model_name"] == "qwen-plus"
    assert payload["judge_model_name"] == ""


# --------------------------------------------------------------------------
# Scenario 7 — REST create without model fields stays backward-compatible
# --------------------------------------------------------------------------
def test_rest_create_without_model_fields_is_backward_compat(flask_client):
    """A POST that omits the three new keys must still succeed and the
    subsequent GET must return all three as ``""``."""
    create_response = flask_client.post(
        "/api/simulation/create",
        json={"project_id": "p1", "graph_id": "g1"},
    )
    assert create_response.status_code == 200
    sim_id = create_response.get_json()["data"]["simulation_id"]

    get_response = flask_client.get(f"/api/simulation/{sim_id}")
    assert get_response.status_code == 200
    payload = get_response.get_json()["data"]
    assert payload["builder_model_name"] == ""
    assert payload["swarm_model_name"] == ""
    assert payload["judge_model_name"] == ""


# --------------------------------------------------------------------------
# Scenario 8 — REST partial update mutates only the supplied field
# --------------------------------------------------------------------------
def test_rest_update_mutates_only_supplied_model_fields(flask_client):
    """Create a simulation with all three model fields set; PATCH with only
    ``swarm_model_name``; GET must show builder/judge unchanged and swarm
    updated. Tests the new ``PATCH /<simulation_id>`` endpoint."""
    create_response = flask_client.post(
        "/api/simulation/create",
        json={
            "project_id": "p1",
            "graph_id": "g1",
            "builder_model_name": "b1",
            "swarm_model_name": "s1",
            "judge_model_name": "j1",
        },
    )
    assert create_response.status_code == 200
    sim_id = create_response.get_json()["data"]["simulation_id"]

    patch_response = flask_client.patch(
        f"/api/simulation/{sim_id}",
        json={"swarm_model_name": "haiku"},
    )
    assert patch_response.status_code == 200

    get_response = flask_client.get(f"/api/simulation/{sim_id}")
    assert get_response.status_code == 200
    payload = get_response.get_json()["data"]
    assert payload["builder_model_name"] == "b1"
    assert payload["swarm_model_name"] == "haiku"
    assert payload["judge_model_name"] == "j1"


# --------------------------------------------------------------------------
# Scenario 9 — defensive coercion of explicit JSON null on load
# --------------------------------------------------------------------------
def test_load_coerces_null_in_json_to_empty_string(tmp_sim_dir):
    """A buggy client may persist ``"builder_model_name": null``. The load
    path must coerce ``None`` to ``""`` so the empty-string sentinel
    contract holds end-to-end."""
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
        "builder_model_name": None,
    }
    with open(os.path.join(sim_dir, "state.json"), "w", encoding="utf-8") as f:
        json.dump(payload_with_null, f)

    manager = SimulationManager()
    state = manager._load_simulation_state(sim_id)

    assert state is not None
    assert state.builder_model_name == ""
    assert state.builder_model_name is not None


# --------------------------------------------------------------------------
# Scenario 10 — regression guard: every existing field survives round-trip
# --------------------------------------------------------------------------
def test_round_trip_preserves_all_existing_simulation_state_fields(tmp_sim_dir):
    """Construct a ``SimulationState`` with every field set to a non-default
    value, save, reload, and assert serializer-level equality. Comparing
    ``to_dict()`` (rather than dataclass instances) sidesteps the
    ``updated_at`` mutation in ``_save_simulation_state`` (line 150)."""
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
    )
    manager._save_simulation_state(state)
    expected_payload = state.to_dict()

    reloaded_manager = SimulationManager()
    reloaded = reloaded_manager._load_simulation_state("s10")

    assert reloaded is not None
    assert reloaded.to_dict() == expected_payload

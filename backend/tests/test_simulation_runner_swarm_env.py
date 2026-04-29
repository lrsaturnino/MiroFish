"""
Tests for the SWARM-role LLM env wiring inside
``SimulationRunner.start_simulation``.

The wiring under test sits between the existing
``env = os.environ.copy()`` block and the ``subprocess.Popen(...)``
call in ``simulation_runner.py``. The implementation must populate
the subprocess's ``env`` dict with the SWARM-role credentials so the
OASIS subprocess (``run_twitter_simulation.py`` / ``run_parallel_simulation.py``)
sees per-role overrides instead of the parent's plain ``LLM_*`` globals.

**Belt-and-suspenders contract**:
The implementation writes BOTH name pairs into the subprocess env:

    * ``LLM_API_KEY``        AND  ``OPENAI_API_KEY``
    * ``LLM_BASE_URL``       AND  ``OPENAI_API_BASE_URL``
    * ``LLM_MODEL_NAME``     (no OPENAI counterpart required)

The ``LLM_*`` pair is what the OASIS subprocess actually reads at
``run_twitter_simulation.py:437-453``; the ``OPENAI_*`` pair makes the
SWARM credentials visible under the OpenAI-style name as well. Both
are written from the same ``Config.llm_for("swarm")`` triple so the
subprocess's runtime behavior matches the upstream intent.

**Source-presence guard**:
``Config.llm_for("swarm")`` returns non-``None`` defaults
(``https://api.openai.com/v1``, ``gpt-4o-mini``) for ``base_url``
and ``model`` even when no env var is set. The implementation must
NOT inject those defaults into the subprocess env when the underlying
source env var (``SWARM_LLM_*`` or the global ``LLM_*``) is absent —
otherwise the "no env vars set" baseline would gain spurious keys.

**LLM_BOOST_* layering** (note-only — final test in this module):
The ``LLM_BOOST_*`` precedence inside
``run_parallel_simulation.create_model(use_boost=True)`` is a
subprocess-side concern that reads different env-var names. SWARM
upstream wiring is orthogonal and never collides. The ``LLM_BOOST_*``
override continues to win over SWARM defaults by non-modification of
the subprocess scripts.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.services.simulation_runner import SimulationRunner


# --------------------------------------------------------------------------
# Per-test string constants — chosen to be obviously distinct so any
# accidental field-swap surfaces immediately in assertions.
# --------------------------------------------------------------------------
GLOBAL_API = "k"
GLOBAL_URL = "b"
GLOBAL_MODEL = "m"

SWARM_API = "sk"
SWARM_URL = "sb"
SWARM_MODEL = "sm"

SIM_ID = "test-sim-t003"


# --------------------------------------------------------------------------
# Shared fixtures
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_openai_env(monkeypatch):
    """The shared ``conftest.py`` autouse fixture cleans ``LLM_*`` and
    role-prefixed vars but does NOT touch ``OPENAI_*``. The
    belt-and-suspenders contract asserts on ``OPENAI_API_KEY`` /
    ``OPENAI_API_BASE_URL``, so the parent env must be a known clean
    slate for those names too — otherwise a stray shell-export silently
    flips assertions. ``raising=False`` keeps it idempotent."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE_URL", raising=False)
    yield


@pytest.fixture
def setup_sim_dir(monkeypatch, tmp_path):
    """Redirect ``SimulationRunner.RUN_STATE_DIR`` to a pytest-managed
    ``tmp_path`` and pre-create the minimal ``simulation_config.json``
    that ``start_simulation`` reads at line 343-347.

    Yields the per-simulation directory.

    The finalizer pops the test's ``SIM_ID`` from every class-level dict
    on ``SimulationRunner`` so subsequent tests observe a clean slate.
    """
    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(tmp_path))

    sim_dir = tmp_path / SIM_ID
    sim_dir.mkdir(parents=True, exist_ok=True)

    config_path = sim_dir / "simulation_config.json"
    config_path.write_text(
        json.dumps(
            {
                "time_config": {
                    "total_simulation_hours": 1,
                    "minutes_per_round": 30,
                }
            }
        ),
        encoding="utf-8",
    )

    yield sim_dir

    # Class-level state pollution is the #1 source of order-dependent
    # flakes in pytest. Drop every dict entry the test could have
    # populated. ``_graph_memory_enabled`` is the one the pre-impl
    # finalizer list omitted — keep it here.
    for attr in (
        "_processes",
        "_action_queues",
        "_stdout_files",
        "_stderr_files",
        "_monitor_threads",
        "_run_states",
        "_graph_memory_enabled",
    ):
        getattr(SimulationRunner, attr).pop(SIM_ID, None)


@pytest.fixture
def mock_popen_and_thread(monkeypatch):
    """Patch ``subprocess.Popen`` and ``threading.Thread`` at the
    module-relative paths so ``start_simulation``'s side effects are
    fully neutralised in-test.

    ``mock_popen.return_value.pid`` is set to a real ``int`` because
    ``start_simulation`` assigns it to ``state.process_pid`` which is
    serialised through ``json.dump`` by ``_save_run_state``. A bare
    MagicMock would raise ``TypeError: not JSON serializable``.
    """
    mock_popen = MagicMock()
    mock_popen.return_value.pid = 99999

    monkeypatch.setattr(
        "app.services.simulation_runner.subprocess.Popen", mock_popen
    )
    monkeypatch.setattr(
        "app.services.simulation_runner.threading.Thread", MagicMock()
    )

    return mock_popen


# --------------------------------------------------------------------------
# Scenario 1 — SWARM unset, globals only: env carries the global LLM_*
# AND the OPENAI_* twins (belt-and-suspenders writes both pairs).
# --------------------------------------------------------------------------
def test_globals_only_swarm_unset_writes_both_name_pairs(
    monkeypatch, setup_sim_dir, mock_popen_and_thread
):
    """With only ``LLM_*`` globals set and no ``SWARM_LLM_*``, the
    subprocess env must carry the global values under BOTH the
    ``LLM_*`` and ``OPENAI_*`` names. The OPENAI_* assertions are the
    substantive RED failure — they are not in ``os.environ.copy()``
    today (verified: ``OPENAI_API_KEY`` not in parent env) so only the
    new SWARM block can introduce them."""
    monkeypatch.setenv("LLM_API_KEY", GLOBAL_API)
    monkeypatch.setenv("LLM_BASE_URL", GLOBAL_URL)
    monkeypatch.setenv("LLM_MODEL_NAME", GLOBAL_MODEL)

    SimulationRunner.start_simulation(SIM_ID, platform="parallel")

    env = mock_popen_and_thread.call_args.kwargs["env"]

    assert env["LLM_API_KEY"] == GLOBAL_API
    assert env["LLM_BASE_URL"] == GLOBAL_URL
    assert env["LLM_MODEL_NAME"] == GLOBAL_MODEL
    assert env["OPENAI_API_KEY"] == GLOBAL_API
    assert env["OPENAI_API_BASE_URL"] == GLOBAL_URL


# --------------------------------------------------------------------------
# Scenario 2 — Full SWARM group set: SWARM values win for every field
# across both name pairs.
# --------------------------------------------------------------------------
def test_full_swarm_group_wins_across_both_pairs(
    monkeypatch, setup_sim_dir, mock_popen_and_thread
):
    """Distinct values for global ``LLM_*`` and ``SWARM_LLM_*`` prove
    the SWARM triple wins per-field, and that wins propagate to BOTH
    the ``LLM_*`` and ``OPENAI_*`` env keys."""
    monkeypatch.setenv("LLM_API_KEY", GLOBAL_API)
    monkeypatch.setenv("LLM_BASE_URL", GLOBAL_URL)
    monkeypatch.setenv("LLM_MODEL_NAME", GLOBAL_MODEL)

    monkeypatch.setenv("SWARM_LLM_API_KEY", SWARM_API)
    monkeypatch.setenv("SWARM_LLM_BASE_URL", SWARM_URL)
    monkeypatch.setenv("SWARM_LLM_MODEL_NAME", SWARM_MODEL)

    SimulationRunner.start_simulation(SIM_ID, platform="parallel")

    env = mock_popen_and_thread.call_args.kwargs["env"]

    assert env["LLM_API_KEY"] == SWARM_API
    assert env["LLM_BASE_URL"] == SWARM_URL
    assert env["LLM_MODEL_NAME"] == SWARM_MODEL
    assert env["OPENAI_API_KEY"] == SWARM_API
    assert env["OPENAI_API_BASE_URL"] == SWARM_URL

    # Lock the override: the global values must NOT survive in either
    # name pair.
    assert env["LLM_API_KEY"] != GLOBAL_API
    assert env["OPENAI_API_KEY"] != GLOBAL_API


# --------------------------------------------------------------------------
# Scenario 3 — Partial SWARM (model only): only LLM_MODEL_NAME flips;
# the api_key/base_url fall through to globals across both pairs.
# --------------------------------------------------------------------------
def test_partial_swarm_model_only_other_fields_inherit_globals(
    monkeypatch, setup_sim_dir, mock_popen_and_thread
):
    """A partially-set SWARM role must not zero out other inherited
    fields (conditional-set semantics). With only ``SWARM_LLM_MODEL_NAME``
    set, ``api_key``/``base_url`` keep their global values in BOTH name
    pairs."""
    monkeypatch.setenv("LLM_API_KEY", GLOBAL_API)
    monkeypatch.setenv("LLM_BASE_URL", GLOBAL_URL)
    monkeypatch.setenv("LLM_MODEL_NAME", GLOBAL_MODEL)
    monkeypatch.setenv("SWARM_LLM_MODEL_NAME", SWARM_MODEL)

    SimulationRunner.start_simulation(SIM_ID, platform="parallel")

    env = mock_popen_and_thread.call_args.kwargs["env"]

    assert env["LLM_API_KEY"] == GLOBAL_API
    assert env["LLM_BASE_URL"] == GLOBAL_URL
    assert env["LLM_MODEL_NAME"] == SWARM_MODEL
    assert env["OPENAI_API_KEY"] == GLOBAL_API
    assert env["OPENAI_API_BASE_URL"] == GLOBAL_URL


# --------------------------------------------------------------------------
# Scenario 4 — Partial SWARM (api_key only): only LLM_API_KEY /
# OPENAI_API_KEY flip; base_url/model inherit globals.
# --------------------------------------------------------------------------
def test_partial_swarm_api_key_only_other_fields_inherit_globals(
    monkeypatch, setup_sim_dir, mock_popen_and_thread
):
    """Mirror of Test 3 for the api_key field. Locks symmetry — a
    partial role override touches only the field it specifies."""
    monkeypatch.setenv("LLM_API_KEY", GLOBAL_API)
    monkeypatch.setenv("LLM_BASE_URL", GLOBAL_URL)
    monkeypatch.setenv("LLM_MODEL_NAME", GLOBAL_MODEL)
    monkeypatch.setenv("SWARM_LLM_API_KEY", SWARM_API)

    SimulationRunner.start_simulation(SIM_ID, platform="parallel")

    env = mock_popen_and_thread.call_args.kwargs["env"]

    assert env["LLM_API_KEY"] == SWARM_API
    assert env["LLM_BASE_URL"] == GLOBAL_URL
    assert env["LLM_MODEL_NAME"] == GLOBAL_MODEL
    assert env["OPENAI_API_KEY"] == SWARM_API
    assert env["OPENAI_API_BASE_URL"] == GLOBAL_URL


# --------------------------------------------------------------------------
# Scenario 5 — UTF-8 regression sentinel: SWARM block must not shadow
# or remove PYTHONUTF8 / PYTHONIOENCODING.
# --------------------------------------------------------------------------
def test_utf8_env_vars_preserved_after_swarm_block(
    monkeypatch, setup_sim_dir, mock_popen_and_thread
):
    """A trivial regression check that the new SWARM wiring sits AFTER
    the UTF-8 lines and does not delete or shadow them. Pre-existing
    runtime guarantee — locks it post-GREEN."""
    monkeypatch.setenv("LLM_API_KEY", GLOBAL_API)
    monkeypatch.setenv("LLM_BASE_URL", GLOBAL_URL)
    monkeypatch.setenv("LLM_MODEL_NAME", GLOBAL_MODEL)

    SimulationRunner.start_simulation(SIM_ID, platform="parallel")

    env = mock_popen_and_thread.call_args.kwargs["env"]

    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"


# --------------------------------------------------------------------------
# Scenario 6 — SWARM_LLM_API_KEY overwrites a legacy LLM_API_KEY
# inherited from os.environ — proves SWARM wins over inheritance, not
# just over the absence of a value.
# --------------------------------------------------------------------------
def test_swarm_overrides_inherited_global_api_key(
    monkeypatch, setup_sim_dir, mock_popen_and_thread
):
    """The parent's ``os.environ`` already contains
    ``LLM_API_KEY="legacy"`` (e.g. from the loaded ``.env``).
    ``os.environ.copy()`` brings ``LLM_API_KEY="legacy"`` into the
    subprocess env. The SWARM block must overwrite it with the
    SWARM-role value — for both name pairs."""
    monkeypatch.setenv("LLM_API_KEY", "legacy")
    monkeypatch.setenv("SWARM_LLM_API_KEY", SWARM_API)
    # Globals for base_url / model are intentionally absent — this
    # test isolates the api_key inheritance-overwrite.

    SimulationRunner.start_simulation(SIM_ID, platform="parallel")

    env = mock_popen_and_thread.call_args.kwargs["env"]

    assert env["LLM_API_KEY"] == SWARM_API
    assert env["LLM_API_KEY"] != "legacy"
    assert env["OPENAI_API_KEY"] == SWARM_API
    assert env["OPENAI_API_KEY"] != "legacy"


# --------------------------------------------------------------------------
# Scenario 7 — Byte-identical baseline sentinel: with NOTHING set (no
# SWARM_LLM_*, no global LLM_*), the SWARM block must NOT inject the
# resolver's non-None defaults into env.
# --------------------------------------------------------------------------
def test_no_env_vars_set_does_not_inject_resolver_defaults(
    monkeypatch, setup_sim_dir, mock_popen_and_thread
):
    """``Config.llm_for("swarm")`` returns ``(None, "https://api.openai.com/v1",
    "gpt-4o-mini")`` when no env vars are set — the resolver invents
    defaults for ``base_url`` and ``model``. The "byte-identical config"
    baseline (no new env vars set) requires those defaults stay OUT of
    the subprocess env. The implementation must guard the conditional
    write on the SOURCE env-var presence, NOT on the resolver output's
    truthiness.

    This test is the canonical lock for that guard. Without it, a
    naive implementation that unconditionally writes the resolver
    tuple would slip past every other test in this file."""
    # All LLM_* env vars are pre-cleaned by the shared autouse
    # fixture. Be explicit anyway so the test is self-describing.
    for name in (
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL_NAME",
        "SWARM_LLM_API_KEY",
        "SWARM_LLM_BASE_URL",
        "SWARM_LLM_MODEL_NAME",
    ):
        monkeypatch.delenv(name, raising=False)

    SimulationRunner.start_simulation(SIM_ID, platform="parallel")

    env = mock_popen_and_thread.call_args.kwargs["env"]

    # None of the five LLM-related keys must appear in the subprocess
    # env when the parent had nothing set. This locks the
    # source-presence guard against the resolver's invented defaults.
    assert "LLM_API_KEY" not in env
    assert "LLM_BASE_URL" not in env
    assert "LLM_MODEL_NAME" not in env
    assert "OPENAI_API_KEY" not in env
    assert "OPENAI_API_BASE_URL" not in env


# --------------------------------------------------------------------------
# Scenario 8 — Source-presence guard: only SWARM_LLM_API_KEY set, no
# globals at all. Only the api_key fields appear; base_url/model fields
# stay absent (resolver returned defaults but source check suppresses
# the write).
# --------------------------------------------------------------------------
def test_only_swarm_api_key_set_no_globals_suppresses_default_fields(
    monkeypatch, setup_sim_dir, mock_popen_and_thread
):
    """Stronger source-presence-guard lock than the api-key-only partial
    test above. With ONLY ``SWARM_LLM_API_KEY`` set and zero globals,
    ``Config.llm_for("swarm")`` returns
    ``("sk", "https://api.openai.com/v1", "gpt-4o-mini")`` — the
    api_key is real, but base_url/model are resolver-invented defaults.

    The implementation must:
      * Write ``LLM_API_KEY`` + ``OPENAI_API_KEY`` (api_key has a real source).
      * NOT write ``LLM_BASE_URL`` / ``OPENAI_API_BASE_URL`` / ``LLM_MODEL_NAME``
        (base_url and model have no source env var present).
    """
    for name in (
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL_NAME",
        "SWARM_LLM_BASE_URL",
        "SWARM_LLM_MODEL_NAME",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SWARM_LLM_API_KEY", SWARM_API)

    SimulationRunner.start_simulation(SIM_ID, platform="parallel")

    env = mock_popen_and_thread.call_args.kwargs["env"]

    # api_key fields ARE written (real source).
    assert env["LLM_API_KEY"] == SWARM_API
    assert env["OPENAI_API_KEY"] == SWARM_API

    # base_url / model fields are NOT written (resolver default,
    # source check suppresses).
    assert "LLM_BASE_URL" not in env
    assert "OPENAI_API_BASE_URL" not in env
    assert "LLM_MODEL_NAME" not in env


# --------------------------------------------------------------------------
# Scenario 9 (note-only) — LLM_BOOST_* layering is a subprocess concern.
# --------------------------------------------------------------------------
def test_llm_boost_layering_is_a_subprocess_concern():
    """Documentation-only sentinel for the LLM_BOOST_* layering contract.

    ``LLM_BOOST_*`` precedence is implemented inside
    ``backend/scripts/run_parallel_simulation.py:984-1030`` (function
    ``create_model(use_boost=True)``). That code reads
    ``LLM_BOOST_API_KEY`` / ``LLM_BOOST_BASE_URL`` /
    ``LLM_BOOST_MODEL_NAME`` BEFORE falling back to ``LLM_*``.

    SWARM upstream wiring writes ``LLM_*`` (and the ``OPENAI_*`` twins)
    into the subprocess env. ``LLM_BOOST_*`` are a separate, finer
    per-call override that wins inside ``create_model`` regardless of
    what SWARM layered upstream — they read different env-var names
    and never collide.

    The contract "LLM_BOOST_* override continues to win over SWARM
    defaults when both are set" therefore holds by NON-MODIFICATION of
    the subprocess scripts."""
    pass

"""
Tests for ``Config.llm_for(role)`` resolver and ``Config.validate()``
regression sentinels.

The resolver and the validator have **different read semantics** by design:

    * ``Config.llm_for(role)``   reads ``os.environ`` at call time, so
                                  ``monkeypatch.setenv`` is the correct
                                  technique for scenarios 1-9.
    * ``Config.validate()``      reads the class attributes
                                  (``cls.LLM_API_KEY``, ``cls.ZEP_API_KEY``)
                                  which are bound once at module import.
                                  Scenarios 10-12 therefore use
                                  ``monkeypatch.setattr(Config, ...)`` and
                                  must NOT use ``setenv`` for the values
                                  under assertion.

Mixing the two techniques up will produce confusing, environment-dependent
failures. Each scenario-10/11/12 test carries an inline reminder.

Default values exercised by the resolver fallback:
    * ``LLM_BASE_URL``    -> ``https://api.openai.com/v1``
    * ``LLM_MODEL_NAME``  -> ``gpt-4o-mini``
    * ``LLM_API_KEY``     -> ``None``
"""

import pytest

from app.config import Config


# Per-test string constants — chosen to be obviously distinct from each
# other so an accidental field-swap surfaces immediately in assertions.
GLOBAL_API = "k"
GLOBAL_URL = "b"
GLOBAL_MODEL = "m"


# --------------------------------------------------------------------------
# Scenario 1 — All role envs unset, every role falls back to LLM_*
# --------------------------------------------------------------------------
@pytest.mark.parametrize("role", ["builder", "swarm", "judge"])
def test_llm_for_returns_global_when_role_envs_unset(monkeypatch, role):
    monkeypatch.setenv("LLM_API_KEY", GLOBAL_API)
    monkeypatch.setenv("LLM_BASE_URL", GLOBAL_URL)
    monkeypatch.setenv("LLM_MODEL_NAME", GLOBAL_MODEL)

    assert Config.llm_for(role) == (GLOBAL_API, GLOBAL_URL, GLOBAL_MODEL)


# --------------------------------------------------------------------------
# Scenario 2 — Single-field override: only model
# --------------------------------------------------------------------------
def test_llm_for_overrides_only_model_when_only_model_env_set(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", GLOBAL_API)
    monkeypatch.setenv("LLM_BASE_URL", GLOBAL_URL)
    monkeypatch.setenv("LLM_MODEL_NAME", GLOBAL_MODEL)
    monkeypatch.setenv("SWARM_LLM_MODEL_NAME", "m_swarm")

    assert Config.llm_for("swarm") == (GLOBAL_API, GLOBAL_URL, "m_swarm")


# --------------------------------------------------------------------------
# Scenario 3 — Single-field override: only base_url
# --------------------------------------------------------------------------
def test_llm_for_overrides_only_base_url_when_only_base_url_env_set(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", GLOBAL_API)
    monkeypatch.setenv("LLM_BASE_URL", GLOBAL_URL)
    monkeypatch.setenv("LLM_MODEL_NAME", GLOBAL_MODEL)
    monkeypatch.setenv("JUDGE_LLM_BASE_URL", "jb")

    assert Config.llm_for("judge") == (GLOBAL_API, "jb", GLOBAL_MODEL)


# --------------------------------------------------------------------------
# Scenario 4 — Single-field override: only api_key
# --------------------------------------------------------------------------
def test_llm_for_overrides_only_api_key_when_only_api_key_env_set(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", GLOBAL_API)
    monkeypatch.setenv("LLM_BASE_URL", GLOBAL_URL)
    monkeypatch.setenv("LLM_MODEL_NAME", GLOBAL_MODEL)
    monkeypatch.setenv("BUILDER_LLM_API_KEY", "bk")

    assert Config.llm_for("builder") == ("bk", GLOBAL_URL, GLOBAL_MODEL)


# --------------------------------------------------------------------------
# Scenario 5 — Full role group wins per-field
# --------------------------------------------------------------------------
def test_llm_for_full_role_group_wins_per_field(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", GLOBAL_API)
    monkeypatch.setenv("LLM_BASE_URL", GLOBAL_URL)
    monkeypatch.setenv("LLM_MODEL_NAME", GLOBAL_MODEL)
    monkeypatch.setenv("BUILDER_LLM_API_KEY", "bk")
    monkeypatch.setenv("BUILDER_LLM_BASE_URL", "bb")
    monkeypatch.setenv("BUILDER_LLM_MODEL_NAME", "bm")

    assert Config.llm_for("builder") == ("bk", "bb", "bm")


# --------------------------------------------------------------------------
# Scenario 6 — Cross-role isolation: builder's override does not leak to swarm
# --------------------------------------------------------------------------
def test_llm_for_role_envs_do_not_leak_across_roles(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", GLOBAL_API)
    monkeypatch.setenv("LLM_BASE_URL", GLOBAL_URL)
    monkeypatch.setenv("LLM_MODEL_NAME", GLOBAL_MODEL)
    monkeypatch.setenv("BUILDER_LLM_API_KEY", "bk")

    swarm_api_key, _, _ = Config.llm_for("swarm")

    assert swarm_api_key == GLOBAL_API
    assert swarm_api_key != "bk"


# --------------------------------------------------------------------------
# Scenario 7 — All three roles fully set: each returns its own triple
# --------------------------------------------------------------------------
def test_llm_for_returns_distinct_triples_for_each_fully_set_role(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", GLOBAL_API)
    monkeypatch.setenv("LLM_BASE_URL", GLOBAL_URL)
    monkeypatch.setenv("LLM_MODEL_NAME", GLOBAL_MODEL)

    monkeypatch.setenv("BUILDER_LLM_API_KEY", "bk")
    monkeypatch.setenv("BUILDER_LLM_BASE_URL", "bb")
    monkeypatch.setenv("BUILDER_LLM_MODEL_NAME", "bm")

    monkeypatch.setenv("SWARM_LLM_API_KEY", "sk")
    monkeypatch.setenv("SWARM_LLM_BASE_URL", "sb")
    monkeypatch.setenv("SWARM_LLM_MODEL_NAME", "sm")

    monkeypatch.setenv("JUDGE_LLM_API_KEY", "jk")
    monkeypatch.setenv("JUDGE_LLM_BASE_URL", "jb")
    monkeypatch.setenv("JUDGE_LLM_MODEL_NAME", "jm")

    assert Config.llm_for("builder") == ("bk", "bb", "bm")
    assert Config.llm_for("swarm") == ("sk", "sb", "sm")
    assert Config.llm_for("judge") == ("jk", "jb", "jm")


# --------------------------------------------------------------------------
# Scenario 8 — Empty-string role env is treated as unset
# --------------------------------------------------------------------------
def test_llm_for_treats_empty_string_role_env_as_unset(monkeypatch):
    # An empty SWARM_LLM_API_KEY must NOT shadow the global LLM_API_KEY.
    # This locks the falsy-``or`` short-circuit semantics in the resolver.
    monkeypatch.setenv("LLM_API_KEY", GLOBAL_API)
    monkeypatch.setenv("LLM_BASE_URL", GLOBAL_URL)
    monkeypatch.setenv("LLM_MODEL_NAME", GLOBAL_MODEL)
    monkeypatch.setenv("SWARM_LLM_API_KEY", "")

    api_key, _, _ = Config.llm_for("swarm")

    assert api_key == GLOBAL_API
    assert api_key != ""


# --------------------------------------------------------------------------
# Scenario 9 — Invalid role raises ValueError
# --------------------------------------------------------------------------
def test_llm_for_invalid_role_raises_valueerror(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", GLOBAL_API)

    with pytest.raises(ValueError, match="unknown role"):
        Config.llm_for("unknown")


# --------------------------------------------------------------------------
# Scenarios 10-12 — ``Config.validate()`` regression sentinels
#
# These tests use ``monkeypatch.setattr`` on the Config class — NOT setenv —
# because ``validate()`` reads class attributes that were bound at module
# import time. Switching to ``setenv`` here will silently produce
# environment-dependent failures.
# --------------------------------------------------------------------------


# Scenario 10 — happy path
def test_validate_returns_empty_when_globals_set_and_no_role_envs(monkeypatch):
    # ``setattr`` because ``validate()`` reads ``cls.LLM_API_KEY`` /
    # ``cls.ZEP_API_KEY`` (class attrs frozen at import).
    monkeypatch.setattr(Config, "LLM_API_KEY", "k")
    monkeypatch.setattr(Config, "ZEP_API_KEY", "z")

    assert Config.validate() == []


# Scenario 11 — missing global LLM_API_KEY
def test_validate_flags_missing_global_llm_api_key(monkeypatch):
    # ``setattr`` because ``validate()`` reads class attrs, not env.
    monkeypatch.setattr(Config, "LLM_API_KEY", None)
    monkeypatch.setattr(Config, "ZEP_API_KEY", "z")

    errors = Config.validate()

    assert "LLM_API_KEY 未配置" in errors


# Scenario 12 — role envs do not satisfy the global LLM_API_KEY check
def test_validate_unchanged_by_role_only_envs(monkeypatch):
    # ``setattr`` flips the class attrs that ``validate()`` actually reads.
    # The ``setenv`` call below proves a role var IS present in os.environ
    # but is correctly ignored by ``validate()``.
    monkeypatch.setattr(Config, "LLM_API_KEY", None)
    monkeypatch.setattr(Config, "ZEP_API_KEY", "z")
    monkeypatch.setenv("BUILDER_LLM_API_KEY", "bk")

    errors = Config.validate()

    assert "LLM_API_KEY 未配置" in errors

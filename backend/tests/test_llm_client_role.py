"""
Tests for ``LLMClient(role=...)`` precedence semantics and the four
service-layer call sites that wire their default credential source.

The 15 scenarios split into three logical groups, each with a different
monkeypatching technique:

    * Scenarios 1-8, 15 — ``LLMClient.__init__`` precedence matrix.
        - Patch ``app.utils.llm_client.OpenAI`` to ``MagicMock`` so the
          constructor does not fire a real network init.
        - For scenarios that depend on the **no-role** branch reaching
          ``Config.LLM_*`` (1, 8, 15), patch the class attributes via
          ``monkeypatch.setattr(Config, ...)``. The class attrs are bound
          at module import time and the autouse ``_clean_llm_env``
          fixture only purges ``os.environ`` — so ``setenv`` alone does
          NOT make ``self.api_key`` deterministic on a developer machine
          with a populated ``.env``.
        - Scenarios that route through ``Config.llm_for(role)`` (2-7) use
          ``setenv`` because the resolver re-reads ``os.environ`` at call
          time. This mirrors T-001's scenario-1-9 / scenario-10-12 split.

    * Scenarios 9, 10 — service default-role wiring.
        - Replace ``LLMClient.__init__`` with a kwargs-capturing stub.
          The body never runs, so neither the api-key guard nor the
          ``OpenAI(...)`` line fires. No ``OpenAI`` patch needed.
        - Scenario 10 also passes ``zep_tools=MagicMock()`` to short-
          circuit ``ZepToolsService()`` lazy init, which would otherwise
          construct a real ``Zep`` client.

    * Scenarios 11-14 — light-touch services.
        - Patch ``Config.llm_for`` to a sentinel triple AND patch the
          service module's own ``OpenAI`` symbol. The two services
          (``OasisProfileGenerator`` / ``SimulationConfigGenerator``)
          construct ``OpenAI(...)`` directly from their own module-level
          import — patching only ``app.utils.llm_client.OpenAI`` is
          invisible to them.

Test-name convention follows T-002.md "Test Specifications (Basic)".
Per-scenario string constants are chosen to be obviously distinct so an
accidental field-swap surfaces immediately in the failure message.
"""

from unittest.mock import MagicMock

import pytest

from app.config import Config
from app.services.oasis_profile_generator import OasisProfileGenerator
from app.services.ontology_generator import OntologyGenerator
from app.services.report_agent import ReportAgent
from app.services.simulation_config_generator import SimulationConfigGenerator
from app.utils.llm_client import LLMClient


# Per-test string constants — all distinct so a field-swap fails loudly.
GLOBAL_API = "k"
GLOBAL_URL = "b"
GLOBAL_MODEL = "m"


# --------------------------------------------------------------------------
# Scenario 1 — No role, no kwargs: v0 behaviour falls through to Config.LLM_*
#
# Why setattr (not setenv): ``LLMClient.__init__`` reads
# ``Config.LLM_API_KEY``/``Config.LLM_BASE_URL``/``Config.LLM_MODEL_NAME``
# which are class attributes bound at module import. The autouse fixture
# clears ``os.environ`` but the class attrs retain whatever ``.env`` had
# at import time. ``setenv`` alone is invisible to this code path.
# --------------------------------------------------------------------------
def test_llmclient_no_role_uses_global_llm_envs(monkeypatch):
    monkeypatch.setattr(Config, "LLM_API_KEY", GLOBAL_API)
    monkeypatch.setattr(Config, "LLM_BASE_URL", GLOBAL_URL)
    monkeypatch.setattr(Config, "LLM_MODEL_NAME", GLOBAL_MODEL)
    monkeypatch.setattr("app.utils.llm_client.OpenAI", MagicMock())

    client = LLMClient()

    assert client.api_key == GLOBAL_API
    assert client.base_url == GLOBAL_URL
    assert client.model == GLOBAL_MODEL


# --------------------------------------------------------------------------
# Scenario 2 — role="builder" routes through Config.llm_for, picks up
# BUILDER_LLM_MODEL_NAME and falls back to LLM_API_KEY for api_key.
# --------------------------------------------------------------------------
def test_llmclient_role_builder_resolves_via_config_llm_for(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", GLOBAL_API)
    monkeypatch.setenv("BUILDER_LLM_MODEL_NAME", "bm")
    monkeypatch.setattr("app.utils.llm_client.OpenAI", MagicMock())

    client = LLMClient(role="builder")

    assert client.api_key == GLOBAL_API
    assert client.model == "bm"


# --------------------------------------------------------------------------
# Scenario 3 — role="swarm" picks up the full SWARM_LLM_* triple,
# isolated from any builder/judge envs.
# --------------------------------------------------------------------------
def test_llmclient_role_swarm_resolves_to_swarm_envs(monkeypatch):
    monkeypatch.setenv("SWARM_LLM_API_KEY", "sk")
    monkeypatch.setenv("SWARM_LLM_BASE_URL", "sb")
    monkeypatch.setenv("SWARM_LLM_MODEL_NAME", "sm")
    monkeypatch.setattr("app.utils.llm_client.OpenAI", MagicMock())

    client = LLMClient(role="swarm")

    assert client.api_key == "sk"
    assert client.base_url == "sb"
    assert client.model == "sm"


# --------------------------------------------------------------------------
# Scenario 4 — role="judge" picks up the full JUDGE_LLM_* triple,
# isolated from any builder/swarm envs.
# --------------------------------------------------------------------------
def test_llmclient_role_judge_resolves_to_judge_envs(monkeypatch):
    monkeypatch.setenv("JUDGE_LLM_API_KEY", "jk")
    monkeypatch.setenv("JUDGE_LLM_BASE_URL", "jb")
    monkeypatch.setenv("JUDGE_LLM_MODEL_NAME", "jm")
    monkeypatch.setattr("app.utils.llm_client.OpenAI", MagicMock())

    client = LLMClient(role="judge")

    assert client.api_key == "jk"
    assert client.base_url == "jb"
    assert client.model == "jm"


# --------------------------------------------------------------------------
# Scenario 5 — Explicit api_key kwarg dominates the role group.
# base_url and model still come from BUILDER_LLM_*.
# --------------------------------------------------------------------------
def test_llmclient_explicit_api_key_kwarg_beats_role(monkeypatch):
    monkeypatch.setenv("BUILDER_LLM_API_KEY", "bk")
    monkeypatch.setenv("BUILDER_LLM_BASE_URL", "bb")
    monkeypatch.setenv("BUILDER_LLM_MODEL_NAME", "bm")
    monkeypatch.setattr("app.utils.llm_client.OpenAI", MagicMock())

    client = LLMClient(api_key="x", role="builder")

    assert client.api_key == "x"
    assert client.base_url == "bb"
    assert client.model == "bm"


# --------------------------------------------------------------------------
# Scenario 6 — Explicit base_url kwarg dominates the role group.
# api_key and model still come from BUILDER_LLM_*.
# --------------------------------------------------------------------------
def test_llmclient_explicit_base_url_kwarg_beats_role(monkeypatch):
    monkeypatch.setenv("BUILDER_LLM_API_KEY", "bk")
    monkeypatch.setenv("BUILDER_LLM_BASE_URL", "bb")
    monkeypatch.setenv("BUILDER_LLM_MODEL_NAME", "bm")
    monkeypatch.setattr("app.utils.llm_client.OpenAI", MagicMock())

    client = LLMClient(base_url="x", role="builder")

    assert client.api_key == "bk"
    assert client.base_url == "x"
    assert client.model == "bm"


# --------------------------------------------------------------------------
# Scenario 7 — Explicit model kwarg dominates the role group.
# api_key and base_url still come from BUILDER_LLM_*.
# --------------------------------------------------------------------------
def test_llmclient_explicit_model_kwarg_beats_role(monkeypatch):
    monkeypatch.setenv("BUILDER_LLM_API_KEY", "bk")
    monkeypatch.setenv("BUILDER_LLM_BASE_URL", "bb")
    monkeypatch.setenv("BUILDER_LLM_MODEL_NAME", "bm")
    monkeypatch.setattr("app.utils.llm_client.OpenAI", MagicMock())

    client = LLMClient(model="x", role="builder")

    assert client.api_key == "bk"
    assert client.base_url == "bb"
    assert client.model == "x"


# --------------------------------------------------------------------------
# Scenario 8 — Missing api_key with role still raises the existing
# ``"LLM_API_KEY 未配置"`` error. Locks the public error-message contract.
#
# Why setattr (not setenv): the precedence chain ends with
# ``or Config.LLM_API_KEY``. The class attr is populated from ``.env`` at
# import time, so without ``setattr(Config, "LLM_API_KEY", None)`` the
# fallback finds a truthy value and the raise never fires. Mirrors the
# scenario-10-12 trap from T-001.
# --------------------------------------------------------------------------
def test_llmclient_missing_api_key_with_role_raises(monkeypatch):
    monkeypatch.setattr(Config, "LLM_API_KEY", None)
    monkeypatch.setattr("app.utils.llm_client.OpenAI", MagicMock())

    with pytest.raises(ValueError, match="LLM_API_KEY"):
        LLMClient(role="builder")


# --------------------------------------------------------------------------
# Scenario 9 — OntologyGenerator() default routes LLMClient(role="builder").
#
# Why kwargs-capture stub: the assertion is on the role kwarg passed to
# ``LLMClient.__init__``, not on the resulting ``self.api_key``. Replacing
# ``__init__`` with a stub skips both the api-key guard and the
# ``OpenAI(...)`` call, so no further patching is needed.
# --------------------------------------------------------------------------
def test_ontology_generator_defaults_to_builder_role(monkeypatch):
    captured = {}

    def fake_init(self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(LLMClient, "__init__", fake_init)

    OntologyGenerator()

    assert captured.get("role") == "builder"


# --------------------------------------------------------------------------
# Scenario 10 — ReportAgent() default routes LLMClient(role="judge").
#
# Why zep_tools=MagicMock(): ``ReportAgent.__init__`` also calls
# ``ZepToolsService()`` at line 907, which constructs a real ``Zep``
# client from ``Config.ZEP_API_KEY`` (populated by ``.env``). Passing an
# explicit ``zep_tools`` short-circuits the lazy default — scoping the
# test to the LLM kwarg only.
# --------------------------------------------------------------------------
def test_report_agent_defaults_to_judge_role(monkeypatch):
    captured = {}

    def fake_init(self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(LLMClient, "__init__", fake_init)

    ReportAgent(
        graph_id="g",
        simulation_id="s",
        simulation_requirement="r",
        zep_tools=MagicMock(),
    )

    assert captured.get("role") == "judge"


# --------------------------------------------------------------------------
# Scenario 11 — OasisProfileGenerator() seeds api_key/base_url/model_name
# from Config.llm_for("builder").
#
# Why patch the service module's OpenAI: the service imports OpenAI at
# its own module top-level and constructs OpenAI(...) directly. A
# monkeypatch on app.utils.llm_client.OpenAI is invisible here.
# --------------------------------------------------------------------------
def test_oasis_profile_generator_defaults_seeded_from_builder_role(monkeypatch):
    monkeypatch.setattr(
        Config,
        "llm_for",
        classmethod(lambda cls, role: ("sentinel_key", "sentinel_url", "sentinel_model")),
    )
    monkeypatch.setattr(
        "app.services.oasis_profile_generator.OpenAI", MagicMock()
    )

    gen = OasisProfileGenerator()

    assert gen.api_key == "sentinel_key"
    assert gen.base_url == "sentinel_url"
    assert gen.model_name == "sentinel_model"


# --------------------------------------------------------------------------
# Scenario 12 — SimulationConfigGenerator() seeds api_key/base_url/
# model_name from Config.llm_for("builder"). Same shape as scenario 11
# but patches the simulation_config_generator module's OpenAI symbol.
# --------------------------------------------------------------------------
def test_simulation_config_generator_defaults_seeded_from_builder_role(monkeypatch):
    monkeypatch.setattr(
        Config,
        "llm_for",
        classmethod(lambda cls, role: ("sentinel_key", "sentinel_url", "sentinel_model")),
    )
    monkeypatch.setattr(
        "app.services.simulation_config_generator.OpenAI", MagicMock()
    )

    gen = SimulationConfigGenerator()

    assert gen.api_key == "sentinel_key"
    assert gen.base_url == "sentinel_url"
    assert gen.model_name == "sentinel_model"


# --------------------------------------------------------------------------
# Scenario 13 — Explicit api_key kwarg to OasisProfileGenerator wins.
# base_url and model_name still come from the patched llm_for sentinel.
# --------------------------------------------------------------------------
def test_oasis_profile_generator_explicit_api_key_kwarg_wins(monkeypatch):
    monkeypatch.setattr(
        Config,
        "llm_for",
        classmethod(lambda cls, role: ("sentinel_key", "sentinel_url", "sentinel_model")),
    )
    monkeypatch.setattr(
        "app.services.oasis_profile_generator.OpenAI", MagicMock()
    )

    gen = OasisProfileGenerator(api_key="explicit")

    assert gen.api_key == "explicit"
    assert gen.base_url == "sentinel_url"
    assert gen.model_name == "sentinel_model"


# --------------------------------------------------------------------------
# Scenario 14 — Explicit api_key kwarg to SimulationConfigGenerator wins.
# base_url and model_name still come from the patched llm_for sentinel.
# --------------------------------------------------------------------------
def test_simulation_config_generator_explicit_api_key_kwarg_wins(monkeypatch):
    monkeypatch.setattr(
        Config,
        "llm_for",
        classmethod(lambda cls, role: ("sentinel_key", "sentinel_url", "sentinel_model")),
    )
    monkeypatch.setattr(
        "app.services.simulation_config_generator.OpenAI", MagicMock()
    )

    gen = SimulationConfigGenerator(api_key="explicit")

    assert gen.api_key == "explicit"
    assert gen.base_url == "sentinel_url"
    assert gen.model_name == "sentinel_model"


# --------------------------------------------------------------------------
# Scenario 15 — LLMClient(role=None) is byte-for-byte identical to
# LLMClient() — locks AC-2 backward-compat for the explicit-None call.
#
# Why setattr (not setenv): same trap as scenario 1 — the no-role branch
# reads class attrs directly.
# --------------------------------------------------------------------------
def test_llmclient_role_none_is_identical_to_no_role(monkeypatch):
    monkeypatch.setattr(Config, "LLM_API_KEY", GLOBAL_API)
    monkeypatch.setattr(Config, "LLM_BASE_URL", GLOBAL_URL)
    monkeypatch.setattr(Config, "LLM_MODEL_NAME", GLOBAL_MODEL)
    monkeypatch.setattr("app.utils.llm_client.OpenAI", MagicMock())

    no_role = LLMClient()
    explicit_none = LLMClient(role=None)

    assert no_role.api_key == explicit_none.api_key
    assert no_role.base_url == explicit_none.base_url
    assert no_role.model == explicit_none.model

"""
Shared pytest fixtures for the backend test package.

The repository ships a populated ``.env`` at ``MiroFish/.env`` which
``backend/app/config.py`` loads via ``dotenv.load_dotenv(..., override=True)``
at module import. Without explicit hygiene the live secrets contaminate any
test that exercises ``Config.llm_for`` or related env-driven logic, masking
real failures behind a developer's local credentials.

The autouse fixture below clears every LLM-related env var before each test
runs. Tests that require specific values must ``monkeypatch.setenv`` them
explicitly inside the test body — making each test self-describing.

The fixture is intentionally scoped to ``backend/tests/`` (not ``backend/``)
so that ``backend/scripts/test_profile_format.py`` is not touched.
"""

import pytest


_LLM_ENV_VARS = (
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL_NAME",
    "BUILDER_LLM_API_KEY",
    "BUILDER_LLM_BASE_URL",
    "BUILDER_LLM_MODEL_NAME",
    "SWARM_LLM_API_KEY",
    "SWARM_LLM_BASE_URL",
    "SWARM_LLM_MODEL_NAME",
    "JUDGE_LLM_API_KEY",
    "JUDGE_LLM_BASE_URL",
    "JUDGE_LLM_MODEL_NAME",
)


@pytest.fixture(autouse=True)
def _clean_llm_env(monkeypatch):
    """Remove every LLM-related env var so each test starts from a known
    clean slate. ``raising=False`` keeps this idempotent on machines where
    the variable was never present."""
    for name in _LLM_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    yield

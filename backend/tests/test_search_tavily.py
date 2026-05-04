"""
Tests for the search-provider abstraction at
``app.services.search`` — ``SearchProvider`` ABC, ``SearchResult``
TypedDict, and the ``TavilyProvider`` REST client.

The 13 scenarios split into four logical groups, each with a different
mocking shape:

    * Scenarios 1, 2 — pure-Python contract checks. No HTTP, no env.
        - ABC enforces ``search`` (instantiating a subclass that does
          not implement it raises ``TypeError``).
        - ``SearchResult`` TypedDict shape carries the three required
          string fields.

    * Scenarios 3, 4, 13 — constructor api_key resolution.
        - Explicit kwarg dominates env.
        - ``TAVILY_API_KEY`` env populates ``self.api_key`` when no
          kwarg given.
        - When neither is set, ``self.api_key is None``; ``search()``
          early-returns ``[]`` without ever calling ``requests.post``.

    * Scenarios 5, 6, 7, 11 — happy path + body inspection +
      truncation + missing-field defensive coercion.
        - Mock ``app.services.search.tavily.requests.post`` returning
          a ``MagicMock`` whose ``.status_code`` is 200 and whose
          ``.json()`` returns a Tavily-shaped dict.
        - Verify the request body carries ``api_key``, ``query``,
          ``max_results``.
        - Verify the response list is truncated to ``[:k]``.
        - Verify missing/null Tavily fields degrade to ``""``.

    * Scenarios 8, 9, 10, 12 — failure modes.
        - HTTP non-200, ``requests.Timeout``, malformed JSON each
          return ``[]`` and log a ``WARNING``.
        - Scenario 12 is the umbrella: every failure path is
          re-checked under one parametrize and ``pytest.raises`` is
          NOT triggered (no exception escapes ``search()``).

The autouse ``_clean_llm_env`` fixture in ``conftest.py`` clears 12
LLM env vars but does NOT clear ``TAVILY_API_KEY``. Every test that
exercises the env-resolution path explicitly calls
``monkeypatch.delenv("TAVILY_API_KEY", raising=False)`` to avoid
contamination from a developer's local ``.env``.

Mock target convention (locked): every test that exercises ``.search``
patches ``app.services.search.tavily.requests.post`` — never the
global ``requests.post``. The ``requests`` module is imported at
module level into ``app.services.search.tavily``; patching the
attribute on that import binds the mock at the call site.

Test-name and per-scenario string constants follow the
``test_llm_client_role.py`` precedent.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from app.services.search.base import SearchProvider, SearchResult
from app.services.search.tavily import TavilyProvider


# Per-test string constants — distinct so a field-swap fails loudly.
TAVILY_KWARG_KEY = "kwarg-key"
TAVILY_ENV_KEY = "env-key"

TAVILY_MOCK_TARGET = "app.services.search.tavily.requests.post"


# Tavily-shaped sample response — three results so scenarios 5, 6, 7
# can exercise list-length, truncation, and body inspection cleanly.
SAMPLE_TAVILY_RESPONSE = {
    "results": [
        {"title": "t1", "url": "https://u1", "content": "s1"},
        {"title": "t2", "url": "https://u2", "content": "s2"},
        {"title": "t3", "url": "https://u3", "content": "s3"},
    ]
}


def _mock_response(status_code=200, json_payload=None, json_side_effect=None):
    """Helper — build a MagicMock shaped like a ``requests.Response``."""
    response = MagicMock()
    response.status_code = status_code
    if json_side_effect is not None:
        response.json.side_effect = json_side_effect
    else:
        response.json.return_value = json_payload if json_payload is not None else {}
    return response


# --------------------------------------------------------------------------
# Scenario 1 — ABC enforces ``search`` method.
#
# Subclassing ``SearchProvider`` without implementing ``search`` and
# attempting to instantiate must raise ``TypeError``. This is the
# default Python ABC contract — no extra implementation needed in
# ``base.py`` beyond ``@abstractmethod``.
# --------------------------------------------------------------------------
def test_abc_enforces_search_method():
    class IncompleteProvider(SearchProvider):
        # Deliberately omits ``search`` — instantiation must fail.
        pass

    with pytest.raises(TypeError, match="search"):
        IncompleteProvider()


# --------------------------------------------------------------------------
# Scenario 2 — ``SearchResult`` TypedDict shape carries the three
# required string fields. Constructing a dict-literal with the three
# keys typed as ``str`` must satisfy the TypedDict contract.
#
# Note: TypedDict is structural at runtime — the assertion is the
# shape (keys present, str values) rather than a class-level check.
# --------------------------------------------------------------------------
def test_search_result_typeddict_shape():
    result: SearchResult = {"title": "t", "url": "u", "snippet": "s"}

    assert result["title"] == "t"
    assert result["url"] == "u"
    assert result["snippet"] == "s"
    assert set(result.keys()) == {"title", "url", "snippet"}
    for value in result.values():
        assert isinstance(value, str)


# --------------------------------------------------------------------------
# Scenario 3 — ``TavilyProvider(api_key="x")`` reads api_key from kwarg.
#
# Why monkeypatch.delenv first: the autouse ``_clean_llm_env`` fixture
# does NOT clear ``TAVILY_API_KEY``. A developer's local ``.env`` could
# silently set the env value and mask whether the kwarg was actually
# read — explicit delenv guarantees the kwarg path is the only source.
# --------------------------------------------------------------------------
def test_tavily_provider_reads_api_key_from_kwarg(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    provider = TavilyProvider(api_key=TAVILY_KWARG_KEY)

    assert provider.api_key == TAVILY_KWARG_KEY


# --------------------------------------------------------------------------
# Scenario 4 — ``TavilyProvider()`` reads api_key from
# ``TAVILY_API_KEY`` env var when no kwarg is given.
# --------------------------------------------------------------------------
def test_tavily_provider_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", TAVILY_ENV_KEY)

    provider = TavilyProvider()

    assert provider.api_key == TAVILY_ENV_KEY


# --------------------------------------------------------------------------
# Scenario 5 — ``TavilyProvider.search`` happy path.
#
# Mock returns a 3-result Tavily response; call ``search("q", k=3)``;
# expect a list of 3 ``SearchResult`` with mapped fields
# (``content`` → ``snippet``).
# --------------------------------------------------------------------------
def test_tavily_search_happy_path(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with patch(TAVILY_MOCK_TARGET) as mock_post:
        mock_post.return_value = _mock_response(
            status_code=200, json_payload=SAMPLE_TAVILY_RESPONSE
        )

        provider = TavilyProvider(api_key=TAVILY_KWARG_KEY)
        results = provider.search("q", k=3)

    assert len(results) == 3
    assert results[0]["title"] == "t1"
    assert results[0]["url"] == "https://u1"
    assert results[0]["snippet"] == "s1"
    assert results[2]["title"] == "t3"
    assert results[2]["url"] == "https://u3"
    assert results[2]["snippet"] == "s3"


# --------------------------------------------------------------------------
# Scenario 6 — ``search`` respects ``k`` cap.
#
# Mock returns 3 results; call ``search("q", k=2)``; expect list
# length ≤ 2. The provider relies on Tavily's ``max_results`` param
# in the request body but defensively truncates if Tavily over-returns.
# --------------------------------------------------------------------------
def test_tavily_search_truncates_to_k(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with patch(TAVILY_MOCK_TARGET) as mock_post:
        mock_post.return_value = _mock_response(
            status_code=200, json_payload=SAMPLE_TAVILY_RESPONSE
        )

        provider = TavilyProvider(api_key=TAVILY_KWARG_KEY)
        results = provider.search("q", k=2)

    assert len(results) <= 2
    assert len(results) == 2


# --------------------------------------------------------------------------
# Scenario 7 — ``search`` sends correct JSON body.
#
# Inspect the kwargs passed to the mocked ``requests.post``: the
# ``json`` payload must contain ``api_key``, ``query``, ``max_results``.
# --------------------------------------------------------------------------
def test_tavily_search_sends_correct_json_body(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with patch(TAVILY_MOCK_TARGET) as mock_post:
        mock_post.return_value = _mock_response(
            status_code=200, json_payload={"results": []}
        )

        provider = TavilyProvider(api_key=TAVILY_KWARG_KEY)
        provider.search("q", k=5)

    assert mock_post.call_count == 1
    _, kwargs = mock_post.call_args
    body = kwargs["json"]
    assert body["api_key"] == TAVILY_KWARG_KEY
    assert body["query"] == "q"
    assert body["max_results"] == 5


# --------------------------------------------------------------------------
# Scenario 8 — non-200 HTTP response → safe failure.
#
# Mock returns HTTP 500; provider must return ``[]`` and not raise.
# A WARNING is emitted; assert via ``caplog`` substring (do NOT couple
# to exact message text).
# --------------------------------------------------------------------------
def test_tavily_search_non_200_returns_empty(monkeypatch, caplog):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with patch(TAVILY_MOCK_TARGET) as mock_post:
        mock_post.return_value = _mock_response(status_code=500, json_payload={})

        provider = TavilyProvider(api_key=TAVILY_KWARG_KEY)
        with caplog.at_level("WARNING"):
            results = provider.search("q", k=3)

    assert results == []
    assert any(
        record.levelname == "WARNING" and "500" in record.getMessage()
        for record in caplog.records
    )


# --------------------------------------------------------------------------
# Scenario 9 — ``requests.Timeout`` → safe failure.
#
# Mock raises ``requests.Timeout``; provider must return ``[]`` and
# not let the exception escape. WARNING substring "timeout" expected.
# --------------------------------------------------------------------------
def test_tavily_search_timeout_returns_empty(monkeypatch, caplog):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with patch(TAVILY_MOCK_TARGET) as mock_post:
        mock_post.side_effect = requests.Timeout()

        provider = TavilyProvider(api_key=TAVILY_KWARG_KEY)
        with caplog.at_level("WARNING"):
            results = provider.search("q", k=3)

    assert results == []
    assert any(
        record.levelname == "WARNING" and "timeout" in record.getMessage().lower()
        for record in caplog.records
    )


# --------------------------------------------------------------------------
# Scenario 10 — malformed JSON in 200 response → safe failure.
#
# Mock returns HTTP 200 but ``response.json()`` raises ``ValueError``
# (the runtime shape ``requests`` exposes for ``JSONDecodeError``).
# Provider must return ``[]`` and not raise. WARNING substring
# "malformed" or "json" expected.
# --------------------------------------------------------------------------
def test_tavily_search_malformed_json_returns_empty(monkeypatch, caplog):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with patch(TAVILY_MOCK_TARGET) as mock_post:
        mock_post.return_value = _mock_response(
            status_code=200, json_side_effect=ValueError("not json")
        )

        provider = TavilyProvider(api_key=TAVILY_KWARG_KEY)
        with caplog.at_level("WARNING"):
            results = provider.search("q", k=3)

    assert results == []
    assert any(
        record.levelname == "WARNING"
        and ("json" in record.getMessage().lower() or "malformed" in record.getMessage().lower())
        for record in caplog.records
    )


# --------------------------------------------------------------------------
# Scenario 11 — missing fields in result item → graceful degradation.
#
# Mock returns a 200 response whose first result has only ``title``
# (no ``url``, no ``content``). Provider must populate the missing
# fields with ``""`` (empty string) — never ``None``, never raise.
# --------------------------------------------------------------------------
def test_tavily_search_missing_fields_default_to_empty_string(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    payload = {"results": [{"title": "only-title"}]}
    with patch(TAVILY_MOCK_TARGET) as mock_post:
        mock_post.return_value = _mock_response(status_code=200, json_payload=payload)

        provider = TavilyProvider(api_key=TAVILY_KWARG_KEY)
        results = provider.search("q", k=3)

    assert len(results) == 1
    assert results[0]["title"] == "only-title"
    assert results[0]["url"] == ""
    assert results[0]["snippet"] == ""


# --------------------------------------------------------------------------
# Scenario 12 — umbrella: provider does not raise out of ``search``
# for any covered failure path.
#
# Parametrized over the four documented failure modes (non-200,
# Timeout, RequestException sibling, malformed JSON). For each, the
# call MUST return ``[]`` and MUST NOT raise. ``pytest.raises`` is
# explicitly NOT used — any escaping exception fails the test.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "failure_kind",
    ["non_200", "timeout", "network_error", "malformed_json"],
)
def test_tavily_search_never_raises_on_any_failure(monkeypatch, failure_kind):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with patch(TAVILY_MOCK_TARGET) as mock_post:
        if failure_kind == "non_200":
            mock_post.return_value = _mock_response(status_code=500, json_payload={})
        elif failure_kind == "timeout":
            mock_post.side_effect = requests.Timeout()
        elif failure_kind == "network_error":
            mock_post.side_effect = requests.ConnectionError()
        elif failure_kind == "malformed_json":
            mock_post.return_value = _mock_response(
                status_code=200, json_side_effect=ValueError("not json")
            )

        provider = TavilyProvider(api_key=TAVILY_KWARG_KEY)
        # No try/except, no pytest.raises — any escaping exception
        # fails this test by design.
        results = provider.search("q", k=3)

    assert results == []


# --------------------------------------------------------------------------
# Scenario 13 (advisor's implicit case) — api_key fully unset
# (kwarg None AND env unset) → ``[]`` + WARNING + ``requests.post``
# is NEVER called.
#
# This locks the never-raises contract for the both-unset branch:
# the constructor does NOT raise; ``search()`` early-returns ``[]``
# without making an HTTP call.
# --------------------------------------------------------------------------
def test_tavily_search_no_api_key_returns_empty_without_http_call(monkeypatch, caplog):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with patch(TAVILY_MOCK_TARGET) as mock_post:
        provider = TavilyProvider()
        with caplog.at_level("WARNING"):
            results = provider.search("anything", k=3)

    assert results == []
    assert mock_post.call_count == 0
    assert any(
        record.levelname == "WARNING"
        and ("api_key" in record.getMessage().lower() or "tavily" in record.getMessage().lower())
        for record in caplog.records
    )

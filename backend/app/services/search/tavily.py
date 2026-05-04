"""
Tavily search-provider implementation.

HTTP transport
--------------
Direct REST via ``requests`` (transitive via ``huggingface-hub`` /
``transformers`` / ``tiktoken`` / others). No new top-level dependency
added to ``backend/pyproject.toml``: the existing transitive pin is
reused. Synchronous ``requests.post`` matches the per-agent serial
search loop in ``AgentResearchService.run`` (no inner concurrency at
the provider layer).

Failure-mode contract
---------------------
Every failure path returns ``[]`` and logs ``WARNING``. The provider
MUST NOT raise out of ``search()``. This contract removes the burden
of ``try/except`` from the caller (``AgentResearchService``), which
inspects the empty list — not exceptions — to decide whether the
agent gets research context for this turn. Failure paths covered:

    * ``api_key`` fully unset (kwarg ``None`` AND env unset)
    * HTTP non-200 response
    * ``requests.Timeout``
    * ``requests.RequestException`` (network / DNS / connection
      refused / etc.)
    * Malformed JSON in a 200 response (``ValueError`` /
      ``JSONDecodeError``)

Field mapping
-------------
Tavily's response uses ``content`` for the per-result text snippet;
the abstraction calls it ``snippet``. The rename happens here. Other
keys map identically: ``title -> title``, ``url -> url``. Missing or
null Tavily fields degrade to ``""`` (empty string) — never ``None``,
never raise.

Truncation
----------
Even though ``max_results=k`` is sent in the request body, the
provider defensively truncates to ``[:k]`` on response. If Tavily
ever over-returns, the caller still sees a list of length ``<= k``.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import requests  # imported at module level — mock target convention

from .base import SearchProvider, SearchResult


logger = logging.getLogger(__name__)


# Module-level constants — single edit point if Tavily changes endpoint
# or the agreed timeout shifts.
TAVILY_API_URL = "https://api.tavily.com/search"
TAVILY_REQUEST_TIMEOUT_SECONDS = 10


class TavilyProvider(SearchProvider):
    """
    Tavily REST search provider.

    Resolves the API key from an explicit kwarg first, then the
    ``TAVILY_API_KEY`` environment variable. When both are unset,
    ``self.api_key`` is ``None`` and ``search()`` early-returns ``[]``
    without making an HTTP call — preserving the never-raises contract.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Construct the provider.

        Args:
            api_key: Explicit API key. Wins over ``TAVILY_API_KEY``
                env var when provided. ``None`` (default) falls back
                to the env var, then to ``None`` if env is also unset.
        """
        # Precedence: explicit kwarg > env var > None.
        # No HTTP client constructed at init — requests.post is per-call.
        self.api_key: Optional[str] = api_key or os.environ.get("TAVILY_API_KEY")

    def search(self, query: str, k: int = 1) -> list[SearchResult]:
        """
        Query Tavily and return up to ``k`` ``SearchResult`` items.

        Behaviour contracts:

        * api_key fully unset -> ``[]`` + WARNING (no HTTP call).
        * Non-200 response    -> ``[]`` + WARNING.
        * Timeout / network   -> ``[]`` + WARNING.
        * Malformed JSON      -> ``[]`` + WARNING.
        * Missing item fields -> ``""`` per missing field (not skipped).
        * Tavily over-returns -> truncated to ``[:k]``.

        Args:
            query: Free-text search query.
            k: Maximum number of results requested. Sent as
                ``max_results`` in the JSON body and re-enforced via
                ``[:k]`` truncation on the response.

        Returns:
            List of ``SearchResult`` items, length ``<= k``. Empty
            list on any failure path or zero-hit response.
        """
        # Truncate query in log lines to avoid log bloat from long inputs.
        query_for_log = query[:80]

        if not self.api_key:
            logger.warning("Tavily search skipped: TAVILY_API_KEY not configured")
            return []

        try:
            response = requests.post(
                TAVILY_API_URL,
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "max_results": k,
                },
                timeout=TAVILY_REQUEST_TIMEOUT_SECONDS,
            )

            if response.status_code != 200:
                logger.warning(
                    "Tavily HTTP %s for query=%r",
                    response.status_code,
                    query_for_log,
                )
                return []

            data = response.json()
        except requests.Timeout:
            logger.warning("Tavily timeout for query=%r", query_for_log)
            return []
        except requests.RequestException as exc:
            # Catches ConnectionError, HTTPError, and other network siblings.
            logger.warning(
                "Tavily network error for query=%r: %s",
                query_for_log,
                exc,
            )
            return []
        except (ValueError, TypeError) as exc:
            # ``response.json()`` raises ``requests.exceptions.JSONDecodeError``,
            # a ``ValueError`` subclass — caught here. ``TypeError`` covers
            # the rare case of a non-string body.
            logger.warning(
                "Tavily malformed JSON for query=%r: %s",
                query_for_log,
                exc,
            )
            return []

        # ``data.get("results")`` may be missing or ``None`` — both
        # collapse to an empty list so the comprehension below is safe.
        items: list[dict] = data.get("results") or []

        # Defensive truncation — ``max_results=k`` is requested but not
        # trusted; if Tavily ever over-returns, the caller still sees a
        # list of length ``<= k``. Per-field ``or ""`` coerces both
        # missing keys AND explicit ``None`` values to ``""`` — the
        # ``SearchResult`` contract is "always a string, never ``None``".
        return [
            SearchResult(
                title=item.get("title") or "",
                url=item.get("url") or "",
                snippet=item.get("content") or "",
            )
            for item in items
        ][:k]

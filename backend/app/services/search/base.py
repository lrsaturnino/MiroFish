"""
Search-provider abstraction.

Defines the contract every search provider implements. ``SearchResult``
is the lingua franca consumed by ``AgentResearchService.run``: every
provider — current ``TavilyProvider`` and future Serper / Brave /
DuckDuckGo implementations — produces lists of ``SearchResult`` with
the same three string fields.

The ``SearchProvider`` ABC enforces the ``search`` method contract.
Subclasses that omit ``search`` raise ``TypeError`` at construction
(default Python ABC behaviour); concrete providers must implement
``search(self, query: str, k: int = 1) -> list[SearchResult]``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict


class SearchResult(TypedDict):
    """
    Canonical search-result shape exchanged across the seam.

    Every provider maps its native response into these three required
    string fields. Missing or null source fields degrade to ``""`` in
    the provider implementation — never ``None``.
    """

    title: str
    url: str
    snippet: str


class SearchProvider(ABC):
    """
    Abstract base class for search providers.

    Concrete providers (``TavilyProvider``, future Serper/Brave/etc.)
    implement ``search`` to return up to ``k`` ``SearchResult`` items
    for the given query. The contract: ``search`` MUST NOT raise on
    network or transport failures — graceful degradation to ``[]`` is
    the agreed-upon behaviour with the calling layer.
    """

    @abstractmethod
    def search(self, query: str, k: int = 1) -> list[SearchResult]:
        """
        Execute a search for ``query`` and return up to ``k`` results.

        Args:
            query: Free-text search query.
            k: Maximum number of results to return. Defaults to ``1``.

        Returns:
            List of ``SearchResult`` items, length ``<= k``. Empty
            list signals either zero hits or a transport failure;
            providers MUST NOT raise.
        """
        ...

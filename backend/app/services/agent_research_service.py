"""
Agent web research orchestration.

This module ships the per-agent search-budget helper and a
dependency-injected class skeleton. The ``budget`` staticmethod is the
only behavioural surface implemented here; ``is_enabled`` and ``run``
are present as stubs that raise ``NotImplementedError`` and will be
filled by sibling tasks (``run`` next, ``is_enabled`` and the JSONL
logger after that).

The four collaborator parameters on the constructor (``search_provider``,
``cache``, ``llm_client``, ``jsonl_logger``) all default to ``None`` and
use forward-quoted string type annotations so the module imports cleanly
even when the search provider, cache, and JSONL logger modules are not
yet present in the package.
"""

import os
from typing import Optional


class AgentResearchService:
    """Per-agent web research orchestrator.

    Wires four optional collaborators via dependency injection and
    exposes the static ``budget`` helper used by callers to decide how
    many search queries each agent should issue. The skip predicate
    short-circuits silent observers (``activity_level < 0.1``) before
    any arithmetic runs on the influence weight.
    """

    def __init__(
        self,
        search_provider: Optional["SearchProvider"] = None,
        cache: Optional["QueryCache"] = None,
        llm_client: Optional["LLMClient"] = None,
        jsonl_logger: Optional["ResearchJsonlLogger"] = None,
    ):
        self.search_provider = search_provider
        self.cache = cache
        self.llm_client = llm_client
        self.jsonl_logger = jsonl_logger

    def is_enabled(self) -> bool:
        raise NotImplementedError("filled by T-010")

    def run(self, project_id, profiles, activity_by_user_id, topic_seed):
        raise NotImplementedError("filled by T-009")

    @staticmethod
    def budget(influence_weight: float, activity_level: float) -> int:
        """Return the per-agent search-query budget.

        Returns ``0`` for silent observers (``activity_level < 0.1``);
        otherwise ``clamp(round(RESEARCH_BASE_K * influence_weight), 1,
        MAX_RESEARCH_QUERIES_PER_AGENT)``. ``RESEARCH_BASE_K`` defaults
        to ``3`` and ``MAX_RESEARCH_QUERIES_PER_AGENT`` defaults to
        ``20``; both are read from ``os.environ`` inside the function
        body so callers (and tests) can override them at call time.

        Uses Python's built-in ``round()`` which is banker's rounding
        (round-half-to-even). For example ``round(4.5) == 4``. The
        floor of ``1`` guarantees every active agent issues at least
        one query, even when the rounded product would be zero or
        negative.
        """
        if activity_level < 0.1:
            return 0
        base_k = int(os.environ.get("RESEARCH_BASE_K", "3"))
        max_q = int(os.environ.get("MAX_RESEARCH_QUERIES_PER_AGENT", "20"))
        return max(1, min(round(base_k * influence_weight), max_q))

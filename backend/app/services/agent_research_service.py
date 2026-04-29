"""
Agent web research orchestration.

This module ships the per-agent search-budget helper and the
dependency-injected ``AgentResearchService`` class. Two behavioural
surfaces are implemented here:

* ``budget`` — a pure-arithmetic staticmethod that returns the per-agent
  search-query budget (zero for silent observers, otherwise the clamped
  ``round(RESEARCH_BASE_K * influence_weight)``).
* ``run`` — the orchestration entry point that walks the profile list,
  runs query-generation + cache-first search + opinion synthesis for
  every active agent, and mutates ``profile.persona`` in place. Per-agent
  failures are captured in ``self._last_run_errors`` instead of bubbling
  out, so one broken agent never aborts the whole run.

``is_enabled`` remains a stub raising ``NotImplementedError`` and will
be filled by the sibling ``RESEARCH_ENABLED`` / ``TAVILY_API_KEY`` env
gating task.

The four collaborator parameters on the constructor (``search_provider``,
``cache``, ``llm_client``, ``jsonl_logger``) all default to ``None`` and
use forward-quoted string type annotations so the module imports cleanly
even when the search provider, cache, and JSONL logger modules are not
yet present in the package. The ``max_persona_append_chars`` knob caps
the appended synthesis length (default 600 chars) and is exposed as a
constructor kwarg so per-instance overrides are possible without touching
environment configuration.
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

    ``run`` is the orchestration entry point — it iterates the profile
    list, computes ``K`` per agent via ``budget``, calls the BUILDER LLM
    twice per active agent (once for query generation, once for opinion
    synthesis), runs the cache-first search loop in between, and appends
    a length-capped opinion paragraph to ``profile.persona`` in place.

    Per-agent error isolation is the load-bearing contract: any
    exception raised inside the per-agent block is captured in
    ``self._last_run_errors`` and the loop continues. The error list is
    reset at the top of every ``run`` call so stale errors never leak
    across consecutive invocations.
    """

    def __init__(
        self,
        search_provider: Optional["SearchProvider"] = None,
        cache: Optional["QueryCache"] = None,
        llm_client: Optional["LLMClient"] = None,
        jsonl_logger: Optional["ResearchJsonlLogger"] = None,
        max_persona_append_chars: int = 600,
    ):
        self.search_provider = search_provider
        self.cache = cache
        self.llm_client = llm_client
        self.jsonl_logger = jsonl_logger
        self.max_persona_append_chars = max_persona_append_chars
        self._last_run_errors: list[dict] = []

    def is_enabled(self) -> bool:
        raise NotImplementedError("filled by T-010")

    def run(self, project_id, profiles, activity_by_user_id, topic_seed):
        """Walk ``profiles`` and append a synthesized opinion to each active agent.

        For every profile, look up its activity record by ``user_id``,
        compute ``K`` via ``self.budget``, skip silent observers
        (``K == 0``), and otherwise run the per-agent block: generate
        ``K`` confirmation-bias queries via the BUILDER LLM, run the
        cache-first search loop, synthesize a 2–3 sentence opinion, and
        append the truncated opinion to ``profile.persona`` with a
        ``"\\n"`` separator.

        Per-agent error isolation: any exception raised inside the
        per-agent block is captured as ``{"agent_id": user_id, "error":
        str(exc)}`` in ``self._last_run_errors`` and the loop continues.
        The error list is reset at the top of every call so stale
        errors never leak across consecutive invocations.

        Returns the same ``profiles`` list reference (identity, not a
        copy) so callers can chain.
        """
        # Per-call reset — previous-run errors must not leak into this run.
        self._last_run_errors = []
        for profile in profiles:
            activity = activity_by_user_id.get(profile.user_id)
            if activity is None:
                # Defensive: pipeline contract should always provide one.
                continue
            k = self.budget(activity.influence_weight, activity.activity_level)
            if k == 0:
                # Silent observer — persona stays untouched.
                continue
            try:
                queries = self._generate_queries(profile, activity, topic_seed, k)
                results = self._run_search_loop(queries)
                opinion = self._synthesize_opinion(
                    profile, activity, topic_seed, results
                )
                truncated = self._truncate_to_cap(opinion)
                profile.persona = profile.persona + "\n" + truncated
            except Exception as exc:
                # Capture and continue — never re-raise. T-010's JSONL
                # logger reads this list after run() returns.
                self._last_run_errors.append(
                    {"agent_id": profile.user_id, "error": str(exc)}
                )
        return profiles

    def _generate_queries(self, profile, activity, topic_seed, k) -> list[str]:
        """Return ``k`` confirmation-bias-shaped queries from the BUILDER LLM.

        Builds a system + user message pair where ``stance`` and the
        first interested topic are interpolated verbatim — no
        case-folding, no transformation. The downstream confirmation-
        bias smoke test asserts literal substring presence in the
        prompt.

        ``chat_json`` returns a JSON object (the OpenAI SDK forces a
        dict in JSON mode). The contract is ``{"queries": [str, ...]}``;
        a missing or empty list raises ``ValueError`` which the caller
        captures as a per-agent error.
        """
        topics_repr = ", ".join(profile.interested_topics)
        system = (
            f"You are a character with stance={activity.stance}. "
            f"Generate {k} search queries that reflect this stance on the topic. "
            f'Return JSON: {{"queries": ["q1", "q2", ...]}}.'
        )
        user = (
            f"bio: {profile.bio}\n"
            f"interested_topics: {topics_repr}\n"
            f"stance: {activity.stance}\n"
            f"topic_seed: {topic_seed}\n"
            f"K: {k}"
        )
        result = self.llm_client.chat_json(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )
        queries = result.get("queries") if isinstance(result, dict) else None
        if not queries or not isinstance(queries, list):
            raise ValueError("query-gen returned no queries")
        return queries

    def _run_search_loop(self, queries: list[str]) -> list[dict]:
        """Aggregate cache-first search results across the query list.

        For every query: ``cache.get`` first; on miss, call
        ``provider.search(q, k=1)`` and write the result back via
        ``cache.set``. The cache's normalisation is internal to
        ``QueryCache``; callers pass the unmodified query string.

        Returns the flattened list of ``SearchResult`` dicts (each
        carrying ``title``, ``url``, ``snippet`` per ``T-006``'s
        TypedDict). Empty per-query results are collected verbatim — an
        all-empty aggregation still flows downstream so synthesis runs
        on an empty snippets section instead of being short-circuited.
        """
        aggregated: list[dict] = []
        for query in queries:
            cached = self.cache.get(query)
            if cached is None:
                results = self.search_provider.search(query, k=1)
                self.cache.set(query, results)
            else:
                results = cached
            aggregated.extend(results)
        return aggregated

    def _synthesize_opinion(self, profile, activity, topic_seed, results) -> str:
        """Collapse the aggregated snippets into a 2–3 sentence opinion.

        One BUILDER LLM ``chat`` call. An empty ``results`` list is
        passed through verbatim — synthesis still runs (the LLM gets an
        empty snippets section and produces an opinion based on persona
        and stance alone). Returning early on empty results would
        introduce a hidden conditional branch and inconsistent
        observability.
        """
        snippets = "\n\n".join(r.get("snippet", "") for r in results)
        system = (
            f"You are a character with stance={activity.stance}. "
            "Given the search snippets below, summarize in 2-3 sentences "
            "what this character now believes about the topic."
        )
        user = (
            f"bio: {profile.bio}\n"
            f"topic_seed: {topic_seed}\n"
            f"snippets:\n{snippets}"
        )
        return self.llm_client.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )

    def _truncate_to_cap(self, opinion: str) -> str:
        """Return ``opinion`` truncated to ``self.max_persona_append_chars``.

        Plain index-slicing — no word-boundary detection, no regex.
        Mid-word truncation is acceptable; downstream LLM consumers
        tolerate hanging tokens.
        """
        return opinion[: self.max_persona_append_chars]

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

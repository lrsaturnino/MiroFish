"""
Agent web research orchestration.

This module ships the per-agent search-budget helper, the
dependency-injected ``AgentResearchService`` class, and the colocated
``ResearchJsonlLogger`` helper. Three behavioural surfaces are
implemented here:

* ``budget`` — a pure-arithmetic staticmethod that returns the per-agent
  search-query budget (zero for silent observers, otherwise the clamped
  ``round(RESEARCH_BASE_K * influence_weight)``).
* ``is_enabled`` — env-driven gate (``RESEARCH_ENABLED=true`` AND
  ``TAVILY_API_KEY`` set, both case-insensitively / non-empty after
  strip). Read once at the top of every ``run`` call so a mid-run env
  flip cannot half-write the JSONL file.
* ``run`` — the orchestration entry point that walks the profile list,
  runs query-generation + cache-first search + opinion synthesis for
  every active agent, and mutates ``profile.persona`` in place. Per-agent
  failures are captured in ``self._last_run_errors`` instead of bubbling
  out, so one broken agent never aborts the whole run. Each active
  agent's run also produces one JSONL record (success or error path)
  written via ``ResearchJsonlLogger``.

``ResearchJsonlLogger`` is a small append-only JSONL writer colocated in
this module. It writes one JSON object per line, UTF-8, to
``<project_dir>/agent_research.jsonl``. Schema (eight success-path keys
plus optional ``error`` discriminator on the failure path):
``agent_id``, ``queries``, ``search_results_summary``,
``synthesized_opinion``, ``latency_ms``, ``tokens``, ``cache_hits``,
``ts``. ``tokens`` is currently a placeholder dict
(``{"prompt": 0, "completion": 0}``) because ``LLMClient.chat`` /
``chat_json`` do not surface ``response.usage``; surfacing real token
counts is deferred and out of scope for this module.

The four collaborator parameters on the constructor (``search_provider``,
``cache``, ``llm_client``, ``jsonl_logger``) all default to ``None`` and
use forward-quoted string type annotations so the module imports cleanly
even when the search provider, cache, and JSONL logger modules are not
yet present in the package. The ``max_persona_append_chars`` knob caps
the appended synthesis length (default 600 chars) and is exposed as a
constructor kwarg so per-instance overrides are possible without touching
environment configuration.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


class ResearchJsonlLogger:
    """Append-only JSONL writer for per-agent research records.

    One JSON object per line, UTF-8, to
    ``<project_dir>/agent_research.jsonl``. The directory is created on
    demand (``mkdir(parents=True, exist_ok=True)``) so callers do not
    have to pre-create it. Each ``write`` call opens the file in append
    mode, writes one line, and closes the handle — there is no shared
    file object, which keeps the writer crash-safe between calls.

    The writer auto-fills ``ts`` with the current ISO-8601 UTC timestamp
    when the caller did not supply it; an explicit caller-supplied
    ``ts`` always wins. ``json.dumps`` is called with
    ``ensure_ascii=False`` so non-ASCII content (e.g. Chinese text in
    ``synthesized_opinion``) round-trips losslessly.
    """

    def __init__(self, project_dir):
        """Store the project directory path; resolve the JSONL log path.

        ``project_dir`` accepts ``str`` or ``pathlib.Path`` — both are
        normalised to ``Path``. The directory is NOT created here;
        creation is deferred to the first ``write`` call so constructing
        a logger never has filesystem side-effects.
        """
        self.project_dir = Path(project_dir)
        self.log_path = self.project_dir / "agent_research.jsonl"

    def write(self, record: dict) -> None:
        """Append one JSON record to the JSONL file.

        Auto-fills ``ts`` with ``datetime.now(timezone.utc).isoformat()``
        when absent (caller-supplied ``ts`` wins via ``setdefault``).
        Ensures the project directory exists before opening the file.
        """
        record.setdefault("ts", datetime.now(timezone.utc).isoformat())
        self.project_dir.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


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

    The env-driven ``is_enabled`` gate is read exactly once at the top
    of each ``run`` call. When disabled, ``run`` emits one ``WARNING``
    log line via the module logger and returns the same ``profiles``
    list reference unchanged — no per-agent work, no JSONL file.
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
        """Return ``True`` iff research is enabled by the environment.

        The gate requires BOTH halves: ``RESEARCH_ENABLED`` must be the
        literal string ``"true"`` after ``.strip().lower()`` (so
        ``"TRUE"``, ``"True"``, ``" true "`` all enable; ``"1"``,
        ``"yes"``, ``"on"`` do NOT — strict policy for least surprise),
        AND ``TAVILY_API_KEY`` must be non-empty after strip (whitespace
        sentinels are rejected).

        Env values are read at call time so ``monkeypatch.setenv`` works
        deterministically and operators can flip the gate without
        restarting the process.
        """
        enabled = os.environ.get("RESEARCH_ENABLED", "").strip().lower() == "true"
        has_key = bool(os.environ.get("TAVILY_API_KEY", "").strip())
        return enabled and has_key

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

        Each active agent (success or error path) also produces one
        JSONL record written via ``self.jsonl_logger`` — auto-built
        lazily from ``Config.UPLOAD_FOLDER`` and ``project_id`` when the
        caller did not inject one.

        Side-effect — companion ``agent_research.meta.json`` file is
        written once at entry (after the gate, before the per-agent
        loop) at ``<project_dir>/agent_research.meta.json``. Schema:
        ``{"total_active_agents": N, "started_at": iso_ts}`` where ``N``
        is the count of profiles whose activity record exists and whose
        ``activity_level >= 0.1`` (the silent-observer skip threshold).
        The file gives the progress endpoint a denominator without
        re-reading project state. The file is NOT written on the
        disabled path so an off-mode run leaves no artifacts.

        Env-driven gate: ``self.is_enabled()`` is read exactly once at
        entry. When disabled, the function emits one ``WARNING`` log
        line and returns ``profiles`` unchanged — no LLM, no provider,
        no JSONL, no meta file.

        Returns the same ``profiles`` list reference (identity, not a
        copy) so callers can chain.
        """
        # Per-call reset — previous-run errors must not leak into this run.
        # MUST happen before the gate check so the disabled path leaves
        # the error list empty (test #19 contract).
        self._last_run_errors = []

        # Gate read once at entry. A mid-run env flip cannot half-write
        # because the loop body never re-reads the gate.
        if not self.is_enabled():
            logger.warning(
                "research disabled: RESEARCH_ENABLED=%r TAVILY_API_KEY set: %s",
                os.environ.get("RESEARCH_ENABLED", ""),
                bool(os.environ.get("TAVILY_API_KEY", "").strip()),
            )
            return profiles

        # Resolve the JSONL logger AFTER the gate — keeps the disabled
        # path zero-cost and avoids creating directories in production
        # when research is off.
        jsonl_logger = self.jsonl_logger
        if jsonl_logger is None:
            # Local import: keep module top free of the Flask import
            # chain so ``from app.services.agent_research_service import
            # AgentResearchService, ResearchJsonlLogger`` stays smoke-
            # importable without pulling Config (and transitively Flask).
            from app.config import Config

            project_dir = os.path.join(
                Config.UPLOAD_FOLDER, "projects", str(project_id)
            )
            jsonl_logger = ResearchJsonlLogger(project_dir)

        # Companion meta file — gives the progress endpoint a denominator
        # decoupled from in-memory state. Written once at entry, after
        # the gate, before any per-agent work. Count of "active" agents
        # mirrors the per-agent skip in the loop body: an activity
        # record must exist AND ``activity_level >= 0.1`` (silent-
        # observer threshold). Best-effort: a write failure here logs a
        # warning but never aborts the run.
        try:
            total_active_agents = sum(
                1
                for p in profiles
                if (
                    (a := activity_by_user_id.get(p.user_id)) is not None
                    and a.activity_level >= 0.1
                )
            )
            jsonl_logger.project_dir.mkdir(parents=True, exist_ok=True)
            meta_path = jsonl_logger.project_dir / "agent_research.meta.json"
            meta_payload = {
                "total_active_agents": total_active_agents,
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_payload, f, ensure_ascii=False)
        except OSError as exc:
            logger.warning("research meta file write failed: %s", exc)

        for profile in profiles:
            activity = activity_by_user_id.get(profile.user_id)
            if activity is None:
                # Defensive: pipeline contract should always provide one.
                continue
            k = self.budget(activity.influence_weight, activity.activity_level)
            if k == 0:
                # Silent observer — persona stays untouched.
                continue

            # Per-agent locals are pre-initialised so the ``finally`` block
            # always sees well-formed values when building the JSONL record
            # — even when an exception fires before the corresponding
            # success-path assignment runs. ``error_message`` doubles as the
            # success/error discriminator at record-write time.
            start = time.perf_counter()
            cache_hits = 0
            queries: list[str] = []
            results: list[dict] = []
            truncated = ""
            error_message: Optional[str] = None
            try:
                queries = self._generate_queries(profile, activity, topic_seed, k)
                results, cache_hits = self._run_search_loop(queries)
                opinion = self._synthesize_opinion(
                    profile, activity, topic_seed, results
                )
                truncated = self._truncate_to_cap(opinion)
                profile.persona = profile.persona + "\n" + truncated
            except Exception as exc:
                # Capture and continue — never re-raise. The JSONL record
                # below carries the same eight success-path keys plus an
                # additive ``error`` discriminator (D4 schema).
                error_message = str(exc)
                self._last_run_errors.append(
                    {"agent_id": profile.user_id, "error": error_message}
                )
            finally:
                latency_ms = int((time.perf_counter() - start) * 1000)
                record: dict = {
                    "agent_id": profile.user_id,
                    "queries": queries,
                    "search_results_summary": self._project_results(results),
                    "synthesized_opinion": truncated,
                    "latency_ms": latency_ms,
                    # ``tokens`` is a placeholder: LLMClient.chat /
                    # chat_json do not surface response.usage today.
                    # Surfacing real token counts is deferred.
                    "tokens": {"prompt": 0, "completion": 0},
                    "cache_hits": cache_hits,
                }
                if error_message is not None:
                    record["error"] = error_message
                jsonl_logger.write(record)
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

    def _run_search_loop(self, queries: list[str]) -> tuple[list[dict], int]:
        """Aggregate cache-first search results across the query list.

        For every query: ``cache.get`` first; on miss, call
        ``provider.search(q, k=1)`` and write the result back via
        ``cache.set``. The cache's normalisation is internal to
        ``QueryCache``; callers pass the unmodified query string.

        Returns ``(aggregated_results, cache_hits)`` where
        ``cache_hits`` is the count of queries served from the cache.
        Empty per-query results are collected verbatim — an all-empty
        aggregation still flows downstream so synthesis runs on an empty
        snippets section instead of being short-circuited.
        """
        aggregated: list[dict] = []
        cache_hits = 0
        for query in queries:
            cached = self.cache.get(query)
            if cached is None:
                results = self.search_provider.search(query, k=1)
                self.cache.set(query, results)
            else:
                results = cached
                cache_hits += 1
            aggregated.extend(results)
        return aggregated, cache_hits

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
    def _project_results(results: list[dict]) -> list[dict]:
        """Project full ``SearchResult`` dicts down to the JSONL summary shape.

        The JSONL ``search_results_summary`` field carries only ``title``
        and ``url`` per the D4 schema — the full ``snippet`` is fed to
        ``_synthesize_opinion`` but deliberately NOT persisted to the
        log to keep records compact and PII-free. Missing keys default
        to empty strings so downstream consumers never see ``None``.
        """
        return [
            {"title": r.get("title", ""), "url": r.get("url", "")}
            for r in results
        ]

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

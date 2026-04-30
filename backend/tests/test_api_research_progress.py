"""
Tests for ``GET /api/research/progress/<project_id>`` — the on-disk
research-progress endpoint that powers the per-agent progress bar in
``Step2EnvSetup.vue``.

The endpoint is a pure file reader: it inspects two artifacts inside
``Config.UPLOAD_FOLDER/projects/<project_id>/`` —
``agent_research.jsonl`` (T-010 schema, eight keys per record) and the
companion ``agent_research.meta.json`` (``{"total_active_agents": int,
"started_at": iso_ts}``) — and returns the four documented counters
(``processed_agents``, ``total_active_agents``, ``last_agent_id``,
``last_ts``) plus a diagnostic ``malformed_count``.

Coverage map (10 scenarios):

    1. Both artifacts absent, project dir exists → zeros.
    2. Meta present, jsonl empty → total only.
    3. Partial jsonl → counts and last-record fields.
    4. Complete jsonl → full count.
    5. Malformed line → skipped, no 500, ``malformed_count`` reported.
    6. Unknown project_id → 404.
    7. Path-traversal attempt (``..``) → 400, never escapes uploads root.
    8. Performance: 100 lines responds in <100 ms (warmed-up call).
    9. UTF-8 (Chinese) content parses cleanly.
   10. ``project_id`` exists as a file, not a directory → 404.

Performance target: full file completes in <10 seconds. All filesystem
I/O is scoped to ``tmp_path``; the Flask test client runs in-process.

The autouse ``_clean_llm_env`` fixture in ``conftest.py`` clears LLM env
vars and seeds research-gate defaults — both harmless here because the
endpoint never calls into ``AgentResearchService``.
"""

import json
import os
import time

import pytest

from app.config import Config


# --------------------------------------------------------------------------
# Test-local fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def tmp_uploads_dir(tmp_path, monkeypatch):
    """Redirect ``Config.UPLOAD_FOLDER`` to a per-test ``tmp_path`` so the
    endpoint reads from the sandboxed projects subtree.

    The endpoint resolves
    ``Config.UPLOAD_FOLDER/projects/<project_id>/`` at request time, so
    swapping the class attribute via ``monkeypatch.setattr`` is the only
    redirection needed. Yields the ``projects`` parent so tests can
    seed per-project subdirectories directly.
    """
    monkeypatch.setattr(Config, "UPLOAD_FOLDER", str(tmp_path))
    projects_root = tmp_path / "projects"
    projects_root.mkdir(parents=True, exist_ok=True)
    return projects_root


@pytest.fixture
def flask_client(tmp_uploads_dir):
    """Build a Flask test client bound to the sandboxed uploads root.

    Function-scoped so the ``Config.UPLOAD_FOLDER`` monkeypatch is in
    effect when ``create_app()`` runs. No project / simulation stubs
    needed — the endpoint reads only on-disk state.
    """
    from app import create_app

    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _seed_project_dir(projects_root, project_id: str):
    """Create ``projects_root/<project_id>/`` and return the path."""
    project_dir = projects_root / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    return project_dir


def _write_meta(project_dir, total_active_agents: int, started_at: str = "2026-04-30T00:00:00+00:00"):
    """Seed ``agent_research.meta.json`` inside ``project_dir``."""
    meta_path = project_dir / "agent_research.meta.json"
    meta_path.write_text(
        json.dumps({"total_active_agents": total_active_agents, "started_at": started_at}),
        encoding="utf-8",
    )
    return meta_path


def _write_jsonl_records(project_dir, records: list[dict]):
    """Seed ``agent_research.jsonl`` with one JSON object per record."""
    jsonl_path = project_dir / "agent_research.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return jsonl_path


def _make_record(agent_id, ts: str, opinion: str = "ok") -> dict:
    """Build one JSONL record matching ``ResearchJsonlLogger`` write shape."""
    return {
        "agent_id": agent_id,
        "queries": ["q1"],
        "search_results_summary": [],
        "synthesized_opinion": opinion,
        "latency_ms": 100,
        "tokens": {"prompt": 0, "completion": 0},
        "cache_hits": 0,
        "ts": ts,
    }


# --------------------------------------------------------------------------
# Scenario 1 — both artifacts absent, project dir exists → zeros
# --------------------------------------------------------------------------
def test_no_jsonl_no_meta_returns_zeros(flask_client, tmp_uploads_dir):
    """Project dir exists but neither meta nor jsonl present — endpoint
    returns the zero-state envelope (research not yet started)."""
    _seed_project_dir(tmp_uploads_dir, "p_known")

    resp = flask_client.get("/api/research/progress/p_known")
    assert resp.status_code == 200, resp.data
    data = resp.get_json()
    assert data["processed_agents"] == 0
    assert data["total_active_agents"] == 0
    assert data["last_agent_id"] is None
    assert data["last_ts"] is None
    assert data["malformed_count"] == 0


# --------------------------------------------------------------------------
# Scenario 2 — meta present, jsonl empty → total only
# --------------------------------------------------------------------------
def test_meta_present_jsonl_empty_returns_total_only(flask_client, tmp_uploads_dir):
    """Meta file fixes the denominator at 10; empty jsonl keeps the
    numerator at 0; last-record fields stay null."""
    project_dir = _seed_project_dir(tmp_uploads_dir, "p_meta_only")
    _write_meta(project_dir, total_active_agents=10)
    (project_dir / "agent_research.jsonl").write_text("", encoding="utf-8")

    resp = flask_client.get("/api/research/progress/p_meta_only")
    assert resp.status_code == 200, resp.data
    data = resp.get_json()
    assert data["processed_agents"] == 0
    assert data["total_active_agents"] == 10
    assert data["last_agent_id"] is None
    assert data["last_ts"] is None
    assert data["malformed_count"] == 0


# --------------------------------------------------------------------------
# Scenario 3 — partial jsonl → counts and last-record fields
# --------------------------------------------------------------------------
def test_partial_jsonl_returns_count_and_last_record_fields(flask_client, tmp_uploads_dir):
    """Meta says 10, jsonl has 4 valid records. Counts come from line
    enumeration; ``last_agent_id`` / ``last_ts`` come from the LAST line."""
    project_dir = _seed_project_dir(tmp_uploads_dir, "p_partial")
    _write_meta(project_dir, total_active_agents=10)
    _write_jsonl_records(
        project_dir,
        [
            _make_record(0, "2026-04-30T00:00:00+00:00"),
            _make_record(1, "2026-04-30T00:00:01+00:00"),
            _make_record(2, "2026-04-30T00:00:02+00:00"),
            _make_record(3, "2026-04-30T00:00:03+00:00"),
        ],
    )

    resp = flask_client.get("/api/research/progress/p_partial")
    assert resp.status_code == 200, resp.data
    data = resp.get_json()
    assert data["processed_agents"] == 4
    assert data["total_active_agents"] == 10
    assert data["last_agent_id"] == 3
    assert data["last_ts"] == "2026-04-30T00:00:03+00:00"
    assert data["malformed_count"] == 0


# --------------------------------------------------------------------------
# Scenario 4 — complete jsonl → full count
# --------------------------------------------------------------------------
def test_complete_jsonl_returns_full_count(flask_client, tmp_uploads_dir):
    """All 10 lines written; numerator equals denominator; final agent
    id and timestamp surface as last-record metadata."""
    project_dir = _seed_project_dir(tmp_uploads_dir, "p_complete")
    _write_meta(project_dir, total_active_agents=10)
    _write_jsonl_records(
        project_dir,
        [_make_record(i, f"2026-04-30T00:00:{i:02d}+00:00") for i in range(10)],
    )

    resp = flask_client.get("/api/research/progress/p_complete")
    assert resp.status_code == 200, resp.data
    data = resp.get_json()
    assert data["processed_agents"] == 10
    assert data["total_active_agents"] == 10
    assert data["last_agent_id"] == 9
    assert data["last_ts"] == "2026-04-30T00:00:09+00:00"
    assert data["malformed_count"] == 0


# --------------------------------------------------------------------------
# Scenario 5 — malformed line skipped, no 500, malformed_count reported
# --------------------------------------------------------------------------
def test_malformed_jsonl_line_skipped_no_500(flask_client, tmp_uploads_dir):
    """3 valid + 1 garbage line. Endpoint must not 500; ``malformed_count``
    surfaces 1; counts only enumerate valid lines; ``last_agent_id`` is
    the last VALID record's id (garbage line never updates it)."""
    project_dir = _seed_project_dir(tmp_uploads_dir, "p_malformed")
    _write_meta(project_dir, total_active_agents=10)

    jsonl_path = project_dir / "agent_research.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(_make_record(0, "2026-04-30T00:00:00+00:00")) + "\n")
        f.write(json.dumps(_make_record(1, "2026-04-30T00:00:01+00:00")) + "\n")
        f.write(json.dumps(_make_record(2, "2026-04-30T00:00:02+00:00")) + "\n")
        f.write("{not valid json\n")

    resp = flask_client.get("/api/research/progress/p_malformed")
    assert resp.status_code == 200, resp.data
    data = resp.get_json()
    assert data["processed_agents"] == 3
    assert data["total_active_agents"] == 10
    assert data["last_agent_id"] == 2
    assert data["last_ts"] == "2026-04-30T00:00:02+00:00"
    assert data["malformed_count"] == 1


# --------------------------------------------------------------------------
# Scenario 6 — unknown project_id → 404
# --------------------------------------------------------------------------
def test_unknown_project_id_returns_404(flask_client, tmp_uploads_dir):
    """Discriminator is project-directory existence, NOT artifact
    existence. With no ``unknown_proj/`` dir created, endpoint returns
    404 in the spec error envelope (NOT Flask's default HTML 404)."""
    # Intentionally do not create tmp_uploads_dir / "unknown_proj"
    resp = flask_client.get("/api/research/progress/unknown_proj")
    assert resp.status_code == 404, resp.data

    # Body MUST be JSON in the spec error shape — proves the request
    # reached the endpoint handler and did not fall through to Flask's
    # default routing 404 (which would return HTML).
    data = resp.get_json()
    assert data is not None, "404 response must be JSON, not HTML"
    assert data.get("success") is False
    assert "error" in data


# --------------------------------------------------------------------------
# Scenario 7 — path-traversal attempt → 400
# --------------------------------------------------------------------------
def test_path_traversal_attempt_rejected(flask_client, tmp_uploads_dir):
    """``project_id=".."`` resolves to ``Config.UPLOAD_FOLDER/projects/..`` →
    parent of the uploads root. The ``commonpath`` guard must reject it
    with 400 BEFORE any file is opened.

    Note: a literal ``"../etc/passwd"`` URL is normalized by the
    WSGI/Werkzeug routing layer before reaching the handler — using
    just ``..`` (a single segment) is what actually exercises the
    handler's path-traversal guard.
    """
    resp = flask_client.get("/api/research/progress/..")
    assert resp.status_code == 400, resp.data
    data = resp.get_json()
    assert data is not None, "400 response must be JSON"
    assert data.get("success") is False
    assert "error" in data


# --------------------------------------------------------------------------
# Scenario 8 — performance: 100 lines responds in <100 ms (warmed-up)
# --------------------------------------------------------------------------
def test_response_time_under_100ms_for_100_line_jsonl(flask_client, tmp_uploads_dir):
    """A 100-line jsonl must respond in under 100 ms.

    Cold imports + Flask routing setup are excluded from the budget by
    making ONE warm-up call before timing the second call. Mitigates
    the cold-start flakiness flagged in the pre-implementation risk
    assessment.
    """
    project_dir = _seed_project_dir(tmp_uploads_dir, "p_perf")
    _write_meta(project_dir, total_active_agents=100)
    _write_jsonl_records(
        project_dir,
        [_make_record(i, f"2026-04-30T00:00:00+00:00") for i in range(100)],
    )

    # Warm-up: prime imports, route table, file-handle path.
    warmup = flask_client.get("/api/research/progress/p_perf")
    assert warmup.status_code == 200, warmup.data

    # Timed call.
    start = time.perf_counter()
    resp = flask_client.get("/api/research/progress/p_perf")
    elapsed = time.perf_counter() - start

    assert resp.status_code == 200, resp.data
    data = resp.get_json()
    assert data["processed_agents"] == 100
    assert elapsed < 0.1, f"endpoint took {elapsed*1000:.1f}ms (budget 100ms)"


# --------------------------------------------------------------------------
# Scenario 9 — UTF-8 (Chinese) content parses cleanly
# --------------------------------------------------------------------------
def test_utf8_chinese_in_synthesized_opinion_parses_cleanly(flask_client, tmp_uploads_dir):
    """A jsonl record carrying Chinese text in ``synthesized_opinion``
    must round-trip through ``json.loads`` without ``UnicodeDecodeError``
    or 500. Mirrors the writer side at
    ``agent_research_service.py:94`` which uses ``encoding="utf-8"`` and
    ``ensure_ascii=False``."""
    project_dir = _seed_project_dir(tmp_uploads_dir, "p_utf8")
    _write_meta(project_dir, total_active_agents=2)
    _write_jsonl_records(
        project_dir,
        [
            _make_record(0, "2026-04-30T00:00:00+00:00", opinion="我相信这是真的"),
            _make_record(1, "2026-04-30T00:00:01+00:00", opinion="另一个观点"),
        ],
    )

    resp = flask_client.get("/api/research/progress/p_utf8")
    assert resp.status_code == 200, resp.data
    data = resp.get_json()
    assert data["processed_agents"] == 2
    assert data["total_active_agents"] == 2
    assert data["last_agent_id"] == 1
    assert data["malformed_count"] == 0


# --------------------------------------------------------------------------
# Scenario 10 — project path is a file, not a directory → 404
# --------------------------------------------------------------------------
def test_project_path_is_a_file_returns_404(flask_client, tmp_uploads_dir):
    """Defensive boundary: ``projects/<id>`` exists but is a regular
    file, not a directory. The handler must use ``os.path.isdir`` (NOT
    ``os.path.exists``) for the existence check, returning 404 in the
    spec error envelope."""
    weird_path = tmp_uploads_dir / "p_weird"
    weird_path.write_text("not a directory", encoding="utf-8")

    resp = flask_client.get("/api/research/progress/p_weird")
    assert resp.status_code == 404, resp.data
    data = resp.get_json()
    assert data is not None, "404 response must be JSON"
    assert data.get("success") is False
    assert "error" in data

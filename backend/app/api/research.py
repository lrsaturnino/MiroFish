"""
Research progress API route.

Single GET handler that powers the per-agent web-research progress bar
in ``Step2EnvSetup.vue``. The endpoint is a pure file reader: it inspects
two on-disk artifacts inside ``Config.UPLOAD_FOLDER/projects/<project_id>/``
and returns four counters (plus a diagnostic ``malformed_count``).

Artifacts inspected:

* ``agent_research.jsonl`` — append-only JSONL produced by
  ``ResearchJsonlLogger`` (one record per processed agent, eight keys per
  record). The numerator (``processed_agents``) is the count of valid
  JSON lines.
* ``agent_research.meta.json`` — companion file written at the top of
  ``AgentResearchService.run`` after the ``is_enabled()`` gate passes.
  Schema: ``{"total_active_agents": int, "started_at": iso_ts}``. The
  denominator (``total_active_agents``) is read from this file. The
  ``started_at`` field is informational only and is not consumed by the
  endpoint.

Transport choice — short-poll. The frontend polls every 1.5s while
research is in progress. SSE (planning §5.9) is the alternative; v1
picks short-poll for simplicity (single Flask handler, no streaming
infrastructure, mirrors the existing polling cadence already used by
the configuration polling helper in ``Step2EnvSetup.vue``).

Status code policy:

* ``200`` — project directory exists; counts are returned (zeros when
  neither artifact is present, i.e. research has not started yet).
* ``404`` — project directory does not exist OR the path resolves to a
  regular file (not a directory). Discriminator is project-directory
  existence, NOT artifact presence. Existence check uses
  ``os.path.isdir`` (NOT ``os.path.exists``) so a regular file at the
  expected directory path is treated as non-existent.
* ``400`` — path-traversal guard rejected the ``project_id`` (escapes
  the uploads root after ``realpath`` normalization).

All non-200 responses follow the spec error envelope ``{"success":
False, "error": "..."}`` so the frontend can render a clear message
without parsing HTML.

Malformed-line handling — the JSONL file is opened with
``encoding="utf-8", errors="replace"``. Each non-empty stripped line is
fed to ``json.loads`` inside a ``try``; on ``JSONDecodeError`` the
``malformed_count`` is incremented and the line is skipped — the
endpoint never 500s on bad input. ``last_agent_id`` and ``last_ts``
update only on the success branch, so a garbage line never clobbers
the last-valid-record metadata.

Path-traversal guard — the resolved real path of
``UPLOAD_FOLDER/projects/<project_id>`` must share a common prefix with
the resolved real path of ``UPLOAD_FOLDER/projects``. ``commonpath``
can raise ``ValueError`` when the two paths live on different drives
(Windows) or are otherwise incomparable; the guard wraps the check in
``try / except`` and returns 400 in that defensive branch too. The
guard runs BEFORE any file open.
"""

import json
import os
import time
import traceback

from flask import jsonify

from . import research_bp
from ..config import Config
from ..utils.logger import get_logger

logger = get_logger('mirofish.api.research')


def _resolve_safe_project_dir(project_id: str):
    """Return the resolved absolute project directory, or ``None`` on escape.

    Builds ``Config.UPLOAD_FOLDER/projects/<project_id>``, resolves it
    via ``os.path.realpath``, and asserts the resolved path stays under
    the resolved ``projects`` root via ``os.path.commonpath``. Returns
    the resolved path on success, ``None`` on any escape attempt or
    incomparable-paths edge case (different drives on Windows, empty
    path, etc.). Also rejects the projects root itself — only proper
    descendants are valid project directories.
    """
    projects_root_raw = os.path.join(Config.UPLOAD_FOLDER, "projects")
    uploads_root = os.path.realpath(projects_root_raw)
    candidate = os.path.realpath(os.path.join(projects_root_raw, project_id))
    try:
        if os.path.commonpath([candidate, uploads_root]) != uploads_root:
            return None
    except ValueError:
        # Different drives, empty path, or other incomparable case.
        return None
    # Reject the projects root itself — only descendants are valid.
    if candidate == uploads_root:
        return None
    return candidate


@research_bp.route('/progress/<project_id>', methods=['GET'])
def get_research_progress(project_id: str):
    """
    Read on-disk research artifacts and return progress counters.

    Path: ``GET /api/research/progress/<project_id>``

    Response (200):
        {
            "processed_agents": int,       # valid JSONL lines counted
            "total_active_agents": int,    # from meta file (0 if absent)
            "last_agent_id": any | null,   # from last VALID JSONL line
            "last_ts": str | null,         # from last VALID JSONL line
            "malformed_count": int         # diagnostic (always present)
        }

    Response (404): project directory does not exist or is not a
    directory. Body: ``{"success": False, "error": "..."}``.

    Response (400): ``project_id`` failed the path-traversal guard.
    Body: ``{"success": False, "error": "..."}``.

    See module docstring for the full status-code policy, malformed-line
    handling, and path-traversal guard rationale.
    """
    try:
        project_dir = _resolve_safe_project_dir(project_id)
        if project_dir is None:
            return jsonify({
                "success": False,
                "error": f"Invalid project_id: {project_id!r}"
            }), 400

        if not os.path.isdir(project_dir):
            return jsonify({
                "success": False,
                "error": f"Project not found: {project_id!r}"
            }), 404

        meta_path = os.path.join(project_dir, "agent_research.meta.json")
        jsonl_path = os.path.join(project_dir, "agent_research.jsonl")

        # Initialize ALL counters before either optional read so the
        # zero-state response (both artifacts absent) carries the full
        # documented shape.
        processed = 0
        total = 0
        last_id = None
        last_ts = None
        malformed = 0

        # Read the meta file when present. Defensive: bad JSON or
        # missing key falls back to total=0 so the endpoint never 500s
        # on a corrupted meta file.
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if isinstance(meta, dict):
                    total_value = meta.get("total_active_agents", 0)
                    if isinstance(total_value, int):
                        total = total_value
            except (OSError, json.JSONDecodeError) as exc:
                # Log and continue — surface as total=0 to the client.
                logger.warning("research meta parse failed for %s: %s", project_id, exc)

        # Tail the JSONL file when present. Each valid line increments
        # processed and updates last_agent_id / last_ts. Malformed lines
        # increment malformed_count and are skipped — last_* never
        # updates from a garbage line.
        if os.path.exists(jsonl_path):
            try:
                with open(jsonl_path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        stripped = line.strip()
                        if not stripped:
                            continue
                        try:
                            record = json.loads(stripped)
                        except json.JSONDecodeError:
                            malformed += 1
                            continue
                        processed += 1
                        if isinstance(record, dict):
                            last_id = record.get("agent_id", last_id)
                            last_ts = record.get("ts", last_ts)
            except OSError as exc:
                # File disappeared between exists() and open(), permission
                # error, etc. Log but return whatever we have so far.
                logger.warning("research jsonl read failed for %s: %s", project_id, exc)

        return jsonify({
            "processed_agents": processed,
            "total_active_agents": total,
            "last_agent_id": last_id,
            "last_ts": last_ts,
            "malformed_count": malformed,
        })

    except Exception as exc:
        logger.error(f"research progress query failed: {exc}")
        return jsonify({
            "success": False,
            "error": str(exc),
            "traceback": traceback.format_exc()
        }), 500

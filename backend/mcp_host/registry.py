"""
OrbixAI MCP server registry.

A single source of truth for *which* MCP servers exist, what they do, and whether
the user has them enabled. The host (client.py) asks this module for the roster of
**enabled** servers to spawn; the frontend Integrations page reads/writes it through
the /mcp/servers endpoints.

Two layers:
  1. CATALOG  — static definitions (id, display metadata, how to spawn, what config
                keys it accepts). Adding a new server = one entry here.
  2. STATE    — user's per-server {enabled, config}, persisted to backend/mcp_state.json
                so a toggle survives restarts.

Robustness contract (important): turning a server off, or leaving one misconfigured,
must never break the backend. This module only records intent + builds spawn specs;
the client tolerates a server that fails to start (see client.py). Read tools stay
available even if an action server is down.
"""

from __future__ import annotations

import os
import sys
import json
import logging
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

_BACKEND = Path(__file__).resolve().parents[1]
_SERVERS_DIR = _BACKEND / "mcp_servers"
_STATE_FILE = _BACKEND / "mcp_state.json"
_PY = sys.executable  # spawn every server with the exact interpreter running us

_lock = Lock()


def _server(path: str) -> str:
    return str(_SERVERS_DIR / path)


# --------------------------------------------------------------------------- #
# CATALOG — the servers OrbixAI actually knows how to run.
#
# Each entry:
#   id            stable key (also the MCP server name the host registers)
#   name/category/description/data/auth/steps  → shown on the Integrations page
#   command/args  → how to spawn it over stdio
#   requires      → "local" | "oauth" | "neo4j" | "apikey"  (informational)
#   default       → enabled out of the box?
#   locked        → cannot be turned off from the UI (core dependency)
#   config        → [{key,label,placeholder,secret}] editable fields; each maps to an
#                   env var (ORBIX_<ID>_<KEY> upper-cased) injected into the subprocess
# --------------------------------------------------------------------------- #
CATALOG: list[dict] = [
    {
        "id": "orbix-memory",
        "name": "Neo4j Graph Memory",
        "category": "Memory",
        "icon": "neo4j",
        "description": "Long-term memory in a Neo4j knowledge graph — recalls people, "
                       "projects and facts, and updates nodes in place instead of duplicating.",
        "data": "Read/write entities, relations & observations in Neo4j",
        "auth": "Neo4j URI + credentials (backend/.env)",
        "requires": "neo4j",
        "default": True,
        "locked": False,
        "command": _PY,
        "args": [_server("memory_server.py")],
        "config": [],
        "steps": [
            "Start Neo4j (local Docker or Aura free tier)",
            "Set NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD in backend/.env",
            "Enable — recall/search/remember tools come online",
        ],
    },
    {
        "id": "orbix-gsuite",
        "name": "Gmail · Meet · Calendar",
        "category": "Your data",
        "icon": "mail",
        "description": "Send email, spin up Google Meet links and manage your calendar — "
                       "the assistant's hands for getting things done.",
        "data": "Send mail, read inbox, create Meet links, read/write calendar events",
        "auth": "OAuth (Google)",
        "requires": "oauth",
        "default": True,
        "locked": False,
        "command": _PY,
        "args": [_server("gsuite_server.py")],
        "config": [],
        "steps": [
            "Click Connect with Google",
            "Approve Gmail, Calendar and Meet access",
            "Token is saved locally in backend/token.json",
        ],
    },
    {
        "id": "orbix-web",
        "name": "Web Fetch & Search",
        "category": "Web & actions",
        "icon": "fetch",
        "description": "Read any web page as clean text and run a live web search — "
                       "so the assistant can research and cite current information.",
        "data": "Public web pages and search results",
        "auth": "Local — no login (keyless DuckDuckGo search)",
        "requires": "local",
        "default": True,
        "locked": False,
        "command": _PY,
        "args": [_server("web_server.py")],
        "config": [],
        "steps": [
            "Enable the server",
            "Ask the assistant to look something up or summarise a URL",
        ],
    },
    {
        "id": "orbix-files",
        "name": "Filesystem",
        "category": "Local & system",
        "icon": "files",
        "description": "Read, write and search all your files automatically — everything "
                       "except Windows/system folders and secrets. No setup needed.",
        "data": "All your drives (C:\\, D:\\, E:\\ …) except Windows/system & secret files",
        "auth": "Local — no login",
        "requires": "local",
        "default": True,
        "locked": False,
        "command": _PY,
        "args": [_server("filesystem_server.py")],
        "config": [
            {"key": "roots", "label": "Restrict to these folders (optional — blank = all files)",
             "placeholder": r"leave blank for all files, or e.g. D:\, C:\Users\you\Documents",
             "secret": False, "required": False},
        ],
        "steps": [
            "It's on by default — the assistant can already read/write your files",
            "Windows, Program Files, ProgramData and secrets (.ssh, tokens, .env) are always blocked",
            "Optionally restrict it to specific folders above",
        ],
    },
    {
        "id": "orbix-tasks",
        "name": "Tasks & To-Do",
        "category": "Productivity",
        "icon": "check",
        "description": "A shared to-do list the assistant can add to and complete — "
                       "so 'remind me to…' and follow-ups land on your task list automatically.",
        "data": "Local to-do items (backend/tasks_store.json)",
        "auth": "Local — no login",
        "requires": "local",
        "default": True,
        "locked": False,
        "command": _PY,
        "args": [_server("tasks_server.py")],
        "config": [],
        "steps": [
            "Enable the server",
            "Ask the assistant to add or finish tasks — they sync to the To-Do panel",
        ],
    },
    {
        "id": "orbix-gdrive",
        "name": "Google Drive",
        "category": "Your data",
        "icon": "drive",
        "description": "Search and read your Google Drive — find a document, pull its "
                       "text, and use it in the conversation. Reuses your Google sign-in.",
        "data": "Search & read Drive files (read-only)",
        "auth": "OAuth (Google) — extra Drive scope",
        "requires": "oauth",
        "default": False,
        "locked": False,
        "command": _PY,
        "args": [_server("gdrive_server.py")],
        "config": [],
        "steps": [
            "Enable the server",
            "Click Connect with Google again to grant Drive access (one-time re-consent)",
            "Ask the assistant to find or read a Drive document",
        ],
    },
    {
        "id": "orbix-cypher",
        "name": "Neo4j Cypher",
        "category": "Database",
        "icon": "neo4j",
        "description": "Run read and (confirmed) write Cypher against your graph and "
                       "explore its schema — power access beyond the memory tools.",
        "data": "Any Cypher query against your Neo4j database",
        "auth": "Neo4j URI + credentials (backend/.env)",
        "requires": "neo4j",
        "default": False,
        "locked": False,
        "command": _PY,
        "args": [_server("neo4j_cypher_server.py")],
        "config": [],
        "steps": [
            "Neo4j must be running (same credentials as Graph Memory)",
            "Enable the server",
            "Reads run freely; writes require write=true confirmation",
        ],
    },
    {
        "id": "orbix-github",
        "name": "GitHub",
        "category": "Developer",
        "icon": "github",
        "description": "Search repos and code, read and open issues, and inspect "
                       "repositories on GitHub.",
        "data": "Repos, code search, issues, PRs (via your token's scope)",
        "auth": "Personal access token",
        "requires": "apikey",
        "default": False,
        "locked": False,
        "command": _PY,
        "args": [_server("github_server.py")],
        "config": [
            {"key": "token", "label": "GitHub personal access token",
             "placeholder": "ghp_…", "secret": True, "required": True},
        ],
        "steps": [
            "Create a scoped PAT at github.com/settings/tokens",
            "Paste it here and Save (stored in backend/mcp_state.json, gitignored)",
            "Enable the server",
        ],
    },
    {
        "id": "orbix-git",
        "name": "Git (local)",
        "category": "Developer",
        "icon": "git",
        "description": "Inspect your local repositories — status, log, diff, grep and "
                       "branches — scoped to folders you allow. Read-only.",
        "data": "Local git repos inside your allowed folders",
        "auth": "Local — no login (scoped)",
        "requires": "local",
        "default": False,
        "locked": False,
        "command": _PY,
        "args": [_server("git_server.py")],
        "config": [
            {"key": "roots", "label": "Repo folders (comma-separated)",
             "placeholder": r"D:\PES\Capstone\Github-capstone", "secret": False,
             "required": True},
        ],
        "steps": [
            "Add the repo folders the assistant may inspect",
            "Enable the server (needs git on PATH)",
            "Ask about status, recent commits, or search the code",
        ],
    },
    {
        "id": "orbix-maps",
        "name": "Maps / Places",
        "category": "Web & actions",
        "icon": "maps",
        "description": "Geocode addresses, find places and get directions & distances "
                       "for location-aware requests.",
        "data": "Geocoding, places, directions (Google Maps)",
        "auth": "API key (Google Maps Platform)",
        "requires": "apikey",
        "default": False,
        "locked": False,
        "command": _PY,
        "args": [_server("maps_server.py")],
        "config": [
            {"key": "api_key", "label": "Google Maps API key", "placeholder": "AIza…",
             "secret": True, "required": True},
        ],
        "steps": [
            "Enable Geocoding/Places/Directions APIs in Google Cloud",
            "Paste the API key here and Save",
            "Enable the server",
        ],
    },
    {
        "id": "orbix-notion",
        "name": "Notion",
        "category": "Productivity",
        "icon": "notion",
        "description": "Search and read your Notion workspace, and append notes to a "
                       "page. Uses an internal integration token — no OAuth app needed.",
        "data": "Pages/databases shared with your integration",
        "auth": "Integration token",
        "requires": "apikey",
        "default": False,
        "locked": False,
        "command": _PY,
        "args": [_server("notion_server.py")],
        "config": [
            {"key": "token", "label": "Notion integration token",
             "placeholder": "secret_… / ntn_…", "secret": True, "required": True},
        ],
        "steps": [
            "Create an integration at notion.so/my-integrations",
            "Share the pages you want accessible with that integration",
            "Paste the token here, Save, and enable the server",
        ],
    },
    {
        "id": "orbix-slack",
        "name": "Slack",
        "category": "Productivity",
        "icon": "slack",
        "description": "List channels, read recent messages and post to a channel in "
                       "your Slack workspace.",
        "data": "Channels & messages the bot can access",
        "auth": "Bot token (xoxb-…)",
        "requires": "apikey",
        "default": False,
        "locked": False,
        "command": _PY,
        "args": [_server("slack_server.py")],
        "config": [
            {"key": "bot_token", "label": "Slack bot token", "placeholder": "xoxb-…",
             "secret": True, "required": True},
        ],
        "steps": [
            "Create a Slack app with channels:read, channels:history, chat:write",
            "Install it and copy the Bot User OAuth token",
            "Paste it here, Save, and enable the server",
        ],
    },
    {
        "id": "orbix-shell",
        "name": "Shell / Desktop",
        "category": "Local & system",
        "icon": "shell",
        "description": "Run terminal commands on this machine. Powerful and risky — off "
                       "by default, and blocks obviously destructive commands.",
        "data": "Terminal & process control (your user's permissions)",
        "auth": "Local — no login  ⚠ high risk",
        "requires": "local",
        "default": False,
        "locked": False,
        "command": _PY,
        "args": [_server("shell_server.py")],
        "config": [
            {"key": "workdir", "label": "Default working directory (optional)",
             "placeholder": r"D:\PES\Capstone", "secret": False},
        ],
        "steps": [
            "Understand the risk — the assistant can run commands as you",
            "Enable only when you need it",
            "Catastrophic patterns are refused; a 30s timeout applies",
        ],
    },
]

_BY_ID = {s["id"]: s for s in CATALOG}


# --------------------------------------------------------------------------- #
# State persistence
# --------------------------------------------------------------------------- #
def _default_state() -> dict:
    return {
        "enabled": {s["id"]: bool(s["default"]) for s in CATALOG},
        "config": {},
    }


def load_state() -> dict:
    """Read persisted {enabled, config}, filling defaults for any new catalog entry."""
    state = _default_state()
    try:
        if _STATE_FILE.exists():
            saved = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
            for sid, val in (saved.get("enabled") or {}).items():
                if sid in _BY_ID:
                    state["enabled"][sid] = bool(val)
            for sid, cfg in (saved.get("config") or {}).items():
                if sid in _BY_ID and isinstance(cfg, dict):
                    state["config"][sid] = cfg
    except Exception as e:  # noqa: BLE001 — never let a bad state file break startup
        logger.error("Failed to read mcp_state.json (%s); using defaults", e)
    return state


def save_state(state: dict) -> None:
    with _lock:
        try:
            _STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to write mcp_state.json: %s", e)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _env_for(sid: str, cfg: dict) -> dict:
    """Build the extra environment a server subprocess should see from its config.
    Config key 'roots' on orbix-files -> ORBIX_FILES_ROOTS, etc."""
    env: dict[str, str] = {}
    prefix = sid.replace("-", "_").upper()  # orbix-files -> ORBIX_FILES
    for k, v in (cfg or {}).items():
        if v is None or v == "":
            continue
        env[f"{prefix}_{k.upper()}"] = str(v)
    return env


def enabled_server_specs() -> dict[str, dict]:
    """
    The roster the host should spawn: {name: {command, args, env}} for every server
    the user has enabled. `env` inherits the parent's environment plus any config the
    user supplied for that server. Disabled servers are simply absent.
    """
    state = load_state()
    specs: dict[str, dict] = {}
    for s in CATALOG:
        sid = s["id"]
        if not state["enabled"].get(sid, s["default"]):
            continue
        env = dict(os.environ)
        # Force UTF-8 stdio — a Windows subprocess otherwise inherits the console's
        # default (non-UTF-8) codepage, and any server printing non-ASCII content
        # (a name, a search result, a file path) can crash outright with
        # UnicodeEncodeError instead of just mis-displaying it. Confirmed live for
        # the travel subprocess (google_service/travel_planner.py has the same fix);
        # applying it here too since any of these servers could hit the same class
        # of bug with the right input.
        env["PYTHONIOENCODING"] = "utf-8"
        env.update(_env_for(sid, state["config"].get(sid, {})))
        specs[sid] = {"command": s["command"], "args": list(s["args"]), "env": env}
    return specs


def list_servers() -> list[dict]:
    """Full catalog decorated with the user's enabled flag + saved (non-secret) config,
    for the Integrations page. Secret config values are redacted."""
    state = load_state()
    out = []
    for s in CATALOG:
        sid = s["id"]
        saved_cfg = state["config"].get(sid, {})
        cfg_fields = []
        # A field is "required" if it's a secret (token/key) or a scoping root — the
        # server can't do anything useful without it. workdir-style extras are optional.
        required_keys = [f["key"] for f in s.get("config", []) if f.get("required")]
        configured = all(str(saved_cfg.get(k, "")).strip() for k in required_keys)
        for field in s.get("config", []):
            val = saved_cfg.get(field["key"], "")
            if field.get("secret") and val:
                val = "••••••••"  # never echo secrets back to the browser
            cfg_fields.append({**field, "value": val})
        out.append({
            "configured": configured,
            "id": sid,
            "name": s["name"],
            "category": s["category"],
            "icon": s.get("icon", ""),
            "description": s["description"],
            "data": s["data"],
            "auth": s["auth"],
            "requires": s["requires"],
            "locked": s.get("locked", False),
            "enabled": bool(state["enabled"].get(sid, s["default"])),
            "steps": s.get("steps", []),
            "config": cfg_fields,
        })
    return out


def set_enabled(sid: str, enabled: bool) -> dict | None:
    """Toggle a server on/off. Returns the updated server view, or None if unknown."""
    if sid not in _BY_ID:
        return None
    if _BY_ID[sid].get("locked") and not enabled:
        # locked servers can't be disabled, but report current state rather than erroring
        return _one(sid)
    state = load_state()
    state["enabled"][sid] = bool(enabled)
    save_state(state)
    logger.info("MCP server %s %s", sid, "enabled" if enabled else "disabled")
    return _one(sid)


def set_config(sid: str, config: dict) -> dict | None:
    """Merge in config values (ignoring redacted placeholders). Returns updated view."""
    if sid not in _BY_ID:
        return None
    state = load_state()
    cur = dict(state["config"].get(sid, {}))
    for k, v in (config or {}).items():
        if v == "••••••••":  # redacted placeholder — keep existing value
            continue
        cur[k] = v
    state["config"][sid] = cur
    save_state(state)
    return _one(sid)


def _one(sid: str) -> dict:
    for srv in list_servers():
        if srv["id"] == sid:
            return srv
    return {}

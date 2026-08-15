"""
Adapter that lets a non-tool-calling model drive the SAME agent loop as
llama3.1:8b — used for `gpraneeth555/llama-3-13k`, whose Ollama template has
no {{ .Tools }}/{{ .ToolCalls }} rendering (checked via `ollama show <model>
--modelfile`) and so never receives/returns Ollama's native tool_calls field.

Empirically (one live test call, see PERFORMANCE_AUDIT.md-style note in the
plan), it reliably answers action requests with a parseable block:

    ### Output:
    {"tool": "gcalendar", "action": "create_event",
     "parameters": {"title": "...", "date": "..."}}

— the same shape the old, now-orphaned `backend/llm/hf_client.py` was built
to parse. This module re-targets that parsing at the REAL MCP tool names/
schemas (`mcp.tool_index`) instead of the old internal intent schema, so the
recovered call flows through agent.py's normal `mcp.call_tool(...)` execution,
tracing, and memory recall exactly like a native tool call would.

`agent.py::run_turn` is the only caller: it skips passing `tools=` to Ollama
for a legacy-format model, appends `render_tool_catalog()` to the system
prompt instead, and calls `parse_legacy_calls()` on the reply text.
"""

from __future__ import annotations

import re
import json
import logging
from types import SimpleNamespace

logger = logging.getLogger(__name__)

# (tool, action) as the fine-tune tends to name them -> the real MCP tool name.
# Keys are lower-cased before lookup. Extend this table as more mismatches are
# observed in eval_harness.py runs — it's the main thing that needs tuning.
_TOOL_ALIASES: dict[tuple[str, str], str] = {
    ("gmail", "send_email"): "send_email",
    ("gmail", "send"): "send_email",
    ("gmail", "email"): "send_email",
    ("gmail", "search_emails"): "list_emails",
    ("gmail", "get_emails"): "list_emails",
    ("gmail", "list_emails"): "list_emails",
    ("gmail", "inbox"): "list_emails",
    ("gmeet", "create_meeting"): "create_meet",
    ("gmeet", "schedule_meeting"): "create_meet",
    ("gmeet", "create_meet"): "create_meet",
    ("gcalendar", "create_event"): "create_calendar_event",
    ("gcalendar", "add_event"): "create_calendar_event",
    ("gcalendar", "create_calendar_event"): "create_calendar_event",
    ("gcalendar", "list_events"): "list_calendar_events",
    ("gcalendar", "list_calendar_events"): "list_calendar_events",
    ("gcalendar", "delete_event"): "delete_calendar_event",
    ("calendar", "create_event"): "create_calendar_event",
    ("calendar", "delete_event"): "delete_calendar_event",
    ("web", "search"): "web_search",
    ("web", "fetch"): "fetch_url",
    ("tasks", "add_task"): "add_task",
    ("tasks", "add"): "add_task",
}

# Fallback when only `action` (or only `tool`) is usable on its own.
_ACTION_ALIASES: dict[str, str] = {
    "send_email": "send_email",
    "send": "send_email",
    "create_meeting": "create_meet",
    "schedule_meeting": "create_meet",
    "create_event": "create_calendar_event",
    "add_event": "create_calendar_event",
    "delete_event": "delete_calendar_event",
    "search_emails": "list_emails",
    "get_emails": "list_emails",
    "list_emails": "list_emails",
    "inbox": "list_emails",
    "add_task": "add_task",
    "search": "web_search",
    "fetch": "fetch_url",
}

# real tool name -> {real param name: [accepted incoming aliases, in priority order]}
_PARAM_ALIASES: dict[str, dict[str, list[str]]] = {
    "send_email": {
        "recipient_email": ["recipient_email", "recipient", "to", "attendee_email", "attendee", "email"],
        "subject": ["subject"],
        "body": ["body", "message", "content"],
    },
    "create_meet": {
        "attendee_email": ["attendee_email", "attendee", "recipient", "recipients", "to", "email"],
        "event_title": ["event_title", "title", "summary", "meeting_title"],
        "event_description": ["event_description", "description", "details"],
        "email_link_to": ["email_link_to"],
    },
    "create_calendar_event": {
        "title": ["title", "event_title", "summary"],
        "date": ["date", "start_date", "start_time"],
        "time": ["time"],
        "description": ["description", "event_description", "details"],
        "type": ["type"],
    },
    "delete_calendar_event": {
        "event_id": ["event_id", "id"],
    },
    "list_emails": {
        "max_results": ["max_results"],
    },
    "list_calendar_events": {
        "year": ["year"], "month": ["month"], "upcoming_only": ["upcoming_only"],
    },
    "add_task": {
        "text": ["text", "task", "title", "item_name", "item", "name"],
    },
    "complete_task": {
        "task_id": ["task_id", "id"], "text": ["text"],
    },
    "web_search": {
        "query": ["query", "q"],
        "max_results": ["max_results"],
    },
    "fetch_url": {
        "url": ["url"],
    },
    "read_file": {
        "path": ["path", "file", "filename"],
    },
    "write_file": {
        "path": ["path", "file", "filename"],
        "content": ["content", "text", "body"],
        "append": ["append"],
    },
    "search_files": {
        "query": ["query", "q"], "root": ["root"],
    },
}


def _first_scalar(v):
    """Unwrap a single-item list (e.g. attendees: ["a@b.com"]) to its value."""
    if isinstance(v, list):
        return v[0] if v else None
    return v


def normalize_params(real_tool_name: str, raw_params: dict) -> dict:
    """Map the model's param names onto the real tool's parameter names."""
    if not isinstance(raw_params, dict):
        return {}
    alias_map = _PARAM_ALIASES.get(real_tool_name)
    if not alias_map:
        # Unknown tool (not in our table) — best-effort passthrough of scalars.
        return {k: _first_scalar(v) for k, v in raw_params.items()
                if isinstance(v, (str, int, float, bool)) or isinstance(v, list)}
    out: dict = {}
    for real_key, aliases in alias_map.items():
        for a in aliases:
            if a in raw_params and raw_params[a] not in (None, ""):
                out[real_key] = _first_scalar(raw_params[a])
                break
    return out


# Tier-4 fallback: the fine-tune wasn't trained on OUR tool names, so it uses
# its own vocabulary ("todo_list"/"add_item" observed live for what should be
# add_task) — exact (tool, action) pairs above can't anticipate every variant.
# This is a small rule-based classifier: which real-tool CATEGORY does the
# `tool` field describe, then which specific tool in that category does the
# `action` field's verb pick. Deterministic and explainable (useful for the
# paper's methodology section), not a guess dressed up as certainty.
_CATEGORY_HINTS: list[tuple[set[str], list[str]]] = [
    ({"mail", "email", "gmail"}, ["send_email", "list_emails"]),
    ({"meet", "meeting", "gmeet", "video", "conference"}, ["create_meet"]),
    ({"calendar", "event", "gcalendar", "schedule"},
     ["create_calendar_event", "list_calendar_events", "delete_calendar_event"]),
    ({"task", "todo", "reminder"}, ["add_task", "list_tasks", "complete_task", "delete_task"]),
    ({"web", "search", "browser"}, ["web_search", "fetch_url"]),
    ({"file", "filesystem", "document"}, ["read_file", "write_file", "search_files"]),
]
_SEND_WORDS = {"send", "create", "add", "new", "schedule", "start", "make"}
_LIST_WORDS = {"list", "show", "get", "search", "find", "check", "inbox", "read"}
_DELETE_WORDS = {"delete", "remove", "cancel"}
_COMPLETE_WORDS = {"complete", "done", "finish"}


def _tokens(s: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", s.lower()) if t}


def _category_pick(tool: str, action: str, available: set[str]) -> str | None:
    hay = _tokens(tool) | _tokens(action)
    for hints, candidates in _CATEGORY_HINTS:
        if not (hay & hints):
            continue
        options = [c for c in candidates if c in available]
        if not options:
            continue
        if len(options) == 1:
            return options[0]
        action_tokens = _tokens(action)
        if action_tokens & _DELETE_WORDS:
            pick = next((o for o in options if "delete" in o), None)
        elif action_tokens & _COMPLETE_WORDS:
            pick = next((o for o in options if "complete" in o), None)
        elif action_tokens & _LIST_WORDS:
            pick = next((o for o in options if o.startswith("list")
                        or o.startswith("fetch") or o.startswith("read")), None)
        elif action_tokens & _SEND_WORDS:
            pick = next((o for o in options if o.startswith(("send", "create", "add"))), None)
        else:
            pick = None
        return pick or options[0]  # ambiguous verb — default to the category's primary action
    return None


def resolve_tool_name(tool: str, action: str, available: set[str]) -> str | None:
    """Map a (tool, action) pair from the model's reply to a real, currently
    available MCP tool name — or None if nothing matches."""
    tool = (tool or "").strip().lower()
    action = (action or "").strip().lower()

    # 1) either field might already BE the real tool name.
    for candidate in (action, tool):
        if candidate and candidate in available:
            return candidate

    # 2) (tool, action) alias table.
    mapped = _TOOL_ALIASES.get((tool, action))
    if mapped and mapped in available:
        return mapped

    # 3) action-only alias table (model omitted or misused `tool`).
    mapped = _ACTION_ALIASES.get(action) or _ACTION_ALIASES.get(tool)
    if mapped and mapped in available:
        return mapped

    # 4) category + verb classifier — the generalizable fallback.
    mapped = _category_pick(tool, action, available)
    if mapped:
        return mapped

    return None


_JSON_ARRAY_RE = re.compile(r'\[[\s\S]*\]')
_JSON_OBJECT_RE = re.compile(r'\{[\s\S]*\}')
# Non-actionable outputs the fine-tune sometimes produces instead of a real call.
_NON_ACTIONABLE = frozenset({
    "user_interaction", "ask_clarification", "clarification",
    "confirm", "user_input", "ask_user", "weather_api",
})


def _raw_calls_from_text(content: str) -> list[dict]:
    """Pull whatever JSON object/array of tool-call-shaped dicts is in the reply."""
    for pattern in (_JSON_ARRAY_RE, _JSON_OBJECT_RE):
        m = pattern.search(content)
        if not m:
            continue
        try:
            parsed = json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, list):
            return [c for c in parsed if isinstance(c, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    return []


def parse_legacy_calls(content: str, tool_by_name: dict) -> list[SimpleNamespace]:
    """
    Recover real tool calls from a legacy-format model's plain-text reply.
    Returns a list shaped like Ollama's tool_calls entries (possibly empty) —
    same SimpleNamespace(function=SimpleNamespace(name=..., arguments=...))
    contract as agent.py's `_extract_stray_tool_call`.
    """
    if not content or not content.strip():
        return []
    available = set(tool_by_name.keys())
    calls: list[SimpleNamespace] = []
    for raw in _raw_calls_from_text(content):
        tool = str(raw.get("tool", "") or "")
        action = str(raw.get("action", "") or raw.get("function", "") or "")
        if tool.lower() in _NON_ACTIONABLE or action.lower() in _NON_ACTIONABLE:
            continue
        name = resolve_tool_name(tool, action, available)
        if not name:
            continue
        params = raw.get("parameters") or raw.get("arguments") or raw.get("params") or {}
        args = normalize_params(name, params)
        calls.append(SimpleNamespace(function=SimpleNamespace(name=name, arguments=args)))
    if not calls and _raw_calls_from_text(content):
        logger.info("legacy_tool_adapter: found JSON but couldn't map it to a real tool")
    return calls


def render_tool_catalog(tools: list[dict]) -> str:
    """
    Compact text form of the tool list, appended to the system prompt when the
    active model can't consume Ollama's native `tools=` param. Keeps the
    model's own habitual `{"tool": ..., "parameters": {...}}` framing (that's
    the format it reliably produces) but asks it to use the REAL tool name in
    the `tool` field — `resolve_tool_name` above is the safety net for when it
    reverts to its trained gmail/gmeet/gcalendar-style naming instead.
    """
    if not tools:
        return ""
    lines = ["\n\nAVAILABLE TOOLS — to use one, end your reply with exactly one line:\n"
             '{"tool": "<tool_name>", "parameters": {<args>}}\n'
             "(no other text after that line; omit it entirely if no tool is needed)\n"]
    for t in tools:
        fn = t.get("function", {})
        name = fn.get("name", "")
        desc = (fn.get("description") or "").strip().split("\n")[0][:100]
        params = list((fn.get("parameters", {}) or {}).get("properties", {}).keys())
        lines.append(f"- {name}({', '.join(params)}): {desc}")
    return "\n".join(lines)

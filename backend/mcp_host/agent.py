"""
OrbixAI agent loop — the reasoning brain that connects the local LLM to MCP tools.

One turn:
  1. record the user turn in SQLite working memory (instant)
  2. open the MCP servers, pull the live tool list
  3. give the tool-calling model the system prompt + transcript + tools
  4. loop: model emits tool call(s) -> host runs them via MCP -> feed results back
     -> repeat until the model produces a final answer
  5. record the assistant turn

Memory reads and writes are just tool calls inside this loop, so remembering and
answering use the same mechanism (architecture §7.2). "Respond first, remember
after" background extraction is a separate worker; here the model also writes
durable facts itself via remember_fact / remember_episode when the user states them.

Standalone test (needs Neo4j up + a tool-calling model pulled):
    python backend/mcp_host/agent.py "where do I live?"
    python backend/mcp_host/agent.py            # interactive REPL
"""

import os
import re
import sys
import json
import time
import uuid
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace

from ollama import AsyncClient

# make sibling (client.py, legacy_tool_adapter.py), backend/graph (working_memory,
# memory), backend/llm (model_registry) and backend/ itself (orchestration.metrics)
# importable whether this module is run standalone or imported as backend.mcp_host.agent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "graph"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "llm"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from client import MCPClientManager, default_servers  # type: ignore  # noqa: E402
import working_memory as wm  # type: ignore  # noqa: E402
import legacy_tool_adapter  # type: ignore  # noqa: E402
import model_registry  # type: ignore  # noqa: E402
from orchestration import metrics as turn_metrics  # type: ignore  # noqa: E402

logger = logging.getLogger(__name__)

# A tool-calling model. Override with env ORBIX_AGENT_MODEL.
# Default kept in sync with what's actually pulled on this machine (`ollama
# list` has no "llama3.2:3b" tag, only llama3.2:latest) — currently masked by
# .env's explicit ORBIX_AGENT_MODEL=llama3.1:8b, but fixed here too so this
# module still works correctly if ever run without that override.
AGENT_MODEL = os.environ.get("ORBIX_AGENT_MODEL", "llama3.1:8b")
MAX_STEPS = int(os.environ.get("ORBIX_AGENT_MAX_STEPS", "4"))
# How long Ollama keeps this model (+ its KV cache) resident after the last call.
# Ollama's own default is 5m; measured on this machine, reprocessing the ~20-28
# enabled-tool schemas from a cold/evicted context costs ~90-110s on CPU alone
# (confirmed: prompt_eval_duration dominates total latency), so letting the model
# idle out between turns of the same conversation is a large, avoidable cost for
# a single-user local assistant with RAM to spare. -1 = keep loaded indefinitely
# until explicitly replaced/unloaded. Must be numeric (not a bare "-1" string) —
# Ollama's Go server parses a string value as a duration and requires a unit
# suffix (e.g. "30m"); a plain "-1"/"300" string has none and is rejected.
_keep_alive_raw = os.environ.get("ORBIX_AGENT_KEEP_ALIVE", "-1").strip()
try:
    AGENT_KEEP_ALIVE: float | str = float(_keep_alive_raw)
except ValueError:
    AGENT_KEEP_ALIVE = _keep_alive_raw  # e.g. "30m" — valid Go duration string as-is
# The one server whose WRITE tools the agent must NOT call directly: memory. Its
# durable writes go through the background extractor (memory's single write path).
# Every OTHER enabled server's action tools (gsuite, files, tasks, …) are fair game,
# so the brain can act across any combination of connected integrations.
_MEMORY_SERVER = "orbix-memory"
from llm.ollama_options import ollama_options

# Ollama auto-selects the Apple Metal backend when num_gpu is omitted.
_OPTIONS = ollama_options(
    temperature=0.2,
    # Cap generation length per step — unset previously, so a single step could
    # in principle generate unboundedly long output, adding pure CPU eval time
    # on every step of an already multi-step loop. 512 is generous for a tool
    # call or a concise reply; matches the cap already used elsewhere (hf_client).
    num_predict=int(os.environ.get("ORBIX_AGENT_NUM_PREDICT", "1024")),
    # Ollama's runtime context window defaults to 4096 regardless of the model's
    # true capability (llama3.1:8b supports up to 131072) — confirmed via
    # `ollama ps` reporting context_length: 4096. The tool schemas alone for a
    # handful of enabled servers already cost ~2400-2700 tokens, so a multi-step
    # chained turn (e.g. recall trips -> list calendar -> send_email, each
    # appending a tool result) can overflow 4096 within a single turn. When
    # that happens Ollama silently truncates the OLDEST context to fit — which
    # can drop the system prompt/original user question entirely, producing
    # replies like "I don't see any user input or request." Raised well above
    # the observed tool-schema + multi-step overhead.
    num_ctx=int(os.environ.get("ORBIX_AGENT_NUM_CTX", "4096")),
)

_OWNER = "user:self"
# main.py's lifespan already runs wm.init_db() once at startup before serving any
# request; guard it here too so the (only) other caller — this module's standalone
# CLI — still gets it, without re-running the schema check/rebuild on every turn.
_wm_initialized = False


def _system_prompt() -> str:
    today = datetime.now().strftime("%A, %Y-%m-%d")
    return (
        "You are OrbixAI, a personal assistant with a persistent long-term memory "
        "stored in a knowledge graph, reachable through tools.\n"
        f"Today is {today}. The user (the owner) has the memory key '{_OWNER}'.\n\n"
        "MEMORY RULES — follow strictly:\n"
        "1. Before answering anything personal (the user's life, people, projects, "
        "preferences, plans), FIRST call `recall('user:self')` or `search(...)`. Use "
        "ONLY the facts a tool actually returned — never guess or invent personal "
        "facts — but ALWAYS restate them in your own plain words as a normal sentence. "
        "NEVER paste, quote, or repeat the tool's raw JSON/dict output in your reply — "
        "the user should never see a `{...}` blob, a `key`/`labels`/`relationships` "
        "field name, or anything that looks like the tool's return value itself. If you "
        "find yourself about to write a brace `{` or a field name from the tool result, "
        "stop and rephrase it as a sentence instead (e.g. 'Priya (person:priya.s@"
        "example.com)' → 'Your spouse is Priya').\n"
        "2. `recall('user:self')` returns the user's related people/projects/orgs in "
        "`relationships`, each often with a `details` object (e.g. a family member's "
        "birthday, a contact's email). READ those details to answer questions about "
        "family and contacts — the answer is usually already there. Only call `search` "
        "or another `recall` if the needed entity or field is genuinely missing. Never "
        "invent a key like 'person:123'; only use keys returned by a tool.\n"
        "3. You do NOT save anything yourself — a background process records new facts "
        "from the conversation automatically. When the user tells you something new, "
        "just acknowledge it naturally; do not claim you 'saved' or 'stored' it.\n"
        "4. The recent conversation is in your context, so you already know what the "
        "user just told you this session even if it isn't in the graph yet. If they "
        "refer back to something from EARLIER in this same conversation that you no "
        "longer see above, call `search_transcript(query)` to find it — but treat "
        "what it returns as a quote of what was said, not a verified fact.\n"
        "5. Keep replies concise and natural. Do not mention tools, keys, or memory "
        "internals to the user.\n\n"
        "ACTIONS — you can DO things, not just talk. Your available tools depend on "
        "which integrations are enabled; only ever call a tool that is actually in your "
        "tool list this turn. Common ones:\n"
        "A. Mail/meetings/calendar: `send_email`, `create_meet` (Google Meet link), "
        "`create_calendar_event`, `list_emails`, `list_calendar_events` (pass "
        "upcoming_only=true for 'upcoming'/'future' events — do not try to filter past "
        "events out yourself by reasoning about today's date, the tool does it exactly), "
        "`list_trips` (use this, NOT list_calendar_events, for 'how many trips'/'what "
        "trips are planned' — a multi-day trip is one calendar event per day, so "
        "list_calendar_events would show you e.g. 3 separate day-events for what is "
        "really 1 trip; list_trips is already grouped correctly). Web: "
        "`web_search` to look things up and `fetch_url` to read a page. Files: "
        "`read_file`, `search_files`, `write_file` (scoped folders). Tasks: `add_task`, "
        "`list_tasks`, `complete_task`.\n"
        "A2. ALWAYS use the actual tool-calling mechanism to act — never type out what "
        "tool you're going to call, or a numbered list of function calls, as plain reply "
        "text (e.g. never write something like '1. recall(...) 2. list_calendar_events(...)"
        "' or 'Let me check your calendar' and then stop). If you're about to do "
        "something, DO it via a real tool call in the same turn; only produce plain text "
        "once you're giving the user your actual final answer.\n"
        "B. Only do what the user asked. Do NOT invent extra details — no meeting link "
        "unless they asked for a meeting AND you actually called `create_meet`; no "
        "attachments, dates, or people they didn't mention.\n"
        "C. send_email needs a REAL email address (contains '@'). To find it: FIRST look "
        "in the 'WHAT YOU ALREADY KNOW' block below — the person's email may be listed "
        "there; if so, use it directly. If it isn't there, call `search(<name>)` once. If "
        "you still can't find an email, ASK the user for it in your reply and STOP — do "
        "NOT call send_email with a bare name like 'vikhyath'.\n"
        "C2. If the user says 'myself'/'my own email'/'send it to me', that means YOUR "
        "OWN address — it's the `email` field on the 'You (user:self)' line in 'WHAT YOU "
        "ALREADY KNOW' below (already loaded, no need to recall/search again). Use that "
        "value directly as recipient_email. Never call send_email with an empty or "
        "missing recipient_email and never ask the user for their own email when it's "
        "already listed there.\n"
        "D. COMPLETE THE WHOLE REQUEST — chain tools, don't stop after the first one. A "
        "request often needs SEVERAL tool calls in sequence. After a tool returns, look "
        "at its result and ask yourself 'is every part of what the user asked now done?' "
        "If not, call the next tool before you reply. Examples:\n"
        "   • 'set up a meet with X and email them/me the link' → call `create_meet` with "
        "`email_link_to` set to the recipient's email address. That single call creates "
        "the meet AND emails the link (result has emailed:true). This is the preferred, "
        "reliable way — do NOT just create the meet and stop, and do not claim you emailed "
        "the link unless the result shows emailed:true (or a send_email returned sent:true).\n"
        "   • 'email A and B' → call `send_email` once per recipient.\n"
        "   • 'set up a meeting with X and put it on my calendar and remind me' → "
        "`create_meet` (with email_link_to), THEN `create_calendar_event`, THEN "
        "`add_task` — do every part the user named.\n"
        "Only give your final plain-text reply once NOTHING is left to do.\n"
        "E. NEVER claim an action succeeded unless the tool result confirms it "
        "(send_email returns sent:true; create_meet returns a meet_link). If a tool "
        "returns an {'error': ...}, tell the user plainly that it failed and why — do "
        "not say 'Done'. If the error is needs_auth, tell them to sign in at /auth/login.\n"
        "F. After a successful action, confirm in one short natural sentence describing "
        "exactly what happened (e.g. 'Sent your test email to alice@example.com.')."
    )


def _format_context(recall: dict | None) -> str:
    """
    Turn a recall('user:self') result into a compact 'what you already know' block for
    the system prompt. Pre-loading the user's own facts and — crucially — their known
    contacts' emails means a small tool-calling model can resolve 'email vikhyath'
    directly from context instead of having to chain a separate recall/search step
    (which 3B models do unreliably). If a named person is absent here, the prompt tells
    the model to ASK for the email rather than guess.
    """
    if not recall or not isinstance(recall, dict):
        return ""
    lines: list[str] = []

    props = recall.get("properties") or {}
    self_bits = [f"{k}={v}" for k, v in props.items()
                 if k in ("name", "email", "city", "country", "role") and v]
    if self_bits:
        lines.append("- You (user:self): " + ", ".join(self_bits))

    facts = recall.get("facts") or []
    fact_bits = [f"{f.get('predicate')}: {f.get('value')}"
                 for f in facts if f.get("predicate") and f.get("value")]
    if fact_bits:
        lines.append("- Known facts about you: " + "; ".join(fact_bits[:12]))

    people: list[str] = []
    for r in recall.get("relationships") or []:
        # skip episode/observation nodes — they aren't contacts and just add noise
        key = r.get("key") or ""
        if r.get("rel") in ("HAS_MEMORY", "HAS_OBSERVATION") or key.startswith("mem:"):
            continue
        name = r.get("name") or key
        if not name:
            continue
        details = r.get("details") or {}
        rel = r.get("rel", "")
        extra = ", ".join(f"{k}: {v}" for k, v in details.items() if v)
        label = f"{name} ({rel.lower()})" if rel else str(name)
        people.append(f"    • {label}" + (f" — {extra}" if extra else ""))
    if people:
        lines.append("- People/things you know about (use these emails directly when "
                     "asked to contact them):")
        lines.extend(people[:20])

    if not lines:
        return ""
    return ("\n\nWHAT YOU ALREADY KNOW ABOUT THE USER (from long-term memory — treat as "
            "ground truth, no need to call recall for anything listed here):\n"
            + "\n".join(lines)
            + "\nIf the user asks you to contact or look up someone who is NOT listed "
            "above and whose email you don't have, ASK them for the email — never invent "
            "one.")


def _messages_from_session(session_id: str, context: str = "") -> list[dict]:
    """System prompt (+ pre-loaded memory context) + recent transcript."""
    msgs = [{"role": "system", "content": _system_prompt() + context}]
    # Ten recent messages preserve normal follow-ups while avoiding repeated prompt
    # evaluation over an ever-growing transcript on local models.
    history_limit = int(os.environ.get("ORBIX_AGENT_HISTORY_MESSAGES", "10"))
    for m in wm.get_messages(session_id, limit=history_limit):
        role = m["role"] if m["role"] in ("user", "assistant", "system") else "user"
        msgs.append({"role": role, "content": m["text"]})
    return msgs


def _friendly_step(name: str, args: dict) -> str:
    """Human-readable 'thinking' label for a tool call (for the UI)."""
    # memory tools
    if name == "search":
        return f"Searching memory for “{args.get('query', '')}”…"
    if name in ("recall", "get_entity"):
        return "Recalling what I know…"
    if name == "get_fact":
        return f"Looking up your {args.get('predicate', 'details')}…"
    if name == "fact_history":
        return "Checking the history…"
    if name == "search_transcript":
        return "Checking earlier in this conversation…"
    # gsuite action tools
    if name == "send_email":
        return f"Sending email to {args.get('recipient_email', '')}…"
    if name == "create_meet":
        return "Creating a Google Meet…"
    if name == "create_calendar_event":
        return f"Adding “{args.get('title', 'event')}” to your calendar…"
    if name == "list_emails":
        return "Fetching your emails…"
    if name == "list_calendar_events":
        return "Checking your calendar…"
    if name == "list_trips":
        return "Checking your planned trips…"
    if name == "delete_calendar_event":
        return "Removing a calendar event…"
    # web tools
    if name == "web_search":
        return f"Searching the web for “{args.get('query', '')}”…"
    if name == "fetch_url":
        return f"Reading {args.get('url', 'the page')}…"
    # filesystem tools
    if name == "read_file":
        return f"Reading {args.get('path', 'a file')}…"
    if name == "write_file":
        return f"Saving {args.get('path', 'a file')}…"
    if name == "search_files":
        return f"Searching files for “{args.get('query', '')}”…"
    if name == "list_directory":
        return "Listing files…"
    # task tools
    if name == "add_task":
        return f"Adding a to-do: “{args.get('text', '')}”…"
    if name in ("complete_task", "list_tasks", "delete_task"):
        return "Updating your to-do list…"
    # neo4j cypher tools
    if name == "run_cypher":
        return "Querying the knowledge graph…"
    if name == "get_schema":
        return "Reading the graph schema…"
    # github tools
    if name == "gh_search_repos":
        return f"Searching GitHub repos for “{args.get('query', '')}”…"
    if name == "gh_search_code":
        return f"Searching GitHub code for “{args.get('query', '')}”…"
    if name in ("gh_get_repo", "gh_list_issues"):
        return f"Reading GitHub {args.get('repo', 'repository')}…"
    if name == "gh_create_issue":
        return f"Opening a GitHub issue on {args.get('repo', '')}…"
    if name == "gh_whoami":
        return "Checking your GitHub account…"
    # git (local) tools
    if name in ("git_status", "git_log", "git_diff", "git_branches"):
        return "Inspecting your local repo…"
    if name == "git_grep":
        return f"Searching the repo for “{args.get('query', '')}”…"
    # maps tools
    if name == "geocode":
        return f"Locating “{args.get('address', '')}”…"
    if name == "search_places":
        return f"Finding places for “{args.get('query', '')}”…"
    if name == "directions":
        return f"Getting directions to {args.get('destination', '')}…"
    # notion tools
    if name == "notion_search":
        return f"Searching Notion for “{args.get('query', '')}”…"
    if name == "notion_get_page":
        return "Reading a Notion page…"
    if name == "notion_append_text":
        return "Adding a note to Notion…"
    # slack tools
    if name == "slack_list_channels":
        return "Listing your Slack channels…"
    if name == "slack_read_channel":
        return "Reading a Slack channel…"
    if name == "slack_post_message":
        return "Posting to Slack…"
    # google drive tools
    if name == "drive_search":
        return f"Searching your Drive for “{args.get('query', '')}”…"
    if name == "drive_list_recent":
        return "Listing recent Drive files…"
    if name == "drive_read_file":
        return "Reading a Drive file…"
    return f"Working ({name})…"


def _result_step(name: str, result) -> str | None:
    """Human-readable label describing what a tool call actually DID (for the UI).
    Returns None for read tools whose outcome isn't worth surfacing."""
    err = result.get("error") if isinstance(result, dict) else None
    if err:
        if isinstance(err, str) and "needs_auth" in err:
            return "⚠ Google isn't connected — please sign in"
        short = str(err).split("\n")[0][:80]
        return f"⚠ {name} failed: {short}"
    if name == "send_email" and isinstance(result, dict) and result.get("sent"):
        return f"✓ Email sent to {result.get('recipient', '')}"
    if name == "create_meet" and isinstance(result, dict) and result.get("meet_link"):
        if result.get("emailed"):
            return (f"✓ Google Meet created and link emailed to "
                    f"{result.get('email_recipient', '')}: {result['meet_link']}")
        return f"✓ Google Meet created: {result['meet_link']}"
    if name == "create_calendar_event" and isinstance(result, dict) and result.get("id"):
        return f"✓ Added “{result.get('title', 'event')}” to your calendar"
    if name == "delete_calendar_event":
        return "✓ Removed the calendar event"
    if name == "list_emails" and isinstance(result, dict):
        return f"✓ Found {result.get('count', 0)} emails"
    if name == "list_trips" and isinstance(result, dict):
        return f"✓ Found {result.get('count', 0)} trip(s)"
    if name == "web_search" and isinstance(result, dict):
        return f"✓ Found {result.get('count', 0)} web results"
    if name == "fetch_url" and isinstance(result, dict) and result.get("title"):
        return f"✓ Read “{str(result.get('title'))[:60]}”"
    if name == "write_file" and isinstance(result, dict) and result.get("written"):
        return f"✓ Saved {result.get('path', 'file')}"
    if name == "add_task" and isinstance(result, dict) and result.get("added"):
        return f"✓ Added to-do: “{result.get('text', '')}”"
    if name == "complete_task" and isinstance(result, dict) and result.get("completed"):
        return "✓ Marked a task done"
    if name == "run_cypher" and isinstance(result, dict) and "count" in result:
        return f"✓ Query returned {result.get('count', 0)} row(s)"
    if name == "gh_create_issue" and isinstance(result, dict) and result.get("created"):
        return f"✓ Opened GitHub issue #{result.get('number')}"
    if name in ("gh_search_repos", "gh_search_code") and isinstance(result, dict):
        n = len(result.get("repos") or result.get("results") or [])
        return f"✓ Found {n} GitHub result(s)"
    if name == "drive_search" and isinstance(result, dict) and "count" in result:
        return f"✓ Found {result.get('count', 0)} Drive file(s)"
    if name == "slack_post_message" and isinstance(result, dict) and result.get("posted"):
        return "✓ Posted to Slack"
    if name == "notion_append_text" and isinstance(result, dict) and result.get("appended"):
        return "✓ Added a note to Notion"
    if name == "geocode" and isinstance(result, dict) and result.get("lat") is not None:
        return f"✓ Found {result.get('address', 'location')}"
    return None


def _extract_stray_tool_call(content: str, valid_names: set[str]):
    """This model sometimes emits a tool call as a {"name": ..., "parameters": ...}
    text blob in the plain reply instead of Ollama's structured tool_calls field —
    observed specifically on the 2nd+ action of a chained request (e.g. "schedule
    a meeting AND THEN email them"): the first action runs fine, but the follow-up
    comes back as inert text and is silently dropped. Scan the content for a
    balanced {...} object naming a real tool and recover it as an actual call
    instead of treating that text as the final answer.
    Returns an object shaped like Ollama's tool-call entries (c.function.name /
    c.function.arguments) or None.
    """
    start = content.find('{')
    while start != -1:
        depth = 0
        for i in range(start, len(content)):
            if content[i] == '{':
                depth += 1
            elif content[i] == '}':
                depth -= 1
                if depth == 0:
                    candidate = content[start:i + 1]
                    try:
                        obj = json.loads(candidate)
                    except (json.JSONDecodeError, ValueError):
                        obj = None
                    if isinstance(obj, dict):
                        name = obj.get("name")
                        args = obj.get("parameters", obj.get("arguments"))
                        if name in valid_names and isinstance(args, dict):
                            return SimpleNamespace(
                                function=SimpleNamespace(name=name, arguments=args))
                    break
        start = content.find('{', start + 1)
    return None


_FUNC_CALL_RE = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^()]*)\)')


def _extract_stray_function_call(content: str, tool_by_name: dict):
    """Same failure mode as _extract_stray_tool_call, different shape: this
    model was also observed narrating a call as bare Python-call-style text —
    e.g. a completion-check re-ask coming back as literally `recall('user:self')`
    instead of a real structured tool call. Maps positional/keyword-looking
    args onto the tool's actual first declared parameter (the schemas here are
    all single- or few-argument), since there's no JSON object to parse.
    tool_by_name maps tool name -> its Ollama tool schema (for arg-name lookup).
    """
    for m in _FUNC_CALL_RE.finditer(content):
        name, argstr = m.group(1), m.group(2).strip()
        tool = tool_by_name.get(name)
        if not tool:
            continue
        args: dict = {}
        if argstr:
            kw_matches = re.findall(r'(\w+)\s*=\s*(\'[^\']*\'|"[^"]*"|[\w.\-@]+)', argstr)
            if kw_matches:
                for k, v in kw_matches:
                    args[k] = v.strip("'\"")
            else:
                props = list((tool.get("function", {}).get("parameters", {})
                             .get("properties", {}) or {}).keys())
                if props:
                    args[props[0]] = argstr.strip("'\"")
        return SimpleNamespace(function=SimpleNamespace(name=name, arguments=args))
    return None


def _extract_stray_command_call(content: str, tool_by_name: dict):
    """Recover Mistral's parenthesized command notation.

    Observed live: ``(web_search query 'flights from ...')``.  It names a real
    exposed tool and parameter but is not JSON or Python-call syntax, so the two
    structural recovery helpers above cannot see it.
    """
    match = re.search(
        r"\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+"
        r"(?:'([^']*)'|\"([^\"]*)\")\s*\)",
        content,
    )
    if not match:
        return None
    name, parameter = match.group(1), match.group(2)
    if name not in tool_by_name:
        return None
    properties = (
        tool_by_name[name].get("function", {}).get("parameters", {})
        .get("properties", {}) or {}
    )
    if parameter not in properties:
        return None
    return SimpleNamespace(function=SimpleNamespace(
        name=name, arguments={parameter: match.group(3) or match.group(4) or ""}
    ))


def _looks_like_raw_tool_dump(text: str) -> bool:
    """True if `text` looks like a tool's raw JSON/dict return value rather
    than a prose sentence. A structural/format check (does this parse as data,
    or contain our own internal tool-result field names) — not a semantic
    guess at what the user meant, so it's not subject to the same brittleness
    as a keyword-based intent heuristic."""
    s = text.strip()
    if not s:
        return False
    if s[0] in "{[":
        try:
            json.loads(s)
            return True
        except (json.JSONDecodeError, ValueError):
            pass
    # Even embedded in surrounding prose, these field names are distinctive of
    # this app's own tool-result shapes (recall/search/get_entity), not
    # natural English.
    return bool(re.search(r'"(relationships|properties|subject_key)"\s*:', s))


# Observed live: "mail this to X" / "send that to my assistant" as a follow-up
# turn, referencing something long the assistant itself said a few messages
# back (e.g. a full trip itinerary). The model has that text in its own
# context (it's right there in the transcript passed to it) but doesn't
# reliably retype a long block verbatim into a new tool call's `body` — it
# calls send_email with body left blank, which trips gsuite_server.py's own
# MailGenerator auto-draft fallback and sends a generic "Message from
# OrbixAI" instead of the actual content. Only fires when body is essentially
# EMPTY (not just short) — a genuinely brief intentional email like "I'll be
# 10 min late" must never be overwritten by an old itinerary.
_EMPTY_BODY_THRESHOLD = 20


def _maybe_fill_session_id(name: str, args: dict, session_id: str) -> dict:
    """search_transcript's session_id is host-side plumbing, not something the
    model should have to get right from memory — a small model asked to echo an
    opaque UUID back verbatim is exactly the kind of thing that goes wrong. Force
    it to the real current session regardless of what (if anything) the model
    supplied, same pattern as _maybe_fill_email_body below."""
    if name != "search_transcript":
        return args
    args = dict(args)
    args["session_id"] = session_id
    return args


def _maybe_fill_email_body(name: str, args: dict, session_id: str) -> dict:
    if name != "send_email":
        return args
    body = str(args.get("body") or "").strip()
    if len(body) >= _EMPTY_BODY_THRESHOLD:
        return args
    for m in reversed(wm.get_messages(session_id, limit=10)):
        if m["role"] == "assistant" and len(m["text"].strip()) >= 300:
            args = dict(args)
            args["body"] = m["text"]
            if not str(args.get("subject") or "").strip():
                args["subject"] = "Here's what you asked for"
            logger.info("Auto-filled empty send_email body from the last substantial "
                       "assistant message (%d chars)", len(m["text"]))
            break
    return args


# Observed live: the model answers a question, then narrates an intended
# follow-up action in plain text ("I will now send an email to your
# assistant...") instead of actually calling the tool — the loop takes that
# text as the final answer since no tool_calls were attached, so the user is
# told an action happened that never did. Rule A2/E in the system prompt
# already forbid this explicitly; a small local model doesn't always comply,
# so this is the structural backstop. Deliberately narrow (specific phrase +
# specific tool name), not a general sentiment classifier.
_PROMISE_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Each pattern has two kinds of branches: a FUTURE promise ("I'll ...",
    # "let me ...") and a PAST completion CLAIM ("I've ...", "already ...").
    # Originally only the future branch existed — confirmed live: "I've sent an
    # email with a detailed list..." (told the recipient never got the first
    # email, the model re-confirmed success without ever calling send_email
    # again) didn't match send(?:ing)? at all, since "sent" is a different word,
    # so the backstop below never fired. Same tense gap would apply to any of
    # these four tools, not just email — extended all four for consistency.
    #
    # CRITICAL: the past-tense branch MUST require a first-person claim prefix
    # (i've/i have/already) — a bare "sent"-near-"email" match is NOT safe.
    # Confirmed live (regression, since fixed): "sent to whom?" (a pure
    # read-only question) got answered "The email was sent by G praneeth to
    # yourself" — third-person, describing history, not a first-person claim —
    # and an earlier looser version of this pattern matched it anyway, wrongly
    # triggering the backstop into an actual unwanted send_email call with a
    # fabricated recipient. The other three tools below were written correctly
    # with this prefix from the start; only email's past-tense branches lacked
    # it. Never drop the prefix requirement here even to broaden coverage.
    (re.compile(r"\b(i(?:'ll| will)|let me|going to)\b[^.!?\n]{0,25}\bemail\b"
               r"|\bsend(?:ing)?\s+(?:you\s+|an?\s+)?email\b"
               r"|\b(i(?:'ve| have)|already)\b[^.!?\n]{0,25}\b(sent|sending)\b[^.!?\n]{0,25}\bemail\b"
               r"|\bemail\b[^.!?\n]{0,25}\b(i(?:'ve| have))\b[^.!?\n]{0,10}\bsent\b", re.I), "send_email"),
    (re.compile(r"\b(i(?:'ll| will)|let me|going to)\b[^.!?\n]{0,25}"
               r"\b(set up|creat\w*|schedul\w*)\b[^.!?\n]{0,15}\b(meet|meeting)\b"
               r"|\b(i(?:'ve| have)|already)\b[^.!?\n]{0,25}"
               r"\b(set up|created|scheduled)\b[^.!?\n]{0,15}\b(meet|meeting)\b", re.I),
     "create_meet"),
    (re.compile(r"\b(i(?:'ll| will)|let me|going to)\b[^.!?\n]{0,25}"
               r"\b(add|creat\w*)\b[^.!?\n]{0,15}\b(calendar|event)\b"
               r"|\b(i(?:'ve| have)|already)\b[^.!?\n]{0,25}"
               r"\b(added|created)\b[^.!?\n]{0,15}\b(calendar|event)\b", re.I),
     "create_calendar_event"),
    (re.compile(r"\b(i(?:'ll| will)|let me|going to)\b[^.!?\n]{0,25}"
               r"\b(add|creat\w*)\b[^.!?\n]{0,15}\b(to-?do|task|remind\w*)\b"
               r"|\b(i(?:'ve| have)|already)\b[^.!?\n]{0,25}"
               r"\b(added|created)\b[^.!?\n]{0,15}\b(to-?do|task|remind\w*)\b", re.I),
     "add_task"),
]


def _unfulfilled_promise(reply: str, trace: list[dict], tool_by_name: dict) -> str | None:
    """Return the tool name the reply promised but never actually called
    (successfully) this turn, or None if nothing's amiss."""
    if not reply:
        return None
    for pattern, tool_name in _PROMISE_PATTERNS:
        if tool_name not in tool_by_name:
            continue  # that integration isn't even enabled this turn
        if not pattern.search(reply):
            continue
        done = any(t.get("tool") == tool_name and isinstance(t.get("result"), dict)
                   and not t["result"].get("error") for t in trace)
        if not done:
            return tool_name
    return None


def _synthesize_from_trace(trace: list[dict]) -> str:
    """Build a plain-language confirmation from whatever tool calls actually
    succeeded this turn. Used when the model ends a turn with no text."""
    outcomes = [s for s in (_result_step(t.get("tool"), t.get("result"))
                            for t in trace) if s and s.startswith("✓")]
    return " ".join(dict.fromkeys(outcomes))


async def run_turn(user_text: str, session_id: str | None = None, *,
                   model: str | None = None, max_steps: int = MAX_STEPS,
                   on_event=None, label: str | None = None,
                   allowed_tool_names: set[str] | None = None,
                   allowed_server_names: set[str] | None = None) -> dict:
    """
    Run one full agent turn. Returns {reply, session_id, trace, steps, model, mode}.
    `on_event(ev)` — optional async callback fed {"type":"thinking","step":...}
    events so a caller (e.g. the SSE endpoint) can stream progress live.
    `label` — optional benchmark-prompt id (used by eval_harness.py) so repeated
    runs of the same prompt can be grouped in the metrics log; leave None for
    ordinary interactive turns.
    """
    async def emit(step: str):
        if on_event:
            try:
                await on_event({"type": "thinking", "step": step})
            except Exception:  # noqa: BLE001
                pass

    # Wall-clock instrumentation for latency research — every LLM/tool call and
    # the total turn get an explicit elapsed-time INFO log, so real per-query/
    # per-step latency can be extracted straight from logs.
    _turn_start = time.monotonic()
    global _wm_initialized
    if not _wm_initialized:
        wm.init_db()
        _wm_initialized = True
    session_id = session_id or uuid.uuid4().hex
    wm.start_session(session_id)

    # The old flush-barrier (drain SQLite queue -> Neo4j before replying) is gone —
    # the fast path is now the LangGraph background write (orchestration.background_tasks
    # -> extraction_executor._extract_from_turn, fire-and-forget right after the reply).
    # The user turn is still enqueued here (enqueue defaults to True for role="user")
    # so extract_queue acts as a durable retry safety net: if the fast path fails (e.g.
    # Neo4j briefly down), the row stays 'pending' instead of the fact being lost
    # outright, and the scheduled maintenance loop (main.py) drains it later. The fast
    # path calls wm.mark_done(msg_id) on success so this row is never double-processed.
    user_msg_id = wm.add_message(session_id, "user", user_text)   # working memory (instant)

    # Re-resolved every turn (not frozen at import) so a model toggle in the UI
    # takes effect on the very next request — same "read the registry fresh
    # each time" pattern client.py already uses for the enabled MCP servers.
    model = model or model_registry.get_active()
    mode = model_registry.tool_calling_mode(model)  # "native" or "legacy"
    llm = AsyncClient()
    trace: list[dict] = []

    # Starting every enabled MCP subprocess on every turn adds avoidable startup
    # latency. The router knows the relevant integration, so start only those
    # servers. None retains the full dynamic roster for direct/legacy callers.
    server_specs = None
    if allowed_server_names is not None:
        server_specs = {
            name: spec for name, spec in default_servers().items()
            if name in allowed_server_names
        }

    async with MCPClientManager(servers=server_specs) as mcp:
        # If a previous turn's background extraction hasn't landed yet (e.g. this is
        # the first message of a brand-new session — startNewChat() is 100% client-
        # side, it never calls the backend, so "first message of a session we may
        # not have seen before" is the only point the backend can actually act on),
        # the recall() below would miss facts the user JUST stated in the OLD
        # session — exactly the cross-session gap that makes memory look like it
        # "forgot" something said seconds ago. Single-owner app (one _OWNER), so no
        # per-user filtering is needed: any pending row is relevant to this recall.
        #
        # This is a hard wait, not best-effort: correctness (answer from the FULL
        # graph) matters more than shaving latency off this one turn. The only
        # timeout here is a generous hang-guard (Ollama/Neo4j wedged, not just
        # slow) — it exists to stop the app from hanging forever, not to cap
        # normal-case latency. drain_all() itself already fails fast (no wait) if
        # Neo4j is simply unreachable — see extractor.mem_reachable().
        # Never drain the durable extraction backlog on the user-facing request.
        # Extraction may require a separate model call per row and previously added
        # up to 60 seconds before inference even began. Background extraction plus
        # scheduled maintenance remain the durable retry paths.
        if wm.pending_count() > 0:
            logger.info("Memory extraction backlog pending; leaving it to background maintenance")

        # Pre-load the user's long-term memory (their facts + known contacts' emails)
        # into the system prompt so a small tool-calling model can resolve "email X"
        # directly from context instead of having to chain a separate recall step.
        context = ""
        _t0 = time.monotonic()
        if "orbix-memory" in mcp.sessions:
            try:
                await emit("Recalling what I know…")
                recall = await mcp.call_tool("recall", {"subject_key": _OWNER})
                context = _format_context(recall)
            except Exception:  # noqa: BLE001 — recall is best-effort
                pass
        logger.info("LATENCY recall (pre-load): %.2fs", time.monotonic() - _t0)

        # The brain may READ anything (memory recall, list emails/calendar, web search,
        # files, tasks) and may ACT via any enabled server EXCEPT memory (send_email,
        # create_meet, calendar CRUD, write_file, add_task, …). Memory WRITE tools stay
        # excluded — the background extractor is memory's single write path, so keeping
        # them out avoids races/duplication. Whatever servers the user has enabled is
        # exactly the tool surface here, so integrations compose in any order.
        def _allowed(name: str) -> bool:
            memory_policy = mcp.is_readonly(name) or mcp.server_of(name) != _MEMORY_SERVER
            scope_policy = allowed_tool_names is None or name in allowed_tool_names
            return memory_policy and scope_policy
        tools = [t for t in mcp.ollama_tools()
                 if _allowed(t["function"]["name"])]
        _tool_names = {t["function"]["name"] for t in tools}
        _tool_by_name = {t["function"]["name"]: t for t in tools}
        logger.info("Agent turn: model=%s (%s), %d tools from %d server(s)%s",
                    model, mode, len(tools), len(mcp.sessions),
                    (f"; unavailable: {', '.join(mcp.failed)}" if mcp.failed else ""))

        # "legacy" models (no native Ollama tool-calling support, e.g. the
        # fine-tune) get the tool list as text in the system prompt instead of
        # via `tools=`; their replies are parsed by legacy_tool_adapter instead
        # of read off Ollama's structured tool_calls field.
        if mode == "legacy":
            context += legacy_tool_adapter.render_tool_catalog(tools)
        messages = _messages_from_session(session_id, context)

        reply = ""
        for step in range(max_steps):
            _t0 = time.monotonic()
            resp = await llm.chat(model=model, messages=messages,
                                  tools=(tools if mode == "native" else None),
                                  options=_OPTIONS, keep_alive=AGENT_KEEP_ALIVE)
            logger.info("LATENCY llm.chat step=%d: %.2fs", step, time.monotonic() - _t0)
            msg = resp.message
            calls = msg.tool_calls or []
            content = msg.content or ""
            if not calls and content and mode == "legacy":
                legacy_calls = legacy_tool_adapter.parse_legacy_calls(content, _tool_by_name)
                if legacy_calls:
                    logger.info("Recovered %d legacy-format tool call(s): %s",
                                len(legacy_calls), [c.function.name for c in legacy_calls])
                    calls = legacy_calls
                    content = ""
            if not calls and content:
                stray = (_extract_stray_tool_call(content, _tool_names)
                         or _extract_stray_function_call(content, _tool_by_name)
                         or _extract_stray_command_call(content, _tool_by_name))
                if stray:
                    logger.info("Recovered stray tool call from plain text: %s", stray.function.name)
                    calls = [stray]
                    content = ""  # don't echo the recovered call text back into history

            # reconstruct the assistant message (with any tool calls) into history
            asst: dict = {"role": "assistant", "content": content}
            if calls:
                asst["tool_calls"] = [
                    {"function": {"name": c.function.name,
                                  "arguments": dict(c.function.arguments or {})}}
                    for c in calls
                ]
            messages.append(asst)

            if not calls:
                reply = content.strip()
                break

            # execute each tool call and feed results back
            for c in calls:
                name = c.function.name
                args = dict(c.function.arguments or {})
                args = _maybe_fill_email_body(name, args, session_id)
                args = _maybe_fill_session_id(name, args, session_id)
                await emit(_friendly_step(name, args))
                _t0 = time.monotonic()
                result = await mcp.call_tool(name, args)
                logger.info("LATENCY tool %s step=%d: %.2fs", name, step, time.monotonic() - _t0)
                trace.append({"step": step, "tool": name, "args": args,
                              "readonly": mcp.is_readonly(name), "result": result})
                logger.info("  tool %s(%s) -> %s", name, args,
                            str(result)[:160])
                outcome = _result_step(name, result)
                if outcome:
                    await emit(outcome)
                messages.append({"role": "tool", "tool_name": name,
                                 "content": json.dumps(result, default=str)})

            # A round containing only write/action calls already has everything
            # needed for a truthful confirmation in the verified trace. Avoid a
            # second full model generation whose only job would be to say "done".
            terminal_write_tools = {
                "send_email", "create_meet", "create_calendar_event",
                "delete_calendar_event", "add_task", "complete_task",
                "delete_task", "write_file", "gh_create_issue",
                "notion_append_text", "slack_post_message",
            }
            if calls and all(c.function.name in terminal_write_tools for c in calls):
                reply = _synthesize_from_trace(trace)
                break

            if mode == "legacy":
                # This model's template has no {{ range .Messages }} / tool-result
                # rendering at all (confirmed via its Modelfile) — re-prompting it
                # with a tool result doesn't read as "here's what happened," so it
                # just re-emits a near-duplicate call instead of recognizing the
                # request is done. Observed live: 6 duplicate add_task writes for
                # one request before finally hitting max_steps. For a write action
                # (email, calendar, meet) that would mean real duplicate side
                # effects, so: execute exactly one round of calls, then stop and
                # synthesize the final reply from what actually happened (below)
                # instead of looping this model at all.
                reply = ""
                break
        else:
            # exhausted steps without a final plain answer
            reply = reply or "I wasn't able to complete that — too many tool steps."

        # Structural backstop for the unfulfilled-promise failure mode (see
        # _unfulfilled_promise docstring): give the model exactly ONE more shot,
        # restricted to ONLY the specific tool it already promised — it cannot
        # invent an unrelated action because no other tool schema is even
        # offered this step. That restriction is what makes this safe to do
        # automatically; an earlier unrestricted "did you really finish?" retry
        # caused a real unintended send_email (documented near-miss), which is
        # exactly the failure mode this narrow version can't reproduce.
        if mode == "native":
            promised_tool = _unfulfilled_promise(reply, trace, _tool_by_name)
            if promised_tool:
                try:
                    await emit(f"Actually completing that ({promised_tool})…")
                    messages.append({"role": "assistant", "content": reply})
                    messages.append({"role": "user", "content":
                        f"You said you would use {promised_tool} but never actually "
                        f"called it. Call it now with the right arguments from the "
                        f"conversation above — or, if you genuinely can't (missing "
                        f"info), say so plainly instead."})
                    _t0 = time.monotonic()
                    resp2 = await llm.chat(model=model, messages=messages,
                                           tools=[_tool_by_name[promised_tool]],
                                           options=_OPTIONS, keep_alive=AGENT_KEEP_ALIVE)
                    logger.info("LATENCY llm.chat promise-check: %.2fs", time.monotonic() - _t0)
                    calls2 = resp2.message.tool_calls or []
                    if calls2:
                        c = calls2[0]
                        name, args = c.function.name, dict(c.function.arguments or {})
                        args = _maybe_fill_email_body(name, args, session_id)
                        await emit(_friendly_step(name, args))
                        result = await mcp.call_tool(name, args)
                        trace.append({"step": max_steps, "tool": name, "args": args,
                                      "readonly": mcp.is_readonly(name), "result": result})
                        logger.info("  tool %s(%s) -> %s", name, args, str(result)[:160])
                        outcome = _result_step(name, result)
                        if outcome:
                            await emit(outcome)
                            reply = reply.rstrip() + "\n\n" + outcome
                        elif isinstance(result, dict) and result.get("error"):
                            reply = (reply.rstrip() + "\n\n(I actually wasn't able to "
                                    f"complete that — {result['error']})")
                    else:
                        # Still didn't call it — correct the false promise instead of
                        # letting it reach the user unchallenged.
                        reply = (reply.rstrip() + "\n\n(I said I'd do that but wasn't "
                                "actually able to complete it.)")
                except Exception as e:  # noqa: BLE001 — never let this backstop break a turn
                    logger.warning("Unfulfilled-promise follow-up failed: %s", e)

    # Small models occasionally finish with empty text right after a forced follow-up
    # tool call. Never return a blank turn, and never say "Done" unless the trace
    # proves an action actually succeeded.
    if not reply.strip():
        reply = _synthesize_from_trace(trace) or (
            "I couldn't generate a reliable response for that request. "
            "No external action was confirmed."
        )

    # Never show the private system prompt. A small local model has occasionally
    # echoed its opening instructions (including the internal owner memory key)
    # after a vague follow-up such as "go ahead".
    if ("persistent long-term memory stored in a knowledge graph" in reply.lower()
            and "memory key 'user:self'" in reply.lower()):
        reply = _synthesize_from_trace(trace) or (
            "I lost the context for that follow-up. Please repeat the request; "
            "no external action was confirmed."
        )

    # Never expose invented placeholders as if they were action results. If real
    # tools succeeded, replace the model prose with confirmations synthesized from
    # the trace. If no tool ran, say that plainly.
    placeholder_re = re.compile(
        r"\[(?:google\s+meet\s+)?link\]|\{\s*(?:meet_link|event_id|emailed)\b|"
        r"\b(?:meet_link|event_id)\b(?=\s*[,}:])",
        re.IGNORECASE,
    )
    if placeholder_re.search(reply):
        verified = _synthesize_from_trace(trace)
        reply = verified or (
            "I couldn't complete that action because no tool call was confirmed. "
            "No meeting, email, or other external action was created."
        )

    # Occasionally the model echoes a literal "assistant" role-header as plain
    # text at the start of its reply (e.g. "assistant\n\nI have completed...") —
    # a chat-template leak, more likely under context pressure. Strip it rather
    # than show it to the user.
    reply = re.sub(r'^\s*assistant\s*[:\n]+\s*', '', reply, flags=re.IGNORECASE)

    # Last-resort structural safety net: this model has been observed pasting a
    # tool's raw JSON/dict return value straight into its reply instead of
    # describing it in prose (e.g. dumping recall()'s {"relationships": [...]}
    # verbatim), despite an explicit system-prompt rule against it. This is a
    # shape check (does this look like serialized data), not a semantic guess
    # at meaning — never show the user something that looks like our own
    # internal tool-result format.
    if _looks_like_raw_tool_dump(reply):
        logger.warning("Reply looked like a raw tool-result dump, not prose — replaced with a fallback")
        reply = ("I found that information but had trouble putting it into words clearly "
                 "— could you ask me again, maybe a bit more specifically?")

    wm.add_message(session_id, "assistant", reply, enqueue=False)
    total_elapsed = time.monotonic() - _turn_start
    logger.info("LATENCY total turn (%d step(s)): %.2fs", len(trace), total_elapsed)

    # A turn "succeeded" if every tool call it made actually worked — the
    # metric the paper cares about is tool-call correctness, not just "the
    # model said something." A turn with no tool calls (plain chat) counts
    # as a success by default.
    tool_errors = [t for t in trace if isinstance(t.get("result"), dict)
                   and t["result"].get("error")]
    turn_metrics.record_turn(
        model=model, mode=mode, session_id=session_id, trace=trace,
        success=not tool_errors, elapsed_seconds=round(total_elapsed, 2),
        steps=len(trace), label=label,
    )

    return {"reply": reply, "session_id": session_id, "user_msg_id": user_msg_id,
            "trace": trace, "steps": len(trace), "elapsed_seconds": round(total_elapsed, 2),
            "model": model, "mode": mode}


# --------------------------------------------------------------------------- #
# CLI for manual testing
# --------------------------------------------------------------------------- #
async def _amain() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")

    if len(sys.argv) > 1:  # one-shot
        out = await run_turn(" ".join(sys.argv[1:]))
        print("\n[trace]")
        for t in out["trace"]:
            tag = "R" if t["readonly"] else "W"
            print(f"  ({tag}) {t['tool']}({t['args']}) -> {str(t['result'])[:120]}")
        print("\n[reply]", out["reply"])
        return

    # interactive REPL (one persistent session)
    sid = uuid.uuid4().hex
    print(f"OrbixAI agent — session {sid[:8]} (Ctrl-C to exit)")
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        out = await run_turn(q, session_id=sid)
        for t in out["trace"]:
            tag = "R" if t["readonly"] else "W"
            print(f"  · {tag} {t['tool']}({t['args']})")
        print("orbix>", out["reply"])


if __name__ == "__main__":
    import asyncio
    asyncio.run(_amain())

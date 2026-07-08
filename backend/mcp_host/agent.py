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
import sys
import json
import uuid
import asyncio
import logging
from pathlib import Path
from datetime import datetime

from ollama import AsyncClient

# make sibling (client.py) and backend/graph (working_memory, memory) importable
# whether this module is run standalone or imported as backend.mcp_host.agent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "graph"))

from client import MCPClientManager  # type: ignore  # noqa: E402
import working_memory as wm  # type: ignore  # noqa: E402

logger = logging.getLogger(__name__)

# A tool-calling model. Override with env ORBIX_AGENT_MODEL.
AGENT_MODEL = os.environ.get("ORBIX_AGENT_MODEL", "llama3.2:3b")
MAX_STEPS = int(os.environ.get("ORBIX_AGENT_MAX_STEPS", "8"))
# The one server whose WRITE tools the agent must NOT call directly: memory. Its
# durable writes go through the background extractor (memory's single write path).
# Every OTHER enabled server's action tools (gsuite, files, tasks, …) are fair game,
# so the brain can act across any combination of connected integrations.
_MEMORY_SERVER = "orbix-memory"
# Match the project's CPU-only Ollama setup (CUDA issues on this machine).
_OPTIONS = {"temperature": 0.2, "num_gpu": int(os.environ.get("OLLAMA_NUM_GPU", "0"))}

_OWNER = "user:self"


def _system_prompt() -> str:
    today = datetime.now().strftime("%A, %Y-%m-%d")
    return (
        "You are OrbixAI, a personal assistant with a persistent long-term memory "
        "stored in a knowledge graph, reachable through tools.\n"
        f"Today is {today}. The user (the owner) has the memory key '{_OWNER}'.\n\n"
        "MEMORY RULES — follow strictly:\n"
        "1. Before answering anything personal (the user's life, people, projects, "
        "preferences, plans), FIRST call `recall('user:self')` or `search(...)` and "
        "answer ONLY from what the tools return. Never guess personal facts.\n"
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
        "user just told you this session even if it isn't in the graph yet.\n"
        "5. Keep replies concise and natural. Do not mention tools, keys, or memory "
        "internals to the user.\n\n"
        "ACTIONS — you can DO things, not just talk. Your available tools depend on "
        "which integrations are enabled; only ever call a tool that is actually in your "
        "tool list this turn. Common ones:\n"
        "A. Mail/meetings/calendar: `send_email`, `create_meet` (Google Meet link), "
        "`create_calendar_event`, `list_emails`, `list_calendar_events`. Web: "
        "`web_search` to look things up and `fetch_url` to read a page. Files: "
        "`read_file`, `search_files`, `write_file` (scoped folders). Tasks: `add_task`, "
        "`list_tasks`, `complete_task`. Call the tool — don't just describe what you'd do.\n"
        "B. Only do what the user asked. Do NOT invent extra details — no meeting link "
        "unless they asked for a meeting AND you actually called `create_meet`; no "
        "attachments, dates, or people they didn't mention.\n"
        "C. send_email needs a REAL email address (contains '@'). To find it: FIRST look "
        "in the 'WHAT YOU ALREADY KNOW' block below — the person's email may be listed "
        "there; if so, use it directly. If it isn't there, call `search(<name>)` once. If "
        "you still can't find an email, ASK the user for it in your reply and STOP — do "
        "NOT call send_email with a bare name like 'vikhyath'.\n"
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
    for m in wm.get_messages(session_id, limit=20):
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


import re

# Words in a reply that assert an action was taken — used to decide whether a turn is
# "action-bearing" and therefore worth a completion check.
_ACTION_CLAIM = re.compile(
    r"\b(sent|emailed|e-mailed|mailed|scheduled|created|set up|added|invited|booked|"
    r"arranged|shared)\b", re.I)


def _needs_completion_check(reply: str, trace: list) -> bool:
    """
    True when a turn looks like it performed (or claims) an action and is worth
    re-verifying. Skips plain-chat turns so we don't add latency or nudge the model
    into needless tool calls when the user just chatted.
    """
    called_action = any(not t.get("readonly") for t in (trace or []))
    claims_action = bool(reply and _ACTION_CLAIM.search(reply))
    return called_action or claims_action


def _completion_check_prompt(trace: list) -> str:
    """A generic self-verification nudge listing the successful action tools so far."""
    done = []
    for t in trace or []:
        res = t.get("result")
        if t.get("readonly") or not isinstance(res, dict) or res.get("error"):
            continue
        done.append(t.get("tool"))
    done_str = ", ".join(dict.fromkeys(done)) or "none"
    return (
        "COMPLETION CHECK — do not answer the user yet. Re-read their most recent "
        "request and list, in your head, every action it asks for. Action tools you have "
        f"successfully called this turn: {done_str}. For EACH requested action, confirm "
        "there is a matching successful tool call above. Critically: if they asked you to "
        "email / send / mail something and `send_email` is NOT in that list, then you "
        "have NOT emailed them yet — creating a Google Meet or calendar invite does not "
        "count as sending the email they asked for. If anything is missing, CALL the "
        "missing tool now (e.g. `send_email`, passing the real meet_link URL from "
        "create_meet in its `meet_link` argument). Only if "
        "every requested action is truly backed by a successful tool call, give your "
        "final answer describing exactly what you actually did — nothing you didn't.")


async def run_turn(user_text: str, session_id: str | None = None, *,
                   model: str | None = None, max_steps: int = MAX_STEPS,
                   on_event=None) -> dict:
    """
    Run one full agent turn. Returns {reply, session_id, trace, steps}.
    `on_event(ev)` — optional async callback fed {"type":"thinking","step":...}
    events so a caller (e.g. the SSE endpoint) can stream progress live.
    """
    async def emit(step: str):
        if on_event:
            try:
                await on_event({"type": "thinking", "step": step})
            except Exception:  # noqa: BLE001
                pass

    wm.init_db()
    session_id = session_id or uuid.uuid4().hex
    wm.start_session(session_id)

    # NOTE: the old flush-barrier (drain SQLite queue -> Neo4j before replying) has been
    # removed. Durable memory writes now happen exclusively through the LangGraph
    # background write path (orchestration.background_tasks -> extraction_executor).
    # The user turn is still recorded in working memory for transcript context, but is
    # NOT enqueued for the retired queue-drain pathway.
    wm.add_message(session_id, "user", user_text, enqueue=False)   # working memory (instant)

    model = model or AGENT_MODEL
    llm = AsyncClient()
    trace: list[dict] = []

    async with MCPClientManager() as mcp:
        # Pre-load the user's long-term memory (their facts + known contacts' emails)
        # into the system prompt so a small tool-calling model can resolve "email X"
        # directly from context instead of having to chain a separate recall step.
        context = ""
        try:
            await emit("Recalling what I know…")
            recall = await mcp.call_tool("recall", {"subject_key": _OWNER})
            context = _format_context(recall)
        except Exception:  # noqa: BLE001
            pass
        messages = _messages_from_session(session_id, context)

        # The brain may READ anything (memory recall, list emails/calendar, web search,
        # files, tasks) and may ACT via any enabled server EXCEPT memory (send_email,
        # create_meet, calendar CRUD, write_file, add_task, …). Memory WRITE tools stay
        # excluded — the background extractor is memory's single write path, so keeping
        # them out avoids races/duplication. Whatever servers the user has enabled is
        # exactly the tool surface here, so integrations compose in any order.
        def _allowed(name: str) -> bool:
            return mcp.is_readonly(name) or mcp.server_of(name) != _MEMORY_SERVER
        tools = [t for t in mcp.ollama_tools()
                 if _allowed(t["function"]["name"])]
        logger.info("Agent turn: model=%s, %d tools from %d server(s)%s",
                    model, len(tools), len(mcp.sessions),
                    (f"; unavailable: {', '.join(mcp.failed)}" if mcp.failed else ""))

        reply = ""
        reflected = False  # completion-check runs at most once per turn
        for step in range(max_steps):
            resp = await llm.chat(model=model, messages=messages,
                                  tools=tools, options=_OPTIONS)
            msg = resp.message
            calls = msg.tool_calls or []

            # reconstruct the assistant message (with any tool calls) into history
            asst: dict = {"role": "assistant", "content": msg.content or ""}
            if calls:
                asst["tool_calls"] = [
                    {"function": {"name": c.function.name,
                                  "arguments": dict(c.function.arguments or {})}}
                    for c in calls
                ]
            messages.append(asst)

            if not calls:
                reply = (msg.content or "").strip()
                # Completion check: small models sometimes stop after the first action
                # (or claim one they didn't take — e.g. "emailed you the link" when only
                # create_meet ran, since a Meet invite is itself an email). Once per turn,
                # on an action-bearing turn, ask the model to verify every requested
                # action is backed by a real successful tool call and finish anything
                # missing. This is generic (no per-intent hardcoding) — the model still
                # decides and makes any follow-up MCP call itself.
                if not reflected and _needs_completion_check(reply, trace):
                    reflected = True
                    await emit("Double-checking everything got done…")
                    messages.append({"role": "system",
                                     "content": _completion_check_prompt(trace)})
                    continue
                break

            # execute each tool call and feed results back
            for c in calls:
                name = c.function.name
                args = dict(c.function.arguments or {})
                await emit(_friendly_step(name, args))
                result = await mcp.call_tool(name, args)
                trace.append({"step": step, "tool": name, "args": args,
                              "readonly": mcp.is_readonly(name), "result": result})
                logger.info("  tool %s(%s) -> %s", name, args,
                            str(result)[:160])
                outcome = _result_step(name, result)
                if outcome:
                    await emit(outcome)
                messages.append({"role": "tool", "tool_name": name,
                                 "content": json.dumps(result, default=str)})
        else:
            # exhausted steps without a final plain answer
            reply = reply or "I wasn't able to complete that — too many tool steps."

    # Small models occasionally finish with empty text right after a forced follow-up
    # tool call. Never return a blank turn: synthesize a confirmation from the actions
    # that actually succeeded (falls back to a generic ack if nothing to report).
    if not reply.strip():
        outcomes = [s for s in (_result_step(t.get("tool"), t.get("result"))
                                for t in trace) if s and s.startswith("✓")]
        reply = " ".join(dict.fromkeys(outcomes)) if outcomes else "Done."

    wm.add_message(session_id, "assistant", reply, enqueue=False)
    return {"reply": reply, "session_id": session_id,
            "trace": trace, "steps": len(trace)}


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

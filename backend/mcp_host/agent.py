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
import extractor  # type: ignore  # noqa: E402

logger = logging.getLogger(__name__)

# A tool-calling model. Override with env ORBIX_AGENT_MODEL.
AGENT_MODEL = os.environ.get("ORBIX_AGENT_MODEL", "qwen2.5:3b")
MAX_STEPS = int(os.environ.get("ORBIX_AGENT_MAX_STEPS", "6"))
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
        "internals to the user."
    )


def _messages_from_session(session_id: str) -> list[dict]:
    """System prompt + recent transcript (working memory gives multi-turn context)."""
    msgs = [{"role": "system", "content": _system_prompt()}]
    for m in wm.get_messages(session_id, limit=20):
        role = m["role"] if m["role"] in ("user", "assistant", "system") else "user"
        msgs.append({"role": role, "content": m["text"]})
    return msgs


def _friendly_step(name: str, args: dict) -> str:
    """Human-readable 'thinking' label for a memory tool call (for the UI)."""
    if name == "search":
        return f"Searching memory for “{args.get('query', '')}”…"
    if name in ("recall", "get_entity"):
        return "Recalling what I know…"
    if name == "get_fact":
        return f"Looking up your {args.get('predicate', 'details')}…"
    if name == "fact_history":
        return "Checking the history…"
    return f"Using memory ({name})…"


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

    # FLUSH-BARRIER: persist this session's still-pending turns into Neo4j BEFORE we
    # answer the next query, so recall() sees the latest facts (diagram: "saving into
    # DB before LLM replies next query"). No-op on the first turn / when nothing pending.
    if wm.pending_count(session_id):
        await emit("Saving what you told me…")
        await asyncio.to_thread(extractor.drain, session_id=session_id)

    wm.add_message(session_id, "user", user_text)   # working memory (instant)

    model = model or AGENT_MODEL
    llm = AsyncClient()
    messages = _messages_from_session(session_id)
    trace: list[dict] = []

    async with MCPClientManager() as mcp:
        # READ-ONLY agent: the brain only retrieves; all writes go through the
        # background extractor (single write path -> no races, no duplication).
        #tools = mcp.ollama_tools()
        tools = [
            t for t in mcp.ollama_tools()
            if t["function"]["name"] in (
                "geocode_address",
                "find_ev_charging_stations",
                "find_nearby_places"
            )
        ]
        logger.info("Agent turn: model=%s, %d read tools available", model, len(tools))

        reply = ""
        for step in range(max_steps):
            resp = await llm.chat(model=model, messages=messages,
                                  tools=tools, options=_OPTIONS)
            msg = resp.message
            calls = msg.tool_calls or []

            print("\nSTEP", step)
            print("MODEL SAID:", msg.content)

            if calls:
                print("TOOL CALLS:")
                for c in calls:
                    print("  ", c.function.name, dict(c.function.arguments or {}))

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
                messages.append({"role": "tool", "tool_name": name,
                                 "content": json.dumps(result, default=str)})
        else:
            # exhausted steps without a final plain answer
            reply = reply or "I wasn't able to complete that — too many tool steps."
            
    # Generate short voice-friendly summary
    voice_reply = reply

    if len(reply) > 250:
        try:
            summary_resp = await llm.chat(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Summarize the response for text-to-speech. "
                            "Maximum 3 sentences. "
                            "Maximum 50 words. "
                            "Keep only the most important information."
                        )
                    },
                    {
                        "role": "user",
                        "content": reply
                    }
                ],
                options=_OPTIONS
            )

            voice_reply = summary_resp.message.content.strip()

        except Exception:
            voice_reply = reply[:200]

    wm.add_message(session_id, "assistant", reply, enqueue=False)
    return {"reply": reply, "session_id": session_id, "voice_reply": voice_reply,
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
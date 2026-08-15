"""
OrbixAI Memory MCP server.

A thin Model Context Protocol server that exposes the long-term memory engine
(backend/graph/memory.py) as typed tools, so the agent brain reads and writes the
Neo4j knowledge graph through a safe, intent-named interface instead of raw Cypher.
All correctness (resolve -> reconcile -> upsert, current-only reads) lives in
memory.py; this file only maps functions to MCP tools and advertises read/write
hints so the host can confirm-gate writes (architecture §7.2-§7.4).

Tools
  read  (readOnlyHint):  recall · search · get_entity · get_fact · fact_history
                          search_transcript (raw SQLite session history, not a
                          graph fact — see its docstring)
  write (not readOnly) :  remember_fact · correct · upsert_entity · link
                          remember_episode · forget

The owner is always the singleton `user:self`. To write a fact about the user,
use subject_key="user:self". For other entities, resolve the key first with
`search` or `recall`.

Run (stdio transport — how an MCP host spawns it):
    python backend/mcp_servers/memory_server.py

Register it with a host via JSON (see orbix_mcp_config.example.json):
    { "mcpServers": { "orbix-memory": {
        "command": "python",
        "args": ["backend/mcp_servers/memory_server.py"] } } }
"""

import sys
from pathlib import Path

# memory.py lives in backend/graph/ and imports its siblings by bare name, so put
# that directory on the path and import it directly (same convention as the loader).
_GRAPH_DIR = Path(__file__).resolve().parents[1] / "graph"
sys.path.insert(0, str(_GRAPH_DIR))

import memory as mem  # type: ignore  # noqa: E402
import working_memory as wm  # type: ignore  # noqa: E402

# This server is spawned as its own OS subprocess (stdio transport), so it never
# shares the main backend's in-process init — but session.db is the same on-disk
# SQLite file, and init_db()'s CREATE TABLE IF NOT EXISTS is safe to call again
# from here (idempotent, WAL mode tolerates the concurrent access).
wm.init_db()

from mcp.server.fastmcp import FastMCP  # noqa: E402
from mcp.types import ToolAnnotations  # noqa: E402

mcp = FastMCP(
    "orbix-memory",
    instructions=(
        "OrbixAI long-term memory over a Neo4j knowledge graph. Before answering "
        "anything personal, call `recall('user:self')` or `search(...)` and answer "
        "ONLY from what you find. Before storing, the key recipes are handled for "
        "you — for a changeable user attribute use remember_fact('user:self', "
        "<predicate>, <value>); for things that happened use remember_episode. "
        "Never invent keys; resolve with search/recall first."
    ),
)

_READONLY = ToolAnnotations(readOnlyHint=True)
_WRITE = ToolAnnotations(readOnlyHint=False, idempotentHint=True)
_DESTRUCTIVE = ToolAnnotations(readOnlyHint=False, destructiveHint=True)


# --------------------------------------------------------------------------- #
# READ tools (auto-run — safe, no side effects)
# --------------------------------------------------------------------------- #
@mcp.tool(annotations=_READONLY)
def recall(subject_key: str) -> dict:
    """
    Resolve a subject and return its CURRENT view: live properties, current facts,
    and outgoing relationships. History is not included (use fact_history for that).
    Use subject_key='user:self' for the owner. Returns {found: false} if unknown.
    """
    res = mem.recall(subject_key)
    return res if res is not None else {"found": False, "key": subject_key}


@mcp.tool(annotations=_READONLY)
def search(query: str, limit: int = 10) -> list[dict]:
    """
    Keyword search over entity names/summaries/aliases (prefix + fuzzy). Returns
    the best-matching nodes as [{key, name, labels, score}]. Use this to resolve a
    name to a key before reading or writing.
    """
    return mem.search(query, limit=limit)


@mcp.tool(annotations=_READONLY)
def get_entity(key: str) -> dict:
    """Full node: live properties, current facts, relationships, AND recent
    observation history attached to it. Returns {found: false} if unknown."""
    res = mem.get_entity(key)
    return res if res is not None else {"found": False, "key": key}


@mcp.tool(annotations=_READONLY)
def get_fact(subject_key: str, predicate: str) -> dict:
    """Current value of one mutable fact, e.g. get_fact('user:self','home_city').
    Returns {found: false} if that fact has never been set."""
    res = mem.get_fact(subject_key, predicate)
    return res if res is not None else {"found": False,
                                        "subject_key": subject_key, "predicate": predicate}


@mcp.tool(annotations=_READONLY)
def fact_history(subject_key: str, predicate: str) -> dict:
    """The 'what did I used to...' read: current value plus archived past values.
    Only use when the user explicitly asks about the past."""
    return mem.fact_history(subject_key, predicate)


@mcp.tool(annotations=_READONLY)
def search_transcript(query: str, session_id: str, limit: int = 5) -> list[dict]:
    """
    Keyword search over THIS conversation's older messages — ones that have
    already scrolled out of your visible context window. Returns raw quotes
    ({role, text, ts}), NOT verified facts: present them as "earlier in this chat
    you/I said X," never as ground truth (use recall/search for that). Only call
    this for something specific you can't find in the visible transcript above or
    in "WHAT YOU ALREADY KNOW". The host fills in session_id automatically —
    pass through whatever value you were given for it unchanged.
    """
    return wm.search_messages(session_id, query, limit=limit)


# --------------------------------------------------------------------------- #
# WRITE tools (host should confirm before executing — §7.4)
# --------------------------------------------------------------------------- #
@mcp.tool(annotations=_WRITE)
def remember_fact(subject_key: str, predicate: str, value: str,
                  confidence: float = 0.9) -> dict:
    """
    Upsert a mutable single-valued fact (overwrites in place, archives the old
    value as history). Use for changeable attributes: home_city, employer,
    marital_status, favourite_colour, etc. Example:
    remember_fact('user:self','home_city','Pune').
    Returns {previous, current, changed}.
    """
    return mem.remember_fact(subject_key, predicate, value, confidence=confidence)


@mcp.tool(annotations=_WRITE)
def correct(subject_key: str, predicate: str, value: str) -> dict:
    """Explicit correction of a fact (high confidence). Same effect as
    remember_fact but signals the user is fixing a known-wrong value."""
    return mem.correct(subject_key, predicate, value)


@mcp.tool(annotations=_WRITE)
def upsert_entity(type: str, name: str, props: dict | None = None,
                  sublabels: list[str] | None = None) -> dict:
    """
    Create or update a typed node (deduped on its identity key). `type` is one of:
    person, organization, project, task, meeting, trip, location, topic, note,
    interest, preference, event, document, medication, reminder. `sublabels`
    refine it (e.g. type='person', sublabels=['Family']). Returns {key}.
    """
    key = mem.upsert_entity(type, name, props=props, sublabels=sublabels, source="agent")
    return {"key": key}


@mcp.tool(annotations=_WRITE)
def link(a_key: str, rel: str, b_key: str, props: dict | None = None) -> dict:
    """
    Create or strengthen a relationship a -[rel]-> b (never duplicated). `rel` is an
    UPPER_SNAKE type from the schema (KNOWS, WORKS_AT, WORKS_ON, PARENT_OF,
    SIBLING_OF, MENTIONS, ...). Both keys must already exist. Returns {ok}.
    """
    mem.link(a_key, rel, b_key, props=props)
    return {"ok": True, "edge": f"{a_key} -[:{rel}]-> {b_key}"}


@mcp.tool(annotations=_WRITE)
def remember_episode(text: str, links: list[str] | None = None) -> dict:
    """
    Append an episodic memory (something that happened — these accumulate and never
    contradict). Optionally pass `links`: keys of entities the episode mentions,
    which get a MENTIONS edge. Returns {key}.
    """
    key = mem.remember_episode(text, links=links, source="agent")
    return {"key": key}


@mcp.tool(annotations=_DESTRUCTIVE)
def forget(key: str) -> dict:
    """Soft-delete a node (mark archived + current:false; history is preserved).
    Returns {ok}."""
    return {"ok": mem.forget(key)}


if __name__ == "__main__":
    mcp.run()  # stdio transport

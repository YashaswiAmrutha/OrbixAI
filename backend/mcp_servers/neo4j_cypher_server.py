"""
OrbixAI Neo4j Cypher MCP server.

Power-user access to the same Neo4j graph the memory server uses, but through raw
Cypher: read queries, guarded write queries, and schema discovery. This complements
orbix-memory (which exposes safe, intent-named recall/remember tools) for the cases
those don't cover. Reuses the shared, cached driver from graph/connection.py, so it
needs the same NEO4J_* credentials in the root .env — nothing new to configure.

Tools
  read  (readOnlyHint):  run_cypher (write=false) · get_schema
  write (action)      :  run_cypher (write=true)

Run (stdio transport):
    python backend/mcp_servers/neo4j_cypher_server.py
"""

import sys
from pathlib import Path

# connection.py lives in backend/graph/ and imports siblings by bare name.
_GRAPH_DIR = Path(__file__).resolve().parents[1] / "graph"
sys.path.insert(0, str(_GRAPH_DIR))

import connection  # type: ignore  # noqa: E402

from mcp.server.fastmcp import FastMCP  # noqa: E402
from mcp.types import ToolAnnotations  # noqa: E402

mcp = FastMCP(
    "orbix-cypher",
    instructions=(
        "Raw Cypher access to the user's Neo4j graph. Use run_cypher(query) for reads "
        "and get_schema() to discover labels/relationship types/properties first. Only "
        "set write=true for queries that create/update/delete, and only when the user "
        "clearly asked to change the graph. Prefer the orbix-memory tools for ordinary "
        "recall/remember; use this for ad-hoc graph questions they don't cover."
    ),
)

_READONLY = ToolAnnotations(readOnlyHint=True)
_ACTION = ToolAnnotations(readOnlyHint=False, idempotentHint=False)

_WRITE_HINT = ("create", "merge", "delete", "set ", "remove", "drop", "detach")


@mcp.tool(annotations=_READONLY)
def get_schema() -> dict:
    """Return the graph's labels, relationship types and property keys."""
    try:
        drv = connection.get_driver()
        with drv.session() as s:
            labels = [r["label"] for r in s.run("CALL db.labels() YIELD label RETURN label")]
            rels = [r["relationshipType"] for r in
                    s.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")]
            props = [r["propertyKey"] for r in
                     s.run("CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey")]
        return {"labels": labels, "relationship_types": rels, "property_keys": props}
    except Exception as e:  # noqa: BLE001
        return {"error": f"schema query failed: {e}"}


@mcp.tool(annotations=_ACTION)
def run_cypher(query: str, params: dict | None = None, write: bool = False) -> dict:
    """
    Run a Cypher query. Keep write=false for reads (MATCH/RETURN); set write=true only
    for CREATE/MERGE/SET/DELETE. Returns {rows: [...], count} or {error: ...}. Pass
    query parameters as `params` (e.g. {"name": "Alice"}) — don't string-concatenate.
    """
    query = (query or "").strip()
    if not query:
        return {"error": "query is required"}
    looks_write = any(kw in query.lower() for kw in _WRITE_HINT)
    if looks_write and not write:
        return {"error": "this looks like a write query; call again with write=true to confirm"}
    try:
        drv = connection.get_driver()
        with drv.session() as s:
            result = s.run(query, params or {})
            rows = [dict(r) for r in result][:200]
        return {"rows": rows, "count": len(rows)}
    except Exception as e:  # noqa: BLE001
        return {"error": f"cypher failed: {e}"}


if __name__ == "__main__":
    mcp.run()  # stdio transport

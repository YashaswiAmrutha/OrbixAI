"""
MCP client manager — the OrbixAI host's connection to its MCP servers.

Spawns each configured server over stdio, initializes a session, aggregates every
server's tools into one namespace, and dispatches tool calls to the right server.
Tools are also exposed in **Ollama's tool-schema format** so the agent loop can hand
the live list to a tool-calling model on every request (architecture §7.1-§7.2).

Lifetime model: the manager is an async context manager. Enter it for the duration
of one agent turn (open -> use -> close) — this keeps the anyio stdio subprocesses
inside a single task/cancel scope, which is simple and robust. Connection pooling
across requests is a later optimization.

    async with MCPClientManager() as mcp:
        tools = mcp.ollama_tools()
        result = await mcp.call_tool("recall", {"subject_key": "user:self"})
"""

import sys
import json
import logging
from pathlib import Path
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MEMORY_SERVER = _REPO_ROOT / "backend" / "mcp_servers" / "memory_server.py"

# Default server roster. Use the current interpreter + absolute paths so spawning
# never depends on PATH or the working directory. Add gmail/calendar/web here later.
DEFAULT_SERVERS: dict[str, dict] = {
    "orbix-memory": {
        "command": sys.executable,
        "args": [str(_MEMORY_SERVER)],
    },
}


def load_servers(config_path: str | Path | None = None) -> dict[str, dict]:
    """Load a server roster from a JSON file ({"mcpServers": {...}}), or the default."""
    if config_path is None:
        return DEFAULT_SERVERS
    data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return data.get("mcpServers", data)


def _tool_result_to_python(result) -> object:
    """Turn an MCP CallToolResult into a plain Python value for the model/caller."""
    if getattr(result, "isError", False):
        text = "; ".join(c.text for c in result.content if getattr(c, "type", "") == "text")
        return {"error": text or "tool error"}
    # Prefer structured output; unwrap FastMCP's {"result": X} envelope for list/scalar returns.
    sc = getattr(result, "structuredContent", None)
    if sc is not None:
        if isinstance(sc, dict) and set(sc.keys()) == {"result"}:
            return sc["result"]
        return sc
    # Fall back to concatenated text content (often JSON).
    text = "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


class MCPClientManager:
    def __init__(self, servers: dict[str, dict] | None = None):
        self.servers = servers or DEFAULT_SERVERS
        self._stack = AsyncExitStack()
        self.sessions: dict[str, ClientSession] = {}
        # tool name -> (server_name, mcp.types.Tool)
        self.tool_index: dict[str, tuple[str, object]] = {}

    async def __aenter__(self) -> "MCPClientManager":
        for name, spec in self.servers.items():
            params = StdioServerParameters(
                command=spec["command"], args=spec.get("args", []),
                env=spec.get("env"),
            )
            read, write = await self._stack.enter_async_context(stdio_client(params))
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self.sessions[name] = session
            listed = await session.list_tools()
            for tool in listed.tools:
                if tool.name in self.tool_index:
                    logger.warning("Tool name collision %r (servers %s / %s)",
                                   tool.name, self.tool_index[tool.name][0], name)
                self.tool_index[tool.name] = (name, tool)
            logger.info("MCP server %r up: %d tools", name, len(listed.tools))
        return self

    async def __aexit__(self, *exc) -> None:
        await self._stack.aclose()

    # ---- tool surface ----------------------------------------------------- #
    def ollama_tools(self) -> list[dict]:
        """Every tool as an Ollama function-tool schema (the live list per request)."""
        out = []
        for tname, (_srv, tool) in self.tool_index.items():
            out.append({
                "type": "function",
                "function": {
                    "name": tname,
                    "description": (tool.description or "").strip(),
                    "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                },
            })
        return out

    def is_readonly(self, tool_name: str) -> bool:
        """Whether a tool advertised readOnlyHint (used for write confirm-gating)."""
        entry = self.tool_index.get(tool_name)
        if not entry:
            return False
        ann = getattr(entry[1], "annotations", None)
        return bool(ann and getattr(ann, "readOnlyHint", False))

    async def call_tool(self, name: str, arguments: dict | None = None) -> object:
        """Dispatch a tool call to its owning server; returns a plain Python value."""
        entry = self.tool_index.get(name)
        if not entry:
            return {"error": f"unknown tool: {name}"}
        server_name, _tool = entry
        try:
            result = await self.sessions[server_name].call_tool(name, arguments or {})
            return _tool_result_to_python(result)
        except Exception as e:  # noqa: BLE001
            logger.error("Tool %r failed: %s", name, e)
            return {"error": str(e)}

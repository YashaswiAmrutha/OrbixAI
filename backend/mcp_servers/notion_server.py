"""
OrbixAI Notion MCP server.

Search and read the user's Notion workspace via the Notion API. Auth is an internal
integration token pasted in the config (env ORBIX_NOTION_TOKEN) — no OAuth app needed;
the user creates an integration at notion.so/my-integrations and shares pages with it.
Without a token the tools return a clear message.

Tools (readOnlyHint):  notion_search · notion_get_page
  write (action)     :  notion_append_text

Run (stdio transport):
    python backend/mcp_servers/notion_server.py
"""

import os
import requests

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

mcp = FastMCP(
    "orbix-notion",
    instructions=(
        "Notion workspace access. notion_search(query) finds pages/databases the "
        "integration can see; notion_get_page(page_id) reads a page's text. Remind the "
        "user to share the relevant pages with the integration. If a tool says the token "
        "is missing, tell them to paste it in the Notion integration settings."
    ),
)

_READONLY = ToolAnnotations(readOnlyHint=True)
_ACTION = ToolAnnotations(readOnlyHint=False, idempotentHint=False)
_API = "https://api.notion.com/v1"
_VER = "2022-06-28"
_TIMEOUT = 20
_NO_TOKEN = {"error": "no Notion token configured — paste an integration token in the Notion settings"}


def _headers() -> dict | None:
    token = os.environ.get("ORBIX_NOTION_TOKEN", "").strip()
    if not token:
        return None
    return {"Authorization": f"Bearer {token}", "Notion-Version": _VER,
            "Content-Type": "application/json"}


def _title_of(obj: dict) -> str:
    props = obj.get("properties", {})
    for v in props.values():
        if v.get("type") == "title":
            return "".join(t.get("plain_text", "") for t in v.get("title", [])) or "(untitled)"
    # databases carry their title at the top level
    if obj.get("title"):
        return "".join(t.get("plain_text", "") for t in obj["title"]) or "(untitled)"
    return "(untitled)"


@mcp.tool(annotations=_READONLY)
def notion_search(query: str, limit: int = 8) -> dict:
    """
    Search pages/databases shared with the integration. Returns
    {results:[{id,title,type,url}]}. Each result's `url` is the actual clickable
    Notion link — when you tell the user about a page, include its real url
    value in your reply (e.g. "OrbixAI — https://notion.so/..."). Never say
    "click the respective URLs" or similar without actually stating them; the
    user can't click a link you didn't write out.
    """
    h = _headers()
    if h is None:
        return _NO_TOKEN
    try:
        r = requests.post(f"{_API}/search", headers=h,
                          json={"query": query, "page_size": max(1, min(int(limit or 8), 20))},
                          timeout=_TIMEOUT)
        if r.status_code >= 400:
            return {"error": f"Notion {r.status_code}: {r.text[:200]}"}
        data = r.json()
    except Exception as e:  # noqa: BLE001
        return {"error": f"request failed: {e}"}
    out = [{"id": o["id"], "title": _title_of(o), "type": o.get("object"),
            "url": o.get("url", "")} for o in data.get("results", [])]
    result = {"results": out, "count": len(out)}
    if not out:
        # Notion's Search API only ever returns pages/databases EXPLICITLY shared
        # with this integration (via a page's "..." -> "Connect to" menu) — a
        # valid token grants zero access on its own. A bare {results: [], count: 0}
        # reads to the model (and then the user) as "you have no pages," which is
        # misleading in the common case where pages exist but were never shared.
        # This hint travels with the result so the model can explain the real
        # reason instead of asserting the workspace is empty.
        result["hint"] = ("No results — in Notion, an integration only sees pages "
                          "that have been explicitly shared with it (open the page "
                          "-> '...' menu -> 'Connect to' -> select this integration). "
                          "This is a Notion access-control requirement, not proof the "
                          "workspace is actually empty — tell the user to share the "
                          "relevant pages with the integration if they expected results.")
    return result


@mcp.tool(annotations=_READONLY)
def notion_get_page(page_id: str) -> dict:
    """Read a page's block text. Returns {id, text} or {error}."""
    h = _headers()
    if h is None:
        return _NO_TOKEN
    if not (page_id or "").strip():
        return {"error": "page_id is required"}
    try:
        r = requests.get(f"{_API}/blocks/{page_id}/children", headers=h,
                        params={"page_size": 100}, timeout=_TIMEOUT)
        if r.status_code >= 400:
            return {"error": f"Notion {r.status_code}: {r.text[:200]}"}
        blocks = r.json().get("results", [])
    except Exception as e:  # noqa: BLE001
        return {"error": f"request failed: {e}"}
    lines = []
    for b in blocks:
        bt = b.get("type", "")
        rich = b.get(bt, {}).get("rich_text", []) if isinstance(b.get(bt), dict) else []
        text = "".join(t.get("plain_text", "") for t in rich)
        if text:
            lines.append(text)
    return {"id": page_id, "text": "\n".join(lines)[:6000]}


@mcp.tool(annotations=_ACTION)
def notion_append_text(page_id: str, text: str) -> dict:
    """Append a paragraph of text to a page. Returns {appended: true} or {error}."""
    h = _headers()
    if h is None:
        return _NO_TOKEN
    if not page_id or not text:
        return {"error": "page_id and text are required"}
    payload = {"children": [{"object": "block", "type": "paragraph",
               "paragraph": {"rich_text": [{"type": "text", "text": {"content": text}}]}}]}
    try:
        r = requests.patch(f"{_API}/blocks/{page_id}/children", headers=h, json=payload,
                          timeout=_TIMEOUT)
        if r.status_code >= 400:
            return {"error": f"Notion {r.status_code}: {r.text[:200]}"}
    except Exception as e:  # noqa: BLE001
        return {"error": f"request failed: {e}"}
    return {"appended": True, "page_id": page_id}


if __name__ == "__main__":
    mcp.run()  # stdio transport

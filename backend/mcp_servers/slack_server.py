"""
OrbixAI Slack MCP server.

Read channels and post messages in a Slack workspace via the Web API. Auth is a bot
token (xoxb-…) pasted in the config (env ORBIX_SLACK_BOT_TOKEN) — create a Slack app,
add the bot scopes (channels:read, chat:write, channels:history) and install it. Without
a token the tools return a clear message.

Tools
  read  (readOnlyHint):  slack_list_channels · slack_read_channel
  write (action)      :  slack_post_message

Run (stdio transport):
    python backend/mcp_servers/slack_server.py
"""

import os
import requests

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

mcp = FastMCP(
    "orbix-slack",
    instructions=(
        "Slack workspace access via a bot token. slack_list_channels() to find a channel "
        "id, slack_read_channel(channel) to read recent messages, slack_post_message("
        "channel, text) to post. Only post when the user asks. If a tool says the token "
        "is missing, tell them to paste a bot token in the Slack integration settings."
    ),
)

_READONLY = ToolAnnotations(readOnlyHint=True)
_ACTION = ToolAnnotations(readOnlyHint=False, idempotentHint=False)
_API = "https://slack.com/api"
_TIMEOUT = 20
_NO_TOKEN = {"error": "no Slack token configured — paste a bot token in the Slack integration settings"}


def _token() -> str:
    return os.environ.get("ORBIX_SLACK_BOT_TOKEN", "").strip()


def _call(method: str, params: dict, post: bool = False) -> dict:
    token = _token()
    if not token:
        return _NO_TOKEN
    headers = {"Authorization": f"Bearer {token}"}
    try:
        if post:
            r = requests.post(f"{_API}/{method}", headers=headers, json=params, timeout=_TIMEOUT)
        else:
            r = requests.get(f"{_API}/{method}", headers=headers, params=params, timeout=_TIMEOUT)
        data = r.json()
    except Exception as e:  # noqa: BLE001
        return {"error": f"request failed: {e}"}
    if not data.get("ok"):
        return {"error": f"Slack: {data.get('error', 'unknown error')}"}
    return data


@mcp.tool(annotations=_READONLY)
def slack_list_channels(limit: int = 20) -> dict:
    """List channels the bot can see. Returns {channels: [{id, name}]}."""
    data = _call("conversations.list", {"limit": max(1, min(int(limit or 20), 100)),
                                        "types": "public_channel,private_channel"})
    if data.get("error"):
        return data
    return {"channels": [{"id": c["id"], "name": c.get("name")} for c in data.get("channels", [])]}


@mcp.tool(annotations=_READONLY)
def slack_read_channel(channel: str, limit: int = 15) -> dict:
    """Read recent messages from a channel id. Returns {messages: [{user, text, ts}]}."""
    if not (channel or "").strip():
        return {"error": "channel id is required"}
    data = _call("conversations.history", {"channel": channel,
                                           "limit": max(1, min(int(limit or 15), 50))})
    if data.get("error"):
        return data
    msgs = [{"user": m.get("user"), "text": m.get("text", ""), "ts": m.get("ts")}
            for m in data.get("messages", [])]
    return {"messages": msgs, "count": len(msgs)}


@mcp.tool(annotations=_ACTION)
def slack_post_message(channel: str, text: str) -> dict:
    """Post a message to a channel id. Returns {posted: true, ts} or {error}."""
    if not channel or not text:
        return {"error": "channel and text are required"}
    data = _call("chat.postMessage", {"channel": channel, "text": text}, post=True)
    if data.get("error"):
        return data
    return {"posted": True, "channel": channel, "ts": data.get("ts")}


if __name__ == "__main__":
    mcp.run()  # stdio transport

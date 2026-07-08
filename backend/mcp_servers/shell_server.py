"""
OrbixAI Shell MCP server.  ⚠ HIGH RISK — off by default.

Runs shell commands on the machine and returns their output. This is powerful and
inherently risky (a bad command can damage the system), so it ships DISABLED and must
be turned on explicitly from the Integrations page. As a guardrail it refuses a small
blocklist of obviously catastrophic patterns and enforces a timeout, but it is NOT a
sandbox — only enable it if you understand the risk.

Tool (action):  run_command

Run (stdio transport):
    python backend/mcp_servers/shell_server.py
"""

import os
import re
import subprocess

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

mcp = FastMCP(
    "orbix-shell",
    instructions=(
        "Run shell commands on the user's machine. HIGH RISK — only run a command the "
        "user clearly asked for, prefer read-only/inspection commands, and never run "
        "anything destructive (deleting data, formatting, shutting down) unless the user "
        "explicitly requested exactly that. Report the command's output plainly."
    ),
)

_ACTION = ToolAnnotations(readOnlyHint=False, destructiveHint=True)
_TIMEOUT = 30

# Refuse a few unambiguously catastrophic patterns outright. This is a seatbelt, not a
# sandbox — it does not make arbitrary command execution safe.
_BLOCKED = [
    r"\brm\s+-rf\s+[/~]", r"\bmkfs\b", r"\b:\(\)\s*\{", r"\bdd\s+if=",
    r"format\s+[a-z]:", r"\bdel\s+/[fsq]", r"\brmdir\s+/s", r"\bshutdown\b",
    r"\breboot\b", r"\bdiskpart\b", r">\s*/dev/sd", r"\bchmod\s+-R\s+777\s+/",
]


@mcp.tool(annotations=_ACTION)
def run_command(command: str, workdir: str = "") -> dict:
    """
    Execute a shell command and return {command, exit_code, stdout, stderr}. Prefer safe,
    read-only commands. Refuses obviously destructive commands and times out after 30s.
    `workdir` optionally sets the working directory (defaults to the configured one).
    """
    cmd = (command or "").strip()
    if not cmd:
        return {"error": "command is required"}
    for pat in _BLOCKED:
        if re.search(pat, cmd, re.IGNORECASE):
            return {"error": "refused: that command matches a blocked destructive pattern"}
    cwd = (workdir or os.environ.get("ORBIX_SHELL_WORKDIR", "") or None)
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=_TIMEOUT, cwd=cwd)
    except subprocess.TimeoutExpired:
        return {"error": f"command timed out after {_TIMEOUT}s"}
    except Exception as e:  # noqa: BLE001
        return {"error": f"failed to run: {e}"}
    return {
        "command": cmd,
        "exit_code": proc.returncode,
        "stdout": (proc.stdout or "")[:6000],
        "stderr": (proc.stderr or "")[:2000],
    }


if __name__ == "__main__":
    mcp.run()  # stdio transport

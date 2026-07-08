"""
OrbixAI Git (local) MCP server.

Read and search local git repositories the user allows. Scoped to repo folders set in
the integration config (env ORBIX_GIT_ROOTS, comma-separated) — every repo path is
validated against those roots before running git, so the assistant can inspect your
projects without roaming the whole disk. Read-only by design (status/log/diff/grep/
branches); it never commits or pushes.

Tools (all readOnlyHint):  git_status · git_log · git_diff · git_grep · git_branches

Run (stdio transport):
    python backend/mcp_servers/git_server.py
"""

import os
import subprocess
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

mcp = FastMCP(
    "orbix-git",
    instructions=(
        "Inspect the user's local git repositories (read-only). Use git_status/git_log/"
        "git_diff/git_grep/git_branches on a repo path inside the allowed roots. If a "
        "call errors about allowed folders, tell the user to add a repo folder in the "
        "Git integration settings."
    ),
)

_READONLY = ToolAnnotations(readOnlyHint=True)
_TIMEOUT = 20


def _roots() -> list[Path]:
    raw = os.environ.get("ORBIX_GIT_ROOTS", "").strip()
    out = []
    for part in raw.split(","):
        part = part.strip().strip('"')
        if part:
            try:
                out.append(Path(part).expanduser().resolve())
            except Exception:  # noqa: BLE001
                pass
    return out


def _repo(path: str) -> tuple[Path | None, str | None]:
    roots = _roots()
    if not roots:
        return None, ("No repo folders configured. Add one in the Git integration "
                      "settings first.")
    p = (path or "").strip().strip('"')
    if not p:
        # default to the first root if the model didn't specify a repo
        p = str(roots[0])
    try:
        rp = Path(p).expanduser().resolve()
    except Exception as e:  # noqa: BLE001
        return None, f"invalid path: {e}"
    for root in roots:
        try:
            rp.relative_to(root)
            return rp, None
        except ValueError:
            continue
    return None, f"'{rp}' is outside your allowed git folders"


def _git(repo: Path, *args: str) -> dict:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True, text=True, timeout=_TIMEOUT,
        )
    except FileNotFoundError:
        return {"error": "git is not installed or not on PATH"}
    except subprocess.TimeoutExpired:
        return {"error": "git command timed out"}
    if proc.returncode != 0:
        return {"error": (proc.stderr or proc.stdout or "git error").strip()[:500]}
    return {"output": proc.stdout.strip()[:6000]}


@mcp.tool(annotations=_READONLY)
def git_status(repo: str = "") -> dict:
    """Show `git status` for a repo inside an allowed folder."""
    rp, err = _repo(repo)
    return {"error": err} if err else {"repo": str(rp), **_git(rp, "status", "-sb")}


@mcp.tool(annotations=_READONLY)
def git_log(repo: str = "", limit: int = 15) -> dict:
    """Show the last N commits (oneline) for a repo."""
    rp, err = _repo(repo)
    if err:
        return {"error": err}
    n = max(1, min(int(limit or 15), 100))
    return {"repo": str(rp), **_git(rp, "log", f"-{n}", "--oneline", "--decorate")}


@mcp.tool(annotations=_READONLY)
def git_diff(repo: str = "", staged: bool = False) -> dict:
    """Show the working-tree diff (or the staged diff if staged=true)."""
    rp, err = _repo(repo)
    if err:
        return {"error": err}
    args = ["diff", "--stat", "-p"]
    if staged:
        args.insert(1, "--staged")
    return {"repo": str(rp), **_git(rp, *args)}


@mcp.tool(annotations=_READONLY)
def git_grep(query: str, repo: str = "") -> dict:
    """Search tracked files for `query` (git grep). Returns matching lines."""
    q = (query or "").strip()
    if not q:
        return {"error": "query is required"}
    rp, err = _repo(repo)
    if err:
        return {"error": err}
    return {"repo": str(rp), **_git(rp, "grep", "-n", "-I", q)}


@mcp.tool(annotations=_READONLY)
def git_branches(repo: str = "") -> dict:
    """List branches for a repo."""
    rp, err = _repo(repo)
    return {"error": err} if err else {"repo": str(rp), **_git(rp, "branch", "-a")}


if __name__ == "__main__":
    mcp.run()  # stdio transport

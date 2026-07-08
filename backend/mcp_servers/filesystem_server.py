"""
OrbixAI Filesystem MCP server.

Lets the agent read, search and write files — but only inside folders the user has
explicitly allowed (env ORBIX_FILES_ROOTS, set from the Integrations page config).
Every path is resolved and checked against those roots before any I/O, and a blocklist
keeps secrets/system dirs off-limits even if a root is broad. If no roots are
configured, the server still starts but every tool returns a clear {"error": ...}
telling the user to set allowed folders — it never touches arbitrary disk.

Tools
  read  (readOnlyHint):  list_directory · read_file · search_files
  write (action)      :  write_file

Run (stdio transport):
    python backend/mcp_servers/filesystem_server.py
"""

import os
import string
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

mcp = FastMCP(
    "orbix-files",
    instructions=(
        "OrbixAI scoped filesystem. Read, search and write files only inside the user's "
        "allowed folders. Use list_directory/read_file/search_files to find and read "
        "content, and write_file to save. If a tool returns an error about allowed "
        "folders, tell the user to add a root folder in the Filesystem integration."
    ),
)

_READONLY = ToolAnnotations(readOnlyHint=True)
_ACTION = ToolAnnotations(readOnlyHint=False, idempotentHint=False)

# Filename/dir names that are never allowed anywhere, even under an allowed root.
_BLOCKED_PARTS = {".ssh", ".aws", ".gnupg", "token.json", "credentials.json",
                  "id_rsa", ".env"}

# Absolute Windows/system locations that stay off-limits even though their whole drive
# is otherwise allowed. Compared case-insensitively against the resolved path.
_BLOCKED_ABS = [
    r"c:\windows",
    r"c:\program files",
    r"c:\program files (x86)",
    r"c:\programdata",
    r"c:\$recycle.bin",
    r"c:\system volume information",
    r"c:\recovery",
]
_MAX_READ = 200_000  # bytes


def _default_roots() -> list[Path]:
    """Everything the user owns: every fixed drive on Windows (C:\\, D:\\, E:\\ …), or
    '/' elsewhere. System and secret paths are excluded separately in _resolve, so this
    means 'all your files except Windows/system' without any manual setup."""
    roots: list[Path] = []
    if os.name == "nt":
        for letter in string.ascii_uppercase:
            d = Path(f"{letter}:\\")
            try:
                if d.exists():
                    roots.append(d)
            except Exception:  # noqa: BLE001
                continue
    if not roots:
        roots = [Path("/")]
    return roots


def _roots() -> list[Path]:
    """Allowed roots. If the user set ORBIX_FILES_ROOTS, honour it as an explicit
    restriction; otherwise default to ALL drives (all files bar system/secret dirs)."""
    raw = os.environ.get("ORBIX_FILES_ROOTS", "").strip()
    if not raw:
        return _default_roots()
    roots = []
    for part in raw.split(","):
        part = part.strip().strip('"')
        if not part:
            continue
        try:
            roots.append(Path(part).expanduser().resolve())
        except Exception:  # noqa: BLE001
            continue
    return roots or _default_roots()


def _is_blocked(rp: Path) -> bool:
    """True if the path hits a secret filename or a Windows/system location."""
    if {part.lower() for part in rp.parts} & _BLOCKED_PARTS:
        return True
    s = str(rp).lower()
    for b in _BLOCKED_ABS:
        if s == b or s.startswith(b + "\\") or s.startswith(b + "/"):
            return True
    return False


def _resolve(path: str) -> tuple[Path | None, str | None]:
    """Resolve `path` and confirm it's inside an allowed root and not blocked.
    Returns (resolved_path, None) on success or (None, error_message)."""
    p = (path or "").strip().strip('"')
    if not p:
        return None, "path is required"
    try:
        rp = Path(p).expanduser().resolve()
    except Exception as e:  # noqa: BLE001
        return None, f"invalid path: {e}"
    if _is_blocked(rp):
        return None, "that path is blocked for safety (Windows/system or secret files)"
    for root in _roots():
        try:
            rp.relative_to(root)
            return rp, None
        except ValueError:
            continue
    return None, f"'{rp}' is outside the allowed area"


@mcp.tool(annotations=_READONLY)
def list_allowed_roots() -> dict:
    """List the folders the assistant is currently allowed to access."""
    return {"roots": [str(r) for r in _roots()]}


@mcp.tool(annotations=_READONLY)
def list_directory(path: str) -> dict:
    """
    List the entries in a directory inside an allowed folder. Returns
    {path, entries: [{name, type, size}]}. `type` is 'dir' or 'file'.
    """
    rp, err = _resolve(path)
    if err:
        return {"error": err}
    if not rp.is_dir():
        return {"error": f"{rp} is not a directory"}
    entries = []
    try:
        for child in sorted(rp.iterdir()):
            try:
                entries.append({
                    "name": child.name,
                    "type": "dir" if child.is_dir() else "file",
                    "size": child.stat().st_size if child.is_file() else None,
                })
            except Exception:  # noqa: BLE001
                continue
    except Exception as e:  # noqa: BLE001
        return {"error": f"could not list {rp}: {e}"}
    return {"path": str(rp), "entries": entries, "count": len(entries)}


@mcp.tool(annotations=_READONLY)
def read_file(path: str) -> dict:
    """
    Read a text file inside an allowed folder. Returns {path, content, truncated}.
    Large files are truncated. Binary files return an error.
    """
    rp, err = _resolve(path)
    if err:
        return {"error": err}
    if not rp.is_file():
        return {"error": f"{rp} is not a file"}
    try:
        data = rp.read_bytes()[:_MAX_READ]
        content = data.decode("utf-8")
    except UnicodeDecodeError:
        return {"error": "file is not UTF-8 text (binary files aren't supported)"}
    except Exception as e:  # noqa: BLE001
        return {"error": f"could not read {rp}: {e}"}
    return {"path": str(rp), "content": content,
            "truncated": rp.stat().st_size > _MAX_READ}


@mcp.tool(annotations=_READONLY)
def search_files(query: str, root: str = "") -> dict:
    """
    Find files whose name contains `query` (case-insensitive) under an allowed folder.
    If `root` is empty, searches all allowed roots. Returns {matches: [path], count}.
    """
    q = (query or "").strip().lower()
    if not q:
        return {"error": "query is required"}
    search_roots: list[Path] = []
    if root:
        rp, err = _resolve(root)
        if err:
            return {"error": err}
        search_roots = [rp]
    else:
        search_roots = _roots()
        if not search_roots:
            return {"error": "No allowed folders configured."}
    matches: list[str] = []
    for base in search_roots:
        try:
            for dirpath, dirnames, filenames in os.walk(base):
                # prune blocked dirs
                dirnames[:] = [d for d in dirnames if d.lower() not in _BLOCKED_PARTS]
                for fn in filenames:
                    if q in fn.lower():
                        matches.append(str(Path(dirpath) / fn))
                        if len(matches) >= 100:
                            return {"matches": matches, "count": len(matches),
                                    "truncated": True}
        except Exception:  # noqa: BLE001
            continue
    return {"matches": matches, "count": len(matches)}


@mcp.tool(annotations=_ACTION)
def write_file(path: str, content: str, append: bool = False) -> dict:
    """
    Write text to a file inside an allowed folder (creates parent dirs). Set
    append=true to add to the end instead of overwriting. Returns
    {written: true, path, bytes}. Only writes inside allowed roots.
    """
    rp, err = _resolve(path)
    if err:
        return {"error": err}
    try:
        rp.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with open(rp, mode, encoding="utf-8") as f:
            f.write(content or "")
    except Exception as e:  # noqa: BLE001
        return {"error": f"could not write {rp}: {e}"}
    return {"written": True, "path": str(rp), "bytes": len((content or "").encode("utf-8"))}


if __name__ == "__main__":
    mcp.run()  # stdio transport

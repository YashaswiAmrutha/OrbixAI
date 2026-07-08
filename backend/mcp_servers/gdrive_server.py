"""
OrbixAI Google Drive MCP server.

Search and read the user's Google Drive, reusing the SAME Google sign-in as Gmail
(backend/token.json). Drive needs an extra scope, so after enabling this the user must
re-connect Google once (click Connect with Google) to grant Drive access — until then
the tools return a clear needs_auth message instead of failing.

Tools (all readOnlyHint):  drive_search · drive_read_file · drive_list_recent

Run (stdio transport):
    python backend/mcp_servers/gdrive_server.py
"""

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

_BACKEND = Path(__file__).resolve().parents[1]
_TOKEN = str(_BACKEND / "token.json")

# The scopes the shared Google sign-in requests (kept in sync with gmail_client.SCOPES).
_SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/drive.readonly",
]

os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")

mcp = FastMCP(
    "orbix-gdrive",
    instructions=(
        "Google Drive access (read-only). drive_search(query) finds files, "
        "drive_read_file(file_id) reads a document's text, drive_list_recent() lists "
        "recent files. If a tool returns needs_auth, tell the user to click Connect with "
        "Google on the Drive integration to grant Drive access."
    ),
)

_READONLY = ToolAnnotations(readOnlyHint=True)
_NEEDS_AUTH = {"error": "needs_auth", "auth_url": "/auth/login",
               "message": "Drive access not granted. Ask the user to re-connect Google "
                          "(Connect with Google) to enable Drive."}

_service = None


def _drive():
    """Build a Drive service from the shared token, or None if Drive scope isn't granted."""
    global _service
    if _service is not None:
        return _service
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
    except Exception:  # noqa: BLE001
        return None
    if not os.path.exists(_TOKEN):
        return None
    try:
        creds = Credentials.from_authorized_user_file(_TOKEN, _SCOPES)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        # Confirm Drive scope is actually present before trusting the service
        granted = set(getattr(creds, "scopes", None) or [])
        if not any("drive" in s for s in granted):
            return None
        _service = build("drive", "v3", credentials=creds)
        return _service
    except Exception:  # noqa: BLE001
        return None


@mcp.tool(annotations=_READONLY)
def drive_search(query: str, limit: int = 10) -> dict:
    """Search Drive by file name / full text. Returns {files: [{id, name, type, url}]}."""
    svc = _drive()
    if svc is None:
        return _NEEDS_AUTH
    q = (query or "").strip().replace("'", "\\'")
    if not q:
        return {"error": "query is required"}
    try:
        res = svc.files().list(
            q=f"fullText contains '{q}' or name contains '{q}'",
            pageSize=max(1, min(int(limit or 10), 25)),
            fields="files(id,name,mimeType,webViewLink)",
        ).execute()
    except Exception as e:  # noqa: BLE001
        return {"error": f"drive search failed: {e}"}
    files = [{"id": f["id"], "name": f["name"], "type": f.get("mimeType", ""),
              "url": f.get("webViewLink", "")} for f in res.get("files", [])]
    return {"files": files, "count": len(files)}


@mcp.tool(annotations=_READONLY)
def drive_list_recent(limit: int = 10) -> dict:
    """List the most recently modified files. Returns {files: [{id, name, type, url}]}."""
    svc = _drive()
    if svc is None:
        return _NEEDS_AUTH
    try:
        res = svc.files().list(
            orderBy="modifiedTime desc",
            pageSize=max(1, min(int(limit or 10), 25)),
            fields="files(id,name,mimeType,webViewLink)",
        ).execute()
    except Exception as e:  # noqa: BLE001
        return {"error": f"drive list failed: {e}"}
    files = [{"id": f["id"], "name": f["name"], "type": f.get("mimeType", ""),
              "url": f.get("webViewLink", "")} for f in res.get("files", [])]
    return {"files": files, "count": len(files)}


@mcp.tool(annotations=_READONLY)
def drive_read_file(file_id: str) -> dict:
    """
    Read a file's text. Google Docs are exported as plain text; other text files are
    downloaded directly. Returns {id, text, truncated} or {error}.
    """
    svc = _drive()
    if svc is None:
        return _NEEDS_AUTH
    if not (file_id or "").strip():
        return {"error": "file_id is required"}
    try:
        meta = svc.files().get(fileId=file_id, fields="mimeType,name").execute()
        mime = meta.get("mimeType", "")
        if mime == "application/vnd.google-apps.document":
            data = svc.files().export(fileId=file_id, mimeType="text/plain").execute()
        elif mime.startswith("text/") or mime in ("application/json",):
            data = svc.files().get_media(fileId=file_id).execute()
        else:
            return {"error": f"can't read '{meta.get('name')}' as text (type {mime})"}
        text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else str(data)
    except Exception as e:  # noqa: BLE001
        return {"error": f"drive read failed: {e}"}
    return {"id": file_id, "text": text[:6000], "truncated": len(text) > 6000}


if __name__ == "__main__":
    mcp.run()  # stdio transport

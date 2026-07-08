"""
OrbixAI GitHub MCP server.

Work with GitHub — search repos/code, read and open issues, inspect a repo — via the
GitHub REST API. Auth is a Personal Access Token pasted in the integration config
(env ORBIX_GITHUB_TOKEN). Without a token the tools return a clear message instead of
failing, so enabling the server never breaks a turn.

Tools
  read  (readOnlyHint):  gh_whoami · gh_search_repos · gh_get_repo · gh_list_issues · gh_search_code
  write (action)      :  gh_create_issue

Run (stdio transport):
    python backend/mcp_servers/github_server.py
"""

import os
import requests

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

mcp = FastMCP(
    "orbix-github",
    instructions=(
        "GitHub access via a personal access token. Search repos/code, read repo info "
        "and issues, and open issues when asked. `repo` arguments are 'owner/name'. If a "
        "tool says the token is missing, tell the user to paste a PAT in the GitHub "
        "integration settings."
    ),
)

_READONLY = ToolAnnotations(readOnlyHint=True)
_ACTION = ToolAnnotations(readOnlyHint=False, idempotentHint=False)

_API = "https://api.github.com"
_TIMEOUT = 20
_NO_TOKEN = {"error": "no GitHub token configured — paste a PAT in the GitHub integration settings"}


def _headers() -> dict | None:
    token = os.environ.get("ORBIX_GITHUB_TOKEN", "").strip()
    if not token:
        return None
    return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"}


def _get(path: str, params: dict | None = None) -> dict | list:
    h = _headers()
    if h is None:
        return _NO_TOKEN
    try:
        r = requests.get(f"{_API}{path}", headers=h, params=params or {}, timeout=_TIMEOUT)
        if r.status_code >= 400:
            return {"error": f"GitHub {r.status_code}: {r.json().get('message', r.text)[:200]}"}
        return r.json()
    except Exception as e:  # noqa: BLE001
        return {"error": f"request failed: {e}"}


@mcp.tool(annotations=_READONLY)
def gh_whoami() -> dict:
    """Return the authenticated GitHub user {login, name}."""
    data = _get("/user")
    if isinstance(data, dict) and data.get("error"):
        return data
    return {"login": data.get("login"), "name": data.get("name")}


@mcp.tool(annotations=_READONLY)
def gh_search_repos(query: str, limit: int = 5) -> dict:
    """Search repositories. Returns {repos: [{full_name, description, stars, url}]}."""
    data = _get("/search/repositories", {"q": query, "per_page": max(1, min(int(limit or 5), 20))})
    if isinstance(data, dict) and data.get("error"):
        return data
    items = data.get("items", []) if isinstance(data, dict) else []
    return {"repos": [{"full_name": i["full_name"], "description": i.get("description"),
                       "stars": i.get("stargazers_count"), "url": i["html_url"]} for i in items]}


@mcp.tool(annotations=_READONLY)
def gh_get_repo(repo: str) -> dict:
    """Get repo info for 'owner/name': {full_name, description, stars, language, url, open_issues}."""
    data = _get(f"/repos/{repo}")
    if isinstance(data, dict) and data.get("error"):
        return data
    return {"full_name": data.get("full_name"), "description": data.get("description"),
            "stars": data.get("stargazers_count"), "language": data.get("language"),
            "open_issues": data.get("open_issues_count"), "url": data.get("html_url")}


@mcp.tool(annotations=_READONLY)
def gh_list_issues(repo: str, limit: int = 10) -> dict:
    """List open issues for 'owner/name'. Returns {issues: [{number, title, url}]}."""
    data = _get(f"/repos/{repo}/issues", {"state": "open", "per_page": max(1, min(int(limit or 10), 30))})
    if isinstance(data, dict) and data.get("error"):
        return data
    issues = [i for i in data if "pull_request" not in i] if isinstance(data, list) else []
    return {"issues": [{"number": i["number"], "title": i["title"], "url": i["html_url"]} for i in issues]}


@mcp.tool(annotations=_READONLY)
def gh_search_code(query: str, repo: str = "", limit: int = 5) -> dict:
    """Search code (optionally within 'owner/name'). Returns {results: [{path, repo, url}]}."""
    q = query + (f" repo:{repo}" if repo else "")
    data = _get("/search/code", {"q": q, "per_page": max(1, min(int(limit or 5), 20))})
    if isinstance(data, dict) and data.get("error"):
        return data
    items = data.get("items", []) if isinstance(data, dict) else []
    return {"results": [{"path": i["path"], "repo": i["repository"]["full_name"],
                         "url": i["html_url"]} for i in items]}


@mcp.tool(annotations=_ACTION)
def gh_create_issue(repo: str, title: str, body: str = "") -> dict:
    """Open an issue on 'owner/name'. Returns {created, number, url} or {error}."""
    h = _headers()
    if h is None:
        return _NO_TOKEN
    if not repo or not title:
        return {"error": "repo ('owner/name') and title are required"}
    try:
        r = requests.post(f"{_API}/repos/{repo}/issues", headers=h,
                          json={"title": title, "body": body}, timeout=_TIMEOUT)
        if r.status_code >= 400:
            return {"error": f"GitHub {r.status_code}: {r.json().get('message', r.text)[:200]}"}
        d = r.json()
        return {"created": True, "number": d["number"], "url": d["html_url"]}
    except Exception as e:  # noqa: BLE001
        return {"error": f"request failed: {e}"}


if __name__ == "__main__":
    mcp.run()  # stdio transport

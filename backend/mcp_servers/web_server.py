"""
OrbixAI Web MCP server.

Gives the agent brain eyes on the open web — read a page as clean text, or run a
live search — through two safe, read-only tools. No API key required: search uses
DuckDuckGo's keyless HTML endpoint, fetch uses plain HTTP. Everything degrades to a
clear {"error": ...} instead of raising, so a network hiccup never breaks a turn.

Tools
  read (readOnlyHint):  fetch_url · web_search

Run (stdio transport — how the host spawns it):
    python backend/mcp_servers/web_server.py
"""

import re
import html as _html

import requests
from bs4 import BeautifulSoup

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

mcp = FastMCP(
    "orbix-web",
    instructions=(
        "OrbixAI web access. Use `web_search(query)` to find current information and "
        "`fetch_url(url)` to read a specific page as text. Both are read-only and safe "
        "to call whenever you need facts you don't already know. Summarise what you find "
        "in your own words and cite the source URL."
    ),
)

_READONLY = ToolAnnotations(readOnlyHint=True)

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) OrbixAI/1.0 "
                   "(personal assistant)"),
    "Accept-Language": "en-US,en;q=0.9",
}
_TIMEOUT = 15
_MAX_CHARS = 6000  # keep tool output small enough for a local model's context


def _clean_html(raw: str) -> str:
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "svg"]):
        tag.decompose()
    text = soup.get_text("\n")
    # collapse blank runs
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip()


@mcp.tool(annotations=_READONLY)
def fetch_url(url: str) -> dict:
    """
    Fetch a web page and return it as clean, readable text (scripts/markup stripped).
    Returns {url, title, text, truncated}. Use this to read an article, doc or any
    page the user references. Returns {error: ...} on a bad URL or network failure.
    """
    url = (url or "").strip()
    if not url:
        return {"error": "url is required"}
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:  # noqa: BLE001
        return {"error": f"could not fetch {url}: {e}"}

    ctype = resp.headers.get("Content-Type", "")
    if "html" in ctype or "<html" in resp.text[:500].lower():
        soup = BeautifulSoup(resp.text, "html.parser")
        title = (soup.title.string.strip() if soup.title and soup.title.string else url)
        text = _clean_html(resp.text)
    else:
        title = url
        text = resp.text
    truncated = len(text) > _MAX_CHARS
    return {"url": url, "title": title, "text": text[:_MAX_CHARS], "truncated": truncated}


@mcp.tool(annotations=_READONLY)
def web_search(query: str, max_results: int = 6) -> dict:
    """
    Run a live web search (keyless DuckDuckGo) and return the top results as
    {query, results: [{title, url, snippet}], count}. Use this to answer questions
    about current events or anything you're unsure of, then optionally fetch_url the
    most relevant result. Returns {error: ...} if the search can't be reached.
    """
    query = (query or "").strip()
    if not query:
        return {"error": "query is required"}
    try:
        resp = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query}, headers=_HEADERS, timeout=_TIMEOUT,
        )
        resp.raise_for_status()
    except Exception as e:  # noqa: BLE001
        return {"error": f"search failed: {e}"}

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    for res in soup.select(".result")[: max(1, min(int(max_results or 6), 15))]:
        a = res.select_one(".result__a")
        if not a:
            continue
        title = a.get_text(" ", strip=True)
        href = a.get("href", "")
        # DDG wraps links in a redirect; pull out the real target if present
        m = re.search(r"uddg=([^&]+)", href)
        if m:
            from urllib.parse import unquote
            href = unquote(m.group(1))
        snip_el = res.select_one(".result__snippet")
        snippet = _html.unescape(snip_el.get_text(" ", strip=True)) if snip_el else ""
        if title and href:
            results.append({"title": title, "url": href, "snippet": snippet})
    return {"query": query, "results": results, "count": len(results)}


if __name__ == "__main__":
    mcp.run()  # stdio transport

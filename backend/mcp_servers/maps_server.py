"""
OrbixAI Maps / Places MCP server.

Location intelligence via the Google Maps Platform: geocode an address, find nearby
places, and get directions/distance. Auth is a Google Maps API key pasted in the
integration config (env ORBIX_MAPS_API_KEY). Without a key the tools return a clear
message rather than failing.

Tools (all readOnlyHint):  geocode · search_places · directions

Run (stdio transport):
    python backend/mcp_servers/maps_server.py
"""

import os
import requests

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

mcp = FastMCP(
    "orbix-maps",
    instructions=(
        "Google Maps location tools. geocode(address) → coordinates; search_places(query) "
        "→ nearby/matching places; directions(origin, destination) → route + distance/time. "
        "If a tool says the API key is missing, tell the user to paste a Google Maps key in "
        "the Maps integration settings."
    ),
)

_READONLY = ToolAnnotations(readOnlyHint=True)
_BASE = "https://maps.googleapis.com/maps/api"
_TIMEOUT = 20
_NO_KEY = {"error": "no Google Maps API key configured — paste one in the Maps integration settings"}


def _key() -> str:
    return os.environ.get("ORBIX_MAPS_API_KEY", "").strip()


def _get(path: str, params: dict) -> dict:
    key = _key()
    if not key:
        return _NO_KEY
    params = {**params, "key": key}
    try:
        r = requests.get(f"{_BASE}{path}", params=params, timeout=_TIMEOUT)
        data = r.json()
    except Exception as e:  # noqa: BLE001
        return {"error": f"request failed: {e}"}
    status = data.get("status")
    if status not in ("OK", "ZERO_RESULTS"):
        return {"error": f"Maps API: {status} — {data.get('error_message', '')}"}
    return data


@mcp.tool(annotations=_READONLY)
def geocode(address: str) -> dict:
    """Geocode an address/place name. Returns {address, lat, lng} or {error}."""
    if not (address or "").strip():
        return {"error": "address is required"}
    data = _get("/geocode/json", {"address": address})
    if data.get("error"):
        return data
    results = data.get("results", [])
    if not results:
        return {"error": "no match"}
    loc = results[0]["geometry"]["location"]
    return {"address": results[0]["formatted_address"], "lat": loc["lat"], "lng": loc["lng"]}


@mcp.tool(annotations=_READONLY)
def search_places(query: str, limit: int = 5) -> dict:
    """Find places matching a text query. Returns {places: [{name, address, rating}]}."""
    if not (query or "").strip():
        return {"error": "query is required"}
    data = _get("/place/textsearch/json", {"query": query})
    if data.get("error"):
        return data
    results = data.get("results", [])[: max(1, min(int(limit or 5), 20))]
    return {"places": [{"name": r.get("name"), "address": r.get("formatted_address"),
                        "rating": r.get("rating")} for r in results]}


@mcp.tool(annotations=_READONLY)
def directions(origin: str, destination: str, mode: str = "driving") -> dict:
    """
    Get directions between two places. `mode` is driving/walking/bicycling/transit.
    Returns {distance, duration, summary} or {error}.
    """
    if not origin or not destination:
        return {"error": "origin and destination are required"}
    data = _get("/directions/json", {"origin": origin, "destination": destination, "mode": mode})
    if data.get("error"):
        return data
    routes = data.get("routes", [])
    if not routes:
        return {"error": "no route found"}
    leg = routes[0]["legs"][0]
    return {"distance": leg["distance"]["text"], "duration": leg["duration"]["text"],
            "summary": routes[0].get("summary", ""),
            "start": leg["start_address"], "end": leg["end_address"]}


if __name__ == "__main__":
    mcp.run()  # stdio transport

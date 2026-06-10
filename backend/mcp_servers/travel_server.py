from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from google_service.travel_planner import (
    search_flights,
    search_hotels,
    get_attractions,
)

mcp = FastMCP("orbix-travel")

_READONLY = ToolAnnotations(readOnlyHint=True)


@mcp.tool(annotations=_READONLY)
def find_flights(
    from_city: str,
    to_city: str,
    departure_date: str,
    adults: int = 1,
):
    return search_flights(
        from_city,
        to_city,
        departure_date,
        adults,
    )


@mcp.tool(annotations=_READONLY)
def find_hotels(
    city: str,
    check_in: str,
    check_out: str,
    adults: int = 1,
):
    return search_hotels(
        city,
        check_in,
        check_out,
        adults,
    )


@mcp.tool(annotations=_READONLY)
def find_attractions(
    city: str,
    max_attractions: int = 20,
):
    return get_attractions(
        city,
        max_attractions,
    )


if __name__ == "__main__":
    mcp.run()
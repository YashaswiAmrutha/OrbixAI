#mcp_travel_server.py:
import sys
print("MCP SERVER STARTED", file=sys.stderr)
print(__file__, file=sys.stderr)
import asyncio
import json
from typing import Optional

from mcp.server.fastmcp import FastMCP
from browser.flights import search_flights as browser_search_flights
from browser.hotels import search_hotels as browser_search_hotels

from google_service.travel_services import (
    generate_itinerary,
    get_attractions,
)

mcp = FastMCP("TravelPlanner")



@mcp.tool()
async def search_flights_tool(
    from_city,
    to_city,
    departure_date,
    return_date=None
):
    print("TOOL ENTERED", file=sys.stderr)
    print("PLAYWRIGHT TOOL ENTERED", file=sys.stderr)

    try:
        result = await asyncio.wait_for(browser_search_flights(
                from_city,
                to_city,
                departure_date,
                return_date
            ), timeout=150)
    except Exception:
        import traceback
        traceback.print_exc()
        result = []

    print("RESULT =", result, file=sys.stderr)
    return json.dumps(result)

@mcp.tool()
async def search_hotels_tool(
    city: str,
    check_in: str,
    check_out: str,
    num_adults: int = 1
):
    print("HOTEL TOOL ENTERED", file=sys.stderr)

    try:
        result = await asyncio.wait_for(browser_search_hotels(
            city,
            check_in,
            check_out,
            num_adults
        ), timeout=75)

        print("HOTEL RESULT:", result, file=sys.stderr)

        return json.dumps(result)

    except Exception as e:
        import traceback
        traceback.print_exc()

        return json.dumps([])



@mcp.tool()
async def attractions_tool(city: str):
    import asyncio
    return json.dumps(await asyncio.to_thread(get_attractions, city))


@mcp.tool()
async def itinerary_tool(
    city: str,
    attractions: list,
    num_days: int,
    check_in: str,
    num_adults: int = 1,
    flight_summary: Optional[str] = None,
    hotel_summary: Optional[str] = None
):
    print("ITINERARY TOOL ENTERED", file=sys.stderr)

    try:
        import asyncio
        result = await asyncio.to_thread(
            generate_itinerary,
            city=city, attractions=attractions, num_days=num_days,
            check_in=check_in, num_adults=num_adults,
            flight_summary=flight_summary, hotel_summary=hotel_summary,
        )

        print("ITINERARY RESULT:", result, file=sys.stderr)

        return json.dumps(result)

    except Exception as e:
        import traceback
        traceback.print_exc()

        return json.dumps({
            "error": str(e)
        })



if __name__ == "__main__":
    mcp.run(transport="stdio")

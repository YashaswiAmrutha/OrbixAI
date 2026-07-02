# mcp_travel_server.py
import sys
print("MCP SERVER STARTED")
import json
import inspect


from mcp.server.fastmcp import FastMCP
from google_service.travel_services import (
    search_flights,
    search_hotels,
    generate_itinerary,
    get_attractions,
)

mcp = FastMCP("TravelPlanner")



@mcp.tool()
def search_flights_tool(
    from_city,
    to_city,
    departure_date,
    adults=1,
):
    print("TOOL ENTERED")

    result = search_flights(
        from_city,
        to_city,
        departure_date,
        adults
    )

    print("RESULT =", result)

    return json.dumps(result)

@mcp.tool()
def search_hotels_tool(
    city: str,
    check_in: str,
    check_out: str,
    num_adults: int = 1
):
    print("HOTEL TOOL ENTERED")

    try:
        result = search_hotels(
            city,
            check_in,
            check_out,
            num_adults
        )

        print("HOTEL RESULT:", result)

        return json.dumps(result)

    except Exception as e:
        import traceback
        traceback.print_exc()

        return json.dumps([])



@mcp.tool()
def attractions_tool(city: str):
    return json.dumps(get_attractions(city))


@mcp.tool()
def itinerary_tool(
    city: str,
    attractions: list,
    num_days: int,
    check_in: str,
    num_adults: int = 1,
    flight_summary: str = None,
    hotel_summary: str = None
):

    result = generate_itinerary(
        city=city,
        attractions=attractions,
        num_days=num_days,
        check_in=check_in,
        num_adults=num_adults,
        flight_summary=flight_summary,
        hotel_summary=hotel_summary
    )

    return json.dumps(result)



if __name__ == "__main__":
    mcp.run(transport="stdio")
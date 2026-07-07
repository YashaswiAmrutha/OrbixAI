# mcp_travel_client.py

from mcp.client.session import ClientSession
import json

async def search_flights_mcp(
    session,
    from_city,
    to_city,
    departure_date
):
    print(">>> search_flights_tool called <<<")
    print(from_city, to_city, departure_date)

    result = await session.call_tool(
        "search_flights_tool",
        arguments={
            "from_city": from_city,
            "to_city": to_city,
            "departure_date": departure_date,
        }
    )

    print("MCP RESULT:", result)

    if result.content:
        print("CONTENT:", result.content[0].text)
        return json.loads(result.content[0].text)

    return []


async def search_hotels_mcp(
    session,
    city,
    check_in,
    check_out,
    num_adults=1
):
    result = await session.call_tool(
        "search_hotels_tool",
        arguments={
            "city": city,
            "check_in": check_in,
            "check_out": check_out,
            "num_adults": num_adults
        }
    )
    if result.content:
        return json.loads(result.content[0].text)

    return []


async def get_attractions_mcp(
    session,
    city
):
    result =await session.call_tool(
        "attractions_tool",
        arguments={
            "city": city
        }
    )
    if result.content:
        return json.loads(result.content[0].text)

    return []


async def generate_itinerary_mcp(
    session,
    city,
    attractions,
    num_days,
    check_in,
    num_adults=1,
    flight_summary=None,
    hotel_summary=None
):
    result = await session.call_tool(
        "itinerary_tool",
        arguments={
            "city": city,
            "attractions": attractions,
            "num_days": num_days,
            "check_in": check_in,
            "num_adults": num_adults,
            "flight_summary": flight_summary,
            "hotel_summary": hotel_summary
        }
    )

    # Defensive: the itinerary is a plain string wrapped as JSON. If it comes back
    # empty or non-JSON, return the raw text (or "") instead of crashing.
    if result.content:
        text = getattr(result.content[0], "text", "") or ""
        try:
            return json.loads(text)
        except Exception:
            return text
    return ""
    